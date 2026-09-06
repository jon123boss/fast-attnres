"""Small BF16-only OBPM-shaped training fixture.

The fixture keeps the model surrounding Attention Residuals close to the
canonical OBPM layout while deliberately leaving the residual operator
injectable.  Full and sequential Block use the same public ``op`` call.  The
model passes ordered source tuples, so the CUDA operator can retain its source
list path; Block summaries are ordinary BF16 additions made by the caller.

This module has no CPU/FP32 model backend.  CPU tests inject a small BF16
equation oracle instead of using the public operator's CPU implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from attnres import attnres

MODEL_DTYPE = torch.bfloat16
ATTNRES_EPS = 2**-23
Schedule = Literal["full", "block"]


def _check_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rms_norm(x: Tensor) -> Tensor:
    """Apply the unparameterized RMSNorm used by canonical ``model.py``."""

    return F.rms_norm(x, (x.shape[-1],), eps=None)


@dataclass
class Config:
    """Architecture and schedule settings for :class:`Model`.

    The defaults are the requested training shape: 24 layers, width 1536,
    24 heads, SwiGLU hidden width 4224, vocabulary 100277, and context 2048.
    ``rank`` defaults to width for standard AttnRes; ``rank < width`` selects
    sliced output-tail routing. Parameters and carried residuals are BF16.
    ``norm_pos`` and ``qk_norm`` are retained as explicit fields so checkpoint
    conversion can reject a canonical checkpoint with a different architecture.
    """

    layers: int = 24
    width: int = 1536
    heads: int = 24
    ffn: int = 4224
    vocab: int = 100277
    context: int = 2048
    # ``rank == width`` is standard AttnRes.  A smaller rank uses the final
    # rank value coordinates as the implicit S-LR key through the same public
    # operator call.
    rank: int | None = None
    mode: Schedule = "full"
    block_count: int = 8
    activation_checkpointing: bool = False
    rope_theta: float = 500000.0
    norm_pos: Literal["before"] = "before"
    qk_norm: bool = True
    attnres_eps: float = ATTNRES_EPS
    attnres_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "layers",
            "width",
            "heads",
            "ffn",
            "vocab",
            "context",
            "block_count",
        ):
            _check_positive_int(name, getattr(self, name))
        if self.rank is None:
            self.rank = self.width
        _check_positive_int("rank", self.rank)
        if self.rank > self.width:
            raise ValueError("rank must satisfy 1 <= rank <= width")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if (self.width // self.heads) % 2:
            raise ValueError("per-head width must be even for rotary embeddings")
        self.mode = str(self.mode).lower()  # type: ignore[assignment]
        if self.mode not in {"full", "block"}:
            raise ValueError("mode must be 'full' or 'block'")
        self.norm_pos = str(self.norm_pos).lower()  # type: ignore[assignment]
        if self.norm_pos != "before":
            raise ValueError("this fixture only supports canonical norm_pos='before'")
        if self.qk_norm is not True:
            raise ValueError("this fixture requires canonical qk_norm=True")
        if not isinstance(self.activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be a bool")
        if not math.isfinite(float(self.rope_theta)) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(float(self.attnres_eps)) or self.attnres_eps <= 0:
            raise ValueError("attnres_eps must be finite and positive")
        if not math.isfinite(float(self.attnres_scale)):
            raise ValueError("attnres_scale must be finite")

    @classmethod
    def small(cls, **overrides: Any) -> Config:
        """Return a CPU-sized BF16 configuration for structural tests."""

        if "sequence" in overrides and "context" not in overrides:
            overrides["context"] = overrides.pop("sequence")
        if "blocks" in overrides and "block_count" not in overrides:
            overrides["block_count"] = overrides.pop("blocks")
        values: dict[str, Any] = {
            "layers": 2,
            "width": 32,
            "heads": 4,
            "ffn": 64,
            "vocab": 97,
            "context": 16,
            "block_count": 2,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_obpm_args(cls, model_args: Any) -> Config:
        return _config_from_obpm_args(cls, model_args)

    # Canonical-name aliases make shape checks and runner integration clear.
    @property
    def n_layer(self) -> int:
        return self.layers

    @property
    def n_embd(self) -> int:
        return self.width

    @property
    def n_head(self) -> int:
        return self.heads

    @property
    def mlp_hidden_dim(self) -> int:
        return self.ffn

    @property
    def vocab_size(self) -> int:
        return self.vocab

    @property
    def block_size(self) -> int:
        return self.context

    @property
    def sequence(self) -> int:
        return self.context

    @property
    def blocks(self) -> int:
        return self.block_count

    @property
    def dtype(self) -> torch.dtype:
        return MODEL_DTYPE

    @property
    def is_sliced(self) -> bool:
        """Whether residual routing uses an output-tail rank smaller than D."""

        return self.rank < self.width


class _RotaryEmbedding(nn.Module):
    """Interleaved rotary embedding matching canonical ``RotaryEmbedding``."""

    def __init__(self, head_width: int, context: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_width, 2, dtype=torch.float32) / head_width)
        )
        positions = torch.arange(context, dtype=torch.float32)
        frequency = torch.outer(positions, inv_freq)
        self.register_buffer("sin", frequency.sin()[None, None].to(MODEL_DTYPE))
        self.register_buffer("cos", frequency.cos()[None, None].to(MODEL_DTYPE))

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        sequence = q.shape[-2]
        sin = self.sin[:, :, :sequence].to(device=q.device, dtype=q.dtype)
        cos = self.cos[:, :, :sequence].to(device=q.device, dtype=q.dtype)
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]
        q_rotated = torch.stack(
            (cos * q_even - sin * q_odd, sin * q_even + cos * q_odd), dim=-1
        ).flatten(-2)
        k_rotated = torch.stack(
            (cos * k_even - sin * k_odd, sin * k_even + cos * k_odd), dim=-1
        ).flatten(-2)
        return q_rotated, k_rotated


class _Attention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.width = config.width
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.rope = _RotaryEmbedding(self.head_width, config.context, config.rope_theta)
        self.c_attn = nn.Linear(
            config.width, 3 * config.width, bias=False, dtype=MODEL_DTYPE
        )
        self.c_proj = nn.Linear(
            config.width, config.width, bias=False, dtype=MODEL_DTYPE
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, sequence, _ = x.shape
        q, k, v = self.c_attn(x).split(self.width, dim=-1)
        q = q.view(batch, sequence, self.heads, self.head_width).transpose(1, 2)
        k = k.view(batch, sequence, self.heads, self.head_width).transpose(1, 2)
        v = v.view(batch, sequence, self.heads, self.head_width).transpose(1, 2)
        q = _rms_norm(q)
        k = _rms_norm(k)
        q, k = self.rope(q, k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, self.width)
        return self.c_proj(attended)


class _MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            config.width, 2 * config.ffn, bias=False, dtype=MODEL_DTYPE
        )
        self.fc2 = nn.Linear(
            config.ffn, config.width, bias=False, dtype=MODEL_DTYPE
        )

    def forward(self, x: Tensor) -> Tensor:
        up, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class _Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn = _Attention(config)
        self.mlp = _MLP(config)

    def attention(self, x: Tensor) -> Tensor:
        return self.attn(_rms_norm(x))

    def mlp_forward(self, x: Tensor) -> Tensor:
        return self.mlp(_rms_norm(x))


class _DepthQuery(nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        # Canonical training initializes depth queries to zero.  Keep the
        # parameter BF16 so the operator sees BF16 query values and gradients.
        self.query = nn.Parameter(torch.zeros(rank, dtype=MODEL_DTYPE))


def _block_ends(events: int, block_count: int) -> tuple[int, ...]:
    count = min(events, block_count)
    return tuple(math.ceil(events * index / count) for index in range(1, count + 1))


class Model(nn.Module):
    """The exact 24-layer training fixture with an injected residual operator."""

    def __init__(self, config: Config, op: Callable[..., Tensor] = attnres) -> None:
        super().__init__()
        if not isinstance(config, Config):
            raise TypeError("config must be a Config")
        if not callable(op):
            raise TypeError("op must be callable")
        self.config = config
        # The operator is an injected execution dependency, not model state.
        # Bypass Module.__setattr__ so a callable nn.Module cannot accidentally
        # add optimizer parameters or checkpoint keys to this fixture.
        self.__dict__["op"] = op
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab, config.width, dtype=MODEL_DTYPE),
                "layers": nn.ModuleList([_Block(config) for _ in range(config.layers)]),
                "attn_residuals": nn.ModuleList(
                    [_DepthQuery(config.rank) for _ in range(2 * config.layers)]
                ),
            }
        )
        # Untied output is intentional and matches canonical OBPM checkpoints.
        self.lm_head = nn.Linear(
            config.width, config.vocab, bias=False, dtype=MODEL_DTYPE
        )
        self._initialize()

    def _initialize(self) -> None:
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_writer(self, fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
        if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def _read(self, sources: Sequence[Tensor], query_index: int) -> Tensor:
        if not sources:
            raise RuntimeError("a residual read requires at least one source")
        if not 0 <= query_index < len(self.transformer.attn_residuals):
            raise IndexError("residual query index is out of range")
        source_tuple = tuple(sources)
        first = source_tuple[0]
        if first.dtype is not MODEL_DTYPE:
            raise TypeError("residual values must use BF16 storage")
        if any(source.dtype is not MODEL_DTYPE for source in source_tuple):
            raise TypeError("all residual values must use BF16 storage")
        query = self.transformer.attn_residuals[query_index].query
        if query.dtype is not MODEL_DTYPE:
            raise TypeError("residual queries must use BF16 storage")
        if query.numel() != self.config.rank:
            raise RuntimeError(
                f"residual query has rank {query.numel()}, expected {self.config.rank}"
            )
        output = self.op(
            source_tuple,
            query,
            eps=self.config.attnres_eps,
            scale=self.config.attnres_scale,
        )
        if not isinstance(output, Tensor):
            raise TypeError("the residual operator must return a tensor")
        if tuple(output.shape) != tuple(first.shape):
            raise ValueError(
                f"the residual operator returned {tuple(output.shape)}, "
                f"expected {tuple(first.shape)}"
            )
        if output.dtype is not MODEL_DTYPE:
            raise TypeError("the residual operator must return BF16 storage")
        return output

    def _write(self, event_index: int, x: Tensor) -> Tensor:
        layer = self.transformer.layers[event_index // 2]
        if event_index % 2 == 0:
            return self._run_writer(layer.attention, x)
        return self._run_writer(layer.mlp_forward, x)

    def _forward_full(self, embedding: Tensor) -> Tensor:
        sources: list[Tensor] = [embedding]
        event_index = 0
        for _ in self.transformer.layers:
            read_input = (
                embedding
                if event_index == 0
                else self._read(sources, event_index - 1)
            )
            sources.append(self._write(event_index, read_input))
            event_index += 1

            read_input = self._read(sources, event_index - 1)
            sources.append(self._write(event_index, read_input))
            event_index += 1

        if event_index != 2 * self.config.layers:
            raise RuntimeError("Full schedule did not emit every sublayer")
        return self._read(sources, event_index - 1)

    def _forward_block(self, embedding: Tensor) -> Tensor:
        events = 2 * self.config.layers
        completed: list[Tensor] = [embedding]
        partial: Tensor | None = None
        event_index = 0
        for block_end in _block_ends(events, self.config.block_count):
            while event_index < block_end:
                if event_index == 0:
                    read_input = embedding
                else:
                    read_sources = (
                        tuple(completed)
                        if partial is None
                        else (*completed, partial)
                    )
                    read_input = self._read(read_sources, event_index - 1)
                emitted = self._write(event_index, read_input)
                partial = emitted if partial is None else partial + emitted
                event_index += 1

            if partial is None:
                raise RuntimeError("Block schedule completed an empty block")
            completed.append(partial)
            partial = None

        if event_index != events or partial is not None:
            raise RuntimeError("Block schedule did not emit every sublayer")
        return self._read(completed, event_index - 1)

    def forward(self, tokens: Tensor) -> Tensor:
        if not isinstance(tokens, Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.dtype not in (
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
        ):
            raise TypeError("tokens must contain integer token ids")
        if not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError("token sequence must be within configured context")
        embedding = self.transformer.wte(tokens)
        hidden = (
            self._forward_full(embedding)
            if self.config.mode == "full"
            else self._forward_block(embedding)
        )
        return self.lm_head(_rms_norm(hidden))

    def load_obpm_checkpoint(
        self, checkpoint: Mapping[str, Any], *, strict: bool = True
    ) -> Model:
        """Load a canonical OBPM payload after strict architecture/key checks."""

        if not strict:
            raise ValueError("OBPM conversion is always strict")
        mapped = convert_obpm_state_dict(checkpoint, self)
        self.load_state_dict(mapped, strict=True)
        return self

    def load_obpm_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        *,
        model_args: Mapping[str, Any] | Any | None = None,
    ) -> Model:
        """Load only the canonical ``model`` mapping, still strictly."""

        payload: dict[str, Any] = {"model": state_dict}
        if model_args is not None:
            payload["model_args"] = model_args
        return self.load_obpm_checkpoint(payload)

    @classmethod
    def from_obpm_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        config: Config | None = None,
        op: Callable[..., Tensor] = attnres,
    ) -> Model:
        """Construct and strictly load a model from a canonical OBPM payload."""

        _, model_args = _checkpoint_parts(checkpoint)
        if config is None:
            if model_args is None:
                raise ValueError("config is required when checkpoint has no model_args")
            config = Config.from_obpm_args(model_args)
        model = cls(config, op=op)
        model.load_obpm_checkpoint(checkpoint)
        return model

    from_canonical_checkpoint = from_obpm_checkpoint


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"{name} must be a mapping or an object with attributes")


def _reject_true(args: Mapping[str, Any], name: str, reason: str) -> None:
    if bool(args.get(name, False)):
        raise ValueError(f"unsupported OBPM architecture flag {name!r}: {reason}")


def _require_int(args: Mapping[str, Any], name: str) -> int:
    if name not in args:
        raise KeyError(f"canonical model_args is missing {name!r}")
    return _check_positive_int(name, args[name])


def _require_false(args: Mapping[str, Any], name: str, reason: str) -> None:
    if bool(args.get(name, False)):
        raise ValueError(f"unsupported OBPM architecture flag {name!r}: {reason}")


def _config_architecture_values(config: Config) -> dict[str, Any]:
    return {
        "layers": config.layers,
        "width": config.width,
        "heads": config.heads,
        "ffn": config.ffn,
        "vocab": config.vocab,
        "context": config.context,
        "rank": config.rank,
        "mode": config.mode,
        # Full never observes block boundaries, so its count is not part of
        # the checkpoint architecture.  Block does use it for source layout.
        "block_count": config.block_count if config.mode == "block" else None,
        "rope_theta": float(config.rope_theta),
        "norm_pos": config.norm_pos,
        "qk_norm": config.qk_norm,
    }


def _validate_model_args_match(config: Config, model_args: Any) -> None:
    parsed = Config.from_obpm_args(model_args)
    expected = _config_architecture_values(config)
    actual = _config_architecture_values(parsed)
    mismatches = {
        name: (expected[name], actual[name])
        for name in expected
        if expected[name] != actual[name]
    }
    if mismatches:
        raise ValueError(f"checkpoint architecture does not match Config: {mismatches}")


def _config_from_obpm_args(cls: type[Config], model_args: Any) -> Config:
    """Build a fixture config while rejecting non-public OBPM variants."""

    args = _as_mapping(model_args, "model_args")
    if not bool(args.get("use_attnres", True)):
        raise ValueError("canonical checkpoint must enable Attention Residuals")
    mode = str(args.get("attnres_type", "full")).lower()
    if mode not in {"full", "block"}:
        raise ValueError("canonical attnres_type must be 'full' or 'block'")
    if str(args.get("norm_pos", "before")).lower() != "before":
        raise ValueError("unsupported OBPM architecture flag 'norm_pos': expected 'before'")
    if args.get("qk_norm", True) is not True:
        raise ValueError("unsupported OBPM architecture flag 'qk_norm': expected True")
    if args.get("attnres_key_norm", True) is not True:
        raise ValueError(
            "unsupported OBPM architecture flag 'attnres_key_norm': expected True"
        )
    if str(args.get("attn_res_query_init", "zero")).lower() != "zero":
        raise ValueError(
            "unsupported OBPM architecture flag 'attn_res_query_init': expected 'zero'"
        )
    _require_false(args, "weight_tying", "the fixture has an untied output head")
    _require_false(args, "use_fused_attnres", "reads must use the injected public callable")
    _require_false(args, "attn_res_query_norm", "queries are static and unnormalized")

    use_lrid = bool(args.get("use_lrid", False))
    output_tail = bool(args.get("lrid_key_from_output_tail", False))
    if use_lrid != output_tail:
        if use_lrid:
            raise ValueError(
                "unsupported LRID combination: only static output-tail S-LR is supported"
            )
        raise ValueError(
            "lrid_key_from_output_tail requires use_lrid=True"
        )

    rank: int | None = None
    if use_lrid:
        rank = _check_positive_int("lrid_rank", args.get("lrid_rank", 64))
        num_heads = _check_positive_int(
            "lrid_num_heads", args.get("lrid_num_heads", 1)
        )
        if num_heads != 1:
            raise ValueError(
                "unsupported OBPM architecture flag 'lrid_num_heads': expected 1"
            )
        projection_rank = args.get("lrid_projection_rank")
        if projection_rank is not None:
            projection_rank = _check_positive_int(
                "lrid_projection_rank", projection_rank
            )
            if projection_rank != rank:
                raise ValueError(
                    "unsupported OBPM LRID projection rank: output-tail S-LR "
                    "requires lrid_projection_rank == lrid_rank"
                )
        for name in (
            "lrid_input_dependent_query",
            "lrid_static_embedding_key",
            "lrid_add_static_embedding_key",
            "lrid_add_static_source_key",
            "lrid_key_from_value",
            "lrid_key_from_value_shared",
            "lrid_query_from_value",
            "lrid_query_from_value_shared",
            "lrid_signed_depth",
        ):
            _reject_true(args, name, "only static output-tail S-LR is supported")
    else:
        for name in (
            "use_lrid",
            "lrid_input_dependent_query",
            "lrid_static_embedding_key",
            "lrid_add_static_embedding_key",
            "lrid_add_static_source_key",
            "lrid_key_from_value",
            "lrid_key_from_value_shared",
            "lrid_key_from_output_tail",
            "lrid_query_from_value",
            "lrid_query_from_value_shared",
            "lrid_signed_depth",
        ):
            _reject_true(args, name, "projected or dynamic LRID paths are not in this fixture")

    if mode == "block":
        for name, reason in (
            ("attnres_block_average", "Block values must be raw caller-supplied sums"),
            ("attnres_block_count_prior", "the public callable has no source-count prior"),
            ("attnres_block_split_sublayers", "only one partial sum is supported"),
            ("attnres_block_learned_scale", "learned Block scaling is not in the API"),
            ("attnres_block_value_norm", "Block values are passed without an extra norm"),
            ("attnres_block_alpha_learned", "Block powers are not in the API"),
            ("attnres_block_beta_learned", "Block powers are not in the API"),
        ):
            _reject_true(args, name, reason)
        if str(args.get("attnres_block_average_mode", "count")).lower() != "count":
            raise ValueError(
                "unsupported OBPM architecture flag 'attnres_block_average_mode'"
            )

    width = _require_int(args, "n_embd")
    ffn = _require_int(args, "mlp_hidden_dim")
    if rank is None:
        rank = width

    if use_lrid:
        if args.get("lrid_use_logit_scale", True) is False:
            residual_scale = 1.0
        else:
            raw_scale = args.get("lrid_logit_scale")
            residual_scale = (
                1.0 / math.sqrt(rank)
                if raw_scale is None
                else float(raw_scale)
            )
            if not math.isfinite(residual_scale) or residual_scale <= 0:
                raise ValueError("lrid_logit_scale must be finite and positive")
    else:
        residual_scale = 1.0

    return cls(
        layers=_require_int(args, "n_layer"),
        width=width,
        heads=_require_int(args, "n_head"),
        ffn=ffn,
        vocab=_require_int(args, "vocab_size"),
        context=_require_int(args, "block_size"),
        rank=rank,
        mode=mode,
        block_count=_require_int(args, "attnres_num_blocks")
        if mode == "block"
        else int(args.get("attnres_num_blocks", 1)),
        rope_theta=float(args.get("rope_theta", 500000.0)),
        attnres_scale=residual_scale,
    )

def _checkpoint_parts(
    checkpoint: Mapping[str, Any],
) -> tuple[Mapping[str, Tensor], Mapping[str, Any] | Any | None]:
    payload = _as_mapping(checkpoint, "checkpoint")
    if "model" in payload:
        state = payload["model"]
        model_args = payload.get("model_args")
    elif "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        state = payload["state_dict"]
        model_args = payload.get("model_args")
    else:
        state = payload
        model_args = None
    state_mapping = _as_mapping(state, "checkpoint model state")
    if any(not isinstance(name, str) for name in state_mapping):
        raise TypeError("checkpoint model keys must be strings")
    return state_mapping, model_args


def _checkpoint_is_sliced(
    source: Mapping[str, Any],
    model_args: Mapping[str, Any] | Any | None,
    config: Config,
) -> bool:
    """Select the strict standard or canonical output-tail state topology."""

    if model_args is not None:
        args = _as_mapping(model_args, "model_args")
        return bool(args.get("use_lrid", False))
    # Without model_args, the canonical state names are unambiguous.  A
    # smaller target rank also requires the output-tail topology; standard
    # width-D query tensors are rejected by the shape/key checks below.
    has_lrid_queries = any(
        name.startswith("transformer.lrid_queries.") for name in source
    )
    has_lrid_projections = any(
        name.endswith((".attn.c_proj.proj.weight", ".mlp.fc2.proj.weight"))
        for name in source
    )
    if has_lrid_queries != has_lrid_projections:
        raise KeyError(
            "mixed canonical standard and output-tail S-LR state keys are not supported"
        )
    return config.is_sliced or has_lrid_queries


def _obpm_key_mapping(
    config: Config, *, sliced: bool | None = None
) -> dict[str, str]:
    """Enumerate canonical source names and their fixture target names."""

    if sliced is None:
        sliced = config.is_sliced
    mapping: dict[str, str] = {
        "transformer.wte.weight": "transformer.wte.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for layer_index in range(config.layers):
        prefix = f"transformer.layers.{layer_index}"
        for suffix in ("attn.rope.sin", "attn.rope.cos", "attn.c_attn.weight"):
            key = f"{prefix}.{suffix}"
            mapping[key] = key
        if sliced:
            mapping[f"{prefix}.attn.c_proj.proj.weight"] = (
                f"{prefix}.attn.c_proj.weight"
            )
            mapping[f"{prefix}.mlp.fc2.proj.weight"] = f"{prefix}.mlp.fc2.weight"
        else:
            mapping[f"{prefix}.attn.c_proj.weight"] = f"{prefix}.attn.c_proj.weight"
            mapping[f"{prefix}.mlp.fc2.weight"] = f"{prefix}.mlp.fc2.weight"
        mapping[f"{prefix}.mlp.fc1.weight"] = f"{prefix}.mlp.fc1.weight"
    if sliced:
        for read_index in range(2 * config.layers):
            mapping[f"transformer.lrid_queries.{read_index}"] = (
                f"transformer.attn_residuals.{read_index}.query"
            )
    else:
        for read_index in range(2 * config.layers):
            key = f"transformer.attn_residuals.{read_index}.query"
            mapping[key] = key
    return mapping


def convert_obpm_state_dict(
    checkpoint: Mapping[str, Any], model: Model
) -> dict[str, Tensor]:
    """Map a canonical OBPM state into ``model`` with strict key/shape checks."""

    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    source, model_args = _checkpoint_parts(checkpoint)
    if model_args is not None:
        _validate_model_args_match(model.config, model_args)

    target_state = model.state_dict()
    mapping = _obpm_key_mapping(
        model.config,
        sliced=_checkpoint_is_sliced(source, model_args, model.config),
    )
    target_names = set(target_state)
    mapped_names = set(mapping.values())
    if target_names != mapped_names:
        raise RuntimeError(
            "fixture state topology changed without updating the explicit OBPM mapping: "
            f"unmapped target keys={sorted(target_names - mapped_names)}, "
            f"stale mapping keys={sorted(mapped_names - target_names)}"
        )
    source_names = set(source)
    canonical_names = set(mapping)
    missing = canonical_names - source_names
    unexpected = source_names - canonical_names
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing canonical keys: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected canonical keys: {sorted(unexpected)}")
        raise KeyError("; ".join(details))

    converted: dict[str, Tensor] = {}
    for source_name, target_name in mapping.items():
        value = source[source_name]
        target = target_state[target_name]
        if not isinstance(value, Tensor):
            raise TypeError(f"checkpoint entry {source_name!r} must be a tensor")
        if source_name.startswith("transformer.lrid_queries."):
            if value.ndim != 2 or value.shape[0] != 1:
                raise ValueError(
                    f"checkpoint entry {source_name!r} must have canonical shape [1, R], "
                    f"got {tuple(value.shape)}"
                )
            value = value.reshape(-1)
        if value.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(
                f"checkpoint entry {source_name!r} has unsupported dtype {value.dtype}"
            )
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"checkpoint entry {source_name!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(target.shape)}"
            )
        converted[target_name] = value.detach().to(
            device=target.device, dtype=target.dtype
        ).clone()
    return converted


def load_obpm_checkpoint(
    model: Model, checkpoint: Mapping[str, Any], *, strict: bool = True
) -> Model:
    """Functional wrapper around :meth:`Model.load_obpm_checkpoint`."""

    return model.load_obpm_checkpoint(checkpoint, strict=strict)


def model_from_obpm_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    config: Config | None = None,
    op: Callable[..., Tensor] = attnres,
) -> Model:
    return Model.from_obpm_checkpoint(checkpoint, config=config, op=op)


__all__ = [
    "ATTNRES_EPS",
    "MODEL_DTYPE",
    "Config",
    "Model",
    "convert_obpm_state_dict",
    "load_obpm_checkpoint",
    "model_from_obpm_checkpoint",
]
