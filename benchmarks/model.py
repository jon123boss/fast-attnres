"""A compact causal language model exercising the AttnRes training path.

The model deliberately keeps the surrounding Transformer implementation in
ordinary PyTorch.  The only backend-dependent calls are residual reads through
``attnres``.  Full and Block share that operator; Block changes only when the
reads occur and which sources they see.  This makes a state dict produced by
the reference path directly usable by the kernel path.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from typing import Callable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from attnres import attnres


Backend = Literal["kernel"] | Callable[..., Tensor]
Variant = Literal["standard", "sliced"]
Mode = Literal["full", "block"]
SourceLayout = Literal["packed", "list"]


# The max-rank source is now the standard implicit-tail topology.  Keep the
# public constant name for callers while versioning the changed state recipe.
CANONICAL_MAX_RANK_STATE_PROTOCOL = "canonical_implicit_max_rank_v1"


@contextmanager
def _temporary_cpu_seed(seed: int):
    """Run model construction from a private CPU RNG state.

    ``torch.manual_seed`` also touches CUDA generators.  The evaluator state
    protocol only needs deterministic CPU initialization, so use a local CPU
    generator to seed and restore the process CPU state without touching any
    CUDA generator.
    """

    previous = torch.random.get_rng_state()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    torch.random.set_rng_state(generator.get_state())
    try:
        yield
    finally:
        torch.random.set_rng_state(previous)


@dataclass
class TrainingConfig:
    """Shape and architecture settings for :class:`CausalAttnResLM`.

    ``block_count`` is the number of evenly sized source blocks.  Each block
    contains Transformer sublayer outputs; the token embedding is the initial
    completed source.  ``rank`` is the query/key width.  A missing rank uses
    the model width, which gives the ordinary full-width AttnRes equations.
    ``source_layout`` selects packed source tensors or ordered source lists
    for kernel/public residual reads; packed is the historical default. Block
    always invokes the same public operator once per residual read.
    """

    layers: int = 2
    width: int = 128
    heads: int = 4
    ffn: int = 384
    batch: int = 2
    sequence: int = 64
    vocab: int = 512
    block_count: int = 2
    variant: Variant = "standard"
    mode: Mode = "full"
    rank: int | None = None
    source_layout: SourceLayout = "packed"

    def __post_init__(self) -> None:
        integer_fields = (
            "layers",
            "width",
            "heads",
            "ffn",
            "batch",
            "sequence",
            "vocab",
            "block_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        self.variant = str(self.variant).lower()  # type: ignore[assignment]
        self.mode = str(self.mode).lower()  # type: ignore[assignment]
        self.source_layout = str(self.source_layout).lower()  # type: ignore[assignment]
        if self.variant not in {"standard", "sliced"}:
            raise ValueError("variant must be 'standard' or 'sliced'")
        if self.mode not in {"full", "block"}:
            raise ValueError("mode must be 'full' or 'block'")
        if self.source_layout not in {"packed", "list"}:
            raise ValueError("source_layout must be 'packed' or 'list'")
        if self.rank is None:
            self.rank = self.width
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise ValueError("rank must be an integer or None")
        if not 1 <= self.rank <= self.width:
            raise ValueError("rank must satisfy 1 <= rank <= width")
        if self.variant == "standard" and self.rank != self.width:
            raise ValueError("standard AttnRes requires rank == width")


def _block_ends(source_events: int, block_count: int) -> tuple[int, ...]:
    """Return one based source-event boundaries for evenly sized blocks."""

    count = min(source_events, block_count)
    return tuple(math.ceil(source_events * i / count) for i in range(1, count + 1))


def _reference_read(values: Tensor | Sequence[Tensor], query: Tensor) -> Tensor:
    """Independent test oracle with FP32 accumulation and BF16-compatible output."""

    if not isinstance(values, Tensor):
        values = torch.stack(tuple(values), dim=0)

    # Keep this equation local to the training fixture.  In particular, the
    # reference block path must not consume a prepared kernel cache.
    value_f = values.to(torch.float32)
    key_f = values[..., -query.numel() :].to(torch.float32)
    query_f = query.to(torch.float32)
    key_norm = torch.rsqrt(key_f.square().mean(dim=-1, keepdim=True) + 2**-23)
    logits = (key_f * key_norm * query_f).sum(dim=-1)
    probabilities = torch.softmax(logits, dim=0)
    return (probabilities.unsqueeze(-1) * value_f).sum(dim=0).to(values.dtype)


class _TransformerLayer(nn.Module):
    """Pre-norm causal self-attention followed by a SwiGLU MLP."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        width = config.width

        self.norm_attention = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.attention_output = nn.Linear(width, width, bias=False)

        self.norm_mlp = nn.LayerNorm(width)
        self.gate_proj = nn.Linear(width, config.ffn, bias=False)
        self.up_proj = nn.Linear(width, config.ffn, bias=False)
        self.down_proj = nn.Linear(config.ffn, width, bias=False)

        self.width = width
        self.heads = config.heads
        self.head_width = width // config.heads

    def attention(self, x: Tensor) -> Tensor:
        normalized = self.norm_attention(x)
        batch, sequence, _ = normalized.shape
        q = self.q_proj(normalized).view(batch, sequence, self.heads, self.head_width)
        k = self.k_proj(normalized).view(batch, sequence, self.heads, self.head_width)
        v = self.v_proj(normalized).view(batch, sequence, self.heads, self.head_width)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # is_causal=True creates the triangular mask in the SDPA implementation.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, sequence, self.width)
        return self.attention_output(attended)

    def mlp(self, x: Tensor) -> Tensor:
        normalized = self.norm_mlp(x)
        hidden = F.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return self.down_proj(hidden)


class CausalAttnResLM(nn.Module):
    """Small causal Transformer with Full or Block attention residual reads."""

    def __init__(self, config: TrainingConfig, backend: Backend) -> None:
        super().__init__()
        self.config = config
        self.variant = config.variant
        self.mode = config.mode
        self.rank = config.rank
        self.backend = backend

        self.token_embedding = nn.Embedding(config.vocab, config.width)
        self.position_embedding = nn.Parameter(torch.empty(config.sequence, config.width))

        self.layers = nn.ModuleList([_TransformerLayer(config) for _ in range(config.layers)])
        self.lm_head = nn.Linear(config.width, config.vocab, bias=False)

        # The first attention consumes the embedding directly.  Subsequent
        # reads (before every later sublayer and before the LM head) have one
        # static learned query each.  ParameterList keeps all queries visible
        # in state_dicts and gives reference/kernel models identical topology.
        self.queries = nn.ParameterList(
            [nn.Parameter(torch.empty(config.rank)) for _ in range(2 * config.layers)]
        )
        # Native FLA accepts an explicit unit RMS weight.  Keep that weight on
        # the model so it is allocated before compilation/capture and follows
        # the model through ``to(device)``.  It is non-persistent because it is
        # a parameter-free constant and must not alter state matching.
        if getattr(backend, "accepts_rms_weight", False) is True:
            self.register_buffer(
                "_backend_rms_weight",
                torch.ones((config.rank,), dtype=torch.float32),
                persistent=False,
            )
        else:
            self._backend_rms_weight = None
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        for query in self.queries:
            # A nonzero initialization avoids an uninformative uniform read and
            # makes query gradients observable from the first training step.
            nn.init.normal_(query, mean=0.0, std=0.02)

    def _embedding_source(self, tokens: Tensor) -> Tensor:
        embedded = self.token_embedding(tokens)
        sequence = tokens.shape[-1]
        embedded = embedded + self.position_embedding[:sequence].unsqueeze(0)
        if torch.is_autocast_enabled(embedded.device.type):
            embedded = embedded.to(torch.get_autocast_dtype(embedded.device.type))
        return embedded

    def _operator(self) -> Callable[..., Tensor]:
        if callable(self.backend):
            return self.backend
        if isinstance(self.backend, str) and self.backend == "kernel":
            return attnres
        raise ValueError("backend must be 'kernel' or a callable")

    def _operator_inputs(
        self,
        values: Sequence[Tensor],
        operator: Callable[..., Tensor],
    ) -> Tensor | tuple[Tensor, ...]:
        """Build packed or source-list inputs for one residual read.

        The native FLA compile bridge consumes a sequence of contiguous source
        tensors and builds its pointer table inside the opaque CUDA custom op.
        That capability remains list based even when the model's explicit
        ``source_layout`` is ``"packed"``.  Reference reads retain the
        straightforward stacked equation regardless of the model layout.
        """

        accepts_source_list = (
            getattr(operator, "accepts_source_list", False) is True
            and self.variant in {"standard", "sliced"}
            and self.rank == self.config.width
        )
        configured_source_list = self.config.source_layout == "list"
        if accepts_source_list or configured_source_list:
            return tuple(values)

        return torch.stack(tuple(values), dim=0)

    def _read(self, values: Sequence[Tensor], query: Tensor) -> Tensor:
        """Invoke the one residual primitive shared by Full and Block."""

        operator = self._operator()
        operator_values = self._operator_inputs(values, operator)
        if self._backend_rms_weight is not None:
            return operator(
                operator_values,
                query,
                rms_weight=self._backend_rms_weight,
            )
        return operator(operator_values, query)

    def _forward_full(self, embedding_value: Tensor) -> Tensor:
        values: list[Tensor] = [embedding_value]
        event_index = 0
        for layer in self.layers:
            read = (
                values[0]
                if event_index == 0
                else self._read(values, self.queries[event_index - 1])
            )
            attention_output = layer.attention(read)
            values.append(attention_output)
            event_index += 1

            read = self._read(values, self.queries[event_index - 1])
            mlp_output = layer.mlp(read)
            values.append(mlp_output)
            event_index += 1

        return self._read(values, self.queries[event_index - 1])

    def _forward_block(self, embedding_value: Tensor) -> Tensor:
        """Run Block with the same residual read helper used by Full."""

        events = 2 * len(self.layers)
        ends = _block_ends(events, self.config.block_count)
        completed_values: list[Tensor] = [embedding_value]
        partial_value: Tensor | None = None
        previous_end = 0
        event_index = 0
        for end in ends:
            for _ in range(previous_end, end):
                if event_index == 0:
                    output = embedding_value
                else:
                    values = list(completed_values)
                    if partial_value is not None:
                        values.append(partial_value)
                    output = self._read(values, self.queries[event_index - 1])
                layer = self.layers[event_index // 2]
                emitted = layer.attention(output) if event_index % 2 == 0 else layer.mlp(output)
                partial_value = emitted if partial_value is None else partial_value + emitted
                event_index += 1
            completed_values.append(partial_value)  # type: ignore[arg-type]
            partial_value = None
            previous_end = end
        return self._read(completed_values, self.queries[event_index - 1])

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            raise TypeError("tokens must contain integer token ids")
        if tokens.shape[1] > self.config.sequence:
            raise ValueError("token sequence exceeds configured sequence length")
        embedding_value = self._embedding_source(tokens)
        if self.mode == "full":
            hidden = self._forward_full(embedding_value)
        else:
            hidden = self._forward_block(embedding_value)
        return self.lm_head(hidden)


def make_model(config: TrainingConfig, backend: Backend = "kernel") -> CausalAttnResLM:
    """Construct a training model with the requested residual operator backend."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if isinstance(backend, str) and backend != "kernel":
        raise ValueError("backend must be 'kernel' or a callable")
    if not isinstance(backend, str) and not callable(backend):
        raise ValueError("backend must be 'kernel' or a callable")
    return CausalAttnResLM(config, backend)


def canonical_max_rank_state(config: TrainingConfig, seed: int) -> dict[str, Tensor]:
    """Build the ``canonical_implicit_max_rank_v1`` standard ``R=D`` state.

    This is deliberately an evaluator helper rather than part of the model's
    default initializer.  The source model is always constructed on CPU with
    the requested architecture and mode, then its detached state is returned
    as CPU tensors.  The private CPU RNG context leaves both the caller's CPU
    state and every CUDA generator unchanged.
    """

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    canonical_config = replace(config, variant="standard", rank=config.width)
    with _temporary_cpu_seed(seed), torch.device("cpu"):
        source = make_model(canonical_config, backend=_reference_read)
    try:
        return {
            name: value.detach().cpu().clone()
            for name, value in source.state_dict().items()
        }
    finally:
        del source


def map_canonical_max_rank_state(
    model: CausalAttnResLM,
    canonical_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Map canonical standard ``R=D`` tensors into a rank target.

    Standard targets copy the canonical state exactly.  Sliced targets use the
    trailing ``R`` query coordinates because implicit keys are read from the
    value tensor's trailing ``R`` features.
    """

    if not hasattr(model, "config") or not callable(getattr(model, "state_dict", None)):
        raise TypeError("model must expose config and state_dict()")
    if not isinstance(canonical_state, Mapping):
        raise TypeError("canonical_state must be a mapping of tensor names to tensors")

    config = model.config
    target_state = model.state_dict()
    mapped: dict[str, Tensor] = {}
    for name, target in target_state.items():
        if name not in canonical_state:
            raise KeyError(f"canonical state is missing {name!r}")
        source = canonical_state[name]
        if not isinstance(source, Tensor):
            raise TypeError(f"canonical state entry {name!r} is not a tensor")

        if name.startswith("queries."):
            if config.variant == "sliced":
                selected = source[-target.numel() :]
            else:
                selected = source[: target.numel()]
        else:
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"canonical fixed-shape tensor {name!r} has shape "
                    f"{tuple(source.shape)}, expected {tuple(target.shape)}"
                )
            selected = source

        if tuple(selected.shape) != tuple(target.shape):
            raise ValueError(
                f"canonical mapping for {name!r} has shape {tuple(selected.shape)}, "
                f"expected {tuple(target.shape)}"
            )
        if selected.dtype != target.dtype:
            raise ValueError(
                f"canonical mapping for {name!r} has dtype {selected.dtype}, "
                f"expected {target.dtype}"
            )
        mapped[name] = selected.detach().clone()

    return mapped


def load_canonical_max_rank_state(
    model: CausalAttnResLM,
    canonical_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Load a canonical rank-sweep state and return the mapped CPU tensors."""

    mapped = map_canonical_max_rank_state(model, canonical_state)
    model.load_state_dict(mapped, strict=True)
    return mapped


def make_model_with_canonical_state(
    config: TrainingConfig,
    backend: Backend,
    canonical_state: Mapping[str, Tensor],
    seed: int,
) -> CausalAttnResLM:
    """Construct a target model and load a canonical evaluator state.

    Target construction is also CPU-only seeded and restored.  The returned
    model remains on CPU so callers can inspect/hash the initial state before
    moving it to the benchmark device.
    """

    with _temporary_cpu_seed(seed), torch.device("cpu"):
        model = make_model(config, backend=backend)
    load_canonical_max_rank_state(model, canonical_state)
    return model


def _microbatches(tokens: Tensor, targets: Tensor, accumulation: int) -> list[tuple[Tensor, Tensor]]:
    if tokens.shape != targets.shape:
        raise ValueError("tokens and targets must have matching shapes")
    if tokens.ndim == 3:
        if tokens.shape[0] != accumulation:
            raise ValueError("3D token input must have leading dimension equal to accumulation")
        return list(zip(tokens.unbind(0), targets.unbind(0)))
    if tokens.ndim != 2:
        raise ValueError("tokens and targets must have shape [batch, sequence]")
    if accumulation == 1:
        return [(tokens, targets)]
    if tokens.shape[0] < accumulation:
        raise ValueError("batch dimension must be at least accumulation")
    # A plain [batch, sequence] batch can also be split into microbatches.  This
    # keeps the helper convenient for callers that do not materialize a leading
    # microbatch dimension themselves.
    return list(zip(tokens.tensor_split(accumulation, dim=0), targets.tensor_split(accumulation, dim=0)))


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: Tensor,
    targets: Tensor,
    *,
    accumulation: int = 1,
) -> Tensor:
    """Run one optimizer step and return the detached mean cross-entropy loss.

    For gradient accumulation, pass tensors shaped ``[microbatch, batch,
    sequence]``.  The loss is divided before each backward call, while the
    returned detached value is the unscaled mean loss across microbatches.
    """

    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError("accumulation must be a positive integer")
    batches = _microbatches(tokens, targets, accumulation)
    optimizer.zero_grad(set_to_none=True)
    losses: list[Tensor] = []
    divisor = float(len(batches))
    for micro_tokens, micro_targets in batches:
        logits = model(micro_tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), micro_targets.reshape(-1))
        losses.append(loss.detach())
        (loss / divisor).backward()
    optimizer.step()
    return torch.stack(losses).mean().detach()


__all__ = [
    "TrainingConfig",
    "CausalAttnResLM",
    "make_model",
    "training_step",
    "CANONICAL_MAX_RANK_STATE_PROTOCOL",
    "canonical_max_rank_state",
    "map_canonical_max_rank_state",
    "load_canonical_max_rank_state",
    "make_model_with_canonical_state",
]
