from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import pytest
import torch

from benchmarks.bf16_model import ATTNRES_EPS, MODEL_DTYPE, Config, Model


def _oracle(
    values: torch.Tensor | Sequence[torch.Tensor],
    query: torch.Tensor,
    *,
    eps: float,
    scale: float,
) -> torch.Tensor:
    """Test-only BF16 oracle; the fixture itself has no FP32 model backend."""

    if isinstance(values, torch.Tensor):
        sources = tuple(values.unbind(0))
    else:
        sources = tuple(values)
    packed = torch.stack(sources, dim=0)
    values_f = packed.float()
    keys_f = values_f[..., -query.numel() :]
    query_f = query.float()
    logits = (keys_f * torch.rsqrt(keys_f.square().mean(-1, keepdim=True) + eps))
    logits = (logits * query_f).sum(-1) * scale
    weights = logits.softmax(dim=0)
    return (weights.unsqueeze(-1) * values_f).sum(dim=0).to(MODEL_DTYPE)


class _RecordingOracle:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[torch.Tensor, ...], torch.Tensor, float, float]] = []

    def __call__(
        self,
        values: Sequence[torch.Tensor],
        query: torch.Tensor,
        *,
        eps: float,
        scale: float,
    ) -> torch.Tensor:
        sources = tuple(values)
        self.calls.append((sources, query, eps, scale))
        return _oracle(sources, query, eps=eps, scale=scale)


def _canonical_args(config: Config) -> dict[str, object]:
    return {
        "n_layer": config.layers,
        "n_head": config.heads,
        "n_embd": config.width,
        "mlp_hidden_dim": config.ffn,
        "vocab_size": config.vocab,
        "block_size": config.context,
        "rope_theta": config.rope_theta,
        "norm_pos": "before",
        "qk_norm": True,
        "weight_tying": False,
        "use_attnres": True,
        "use_fused_attnres": False,
        "attnres_type": config.mode,
        "attnres_num_blocks": config.block_count,
        "attnres_key_norm": True,
        "attn_res_query_norm": False,
        "attn_res_query_init": "zero",
        "use_lrid": False,
        "attnres_block_average": False,
        "attnres_block_count_prior": False,
        "attnres_block_average_mode": "count",
    }


def test_requested_default_and_small_config_are_bf16_only():
    config = Config()
    assert asdict(config) == {
        "layers": 24,
        "width": 1536,
        "heads": 24,
        "ffn": 4224,
        "vocab": 100277,
        "context": 2048,
        "rank": 1536,
        "mode": "full",
        "block_count": 8,
        "activation_checkpointing": False,
        "rope_theta": 500000.0,
        "norm_pos": "before",
        "qk_norm": True,
        "attnres_eps": ATTNRES_EPS,
        "attnres_scale": 1.0,
    }
    small = Config.small()
    assert small.width == 32 and small.vocab == 97 and small.rank == 32
    assert MODEL_DTYPE is torch.bfloat16


def test_full_uses_one_callable_for_48_style_growing_reads():
    config = Config.small(layers=3, width=16, heads=4, ffn=32, rank=5, block_count=2)
    operator = _RecordingOracle()
    model = Model(config, op=operator)
    tokens = torch.randint(config.vocab, (2, config.context), dtype=torch.int64)

    logits = model(tokens)

    assert logits.shape == (2, config.context, config.vocab)
    assert logits.dtype is MODEL_DTYPE
    assert len(operator.calls) == 2 * config.layers
    assert [len(values) for values, *_ in operator.calls] == list(range(2, 2 * config.layers + 2))
    assert all(query.dtype is MODEL_DTYPE for _, query, _, _ in operator.calls)
    assert all(query.numel() == config.rank for _, query, _, _ in operator.calls)
    assert all(eps == ATTNRES_EPS and scale == 1.0 for _, _, eps, scale in operator.calls)


def test_block_passes_completed_sums_and_partial_to_the_same_callable():
    config = Config.small(
        layers=4,
        width=16,
        heads=4,
        ffn=32,
        rank=5,
        block_count=2,
        mode="block",
    )
    operator = _RecordingOracle()
    model = Model(config, op=operator)
    tokens = torch.randint(config.vocab, (1, 5), dtype=torch.int64)

    logits = model(tokens)

    assert logits.shape == (1, 5, config.vocab)
    assert len(operator.calls) == 2 * config.layers
    assert [len(values) for values, *_ in operator.calls] == [2, 2, 2, 2, 3, 3, 3, 3]
    assert all(query.numel() == config.rank for _, query, _, _ in operator.calls)
    assert all(source.dtype is MODEL_DTYPE for values, *_ in operator.calls for source in values)


@pytest.mark.parametrize("mode", ["full", "block"])
def test_activation_checkpointing_keeps_bf16_operator_gradients(mode: str):
    config = Config.small(
        layers=2,
        width=16,
        heads=4,
        ffn=32,
        mode=mode,
        activation_checkpointing=True,
    )
    operator = _RecordingOracle()
    model = Model(config, op=operator).train()
    tokens = torch.randint(config.vocab, (2, 5), dtype=torch.int64)
    loss = model(tokens).float().square().mean()
    loss.backward()

    assert len(operator.calls) == 2 * config.layers
    assert all(parameter.dtype is MODEL_DTYPE for parameter in model.parameters())
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        query.query.grad is not None and query.query.grad.dtype is MODEL_DTYPE
        for query in model.transformer.attn_residuals
    )


def test_obpm_conversion_maps_canonical_names_and_rejects_key_drift():
    config = Config.small(layers=1, width=16, heads=4, ffn=32, block_count=1)
    source_model = Model(config, op=_oracle)
    checkpoint = {
        "model": {name: value.float().clone() for name, value in source_model.state_dict().items()},
        "model_args": _canonical_args(config),
    }
    target = Model(config, op=_oracle)
    target.load_obpm_checkpoint(checkpoint)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, source_model.state_dict()[name], rtol=0, atol=0)

    missing = {name: value.clone() for name, value in checkpoint["model"].items()}
    missing.pop(next(iter(missing)))
    with pytest.raises(KeyError, match="missing canonical keys"):
        target.load_obpm_state_dict(missing, model_args=_canonical_args(config))

    extra = {name: value.clone() for name, value in checkpoint["model"].items()}
    extra["unexpected.weight"] = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(KeyError, match="unexpected canonical keys"):
        target.load_obpm_state_dict(extra, model_args=_canonical_args(config))


@pytest.mark.parametrize(
    "field,value",
    [("use_lrid", True), ("norm_pos", "after"), ("attnres_block_count_prior", True)],
)
def test_obpm_conversion_rejects_unsupported_architecture_flags(field: str, value: object):
    config = Config.small(mode="block")
    model = Model(config, op=_oracle)
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    args = _canonical_args(config)
    args[field] = value
    with pytest.raises(ValueError, match="unsupported|expected"):
        model.load_obpm_state_dict(state, model_args=args)


def test_obpm_conversion_maps_static_output_tail_sliced_checkpoint():
    config = Config.small(layers=1, width=16, heads=4, ffn=32, block_count=1, rank=5)
    source_model = Model(config, op=_oracle)
    canonical_state: dict[str, torch.Tensor] = {}
    for name, value in source_model.state_dict().items():
        if name.startswith("transformer.attn_residuals."):
            index = name.rsplit(".", 2)[1]
            canonical_state[f"transformer.lrid_queries.{index}"] = value.float().unsqueeze(0)
        elif name.endswith(".attn.c_proj.weight"):
            canonical_state[name.replace(
                ".attn.c_proj.weight", ".attn.c_proj.proj.weight"
            )] = value.float()
        elif name.endswith(".mlp.fc2.weight"):
            canonical_state[name.replace(
                ".mlp.fc2.weight", ".mlp.fc2.proj.weight"
            )] = value.float()
        else:
            canonical_state[name] = value.float()

    args = _canonical_args(config)
    args.update(
        {
            "use_lrid": True,
            "lrid_rank": config.rank,
            "lrid_projection_rank": config.rank,
            "lrid_num_heads": 1,
            "lrid_key_from_output_tail": True,
            "lrid_use_logit_scale": False,
        }
    )
    target = Model(config, op=_oracle)
    target.load_obpm_checkpoint({"model": canonical_state, "model_args": args})
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, source_model.state_dict()[name], rtol=0, atol=0)


def test_obpm_standard_checkpoint_rejects_rank_mismatch():
    standard = Config.small(layers=1, width=16, heads=4, ffn=32, block_count=1)
    sliced = Config.small(layers=1, width=16, heads=4, ffn=32, block_count=1, rank=5)
    source = Model(standard, op=_oracle)
    target = Model(sliced, op=_oracle)
    with pytest.raises(ValueError, match="rank"):
        target.load_obpm_state_dict(
            {name: value.clone() for name, value in source.state_dict().items()},
            model_args=_canonical_args(standard),
        )
