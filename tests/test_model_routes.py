from __future__ import annotations

import torch

from benchmarks import model as model_module
from benchmarks.model import TrainingConfig, make_model
from validation.oracle import oracle


def _oracle_backend(values, query, *, eps=2**-23, scale=1.0, rms_weight=None):
    """Use the frozen BF16 oracle for model routing fixtures."""

    del rms_weight
    if isinstance(values, (list, tuple)):
        values = torch.stack(tuple(values), dim=0)
    return oracle(values, query, eps=eps, scale=scale)


def _config(mode: str, layout: str) -> TrainingConfig:
    return TrainingConfig(
        layers=2,
        width=16,
        heads=4,
        ffn=32,
        batch=1,
        sequence=4,
        vocab=31,
        block_count=2,
        variant="sliced",
        mode=mode,
        rank=4,
        source_layout=layout,
    )


def test_full_and_block_resolve_the_same_public_operator():
    from attnres import attnres

    for layout in ("packed", "list"):
        full = make_model(_config("full", layout), backend="kernel")
        block = make_model(_config("block", layout), backend="kernel")
        assert full._operator() is attnres is block._operator()
        assert not hasattr(full.config, "block_execution")
        assert not hasattr(block.config, "block_execution")


def test_full_and_block_use_the_same_read_scheduler(monkeypatch):
    calls: list[tuple[str, int]] = []
    original = model_module.CausalAttnResLM._read

    def record(self, values, query):
        calls.append((self.mode, len(values)))
        return original(self, values, query)

    monkeypatch.setattr(model_module.CausalAttnResLM, "_read", record)
    tokens = torch.randint(31, (1, 4))
    make_model(_config("full", "packed"), backend=_oracle_backend)(tokens)
    make_model(_config("block", "packed"), backend=_oracle_backend)(tokens)
    assert calls == [
        *(("full", count) for count in (2, 3, 4, 5)),
        *(("block", count) for count in (2, 2, 3, 3)),
    ]


def test_full_and_block_share_read_helper_and_preserve_source_schedules(monkeypatch):
    for layout, expected_kind in (("packed", torch.Tensor), ("list", tuple)):
        calls: list[tuple[type, int]] = []

        def recorder(values, query):
            calls.append((type(values), len(values)))
            packed = values if isinstance(values, torch.Tensor) else torch.stack(values)
            return oracle(packed, query)

        monkeypatch.setattr(model_module, "attnres", recorder)
        tokens = torch.randint(31, (1, 4))
        make_model(_config("full", layout), backend="kernel")(tokens)
        assert calls == [(expected_kind, count) for count in (2, 3, 4, 5)]
        calls.clear()
        make_model(_config("block", layout), backend="kernel")(tokens)
        assert calls == [(expected_kind, count) for count in (2, 2, 3, 3)]


def test_model_backend_rms_weight_is_preallocated_reused_and_nonpersistent():
    seen: list[torch.Tensor] = []

    def backend(values, query, *, rms_weight):
        seen.append(rms_weight)
        packed = values if isinstance(values, torch.Tensor) else torch.stack(values, dim=0)
        return oracle(packed, query)

    backend.accepts_rms_weight = True
    config = _config("full", "list")
    model = make_model(config, backend=backend)

    assert model._backend_rms_weight.shape == (config.rank,)
    assert model._backend_rms_weight.dtype == torch.float32
    assert "_backend_rms_weight" not in model.state_dict()

    model(torch.randint(config.vocab, (config.batch, config.sequence)))
    assert len(seen) == 4
    assert all(weight is model._backend_rms_weight for weight in seen)

    # ``Module.to`` must move and cast the preallocated constant with the
    # model, while state matching remains limited to persistent parameters.
    model.to(dtype=torch.bfloat16)
    assert model._backend_rms_weight.dtype == torch.bfloat16
    model(torch.randint(config.vocab, (config.batch, config.sequence)))
    assert seen[-1] is model._backend_rms_weight
    assert seen[-1].dtype == torch.bfloat16


def test_fla_weight_contract_is_static_and_keeps_direct_call_fallback():
    import inspect

    from benchmarks import fla_compile

    source = inspect.getsource(fla_compile.make_model_backend)
    assert "query.new_ones" not in source
    assert "accepts_rms_weight" in source
    assert "rms_weight is None" in source
