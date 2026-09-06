"""CPU/fullgraph contract tests for the explicit Catswe model adapter."""

from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from benchmarks import catswe
from benchmarks.model import TrainingConfig, make_model


class _CpuNativeCatswe(catswe.CatsweBackend):
    """Use the independent phase1 equation while retaining native ABI checks."""

    @staticmethod
    def _check(values, query, *, require_cuda=True):
        return catswe.CatsweBackend._check(values, query, require_cuda=False)


def _backend():
    call = _CpuNativeCatswe(catswe._cpu_phase1, Path("/pinned/catswe"), native=True)
    comparator = SimpleNamespace(
        available=True,
        call=call,
        vendor_root="/pinned/catswe",
        vendor_revision=catswe.PINNED_REVISION,
        describe=lambda: {
            "name": catswe.NAME,
            "status": "available",
            "vendor_revision": catswe.PINNED_REVISION,
        },
    )
    return catswe.make_model_backend(comparator)


def test_model_backend_requires_verified_native_call_and_declares_public_abi():
    missing = SimpleNamespace(available=False, reason="not pinned")
    with pytest.raises(RuntimeError, match="not pinned"):
        catswe.make_model_backend(missing)

    backend = _backend()
    assert backend.accepts_source_list is True
    assert backend.native_model_source_list is False
    assert backend.supports_full is True
    assert backend.supports_per_read_block is True
    metadata = backend.source_hash_metadata
    assert metadata["model_scope"] == "compiled_training_step"
    assert metadata["cache_api"] == "none"
    assert metadata["prepare_api"] == "none"
    assert metadata["merge_api"] == "none"
    assert metadata["phase2_api"] == "none"
    assert "stack" in metadata["source_list_copy"]
    assert "contiguous" in metadata["packed_copy"]


def test_model_backend_stages_source_list_and_supports_autograd():
    backend = _backend()
    values = [
        torch.randn(1, 1, 4, 8, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    query = torch.randn(8, dtype=torch.float32, requires_grad=True)
    stacked = {"count": 0}
    original_stack = catswe.torch.stack

    def record_stack(*args, **kwargs):
        stacked["count"] += 1
        return original_stack(*args, **kwargs)

    # The model adapter, rather than the caller/model scheduler, owns this
    # source-list materialization boundary.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(catswe.torch, "stack", record_stack)
        output = backend(tuple(values), query)
    assert stacked["count"] == 1
    assert output.shape == (1, 1, 4, 8)
    assert output.dtype == torch.bfloat16
    gradients = torch.autograd.grad(output.float().sum(), [*values, query])
    assert all(gradient is not None for gradient in gradients)


def test_model_backend_runs_inside_a_fullgraph_cpu_model():
    backend = _backend()
    config = TrainingConfig(
        layers=1,
        width=8,
        heads=1,
        ffn=16,
        batch=1,
        sequence=4,
        vocab=16,
        rank=8,
        mode="block",
        block_count=1,
        source_layout="list",
    )
    model = make_model(config, backend=backend)
    compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)
    tokens = torch.randint(0, config.vocab, (config.batch, config.sequence))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = compiled(tokens)
    assert output.shape == (config.batch, config.sequence, config.vocab)
    assert output.dtype == torch.bfloat16
