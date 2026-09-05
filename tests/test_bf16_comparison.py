from types import SimpleNamespace

import pytest

from benchmarks import bf16_comparison as comparison


def setup_memory(monkeypatch, free):
    gib = 2**30
    monkeypatch.setattr(comparison.torch.cuda, "mem_get_info", lambda: (free * gib, 80 * gib))
    monkeypatch.setattr(comparison.torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(comparison.torch.cuda, "memory_allocated", lambda: 0)


def test_capacity_rejection_never_partially_activates(monkeypatch):
    setup_memory(monkeypatch, 5)
    arm = {"memory": {"persistent_incremental_allocated_bytes": 6 * 2**30,
                       "peak_allocated_bytes_incremental": 25 * 2**30}}
    plan = comparison.activate_all([arm], lambda a: pytest.fail("must stay on CPU"), None)
    assert not plan["admitted"]


def test_different_models_cannot_alias_gpu_storage(monkeypatch):
    setup_memory(monkeypatch, 75)
    parameter = SimpleNamespace(is_cuda=True, grad=None,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 1234))
    model = SimpleNamespace(parameters=lambda: [parameter], buffers=lambda: [])
    arms = [{"model": model, "optimizers": [],
             "memory": {"persistent_incremental_allocated_bytes": 2**30,
                        "peak_allocated_bytes_incremental": 2**30}} for _ in range(2)]
    with pytest.raises(AssertionError, match="share GPU storage"):
        comparison.activate_all(arms, lambda a: None, lambda a: [])
