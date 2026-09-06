from __future__ import annotations

import pytest
import torch

from benchmarks import bf16_training


def test_training_progress_when_imported_by_worker(capsys):
    import json
    import time

    started = time.monotonic()
    bf16_training._progress("torch_compile", "qualification", started)
    report = json.loads(capsys.readouterr().out)
    assert report["backend"] == "torch_compile"
    assert report["phase"] == "qualification"
    assert 0 <= report["elapsed_s"] <= time.monotonic() - started


def test_memory_record_separates_incremental_and_global_totals():
    result = bf16_training._memory_record(
        100,
        250,
        180,
        model_incremental=70,
        model_optimizer_incremental=90,
    )

    assert result["peak_allocated_bytes_incremental"] == 150
    assert result["peak_allocated_bytes_global_total"] == 250
    assert result["persistent_incremental_allocated_bytes"] == 80
    assert result["model_incremental_allocated_bytes"] == 70
    assert result["model_optimizer_incremental_allocated_bytes"] == 90


def test_case_backend_filter_preserves_available_order_and_reports_missing():
    selected, missing = bf16_training._case_backend_items(
        {"backends": ["candidate", "missing", "candidate"]},
        {"reference": object(), "candidate": object(), "other": object()},
    )

    assert [name for name, _ in selected] == ["candidate"]
    assert missing == ["missing"]


def test_runtime_validation_stops_before_metadata_without_cuda(monkeypatch):
    monkeypatch.setattr(bf16_training.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        bf16_training,
        "metadata",
        lambda: pytest.fail("metadata must not be read without CUDA"),
    )

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        bf16_training._validate_runtime({"gpu": "H100"})


def test_runtime_validation_rejects_a_capability_substitution(monkeypatch):
    monkeypatch.setattr(bf16_training.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bf16_training.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        bf16_training,
        "metadata",
        lambda: {"gpu": "NVIDIA H100", "capability": [8, 0]},
    )

    with pytest.raises(RuntimeError, match="GPU substitution"):
        bf16_training._validate_runtime({"gpu": "H100"})


def test_dynamo_limits_are_bounded_and_restored():
    import torch._dynamo.config as dynamo_config

    before = {
        name: getattr(dynamo_config, name)
        for name in ("cache_size_limit", "accumulated_cache_size_limit")
    }
    configuration = bf16_training._configure_dynamo({
        "dynamo_cache_size_limit": 12,
        "dynamo_accumulated_cache_size_limit": 48,
    })
    try:
        assert configuration["limits"] == {
            "cache_size_limit": 12,
            "accumulated_cache_size_limit": 48,
            "fullgraph": True,
            "dynamic": False,
        }
        assert dynamo_config.cache_size_limit == 12
        assert dynamo_config.accumulated_cache_size_limit == 48
    finally:
        bf16_training._restore_dynamo(configuration)
    assert dynamo_config.cache_size_limit == before["cache_size_limit"]
    assert dynamo_config.accumulated_cache_size_limit == before[
        "accumulated_cache_size_limit"
    ]


def test_adamw_fallback_label_matches_the_constructor_that_succeeded(monkeypatch):
    model = torch.nn.Linear(3, 2)
    real_adamw = torch.optim.AdamW
    calls = []

    def fallback_adamw(parameters, **kwargs):
        calls.append(kwargs)
        if kwargs.get("fused") or kwargs.get("capturable"):
            raise TypeError("simulated unsupported fused AdamW")
        return real_adamw(parameters, **kwargs)

    monkeypatch.setattr(bf16_training.torch.optim, "AdamW", fallback_adamw)
    optimizers = bf16_training._optimizers(model, {})

    assert bf16_training._optimizer_label(optimizers) == "AdamW(foreach=True)"
    assert calls[:3] == [
        {"lr": .0003, "betas": (.9, .95), "weight_decay": 0.,
         "fused": True, "capturable": True},
        {"lr": .0003, "betas": (.9, .95), "weight_decay": 0., "fused": True},
        {"lr": .0003, "betas": (.9, .95), "weight_decay": 0., "foreach": True},
    ]


def test_save_resume_smoke_round_trips_model_and_optimizer_on_cpu():
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(4, 3)).square().mean().backward()
    optimizer.step()

    result = bf16_training._save_resume_smoke(model, [optimizer])

    assert result["status"] == "passed"
    assert result["model_state"]["max_abs"] == 0.0
    assert result["optimizer_state"]["max_abs"] == 0.0


def test_resume_next_update_restores_rng_buffers_position_and_fresh_optimizers():
    torch.manual_seed(19)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.2))
    model.register_buffer("updates", torch.tensor(0))
    instances, positions = [], []
    def make():
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        instances.append(optimizer)
        return [optimizer]
    optimizers = make()
    def step(position):
        positions.append(position)
        optimizers[0].zero_grad(set_to_none=True)
        loss = model(torch.ones(4, 3) * (position + 1)).square().sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizers[0].step()
        model.updates += 1
        return loss
    step(0)
    initial = bf16_training._cpu_state(model)
    rng = torch.get_rng_state().clone()
    result = bf16_training._resume_next_update(model, optimizers, step, make, next_input=3)
    assert result['status'] == 'passed' and result['state']['max_abs'] == 0
    assert positions == [0, 3, 3]
    assert len({id(x) for x in instances}) == 3
    assert optimizers[0] is instances[-1]
    bf16_training._compare_state_tree(bf16_training._cpu_state(model), initial, strict=True)
    assert torch.equal(torch.get_rng_state(), rng)


def test_resume_detects_optimizer_behavior_missing_from_its_state_dict():
    model = torch.nn.Linear(1, 1, bias=False)
    class IncompleteOptimizer(torch.optim.SGD):
        def __init__(self):
            super().__init__(model.parameters(), lr=.1)
            self.unsaved_updates = 0
        def step(self):
            self.unsaved_updates += 1
            model.weight.grad.mul_(self.unsaved_updates)
            super().step()
    make = lambda: [IncompleteOptimizer()]
    optimizers = make()
    def step(position):
        optimizers[0].zero_grad(set_to_none=True)
        loss = model(torch.ones(1, 1)).sum()
        loss.backward()
        optimizers[0].step()
        return loss
    step(0)
    with pytest.raises(AssertionError):
        bf16_training._resume_next_update(model, optimizers, step, make)


def test_clipped_snapshot_records_the_actual_global_norm_and_gradients():
    model = torch.nn.Linear(1, 1)
    model.weight.grad = torch.tensor([[3.]])
    model.bias.grad = torch.tensor([4.])
    result = bf16_training._clipped_gradients(model)
    assert result['preclip_norm'] == 5
    torch.testing.assert_close(result['gradients']['weight'], torch.tensor([[.6]]))
    torch.testing.assert_close(result['gradients']['bias'], torch.tensor([.8]))


def test_compare_state_tree_rejects_optimizer_state_key_drift():
    with pytest.raises(AssertionError, match="keys changed"):
        bf16_training._compare_state_tree(
            {"state": {}, "param_groups": []},
            {"state": {}, "param_groups": [], "extra": 1},
        )


def test_failure_classification_keeps_compiler_errors_unresolved():
    qualification = bf16_training._failure_record(
        "qualification", AssertionError("output mismatch")
    )
    compile_failure = bf16_training._failure_record(
        "compile", AssertionError("cache limit")
    )

    assert qualification["classification"] == "incorrect"
    assert compile_failure["classification"] == "unresolved"


@pytest.mark.parametrize("count", range(2, 14))
def test_round_schedule_balances_every_backend_pair(count):
    from itertools import combinations
    names = list(range(count))
    orders = [bf16_training._balanced_order(names, i) for i in range(120)]
    assert all(sorted(order) == names for order in orders)
    for a, b in combinations(names, 2):
        assert sum(order.index(a) < order.index(b) for order in orders) == 60
