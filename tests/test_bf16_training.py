from __future__ import annotations

import pytest
import torch

from benchmarks import bf16_training


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
