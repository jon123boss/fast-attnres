"""Regression tests for the isolated distributed-qualification patch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


from benchmarks import bf16_qualification_distributed as distributed


def _optimizer_state():
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    return model, optimizer


def test_restore_requires_exact_serialized_and_loaded_state(monkeypatch):
    model, optimizer = _optimizer_state()
    wrapped = SimpleNamespace(module=model)
    expected = distributed._state_snapshot(wrapped, [optimizer])
    serialized = distributed._clone_cpu(expected)
    with torch.no_grad():
        model.weight.add_(2)
    calls = []

    def reduce(flag, op=None):
        calls.append((tuple(flag.shape), op))

    monkeypatch.setattr(distributed.dist, "all_reduce", reduce)
    metrics = distributed._restore_serialized_state(
        wrapped, [optimizer], expected, serialized, torch.device("cpu")
    )

    assert metrics["exact"] is True
    assert metrics["serialized_exact"] is True
    assert metrics["restored_exact"] is True
    assert len(calls) == 2
    assert distributed._tree_equal(
        distributed._state_snapshot(wrapped, [optimizer]), expected
    )


def test_restore_rejects_round_trip_drift_before_loading(monkeypatch):
    model, optimizer = _optimizer_state()
    wrapped = SimpleNamespace(module=model)
    expected = distributed._state_snapshot(wrapped, [optimizer])
    serialized = distributed._clone_cpu(expected)
    serialized["model"]["weight"].view(-1)[0].add_(1)
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda flag, op=None: None)

    with pytest.raises(AssertionError, match="changed during round-trip"):
        distributed._restore_serialized_state(
            wrapped, [optimizer], expected, serialized, torch.device("cpu")
        )
    assert distributed._tree_equal(
        distributed._state_snapshot(wrapped, [optimizer]), expected
    )


def test_same_input_metrics_keep_bf16_continuation_difference_and_loss():
    uninterrupted = {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    resumed = {"weight": torch.tensor([1.0, 2.03125], dtype=torch.bfloat16)}
    uninterrupted_loss = torch.tensor(1.0, dtype=torch.bfloat16)
    resumed_loss = torch.tensor(1.03125, dtype=torch.bfloat16)

    metrics = distributed._same_input_metrics(
        resumed, uninterrupted, resumed_loss, uninterrupted_loss
    )

    assert metrics["same_inputs"] is True
    assert metrics["exact"] is False
    assert metrics["state"]["max_abs"] > 0
    assert metrics["loss_max_abs"] > 0
    assert metrics["loss_within_bf16_tolerance"] is True
    assert metrics["loss"]["within_bf16_tolerance"] is True


def test_original_distributed_contract_remains_frozen():
    assert distributed.TOLERANCE == {"rtol": 0.05, "atol": 0.05}
    assert distributed.PRIMARY_DEFAULT["width"] == 1536
    assert distributed.PRIMARY_DEFAULT["block_count"] == 8
    assert distributed.PRIMARY_DEFAULT["activation_checkpointing"] is False


def test_restore_rejects_optimizer_that_did_not_load(monkeypatch):
    model, optimizer = _optimizer_state()
    wrapped = SimpleNamespace(module=model)
    expected = distributed._state_snapshot(wrapped, [optimizer])
    serialized = distributed._clone_cpu(expected)
    next(iter(optimizer.state.values()))["exp_avg"].add_(1)
    monkeypatch.setattr(optimizer, "load_state_dict", lambda state: None)
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda flag, op=None: None)
    with pytest.raises(AssertionError, match="restoration diverged"):
        distributed._restore_serialized_state(wrapped, [optimizer], expected, serialized, torch.device("cpu"))


def test_continuation_still_rejects_out_of_oracle_loss():
    state = {"weight": torch.ones(1, dtype=torch.bfloat16)}
    with pytest.raises(AssertionError):
        distributed._same_input_metrics(state, state, torch.tensor(2.), torch.tensor(1.))


def test_exact_fast_path_keeps_finite_and_dtype_checks():
    finite = torch.tensor([1., 2.], dtype=torch.bfloat16)
    assert distributed._compare_tree({"x": finite}, {"x": finite.clone()}, "same") == {
        "tensor_count": 1, "max_abs": 0.0}
    assert not distributed._tree_equal(finite, finite.float())
    for value in (float("inf"), float("nan")):
        invalid = torch.tensor([value], dtype=torch.bfloat16)
        assert not distributed._tree_equal(invalid, invalid.clone())
        with pytest.raises(AssertionError, match="non-finite"):
            distributed._compare_tree(invalid, invalid.clone(), "invalid")


def test_exact_restore_does_not_repeat_approximate_scan(monkeypatch):
    model, optimizer = _optimizer_state()
    wrapped = SimpleNamespace(module=model)
    expected = distributed._state_snapshot(wrapped, [optimizer])
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda flag, op=None: None)
    def unused(*args, **kwargs):
        raise AssertionError("redundant approximate scan")
    monkeypatch.setattr(distributed, "_compare_tree", unused)
    result = distributed._restore_serialized_state(
        wrapped, [optimizer], expected, distributed._clone_cpu(expected), torch.device("cpu"))
    assert result["state"]["max_abs"] == 0.0 and result["state"]["tensor_count"] == 8
