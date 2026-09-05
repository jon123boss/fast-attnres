import gc
import json
from types import SimpleNamespace

import pytest

from benchmarks import bf16_training as training
from benchmarks import bf16_timing_diagnostic as diagnostic


@pytest.mark.parametrize("fails", [False, True])
def test_observer_preserves_evaluator_and_restores_hooks(tmp_path, monkeypatch, fails):
    op = lambda: None
    collector_enabled = gc.isenabled()
    callbacks = list(gc.callbacks)
    original_activate, original_select = training._activate_arm, training._case_backend_items
    monkeypatch.setattr(training, "_activate_arm", lambda arm: None)
    activate = training._activate_arm

    def evaluate(config, checkpoint):
        training._case_backend_items({"backends": ["candidate"]}, {"candidate": op})
        def step(index):
            gc.collect()
            return 42
        arm = {"model": SimpleNamespace(op=op), "step": step}
        training._activate_arm(arm)
        wrapped = arm["step"]
        training._activate_arm(arm)
        assert arm["step"] is wrapped
        assert wrapped(7) == 42
        checkpoint({"status": "running", "sentinel": 17})
        if fails:
            raise RuntimeError("evaluator failure")
        checkpoint({"status": "complete", "sentinel": 17})
        return 17

    monkeypatch.setattr(training, "run_training", evaluate)
    output = tmp_path / "report.json"
    if fails:
        with pytest.raises(RuntimeError, match="evaluator failure"):
            diagnostic.run({}, output)
    else:
        assert diagnostic.run({}, output) == 17
    assert training._activate_arm is activate and training._case_backend_items is original_select
    assert gc.isenabled() is collector_enabled and gc.callbacks == callbacks
    assert json.loads(output.read_text())["sentinel"] == 17
    data = json.loads(output.with_suffix(".gc.json").read_text())
    assert len(data["steps"]) == 1
    step = data["steps"][0]
    assert step["backend"] == "candidate" and step["input_index"] == 7
    assert any(step["start_ns"] <= row["start_ns"] < row["stop_ns"] <= step["stop_ns"]
               for row in data["gc_intervals"])


def test_primary_contract_cannot_be_timed_through_observer(tmp_path):
    with pytest.raises(ValueError, match="diagnostic"):
        diagnostic.run({"primary_contract_sha256": ""}, tmp_path / "report.json")
