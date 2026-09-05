import json
import pytest
import torch

from benchmarks import bf16_qualification_distributed as distributed
from benchmarks import bf16_qualification as qualification


def test_gpu_qualification_contract_is_frozen_and_covers_requested_shapes():
    assert qualification.BF16_TOLERANCE == {"rtol": 0.05, "atol": 0.05}
    assert qualification.GRAPH_REPLAYS == 8
    widths = {case["shape"][2] for case in qualification.DEFAULT_OPERATOR_CASES}
    assert {513, 3072, 4096} <= widths
    assert all(case["shared"] for case in qualification.DEFAULT_OPERATOR_CASES)
    assert all(case["graph"] for case in qualification.DEFAULT_OPERATOR_CASES[:1])


def test_runner_writes_json_checkpoint_and_reports_non_cuda_as_skipped(tmp_path):
    output = tmp_path / "qualification.json"
    report = qualification.run_qualification(
        {
            "device": "cpu",
            "operator_cases": [],
            "training": {"enabled": False},
        },
        output,
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert report == saved
    assert report["status"] == "skipped"
    assert report["failures"][0]["phase"] == "preflight"
    assert report["failed"] == 0


def test_distributed_defaults_preserve_primary_configuration_and_rank_override():
    primary = distributed._model_spec({"primary_rank": 384}, "primary")
    assert (
        primary["layers"],
        primary["width"],
        primary["heads"],
        primary["ffn"],
        primary["vocab"],
        primary["context"],
    ) == (24, 1536, 24, 4224, 100277, 2048)
    assert primary["mode"] == "block"
    assert primary["block_count"] == 8
    assert distributed._model_spec({"primary_rank": 384}, "primary")["rank"] == 384


def test_collective_gradient_mismatch_finishes_all_reductions(monkeypatch):
    from types import SimpleNamespace
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    calls = []
    def reduce(tensor, op=None):
        calls.append(tuple(tensor.shape))
        if tensor.ndim:
            tensor.mul_(2)
            if tensor.ndim == 2:
                tensor.add_(1)
    monkeypatch.setattr(distributed.dist, "all_reduce", reduce)
    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: 2)
    with pytest.raises(AssertionError, match="collective gradient mismatch"):
        distributed._collective_gradients(SimpleNamespace(module=model), torch.device("cpu"))
    assert calls == [(), (2, 2), (2,), (), ()]


def test_snapshot_unwraps_ddp_and_compiled_model():
    from types import SimpleNamespace
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    wrapped = SimpleNamespace(module=SimpleNamespace(_orig_mod=model))
    snapshot = distributed._state_snapshot(wrapped, [])
    assert set(snapshot["model"]) == {"weight", "bias"}
    assert snapshot["model"]["weight"].device.type == "cpu"
    assert snapshot["model"]["weight"].data_ptr() != model.weight.data_ptr()
