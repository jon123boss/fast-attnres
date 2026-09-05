import json

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
