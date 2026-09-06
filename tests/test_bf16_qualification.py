import json

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
