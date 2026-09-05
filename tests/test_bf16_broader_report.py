import copy
import json
from pathlib import Path

import pytest

from benchmarks import bf16_broader_report as report


@pytest.fixture
def evidence(tmp_path):
    contract = report.read(Path(__file__).parents[1] / "configs/bf16_broader.json")
    name = "retained-job"
    snapshot = tmp_path / "snapshots" / name
    runner = snapshot / "runner"
    runner.mkdir(parents=True)
    evaluator = runner / "evaluator.py"
    evaluator.write_text("# immutable test fixture\n")
    contract["evaluator_files"] = {"evaluator.py": report.file_hash(evaluator)}
    config = {k: copy.deepcopy(contract[k]) for k in
              ("kind", "rounds", "warmups", "replays", "cache_autotuning", "torch_baseline")}
    config.update(gpu="B200", cases=contract["cases"][:1], seeds=contract["seeds"][:1],
                  sources={}, competitors={})
    job = {"config": config, "hashes": {"runner": report.tree_hash(runner)}}
    (snapshot / "job.json").write_text(json.dumps(job))
    data = {"kind": "operator", "config": copy.deepcopy(config),
            "runtime": {**contract["runtime"], "gpu": "NVIDIA B200", "capability": [10, 0]},
            "identities": {n: {"content_hash": v} for n, v in contract["identities"].items()},
            "status": "running", "results": [{"case": config["cases"][0],
                "seed": config["seeds"][0], "arms": {n: {"status": "passed", "samples_ms": [1.] * 120}
                for n in config["cases"][0]["backends"]}}]}
    result = tmp_path / "results" / name
    result.mkdir(parents=True)
    (result / "report.json").write_text(json.dumps(data))
    (tmp_path / "ledger.json").write_text(json.dumps({"jobs": [{"id": name, "status": "failed",
                                                             "error": "retained timeout"}]}))
    return tmp_path, name, contract, job, data


def test_completed_rows_survive_parent_timeout(evidence):
    work, name, contract, _, _ = evidence
    result = report.summarize(work, [name], contract)
    assert len(result["cells"]) == 1 and len(result["missing"]) == len(contract["cases"]) * 6 - 1
    assert not result["admission_failures"] and not result["monotonic_pass"]
    assert result["failures"][0]["execution"]["error"] == "retained timeout"


@pytest.mark.parametrize("field", ["runtime", "identities", "snapshot", "evaluator"])
def test_changed_execution_is_rejected(evidence, field):
    work, name, contract, job, data = evidence
    snapshot = work / "snapshots" / name
    if field == "runtime":
        data["runtime"]["torch"] = "other"
    elif field == "identities":
        data["identities"]["release"]["content_hash"] = "other"
    elif field == "snapshot":
        (snapshot / "runner" / "extra.py").write_text("changed\n")
    else:
        contract["evaluator_files"]["evaluator.py"] = "other"
    assert report.admission_errors(data, job, snapshot, contract)


def test_resume_slice_cannot_replace_full_contract(evidence):
    _, _, contract, _, _ = evidence
    contract["cases"] = contract["cases"][:1]
    with pytest.raises(ValueError, match="complete frozen"):
        report.validate_contract(contract)


def test_incomplete_candidate_samples_remain_unresolved(evidence):
    work, name, contract, _, data = evidence
    data["results"][0]["arms"]["candidate"]["samples_ms"].pop()
    (work / "results" / name / "report.json").write_text(json.dumps(data))
    result = report.summarize(work, [name], contract)
    assert result["admission_failures"] and len(result["missing"]) == len(contract["cases"]) * 6
    assert any(row.get("backend") == "candidate" for row in result["failures"])


def test_duplicate_jobs_cannot_select_the_fastest_retry(evidence):
    work, name, contract, _, _ = evidence
    with pytest.raises(ValueError, match="duplicate measurement"):
        report.summarize(work, [name, name], contract)
