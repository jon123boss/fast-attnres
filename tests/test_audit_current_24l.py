from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import audit_current_24l as auditor

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "current_24l"


def _report(gpu: str) -> dict:
    name = "h100-report.json" if gpu == "H100!" else "b200-report.json"
    return json.loads((EVIDENCE / "raw" / name).read_text(encoding="utf-8"))


def _archive() -> dict[str, bytes]:
    return auditor._archive_files(EVIDENCE / "reproduction" / "performance_source.tar.gz")


def test_current_24l_bundle_recomputes_all_six_unpooled_results():
    result = auditor.audit_bundle(EVIDENCE, ROOT)

    assert result["status"] == "passed"
    assert result["reports"]["H100"]["median_advantage_pct"] == pytest.approx(
        5.195676724310161
    )
    assert result["reports"]["B200"]["median_advantage_pct"] == pytest.approx(
        15.519781490590379
    )
    assert [row["seed"] for row in result["reports"]["H100"]["results"]] == list(
        auditor.SEEDS
    )
    assert [row["n"] for row in result["reports"]["B200"]["results"]] == [120, 120, 120]


def test_report_bytes_are_bound_before_structural_validation(tmp_path: Path):
    report = _report("H100!")
    report["campaign_results"][0]["measurements"]["model_timings"]["statistics"][
        "kernel_rank_1024_over_fla_triton_compile_standard_rank_1024"
    ]["estimate"] = 0.01
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(auditor.Current24LAuditError, match="report SHA-256"):
        auditor.audit_report(
            forged,
            gpu="H100!",
            evidence_dir=EVIDENCE,
            repo_root=ROOT,
        )


def test_derived_statistics_are_recomputed_from_raw_rows():
    report = copy.deepcopy(_report("B200"))
    stats = report["campaign_results"][0]["measurements"]["model_timings"]["statistics"]
    stats["kernel_rank_1024_over_fla_triton_compile_standard_rank_1024"]["ci_high"] = 0.99

    with pytest.raises(auditor.Current24LAuditError, match="statistic ci_high"):
        auditor._audit_report_object(report, gpu="B200", archive=_archive(), repo_root=ROOT)


def test_paired_input_identity_and_timing_boundary_fail_closed():
    report = copy.deepcopy(_report("H100!"))
    timing = report["campaign_results"][0]["measurements"]["model_timings"]
    timing["raw_samples"][1]["input_hash"] = "f" * 64
    with pytest.raises(auditor.Current24LAuditError, match="paired input"):
        auditor._audit_report_object(report, gpu="H100!", archive=_archive(), repo_root=ROOT)

    report = copy.deepcopy(_report("H100!"))
    report["campaign_results"][0]["measurements"]["model_timings"][
        "timed_input_identity"
    ]["tensor_byte_hashing"] = True
    with pytest.raises(auditor.Current24LAuditError, match="timed hashing"):
        auditor._audit_report_object(report, gpu="H100!", archive=_archive(), repo_root=ROOT)


def test_cli_emits_canonical_passed_audit(tmp_path: Path):
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.audit_current_24l",
            "--evidence-dir",
            str(EVIDENCE),
            "--repo",
            str(ROOT),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "passed"
    assert output.read_text(encoding="utf-8") == result.stdout
