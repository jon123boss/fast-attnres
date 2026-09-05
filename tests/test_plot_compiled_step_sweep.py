"""Adversarial tests for the strict canonical compiled-step report plotter."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.plot_compiled_step_sweep import (
    SCHEMA,
    SweepPlotError,
    audit_cell,
    load_cells,
    main,
    render_sweep,
    table_rows,
    write_table,
)

_REPORT_ROOT = Path("/private/tmp/compiled-step-autotune-final")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PUBLISHED_SCREEN = _REPO_ROOT / "results" / "adoption" / "compiled_step_screen"
_REPORTS = (
    "compiled-autotune-b200-d1536-s9-v2.json",
    "compiled-autotune-h100-d1536-s9-v2.json",
    "compiled-autotune-b200-full-d1024-s17-v2.json",
    "compiled-autotune-b200-block-d1536-s3-v2.json",
)


@pytest.fixture(autouse=True)
def measured_checkout(historical_release_root, monkeypatch):
    from scripts import compiled_step_sweep
    monkeypatch.setattr(compiled_step_sweep, "PROJECT_ROOT", historical_release_root)


def _drop_worker(payload: dict) -> None:
    payload.pop("worker", None)


def _report(name: str = _REPORTS[0]) -> dict:
    path = _REPORT_ROOT / name
    if not path.is_file():
        pytest.skip(f"genuine worker report is not present: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_published_screen_is_populated_and_matches_its_manifest():
    manifest = json.loads((_PUBLISHED_SCREEN / "manifest.json").read_text(encoding="utf-8"))
    reports = manifest["reports"]
    paths = [_REPO_ROOT / report["path"] for report in reports]
    assert len(paths) == 8
    for report, path in zip(reports, paths, strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report["sha256"]

    cells = load_cells(paths, keep_failures=False)
    rows = table_rows(cells)
    assert len(cells) == 8
    assert len(rows) == 24
    assert Counter(row["status"] for row in rows) == Counter({"OK": 16, "NA": 7, "FAIL": 1})

    for artifact in manifest["derived_artifacts"].values():
        path = _REPO_ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    for gpu in ("h100", "b200"):
        assert "No qualified arms" not in (
            _REPO_ROOT / "docs" / "assets" / f"compiled_step_screen_{gpu}.svg"
        ).read_text(encoding="utf-8")


@pytest.mark.parametrize("mutation", ["delete", "forge"])
def test_published_failure_reason_is_bound_to_traceback(mutation: str):
    path = _PUBLISHED_SCREEN / "raw" / "b200_block_d2048_s9.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    error = payload["benchmark"]["model_timings"]["comparator_failures"][0]["error"]
    if mutation == "delete":
        error.pop("message")
    else:
        error["message"] = "fabricated pass"
    with pytest.raises(SweepPlotError, match="failure"):
        audit_cell(payload, source=path.name)


def test_genuine_worker_reports_bind_outer_model_runtime_schedule_and_catswe():
    cells = [audit_cell(_report(name), source=name) for name in _REPORTS]
    assert all(cell.status == "OK" for cell in cells)
    assert {cell.gpu for cell in cells} == {"H100", "B200"}
    assert any(arm.arm == "catswe" and arm.status == "OK" for cell in cells for arm in cell.arms)
    assert all(cell.warmup == 5 and cell.rounds == 40 for cell in cells)

    rows = table_rows(cells)
    catswe = [row for row in rows if row["competitor"] == "Catswe" and row["status"] == "OK"]
    assert len(catswe) == 1
    assert float(catswe[0]["ratio"]) == pytest.approx(0.870868562, rel=1e-8)


def test_same_geometry_on_h100_and_b200_is_not_a_duplicate(tmp_path: Path):
    paths = []
    for name in (_REPORTS[0], _REPORTS[1]):
        payload = _report(name)
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    cells = load_cells(paths, keep_failures=False)
    assert len(cells) == 2
    assert {cell.gpu for cell in cells} == {"H100", "B200"}
    assert all(cell.status == "OK" for cell in cells)


def test_distinct_block_source_geometry_on_one_gpu_is_not_a_duplicate(tmp_path: Path):
    paths = []
    for name in (_REPORTS[0], _REPORTS[3]):
        payload = _report(name)
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    cells = load_cells(paths, keep_failures=False)
    assert len(cells) == 2
    assert {(cell.event_block_size, cell.smax) for cell in cells} == {(2, 9), (8, 3)}
    assert all(cell.status == "OK" for cell in cells)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p["benchmark"]["model_timings"]["raw_samples"][0].update(input_hash="0" * 64), "input_hash"),
        (lambda p: p["benchmark"]["model_timings"]["statistics"].__getitem__("kernel_rank_1536_over_fla_triton_compile_standard_rank_1536").update(ratio=0.2), "statistics"),
        (lambda p: p["benchmark"]["model_timings"]["graph"]["kernel_rank_1536"]["changed_input_replays"].update(replay_count=1), "replay_input_hashes"),
        (lambda p: p["benchmark"]["model_timings"]["warmup"].__setitem__(0, {"arm": "kernel_rank_1536", "index": 0, "status": "ok", "host_ms": 1.0, "forged": True}), "warmup"),
        (_drop_worker, "worker result fields"),
    ],
)
def test_forged_ratio_hash_graph_warmup_or_wrapper_fails_closed(mutate, match):
    payload = _report()
    mutate(payload)
    with pytest.raises(SweepPlotError, match=match):
        audit_cell(payload)


def test_minimal_direct_shape_with_matching_schema_is_rejected():
    payload = {"schema": SCHEMA, "status": "complete", "gpu": "H100", "cell": {}, "benchmark": {}}
    with pytest.raises(SweepPlotError, match="worker result fields"):
        audit_cell(payload)


def test_actual_failure_without_benchmark_is_retained_without_numbers(tmp_path: Path):
    payload = _report()
    payload["status"] = "failed"
    payload["benchmark"] = None
    payload["worker"] = None
    payload["runtime_preflight"] = {"status": "not_passed"}
    payload["failure"] = {"type": "SweepError", "message": "runtime preflight failed"}
    path = tmp_path / "unavailable.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = table_rows(load_cells([path]))
    assert rows and all(row["status"] == "FAIL" for row in rows)
    assert all(row["ratio"] == "" for row in rows)


def test_outputs_are_deterministic_and_long_form(tmp_path: Path):
    cells = [audit_cell(_report(name), source=name) for name in _REPORTS]
    first, second = tmp_path / "first", tmp_path / "second"
    first_figures = render_sweep(cells, first)
    second_figures = render_sweep(cells, second)
    assert len(first_figures) == len(second_figures) == 4
    for first_figure, second_figure in zip(first_figures, second_figures, strict=True):
        assert first_figure.read_bytes() == second_figure.read_bytes()
    svg_text = "\n".join(
        figure.read_text(encoding="utf-8")
        for figure in first_figures
        if figure.suffix == ".svg"
    )
    assert "H100" in svg_text and "B200" in svg_text
    assert "Catswe" in svg_text
    assert "advantage" in svg_text.lower()
    assert "FAIL/NA" in svg_text
    assert "native tl.arange tile" not in svg_text
    rows = table_rows(cells)
    csv_path, md_path = write_table(rows, first / "rows.csv", first / "rows.md")
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("gpu,phase,mode")
    assert "Catswe" in md_path.read_text(encoding="utf-8")


def test_cli_rejects_invalid_report_before_writing_any_artifact(tmp_path: Path):
    payload = _report()
    stats = payload["benchmark"]["model_timings"]["statistics"]
    stats["kernel_rank_1536_over_fla_triton_compile_standard_rank_1536"]["ratio"] = 0.2
    report = tmp_path / "forged.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "published"

    assert main([str(report), "--output-dir", str(output)]) == 2
    assert not output.exists()
