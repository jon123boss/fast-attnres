"""CPU/static coverage for the standalone Modal Block codegen launcher."""

import ast
import json
from pathlib import Path

import pytest

from benchmarks import modal_fla_block_codegen_probe as launcher


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "benchmarks" / "modal_fla_block_codegen_probe.py"


def test_launcher_imports_without_gpu_or_modal_runtime():
    tree = ast.parse(LAUNCHER_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    assert "torch" not in imported
    assert "triton" not in imported
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "torch.cuda.Event" not in source
    assert "do_bench" not in source
    assert "time." not in source
    assert "selected_fla_block_codegen_probe.run_probe" in source


def test_launcher_requires_explicit_hardware_cache_and_output():
    with pytest.raises(ValueError, match="gpu"):
        launcher._validate_cli_args("", "H100", "/tmp/cache", "/tmp/out.json")
    with pytest.raises(ValueError, match="hardware"):
        launcher._validate_cli_args("H100!", "B200", "/tmp/cache", "/tmp/out.json")
    with pytest.raises(ValueError, match="hardware"):
        launcher._validate_cli_args("H100!", "", "/tmp/cache", "/tmp/out.json")
    with pytest.raises(ValueError, match="--cache"):
        launcher._validate_cli_args("H100!", "H100", "", "/tmp/out.json")
    with pytest.raises(ValueError, match="--output"):
        launcher._validate_cli_args("B200", "B200", "/tmp/cache", "")


@pytest.mark.parametrize(
    "value,expected", [("H100!", "H100!"), ("H100", "H100!"), ("B200", "B200")]
)
def test_gpu_aliases_are_explicit(value, expected):
    assert launcher._normalize_gpu(value) == expected


@pytest.mark.parametrize("value", ["H100", "B200"])
def test_hardware_scopes_are_explicit(value):
    assert launcher._normalize_hardware(value) == value


def test_gpu_selector_rejects_both_and_unknown():
    with pytest.raises(ValueError, match="one of"):
        launcher._normalize_gpu("both")
    with pytest.raises(ValueError, match="one of"):
        launcher._normalize_gpu("sm90")


def test_relative_cache_is_namespaced_and_must_be_empty(tmp_path):
    transport_cache = {"directories": {"triton": str(tmp_path / "transport")}}
    path = launcher._resolve_probe_cache("run-01", transport_cache)
    assert path == (tmp_path / "transport" / "codegen-probe" / "run-01").resolve()
    assert path.is_dir()
    (path / "stale").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        launcher._resolve_probe_cache("run-01", transport_cache)


def test_cache_traversal_and_nonempty_absolute_cache_are_rejected(tmp_path):
    transport_cache = {"directories": {"triton": str(tmp_path / "transport")}}
    with pytest.raises(ValueError, match="inside"):
        launcher._resolve_probe_cache("../../../../outside", transport_cache)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stale").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        launcher._resolve_probe_cache(str(occupied), transport_cache)


def test_report_writer_emits_one_json_report(tmp_path):
    target = tmp_path / "nested" / "report.json"
    report = {"status": "complete", "probe": launcher.PROBE_MODULE}
    written = launcher._write_report(str(target), report)
    assert written == target.resolve()
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert not target.with_suffix(".json.tmp").exists()


def test_modal_entrypoint_reuses_provenance_and_keeps_probe_untimed():
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    for token in (
        "_source_fingerprint",
        "_prepare_cache",
        "_align_cache_runtime",
        "_commit_cache",
        "@app.function(gpu=",
        "--gpu",
        "--hardware",
        "--cache",
        "--output",
        "hardware=PROBE_HARDWARE[expected]",
    ):
        assert token in source
    assert "run_suite" not in source
    assert "run_source_profile" not in source


def test_fallback_main_stays_blocked_without_modal():
    if launcher._modal is not None:
        pytest.skip("Modal is installed; fallback path is not active")
    with pytest.raises((RuntimeError, ValueError), match="Modal|gpu|hardware"):
        launcher.main("", "", "", "")


def test_main_impl_passes_scope_and_writes_one_report(tmp_path, capsys):
    observed = {}

    def remote(payload):
        observed.update(payload)
        return {"status": "complete", "probe": launcher.PROBE_MODULE}

    output = tmp_path / "report.json"
    assert (
        launcher._main_impl("H100!", "H100", "run-01", str(output), remote)
        == 0
    )
    assert observed == {"cache": "run-01", "hardware": "H100"}
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"
    assert json.loads(capsys.readouterr().out)["probe"] == launcher.PROBE_MODULE


def test_main_impl_writes_failed_report_when_modal_call_raises(tmp_path, capsys):
    def remote(_payload):
        raise RuntimeError("transport unavailable")

    output = tmp_path / "failed.json"
    with pytest.raises(SystemExit, match="report written"):
        launcher._main_impl("B200", "B200", "run-02", str(output), remote)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["requested_gpu"] == "B200"
    assert "transport unavailable" in report["error"]
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
