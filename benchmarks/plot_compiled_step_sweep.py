"""Plot and tabulate audited ``compiled_step_sweep`` cell results.

The input is the canonical cell envelope written by
``scripts.compiled_step_sweep``.  This module performs the small, offline
audit needed by the figure: it checks the complete model-step status, binds
the exact sweep geometry and source schedule, validates every measured row,
and recomputes the paired mean-of-ratios and simultaneous bootstrap interval.
Failed, incomplete, and unavailable arms remain visible as ``FAIL``/``NA``
rows but never contribute a number.

It deliberately has no knowledge of any other benchmark report format.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # allow ``python benchmarks/plot_compiled_step_sweep.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import compiled_step_sweep as sweep

SCHEMA = sweep.SCHEMA
SUPPORTED_GPUS = ("H100", "B200")
SCREEN = (5, 40)
RELEASE = (10, 120)
PNG_DPI = 160
DEFAULT_OUTPUT_DIR = Path("docs/assets")
DEFAULT_SVG_NAME = "compiled_step_sweep.svg"
DEFAULT_PNG_NAME = "compiled_step_sweep.png"
DEFAULT_CSV_NAME = "compiled_step_sweep.csv"
DEFAULT_MD_NAME = "compiled_step_sweep.md"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_SCREEN_MANIFEST = (
    PROJECT_ROOT / "results" / "adoption" / "compiled_step_screen" / "manifest.json"
)

ARM_LABELS = {"attnres": "Fast-AttnRes", "fla": "FLA Triton", "liger": "Liger", "catswe": "Catswe"}
ARM_COLORS = {
    "attnres": "#0072B2",  # Okabe-Ito blue
    "fla": "#D55E00",      # vermillion
    "liger": "#009E73",    # green
    "catswe": "#CC79A7",   # purple
    "fail": "#B22222",
    "na": "#66727A",
}
DARK_ARM_COLORS = {
    "attnres": "#2AB7F6",
    "fla": "#FF8B2C",
    "liger": "#42D6A4",
    "catswe": "#D987B3",
    "fail": "#FF6B6B",
    "na": "#A8B6C2",
}
MODE_MARKERS = {"full": "o", "block": "s"}
BLOCK_COLORS = {8: "#56B4E9", 4: "#E69F00", 2: "#009E73", 1: "#CC79A7"}
TABLE_COLUMNS = (
    "gpu", "phase", "mode", "event_block_size", "smax", "width_d", "rank_r",
    "rank_relation", "equation", "competitor", "status", "reason", "warmup",
    "rounds", "n", "attnres_mean_ms", "competitor_mean_ms", "ratio", "ci_low",
    "ci_high", "advantage_pct", "source",
)
_HEX = re.compile(r"^[0-9a-fA-F]+$")
_HEX64_LOWER = re.compile(r"^[0-9a-f]{64}$")
_HEX40_LOWER = re.compile(r"^[0-9a-f]{40}$")

# ``run_worker`` is deliberately a transport boundary.  A local launcher
# envelope, a hand-written benchmark report, or a historical projection is
# not a cell result and must never be silently unwrapped here.
_RESULT_FIELDS = frozenset({
    "schema", "status", "gpu", "cell", "config", "project_provenance",
    "roots", "run_parameters", "routes", "eligibility", "timing_contract",
    "runtime_preflight", "provenance", "report_identity", "worker",
    "benchmark", "failure",
})
_BENCHMARK_FIELDS = frozenset({
    "status", "config", "contract", "coverage", "environment", "device",
    "fla_checkout", "comparators", "correctness", "operator_timings",
    "model_timings", "failures", "protocol", "comparators_enabled",
    "source_hashes", "hashes",
})
_CORE_MODEL_FIELDS = frozenset({
    "status", "config", "effective_variant", "ranks", "qualification",
    "comparator_qualification", "comparator_failures", "state_protocol",
    "compile_backend_metadata", "architecture_comparisons",
    "qualification_staging", "execution_schedules", "compile",
    "compiled_loss", "optimizer", "complete_step_qualification",
    "pre_timing_gate", "training_step", "timing_method", "frozen_baseline",
    "graph", "canonical_training_step", "reference_timing", "include_fla_model",
    "pairwise", "timing_boundary", "accumulation", "warmup",
    "requested_warmup", "effective_warmup", "requested_rounds",
    "graph_counters", "timed_graph_counters", "changed_inputs",
    "timed_input_identity", "timed_numerical_checks", "raw_samples",
    "statistics", "model_profile", "failures", "include_liger_model",
    "include_catswe_model", "model_comparator_scope",
    "model_comparator_metadata",
})
_RAW_ROW_FIELDS = frozenset({
    "arm", "rank", "backend", "sample_index", "order_index", "input_hash",
    "ms", "status", "timing_method", "replay_count",
})
_RAW_FAILED_ROW_FIELDS = frozenset({
    "arm", "rank", "backend", "sample_index", "order_index", "input_hash",
    "ms", "status", "error",
})


class SweepPlotError(ValueError):
    """Raised when a canonical cell cannot support an audited numeric row."""


@dataclass(frozen=True)
class ArmResult:
    """One comparator row for one cell."""

    arm: str
    status: str
    reason: str
    mean_ms: float | None = None
    ratio: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None
    reported_key: str | None = None


@dataclass(frozen=True)
class CellResult:
    """Validated metadata and comparator rows from one canonical cell."""

    source: str
    gpu: str
    phase: str
    mode: str
    event_block_size: int | None
    smax: int
    width: int
    rank: int
    rank_relation: str
    warmup: int
    rounds: int
    status: str
    reason: str
    arms: tuple[ArmResult, ...]

    @property
    def label(self) -> str:
        block = "" if self.mode == "full" else f" bs={self.event_block_size} Smax={self.smax}"
        return f"{self.mode.title()}{block} · D={self.width} · {self.rank_relation}"


def _error(message: str) -> None:
    raise SweepPlotError(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(f"{path} must be an object")
    return value


def _int(value: Any, path: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        _error(f"{path} must be {'a positive ' if positive else ''}integer")
    return int(value)


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        _error(f"{path} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _nonnegative_map(value: Any, path: str) -> None:
    """Validate the producer's recursively nested max-error maps.

    Changed-input graph evidence records one scalar per parameter for model
    and gradient deltas, and a scalar map inside each optimizer-state entry.
    Requiring every leaf to be a finite nonnegative number prevents a forged
    string/boolean/signed summary from being accepted as numerical evidence.
    """

    mapping = _mapping(value, path)
    if not mapping:
        _error(f"{path} must be a nonempty numeric map")
    for key, item in mapping.items():
        if not isinstance(key, str) or not key:
            _error(f"{path} contains a malformed parameter key")
        if isinstance(item, Mapping):
            _nonnegative_map(item, f"{path}.{key}")
        else:
            _number(item, f"{path}.{key}")


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _read_json(path: Path | str) -> Mapping[str, Any]:
    target = Path(path).expanduser().resolve()
    try:
        value = json.loads(
            target.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant {item}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SweepPlotError(f"cannot read canonical cell {target}: {exc}") from exc
    return _mapping(value, str(target))


def _hex(value: Any, pattern: re.Pattern[str], path: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _error(f"{path} must be a lowercase hexadecimal digest")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _error(f"value is not canonical JSON: {exc}")
    return _sha256_bytes(encoded)


def _canonical_frozen_hashes() -> dict[str, str]:
    """Read the checked-in frozen file map used by the producer."""

    path = sweep.PROJECT_ROOT / sweep.FROZEN_MANIFEST_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _error(f"cannot read current frozen manifest: {exc}")
    if not isinstance(value, Mapping) or not value:
        _error("current frozen manifest is not a nonempty object")
    if any(not isinstance(key, str) or not isinstance(digest, str) or _HEX64_LOWER.fullmatch(digest) is None for key, digest in value.items()):
        _error("current frozen manifest contains malformed hashes")
    return {str(key): str(digest) for key, digest in value.items()}


def _report_bound_frozen_hashes(project: Mapping[str, Any]) -> dict[str, str]:
    """Load the frozen map from the exact revision attested by a report."""

    expected = str(_mapping(project.get("frozen_manifest"), "project_provenance.frozen_manifest")["sha256"])
    current_path = sweep.PROJECT_ROOT / sweep.FROZEN_MANIFEST_PATH
    try:
        raw = current_path.read_bytes()
    except OSError as exc:
        _error(f"cannot read current frozen manifest: {exc}")
    if _sha256_bytes(raw) != expected:
        if _matches_published_measurement_binding(project):
            try:
                manifest = json.loads(PUBLISHED_SCREEN_MANIFEST.read_text(encoding="utf-8"))
                artifact = _mapping(
                    _mapping(manifest.get("derived_artifacts"), "published derived_artifacts").get(
                        "measurement_frozen_manifest"
                    ),
                    "published measurement_frozen_manifest",
                )
                relative = Path(str(artifact.get("path")))
                if relative.is_absolute() or ".." in relative.parts:
                    _error("published measurement frozen manifest path is unsafe")
                raw = (PROJECT_ROOT / relative).read_bytes()
                if artifact.get("sha256") != expected:
                    _error("published measurement frozen manifest digest is not report-bound")
            except (OSError, UnicodeError, json.JSONDecodeError, SweepPlotError) as exc:
                _error(f"cannot load published measurement frozen manifest: {exc}")
        else:
            try:
                raw = subprocess.run(
                    ["git", "show", f"{project['revision']}:{sweep.FROZEN_MANIFEST_PATH}"],
                    cwd=sweep.PROJECT_ROOT,
                    capture_output=True,
                    check=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as exc:
                _error(f"cannot load report-bound frozen manifest: {exc}")
    if _sha256_bytes(raw) != expected:
        _error("report-bound frozen manifest bytes do not match the attested digest")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        _error(f"report-bound frozen manifest is invalid JSON: {exc}")
    if not isinstance(value, Mapping) or not value:
        _error("report-bound frozen manifest is not a nonempty object")
    if any(not isinstance(key, str) or not isinstance(digest, str) or _HEX64_LOWER.fullmatch(digest) is None for key, digest in value.items()):
        _error("report-bound frozen manifest contains malformed hashes")
    return {str(key): str(digest) for key, digest in value.items()}


def _validate_exact_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact producer cell, rather than a geometry lookalike."""

    try:
        value = sweep._validate_cell(dict(cell))
    except Exception as exc:  # producer uses SweepError, but keep the loader ABI stable
        _error(f"cell is not the exact sealed producer cell: {exc}")
    return dict(value)


def _matches_published_measurement_binding(project: Mapping[str, Any]) -> bool:
    """Accept the checked-in measurement binding without prior Git objects."""

    try:
        manifest = json.loads(PUBLISHED_SCREEN_MANIFEST.read_text(encoding="utf-8"))
        expected = _mapping(manifest.get("project_provenance"), "published project_provenance")
    except (OSError, UnicodeError, json.JSONDecodeError, SweepPlotError):
        return False
    keys = ("revision", "tree", "clean", "clean_required", "frozen_manifest", "kernel_sha256")
    return all(key in expected and _same(project.get(key), expected[key]) for key in keys)


def _validate_project_provenance(project_value: Any, *, catswe_required: bool) -> dict[str, Any]:
    """Validate the report's immutable project/frozen/kernel/vendor binding.

    The worker's checkout revision is intentionally compared only to the
    nested attestation and digest contracts.  A plotter may run from a later
    local commit than the remote worker, while the frozen manifest and kernel
    bytes remain the same project contract.
    """

    project = _mapping(project_value, "project_provenance")
    base = {"revision", "tree", "clean", "clean_required", "frozen_manifest", "kernel_sha256"}
    expected_fields = base | ({"catswe"} if catswe_required else set())
    if set(project) != expected_fields:
        _error("project_provenance fields are not the exact worker contract")
    _hex(project.get("revision"), _HEX40_LOWER, "project_provenance.revision")
    _hex(project.get("tree"), _HEX40_LOWER, "project_provenance.tree")
    if project.get("clean") is not True or project.get("clean_required") is not True:
        _error("project provenance must attest a clean required checkout")
    frozen = _mapping(project.get("frozen_manifest"), "project_provenance.frozen_manifest")
    if set(frozen) != {"path", "sha256"} or frozen.get("path") != sweep.FROZEN_MANIFEST_PATH:
        _error("project frozen manifest path is not canonical")
    frozen_digest = _hex(frozen.get("sha256"), _HEX64_LOWER, "project_provenance.frozen_manifest.sha256")
    try:
        current_frozen_digest = _sha256_bytes((sweep.PROJECT_ROOT / sweep.FROZEN_MANIFEST_PATH).read_bytes())
    except OSError as exc:
        _error(f"cannot read current frozen manifest bytes: {exc}")
    if frozen_digest != current_frozen_digest and not _matches_published_measurement_binding(project):
        revision = str(project["revision"])
        tree = str(project["tree"])
        try:
            resolved_tree = subprocess.run(
                ["git", "rev-parse", f"{revision}^{{tree}}"],
                cwd=sweep.PROJECT_ROOT,
                capture_output=True,
                check=True,
            ).stdout.decode("ascii").strip()
            historical = subprocess.run(
                ["git", "show", f"{revision}:{sweep.FROZEN_MANIFEST_PATH}"],
                cwd=sweep.PROJECT_ROOT,
                capture_output=True,
                check=True,
            ).stdout
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
                cwd=sweep.PROJECT_ROOT,
                capture_output=True,
                check=True,
            )
        except (OSError, UnicodeError, subprocess.CalledProcessError) as exc:
            _error(f"cannot verify report-bound historical frozen manifest: {exc}")
        if resolved_tree != tree or _sha256_bytes(historical) != frozen_digest:
            _error("project frozen manifest digest differs from both current and report revision")
    kernels = _mapping(project.get("kernel_sha256"), "project_provenance.kernel_sha256")
    if set(kernels) != set(sweep.KERNEL_PATHS):
        _error("project kernel provenance paths are not exact")
    for relative in sweep.KERNEL_PATHS:
        digest = _hex(kernels.get(relative), _HEX64_LOWER, f"project_provenance.kernel_sha256.{relative}")
        try:
            current = sweep._sha256_file(sweep.PROJECT_ROOT / relative, f"current kernel {relative}")
        except Exception as exc:
            _error(f"cannot hash current kernel {relative}: {exc}")
        if digest != current:
            _error(f"project kernel digest differs for {relative}")
    if catswe_required:
        try:
            expected_catswe = sweep._catswe_provenance_contract()
        except Exception as exc:
            _error(f"cannot load pinned Catswe vendor contract: {exc}")
        if not _same(project.get("catswe"), expected_catswe):
            _error("project Catswe vendor provenance differs from the pinned contract")
    return dict(project)


def _validate_worker_binding(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every field copied from the producer payload into a result."""

    cell = _validate_exact_cell(_mapping(payload.get("cell"), "cell"))
    gpu = payload.get("gpu")
    if gpu not in SUPPORTED_GPUS:
        _error(f"result.gpu must be H100 or B200, got {gpu!r}")
    config = _mapping(payload.get("config"), "config")
    roots = _mapping(payload.get("roots"), "roots")
    if set(roots) != {"remote_repo", "remote_fla_root", "remote_liger_root", "remote_catswe_root", "triton_cache_dir"}:
        _error("result roots are not the exact worker root binding")
    if any(not isinstance(value, str) or not value for value in roots.values()):
        _error("result roots must be nonempty strings")
    params = _mapping(payload.get("run_parameters"), "run_parameters")
    expected_params = {"seed", "warmup", "rounds", "bootstrap_samples", "batch", "sequence", "vocab"}
    if set(params) != expected_params:
        _error("result run_parameters are not the exact worker parameter binding")
    if type(params["seed"]) is not int or params["seed"] <= 0:
        _error("result run_parameters.seed must be a positive integer")
    if type(params["warmup"]) is not int or params["warmup"] < 0:
        _error("result run_parameters.warmup must be a nonnegative integer")
    if type(params["rounds"]) is not int or params["rounds"] <= 0:
        _error("result run_parameters.rounds must be a positive integer")
    if type(params["bootstrap_samples"]) is not int or params["bootstrap_samples"] <= 0:
        _error("result run_parameters.bootstrap_samples must be a positive integer")
    for key, expected in (("batch", sweep.BATCH), ("sequence", sweep.SEQUENCE), ("vocab", sweep.VOCAB)):
        if type(params[key]) is not int or params[key] != expected:
            _error(f"result run_parameters.{key} differs from the fixed worker profile")
    try:
        expected_config = sweep.make_worker_config(
            cell,
            seed=params["seed"], warmup=params["warmup"], rounds=params["rounds"],
            bootstrap_samples=params["bootstrap_samples"], batch=params["batch"],
            sequence=params["sequence"], vocab=params["vocab"],
            remote_repo=roots["remote_repo"], remote_fla_root=roots["remote_fla_root"],
            remote_liger_root=roots["remote_liger_root"], remote_catswe_root=roots["remote_catswe_root"],
        )
    except Exception as exc:
        _error(f"cannot derive expected worker config: {exc}")
    if not _same(config, expected_config):
        _error("result config differs from the exact payload-bound worker config")
    project = _validate_project_provenance(
        payload.get("project_provenance"),
        catswe_required=bool(config["include_catswe_model"]),
    )
    if not _same(payload.get("cell"), config.get("sweep_cell")):
        _error("result cell and config.sweep_cell differ")
    binding = {
        "config": dict(config),
        "project_provenance": project,
        "roots": dict(roots),
        "run_parameters": dict(params),
        "routes": sweep._worker_result_routes(config),
        "eligibility": dict(cell["competitors"]),
        "timing_contract": dict(config["sweep_timing_contract"]),
    }
    for key in ("routes", "eligibility", "timing_contract"):
        actual = _mapping(payload.get(key), key)
        if not _same(actual, binding[key]):
            _error(f"result {key} differs from the exact worker binding")
    return cell, binding


def _validate_runtime(runtime: Any, gpu: str) -> dict[str, Any]:
    try:
        value = sweep._validate_worker_runtime(runtime, gpu, allow_not_passed=False)
    except Exception as exc:
        _error(f"runtime_preflight is not the exact passed runtime contract: {exc}")
    return dict(value)


def _validate_result_provenance(actual: Any, project: Mapping[str, Any], *, catswe_required: bool) -> None:
    value = _mapping(actual, "provenance")
    expected_fields = {"project", "catswe"} if catswe_required else {"project"}
    if set(value) != expected_fields:
        _error("result provenance fields are not exact")
    # The producer's project attestation deliberately omits the optional
    # Catswe contract; that vendor is attested in its own sibling object.
    base = {key: value for key, value in project.items() if key != "catswe"}
    expected_project = {"status": "verified", **base}
    if not _same(value.get("project"), expected_project):
        _error("result project attestation differs from project_provenance")
    if catswe_required:
        catswe = _mapping(value.get("catswe"), "provenance.catswe")
        expected = _mapping(project.get("catswe"), "project_provenance.catswe")
        fields = {
            "status", "transport", "revision", "tree", "clean", "origin", "license",
            "license_file", "license_sha256", "source_hashes", "vendor_file_sha256",
        }
        if set(catswe) != fields:
            _error("result Catswe attestation fields are not exact")
        if catswe.get("status") != "verified" or catswe.get("transport") not in {"git_checkout", "host_git_preflight+remote_bytes"}:
            _error("result Catswe attestation is not verified")
        for key in ("revision", "tree", "license", "license_file", "license_sha256", "source_hashes", "vendor_file_sha256"):
            if not _same(catswe.get(key), expected.get(key)):
                _error(f"result Catswe attestation {key} differs from pinned vendor provenance")
        if catswe.get("clean") is not True:
            _error("result Catswe attestation is not clean")
        # Git and remote byte transports may spell an origin with a trailing
        # .git; the producer accepts that normalization, so mirror it here.
        if sweep._normalise_origin(catswe.get("origin")) != sweep._normalise_origin(expected.get("origin")):
            _error("result Catswe attestation origin differs")


def _validate_worker_identity(payload: Mapping[str, Any], benchmark: Mapping[str, Any], *, complete: bool) -> None:
    worker = _mapping(payload.get("worker"), "worker")
    fields = {"run_id", "started_unix_s", "finished_unix_s", "elapsed_s", "timed_tensor_hashing", "timed_input_copy", "timed_qualification"}
    if set(worker) != fields:
        _error("worker identity fields are not exact")
    if not isinstance(worker["run_id"], str) or not worker["run_id"]:
        _error("worker.run_id is missing")
    for key in ("started_unix_s", "finished_unix_s", "elapsed_s"):
        if isinstance(worker[key], bool) or not isinstance(worker[key], (int, float)) or not math.isfinite(float(worker[key])):
            _error(f"worker.{key} is malformed")
    if worker["finished_unix_s"] < worker["started_unix_s"] or worker["elapsed_s"] < 0:
        _error("worker timing is malformed")
    for key in ("timed_tensor_hashing", "timed_input_copy", "timed_qualification"):
        if worker[key] is not False:
            _error(f"worker.{key} must be false")
    try:
        expected = sweep._report_identity(benchmark)
    except Exception as exc:
        _error(f"cannot derive report identity: {exc}")
    if not _same(payload.get("report_identity"), expected):
        _error("report_identity is not the producer-derived identity")


def _validate_environment_and_hashes(benchmark: Mapping[str, Any], project: Mapping[str, Any], runtime: Mapping[str, Any], gpu: str, config: Mapping[str, Any]) -> None:
    environment = _mapping(benchmark.get("environment"), "benchmark.environment")
    if set(environment) != {"python", "platform", "machine", "hostname", "torch", "cuda_runtime", "triton", "git", "env"}:
        _error("benchmark.environment fields are not exact")
    for key in ("python", "platform", "machine", "hostname"):
        if not isinstance(environment[key], str) or not environment[key]:
            _error(f"benchmark.environment.{key} is missing")
    if environment["torch"] != runtime["torch"] or environment["cuda_runtime"] != runtime["cuda"] or environment["triton"] != runtime["triton"]:
        _error("benchmark environment runtime differs from runtime_preflight")
    git = _mapping(environment.get("git"), "benchmark.environment.git")
    if set(git) != {"revision", "branch", "dirty"} or git.get("revision") != project.get("revision") or git.get("dirty") is not False:
        _error("benchmark environment git provenance differs from the worker project")
    if not isinstance(git.get("branch"), str) or not git["branch"]:
        _error("benchmark.environment.git.branch is missing")
    env = _mapping(environment.get("env"), "benchmark.environment.env")
    if set(env) - {"CUDA_VISIBLE_DEVICES", "FLA_ATTNRES_GLUON"} or any(not isinstance(value, str) for value in env.values()):
        _error("benchmark environment variable provenance is malformed")
    device = _mapping(benchmark.get("device"), "benchmark.device")
    expected_device_fields = {"requested", "type", "available", "index", "count", "name", "capability", "total_memory", "multi_processor_count"}
    if set(device) != expected_device_fields:
        _error("benchmark.device fields are not exact")
    if device.get("requested") != config.get("device") or device.get("type") != "cuda" or device.get("available") is not True or device.get("index") != 0 or device.get("count") != 1:
        _error("benchmark device selection is not the canonical single GPU")
    if device.get("name") != runtime.get("name") or device.get("capability") != runtime.get("compute_capability") or device.get("total_memory") != runtime.get("total_memory"):
        _error("benchmark device differs from runtime_preflight")
    if type(device.get("multi_processor_count")) is not int or device["multi_processor_count"] <= 0:
        _error("benchmark device multiprocessor count is malformed")
    source = _mapping(benchmark.get("source_hashes"), "benchmark.source_hashes")
    if set(source) != {"frozen", "project", "vendor", "software_hash"}:
        _error("benchmark source hash fields are not exact")
    frozen_hashes = _report_bound_frozen_hashes(project)
    if not _same(source.get("frozen"), frozen_hashes):
        _error("benchmark frozen hashes differ from the checked-in frozen contract")
    project_hashes = _mapping(source.get("project"), "benchmark.source_hashes.project")
    if not project_hashes or any(not isinstance(key, str) or not _HEX64_LOWER.fullmatch(str(value)) for key, value in project_hashes.items()):
        _error("benchmark project source hashes are malformed")
    for relative in sweep.KERNEL_PATHS:
        if project_hashes.get(relative) != project["kernel_sha256"][relative]:
            _error(f"benchmark project source hash differs for {relative}")
    vendor = _mapping(source.get("vendor"), "benchmark.source_hashes.vendor")
    if set(vendor) != {"path", "git_revision", "dispatch_environment"} or vendor.get("path") is not None or vendor.get("git_revision") is not None:
        _error("worker FLA vendor discovery must remain the disabled operator route")
    if vendor.get("dispatch_environment") != env.get("FLA_ATTNRES_GLUON"):
        _error("benchmark vendor environment provenance differs")
    software_input = {"frozen": dict(source["frozen"]), "project": dict(project_hashes), "vendor": dict(vendor)}
    if source.get("software_hash") != _sha256_bytes(json.dumps(software_input, sort_keys=True).encode("utf-8")):
        _error("benchmark software hash does not bind frozen/project/vendor source hashes")
    hashes = _mapping(benchmark.get("hashes"), "benchmark.hashes")
    if set(hashes) != {"protocol", "software", "hardware"}:
        _error("benchmark hashes fields are not exact")
    if not _same(hashes.get("protocol"), source["frozen"]) or hashes.get("software") != source["software_hash"]:
        _error("benchmark aggregate hashes are not bound to source hashes")
    if hashes.get("hardware") != _sha256_bytes(json.dumps(dict(device), sort_keys=True).encode("utf-8")):
        _error("benchmark hardware hash does not bind the audited device")


def _geometry(cell: Mapping[str, Any]) -> tuple[str, int | None, int, int, int, str]:
    value = _validate_exact_cell(cell)
    mode = str(value["mode"])
    event = value["event_block_size"]
    width = int(value["width"])
    rank = int(value["rank"])
    relation = str(value["rank_relation"])
    return mode, event, int(value["max_read_sources"]), width, rank, relation


def _phase(model: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[str, int, int]:
    warmup = model.get("requested_warmup", config.get("model_warmup"))
    rounds = model.get("requested_rounds", config.get("model_rounds"))
    warmup = _int(warmup, "model_timings.requested_warmup")
    rounds = _int(rounds, "model_timings.requested_rounds", positive=True)
    pair = (warmup, rounds)
    if pair == SCREEN:
        return "screen", warmup, rounds
    if pair == RELEASE:
        return "release", warmup, rounds
    _error("warmup/rounds must be 5/40 (screen) or 10/120 (release)")


def _canonical_arm_names(raw: Sequence[Mapping[str, Any]], rank: int, width: int) -> dict[str, str]:
    """Bind raw rows to the four names emitted by ``run_worker``.

    Backend and rank fields are checked separately below.  Keeping this map
    literal is intentional: accepting a wrapper or a similarly named direct
    benchmark arm would make a forged ratio look like a worker result.
    """

    allowed = {
        f"kernel_rank_{rank}": "attnres",
        f"fla_triton_compile_standard_rank_{width}": "fla",
        f"liger_rank_{width}": "liger",
        f"catswe_phase1_model_rank_{rank}": "catswe",
    }
    names: dict[str, str] = {}
    for index, row in enumerate(raw):
        name = row.get("arm")
        if not isinstance(name, str) or name not in allowed:
            _error(f"raw_samples[{index}].arm is not a canonical worker arm")
        kind = allowed[name]
        if kind in names and names[kind] != name:
            _error(f"multiple raw arm names map to {kind}")
        names[kind] = name
    if "attnres" not in names or "fla" not in names:
        _error("a complete worker cell needs candidate and standard FLA raw arms")
    return names


def _balanced_schedule(arms: Sequence[str], rounds: int, seed: int) -> list[list[str]]:
    first = list(arms)
    rng = random.Random(int(seed) + 771)
    # ``_model_timings`` shuffles the warmup order with this same RNG before
    # calling ``_balanced_orders``.  Consume that draw so the persisted raw
    # rows are checked against the producer's actual ABBA schedule.
    warmup_order = list(arms)
    rng.shuffle(warmup_order)
    rng.shuffle(first)
    second = list(reversed(first))
    if len(first) < 3:
        return [list(first if index % 2 == 0 else second) for index in range(rounds)]
    count = len(first)
    result: list[list[str]] = []
    for index in range(rounds):
        offset = (index // 2) % count
        order = first[offset:] + first[:offset]
        if index % 2:
            order.reverse()
        result.append(order)
    return result


def _logical_input_hash(seed: int, sample: int, model_config: Mapping[str, Any]) -> str:
    value = {
        "protocol": "logical_model_sample_v1",
        "seed": int(seed),
        "sample_index": int(sample),
        "batch": int(model_config["batch"]),
        "sequence": int(model_config["sequence"]),
        "vocab": int(model_config["vocab"]),
    }
    return _sha256_json(value)


def _raw_vectors(
    raw: Sequence[Mapping[str, Any]],
    names: Mapping[str, str],
    rounds: int,
    *,
    seed: int | None = None,
    model_config: Mapping[str, Any] | None = None,
    optional_failure_names: set[str] | None = None,
) -> dict[str, list[float]]:
    """Validate the producer's exact paired schedule and return OK vectors.

    Optional comparator failures are represented by one ``failed`` row and
    subsequent ``skipped_due_to_failure`` rows.  They remain auditable, but
    only complete OK vectors can enter a ratio estimate.
    """

    active = [names[k] for k in ("attnres", "liger", "catswe", "fla") if k in names]
    optional_failure_names = set(optional_failure_names or ())
    by_arm: dict[str, list[Mapping[str, Any]]] = {name: [] for name in active}
    for index, row in enumerate(raw):
        name = row.get("arm")
        if name not in by_arm:
            _error(f"raw_samples[{index}].arm is outside the canonical active arms")
        status = row.get("status")
        if status == "ok":
            if set(row) != _RAW_ROW_FIELDS:
                _error(f"raw_samples[{index}] has noncanonical successful row fields")
            if row.get("timing_method") != "cuda_graph" or row.get("replay_count") != 1:
                _error(f"raw_samples[{index}] is missing the CUDA Graph replay contract")
            _int(row.get("order_index"), f"raw_samples[{index}].order_index")
            if int(row["order_index"]) < 0:
                _error(f"raw_samples[{index}].order_index is negative")
            _number(row.get("ms"), f"raw_samples[{index}].ms", positive=True)
        elif status == "failed":
            if names.get("liger") != name and names.get("catswe") != name:
                _error(f"core raw arm {name!r} failed; core timing is not complete")
            if set(row) != _RAW_FAILED_ROW_FIELDS or row.get("ms") is not None:
                _error(f"raw_samples[{index}] has noncanonical failed row fields")
            _int(row.get("order_index"), f"raw_samples[{index}].order_index")
            if int(row["order_index"]) < 0:
                _error(f"raw_samples[{index}].order_index is negative")
            error = _mapping(row.get("error"), f"raw_samples[{index}].error")
            if set(error) != {"type", "message", "traceback"} or any(not isinstance(error[key], str) or not error[key] for key in error):
                _error(f"raw_samples[{index}].error is not producer exception evidence")
        elif status == "skipped_due_to_failure":
            if names.get("liger") != name and names.get("catswe") != name:
                _error(f"core raw arm {name!r} was skipped")
            if set(row) != _RAW_ROW_FIELDS - {"timing_method", "replay_count"} or row.get("order_index") is not None or row.get("ms") is not None:
                _error(f"raw_samples[{index}] has noncanonical skipped row fields")
        else:
            _error(f"raw_samples[{index}].status is not a worker timing status")
        sample = _int(row.get("sample_index"), f"raw_samples[{index}].sample_index")
        if not 0 <= sample < rounds:
            _error(f"raw_samples[{index}].sample_index is outside 0..rounds-1")
        digest = _hex(row.get("input_hash"), _HEX64_LOWER, f"raw_samples[{index}].input_hash")
        if seed is not None and model_config is not None and digest != _logical_input_hash(seed, sample, model_config):
            _error(f"raw_samples[{index}].input_hash is not the canonical logical sample identity")
        by_arm[name].append(row)
    if len(raw) != len(active) * rounds:
        _error(f"raw_samples must contain exactly {len(active)} arms × {rounds} rounds")
    for name, rows in by_arm.items():
        if len(rows) != rounds:
            _error(f"raw arm {name} has {len(rows)} rows; expected {rounds}")
        if [int(row["sample_index"]) for row in rows] != list(range(rounds)):
            _error(f"raw arm {name} does not contain samples in exact order")

    # Exact backend/rank identities are part of the worker ABI.  In
    # particular, a direct FLA arm or a differently ranked kernel cannot be
    # relabelled as a canonical result.
    expected_fields = {
        names["attnres"]: ("kernel", None),
        names["fla"]: ("fla_triton_compile", None),
    }
    if "liger" in names:
        expected_fields[names["liger"]] = ("liger", None)
    if "catswe" in names:
        expected_fields[names["catswe"]] = ("catswe_phase1", None)
    for kind, name in (("attnres", names["attnres"]), ("fla", names["fla"]), ("liger", names.get("liger")), ("catswe", names.get("catswe"))):
        if name is None:
            continue
        for row in by_arm[name]:
            expected_backend = expected_fields[name][0]
            if row.get("backend") != expected_backend:
                _error(f"raw arm {name} backend does not match its canonical route")
            expected_rank = int(name.rsplit("_", 1)[1])
            if row.get("rank") != expected_rank:
                _error(f"raw arm {name} rank does not match its canonical name")

    # Identify the first optional failure.  The producer removes that arm
    # beginning with the following sample, while rows at the failure sample
    # still follow the full scheduled order.
    failed_at: dict[str, int] = {}
    for name in active:
        failures = [int(row["sample_index"]) for row in by_arm[name] if row.get("status") == "failed"]
        skipped = [int(row["sample_index"]) for row in by_arm[name] if row.get("status") == "skipped_due_to_failure"]
        if failures:
            if len(failures) != 1 or skipped != list(range(failures[0] + 1, rounds)):
                _error(f"optional raw arm {name} has an invalid failure/skip transition")
            failed_at[name] = failures[0]
        elif skipped:
            # A comparator may fail before the first timed round (discovery,
            # compile, warmup, or graph qualification).  The producer then
            # emits one skipped row for every sample and records the failure
            # in comparator_failures rather than manufacturing a failed timing
            # row.  Treat that transition as occurring before sample zero,
            # while still requiring the explicit producer failure evidence.
            if name not in optional_failure_names or skipped != list(range(rounds)):
                _error(f"optional raw arm {name} was skipped without a failure")
            failed_at[name] = -1
    if any(name in failed_at for name in (names["attnres"], names["fla"])):
        _error("core raw arms cannot fail or be skipped")

    # Check row order, order_index positions, and exact ABBA/rotation schedule.
    schedule = _balanced_schedule(active, rounds, int(seed or 0)) if seed is not None else None
    for sample in range(rounds):
        rows = [row for row in raw if int(row["sample_index"]) == sample]
        expected_present = [name for name in (schedule[sample] if schedule is not None else active) if failed_at.get(name, rounds) >= sample]
        expected_skipped = [name for name in active if name not in expected_present]
        expected_names = expected_present + expected_skipped
        if [str(row["arm"]) for row in rows] != expected_names:
            _error(f"raw_samples rows do not follow the exact paired ABBA schedule at sample {sample}")
        for position, row in enumerate(rows[:len(expected_present)]):
            if row.get("order_index") != position:
                _error(f"raw_samples order_index is not contiguous at sample {sample}")
            if row.get("status") == "failed" and int(row["sample_index"]) != failed_at.get(str(row["arm"])):
                _error(f"raw_samples failure transition is inconsistent at sample {sample}")
        for row in rows[len(expected_present):]:
            if row.get("order_index") is not None or row.get("status") != "skipped_due_to_failure":
                _error(f"raw_samples skipped rows are malformed at sample {sample}")

    # Every active arm shares one logical sample identity, including optional
    # rows that were skipped after a comparator failure.
    for sample in range(rounds):
        hashes = {str(row["input_hash"]) for row in raw if int(row["sample_index"]) == sample}
        if len(hashes) != 1:
            _error(f"input hash differs across arms at sample {sample}")

    vectors: dict[str, list[float]] = {}
    for name, rows in by_arm.items():
        if all(row.get("status") == "ok" for row in rows):
            vectors[name] = [float(row["ms"]) for row in rows]
    return vectors


def _statistics(
    raw: Sequence[Mapping[str, Any]],
    names: Mapping[str, str],
    rounds: int,
    reported: Mapping[str, Any],
    seed: int,
    *,
    sample_seed: int,
    model_config: Mapping[str, Any],
    bootstrap_samples: int,
    optional_failure_names: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Recompute every reported ratio and familywise interval from raw rows."""

    if not isinstance(reported, Mapping) or not reported:
        _error("model_timings.statistics is empty")
    vectors = _raw_vectors(
        raw,
        names,
        rounds,
        seed=sample_seed,
        model_config=model_config,
        optional_failure_names=optional_failure_names,
    )
    core_key = f"kernel_rank_{names['attnres'].rsplit('_', 1)[1]}_over_{names['fla']}"
    pairs: dict[str, tuple[str, str]] = {
        core_key: (names["fla"], names["attnres"]),
    }
    for kind in ("liger", "catswe"):
        name = names.get(kind)
        if name is not None and name in vectors:
            pairs[f"{names['attnres']}_over_{name}"] = (name, names["attnres"])
    if set(reported) != set(pairs):
        missing = sorted(set(pairs) - set(reported))
        extra = sorted(set(reported) - set(pairs))
        _error(f"statistics keys do not match complete canonical comparisons (missing={missing}, extra={extra})")
    try:
        from benchmarks.statistics import simultaneous_paired_ratio_bootstrap
        expected = simultaneous_paired_ratio_bootstrap(
            {key: (vectors[base], vectors[candidate]) for key, (base, candidate) in pairs.items()},
            samples=int(bootstrap_samples),
            seed=int(seed),
            confidence=0.95,
            margin=0.01,
        )
    except Exception as exc:
        _error(f"cannot recompute paired statistics: {exc}")
    fields = {
        "n", "estimate", "ratio", "ci", "ci_low", "ci_high", "confidence",
        "bootstrap_samples", "simultaneous", "classification",
    }
    for key, expected_item in expected.items():
        observed = _mapping(reported.get(key), f"statistics.{key}")
        if set(observed) != fields:
            _error(f"statistics.{key} fields are not exact")
        for field, value in expected_item.items():
            actual = observed.get(field)
            if isinstance(value, list):
                if (
                    not isinstance(actual, Sequence)
                    or isinstance(actual, (str, bytes))
                    or len(actual) != len(value)
                    or any(
                        isinstance(item, bool)
                        or not isinstance(item, (int, float))
                        or not math.isfinite(float(item))
                        or not math.isclose(float(item), float(expected_value), rel_tol=2e-10, abs_tol=2e-12)
                        for item, expected_value in zip(actual, value, strict=True)
                    )
                ):
                    _error(f"statistics.{key}.{field} disagrees with raw paired samples")
            elif isinstance(value, float):
                if (
                    isinstance(actual, bool)
                    or not isinstance(actual, (int, float))
                    or not math.isfinite(float(actual))
                    or not math.isclose(float(actual), value, rel_tol=2e-10, abs_tol=2e-12)
                ):
                    _error(f"statistics.{key}.{field} disagrees with raw paired samples")
            elif actual != value:
                _error(f"statistics.{key}.{field} disagrees with raw paired samples")
    return {str(key): dict(value) for key, value in expected.items()}

def _validate_state_protocol(state: Any, model_config: Mapping[str, Any], seed: int, mode: str, width: int) -> None:
    value = _mapping(state, "model_timings.state_protocol")
    fields = {"name", "semantics", "seed", "mode", "canonical_source", "mapping", "arms"}
    if set(value) != fields or value.get("name") != "canonical_implicit_max_rank_v1" or value.get("seed") != seed or value.get("mode") != mode:
        _error("model state protocol is not the canonical worker protocol")
    source = _mapping(value.get("canonical_source"), "state_protocol.canonical_source")
    required_source = {"device", "backend", "variant", "rank", "key_mode", "config", "initial_state_hash", "shape_metadata", "common_fixed_state_hash"}
    if set(source) != required_source or source.get("device") != "cpu" or source.get("backend") != "reference" or source.get("variant") != "standard" or source.get("rank") != width or source.get("key_mode") != "implicit_value_tail":
        _error("state protocol canonical source is forged")
    source_config = _mapping(source.get("config"), "state_protocol.canonical_source.config")
    expected = {name: model_config[name] for name in ("layers", "width", "heads", "ffn", "batch", "sequence", "vocab", "block_count", "mode")}
    expected.update({"variant": "standard", "rank": width})
    if not _same(source_config, expected):
        _error("state protocol canonical source geometry differs")
    _hex(source.get("initial_state_hash"), _HEX64_LOWER, "state_protocol.initial_state_hash")
    _hex(source.get("common_fixed_state_hash"), _HEX64_LOWER, "state_protocol.common_fixed_state_hash")
    shape = _mapping(source.get("shape_metadata"), "state_protocol.shape_metadata")
    if not shape:
        _error("state protocol shape metadata is empty")
    mapping = _mapping(value.get("mapping"), "state_protocol.mapping")
    expected_mapping = {"fixed_shape_tensors", "standard", "sliced.queries.*", "cuda_generators"}
    if set(mapping) != expected_mapping:
        _error("state protocol mapping fields are not exact")
    if not isinstance(value.get("arms"), Mapping):
        _error("state protocol arm records are missing")


def _validate_fla_backend_metadata(metadata: Any) -> None:
    value = _mapping(metadata, "compile_backend_metadata.fla_triton_compile")
    required = {
        "bridge", "implementation", "checkpoint_level", "qualification_eligible",
        "vendor_revision", "expected_vendor_revision", "vendor_package_sha256",
        "expected_vendor_package_sha256", "vendor_file_hashes", "expected_vendor_file_hashes",
        "vendor_license_sha256", "expected_vendor_license_sha256", "vendor_git_dirty",
        "vendor_origin", "expected_origin", "adapter_file", "adapter_sha256",
    }
    if not required.issubset(set(value)):
        _error("FLA compile backend provenance is incomplete")
    try:
        from benchmarks.competitors import (
            FLA_LICENSE_SHA256, FLA_PACKAGE_SHA256, FLA_REPOSITORY,
            FLA_REVISION, FLA_SOURCE_HASHES,
        )
    except Exception as exc:
        _error(f"cannot load pinned FLA vendor contract: {exc}")
    checks = {
        "bridge": "fla_native_compile_custom_op",
        "implementation": "triton",
        "checkpoint_level": 1,
        "qualification_eligible": True,
        "vendor_revision": FLA_REVISION,
        "expected_vendor_revision": FLA_REVISION,
        "vendor_package_sha256": FLA_PACKAGE_SHA256,
        "expected_vendor_package_sha256": FLA_PACKAGE_SHA256,
        "vendor_file_hashes": dict(FLA_SOURCE_HASHES),
        "expected_vendor_file_hashes": dict(FLA_SOURCE_HASHES),
        "vendor_license_sha256": FLA_LICENSE_SHA256,
        "expected_vendor_license_sha256": FLA_LICENSE_SHA256,
        "vendor_git_dirty": False,
        "expected_origin": FLA_REPOSITORY,
    }
    for key, expected in checks.items():
        if not _same(value.get(key), expected):
            _error(f"FLA compile backend provenance {key} differs from pinned vendor contract")
    if sweep._normalise_origin(value.get("vendor_origin")) != sweep._normalise_origin(FLA_REPOSITORY):
        _error("FLA compile backend vendor origin differs from pinned contract")
    _hex(value.get("adapter_sha256"), _HEX64_LOWER, "compile_backend_metadata adapter_sha256")
    if not isinstance(value.get("adapter_file"), str) or not value["adapter_file"]:
        _error("compile backend adapter_file is missing")
    for key in ("vendor_file_hashes", "expected_vendor_file_hashes"):
        hashes = _mapping(value[key], f"compile_backend_metadata.{key}")
        if any(_HEX64_LOWER.fullmatch(str(item)) is None for item in hashes.values()):
            _error(f"compile_backend_metadata.{key} contains malformed hashes")


def _validate_model_qualification(value: Any, path: str) -> None:
    """Require the complete core model qualification record."""

    item = _mapping(value, path)
    fields = {
        "gradient_max_abs", "loss_max_abs", "output_max_abs", "parameter_count",
        "reference_evidence_device", "status", "tolerance",
    }
    if set(item) != fields or item.get("status") != "qualified" or item.get("reference_evidence_device") != "cpu":
        _error(f"{path} is not a complete qualified model record")
    gradients = item.get("gradient_max_abs")
    if not isinstance(gradients, Sequence) or isinstance(gradients, (str, bytes)) or not gradients:
        _error(f"{path}.gradient_max_abs is incomplete")
    for index, error in enumerate(gradients):
        _number(error, f"{path}.gradient_max_abs[{index}]")
    _number(item.get("loss_max_abs"), f"{path}.loss_max_abs")
    _number(item.get("output_max_abs"), f"{path}.output_max_abs")
    if type(item.get("parameter_count")) is not int or item["parameter_count"] <= 0:
        _error(f"{path}.parameter_count is malformed")
    tolerance = _mapping(item.get("tolerance"), f"{path}.tolerance")
    if tolerance != {"rtol": 0.05, "atol": 0.05}:
        _error(f"{path}.tolerance is not the BF16 qualification tolerance")


def _validate_graph_evidence(value: Any, path: str) -> None:
    graph = _mapping(value, path)
    required = {"status", "host_ms", "state_restored_before_replay", "state_restored_model_and_optimizer", "side_stream_warmup", "complete_step", "counters", "stable_capture", "changed_input_replays"}
    if set(graph) != required:
        _error(f"{path} fields are not the exact CUDA Graph evidence")
    if graph.get("status") != "ok" or graph.get("complete_step") is not True or graph.get("stable_capture") is not True or graph.get("state_restored_before_replay") is not True or graph.get("state_restored_model_and_optimizer") is not True:
        _error(f"{path} does not prove a restored complete CUDA Graph step")
    _number(graph.get("host_ms"), f"{path}.host_ms", positive=True)
    if graph.get("side_stream_warmup") != 2 or not isinstance(graph.get("counters"), Mapping):
        _error(f"{path} CUDA Graph capture metadata is malformed")
    replay = _mapping(graph.get("changed_input_replays"), f"{path}.changed_input_replays")
    replay_fields = {"capture_input_hash", "dynamo_delta", "replay_count", "replay_input_hashes", "replays", "state_restored", "status", "tolerance"}
    if set(replay) != replay_fields or replay.get("status") != "qualified":
        _error(f"{path} changed input replay qualification did not pass")
    _hex(replay.get("capture_input_hash"), _HEX64_LOWER, f"{path}.changed_input_replays.capture_input_hash")
    hashes = replay.get("replay_input_hashes")
    if not isinstance(hashes, Sequence) or isinstance(hashes, (str, bytes)) or replay.get("replay_count") != 2 or len(hashes) != 2 or len(set(hashes)) != 2 or any(_HEX64_LOWER.fullmatch(str(item)) is None for item in hashes) or replay.get("capture_input_hash") in hashes:
        _error(f"{path}.changed_input_replays.replay_input_hashes is incomplete")
    if replay.get("state_restored") is not True or not isinstance(replay.get("dynamo_delta"), Mapping):
        _error(f"{path}.changed_input_replays state restoration evidence is malformed")
    replays = replay.get("replays")
    if not isinstance(replays, Sequence) or isinstance(replays, (str, bytes)) or len(replays) != 2:
        _error(f"{path}.changed_input_replays.replays is incomplete")
    for index, item in enumerate(replays):
        row = _mapping(item, f"{path}.changed_input_replays.replays[{index}]")
        required_replay = {"candidate_optimizer_updates", "candidate_parameter_updates", "gradient_max_abs", "index", "loss_max_abs", "model_state_max_abs", "optimizer_groups_match", "optimizer_state_max_abs", "reference_optimizer_updates", "reference_parameter_updates"}
        if set(row) != required_replay or row.get("index") != index + 1 or row.get("optimizer_groups_match") is not True:
            _error(f"{path}.changed_input_replays replay evidence is malformed")
        _number(row.get("loss_max_abs"), f"{path}.changed_input_replays.replays[{index}].loss_max_abs")
        _nonnegative_map(row.get("model_state_max_abs"), f"{path}.changed_input_replays.replays[{index}].model_state_max_abs")
        _nonnegative_map(row.get("gradient_max_abs"), f"{path}.changed_input_replays.replays[{index}].gradient_max_abs")
        _nonnegative_map(row.get("optimizer_state_max_abs"), f"{path}.changed_input_replays.replays[{index}].optimizer_state_max_abs")
        for update_key in ("candidate_parameter_updates", "reference_parameter_updates", "candidate_optimizer_updates", "reference_optimizer_updates"):
            updates = row.get(update_key)
            if not isinstance(updates, list) or not updates or any(not isinstance(item, str) or not item for item in updates):
                _error(f"{path}.changed_input_replays.replays[{index}].{update_key} is malformed")
        if row["candidate_parameter_updates"] != row["reference_parameter_updates"] or row["candidate_optimizer_updates"] != row["reference_optimizer_updates"]:
            _error(f"{path}.changed_input_replays.replays[{index}] update sets differ between candidate and reference")
    tolerance = _mapping(replay.get("tolerance"), f"{path}.changed_input_replays.tolerance")
    if set(tolerance) != {"rtol", "atol"} or tolerance != {"rtol": 0.05, "atol": 0.05}:
        _error(f"{path}.changed_input_replays.tolerance is malformed")
    _number(tolerance["rtol"], f"{path}.changed_input_replays.tolerance.rtol")
    _number(tolerance["atol"], f"{path}.changed_input_replays.tolerance.atol")


def _validate_model_report(
    model: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    cell: Mapping[str, Any],
    names: Mapping[str, str],
    rounds: int,
    warmup: int,
) -> None:
    expected_model_fields = set(_CORE_MODEL_FIELDS)
    if "model_only_admission" in config:
        expected_model_fields.add("model_only_admission")
    if set(model) != expected_model_fields:
        missing = sorted(expected_model_fields - set(model))
        extra = sorted(set(model) - expected_model_fields)
        _error(f"model_timings fields are not the exact worker report (missing={missing}, extra={extra})")
    if model.get("status") not in {"complete", "incomplete"}:
        _error("model_timings.status is not a valid worker status")
    if model.get("failures") != []:
        _error("model_timings contains a core failure")
    comparator_failures = model.get("comparator_failures")
    if not isinstance(comparator_failures, list):
        _error("model_timings.comparator_failures must be a list")
    # An optional arm can fail before it ever reaches ``raw_samples`` (for
    # example during discovery or compile), so bind failures to the sealed
    # names derived from the cell as well as to names present in raw rows.
    canonical_optional_names = {
        f"liger_rank_{cell['rank']}",
        *( {f"catswe_phase1_model_rank_{cell['rank']}"} if config["include_catswe_model"] else set() ),
    }
    for index, failure in enumerate(comparator_failures):
        item = _mapping(failure, f"model_timings.comparator_failures[{index}]")
        if set(item) != {"arm", "error", "phase", "status"}:
            _error("model comparator failure fields are incomplete")
        if item.get("status") != "failed" or not isinstance(item.get("phase"), str) or not item["phase"] or item.get("arm") not in canonical_optional_names:
            _error("model comparator failure is not bound to an optional comparator arm")
        error = _mapping(item.get("error"), f"model_timings.comparator_failures[{index}].error")
        if set(error) != {"message", "traceback", "type"}:
            _error("model comparator failure error evidence is incomplete")
        error_type = error.get("type")
        message = error.get("message")
        traceback = error.get("traceback")
        if any(not isinstance(value, str) or not value for value in (error_type, message, traceback)):
            _error("model comparator failure error evidence is malformed")
        if f"{error_type}: {message}" not in traceback:
            _error("model comparator failure message is not bound to its traceback")
    if model.get("status") == "complete" and comparator_failures:
        _error("complete model timings contain comparator failures")
    if model.get("status") == "incomplete" and not comparator_failures:
        _error("incomplete model timings lack optional comparator failure evidence")
    # ``benchmarks.run._model_config`` adds the canonical source rank (R=D)
    # to the report's model geometry even when the worker times an LR rank.
    expected_model = {**sweep.make_model_config(cell), "rank": int(cell["width"])}
    if "model_only_admission" in expected_model_fields:
        try:
            from benchmarks import run as benchmark_run
            protocol, _ = benchmark_run.load_protocol()
            _, expected_admission, admission_error = benchmark_run._model_only_rank_admission(
                config, expected_model, [int(rank) for rank in protocol["ranks"]]
            )
        except Exception as exc:
            _error(f"cannot derive model-only admission evidence: {exc}")
        if admission_error is not None or expected_admission is None or not _same(model.get("model_only_admission"), expected_admission):
            _error("model-only admission evidence differs from the worker binding")
    if not _same(model.get("config"), expected_model):
        _error("model_timings.config differs from the canonical cell model geometry")
    if model.get("effective_variant") != "sliced" or model.get("ranks") != [int(cell["rank"])] or model.get("timing_method") != "cuda_graph":
        _error("model timing variant/rank/method differs from the canonical worker route")
    if model.get("training_step") != "benchmarks.training_graph.CapturedTrainingStep.replay" or model.get("canonical_training_step") != "benchmarks.model.training_step (validated, not timed)":
        _error("model training step route is not canonical")
    if model.get("requested_warmup") != config["model_warmup"] or model.get("requested_rounds") != config["model_rounds"] or model.get("effective_warmup") != max(1, warmup) or model.get("accumulation") != 1:
        _error("model timing parameters differ from the worker config")
    if model.get("reference_timing") is not False or model.get("include_fla_model") is not False or model.get("pairwise") is not False:
        _error("reference, FLA model, or pairwise route was unexpectedly enabled")
    if model.get("frozen_baseline") is not None:
        _error("historical or frozen baseline data is not allowed in this worker report")
    _validate_state_protocol(model.get("state_protocol"), expected_model, int(config["seed"]), str(cell["mode"]), int(cell["width"]))
    scope = _mapping(model.get("model_comparator_scope"), "model_timings.model_comparator_scope")
    expected_scope: dict[str, Any] = {}
    try:
        from benchmarks import run as benchmark_run
        expected_scope[f"liger_rank_{cell['rank']}"] = benchmark_run._liger_model_eligibility(expected_model, int(cell["rank"]))
        if config["include_catswe_model"]:
            expected_scope[f"catswe_phase1_model_rank_{cell['rank']}"] = benchmark_run._catswe_model_eligibility(expected_model, int(cell["rank"]))
    except Exception as exc:
        _error(f"cannot derive model comparator eligibility: {exc}")
    if set(scope) != set(expected_scope):
        _error("model comparator scope keys are not exact")
    for key, expected in expected_scope.items():
        if not _same(scope.get(key), expected):
            _error(f"model comparator eligibility differs for {key}")
    qualification = _mapping(model.get("qualification"), "model_timings.qualification")
    if set(qualification) != {f"rank_{cell['rank']}"}:
        _error("candidate qualification keys are not exact")
    candidate_qualification = _mapping(qualification[f"rank_{cell['rank']}"], "candidate qualification")
    _validate_model_qualification(candidate_qualification, "candidate qualification")
    comparator_qualification = _mapping(model.get("comparator_qualification"), "model_timings.comparator_qualification")
    expected_qualification_keys = {f"fla_triton_compile_standard_rank_{cell['width']}", f"liger_rank_{cell['rank']}"}
    if config["include_catswe_model"]:
        expected_qualification_keys.add(f"catswe_phase1_model_rank_{cell['rank']}")
    if set(comparator_qualification) != expected_qualification_keys:
        _error("comparator qualification keys are not exact")
    fla_qualification = _mapping(comparator_qualification[f"fla_triton_compile_standard_rank_{cell['width']}"], "FLA standard qualification")
    _validate_model_qualification(fla_qualification, "FLA standard qualification")
    for key, expected in expected_scope.items():
        item = _mapping(comparator_qualification[key], f"{key} qualification")
        if not expected.get("eligible", False):
            if not _same(item, expected):
                _error(f"ineligible comparator qualification is forged for {key}")
        elif item.get("eligibility") is not None and not _same(item.get("eligibility"), expected):
            _error(f"eligible comparator qualification is not bound for {key}")
    compile_meta = _mapping(model.get("compile_backend_metadata"), "model_timings.compile_backend_metadata")
    if set(compile_meta) != {"fla_triton_compile"}:
        _error("compile backend metadata keys are not exact")
    _validate_fla_backend_metadata(compile_meta["fla_triton_compile"])
    architecture = _mapping(model.get("architecture_comparisons"), "model_timings.architecture_comparisons")
    architecture_key = f"fla_triton_compile_standard_rank_{cell['width']}"
    if set(architecture) != {architecture_key}:
        _error("standard FLA architecture comparison evidence is missing")
    architecture_item = _mapping(architecture[architecture_key], f"architecture_comparisons.{architecture_key}")
    expected_candidate_config = dict(expected_model, rank=int(cell["rank"]), variant="sliced", mode=cell["mode"])
    expected_standard_config = dict(expected_model, rank=int(cell["width"]), variant="standard", mode=cell["mode"])
    for key, expected in {
        "candidate_configs": [expected_candidate_config],
        "standard_config": expected_standard_config,
        "candidate_variant": "sliced",
        "standard_variant": "standard",
        "role": "sliced LR candidate versus standard R=D AttnRes",
        "qualification": "each architecture against its own equation reference",
        "comparison_kind_by_rank": {str(cell["rank"]): "architectural_lr_vs_standard" if cell["rank"] != cell["width"] else "same_equation_different_execution"},
    }.items():
        if not _same(architecture_item.get(key), expected):
            _error(f"architecture comparison {key} is not canonical")
    schedules = _mapping(architecture_item.get("schedules"), "architecture comparison schedules")
    if schedules.get("mode") != cell["mode"] or not isinstance(schedules.get("candidate_kernel"), str) or not isinstance(schedules.get("standard_fla"), str):
        _error("architecture source schedule evidence is incomplete")
    expected_schedules = {"kernel": "public attnres.attnres for each Block read" if cell["mode"] == "block" else "per-read aggregation", "fla": "fused per-read aggregation"}
    expected_schedules.update({"liger": "native Liger per-read aggregation; source lists are stacked and made contiguous inside the adapter call"})
    if config["include_catswe_model"]:
        expected_schedules["catswe_phase1"] = sweep.CATSWE_MODEL_SCHEDULE
    if set(schedules := _mapping(model.get("execution_schedules"), "model_timings.execution_schedules")) != set(expected_schedules) or any(schedules.get(key) != expected for key, expected in expected_schedules.items()):
        _error("model execution schedules differ from the canonical source schedule")
    if model.get("qualification_staging") != "CPU between qualifications; restored before compile/optimizer":
        _error("model qualification staging is not the canonical untimed path")
    compiled_loss = _mapping(model.get("compiled_loss"), "model_timings.compiled_loss")
    if compiled_loss != {"status": "ok", "fullgraph": True, "dynamic": False, "function": "torch.nn.functional.cross_entropy"}:
        _error("compiled loss evidence is not exact")
    active_names = set(names.values())
    optional_failure_names = {
        str(item["arm"])
        for item in comparator_failures
        if isinstance(item, Mapping) and isinstance(item.get("arm"), str)
    }
    compile_rows = _mapping(model.get("compile"), "model_timings.compile")
    optimizer_rows = _mapping(model.get("optimizer"), "model_timings.optimizer")
    for label, rows in (("compile", compile_rows), ("optimizer", optimizer_rows)):
        unknown = set(rows) - (active_names | optional_failure_names)
        if unknown:
            _error(f"{label} evidence contains unknown worker arms")
        missing = active_names - set(rows)
        if missing:
            _error(f"{label} evidence does not cover every active worker arm")
    for arm in active_names:
        comp = _mapping(compile_rows[arm], f"compile.{arm}")
        opt = _mapping(optimizer_rows[arm], f"optimizer.{arm}")
        if comp.get("status") == "ok":
            if comp.get("fullgraph") is not True or comp.get("dynamic") is not False:
                _error(f"compile evidence failed for {arm}")
        elif arm not in optional_failure_names or comp.get("status") != "failed":
            _error(f"compile evidence failed for {arm}")
        if opt.get("status") == "ok":
            if opt.get("implementation") != "AdamW(fused=True,capturable=True)" or opt.get("state_initialized_during_warmup") is not True:
                _error(f"optimizer evidence failed for {arm}")
        elif arm not in optional_failure_names or opt.get("status") != "failed":
            _error(f"optimizer evidence failed for {arm}")
    for arm in (set(compile_rows) | set(optimizer_rows)) - active_names:
        if arm not in optional_failure_names:
            _error(f"optional compile evidence is not bound to a comparator failure: {arm}")
        if arm in compile_rows and _mapping(compile_rows[arm], f"compile.{arm}").get("status") != "failed":
            _error(f"optional compile evidence is not a failed producer row: {arm}")
        if arm in optimizer_rows and _mapping(optimizer_rows[arm], f"optimizer.{arm}").get("status") != "failed":
            _error(f"optional optimizer evidence is not a failed producer row: {arm}")
    complete = _mapping(model.get("complete_step_qualification"), "model_timings.complete_step_qualification")
    gate = _mapping(model.get("pre_timing_gate"), "model_timings.pre_timing_gate")
    if set(complete) != set(names.values()) or not _same(complete, gate):
        _error("complete-step qualification gate does not cover the active arms exactly")
    for arm in names.values():
        item = _mapping(complete[arm], f"complete_step_qualification.{arm}")
        if item.get("status") != "qualified":
            if arm not in optional_failure_names or item.get("status") not in {"failed", "not_run", "skipped"}:
                _error(f"complete-step qualification failed for {arm}")
            continue
        compiled_step = _mapping(item.get("compiled_step"), f"complete_step_qualification.{arm}.compiled_step")
        required_evidence = {"loss_max_abs", "model_state_max_abs", "gradient_max_abs", "optimizer_state_max_abs", "optimizer_groups_match"}
        if not required_evidence.issubset(set(compiled_step)) or compiled_step.get("optimizer_groups_match") is not True:
            _error(f"complete-step evidence is incomplete for {arm}")
    graph = _mapping(model.get("graph"), "model_timings.graph")
    if not set(names.values()).issuperset(set(graph)):
        _error("CUDA Graph evidence contains unknown worker arms")
    missing_graph = set(names.values()) - set(graph)
    if missing_graph - optional_failure_names:
        _error("CUDA Graph evidence does not cover every core worker arm")
    for arm in names.values():
        if arm not in graph:
            continue
        graph_item = _mapping(graph[arm], f"model_timings.graph.{arm}")
        if graph_item.get("status") == "ok":
            _validate_graph_evidence(graph_item, f"model_timings.graph.{arm}")
        elif arm not in optional_failure_names or graph_item.get("status") != "failed":
            _error(f"CUDA Graph evidence failed for {arm}")
        else:
            # Capture/replay failures are retained as producer exception
            # evidence; there is intentionally no fabricated graph payload
            # for an arm that never qualified.
            if not isinstance(graph_item.get("error"), Mapping):
                _error(f"CUDA Graph failure evidence is incomplete for {arm}")
    counters = _mapping(model.get("timed_graph_counters"), "model_timings.timed_graph_counters")
    if counters.get("stable") is not True or any(counters.get(key) != 0 for key in ("graph_breaks", "recompiles", "new_unique_graphs")):
        _error("timed CUDA Graph counters are not stable")
    if model.get("changed_inputs") is not True or model.get("timed_input_identity") != {"kind": "logical_model_sample_v1", "tensor_byte_hashing": False, "device_to_host_copy": False, "shared_tensor_objects_across_arms": True}:
        _error("timed input identity is not the canonical no-copy contract")
    if model.get("timed_numerical_checks") != "pre_timing_complete_step_and_changed_input_graph_gate_only":
        _error("timed numerical-check boundary differs from the worker contract")
    expected_boundary = {
        "steady_step_includes": ["BF16 autocast", "zero_grad", "model forward", "cross_entropy loss", "backward", "gradient accumulation", "AdamW optimizer.step"],
        "excluded": ["input copies", "torch.compile", "optimizer construction", "warmup", "graph capture"],
        "loss_owner": "benchmarks.training_graph._cross_entropy",
        "backward_orchestration": "captured complete step including optimizer update",
        "optimizer_construction": "before warmup; state initialized during warmup",
    }
    if not _same(model.get("timing_boundary"), expected_boundary):
        _error("timing boundary is not the exact captured complete-step contract")
    warmup_rows = model.get("warmup")
    if not isinstance(warmup_rows, list):
        _error("warmup evidence must be a list")
    per_arm_rows: dict[str, list[Mapping[str, Any]]] = {name: [] for name in names.values()}
    for index, row in enumerate(warmup_rows):
        item = _mapping(row, f"warmup[{index}]")
        arm = item.get("arm")
        if arm not in per_arm_rows:
            _error("warmup evidence contains an unknown worker arm")
        per_arm_rows[arm].append(item)
        status = item.get("status")
        if status == "ok":
            if set(item) != {"arm", "index", "status", "host_ms"}:
                _error("warmup evidence row is not canonical")
            _number(item.get("host_ms"), f"warmup[{index}].host_ms", positive=True)
        elif status == "failed":
            if arm not in optional_failure_names or set(item) != {"arm", "index", "status", "error"}:
                _error("core or malformed warmup failure evidence")
            error = _mapping(item.get("error"), f"warmup[{index}].error")
            if set(error) != {"type", "message", "traceback"} or any(not isinstance(error[key], str) or not error[key] for key in error):
                _error("warmup failure evidence is not canonical")
        else:
            _error("warmup evidence has an invalid status")
        _int(item.get("index"), f"warmup[{index}].index")
    for arm, rows in per_arm_rows.items():
        indices = [int(item["index"]) for item in rows]
        expected_count = max(1, warmup)
        if arm in optional_failure_names and any(item.get("status") == "failed" for item in rows):
            failure_positions = [index for index, item in enumerate(rows) if item.get("status") == "failed"]
            if len(failure_positions) != 1 or failure_positions[0] != len(rows) - 1 or not 0 <= int(rows[-1]["index"]) < expected_count:
                _error(f"optional warmup failure transition is not exact for {arm}")
            expected_indices = list(range(len(rows)))
        else:
            if len(rows) != expected_count or any(item.get("status") != "ok" for item in rows):
                if arm in optional_failure_names:
                    _error(f"optional warmup evidence is incomplete for {arm}")
                _error(f"warmup evidence does not cover core arm {arm}")
            expected_indices = list(range(expected_count))
        if indices != expected_indices:
            _error(f"warmup evidence indices are not exact for {arm}")
    # ``run.py`` uses the same seeded RNG for warmup arm order and then for
    # the timed ABBA order.  Bind the persisted warmup groups to that order
    # too, so a report cannot splice together rows from another schedule.
    warmup_arm_order = []
    for item in warmup_rows:
        arm = str(item["arm"])
        if not warmup_arm_order or warmup_arm_order[-1] != arm:
            warmup_arm_order.append(arm)
    expected_warmup_order = [
        names["attnres"],
        *(name for name in (names.get("liger"), names.get("catswe")) if name is not None),
        names["fla"],
    ]
    warmup_rng = random.Random(int(config["seed"]) + 771)
    warmup_rng.shuffle(expected_warmup_order)
    if warmup_arm_order != expected_warmup_order:
        _error("warmup evidence does not follow the producer seeded arm order")
    for key in ("graph_counters",):
        if set(model.get(key, {})) != set(names.values()):
            _error(f"{key} does not cover active arms")
    graph_counter_values = _mapping(model.get("graph_counters"), "model_timings.graph_counters")
    for arm in names.values():
        item = _mapping(graph_counter_values[arm], f"graph_counters.{arm}")
        if not {"before", "after_warmup", "delta", "graph_breaks", "recompiles", "new_unique_graphs"}.issubset(set(item)):
            _error(f"graph counter evidence is incomplete for {arm}")
    profile = _mapping(model.get("model_profile"), "model_timings.model_profile")
    if profile.get("enabled") is not False or profile.get("status") != "disabled" or profile.get("requested") is not False or profile.get("failures") != []:
        _error("model profiling must remain disabled in the worker report")


def _optional_reason(model: Mapping[str, Any], cell: Mapping[str, Any], kind: str, names: Mapping[str, str]) -> tuple[str, str]:
    """Map a missing/failed optional arm to an explicit display status."""

    def failure_text(item: Mapping[str, Any] | None, fallback: str) -> str:
        if not item:
            return fallback
        value: Any = item.get("reason") or item.get("error")
        if isinstance(value, Mapping):
            value = value.get("message") or value.get("type")
        text = " ".join(str(value or fallback).split())
        return text or fallback

    name = names.get(kind)
    if name is not None:
        raw = model.get("raw_samples", [])
        if any(isinstance(row, Mapping) and row.get("arm") == name and row.get("status") == "failed" for row in raw):
            failures = [item for item in model.get("comparator_failures", []) if isinstance(item, Mapping) and item.get("arm") == name]
            return "FAIL", failure_text(failures[0] if failures else None, f"{ARM_LABELS[kind]} timing failed")
        return "OK", ""
    competitor = _mapping(cell.get("competitors"), "cell.competitors").get("liger" if kind == "liger" else "catswe_phase1")
    if isinstance(competitor, Mapping) and competitor.get("status") != "model_step_arm":
        return "NA", str(competitor.get("reason") or f"{ARM_LABELS[kind]} model arm is not eligible for this cell")
    failures = [item for item in model.get("comparator_failures", []) if isinstance(item, Mapping) and str(item.get("arm", "")).startswith("liger" if kind == "liger" else "catswe_phase1")]
    if failures:
        return "FAIL", failure_text(failures[0], f"{ARM_LABELS[kind]} qualification failed")
    qualification = _mapping(model.get("comparator_qualification"), "model_timings.comparator_qualification")
    item = qualification.get(f"liger_rank_{cell['rank']}" if kind == "liger" else f"catswe_phase1_model_rank_{cell['rank']}")
    if isinstance(item, Mapping) and item.get("status") in {"failed", "missing", "incomplete"}:
        return "FAIL", failure_text(item, f"{ARM_LABELS[kind]} qualification failed")
    return "FAIL", f"{ARM_LABELS[kind]} was eligible but has no complete timed arm"


def _unwrap(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], str, str, Mapping[str, Any]]:
    """Validate and unwrap only the exact ``run_worker`` result ABI."""

    value = _mapping(payload, "worker result")
    if set(value) != _RESULT_FIELDS:
        _error("worker result fields are not the exact canonical run_worker ABI")
    if value.get("schema") != SCHEMA:
        _error(f"canonical worker result schema must be {SCHEMA!r}")
    if value.get("status") not in {"complete", "failed"}:
        _error("worker result status is invalid")
    cell, binding = _validate_worker_binding(value)
    gpu = str(value["gpu"])
    runtime = _validate_runtime(value.get("runtime_preflight"), gpu)
    benchmark_value = value.get("benchmark")
    if benchmark_value is None:
        _error("worker result has no benchmark report")
    benchmark = _mapping(benchmark_value, "benchmark")
    if set(benchmark) != _BENCHMARK_FIELDS:
        missing = sorted(_BENCHMARK_FIELDS - set(benchmark))
        extra = sorted(set(benchmark) - _BENCHMARK_FIELDS)
        _error(f"benchmark fields are not the exact run_suite report (missing={missing}, extra={extra})")
    if not _same(benchmark.get("config"), binding["config"]):
        _error("benchmark.config differs from the payload-bound worker config")
    if benchmark.get("status") not in {"complete", "incomplete"}:
        _error("benchmark.status is not a valid model-only worker report status")
    if benchmark.get("failures") != []:
        _error("benchmark contains a core suite failure")
    if benchmark.get("fla_checkout") != {"status": "not_required"}:
        _error("worker FLA operator checkout route is not disabled as required")
    if benchmark.get("comparators_enabled") is not True:
        _error("worker comparator route is not enabled")
    comparators = _mapping(benchmark.get("comparators"), "benchmark.comparators")
    expected_comparators = {"liger"} | ({"catswe_phase1"} if binding["config"]["include_catswe_model"] else set())
    if set(comparators) != expected_comparators or any(not isinstance(item, Mapping) for item in comparators.values()):
        _error("benchmark comparator descriptors are not the exact worker routes")
    coverage = _mapping(benchmark.get("coverage"), "benchmark.coverage")
    expected_coverage_fields = {"scope", "claims_full_suite", "operator_cases_requested", "operator_cases_valid", "model", "comparators_enabled", "model_reference_timing", "include_fla_model", "include_liger_model"}
    if binding["config"]["include_catswe_model"]:
        expected_coverage_fields.add("include_catswe_model")
    if set(coverage) != expected_coverage_fields:
        _error("benchmark coverage fields are not the exact worker coverage ABI")
    # ``run_suite`` reports the canonical source model (R=D), then records the
    # worker candidate rank in ``model_timings.ranks``.
    expected_model = {**sweep.make_model_config(cell), "rank": int(cell["width"])}
    if coverage.get("scope") != "custom" or coverage.get("claims_full_suite") is not False or coverage.get("model") != expected_model or coverage.get("comparators_enabled") is not True or coverage.get("model_reference_timing") is not False or coverage.get("include_fla_model") is not False or coverage.get("include_liger_model") is not True or coverage.get("operator_cases_requested") != 0 or coverage.get("operator_cases_valid") != 0:
        _error("benchmark coverage does not bind the canonical model-only route")
    if binding["config"]["include_catswe_model"] and coverage.get("include_catswe_model") is not True:
        _error("benchmark Catswe coverage route is missing")
    for phase in ("correctness", "operator_timings"):
        phase_value = _mapping(benchmark.get(phase), f"benchmark.{phase}")
        if phase_value.get("status") != "not_run" or phase_value.get("failures") != []:
            _error(f"benchmark.{phase} unexpectedly contains an executed comparator phase")
    contract = _mapping(benchmark.get("contract"), "benchmark.contract")
    protocol = _mapping(benchmark.get("protocol"), "benchmark.protocol")
    frozen_hashes = _report_bound_frozen_hashes(binding["project_provenance"])
    if contract != {"status": "verified", "frozen_hashes": frozen_hashes} or protocol.get("frozen_hashes") != frozen_hashes or set(protocol) != {"version", "frozen_hashes"}:
        _error("benchmark frozen protocol provenance is not exact")
    _validate_environment_and_hashes(benchmark, binding["project_provenance"], runtime, gpu, binding["config"])
    _validate_result_provenance(value.get("provenance"), binding["project_provenance"], catswe_required=bool(binding["config"]["include_catswe_model"]))
    _validate_worker_identity(value, benchmark, complete=False)
    model = _mapping(benchmark.get("model_timings"), "benchmark.model_timings")
    raw_for_names = model.get("raw_samples")
    if not isinstance(raw_for_names, Sequence) or isinstance(raw_for_names, (str, bytes)):
        _error("benchmark.model_timings.raw_samples must be a list")
    names = _canonical_arm_names([_mapping(row, "raw_samples item") for row in raw_for_names], int(cell["rank"]), int(cell["width"]))
    # The benchmark report must be complete in its core model path.  Optional
    # comparator failures are represented by the producer's incomplete model
    # status and the outer failed status, while benchmark.status remains
    # incomplete because correctness/operator phases were intentionally omitted.
    warmup = model.get("requested_warmup")
    rounds = model.get("requested_rounds")
    _phase(model, binding["config"])
    if type(warmup) is not int or type(rounds) is not int:
        _error("model timing warmup/rounds are missing")
    _validate_model_report(model, config=binding["config"], cell=cell, names=names, rounds=rounds, warmup=warmup)
    if value["status"] == "complete":
        if model.get("status") != "complete" or model.get("comparator_failures") != [] or value.get("failure") is not None:
            _error("complete worker status is inconsistent with model comparator failures")
    else:
        if model.get("status") == "complete":
            _error("failed worker status lacks a model failure")
        failure = _mapping(value.get("failure"), "worker failure")
        if set(failure) != {"type", "message"} or failure.get("type") != "IncompleteModelStep" or not isinstance(failure.get("message"), str) or not failure["message"]:
            _error("failed complete-core worker failure is not the producer IncompleteModelStep record")
    return cell, benchmark, gpu, str(value["status"]), binding


def _qualification_reason(model: Mapping[str, Any], cell: Mapping[str, Any], names: Mapping[str, str], arm: str) -> tuple[str, str]:
    return _optional_reason(model, cell, arm, names)


def audit_cell(payload: Mapping[str, Any], *, source: str = "<memory>") -> CellResult:
    """Audit one complete canonical worker result and derive display values."""

    cell, benchmark, gpu, _root_status, binding = _unwrap(payload)
    mode, event, smax, width, rank, relation = _geometry(cell)
    model = _mapping(benchmark.get("model_timings"), "benchmark.model_timings")
    raw = model.get("raw_samples")
    raw = tuple(_mapping(row, "raw_samples item") for row in raw)
    names = _canonical_arm_names(raw, rank, width)
    seed = _int(binding["run_parameters"].get("seed"), "run_parameters.seed", positive=True)
    rounds = _int(model.get("requested_rounds"), "model_timings.requested_rounds", positive=True)
    warmup = _int(model.get("requested_warmup"), "model_timings.requested_warmup")
    phase, phase_warmup, phase_rounds = _phase(model, binding["config"])
    if phase_warmup != warmup or phase_rounds != rounds:
        _error("model phase timing parameters are inconsistent")
    optional_failure_names = {
        str(item["arm"])
        for item in model.get("comparator_failures", [])
        if isinstance(item, Mapping) and isinstance(item.get("arm"), str)
    }
    _raw_vectors(
        raw,
        names,
        rounds,
        seed=seed,
        model_config=model["config"],
        optional_failure_names=optional_failure_names,
    )
    reports = _statistics(
        raw,
        names,
        rounds,
        _mapping(model.get("statistics"), "model_timings.statistics"),
        seed + 18000,
        sample_seed=seed,
        model_config=model["config"],
        bootstrap_samples=int(binding["run_parameters"]["bootstrap_samples"]),
        optional_failure_names=optional_failure_names,
    )
    vectors = _raw_vectors(
        raw,
        names,
        rounds,
        seed=seed,
        model_config=model["config"],
        optional_failure_names=optional_failure_names,
    )
    arms: list[ArmResult] = [ArmResult("attnres", "OK", "", mean_ms=float(sum(vectors[names["attnres"]]) / rounds), n=rounds)]
    core_key = f"kernel_rank_{rank}_over_{names['fla']}"
    fla_stat = reports[core_key]
    arms.append(ArmResult("fla", "OK", "", mean_ms=float(sum(vectors[names["fla"]]) / rounds), ratio=float(fla_stat["ratio"]), ci_low=float(fla_stat["ci_low"]), ci_high=float(fla_stat["ci_high"]), n=rounds, reported_key=core_key))
    for kind in ("liger", "catswe"):
        name = names.get(kind)
        if name is not None and name in vectors:
            key = f"{names['attnres']}_over_{name}"
            stat = reports[key]
            arms.append(ArmResult(kind, "OK", "", mean_ms=float(sum(vectors[name]) / rounds), ratio=float(stat["ratio"]), ci_low=float(stat["ci_low"]), ci_high=float(stat["ci_high"]), n=rounds, reported_key=key))
        else:
            status, reason = _qualification_reason(model, cell, names, kind)
            arms.append(ArmResult(kind, status, reason))
    # Keep a fixed ordering in the long-form artifact regardless of active-arm
    # insertion order in the producer report.
    by_kind = {item.arm: item for item in arms}
    ordered = tuple(by_kind[kind] for kind in ("attnres", "fla", "liger", "catswe"))
    cell_status = "OK"
    reason = ""
    return CellResult(str(source), gpu, phase, mode, event, smax, width, rank, relation, warmup, rounds, cell_status, reason, ordered)

def _fallback_identity(payload: Mapping[str, Any], source: str, reason: str) -> CellResult:
    """Best-effort identity for a rejected cell; all latency fields stay null."""

    cell = payload.get("cell") if isinstance(payload.get("cell"), Mapping) else {}
    gpu = str(payload.get("gpu", "?"))
    mode = str(cell.get("mode", "?"))
    event = cell.get("event_block_size") if type(cell.get("event_block_size")) is int else None
    width = int(cell["width"]) if type(cell.get("width")) is int else 0
    rank = int(cell["rank"]) if type(cell.get("rank")) is int else 0
    relation = str(cell.get("rank_relation", "?"))
    smax = int(cell["max_read_sources"]) if type(cell.get("max_read_sources")) is int else 0
    phase = "unknown"
    arms = tuple(ArmResult(arm, "FAIL", reason) for arm in ("attnres", "fla", "liger", "catswe"))
    return CellResult(str(source), gpu, phase, mode, event, smax, width, rank, relation, 0, 0, "FAIL", reason, arms)


def load_cell(path: Path | str) -> CellResult:
    """Read and strictly audit one canonical cell JSON."""

    target = Path(path).expanduser().resolve()
    return audit_cell(_read_json(target), source=str(target))


def load_cells(paths: Sequence[Path | str], *, keep_failures: bool = True) -> tuple[CellResult, ...]:
    """Load cells in deterministic path order, retaining explicit failures."""

    results: list[CellResult] = []
    # A cell geometry is intentionally reusable across GPUs.  Keep the
    # hardware identity in the de-duplication key so the H100 and B200 copies
    # of one geometry remain two independently audited observations.  Include
    # the source geometry too: a Block cell's event block size and Smax are
    # controlled dimensions, so they are not duplicate observations either.
    seen: set[tuple[str, str, str, int | None, int, int, int, str]] = set()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        try:
            result = load_cell(path)
        except SweepPlotError as exc:
            if not keep_failures:
                raise
            try:
                payload = _read_json(path)
            except SweepPlotError:
                payload = {}
            result = _fallback_identity(payload, str(Path(path).expanduser().resolve()), str(exc))
        key = (
            str(result.gpu), result.phase, result.mode, result.event_block_size,
            result.smax, result.width, result.rank, result.rank_relation,
        )
        if key in seen:
            result = replace(result, status="FAIL", reason="duplicate canonical cell identity", arms=tuple(ArmResult(arm.arm, "FAIL", "duplicate canonical cell identity") for arm in result.arms))
        seen.add(key)
        results.append(result)
    return tuple(results)


def _equation(cell: CellResult, competitor: str) -> str:
    if cell.rank_relation == "R=D/4":
        return "LR R=D/4 vs standard FLA R=D (different equation)"
    return "standard R=D (same equation)"


def _display_source(source: str) -> str:
    """Use a repository-relative report path in published tables."""

    path = Path(source)
    if not path.is_absolute():
        return source
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return source


def table_rows(cells: Sequence[CellResult]) -> list[dict[str, Any]]:
    """Return exact long-form rows with blanks for non-numeric statuses."""

    rows: list[dict[str, Any]] = []
    gpu_order = {gpu: index for index, gpu in enumerate(SUPPORTED_GPUS)}
    phase_order = {"screen": 0, "release": 1, "unknown": 2}
    mode_order = {"full": 0, "block": 1}
    for cell in sorted(cells, key=lambda item: (gpu_order.get(item.gpu, 99), phase_order.get(item.phase, 99), mode_order.get(item.mode, 99), item.event_block_size or 0, item.width, item.rank)):
        candidate = next((item for item in cell.arms if item.arm == "attnres"), None)
        for arm in ("fla", "liger", "catswe"):
            result = next(item for item in cell.arms if item.arm == arm)
            status = result.status
            reason = result.reason or cell.reason
            numeric = status == "OK" and result.ratio is not None and candidate is not None and candidate.mean_ms is not None and result.mean_ms is not None
            row: dict[str, Any] = {
                key: "" for key in TABLE_COLUMNS
            }
            row.update({
                "gpu": cell.gpu, "phase": f"{cell.phase} ({cell.warmup}/{cell.rounds})" if cell.warmup else cell.phase,
                "mode": cell.mode, "event_block_size": "" if cell.event_block_size is None else cell.event_block_size,
                "smax": cell.smax, "width_d": cell.width, "rank_r": cell.rank,
                "rank_relation": cell.rank_relation, "equation": _equation(cell, arm),
                "competitor": ARM_LABELS[arm], "status": status, "reason": reason,
                "warmup": cell.warmup or "", "rounds": cell.rounds or "", "source": _display_source(cell.source),
            })
            if numeric:
                row.update({
                    "n": result.n, "attnres_mean_ms": f"{candidate.mean_ms:.9f}",
                    "competitor_mean_ms": f"{result.mean_ms:.9f}", "ratio": f"{result.ratio:.9f}",
                    "ci_low": f"{result.ci_low:.9f}", "ci_high": f"{result.ci_high:.9f}",
                    "advantage_pct": f"{100.0 * (1.0 - float(result.ratio)):.4f}",
                })
            rows.append(row)
    return rows


def write_table(rows: Sequence[Mapping[str, Any]], csv_path: Path | str, md_path: Path | str) -> tuple[Path, Path]:
    """Write deterministic CSV and Markdown copies of the long-form table."""

    csv_target, md_target = Path(csv_path).expanduser().resolve(), Path(md_path).expanduser().resolve()
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in TABLE_COLUMNS} for row in rows)
    lines = ["| " + " | ".join(TABLE_COLUMNS) + " |", "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"]
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in TABLE_COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    md_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_target, md_target


def _style(ax: Any, *, dark: bool) -> None:
    text = "#F7FAFC" if dark else "#1F2933"
    muted = "#A8B6C2" if dark else "#52636B"
    grid = "#28384A" if dark else "#D9E1E5"
    ax.set_facecolor("#111C2B" if dark else "#FFFFFF")
    ax.tick_params(colors=muted, labelsize=10.5)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)
    ax.title.set_color(text)
    ax.grid(axis="x", color=grid, linewidth=.65, alpha=.8)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_color(grid)


def _compact_plot_reason(status: str, reason: str) -> str:
    """Keep status annotations short while preserving full table reasons."""

    text = " ".join(str(reason).split())
    if status == "NA":
        if "power-of-two D" in text:
            return "D not power of two"
        if "not eligible" in text or "not applicable" in text:
            return "not eligible"
        if text.startswith("Catswe model capability rejects this cell"):
            return "not eligible"
    if status == "FAIL" and ("Tensor-likes are not close" in text or "Mismatched elements" in text):
        return "strict numerical gate failed"
    if text == "duplicate canonical cell identity":
        return "duplicate cell"
    if len(text) > 36:
        return text[:33].rstrip() + "..."
    return text


def _draw_cell_bars(ax: Any, cell: CellResult, *, dark: bool) -> None:
    """Draw actual complete-step latency for one measured configuration."""

    _style(ax, dark=dark)
    entries = [arm for arm in cell.arms if arm.status == "OK" and arm.mean_ms is not None]
    text = "#F7FAFC" if dark else "#1F2933"
    muted = "#A8B6C2" if dark else "#52636B"
    palette = DARK_ARM_COLORS if dark else ARM_COLORS
    if not entries:
        ax.text(.5, .5, "No qualified arms", transform=ax.transAxes, ha="center", va="center", color=ARM_COLORS["na"])
        ax.set_xticks([]); ax.set_yticks([])
        return
    labels = ["AttnRes" if arm.arm == "attnres" else ARM_LABELS[arm.arm] for arm in entries]
    values = [float(arm.mean_ms) for arm in entries]
    bars = ax.bar(
        range(len(entries)),
        values,
        color=[palette[arm.arm] for arm in entries],
        width=.62,
        zorder=2,
    )
    candidate = next(float(arm.mean_ms) for arm in entries if arm.arm == "attnres")
    # Reserve three separate vertical bands: milliseconds directly over each
    # bar, comparator advantage above that, and FAIL/NA status at the panel
    # ceiling.  Keeping these bands distinct prevents the annotations from
    # colliding on near-parity rows.
    ceiling = max(values) * 1.42
    for index, (arm, bar, value) in enumerate(zip(entries, bars, values)):
        ax.text(
            index,
            value + ceiling * .025,
            f"{value:.3f} ms",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=text,
        )
        if arm.arm != "attnres":
            advantage = 100.0 * (1.0 - candidate / value)
            if abs(advantage) < .05:
                delta = "parity"
                color = muted
            else:
                delta = f"{advantage:+.1f}%"
                color = palette["liger"] if advantage > 0 else palette["fail"]
            ax.text(
                index,
                value + ceiling * .095,
                delta,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color=color,
            )
    ax.set_ylim(0, ceiling)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=11.5, fontweight="bold")
    ax.set_ylabel("ms / complete step", fontsize=11.5, fontweight="bold")
    schedule = (
        f"Full · Smax={cell.smax}"
        if cell.mode == "full"
        else f"Block · bs={cell.event_block_size} · Smax={cell.smax}"
    )
    geometry = (
        f"D=R={cell.width}"
        if cell.rank_relation == "R=D"
        else f"LR · D/R={cell.width}/{cell.rank}"
    )
    title = f"{schedule} · {geometry}" if cell.mode == "full" else f"{schedule}\n{geometry}"
    ax.set_title(title, loc="left", fontweight="bold", fontsize=15, pad=13, color=text)
    excluded = [
        f"{ARM_LABELS[arm.arm]} {arm.status}"
        for arm in cell.arms
        if arm.arm != "attnres" and arm.status in {"FAIL", "NA"}
    ]
    if excluded:
        ax.text(
            .985,
            .975,
            " · ".join(excluded),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color=palette["fail"] if any("FAIL" in item for item in excluded) else muted,
        )


def _gpu_output_name(name: str, gpu: str) -> str:
    """Append a stable GPU suffix to a validated plain output name."""

    value = Path(name)
    return f"{value.stem}_{gpu.lower()}{value.suffix}"


def render_sweep(
    cells: Sequence[CellResult],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    svg_name: str = DEFAULT_SVG_NAME,
    png_name: str = DEFAULT_PNG_NAME,
    theme: str = "dark",
) -> tuple[Path, Path, Path, Path]:
    """Render separate, large H100 and B200 SVG/PNG small multiples.

    The split is intentional: placing all eight configurations on one canvas
    made labels too small at README width.  Each returned device figure keeps
    one vertical bar chart per configuration while retaining FAIL and NA arms.
    """

    if theme not in {"light", "dark"}:
        _error("theme must be light or dark")
    try:
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover
        raise SweepPlotError("Matplotlib is required to render the sweep") from exc
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    dark = theme == "dark"
    face = "#0B1220" if dark else "#FFFFFF"
    text = "#F7FAFC" if dark else "#1F2933"
    muted = "#A8B6C2" if dark else "#52636B"
    palette = DARK_ARM_COLORS if dark else ARM_COLORS
    outputs: list[Path] = []
    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Avenir", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "attnres-compiled-step-sweep-v2",
            "axes.axisbelow": True,
            "figure.facecolor": face,
            "savefig.facecolor": face,
        }
    ):
        for gpu in SUPPORTED_GPUS:
            subset = sorted(
                (cell for cell in cells if cell.gpu == gpu),
                key=lambda cell: (
                    cell.mode != "full",
                    cell.rank_relation != "R=D",
                    cell.width,
                    cell.smax,
                ),
            )
            if not subset:
                _error(f"no {gpu} cells were supplied")
            rows, columns = ((1, 3) if gpu == "H100" else (2, 3))
            size = (17, 6.2) if gpu == "H100" else (17, 10.8)
            fig, axes = plt.subplots(rows, columns, figsize=size, squeeze=False, facecolor=face)
            fig.subplots_adjust(
                left=.06,
                right=.985,
                top=.75 if gpu == "H100" else .78,
                bottom=.15 if gpu == "H100" else .12,
                hspace=.52,
                wspace=.25,
            )
            fig.text(
                .055,
                .95,
                f"{gpu} · compiled BF16 training steps",
                ha="left",
                va="top",
                fontsize=25,
                fontweight="bold",
                color=text,
            )
            fig.text(
                .055,
                .875,
                "One large chart per configuration · lower is faster",
                ha="left",
                va="top",
                fontsize=14,
                color=muted,
            )
            flat = list(axes.flat)
            for ax, cell in zip(flat, subset):
                _draw_cell_bars(ax, cell, dark=dark)
            for ax in flat[len(subset):]:
                ax.axis("off")
            handles = [
                Line2D([0], [0], color=palette["attnres"], linewidth=8, label="Fast-AttnRes"),
                Line2D([0], [0], color=palette["fla"], linewidth=8, label="FLA ckpt-1"),
                Line2D([0], [0], color=palette["liger"], linewidth=8, label="Liger 0.8.2"),
                Line2D([0], [0], color=palette["catswe"], linewidth=8, label="Catswe phase 1"),
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=4,
                frameon=False,
                fontsize=12.5,
                labelcolor=text,
            )
            svg = out / _gpu_output_name(svg_name, gpu)
            png = out / _gpu_output_name(png_name, gpu)
            metadata = {
                "Title": f"{gpu} BF16 compiled complete-training-step latency",
                "Creator": "benchmarks/plot_compiled_step_sweep.py",
                "Description": (
                    f"Large vertical latency bar charts for each {gpu} complete "
                    "training-step configuration; comparator labels show "
                    "Fast-AttnRes advantage and FAIL/NA arms remain explicit."
                ),
                "Date": None,
            }
            fig.savefig(svg, format="svg", metadata=metadata)
            svg.write_bytes(
                b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines())
                + b"\n"
            )
            fig.savefig(
                png,
                format="png",
                dpi=PNG_DPI,
                metadata={
                    "Software": "Fast-AttnRes compiled-step sweep",
                    "Date": None,
                },
            )
            plt.close(fig)
            outputs.extend((svg, png))
    return outputs[0], outputs[1], outputs[2], outputs[3]


def _plain_name(name: str, suffix: str) -> str:
    value = Path(name)
    if value.name != name or value.suffix.lower() != suffix:
        _error(f"output name must be a plain {suffix} filename")
    return name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="*", type=Path, help="canonical scripts.compiled_step_sweep cell JSONs")
    parser.add_argument("--input", "--cell", dest="named_cells", action="append", type=Path, default=[], help="canonical cell JSON (repeatable)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--svg-name", default=DEFAULT_SVG_NAME)
    parser.add_argument("--png-name", default=DEFAULT_PNG_NAME)
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME)
    parser.add_argument("--md-name", default=DEFAULT_MD_NAME)
    parser.add_argument("--theme", choices=("light", "dark"), default="dark")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = tuple(args.cells) + tuple(args.named_cells)
    if not paths:
        print("error: at least one canonical cell JSON is required")
        return 2
    try:
        for name, suffix in ((args.svg_name, ".svg"), (args.png_name, ".png"), (args.csv_name, ".csv"), (args.md_name, ".md")):
            _plain_name(name, suffix)
        # Publication is fail closed.  Programmatic callers can opt into
        # retained FAIL placeholders through ``load_cells(...,
        # keep_failures=True)``, but the CLI must never overwrite a populated
        # release figure with an all-empty audit-failure render.
        cells = load_cells(paths, keep_failures=False)
        rows = table_rows(cells)
        out = Path(args.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        figures = render_sweep(
            cells,
            out,
            svg_name=args.svg_name,
            png_name=args.png_name,
            theme=args.theme,
        )
        csv_path, md_path = write_table(rows, out / args.csv_name, out / args.md_name)
    except SweepPlotError as exc:
        print(f"error: {exc}")
        return 2
    for figure in figures:
        print(f"wrote {figure}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


__all__ = [
    "ARM_COLORS",
    "ARM_LABELS",
    "TABLE_COLUMNS",
    "ArmResult",
    "CellResult",
    "SweepPlotError",
    "audit_cell",
    "load_cell",
    "load_cells",
    "main",
    "render_sweep",
    "table_rows",
    "write_table",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
