"""Fail-closed audit for the compiled model timing campaign.

The campaign runner emits a direct suite result whose model phase contains a
complete CUDA-Graph training-step comparison.  This module is the offline
boundary for that result.  It validates the report schema, binds every
reported source digest to a checkout supplied by the caller, reconstructs the
logical input IDs and deterministic ABBA schedule, and recomputes the paired
statistics from raw rows.  It never imports CUDA, Torch, Triton, or a vendor
package.

``timing_verified`` means that the model-only timing sub-artifact is internally
consistent.  Such a report is intentionally not a complete release: the
runner rolls it up as ``incomplete`` while correctness and operator phases are
``not_run``.  Consequently ``release_promotable`` remains false until a full
release auditor accepts the other phases.  A separately hashed hardware/vendor
attestation can be required by callers that want a release-bound provenance
check; it does not turn a model-only report into a full release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "attnres.compiled_step_campaign.report.v1"
AUDIT_SCHEMA = "attnres.compiled_step_campaign.audit.v1"
ATTESTATION_SCHEMA = "attnres.compiled_step_campaign.attestation.v1"
MANIFEST_SCHEMA = "attnres.compiled_step_campaign.manifest.v1"
CAMPAIGN_SCHEMA = "attnres.compiled_step_campaign.v2"
RUNTIME_PREFLIGHT_SCHEMA = "attnres.compiled_step_runtime_preflight.v2"

SUPPORTED_GPUS = ("H100", "B200")
SUPPORTED_GPU_ALIASES = {"H100!": "H100", "H100": "H100", "B200": "B200"}
SUPPORTED_SEEDS = (20260827, 20260903, 20260911)
SEED_LABELS = {seed: str(seed) for seed in SUPPORTED_SEEDS}
PROJECTION_SCHEMA = "attnres.compiled_step_hero_projection.v1"
EXPECTED_ROUNDS = 120
EXPECTED_WARMUP = 10
EXPECTED_BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED_OFFSET = 18_000
EXPECTED_CONFIDENCE = 0.95
EXPECTED_MARGIN = 0.01
EXPECTED_PARAMETER_COUNT = 315
BF16_MAX_FINITE = 3.3895313892515355e38

EXPECTED_MODEL_CONFIG = {
    "batch": 2,
    "block_count": 8,
    "ffn": 2816,
    "heads": 16,
    "layers": 24,
    "mode": "full",
    "sequence": 1024,
    "source_layout": "list",
    "variant": "sliced",
    "vocab": 32768,
    "width": 1024,
}
EXPECTED_TIMING_CONFIG = {**EXPECTED_MODEL_CONFIG, "rank": 1024}
EXPECTED_BF16_TOLERANCE = {"atol": 0.05, "rtol": 0.05}
EXPECTED_FLA_REVISION = "5e02dd3a7651f5f2797eb8b12bbec401826031e1"
EXPECTED_FLA_PACKAGE_SHA256 = "2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781"
EXPECTED_FLA_ORIGIN = "https://github.com/fla-org/flash-linear-attention.git"
EXPECTED_VENDOR_FILES = {
    "fla/ops/attnres/backends/gluon.py": "f8f163fb7ebb8d035236674aeb668483812fb4e9a29572ed2ae937c626990190",
    "fla/ops/attnres/fused.py": "0e4683ab291086a9c3919d7352e2a998112973c94f5363e58f76ea7efea114f3",
}
EXPECTED_KERNEL_PATHS = (
    "src/attnres/_kernels/fixed_tail.py",
    "src/attnres/_kernels/fixed_tail_sources.py",
    "src/attnres/_kernels/fla_full_sources.py",
)
EXPECTED_GPU = {
    "H100": {"name": "NVIDIA H100 80GB HBM3", "capability": [9, 0]},
    "B200": {"name": "NVIDIA B200", "capability": [10, 0]},
}
EXPECTED_RUNTIME = {
    "torch": "2.13.0+cu130",
    "cuda_runtime": "13.0",
    "triton": "3.7.1",
}
EXPECTED_CAMPAIGN_BASE_REVISION = "81dffbfeb0f84470513e846e3df8080e8ffb563d"
EXPECTED_RMS_WEIGHT_MANIFEST = {
    "contract": "native_fla_unit_rms_v1",
    "value": "ones",
    "model_rms_weight_allocation": "nonpersistent_buffer",
    "model_rms_weight_name": "_backend_rms_weight",
    "model_rms_weight_reuse": "one_buffer_per_model",
    "model_rms_weight_shape": "[R]",
    "model_rms_weight_dtype": "float32",
    "model_rms_weight_requires_grad": False,
    "allocation_phase": "model_construction_before_compile_and_capture",
    "fill_phase": "model_construction_only",
    "compiled_model_fill_launches_per_step": 0,
    "compiled_model_fill_launches_avoided_per_step": 1,
    "direct_call_fallback": "query_ones",
    "output_rms_weight": None,
    "included_timed_operation": "native_kernel_reads_preallocated_buffer",
}
# Compatibility aliases for the fair v2 fixture/auditor.  The compact value
# is the runtime preflight field; the manifest carries the expanded lifecycle
# record above.
EXPECTED_RMS_WEIGHT_LIFECYCLE = "preallocated_nonpersistent_model_buffer"
EXPECTED_REPO_HEAD = EXPECTED_CAMPAIGN_BASE_REVISION
EXPECTED_RUNNER_SHA256 = "a212da2bf7631061659a59046a83f98ccd47ff3a8311fce03b1b1ba38f273c92"
EXPECTED_FLA_ADAPTER_SHA256 = "96e98ca3f488a36832aa767d5c3b12a5ae3544d8fc12d042c734037b62a25f75"
EXPECTED_MODEL_SHA256 = "a921d49ed4e4c2e12113d87c2cda9743e7a297bd26d4c31e77cab71dc254c21d"
EXPECTED_FROZEN_MANIFEST_SHA256 = "45e43c61511969f35d665851039985b03603e5697baf3ec99cd70a76ee0fb6f5"
EXPECTED_WRAPPER_SHA256 = "4893166e1c03db1ef53cc9b2f4469f87c29592aac033f583467eba53decbfbb3"
EXPECTED_FAIR_KERNELS = {
    "src/attnres/_kernels/fixed_tail.py": "2333b3034e3c0e6493855b1246280ed91e65d29a962ce1d150beff71e8bbd34e",
    "src/attnres/_kernels/fixed_tail_sources.py": "20fa0206fcbf6cc6b28a2973ac280575b6e8e378b09e0903449bf423d9812196",
    "src/attnres/_kernels/fla_full_sources.py": "2cd7ac89b15faeb13640bff4a7948e437453b69446bfc8c7922511e341843e10",
}
EXPECTED_NVIDIA_SMI_KEYS = frozenset(
    {
        "name",
        "uuid",
        "driver_version",
        "pstate",
        "pci.bus_id",
        "power.limit",
        "clocks.max.sm",
        "memory.total",
    }
)
EXPECTED_RMS_WEIGHT_CAMPAIGN = {
    "lifecycle": "preallocated_nonpersistent_model_buffer",
    "allocated_before_compile_capture_timing": True,
    "fill_launches_inside_step": 0,
    "direct_operator_fallback": "query_ones",
}
EXPECTED_SOURCE_PATHS = (
    "benchmarks/run.py",
    "benchmarks/training_graph.py",
    "benchmarks/fla_compile.py",
    "benchmarks/model.py",
    "benchmarks/competitors.py",
    "src/attnres/_kernels/fixed_tail.py",
    "src/attnres/_kernels/fixed_tail_sources.py",
    "src/attnres/_kernels/fla_full_sources.py",
)

ROOT_KEYS = frozenset(
    {
        "comparators",
        "comparators_enabled",
        "compiled_step_execution_status",
        "compiled_step_runtime_preflight",
        "config",
        "contract",
        "correctness",
        "coverage",
        "device",
        "environment",
        "failures",
        "fla_checkout",
        "hashes",
        "model_timings",
        "operator_timings",
        "protocol",
        "source_hashes",
        "status",
        "timing_subartifact",
    }
)
LEGACY_ROOT_KEYS = ROOT_KEYS - {"timing_subartifact"}
CONFIG_KEYS = frozenset(
    {
        "accumulation",
        "betas",
        "bootstrap_samples",
        "compiled_step_campaign",
        "fla_compile_backends",
        "include_baseline",
        "include_fla",
        "include_fla_compile",
        "include_fla_model",
        "include_packed_comparison",
        "lr",
        "mode",
        "model_config",
        "model_profile",
        "model_progress",
        "model_rounds",
        "model_state_protocol",
        "model_timing",
        "model_warmup",
        "pairwise",
        "phases",
        "production_ladder",
        "ranks",
        "reference_timing",
        "schema",
        "scope",
        "seed",
        "standard_fla_comparison",
        "variant",
        "vendor_root",
        "weight_decay",
    }
)
MODEL_TIMING_KEYS = frozenset(
    {
        "accumulation",
        "architecture_comparisons",
        "canonical_training_step",
        "changed_inputs",
        "comparator_failures",
        "comparator_qualification",
        "compile",
        "compile_backend_metadata",
        "compiled_loss",
        "complete_step_qualification",
        "config",
        "effective_variant",
        "effective_warmup",
        "execution_schedules",
        "failures",
        "frozen_baseline",
        "graph",
        "graph_counters",
        "include_fla_model",
        "model_profile",
        "optimizer",
        "pairwise",
        "pre_timing_gate",
        "qualification",
        "qualification_staging",
        "ranks",
        "raw_samples",
        "reference_timing",
        "requested_rounds",
        "requested_warmup",
        "state_protocol",
        "statistics",
        "status",
        "timed_graph_counters",
        "timed_input_identity",
        "timed_numerical_checks",
        "timing_boundary",
        "timing_method",
        "training_step",
        "warmup",
    }
)
RAW_SAMPLE_KEYS = frozenset(
    {"arm", "backend", "input_hash", "ms", "order_index", "rank", "replay_count", "sample_index", "status", "timing_method"}
)
WARMUP_KEYS = frozenset({"arm", "host_ms", "index", "status"})
STAT_KEYS = frozenset(
    {"bootstrap_samples", "ci", "ci_high", "ci_low", "classification", "confidence", "estimate", "n", "ratio", "simultaneous"}
)


class CompiledStepAuditError(ValueError):
    """Raised when a report cannot support a compiled-step timing claim."""


# Convenient compatibility names for callers that use the older auditors.
ReportAuditError = CompiledStepAuditError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompiledStepAuditError(message)


def _same(left: Any, right: Any) -> bool:
    """Compare JSON values without treating bool and int as interchangeable."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{path} must be a nonempty string")
    return value


def _int(value: Any, path: str, *, minimum: int | None = None) -> int:
    _require(type(value) is int, f"{path} must be an integer")
    if minimum is not None:
        _require(value >= minimum, f"{path} must be >= {minimum}")
    return value


def _finite(value: Any, path: str, *, positive: bool = False) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{path} must be numeric")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CompiledStepAuditError(f"{path} must be numeric") from exc
    _require(math.isfinite(number), f"{path} must be finite")
    _require(number > 0 if positive else number >= 0, f"{path} must be {'positive' if positive else 'nonnegative'}")
    return number


def _sha256_hex(value: Any, path: str) -> str:
    value = _string(value, path)
    _require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{path} must be lowercase SHA-256")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    _require(set(value) == set(expected), f"{path} fields are not exact")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompiledStepAuditError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise CompiledStepAuditError(f"JSON constant {value!r} is not permitted")


def _reject_nonfinite(value: Any, path: str = "report") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{path} must be finite")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def strict_json_loads(value: str, label: str = "JSON input") -> Any:
    """Parse JSON while rejecting duplicate object keys and nonfinite values."""

    try:
        result = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except CompiledStepAuditError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CompiledStepAuditError(f"cannot parse {label}: {exc}") from exc
    _reject_nonfinite(result, label)
    return result


def read_report(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    try:
        info = path.lstat()
    except OSError as exc:
        raise CompiledStepAuditError(f"cannot inspect report {path}: {exc}") from exc
    _require(not stat.S_ISLNK(info.st_mode) and stat.S_ISREG(info.st_mode), f"report must be a regular file: {path}")
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"), str(path))
    except (OSError, UnicodeError) as exc:
        raise CompiledStepAuditError(f"cannot read report {path}: {exc}") from exc
    _require(isinstance(value, Mapping), "report must contain an object")
    return dict(value)


def read_campaign_manifest(path: str | Path) -> dict[str, Any]:
    """Read a strict source/revision manifest used to bind a campaign."""

    path = Path(path).expanduser()
    try:
        info = path.lstat()
    except OSError as exc:
        raise CompiledStepAuditError(f"cannot inspect campaign manifest {path}: {exc}") from exc
    _require(not stat.S_ISLNK(info.st_mode) and stat.S_ISREG(info.st_mode), f"campaign manifest must be a regular file: {path}")
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"), str(path))
    except (OSError, UnicodeError) as exc:
        raise CompiledStepAuditError(f"cannot read campaign manifest {path}: {exc}") from exc
    _require(isinstance(value, Mapping), "campaign manifest must contain an object")
    return dict(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_digest(value: Any) -> str:
    # Match the campaign runner's aggregate hashes (``json.dumps(...,
    # sort_keys=True)`` with its default separators).  The logical input IDs
    # and config digest have their own compact canonicalization below.
    return _sha256_bytes(json.dumps(value, sort_keys=True).encode("utf-8"))


def _resolve_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    _require(path.is_dir(), f"repository root is not a directory: {path}")
    return path


def _safe_child(root: Path, relative: str, label: str) -> Path:
    _require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), f"{label} path is unsafe")
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CompiledStepAuditError(f"{label} path escapes repository: {relative!r}") from exc
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise CompiledStepAuditError(f"cannot inspect {label}: {candidate}") from exc
    _require(not stat.S_ISLNK(info.st_mode), f"{label} must not be a symlink: {relative}")
    _require(stat.S_ISREG(info.st_mode), f"{label} must be a regular file: {relative}")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CompiledStepAuditError(f"cannot inspect repository git state: {' '.join(args)}") from exc
    return completed.stdout.strip()


def expected_model_schedule(seed: int) -> tuple[list[str], list[list[str]]]:
    """Reproduce the runner's warmup RNG advance and two-arm ABBA schedule."""

    active = ["kernel_rank_1024", "fla_triton_compile_standard_rank_1024"]
    rng = random.Random(int(seed) + 771)
    warmup = list(active)
    rng.shuffle(warmup)
    first = list(active)
    rng.shuffle(first)
    reverse = list(reversed(first))
    orders = [list(first if index % 2 == 0 else reverse) for index in range(EXPECTED_ROUNDS)]
    return warmup, orders


def _logical_input_id(seed: int, sample_index: int) -> str:
    payload = {
        "protocol": "logical_model_sample_v1",
        "seed": int(seed),
        "sample_index": int(sample_index),
        "batch": EXPECTED_MODEL_CONFIG["batch"],
        "sequence": EXPECTED_MODEL_CONFIG["sequence"],
        "vocab": EXPECTED_MODEL_CONFIG["vocab"],
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _check_root_shape(report: Mapping[str, Any]) -> None:
    _require(set(report) in (set(ROOT_KEYS), set(LEGACY_ROOT_KEYS)), "report fields are not an accepted compiled-step envelope")
    _require(report.get("status") == "incomplete", "model-only report must roll up as incomplete")
    _require(report.get("compiled_step_execution_status") == "complete", "compiled step execution did not complete")
    if "timing_subartifact" in report:
        timing_subartifact = _mapping(report.get("timing_subartifact"), "timing_subartifact")
        _exact_keys(
            timing_subartifact,
            frozenset({"model_status", "phase", "release_promotable", "seed", "status"}),
            "timing_subartifact",
        )
        _require(
            timing_subartifact.get("status") == "complete"
            and timing_subartifact.get("phase") == "model_timings"
            and timing_subartifact.get("model_status") == "complete"
            and timing_subartifact.get("release_promotable") is False
            and type(timing_subartifact.get("seed")) is int,
            "timing sub-artifact is not an explicit complete model-only result",
        )
    _require(report.get("comparators") == {} and report.get("comparators_enabled") is False, "unrequested comparators are present")
    _require(report.get("failures") == [], "report retains failures")
    for name in ("correctness", "operator_timings"):
        phase = _mapping(report.get(name), f"report.{name}")
        _exact_keys(phase, frozenset({"failures", "reason", "status"}), f"report.{name}")
        _require(phase.get("status") == "not_run" and phase.get("reason") == "not started" and phase.get("failures") == [], f"{name} must be explicitly not_run")
    coverage = _mapping(report.get("coverage"), "report.coverage")
    _exact_keys(coverage, frozenset({"claims_full_suite", "comparators_enabled", "include_fla_model", "model", "model_reference_timing", "operator_cases_requested", "operator_cases_valid", "scope"}), "report.coverage")
    _require(coverage.get("claims_full_suite") is False and coverage.get("scope") == "primary", "coverage must be model-only primary scope")
    _require(coverage.get("comparators_enabled") is False and coverage.get("include_fla_model") is False and coverage.get("model_reference_timing") is False, "coverage includes unrequested phases")
    _int(coverage.get("operator_cases_requested"), "coverage.operator_cases_requested", minimum=0)
    _int(coverage.get("operator_cases_valid"), "coverage.operator_cases_valid", minimum=0)
    model_coverage = _mapping(coverage.get("model"), "coverage.model")
    _require(model_coverage.get("sequence") == EXPECTED_MODEL_CONFIG["sequence"], "coverage model sequence differs")
    for key, value in EXPECTED_MODEL_CONFIG.items():
        _require(model_coverage.get(key) == value, f"coverage.model.{key} differs from exact model configuration")
    _require(model_coverage.get("rank") == 1024, "coverage model rank differs")


def _check_config(report: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    config = _mapping(report.get("config"), "config")
    _exact_keys(config, CONFIG_KEYS, "config")
    expected_scalars = {
        "schema": "attnres.production_ladder_config.v1",
        "scope": "primary",
        "phases": ["model"],
        "variant": "sliced",
        "mode": "full",
        "ranks": [1024],
        "pairwise": False,
        "reference_timing": False,
        "include_fla": False,
        "include_fla_model": False,
        "include_fla_compile": True,
        "fla_compile_backends": ["triton"],
        "standard_fla_comparison": True,
        "include_baseline": False,
        "include_packed_comparison": False,
        "model_state_protocol": "canonical_implicit_max_rank_v1",
        "model_timing": "cuda_graph",
        "model_warmup": EXPECTED_WARMUP,
        "model_rounds": EXPECTED_ROUNDS,
        "accumulation": 1,
        "lr": 0.0003,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "seed": seed,
        "bootstrap_samples": EXPECTED_BOOTSTRAP_SAMPLES,
        "model_profile": False,
        "model_progress": True,
    }
    for key, value in expected_scalars.items():
        _require(_same(config.get(key), value), f"config.{key} differs from exact compiled-step contract")
    model = _mapping(config.get("model_config"), "config.model_config")
    _require(_same(dict(model), EXPECTED_MODEL_CONFIG), "config.model_config is not exact B2/T1024/D1024 Full BF16 geometry")
    campaign = _mapping(config.get("compiled_step_campaign"), "config.compiled_step_campaign")
    _exact_keys(campaign, frozenset({"comparison", "dtype", "fla_unit_rms_weight", "h100_memory_fit_profile", "hashing_inside_timing", "input_copy_inside_timing", "metric", "mode", "per_round_numerical_checks", "pool_gpus", "pool_seeds", "pre_timing_complete_step_gate", "qualification_inside_timing", "schema", "scope", "seed", "timed_tensor_hashing"}), "config.compiled_step_campaign")
    expected_campaign = {
        "schema": "attnres.compiled_step_campaign.v2",
        "scope": "bf16_complete_training_step",
        "comparison": "candidate_over_native_fla_checkpoint1",
        "dtype": "bf16_autocast",
        "h100_memory_fit_profile": "B2_T1024_N2048_D1024_L24",
        "metric": "captured_complete_training_step_device_time",
        "mode": "full",
        "seed": seed,
        "pre_timing_complete_step_gate": True,
        "timed_tensor_hashing": False,
        "hashing_inside_timing": False,
        "input_copy_inside_timing": False,
        "qualification_inside_timing": False,
        "per_round_numerical_checks": False,
        "pool_gpus": False,
        "pool_seeds": False,
        "fla_unit_rms_weight": EXPECTED_RMS_WEIGHT_CAMPAIGN,
    }
    for key, value in expected_campaign.items():
        _require(_same(campaign.get(key), value), f"config.compiled_step_campaign.{key} differs")
    ladder = _mapping(config.get("production_ladder"), "config.production_ladder")
    _require(ladder.get("state_protocol") == "canonical_implicit_max_rank_v1", "production ladder state protocol differs")
    _require(ladder.get("input_protocol") == "shared_per_sample_timed_inputs_v1", "production ladder input protocol differs")
    _require(ladder.get("source_layout") == "list" and ladder.get("cached_block") is False, "production ladder source contract differs")
    _require(ladder.get("fla_anchor") == {"checkpoint_level": 1, "implementation": "triton", "rank": 1024, "scope": "R=D anchor only"}, "FLA anchor differs")
    fla_expected = _mapping(ladder.get("fla_checkout"), "production_ladder.fla_checkout")
    _require(fla_expected.get("revision") == EXPECTED_FLA_REVISION and fla_expected.get("package_sha256") == EXPECTED_FLA_PACKAGE_SHA256 and fla_expected.get("required_clean") is True, "production ladder FLA checkout differs")
    resident = _mapping(ladder.get("resident_candidate"), "production_ladder.resident_candidate")
    _require(resident.get("selection") == "exact source gate candidate; no source or evaluator edits", "resident candidate selection differs")
    return config


def _check_model_timing(report: Mapping[str, Any], seed: int) -> tuple[Mapping[str, Any], dict[str, list[float]]]:
    timing = _mapping(report.get("model_timings"), "model_timings")
    _exact_keys(timing, MODEL_TIMING_KEYS, "model_timings")
    _require(timing.get("status") == "complete" and timing.get("failures") == [] and timing.get("comparator_failures") == [], "model timing is not complete")
    _require(_same(timing.get("config"), EXPECTED_TIMING_CONFIG), "model_timings.config differs from exact geometry")
    _require(timing.get("effective_variant") == "sliced" and timing.get("ranks") == [1024], "model timing rank or variant differs")
    for key, value in {
        "accumulation": 1,
        "effective_warmup": EXPECTED_WARMUP,
        "requested_warmup": EXPECTED_WARMUP,
        "requested_rounds": EXPECTED_ROUNDS,
        "timing_method": "cuda_graph",
        "training_step": "benchmarks.training_graph.CapturedTrainingStep.replay",
        "canonical_training_step": "benchmarks.model.training_step (validated, not timed)",
        "reference_timing": False,
        "include_fla_model": False,
        "pairwise": False,
        "changed_inputs": True,
        "qualification_staging": "CPU between qualifications; restored before compile/optimizer",
        "timed_numerical_checks": "pre_timing_complete_step_and_changed_input_graph_gate_only",
    }.items():
        _require(_same(timing.get(key), value), f"model_timings.{key} differs")
    _require(timing.get("frozen_baseline") is None, "frozen baseline is unexpectedly present")
    _require(timing.get("timed_input_identity") == {"kind": "logical_model_sample_v1", "tensor_byte_hashing": False, "device_to_host_copy": False, "shared_tensor_objects_across_arms": True}, "timed input identity contract differs")
    _require(timing.get("timing_boundary") == {"steady_step_includes": ["BF16 autocast", "zero_grad", "model forward", "cross_entropy loss", "backward", "gradient accumulation", "AdamW optimizer.step"], "excluded": ["input copies", "torch.compile", "optimizer construction", "warmup", "graph capture"], "loss_owner": "benchmarks.training_graph._cross_entropy", "backward_orchestration": "captured complete step including optimizer update", "optimizer_construction": "before warmup; state initialized during warmup"}, "timing boundary differs")
    _require(timing.get("execution_schedules") == {"kernel": "per-read aggregation", "fla": "fused per-read aggregation"}, "execution schedule differs")
    loss = _mapping(timing.get("compiled_loss"), "compiled_loss")
    _require(loss == {"status": "ok", "fullgraph": True, "dynamic": False, "function": "torch.nn.functional.cross_entropy"}, "compiled loss contract differs")
    profile = _mapping(timing.get("model_profile"), "model_profile")
    _require(profile.get("status") == "disabled" and profile.get("enabled") is False and profile.get("requested") is False and profile.get("failures") == [], "model profile is not explicitly disabled")
    _check_architecture(timing)
    arrays = _check_gates_and_raw(timing, seed)
    return timing, arrays


def _check_architecture(timing: Mapping[str, Any]) -> None:
    comparisons = _mapping(timing.get("architecture_comparisons"), "architecture_comparisons")
    _require(set(comparisons) == {"fla_triton_compile_standard_rank_1024"}, "architecture comparison coverage differs")
    item = _mapping(comparisons["fla_triton_compile_standard_rank_1024"], "architecture comparison")
    expected_candidate = {**EXPECTED_MODEL_CONFIG, "rank": 1024}
    expected_standard = {**EXPECTED_MODEL_CONFIG, "rank": 1024, "variant": "standard"}
    _require(item.get("candidate_configs") == [expected_candidate], "candidate architecture config differs")
    _require(item.get("standard_config") == expected_standard, "standard architecture config differs")
    _require(item.get("candidate_variant") == "sliced" and item.get("standard_variant") == "standard", "architecture variants differ")
    _require(item.get("comparison_kind_by_rank") == {"1024": "same_equation_different_execution"}, "architecture comparison kind differs")
    _require(item.get("role") == "sliced LR candidate versus standard R=D AttnRes", "architecture comparison role differs")
    _require(item.get("qualification") == "each architecture against its own equation reference", "architecture qualification differs")
    _require(item.get("schedules") == {"candidate_kernel": "per-read ordered source-list input", "standard_fla": "fused read of ordered source-list input", "mode": "full"}, "architecture schedules differ")


def _check_state_protocol(timing: Mapping[str, Any], seed: int) -> set[str]:
    state = _mapping(timing.get("state_protocol"), "state_protocol")
    _exact_keys(state, frozenset({"arms", "canonical_source", "mapping", "mode", "name", "seed", "semantics"}), "state_protocol")
    _require(state.get("name") == "canonical_implicit_max_rank_v1" and state.get("mode") == "full" and state.get("seed") == seed, "state protocol identity differs")
    _require(state.get("semantics") == "standard R=D canonical source with implicit value-tail keys; sliced targets map the trailing R query coordinates", "state protocol semantics differ")
    _require(state.get("mapping") == {"fixed_shape_tensors": "exact source tensor copy", "standard": "exact source tensor copy", "sliced.queries.*": "source[-R:]", "cuda_generators": "untouched"}, "state protocol mapping differs")
    canonical = _mapping(state.get("canonical_source"), "state_protocol.canonical_source")
    _exact_keys(canonical, frozenset({"backend", "common_fixed_state_hash", "config", "device", "initial_state_hash", "key_mode", "rank", "shape_metadata", "variant"}), "canonical_source")
    _require(canonical.get("backend") == "reference" and canonical.get("device") == "cpu" and canonical.get("variant") == "standard" and canonical.get("rank") == 1024 and canonical.get("key_mode") == "implicit_value_tail", "canonical source identity differs")
    canonical_config = _mapping(canonical.get("config"), "canonical_source.config")
    _require(canonical_config == {key: value for key, value in {**EXPECTED_MODEL_CONFIG, "rank": 1024, "variant": "standard"}.items() if key != "source_layout"}, "canonical source config differs")
    _sha256_hex(canonical.get("common_fixed_state_hash"), "canonical_source.common_fixed_state_hash")
    canonical_initial = _sha256_hex(canonical.get("initial_state_hash"), "canonical_source.initial_state_hash")
    shape = _mapping(canonical.get("shape_metadata"), "canonical_source.shape_metadata")
    _require(len(shape) == EXPECTED_PARAMETER_COUNT, "canonical source shape metadata does not cover every parameter")
    for name, sizes in shape.items():
        _string(name, "canonical_source.shape_metadata key")
        _require(isinstance(sizes, list) and sizes and all(type(size) is int and size > 0 for size in sizes), f"canonical source shape for {name} is malformed")
    arms = _mapping(state.get("arms"), "state_protocol.arms")
    expected_arms = {"kernel_rank_1024", "reference_rank_1024", "fla_triton_compile_standard_rank_1024", "standard_reference_rank_1024"}
    _require(set(arms) == expected_arms, "state protocol arms do not cover exactly the model job")
    for name in sorted(arms):
        record = _mapping(arms[name], f"state_protocol.arms.{name}")
        _exact_keys(record, frozenset({"arm", "common_fixed_state_hash", "initial_state_hash", "mode", "protocol", "rank", "shape_metadata", "variant"}), f"state_protocol.arms.{name}")
        expected_variant = "standard" if name in {"standard_reference_rank_1024", "fla_triton_compile_standard_rank_1024"} else "sliced"
        _require(record.get("arm") == name and record.get("rank") == 1024 and record.get("mode") == "full" and record.get("protocol") == "canonical_implicit_max_rank_v1" and record.get("variant") == expected_variant, f"state protocol arm identity differs for {name}")
        _require(record.get("common_fixed_state_hash") == canonical.get("common_fixed_state_hash"), f"state fixed hash differs for {name}")
        _require(record.get("initial_state_hash") == canonical_initial, f"state initial hash differs for {name}")
        _require(record.get("shape_metadata") == shape, f"state shape metadata differs for {name}")
    return set(shape)


def _check_qualification_record(
    value: Any,
    path: str,
    *,
    complete: bool = False,
    expected_parameter_names: set[str] | None = None,
) -> None:
    item = _mapping(value, path)
    if complete:
        _exact_keys(item, frozenset({"candidate_optimizer_updates", "candidate_parameter_updates", "dynamo_delta", "gradient_max_abs", "loss_max_abs", "model_state_max_abs", "optimizer_groups_match", "optimizer_state_max_abs", "reference_evidence_device", "reference_optimizer_updates", "reference_parameter_updates", "state_restored", "status", "tolerance"}), path)
        _require(item.get("status") == "qualified" and item.get("state_restored") is True and item.get("optimizer_groups_match") is True and item.get("reference_evidence_device") == "cpu" and item.get("dynamo_delta") == {}, f"{path} complete-step gate failed")
        _require(item.get("tolerance") == EXPECTED_BF16_TOLERANCE, f"{path}.tolerance differs")
        for key in ("candidate_parameter_updates", "reference_parameter_updates", "candidate_optimizer_updates", "reference_optimizer_updates"):
            _parameter_names(item.get(key), f"{path}.{key}")
        _require(item["candidate_parameter_updates"] == item["reference_parameter_updates"] and item["candidate_optimizer_updates"] == item["reference_optimizer_updates"], f"{path} update parameter identities differ")
        names = set(item["candidate_parameter_updates"])
        if expected_parameter_names is not None:
            _require(names == expected_parameter_names, f"{path} update names differ from state shape metadata")
        _check_error_map(item.get("gradient_max_abs"), f"{path}.gradient_max_abs", names)
        _check_bf16_error(item.get("loss_max_abs"), f"{path}.loss_max_abs")
        _check_error_map(item.get("model_state_max_abs"), f"{path}.model_state_max_abs", names)
        _check_optimizer_error_map(item.get("optimizer_state_max_abs"), f"{path}.optimizer_state_max_abs", names)
    else:
        _exact_keys(item, frozenset({"gradient_max_abs", "loss_max_abs", "output_max_abs", "parameter_count", "reference_evidence_device", "status", "tolerance"}), path)
        _require(item.get("status") == "qualified" and item.get("parameter_count") == EXPECTED_PARAMETER_COUNT and item.get("reference_evidence_device") == "cpu" and item.get("tolerance") == EXPECTED_BF16_TOLERANCE, f"{path} qualification failed")
        for key in ("output_max_abs", "loss_max_abs"):
            _check_bf16_error(item.get(key), f"{path}.{key}")
        gradients = item.get("gradient_max_abs")
        _require(isinstance(gradients, list) and len(gradients) == EXPECTED_PARAMETER_COUNT, f"{path}.gradient_max_abs must cover every parameter")
        for index, error in enumerate(gradients):
            _check_bf16_error(error, f"{path}.gradient_max_abs[{index}]")


def _parameter_names(value: Any, path: str) -> list[str]:
    _require(isinstance(value, list) and len(value) == EXPECTED_PARAMETER_COUNT, f"{path} must cover every parameter")
    _require(all(isinstance(name, str) and bool(name) for name in value), f"{path} contains an invalid parameter name")
    _require(len(set(value)) == EXPECTED_PARAMETER_COUNT, f"{path} contains duplicate parameter names")
    return value


def _check_error_map(value: Any, path: str, names: set[str]) -> None:
    item = _mapping(value, path)
    _require(set(item) == names, f"{path} parameter coverage differs")
    for name, error in item.items():
        _check_bf16_error(error, f"{path}.{name}")


def _check_optimizer_error_map(value: Any, path: str, names: set[str]) -> None:
    item = _mapping(value, path)
    _require(set(item) == names, f"{path} parameter coverage differs")
    for name, state in item.items():
        state = _mapping(state, f"{path}.{name}")
        _exact_keys(state, frozenset({"exp_avg", "exp_avg_sq", "step"}), f"{path}.{name}")
        for key, error in state.items():
            _check_bf16_error(error, f"{path}.{name}.{key}")


def _check_bf16_error(value: Any, path: str) -> float:
    """Require a finite error metric representable by BF16."""

    number = _finite(value, path)
    _require(number <= BF16_MAX_FINITE, f"{path} exceeds BF16 finite range")
    return number


def _check_graph_replay(
    value: Any,
    path: str,
    *,
    expected_parameter_names: set[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    item = _mapping(value, path)
    _exact_keys(item, frozenset({"capture_input_hash", "dynamo_delta", "replay_count", "replay_input_hashes", "replays", "state_restored", "status", "tolerance"}), path)
    _require(item.get("status") == "qualified" and item.get("replay_count") == 2 and item.get("state_restored") is True and item.get("dynamo_delta") == {} and item.get("tolerance") == EXPECTED_BF16_TOLERANCE, f"{path} graph replay gate failed")
    capture = _sha256_hex(item.get("capture_input_hash"), f"{path}.capture_input_hash")
    hashes = item.get("replay_input_hashes")
    _require(isinstance(hashes, list) and len(hashes) == 2, f"{path}.replay_input_hashes must contain two entries")
    replay_hashes = [_sha256_hex(entry, f"{path}.replay_input_hashes[{i}]") for i, entry in enumerate(hashes)]
    _require(len({capture, *replay_hashes}) == 3, f"{path} changed-input hashes are not distinct")
    replays = item.get("replays")
    _require(isinstance(replays, list) and len(replays) == 2, f"{path}.replays must contain two entries")
    expected_replay_keys = frozenset({"candidate_optimizer_updates", "candidate_parameter_updates", "gradient_max_abs", "index", "loss_max_abs", "model_state_max_abs", "optimizer_groups_match", "optimizer_state_max_abs", "reference_optimizer_updates", "reference_parameter_updates"})
    for index, replay in enumerate(replays, start=1):
        replay = _mapping(replay, f"{path}.replays[{index - 1}]")
        _exact_keys(replay, expected_replay_keys, f"{path}.replays[{index - 1}]")
        _require(replay.get("index") == index and replay.get("optimizer_groups_match") is True, f"{path}.replays[{index - 1}] identity differs")
        for key in ("candidate_parameter_updates", "reference_parameter_updates", "candidate_optimizer_updates", "reference_optimizer_updates"):
            _parameter_names(replay.get(key), f"{path}.replays[{index - 1}].{key}")
        _require(replay["candidate_parameter_updates"] == replay["reference_parameter_updates"] and replay["candidate_optimizer_updates"] == replay["reference_optimizer_updates"], f"{path}.replays[{index - 1}] update parameter identities differ")
        names = set(replay["candidate_parameter_updates"])
        if expected_parameter_names is not None:
            _require(names == expected_parameter_names, f"{path}.replays[{index - 1}] update names differ from state shape metadata")
        _check_error_map(replay.get("gradient_max_abs"), f"{path}.replays[{index - 1}].gradient_max_abs", names)
        _finite(replay.get("loss_max_abs"), f"{path}.replays[{index - 1}].loss_max_abs")
        _check_error_map(replay.get("model_state_max_abs"), f"{path}.replays[{index - 1}].model_state_max_abs", names)
        _check_optimizer_error_map(replay.get("optimizer_state_max_abs"), f"{path}.replays[{index - 1}].optimizer_state_max_abs", names)
    return capture, tuple(replay_hashes)


def _counter_tree(value: Any, path: str) -> None:
    value = _mapping(value, path)
    for key, item in value.items():
        _string(key, f"{path} key")
        if isinstance(item, Mapping):
            _counter_tree(item, f"{path}.{key}")
        else:
            _int(item, f"{path}.{key}", minimum=0)


def _check_gates_and_raw(timing: Mapping[str, Any], seed: int) -> dict[str, list[float]]:
    active = ["kernel_rank_1024", "fla_triton_compile_standard_rank_1024"]
    expected_parameter_names = _check_state_protocol(timing, seed)
    qualifications = _mapping(timing.get("qualification"), "qualification")
    _require(set(qualifications) == {"rank_1024"}, "candidate qualification coverage differs")
    _check_qualification_record(qualifications["rank_1024"], "qualification.rank_1024")
    comparator_qualifications = _mapping(timing.get("comparator_qualification"), "comparator_qualification")
    _require(set(comparator_qualifications) == {active[1]}, "FLA qualification coverage differs")
    _check_qualification_record(comparator_qualifications[active[1]], f"comparator_qualification.{active[1]}")
    _require(timing.get("compile_backend_metadata") and set(timing["compile_backend_metadata"]) == {"fla_triton_compile"}, "FLA backend metadata coverage differs")
    backend = _mapping(timing["compile_backend_metadata"]["fla_triton_compile"], "compile_backend_metadata.fla_triton_compile")
    for key, value in {
        "implementation": "triton",
        "checkpoint_level": 1,
        "qualification_eligible": True,
        "bridge": "fla_native_compile_custom_op",
        "model_forced_source_stack": False,
        "accepts_source_list": True,
        "source_table": "native_address_only_pointer_table",
        "equation_dtype": "FP32_native_kernel_accumulation",
        "storage": "BF16_or_FP32",
        "rms_weight": "ones",
        "model_rms_weight_allocation": "nonpersistent_buffer",
        "model_rms_weight_reuse": "one_buffer_per_model",
        "direct_call_fallback": "query_ones",
        "compiled_model_fill_launches_per_step": 0,
        "compiled_model_fill_launches_avoided_per_step": 1,
        "output_rms_weight": None,
        "rms_eps": 1.1920928955078125e-07,
        "scale": 1.0,
        "native_functions": ["fused_attnres_fwd", "fused_attnres_bwd"],
        "model_source_argument": "sequence_of_contiguous_source_tensors",
    }.items():
        _require(_same(backend.get(key), value), f"FLA backend metadata {key} differs")
    _require(_same(backend.get("expected_vendor_file_hashes"), EXPECTED_VENDOR_FILES) and _same(backend.get("vendor_file_hashes"), EXPECTED_VENDOR_FILES), "FLA vendor file hashes differ")
    for key, value in {"expected_vendor_revision": EXPECTED_FLA_REVISION, "vendor_revision": EXPECTED_FLA_REVISION, "expected_vendor_package_sha256": EXPECTED_FLA_PACKAGE_SHA256, "vendor_package_sha256": EXPECTED_FLA_PACKAGE_SHA256, "expected_origin": EXPECTED_FLA_ORIGIN, "vendor_origin": EXPECTED_FLA_ORIGIN, "vendor_git_dirty": False}.items():
        _require(_same(backend.get(key), value), f"FLA backend metadata {key} differs")
    compile_records = _mapping(timing.get("compile"), "compile")
    optimizer_records = _mapping(timing.get("optimizer"), "optimizer")
    graph_records = _mapping(timing.get("graph"), "graph")
    complete_records = _mapping(timing.get("complete_step_qualification"), "complete_step_qualification")
    pre_timing_records = _mapping(timing.get("pre_timing_gate"), "pre_timing_gate")
    _require(set(compile_records) == set(active) and set(optimizer_records) == set(active) and set(graph_records) == set(active), "model gate arm coverage differs")
    _require(set(complete_records) == set(active) and set(pre_timing_records) == set(active), "complete-step gate arm coverage differs")
    graph_input_identity: tuple[str, tuple[str, ...]] | None = None
    for name in active:
        compile_row = _mapping(compile_records.get(name), f"compile.{name}")
        _exact_keys(compile_row, frozenset({"dynamic", "fullgraph", "host_ms", "status"}), f"compile.{name}")
        _require(compile_row.get("status") == "ok" and compile_row.get("fullgraph") is True and compile_row.get("dynamic") is False, f"compile contract failed for {name}")
        _finite(compile_row.get("host_ms"), f"compile.{name}.host_ms", positive=True)
        optimizer = _mapping(optimizer_records.get(name), f"optimizer.{name}")
        _exact_keys(optimizer, frozenset({"host_ms", "implementation", "state_initialized_during_warmup", "status"}), f"optimizer.{name}")
        _require(optimizer.get("status") == "ok" and optimizer.get("implementation") == "AdamW(fused=True,capturable=True)" and optimizer.get("state_initialized_during_warmup") is True, f"optimizer contract failed for {name}")
        _finite(optimizer.get("host_ms"), f"optimizer.{name}.host_ms", positive=True)
        graph = _mapping(graph_records.get(name), f"graph.{name}")
        _exact_keys(graph, frozenset({"changed_input_replays", "complete_step", "counters", "host_ms", "side_stream_warmup", "stable_capture", "state_restored_before_replay", "state_restored_model_and_optimizer", "status"}), f"graph.{name}")
        _require(graph.get("status") == "ok" and graph.get("complete_step") is True and graph.get("stable_capture") is True and graph.get("state_restored_before_replay") is True and graph.get("state_restored_model_and_optimizer") is True and graph.get("side_stream_warmup") == 2, f"graph contract failed for {name}")
        _finite(graph.get("host_ms"), f"graph.{name}.host_ms", positive=True)
        _counter_tree(graph.get("counters"), f"graph.{name}.counters")
        _require(_counter_nonzero(graph.get("counters"), "graph_break") is False and _counter_nonzero(graph.get("counters"), "recompil") is False, f"graph counters unstable for {name}")
        complete = _mapping(complete_records.get(name), f"complete_step_qualification.{name}")
        _exact_keys(complete, frozenset({"compiled_step", "graph_reference_precompute", "graph_replay", "status"}), f"complete_step_qualification.{name}")
        _require(complete.get("status") == "qualified", f"complete-step qualification failed for {name}")
        _check_qualification_record(complete.get("compiled_step"), f"complete_step_qualification.{name}.compiled_step", complete=True, expected_parameter_names=expected_parameter_names)
        _require(complete.get("graph_reference_precompute") == {"evidence_device": "cpu", "replay_count": 2, "status": "qualified"}, f"graph reference precompute failed for {name}")
        current_graph_identity = _check_graph_replay(complete.get("graph_replay"), f"complete_step_qualification.{name}.graph_replay", expected_parameter_names=expected_parameter_names)
        if graph_input_identity is None:
            graph_input_identity = current_graph_identity
        else:
            _require(current_graph_identity == graph_input_identity, f"changed-input graph evidence differs between arms for {name}")
        _require(pre_timing_records.get(name) == complete, f"pre-timing gate differs from complete-step gate for {name}")
        _check_graph_replay(graph["changed_input_replays"], f"graph.{name}.changed_input_replays", expected_parameter_names=expected_parameter_names)
        _require(graph["changed_input_replays"] == timing["complete_step_qualification"][name]["graph_replay"], f"graph replay evidence differs from pre-timing evidence for {name}")
    _check_graph_counters(timing, active)
    return _check_raw_samples(timing, seed, active)


def _counter_nonzero(value: Any, needle: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    for key, item in value.items():
        if needle.lower() in str(key).lower() and isinstance(item, (int, float)) and not isinstance(item, bool) and item != 0:
            return True
        if _counter_nonzero(item, needle):
            return True
    return False


def _check_graph_counters(timing: Mapping[str, Any], active: Sequence[str]) -> None:
    counters = _mapping(timing.get("graph_counters"), "graph_counters")
    _require(set(counters) == set(active), "graph counter coverage differs")
    for name in active:
        item = _mapping(counters[name], f"graph_counters.{name}")
        _exact_keys(item, frozenset({"after_warmup", "before", "delta", "graph_breaks", "new_unique_graphs", "recompiles"}), f"graph_counters.{name}")
        for phase in ("before", "after_warmup", "delta"):
            _counter_tree(item[phase], f"graph_counters.{name}.{phase}")
        _require(item.get("graph_breaks") == 0 and item.get("recompiles") == 0 and item.get("new_unique_graphs") == 1, f"graph capture counters failed for {name}")
    timed = _mapping(timing.get("timed_graph_counters"), "timed_graph_counters")
    _exact_keys(timed, frozenset({"after", "before", "delta", "graph_breaks", "new_unique_graphs", "recompiles", "stable"}), "timed_graph_counters")
    for phase in ("before", "after", "delta"):
        _counter_tree(timed[phase], f"timed_graph_counters.{phase}")
    _require(timed.get("delta") == {} and timed.get("before") == timed.get("after") and timed.get("graph_breaks") == 0 and timed.get("recompiles") == 0 and timed.get("new_unique_graphs") == 0 and timed.get("stable") is True, "timed CUDA Graph counters are unstable")


def _check_raw_samples(timing: Mapping[str, Any], seed: int, active: Sequence[str]) -> dict[str, list[float]]:
    warmup_order, orders = expected_model_schedule(seed)
    warmup = timing.get("warmup")
    _require(isinstance(warmup, list) and len(warmup) == EXPECTED_WARMUP * len(active), "warmup matrix is incomplete")
    for index, value in enumerate(warmup):
        row = _mapping(value, f"warmup[{index}]")
        _exact_keys(row, WARMUP_KEYS, f"warmup[{index}]")
        expected_arm = warmup_order[index // EXPECTED_WARMUP]
        _require(row.get("arm") == expected_arm and row.get("index") == index % EXPECTED_WARMUP and row.get("status") == "ok", f"warmup schedule differs at row {index}")
        _finite(row.get("host_ms"), f"warmup[{index}].host_ms", positive=True)
    rows = timing.get("raw_samples")
    _require(isinstance(rows, list) and len(rows) == EXPECTED_ROUNDS * len(active), "raw timing matrix is incomplete")
    arrays = {arm: [None] * EXPECTED_ROUNDS for arm in active}
    hashes: list[str | None] = [None] * EXPECTED_ROUNDS
    for index, value in enumerate(rows):
        row = _mapping(value, f"raw_samples[{index}]")
        _exact_keys(row, RAW_SAMPLE_KEYS, f"raw_samples[{index}]")
        sample = index // len(active)
        position = index % len(active)
        arm = orders[sample][position]
        _require(row.get("sample_index") == sample and row.get("order_index") == position and row.get("arm") == arm, f"raw ABBA schedule differs at row {index}")
        _require(row.get("status") == "ok" and row.get("timing_method") == "cuda_graph" and row.get("replay_count") == 1, f"raw timing contract failed at row {index}")
        expected_rank = 1024
        expected_backend = "kernel" if arm == "kernel_rank_1024" else "fla_triton_compile"
        _require(row.get("rank") == expected_rank and row.get("backend") == expected_backend, f"raw arm metadata differs at row {index}")
        digest = _sha256_hex(row.get("input_hash"), f"raw_samples[{index}].input_hash")
        expected_digest = _logical_input_id(seed, sample)
        _require(digest == expected_digest, f"raw input ID differs from logical_model_sample_v1 at sample {sample}")
        if hashes[sample] is None:
            hashes[sample] = digest
        _require(hashes[sample] == digest, f"timed arms do not share one logical input ID at sample {sample}")
        arrays[arm][sample] = _finite(row.get("ms"), f"raw_samples[{index}].ms", positive=True)
    _require(all(item is not None for item in hashes) and len(set(hashes)) == EXPECTED_ROUNDS, "timed inputs are not unique per sample")
    return {arm: [float(item) for item in values if item is not None] for arm, values in arrays.items()}


def _classify(low: float, high: float) -> str:
    if high < 1.0:
        return "gain"
    if 1.0 - EXPECTED_MARGIN <= low <= 1.0 <= high <= 1.0 + EXPECTED_MARGIN:
        return "plateau"
    if low > 1.0:
        return "slowdown"
    return "inconclusive"


def _recompute_statistics(arrays: Mapping[str, Sequence[float]], seed: int) -> dict[str, dict[str, Any]]:
    """Recompute mean paired ratios and the common-index max-deviation CI."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a test dependency
        raise CompiledStepAuditError("numpy is required to recompute compiled-step statistics") from exc
    candidate = np.asarray(list(arrays["kernel_rank_1024"]), dtype=np.float64)
    baseline = np.asarray(list(arrays["fla_triton_compile_standard_rank_1024"]), dtype=np.float64)
    _require(candidate.size == baseline.size == EXPECTED_ROUNDS, "paired timing vectors must contain 120 samples")
    _require(np.isfinite(candidate).all() and np.isfinite(baseline).all() and (candidate > 0).all() and (baseline > 0).all(), "timing vectors must be finite and positive")
    ratios = candidate / baseline
    rng = np.random.default_rng(int(seed) + BOOTSTRAP_SEED_OFFSET)
    indices = rng.integers(0, EXPECTED_ROUNDS, size=(EXPECTED_BOOTSTRAP_SAMPLES, EXPECTED_ROUNDS))
    estimates = ratios[indices].mean(axis=1)
    point = float(ratios.mean())
    width = float(np.quantile(np.abs(estimates - point), 0.975, method="linear"))
    low, high = point - width, point + width
    item = {"n": EXPECTED_ROUNDS, "estimate": point, "ratio": point, "ci": [float(low), float(high)], "ci_low": float(low), "ci_high": float(high), "confidence": EXPECTED_CONFIDENCE, "bootstrap_samples": EXPECTED_BOOTSTRAP_SAMPLES, "simultaneous": True, "classification": _classify(low, high)}
    return {"kernel_rank_1024_over_fla_triton_compile_standard_rank_1024": item}


def _check_statistics(timing: Mapping[str, Any], arrays: Mapping[str, Sequence[float]], seed: int) -> dict[str, dict[str, Any]]:
    expected = _recompute_statistics(arrays, seed)
    observed = _mapping(timing.get("statistics"), "model_timings.statistics")
    _require(set(observed) == set(expected), "statistics comparison IDs differ from raw arms")
    for name, item in expected.items():
        value = _mapping(observed.get(name), f"statistics.{name}")
        _exact_keys(value, STAT_KEYS, f"statistics.{name}")
        _require(_same(value, item), f"statistics.{name} disagrees with raw samples")
    return expected


def _check_campaign_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    revision: str,
    project: Mapping[str, str],
    frozen: Mapping[str, str],
    runner_sha: str,
    kernel_hashes: Mapping[str, str],
) -> None:
    manifest = _mapping(manifest, "campaign manifest")
    # Keep accepting the compact fair v2 attestation used by the already
    # running wrapper.  New repository generated reports use the sealed
    # six-job manifest below; the compact form is only a byte binding for one
    # historical report and cannot weaken the report's own runtime checks.
    legacy_keys = frozenset({"kernel_sha256", "project", "frozen", "repo_head", "runner_sha256", "schema"})
    if set(manifest) == set(legacy_keys):
        _require(manifest.get("schema") == MANIFEST_SCHEMA, "campaign manifest schema differs")
        _require(manifest.get("repo_head") == revision, "campaign manifest repo HEAD differs from local checkout")
        _require(_same(manifest.get("project"), project), "campaign manifest project hashes differ from local checkout")
        _require(_same(manifest.get("frozen"), frozen), "campaign manifest frozen hashes differ from local checkout")
        _require(manifest.get("runner_sha256") == runner_sha, "campaign manifest runner hash differs from local checkout")
        _require(_same(manifest.get("kernel_sha256"), kernel_hashes), "campaign manifest kernel hashes differ from local checkout")
        return
    _exact_keys(
        manifest,
        frozenset({
            "config_path",
            "config_sha256",
            "fla",
            "gpus",
            "jobs",
            "repo_base_revision",
            "rms_weight_lifecycle",
            "runtime",
            "schema",
            "seeds",
            "source_sha256",
            "timing_contract",
        }),
        "campaign manifest",
    )
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "campaign manifest schema differs")
    base_revision = _string(manifest.get("repo_base_revision"), "campaign manifest.repo_base_revision")
    _require(re.fullmatch(r"[0-9a-f]{40}", base_revision) is not None, "campaign manifest.repo_base_revision must be a git SHA")
    _require(base_revision == EXPECTED_CAMPAIGN_BASE_REVISION, "campaign manifest base revision differs")
    try:
        _git(root, "merge-base", "--is-ancestor", base_revision, revision)
    except CompiledStepAuditError:
        # The caller's checkout is authoritative; the campaign base is a
        # sealed minimum revision and may be followed by this audit commit.
        _require(revision == base_revision, "campaign manifest base revision is not an ancestor of the report checkout")
    _sha256_hex(manifest.get("config_sha256"), "campaign manifest.config_sha256")
    _require(manifest.get("config_path") == "configs/compiled_step_campaign.json", "campaign manifest config path differs")
    _require(manifest.get("config_sha256") == _sha256_file(_safe_child(root, "configs/compiled_step_campaign.json", "campaign config")), "campaign manifest config digest differs from local sealed bytes")
    _require(_same(manifest.get("runtime"), EXPECTED_RUNTIME), "campaign manifest runtime differs")
    source = _mapping(manifest.get("source_sha256"), "campaign manifest.source_sha256")
    _require(set(source) == set(EXPECTED_SOURCE_PATHS), "campaign manifest source hash coverage differs")
    for relative in EXPECTED_SOURCE_PATHS:
        _sha256_hex(source.get(relative), f"campaign manifest.source_sha256.{relative}")
        _require(source.get(relative) == _sha256_file(_safe_child(root, relative, "campaign source")), f"campaign manifest source hash differs for {relative}")
    _require(_same(manifest.get("rms_weight_lifecycle"), EXPECTED_RMS_WEIGHT_MANIFEST), "campaign manifest RMS-weight lifecycle differs")
    _require(isinstance(manifest.get("timing_contract"), Mapping), "campaign manifest timing contract is missing")
    _require(_same(manifest.get("fla"), {"origin": EXPECTED_FLA_ORIGIN, "package_sha256": EXPECTED_FLA_PACKAGE_SHA256, "revision": EXPECTED_FLA_REVISION, "source_sha256": EXPECTED_VENDOR_FILES}), "campaign manifest FLA identity differs")
    _require(_same(manifest.get("gpus"), {gpu: {"capability": EXPECTED_GPU[gpu]["capability"], "name": EXPECTED_GPU[gpu]["name"], "selector": gpu} for gpu in SUPPORTED_GPUS}), "campaign manifest GPU matrix differs")
    _require(manifest.get("seeds") == list(SUPPORTED_SEEDS), "campaign manifest seed matrix differs")
    jobs = manifest.get("jobs")
    _require(isinstance(jobs, list) and len(jobs) == len(SUPPORTED_GPUS) * len(SUPPORTED_SEEDS), "campaign manifest job matrix is incomplete")
    expected_jobs = {(gpu, seed) for gpu in SUPPORTED_GPUS for seed in SUPPORTED_SEEDS}
    actual_jobs = set()
    for index, job in enumerate(jobs):
        row = _mapping(job, f"campaign manifest.jobs[{index}]")
        _exact_keys(row, frozenset({"gpu", "seed", "filename"}), f"campaign manifest.jobs[{index}]")
        selected = SUPPORTED_GPU_ALIASES.get(row.get("gpu"))
        seed = row.get("seed")
        _require(selected in SUPPORTED_GPUS and type(seed) is int and seed in SUPPORTED_SEEDS, f"campaign manifest.jobs[{index}] identity is invalid")
        _require(isinstance(row.get("filename"), str) and row["filename"].endswith(".json") and "/" not in row["filename"] and "\\" not in row["filename"], f"campaign manifest.jobs[{index}].filename is unsafe")
        actual_jobs.add((selected, seed))
    _require(actual_jobs == expected_jobs, "campaign manifest job matrix differs")


def _check_runtime_and_hashes(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    root: Path,
    gpu: str,
    campaign_manifest: Mapping[str, Any] | None = None,
    campaign_manifest_digest: str | None = None,
) -> None:
    pre = _mapping(report.get("compiled_step_runtime_preflight"), "compiled_step_runtime_preflight")
    fair_preflight_keys = frozenset({
        "compute_capability", "config_sha256", "cuda_runtime", "fla_adapter_sha256",
        "fla_clean", "fla_fill_launches_inside_step", "fla_revision",
        "fla_unit_rms_weight_lifecycle", "frozen_manifest_sha256", "gpu_name",
        "gpu_selector", "kernel_sha256", "model_sha256", "nvidia_smi", "repo_clean",
        "repo_head", "runner_sha256", "schema", "started_unix_s", "status",
        "timed_input_copy", "timed_qualification", "timed_tensor_hashing", "torch",
        "triton", "finished_unix_s", "wrapper_sha256",
    })
    extended_preflight_keys = fair_preflight_keys | frozenset({"fla_origin", "fla_package_sha256", "manifest_sha256", "source_sha256"})
    _require(set(pre) in (set(fair_preflight_keys), set(extended_preflight_keys)), "compiled_step_runtime_preflight fields are not an accepted v2 envelope")
    _require(pre.get("schema") == "attnres.compiled_step_runtime_preflight.v2", "runtime preflight schema differs")
    expected_gpu = EXPECTED_GPU[gpu]
    _require(pre.get("status") == "passed" and pre.get("gpu_selector") == gpu and pre.get("gpu_name") == expected_gpu["name"] and pre.get("compute_capability") == expected_gpu["capability"], "runtime preflight GPU identity differs")
    _require(pre.get("torch") == EXPECTED_RUNTIME["torch"] and pre.get("triton") == EXPECTED_RUNTIME["triton"] and pre.get("cuda_runtime") == EXPECTED_RUNTIME["cuda_runtime"] and pre.get("fla_revision") == EXPECTED_FLA_REVISION and pre.get("fla_clean") is True and pre.get("repo_clean") is True, "runtime preflight software/provenance differs")
    _require(pre.get("fla_adapter_sha256") == _sha256_file(_safe_child(root, "benchmarks/fla_compile.py", "FLA adapter")), "runtime FLA adapter hash differs")
    _require(pre.get("model_sha256") == _sha256_file(_safe_child(root, "benchmarks/model.py", "model")), "runtime model hash differs")
    _require(pre.get("fla_unit_rms_weight_lifecycle") == "preallocated_nonpersistent_model_buffer" and pre.get("fla_fill_launches_inside_step") == 0, "runtime RMS-weight lifecycle differs")
    _require(pre.get("timed_tensor_hashing") is False and pre.get("timed_input_copy") is False and pre.get("timed_qualification") is False, "runtime preflight timed boundary flags permit hidden work")
    _sha256_hex(pre.get("runner_sha256"), "runtime preflight.runner_sha256")
    _sha256_hex(pre.get("wrapper_sha256"), "runtime preflight.wrapper_sha256")
    _finite(pre.get("started_unix_s"), "runtime preflight.started_unix_s")
    _finite(pre.get("finished_unix_s"), "runtime preflight.finished_unix_s")
    _require(pre.get("finished_unix_s") >= pre.get("started_unix_s"), "runtime preflight timestamps are reversed")
    _require(isinstance(pre.get("nvidia_smi"), Mapping), "runtime preflight nvidia_smi attestation is missing")
    _sha256_hex(pre.get("frozen_manifest_sha256"), "runtime preflight.frozen_manifest_sha256")
    _require(pre.get("frozen_manifest_sha256") == _sha256_file(_safe_child(root, "validation/frozen.json", "frozen manifest")), "runtime frozen manifest digest differs from local bytes")
    revision = _git(root, "rev-parse", "HEAD")
    _require(pre.get("repo_head") == revision, "report repo HEAD does not match local checkout")
    _require(_git(root, "status", "--porcelain", "--untracked-files=all") == "", "local checkout is dirty; rerun from the sealed commit")
    environment = _mapping(report.get("environment"), "environment")
    _exact_keys(environment, frozenset({"cuda_runtime", "env", "git", "hostname", "machine", "platform", "python", "torch", "triton"}), "environment")
    _require(environment.get("torch") == pre.get("torch") and environment.get("triton") == pre.get("triton") and environment.get("cuda_runtime") == pre.get("cuda_runtime") and environment.get("env") == {}, "environment runtime differs from preflight")
    git_info = _mapping(environment.get("git"), "environment.git")
    _require(git_info.get("revision") == revision and git_info.get("dirty") is False, "environment git provenance differs")
    device = _mapping(report.get("device"), "device")
    _exact_keys(device, frozenset({"available", "capability", "count", "index", "multi_processor_count", "name", "requested", "total_memory", "type"}), "device")
    _require(device.get("available") is True and device.get("type") == "cuda" and device.get("requested") == "cuda:0" and device.get("index") == 0 and device.get("count") == 1 and device.get("name") == expected_gpu["name"] and device.get("capability") == expected_gpu["capability"], "device attestation differs")
    _int(device.get("total_memory"), "device.total_memory", minimum=1)
    _int(device.get("multi_processor_count"), "device.multi_processor_count", minimum=1)
    hashes = _mapping(report.get("hashes"), "hashes")
    _exact_keys(hashes, frozenset({"hardware", "protocol", "software"}), "hashes")
    _require(hashes.get("hardware") == _json_digest(device), "hardware aggregate hash differs")
    source_hashes = _mapping(report.get("source_hashes"), "source_hashes")
    _exact_keys(source_hashes, frozenset({"frozen", "project", "software_hash", "vendor"}), "source_hashes")
    frozen = _local_frozen_hashes(root)
    _require(_same(source_hashes.get("frozen"), frozen), "source_hashes.frozen differs from local validation/frozen.json")
    contract = _mapping(report.get("contract"), "contract")
    protocol = _mapping(report.get("protocol"), "protocol")
    _exact_keys(contract, frozenset({"frozen_hashes", "status"}), "contract")
    _exact_keys(protocol, frozenset({"frozen_hashes", "version"}), "protocol")
    _require(contract.get("status") == "verified" and protocol.get("version") == 1, "contract/protocol status differs")
    _require(_same(contract.get("frozen_hashes"), frozen) and _same(protocol.get("frozen_hashes"), frozen) and _same(hashes.get("protocol"), frozen), "frozen hash copies disagree")
    project = _local_project_hashes(root)
    reported_project = _mapping(source_hashes.get("project"), "source_hashes.project")
    if "timing_subartifact" in report:
        _require(_same(reported_project, project), "source_hashes.project differs from local checkout")
    else:
        # The live fair wrapper predates this repository-integrated tooling;
        # permit its report to omit only newer audit/CLI modules while still
        # binding every path it actually reported to local bytes.
        _require(set(reported_project) <= set(project), "source_hashes.project contains an unknown local path")
        for relative, digest in reported_project.items():
            _require(project.get(relative) == digest, f"source_hashes.project differs from local checkout for {relative}")
    vendor = source_hashes.get("vendor")
    _require(vendor == {"dispatch_environment": None, "git_revision": None, "path": None}, "source_hashes.vendor must remain explicitly external")
    software_payload = {"frozen": frozen, "project": project, "vendor": vendor}
    _require(source_hashes.get("software_hash") == _json_digest(software_payload) and hashes.get("software") == _json_digest(software_payload), "software aggregate hash differs")
    runner_sha = _sha256_file(_safe_child(root, "benchmarks/run.py", "runner"))
    _require(pre.get("runner_sha256") == runner_sha and source_hashes["project"].get("benchmarks/run.py") == runner_sha, "runner hash differs from local checkout")
    kernel_hashes = _mapping(pre.get("kernel_sha256"), "runtime preflight kernel_sha256")
    _require(set(kernel_hashes) == set(EXPECTED_KERNEL_PATHS), "runtime kernel hash coverage differs")
    for relative in EXPECTED_KERNEL_PATHS:
        actual = _sha256_file(_safe_child(root, relative, "kernel"))
        _require(kernel_hashes.get(relative) == actual and source_hashes["project"].get(relative) == actual, f"kernel hash differs for {relative}")
    if "source_sha256" in pre:
        source_digest = _mapping(pre.get("source_sha256"), "runtime preflight source_sha256")
        _require(set(source_digest) == set(EXPECTED_SOURCE_PATHS), "runtime source hash coverage differs")
        for relative in EXPECTED_SOURCE_PATHS:
            actual = _sha256_file(_safe_child(root, relative, "source"))
            _require(source_digest.get(relative) == actual and source_hashes["project"].get(relative) == actual, f"source hash differs for {relative}")
    else:
        _require(source_hashes["project"].get("benchmarks/fla_compile.py") == pre.get("fla_adapter_sha256") and source_hashes["project"].get("benchmarks/model.py") == pre.get("model_sha256"), "runtime source hash evidence is incomplete")
        for relative in EXPECTED_KERNEL_PATHS:
            _require(source_hashes["project"].get(relative) == kernel_hashes.get(relative), f"runtime kernel source evidence is incomplete for {relative}")
    if "fla_origin" in pre:
        _require(pre.get("fla_origin") == EXPECTED_FLA_ORIGIN and pre.get("fla_package_sha256") == EXPECTED_FLA_PACKAGE_SHA256, "extended runtime FLA identity differs")
    else:
        _require(_mapping(config.get("production_ladder"), "production_ladder").get("fla_checkout", {}).get("package_sha256") == EXPECTED_FLA_PACKAGE_SHA256, "runtime FLA package identity is not bound by config")
    if "manifest_sha256" in pre:
        _sha256_hex(pre.get("manifest_sha256"), "runtime preflight.manifest_sha256")
        if campaign_manifest_digest is not None:
            _require(campaign_manifest_digest == pre.get("manifest_sha256"), "audited campaign manifest bytes differ from runtime preflight")
    if campaign_manifest is not None and "config_sha256" in campaign_manifest:
        _require(_sha256_hex(campaign_manifest.get("config_sha256"), "campaign manifest.config_sha256") == pre.get("config_sha256"), "runtime config hash differs from campaign manifest")
    if campaign_manifest is not None:
        _check_campaign_manifest(campaign_manifest, root=root, revision=revision, project=project, frozen=frozen, runner_sha=runner_sha, kernel_hashes=dict(kernel_hashes))
    _check_fla(report, config)


def _local_project_hashes(root: Path) -> dict[str, str]:
    paths = sorted([*root.joinpath("src").rglob("*.py"), *root.joinpath("benchmarks").rglob("*.py")])
    _require(paths, "local project source set is empty")
    result: dict[str, str] = {}
    for path in paths:
        relative = str(path.relative_to(root))
        _safe_child(root, relative, "project source")
        result[relative] = _sha256_file(path)
    return result


def _local_frozen_hashes(root: Path) -> dict[str, str]:
    manifest = _safe_child(root, "validation/frozen.json", "frozen manifest")
    try:
        value = strict_json_loads(manifest.read_text(encoding="utf-8"), str(manifest))
    except (OSError, UnicodeError) as exc:
        raise CompiledStepAuditError(f"cannot read frozen manifest: {exc}") from exc
    _require(isinstance(value, Mapping) and value, "validation/frozen.json must contain a nonempty object")
    result: dict[str, str] = {}
    for relative, digest in value.items():
        _sha256_hex(digest, f"validation/frozen.json[{relative!r}]")
        result[str(relative)] = str(digest)
        actual = _sha256_file(_safe_child(root, str(relative), "frozen source"))
        _require(actual == digest, f"local frozen source hash differs for {relative}")
    return result


def _check_fla(report: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    checkout = _mapping(report.get("fla_checkout"), "fla_checkout")
    _exact_keys(checkout, frozenset({"actual", "anchor", "expected", "status"}), "fla_checkout")
    _require(checkout.get("status") == "verified", "FLA checkout is not verified")
    expected = _mapping(checkout.get("expected"), "fla_checkout.expected")
    _require(expected == {"environment": "ATTNRES_FLA_DIR", "layout": "clean checkout containing fla/", "package_sha256": EXPECTED_FLA_PACKAGE_SHA256, "required_clean": True, "revision": EXPECTED_FLA_REVISION}, "FLA expected checkout identity differs")
    actual = _mapping(checkout.get("actual"), "fla_checkout.actual")
    _exact_keys(actual, frozenset({"git_dirty", "origin", "package_file_count", "package_sha256", "path", "revision"}), "fla_checkout.actual")
    _require(actual.get("git_dirty") is False and actual.get("origin") == EXPECTED_FLA_ORIGIN and actual.get("package_sha256") == EXPECTED_FLA_PACKAGE_SHA256 and actual.get("revision") == EXPECTED_FLA_REVISION, "FLA actual checkout identity differs")
    _int(actual.get("package_file_count"), "fla_checkout.actual.package_file_count", minimum=1)
    _string(actual.get("path"), "fla_checkout.actual.path")
    _require(checkout.get("anchor") == {"checkpoint_level": 1, "implementation": "triton", "rank": 1024, "scope": "R=D anchor only"}, "FLA anchor differs")
    configured = _mapping(_mapping(config.get("production_ladder"), "production_ladder").get("fla_checkout"), "production_ladder.fla_checkout")
    _require(configured.get("revision") == actual.get("revision") and configured.get("package_sha256") == actual.get("package_sha256"), "config and FLA checkout identities disagree")


def _check_attestation(attestation: Mapping[str, Any], report: Mapping[str, Any], report_sha256: str | None, gpu: str) -> None:
    _exact_keys(attestation, frozenset({"hardware", "hashes", "report_sha256", "schema", "vendor"}), "release attestation")
    _require(attestation.get("schema") == ATTESTATION_SCHEMA, "release attestation schema differs")
    _require(report_sha256 is not None and attestation.get("report_sha256") == report_sha256, "release attestation is not bound to report bytes")
    hardware = _mapping(attestation.get("hardware"), "release attestation.hardware")
    _exact_keys(hardware, frozenset({"capability", "gpu", "multi_processor_count", "name", "total_memory"}), "release attestation.hardware")
    device = _mapping(report.get("device"), "device")
    expected_hardware = {"gpu": gpu, "name": device["name"], "capability": device["capability"], "total_memory": device["total_memory"], "multi_processor_count": device["multi_processor_count"]}
    _require(_same(hardware, expected_hardware), "release hardware attestation disagrees with report device")
    vendor = _mapping(attestation.get("vendor"), "release attestation.vendor")
    _exact_keys(vendor, frozenset({"git_dirty", "origin", "package_file_count", "package_sha256", "revision"}), "release attestation.vendor")
    actual = _mapping(_mapping(report.get("fla_checkout"), "fla_checkout").get("actual"), "fla_checkout.actual")
    expected_vendor = {key: actual[key] for key in ("git_dirty", "origin", "package_file_count", "package_sha256", "revision")}
    _require(_same(vendor, expected_vendor), "release vendor attestation disagrees with report FLA checkout")
    hashes = _mapping(attestation.get("hashes"), "release attestation.hashes")
    _exact_keys(hashes, frozenset({"hardware_sha256", "vendor_sha256"}), "release attestation.hashes")
    _require(hashes.get("hardware_sha256") == _json_digest(hardware) and hashes.get("vendor_sha256") == _json_digest(vendor), "release attestation component hash mismatch")
    _sha256_hex(attestation.get("report_sha256"), "release attestation.report_sha256")


def audit_compiled_step_report(
    report: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
    gpu: str,
    seed: int | None = None,
    require_release_attestation: bool = False,
    release_attestation: Mapping[str, Any] | None = None,
    report_sha256: str | None = None,
    campaign_manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Audit one direct model-only report without running GPU code."""

    _require(isinstance(report, Mapping), "report must be an object")
    _reject_nonfinite(report)
    _require(isinstance(gpu, str), "GPU selector must be a string")
    selected_gpu = SUPPORTED_GPU_ALIASES.get(gpu)
    _require(selected_gpu in SUPPORTED_GPUS, f"unsupported GPU selector {gpu!r}")
    _check_root_shape(report)
    config_seed = _int(_mapping(report.get("config"), "config").get("seed"), "config.seed")
    _require(config_seed in SUPPORTED_SEEDS, "config.seed is not one of the three fixed campaign seeds")
    if seed is not None:
        _require(type(seed) is int and seed == config_seed, "requested seed differs from report seed")
    config = _check_config(report, config_seed)
    if "timing_subartifact" in report:
        timing_subartifact = _mapping(report.get("timing_subartifact"), "timing_subartifact")
        _require(timing_subartifact.get("seed") == config_seed, "timing sub-artifact seed differs from config.seed")
    pre = _mapping(report.get("compiled_step_runtime_preflight"), "compiled_step_runtime_preflight")
    _require(pre.get("gpu_selector") == selected_gpu, "requested GPU differs from runtime preflight")
    timing, arrays = _check_model_timing(report, config_seed)
    stats = _check_statistics(timing, arrays, config_seed)
    root = _resolve_root(repo_root)
    campaign_manifest_digest = None
    if isinstance(campaign_manifest, (str, Path)):
        manifest_path = Path(campaign_manifest).expanduser()
        try:
            campaign_manifest_digest = _sha256_bytes(manifest_path.read_bytes())
        except (OSError, UnicodeError) as exc:
            raise CompiledStepAuditError(f"cannot read campaign manifest bytes: {exc}") from exc
        campaign_manifest = read_campaign_manifest(manifest_path)
    elif campaign_manifest is not None:
        _require(isinstance(campaign_manifest, Mapping), "campaign manifest must be an object or JSON path")
    _check_runtime_and_hashes(report, config, root, selected_gpu, campaign_manifest=campaign_manifest, campaign_manifest_digest=campaign_manifest_digest)
    attestation_verified = False
    if release_attestation is not None:
        _check_attestation(release_attestation, report, report_sha256, selected_gpu)
        attestation_verified = True
    elif require_release_attestation:
        raise CompiledStepAuditError("release promotion requires a separately hashed hardware/vendor attestation")
    blockers = ["model_only_subartifact", "correctness_not_run", "operator_timings_not_run", "full_suite_claim_false"]
    if require_release_attestation is False and not attestation_verified:
        blockers.append("release_attestation_not_supplied")
    return {
        "schema": AUDIT_SCHEMA,
        "status": "timing_verified",
        "timing_verified": True,
        "release_promotable": False,
        "release_blockers": blockers,
        "gpu": selected_gpu,
        "seed": config_seed,
        "mode": "full",
        "sequence": EXPECTED_MODEL_CONFIG["sequence"],
        "rounds": EXPECTED_ROUNDS,
        "warmup": EXPECTED_WARMUP,
        "timing_rows": len(timing["raw_samples"]),
        "timing_means_ms": {
            "candidate": sum(arrays["kernel_rank_1024"]) / EXPECTED_ROUNDS,
            "baseline": sum(arrays["fla_triton_compile_standard_rank_1024"]) / EXPECTED_ROUNDS,
        },
        "statistics": stats,
        "attestation_verified": attestation_verified,
        "report_sha256": report_sha256,
    }


def build_hero_projection(
    report_paths: Mapping[str, Mapping[int, str | Path]],
    *,
    repo_root: str | Path = ".",
    campaign_manifest: Mapping[str, Any] | str | Path | None = None,
    release_attestation_paths: Mapping[str, Mapping[int, str | Path]] | None = None,
) -> dict[str, Any]:
    """Audit exactly six reports and emit the renderer's compact projection.

    ``report_paths`` is keyed by the semantic GPU selector (``H100`` or
    ``B200``), then by one of the three fixed integer seeds.  Paths are only
    inputs to :func:`audit_path`; their filenames are never interpreted.  The
    output contains no raw rows and no pooled interval.  Its source digest is
    a deterministic hash of all six audited report byte digests plus their
    semantic identities, so changing any source report changes the projection
    provenance.
    """

    _require(isinstance(report_paths, Mapping), "hero report paths must be an object")
    _require(set(report_paths) == set(SUPPORTED_GPUS), "hero projection requires exactly H100 and B200 reports")
    if release_attestation_paths is not None:
        _require(
            isinstance(release_attestation_paths, Mapping)
            and set(release_attestation_paths) == set(SUPPORTED_GPUS),
            "hero attestation paths require exactly H100 and B200 records",
        )
    measurements: dict[str, dict[str, Any]] = {}
    source_records: list[dict[str, Any]] = []
    for gpu in SUPPORTED_GPUS:
        paths = _mapping(report_paths.get(gpu), f"hero report paths.{gpu}")
        _require(set(paths) == set(SUPPORTED_SEEDS), f"hero projection requires all three seeds for {gpu}")
        attestation_paths = None
        if release_attestation_paths is not None:
            attestation_paths = _mapping(
                release_attestation_paths.get(gpu), f"hero attestation paths.{gpu}"
            )
            _require(
                set(attestation_paths) == set(SUPPORTED_SEEDS),
                f"hero attestation paths require all three seeds for {gpu}",
            )
        ratios: list[dict[str, Any]] = []
        mean_values: list[dict[str, float]] = []
        for seed in SUPPORTED_SEEDS:
            path = paths.get(seed)
            _require(isinstance(path, (str, Path)), f"hero report path {gpu}/{seed} is invalid")
            attestation_path = None if attestation_paths is None else attestation_paths.get(seed)
            if attestation_paths is not None:
                _require(
                    isinstance(attestation_path, (str, Path)),
                    f"hero attestation path {gpu}/{seed} is invalid",
                )
            audit_kwargs = {
                "repo_root": repo_root,
                "gpu": gpu,
                "seed": seed,
                "campaign_manifest": campaign_manifest,
            }
            if attestation_path is not None:
                audit_kwargs.update(
                    {
                        "release_attestation_path": attestation_path,
                        "require_release_attestation": True,
                    }
                )
            result = audit_path(path, **audit_kwargs)
            _require(result.get("status") == "timing_verified" and result.get("timing_verified") is True and result.get("release_promotable") is False, f"audit {gpu}/{seed} is not a model-only timing verification")
            if attestation_path is not None:
                _require(
                    result.get("attestation_verified") is True,
                    f"audit {gpu}/{seed} did not verify its release attestation",
                )
            current_means = _mapping(result.get("timing_means_ms"), f"audit {gpu}/{seed}.timing_means_ms")
            _require(set(current_means) == {"candidate", "baseline"}, f"audit {gpu}/{seed} means are incomplete")
            means = {key: _finite(current_means.get(key), f"audit {gpu}/{seed}.timing_means_ms.{key}", positive=True) for key in ("candidate", "baseline")}
            mean_values.append(means)
            comparison = result["statistics"]["kernel_rank_1024_over_fla_triton_compile_standard_rank_1024"]
            ratio = _finite(comparison["ratio"], f"audit {gpu}/{seed}.ratio", positive=True)
            ratio_item = {
                "seed": SEED_LABELS[seed],
                "ratio": ratio,
                "ci_low": _finite(comparison["ci_low"], f"audit {gpu}/{seed}.ci_low", positive=True),
                "ci_high": _finite(comparison["ci_high"], f"audit {gpu}/{seed}.ci_high", positive=True),
            }
            _require(ratio_item["ci_low"] <= ratio <= ratio_item["ci_high"], f"audit {gpu}/{seed} CI does not contain estimate")
            ratios.append(ratio_item)
            source_record = {
                "gpu": gpu,
                "seed": seed,
                "report_sha256": _sha256_hex(result.get("report_sha256"), f"audit {gpu}/{seed}.report_sha256"),
            }
            if attestation_path is not None:
                source_record["attestation_sha256"] = _sha256_bytes(
                    Path(attestation_path).expanduser().read_bytes()
                )
            source_records.append(source_record)
        _require(len(mean_values) == len(SUPPORTED_SEEDS), f"hero projection has no timing means for {gpu}")
        measurements[gpu] = {
            "absolute_ms": {
                "attnres": sum(item["candidate"] for item in mean_values) / len(mean_values),
                "fla_ckpt1": sum(item["baseline"] for item in mean_values) / len(mean_values),
            },
            "ratios": ratios,
        }
    source_digest = _json_digest(source_records)
    devices = {"H100 SXM": measurements["H100"], "B200": measurements["B200"]}
    return {
        "schema": PROJECTION_SCHEMA,
        "status": "audited",
        "provenance": {
            "generator": "benchmarks/audit_compiled_step.py",
            "audit_schema": AUDIT_SCHEMA,
            "audit_status": "passed",
            "source_digest": source_digest,
        },
        "campaign": {
            "mode": "full",
            "dtype": "bf16",
            "rank_relation": "R=D",
            "timing_method": "cuda_graph",
            "baseline": "native FLA Triton checkpoint 1",
            "optimizer": "AdamW (fused, capturable)",
            "rounds": EXPECTED_ROUNDS,
            "warmup": EXPECTED_WARMUP,
            "confidence": EXPECTED_CONFIDENCE,
            "seeds": [SEED_LABELS[seed] for seed in SUPPORTED_SEEDS],
            "schedule": "ordered source-list; 48 reads (S=2…49); deterministic order/reverse ABBA",
            "model": {
                "layers": EXPECTED_MODEL_CONFIG["layers"],
                "width": EXPECTED_MODEL_CONFIG["width"],
                "heads": EXPECTED_MODEL_CONFIG["heads"],
                "ffn": EXPECTED_MODEL_CONFIG["ffn"],
                "batch": EXPECTED_MODEL_CONFIG["batch"],
                "sequence": EXPECTED_MODEL_CONFIG["sequence"],
                "vocab": EXPECTED_MODEL_CONFIG["vocab"],
                "block_count": EXPECTED_MODEL_CONFIG["block_count"],
            },
        },
        "devices": devices,
    }


def write_hero_projection(
    report_paths: Mapping[str, Mapping[int, str | Path]],
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
    campaign_manifest: Mapping[str, Any] | str | Path | None = None,
    release_attestation_paths: Mapping[str, Mapping[int, str | Path]] | None = None,
) -> dict[str, Any]:
    """Audit six reports, write a deterministic projection, and return it."""

    projection = build_hero_projection(
        report_paths,
        repo_root=repo_root,
        campaign_manifest=campaign_manifest,
        release_attestation_paths=release_attestation_paths,
    )
    path = Path(output_path).expanduser()
    _require(path.suffix.lower() == ".json", "hero projection output must be JSON")
    path.write_text(json.dumps(projection, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return projection


def audit_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`audit_compiled_step_report`."""

    return audit_compiled_step_report(*args, **kwargs)


def audit_path(
    report_path: str | Path,
    *,
    repo_root: str | Path = ".",
    gpu: str,
    seed: int | None = None,
    release_attestation_path: str | Path | None = None,
    require_release_attestation: bool = False,
    campaign_manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    path = Path(report_path).expanduser()
    report = read_report(path)
    digest = _sha256_bytes(path.read_bytes())
    attestation = read_report(release_attestation_path) if release_attestation_path is not None else None
    return audit_compiled_step_report(report, repo_root=repo_root, gpu=gpu, seed=seed, require_release_attestation=require_release_attestation, release_attestation=attestation, report_sha256=digest, campaign_manifest=campaign_manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--repo", type=Path, default=Path("."), help="local checkout to hash and compare")
    parser.add_argument("--gpu", choices=SUPPORTED_GPUS, required=True)
    parser.add_argument("--seed", type=int, choices=SUPPORTED_SEEDS)
    parser.add_argument("--release-attestation", type=Path)
    parser.add_argument("--require-release-attestation", action="store_true")
    parser.add_argument("--campaign-manifest", type=Path, help="sealed source/revision manifest for the campaign")
    args = parser.parse_args(argv)
    try:
        result = audit_path(args.report, repo_root=args.repo, gpu=args.gpu, seed=args.seed, release_attestation_path=args.release_attestation, require_release_attestation=args.require_release_attestation, campaign_manifest=args.campaign_manifest)
    except (CompiledStepAuditError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTESTATION_SCHEMA",
    "AUDIT_SCHEMA",
    "BF16_MAX_FINITE",
    "CAMPAIGN_SCHEMA",
    "EXPECTED_FAIR_KERNELS",
    "EXPECTED_FLA_ADAPTER_SHA256",
    "EXPECTED_FLA_ORIGIN",
    "EXPECTED_FLA_PACKAGE_SHA256",
    "EXPECTED_FLA_REVISION",
    "EXPECTED_FROZEN_MANIFEST_SHA256",
    "EXPECTED_GPU",
    "EXPECTED_MODEL_CONFIG",
    "EXPECTED_MODEL_SHA256",
    "EXPECTED_REPO_HEAD",
    "EXPECTED_RMS_WEIGHT_LIFECYCLE",
    "EXPECTED_RUNNER_SHA256",
    "EXPECTED_RUNTIME",
    "EXPECTED_WRAPPER_SHA256",
    "MANIFEST_SCHEMA",
    "PROJECTION_SCHEMA",
    "REPORT_SCHEMA",
    "RUNTIME_PREFLIGHT_SCHEMA",
    "SUPPORTED_GPUS",
    "SUPPORTED_SEEDS",
    "CompiledStepAuditError",
    "audit_compiled_step_report",
    "audit_path",
    "audit_report",
    "build_hero_projection",
    "expected_model_schedule",
    "read_campaign_manifest",
    "read_report",
    "strict_json_loads",
    "write_hero_projection",
]
