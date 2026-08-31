"""Reproducible, fail-closed runner for the BF16 compiled-step campaign.

The normal benchmark entry point is intentionally broad.  This module seals
one six-job comparison around that entry point: the sliced Full candidate at
``R=D=1024`` against the native FLA Triton checkpoint-1 arm.  It performs all
identity checks before importing the CUDA benchmark stack, then lets
``benchmarks.run.run_suite`` own model construction, qualification, CUDA
Graph capture, and paired replay timing.

The CUDA event surrounds only ``CapturedTrainingStep.replay``.  Input copies,
hashes, compilation, optimizer construction, warmups, graph capture, and
qualification happen before the event or after synchronization.  The runner
does not add any timed work.  Reports are audited before they are atomically
published; an incomplete or provenance-mismatched result cannot replace an
existing output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .audit_compiled_step import (
    AUDIT_SCHEMA,
    CONFIG_KEYS,
    EXPECTED_FLA_ORIGIN,
    EXPECTED_FLA_PACKAGE_SHA256,
    EXPECTED_FLA_REVISION,
    EXPECTED_GPU,
    EXPECTED_KERNEL_PATHS,
    EXPECTED_MODEL_CONFIG,
    EXPECTED_RMS_WEIGHT_CAMPAIGN,
    EXPECTED_ROUNDS,
    EXPECTED_RUNTIME,
    MANIFEST_SCHEMA,
    SUPPORTED_GPUS,
    SUPPORTED_SEEDS,
)

CONFIG_SCHEMA = "attnres.production_ladder_config.v1"
CAMPAIGN_SCHEMA = "attnres.compiled_step_campaign.v2"
PREFLIGHT_SCHEMA = "attnres.compiled_step_runtime_preflight.v2"
AGGREGATE_SCHEMA = "attnres.compiled_step_campaign.aggregate.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "compiled_step_campaign.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "compiled_step_campaign_manifest.json"
BASE_REVISION = "81dffbfeb0f84470513e846e3df8080e8ffb563d"
EXPECTED_FROZEN_COUNT = 62
SOURCE_PATHS = (
    "benchmarks/run.py",
    "benchmarks/training_graph.py",
    "benchmarks/fla_compile.py",
    "benchmarks/model.py",
    "benchmarks/competitors.py",
    "src/attnres/_kernels/fixed_tail.py",
    "src/attnres/_kernels/fixed_tail_sources.py",
    "src/attnres/_kernels/fla_full_sources.py",
)
FILENAME_RE = re.compile(r"^(?:h100|b200)-full-seed(?:20260827|20260903|20260911)\.json$")
TIMING_CONTRACT = {
    "event_call": "CUDA events around CapturedTrainingStep.replay only",
    "inside_cuda_event": [
        "BF16 autocast",
        "zero_grad",
        "model forward",
        "cross_entropy loss",
        "backward",
        "gradient accumulation",
        "AdamW optimizer.step",
    ],
    "outside_cuda_event": [
        "input copies via CapturedTrainingStep.copy_inputs",
        "logical input ID assignment; no tensor byte hashing",
        "device-to-host validation and report serialization",
        "torch.compile for the model and loss",
        "optimizer construction and state initialization",
        "warmup replays",
        "CUDA Graph capture",
        "CUDA event synchronization",
        "pre-timing complete-step and changed-input graph gates",
    ],
    "per_round_numerical_checks": False,
    "timed_tensor_hashing": False,
    "timed_input_copy": False,
    "timed_qualification": False,
}
RMS_WEIGHT_LIFECYCLE = {
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


class CampaignError(ValueError):
    """Raised when a campaign input, preflight, or report is unsafe."""


class CampaignPreflightError(CampaignError):
    """Raised when host/runtime identity does not match the sealed campaign."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CampaignError(f"cannot inspect {label}: {path}: {exc}") from exc
    _require(not stat.S_ISLNK(info.st_mode) and stat.S_ISREG(info.st_mode), f"{label} must be a regular file: {path}")
    return path


def _absolute_leaf(path: str | os.PathLike[str]) -> Path:
    """Make a path absolute while preserving the final symlink for lstat."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent.resolve() / candidate.name


def _safe_relative(root: Path, relative: str, label: str) -> Path:
    _require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), f"{label} path is unsafe")
    candidate = root / relative
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise CampaignError(f"{label} path escapes checkout: {relative!r}") from exc
    return _regular_file(candidate, label)


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CampaignPreflightError(f"cannot inspect checkout git state: {' '.join(args)}") from exc
    return completed.stdout.strip()


def _git_ancestor(root: Path, base: str, head: str) -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", base, head],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def normalize_gpu(value: str) -> str:
    aliases = {"H100!": "H100", "H100": "H100", "B200": "B200"}
    try:
        return aliases[value]
    except KeyError as exc:
        raise CampaignError(f"unsupported GPU selector {value!r}; choose H100 or B200") from exc


def _read_sealed(path: str | os.PathLike[str], label: str) -> dict[str, Any]:
    candidate = _absolute_leaf(path)
    _regular_file(candidate, label)
    try:
        from .audit_compiled_step import strict_json_loads

        value = strict_json_loads(candidate.read_text(encoding="utf-8"), str(candidate))
    except (OSError, UnicodeError) as exc:
        raise CampaignError(f"cannot read {label}: {candidate}: {exc}") from exc
    _require(isinstance(value, Mapping), f"{label} must contain an object")
    return dict(value)


def _expected_campaign(seed: int) -> dict[str, Any]:
    return {
        "schema": CAMPAIGN_SCHEMA,
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
        "fla_unit_rms_weight": dict(EXPECTED_RMS_WEIGHT_CAMPAIGN),
    }


def validate_campaign_config(config: Mapping[str, Any], *, seed: int | None = None) -> dict[str, Any]:
    """Validate the exact per-job suite configuration before any model import."""

    _require(isinstance(config, Mapping), "campaign config must be an object")
    _require(set(config) == set(CONFIG_KEYS), "campaign config fields are not exact")
    expected_scalars = {
        "schema": CONFIG_SCHEMA,
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
        "model_warmup": 10,
        "model_rounds": 120,
        "accumulation": 1,
        "lr": 0.0003,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "bootstrap_samples": 20_000,
        "model_profile": False,
        "model_progress": True,
    }
    for key, value in expected_scalars.items():
        _require(_same(config.get(key), value), f"campaign config.{key} differs from sealed contract")
    current_seed = config.get("seed")
    _require(type(current_seed) is int and current_seed in SUPPORTED_SEEDS, "campaign config.seed is not one of the fixed seeds")
    if seed is not None:
        _require(type(seed) is int and current_seed == seed, "campaign config seed differs from requested job")
    model = config.get("model_config")
    _require(_same(model, EXPECTED_MODEL_CONFIG), "campaign model_config differs from exact BF16 Full geometry")
    campaign = config.get("compiled_step_campaign")
    _require(_same(campaign, _expected_campaign(int(current_seed))), "compiled_step_campaign metadata differs from v2 contract")
    production = config.get("production_ladder")
    _require(isinstance(production, Mapping), "campaign production_ladder is missing")
    _require(production.get("state_protocol") == "canonical_implicit_max_rank_v1", "campaign state protocol differs")
    _require(production.get("input_protocol") == "shared_per_sample_timed_inputs_v1", "campaign input protocol differs")
    _require(production.get("source_layout") == "list" and production.get("cached_block") is False, "campaign source contract differs")
    _require(production.get("fla_anchor") == {"checkpoint_level": 1, "implementation": "triton", "rank": 1024, "scope": "R=D anchor only"}, "campaign FLA anchor differs")
    checkout = production.get("fla_checkout")
    _require(checkout == {"environment": "ATTNRES_FLA_DIR", "layout": "clean checkout containing fla/", "revision": EXPECTED_FLA_REVISION, "package_sha256": EXPECTED_FLA_PACKAGE_SHA256, "required_clean": True}, "campaign FLA checkout pin differs")
    candidate = production.get("resident_candidate")
    _require(isinstance(candidate, Mapping) and candidate.get("fixed_tail_sources_sha256") == _sha256_placeholder("fixed_tail_sources"), "campaign fixed-tail source pin differs")
    _require(isinstance(config.get("vendor_root"), (str, type(None))), "campaign vendor_root must be null or a path")
    return dict(config)


def _sha256_placeholder(name: str) -> str:
    """Return the active source digest used in the campaign metadata."""

    if name == "fixed_tail_sources":
        return "1373614c93d7291ad96697b1b8ff627120590b75f63f7e38bd65d50b19fcfb4a"
    if name == "fla_full_sources":
        return "8749c72c4714145214e33e8bc7d37f57b47a79b67f2e83044205db72cda416fa"
    raise CampaignError(f"unknown source digest {name!r}")


def validate_campaign_manifest(
    manifest: Mapping[str, Any],
    *,
    root: str | os.PathLike[str] = PROJECT_ROOT,
    config: Mapping[str, Any] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate the sealed six-job matrix and every preflight identity pin."""

    root_path = Path(root).expanduser().resolve()
    _require(isinstance(manifest, Mapping), "campaign manifest must be an object")
    expected_keys = {"config_path", "config_sha256", "fla", "gpus", "jobs", "repo_base_revision", "rms_weight_lifecycle", "runtime", "schema", "seeds", "source_sha256", "timing_contract"}
    _require(set(manifest) == expected_keys, "campaign manifest fields are not exact")
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "campaign manifest schema differs")
    _require(manifest.get("config_path") == "configs/compiled_step_campaign.json", "campaign manifest config path differs")
    _require(manifest.get("repo_base_revision") == BASE_REVISION, "campaign manifest base revision differs")
    _require(manifest.get("runtime") == EXPECTED_RUNTIME, "campaign manifest runtime differs")
    _require(manifest.get("seeds") == list(SUPPORTED_SEEDS), "campaign manifest seeds differ")
    _require(manifest.get("rms_weight_lifecycle") == RMS_WEIGHT_LIFECYCLE, "campaign manifest RMS-weight lifecycle differs")
    _require(manifest.get("timing_contract") == TIMING_CONTRACT, "campaign manifest timing contract differs")
    _require(manifest.get("fla") == {"origin": EXPECTED_FLA_ORIGIN, "package_sha256": EXPECTED_FLA_PACKAGE_SHA256, "revision": EXPECTED_FLA_REVISION, "source_sha256": {"fla/ops/attnres/backends/gluon.py": "f8f163fb7ebb8d035236674aeb668483812fb4e9a29572ed2ae937c626990190", "fla/ops/attnres/fused.py": "0e4683ab291086a9c3919d7352e2a998112973c94f5363e58f76ea7efea114f3"}}, "campaign FLA identity differs")
    expected_gpus = {gpu: {"capability": EXPECTED_GPU[gpu]["capability"], "name": EXPECTED_GPU[gpu]["name"], "selector": gpu} for gpu in SUPPORTED_GPUS}
    _require(manifest.get("gpus") == expected_gpus, "campaign GPU matrix differs")
    source = manifest.get("source_sha256")
    _require(isinstance(source, Mapping) and set(source) == set(SOURCE_PATHS), "campaign source hash coverage differs")
    for relative in SOURCE_PATHS:
        expected = source.get(relative)
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"campaign source hash is malformed for {relative}")
        actual = sha256_file(_safe_relative(root_path, relative, "campaign source"))
        _require(expected == actual, f"campaign source hash differs for {relative}")
    config_candidate = _absolute_leaf(config_path) if config_path is not None else _safe_relative(root_path, str(manifest["config_path"]), "campaign config")
    _regular_file(config_candidate, "campaign config")
    _require(manifest.get("config_sha256") == sha256_file(config_candidate), "campaign config digest differs from sealed bytes")
    if config is not None:
        validate_campaign_config(config)
    jobs = manifest.get("jobs")
    _require(isinstance(jobs, list) and len(jobs) == 6, "campaign manifest must contain exactly six jobs")
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(jobs):
        _require(isinstance(row, Mapping) and set(row) == {"filename", "gpu", "seed"}, f"campaign manifest.jobs[{index}] fields are not exact")
        gpu = normalize_gpu(row.get("gpu"))
        seed = row.get("seed")
        filename = row.get("filename")
        _require(type(seed) is int and seed in SUPPORTED_SEEDS, f"campaign manifest.jobs[{index}] seed is invalid")
        _require(isinstance(filename, str) and FILENAME_RE.fullmatch(filename) is not None, f"campaign manifest.jobs[{index}] filename is invalid")
        _require((gpu, seed) not in seen, f"campaign manifest.jobs[{index}] duplicates a job")
        seen.add((gpu, seed))
    _require(seen == {(gpu, seed) for gpu in SUPPORTED_GPUS for seed in SUPPORTED_SEEDS}, "campaign manifest job matrix differs")
    return dict(manifest)


def load_sealed_campaign(
    *,
    root: str | os.PathLike[str] = PROJECT_ROOT,
    config_path: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    root_path = Path(root).expanduser().resolve()
    config_file = _absolute_leaf(config_path) if config_path is not None else root_path / DEFAULT_CONFIG.relative_to(PROJECT_ROOT)
    manifest_file = _absolute_leaf(manifest_path) if manifest_path is not None else root_path / DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)
    config = _read_sealed(config_file, "campaign config")
    manifest = _read_sealed(manifest_file, "campaign manifest")
    validate_campaign_config(config)
    validate_campaign_manifest(manifest, root=root_path, config=config, config_path=config_file)
    _require(manifest_file == root_path / manifest["config_path"].replace("compiled_step_campaign.json", "compiled_step_campaign_manifest.json"), "campaign manifest path is not sealed")
    return config, manifest, config_file, manifest_file


def build_job_config(config: Mapping[str, Any], seed: int, *, vendor_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Create one job config without changing any sealed campaign field."""

    validate_campaign_config(config)
    _require(type(seed) is int and seed in SUPPORTED_SEEDS, "job seed is not one of the fixed campaign seeds")
    job = copy.deepcopy(dict(config))
    job["seed"] = seed
    job["compiled_step_campaign"]["seed"] = seed
    if vendor_root is not None:
        job["vendor_root"] = str(Path(vendor_root).expanduser().resolve())
    validate_campaign_config(job, seed=seed)
    return job


def _nvidia_smi() -> dict[str, str]:
    query = "name,uuid,driver_version,pstate,pci.bus_id,power.limit,clocks.max.sm,memory.total"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CampaignPreflightError("nvidia-smi preflight failed") from exc
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    _require(len(rows) == 1, "campaign requires exactly one visible GPU in nvidia-smi")
    values = [part.strip() for part in rows[0].split(",")]
    keys = query.split(",")
    _require(len(values) == len(keys), "nvidia-smi returned an unexpected CSV shape")
    return dict(zip(keys, values, strict=True))


def _source_digests(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        result[relative] = sha256_file(_safe_relative(root, relative, "campaign source"))
    return result


def runtime_preflight(
    *,
    root: str | os.PathLike[str],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    gpu: str,
    vendor_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Verify checkout, vendor, runtime, hardware, and timing identity first."""

    started = time.time()
    root_path = Path(root).expanduser().resolve()
    selected_gpu = normalize_gpu(gpu)
    try:
        validate_campaign_config(config)
        validate_campaign_manifest(manifest, root=root_path, config=config, config_path=config_path)
        _require(_git(root_path, "status", "--porcelain", "--untracked-files=all") == "", "checkout is dirty; run from the sealed campaign commit")
        head = _git(root_path, "rev-parse", "HEAD")
        _require(_git_ancestor(root_path, BASE_REVISION, head), "checkout HEAD is not descended from the sealed campaign base")
        source_sha256 = _source_digests(root_path)
        runner_sha256 = sha256_file(_safe_relative(root_path, "benchmarks/run.py", "runner"))
        kernel_sha256 = {relative: source_sha256[relative] for relative in EXPECTED_KERNEL_PATHS}
        frozen_manifest_sha256 = sha256_file(_safe_relative(root_path, "validation/frozen.json", "frozen manifest"))
        config_sha256 = sha256_file(_regular_file(_absolute_leaf(config_path), "campaign config"))
        manifest_sha256 = sha256_file(_regular_file(_absolute_leaf(manifest_path), "campaign manifest"))
        _require(config_sha256 == manifest["config_sha256"], "sealed config digest differs from manifest")
        _require(selected_gpu in SUPPORTED_GPUS, "unsupported campaign GPU")

        from .fla_checkout import verify_runtime_fla_config

        configured = vendor_root if vendor_root is not None else config.get("vendor_root")
        verification = verify_runtime_fla_config(config, project_root=root_path, configured=configured)
        _require(verification.get("status") == "verified", f"FLA checkout preflight failed: {verification.get('error', verification)}")
        actual_fla = verification.get("actual")
        _require(isinstance(actual_fla, Mapping), "FLA preflight did not return actual checkout metadata")
        _require(actual_fla.get("revision") == EXPECTED_FLA_REVISION and actual_fla.get("package_sha256") == EXPECTED_FLA_PACKAGE_SHA256 and actual_fla.get("origin") == EXPECTED_FLA_ORIGIN and actual_fla.get("git_dirty") is False, "FLA checkout identity differs from sealed manifest")

        # CUDA/Triton are imported only after every repository and vendor byte
        # check above has passed.  This keeps setup failures from constructing
        # a model or compiling a kernel with an unsealed checkout.
        try:
            import torch
            import triton
        except Exception as exc:
            raise CampaignPreflightError("pinned Torch/Triton runtime could not be imported") from exc
        _require(str(torch.__version__) == EXPECTED_RUNTIME["torch"], f"Torch runtime must be {EXPECTED_RUNTIME['torch']}")
        _require(str(torch.version.cuda) == EXPECTED_RUNTIME["cuda_runtime"], f"CUDA runtime must be {EXPECTED_RUNTIME['cuda_runtime']}")
        _require(str(triton.__version__) == EXPECTED_RUNTIME["triton"], f"Triton runtime must be {EXPECTED_RUNTIME['triton']}")
        _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "campaign requires exactly one available CUDA device")
        name = str(torch.cuda.get_device_name(0))
        capability = list(torch.cuda.get_device_capability(0))
        _require(selected_gpu in name and capability == EXPECTED_GPU[selected_gpu]["capability"], f"visible GPU does not match {selected_gpu}: {name} cc={capability}")
        smi = _nvidia_smi()
        finished = time.time()
        return {
            "schema": PREFLIGHT_SCHEMA,
            "status": "passed",
            "gpu_selector": selected_gpu,
            "gpu_name": name,
            "compute_capability": capability,
            "nvidia_smi": smi,
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "triton": str(triton.__version__),
            "repo_head": head,
            "repo_clean": True,
            "runner_sha256": runner_sha256,
            "fla_adapter_sha256": source_sha256["benchmarks/fla_compile.py"],
            "model_sha256": source_sha256["benchmarks/model.py"],
            "kernel_sha256": kernel_sha256,
            "source_sha256": source_sha256,
            "frozen_manifest_sha256": frozen_manifest_sha256,
            "fla_revision": str(actual_fla["revision"]),
            "fla_clean": actual_fla["git_dirty"] is False,
            "fla_package_sha256": str(actual_fla["package_sha256"]),
            "fla_origin": str(actual_fla["origin"]),
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "wrapper_sha256": sha256_file(Path(__file__).resolve()),
            "started_unix_s": started,
            "finished_unix_s": finished,
            "timed_tensor_hashing": False,
            "timed_input_copy": False,
            "timed_qualification": False,
            "fla_unit_rms_weight_lifecycle": EXPECTED_RMS_WEIGHT_CAMPAIGN["lifecycle"],
            "fla_fill_launches_inside_step": EXPECTED_RMS_WEIGHT_CAMPAIGN["fill_launches_inside_step"],
        }
    except CampaignPreflightError:
        raise
    except CampaignError as exc:
        raise CampaignPreflightError(str(exc)) from exc
    except Exception as exc:
        raise CampaignPreflightError(str(exc)) from exc


def _execution_complete(result: Mapping[str, Any]) -> bool:
    model = result.get("model_timings")
    return isinstance(model, Mapping) and model.get("status") == "complete" and model.get("failures") == [] and model.get("comparator_failures") == []


def attach_timing_subartifact(result: Mapping[str, Any], *, seed: int, blocked: bool = False) -> dict[str, Any]:
    """Attach an explicit model timing status without changing timed rows."""

    report = dict(result)
    model = result.get("model_timings")
    model_status = model.get("status", "not_run") if isinstance(model, Mapping) else "not_run"
    if blocked:
        status, phase = "blocked", "preflight"
    elif model_status == "complete":
        status, phase = "complete", "model_timings"
    elif model_status == "failed":
        status, phase = "failed", "model_timings"
    else:
        status, phase = "incomplete", "model_timings"
    report["timing_subartifact"] = {
        "model_status": model_status,
        "phase": phase,
        "release_promotable": False,
        "seed": seed,
        "status": status,
    }
    return report


def _failure_record(phase: str, message: str, *, error_type: str = "CampaignError") -> dict[str, Any]:
    return {"phase": phase, "error": {"type": error_type, "message": message}}


def blocked_report(config: Mapping[str, Any], *, seed: int, message: str, preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create a deterministic report when preflight prevents model creation."""

    report: dict[str, Any] = {
        "status": "failed",
        "compiled_step_execution_status": "blocked",
        "config": dict(config),
        "contract": {"status": "not_run"},
        "coverage": {"scope": "primary", "claims_full_suite": False},
        "environment": {},
        "device": {"requested": "cuda", "type": "cuda", "available": False, "count": 0},
        "fla_checkout": {"status": "not_run"},
        "comparators": {},
        "comparators_enabled": False,
        "correctness": {"status": "not_run", "reason": "preflight blocked", "failures": []},
        "operator_timings": {"status": "not_run", "reason": "preflight blocked", "failures": []},
        "model_timings": {"status": "not_run", "reason": "preflight blocked", "failures": [_failure_record("preflight", message)]},
        "failures": [_failure_record("preflight", message)],
        "protocol": {},
        "hashes": {},
        "source_hashes": {},
        "compiled_step_runtime_preflight": dict(preflight or {"schema": PREFLIGHT_SCHEMA, "status": "failed", "error": message}),
    }
    return attach_timing_subartifact(report, seed=seed, blocked=True)


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Atomically publish a finite JSON object, refusing symlink targets."""

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.parent.resolve() / target.name
    _require(target.suffix.lower() == ".json", "campaign output must have a .json suffix")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _regular_file(target, "campaign output")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
    return target


def _manifest_job(manifest: Mapping[str, Any], gpu: str, seed: int) -> Mapping[str, Any]:
    selected = normalize_gpu(gpu)
    for row in manifest["jobs"]:
        if normalize_gpu(row["gpu"]) == selected and row["seed"] == seed:
            return row
    raise CampaignError(f"manifest has no job for {selected}/{seed}")


def _run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    # Importing run.py imports Torch; keep this after runtime_preflight.
    from .run import run_suite

    value = run_suite(dict(config))
    _require(isinstance(value, Mapping), "benchmark runner returned a non-object report")
    return dict(value)


def run_job(
    *,
    gpu: str,
    seed: int,
    output: str | os.PathLike[str],
    root: str | os.PathLike[str] = PROJECT_ROOT,
    config_path: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
    audit: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one sealed GPU/seed job and publish it only after offline audit."""

    root_path = Path(root).expanduser().resolve()
    selected_gpu = normalize_gpu(gpu)
    _require(type(seed) is int and seed in SUPPORTED_SEEDS, "job seed is not one of the fixed campaign seeds")
    config, manifest, config_file, manifest_file = load_sealed_campaign(root=root_path, config_path=config_path, manifest_path=manifest_path)
    job_row = _manifest_job(manifest, selected_gpu, seed)
    # Preserve the final path component so atomic_write_json can reject a
    # pre-existing symlink instead of resolving through it and overwriting its
    # target.
    output_path = _absolute_leaf(output)
    _require(output_path.name == job_row["filename"], f"output filename must be {job_row['filename']!r} for {selected_gpu}/{seed}")
    job_config = build_job_config(config, seed, vendor_root=vendor_root)
    try:
        preflight = runtime_preflight(root=root_path, config=job_config, manifest=manifest, config_path=config_file, manifest_path=manifest_file, gpu=selected_gpu, vendor_root=vendor_root)
    except Exception as exc:  # noqa: BLE001 - a failed benchmark is an explicit failed report
        report = blocked_report(job_config, seed=seed, message=str(exc))
        atomic_write_json(output_path, report)
        return report
    try:
        report = _run_suite(job_config)
    except Exception as exc:  # noqa: BLE001 - a failed benchmark is an explicit failed report
        report = blocked_report(job_config, seed=seed, message=f"benchmark runner failed after preflight: {exc}", preflight=preflight)
        report["compiled_step_execution_status"] = "failed"
        report["timing_subartifact"]["status"] = "failed"
        atomic_write_json(output_path, report)
        return report
    report["compiled_step_execution_status"] = "complete" if _execution_complete(report) else "failed"
    report["compiled_step_runtime_preflight"] = preflight
    report = attach_timing_subartifact(report, seed=seed)
    if audit is None:
        from .audit_compiled_step import audit_compiled_step_report

        audit = audit_compiled_step_report
    audit(report, repo_root=root_path, gpu=selected_gpu, seed=seed, campaign_manifest=manifest_file)
    atomic_write_json(output_path, report)
    return report


def aggregate_campaign(
    *,
    reports_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
    root: str | os.PathLike[str] = PROJECT_ROOT,
    config_path: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Audit all six independent jobs and emit unpooled per-GPU/seed stats."""

    root_path = Path(root).expanduser().resolve()
    _config, manifest, config_file, manifest_file = load_sealed_campaign(root=root_path, config_path=config_path, manifest_path=manifest_path)
    reports_root = Path(reports_dir).expanduser().resolve()
    _require(reports_root.is_dir(), f"reports directory is not a directory: {reports_root}")
    from .audit_compiled_step import audit_path

    jobs: list[dict[str, Any]] = []
    per_gpu_seed: dict[str, dict[str, Any]] = {gpu: {} for gpu in SUPPORTED_GPUS}
    for row in manifest["jobs"]:
        gpu = normalize_gpu(row["gpu"])
        seed = int(row["seed"])
        report_path = reports_root / row["filename"]
        _regular_file(report_path, "job report")
        audited = audit_path(report_path, repo_root=root_path, gpu=gpu, seed=seed, campaign_manifest=manifest_file)
        _require(audited.get("status") == "timing_verified" and audited.get("release_promotable") is False, f"job audit is not a complete model timing sub-artifact: {gpu}/{seed}")
        means = dict(audited["timing_means_ms"])
        statistics = dict(audited["statistics"])
        record = {
            "gpu": gpu,
            "seed": seed,
            "report": row["filename"],
            "report_sha256": audited["report_sha256"],
            "timing_rows": audited["timing_rows"],
            "timing_means_ms": means,
            "statistics": statistics,
        }
        jobs.append(record)
        per_gpu_seed[gpu][str(seed)] = {
            "timing_means_ms": means,
            "statistics": statistics,
            "report": row["filename"],
            "report_sha256": audited["report_sha256"],
        }
    _require(len(jobs) == 6, "aggregate requires exactly six audited jobs")
    config_digest = sha256_file(config_file)
    manifest_digest = sha256_file(manifest_file)
    aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete",
        "timing_subartifact": {
            "model_status": "complete",
            "phase": "all_model_timings",
            "release_promotable": False,
            "seed": None,
            "status": "complete",
            "job_count": 6,
            "required_job_count": 6,
        },
        "campaign": {
            "mode": "full",
            "dtype": "bf16_autocast",
            "rank_relation": "R=D",
            "model": dict(EXPECTED_MODEL_CONFIG),
            "gpus": list(SUPPORTED_GPUS),
            "seeds": list(SUPPORTED_SEEDS),
            "warmup": 10,
            "rounds": EXPECTED_ROUNDS,
            "timing_method": "cuda_graph",
            "baseline": "native FLA Triton checkpoint 1",
            "candidate": "sliced Full AttnRes kernel",
        },
        "jobs": jobs,
        "per_gpu_seed": per_gpu_seed,
        "statistics_scope": "unpooled per GPU and seed; no cross-GPU or cross-seed pairing",
        "provenance": {
            "audit_schema": AUDIT_SCHEMA,
            "config_sha256": config_digest,
            "manifest_sha256": manifest_digest,
            "repo_base_revision": manifest["repo_base_revision"],
            "source_sha256": dict(manifest["source_sha256"]),
            "rms_weight_lifecycle": dict(manifest["rms_weight_lifecycle"]),
            "timing_contract": dict(manifest["timing_contract"]),
        },
    }
    atomic_write_json(output, aggregate)
    return aggregate


def _cli_run(args: argparse.Namespace) -> int:
    report = run_job(
        gpu=args.gpu,
        seed=args.seed,
        output=args.output,
        root=args.repo,
        config_path=args.config,
        manifest_path=args.manifest,
        vendor_root=args.vendor_root,
    )
    timing = report.get("timing_subartifact", {})
    print(json.dumps({"status": report.get("status"), "compiled_step_execution_status": report.get("compiled_step_execution_status"), "timing_subartifact": timing, "output": str(Path(args.output).expanduser().resolve())}, sort_keys=True), flush=True)
    return 0 if report.get("compiled_step_execution_status") == "complete" and timing.get("status") == "complete" else 1


def _cli_aggregate(args: argparse.Namespace) -> int:
    aggregate = aggregate_campaign(
        reports_dir=args.reports_dir,
        output=args.output,
        root=args.repo,
        config_path=args.config,
        manifest_path=args.manifest,
    )
    print(json.dumps({"status": aggregate["status"], "jobs": len(aggregate["jobs"]), "output": str(Path(args.output).expanduser().resolve())}, sort_keys=True), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one sealed GPU/seed job")
    run_parser.add_argument("--gpu", choices=("H100", "H100!", "B200"), required=True)
    run_parser.add_argument("--seed", type=int, choices=SUPPORTED_SEEDS, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument("--manifest", type=Path)
    run_parser.add_argument("--vendor-root", type=Path)
    run_parser.set_defaults(handler=_cli_run)
    aggregate_parser = subparsers.add_parser("aggregate", help="audit and aggregate all six jobs")
    aggregate_parser.add_argument("--reports-dir", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    aggregate_parser.add_argument("--config", type=Path)
    aggregate_parser.add_argument("--manifest", type=Path)
    aggregate_parser.set_defaults(handler=_cli_aggregate)
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (CampaignError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
