"""Sealed matched-competitor benchmark protocol.

This module is deliberately CPU-only.  It loads and validates the checked-in
protocol, emits deterministic paired orders and planned cells, and validates
raw rows without discarding failures.  A benchmark runner may use these
helpers before importing CUDA or Triton; this module never launches a GPU job.

The protocol is a *design contract*, not a claim that the requested H100 or
B200 measurements have already run.  Selection, input/state pairing, oracle
gates, timing counts, and failure handling are fixed in the JSON artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from .comparator_registry import capability_for, eligibility_for as _registry_eligibility


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "matched_competitor_benchmark.json"
SCHEMA_PATH = PROJECT_ROOT / "configs" / "matched_competitor_benchmark.schema.json"

CONFIG_SCHEMA = "attnres.matched_competitor_benchmark.v1"
# The config is selected against this protocol commit.  Keep the anchor in
# code as well as JSON so a temporary or hand-edited config cannot claim an
# unrelated historical revision while still passing the shape checks.
PROTOCOL_BASE_REVISION = "3f2da420d8dc1394c6331ba59c2a61eca276062b"
HARDWARE_ORDER = ("H100!", "B200")
DTYPES = ("bf16", "fp32")
SEEDS = (20260827, 20260903, 20260911)
LR_RANKS = (16, 64, 128, 512, 1024)
WARMUP_ROUNDS = 10
TIMED_ROUNDS = 120
BOOTSTRAP_SAMPLES = 20_000
CONFIDENCE = 0.95
PLATEAU_MARGIN = 0.01

# This is the byte hash of the checked-in JSON artifact.  ``load_config``
# checks it for the default path so a result cannot silently use a locally
# edited protocol.  Callers loading a deliberate temporary config can still
# pass a path and use ``validate_config`` directly.
CONFIG_SHA256 = "e479b724db63bac693ae4344747e94a83fb907aec70ed10309d56661e9aff040"

_FAILURE_STATUSES = frozenset(
    {"failed", "skipped_due_to_failure", "not_applicable"}
)
_ROW_STATUSES = frozenset({"ok", *_FAILURE_STATUSES})
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ProtocolError(ValueError):
    """Raised when a protocol/configuration or raw result violates the seal."""


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read protocol JSON {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"protocol JSON {target} must contain an object")
    return value


def _same(actual: Any, expected: Any) -> bool:
    """Compare JSON values while rejecting bool/int equivalence."""

    if type(actual) is not type(expected):
        return False
    if isinstance(actual, Mapping):
        return set(actual) == set(expected) and all(
            _same(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _same(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _expect(actual: Any, expected: Any, path: str) -> None:
    if not _same(actual, expected):
        raise ProtocolError(f"{path} must equal {expected!r}; got {actual!r}")


def _keys(value: Mapping[str, Any], required: Sequence[str], path: str) -> None:
    names = set(value)
    required_set = set(required)
    missing = sorted(required_set - names)
    extra = sorted(names - required_set)
    if missing:
        raise ProtocolError(f"{path} is missing keys: {', '.join(missing)}")
    if extra:
        raise ProtocolError(f"{path} has unexpected keys: {', '.join(extra)}")


_TOP_LEVEL_KEYS = (
    "$schema",
    "schema",
    "version",
    "status",
    "sealed",
    "description",
    "hardware_order",
    "gpus",
    "hardware",
    "runtime",
    "dtypes",
    "bf16",
    "fp32",
    "oracle",
    "model",
    "operator_cases",
    "ranks",
    "arms",
    "competitors",
    "comparison_families",
    "seeds",
    "warmup",
    "warmup_rounds",
    "rounds",
    "timed_rounds",
    "schedule",
    "statistics",
    "failure_policy",
    "gpu_launch",
    "launch_policy",
    "provenance",
)

_COMPETITOR_BASE = {
    "native_fla_triton_checkpoint1": {
        "implementation": "fla_native",
        "backend": "triton",
        "checkpoint": 1,
        "role": "eligible",
        "eligible_denominator": True,
        "qualification": "same_oracle_and_all_parameter_gradients",
        "source": "fla-org/flash-linear-attention",
    },
    "native_fla_gluon": {
        "implementation": "fla_native",
        "backend": "gluon",
        "checkpoint": None,
        "role": "conditional_eligible",
        "eligible_denominator": True,
        "qualification": "include_only_after_same_oracle_and_all_parameter_gradients",
        "source": "fla-org/flash-linear-attention",
    },
    "native_fla_triton_checkpoint0": {
        "implementation": "fla_native",
        "backend": "triton",
        "checkpoint": 0,
        "role": "diagnostic",
        "eligible_denominator": False,
        "qualification": "retain_known_gradient_failure",
        "source": "fla-org/flash-linear-attention",
    },
    "liger": {
        "implementation": "liger_native",
        "backend": "triton",
        "checkpoint": None,
        "role": "conditional_eligible",
        "eligible_denominator": True,
        "qualification": "include_only_after_same_oracle_and_all_parameter_gradients",
        "source": "linkedin/Liger-Kernel",
    },
    "catswe_phase1": {
        "implementation": "catswe_native",
        "backend": "triton",
        "checkpoint": None,
        "role": "external_comparator",
        "eligible_denominator": True,
        "qualification": "phase1_standard_operator_oracle_before_external_timing",
        "source": "catswe/flash-attn-res",
    },
    "manish_hydra_2p": {
        "implementation": "manish_hydra_2p",
        "backend": "triton",
        "checkpoint": None,
        "role": "small_d_panel",
        "eligible_denominator": True,
        "qualification": "small_d_standard_operator_and_block_panel_oracle",
        "source": "manishklach/attnres-kernel-lab",
    },
}

_COMPETITOR_ELIGIBILITY = {
    "native_fla_triton_checkpoint1": {
        "scope": "R=D Full and per-read Block",
        "predicate": "CUDA; R=D; 1<=S<=129; 1<=D<=8192",
        "status": "eligible",
    },
    "native_fla_gluon": {
        "scope": "R=D Full and per-read Block after independent qualification",
        "predicate": (
            "CUDA; R=D; 1<=S<=129; 1<=D<=8192; "
            "Gluon compile envelope BD=nextpow2(D)<=4096; "
            "S*BD<=262144; checkpoint1 static work=33*S*BD<=8650752"
        ),
        "status": "conditional",
    },
    "native_fla_triton_checkpoint0": {
        "scope": "diagnostic R=D Full and per-read Block only",
        "predicate": "CUDA; R=D; 1<=S<=129; 1<=D<=8192",
        "status": "diagnostic_only",
    },
    "liger": {
        "scope": "R=D; per-read Block and Full only when the cell S<=32",
        "predicate": "CUDA; R=D; 1<=S<=32; 1<=D<=8192",
        "status": "conditional",
    },
    "catswe_phase1": {
        "scope": "standard CUDA phase1 standard operator only",
        "predicate": (
            "CUDA BF16; R=D; 1<=S<=129; 1<=D<=8192; "
            "D is power-of-two; nextpow2(S)*D<=1048576; native phase1"
        ),
        "status": "external_comparator",
    },
    "manish_hydra_2p": {
        "scope": "small-D standard operator and Block panel only",
        "predicate": "CUDA; R=D; 1<=S<=129; 1<=D<=8192; native timing D<=256",
        "status": "small_d_panel_only",
    },
}

def _protocol_capability(name: str) -> dict[str, Any]:
    """Keep the sealed JSON capability surface compact but self describing."""

    capability = capability_for(name)
    keys = (
        "family",
        "adapter",
        "eligible_denominator",
        "rank_scope",
        "max_sources",
        "max_width",
        "timing_max_width",
        "max_program_elements",
        "compile_envelope",
        "requires_power_of_two_width",
        "dtypes",
        "modes",
        "schedules",
        "external_route",
        "model_scope",
        "supports_per_read_block",
        "block_scope",
        "cuda_required",
        "oracle",
        "revision",
        "tree",
        "origin",
        "source",
        "source_hashes",
        "package_sha256",
        "license",
        "license_sha256",
        "notice_sha256",
        "pyproject_sha256",
    )
    return {key: capability[key] for key in keys if key in capability}


_COMPETITORS = {
    name: {
        **entry,
        "capability": _protocol_capability(name),
        "eligibility": dict(_COMPETITOR_ELIGIBILITY[name]),
    }
    for name, entry in _COMPETITOR_BASE.items()
}

_COMPARISON_FAMILIES = {
    "primary": {
        "competitors": [
            "native_fla_triton_checkpoint1",
            "native_fla_gluon",
        ],
        "eligibility": "every_required_arm_has_120_ok_rows",
        "unqualified_competitor_policy": "incomplete_unavailable",
        "per_seed_gate": True,
        "pool_seeds": False,
        "pool_gpus": False,
    },
    "diagnostic": {
        "competitors": ["native_fla_triton_checkpoint0"],
        "eligibility": "never_in_primary_denominator",
        "unqualified_competitor_policy": "retain_failure_only",
        "per_seed_gate": False,
        "pool_seeds": False,
        "pool_gpus": False,
    },
    "liger": {
        "competitors": ["liger"],
        "eligibility": "R_equals_D_and_full_cells_require_S_at_most_32",
        "unqualified_competitor_policy": "incomplete_unavailable",
        "per_seed_gate": True,
        "pool_seeds": False,
        "pool_gpus": False,
    },
    "catswe_external": {
        "competitors": ["catswe_phase1"],
        "eligibility": "native_phase1_standard_operator_only_cached_schedule_excluded",
        "unqualified_competitor_policy": "external_route_incomplete_unavailable",
        "per_seed_gate": True,
        "pool_seeds": False,
        "pool_gpus": False,
    },
    "manish_small_d": {
        "competitors": ["manish_hydra_2p"],
        "eligibility": "standard_operator_and_block_panel_only_D_at_most_256",
        "unqualified_competitor_policy": "small_d_panel_incomplete_unavailable",
        "per_seed_gate": True,
        "pool_seeds": False,
        "pool_gpus": False,
    },
}


def _validate_tolerance(value: Any, expected: Mapping[str, float], path: str) -> None:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{path} must be an object")
    _keys(value, ("rtol", "atol"), path)
    _expect(value, dict(expected), path)


def _validate_operator_cases(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ProtocolError("operator_cases must be an object")
    _keys(value, ("smoke", "primary", "heldout"), "operator_cases")
    for scope in ("smoke", "primary", "heldout"):
        cases = value[scope]
        if not isinstance(cases, list) or not cases:
            raise ProtocolError(f"operator_cases.{scope} must be a nonempty list")
        for index, case in enumerate(cases):
            path = f"operator_cases.{scope}[{index}]"
            if not isinstance(case, Mapping):
                raise ProtocolError(f"{path} must be an object with S/N/D/R/dtype")
            _keys(case, ("S", "N", "D", "R", "dtype"), path)
            for name in ("S", "N", "D", "R"):
                if type(case[name]) is not int or case[name] < 1:
                    raise ProtocolError(f"{path}.{name} must be a positive integer")
            if case["R"] > case["D"]:
                raise ProtocolError(f"{path}.R must be no greater than {path}.D")
            if type(case["dtype"]) is not str or case["dtype"] not in DTYPES:
                raise ProtocolError(
                    f"{path}.dtype must be one of {list(DTYPES)!r}"
                )


def _validate_provenance(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ProtocolError("provenance must be an object")
    _keys(
        value,
        (
            "base_revision",
            "config_hash_is_content_addressed",
            "mutable_overrides",
            "results_may_not_modify_config",
            "selection_must_precede_results",
        ),
        "provenance",
    )
    revision = value["base_revision"]
    if not isinstance(revision, str) or _HEX40.fullmatch(revision) is None:
        raise ProtocolError("provenance.base_revision must be a 40-character SHA-1")
    _expect(revision, PROTOCOL_BASE_REVISION, "provenance.base_revision")
    _expect(
        value["config_hash_is_content_addressed"],
        True,
        "provenance.config_hash_is_content_addressed",
    )
    _expect(value["mutable_overrides"], [], "provenance.mutable_overrides")
    _expect(
        value["results_may_not_modify_config"],
        True,
        "provenance.results_may_not_modify_config",
    )
    _expect(
        value["selection_must_precede_results"],
        True,
        "provenance.selection_must_precede_results",
    )


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach one immutable matched-competitor config.

    The returned object is a deep copy, so adding runner metadata to it cannot
    mutate the sealed input.  Validation intentionally does not import
    ``torch`` or ``triton`` and does not inspect GPU availability.
    """

    if not isinstance(config, Mapping):
        raise TypeError("matched-competitor config must be a mapping")
    result = deepcopy(dict(config))
    _keys(result, _TOP_LEVEL_KEYS, "config")

    _expect(result["$schema"], "https://json-schema.org/draft/2020-12/schema", "$schema")
    _expect(result["schema"], CONFIG_SCHEMA, "schema")
    _expect(result["version"], 1, "version")
    _expect(result["status"], "sealed", "status")
    _expect(result["sealed"], True, "sealed")
    if not isinstance(result["description"], str) or not result["description"].strip():
        raise ProtocolError("description must be a nonempty string")

    _expect(result["hardware_order"], list(HARDWARE_ORDER), "hardware_order")
    _expect(result["gpus"], list(HARDWARE_ORDER), "gpus")
    hardware = result["hardware"]
    if not isinstance(hardware, Mapping):
        raise ProtocolError("hardware must be an object")
    _keys(hardware, HARDWARE_ORDER, "hardware")
    _expect(
        hardware["H100!"],
        {"modal_name": "H100!", "compute_capability": "sm90", "same_device_pairing": True},
        "hardware.H100!",
    )
    _expect(
        hardware["B200"],
        {"modal_name": "B200", "compute_capability": "sm100", "same_device_pairing": True},
        "hardware.B200",
    )

    runtime = result["runtime"]
    if not isinstance(runtime, Mapping):
        raise ProtocolError("runtime must be an object")
    _expect(
        runtime,
        {
            "pytorch": "2.13.0",
            "triton": "3.7.1",
            "same_runtime_for_all_arms": True,
            "runtime_selection_is_immutable": True,
        },
        "runtime",
    )
    _expect(result["dtypes"], list(DTYPES), "dtypes")
    _validate_tolerance(result["bf16"], {"rtol": 0.05, "atol": 0.05}, "bf16")
    _validate_tolerance(result["fp32"], {"rtol": 0.001, "atol": 0.0001}, "fp32")

    oracle = result["oracle"]
    if not isinstance(oracle, Mapping):
        raise ProtocolError("oracle must be an object")
    _expect(
        oracle,
        {
            "path": "validation/oracle.py",
            "entrypoint": "validation.oracle.oracle",
            "math_dtype": "fp32",
            "checks": ["output", "values_gradient", "query_gradient"],
            "must_pass_before_timing": True,
            "tolerances": {
                "bf16": {"rtol": 0.05, "atol": 0.05},
                "fp32": {"rtol": 0.001, "atol": 0.0001},
            },
        },
        "oracle",
    )

    _expect(
        result["model"],
        {
            "layers": 24,
            "width": 1024,
            "heads": 16,
            "ffn": 2816,
            "batch": 2,
            "sequence": 2048,
            "vocab": 32768,
            "block_count": 8,
            "source_layout": "list",
            "timing_scope": "complete_training_step",
            "optimizer": "capturable_adamw",
            "accumulation": 1,
        },
        "model",
    )
    _validate_operator_cases(result["operator_cases"])

    _expect(result["ranks"], list(LR_RANKS), "ranks")
    _expect(
        result["arms"],
        {
            "standard_operator": {
                "kind": "operator",
                "variant": "standard",
                "mode": "forward_backward",
                "rank": 1024,
                "rank_equals_width": True,
                "source_layout": "list",
            },
            "full": {
                "kind": "model",
                "variant": "full",
                "mode": "full",
                "rank": 1024,
                "source_layout": "list",
                "block_path": "public_attnres_per_read",
            },
            "block_per_read": {
                "kind": "model",
                "variant": "block",
                "mode": "block",
                "rank": 1024,
                "source_layout": "list",
                "block_path": "public_attnres_per_read",
            },
            "lr_ranks": {
                "kind": "model",
                "variant": "sliced",
                "mode": "full_and_block",
                "ranks": list(LR_RANKS),
                "source_layout": "list",
                "implicit_key": "last_R_value_coordinates",
                "value_width": "full_width",
            },
        },
        "arms",
    )
    _expect(result["competitors"], _COMPETITORS, "competitors")
    _expect(result["comparison_families"], _COMPARISON_FAMILIES, "comparison_families")

    _expect(result["seeds"], list(SEEDS), "seeds")
    _expect(result["warmup"], WARMUP_ROUNDS, "warmup")
    _expect(result["warmup_rounds"], WARMUP_ROUNDS, "warmup_rounds")
    _expect(result["rounds"], TIMED_ROUNDS, "rounds")
    _expect(result["timed_rounds"], TIMED_ROUNDS, "timed_rounds")

    _expect(
        result["schedule"],
        {
            "paired": True,
            "order": "balanced_forward_reverse_pairs",
            "round_order_rule": (
                "one deterministic arm permutation per cell; even rounds use it and "
                "odd rounds use its exact reverse"
            ),
            "shared_inputs_per_pair": True,
            "shared_inputs_across_arms": True,
            "same_initial_state_across_arms": True,
            "same_device_for_pair": True,
            "synchronize_device_before_measurement": True,
            "warmup_excluded_from_timing": True,
            "timing_boundary": {
                "operator": "operator_forward_backward",
                "model": "complete_compiled_training_step",
            },
            "order_selection_after_results": False,
        },
        "schedule",
    )
    _expect(
        result["statistics"],
        {
            "estimator": "simultaneous_paired_ratio_bootstrap",
            "ratio": "candidate_over_baseline",
            "confidence": CONFIDENCE,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "common_resample_indices": True,
            "familywise_scope": (
                "all planned comparisons within each "
                "GPU_seed_predeclared_competitor_family_cell"
            ),
            "per_seed_gate": True,
            "pool_seeds": False,
            "pool_gpus": False,
            "plateau_margin": PLATEAU_MARGIN,
            "comparison_plan": [
                {
                    "id": "full_over_standard_operator",
                    "baseline": "standard_operator",
                    "candidate": "full",
                },
                {
                    "id": "block_per_read_over_full",
                    "baseline": "full",
                    "candidate": "block_per_read",
                },
                {
                    "id": "lr_rank_over_standard_operator",
                    "baseline": "standard_operator",
                    "candidate": "lr_ranks",
                },
                {
                    "id": "lr_adjacent_rank_edges",
                    "baseline": "larger_lr_rank",
                    "candidate": "smaller_lr_rank",
                },
            ],
            "selection_after_results": False,
        },
        "statistics",
    )
    _expect(
        result["failure_policy"],
        {
            "retain_failures": True,
            "retain_raw_samples": True,
            "drop_failures": False,
            "drop_incomplete_cells": False,
            "missing_sample_is_failure": True,
            "statistics_require_complete_ok": True,
            "failure_statuses": ["failed", "skipped_due_to_failure", "not_applicable"],
            "skipped_rounds_are_recorded": True,
            "required_row_fields": [
                "seed",
                "gpu",
                "round_index",
                "order_index",
                "input_hash",
                "arm",
                "status",
                "latency_ms",
                "failure_phase",
                "failure_reason",
                "failure_at_round",
                "failure_at_order",
            ],
            "raw_artifact_format": "jsonl",
            "no_interpolation": True,
            "no_retry_of_unchanged_failure": True,
        },
        "failure_policy",
    )
    _expect(result["gpu_launch"], False, "gpu_launch")
    _expect(
        result["launch_policy"],
        {"enabled": False, "dry_run_only": True, "owner": "root", "agent_may_launch": False},
        "launch_policy",
    )
    _validate_provenance(result["provenance"])
    return result


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes for content addressing."""

    if not isinstance(value, Mapping):
        raise TypeError("canonical_json expects a mapping")
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def config_digest(config: Mapping[str, Any]) -> str:
    """Hash a config's canonical JSON representation."""

    return hashlib.sha256(canonical_json(config)).hexdigest()


def file_digest(path: str | Path) -> str:
    """Hash the exact bytes of a checked-in protocol artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(
    path: str | Path | None = None,
    *,
    verify_digest: bool = True,
) -> dict[str, Any]:
    """Load and validate the sealed config without importing GPU runtimes."""

    target = CONFIG_PATH if path is None else Path(path)
    if verify_digest and target.resolve() == CONFIG_PATH.resolve():
        actual = file_digest(target)
        if actual != CONFIG_SHA256:
            raise ProtocolError(
                f"sealed config digest mismatch: expected {CONFIG_SHA256}, got {actual}"
            )
    return validate_config(_read_json(target))


def load_schema(path: str | Path | None = None) -> dict[str, Any]:
    """Load the JSON Schema artifact; no optional schema library is required."""

    return _read_json(SCHEMA_PATH if path is None else path)


def assert_sealed(path: str | Path | None = None) -> str:
    """Verify the default config's byte hash and return it."""

    target = CONFIG_PATH if path is None else Path(path)
    expected = CONFIG_SHA256 if target.resolve() == CONFIG_PATH.resolve() else None
    actual = file_digest(target)
    if expected is not None and actual != expected:
        raise ProtocolError(
            f"sealed config digest mismatch: expected {expected}, got {actual}"
        )
    return actual


def _lr_modes(arm_name: str) -> tuple[str, ...]:
    return ("full", "block") if arm_name == "lr_ranks" else (arm_name,)


def planned_cells(config: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], ...]:
    """Expand the sealed arm/rank matrix into deterministic benchmark cells.

    A cell is one GPU, dtype, seed, architecture arm/mode and rank.  The
    ``lr_ranks`` arm expands to both Full and per-read Block modes while its
    five ranks remain exactly the selected ladder.
    """

    cfg = load_config() if config is None else validate_config(config)
    cells: list[dict[str, Any]] = []
    model = cfg["model"]
    total_sources = 1 + 2 * int(model["layers"])
    per_read_sources = 1 + int(model["block_count"])
    for gpu in cfg["hardware_order"]:
        for dtype in cfg["dtypes"]:
            for seed in cfg["seeds"]:
                for arm_name, arm in cfg["arms"].items():
                    # Operator arms are expanded from the explicit
                    # ``operator_cases`` matrix below.  Treating them as model
                    # arms invents model coverage for operator-only adapters.
                    if arm.get("kind") == "operator":
                        continue
                    ranks = arm["ranks"] if "ranks" in arm else [arm["rank"]]
                    modes = _lr_modes(arm_name)
                    if arm_name != "lr_ranks":
                        modes = (arm["mode"],)
                    for mode in modes:
                        for rank in ranks:
                            cells.append(
                                {
                                    "cell_id": (
                                        f"{gpu}:{dtype}:seed{seed}:{arm_name}:{mode}:R{rank}"
                                    ),
                                    "scope": "model",
                                    "gpu": gpu,
                                    "dtype": dtype,
                                    "seed": seed,
                                    "arm": arm_name,
                                    "mode": mode,
                                    "rank": rank,
                                    "S": (
                                        per_read_sources
                                        if mode == "block"
                                        else total_sources
                                    ),
                                    "N": int(model["sequence"]),
                                    "D": int(model["width"]),
                                    "R": int(rank),
                                    "source_count": total_sources,
                                    "read_source_count": (
                                        per_read_sources if mode == "block" else None
                                    ),
                                }
                            )
    return tuple(cells)


def _operator_cells(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expand explicit operator geometry into deterministic paired cells."""

    cells: list[dict[str, Any]] = []
    for scope in ("smoke", "primary", "heldout"):
        for index, case in enumerate(config["operator_cases"][scope]):
            # ``validate_config`` has already checked the exact mapping and
            # scalar types.  Repeating the names here keeps the emitted plan
            # self describing and independent of runner-side tuple parsing.
            geometry = {
                "S": int(case["S"]),
                "N": int(case["N"]),
                "D": int(case["D"]),
                "R": int(case["R"]),
                "dtype": str(case["dtype"]),
            }
            for gpu in config["hardware_order"]:
                for seed in config["seeds"]:
                    case_id = f"operator_{scope}_{index}"
                    cells.append(
                        {
                            "cell_id": (
                                f"{gpu}:{geometry['dtype']}:seed{seed}:"
                                f"{case_id}:standard_operator:R{geometry['R']}"
                            ),
                            "scope": "operator",
                            "operator_scope": scope,
                            "operator_case_id": case_id,
                            "operator_case_index": index,
                            "gpu": gpu,
                            "seed": seed,
                            "arm": "standard_operator",
                            "mode": "standard_operator",
                            "S": geometry["S"],
                            "N": geometry["N"],
                            "D": geometry["D"],
                            "R": geometry["R"],
                            "rank": geometry["R"],
                            "dtype": geometry["dtype"],
                            "source_count": geometry["S"],
                            "read_source_count": None,
                        }
                    )
    return tuple(cells)


def competitor_capability(name: str) -> dict[str, Any]:
    """Return one dependency-free capability record from the registry."""

    return capability_for(name)


def competitor_capabilities() -> dict[str, dict[str, Any]]:
    """Return all registered capability records without importing adapters."""

    names = tuple(_COMPETITORS)
    return {name: capability_for(name) for name in names}


def competitor_eligibility(
    name: str,
    case: Mapping[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Evaluate a comparator cell against the sealed capability registry."""

    merged = dict(case or {})
    merged.update(fields)
    return _registry_eligibility(name, merged)


def _comparison_case(
    config: Mapping[str, Any],
    cell: Mapping[str, Any],
    competitor: str,
) -> dict[str, Any]:
    """Derive the geometry visible to the dependency-free eligibility gate."""

    arm_name = str(cell["arm"])
    mode = str(cell["mode"])
    if cell.get("scope") == "operator":
        normalized_mode = "standard_operator"
        source_count = int(cell["S"])
        read_source_count = None
    elif arm_name == "standard_operator":
        normalized_mode = "standard_operator"
        source_count = int(cell["source_count"])
        read_source_count = None
    elif mode == "full":
        normalized_mode = "full"
        source_count = int(cell["source_count"])
        read_source_count = None
    else:
        normalized_mode = "block_per_read"
        # The external primitive sees only the sources supplied at this read.
        # Total model history remains cell metadata, but it is not an alias
        # for the per-read S passed to the capability gate.
        source_count = int(cell["read_source_count"])
        read_source_count = int(cell["read_source_count"])

    # Manish/Hydra is deliberately registered as a small-D operator panel;
    # the model arms are therefore explicit not_applicable rows rather than a
    # silent promotion of its callable into the model runner.
    if competitor == "manish_hydra_2p" and normalized_mode == "block_per_read":
        normalized_mode = "block_panel"

    return {
        "mode": normalized_mode,
        "rank": int(cell["R"]),
        "width": int(cell["D"]),
        "dtype": str(cell["dtype"]),
        "N": int(cell["N"]),
        "S": int(cell["S"]),
        "source_count": source_count,
        "read_source_count": read_source_count,
        "rank_equals_width": int(cell["R"]) == int(cell["D"]),
        "external_route": False,
        "timing": True,
    }


def planned_comparison_cells(
    config: Mapping[str, Any] | None = None,
    *,
    include_not_applicable: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Return comparator cells at the architecture/capability intersection.

    Unsupported combinations are retained only when ``include_not_applicable``
    is requested.  They carry an explicit ``status='not_applicable'`` and a
    false denominator flag; they are never represented as failed timings.
    Native discovery, oracle gates, and complete-row requirements remain
    separate runtime checks.
    """

    cfg = load_config() if config is None else validate_config(config)
    cells: list[dict[str, Any]] = []
    for cell in (*_operator_cells(cfg), *planned_cells(cfg)):
        for family_name, family in cfg["comparison_families"].items():
            for competitor in family["competitors"]:
                eligibility = competitor_eligibility(
                    competitor, _comparison_case(cfg, cell, competitor)
                )
                applicable = bool(eligibility["eligible"])
                if not applicable and not include_not_applicable:
                    continue
                cells.append(
                    {
                        **cell,
                        "comparison_family": family_name,
                        "competitor": competitor,
                        "comparison_cell_id": (
                            f"{cell['cell_id']}:{family_name}:{competitor}"
                        ),
                        # Every protocol cell is intended for timing.  Carry
                        # this flag into the materialized cell so a public
                        # runner cannot accidentally re-evaluate a timing-only
                        # capability limit as an untimed correctness case.
                        "timing": True,
                        # Operator cells are measured as complete
                        # forward+backward calls.  Carry this explicitly
                        # so a direct runner cannot silently default them
                        # to a forward-only timing boundary.
                        **(
                            {"timing_mode": "forward_backward"}
                            if cell["scope"] == "operator"
                            else {}
                        ),
                        "status": "planned" if applicable else "not_applicable",
                        "eligible": applicable,
                        "eligible_denominator": bool(
                            applicable and eligibility["eligible_denominator"]
                        ),
                        "eligibility": eligibility,
                        "eligibility_reason": eligibility["reason"],
                    }
                )
    return tuple(cells)


def comparison_plan(
    config: Mapping[str, Any] | None = None,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return planned intersections and explicit inapplicable audit rows."""

    all_cells = planned_comparison_cells(config, include_not_applicable=True)
    return {
        "planned": tuple(cell for cell in all_cells if cell["status"] == "planned"),
        "not_applicable": tuple(
            cell for cell in all_cells if cell["status"] == "not_applicable"
        ),
    }


def paired_orders(
    arms: Sequence[str],
    rounds: int = TIMED_ROUNDS,
    *,
    seed: int = SEEDS[0],
) -> tuple[tuple[str, ...], ...]:
    """Return balanced forward/reverse arm orders for paired rounds.

    One seeded permutation is drawn for the cell.  Every even round uses that
    order and every odd round uses its exact reverse, so each arm occupies the
    corresponding positions equally when ``rounds`` is even.
    """

    if isinstance(arms, (str, bytes)):
        raise TypeError("arms must be a sequence of distinct arm names")
    names = tuple(str(arm) for arm in arms)
    if not names or len(set(names)) != len(names):
        raise ProtocolError("paired order needs one or more distinct arm names")
    if type(rounds) is not int or rounds < 1:
        raise ProtocolError("rounds must be a positive integer")
    if type(seed) is not int:
        raise ProtocolError("paired-order seed must be an integer")
    first = list(names)
    random.Random(seed).shuffle(first)
    reverse = list(reversed(first))
    return tuple(tuple(first if index % 2 == 0 else reverse) for index in range(rounds))


def paired_order(
    arms: Sequence[str], round_index: int, *, seed: int = SEEDS[0]
) -> tuple[str, ...]:
    """Return one round's deterministic paired order."""

    if type(round_index) is not int or round_index < 0:
        raise ProtocolError("round_index must be a nonnegative integer")
    return paired_orders(arms, round_index + 1, seed=seed)[round_index]


def retain_failure(
    arm: str,
    round_index: int,
    *,
    phase: str = "timing",
    error: Mapping[str, Any] | str | None = None,
    status: str = "failed",
    seed: int | None = None,
    gpu: str | None = None,
    order_index: int | None = None,
    input_hash: str | None = None,
    latency_ms: float | None = None,
    failure_reason: str | None = None,
    failure_at_round: int | None = None,
    failure_at_order: int | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Construct a retained raw failure row; no sample is silently removed."""

    if not isinstance(arm, str) or not arm:
        raise ProtocolError("failure arm must be a nonempty string")
    if type(round_index) is not int or round_index < 0:
        raise ProtocolError("failure round must be a nonnegative integer")
    if status not in _FAILURE_STATUSES:
        raise ProtocolError(f"invalid retained failure status {status!r}")
    if type(seed) is not int:
        raise ProtocolError("failure seed must be an integer")
    if gpu not in HARDWARE_ORDER:
        raise ProtocolError(f"failure GPU must be one of {list(HARDWARE_ORDER)!r}")
    if type(order_index) is not int or order_index < 0:
        raise ProtocolError("failure order_index must be a nonnegative integer")
    if not isinstance(input_hash, str) or not input_hash.strip():
        raise ProtocolError("failure input_hash must be a nonempty string")
    if not isinstance(phase, str) or not phase.strip():
        raise ProtocolError("failure phase must be a nonempty string")
    if latency_ms is not None and (
        isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)) or latency_ms <= 0
    ):
        raise ProtocolError("failure latency_ms must be positive or None")
    if failure_at_round is not None and (
        type(failure_at_round) is not int or failure_at_round < 0
    ):
        raise ProtocolError("failure_at_round must be a nonnegative integer")
    if failure_at_order is not None and (
        type(failure_at_order) is not int or failure_at_order < 0
    ):
        raise ProtocolError("failure_at_order must be a nonnegative integer")
    if failure_reason is None:
        if isinstance(error, str):
            failure_reason = error
        elif isinstance(error, Mapping):
            failure_reason = str(error.get("message") or error.get("reason") or "") or None
    if not isinstance(failure_reason, str) or not failure_reason.strip():
        raise ProtocolError("retained failures require a nonempty failure_reason")
    failure_at_round = round_index if failure_at_round is None else failure_at_round
    failure_at_order = order_index if failure_at_order is None else failure_at_order
    reserved = {
        "seed",
        "gpu",
        "round_index",
        "order_index",
        "input_hash",
        "arm",
        "status",
        "latency_ms",
        "failure_phase",
        "failure_reason",
        "failure_at_round",
        "failure_at_order",
    }
    overlap = sorted(reserved.intersection(metadata))
    if overlap:
        raise ProtocolError(
            "failure metadata may not override required fields: "
            + ", ".join(overlap)
        )
    row: dict[str, Any] = {
        "seed": seed,
        "gpu": gpu,
        "round_index": round_index,
        "order_index": order_index,
        "input_hash": input_hash,
        "arm": arm,
        "status": status,
        "latency_ms": latency_ms,
        "failure_phase": phase,
        "failure_reason": failure_reason,
        "failure_at_round": failure_at_round,
        "failure_at_order": failure_at_order,
    }
    if error is not None:
        row["error"] = deepcopy(error)
    row.update(deepcopy(metadata))
    return row


def _normalise_planned_eligibility(
    value: Mapping[str, Any] | Sequence[Any] | None,
) -> dict[str, bool] | None:
    """Normalize a plan or arm map to ``arm -> eligible`` booleans."""

    if value is None:
        return None
    result: dict[str, bool] = {}

    def add(name: Any, decision: Any) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ProtocolError("planned eligibility needs a nonempty arm name")
        if type(decision) is not bool:
            raise ProtocolError(
                f"planned eligibility for {name!r} must be a boolean"
            )
        name = name.strip()
        if name in result and result[name] is not decision:
            raise ProtocolError(f"planned eligibility conflicts for arm {name!r}")
        result[name] = decision

    if isinstance(value, Mapping):
        # Also accept one planned-comparison row as a convenience for callers
        # validating a single cell.
        if "eligible" in value and (
            "arm" in value or "competitor" in value
        ):
            add(value.get("arm", value.get("competitor")), value["eligible"])
        else:
            for name, decision in value.items():
                if isinstance(decision, Mapping):
                    if "eligible" not in decision:
                        raise ProtocolError(
                            f"planned eligibility for {name!r} lacks 'eligible'"
                        )
                    decision = decision["eligible"]
                add(name, decision)
        return result

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("planned eligibility must be a mapping or sequence")
    for item in value:
        if isinstance(item, Mapping):
            name = item.get("arm", item.get("competitor"))
            if "eligible" not in item:
                raise ProtocolError("planned eligibility rows need an 'eligible' field")
            add(name, item["eligible"])
        else:
            # A sequence of names is an explicit eligible-arm set.  The
            # validator treats arms omitted from that set as inapplicable.
            add(item, True)
    return result


def validate_raw_samples(
    rows: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    *,
    rounds: int = TIMED_ROUNDS,
    seed: int | None = None,
    gpu: str | None = None,
    require_metadata: bool = True,
    eligibility: Mapping[str, Any] | Sequence[Any] | None = None,
    planned_eligibility: Mapping[str, Any] | Sequence[Any] | None = None,
    planned_cells: Mapping[str, Any] | Sequence[Any] | None = None,
    require_eligible_ok: bool = True,
) -> list[dict[str, Any]]:
    """Validate a complete raw matrix while retaining every status row.

    Every row must carry its seed, GPU, round, paired order, shared input
    hash, latency field, and failure provenance fields.  The recorded arm
    order is checked against :func:`paired_orders`, rather than merely
    checking that order indexes form a set.  ``eligibility`` (or either
    spelling of ``planned_eligibility``) may be an arm map or planned-cell
    rows.  When supplied, eligible arms must be ``ok`` and only predeclared
    ineligible arms may be ``not_applicable``.  Failures without a planned
    eligibility map remain retained audit rows.

    ``require_metadata`` is retained for source compatibility; the sealed
    protocol now always requires the metadata needed to reconstruct pairing.
    """

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of mappings")
    if isinstance(arms, (str, bytes)):
        raise TypeError("arms must be a sequence of names")
    names = tuple(str(arm) for arm in arms)
    if not names or len(set(names)) != len(names):
        raise ProtocolError("raw sample validation needs distinct arm names")
    if type(rounds) is not int or rounds < 1:
        raise ProtocolError("rounds must be a positive integer")
    if seed is not None and type(seed) is not int:
        raise ProtocolError("seed must be an integer")
    if gpu is not None and gpu not in HARDWARE_ORDER:
        raise ProtocolError(f"GPU must be one of {list(HARDWARE_ORDER)!r}")
    if type(require_metadata) is not bool:
        raise ProtocolError("require_metadata must be a boolean")
    if type(require_eligible_ok) is not bool:
        raise ProtocolError("require_eligible_ok must be a boolean")

    supplied_plans = [
        item
        for item in (eligibility, planned_eligibility, planned_cells)
        if item is not None
    ]
    if len(supplied_plans) > 1:
        raise ProtocolError(
            "provide only one of eligibility, planned_eligibility, or planned_cells"
        )
    eligibility_map = _normalise_planned_eligibility(
        supplied_plans[0] if supplied_plans else None
    )
    inferred_eligibility: dict[str, bool] = dict(eligibility_map or {})
    plan_supplied = eligibility_map is not None

    expected = {(name, index) for name in names for index in range(rounds)}
    seen: set[tuple[str, int]] = set()
    round_orders: dict[int, dict[int, str]] = {}
    round_hashes: dict[int, str] = {}
    detached: list[dict[str, Any]] = []
    matrix_seed = seed
    matrix_gpu = gpu
    required_fields = (
        "seed",
        "gpu",
        "round_index",
        "order_index",
        "input_hash",
        "latency_ms",
        "failure_phase",
        "failure_reason",
        "failure_at_round",
        "failure_at_order",
    )
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ProtocolError(f"raw row {index} must be an object")
        row = deepcopy(dict(raw))
        missing_fields = [field for field in required_fields if field not in row]
        if missing_fields:
            raise ProtocolError(
                f"raw row {index} is missing required fields: {', '.join(missing_fields)}"
            )
        arm = row.get("arm")
        sample = row.get("round_index")
        status = row.get("status")
        if arm not in names:
            raise ProtocolError(f"raw row {index} has unknown arm {arm!r}")
        if type(sample) is not int or not 0 <= sample < rounds:
            raise ProtocolError(f"raw row {index} has invalid round {sample!r}")
        if status not in _ROW_STATUSES:
            raise ProtocolError(f"raw row {index} has invalid status {status!r}")
        row_seed = row["seed"]
        row_gpu = row["gpu"]
        order_index = row["order_index"]
        input_hash = row["input_hash"]
        latency_ms = row["latency_ms"]
        if type(row_seed) is not int:
            raise ProtocolError(f"raw row {index} seed must be an integer")
        if row_gpu not in HARDWARE_ORDER:
            raise ProtocolError(f"raw row {index} has invalid GPU {row_gpu!r}")
        if matrix_seed is None:
            matrix_seed = row_seed
        elif row_seed != matrix_seed:
            raise ProtocolError(f"raw row {index} seed differs from the matrix seed")
        if matrix_gpu is None:
            matrix_gpu = row_gpu
        elif row_gpu != matrix_gpu:
            raise ProtocolError(f"raw row {index} GPU differs from the matrix GPU")
        if type(order_index) is not int or not 0 <= order_index < len(names):
            raise ProtocolError(f"raw row {index} has invalid order_index")
        if not isinstance(input_hash, str) or not input_hash.strip():
            raise ProtocolError(f"raw row {index} has an empty input_hash")
        if status == "ok":
            if (
                isinstance(latency_ms, bool)
                or not isinstance(latency_ms, (int, float))
                or latency_ms <= 0
            ):
                raise ProtocolError(
                    f"raw row {index} needs a positive latency_ms when status is ok"
                )
        elif latency_ms is not None and (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, (int, float))
            or latency_ms <= 0
        ):
            raise ProtocolError(
                f"raw row {index} failure latency_ms must be positive or None"
            )

        failure_fields = {
            "failure_phase": row["failure_phase"],
            "failure_reason": row["failure_reason"],
            "failure_at_round": row["failure_at_round"],
            "failure_at_order": row["failure_at_order"],
        }
        if status in _FAILURE_STATUSES:
            if any(
                not isinstance(failure_fields[field], str)
                or not failure_fields[field].strip()
                for field in ("failure_phase", "failure_reason")
            ):
                raise ProtocolError(
                    f"raw row {index} retained failure needs nonempty failure provenance"
                )
            failure_at_round = failure_fields["failure_at_round"]
            failure_at_order = failure_fields["failure_at_order"]
            if type(failure_at_round) is not int or not 0 <= failure_at_round < rounds:
                raise ProtocolError(
                    f"raw row {index} has invalid failure_at_round provenance"
                )
            if type(failure_at_order) is not int or not 0 <= failure_at_order < len(names):
                raise ProtocolError(
                    f"raw row {index} has invalid failure_at_order provenance"
                )
        else:
            # Successful rows may carry null failure fields, but a non-null
            # provenance value must still have the sealed scalar type.
            phase = failure_fields["failure_phase"]
            reason = failure_fields["failure_reason"]
            if phase is not None and (
                not isinstance(phase, str) or not phase.strip()
            ):
                raise ProtocolError(f"raw row {index} has invalid failure_phase")
            if reason is not None and (
                not isinstance(reason, str) or not reason.strip()
            ):
                raise ProtocolError(f"raw row {index} has invalid failure_reason")
            for field, upper in (
                ("failure_at_round", rounds),
                ("failure_at_order", len(names)),
            ):
                value = failure_fields[field]
                if value is not None and (
                    type(value) is not int or not 0 <= value < upper
                ):
                    raise ProtocolError(f"raw row {index} has invalid {field}")

        row_eligible = row.get("eligible", None)
        if "eligible" in row:
            if type(row_eligible) is not bool:
                raise ProtocolError(f"raw row {index} eligible must be a boolean")
            if plan_supplied:
                if arm not in inferred_eligibility:
                    raise ProtocolError(
                        f"raw row {index} lacks planned eligibility for arm {arm!r}"
                    )
                if inferred_eligibility[arm] is not row_eligible:
                    raise ProtocolError(
                        f"raw row {index} conflicts with planned eligibility"
                    )
        if plan_supplied:
            if arm not in inferred_eligibility:
                raise ProtocolError(
                    f"raw row {index} lacks planned eligibility for arm {arm!r}"
                )
            eligible = inferred_eligibility[arm]
            if eligible:
                if require_eligible_ok and status != "ok":
                    raise ProtocolError(
                        f"eligible arm {arm!r} must have status 'ok', got {status!r}"
                    )
                if status == "not_applicable":
                    raise ProtocolError(
                        f"eligible arm {arm!r} cannot be marked not_applicable"
                    )
            elif status != "not_applicable":
                raise ProtocolError(
                    f"arm {arm!r} is predeclared ineligible and must be not_applicable"
                )
        elif status == "not_applicable":
            raise ProtocolError(
                "not_applicable rows require predeclared planned eligibility"
            )

        key = (arm, sample)
        if key in seen:
            raise ProtocolError(f"duplicate raw row for arm={arm!r}, round={sample}")
        seen.add(key)
        round_order = round_orders.setdefault(sample, {})
        if order_index in round_order:
            raise ProtocolError(
                f"round {sample} has duplicate order_index {order_index}"
            )
        round_order[order_index] = arm
        prior_hash = round_hashes.setdefault(sample, input_hash)
        if prior_hash != input_hash:
            raise ProtocolError(f"raw rows for round {sample} do not share one input_hash")
        detached.append(row)

    if matrix_seed is None or matrix_gpu is None:
        raise ProtocolError("raw samples require a seed and GPU in every row")
    missing = sorted(expected - seen)
    if missing:
        preview = ", ".join(f"{arm}:{sample}" for arm, sample in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ProtocolError(f"missing raw failure/sample rows ({preview}{suffix})")
    expected_orders = paired_orders(names, rounds, seed=matrix_seed)
    for sample in range(rounds):
        order_map = round_orders.get(sample, {})
        if set(order_map) != set(range(len(names))):
            raise ProtocolError(
                f"round {sample} does not contain one order_index per arm"
            )
        actual = tuple(order_map[index] for index in range(len(names)))
        if actual != expected_orders[sample]:
            raise ProtocolError(
                f"round {sample} arm order {actual!r} does not match paired_orders"
            )
    if plan_supplied and set(inferred_eligibility) != set(names):
        missing_plans = sorted(set(names) - set(inferred_eligibility))
        extra_plans = sorted(set(inferred_eligibility) - set(names))
        details = []
        if missing_plans:
            details.append("missing=" + ",".join(missing_plans))
        if extra_plans:
            details.append("extra=" + ",".join(extra_plans))
        raise ProtocolError("planned eligibility arm mismatch (" + "; ".join(details) + ")")
    return detached


# Explicit aliases make the retention contract discoverable to report writers
# without introducing a second implementation.
validate_result_rows = validate_raw_samples
record_failure = retain_failure


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "CONFIG_PATH",
    "CONFIG_SCHEMA",
    "CONFIG_SHA256",
    "CONFIDENCE",
    "DTYPES",
    "HARDWARE_ORDER",
    "LR_RANKS",
    "PLATEAU_MARGIN",
    "PROTOCOL_BASE_REVISION",
    "PROJECT_ROOT",
    "ProtocolError",
    "SCHEMA_PATH",
    "SEEDS",
    "TIMED_ROUNDS",
    "WARMUP_ROUNDS",
    "assert_sealed",
    "canonical_json",
    "comparison_plan",
    "competitor_capabilities",
    "competitor_capability",
    "competitor_eligibility",
    "config_digest",
    "file_digest",
    "load_config",
    "load_schema",
    "paired_order",
    "paired_orders",
    "planned_cells",
    "planned_comparison_cells",
    "record_failure",
    "retain_failure",
    "validate_config",
    "validate_raw_samples",
    "validate_result_rows",
]
