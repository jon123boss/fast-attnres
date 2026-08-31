#!/usr/bin/env python3
"""Inspect selected FLA-derived source-list Block binaries.

This is an external, non-timing diagnostic for the scalar-compact FLA-derived
source-list kernel at the frozen ``25a85a9`` implementation. For each of the
six production Block geometries ``D=1024``, ``R in (128, 512, 1024)`` and
``S in (2, 9)`` it makes one warm-up call in each direction, observes the
exact Triton 3.6 ``Autotuner.run`` result, captures each route in a CUDA
Graph, and only then reads and hashes TTIR, TTGIR, LLIR, and PTX artifacts.
The backward tuner has three explicit geometry/layout families; family 2
contains the source-serial compact-prefix candidate; the report records which family was
selected for the exact full dtype-bearing key.  The observer stores in-memory
identity and launch facts only; it is removed before graph capture and
artifact inspection.  No benchmark timing, evaluator import, or dispatch
input is changed.

The module imports only the Python standard library at import time.  The GPU
phase requires ``--allow-gpu`` explicitly and accepts H100/SM90 or B200/SM100.
CPU/static tests exercise the report and matching logic without importing
Triton or launching CUDA.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = Path(__file__).resolve()
RESULT_MARKER = "SELECTED_FLA_BLOCK_CODEGEN_RESULT="

# Candidate F2 is intentionally pinned to the scalar source-tile compact-prefix
# implementation. This probe does not mutate the repository's frozen gate.
BASE_COMMIT = "134d9d3a206b185a83a6e4d5a5765790ee675201"
IMPLEMENTATION_TREE = "74f0b86eac24c2ff85ad01d7a77039dcaf84044c"
IMPLEMENTATION_COMMIT = "25a85a9b99985ac90d69ce636d6b42b5f636a129"
SOURCE_REVISION = IMPLEMENTATION_COMMIT
SOURCE_SHA256 = "2cd7ac89b15faeb13640bff4a7948e437453b69446bfc8c7922511e341843e10"
SOURCE_MODULE = "attnres._kernels.fla_full_sources"
FORWARD_KERNEL_NAME = "_fla_source_forward_kernel"
BACKWARD_KERNEL_NAME = "_fla_source_backward_kernel"

SOURCE_COUNTS = (2, 9)
RANKS = (128, 512, 1024)
BATCH_SIZE = 2
TOKEN_COUNT = 2048
VALUE_WIDTH = 1024
PRODUCTION_EPS = 2**-23
PRODUCTION_SCALE = 1.0
VALUE_DTYPE = "torch.bfloat16"
QUERY_DTYPE = "torch.float32"
AUTOTUNER_KEY_FIELDS = ("L2", "D", "R")
FORWARD_LAYOUT_FAMILIES = (0, 1)
BACKWARD_LAYOUT_FAMILIES = (0, 1, 2)
FORWARD_DTYPE_SUFFIXES = (
    QUERY_DTYPE,
    VALUE_DTYPE,
    "torch.float32",
    "torch.float32",
    "torch.float32",
    "torch.float32",
)
BACKWARD_DTYPE_SUFFIXES = (
    QUERY_DTYPE,
    "torch.float32",
    VALUE_DTYPE,
    "torch.float32",
    "torch.float32",
    "torch.float32",
    "torch.float32",
)

# The names are an observation scope, not a dispatch table.  The probe never
# chooses a kernel configuration from this mapping; it only checks that the
# device reported by CUDA matches the explicitly requested scope.
HARDWARE = {
    "H100": {"capability": (9, 0), "sm": "sm90", "arch": 90},
    "B200": {"capability": (10, 0), "sm": "sm100", "arch": 100},
}


class ProbeError(RuntimeError):
    """A missing or ambiguous observation that must fail closed."""


def _jsonable(value: Any) -> Any:
    """Convert runtime values to deterministic JSON without device access."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": _sha256_bytes(value)}
    as_dict = getattr(value, "_asdict", None)
    if callable(as_dict):
        try:
            return _jsonable(as_dict())
        except Exception:
            pass
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    # Triton's constexpr values are useful as their underlying value when it
    # is exposed.  This branch is intentionally conservative for opaque
    # compiler objects and never follows filesystem paths.
    if type(value).__name__ == "constexpr":
        try:
            return {"constexpr": _jsonable(value.value)}
        except Exception:
            return {"type": f"{type(value).__module__}.{type(value).__name__}"}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return {"type": f"{type(value).__module__}.{type(value).__name__}"}


def _unwrap_constexpr(value: Any) -> Any:
    """Unwrap Triton constexpr wrappers for strict constant comparisons."""

    if type(value).__name__ == "constexpr":
        try:
            return _unwrap_constexpr(value.value)
        except Exception:
            return value
    if isinstance(value, (tuple, list)):
        return tuple(_unwrap_constexpr(item) for item in value)
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (int(value) - 1).bit_length()


def _validate_case(source_count: int, rank: int) -> None:
    if source_count not in SOURCE_COUNTS:
        raise ValueError(f"source_count must be one of {SOURCE_COUNTS}, got {source_count!r}")
    if rank not in RANKS:
        raise ValueError(f"rank must be one of {RANKS}, got {rank!r}")


def _expected_l2(source_count: int) -> int:
    _validate_case(source_count, RANKS[0])
    return max(8, _next_power_of_two(source_count))


def _expected_tuning_key(
    direction: str, source_count: int, rank: int,
) -> tuple[Any, ...]:
    """Return Triton 3.6's complete direction-specific Autotuner key."""

    _validate_case(source_count, rank)
    if direction not in {"forward", "backward"}:
        raise ValueError(f"unknown direction {direction!r}")
    # Autotuner.run appends str(arg.dtype) for tensor arguments in its
    # insertion order.  Source and gradient-source tuples have no dtype;
    # the scalar and constexpr arguments do not contribute suffixes.
    suffixes = (
        FORWARD_DTYPE_SUFFIXES if direction == "forward" else BACKWARD_DTYPE_SUFFIXES
    )
    return (_expected_l2(source_count), VALUE_WIDTH, rank, *suffixes)


def _expected_launch_constants(
    direction: str, source_count: int, rank: int, block: int, layout_family: int,
) -> dict[str, Any]:
    """Return constants expected for one selected direction/config."""

    _validate_case(source_count, rank)
    if block not in (1, 2, 4, 8):
        raise ValueError(f"unexpected source tile {block!r}")
    allowed_families = (
        FORWARD_LAYOUT_FAMILIES
        if direction == "forward"
        else BACKWARD_LAYOUT_FAMILIES
    )
    if layout_family not in allowed_families:
        raise ValueError(
            f"unexpected {direction} layout family {layout_family!r}; "
            f"expected one of {allowed_families!r}"
        )
    l2 = _expected_l2(source_count)
    common = {
        "D": VALUE_WIDTH,
        "R": rank,
        "BLOCK_D": VALUE_WIDTH,
        "BLOCK_R": _next_power_of_two(rank),
        "BL": block,
        "LAYOUT_FAMILY": layout_family,
        "L2": l2,
    }
    if direction == "forward":
        common.update({
            "ROW_STRIDES": (VALUE_WIDTH,) * l2,
            "FEATURE_STRIDES": (1,) * l2,
        })
    elif direction == "backward":
        common.update({
            "BLOCK_PREFIX": _next_power_of_two(max(1, VALUE_WIDTH - rank)),
            "VALUE_ROW_STRIDES": (VALUE_WIDTH,) * l2,
            "VALUE_FEATURE_STRIDES": (1,) * l2,
            "GRAD_VALUE_ROW_STRIDES": (VALUE_WIDTH,) * l2,
            "GRAD_VALUE_FEATURE_STRIDES": (1,) * l2,
        })
    else:
        raise ValueError(f"unknown direction {direction!r}")
    return common


def _config_summary(config: Any) -> dict[str, Any]:
    """Copy one Triton Config without retaining a live compiler object."""

    kwargs = getattr(config, "kwargs", {})
    kwargs = dict(kwargs) if isinstance(kwargs, Mapping) else {}
    all_kwargs = getattr(config, "all_kwargs", None)
    if callable(all_kwargs):
        try:
            all_values = all_kwargs()
        except Exception:
            all_values = kwargs
    else:
        all_values = kwargs
    return {
        "kwargs": _jsonable(kwargs),
        "all_kwargs": _jsonable(all_values),
        "num_warps": _jsonable(getattr(config, "num_warps", None)),
        "num_stages": _jsonable(getattr(config, "num_stages", None)),
        "num_ctas": _jsonable(getattr(config, "num_ctas", None)),
        "maxnreg": _jsonable(getattr(config, "maxnreg", None)),
        "has_pre_hook": getattr(config, "pre_hook", None) is not None,
        "ir_override": _jsonable(getattr(config, "ir_override", None)),
    }


def _config_signature(config: Any) -> str:
    return json.dumps(_config_summary(config), sort_keys=True, separators=(",", ":"))


def _value_summary(value: Any) -> dict[str, Any]:
    """Summarize a call value without copying tensor storage."""

    if hasattr(value, "dtype"):
        shape = getattr(value, "shape", None)
        return {
            "type": f"{type(value).__module__}.{type(value).__name__}",
            "dtype": str(getattr(value, "dtype")),
            "shape": _jsonable(tuple(shape)) if shape is not None else None,
        }
    if isinstance(value, (tuple, list)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "item_dtypes": [str(getattr(item, "dtype")) for item in value[:16]
                            if hasattr(item, "dtype")],
        }
    return {"type": f"{type(value).__module__}.{type(value).__name__}", "value": _jsonable(value)}


def _autotuner_call_binding(
    target: Any, args: Sequence[Any], kwargs: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Reconstruct the exact key made by Triton 3.6 Autotuner.run."""

    arg_names = tuple(getattr(target, "arg_names", ()))
    if not arg_names:
        raise ProbeError("selected Autotuner has no argument names")
    named = dict(zip(arg_names, args))
    all_args = dict(named)
    all_args.update(kwargs)
    key_fields = tuple(getattr(target, "keys", ()))
    if key_fields != AUTOTUNER_KEY_FIELDS:
        raise ProbeError(
            "FLA Autotuner key fields changed or are ambiguous: "
            f"expected {AUTOTUNER_KEY_FIELDS!r}, got {key_fields!r}"
        )
    missing = [name for name in key_fields if name not in all_args]
    if missing:
        raise ProbeError(f"selected Autotuner call omitted key fields: {missing!r}")
    tuning_key = [all_args[name] for name in key_fields]
    dtype_suffixes = [str(value.dtype) for value in all_args.values() if hasattr(value, "dtype")]
    tuning_key.extend(dtype_suffixes)
    binding = {
        "arg_names": list(arg_names),
        "key_fields": list(key_fields),
        "key_values": _jsonable(tuning_key[:len(key_fields)]),
        "dtype_suffixes": dtype_suffixes,
        "tuning_key": _jsonable(tuple(tuning_key)),
        "values": {name: _value_summary(value) for name, value in all_args.items()
                   if name in arg_names},
        "options": {
            name: _jsonable(all_args[name]) for name in
            ("num_warps", "num_stages", "num_ctas", "maxnreg") if name in all_args
        },
    }
    return tuple(tuning_key), binding


def _install_autotuner_observer(
    target: Any, observations: dict[str, Any], direction: str,
) -> Any:
    """Observe one outer Autotuner call, with no file I/O or timing hooks."""

    target_type = type(target)
    original_run = target_type.run

    def observed_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self is not target:
            return original_run(self, *args, **kwargs)
        if observations["calls"]:
            raise ProbeError(f"selected FLA {direction} Autotuner ran more than once")
        tuning_key, binding = _autotuner_call_binding(self, args, kwargs)
        event = {
            "direction": direction,
            "binding": binding,
            "tuning_key": tuning_key,
            "status": "running",
        }
        observations["calls"].append(event)
        # Keep this wrapper around only the outer Autotuner call.  In
        # particular, do not wrap Autotuner._bench: doing so would alter the
        # config selection measurements made inside the call.
        try:
            result = original_run(self, *args, **kwargs)
        except Exception as exc:
            event["status"] = "error"
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        event["status"] = "returned"
        event["compiled_kernel"] = result
        event["object_id"] = f"0x{id(result):x}" if result is not None else None
        return result

    target_type.run = observed_run

    def restore() -> None:
        if target_type.run is observed_run:
            target_type.run = original_run

    return restore


def _cache_bundles(
    function: Any, device_index: int | None = None, direction: str = "kernel",
) -> list[dict[str, Any]]:
    caches = getattr(function, "device_caches", None)
    if not isinstance(caches, Mapping):
        raise ProbeError(f"selected {direction} JIT function has no device_caches mapping")
    bundles = []
    for device, bundle in caches.items():
        try:
            same_device = device_index is None or int(device) == int(device_index)
        except (TypeError, ValueError):
            same_device = device_index is None or str(device) in {
                str(device_index),
                f"cuda:{device_index}",
            }
        if not same_device or not isinstance(bundle, (tuple, list)) or len(bundle) < 2:
            continue
        kernel_cache, key_cache = bundle[0], bundle[1]
        if isinstance(kernel_cache, Mapping) and isinstance(key_cache, Mapping):
            bundles.append({
                "device": _jsonable(device),
                "kernel_cache": kernel_cache,
                "key_cache": key_cache,
            })
    if not bundles:
        raise ProbeError(f"selected {direction} JIT function has no usable current-device cache")
    return bundles


def _cache_entries(
    function: Any, device_index: int | None = None, direction: str = "kernel",
) -> list[dict[str, Any]]:
    """Join each direct JIT cache binary to one exact specialization entry."""

    entries: list[dict[str, Any]] = []
    for bundle in _cache_bundles(function, device_index, direction):
        kernel_cache = bundle["kernel_cache"]
        key_cache = bundle["key_cache"]
        for cache_key, compiled in kernel_cache.items():
            matches = []
            for key_entry, value in key_cache.items():
                if value != cache_key or not isinstance(key_entry, tuple) or len(key_entry) != 2:
                    continue
                specialization, options = key_entry
                matches.append({"specialization": specialization, "options": options})
            if len(matches) != 1:
                raise ProbeError(
                    "direct JIT cache key has ambiguous specialization mapping: "
                    f"{cache_key!r} ({len(matches)} matches)"
                )
            entries.append({
                "device": bundle["device"],
                "cache_key": cache_key,
                "compiled": compiled,
                "specialization": matches[0]["specialization"],
                "options": matches[0]["options"],
            })
    if not entries:
        raise ProbeError(f"selected {direction} JIT cache contains no CompiledKernel entries")
    return entries


def _metadata_value(metadata: Any, name: str) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    try:
        return getattr(metadata, name)
    except Exception:
        return None


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compiled_resources(compiled: Any) -> dict[str, Any]:
    """Return compiler and launch resource metadata from one binary."""

    metadata = getattr(compiled, "metadata", None)
    resources: dict[str, Any] = {}
    for name in ("n_regs", "n_spills", "n_max_threads"):
        value = getattr(compiled, name, None)
        if value is not None:
            resources[name] = _jsonable(value)
    for name in ("shared", "num_warps", "num_stages", "num_ctas", "tmem_size"):
        value = _metadata_value(metadata, name)
        if value is not None:
            resources[name] = _jsonable(value)
    resources["metadata"] = _jsonable(metadata)
    return resources


def _selected_config(
    target: Any, tuning_key: tuple[Any, ...], direction: str,
) -> dict[str, Any]:
    """Resolve one Autotuner key/config pair and reject all ambiguity."""

    cache = getattr(target, "cache", None)
    if not isinstance(cache, Mapping):
        raise ProbeError(f"selected {direction} Autotuner has no in-memory cache mapping")
    matches = [(key, value) for key, value in cache.items() if key == tuning_key]
    if len(matches) != 1:
        raise ProbeError(
            "selected Autotuner key is missing or ambiguous: "
            f"expected one exact key, found {len(matches)}"
        )
    cache_key, config = matches[0]
    if config is None:
        raise ProbeError(f"selected {direction} Autotuner key does not map to a Config")
    best = getattr(target, "best_config", None)
    if best is None or best is not config:
        raise ProbeError(
            f"{direction} Autotuner.best_config does not identify the exact cached Config"
        )
    configs = list(getattr(target, "configs", ()))
    same = [
        candidate
        for candidate in configs
        if _config_signature(candidate) == _config_signature(config)
    ]
    if len(same) != 1:
        raise ProbeError(
            "selected Autotuner Config is missing or duplicated in candidate list: "
            f"found {len(same)} matching configs"
        )
    if len(cache) != 1:
        raise ProbeError(
            "selected Autotuner cache contains more than the one observed production key"
        )
    kwargs = getattr(config, "kwargs", {})
    if not isinstance(kwargs, Mapping) or set(kwargs) != {"BL", "LAYOUT_FAMILY"}:
        raise ProbeError(f"selected FLA {direction} config has unexpected kwargs: {kwargs!r}")
    block, layout_family = kwargs["BL"], kwargs["LAYOUT_FAMILY"]
    allowed_families = (
        FORWARD_LAYOUT_FAMILIES
        if direction == "forward"
        else BACKWARD_LAYOUT_FAMILIES
    )
    if block not in (1, 2, 4, 8) or layout_family not in allowed_families:
        raise ProbeError(f"selected FLA {direction} config has invalid geometry: {kwargs!r}")
    family_counts = {family: 0 for family in allowed_families}
    for candidate in configs:
        candidate_kwargs = getattr(candidate, "kwargs", {})
        if isinstance(candidate_kwargs, Mapping):
            family = candidate_kwargs.get("LAYOUT_FAMILY")
            if family in family_counts:
                family_counts[family] += 1
    if any(count == 0 for count in family_counts.values()):
        raise ProbeError(
            f"selected {direction} Autotuner candidate list omits a required "
            f"layout family: {family_counts!r}"
        )
    return {
        "key": _jsonable(cache_key),
        "config": _config_summary(config),
        "block": int(block),
        "layout_family": int(layout_family),
        "candidate_count": len(configs),
        "allowed_layout_families": list(allowed_families),
        "candidate_family_counts": {
            str(family): count for family, count in family_counts.items()
        },
        "candidate_configs": [_config_summary(candidate) for candidate in configs],
        "configs_timings": _jsonable(getattr(target, "configs_timings", None)),
    }


def _constant_map(compiled: Any) -> dict[str, Any]:
    source = getattr(compiled, "src", None)
    function = getattr(source, "fn", None)
    names = tuple(getattr(function, "arg_names", ()))
    constants = getattr(source, "constants", None)
    if not isinstance(constants, Mapping):
        raise ProbeError("selected CompiledKernel has no constexpr constants mapping")
    result: dict[str, Any] = {}
    for raw_index, value in constants.items():
        index = raw_index[0] if isinstance(raw_index, tuple) and len(raw_index) == 1 else raw_index
        if isinstance(index, int) and 0 <= index < len(names):
            result[names[index]] = _unwrap_constexpr(value)
    return result


def _validate_selected_binary(
    target: Any,
    compiled: Any,
    entries: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    direction: str,
    source_count: int,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one returned binary and its exact direct JIT cache entry."""

    if type(compiled).__name__ != "CompiledKernel":
        raise ProbeError(
            f"{direction} Autotuner did not return a Triton CompiledKernel: "
            f"{type(compiled).__name__}"
        )
    object_matches = [entry for entry in entries if entry["compiled"] is compiled]
    if len(object_matches) != 1:
        raise ProbeError(
            f"selected {direction} CompiledKernel identity is missing or ambiguous: "
            f"found {len(object_matches)} direct cache entries"
        )
    entry = object_matches[0]
    compiler_hash = getattr(compiled, "hash", None)
    if not isinstance(compiler_hash, str) or not compiler_hash:
        raise ProbeError("selected CompiledKernel has no compiler hash")
    hash_matches = [
        entry
        for entry in entries
        if getattr(entry["compiled"], "hash", None) == compiler_hash
    ]
    if len(hash_matches) != 1:
        raise ProbeError(
            f"selected {direction} CompiledKernel compiler hash is ambiguous: "
            f"found {len(hash_matches)} direct cache entries"
        )
    metadata_hash = _metadata_value(getattr(compiled, "metadata", None), "hash")
    if metadata_hash != compiler_hash:
        raise ProbeError(
            "selected CompiledKernel hash mismatch: "
            f"object={compiler_hash!r}, metadata={metadata_hash!r}"
        )

    source = getattr(compiled, "src", None)
    source_function = getattr(source, "fn", None)
    target_function = getattr(target, "fn", None)
    same_object = source_function is target_function
    source_cache_key = getattr(source_function, "cache_key", None)
    target_cache_key = getattr(target_function, "cache_key", None)
    same_cache_key = source_cache_key is not None and source_cache_key == target_cache_key
    if not (same_object or same_cache_key):
        raise ProbeError(
            f"selected CompiledKernel is not bound to the target FLA {direction} JIT function"
        )

    constants = _constant_map(compiled)
    expected = _expected_launch_constants(
        direction,
        source_count,
        rank,
        int(selected["block"]),
        int(selected["layout_family"]),
    )
    missing = [name for name in expected if name not in constants]
    if missing:
        raise ProbeError(f"selected CompiledKernel omitted constexpr constants: {missing!r}")
    mismatches = {
        name: {"expected": _jsonable(value), "observed": _jsonable(constants[name])}
        for name, value in expected.items()
        if _unwrap_constexpr(constants[name]) != _unwrap_constexpr(value)
    }
    if mismatches:
        raise ProbeError(
            f"selected {direction} CompiledKernel constants do not match the call: {mismatches!r}"
        )

    metadata = getattr(compiled, "metadata", None)
    selected_config = selected["config"]
    for name in ("num_warps", "num_stages"):
        configured = selected_config.get(name)
        observed = _metadata_value(metadata, name)
        if observed is not None and configured != observed:
            raise ProbeError(
                f"selected Config/{name} mismatch: config={configured!r}, metadata={observed!r}"
            )
    for name in ("n_regs", "n_spills", "n_max_threads"):
        if not _numeric(getattr(compiled, name, None)):
            raise ProbeError(f"selected CompiledKernel missing numeric resource {name}")
    for name in ("shared", "num_warps"):
        if not _numeric(_metadata_value(metadata, name)):
            raise ProbeError(f"selected CompiledKernel metadata missing numeric resource {name}")

    identity = {
        "class": f"{type(compiled).__module__}.{type(compiled).__name__}",
        "object_id": f"0x{id(compiled):x}",
        "compiler_hash": compiler_hash,
        "kernel_sha256": _sha256_bytes(getattr(compiled, "kernel", b""))
        if isinstance(getattr(compiled, "kernel", None), bytes)
        else None,
        "name": _jsonable(getattr(compiled, "name", None)),
        "metadata_hash": _jsonable(metadata_hash),
        "jit_cache_key": _jsonable(entry["cache_key"]),
        "jit_specialization": _jsonable(entry["specialization"]),
        "jit_options": _jsonable(entry["options"]),
        "source_binding": {
            "same_jit_object": same_object,
            "same_jit_cache_key": same_cache_key,
        },
        "constants": _jsonable(constants),
        "resources": _compiled_resources(compiled),
    }
    if not identity["kernel_sha256"]:
        raise ProbeError("selected CompiledKernel has no binary bytes to hash")
    return identity, dict(entry)


def _artifact_text(compiled: Any, extension: str) -> str:
    asm = getattr(compiled, "asm", None)
    if isinstance(asm, Mapping) and extension in asm:
        value = asm[extension]
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ProbeError(f"selected {extension} artifact is not UTF-8 text: {exc}") from exc
        if isinstance(value, str):
            return value
        raise ProbeError(f"selected {extension} artifact is not text")

    # A cache hit can expose an artifact through metadata_group even when a
    # backend-specific AsmDict omits the intermediate from its public keys.
    metadata_group = getattr(compiled, "metadata_group", None)
    if isinstance(metadata_group, Mapping):
        matches = [
            Path(str(path)).expanduser().resolve()
            for logical, path in metadata_group.items()
            if Path(str(logical)).suffix.lower() == f".{extension}"
        ]
        if len(matches) == 1 and matches[0].is_file():
            try:
                return matches[0].read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ProbeError(f"selected {extension} artifact is not UTF-8 text: {exc}") from exc
    raise ProbeError(f"selected CompiledKernel does not expose {extension.upper()} artifact")


def _artifact_record(compiled: Any, extension: str, text: str) -> dict[str, Any]:
    """Hash one compiler artifact after graph capture has completed."""

    text_bytes = text.encode("utf-8")
    metadata_group = getattr(compiled, "metadata_group", None)
    paths = []
    if isinstance(metadata_group, Mapping):
        paths = [
            (str(logical), Path(str(path)).expanduser().resolve())
            for logical, path in metadata_group.items()
            if Path(str(logical)).suffix.lower() == f".{extension}"
        ]
    path_record: dict[str, Any] = {}
    if len(paths) == 1 and paths[0][1].is_file():
        logical, path = paths[0]
        data = path.read_bytes()
        path_record = {
            "logical_name": logical,
            "path": str(path),
            "path_bytes": len(data),
            "path_sha256": _sha256_bytes(data),
            "path_matches_asm": data == text_bytes,
        }
    return {
        "extension": extension,
        "bytes": len(text_bytes),
        "sha256": _sha256_bytes(text_bytes),
        "line_count": len(text.splitlines()),
        "source": "CompiledKernel.asm" if isinstance(getattr(compiled, "asm", None), Mapping)
        and extension in getattr(compiled, "asm", {}) else "metadata_group",
        **path_record,
    }


def _strip_artifact_comments(text: str) -> str:
    return "\n".join(
        re.sub(r"//.*$|#.*$", "", line) for line in text.splitlines()
    )


def _artifact_analysis(text: str, extension: str) -> dict[str, Any]:
    """Count layout/memory/synchronization markers in one IR artifact."""

    clean = _strip_artifact_comments(text)
    patterns = {
        "convert_layout": r"(?:\bconvert_layout\b|(?:triton_gpu|ttg)\.convert_layout)",
        "shared": r"(?:\bshared\b|addrspace\(3\)|\.shared\b|\blocal_(?:alloc|load|store)\b)",
        "barrier": r"(?:\bbarrier\b|bar\.sync|barrier0|llvm\.nvvm\.barrier)",
        "shuffle": r"(?:\bshuffle\b|shfl\.|llvm\.nvvm\.shfl)",
        "local_loads": r"(?:ld\.local\b|\blocal_load\b|load[^\n]*addrspace\(5\))",
        "global_loads": r"(?:ld\.global\b|\bglobal_load\b|load[^\n]*addrspace\(1\))",
    }
    counts = {name: len(re.findall(pattern, clean)) for name, pattern in patterns.items()}
    instruction_counts: dict[str, int] = {}
    if extension == "ptx":
        for line in clean.splitlines():
            line = re.sub(r"^@!?%[A-Za-z0-9_]+\s+", "", line.strip())
            match = re.match(r"([A-Za-z_][A-Za-z0-9_.]*)", line)
            if match:
                instruction = match.group(1)
                instruction_counts[instruction] = instruction_counts.get(instruction, 0) + 1
    return {
        "op_counts": counts,
        "instruction_counts": instruction_counts,
        "flags": {
            "convert_layout": counts["convert_layout"] > 0,
            "shared": counts["shared"] > 0,
            "barrier": counts["barrier"] > 0,
            "shuffle": counts["shuffle"] > 0,
            "local_loads": counts["local_loads"] > 0,
            "global_loads": counts["global_loads"] > 0,
        },
        "instruction_count": sum(instruction_counts.values()),
    }


def _record_codegen(compiled: Any) -> dict[str, Any]:
    """Record all requested compiler artifacts after graph capture."""

    artifacts = {}
    for extension in ("ttir", "ttgir", "llir", "ptx"):
        text = _artifact_text(compiled, extension)
        artifacts[extension] = {
            **_artifact_record(compiled, extension, text),
            "analysis": _artifact_analysis(text, extension),
        }
    return {"artifacts": artifacts}


def _source_binding(target: Any, direction: str) -> dict[str, Any]:
    raw_function = getattr(getattr(target, "fn", None), "fn", None)
    if raw_function is None:
        raise ProbeError(f"cannot resolve selected FLA {direction} Python source function")
    try:
        path = Path(inspect.getsourcefile(raw_function) or "").resolve()
        line = inspect.getsourcelines(raw_function)[1]
    except (OSError, TypeError) as exc:
        raise ProbeError(f"cannot resolve selected FLA {direction} source binding: {exc}") from exc
    if not path.is_file():
        raise ProbeError(f"selected FLA {direction} source is not a file: {path}")
    return {
        "module": _jsonable(getattr(raw_function, "__module__", None)),
        "qualname": _jsonable(getattr(raw_function, "__qualname__", None)),
        "source_file": str(path),
        "source_line": line,
        "source_sha256": _sha256_file(path),
        "direction": direction,
    }


def _hardware_scope(torch_module: Any, requested: str | None) -> dict[str, Any]:
    capability = tuple(int(item) for item in torch_module.cuda.get_device_capability())
    name = str(torch_module.cuda.get_device_name())
    matching = [
        scope
        for scope, facts in HARDWARE.items()
        if tuple(facts["capability"]) == capability
    ]
    if len(matching) != 1:
        raise ProbeError(f"unsupported or ambiguous CUDA capability for this probe: {capability!r}")
    scope = matching[0]
    if scope not in name:
        raise ProbeError(
            f"CUDA capability {capability!r} maps to {scope}, but device name is {name!r}"
        )
    if requested is not None and requested != scope:
        raise ProbeError(f"requested hardware {requested} but observed {scope} ({name})")
    return {
        "scope": scope,
        "device_name": name,
        "capability": list(capability),
        "sm": HARDWARE[scope]["sm"],
    }


def _make_inputs(
    source_count: int, rank: int, torch_module: Any,
) -> tuple[dict[str, Any], tuple[Any, ...], Any]:
    _validate_case(source_count, rank)
    torch_module.manual_seed(20260830 + source_count * 100 + rank)
    query = torch_module.randn(rank, device="cuda", dtype=torch_module.float32)
    sources = tuple(
        torch_module.randn(
            BATCH_SIZE,
            TOKEN_COUNT,
            VALUE_WIDTH,
            device="cuda",
            dtype=torch_module.bfloat16,
        )
        for _ in range(source_count)
    )
    return {
        "call_function": f"{SOURCE_MODULE}.forward and {SOURCE_MODULE}.backward",
        "input_layout": "source_list",
        "source_count": source_count,
        "shape": {
            "batch": BATCH_SIZE,
            "tokens": TOKEN_COUNT,
            "D": VALUE_WIDTH,
            "R": rank,
        },
        "source_dtype": str(sources[0].dtype),
        "query_dtype": str(query.dtype),
        "device": str(sources[0].device),
        "eps": PRODUCTION_EPS,
        "scale": PRODUCTION_SCALE,
    }, sources, query


def _finite_tensors(tensors: Sequence[Any], torch_module: Any) -> bool:
    return all(
        bool(torch_module.isfinite(tensor).all().item())
        for tensor in tensors
        if isinstance(tensor, torch_module.Tensor)
    )


def _capture_forward(
    fla_module: Any, sources: Sequence[Any], query: Any, torch_module: Any,
    eps: float, scale: float,
) -> tuple[Any, dict[str, Any]]:
    """Capture and replay the direct production FLA forward route."""

    with torch_module.no_grad():
        warmup = fla_module.forward(sources, query, eps, scale)
    torch_module.cuda.synchronize()
    graph = torch_module.cuda.CUDAGraph()
    with torch_module.cuda.graph(graph):
        captured = fla_module.forward(sources, query, eps, scale)
    graph.replay()
    torch_module.cuda.synchronize()
    if not _finite_tensors(captured, torch_module):
        raise ProbeError("captured FLA forward outputs contain non-finite values")
    return captured, {
        "status": "captured_and_replayed",
        "warmup_output_count": len(warmup),
        "captured_output_count": len(captured),
        "replay_count": 1,
        "timed": False,
    }


def _capture_backward(
    fla_module: Any, sources: Sequence[Any], query: Any, forward_outputs: Sequence[Any],
    torch_module: Any, eps: float, scale: float,
) -> tuple[Any, dict[str, Any]]:
    """Capture and replay the direct production FLA backward route."""

    del eps  # Forward epsilon is already represented in the saved tensors.
    grad_output = torch_module.ones_like(forward_outputs[0])
    saved = tuple(forward_outputs[1:])
    with torch_module.no_grad():
        warmup = fla_module.backward(
            sources, query, saved[0], grad_output, saved[1], saved[2], saved[3], scale
        )
    torch_module.cuda.synchronize()
    graph = torch_module.cuda.CUDAGraph()
    with torch_module.cuda.graph(graph):
        captured = fla_module.backward(
            sources, query, saved[0], grad_output, saved[1], saved[2], saved[3], scale
        )
    graph.replay()
    torch_module.cuda.synchronize()
    if not _finite_tensors(captured, torch_module):
        raise ProbeError("captured FLA backward outputs contain non-finite values")
    return captured, {
        "status": "captured_and_replayed",
        "warmup_output_count": len(warmup),
        "captured_output_count": len(captured),
        "replay_count": 1,
        "grad_output_shape": list(grad_output.shape),
        "grad_output_dtype": str(grad_output.dtype),
        "timed": False,
    }


def _empty_cache_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ProbeError(f"cache directory must be new and empty: {path}")
    else:
        path.mkdir(parents=True)


def _run_case(
    source_count: int,
    rank: int,
    cache_dir: Path,
    requested_hardware: str | None = None,
) -> dict[str, Any]:
    """Observe, capture, and inspect both production FLA directions."""

    _validate_case(source_count, rank)
    cache_dir = cache_dir.expanduser().resolve()
    _empty_cache_dir(cache_dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
    # Keep all intermediate compiler artifacts in the fresh diagnostic cache.
    os.environ["TRITON_STORE_BINARY_ONLY"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import torch
    import triton

    if not str(triton.__version__).startswith("3.6."):
        raise ProbeError(
            "selected FLA codegen probe requires Triton 3.6.x, "
            f"got {triton.__version__}"
        )
    if not torch.cuda.is_available():
        raise ProbeError("--allow-gpu was supplied, but torch.cuda.is_available() is false")
    if os.environ.get("TRITON_INTERPRET", "0").lower() in {"1", "true", "yes"}:
        raise ProbeError("TRITON_INTERPRET disables selected binary observation")
    hardware = _hardware_scope(torch, requested_hardware)

    from attnres._kernels import fla_full_sources

    call_facts, sources, query = _make_inputs(source_count, rank, torch)
    eps = float(call_facts["eps"])
    scale = float(call_facts["scale"])

    def observe_call(
        target: Any, direction: str, call: Any,
    ) -> tuple[Any, dict[str, Any]]:
        if type(target).__name__ != "Autotuner":
            raise ProbeError(
                f"{SOURCE_MODULE}.{direction} target is not the expected Triton Autotuner"
            )
        observations: dict[str, Any] = {"calls": []}
        restore = _install_autotuner_observer(target, observations, direction)
        try:
            result = call()
        finally:
            restore()
        if len(observations["calls"]) != 1:
            raise ProbeError(
                f"expected one selected FLA {direction} Autotuner call, "
                f"got {len(observations['calls'])}"
            )
        event = observations["calls"][0]
        if event.get("status") != "returned":
            raise ProbeError(
                f"selected FLA {direction} Autotuner call did not return: {event!r}"
            )
        return result, event

    def inspect_direction(
        direction: str, target: Any, event: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        expected_key = _expected_tuning_key(direction, source_count, rank)
        tuning_key = tuple(event["tuning_key"])
        if tuning_key != expected_key:
            raise ProbeError(
                f"selected FLA {direction} Autotuner key mismatch: "
                f"expected {_jsonable(expected_key)!r}, got {_jsonable(tuning_key)!r}"
            )
        expected_suffixes = (
            FORWARD_DTYPE_SUFFIXES
            if direction == "forward"
            else BACKWARD_DTYPE_SUFFIXES
        )
        if event["binding"]["dtype_suffixes"] != list(expected_suffixes):
            raise ProbeError(
                f"selected FLA {direction} dtype suffix mismatch: "
                f"expected {expected_suffixes!r}, "
                f"got {event['binding']['dtype_suffixes']!r}"
            )
        selected = _selected_config(target, tuning_key, direction)
        device_index = int(torch.cuda.current_device())
        entries = _cache_entries(
            getattr(target, "fn", None), device_index, direction
        )
        identity, direct_entry = _validate_selected_binary(
            target,
            event.get("compiled_kernel"),
            entries,
            selected,
            direction,
            source_count,
            rank,
        )
        metadata_target = _metadata_value(
            getattr(event["compiled_kernel"], "metadata", None), "target"
        )
        metadata_arch = _metadata_value(metadata_target, "arch")
        expected_arch = HARDWARE[hardware["scope"]]["arch"]
        try:
            numeric_arch = int(str(metadata_arch).replace("sm", ""))
        except (TypeError, ValueError):
            numeric_arch = None
        if numeric_arch != expected_arch:
            raise ProbeError(
                f"selected FLA {direction} binary target arch mismatch: "
                f"expected {expected_arch}, got {metadata_arch!r}"
            )
        return (
            {
                "kernel": (
                    FORWARD_KERNEL_NAME
                    if direction == "forward"
                    else BACKWARD_KERNEL_NAME
                ),
                "autotuner_call_count": 1,
                "key_fields": list(AUTOTUNER_KEY_FIELDS),
                "tuning_key": _jsonable(tuning_key),
                "key_values": _jsonable(
                    tuning_key[: len(AUTOTUNER_KEY_FIELDS)]
                ),
                "dtype_suffixes": list(event["binding"]["dtype_suffixes"]),
                "call_binding": event["binding"],
                "selected": selected,
                "compiled_kernel": identity,
                "direct_cache_entry": {
                    "device": direct_entry["device"],
                    "jit_cache_key": _jsonable(direct_entry["cache_key"]),
                    "jit_specialization": _jsonable(
                        direct_entry["specialization"]
                    ),
                    "jit_options": _jsonable(direct_entry["options"]),
                },
            },
            identity,
        )

    forward_target = getattr(fla_full_sources, FORWARD_KERNEL_NAME, None)
    if forward_target is None:
        raise ProbeError(f"missing {FORWARD_KERNEL_NAME}")
    forward_result, forward_event = observe_call(
        forward_target,
        "forward",
        lambda: fla_full_sources.forward(sources, query, eps, scale),
    )
    torch.cuda.synchronize()
    forward_info, _forward_identity = inspect_direction(
        "forward", forward_target, forward_event
    )
    forward_outputs, forward_graph = _capture_forward(
        fla_full_sources, sources, query, torch, eps, scale
    )

    backward_target = getattr(fla_full_sources, BACKWARD_KERNEL_NAME, None)
    if backward_target is None:
        raise ProbeError(f"missing {BACKWARD_KERNEL_NAME}")
    grad_output = torch.ones_like(forward_outputs[0])
    saved = tuple(forward_outputs[1:])
    backward_result, backward_event = observe_call(
        backward_target,
        "backward",
        lambda: fla_full_sources.backward(
            sources,
            query,
            saved[0],
            grad_output,
            saved[1],
            saved[2],
            saved[3],
            scale,
        ),
    )
    torch.cuda.synchronize()
    backward_info, _backward_identity = inspect_direction(
        "backward", backward_target, backward_event
    )
    # Keep the selected forward graph outputs as the static saved tensors for
    # this capture.  Artifact reads happen only after both graph operations.
    backward_outputs, backward_graph = _capture_backward(
        fla_full_sources,
        sources,
        query,
        forward_outputs,
        torch,
        eps,
        scale,
    )

    forward_binding = _source_binding(forward_target, "forward")
    backward_binding = _source_binding(backward_target, "backward")
    for direction, binding in (
        ("forward", forward_binding),
        ("backward", backward_binding),
    ):
        if binding["source_sha256"] != SOURCE_SHA256:
            raise ProbeError(
                f"selected FLA {direction} source hash differs from exact "
                f"implementation {IMPLEMENTATION_COMMIT}: expected {SOURCE_SHA256}, "
                f"got {binding['source_sha256']}"
            )

    # This is deliberately after graph capture/replay and after every observer
    # is restored.  No artifact I/O participates in any route launch.
    forward_info["post_capture_codegen"] = _record_codegen(
        forward_event["compiled_kernel"]
    )
    backward_info["post_capture_codegen"] = _record_codegen(
        backward_event["compiled_kernel"]
    )

    return {
        "status": "passed",
        "probe": "selected_fla_block_codegen",
        "source_revision": SOURCE_REVISION,
        "base_commit": BASE_COMMIT,
        "implementation_tree": IMPLEMENTATION_TREE,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "triton": str(triton.__version__),
        "torch": str(torch.__version__),
        "hardware": hardware,
        "production_geometry": {
            "source_count": source_count,
            "L2": _expected_l2(source_count),
            "D": VALUE_WIDTH,
            "R": rank,
            "is_r_equals_d": rank == VALUE_WIDTH,
            "batch": BATCH_SIZE,
            "tokens": TOKEN_COUNT,
        },
        "call": call_facts,
        "directions": {
            "forward": forward_info,
            "backward": backward_info,
        },
        "selected_backward": {
            "family": backward_info["selected"]["layout_family"],
            "BL": backward_info["selected"]["block"],
            "num_warps": backward_info["selected"]["config"]["num_warps"],
            "num_stages": backward_info["selected"]["config"]["num_stages"],
            "candidate_count": backward_info["selected"]["candidate_count"],
            "candidate_family_counts": backward_info["selected"][
                "candidate_family_counts"
            ],
            "dtype_key": backward_info["tuning_key"],
        },
        "graph": {
            "forward": forward_graph,
            "backward": backward_graph,
        },
        "outputs": {
            "warmup_forward_count": len(forward_result),
            "captured_forward_count": len(forward_outputs),
            "warmup_backward_count": len(backward_result),
            "captured_backward_count": len(backward_outputs),
            "captured_forward_finite": _finite_tensors(forward_outputs, torch),
            "captured_backward_finite": _finite_tensors(backward_outputs, torch),
        },
        "timing": {
            "status": "not_timed",
            "timed_regions": 0,
            "boundary": "none; warm-up plus graph capture/replay only",
            "autotuner_benchmarking": (
                "Triton's internal selection was observed; no external timing "
                "or benchmark wrapper was installed"
            ),
        },
        "source_bindings": {
            "forward": forward_binding,
            "backward": backward_binding,
        },
        "cache_dir": str(cache_dir),
    }

def _subprocess_env(cache_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["TRITON_CACHE_DIR"] = str(cache_dir)
    environment["TRITON_PRINT_AUTOTUNING"] = "0"
    environment["TRITON_STORE_BINARY_ONLY"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    pythonpath = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return environment


def _run_subprocess(
    source_count: int,
    rank: int,
    cache_dir: Path,
    hardware: str | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROBE_PATH),
        "--source-count",
        str(source_count),
        "--rank",
        str(rank),
        "--cache-dir",
        str(cache_dir),
        "--allow-gpu",
    ]
    if hardware is not None:
        command.extend(("--hardware", hardware))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(cache_dir),
        capture_output=True,
        text=True,
        timeout=720,
        check=False,
    )
    if completed.returncode:
        raise ProbeError(
            f"S={source_count} R={rank} subprocess failed with {completed.returncode}:\n"
            f"stdout:\n{completed.stdout[-8000:]}\n"
            f"stderr:\n{completed.stderr[-8000:]}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                return json.loads(line[len(RESULT_MARKER):])
            except json.JSONDecodeError as exc:
                raise ProbeError(f"malformed selected codegen result: {exc}") from exc
    raise ProbeError(
        f"S={source_count} R={rank} emitted no {RESULT_MARKER!r}:\n"
        f"stdout:\n{completed.stdout[-8000:]}\n"
        f"stderr:\n{completed.stderr[-8000:]}"
    )


def run_probe(
    cache_root: str | os.PathLike[str] | None = None,
    *,
    hardware: str | None = None,
    source_counts: Sequence[int] = SOURCE_COUNTS,
    ranks: Sequence[int] = RANKS,
) -> dict[str, Any]:
    """Run one fresh process for every production S/R geometry."""

    if hardware is not None and hardware not in HARDWARE:
        raise ValueError(f"hardware must be one of {tuple(HARDWARE)}, got {hardware!r}")
    source_counts = tuple(int(value) for value in source_counts)
    ranks = tuple(int(value) for value in ranks)
    for source_count in source_counts:
        for rank in ranks:
            _validate_case(source_count, rank)
    root = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else Path("/tmp") / f"attnres-selected-fla-codegen-{os.getpid()}"
    )
    _empty_cache_dir(root)
    rows: dict[str, Any] = {}
    for source_count in source_counts:
        rows[str(source_count)] = {}
        for rank in ranks:
            case_cache = root / f"S{source_count}_R{rank}"
            case_cache.mkdir()
            rows[str(source_count)][str(rank)] = _run_subprocess(
                source_count, rank, case_cache, hardware
            )
    return {
        "status": "passed",
        "probe": "selected_fla_block_codegen",
        "base_commit": BASE_COMMIT,
        "implementation_tree": IMPLEMENTATION_TREE,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "source_revision": SOURCE_REVISION,
        "hardware_scope": hardware or "observed_per_case",
        "source_counts": list(source_counts),
        "ranks": list(ranks),
        "cases": rows,
        "timing": "not_timed",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-count", type=int, choices=SOURCE_COUNTS, required=True)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--hardware", choices=tuple(HARDWARE), help="assert H100 or B200 scope")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="required explicit consent to import CUDA/Triton and launch one public call",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.allow_gpu:
        raise SystemExit("refusing a GPU launch; pass --allow-gpu when running the probe")
    try:
        result = _run_case(args.source_count, args.rank, args.cache_dir, args.hardware)
    except ProbeError as exc:
        raise SystemExit(
            f"selected FLA forward/backward codegen probe failed: {exc}"
        ) from exc
    encoded = json.dumps(result, sort_keys=True)
    if args.output is not None:
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(RESULT_MARKER + encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
