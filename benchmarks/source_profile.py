"""Paired diagnostic for source-list and packed Full Attention Residuals.

This is separate from the frozen acceptance runner. Every arm is qualified
with ``validation.oracle`` before timing, and every raw timing row is kept,
including failures and skipped rows. This profile makes no model-level speed
claim.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from attnres import attnres

from .run import (
    PROJECT_ROOT,
    _exception,
    _finite,
    _jsonable,
    _max_abs,
    _operator_digest,
    _paired_samples,
    _seeded_randn,
    _tolerance,
    load_protocol,
)
from .statistics import simultaneous_paired_ratio_bootstrap


MODES = ("forward", "forward_backward")
METHODS = ("eager", "cuda_graph")
EPS = 2**-23


def _fail(phase: str, **fields: Any) -> dict[str, Any]:
    return {"phase": phase, **fields}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True).encode()).hexdigest()


def _case(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise TypeError("each case must be a mapping with variant and shape")
    variant = str(item.get("variant", "")).lower()
    if variant not in {"standard", "sliced"}:
        raise ValueError("case variant must be standard or sliced")
    shape = tuple(int(value) for value in item.get("shape", ()))
    if len(shape) != 4:
        raise ValueError("case shape must be [S,N,D,R]")
    if min(shape) < 1 or shape[3] > shape[2]:
        raise ValueError("case shape must satisfy positive dimensions and R<=D")
    if variant == "standard" and shape[3] != shape[2]:
        raise ValueError("standard cases require R=D")
    result = {"id": str(item.get("id", f"case_{index}")), "variant": variant,
              "shape": list(shape)}
    return result


def _leaf(shape: Sequence[int], dtype: torch.dtype, device: torch.device, seed: int, grad: bool) -> torch.Tensor:
    return _seeded_randn(shape, dtype=dtype, device=device, seed=seed).detach().clone().requires_grad_(grad)


def _clone(tensor: torch.Tensor, grad: bool) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(grad)


def _pack(values: Sequence[torch.Tensor], grad: bool = True) -> torch.Tensor:
    # The detach/clone makes an independent packed leaf; source-list values
    # are never produced by unbinding this tensor.
    return torch.stack(tuple(value.detach() for value in values), dim=0).detach().clone().requires_grad_(grad)


def _make_arms(
    case: Mapping[str, Any], device: torch.device, seed: int, baseline: bool,
) -> dict[str, dict[str, Any]]:
    sources, rows, width, rank = tuple(case["shape"])
    dtype = torch.bfloat16
    values = tuple(_leaf((rows, width), dtype, device, seed + 11 + 17 * index, True) for index in range(sources))
    query_data = _leaf((rank,), dtype, device, seed + 401, False)
    with torch.no_grad():
        query_data.mul_(0.02)
    upstream_data = _leaf((rows, width), dtype, device, seed + 402, False)
    packed_values = _pack(values)

    def arm(v: Any, q: torch.Tensor, u: torch.Tensor) -> dict[str, Any]:
        return {"values": v, "query": q, "upstream": u}

    result = {
        "source_list": arm(
            values,
            _clone(query_data, True),
            _clone(upstream_data, False),
        ),
        "packed": arm(
            packed_values,
            _clone(query_data, True),
            _clone(upstream_data, False),
        ),
    }
    if baseline:
        result["frozen_source"] = arm(
            tuple(_clone(value, True) for value in values),
            _clone(query_data, True),
            _clone(upstream_data, False),
        )
    return result


def _tensors(arm: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    values = arm["values"] if isinstance(arm["values"], (tuple, list)) else (arm["values"],)
    return (*values, arm["query"])


def _leaves(arm: Mapping[str, Any]) -> dict[str, Any]:
    values = arm["values"] if isinstance(arm["values"], (tuple, list)) else (arm["values"],)
    result = {
        "all_leaves": all(tensor.is_leaf and tensor.grad_fn is None for tensor in _tensors(arm)),
        "source_value_count": len(values),
        "value_strides": [list(tensor.stride()) for tensor in values],
        "query_stride": list(arm["query"].stride()),
    }
    return result


def _clone_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Clone an arm while preserving independent source leaves."""
    values = arm["values"] if isinstance(arm["values"], (tuple, list)) else _clone(arm["values"], True)
    if isinstance(arm["values"], (tuple, list)):
        values = tuple(_clone(value, True) for value in arm["values"])
    return {
        "values": values,
        "query": _clone(arm["query"], True),
        "upstream": _clone(arm["upstream"], False),
    }


def _call(function: Callable[..., Any], arm: Mapping[str, Any]) -> torch.Tensor:
    output = function(arm["values"], arm["query"], eps=EPS, scale=1.0)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"operator returned {type(output).__name__}; expected a tensor")
    return output


def _oracle_reference(
    arm: Mapping[str, Any], *, with_gradients: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Build an independent oracle over the same source values and query."""
    from validation.oracle import oracle

    values = arm["values"] if isinstance(arm["values"], (tuple, list)) else (arm["values"],)
    oracle_values = _pack(values) if isinstance(arm["values"], (tuple, list)) else _clone(arm["values"], True)
    expected_query = _clone(arm["query"], True)
    with torch.enable_grad():
        expected = oracle(oracle_values, expected_query, eps=EPS, scale=1.0)
        if not with_gradients:
            return expected, ()
        oracle_grads = torch.autograd.grad(
            expected,
            (oracle_values, expected_query),
            arm["upstream"],
            allow_unused=False,
        )
    expected_inputs = tuple(oracle_grads[0][index] for index in range(len(values))) if isinstance(arm["values"], (tuple, list)) else (oracle_grads[0],)
    return expected, expected_inputs + (oracle_grads[1],)


def _qualify(function: Callable[..., Any], arm: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    actual = _clone_arm(arm)
    with torch.enable_grad():
        output = _call(function, actual)
    expected, expected_inputs = _oracle_reference(actual, with_gradients=True)
    tolerance = _tolerance(protocol, torch.bfloat16)
    _finite(output, "operator output")
    _finite(expected, "oracle output")
    torch.testing.assert_close(output, expected, **tolerance)
    actual_grads = torch.autograd.grad(output, _tensors(actual), actual["upstream"], allow_unused=False)
    if len(actual_grads) != len(expected_inputs):
        raise RuntimeError(f"gradient list length mismatch: {len(actual_grads)} != {len(expected_inputs)}")
    errors = []
    for index, (actual_grad, expected_grad) in enumerate(zip(actual_grads, expected_inputs)):
        _finite(actual_grad, f"operator gradient {index}")
        _finite(expected_grad, f"oracle gradient {index}")
        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)
        errors.append(_max_abs(actual_grad, expected_grad))
    return {"status": "qualified", "oracle": "validation.oracle.oracle", "tolerance": tolerance, "output_max_abs": _max_abs(output, expected), "gradient_max_abs": errors, "input_leaves": _leaves(actual)}


def _step(function: Callable[..., Any], arm: Mapping[str, Any], mode: str) -> Any:
    if mode == "forward":
        with torch.no_grad():
            return _call(function, arm)
    with torch.enable_grad():
        output = _call(function, arm)
        gradients = torch.autograd.grad(output, _tensors(arm), arm["upstream"], allow_unused=False)
        return output, gradients


def _event(function: Callable[[], Any], device: torch.device) -> tuple[float, Any]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(device):
        start.record()
        value = function()
        end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)), value


def _graph(function: Callable[..., Any], arm: Mapping[str, Any], mode: str, device: torch.device, warmup: int) -> dict[str, Any]:
    static = _clone_arm(arm)
    stream, current = torch.cuda.Stream(device=device), torch.cuda.current_stream(device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(max(2, warmup)):
            warmup_result = _step(function, static, mode)
            del warmup_result
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        captured = _step(function, static, mode)
    stream.synchronize()
    current.wait_stream(stream)
    if mode == "forward_backward":
        output, gradients = captured
    else:
        output, gradients = captured, ()
    return {"graph": graph, "static": static, "output": output, "grads": gradients, "capture_host_ms": (time.perf_counter() - started) * 1000.0}


def _check_captured(state, mode, protocol):
    """Compare captured results with independent leaves and the frozen oracle."""
    static = state["static"]
    expected, expected_grads = _oracle_reference(
        static, with_gradients=mode == "forward_backward"
    )
    tolerance = _tolerance(protocol, torch.bfloat16)
    _finite(state["output"], "captured output")
    torch.testing.assert_close(state["output"], expected, **tolerance)
    if mode == "forward_backward":
        if len(state["grads"]) != len(expected_grads):
            raise AssertionError("captured gradient count differs from oracle")
        for actual, expected_gradient in zip(state["grads"], expected_grads):
            _finite(actual, "captured gradient")
            torch.testing.assert_close(actual, expected_gradient, **tolerance)


def _verify_graph(state, original, mode, protocol):
    """Check initial and changed replay, then restore the exact timed inputs."""
    static = state["static"]
    tensors = (*_tensors(static), static["upstream"])
    originals = (*_tensors(original), original["upstream"])
    if len(tensors) != len(originals):
        raise AssertionError("graph input count changed")
    state["graph"].replay()
    _check_captured(state, mode, protocol)
    try:
        with torch.no_grad():
            for tensor in tensors:
                tensor.mul_(0.75).add_(0.03125)
        state["graph"].replay()
        _check_captured(state, mode, protocol)
    finally:
        with torch.no_grad():
            for tensor, initial in zip(tensors, originals):
                tensor.copy_(initial)
    state["graph"].replay()
    result = {
        "initial_and_changed_inputs": "qualified",
        "timed_inputs_restored": True,
        "gradient_count": len(state["grads"]),
        "changed_input_transform": "x * 0.75 + 0.03125; no RNG consumed",
    }
    return result


def _ptx_counts(compiled: Any) -> dict[str, int]:
    try:
        asm = getattr(compiled, "asm")
        ptx = asm.get("ptx") if isinstance(asm, Mapping) else None
    except Exception:
        return {}
    if not isinstance(ptx, str):
        return {}
    counts: dict[str, int] = {}
    for line in ptx.splitlines():
        tokens = line.split("//", 1)[0].split()
        if tokens and tokens[0].startswith("@"):
            tokens = tokens[1:]
        token = tokens[0] if tokens else ""
        if token.startswith(("ld.global", "ld.local", "st.global", "st.local")):
            counts[token] = counts.get(token, 0) + 1
    return counts


def _resource_metadata(variants: Sequence[str]) -> dict[str, Any]:
    """Read numeric fields from cached Triton CompiledKernel metadata only."""
    rows, attempted, errors = [], [], []
    # Inspect the shared packed/source fixed-tail core only when an execution
    # path has already loaded it; reading it from ``sys.modules`` avoids
    # importing or launching it merely to populate a cache.  The baseline
    # loader gives frozen packages an ``_attnres_frozen_<hash>`` namespace, so
    # retain any already-loaded core aliases for the existing ``frozen_source``
    # comparator as well.
    fixed_tail_modules = []
    if sys.modules.get("attnres._kernels.fixed_tail") is not None:
        fixed_tail_modules.append("attnres._kernels.fixed_tail")
    fixed_tail_modules.extend(
        name
        for name in sorted(sys.modules)
        if name.startswith("_attnres_frozen_")
        and name.endswith("._kernels.fixed_tail")
        and sys.modules.get(name) is not None
    )
    module_names = fixed_tail_modules
    seen_modules = set()
    for module_name in module_names:
        if module_name in seen_modules:
            continue
        seen_modules.add(module_name)
        try:
            module = sys.modules[module_name]
        except Exception as exc:
            errors.append({"module": module_name, "error": _exception(exc)})
            continue
        for name, function in vars(module).items():
            if type(function).__name__ == "Autotuner":
                function = function.fn
            if "kernel" not in name.lower() or type(function).__name__ != "JITFunction":
                continue
            kernel_name = f"{module_name}.{name}"
            attempted.append(kernel_name)
            caches = []
            for attribute in ("device_caches", "cache"):
                try:
                    value = getattr(function, attribute)
                except Exception:
                    continue
                if isinstance(value, Mapping):
                    for index, cached in enumerate(value.values()):
                        if index >= 64:
                            break
                        # Triton 3.6 stores device_caches[device] as
                        # (kernel_cache, key_cache, target, backend, binder).
                        if isinstance(cached, (tuple, list)) and cached and isinstance(cached[0], Mapping):
                            caches.append(cached[0])
                        else:
                            caches.append(cached)
            compiled = []
            for value in caches:
                if type(value).__name__ == "CompiledKernel":
                    compiled.append(value)
                elif isinstance(value, Mapping):
                    for index, item in enumerate(value.values()):
                        if index >= 64:
                            break
                        if type(item).__name__ == "CompiledKernel":
                            compiled.append(item)
            for item in compiled:
                resources = {}
                for field in ("n_regs", "n_spills"):
                    try:
                        value = getattr(item, field)
                    except Exception:
                        value = None
                    if isinstance(value, (bool, int, float)):
                        resources[field] = value
                try:
                    metadata = getattr(item, "metadata")
                except Exception:
                    metadata = None
                if isinstance(metadata, Mapping):
                    for field in ("shared", "num_warps", "num_stages"):
                        value = metadata.get(field)
                        if isinstance(value, (bool, int, float)):
                            resources[field] = value
                else:
                    for field in ("shared", "num_warps", "num_stages"):
                        try:
                            value = getattr(metadata, field)
                        except Exception:
                            value = None
                        if isinstance(value, (bool, int, float)):
                            resources[field] = value
                # Constants identify source-list versus packed specializations
                # when the same JITFunction owns both cache entries.
                try:
                    source = getattr(item, "src")
                    constants = getattr(source, "constants")
                    arg_names = getattr(getattr(source, "fn"), "arg_names")
                except Exception:
                    constants, arg_names = None, ()
                if not isinstance(arg_names, (tuple, list)):
                    arg_names = ()
                if isinstance(constants, Mapping):
                    selected = {}
                    for key, value in constants.items():
                        index = key[0] if isinstance(key, tuple) and len(key) == 1 else key
                        name = arg_names[index] if isinstance(index, int) and 0 <= index < len(arg_names) else str(key)
                        if name in {"D", "R", "S", "N_SOURCES", "L2", "LIST_SOURCES", "TOKEN_BLOCK", "BL", "SOURCE_TILE", "BLOCK_S", "BLOCK_D", "BLOCK_R", "BLOCK_N"} and isinstance(value, (bool, int, float)):
                            selected[name] = value
                    if selected:
                        resources["constants"] = selected
                ptx_counts = _ptx_counts(item)
                if ptx_counts:
                    resources["ptx_instruction_counts"] = ptx_counts
                if resources:
                    rows.append({"kernel": kernel_name, "resources": resources, "resource_units": {"n_spills": "compiler-reported; unit unspecified"} if "n_spills" in resources else {}, "resource_source": "triton_callable_cache"})
    if rows:
        return {"status": "available", "resource_source": "triton_callable_cache", "resource_scope": "all cached compiled variants; not a runtime trace or selected-autotune claim", "kernels": rows, "attempted_kernels": attempted, "errors": errors}
    return {"status": "unavailable", "resource_source": "triton_callable_cache", "kernels": [], "attempted_kernels": attempted, "errors": errors, "reason": "no cached CompiledKernel numeric metadata exposed"}


def _metric(functions: Mapping[str, Callable[..., Any]], arms: Mapping[str, Mapping[str, Any]], names: Sequence[str], active: Sequence[str], case: Mapping[str, Any], input_hash: str, mode: str, method: str, device: torch.device, rounds: int, warmup: int, replays: int, seed: int, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ready = list(active)
    failures: list[dict[str, Any]] = []
    states, graph_info = {}, {}
    for name in list(ready):
        try:
            if method == "cuda_graph":
                states[name] = _graph(functions[name], arms[name], mode, device, warmup)
                verified = _verify_graph(states[name], arms[name], mode, protocol)
                graph_info[name] = {"status": "captured", "capture_host_ms": states[name]["capture_host_ms"], "qualification": verified}
            else:
                for _ in range(max(1, warmup)):
                    warmup_result = _step(functions[name], arms[name], mode)
                    del warmup_result
                    torch.cuda.synchronize(device)
        except Exception as exc:
            ready.remove(name)
            failures.append(_fail("timing_setup", case=case["id"], variant=case["variant"], arm=name, mode=mode, method=method, status="failed", error=_exception(exc)))
            if method == "cuda_graph":
                graph_info[name] = {"status": "failed", "error": _exception(exc)}
    failed = set(names) - set(ready)
    rng = random.Random(seed + sum(ord(char) for char in f"{method}:{mode}"))

    def row(name: str, sample: int, order: int | None) -> dict[str, Any]:
        if method == "eager":
            boundary = "one eager forward operator call; host setup outside event" if mode == "forward" else "one eager forward plus torch.autograd.grad; host setup outside event"
        else:
            boundary = "fixed graph forward replay(s); input copies/capture outside event" if mode == "forward" else "fixed graph forward plus torch.autograd.grad replay(s); input copies/capture outside event and returned gradients retained"
        row = {"case": case["id"], "variant": case["variant"], "shape": case["shape"], "input_hash": input_hash, "arm": name, "mode": mode, "timing_method": method, "sample_index": sample, "order_index": order, "replay_count": 1 if method == "eager" else replays, "status": "pending", "ms": None, "timing_boundary": boundary}
        return row

    def measure(name: str, _sample: int) -> Mapping[str, Any]:
        if method == "eager":
            elapsed, timed_result = _event(lambda: _step(functions[name], arms[name], mode), device)
            del timed_result
        else:
            elapsed, _ = _event(lambda: [states[name]["graph"].replay() for _ in range(replays)], device)
        normalized = elapsed / replays if method == "cuda_graph" else elapsed
        return {"status": "ok", "elapsed_ms": elapsed, "normalized_ms": normalized, "ms": normalized}

    raw = _paired_samples(names, ready, rounds, rng, failed, row, measure, lambda _name, _row: failures, "timing")
    packed = [item["ms"] for item in raw if item["arm"] == "packed" and item["status"] == "ok"]
    source = [item["ms"] for item in raw if item["arm"] == "source_list" and item["status"] == "ok"]
    frozen = [item["ms"] for item in raw if item["arm"] == "frozen_source" and item["status"] == "ok"]
    comparisons = {}
    if len(packed) == len(source) == rounds:
        comparisons["source_over_packed"] = (packed, source)
    if len(frozen) == len(source) == rounds:
        comparisons["source_over_frozen_source"] = (frozen, source)
    if comparisons:
        statistics = {"status": "complete", "comparisons": simultaneous_paired_ratio_bootstrap(comparisons, samples=int(protocol["bootstrap_samples"]), seed=seed + 17, margin=float(protocol["plateau_margin"]))}
    else:
        statistics = {"status": "incomplete", "reason": "missing timing samples; no cases are inferred", "sample_counts": {name: sum(item["arm"] == name and item["status"] == "ok" for item in raw) for name in names}}
    metric = {"status": "complete" if not failures else "failed", "raw_samples": raw, "statistics": statistics, "graph": graph_info if method == "cuda_graph" else {}}
    return metric, failures


def run_source_profile(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the fixed BF16 Full source-list diagnostic from one config."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    try:
        protocol, frozen = load_protocol(PROJECT_ROOT)
    except Exception as exc:
        return {"status": "failed", "profile": "source_list_full_diagnostic", "failures": [_fail("contract", error=_exception(exc))]}
    failures: list[dict[str, Any]] = []
    cases, invalid_case_rows = [], []
    try:
        for index, item in enumerate(config.get("cases", ())):
            try:
                cases.append(_case(item, index))
            except Exception as exc:
                error = _exception(exc)
                failures.append(_fail("case", case=f"case_{index}", status="failed", error=error))
                invalid_case_rows.append({"case": f"case_{index}", "status": "failed", "error": error})
        seed, warmup = int(config.get("seed", 20260827)), int(config.get("warmup", protocol["warmup"]))
        rounds, replays = int(config.get("rounds", protocol["smoke_rounds"])), int(config.get("graph_replays", 10))
        include_baseline = bool(config.get("include_baseline", False))
        if warmup < 0 or rounds < 1 or replays < 1:
            raise ValueError("warmup must be non-negative; rounds and graph_replays must be positive")
    except Exception as exc:
        failures.append(_fail("config", error=_exception(exc)))
        cases, seed, warmup, rounds, replays, include_baseline = [], 20260827, 1, 1, 1, False
    baseline, baseline_info = None, {"requested": include_baseline, "alias": "frozen_source", "status": "not_requested" if not include_baseline else "pending"}
    if include_baseline:
        try:
            from .baseline import load_baseline

            baseline = load_baseline()
            baseline_info.update(status="loaded", metadata=baseline.metadata, source_hash=_digest(baseline.metadata))
        except Exception as exc:
            baseline_info.update(status="failed", error=_exception(exc))
            failures.append(_fail("baseline", status="failed", error=_exception(exc)))
    result: dict[str, Any] = {
        "status": "incomplete",
        "profile": "source_list_full_diagnostic",
        "operator": "Full",
        "dtype": "torch.bfloat16",
        "oracle": "validation.oracle.oracle",
        "input_contract": {
            "source_values": "independent leaf tensors per source; no unbind views",
            "packed_values": "detach/clone packed leaf with identical logical data",
            "query_upstream": "same nonzero logical data in independent tensors",
            "query_dtype": "torch.bfloat16",
            "query_init_std": 0.02,
            "upstream_dtype": "torch.bfloat16",
            "input_hash_schema": "values-query-upstream-v1",
            "comparison_boundary": "standalone operator; packing and producer-gradient assembly excluded",
        },
        "protocol": {"path": str(PROJECT_ROOT / "validation/protocol.json"), "frozen": frozen, "tolerance": protocol["bf16"]},
        "config": _jsonable(dict(config)),
        "normalized": {"seed": seed, "warmup": warmup, "rounds": rounds, "graph_replays": replays, "include_baseline": include_baseline},
        "baseline": baseline_info,
        "cases": list(invalid_case_rows),
        "failures": failures,
    }
    if not cases:
        result["failures"].append(_fail("config", status="incomplete", reason="no cases requested"))
        result["status"] = "failed" if invalid_case_rows else "incomplete"
        return _jsonable(result)
    if not torch.cuda.is_available():
        reason = "CUDA is unavailable; no qualification or timing was run"
        for case in cases:
            result["cases"].append({"case": case["id"], **case, "status": "unavailable", "reason": reason, "qualification": {}, "timing": {}, "failures": [_fail("device", status="unavailable", reason=reason)]})
        result["failures"].append(_fail("device", status="unavailable", reason=reason))
        result["resource_metadata"] = _resource_metadata(sorted({case["variant"] for case in cases}))
        return _jsonable(result)
    for index, case in enumerate(cases):
        device = torch.device("cuda")
        arms = _make_arms(case, device, seed + index * 1000, baseline is not None)
        input_hash = _operator_digest(
            arms["packed"]["values"], arms["packed"]["query"],
            arms["packed"]["upstream"],
        )
        functions: dict[str, Callable[..., Any]] = {"source_list": attnres, "packed": attnres}
        names = ["source_list", "packed"]
        if include_baseline:
            names.append("frozen_source")
            if baseline is not None:
                functions["frozen_source"] = baseline
        qualification, row_failures, active = {}, [], []
        for name in names:
            if name not in functions:
                qualification[name] = {"status": "unavailable", "reason": "requested frozen baseline was not loaded"}
                row_failures.append(_fail("qualification", case=case["id"], variant=case["variant"], arm=name, status="failed", reason="requested frozen baseline was not loaded"))
                continue
            try:
                qualification[name] = _qualify(functions[name], arms[name], protocol)
                active.append(name)
            except Exception as exc:
                qualification[name] = {"status": "failed", "error": _exception(exc)}
                row_failures.append(_fail("qualification", case=case["id"], variant=case["variant"], arm=name, status="failed", error=_exception(exc)))
        timings, timing_failures = {}, []
        for method in METHODS:
            timings[method] = {}
            for mode in MODES:
                metric, metric_failures = _metric(functions, arms, names, active, case, input_hash, mode, method, device, rounds, warmup, replays, seed + index * 1000, protocol)
                timings[method][mode] = metric
                timing_failures.extend(metric_failures)
        row_failures.extend(timing_failures)
        result["cases"].append({"case": case["id"], **case, "input_hash": input_hash, "status": "complete" if not row_failures and len(active) == len(names) else "failed", "qualification": qualification, "leaf_checks": {name: _leaves(arms[name]) if name in arms else {"status": "unavailable", "reason": "requested frozen baseline was not loaded"} for name in names}, "timing": timings, "failures": row_failures})
        result["failures"].extend(row_failures)
    result["resource_metadata"] = _resource_metadata(sorted({case["variant"] for case in cases}))
    result["status"] = "complete" if not result["failures"] and all(case["status"] == "complete" for case in result["cases"]) else "failed"
    return _jsonable(result)


__all__ = ["run_source_profile"]
