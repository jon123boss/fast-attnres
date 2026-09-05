"""Frozen AttnRes correctness and timing runner.

CUDA and optional comparator/model imports are phase local.  This module never
launches Modal
benchmarks.modal_runner transports a JSON configuration here.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
import torch
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "benchmarks"
from .statistics import simultaneous_paired_ratio_bootstrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FILES = (
    "EVALUATION.md",
    "validation/oracle.py",
    "validation/protocol.json",
    "tests/test_reference.py",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, torch.dtype): return str(value).removeprefix("torch.")
    if isinstance(value, Mapping): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, str)): return value
    if isinstance(value, float): return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        try: return _jsonable(item())
        except Exception: pass
    return str(value)


def _exception(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc, limit=8))}


def _failure(phase: str, **fields: Any) -> dict[str, Any]: return {"phase": phase, **fields}


def _model_progress_logger(config: Mapping[str, Any]) -> Callable[[str, str | None], None]:
    """Return an opt-in host-only progress logger for model orchestration."""

    if not bool(config.get("model_progress", False)):
        return lambda stage, arm=None: None
    started = time.monotonic()

    def log(stage: str, arm: str | None = None) -> None:
        try:
            row = {"stage": stage, "arm": arm, "elapsed_s": time.monotonic() - started}
            print(json.dumps(row, separators=(",", ":"), sort_keys=True), flush=True)
        except Exception:
            # Progress output must not change benchmark failure behavior.
            pass

    return log


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_hashes(project_root: str | os.PathLike[str] | None = None) -> dict[str, str]:
    root = Path(project_root or PROJECT_ROOT).resolve()
    expected = json.loads((root / "validation/frozen.json").read_text())
    actual, mismatches = {}, []
    for name, digest in expected.items():
        path = root / name
        if not path.is_file(): mismatches.append(f"{name}: missing")
        else:
            actual[name] = sha256_file(path)
            if actual[name] != digest: mismatches.append(f"{name}: expected {digest}, got {actual[name]}")
    if mismatches: raise RuntimeError("frozen validation contract mismatch: " + "; ".join(mismatches))
    return actual


def load_protocol(project_root: str | os.PathLike[str] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    root = Path(project_root or PROJECT_ROOT).resolve()
    hashes = assert_frozen_hashes(root)
    return json.loads((root / "validation/protocol.json").read_text()), hashes


def _git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try: return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError): return None
    return {"revision": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def _environment(root: Path) -> dict[str, Any]:
    try: triton = importlib.import_module("triton").__version__
    except Exception: triton = None
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "hostname": socket.gethostname(), "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "triton": triton, "git": _git_info(root),
            "env": {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "FLA_ATTNRES_GLUON") if k in os.environ}}


def _device_info(device: torch.device | None = None) -> dict[str, Any]:
    result = {"requested": str(device) if device is not None else "cuda", "type": "cuda", "available": bool(torch.cuda.is_available())}
    if not result["available"]:
        result["count"] = 0
        return result
    selected = device or torch.device("cuda", torch.cuda.current_device())
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        props = torch.cuda.get_device_properties(index)
        result.update(index=index, count=torch.cuda.device_count(), name=torch.cuda.get_device_name(index),
                      capability=list(torch.cuda.get_device_capability(index)), total_memory=int(props.total_memory),
                      multi_processor_count=int(props.multi_processor_count))
    return result


def _source_hashes(root: Path, frozen: Mapping[str, str], vendor: Mapping[str, Any] | None = None) -> dict[str, Any]:
    paths = sorted([*(root / "src").rglob("*.py"), *(root / "benchmarks").rglob("*.py")])
    project = {str(path.relative_to(root)): sha256_file(path) for path in paths}
    result = {"frozen": dict(frozen), "project": project, "vendor": dict(vendor or {})}
    result["software_hash"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def _hardware_hash(device: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(device), sort_keys=True).encode()).hexdigest()


def _dtype(value: Any, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    if isinstance(value, torch.dtype):
        if value == torch.bfloat16:
            return value
        raise ValueError(f"benchmark dtype must be BF16; got {value!r}")
    name = str(default if value is None else value).lower().replace("torch.", "")
    if name in {"bf16", "bfloat16"}: return torch.bfloat16
    raise ValueError(f"benchmark dtype must be BF16; got {value!r}")


def _tolerance(protocol: Mapping[str, Any], dtype: torch.dtype) -> dict[str, float]:
    if dtype != torch.bfloat16:
        raise ValueError("benchmark tolerance is defined for BF16 only")
    values = protocol["bf16"]
    return {"rtol": float(values["rtol"]), "atol": float(values["atol"])}


def _seeded_randn(shape: Sequence[int], *, dtype: torch.dtype, device: torch.device, seed: int, requires_grad: bool = False) -> torch.Tensor:
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        result = torch.randn(tuple(shape), generator=generator, device=device, dtype=dtype)
    except (RuntimeError, TypeError):
        previous = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        result = torch.randn(tuple(shape), device=device, dtype=dtype)
        torch.random.set_rng_state(previous)
    return result.requires_grad_(requires_grad)


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item(): raise FloatingPointError(f"{name} contains non-finite values")


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.detach().float() - expected.detach().float()).abs().max().item())


# Operator cases and independent correctness.

def _operator_case(raw: Any, index: int) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        def pick(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in raw: return raw[name]
            return default
        sources, rows = pick("S", "sources", "source_count"), pick("N", "rows", "tokens", "T", default=1)
        width, rank, shape_format = pick("D", "width", "hidden"), pick("R", "rank", default=None), "mapping"
    else:
        fields = list(raw)
        if len(fields) == 4:
            sources, rows, width, rank = fields
            shape_format = "S,N,D,R"
        elif len(fields) == 3:
            sources, rows, width = fields
            rank, shape_format = width, "S,N,D (R=D)"
        else: raise ValueError(f"operator case {index} needs 3 or 4 fields")
    rank = width if rank is None else rank
    result = {"id": f"operator_{index}", "S": int(sources), "N": int(rows), "D": int(width), "R": int(rank), "shape_format": shape_format}
    if min(result[k] for k in ("S", "N", "D", "R")) <= 0 or result["R"] > result["D"]:
        raise ValueError(f"invalid operator dimensions in case {index}: {result}")
    return result


def _operator_cases(protocol: Mapping[str, Any], config: Mapping[str, Any], scope: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = config.get("operator_cases", config.get("shapes"))
    raw = protocol.get(f"operator_{scope}", []) if raw is None else raw
    cases, failures = [], []
    for index, value in enumerate(raw):
        try: cases.append(_operator_case(value, index))
        except Exception as exc: failures.append({"id": f"operator_{index}", "status": "failed", "error": _exception(exc)})
    return cases, failures


def _project_ops() -> tuple[Callable[..., Any], Callable[..., Any]]:
    source = str(PROJECT_ROOT / "src")
    if source not in sys.path: sys.path.insert(0, source)
    module = importlib.import_module("attnres")
    from benchmarks.bf16_device import bf16_torch
    return module.attnres, bf16_torch


def _project_call(arm: str, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    kernel, reference = _project_ops()
    operator = kernel if arm == "kernel" else reference
    result = operator(values, query)
    if not isinstance(result, torch.Tensor): raise TypeError(f"{arm} returned {type(result).__name__}; expected a tensor")
    return result


def _operator_function(name: str, comparator: Any | None = None) -> Callable[..., Any]:
    if name in {"kernel", "reference"}: return lambda values, query: _project_call(name, values, query)
    from .competitors import invoke_comparator
    return lambda values, query: invoke_comparator(comparator, values, query)


def _qualify_operator(function: Callable[..., Any], values: torch.Tensor, query: torch.Tensor, protocol: Mapping[str, Any]) -> dict[str, Any]:
    from validation.oracle import oracle
    actual = [values.detach().clone().requires_grad_(True), query.detach().clone().requires_grad_(True)]
    expected = [values.detach().clone().requires_grad_(True), query.detach().clone().requires_grad_(True)]
    actual_out = function(*actual)
    expected_out = oracle(*expected)
    if not isinstance(actual_out, torch.Tensor): raise TypeError(f"operator returned {type(actual_out).__name__}; expected a tensor")
    _finite(actual_out, "operator output")
    _finite(expected_out, "oracle output")
    tolerance = _tolerance(protocol, values.dtype)
    torch.testing.assert_close(actual_out, expected_out, **tolerance)
    upstream = _seeded_randn(actual_out.shape, dtype=actual_out.dtype, device=actual_out.device, seed=91417)
    actual_grads = torch.autograd.grad(actual_out, actual, upstream, allow_unused=False)
    expected_grads = torch.autograd.grad(expected_out, expected, upstream, allow_unused=False)
    errors = []
    for index, (actual_grad, expected_grad) in enumerate(zip(actual_grads, expected_grads)):
        _finite(actual_grad, f"operator gradient {index}")
        _finite(expected_grad, f"oracle gradient {index}")
        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)
        errors.append(_max_abs(actual_grad, expected_grad))
    return {"status": "qualified", "output_max_abs": _max_abs(actual_out, expected_out), "gradient_max_abs": errors, "tolerance": tolerance}


def _make_operator_inputs(case: Mapping[str, int], dtype: torch.dtype, device: torch.device, seed: int, *, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype != torch.bfloat16:
        raise ValueError("operator inputs require BF16 storage")
    values = _seeded_randn(
        (case["S"], case["N"], case["D"]),
        dtype=dtype,
        device=device,
        seed=seed,
        requires_grad=requires_grad,
    )
    query = _seeded_randn(
        (case["R"],),
        dtype=dtype,
        device=device,
        seed=seed + 1,
        requires_grad=requires_grad,
    )
    return values, query


def _operator_correctness(protocol: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], device: torch.device, seed: int, comparators: Mapping[str, Any]) -> dict[str, Any]:
    rows, failures = [], []
    for case_index, case in enumerate(cases):
        dtype = torch.bfloat16
        case_seed = seed + case_index * 1000
        values, query = _make_operator_inputs(case, dtype, device, case_seed)
        applicable = case["R"] == case["D"]
        for name in ["kernel", "reference"] + (list(comparators) if applicable else []):
            row = {"case": dict(case), "dtype": str(dtype), "arm": name}
            try:
                comparator = comparators.get(name)
                if name not in {"kernel", "reference"} and not comparator.available:
                    row.update(status=comparator.status, reason=comparator.reason)
                else: row.update(_qualify_operator(_operator_function(name, comparator), values, query, protocol))
            except Exception as exc:
                row.update(status="failed", error=_exception(exc))
                failures.append(_failure("operator_correctness", **row))
            rows.append(row)
        if not applicable:
            rows.extend({"case": dict(case), "dtype": str(dtype), "arm": name,
                         "status": "not_applicable", "reason": "FLA comparator requires implicit standard R=D inputs"} for name in comparators)
    return {"status": "failed" if failures else ("complete" if rows else "incomplete"), "cases": rows, "failures": failures, "requested_cases": len(cases)}


# Operator timing and changed-input CUDA graph qualification.

def _cuda_event_call(function: Callable[[], Any], device: torch.device) -> tuple[float, Any]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(device):
        start.record()
        output = function()
        end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)), output


def _balanced_orders(arms: Sequence[str], rounds: int, rng: Any) -> list[list[str]]:
    first = list(arms)
    rng.shuffle(first)
    second = list(reversed(first))
    if len(first) < 3:
        return [list(first if i % 2 == 0 else second) for i in range(rounds)]
    count = len(first)
    schedule = []
    for i in range(rounds):
        offset = (i // 2) % count
        order = first[offset:] + first[:offset]
        if i % 2:
            order.reverse()
        schedule.append(order)
    return schedule


def _operator_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _operator_step(function: Callable[..., Any], values: torch.Tensor, query: torch.Tensor, mode: str, upstream: torch.Tensor) -> torch.Tensor:
    if mode == "forward": output = function(values, query)
    else:
        for tensor in (values, query):
            tensor.grad = None
        output = function(values, query)
        if isinstance(output, torch.Tensor): output.backward(upstream)
    if not isinstance(output, torch.Tensor): raise TypeError(f"operator returned {type(output).__name__}; expected a tensor")
    return output


def _operator_graph_step(function: Callable[..., Any], values: torch.Tensor, query: torch.Tensor, mode: str, upstream: torch.Tensor) -> torch.Tensor:
    """Run a graph-captured step with stable, preallocated gradient buffers."""
    if mode == "forward":
        output = function(values, query)
    else:
        for tensor in (values, query):
            if tensor.grad is None:
                tensor.grad = torch.zeros_like(tensor)
            else:
                tensor.grad.zero_()
        output = function(values, query)
        if isinstance(output, torch.Tensor):
            output.backward(upstream)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"operator returned {type(output).__name__}; expected a tensor")
    return output


def _capture_operator_graph(function: Callable[..., Any], values: torch.Tensor, query: torch.Tensor, upstream: torch.Tensor, mode: str, device: torch.device, warmup: int) -> dict[str, Any]:
    side, current = torch.cuda.Stream(device=device), torch.cuda.current_stream(device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        static_values = values.detach().clone().requires_grad_(True)
        static_query = query.detach().clone().requires_grad_(True)
        static_upstream = upstream.detach().clone()
        if mode == "forward_backward":
            for tensor in (static_values, static_query):
                tensor.grad = torch.zeros_like(tensor)
        for _ in range(max(2, warmup)):
            _operator_graph_step(function, static_values, static_query, mode, static_upstream)
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph, stream=side): output = _operator_graph_step(function, static_values, static_query, mode, static_upstream)
    side.synchronize()
    current.wait_stream(side)
    return {"graph": graph, "values": static_values, "query": static_query, "upstream": static_upstream,
            "output": output.detach(), "capture_host_ms": (time.perf_counter() - started) * 1000.0,
            "side_stream_warmup": max(2, warmup), "mode": mode}


def _copy_operator_graph_inputs(graph: Mapping[str, Any], sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    values, query, upstream = sample
    with torch.no_grad():
        graph["values"].copy_(values)
        graph["query"].copy_(query)
        graph["upstream"].copy_(upstream)


def _check_operator_graph_parity(function: Callable[..., Any], graph: Mapping[str, Any], samples: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], protocol: Mapping[str, Any], dtype: torch.dtype, device: torch.device) -> dict[str, Any]:
    del function
    if len(samples) < 2: raise ValueError("CUDA graph replay parity requires at least two inputs")
    hashes = [_operator_digest(*sample) for sample in samples]
    if len(set(hashes)) != len(hashes): raise RuntimeError("CUDA graph parity inputs did not change")
    from validation.oracle import oracle
    tolerance, output_errors, gradient_errors = _tolerance(protocol, dtype), [], []
    for sample in samples:
        _copy_operator_graph_inputs(graph, sample)
        graph["graph"].replay()
        torch.cuda.synchronize(device)
        replayed = graph["output"].detach().clone()
        oracle_values = sample[0].detach().clone().requires_grad_(True)
        oracle_query = sample[1].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            expected = oracle(oracle_values, oracle_query)
            if graph["mode"] == "forward_backward": expected.backward(sample[2])
        _finite(replayed, "CUDA graph output")
        _finite(expected, "CUDA graph oracle output")
        torch.testing.assert_close(replayed, expected, **tolerance)
        output_errors.append(_max_abs(replayed, expected))
        if graph["mode"] == "forward_backward":
            errors = []
            for graph_tensor, oracle_tensor in ((graph["values"], oracle_values), (graph["query"], oracle_query)):
                if graph_tensor.grad is None or oracle_tensor.grad is None: raise RuntimeError("CUDA graph replay did not produce every input gradient")
                _finite(graph_tensor.grad, "CUDA graph gradient")
                _finite(oracle_tensor.grad, "CUDA graph oracle gradient")
                torch.testing.assert_close(graph_tensor.grad, oracle_tensor.grad, **tolerance)
                errors.append(_max_abs(graph_tensor.grad, oracle_tensor.grad))
            gradient_errors.append(errors)
    return {"status": "qualified", "input_hashes": hashes, "output_max_abs": max(output_errors), "gradient_max_abs": gradient_errors, "tolerance": tolerance}


def _op_setup(dtype: Any, rounds: int, requested_warmup: int, method: str, replays: int | None, message: str) -> dict[str, Any]:
    return {"status": "failed", "dtype": str(dtype), "rounds": rounds, "warmup": max(1, requested_warmup), "requested_warmup": requested_warmup,
            "timing_method": method, "graph_replays": replays, "cases": [], "failures": [_failure("operator_setup", error={"message": message})]}


def _op_sample(case_id: str, case: Mapping[str, Any], dtype: torch.dtype, mode: str, name: str, sample: int, order: int | None, input_hash: str | None, method: str, replays: int) -> dict[str, Any]:
    return {"case": case_id, "shape": dict(case), "dtype": str(dtype), "mode": mode, "arm": name,
            "sample_index": sample, "order_index": order, "input_hash": input_hash,
            "timing_method": "cuda_event" if method == "eager" else "cuda_graph", "replay_count": 1 if method == "eager" else replays,
            "elapsed_ms": None, "normalized_ms": None, "ms": None}


def _paired_samples(names: Sequence[str], active: Sequence[str], rounds: int, rng: random.Random, failed: set[str], row_factory: Callable[[str, int, int | None], dict[str, Any]], measure: Callable[[str, int], Mapping[str, Any]], sink: Callable[[str, Mapping[str, Any]], list[dict[str, Any]]], phase: str) -> list[dict[str, Any]]:
    raw = []
    schedule = _balanced_orders(list(active), rounds, rng)
    for sample, scheduled in enumerate(schedule):
        order = [name for name in scheduled if name not in failed]
        present = set()
        for position, name in enumerate(order):
            present.add(name)
            row = row_factory(name, sample, position)
            try: row.update(measure(name, sample))
            except Exception as exc:
                failed.add(name)
                row.update(status="failed", error=_exception(exc))
                sink(name, row).append(_failure(phase, **row))
            raw.append(row)
        for name in names:
            if name not in present: raw.append(row_factory(name, sample, None) | {"status": "skipped_due_to_failure"})
    return raw


def _operator_timings(protocol: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], config: Mapping[str, Any], device: torch.device, seed: int, comparators: Mapping[str, Any]) -> dict[str, Any]:
    baseline = None
    if config.get("include_baseline", False):
        from .baseline import load_baseline
        baseline = load_baseline(config.get("baseline_root"))
    scope = str(config.get("scope", "smoke"))
    default_rounds = protocol["rounds"] if scope in {"primary", "heldout"} else protocol["smoke_rounds"]
    rounds = int(config.get("operator_rounds", config.get("rounds", default_rounds)))
    requested_warmup = int(config.get("operator_warmup", config.get("warmup", protocol["warmup"])))
    warmup = max(1, requested_warmup)
    modes = tuple(config.get("operator_modes", ("forward", "forward_backward")))
    method = str(config.get("operator_timing", "eager")).lower()
    try: replays = int(config.get("graph_replays", 10))
    except (TypeError, ValueError): replays = 0
    requested_dtype = config.get("operator_dtype", "bf16")
    try:
        dtype = _dtype(requested_dtype)
    except (TypeError, ValueError) as exc:
        return _op_setup(str(requested_dtype), rounds, requested_warmup, method,
                         replays if method == "cuda_graph" else None,
                         str(exc))
    if rounds < 1 or requested_warmup < 0: return _op_setup(dtype, rounds, requested_warmup, method, replays if method == "cuda_graph" else None, "rounds must be positive and warmup must be non-negative")
    if method not in {"eager", "cuda_graph"}: return _op_setup(dtype, rounds, requested_warmup, method, None, f"unsupported operator timing method: {method!r}")
    if method == "cuda_graph" and replays < 1: return _op_setup(dtype, rounds, requested_warmup, method, replays, "graph_replays must be positive for CUDA graph timing")
    invalid = sorted(set(modes) - {"forward", "forward_backward"})
    if not modes or invalid: return _op_setup(dtype, rounds, requested_warmup, method, replays if method == "cuda_graph" else None, f"unsupported operator modes: {invalid or 'none'}")
    cases_out, failures = [], []
    for case_index, case in enumerate(cases):
        case_id, case_seed = f"{case['id']}_implicit", seed + 30000 + case_index * 1000
        case_failures = []
        try:
            values, query = _make_operator_inputs(case, dtype, device, case_seed)
            arms = {"kernel": _operator_function("kernel"), "reference": _operator_function("reference")}
            if baseline is not None:
                arms["frozen_baseline"] = baseline
            if case["R"] == case["D"]:
                arms.update({name: _operator_function(name, comparator) for name, comparator in comparators.items() if comparator.available})
            qualification, inputs, active = {}, {}, []
            for name, function in arms.items():
                try:
                    qualification[name] = _qualify_operator(function, values, query, protocol)
                    input_values = values.detach().clone().requires_grad_(True)
                    input_query = query.detach().clone().requires_grad_(True)
                    upstream = _seeded_randn((case["N"], case["D"]), dtype=dtype, device=device, seed=case_seed + 701)
                    inputs[name], active = (input_values, input_query, upstream), active + [name]
                except Exception as exc:
                    qualification[name] = {"status": "failed", "error": _exception(exc)}
                    failure = _failure("operator_qualification", case=case_id, arm=name, **qualification[name])
                    failures.append(failure)
                    case_failures.append(failure)
            for name, comparator in comparators.items():
                if name not in arms:
                    qualification[name] = {"status": comparator.status, "reason": comparator.reason} if case["R"] == case["D"] else {"status": "not_applicable", "reason": "FLA comparator requires implicit standard R=D inputs"}
            failed_arms = {name for name in arms if name not in active}
            warmup_rows = []
            warmup_mode = "forward_backward" if "forward_backward" in modes else "forward"
            for name in active:
                for index in range(warmup):
                    try:
                        started = time.perf_counter()
                        input_values, input_query, upstream = inputs[name]
                        with torch.enable_grad(): output = _operator_step(arms[name], input_values, input_query, warmup_mode, upstream)
                        torch.cuda.synchronize(device)
                        warmup_rows.append({"arm": name, "index": index, "status": "ok", "host_ms": (time.perf_counter() - started) * 1000.0})
                        del output
                    except Exception as exc:
                        failed_arms.add(name)
                        row = {"arm": name, "index": index, "status": "failed", "error": _exception(exc)}
                        warmup_rows.append(row)
                        failure = _failure("operator_warmup", case=case_id, **row)
                        failures.append(failure)
                        case_failures.append(failure)
                        break
            graphs = {mode: {} for mode in modes}
            graph_reports = {mode: {} for mode in modes}
            graph_failed = {mode: set() for mode in modes}
            if method == "cuda_graph":
                parity = []
                for index in range(2):
                    parity_values, parity_query = _make_operator_inputs(case, dtype, device, case_seed + 100000 + index)
                    parity_upstream = _seeded_randn((case["N"], case["D"]), dtype=dtype, device=device, seed=case_seed + 101000 + index)
                    parity.append((parity_values, parity_query, parity_upstream))
                for mode in modes:
                    for name in active:
                        if name in failed_arms:
                            graph_reports[mode][name] = {"status": "skipped_due_to_failure", "reason": "operator warmup failed"}
                            graph_failed[mode].add(name)
                            continue
                        captured = False
                        try:
                            graph = _capture_operator_graph(arms[name], *inputs[name], mode, device, warmup)
                            graphs[mode][name] = graph
                            captured = True
                            parity_report = _check_operator_graph_parity(arms[name], graph, parity, protocol, dtype, device)
                            graph_reports[mode][name] = {"status": "qualified", "capture_host_ms": graph["capture_host_ms"], "side_stream_warmup": graph["side_stream_warmup"], "parity": parity_report}
                        except Exception as exc:
                            graph_failed[mode].add(name)
                            phase = "operator_graph_parity" if captured else "operator_graph_capture"
                            failure = _failure(phase, case=case_id, arm=name, mode=mode, status="failed", error=_exception(exc))
                            failures.append(failure)
                            case_failures.append(failure)
                            graph_reports[mode][name] = {"status": "failed", "phase": phase, "error": _exception(exc)}
            rng, all_names, raw = random.Random(case_seed), list(arms), []
            # Operator leaves are fixed throughout timing (only .grad
            # changes). Hash their bytes once, outside every event, rather
            # than copying the same large tensors to the CPU each round.
            input_hashes = {name: _operator_digest(*sample) for name, sample in inputs.items()}
            for mode in modes:
                mode_failed = set(failed_arms) | graph_failed[mode]
                def row_factory(name: str, sample: int, order: int | None, mode: str = mode) -> dict[str, Any]:
                    return _op_sample(case_id, case, dtype, mode, name, sample, order, input_hashes.get(name), method, replays)
                def measure(name: str, sample: int, mode: str = mode) -> Mapping[str, Any]:
                    input_values, input_query, upstream = inputs[name]
                    if method == "eager":
                        elapsed, output = _cuda_event_call(lambda: _operator_step(arms[name], input_values, input_query, mode, upstream), device)
                        del output
                        return {"status": "ok", "elapsed_ms": elapsed, "normalized_ms": elapsed, "ms": elapsed}
                    graph = graphs[mode][name]
                    _copy_operator_graph_inputs(graph, inputs[name])
                    elapsed, _ = _cuda_event_call(lambda: [graph["graph"].replay() for _ in range(replays)], device)
                    return {"status": "ok", "elapsed_ms": elapsed, "normalized_ms": elapsed / replays, "ms": elapsed / replays}
                raw.extend(_paired_samples(all_names, [name for name in active if name not in mode_failed], rounds, rng, mode_failed, row_factory, measure, lambda _name, _row: failures, "operator_graph_timing" if method == "cuda_graph" else "operator_timing"))
            statistics = {}
            for mode in modes:
                kernel_samples = [r["ms"] for r in raw if r["mode"] == mode and r["arm"] == "kernel" and r["status"] == "ok"]
                comparisons = {}
                for name in active:
                    candidate = [r["ms"] for r in raw if r["mode"] == mode and r["arm"] == name and r["status"] == "ok"]
                    if name != "kernel" and len(kernel_samples) == rounds == len(candidate): comparisons[name] = (kernel_samples, candidate)
                if comparisons: statistics[mode] = simultaneous_paired_ratio_bootstrap(comparisons, samples=int(config.get("bootstrap_samples", protocol["bootstrap_samples"])), seed=case_seed + sum((i + 1) * ord(c) for i, c in enumerate(mode)) % 10000, margin=float(protocol["plateau_margin"]))
            bad = any(v.get("status") == "failed" for v in qualification.values()) or any(f.get("case") == case_id for f in failures)
            case_status = "failed" if bad else ("incomplete" if any(v.get("status") == "missing" for v in qualification.values()) else "complete")
            cases_out.append(
                {
                    "case": case_id,
                    "shape": dict(case),
                    "dtype": str(dtype),
                    "status": case_status,
                    "requested_rounds": rounds,
                    "warmup_rounds": warmup,
                    "arms": all_names,
                    "qualification": qualification,
                    "warmup": warmup_rows,
                    "timing_method": method,
                    "graph_replays": replays if method == "cuda_graph" else None,
                    "graph": graph_reports if method == "cuda_graph" else {},
                    "raw_samples": raw,
                    "statistics": statistics,
                }
            )
        except Exception as exc:
            row = {"case": case_id, "status": "failed", "error": _exception(exc)}
            cases_out.append(row)
            failures.append(_failure("operator_setup", **row))
    status = "failed" if failures else (
        "complete"
        if cases_out and all(row.get("status") == "complete" for row in cases_out)
        else "incomplete"
    )
    return {
        "status": status,
        "dtype": str(dtype),
        "rounds": rounds,
        "warmup": warmup,
        "requested_warmup": requested_warmup,
        "timing_method": method,
        "frozen_baseline": baseline.metadata if baseline is not None else None,
        "graph_replays": replays if method == "cuda_graph" else None,
        "timing_boundary": {
            "eager": "CUDA event interval around one eager operator call; device idle time inside remains measured",
            "cuda_graph": "CUDA event interval around fixed graph replays; copies and capture/parity are outside",
        },
        "cases": cases_out,
        "failures": failures,
    }


# Compiled model timing.

def _model_config(protocol: Mapping[str, Any], config: Mapping[str, Any], scope: str) -> dict[str, Any]:
    selected = config.get("model_config", config.get("model"))
    selected = protocol.get(f"{scope}_model", protocol.get("smoke_model", {})) if not isinstance(selected, Mapping) else selected
    result = dict(selected)
    for name in ("layers", "width", "heads", "ffn", "batch", "sequence", "vocab", "block_count", "variant", "mode", "rank", "source_layout"):
        if name in config: result[name] = config[name]
    result.setdefault("variant", "standard")
    result.setdefault("mode", "full")
    result.setdefault("rank", result.get("width"))
    return result


def _effective_variant(config: Mapping[str, Any], model_data: Mapping[str, Any], ranks: Sequence[int]) -> str:
    selected = config.get("model_config", config.get("model"))
    explicit = "variant" in config or isinstance(selected, Mapping) and "variant" in selected
    variant = str(model_data["variant"]).lower()
    return "sliced" if variant == "standard" and not explicit and any(r != int(model_data["width"]) for r in ranks) else variant


def _model_inputs(model_data: Mapping[str, Any], device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    shape = (int(model_data["batch"]), int(model_data["sequence"]))
    tokens = torch.randint(int(model_data["vocab"]), shape, generator=generator, device=device)
    generator.manual_seed(int(seed) + 1)
    targets = torch.randint(int(model_data["vocab"]), shape, generator=generator, device=device)
    return tokens, targets


def _model_read_source_counts(model_data: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the exact source count seen by each model residual read.

    External model comparators are constrained by the geometry of each
    individual public ``attnres`` call.  In particular, Liger's ``S<=32``
    limit applies to every Full read and every Block read, rather than to a
    synthetic total assembled from the model configuration.  Keeping this
    scheduler calculation next to the model runner makes that gate agree with
    the actual Full/Block source construction in :mod:`benchmarks.model`.
    """

    layers = model_data.get("layers")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("model layers must be a positive integer")
    mode = str(model_data.get("mode", "full")).lower()
    events = 2 * layers
    if mode == "full":
        # The first attention consumes the embedding directly.  Every later
        # residual event reads all previous sources, followed by the final
        # read before the LM head.
        return tuple(range(2, events + 2))
    if mode != "block":
        raise ValueError(f"unsupported model mode {mode!r}")

    # Keep the source scheduler definition shared with the model rather than
    # reproducing its evenly partitioned block arithmetic here.
    from .model import _block_ends

    block_count = model_data.get("block_count")
    if (
        isinstance(block_count, bool)
        or not isinstance(block_count, int)
        or block_count < 1
    ):
        raise ValueError("model block_count must be a positive integer")
    ends = _block_ends(events, block_count)
    completed_count = 1
    partial_exists = False
    previous_end = 0
    read_counts: list[int] = []
    for end in ends:
        for event_index in range(previous_end, end):
            if event_index != 0:
                read_counts.append(completed_count + int(partial_exists))
            partial_exists = True
        completed_count += 1
        partial_exists = False
        previous_end = end
    # Block's terminal read sees one completed source per completed block,
    # plus the embedding source.
    read_counts.append(completed_count)
    return tuple(read_counts)


def _liger_model_eligibility(
    model_data: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    """Apply the registry gate to every Liger model read before allocation.

    The compiled model runner is a BF16 autocast path.  Liger itself also has
    an FP32 storage capability, but accepting a user supplied FP32 model
    label here would misdescribe this runner's actual timed computation, so
    this model arm is explicitly tied to the BF16 campaign.
    """

    from .comparator_registry import eligibility_for

    width = model_data.get("width")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        return {
            "competitor": "liger",
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "reason": "Liger model eligibility requires a positive model width D",
        }
    if type(rank) is not int or rank < 1:
        return {
            "competitor": "liger",
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "reason": "Liger model eligibility requires a positive rank R",
        }
    if rank != width:
        return {
            "competitor": "liger",
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "rank": rank,
            "width": width,
            "dtype": "bf16",
            "reason": f"Liger is restricted to standard R=D (got R={rank}, D={width})",
        }
    try:
        read_counts = _model_read_source_counts(model_data)
    except (TypeError, ValueError) as exc:
        return {
            "competitor": "liger",
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "rank": rank,
            "width": width,
            "dtype": "bf16",
            "reason": str(exc),
        }
    mode = str(model_data.get("mode", "full")).lower()
    registry_mode = "full" if mode == "full" else "block_per_read"
    per_read: list[dict[str, Any]] = []
    for read_index, source_count in enumerate(read_counts):
        fields: dict[str, Any] = {
            "mode": registry_mode,
            "rank": rank,
            "width": width,
            "dtype": "bf16",
            "timing": True,
        }
        if registry_mode == "full":
            fields["source_count"] = source_count
        else:
            fields["read_source_count"] = source_count
        decision = eligibility_for("liger", **fields)
        per_read.append({"read_index": read_index, **decision})
        if not decision.get("eligible", False):
            return {
                "competitor": "liger",
                "status": "not_applicable",
                "eligible": False,
                "eligible_denominator": False,
                "rank": rank,
                "width": width,
                "dtype": "bf16",
                "mode": registry_mode,
                "read_source_counts": list(read_counts),
                "failed_read_index": read_index,
                "per_read": per_read,
                "reason": decision.get("reason", "Liger read is outside its declared capability"),
                "capability": decision.get("capability"),
            }
    return {
        "competitor": "liger",
        "status": "eligible",
        "eligible": True,
        "eligible_denominator": True,
        "rank": rank,
        "width": width,
        "dtype": "bf16",
        "mode": registry_mode,
        "read_source_counts": list(read_counts),
        "max_read_source_count": max(read_counts, default=0),
        "per_read": per_read,
        "reason": "every compiled BF16 model read is inside Liger's declared capability",
        "capability": per_read[0].get("capability") if per_read else None,
    }


def _catswe_model_eligibility(
    model_data: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    """Apply the separate registry capability to every actual model read.

    The operator registry row remains ``standard_operator_only``.  This arm
    is selected through the explicit model capability scope and invokes the
    vendor public phase1 operation once per Full or Block read.  All read
    decisions happen before constructing the comparator model.
    """

    from .comparator_registry import capability_for, eligibility_for

    width = model_data.get("width")
    model_capability = capability_for("catswe_phase1", scope="model")
    operator_capability = capability_for("catswe_phase1")
    base: dict[str, Any] = {
        "competitor": "catswe_phase1",
        "adapter": model_capability["adapter"],
        "model_scope": model_capability["model_scope"],
        "operator_capability_scope": operator_capability["model_scope"],
        "rank": int(rank) if type(rank) is int else rank,
        "width": int(width) if type(width) is int else width,
        "dtype": "bf16",
        "timing": True,
        "capability_scope": "model",
    }
    if type(width) is not int or width < 1:
        return {
            **base,
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "reason": "Catswe model eligibility requires a positive model width D",
        }
    if type(rank) is not int or rank < 1:
        return {
            **base,
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "reason": "Catswe model eligibility requires a positive rank R",
        }
    mode = str(model_data.get("mode", "full")).lower()
    if mode not in {"full", "block"}:
        return {
            **base,
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "mode": mode,
            "reason": f"Catswe compiled model arm does not support mode {mode!r}",
        }
    try:
        read_counts = _model_read_source_counts(model_data)
    except (TypeError, ValueError) as exc:
        return {
            **base,
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "mode": mode,
            "reason": str(exc),
        }

    registry_mode = "full" if mode == "full" else "block_per_read"
    per_read: list[dict[str, Any]] = []
    for read_index, source_count in enumerate(read_counts):
        fields: dict[str, Any] = {
            "mode": registry_mode,
            "rank": rank,
            "width": width,
            "dtype": "bf16",
            "timing": True,
        }
        if registry_mode == "full":
            fields["source_count"] = source_count
        else:
            fields["read_source_count"] = source_count
        decision = eligibility_for("catswe_phase1", scope="model", **fields)
        row = {"read_index": read_index, **decision}
        per_read.append(row)
        if not decision.get("eligible", False):
            return {
                **base,
                "status": "not_applicable",
                "eligible": False,
                "eligible_denominator": False,
                "mode": registry_mode,
                "read_source_counts": list(read_counts),
                "max_read_source_count": max(read_counts, default=0),
                "failed_read_index": read_index,
                "per_read": per_read,
                "reason": decision.get(
                    "reason", "Catswe read is outside its declared model capability"
                ),
                "capability": decision.get("capability", model_capability),
            }
    return {
        **base,
        "status": "eligible",
        "eligible": True,
        "eligible_denominator": True,
        "mode": registry_mode,
        "read_source_counts": list(read_counts),
        "max_read_source_count": max(read_counts, default=0),
        "per_read": per_read,
        "reason": (
            "every compiled BF16 model read is inside Catswe's separate "
            "public phase1 model capability; no cache/prepare/merge/phase2 route"
        ),
        "capability": per_read[0].get("capability", model_capability)
        if per_read
        else model_capability,
    }


def _catswe_model_discovery_eligible(
    model_data: Mapping[str, Any],
    raw_ranks: Any,
) -> bool:
    """Return whether Catswe discovery is needed for any requested model rank.

    This is a host only gate.  It uses the same public model capability
    predicate as ``_model_timings`` and intentionally returns false for
    malformed ranks, so an opt-in cannot trigger native vendor discovery
    before the actual model eligibility check.
    """

    try:
        if isinstance(raw_ranks, bool):
            return False
        ranks = (
            [int(raw_ranks)]
            if isinstance(raw_ranks, int)
            else [int(rank) for rank in raw_ranks]
        )
    except (TypeError, ValueError):
        return False
    if not ranks:
        return False
    try:
        return any(
            _catswe_model_eligibility(model_data, rank).get("eligible", False)
            is True
            for rank in ranks
        )
    except Exception:
        # Discovery is optional and must remain fail closed when the
        # capability preflight cannot establish eligibility.
        return False


def _model_only_rank_admission(
    config: Mapping[str, Any],
    model_data: Mapping[str, Any],
    protocol_ranks: Sequence[int],
) -> tuple[tuple[int, ...], dict[str, Any] | None, str | None]:
    """Validate the opt-in rank extension for standalone model sweeps.

    The frozen protocol ladder remains the default.  A model-only campaign
    may explicitly add only the standard pairs ``D=R=2048`` and
    ``D=R=4096``.  This narrow, sealed admission lets width sweeps exercise a
    standard external comparator without turning those ranks into operator or
    LR protocol cells.  The returned digest is recorded with the model
    result, so a report cannot silently grow a rank outside its declared
    admission object.
    """

    raw = config.get("model_only_admission")
    frozen = tuple(int(rank) for rank in protocol_ranks)
    if raw is None:
        return frozen, None, None
    if not isinstance(raw, Mapping):
        return frozen, None, "model_only_admission must be a mapping"
    if raw.get("enabled") is not True:
        return frozen, None, "model_only_admission.enabled must be true"
    if raw.get("sealed") is not True:
        return frozen, None, "model_only_admission.sealed must be true"
    if raw.get("scope") != "model_only":
        return frozen, None, "model_only_admission.scope must be 'model_only'"
    unknown = set(raw) - {
        "enabled",
        "sealed",
        "scope",
        "width_rank_pairs",
        "purpose",
    }
    if unknown:
        return frozen, None, (
            "model_only_admission contains unknown fields: "
            f"{sorted(str(value) for value in unknown)}"
        )
    if "purpose" in raw and (
        not isinstance(raw["purpose"], str) or not raw["purpose"].strip()
    ):
        return frozen, None, "model_only_admission.purpose must be a nonempty string"
    pairs = raw.get("width_rank_pairs")
    if not isinstance(pairs, list) or not pairs:
        return frozen, None, "model_only_admission.width_rank_pairs must be a non-empty list"
    normalized: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(value) is not int or value < 1 for value in pair)
        ):
            return frozen, None, "model_only_admission pairs must be [width, rank] integer lists"
        width, rank = int(pair[0]), int(pair[1])
        if (width, rank) in seen:
            return frozen, None, "model_only_admission width/rank pairs must be unique"
        if (width, rank) not in {(2048, 2048), (4096, 4096)}:
            return frozen, None, (
                "model_only_admission permits only [2048, 2048] and [4096, 4096]"
            )
        seen.add((width, rank))
        normalized.append([width, rank])
    width = model_data.get("width")
    if type(width) is not int or width < 1:
        return frozen, None, "model width must be a positive integer"
    admitted_pairs = {(pair[0], pair[1]) for pair in normalized}
    extra_ranks = {
        rank for admitted_width, rank in admitted_pairs
        if admitted_width == width and rank not in frozen
    }
    requested = config.get("ranks", protocol_ranks)
    if type(requested) is int:
        requested_ranks = (requested,)
    elif isinstance(requested, (list, tuple)) and all(
        type(value) is int for value in requested
    ):
        requested_ranks = tuple(requested)
    else:
        return frozen, None, "requested model ranks must be integer or an integer list"
    invalid_extra = {
        rank for rank in requested_ranks
        if rank not in frozen and (width, rank) not in admitted_pairs
    }
    if invalid_extra:
        return frozen, None, (
            "requested rank is outside the sealed model-only width/rank admission: "
            f"{sorted(invalid_extra)}"
        )
    admission = {
        "status": "sealed",
        "scope": "model_only",
        "enabled": True,
        "sealed": True,
        "protocol_ranks": list(frozen),
        "width_rank_pairs": normalized,
        "requested_width": width,
        "requested_ranks": list(requested_ranks),
        "admitted_extra_ranks": sorted(extra_ranks),
        "model_geometry": {
            name: model_data.get(name)
            for name in (
                "layers",
                "width",
                "heads",
                "ffn",
                "batch",
                "sequence",
                "vocab",
                "block_count",
                "variant",
                "mode",
                "source_layout",
            )
        },
    }
    if "purpose" in raw:
        admission["purpose"] = raw["purpose"]
    admission["digest"] = hashlib.sha256(
        json.dumps(admission, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    allowed = tuple(dict.fromkeys((*frozen, *sorted(extra_ranks))))
    return allowed, admission, None


def _copy_model_state(source: Any, target: Any) -> None: target.load_state_dict({k: v.detach().clone() for k, v in source.state_dict().items()}, strict=True)


def _state_digest(state: Mapping[str, torch.Tensor], names: Sequence[str] | None = None) -> str:
    """Hash named tensor bytes and shapes without serializing them in reports."""

    digest = hashlib.sha256()
    selected = sorted(state if names is None else names)
    for name in selected:
        tensor = state[name].detach()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_shape_metadata(state: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: [int(size) for size in state[name].shape] for name in sorted(state)}


def _common_state_names(state: Mapping[str, torch.Tensor]) -> list[str]:
    return [
        name
        for name in sorted(state)
        if not name.startswith("queries.")
    ]


def _model_state_record(
    model: Any,
    *,
    arm: str,
    rank: int,
    variant: str,
    mode: str,
    protocol_name: str,
) -> dict[str, Any]:
    state = model.state_dict()
    names = _common_state_names(state)
    return {
        "arm": arm,
        "rank": int(rank),
        "variant": variant,
        "mode": mode,
        "initial_state_hash": _state_digest(state),
        "shape_metadata": _state_shape_metadata(state),
        "common_fixed_state_hash": _state_digest(state, names),
        "protocol": protocol_name,
    }


def _release_qualification_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _model_qualification(reference: Any, candidate: Any, tokens: torch.Tensor, targets: torch.Tensor, protocol: Mapping[str, Any], loss_function: Callable[..., Any]) -> dict[str, Any]:
    def forward_backward(model):
        parameters = [p for p in model.parameters() if p.requires_grad]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens)
            if not isinstance(logits, torch.Tensor):
                raise TypeError("model arms must return logits tensors")
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        _finite(logits, "model output")
        _finite(loss, "model loss")
        gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
        return logits.detach(), loss.detach(), gradients

    reference.train(False)
    candidate.train(False)
    reference_device = next(reference.parameters()).device
    candidate_device = next(candidate.parameters()).device
    try:
        # Only one full model may reside on an 80 GiB H100 while its exact
        # B=2,T=2048 training graph is live.  Stage the peer on CPU; this is an
        # untimed oracle and both models are restored before compilation.
        candidate.to("cpu")
        _release_qualification_memory(tokens.device)
        reference_logits, reference_loss, reference_grads = forward_backward(reference)
        # Full production geometry nearly fills an 80 GiB H100.  Retaining a
        # second model's full-width logits and parameter gradients on device
        # while building the candidate graph can turn an untimed oracle into
        # an artificial OOM.  Preserve the exact evidence on CPU instead.
        reference_logits = reference_logits.cpu()
        reference_loss = reference_loss.cpu()
        reference_grads = tuple(gradient.cpu() for gradient in reference_grads)
        reference.to("cpu")
        candidate.to(candidate_device)
        if tokens.device.type == "cuda":
            torch.cuda.empty_cache()
        candidate_logits, candidate_loss, candidate_grads = forward_backward(candidate)
        tolerance = _tolerance(protocol, torch.bfloat16)
        candidate_logits_cpu = candidate_logits.cpu()
        candidate_loss_cpu = candidate_loss.cpu()
        torch.testing.assert_close(candidate_logits_cpu, reference_logits, **tolerance)
        torch.testing.assert_close(candidate_loss_cpu, reference_loss, **tolerance)
        if len(reference_grads) != len(candidate_grads): raise RuntimeError("model parameter counts differ")
        errors = []
        for index, (candidate_grad, reference_grad) in enumerate(zip(candidate_grads, reference_grads)):
            _finite(candidate_grad, f"candidate gradient {index}")
            _finite(reference_grad, f"reference gradient {index}")
            candidate_grad_cpu = candidate_grad.cpu()
            torch.testing.assert_close(candidate_grad_cpu, reference_grad, **tolerance)
            errors.append(_max_abs(candidate_grad_cpu, reference_grad))
        return {"status": "qualified", "output_max_abs": _max_abs(candidate_logits_cpu, reference_logits), "loss_max_abs": _max_abs(candidate_loss_cpu, reference_loss), "gradient_max_abs": errors, "tolerance": tolerance, "parameter_count": len(candidate_grads), "reference_evidence_device": "cpu"}
    finally:
        if next(reference.parameters()).device != reference_device:
            reference.to(reference_device)
        if next(candidate.parameters()).device != candidate_device:
            candidate.to(candidate_device)
        reference.train(True)
        candidate.train(True)


def _clone_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _clone_state_value_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_state_value_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_value_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value_cpu(item) for item in value)
    return copy.deepcopy(value)


def _clone_state_value_to_device(value: Any, device: torch.device | str) -> Any:
    """Clone recursive state while placing tensor leaves on ``device``."""

    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_state_value_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_state_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value_to_device(item, device) for item in value)
    return copy.deepcopy(value)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device | str) -> None:
    """Move all optimizer tensor leaves in place while retaining its object."""

    target = torch.device(device)

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=target)
        if isinstance(value, Mapping):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for parameter, values in list(optimizer.state.items()):
        optimizer.state[parameter] = move(values)
    for group in optimizer.param_groups:
        for key, value in list(group.items()):
            if key != "params" and isinstance(value, torch.Tensor):
                group[key] = value.detach().to(device=target)


def _move_model_optimizer(
    model: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> None:
    """Move one candidate model and optimizer as a single offload unit."""

    model.to(device)
    _move_optimizer_state(optimizer, device)


def _validate_finite_state(value: Any, label: str) -> None:
    """Reject nonfinite tensor or floating scalar leaves recursively."""

    if isinstance(value, torch.Tensor):
        _finite(value, label)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_state(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_state(item, f"{label}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"{label} contains non-finite values")


def _require_cpu_evidence(value: Any, label: str) -> None:
    """Ensure evidence has no live device tensors before comparison."""

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise RuntimeError(f"{label} contains non-CPU evidence")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_cpu_evidence(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_cpu_evidence(item, f"{label}[{index}]")



def _clone_model_checkpoint(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _restore_model_checkpoint(model: Any, snapshot: Mapping[str, torch.Tensor]) -> None:
    from .training_graph import _restore_module_state

    _restore_module_state(model, snapshot)


def _clone_optimizer_checkpoint(optimizer: torch.optim.Optimizer) -> dict[Any, dict[str, Any]]:
    return {
        parameter: {
            key: _clone_state_value_cpu(value)
            for key, value in values.items()
        }
        for parameter, values in optimizer.state.items()
    }


def _restore_optimizer_checkpoint(
    optimizer: torch.optim.Optimizer,
    snapshot: Mapping[Any, Mapping[str, Any]],
    *,
    preserve_allocations: bool = False,
) -> None:
    if preserve_allocations:
        # A captured optimizer owns pointers to its state tensors.  Keep those
        # allocations in place after a graph qualification replay.
        from .training_graph import _restore_optimizer_state

        _restore_optimizer_state(optimizer, snapshot)
        return

    # The ordinary (pre-capture) gate must restore the optimizer exactly.  In
    # particular, remove state entries allocated by a failed/no-op candidate
    # step; the graph helper intentionally retains those entries for pointer
    # stability and is therefore not sufficient here.
    for parameter in list(optimizer.state):
        if parameter not in snapshot:
            del optimizer.state[parameter]
    for parameter, original_values in snapshot.items():
        current_values = optimizer.state.get(parameter)
        if current_values is None:
            optimizer.state[parameter] = {
                key: _clone_state_value_to_device(value, parameter.device)
                for key, value in original_values.items()
            }
            continue
        for key in list(current_values):
            if key not in original_values:
                del current_values[key]
        for key, original in original_values.items():
            current = current_values.get(key)
            if isinstance(original, torch.Tensor):
                if (
                    isinstance(current, torch.Tensor)
                    and current.shape == original.shape
                    and current.dtype == original.dtype
                ):
                    current.copy_(original)
                else:
                    current_values[key] = original.detach().to(parameter.device).clone()
            else:
                current_values[key] = _clone_state_value_to_device(original, parameter.device)


def _restore_named_optimizer_state(
    model: Any,
    optimizer: torch.optim.Optimizer,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    """Restore CPU optimizer evidence onto an independent model by name."""

    parameters = dict(model.named_parameters())
    if set(snapshot) - set(parameters):
        raise RuntimeError("optimizer evidence contains an unknown model parameter")
    names_by_id = {id(parameter): name for name, parameter in parameters.items()}
    for parameter in list(optimizer.state):
        if names_by_id.get(id(parameter)) not in snapshot:
            del optimizer.state[parameter]
    for name, values in snapshot.items():
        parameter = parameters[name]
        optimizer.state[parameter] = {
            key: _clone_state_value_to_device(value, parameter.device)
            for key, value in values.items()
        }


def _clone_named_gradients(model: Any) -> dict[str, torch.Tensor | None]:
    return {
        name: parameter.grad.detach().cpu().clone()
        if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }


def _restore_named_gradients(
    model: Any,
    snapshot: Mapping[str, torch.Tensor | None],
) -> None:
    parameters = dict(model.named_parameters())
    if set(parameters) != set(snapshot):
        raise RuntimeError("model parameter names changed during qualification")
    for name, original in snapshot.items():
        parameter = parameters[name]
        if original is None:
            parameter.grad = None
        elif (
            isinstance(parameter.grad, torch.Tensor)
            and parameter.grad.shape == original.shape
            and parameter.grad.dtype == original.dtype
        ):
            parameter.grad.copy_(original)
        else:
            parameter.grad = original.detach().to(parameter.device).clone()


def _named_optimizer_state(
    model: Any,
    optimizer: torch.optim.Optimizer,
) -> dict[str, dict[str, Any]]:
    """Return optimizer state by parameter name and validate full coverage."""

    parameters = dict(model.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in parameters.items()}
    grouped = [parameter for group in optimizer.param_groups for parameter in group.get("params", ())]
    grouped_ids = [id(parameter) for parameter in grouped]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("optimizer contains duplicate parameters")
    if set(grouped_ids) != set(names_by_id):
        missing = sorted(set(names_by_id) - set(grouped_ids))
        foreign = sorted(set(grouped_ids) - set(names_by_id))
        raise RuntimeError(
            f"optimizer parameter coverage differs from model (missing={missing}, foreign={foreign})"
        )
    result: dict[str, dict[str, Any]] = {}
    for parameter, values in optimizer.state.items():
        name = names_by_id.get(id(parameter))
        if name is None:
            raise RuntimeError("optimizer state contains a parameter outside the model")
        if not isinstance(values, Mapping):
            raise RuntimeError(f"optimizer state for {name} is not a mapping")
        result[name] = {key: _clone_state_value_cpu(value) for key, value in values.items()}
    return result


def _named_optimizer_groups(
    model: Any,
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    """Snapshot optimizer groups with parameter identities replaced by names."""

    parameters = dict(model.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in parameters.items()}
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        names = []
        for parameter in group.get("params", ()):
            name = names_by_id.get(id(parameter))
            if name is None:
                raise RuntimeError("optimizer group contains a parameter outside the model")
            names.append(name)
        groups.append({
            key: _clone_state_value(value)
            for key, value in group.items()
            if key != "params"
        } | {"params": names})
    return groups


def _restore_optimizer_groups(
    model: Any,
    optimizer: torch.optim.Optimizer,
    snapshot: Sequence[Mapping[str, Any]],
) -> None:
    if len(optimizer.param_groups) != len(snapshot):
        raise RuntimeError("optimizer parameter-group topology changed during qualification")
    current_names = _named_optimizer_groups(model, optimizer)
    for index, original in enumerate(snapshot):
        current = optimizer.param_groups[index]
        wanted_keys = set(original)
        for key in list(current):
            if key != "params" and key not in wanted_keys:
                del current[key]
        for key, value in original.items():
            if key == "params":
                continue
            group_device = next(
                (parameter.device for parameter in current.get("params", ())
                 if isinstance(parameter, torch.Tensor)),
                torch.device("cpu"),
            )
            current[key] = _clone_state_value_to_device(value, group_device)
        # Parameter membership and ordering are validated separately; the
        # optimizer keeps its live Parameter objects rather than name strings.
        if current_names[index].get("params") != original.get("params"):
            raise RuntimeError("optimizer parameter-group membership changed during qualification")


def _value_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        return (
            isinstance(actual, torch.Tensor)
            and isinstance(expected, torch.Tensor)
            and torch.equal(actual.detach().cpu(), expected.detach().cpu())
        )
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping) or set(actual) != set(expected):
            return False
        return all(_value_equal(actual[name], expected[name]) for name in actual)
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, type(expected))
            and len(actual) == len(expected)
            and all(_value_equal(a, e) for a, e in zip(actual, expected))
        )
    return actual == expected


def _compare_state_values(
    actual: Any,
    expected: Any,
    tolerance: Mapping[str, float],
    *,
    label: str,
    exact: bool = False,
) -> Any:
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            raise RuntimeError(f"{label} tensor type differs")
        _finite(actual, f"{label} candidate")
        _finite(expected, f"{label} reference")
        compare = {"rtol": 0.0, "atol": 0.0} if exact else dict(tolerance)
        actual_compare = actual.detach().cpu()
        expected_compare = expected.detach().cpu()
        torch.testing.assert_close(actual_compare, expected_compare, **compare)
        return _max_abs(actual_compare, expected_compare)
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise RuntimeError(f"{label} container type differs")
        if set(actual) != set(expected):
            raise RuntimeError(
                f"{label} names differ (candidate={sorted(actual)}, reference={sorted(expected)})"
            )
        return {
            str(name): _compare_state_values(
                actual[name], expected[name], tolerance,
                label=f"{label}.{name}", exact=exact,
            )
            for name in sorted(expected)
        }
    if not _value_equal(actual, expected):
        raise RuntimeError(f"{label} differs")
    return 0.0


def _compare_model_checkpoint(
    candidate: Any,
    reference: Any,
    tolerance: Mapping[str, float],
) -> dict[str, float]:
    return _compare_state_values(
        candidate.state_dict(), reference.state_dict(), tolerance, label="model state"
    )


def _compare_named_gradients(
    candidate: Any,
    reference: Any,
    tolerance: Mapping[str, float],
) -> dict[str, float]:
    candidate_parameters = dict(candidate.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    if set(candidate_parameters) != set(reference_parameters):
        raise RuntimeError("candidate and reference parameter names differ")
    result: dict[str, float] = {}
    for name in sorted(reference_parameters):
        candidate_gradient = candidate_parameters[name].grad
        reference_gradient = reference_parameters[name].grad
        if candidate_gradient is None or reference_gradient is None:
            raise RuntimeError(f"missing gradient for parameter {name}")
        result[name] = _compare_state_values(
            candidate_gradient, reference_gradient, tolerance,
            label=f"gradient {name}",
        )
    return result


def _compare_named_optimizer_states(
    candidate_model: Any,
    candidate_optimizer: torch.optim.Optimizer,
    reference_model: Any,
    reference_optimizer: torch.optim.Optimizer,
    tolerance: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    candidate = _named_optimizer_state(candidate_model, candidate_optimizer)
    reference = _named_optimizer_state(reference_model, reference_optimizer)
    if set(candidate) != set(reference):
        raise RuntimeError(
            f"optimizer state parameter names differ (candidate={sorted(candidate)}, reference={sorted(reference)})"
        )
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(reference):
        candidate_values, reference_values = candidate[name], reference[name]
        if set(candidate_values) != set(reference_values):
            raise RuntimeError(f"optimizer state keys differ for {name}")
        result[name] = {}
        for key in sorted(reference_values):
            # AdamW's step counter is a tensor on CUDA, but it is a discrete
            # counter and must match exactly rather than pass a BF16 tolerance.
            result[name][key] = _compare_state_values(
                candidate_values[key], reference_values[key], tolerance,
                label=f"optimizer state {name}.{key}", exact=key == "step",
            )
    return result


def _changed_names(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    changed: list[str] = []
    for name in sorted(set(before) | set(after)):
        if not _value_equal(before.get(name, {}), after.get(name, {})):
            changed.append(name)
    return changed


def _changed_parameters(
    before: Mapping[str, torch.Tensor],
    model: Any,
) -> list[str]:
    parameters = dict(model.named_parameters())
    return [
        name for name in sorted(parameters)
        if name not in before or not _value_equal(before[name], parameters[name].detach())
    ]


def _restore_and_verify_candidate(
    model: Any,
    optimizer: torch.optim.Optimizer,
    model_state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[Any, Mapping[str, Any]],
    gradient_state: Mapping[str, torch.Tensor | None],
    named_optimizer_state: Mapping[str, Mapping[str, Any]],
    optimizer_groups: Sequence[Mapping[str, Any]],
    *,
    preserve_optimizer_allocations: bool = False,
) -> None:
    _restore_model_checkpoint(model, model_state)
    _restore_optimizer_checkpoint(
        optimizer,
        optimizer_state,
        preserve_allocations=preserve_optimizer_allocations,
    )
    _restore_optimizer_groups(model, optimizer, optimizer_groups)
    _restore_named_gradients(model, gradient_state)
    _compare_state_values(
        model.state_dict(), model_state, {"rtol": 0.0, "atol": 0.0},
        label="restored model state", exact=True,
    )
    restored_optimizer_state = _named_optimizer_state(model, optimizer)
    if not _value_equal(restored_optimizer_state, named_optimizer_state):
        raise RuntimeError("candidate optimizer state was not restored exactly")
    if not _value_equal(_named_optimizer_groups(model, optimizer), optimizer_groups):
        raise RuntimeError("candidate optimizer groups were not restored exactly")
    restored_gradients = _clone_named_gradients(model)
    if not _value_equal(restored_gradients, gradient_state):
        raise RuntimeError("candidate gradients were not restored exactly")


def _oracle_complete_training_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    accumulation: int,
) -> torch.Tensor:
    """Run one readable independent reference update with BF16 autocast."""

    from .model import _microbatches

    batches = _microbatches(tokens, targets, accumulation)
    optimizer.zero_grad(set_to_none=True)
    result: torch.Tensor | None = None
    divisor = float(len(batches))
    with torch.autocast(device_type=tokens.device.type, dtype=torch.bfloat16):
        for micro_tokens, micro_targets in batches:
            logits = model(micro_tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), micro_targets.reshape(-1)
            )
            result = loss
            (loss / divisor).backward()
    optimizer.step()
    if result is None:
        raise RuntimeError("reference complete step produced no microbatches")
    return result.detach()


def _compare_complete_step(
    *,
    candidate_model: Any,
    candidate_optimizer: torch.optim.Optimizer,
    candidate_loss: torch.Tensor,
    reference_model: Any,
    reference_optimizer: torch.optim.Optimizer,
    reference_loss: torch.Tensor,
    tolerance: Mapping[str, float],
    label: str,
) -> dict[str, Any]:
    if not isinstance(candidate_loss, torch.Tensor) or not isinstance(reference_loss, torch.Tensor):
        raise TypeError(f"{label} must return a loss tensor")
    _finite(candidate_loss, f"candidate {label} loss")
    _finite(reference_loss, f"reference {label} loss")
    torch.testing.assert_close(candidate_loss, reference_loss, **dict(tolerance))
    candidate_groups = _named_optimizer_groups(candidate_model, candidate_optimizer)
    reference_groups = _named_optimizer_groups(reference_model, reference_optimizer)
    if not _value_equal(candidate_groups, reference_groups):
        raise RuntimeError(f"{label} optimizer parameter groups differ")
    return {
        "loss_max_abs": _max_abs(candidate_loss, reference_loss),
        "model_state_max_abs": _compare_model_checkpoint(
            candidate_model, reference_model, tolerance
        ),
        "gradient_max_abs": _compare_named_gradients(
            candidate_model, reference_model, tolerance
        ),
        "optimizer_state_max_abs": _compare_named_optimizer_states(
            candidate_model, candidate_optimizer,
            reference_model, reference_optimizer, tolerance,
        ),
        "optimizer_groups_match": True,
    }


def _capture_complete_step_evidence(
    model: Any,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    *,
    label: str,
    before_model: Mapping[str, torch.Tensor] | None = None,
    before_optimizer: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Capture one complete update as finite, CPU-owned evidence."""

    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"{label} must return a loss tensor")
    _finite(loss, f"{label} loss")
    model_state = _clone_model_checkpoint(model)
    gradients = _clone_named_gradients(model)
    parameters = dict(model.named_parameters())
    for name in parameters:
        if gradients[name] is None:
            raise RuntimeError(f"{label} missing gradient for named parameter {name}")
    optimizer_state = _named_optimizer_state(model, optimizer)
    optimizer_groups = _clone_state_value_cpu(_named_optimizer_groups(model, optimizer))
    evidence = {
        "loss": loss.detach().cpu().clone(),
        "model_state": model_state,
        "gradients": gradients,
        "optimizer_state": optimizer_state,
        "optimizer_groups": optimizer_groups,
    }
    _validate_finite_state(evidence, label)
    if before_model is not None:
        evidence["parameter_updates"] = [
            name for name in sorted(set(before_model) | set(model_state))
            if not _value_equal(before_model.get(name), model_state.get(name))
        ]
    if before_optimizer is not None:
        evidence["optimizer_updates"] = _changed_names(before_optimizer, optimizer_state)
    return evidence


def _compare_complete_step_evidence(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    tolerance: Mapping[str, float],
    *,
    label: str,
) -> dict[str, Any]:
    """Compare two complete-step records after both models leave the GPU."""

    _require_cpu_evidence(candidate, f"{label} candidate evidence")
    _require_cpu_evidence(reference, f"{label} reference evidence")
    for evidence_name, evidence in (("candidate", candidate), ("reference", reference)):
        for field in ("loss", "model_state", "gradients", "optimizer_state", "optimizer_groups"):
            if field not in evidence:
                raise RuntimeError(f"{label} {evidence_name} evidence lacks {field}")
    loss_max_abs = _compare_state_values(
        candidate["loss"], reference["loss"], tolerance,
        label=f"{label} loss",
    )
    model_max_abs = _compare_state_values(
        candidate["model_state"], reference["model_state"], tolerance,
        label=f"{label} model state",
    )
    gradient_max_abs = _compare_state_values(
        candidate["gradients"], reference["gradients"], tolerance,
        label=f"{label} gradients",
    )
    candidate_optimizer = candidate["optimizer_state"]
    reference_optimizer = reference["optimizer_state"]
    if set(candidate_optimizer) != set(reference_optimizer):
        raise RuntimeError(f"{label} optimizer state parameter names differ")
    optimizer_max_abs: dict[str, dict[str, float]] = {}
    for name in sorted(reference_optimizer):
        candidate_values = candidate_optimizer[name]
        reference_values = reference_optimizer[name]
        if set(candidate_values) != set(reference_values):
            raise RuntimeError(f"{label} optimizer state keys differ for {name}")
        optimizer_max_abs[name] = {}
        for key in sorted(reference_values):
            optimizer_max_abs[name][key] = _compare_state_values(
                candidate_values[key], reference_values[key], tolerance,
                label=f"{label} optimizer state {name}.{key}",
                exact=key == "step",
            )
    if not _value_equal(candidate["optimizer_groups"], reference["optimizer_groups"]):
        raise RuntimeError(f"{label} optimizer parameter groups differ")
    return {
        "loss_max_abs": loss_max_abs,
        "model_state_max_abs": model_max_abs,
        "gradient_max_abs": gradient_max_abs,
        "optimizer_state_max_abs": optimizer_max_abs,
        "optimizer_groups_match": True,
    }


def _complete_step_qualification(
    *,
    candidate_model: Any,
    candidate_optimizer: torch.optim.Optimizer,
    candidate_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    reference_factory: Callable[[Any, torch.device], Any],
    optimizer_config: Mapping[str, Any],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    accumulation: int,
    protocol: Mapping[str, Any],
    device: torch.device,
    cuda_graph: bool,
    label: str,
) -> dict[str, Any]:
    """Run one untimed complete step with the reference on the GPU alone."""

    tolerance = _tolerance(protocol, torch.bfloat16)
    cpu = torch.device("cpu")
    candidate_device = next(
        (parameter.device for parameter in candidate_model.parameters()), device
    )
    before_model = _clone_model_checkpoint(candidate_model)
    before_optimizer = _clone_optimizer_checkpoint(candidate_optimizer)
    before_named_optimizer = _named_optimizer_state(candidate_model, candidate_optimizer)
    before_optimizer_groups = _named_optimizer_groups(candidate_model, candidate_optimizer)
    before_gradients = _clone_named_gradients(candidate_model)
    reference = None
    reference_optimizer = None
    reference_before_model = None
    reference_before_gradients = None
    dynamo_before = _dynamo_counters()
    try:
        # The factory is asked for CPU explicitly.  Accept a factory that
        # ignores its device argument, but move it off the GPU before the
        # candidate step so the Full model is never duplicated there.
        reference = reference_factory(getattr(candidate_model, "config", None), cpu)
        if reference is candidate_model:
            raise RuntimeError("candidate and reference models must be independent")
        reference.to(cpu)
        reference_before_model = _clone_model_checkpoint(reference)
        reference_before_gradients = _clone_named_gradients(reference)

        candidate_loss = candidate_step(tokens, targets)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        candidate_evidence = _capture_complete_step_evidence(
            candidate_model,
            candidate_optimizer,
            candidate_loss,
            label=label,
            before_model=before_model,
            before_optimizer=before_named_optimizer,
        )

        _move_model_optimizer(candidate_model, candidate_optimizer, cpu)
        if device.type == "cuda":
            _release_qualification_memory(device)

        reference.to(device)
        _restore_model_checkpoint(reference, before_model)
        reference_optimizer, _ = _adamw(
            reference.parameters(), optimizer_config, cuda_graph=cuda_graph
        )
        _restore_named_optimizer_state(reference, reference_optimizer, before_named_optimizer)
        _restore_optimizer_groups(reference, reference_optimizer, before_optimizer_groups)
        reference_loss = _oracle_complete_training_step(
            reference, reference_optimizer, tokens, targets, accumulation
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        reference_evidence = _capture_complete_step_evidence(
            reference,
            reference_optimizer,
            reference_loss,
            label=f"{label} reference",
            before_model=before_model,
            before_optimizer=before_named_optimizer,
        )
        expected_parameter_names = set(dict(reference.named_parameters()))
        for evidence_name, evidence in (("candidate", candidate_evidence), ("reference", reference_evidence)):
            if set(evidence["optimizer_state"]) != expected_parameter_names:
                raise RuntimeError(
                    f"{label} {evidence_name} did not initialize optimizer state for every model parameter"
                )
        actual_changed = list(candidate_evidence["parameter_updates"])
        expected_changed = list(reference_evidence["parameter_updates"])
        if expected_changed and not actual_changed:
            raise RuntimeError(f"{label} made no model parameter update")
        candidate_optimizer_changed = list(candidate_evidence["optimizer_updates"])
        reference_optimizer_changed = list(reference_evidence["optimizer_updates"])
        if reference_optimizer_changed and not candidate_optimizer_changed:
            raise RuntimeError(f"{label} made no optimizer-state update")

        # Drop the reference's device allocations before comparing or restoring
        # the candidate.  All comparison leaves below are CPU-owned evidence.
        reference.to(cpu)
        del reference_optimizer
        reference_optimizer = None
        dynamo_delta = _counter_delta(dynamo_before, _dynamo_counters())
        if _graph_breaks(dynamo_delta) or _recompiles(dynamo_delta):
            raise RuntimeError(f"{label} introduced a graph break or recompilation")
        return {
            "status": "qualified",
            **_compare_complete_step_evidence(
                candidate_evidence,
                reference_evidence,
                tolerance,
                label=label,
            ),
            "candidate_parameter_updates": actual_changed,
            "reference_parameter_updates": expected_changed,
            "candidate_optimizer_updates": candidate_optimizer_changed,
            "reference_optimizer_updates": reference_optimizer_changed,
            "state_restored": True,
            "dynamo_delta": dynamo_delta,
            "tolerance": tolerance,
            "reference_evidence_device": "cpu",
        }
    finally:
        try:
            if reference is not None:
                try:
                    if reference_optimizer is not None:
                        del reference_optimizer
                        reference_optimizer = None
                    reference.to(cpu)
                    if reference_before_model is not None:
                        _restore_model_checkpoint(reference, reference_before_model)
                    if reference_before_gradients is not None:
                        _restore_named_gradients(reference, reference_before_gradients)
                finally:
                    del reference
        finally:
            # Candidate restoration is kept in an outer finally so a
            # reference cleanup error cannot strand the candidate on CPU.
            _move_model_optimizer(candidate_model, candidate_optimizer, candidate_device)
            _restore_and_verify_candidate(
                candidate_model,
                candidate_optimizer,
                before_model,
                before_optimizer,
                before_gradients,
                before_named_optimizer,
                before_optimizer_groups,
            )
            gc.collect()


def _changed_graph_inputs(
    tokens: torch.Tensor,
    targets: torch.Tensor,
    vocab: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if vocab <= 2:
        raise RuntimeError("graph qualification requires vocab > 2")
    samples = [
        (torch.remainder(tokens + offset, vocab),
         torch.remainder(targets + offset + 1, vocab))
        for offset in (1, 2)
    ]
    if any(torch.equal(sample_tokens, tokens) or torch.equal(sample_targets, targets)
           for sample_tokens, sample_targets in samples):
        raise RuntimeError("graph qualification inputs did not change")
    if torch.equal(samples[0][0], samples[1][0]) or torch.equal(samples[0][1], samples[1][1]):
        raise RuntimeError("graph qualification inputs are not distinct")
    return samples


def _graph_replay_qualification(
    *,
    candidate_model: Any,
    candidate_optimizer: torch.optim.Optimizer,
    graph_step: Any,
    reference_factory: Callable[[Any, torch.device], Any],
    optimizer_config: Mapping[str, Any],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    accumulation: int,
    vocab: int,
    protocol: Mapping[str, Any],
    device: torch.device,
    capture_inputs: tuple[torch.Tensor, torch.Tensor],
    reference_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare two changed-input graph replays against CPU reference evidence."""

    if graph_step is None:
        raise RuntimeError("missing captured complete-step graph")
    changed_inputs = _changed_graph_inputs(tokens, targets, vocab)
    if reference_evidence is None:
        if device.type == "cuda":
            raise RuntimeError(
                "CUDA Graph reference evidence must be precomputed before capture"
            )
        reference = reference_factory(getattr(candidate_model, "config", None), torch.device("cpu"))
        try:
            reference.to("cpu")
            reference_evidence = _precompute_graph_reference_evidence(
                candidate_model=candidate_model,
                candidate_optimizer=candidate_optimizer,
                reference=reference,
                optimizer_config=optimizer_config,
                tokens=tokens,
                targets=targets,
                accumulation=accumulation,
                vocab=vocab,
                device=device,
            )
        finally:
            del reference
    if len(reference_evidence) != len(changed_inputs) or len(reference_evidence) != 2:
        raise RuntimeError("graph qualification requires exactly two reference evidence records")
    for index, record in enumerate(reference_evidence, start=1):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"CUDA Graph replay {index} reference evidence is not a mapping")
        _require_cpu_evidence(record, f"CUDA Graph replay {index} reference evidence")
        if any(field not in record for field in (
            "loss", "model_state", "gradients", "optimizer_state", "optimizer_groups",
        )):
            raise RuntimeError(f"CUDA Graph replay {index} reference evidence is incomplete")
    tolerance = _tolerance(protocol, torch.bfloat16)
    before_model = _clone_model_checkpoint(candidate_model)
    before_optimizer = _clone_optimizer_checkpoint(candidate_optimizer)
    before_named_optimizer = _named_optimizer_state(candidate_model, candidate_optimizer)
    before_optimizer_groups = _named_optimizer_groups(candidate_model, candidate_optimizer)
    before_gradients = _clone_named_gradients(candidate_model)
    dynamo_before = _dynamo_counters()
    replay_metrics = []
    capture_hash = _input_digest(*capture_inputs)
    try:
        for index, (changed_tokens, changed_targets) in enumerate(changed_inputs, start=1):
            if _input_digest(changed_tokens, changed_targets) == capture_hash:
                raise RuntimeError("graph qualification replay reused capture inputs")
            replay_before_model = _clone_model_checkpoint(candidate_model)
            replay_before_optimizer = _named_optimizer_state(candidate_model, candidate_optimizer)
            graph_step.copy_inputs(changed_tokens, changed_targets)
            candidate_loss = graph_step.replay()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            candidate_evidence = _capture_complete_step_evidence(
                candidate_model,
                candidate_optimizer,
                candidate_loss,
                label=f"CUDA Graph replay {index}",
                before_model=replay_before_model,
                before_optimizer=replay_before_optimizer,
            )
            reference_record = reference_evidence[index - 1]
            expected_changed = list(reference_record.get("parameter_updates", ()))
            actual_changed = list(candidate_evidence.get("parameter_updates", ()))
            if expected_changed and not actual_changed:
                raise RuntimeError(f"CUDA Graph replay {index} made no model parameter update")
            actual_named_optimizer = candidate_evidence["optimizer_state"]
            reference_named_optimizer = reference_record["optimizer_state"]
            expected_parameter_names = set(dict(candidate_model.named_parameters()))
            if set(actual_named_optimizer) != expected_parameter_names:
                raise RuntimeError(
                    f"CUDA Graph replay {index} did not initialize optimizer state for every model parameter"
                )
            if set(reference_named_optimizer) != expected_parameter_names:
                raise RuntimeError(
                    f"CUDA Graph replay {index} reference did not initialize optimizer state for every model parameter"
                )
            candidate_optimizer_changed = list(candidate_evidence.get("optimizer_updates", ()))
            reference_optimizer_changed = list(reference_record.get("optimizer_updates", ()))
            if reference_optimizer_changed and not candidate_optimizer_changed:
                raise RuntimeError(f"CUDA Graph replay {index} made no optimizer-state update")
            replay_metrics.append({
                "index": index,
                **_compare_complete_step_evidence(
                    candidate_evidence,
                    reference_record,
                    tolerance,
                    label=f"CUDA Graph replay {index}",
                ),
                "candidate_parameter_updates": actual_changed,
                "reference_parameter_updates": expected_changed,
                "candidate_optimizer_updates": candidate_optimizer_changed,
                "reference_optimizer_updates": reference_optimizer_changed,
            })
        dynamo_delta = _counter_delta(dynamo_before, _dynamo_counters())
        if dynamo_delta:
            raise RuntimeError("CUDA Graph qualification changed Dynamo counters")
        return {
            "status": "qualified",
            "replays": replay_metrics,
            "replay_count": len(replay_metrics),
            "capture_input_hash": capture_hash,
            "replay_input_hashes": [
                _input_digest(*sample) for sample in changed_inputs
            ],
            "dynamo_delta": dynamo_delta,
            "tolerance": tolerance,
            "state_restored": True,
        }
    finally:
        _restore_and_verify_candidate(
            candidate_model,
            candidate_optimizer,
            before_model,
            before_optimizer,
            before_gradients,
            before_named_optimizer,
            before_optimizer_groups,
            preserve_optimizer_allocations=True,
        )
        gc.collect()


def _precompute_graph_reference_evidence(
    *,
    candidate_model: Any,
    candidate_optimizer: torch.optim.Optimizer,
    reference: Any,
    optimizer_config: Mapping[str, Any],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    accumulation: int,
    vocab: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run exactly two changed-input reference steps before graph capture.

    The candidate and optimizer are offloaded to CPU while the independent
    reference occupies the requested device.  Only detached CPU evidence is
    returned, allowing later graph replays to run without a concurrent oracle.
    """

    if candidate_model is reference:
        raise RuntimeError("candidate and reference models must be independent")
    cpu = torch.device("cpu")
    candidate_device = next(
        (parameter.device for parameter in candidate_model.parameters()), device
    )
    before_model = _clone_model_checkpoint(candidate_model)
    before_optimizer = _clone_optimizer_checkpoint(candidate_optimizer)
    before_named_optimizer = _named_optimizer_state(candidate_model, candidate_optimizer)
    before_optimizer_groups = _named_optimizer_groups(candidate_model, candidate_optimizer)
    before_gradients = _clone_named_gradients(candidate_model)
    reference_before_model = _clone_model_checkpoint(reference)
    reference_before_gradients = _clone_named_gradients(reference)
    reference_optimizer = None
    evidence: list[dict[str, Any]] = []
    try:
        reference.to(cpu)
        _move_model_optimizer(candidate_model, candidate_optimizer, cpu)
        if device.type == "cuda":
            _release_qualification_memory(device)
        reference.to(device)
        _restore_model_checkpoint(reference, before_model)
        reference_optimizer, _ = _adamw(
            reference.parameters(), optimizer_config, cuda_graph=device.type == "cuda"
        )
        _restore_named_optimizer_state(reference, reference_optimizer, before_named_optimizer)
        _restore_optimizer_groups(reference, reference_optimizer, before_optimizer_groups)
        for index, (changed_tokens, changed_targets) in enumerate(
            _changed_graph_inputs(tokens, targets, vocab), start=1
        ):
            step_before_model = _clone_model_checkpoint(reference)
            step_before_optimizer = _named_optimizer_state(reference, reference_optimizer)
            reference_loss = _oracle_complete_training_step(
                reference,
                reference_optimizer,
                changed_tokens,
                changed_targets,
                accumulation,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            evidence.append(_capture_complete_step_evidence(
                reference,
                reference_optimizer,
                reference_loss,
                label=f"reference changed-input step {index}",
                before_model=step_before_model,
                before_optimizer=step_before_optimizer,
            ))
        if len(evidence) != 2:
            raise RuntimeError("changed-input graph qualification must produce exactly two references")
        expected_parameter_names = set(dict(reference.named_parameters()))
        for index, record in enumerate(evidence, start=1):
            if set(record["optimizer_state"]) != expected_parameter_names:
                raise RuntimeError(
                    f"reference changed-input step {index} did not initialize optimizer state for every model parameter"
                )
        return evidence
    finally:
        try:
            if reference_optimizer is not None:
                del reference_optimizer
                reference_optimizer = None
            reference.to(cpu)
            _restore_model_checkpoint(reference, reference_before_model)
            _restore_named_gradients(reference, reference_before_gradients)
        finally:
            _move_model_optimizer(candidate_model, candidate_optimizer, candidate_device)
            _restore_and_verify_candidate(
                candidate_model,
                candidate_optimizer,
                before_model,
                before_optimizer,
                before_gradients,
                before_named_optimizer,
                before_optimizer_groups,
            )
            gc.collect()


def _dynamo_counters() -> dict[str, dict[str, int]]:
    counters = getattr(getattr(torch, "_dynamo", None), "utils", None)
    counters = getattr(counters, "counters", {})
    return {str(section): {str(key): int(value) for key, value in values.items()} for section, values in counters.items() if values}


def _counter_delta(before: Mapping[str, Mapping[str, int]], after: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    result = {}
    for section in set(before) | set(after):
        delta = {key: int(after.get(section, {}).get(key, 0) - before.get(section, {}).get(key, 0)) for key in set(before.get(section, {})) | set(after.get(section, {})) if after.get(section, {}).get(key, 0) != before.get(section, {}).get(key, 0)}
        if delta: result[section] = delta
    return result


def _counter_total(counters: Mapping[str, Mapping[str, int]], needle: str) -> int:
    return sum(int(value) for section, values in counters.items() for key, value in values.items() if needle in f"{section}.{key}".lower())


def _graph_breaks(counters: Mapping[str, Mapping[str, int]]) -> int: return sum(int(v) for v in counters.get("graph_break", {}).values())
def _recompiles(counters: Mapping[str, Mapping[str, int]]) -> int: return _counter_total(counters, "recompil")
def _unique_graphs(counters: Mapping[str, Mapping[str, int]]) -> int: return _counter_total(counters, "unique_graph")


def _input_digest(*tensors: torch.Tensor) -> str: return _operator_digest(*tensors)


def _logical_model_input_id(
    seed: int,
    sample_index: int,
    model_data: Mapping[str, Any],
) -> str:
    """Identify a paired model sample without reading CUDA tensor bytes.

    Token and target tensors are generated once and shared by every arm for a
    sample.  The complete-step qualification proves that changed inputs are
    consumed before timing.  Timed rows therefore need only a deterministic
    pairing identifier, not a device-to-host copy and byte hash.
    """

    payload = {
        "protocol": "logical_model_sample_v1",
        "seed": int(seed),
        "sample_index": int(sample_index),
        "batch": int(model_data["batch"]),
        "sequence": int(model_data["sequence"]),
        "vocab": int(model_data["vocab"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _check_model_gradients(model: Any, name: str) -> int:
    parameters = [p for p in model.parameters() if p.requires_grad]
    missing = [str(i) for i, p in enumerate(parameters) if p.grad is None]
    if missing: raise RuntimeError(f"{name} produced no gradient for parameters: {','.join(missing)}")
    for index, parameter in enumerate(parameters): _finite(parameter.grad, f"{name} gradient {index}")
    return len(parameters)


def _adamw(parameters: Any, config: Mapping[str, Any], *, cuda_graph: bool = False) -> tuple[torch.optim.Optimizer, str]:
    params = list(parameters)
    options = {"lr": float(config.get("lr", 3e-4)), "betas": tuple(config.get("betas", (0.9, 0.95))), "weight_decay": float(config.get("weight_decay", 0.1))}
    if cuda_graph:
        return torch.optim.AdamW(params, fused=True, capturable=True, **options), "AdamW(fused=True,capturable=True)"
    try: return torch.optim.AdamW(params, fused=True, **options), "AdamW(fused=True)"
    except (TypeError, RuntimeError): return torch.optim.AdamW(params, foreach=True, **options), "AdamW(foreach=True)"


def _compiled_training_step(model: Any, optimizer: torch.optim.Optimizer, loss_function: Callable[..., Any], tokens: torch.Tensor, targets: torch.Tensor, accumulation: int) -> torch.Tensor:
    """Run one complete compiled step over the canonical microbatch layout.

    A two-dimensional batch is split along its batch axis, while a three
    dimensional input already has an explicit leading microbatch axis.  Each
    backward call is scaled by the number of actual microbatches.  The final
    microbatch loss is returned for compatibility with runner telemetry; this
    return value does not affect the optimizer update.
    """
    from .model import _microbatches as canonical_microbatches

    batches = canonical_microbatches(tokens, targets, accumulation)
    optimizer.zero_grad(set_to_none=True)
    result: torch.Tensor | None = None
    divisor = float(len(batches))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for micro_tokens, micro_targets in batches:
            logits = model(micro_tokens)
            if not isinstance(logits, torch.Tensor): raise TypeError("compiled model must return a logits tensor")
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), micro_targets.reshape(-1))
            if not isinstance(loss, torch.Tensor): raise TypeError("compiled loss must return a tensor")
            result = loss
            (loss / divisor).backward()
    optimizer.step()
    if result is None: raise RuntimeError("no microbatches were produced")
    return result.detach()


def _cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: return torch.nn.functional.cross_entropy(logits, targets)


def _model_comparisons(raw, arms, ranks, rounds, include_reference, architecture_comparisons):
    comparisons = {}
    for rank in ranks:
        reference = [r["ms"] for r in raw if r["arm"] == f"reference_rank_{rank}" and r["status"] == "ok"]
        kernel = [r["ms"] for r in raw if r["arm"] == f"kernel_rank_{rank}" and r["status"] == "ok"]
        if include_reference and len(reference) == len(kernel) == rounds: comparisons[f"kernel_rank_{rank}_over_reference"] = (reference, kernel)
        for name, arm in arms.items():
            if name in architecture_comparisons or arm["rank"] != rank or arm["backend"] in {"kernel", "reference"}: continue
            candidate = [r["ms"] for r in raw if r["arm"] == name and r["status"] == "ok"]
            if len(candidate) == rounds:
                if include_reference and len(reference) == rounds: comparisons[f"{name}_over_reference"] = (reference, candidate)
                elif not include_reference and len(kernel) == rounds: comparisons[f"kernel_rank_{rank}_over_{name}"] = (candidate, kernel)
                if arm["backend"] == "per_read" and len(kernel) == rounds:
                    comparisons[f"kernel_rank_{rank}_over_{name}"] = (candidate, kernel)
        packed = [r["ms"] for r in raw if r["arm"] == f"packed_rank_{rank}" and r["status"] == "ok"]
        if len(packed) == len(kernel) == rounds:
            # The named statistic is list / packed.  The estimator accepts
            # (baseline, candidate), so keep packed first here.
            comparisons[f"kernel_rank_{rank}_over_packed_rank_{rank}"] = (packed, kernel)
    ordered = sorted(set(ranks))
    for name in architecture_comparisons:
        standard = [r["ms"] for r in raw if r["arm"] == name and r["status"] == "ok"]
        for rank in ranks:
            kernel = [r["ms"] for r in raw if r["arm"] == f"kernel_rank_{rank}" and r["status"] == "ok"]
            if len(standard) == len(kernel) == rounds:
                comparisons[f"kernel_rank_{rank}_over_{name}"] = (standard, kernel)
    for smaller, larger in zip(ordered, ordered[1:]):
        small = [r["ms"] for r in raw if r["arm"] == f"kernel_rank_{smaller}" and r["status"] == "ok"]
        large = [r["ms"] for r in raw if r["arm"] == f"kernel_rank_{larger}" and r["status"] == "ok"]
        if len(small) == len(large) == rounds: comparisons[f"kernel_rank_{smaller}_over_rank_{larger}"] = (large, small)
    return comparisons


def _model_failure_result(model_data: Mapping[str, Any], variant: str, ranks: Sequence[int], **fields: Any) -> dict[str, Any]:
    return {"status": "failed", "config": dict(model_data), "effective_variant": variant, "ranks": list(ranks), **fields}


def _is_core_model_arm(arm: Mapping[str, Any]) -> bool:
    """Return whether an arm failure belongs to the primary model result."""

    return arm.get("backend") in {"kernel", "reference"} and arm.get("comparison") is None


_MODEL_PROFILE_LIMITATIONS = [
    "One existing captured complete-step replay is profiled per successful CUDA Graph arm after timed statistics.",
    "Profiler overhead and synchronization are outside the primary CUDA-event timing interval.",
    "Summed CUDA device durations can overlap and are not a wall-clock complete-step total.",
    "CUDA device durations are grouped by the names reported by PyTorch; names alone do not establish category causality.",
    "Trace coverage depends on PyTorch Kineto CPU/CUDA activity support.",
    "Launch attributes are raw fields from the exported Kineto/CUPTI Chrome trace; unavailable fields are not inferred.",
    "Correlation IDs are bounded samples with omission counts; full launch-group coverage does not imply exhaustive IDs.",
]

_MODEL_PROFILE_MAX_KERNEL_GROUPS = 512
_MODEL_PROFILE_MAX_CORRELATION_IDS = 32
_PROFILE_TRACE_MISSING = object()
_PROFILE_LAUNCH_FIELDS = (("grid", "grid", True), ("block", "block", True),
                          ("registers per thread", "registers per thread", False),
                          ("shared memory", "shared memory", False))


def _profile_cuda_event(event: Any) -> bool:
    device_type = getattr(event, "device_type", None)
    return "cuda" in str(device_type).lower()


def _profile_cuda_duration_us(event: Any) -> float:
    for attribute in ("device_time_total", "self_device_time_total", "cuda_time_total", "self_cuda_time_total"):
        try:
            value = float(getattr(event, attribute))
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0.0:
            return value
    raise RuntimeError("CUDA profiler event did not expose a finite device duration")


def _profile_trace_raw(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        raw = [_profile_trace_raw(item) for item in value]
        return raw if all(item is not None or original is None for item, original in zip(raw, value)) else None
    return None


def _profile_trace_value(value: Any, vector: bool = False) -> tuple[Any, str]:
    if value is _PROFILE_TRACE_MISSING:
        return None, "unavailable"
    if vector:
        valid = (isinstance(value, (list, tuple)) and len(value) == 3
                 and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value))
        normalized = list(value) if valid else _profile_trace_raw(value)
    else:
        valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        normalized = value if valid else _profile_trace_raw(value)
    return normalized, "available" if valid else "unknown"


def _profile_trace_kernel_rows(path: str | os.PathLike[str], limit: int = _MODEL_PROFILE_MAX_KERNEL_GROUPS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    events = payload.get("traceEvents") if isinstance(payload, Mapping) else payload
    if not isinstance(events, list):
        raise RuntimeError("exported profiler trace did not contain a traceEvents list")
    aggregates: dict[str, dict[str, Any]] = {}
    kernel_event_count = 0
    for event in events:
        if (not isinstance(event, Mapping) or event.get("cat") != "kernel"
                or event.get("ph") != "X"):
            continue
        name = event.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("exported CUDA kernel event did not expose a kernel name")
        duration = event.get("dur")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise RuntimeError("exported CUDA kernel event did not expose a finite duration")
        duration_us = float(duration)
        source = dict(event)
        if isinstance(event.get("args"), Mapping): source.update(event["args"])
        metadata, metadata_status = {}, {}
        for field, trace_key, vector in _PROFILE_LAUNCH_FIELDS:
            value, status = _profile_trace_value(source.get(trace_key, _PROFILE_TRACE_MISSING), vector)
            metadata[field], metadata_status[field] = value, status
        correlation, correlation_status = _profile_trace_value(source.get("correlation", _PROFILE_TRACE_MISSING))
        key = json.dumps((str(name), metadata, metadata_status), sort_keys=True, separators=(",", ":"))
        aggregate = aggregates.setdefault(key, {"name": str(name), "count": 0, "device_time_us": 0.0,
            **metadata, "metadata_status": {**metadata_status, "correlation": "unavailable"},
            "correlation_ids": [], "correlation_ids_omitted": 0,
            "correlation_status_counts": {"available": 0, "unavailable": 0, "unknown": 0}})
        aggregate["count"] += 1
        aggregate["device_time_us"] += duration_us
        statuses = aggregate["correlation_status_counts"]
        statuses[correlation_status] += 1
        if correlation_status == "available":
            if len(aggregate["correlation_ids"]) < _MODEL_PROFILE_MAX_CORRELATION_IDS:
                aggregate["correlation_ids"].append(correlation)
            else:
                aggregate["correlation_ids_omitted"] += 1
        active_statuses = [status for status, count in statuses.items() if count]
        aggregate["metadata_status"]["correlation"] = active_statuses[0] if len(active_statuses) == 1 else "mixed"
        kernel_event_count += 1
    if not kernel_event_count:
        raise RuntimeError("exported profiler trace contained no CUDA kernel events")
    kernels = sorted(aggregates.values(), key=lambda row: (-float(row["device_time_us"]), row["name"]))
    retained = kernels[:max(1, int(limit))]
    dropped = len(kernels) - len(retained)
    return retained, {"kernel_event_count": kernel_event_count, "kernel_group_count": len(kernels),
                      "retained_kernel_group_count": len(retained), "dropped_kernel_group_count": dropped,
                      "max_kernel_groups": max(1, int(limit)), "truncated": dropped > 0,
                      "correlation_ids_omitted": sum(row["correlation_ids_omitted"] for row in kernels),
                      "correlation_ids_truncated": any(row["correlation_ids_omitted"] for row in kernels)}


def _profile_cuda_graph_replay(graph_step: Any, device: torch.device) -> dict[str, Any]:
    replayed = False
    trace_path = None
    try:
        torch_profiler = importlib.import_module("torch.profiler")
        activities = [torch_profiler.ProfilerActivity.CPU, torch_profiler.ProfilerActivity.CUDA]
        torch.cuda.synchronize(device)
        with torch_profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
        ) as profiler:
            replayed = True
            if getattr(device, "type", None) == "cuda":
                with torch.cuda.device(device):
                    graph_step.replay()
            else:
                graph_step.replay()
        torch.cuda.synchronize(device)
        events_method = getattr(profiler, "events", None)
        if callable(events_method):
            events = events_method()
        elif events_method is not None:
            events = events_method
        else:
            events = getattr(profiler, "function_events", None)
        if events is None:
            raise RuntimeError("torch.profiler did not expose recorded events")
        aggregates: dict[str, dict[str, float | int]] = {}
        cuda_event_count = 0
        for event in events:
            if not _profile_cuda_event(event):
                continue
            name = getattr(event, "name", None) or getattr(event, "key", None)
            if not name:
                raise RuntimeError("CUDA profiler event did not expose a kernel name")
            try:
                count = max(1, int(getattr(event, "count", 1)))
            except (TypeError, ValueError):
                count = 1
            duration_us = _profile_cuda_duration_us(event)
            aggregate = aggregates.setdefault(str(name), {"count": 0, "device_time_us": 0.0})
            aggregate["count"] = int(aggregate["count"]) + count
            aggregate["device_time_us"] = float(aggregate["device_time_us"]) + duration_us
            cuda_event_count += count
        if not cuda_event_count:
            raise RuntimeError("torch.profiler returned no CUDA device events")
        kernels = [
            {"name": name, "count": int(values["count"]), "device_time_us": float(values["device_time_us"])}
            for name, values in sorted(aggregates.items(), key=lambda item: (-float(item[1]["device_time_us"]), item[0]))
        ]
        result = {
            "status": "complete",
            "activities": ["CPU", "CUDA"],
            "replay_count": 1,
            "cuda_event_count": cuda_event_count,
            "cuda_kernels": kernels,
            "limitations": list(_MODEL_PROFILE_LIMITATIONS),
        }
        export_method = getattr(profiler, "export_chrome_trace", None)
        if callable(export_method):
            try:
                descriptor, trace_path = tempfile.mkstemp(prefix="attnres_model_profile_", suffix=".json")
                os.close(descriptor)
                export_method(trace_path)
                launch_rows, trace_summary = _profile_trace_kernel_rows(trace_path)
                result.update(cuda_launches=launch_rows, trace_source="export_chrome_trace",
                              trace_summary=trace_summary)
                if trace_summary["truncated"]:
                    result["status"] = "incomplete"
                    result["error"] = {"type": "RuntimeError", "message": "kernel launch rows were truncated"}
            except Exception as exc:
                result.update(status="failed", cuda_launches=[],
                              trace_summary={"status": "failed", "error": _exception(exc)})
        else:
            result.update(cuda_launches=[], trace_summary={
                "status": "unavailable",
                "reason": "torch.profiler did not expose export_chrome_trace",
            })
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "activities": ["CPU", "CUDA"],
            "replay_count": 1 if replayed else 0,
            "limitations": list(_MODEL_PROFILE_LIMITATIONS),
            "error": _exception(exc),
        }
    finally:
        if trace_path is not None:
            try:
                os.unlink(trace_path)
            except OSError:
                pass


def _model_profile_report(
    active: Sequence[str],
    failed_arms: set[str],
    raw: Sequence[Mapping[str, Any]],
    rounds: int,
    graph_steps: Mapping[str, Any],
    model_timing: str,
    device: torch.device,
    timed_stability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "enabled": True,
        "requested": True,
        "method": "torch.profiler.profile",
        "limitations": list(_MODEL_PROFILE_LIMITATIONS),
        "arms": {},
        "profiled_arms": [],
        "failures": [],
    }
    if model_timing != "cuda_graph":
        report.update(status="skipped", reason="model_profile requires model_timing='cuda_graph'")
        return report
    if timed_stability is not None and not bool(timed_stability.get("stable", False)):
        report.update(
            status="skipped",
            qualification="unqualified",
            reason="timed graph-counter stability failed",
            timed_graph_counters=dict(timed_stability),
        )
        return report
    successful = [
        name for name in active
        if name in graph_steps
        and name not in failed_arms
        and sum(row.get("arm") == name and row.get("status") == "ok" for row in raw) == rounds
    ]
    if not successful:
        report.update(status="skipped", reason="no successful captured CUDA Graph arms remained after timing")
        return report
    for name in successful:
        try:
            arm_report = _profile_cuda_graph_replay(graph_steps[name], device)
        except Exception as exc:
            arm_report = {
                "status": "failed",
                "activities": ["CPU", "CUDA"],
                "replay_count": 0,
                "limitations": list(_MODEL_PROFILE_LIMITATIONS),
                "error": _exception(exc),
            }
        report["arms"][name] = arm_report
        if arm_report.get("status") != "complete":
            report["failures"].append(_failure(
                "model_profile",
                arm=name,
                status="failed",
                error=arm_report.get("error", {"message": "CUDA profiler did not complete"}),
            ))
    report["status"] = "failed" if report["failures"] else "complete"
    report["profiled_arms"] = successful
    return report


def _source_layout_metadata(
    list_config: Mapping[str, Any],
    packed_config: Mapping[str, Any],
    *,
    variant: str,
    mode: str,
    rank: int,
) -> dict[str, Any]:
    """Describe a list/packed pair without conflating it with architecture arms."""

    kernel_schedule = _kernel_schedule(mode)
    list_schedule = f"{kernel_schedule}; ordered source list passed directly to the kernel"
    packed_schedule = f"{kernel_schedule}; sources supplied through the existing packed tensor path"
    return {
        "kind": "same_equation_source_layout_only",
        "variant": variant,
        "rank": int(rank),
        "width": int(list_config["width"]),
        "list_arm": f"kernel_rank_{rank}",
        "packed_arm": f"packed_rank_{rank}",
        "ratio": "list/packed",
        "list_config": dict(list_config),
        "packed_config": dict(packed_config),
        "configs": {"list": dict(list_config), "packed": dict(packed_config)},
        "list_schedule": list_schedule,
        "packed_schedule": packed_schedule,
        "schedules": {"list": list_schedule, "packed": packed_schedule, "mode": mode},
    }


def _kernel_schedule(mode: str) -> str:
    """Describe the actual project kernel schedule for model metadata."""

    return (
        "public attnres.attnres for each Block read"
        if mode == "block"
        else "per-read aggregation"
    )


def _model_timings(
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    scope: str,
    device: torch.device,
    seed: int,
    comparators: Mapping[str, Any],
    model_comparators: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    progress = _model_progress_logger(config)
    try:
        from .model import (
            CANONICAL_MAX_RANK_STATE_PROTOCOL,
            TrainingConfig,
            canonical_max_rank_state,
            make_model,
            make_model_with_canonical_state,
            training_step,
        )
        from .bf16_device import bf16_torch
        if not callable(training_step): raise TypeError("benchmarks.model.training_step must be callable")
    except Exception as exc: return {"status": "incomplete", "reason": "benchmarks.model unavailable", "failures": [_failure("model_import", error=_exception(exc))]}
    model_data = _model_config(protocol, config, scope)
    selected_model = config.get("model_config", config.get("model"))
    if (
        "block_execution" in config
        or isinstance(selected_model, Mapping) and "block_execution" in selected_model
        or "include_per_read" in config
    ):
        return _model_failure_result(model_data, "", [], failures=[_failure(
            "model_setup",
            reason=(
                "cached Block and the include_per_read ablation were removed; "
                "Block always uses the shared per-read attnres primitive"
            ),
        )])
    baseline = None
    if config.get("include_baseline", False):
        from .baseline import load_baseline
        baseline = load_baseline(config.get("baseline_root"))
    raw_ranks = config.get("ranks", protocol["ranks"])
    ranks = [int(raw_ranks)] if isinstance(raw_ranks, int) else [int(r) for r in raw_ranks]
    include_reference = bool(config.get("reference_timing", config.get("include_reference", True)))
    include_fla_model = bool(config.get("include_fla_model", config.get("include_fla", True))) and bool(config.get("include_fla", True)) and not bool(config.get("no_fla", False))
    include_liger_model = bool(config.get("include_liger_model", config.get("include_liger", False)))
    # External model comparators are independently opt-in.  In particular,
    # enabling Liger must not discover or time Catswe implicitly.
    include_catswe_model = bool(config.get("include_catswe_model", False))
    model_comparators = dict(model_comparators or {})
    pairwise = bool(config.get("pairwise", False))
    protocol_ranks = [int(r) for r in protocol["ranks"]]
    admitted_ranks, model_admission, admission_error = _model_only_rank_admission(
        config, model_data, protocol_ranks
    )
    if admission_error is not None:
        return _model_failure_result(
            model_data,
            "",
            [],
            failures=[_failure("model_setup", reason=admission_error)],
        )
    if not ranks or any(r <= 0 for r in ranks): return _model_failure_result(model_data, "", ranks, failures=[_failure("model_setup", error={"message": "ranks must be positive"})])
    if any(r not in admitted_ranks for r in ranks):
        return _model_failure_result(
            model_data,
            "",
            ranks,
            failures=[_failure(
                "model_setup",
                error={"message": "ranks must be in the frozen protocol ladder or sealed model-only admission"},
            )],
        )
    if pairwise and any(r not in protocol_ranks for r in ranks):
        return _model_failure_result(
            model_data,
            "",
            ranks,
            pairwise=True,
            failures=[_failure(
                "model_setup",
                error={"message": "pairwise model jobs cannot include model-only admitted ranks"},
            )],
        )
    if pairwise and not (len(ranks) == 2 and abs(protocol_ranks.index(ranks[0]) - protocol_ranks.index(ranks[1])) == 1): return _model_failure_result(model_data, "", ranks, pairwise=True, failures=[_failure("model_setup", error={"message": "pairwise model jobs require two adjacent protocol ranks"})])
    default_rounds = protocol["rounds"] if scope in {"primary", "heldout"} else protocol["smoke_rounds"]
    rounds = int(config.get("model_rounds", config.get("rounds", default_rounds)))
    requested_warmup = int(config.get("model_warmup", config.get("warmup", protocol["warmup"])))
    warmup, accumulation = max(1, requested_warmup), int(config.get("accumulation", 1))
    model_timing = str(config.get("model_timing", "cuda_event")).lower()
    if model_timing not in {"cuda_event", "cuda_graph"}:
        return _model_failure_result(model_data, "", ranks, failures=[_failure("model_setup", error={"message": "model_timing must be cuda_event or cuda_graph"})])
    if rounds < 1 or requested_warmup < 0 or accumulation < 1: return _model_failure_result(model_data, "", ranks, failures=[_failure("model_setup", error={"message": "rounds and accumulation must be positive, warmup must be non-negative"})])
    variant = _effective_variant(config, model_data, ranks)
    mode = str(model_data["mode"]).lower()
    include_packed_comparison = bool(config.get("include_packed_comparison", False))
    source_layout = str(model_data.get("source_layout", "packed")).lower()
    if isinstance(model_data.get("source_layout"), str):
        model_data = dict(model_data, source_layout=source_layout)
    if include_packed_comparison and source_layout != "list":
        return _model_failure_result(model_data, variant, ranks, failures=[_failure(
            "model_setup", reason="include_packed_comparison requires model_config.source_layout='list'")])
    standard_fla = bool(config.get("standard_fla_comparison", False))
    if standard_fla and (variant != "sliced" or mode not in {"full", "block"}
                         or not config.get("include_fla_compile", False)):
        return _model_failure_result(model_data, variant, ranks, failures=[_failure(
            "model_setup", reason="standard_fla_comparison requires sliced Full or Block and include_fla_compile")])
    requested_state_protocol = config.get("model_state_protocol")
    if requested_state_protocol is not None and requested_state_protocol != CANONICAL_MAX_RANK_STATE_PROTOCOL:
        return _model_failure_result(model_data, variant, ranks, failures=[_failure(
            "model_setup", reason=(
                "model_state_protocol must be absent or "
                f"{CANONICAL_MAX_RANK_STATE_PROTOCOL!r}"
            )
        )])
    canonical_state = None
    state_protocol_report = None
    state_records: dict[str, Any] = {}
    if requested_state_protocol == CANONICAL_MAX_RANK_STATE_PROTOCOL:
        try:
            canonical_config = TrainingConfig(
                **dict(model_data, variant="standard", mode=mode, rank=int(model_data["width"]))
            )
            canonical_state = canonical_max_rank_state(canonical_config, seed)
            state_protocol_report = {
                "name": CANONICAL_MAX_RANK_STATE_PROTOCOL,
                "semantics": (
                    "standard R=D canonical source with implicit value-tail keys; "
                    "sliced targets map the trailing R query coordinates"
                ),
                "seed": int(seed),
                "mode": mode,
                "canonical_source": {
                    "device": "cpu",
                    "backend": "reference",
                    "variant": "standard",
                    "rank": int(canonical_config.width),
                    "key_mode": "implicit_value_tail",
                    "config": {
                        name: getattr(canonical_config, name)
                        for name in (
                            "layers", "width", "heads", "ffn", "batch",
                            "sequence", "vocab", "block_count", "variant",
                            "mode", "rank",
                        )
                    },
                    "initial_state_hash": _state_digest(canonical_state),
                    "shape_metadata": _state_shape_metadata(canonical_state),
                    "common_fixed_state_hash": _state_digest(
                        canonical_state, _common_state_names(canonical_state)
                    ),
                },
                "mapping": {
                    "fixed_shape_tensors": "exact source tensor copy",
                    "standard": "exact source tensor copy",
                    "sliced.queries.*": "source[-R:]",
                    "cuda_generators": "untouched",
                },
                "arms": state_records,
            }
        except Exception as exc:
            return _model_failure_result(
                model_data,
                variant,
                ranks,
                state_protocol={
                    "name": CANONICAL_MAX_RANK_STATE_PROTOCOL,
                    "semantics": "canonical standard R=D source with implicit value-tail keys",
                    "status": "failed",
                },
                failures=[_failure("model_state_protocol", error=_exception(exc))],
            )

    def _make_model(
        rank_config: Any,
        backend: Any,
        *,
        target_device: torch.device | str | None = None,
    ) -> Any:
        target = device if target_device is None else target_device
        if canonical_state is None:
            torch.manual_seed(int(seed))
            return make_model(rank_config, backend=backend).to(target)
        return make_model_with_canonical_state(
            rank_config, backend, canonical_state, int(seed)
        ).to(target)

    def _record_model(name: str, model: Any, rank: int, target_variant: str) -> None:
        if state_protocol_report is None:
            return
        state_records[name] = _model_state_record(
            model,
            arm=name,
            rank=rank,
            variant=target_variant,
            mode=mode,
            protocol_name=CANONICAL_MAX_RANK_STATE_PROTOCOL,
        )

    def _qualify_model(arm_name: str, role: str, reference: Any, candidate: Any) -> dict[str, Any]:
        progress(f"qualification_{role}_start", arm_name)
        try:
            return _model_qualification(reference, candidate, tokens, targets, protocol, compiled_loss)
        finally:
            progress(f"qualification_{role}_end", arm_name)

    tokens, targets = _model_inputs(model_data, device, seed + 5000)
    try: compiled_loss = torch.compile(_cross_entropy_loss, fullgraph=True, dynamic=False)
    except Exception as exc: return _model_failure_result(model_data, variant, ranks, **({"state_protocol": state_protocol_report} if state_protocol_report is not None else {}), compiled_loss={"status": "failed", "fullgraph": True, "dynamic": False}, failures=[_failure("model_compile_loss", error=_exception(exc))])
    arms, qualification, comparator_qualification, failures, comparator_failures = {}, {}, {}, [], []
    model_comparator_scope: dict[str, Any] = {}
    model_backend_metadata: dict[str, Any] = {}
    source_layout_comparisons = {}
    compile_backends, compile_backend_metadata = {}, {}
    if config.get("include_fla_compile", False):
        from .fla_compile import make_model_backend
        implementations = config.get("fla_compile_backends", ["triton", "gluon"])
        if (not isinstance(implementations, list) or not implementations
                or any(not isinstance(name, str) for name in implementations)
                or len(set(implementations)) != len(implementations)
                or any(name not in {"triton", "gluon"} for name in implementations)):
            return _model_failure_result(model_data, variant, ranks, **({"state_protocol": state_protocol_report} if state_protocol_report is not None else {}), failures=[_failure(
                "model_setup", reason="fla_compile_backends must select unique triton/gluon names")])
        for implementation in implementations:
            name = f"fla_{implementation}_compile"
            try:
                backend = make_model_backend(
                    implementation,
                    vendor_root=config.get("vendor_root", config.get("fla_root")),
                )
                compile_backend_metadata[name] = backend.source_hash_metadata
                compile_backends[name] = backend
            except Exception as exc:
                failure = _failure("model_comparator_discovery", arm=name,
                                   status="failed", error=_exception(exc))
                comparator_qualification[name] = failure
                comparator_failures.append(failure)
    fla_eligible = variant in {"standard", "sliced"} and mode in {"full", "block"}
    # The explicit architectural path constructs the standard R=D arm below.
    # For sliced R=D that arm has the same equation as the ordinary FLA arm;
    # avoid allocating and timing the duplicate when the opt-in is enabled.
    same_rank_fla = fla_eligible and not standard_fla
    source_layout_staging_ran = False
    for rank in ranks:
        rank_data = dict(
            model_data,
            rank=rank,
            variant=variant,
            source_layout=source_layout,
        )
        catswe_arm = f"catswe_phase1_model_rank_{rank}"
        catswe_eligibility = None
        if include_catswe_model:
            # Gate every actual Full/Block read before constructing any model
            # or importing the optional native phase-1 operation.  An
            # ineligible cell remains an explicit model NA row and cannot
            # fall back to the operator comparator.
            catswe_eligibility = _catswe_model_eligibility(model_data, rank)
            model_comparator_scope[catswe_arm] = catswe_eligibility
            if not catswe_eligibility.get("eligible", False):
                comparator_qualification[catswe_arm] = dict(catswe_eligibility)
        packed_name = f"packed_rank_{rank}"
        packed_data = dict(rank_data, source_layout="packed")
        if include_packed_comparison:
            source_layout_comparisons[f"kernel_rank_{rank}_over_packed_rank_{rank}"] = _source_layout_metadata(
                rank_data,
                packed_data,
                variant=variant,
                mode=mode,
                rank=rank,
            )
        reference, kernel, candidate = None, None, None
        packed_reference, packed_kernel = None, None
        packed_staged_models = []
        try:
            rank_config = TrainingConfig(**rank_data)
            reference = _make_model(rank_config, bf16_torch)
            _record_model(f"reference_rank_{rank}", reference, rank, variant)
            kernel = _make_model(rank_config, "kernel")
            _copy_model_state(reference, kernel)
            _record_model(f"kernel_rank_{rank}", kernel, rank, variant)
            qualification[f"rank_{rank}"] = _qualify_model(
                f"kernel_rank_{rank}", "core", reference, kernel)
            if include_reference:
                arms[f"reference_rank_{rank}"] = {"model": reference, "rank": rank, "backend": "reference"}
            kernel_arm = {"model": kernel, "rank": rank, "backend": "kernel"}
            if include_packed_comparison:
                kernel_arm["source_layout"] = "list"
            arms[f"kernel_rank_{rank}"] = kernel_arm
            if include_packed_comparison:
                try:
                    staged_ids = set()
                    for model_name, model in (("reference", reference), ("kernel", kernel)):
                        if model is None or id(model) in staged_ids:
                            continue
                        model.to("cpu")
                        packed_staged_models.append((model_name, model))
                        staged_ids.add(id(model))
                        source_layout_staging_ran = True
                    try:
                        _release_qualification_memory(device)
                    except Exception as cleanup_exc:
                        comparator_failures.append(_failure(
                            "model_comparator_cleanup",
                            arm=packed_name,
                            context="source_layout",
                            status="failed",
                            error=_exception(cleanup_exc),
                        ))
                    packed_config = TrainingConfig(**packed_data)
                    packed_reference = _make_model(packed_config, bf16_torch)
                    _copy_model_state(reference, packed_reference)
                    _record_model(
                        f"packed_reference_rank_{rank}",
                        packed_reference,
                        rank,
                        variant,
                    )
                    packed_kernel = _make_model(packed_config, "kernel")
                    _copy_model_state(packed_reference, packed_kernel)
                    _record_model(packed_name, packed_kernel, rank, variant)
                    comparator_qualification[packed_name] = _qualify_model(
                        packed_name, "comparator", packed_reference, packed_kernel)
                    arms[packed_name] = {
                        "model": packed_kernel,
                        "rank": rank,
                        "backend": "kernel",
                        "source_layout": "packed",
                        "comparison": "source_layout",
                    }
                    packed_kernel = None
                except Exception as exc:
                    info = _failure("model_comparator_qualification", arm=packed_name,
                                    status="failed", error=_exception(exc))
                    comparator_qualification[packed_name] = info
                    comparator_failures.append(info)
                finally:
                    packed_reference = None
                    packed_kernel = None
                    try:
                        _release_qualification_memory(device)
                    except Exception as cleanup_exc:
                        comparator_failures.append(_failure(
                            "model_comparator_cleanup",
                            arm=packed_name,
                            context="source_layout_after_qualification",
                            status="failed",
                            error=_exception(cleanup_exc),
                        ))
                    for model_name, model in packed_staged_models:
                        try:
                            model.to(device)
                        except Exception as restore_exc:
                            failures.append(_failure(
                                "model_qualification_restore",
                                model=model_name,
                                source_layout="list",
                                error=_exception(restore_exc),
                            ))
                    packed_staged_models.clear()
                    model_name, model = None, None
            if baseline is not None:
                name = f"frozen_baseline_rank_{rank}"
                candidate = _make_model(rank_config, baseline)
                _copy_model_state(reference, candidate)
                _record_model(name, candidate, rank, variant)
                comparator_qualification[name] = _qualify_model(
                    name, "comparator", reference, candidate)
                arms[name] = {"model": candidate, "rank": rank, "backend": "frozen_baseline"}
            if same_rank_fla and rank == int(model_data["width"]):
                for name, backend in compile_backends.items():
                    arm_name = f"{name}_rank_{rank}"
                    try:
                        candidate = _make_model(rank_config, backend)
                        _copy_model_state(reference, candidate)
                        _record_model(arm_name, candidate, rank, variant)
                        comparator_qualification[arm_name] = _qualify_model(
                            arm_name, "comparator", reference, candidate)
                        arms[arm_name] = {"model": candidate, "rank": rank, "backend": name}
                    except Exception as exc:
                        info = _failure("model_comparator_qualification", arm=arm_name, status="failed", error=_exception(exc))
                        comparator_qualification[arm_name] = info
                        comparator_failures.append(info)
                    finally:
                        candidate = None
            if include_fla_model and same_rank_fla and rank == int(model_data["width"]):
                from .competitors import model_backend
                for name, comparator in comparators.items():
                    arm_name = f"{name}_rank_{rank}"
                    if not comparator.available:
                        comparator_qualification[arm_name] = {"status": comparator.status, "reason": comparator.reason, "matched_rank_only": True}
                        comparator_failures.append(_failure("model_comparator_discovery", arm=arm_name, status=comparator.status, reason=comparator.reason))
                        continue
                    try:
                        candidate = _make_model(rank_config, model_backend(comparator))
                        _copy_model_state(reference, candidate)
                        _record_model(arm_name, candidate, rank, variant)
                        comparator_qualification[arm_name] = _qualify_model(
                            arm_name, "comparator", reference, candidate)
                        arms[arm_name] = {"model": candidate, "rank": rank, "backend": name}
                    except Exception as exc:
                        info = _failure("model_comparator_qualification", arm=arm_name, status="failed", error=_exception(exc))
                        comparator_qualification[arm_name] = {k: v for k, v in info.items() if k not in {"phase", "arm"}}
                        comparator_failures.append(info)
                    finally:
                        candidate = None
            elif include_fla_model:
                reason = (
                    "standard_fla_comparison uses its separately qualified standard R=D arm"
                    if standard_fla else "FLA model arms require Full/Block-mode implicit R=D"
                )
                for name in comparators: comparator_qualification[f"{name}_rank_{rank}"] = {"status": "not_applicable", "reason": reason}

            # Liger is a native per-read comparator, so it may participate in
            # the complete compiled training step only when *every* read made
            # by this exact model is inside its S<=32/R=D/BF16 envelope.  Do
            # this gate before constructing, compiling, or allocating a
            # comparator model.  A failed read is an explicit NA row rather
            # than a silently truncated model comparison.
            if include_liger_model:
                liger_arm = f"liger_rank_{rank}"
                eligibility = _liger_model_eligibility(model_data, rank)
                model_comparator_scope[liger_arm] = eligibility
                if not eligibility.get("eligible", False):
                    comparator_qualification[liger_arm] = dict(eligibility)
                else:
                    comparator = model_comparators.get("liger")
                    if comparator is None or getattr(comparator, "available", False) is not True:
                        status_value = getattr(comparator, "status", "missing")
                        reason_value = getattr(
                            comparator,
                            "reason",
                            "pinned Liger comparator was not discovered",
                        )
                        info = _failure(
                            "model_comparator_discovery",
                            arm=liger_arm,
                            status=str(status_value),
                            reason=str(reason_value),
                        )
                        comparator_qualification[liger_arm] = {
                            "competitor": "liger",
                            "status": str(status_value),
                            "eligible": True,
                            "eligible_denominator": True,
                            "reason": str(reason_value),
                            "eligibility": eligibility,
                        }
                        comparator_failures.append(info)
                    else:
                        try:
                            from .liger import model_backend as liger_model_backend

                            liger_backend = liger_model_backend(comparator)
                            candidate = _make_model(rank_config, liger_backend)
                            _copy_model_state(reference, candidate)
                            _record_model(liger_arm, candidate, rank, variant)
                            qualification_report = _qualify_model(
                                liger_arm,
                                "comparator",
                                reference,
                                candidate,
                            )
                            comparator_qualification[liger_arm] = {
                                "eligibility": eligibility,
                                **qualification_report,
                            }
                            arms[liger_arm] = {
                                "model": candidate,
                                "rank": rank,
                                "backend": "liger",
                                "source_layout": "list",
                                "comparison": "external_model_comparator",
                            }
                        except Exception as exc:
                            info = _failure(
                                "model_comparator_qualification",
                                arm=liger_arm,
                                status="failed",
                                error=_exception(exc),
                            )
                            comparator_qualification[liger_arm] = {
                                "eligibility": eligibility,
                                "status": "failed",
                                "error": info["error"],
                            }
                            comparator_failures.append(info)
                        finally:
                            candidate = None

            # The explicit model scope adapts only the public phase-1
            # primitive.  No cached Block, prepare, merge, or phase-2 route is
            # available here; stack/contiguous staging stays in the captured
            # adapter call.
            if (
                include_catswe_model
                and catswe_eligibility is not None
                and catswe_eligibility.get("eligible", False)
            ):
                comparator = model_comparators.get("catswe_phase1")
                if comparator is None or getattr(comparator, "available", False) is not True:
                    status_value = getattr(comparator, "status", "missing")
                    reason_value = getattr(
                        comparator,
                        "reason",
                        "pinned Catswe comparator was not discovered",
                    )
                    info = _failure(
                        "model_comparator_discovery",
                        arm=catswe_arm,
                        status=str(status_value),
                        reason=str(reason_value),
                    )
                    comparator_qualification[catswe_arm] = {
                        "competitor": "catswe_phase1",
                        "status": str(status_value),
                        "eligible": True,
                        "eligible_denominator": True,
                        "reason": str(reason_value),
                        "eligibility": catswe_eligibility,
                    }
                    comparator_failures.append(info)
                else:
                    try:
                        from .catswe import make_model_backend as catswe_model_backend

                        catswe_backend = catswe_model_backend(comparator)
                        model_backend_metadata["catswe_phase1"] = dict(
                            getattr(catswe_backend, "source_hash_metadata", {})
                        )
                        candidate = _make_model(rank_config, catswe_backend)
                        _copy_model_state(reference, candidate)
                        _record_model(catswe_arm, candidate, rank, variant)
                        qualification_report = _qualify_model(
                            catswe_arm,
                            "comparator",
                            reference,
                            candidate,
                        )
                        comparator_qualification[catswe_arm] = {
                            "eligibility": catswe_eligibility,
                            **qualification_report,
                        }
                        arms[catswe_arm] = {
                            "model": candidate,
                            "rank": rank,
                            "backend": "catswe_phase1",
                            "source_layout": "list",
                            "comparison": "external_model_comparator",
                        }
                    except Exception as exc:
                        info = _failure(
                            "model_comparator_qualification",
                            arm=catswe_arm,
                            status="failed",
                            error=_exception(exc),
                        )
                        comparator_qualification[catswe_arm] = {
                            "eligibility": catswe_eligibility,
                            "status": "failed",
                            "error": info["error"],
                        }
                        comparator_failures.append(info)
                    finally:
                        candidate = None
        except Exception as exc: failures.append(_failure("model_qualification", rank=rank, status="failed", error=_exception(exc)))
        finally:
            # Successful arms own their models; failed locals must not extend
            # GPU lifetimes into another architecture's qualification.
            reference, kernel, candidate = None, None, None
            packed_reference, packed_kernel = None, None
    architecture_comparisons, schedule_comparisons = {}, {}
    qualification_staging_ran = False
    if standard_fla and compile_backends and not failures:
        standard_data = dict(model_data, rank=int(model_data["width"]), variant="standard", mode=mode)
        candidate_configs = [dict(model_data, rank=rank, variant=variant, mode=mode) for rank in ranks]
        full_input_label = (
            "ordered source-list input" if source_layout == "list"
            else "full source stack"
        )
        schedules = {
            "candidate_kernel": _kernel_schedule(mode) if mode == "block" else f"per-read {full_input_label}",
            "standard_fla": "fused read of completed+partial sources; no cache" if mode == "block" else f"fused read of {full_input_label}",
            "mode": mode,
        }
        comparison_kind_by_rank = {
            str(rank): (
                "same_equation_different_execution"
                if variant == "sliced" and int(rank) == int(model_data["width"])
                else "architectural_lr_vs_standard"
            )
            for rank in ranks
        }
        standard_reference, candidate = None, None
        staged_models = []
        try:
            # These arms have no optimizer or captured graph yet. Keep their
            # exact state on CPU while the untimed standard reference runs.
            for arm in arms.values():
                staged_models.append(arm["model"])
                arm["model"].to("cpu")
                # Record staging only after an actual model has moved.  This
                # remains true when a later move or setup operation fails.
                qualification_staging_ran = True
            try:
                _release_qualification_memory(device)
            except Exception as cleanup_exc:
                comparator_failures.append(_failure(
                    "model_comparator_cleanup",
                    arm=f"standard_reference_rank_{standard_data['rank']}",
                    context="reference",
                    status="failed",
                    error=_exception(cleanup_exc),
                ))
            standard_config = TrainingConfig(**standard_data)
            standard_reference = _make_model(standard_config, bf16_torch)
            _record_model(
                f"standard_reference_rank_{standard_data['rank']}",
                standard_reference,
                int(standard_data["rank"]),
                "standard",
            )
            for name, backend in compile_backends.items():
                arm_name = f"{name}_standard_rank_{standard_data['rank']}"
                try:
                    candidate = _make_model(standard_config, backend)
                    _copy_model_state(standard_reference, candidate)
                    _record_model(arm_name, candidate, int(standard_data["rank"]), "standard")
                    comparator_qualification[arm_name] = _qualify_model(
                        arm_name, "comparator", standard_reference, candidate)
                    candidate.to("cpu")
                    staged_models.append(candidate)
                    arms[arm_name] = {"model": candidate, "rank": standard_data["rank"], "backend": name}
                    architecture_metadata = {
                        "candidate_configs": candidate_configs,
                        "standard_config": standard_data,
                        "candidate_variant": variant,
                        "standard_variant": "standard",
                        "role": "sliced LR candidate versus standard R=D AttnRes",
                        "qualification": "each architecture against its own equation reference",
                        "schedules": schedules,
                        "comparison_kind_by_rank": comparison_kind_by_rank,
                    }
                    architecture_comparisons[arm_name] = architecture_metadata
                except Exception as exc:
                    info = _failure("model_comparator_qualification", arm=arm_name,
                                    status="failed", error=_exception(exc))
                    comparator_qualification[arm_name] = info
                    comparator_failures.append(info)
                finally:
                    candidate = None
                    try:
                        _release_qualification_memory(device)
                    except Exception as cleanup_exc:
                        comparator_failures.append(_failure(
                            "model_comparator_cleanup", arm=arm_name, status="failed",
                            error=_exception(cleanup_exc)))
        except Exception as exc:
            comparator_failures.append(_failure("model_comparator_reference", error=_exception(exc)))
        finally:
            del standard_reference
            try:
                _release_qualification_memory(device)
            except Exception as exc:
                failures.append(_failure("model_qualification_cleanup", error=_exception(exc)))
            for index, model in enumerate(staged_models):
                try:
                    model.to(device)
                except Exception as exc:
                    failures.append(_failure("model_qualification_restore", model_index=index, error=_exception(exc)))
    if include_packed_comparison:
        schedule_comparisons.update(source_layout_comparisons)
    if failures: return _model_failure_result(model_data, variant, ranks, qualification=qualification, comparator_qualification=comparator_qualification, comparator_failures=comparator_failures, **({"state_protocol": state_protocol_report} if state_protocol_report is not None else {}), compiled_loss={"status": "ok", "fullgraph": True, "dynamic": False}, reference_timing=include_reference, include_fla_model=include_fla_model, pairwise=pairwise, failures=failures)
    compile_rows, optimizer_rows, compiled_models, optimizers, active = {}, {}, {}, {}, []
    for name, arm in arms.items():
        try:
            started = time.perf_counter()
            progress("compile_start", name)
            try:
                compiled_models[name] = torch.compile(arm["model"], fullgraph=True, dynamic=False)
            finally:
                progress("compile_end", name)
            torch.cuda.synchronize(device)
            compile_rows[name] = {"status": "ok", "host_ms": (time.perf_counter() - started) * 1000.0, "fullgraph": True, "dynamic": False}
            started = time.perf_counter()
            optimizer, implementation = _adamw(compiled_models[name].parameters(), config, cuda_graph=model_timing == "cuda_graph")
            optimizers[name] = optimizer
            optimizer_rows[name] = {"status": "ok", "implementation": implementation,
                                    "host_ms": (time.perf_counter() - started) * 1000,
                                    "state_initialized_during_warmup": True}
            active.append(name)
        except Exception as exc:
            row = _failure("model_compile", arm=name, status="failed", error=_exception(exc), fullgraph=True, dynamic=False)
            compile_rows[name] = {k: v for k, v in row.items() if k not in {"phase", "arm"}}
            optimizer_rows[name] = {"status": "failed"}
            (failures if _is_core_model_arm(arm) else comparator_failures).append(row)
    if failures: return _model_failure_result(model_data, variant, ranks, qualification=qualification, comparator_qualification=comparator_qualification, comparator_failures=comparator_failures, **({"state_protocol": state_protocol_report} if state_protocol_report is not None else {}), compile=compile_rows, optimizer=optimizer_rows, compiled_loss={"status": "ok", "fullgraph": True, "dynamic": False}, reference_timing=include_reference, include_fla_model=include_fla_model, pairwise=pairwise, failures=failures)
    rng = random.Random(seed + 771)
    warmup_inputs = [_model_inputs(model_data, device, seed + 9000 + i) for i in range(warmup)]
    timed_inputs = [_model_inputs(model_data, device, seed + 12000 + i) for i in range(rounds)]
    input_hashes = [
        _logical_model_input_id(seed, index, model_data)
        for index in range(rounds)
    ]
    warmup_rows, failed_arms, graph_counters = [], set(), {}
    warmup_order = list(active)
    rng.shuffle(warmup_order)
    for name in warmup_order:
        progress("warmup_start", name)
        try:
            before = _dynamo_counters()
            for index, (warmup_tokens, warmup_targets) in enumerate(warmup_inputs):
                try:
                    started = time.perf_counter()
                    output = _compiled_training_step(compiled_models[name], optimizers[name], compiled_loss, warmup_tokens, warmup_targets, accumulation)
                    torch.cuda.synchronize(device)
                    _finite(output, f"{name} warmup result")
                    if index == 0: _check_model_gradients(compiled_models[name], name)
                    warmup_rows.append({"arm": name, "index": index, "status": "ok", "host_ms": (time.perf_counter() - started) * 1000.0})
                except Exception as exc:
                    failed_arms.add(name)
                    row = {"arm": name, "index": index, "status": "failed", "error": _exception(exc)}
                    warmup_rows.append(row)
                    (failures if _is_core_model_arm(arms[name]) else comparator_failures).append(_failure("model_warmup", **row))
                    break
        finally:
            progress("warmup_end", name)
        after = _dynamo_counters()
        delta = _counter_delta(before, after)
        graph_counters[name] = {"before": before, "after_warmup": after, "delta": delta, "graph_breaks": _graph_breaks(delta), "recompiles": _recompiles(delta), "new_unique_graphs": _unique_graphs(delta)}
        if graph_counters[name]["graph_breaks"] or graph_counters[name]["recompiles"]:
            failed_arms.add(name)
            row = _failure("model_compile", arm=name, status="failed", reason="graph break or recompilation observed after warmup", counters=graph_counters[name])
            (failures if _is_core_model_arm(arms[name]) else comparator_failures).append(row)
    # Timing is a privilege of an arm that has passed an untimed complete
    # optimizer step.  Warmup only establishes compiler/optimizer state; it is
    # not a semantic oracle.  Keep this gate after warmup so its checkpoint can
    # be restored without invalidating a captured optimizer allocation.
    complete_step_qualification = {
        name: {"status": "not_run", "reason": "arm failed before complete-step qualification"}
        for name in active
    }
    real_cuda = device.type == "cuda" and bool(torch.cuda.is_available())
    if real_cuda:
        qualification_tokens, qualification_targets = timed_inputs[0]

        def _reference_factory(candidate_config: Any, _device: torch.device) -> Any:
            if candidate_config is None:
                raise RuntimeError("candidate model does not expose its TrainingConfig")
            return _make_model(candidate_config, bf16_torch, target_device=_device)

        for name in active:
            if name in failed_arms:
                continue
            progress("complete_step_qualification_start", name)
            try:
                report = _complete_step_qualification(
                    candidate_model=arms[name]["model"],
                    candidate_optimizer=optimizers[name],
                    candidate_step=(
                        lambda step_tokens, step_targets, arm_name=name:
                        _compiled_training_step(
                            compiled_models[arm_name],
                            optimizers[arm_name],
                            compiled_loss,
                            step_tokens,
                            step_targets,
                            accumulation,
                        )
                    ),
                    reference_factory=_reference_factory,
                    optimizer_config=config,
                    tokens=qualification_tokens,
                    targets=qualification_targets,
                    accumulation=accumulation,
                    protocol=protocol,
                    device=device,
                    cuda_graph=model_timing == "cuda_graph",
                    label=f"{name} complete step",
                )
                complete_step_qualification[name] = {
                    "status": "qualified",
                    "compiled_step": report,
                }
            except Exception as exc:
                failed_arms.add(name)
                row = _failure(
                    "model_complete_step_qualification",
                    arm=name,
                    status="failed",
                    error=_exception(exc),
                )
                complete_step_qualification[name] = {
                    "status": "failed",
                    "error": row["error"],
                }
                (failures if _is_core_model_arm(arms[name]) else comparator_failures).append(row)
            finally:
                progress("complete_step_qualification_end", name)
    else:
        for name in active:
            if name not in failed_arms:
                complete_step_qualification[name] = {
                    "status": "skipped",
                    "reason": "CUDA unavailable for complete-step qualification",
                }
    graph_reference_evidence: dict[str, Sequence[Mapping[str, Any]]] = {}
    if real_cuda and model_timing == "cuda_graph" and not failures:
        # The changed-input oracle is deliberately completed before capture.
        # Its CPU evidence is then consumed by candidate-only graph replays,
        # keeping the reference out of the capture and timed memory boundary.
        for name in active:
            if name in failed_arms:
                continue
            reference = None
            try:
                reference = _reference_factory(
                    getattr(arms[name]["model"], "config", None),
                    torch.device("cpu"),
                )
                graph_reference_evidence[name] = _precompute_graph_reference_evidence(
                    candidate_model=arms[name]["model"],
                    candidate_optimizer=optimizers[name],
                    reference=reference,
                    optimizer_config=config,
                    tokens=qualification_tokens,
                    targets=qualification_targets,
                    accumulation=accumulation,
                    vocab=int(model_data["vocab"]),
                    device=device,
                )
                complete_step_qualification[name]["graph_reference_precompute"] = {
                    "status": "qualified",
                    "replay_count": len(graph_reference_evidence[name]),
                    "evidence_device": "cpu",
                }
            except Exception as exc:
                failed_arms.add(name)
                row = _failure(
                    "model_graph_reference_precompute",
                    arm=name,
                    status="failed",
                    error=_exception(exc),
                )
                complete_step_qualification[name]["graph_reference_precompute"] = {
                    "status": "failed",
                    "error": row["error"],
                }
                (failures if _is_core_model_arm(arms[name]) else comparator_failures).append(row)
            finally:
                del reference
    # A core arm that has not passed the complete-step gate must never reach
    # CUDA Graph capture or timed rounds. Comparator failures remain explicit
    # and are excluded from timing, matching the runner's existing optional
    # comparator policy.
    if failures:
        # Capture failures can be raised on the host before an asynchronous
        # device fault reaches Python.  Drain the device before returning so
        # the retained diagnostics include both failure sources and no latent
        # CUDA error leaks into a later phase.
        try:
            torch.cuda.synchronize(device)
        except Exception as exc:
            failures.append(
                _failure("model_graph_sync", status="failed", error=_exception(exc))
            )
        return _model_failure_result(
            model_data,
            variant,
            ranks,
            qualification=qualification,
            comparator_qualification=comparator_qualification,
            comparator_failures=comparator_failures,
            **(
                {"state_protocol": state_protocol_report}
                if state_protocol_report is not None
                else {}
            ),
            compile=compile_rows,
            compiled_loss={"status": "ok", "fullgraph": True, "dynamic": False},
            optimizer=optimizer_rows,
            complete_step_qualification=complete_step_qualification,
            pre_timing_gate=complete_step_qualification,
            graph={"status": "not_run", "reason": "complete-step qualification failed"},
            graph_counters=graph_counters,
            warmup=warmup_rows,
            failures=failures,
        )
    graph_steps, graph_reports = {}, {}
    if model_timing == "cuda_graph":
        from .training_graph import capture_training_step
        for name in active:
            if name in failed_arms:
                continue
            before = _dynamo_counters()
            capture_model_state = None
            capture_optimizer_state = None
            if real_cuda:
                capture_model_state = _clone_model_checkpoint(arms[name]["model"])
                capture_optimizer_state = _named_optimizer_state(
                    arms[name]["model"], optimizers[name]
                )
            try:
                started = time.perf_counter()
                progress("cuda_graph_capture_start", name)
                try:
                    graph_steps[name] = capture_training_step(
                        arms[name]["model"], optimizers[name], *warmup_inputs[-1],
                        accumulation=accumulation,
                    )
                finally:
                    progress("cuda_graph_capture_end", name)
                if real_cuda:
                    _compare_state_values(
                        arms[name]["model"].state_dict(),
                        capture_model_state,
                        {"rtol": 0.0, "atol": 0.0},
                        label=f"{name} capture model state",
                        exact=True,
                    )
                    if not _value_equal(
                        _named_optimizer_state(arms[name]["model"], optimizers[name]),
                        capture_optimizer_state,
                    ):
                        raise RuntimeError(
                            f"{name} CUDA Graph capture did not restore optimizer state"
                        )
                graph_reports[name] = {
                    "status": "ok", "host_ms": (time.perf_counter() - started) * 1000,
                    "state_restored_before_replay": True,
                    "state_restored_model_and_optimizer": True,
                    "side_stream_warmup": 2,
                    "complete_step": True, "counters": _counter_delta(before, _dynamo_counters()),
                }
                delta = graph_reports[name]["counters"]
                graph_reports[name]["stable_capture"] = not (_graph_breaks(delta) or _recompiles(delta))
                if not graph_reports[name]["stable_capture"]:
                    raise RuntimeError("graph break or recompilation during complete-step capture")
            except Exception as exc:
                failed_arms.add(name)
                row = _failure("model_graph_capture", arm=name, status="failed", error=_exception(exc))
                graph_reports[name] = row
                (failures if _is_core_model_arm(arms[name]) else comparator_failures).append(row)
    if model_timing == "cuda_graph" and real_cuda:
        qualification_tokens, qualification_targets = timed_inputs[0]
        for name in active:
            if name in failed_arms or name not in graph_steps:
                continue
            progress("cuda_graph_qualification_start", name)
            try:
                replay_report = _graph_replay_qualification(
                    candidate_model=arms[name]["model"],
                    candidate_optimizer=optimizers[name],
                    graph_step=graph_steps[name],
                    reference_factory=_reference_factory,
                    optimizer_config=config,
                    tokens=qualification_tokens,
                    targets=qualification_targets,
                    accumulation=accumulation,
                    vocab=int(model_data["vocab"]),
                    protocol=protocol,
                    device=device,
                    capture_inputs=warmup_inputs[-1],
                    reference_evidence=graph_reference_evidence.get(name),
                )
                complete_step_qualification[name]["graph_replay"] = replay_report
                graph_reports[name]["changed_input_replays"] = replay_report
            except Exception as exc:
                failed_arms.add(name)
                row = _failure(
                    "model_graph_qualification",
                    arm=name,
                    status="failed",
                    error=_exception(exc),
                )
                complete_step_qualification[name]["graph_replay"] = {
                    "status": "failed",
                    "error": row["error"],
                }
                graph_reports[name] = {
                    **graph_reports.get(name, {}),
                    "status": "failed",
                    "complete_step": False,
                    "error": row["error"],
                }
                (failures if _is_core_model_arm(arms[name]) else comparator_failures).append(row)
            finally:
                progress("cuda_graph_qualification_end", name)
    if failures:
        # A capture or replay-qualification failure can be raised on the host
        # before an asynchronous device fault reaches Python. Drain the device
        # so both failures are retained and no latent error reaches timing.
        try:
            torch.cuda.synchronize(device)
        except Exception as exc:
            failures.append(
                _failure("model_graph_sync", status="failed", error=_exception(exc))
            )
        return _model_failure_result(
            model_data,
            variant,
            ranks,
            qualification=qualification,
            comparator_qualification=comparator_qualification,
            comparator_failures=comparator_failures,
            **(
                {"state_protocol": state_protocol_report}
                if state_protocol_report is not None
                else {}
            ),
            compile=compile_rows,
            compiled_loss={"status": "ok", "fullgraph": True, "dynamic": False},
            optimizer=optimizer_rows,
            complete_step_qualification=complete_step_qualification,
            pre_timing_gate=complete_step_qualification,
            graph=graph_reports,
            graph_counters=graph_counters,
            warmup=warmup_rows,
            failures=failures,
        )
    try:
        torch.cuda.synchronize(device)
    except Exception as exc:
        # The post-capture synchronize can report an asynchronous device fault
        # without identifying which captured arm caused it.
        sync_failure = _failure(
            "model_graph_sync", status="failed", error=_exception(exc)
        )
        return _model_failure_result(
            model_data,
            variant,
            ranks,
            qualification=qualification,
            comparator_qualification=comparator_qualification,
            comparator_failures=comparator_failures,
            **(
                {"state_protocol": state_protocol_report}
                if state_protocol_report is not None
                else {}
            ),
            compile=compile_rows,
            compiled_loss={"status": "ok", "fullgraph": True, "dynamic": False},
            optimizer=optimizer_rows,
            complete_step_qualification=complete_step_qualification,
            pre_timing_gate=complete_step_qualification,
            graph=graph_reports,
            graph_counters=graph_counters,
            warmup=warmup_rows,
            failures=[*failures, sync_failure],
        )
    timed_before = _dynamo_counters()
    def row_factory(name: str, sample: int, order: int | None) -> dict[str, Any]:
        row = {"arm": name, "rank": arms[name]["rank"], "backend": arms[name]["backend"], "sample_index": sample, "order_index": order, "input_hash": input_hashes[sample], "ms": None}
        if include_packed_comparison:
            layout = arms[name].get("source_layout")
            if name.startswith("reference_rank_"):
                layout = "reference_stack"
            if layout is not None:
                row["source_layout"] = layout
        return row
    def measure(name: str, sample: int) -> Mapping[str, Any]:
        sample_tokens, sample_targets = timed_inputs[sample]
        if model_timing == "cuda_graph":
            with torch.cuda.device(device):
                graph_steps[name].copy_inputs(sample_tokens, sample_targets)
            elapsed, _ = _cuda_event_call(graph_steps[name].replay, device)
        else:
            elapsed, _ = _cuda_event_call(lambda: _compiled_training_step(compiled_models[name], optimizers[name], compiled_loss, sample_tokens, sample_targets, accumulation), device)
        return {"status": "ok", "ms": elapsed, "timing_method": model_timing, "replay_count": 1}
    def sink(name: str, row: Mapping[str, Any]) -> list[dict[str, Any]]: return failures if _is_core_model_arm(arms[name]) else comparator_failures
    progress("paired_timing_start", "paired")
    try:
        raw = _paired_samples(active, active, rounds, rng, failed_arms, row_factory, measure, sink, "model_timing")
    finally:
        progress("paired_timing_end", "paired")
    timed_after = _dynamo_counters()
    timed_delta = _counter_delta(timed_before, timed_after)
    timed_stability = {"before": timed_before, "after": timed_after, "delta": timed_delta, "graph_breaks": _graph_breaks(timed_delta), "recompiles": _recompiles(timed_delta), "new_unique_graphs": _unique_graphs(timed_delta)}
    timed_stability["stable"] = not any(timed_stability[k] for k in ("graph_breaks", "recompiles", "new_unique_graphs"))
    if not timed_stability["stable"]: failures.append(_failure("model_timing", status="failed", reason="Dynamo graph activity changed during timed rounds", counters=timed_stability))
    if len(set(input_hashes)) < 2: failures.append(_failure("model_inputs", status="failed", reason="timed inputs did not change"))
    comparisons = _model_comparisons(raw, arms, ranks, rounds, include_reference, architecture_comparisons)
    statistics = simultaneous_paired_ratio_bootstrap(comparisons, samples=int(config.get("bootstrap_samples", protocol["bootstrap_samples"])), seed=seed + 18000, margin=float(protocol["plateau_margin"])) if comparisons else {}
    status = "failed" if failures else ("incomplete" if comparator_failures or not comparisons or len(statistics) != len(comparisons) else "complete")
    if bool(config.get("model_profile", False)):
        try:
            model_profile = _model_profile_report(
                active,
                failed_arms,
                raw,
                rounds,
                graph_steps,
                model_timing,
                device,
                timed_stability,
            )
        except Exception as exc:
            model_profile = {
                "enabled": True,
                "status": "failed",
                "requested": True,
                "method": "torch.profiler.profile",
                "limitations": list(_MODEL_PROFILE_LIMITATIONS),
                "arms": {},
                "profiled_arms": [],
                "failures": [_failure("model_profile", status="failed", error=_exception(exc))],
            }
    else:
        model_profile = {
            "enabled": False,
            "status": "disabled",
            "requested": False,
            "reason": "model_profile was not requested",
            "method": "torch.profiler.profile",
            "limitations": list(_MODEL_PROFILE_LIMITATIONS),
            "arms": {},
            "profiled_arms": [],
            "failures": [],
        }
    timing_boundary = {
        "steady_step_includes": [
            "BF16 autocast",
            "zero_grad",
            "model forward",
            "cross_entropy loss",
            "backward",
            "gradient accumulation",
            "AdamW optimizer.step",
        ],
        "excluded": ["torch.compile", "optimizer construction", "warmup"],
        "loss_owner": "benchmarks.run._cross_entropy_loss",
        "backward_orchestration": "eager loss.backward over compiled model/loss AOT graphs",
        "optimizer_construction": "before warmup; state initialized during warmup",
    }
    if model_timing == "cuda_graph":
        timing_boundary.update(
            excluded=["input copies", "torch.compile", "optimizer construction", "warmup", "graph capture"],
            loss_owner="benchmarks.training_graph._cross_entropy",
            backward_orchestration="captured complete step including optimizer update",
        )
    model_comparator_metadata: dict[str, Any] = {}
    for name, comparator in model_comparators.items():
        metadata = getattr(comparator, "metadata", None)
        if not isinstance(metadata, Mapping):
            describe = getattr(comparator, "describe", None)
            metadata = describe() if callable(describe) else {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        entry = dict(metadata)
        if name in model_backend_metadata:
            # Keep the standard operator descriptor and the explicit model
            # adapter descriptor distinguishable in the report.
            entry["model_backend"] = model_backend_metadata[name]
        model_comparator_metadata[name] = entry
    report = {
        "status": status,
        "config": model_data,
        "effective_variant": variant,
        "ranks": ranks,
        "qualification": qualification,
        "comparator_qualification": comparator_qualification,
        "comparator_failures": comparator_failures,
        **({"state_protocol": state_protocol_report} if state_protocol_report is not None else {}),
        "compile_backend_metadata": compile_backend_metadata,
        "architecture_comparisons": architecture_comparisons,
        "qualification_staging": (
            "CPU between qualifications; restored before compile/optimizer"
            if qualification_staging_ran
            else "CPU during source-layout qualification; restored before compile/optimizer"
            if source_layout_staging_ran
            else "none"
        ),
        "execution_schedules": {
            "kernel": _kernel_schedule(mode),
            "fla": "fused per-read aggregation",
        },
        **({"schedule_comparisons": schedule_comparisons} if include_packed_comparison else {}),
        "compile": compile_rows,
        "compiled_loss": {
            "status": "ok",
            "fullgraph": True,
            "dynamic": False,
            "function": "torch.nn.functional.cross_entropy",
        },
        "optimizer": optimizer_rows,
        "complete_step_qualification": complete_step_qualification,
        "pre_timing_gate": complete_step_qualification,
        "training_step": ("benchmarks.training_graph.CapturedTrainingStep.replay" if model_timing == "cuda_graph" else "benchmarks.run._compiled_training_step"),
        "timing_method": model_timing,
        "frozen_baseline": baseline.metadata if baseline is not None else None,
        "graph": graph_reports,
        "canonical_training_step": "benchmarks.model.training_step (validated, not timed)",
        "reference_timing": include_reference,
        **({"include_packed_comparison": True} if include_packed_comparison else {}),
        "include_fla_model": include_fla_model,
        "pairwise": pairwise,
        "timing_boundary": timing_boundary,
        "accumulation": accumulation,
        "warmup": warmup_rows,
        "requested_warmup": requested_warmup,
        "effective_warmup": warmup,
        "requested_rounds": rounds,
        "graph_counters": graph_counters,
        "timed_graph_counters": timed_stability,
        "changed_inputs": len(set(input_hashes)) > 1,
        "timed_input_identity": {
            "kind": "logical_model_sample_v1",
            "tensor_byte_hashing": False,
            "device_to_host_copy": False,
            "shared_tensor_objects_across_arms": True,
        },
        "timed_numerical_checks": "pre_timing_complete_step_and_changed_input_graph_gate_only",
        "raw_samples": raw,
        "statistics": statistics,
        "model_profile": model_profile,
        "failures": failures,
    }
    if include_liger_model or include_catswe_model:
        execution_schedules = dict(report["execution_schedules"])
        if include_liger_model:
            execution_schedules["liger"] = (
                "native Liger per-read aggregation; source lists are stacked and "
                "made contiguous inside the adapter call"
            )
        if include_catswe_model:
            execution_schedules["catswe_phase1"] = (
                "native Catswe public phase-1 per-read aggregation for eligible "
                "Full/Block models; source lists are stacked and made contiguous "
                "inside the adapter call, with no cache/prepare/merge/phase2"
            )
        report["execution_schedules"] = execution_schedules
        report.update(
            {
                "include_liger_model": include_liger_model,
                "include_catswe_model": include_catswe_model,
                "model_comparator_scope": model_comparator_scope,
                "model_comparator_metadata": model_comparator_metadata,
            }
        )
    if model_admission is not None:
        report["model_only_admission"] = model_admission
    return report


def _phase_skip(status: str, reason: str) -> dict[str, Any]: return {"status": status, "reason": reason, "failures": []}


def _rollup(result: Mapping[str, Any]) -> str:
    statuses = [result.get("contract", {}).get("status", "failed")] + [result.get(name, {}).get("status", "not_run") for name in ("correctness", "operator_timings", "model_timings")]
    if result.get("comparators_enabled", True): statuses += [item.get("status", "incomplete") for item in result.get("comparators", {}).values()]
    return "failed" if "failed" in statuses else ("incomplete" if any(s in {"incomplete", "missing", "not_run"} for s in statuses) else "complete")


def run_suite(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or {})
    root = Path(config.get("project_root", PROJECT_ROOT)).resolve()
    scope = str(config.get("scope", config.get("suite", "smoke")))
    result = {
        "status": "failed",
        "config": config,
        "contract": {"status": "failed"},
        "coverage": {},
        "environment": {},
        "device": {"requested": str(config.get("device", "cuda")), "type": "cuda", "available": False, "count": 0},
        "fla_checkout": {"status": "not_required"},
        "comparators": {},
        "correctness": _phase_skip("not_run", "not started"),
        "operator_timings": _phase_skip("not_run", "not started"),
        "model_timings": _phase_skip("not_run", "not started"),
        "failures": [],
    }
    if scope not in {"smoke", "primary", "heldout", "custom"}:
        result["coverage"] = {"scope": scope, "claims_full_suite": False}
        result["failures"].append(_failure("config", error={"message": f"unsupported scope {scope!r}"}))
        result["status"] = "failed"
        return _jsonable(result)
    requested = config.get("phases", ("correctness", "operator", "model"))
    requested = (requested,) if isinstance(requested, str) else requested
    aliases = {"operator": "operator_timings", "operators": "operator_timings",
               "model": "model_timings", "training": "model_timings"}
    allowed = {"correctness", "operator_timings", "model_timings", *aliases}
    if not isinstance(requested, (list, tuple)) or any(
        not isinstance(phase, str) or phase not in allowed for phase in requested
    ):
        result["failures"].append(_failure("config", error={"message": f"unsupported phases {requested!r}"}))
        return _jsonable(result)
    phases = {aliases.get(phase, phase) for phase in requested}
    result["environment"] = _environment(root)
    try: protocol, frozen_hashes = load_protocol(root)
    except Exception as exc:
        result["contract"] = {"status": "failed", "error": _exception(exc)}
        result["failures"].append(_failure("contract", error=_exception(exc)))
        result["status"] = _rollup(result)
        return _jsonable(result)
    result["protocol"] = {"version": protocol.get("version"), "frozen_hashes": frozen_hashes}
    result["contract"] = {"status": "verified", "frozen_hashes": frozen_hashes}
    if "production_ladder" in config:
        from .fla_checkout import verify_runtime_fla_config

        configured_fla = config.get("vendor_root", config.get("fla_root"))
        verification = verify_runtime_fla_config(
            config, project_root=root, configured=configured_fla
        )
        result["fla_checkout"] = verification
        if verification["status"] == "failed":
            error = verification.get(
                "error",
                {"type": "FLACheckoutError", "message": "FLA checkout verification failed"},
            )
            result["failures"].append(_failure("fla_checkout_preflight", error=error))
            result["status"] = "failed"
            return _jsonable(result)
        if verification["status"] == "verified":
            config["vendor_root"] = verification["actual"]["path"]
    result["device"] = _device_info()
    include_fla = bool(config.get("include_fla", True)) and not bool(config.get("no_fla", False))
    include_liger_model = bool(config.get("include_liger_model", config.get("include_liger", False)))
    include_catswe_model = bool(config.get("include_catswe_model", False))
    # Derive model geometry and evaluate the explicit Catswe model scope
    # before importing or discovering the optional vendor package.  An
    # ineligible LR or non-power-of-two cell must not execute Catswe discovery
    # merely because a caller forged ``include_catswe_model=True``.
    model_data = _model_config(protocol, config, scope)
    catswe_discovery_eligible = _catswe_model_discovery_eligible(
        model_data,
        config.get("ranks", protocol.get("ranks", [])),
    ) if include_catswe_model else False
    configured_vendor = config.get("vendor_root", config.get("fla_root"))
    from .competitors import discover_comparators, vendor_metadata
    comparator_objects = discover_comparators(root, configured_vendor) if include_fla else {}
    model_comparator_objects: dict[str, Any] = {}
    if include_liger_model:
        from .liger import discover_comparator

        configured_liger = config.get("liger_root", config.get("liger_vendor_root"))
        model_comparator_objects["liger"] = discover_comparator(
            project_root=root,
            vendor_root=configured_liger,
        )
    if include_catswe_model and catswe_discovery_eligible:
        from .catswe import discover_comparator

        configured_catswe = config.get("catswe_root", config.get("catswe_vendor_root"))
        model_comparator_objects["catswe_phase1"] = discover_comparator(
            project_root=root,
            vendor_root=configured_catswe,
        )
    result["comparators_enabled"] = include_fla or include_liger_model or include_catswe_model
    result["comparators"] = {
        n: c.describe() for n, c in comparator_objects.items()
    }
    if include_liger_model:
        liger_comparator = model_comparator_objects["liger"]
        result["comparators"]["liger"] = liger_comparator.describe()
    if include_catswe_model and "catswe_phase1" in model_comparator_objects:
        catswe_comparator = model_comparator_objects["catswe_phase1"]
        result["comparators"]["catswe_phase1"] = catswe_comparator.describe()
    vendor = vendor_metadata(root, configured_vendor) if include_fla else {"path": None, "git_revision": None, "dispatch_environment": os.environ.get("FLA_ATTNRES_GLUON")}
    result["source_hashes"] = _source_hashes(root, frozen_hashes, vendor)
    result["hashes"] = {"protocol": dict(frozen_hashes), "software": result["source_hashes"]["software_hash"], "hardware": _hardware_hash(result["device"])}
    cases, case_failures = _operator_cases(protocol, config, scope)
    result["coverage"] = {
        "scope": scope,
        "claims_full_suite": False,
        "operator_cases_requested": (
            len(protocol.get(f"operator_{scope}", []))
            if not config.get("operator_cases") and not config.get("shapes")
            else len(cases)
        ),
        "operator_cases_valid": len(cases),
        "model": model_data,
        "comparators_enabled": include_fla or include_liger_model or include_catswe_model,
        "model_reference_timing": bool(
            config.get("reference_timing", config.get("include_reference", True))
        ),
        "include_fla_model": bool(
            config.get("include_fla_model", config.get("include_fla", True))
        )
        and include_fla,
    }
    if include_liger_model:
        result["coverage"]["include_liger_model"] = True
    if include_catswe_model:
        result["coverage"]["include_catswe_model"] = True
    result["failures"].extend(case_failures)
    if torch.cuda.is_available():
        device = torch.device(config.get("device", "cuda:0"))
        result["device"] = _device_info(device)
        result["hashes"]["hardware"] = _hardware_hash(result["device"])
        seed = int(config.get("seed", protocol["seeds"][0]))
        if "correctness" in phases:
            try: result["correctness"] = _operator_correctness(protocol, cases, device, seed, comparator_objects)
            except Exception as exc: result["correctness"] = {"status": "failed", "failures": [_failure("correctness", error=_exception(exc))]}
        if "operator_timings" in phases:
            try: result["operator_timings"] = _operator_timings(protocol, cases, {**config, "scope": scope}, device, seed, comparator_objects)
            except Exception as exc: result["operator_timings"] = {"status": "failed", "failures": [_failure("operator", error=_exception(exc))]}
        if "model_timings" in phases:
            try:
                result["model_timings"] = _model_timings(
                    protocol,
                    {**config, "scope": scope},
                    scope,
                    device,
                    seed,
                    comparator_objects,
                    model_comparator_objects,
                )
            except Exception as exc: result["model_timings"] = {"status": "failed", "failures": [_failure("model", error=_exception(exc))]}
    else:
        reason = "CUDA is unavailable; CUDA correctness and timing phases were not run"
        for phase in phases:
            if phase in {"correctness", "operator_timings", "model_timings"}: result[phase] = _phase_skip("incomplete", reason)
    result["status"] = _rollup(result)
    return _jsonable(result)


def _read_cli_config(value: str | None) -> dict[str, Any]:
    if not value: return {}
    path = Path(value)
    parsed = json.loads(path.read_text() if path.is_file() else value)
    if not isinstance(parsed, dict): raise ValueError("benchmark config must be a JSON object")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen AttnRes benchmark suite")
    parser.add_argument("--config", help="JSON object or path to a JSON config file")
    parser.add_argument("--scope", choices=("smoke", "primary", "heldout", "custom"))
    parser.add_argument("--phase", action="append", dest="phases", choices=("correctness", "operator", "model"))
    parser.add_argument("--device")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--operator-timing", choices=("eager", "cuda_graph"))
    parser.add_argument("--model-timing", choices=("cuda_event", "cuda_graph"))
    parser.add_argument("--graph-replays", type=int)
    parser.add_argument("--no-fla", action="store_true", help="skip optional FLA comparator arms")
    parser.add_argument("--out", help="write JSON result to this path")
    args = parser.parse_args(argv)
    config = _read_cli_config(args.config)
    for name in ("scope", "device", "rounds", "warmup", "operator_timing", "model_timing", "graph_replays"):
        value = getattr(args, name)
        if value is not None: config[name] = value
    if args.phases: config["phases"] = args.phases
    if args.no_fla: config["include_fla"] = False
    encoded = json.dumps(run_suite(config), indent=2, sort_keys=True, allow_nan=False)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if json.loads(encoded)["status"] == "failed" else 0


if __name__ == "__main__": raise SystemExit(main())
__all__ = ["assert_frozen_hashes", "load_protocol", "main", "run_suite", "sha256_file"]
