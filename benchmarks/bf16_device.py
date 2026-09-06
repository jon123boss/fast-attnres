"""Operator qualification and changed-input replay checks for cache validation."""
from __future__ import annotations

import os
import sys
import time
import traceback

import torch

from validation.comparison import compare
from validation.oracle import oracle as bf16_torch


def _profile_operator(op, values, query, params, upstream):
    """Attribute CUDA work after timing; profiler samples never select a winner."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True,
    ) as profile:
        for _ in range(5):
            y = op(values, query)
            torch.autograd.grad(y, params, upstream)
        torch.cuda.synchronize()
    module = sys.modules.get(op.__module__.rsplit(".", 1)[0] + "._kernels.fla_full_sources")
    launches = {}
    if module is not None:
        for name in ("_fla_standard_forward_kernel", "_fla_standard_backward_kernel",
                     "_fla_standard_query_reduce_kernel"):
            config = getattr(getattr(module, name, None), "best_config", None)
            if config is not None:
                launches[name] = {"kwargs": config.kwargs, "num_warps": config.num_warps,
                                  "num_stages": config.num_stages}
    compiler = {}
    if module is not None and (directory := os.environ.get("ATTNRES_KERNEL_EXPORTS")):
        from benchmarks.bf16_kernel_export import export_selected
        def call():
            output = op(values, query)
            torch.autograd.grad(output, params, upstream)
        compiler = export_selected(module, call, directory)
        torch.cuda.synchronize()
    return {"iterations": 5, "timing_eligible": False, "launches": launches,
            "compiler": compiler,
            "events": [{"name": event.key, "count": event.count,
                        "self_cpu_us": event.self_cpu_time_total,
                        "self_cuda_us": event.self_device_time_total,
                        "cuda_us": event.device_time_total,
                        "self_cuda_memory_bytes": event.self_device_memory_usage}
                       for event in profile.key_averages()
                       if event.device_time_total or event.self_device_memory_usage]}


def _inputs(case, seed):
    s, n, d, r = case["shape"]
    torch.manual_seed(seed)
    # Physical producers are independent. Shared/view cases retain their true
    # autograd ownership, rather than comparing detached source gradients.
    layout = case.get("layout", "list")
    shared = case.get("shared", False)
    if layout == "strided":
        leaves = tuple(torch.randn(n, d * 2, device="cuda", dtype=torch.bfloat16,
                                   requires_grad=True) for _ in range(s - int(shared)))
        values = tuple(x[:, ::2] for x in leaves)
    elif layout == "packed":
        leaves = (torch.randn(s, n, d, device="cuda", dtype=torch.bfloat16,
                              requires_grad=True),)
        values = leaves[0]
    else:
        leaves = tuple(torch.randn(n, d, device="cuda", dtype=torch.bfloat16,
                                   requires_grad=True) for _ in range(s - int(shared)))
        values = leaves
    if shared:
        if isinstance(values, torch.Tensor):
            raise ValueError("shared packed fixture is not defined")
        values = (*values, values[0])
    q = (torch.randn(r, device="cuda", dtype=torch.bfloat16) *
         case.get("query_scale", 0.05)).requires_grad_()
    # Deliberately exercise a noncontiguous upstream gradient.
    upstream = torch.randn(d, n, device="cuda", dtype=torch.bfloat16).T
    return values, (*leaves, q), q, upstream


def operator_case(case, backends, *, seed, warmups=5, rounds=40, replays=4):
    # Autograd records leaf stream ownership at its first use. Qualification,
    # warmup and capture must therefore share the same non-default stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = _operator_case(case, backends, seed=seed, warmups=warmups,
                                rounds=rounds, replays=replays)
    torch.cuda.current_stream().wait_stream(stream)
    return result


def _operator_case(case, backends, *, seed, warmups, rounds, replays):
    values, params, query, upstream = _inputs(case, seed)
    expected = bf16_torch(values, query)
    expected_grad = torch.autograd.grad(expected, params, upstream)
    arms, failures = {}, {}
    for name, op in backends.items():
        if case.get("backends") is not None and name not in case["backends"]:
            continue
        try:
            start = time.monotonic()
            print(f"operator arm {name} {case} seed={seed}", flush=True)
            out = op(values, query)
            grads = torch.autograd.grad(out, params, upstream)
            errors = [compare(out, expected)] + [compare(a, b) for a, b in zip(grads, expected_grad)]
            if any(x.dtype != torch.bfloat16 for x in (out, *grads)):
                raise AssertionError("operator output/gradient must use BF16 storage")
            def call():
                y = op(values, query)
                return y, torch.autograd.grad(y, params, upstream)
            for _ in range(warmups):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
                result = call()
            arms[name] = {"graph": graph, "result": result, "errors": errors,
                          "compile_warmup_s": time.monotonic() - start, "samples_ms": []}
        except Exception as exc:
            failures[name] = {"status": "failed", "phase": "qualification",
                              "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()}

    # All backends see the same changed leaves on every replay. A replay's
    # buffers must not contain stale outputs or gradients from capture.
    for replay in range(replays):
        torch.manual_seed(seed + 1000 + replay)
        with torch.no_grad():
            for x in params[:-1]:
                x.copy_(torch.randn_like(x))
            query.copy_(torch.randn_like(query) * case.get("query_scale", 0.05))
        ref = bf16_torch(values, query)
        ref_grads = torch.autograd.grad(ref, params, upstream)
        for name, arm in list(arms.items()):
            try:
                arm["graph"].replay()
                torch.cuda.synchronize()
                y, grads = arm["result"]
                compare(y, ref)
                for a, b in zip(grads, ref_grads):
                    compare(a, b)
            except Exception as exc:
                failures[name] = {"status": "failed", "phase": "changed_input",
                                  "replay": replay, "error": f"{type(exc).__name__}: {exc}"}
                del arms[name]
    names = list(arms)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for iteration in range(rounds):
        order = names if iteration % 2 == 0 else list(reversed(names))
        for name in order:
            begin.record()
            for _ in range(10):
                arms[name]["graph"].replay()
            end.record()
            end.synchronize()
            arms[name]["samples_ms"].append(begin.elapsed_time(end) / 10)
    result = {"case": case, "seed": seed,
              "measurement": "CUDA Graph forward+backward including query reduction",
              "arms": {name: {"status": "passed", "samples_ms": a["samples_ms"],
                              "errors": a["errors"], "compile_warmup_s": a["compile_warmup_s"]}
                       for name, a in arms.items()}}
    result["arms"].update(failures)
    if case.get("profile", False):
        for name in arms:
            try:
                result["arms"][name]["profile"] = _profile_operator(
                    backends[name], values, query, params, upstream)
            except Exception as exc:
                result["arms"][name]["profile"] = {"error": f"{type(exc).__name__}: {exc}",
                                                     "timing_eligible": False}
    return result
