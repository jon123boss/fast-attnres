"""GPU measurements for the BF16 campaign; no cloud or account operations."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import torch

from benchmarks.baseline import load_baseline
from validation.oracle import oracle


def source_digest(root):
    files = sorted(Path(root).rglob("*.py"))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if "__pycache__" not in p.parts}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": hashes}


def metadata():
    import triton
    p = torch.cuda.get_device_properties(0)
    return {"torch": str(torch.__version__), "triton": str(triton.__version__),
            "cuda": torch.version.cuda, "python": platform.python_version(),
            "gpu": p.name, "capability": list(torch.cuda.get_device_capability()),
            "memory_bytes": p.total_memory, "sms": p.multi_processor_count}


def bf16_torch(values, query, *, eps=2**-23, scale=1.0):
    """Validation/benchmark fixture only: BF16 storage with stable reductions."""
    if isinstance(values, (tuple, list)):
        values = torch.stack(values)
    return oracle(values, query, eps=eps, scale=scale)


def compare(actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("nonfinite output or gradient")
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)
    difference = (actual.detach().float() - expected.detach().float()).abs()
    return {"max_abs": float(difference.max()),
            "relative_l2": float(torch.linalg.vector_norm(difference) /
                                 torch.linalg.vector_norm(expected.detach().float()).clamp_min(1e-20))}


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
    return result


def run_operator(config, checkpoint):
    """Run an immutable case list and checkpoint each result through the caller."""
    actual = metadata()
    if not actual["torch"].startswith("2.13.0") or actual["triton"] != "3.7.1":
        raise RuntimeError(f"unqualified runtime: {actual}")
    expected_capability = {"H100": [9, 0], "B200": [10, 0]}[config["gpu"]]
    if actual["capability"] != expected_capability or config["gpu"] not in actual["gpu"]:
        raise RuntimeError(f"GPU substitution: {actual}")
    backends, identities = {}, {}
    for name, root in config["sources"].items():
        baseline = load_baseline(root)
        backends[name] = baseline.attnres
        identities[name] = baseline.metadata
    from benchmarks.bf16_competitors import load_all
    competitors, competitor_identities, import_failures = load_all(config.get("competitors", {}))
    backends.update(competitors)
    identities.update(competitor_identities)
    if config.get("torch_baseline", False):
        backends["torch_compile"] = torch.compile(bf16_torch, fullgraph=True, dynamic=False)
    report = {"kind": "operator", "config": config, "runtime": actual,
              "identities": identities, "import_failures": import_failures,
              "results": [], "status": "running"}
    checkpoint(report)
    for case in config["cases"]:
        for seed in config["seeds"]:
            report["in_progress"] = {"case": case, "seed": seed}
            checkpoint(report)
            report["results"].append(operator_case(case, backends, seed=seed,
                warmups=config.get("warmups", 5), rounds=config.get("rounds", 40),
                replays=config.get("replays", 8)))
            report.pop("in_progress", None)
            checkpoint(report)
            gc.collect()
            torch.cuda.empty_cache()
    report["status"] = "complete"
    checkpoint(report)
    return report
