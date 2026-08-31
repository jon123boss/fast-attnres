"""Independent wide per-read Block equations and graph replay gates."""
import gc
import hashlib
import traceback

import torch

from .gpu_checks import _compare, PROTOCOL


# Chosen before evaluating this gate; these are development envelope cases,
# not the untouched held-out protocol cases.
CASES = [(1, 2, 128, 1), (9, 65, 768, 127), (33, 8, 3000, 257),
         (128, 8, 7168, 1024), (128, 8, 8192, 8192)]


def _forward(source, partial, queries, upstream, width, reference):
    from attnres import attnres
    from .oracle import oracle

    values = source[..., :width]
    combined = torch.cat((values, partial[..., :width].unsqueeze(0)), dim=0)
    function = oracle if reference else attnres
    outputs = []
    for index in (0, 1, 1, 2):
        outputs.append(function(combined, queries[index]))
    outputs.append(function(values, queries[0]))
    stacked = torch.stack(outputs)
    loss = (stacked.float() * upstream.float()).sum()
    return stacked, loss


def _input_hash(args):
    digest = hashlib.sha256()
    for value in args:
        digest.update(f"{tuple(value.shape)}:{value.dtype}:".encode())
        digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _case(shape, dtype, graph, metadata=None):
    s, n, d, r = shape
    width = d
    source = torch.randn(s, n, width, device="cuda", dtype=dtype, requires_grad=True)
    partial = torch.randn(n, width, device="cuda", dtype=dtype, requires_grad=True)
    query = (torch.randn(3, r, device="cuda") * 0.25).requires_grad_()
    upstream = torch.randn(5, d, n, device="cuda", dtype=dtype).transpose(1, 2)
    args = (source, partial, query, upstream)
    # Consume the fixed replay draws even if this candidate fails before replay.
    # Otherwise a failure changes the random inputs of every following case.
    replay_args = tuple(torch.randn_like(x) * 0.25 for x in args) if graph else ()
    if metadata is not None:
        metadata["inputs"] = {"initial_sha256": _input_hash(args),
                              "replay_sha256": _input_hash(replay_args) if graph else None}

    def forward(a, b, q, u):
        return _forward(a, b, q, u, d, False)

    function = torch.compile(forward, fullgraph=True, dynamic=False) if graph else forward

    def check(actual, grads):
        expected, loss = _forward(*args, d, True)
        expected_grads = torch.autograd.grad(loss, args[:3])
        return {"output": _compare(actual, expected, dtype),
                "grads": [_compare(a, e, dtype) for a, e in zip(grads, expected_grads)]}

    actual, loss = function(*args)
    grads = torch.autograd.grad(loss, args[:3], retain_graph=True)
    repeat = torch.autograd.grad(loss, args[:3])
    for a, b in zip(grads, repeat):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    result = check(actual, grads)
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            args = tuple(x.detach().clone().requires_grad_(i < 3) for i, x in enumerate(args))
            for _ in range(3):
                _, loss = forward(*args)
                torch.autograd.grad(loss, args[:3])
        torch.cuda.current_stream().wait_stream(stream)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=stream):
            actual, loss = forward(*args)
            grads = torch.autograd.grad(loss, args[:3])
        with torch.no_grad():
            for x, value in zip(args, replay_args):
                x.copy_(value)
        capture.replay()
        torch.cuda.synchronize()
        result["changed_input_replay"] = check(actual, grads)
        result["compiled_fullgraph"] = True
    return result


def run_block_checks(config):
    torch.manual_seed(config.get("seed", PROTOCOL["seeds"][0]))
    dtype = torch.bfloat16 if config.get("dtype", "bf16") == "bf16" else torch.float32
    result = {"passed": 0, "failed": 0, "cases": [], "config": config}
    for shape in config.get("cases", CASES):
        name = f"block_{shape}"
        row = {"name": name}
        try:
            metrics = _case(shape, dtype, config.get("graph", False), row)
            row.update(status="passed", metrics=metrics)
            result["passed"] += 1
        except Exception as exc:
            row.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            result["failed"] += 1
        result["cases"].append(row)
        gc.collect()
        torch.cuda.empty_cache()
    return result
