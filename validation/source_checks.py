"""Root-owned sequence-interface gates with matched BF16 graph boundaries."""
import contextlib

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .gpu_checks import _compare, PROTOCOL
from .oracle import oracle


class _NoSourceStack(TorchDispatchMode):
    """Reject materializing an activation source stack in forward or backward."""

    def __init__(self, shapes, counts):
        self.shapes = set(shapes)
        self.counts = set(counts)

    @classmethod
    def ignore_compile_internals(cls):
        """Inspect the compiled artifact without blocking Dynamo internals.

        PyTorch 2.13 refuses to trace while an ordinary dispatch mode is
        active.  This opt-in masks the mode during compilation and restores it
        for the compiled artifact, which is the boundary this guard intends to
        inspect.
        """

        return True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.stack.default, torch.ops.aten.cat.default) and args[0]:
            tensors = args[0]
            shapes = [tuple(t.shape) for t in tensors]
            matches = all(shape in self.shapes or shape[1:] in self.shapes for shape in shapes)
            if len(tensors) in self.counts and len(set(shapes)) == 1 and matches:
                raise AssertionError("sequence path materialized a source stack")
        return func(*args, **(kwargs or {}))


def source_case(shape, mode, dtype, *, graph=False, shared=False, device="cuda"):
    """Compare source leaves, query and partial gradients, then changed-input replay."""
    from attnres import attnres

    s, n, d, r = shape
    torch.manual_seed(PROTOCOL["seeds"][0])
    width = d
    count = s - 1 if shared and s > 1 else s
    leaves = []
    for i in range(count):
        physical = ((n, 2 * width), (width, n), (n, width + 7))[i % 3]
        leaves.append(torch.randn(*physical, device=device, dtype=dtype, requires_grad=True))
    qshape = (3, 2 * r) if mode == "block" else (2 * r,)
    query = torch.randn(*qshape, device=device, dtype=dtype, requires_grad=True)
    partial = torch.randn(n, width, device=device, dtype=dtype, requires_grad=True)
    params = (*leaves, query, partial) if mode == "block" else (*leaves, query)
    output_count = 5 if mode == "block" else 1
    upstream = torch.randn(output_count, d, n, device=device, dtype=dtype).transpose(1, 2)
    # Draw every replay input before candidate execution, so a failure cannot
    # alter subsequent inputs. Each invocation also resets its own seed.
    replay = tuple(torch.randn_like(x) for x in params)
    replay_upstream = torch.randn_like(upstream)

    def views(args):
        rows = [a[..., ::2] if i % 3 == 0 else a.T if i % 3 == 1 else a[..., :width]
                for i, a in enumerate(args[:count])]
        if shared and s > 1:
            rows.append(rows[0])
        return tuple(a[..., :d] for a in rows)

    def forward(args, weights, reference=False, packed=False):
        values = views(args)
        q = args[count][..., ::2]
        if packed:
            values = torch.stack(values)
        if mode == "full":
            out = (oracle(values, q)
                   if reference else attnres(values, q))
            return (out,), (out.float() * weights[0].float()).sum()
        part = args[-1]
        completed = values
        combined = (*values, part[..., :d])
        outputs = []
        for index in (0, 1, 1, 2):
            outputs.append(
                oracle(combined, q[index])
                if reference
                else attnres(combined, q[index])
            )
        outputs.append(
            oracle(completed, q[0])
            if reference
            else attnres(values, q[0])
        )
        loss = sum((o.float() * w.float()).sum() for o, w in zip(outputs, weights))
        return tuple(outputs), loss

    def compare(outputs, gradients, args, weights, *, packed=False):
        expected, loss = forward(args, weights, True, packed=packed)
        expected_grads = torch.autograd.grad(loss, args)
        return {"outputs": [_compare(a, e, dtype) for a, e in zip(outputs, expected)],
                "grads": [_compare(a, e, dtype) for a, e in zip(gradients, expected_grads)]}

    def no_stack():
        if device == "cuda":
            shapes = {(n, d), (n, r), (n, width)}
            return _NoSourceStack(shapes, (s, count))
        return contextlib.nullcontext()

    with no_stack():
        outputs, loss = forward(params, upstream)
        gradients = torch.autograd.grad(loss, params, retain_graph=True)
        repeat = torch.autograd.grad(loss, params)
    for a, b in zip(gradients, repeat):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    result = {"eager": compare(outputs, gradients, params, upstream)}
    packed_outputs, packed_loss = forward(params, upstream, packed=True)
    packed_gradients = torch.autograd.grad(packed_loss, params)
    # Packing changes where BF16 leaf contributions accumulate. Compare each
    # layout against its own oracle graph, not against a different cast tree.
    result["packed_control"] = compare(
        packed_outputs, packed_gradients, params, upstream, packed=True)
    if not graph:
        return result

    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
    with no_stack():
        outputs, loss = compiled(params, upstream)
        gradients = torch.autograd.grad(loss, params)
    result["compiled"] = compare(outputs, gradients, params, upstream)
    before = dict(torch._dynamo.utils.counters["stats"])
    with torch.no_grad():
        for x, value in zip(params, replay):
            x.copy_(value)
        upstream.copy_(replay_upstream)
    with no_stack():
        outputs, loss = compiled(params, upstream)
        gradients = torch.autograd.grad(loss, params)
    result["compiled_changed_input"] = compare(outputs, gradients, params, upstream)
    assert dict(torch._dynamo.utils.counters["stats"]) == before

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        static = tuple(x.detach().clone().requires_grad_() for x in params)
        weights = upstream.clone()
        with no_stack():
            for _ in range(3):
                _, loss = compiled(static, weights)
                torch.autograd.grad(loss, static)
    torch.cuda.current_stream().wait_stream(stream)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        with no_stack():
            outputs, loss = compiled(static, weights)
            gradients = torch.autograd.grad(loss, static)
    result["graph_changed_input"] = []
    for index in range(8):
        with torch.no_grad():
            for x, value in zip(static, replay):
                x.copy_(value * ((index + 1) / 8))
            weights.copy_(replay_upstream * ((index + 1) / 8))
        capture.replay()
        torch.cuda.synchronize()
        result["graph_changed_input"].append(compare(outputs, gradients, static, weights))
    return result
