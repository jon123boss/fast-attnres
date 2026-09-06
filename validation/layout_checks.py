"""Corrected source/upstream addressing and physical-leaf gradient regressions."""
import torch

from .oracle import oracle


def run_layout_checks(operator):
    results = []
    for layout in ("singleton", "nonaffine"):
        for sources in (2, 4):
            results.append(check_layout(operator, layout, sources))
    return results


def check_layout(operator, layout, sources):
    width, rank = 32, 16
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(20260827)
        rows = 3 if layout == "singleton" else 6
        leaves = tuple(torch.randn(rows * width + 1, device="cuda", dtype=torch.bfloat16,
                                   requires_grad=True) for _ in range(sources - (sources == 4)))
        query = (torch.randn(rank * 2, device="cuda", dtype=torch.bfloat16) * .05).requires_grad_()
        raw_upstream = torch.randn(rows * width, device="cuda", dtype=torch.bfloat16)

        def view(value):
            if layout == "singleton":
                return value[1:].as_strided((3, 1, width), (width, 2 * width, 1))
            return value[1:].view(2, 3, width).transpose(0, 1)

        upstream = (raw_upstream.as_strided((3, 1, width), (width, 3 * width, 1))
                    if layout == "singleton" else raw_upstream.view(2, 3, width).transpose(0, 1))

        def forward(op):
            values = tuple(view(value) for value in leaves)
            if sources == 4:
                values = (*values, values[1])
            return op(values, query[::2], eps=2**-10, scale=-.75)

        def call():
            output = forward(operator)
            return (output, *torch.autograd.grad(output, (*leaves, query), upstream))

        def check(actual):
            expected = forward(oracle)
            gradients = torch.autograd.grad(expected, (*leaves, query), upstream)
            for a, b in zip(actual, (expected, *gradients)):
                assert a.dtype == torch.bfloat16
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                torch.testing.assert_close(a, b, rtol=.05, atol=.05)

        check(call())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = call()
        for _ in range(8):
            with torch.no_grad():
                for value in leaves:
                    value.copy_(torch.randn_like(value))
                query.copy_(torch.randn_like(query) * .05)
                raw_upstream.copy_(torch.randn_like(raw_upstream))
            graph.replay()
            torch.cuda.synchronize()
            check(captured)
    torch.cuda.current_stream().wait_stream(stream)
    return {"layout": layout, "sources": sources, "status": "passed",
            "storage_offset": 1, "query_stride": 2, "leaf_gradients": True,
            "shared_source": sources == 4, "changed_input_replays": 8,
            "eps": 2**-10, "scale": -.75}
