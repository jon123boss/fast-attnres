"""Large-offset softmax regressions, including changed-upstream graph replay."""

import importlib
import os

import torch


def check_equal_logits(operator, *, sources=4, rows=2, width=256, rank=16,
                       packed=False, scale=2**24, expected_router=None):
    """Equal sources must have exact uniform value gradients and zero query gradients."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Views create autograd nodes too: construct them on the capture stream.
    with torch.cuda.stream(stream):
        values = torch.ones(sources, rows, width, device='cuda', dtype=torch.bfloat16,
                            requires_grad=True)
        query = torch.ones(rank, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        inputs = values if packed else tuple(values.unbind())
        upstream = torch.ones_like(values[0])

    def call():
        output = operator(inputs, query, eps=2**-23, scale=scale)
        return output, *torch.autograd.grad(output, (values, query), upstream)

    def check(result):
        output, dv, dq = result
        torch.testing.assert_close(output, values[0], rtol=0, atol=0)
        torch.testing.assert_close(dv, upstream.expand_as(values) / sources, rtol=0, atol=0)
        torch.testing.assert_close(dq, torch.zeros_like(query), rtol=0, atol=0)

    compiler = {}
    with torch.cuda.stream(stream):
        check(call())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = call()
        for factor in (-2.0, 4.0):
            upstream.fill_(factor)
            graph.replay()
            torch.cuda.synchronize()
            check(result)
        if expected_router is not None:
            from benchmarks.bf16_kernel_export import export_selected
            module = importlib.import_module(operator.__module__.rsplit('.', 1)[0]
                                             + '._kernels.fla_full_sources')
            compiler = export_selected(module, call, os.environ['ATTNRES_KERNEL_EXPORTS'])
            assert ('_routing_forward' in compiler) == expected_router
    torch.cuda.current_stream().wait_stream(stream)
    return {'sources': sources, 'rows': rows, 'width': width, 'rank': rank,
            'packed': packed, 'scale': scale, 'compiler': compiler,
            'status': 'passed', 'changed_upstream_replays': 2, 'exact_gradients': True}


def run_equal_logit_checks(operator, cases=None):
    # These defaults cover saved/recomputed paths, not guarded split dispatch.
    if cases is None:
        cases = [dict(sources=s, rows=n, packed=packed, scale=scale)
                 for s, n, packed in ((2, 2, False), (4, 2, False), (4, 2, True), (4, 1024, False))
                 for scale in (2**24, -(2**24))]
    return [check_equal_logits(operator, **case) for case in cases]
