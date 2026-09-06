"""Large-offset softmax regressions, including changed-upstream graph replay."""

import torch


def check_equal_logits(operator, *, sources=4, rows=2, packed=False, scale=2**24):
    """Equal sources must have exact uniform value gradients and zero query gradients."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Views create autograd nodes too: construct them on the capture stream.
    with torch.cuda.stream(stream):
        values = torch.ones(sources, rows, 256, device='cuda', dtype=torch.bfloat16,
                            requires_grad=True)
        query = torch.ones(16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
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
    torch.cuda.current_stream().wait_stream(stream)
    return {'sources': sources, 'rows': rows, 'packed': packed, 'scale': scale,
            'status': 'passed', 'changed_upstream_replays': 2, 'exact_gradients': True}


def run_equal_logit_checks(operator):
    # S=2 exercises recomputation; S=4 exercises saved values; N=1024 also
    # reaches the split-forward prototype. Shapes share compiled scalar variants.
    return [check_equal_logits(operator, sources=s, rows=n, packed=packed, scale=scale)
            for s, n, packed in ((2, 2, False), (4, 2, False), (4, 2, True), (4, 1024, False))
            for scale in (2**24, -(2**24))]
