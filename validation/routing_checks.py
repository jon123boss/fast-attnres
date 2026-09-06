"""Nonuniform routing derivatives at the fused and split-route boundaries."""
from __future__ import annotations

import math

import torch

from .oracle import oracle


def fill_case(values, query, upstream, kind, iteration):
    """Change every independent input while retaining the case's numeric regime."""
    sources, _, width = values.shape
    rank = query.numel()
    with torch.no_grad():
        values.normal_()
        query.normal_(std=.05)
        upstream.normal_()
        # Different non-key values make zero-key routing derivatives observable.
        values[..., :-rank].add_(torch.arange(sources, device=values.device)[:, None, None])
        if kind in ('zero_keys', 'tiny_keys'):
            values[..., -rank:].mul_(0 if kind == 'zero_keys' else 2**-12)
            return 2**-10, -.75
        if kind == 'zero_scale':
            return 2**-10, 0.
        if kind == 'large_logits':
            # Fixed key norms, a large shared score, and unequal finite scores.
            # Only the sign of coordinate 1 differs across sources.
            values[..., -rank:].zero_()
            values[..., -rank] = 1
            values[..., -rank + 1] = torch.tensor(
                [1 if (s + iteration) % 2 else -1 for s in range(sources)],
                device=values.device)[:, None]
            query.zero_()
            query[0] = 1 + iteration / 8
            query[1] = 2**-22
            return 2**-10, 2**20
        if kind != 'negative_scale':
            raise ValueError(f'unknown routing case: {kind}')
        return 2**-10, -.75


def check_routing(operator, *, rows, width, rank, kind, replays=8, compiled=False):
    """Check real input leaves against the FP32 oracle, including changed replay."""
    from benchmarks.bf16_device import compare

    torch.manual_seed(20260906)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        values = torch.empty(4, rows, width, device='cuda', dtype=torch.bfloat16,
                             requires_grad=True)
        query = torch.empty(rank, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        upstream = torch.empty(width, rows, device='cuda', dtype=torch.bfloat16).T
        eps, scale = fill_case(values, query, upstream, kind, 0)

        def forward(v, q):
            return operator(tuple(v.unbind()), q, eps=eps, scale=scale)

        actual_forward = (torch.compile(forward, fullgraph=True, dynamic=False,
                                       options={'triton.cudagraphs': False})
                          if compiled else forward)

        def call():
            output = actual_forward(values, query)
            return output, *torch.autograd.grad(output, (values, query), upstream)

        def check(actual):
            expected = oracle(tuple(values.unbind()), query, eps=eps, scale=scale)
            gradients = torch.autograd.grad(expected, (values, query), upstream)
            errors = [compare(a, b) for a, b in zip(actual, (expected, *gradients))]
            if kind == 'zero_keys':
                # alpha=1/S, k=0: dk=scale*dz*q/sqrt(eps); dq=0.
                v, g, q = values.detach().float(), upstream.float(), query.detach().float()
                mixed = v.mean(0)
                dz = ((v - mixed) * g).sum(-1) / v.shape[0]
                dk = scale * dz[..., None] * q / math.sqrt(eps)
                derivative = (g[None, ..., -rank:] / v.shape[0] + dk).to(values.dtype)
                assert torch.count_nonzero(dk), 'zero-key routing derivative was not exercised'
                compare(actual[1][..., -rank:], derivative)
                torch.testing.assert_close(actual[2], torch.zeros_like(query), rtol=0, atol=0)
            return errors

        errors = check(call())
        if compiled:
            # Exercise Inductor with new inputs, independently of raw graph replay.
            for iteration in range(1, replays + 1):
                fill_case(values, query, upstream, kind, iteration)
                errors = check(call())
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = call()
            for iteration in range(1, replays + 1):
                fill_case(values, query, upstream, kind, iteration)
                graph.replay()
                torch.cuda.synchronize()
                errors = check(captured)
    torch.cuda.current_stream().wait_stream(stream)
    return dict(status='passed', shape=[4, rows, width, rank], kind=kind,
                eps=eps, scale=scale, changed_input_replays=replays,
                compiled_fullgraph=compiled, dynamic=False if compiled else None,
                nonzero_zero_key_derivative=kind == 'zero_keys', errors=errors)
