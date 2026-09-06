"""The numerical fixtures must exercise the derivatives their gates claim."""
import math

import torch

from validation.oracle import oracle
from validation.routing_checks import fill_case


def test_zero_keys_have_nonzero_routing_derivatives():
    torch.manual_seed(20260906)
    values = torch.empty(4, 3, 32, dtype=torch.bfloat16, requires_grad=True)
    query = torch.empty(16, dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.empty(3, 32, dtype=torch.bfloat16)
    eps, scale = fill_case(values, query, upstream, 'zero_keys', 0)
    result = oracle(values, query, eps=eps, scale=scale)
    dv, dq = torch.autograd.grad(result, (values, query), upstream)
    v, g, q = values.detach().float(), upstream.float(), query.detach().float()
    dz = ((v - v.mean(0)) * g).sum(-1) / 4
    dk = scale * dz[..., None] * q / math.sqrt(eps)
    assert torch.count_nonzero(dk) == dk.numel()
    expected = (g[None, ..., -16:] / 4 + dk).to(torch.bfloat16)
    torch.testing.assert_close(dv[..., -16:], expected, rtol=.05, atol=.05)
    torch.testing.assert_close(dq, torch.zeros_like(query), rtol=0, atol=0)


def test_large_logit_fixture_is_finite_and_nonuniform():
    values = torch.empty(4, 2, 32, dtype=torch.bfloat16)
    query = torch.empty(16, dtype=torch.bfloat16)
    upstream = torch.empty(2, 32, dtype=torch.bfloat16)
    eps, scale = fill_case(values, query, upstream, 'large_logits', 0)
    key = values[..., -16:].float()
    logits = (key * query.float()).sum(-1) * torch.rsqrt(key.square().mean(-1) + eps) * scale
    assert torch.isfinite(logits).all() and logits.min() > 1e6
    assert logits.unique().numel() == 2
    assert (logits.softmax(0) - .25).abs().max() > .01


def test_zero_scale_removes_routing_but_keeps_direct_value_gradient():
    values = torch.empty(4, 2, 32, dtype=torch.bfloat16, requires_grad=True)
    query = torch.empty(16, dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.empty(2, 32, dtype=torch.bfloat16)
    eps, scale = fill_case(values, query, upstream, 'zero_scale', 0)
    result = oracle(values, query, eps=eps, scale=scale)
    dv, dq = torch.autograd.grad(result, (values, query), upstream)
    torch.testing.assert_close(dv, upstream.expand_as(values) / 4, rtol=0, atol=0)
    torch.testing.assert_close(dq, torch.zeros_like(query), rtol=0, atol=0)
