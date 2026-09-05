"""BF16 oracle boundaries checked against the analytic routing derivative."""
import pytest
import torch

from validation.oracle import oracle


@pytest.mark.parametrize("rank", [3, 7, 17])
@pytest.mark.parametrize("sequence", [False, True])
def test_bf16_oracle_rounds_complete_input_derivative_once(rank, sequence):
    torch.manual_seed(19)
    packed = torch.randn(5, 3, 17, dtype=torch.bfloat16)
    values = (tuple(x.clone().requires_grad_() for x in packed)
              if sequence else packed.requires_grad_())
    query = torch.randn(rank, dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(3, 17, dtype=torch.bfloat16)
    parameters = (*values, query) if sequence else (values, query)
    output = oracle(values, query, scale=.7)
    gradients = torch.autograd.grad(output, parameters, upstream)

    v, q, dy = packed.float(), query.detach().float(), upstream.float()
    key = v[..., -rank:]
    inverse = (key.square().mean(-1) + 2**-23).rsqrt()
    normalized = key * inverse[..., None]
    logits = .7 * (normalized * q).sum(-1)
    probability = logits.softmax(0)
    dweight = (v * dy).sum(-1)
    dscore = probability * (dweight - (probability * dweight).sum(0))
    expected_v = probability[..., None] * dy
    expected_v[..., -rank:] += inverse[..., None] * (
        .7 * dscore[..., None] * q - normalized * (dscore * logits / rank)[..., None])
    expected_q = (.7 * dscore[..., None] * normalized).sum((0, 1))
    expected = ((*expected_v.to(torch.bfloat16).unbind(0), expected_q.to(torch.bfloat16))
                if sequence else (expected_v.to(torch.bfloat16), expected_q.to(torch.bfloat16)))
    for actual, target in zip(gradients, expected):
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, target, rtol=.05, atol=.05)


@pytest.mark.parametrize("mode", ["full", "block"])
def test_shared_bf16_views_use_the_same_per_read_boundaries(monkeypatch, mode):
    import attnres
    from validation.source_checks import source_case
    monkeypatch.setattr(attnres, "attnres", oracle)
    result = source_case((5, 7, 33, 17), mode, torch.bfloat16, shared=True, device="cpu")
    assert result["eager"]["grads"]
    assert result["packed_control"]["grads"]
