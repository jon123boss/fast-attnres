import pytest
import torch
from attnres import attnres, reference_attnres
from validation.oracle import oracle


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_and_gradients(dtype):
    torch.manual_seed(23)
    v = torch.randn(3, 2, 17, dtype=dtype, requires_grad=True)
    q = torch.randn(5, dtype=torch.float32, requires_grad=True)
    params = [v, q]
    actual = attnres(v, q)
    expected = oracle(v, q)
    tol = dict(rtol=.05, atol=.05) if dtype == torch.bfloat16 else dict(rtol=.001, atol=.0001)
    torch.testing.assert_close(actual, expected, **tol)
    g = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, params, g)
    ge = torch.autograd.grad(expected, params, g)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, **tol)


def test_fp64_reference_gradcheck():
    v = torch.randn(3, 2, 5, dtype=torch.float64, requires_grad=True)
    q = torch.randn(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a,b: reference_attnres(a,b,compute_dtype=torch.float64), (v,q))
