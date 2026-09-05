from pathlib import Path

import pytest
import torch

from attnres import attnres


def _oracle(values, query, *, eps=2**-23, scale=1.0):
    """Test-only BF16 oracle; no reference implementation ships in attnres."""
    values_fp32 = values.float()
    keys_fp32 = values[..., -query.numel():].float()
    query_fp32 = query.float()
    scores = (
        keys_fp32
        * torch.rsqrt(keys_fp32.square().mean(-1, keepdim=True) + eps)
        * query_fp32
    ).sum(-1)
    weights = torch.softmax(scores * scale, dim=0)
    return (weights.unsqueeze(-1) * values_fp32).sum(0).to(values.dtype)


@pytest.mark.cuda
def test_cuda_bf16_operator_and_gradients_match_test_oracle():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    torch.manual_seed(23)
    values = torch.randn(
        3, 2, 17, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    query = torch.randn(5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn_like(values[0])

    actual = attnres(values, query)
    expected = _oracle(values, query)
    gradients = torch.autograd.grad(actual, (values, query), upstream)
    expected_gradients = torch.autograd.grad(expected, (values, query), upstream)

    assert actual.dtype == torch.bfloat16
    assert all(gradient.dtype == torch.bfloat16 for gradient in gradients)
    for actual_value, expected_value in zip(
        (actual, *gradients), (expected, *expected_gradients)
    ):
        torch.testing.assert_close(actual_value, expected_value, rtol=0.05, atol=0.05)


def test_reference_module_is_removed_from_the_distribution():
    package_dir = Path(__file__).resolve().parents[1] / "src" / "attnres"
    assert not (package_dir / "reference.py").exists()
