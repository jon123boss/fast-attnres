"""BF16 boundaries and autocast-independent reference arithmetic."""
import pytest
import torch

from validation.oracle import oracle


@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_reference_bf16_boundaries_are_independent_of_autocast(sequence, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(19)
    values = torch.randn(5, 3, 17, device=device, dtype=torch.bfloat16, requires_grad=True)
    query = torch.randn(7, device=device, dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(3, 17, device=device, dtype=torch.bfloat16)
    results = []
    for enabled in (False, True):
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=enabled):
            inputs = tuple(values.unbind()) if sequence else values
            output = oracle(inputs, query, scale=.7)
        gradients = torch.autograd.grad(output, (values, query), upstream)
        result = (output, *gradients)
        assert all(tensor.dtype == torch.bfloat16 for tensor in result)
        results.append(result)
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_rejects_other_floating_dtypes(dtype):
    with pytest.raises(TypeError, match="requires BF16"):
        oracle(torch.ones(2, 3, 16, dtype=dtype), torch.ones(16, dtype=dtype))


@pytest.mark.parametrize("mode", ["full", "block"])
def test_shared_bf16_views_use_the_same_per_read_boundaries(monkeypatch, mode):
    import attnres
    from validation.source_checks import source_case
    monkeypatch.setattr(attnres, "attnres", oracle)
    result = source_case((5, 7, 33, 17), mode, torch.bfloat16, shared=True, device="cpu")
    assert result["eager"]["grads"]
    assert result["packed_control"]["grads"]
