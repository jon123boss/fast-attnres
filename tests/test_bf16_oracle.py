"""BF16 arithmetic is enforced throughout the independent reference."""
import pytest
import torch

from validation.oracle import oracle


@pytest.mark.parametrize("sequence", [False, True])
def test_reference_keeps_every_floating_intermediate_and_gradient_bf16(sequence):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class BF16Only(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for tensor in tree_leaves(result):
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    assert tensor.dtype == torch.bfloat16, str(func)
            return result

    torch.manual_seed(19)
    values = torch.randn(5, 3, 17, dtype=torch.bfloat16, requires_grad=True)
    query = torch.randn(7, dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(3, 17, dtype=torch.bfloat16)
    with BF16Only():
        inputs = tuple(values.unbind()) if sequence else values
        output = oracle(inputs, query, scale=.7)
        gradients = torch.autograd.grad(output, (values, query), upstream)
    assert output.dtype == torch.bfloat16
    assert all(gradient.dtype == torch.bfloat16 for gradient in gradients)


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
