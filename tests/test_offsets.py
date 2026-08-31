"""Exercise actual addresses beyond signed 32-bit element offsets."""
import pytest
import torch

from attnres import attnres
from validation.gpu_checks import _compare
from validation.oracle import oracle

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device")]


def test_source_offsets_above_int32():
    torch.manual_seed(20260827)
    width, rank = 8192, 127
    stride = 2**31 + 32
    storage = torch.empty(stride + width + rank, device="cuda", dtype=torch.bfloat16)
    values = storage.as_strided((2, 1, width), (stride, width, 1)).detach()
    padding = storage.as_strided((2, 1, rank), (stride, rank, 1), width).detach()
    values.copy_(torch.randn_like(values))
    # Preserve the frozen implicit fixture's RNG sequence before query/upstream.
    padding.copy_(torch.randn_like(padding))
    values.requires_grad_()
    query = (torch.randn(rank, device="cuda") * 0.25).requires_grad_()
    assert (values[1].data_ptr() - values[0].data_ptr()) // values.element_size() > 2**31
    actual = attnres(values, query)
    expected = oracle(values, query)
    upstream = torch.randn_like(actual)
    parameters = (values, query)
    actual_grad = torch.autograd.grad(actual, parameters, upstream)
    expected_grad = torch.autograd.grad(expected, parameters, upstream)
    _compare(actual, expected, values.dtype)
    for a, e in zip(actual_grad, expected_grad):
        _compare(a, e, values.dtype)


@pytest.mark.parametrize("variant", ["standard", "sliced"])
def test_source_list_row_stride_above_int32(variant):
    """Exercise row strides, not just distant source base addresses."""
    torch.manual_seed(20260827)
    width, rank = 17, 17 if variant == "standard" else 5
    stride = 2**31 + 32
    storage = torch.empty(stride + width + rank, device="cuda", dtype=torch.bfloat16)
    value = storage.as_strided((2, width), (stride, 1)).detach()
    value.copy_(torch.randn_like(value))
    value.requires_grad_()
    other = torch.randn(2, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    query = (torch.randn(rank, device="cuda") * .25).requires_grad_()
    values = (value, other)
    assert value.stride(0) > 2**31
    actual = attnres(values, query)
    expected = oracle(torch.stack(values), query)
    upstream = torch.randn_like(actual)
    parameters = (*values, query)
    actual_grad = torch.autograd.grad(actual, parameters, upstream)
    expected_grad = torch.autograd.grad(expected, parameters, upstream)
    _compare(actual, expected, value.dtype)
    for a, e in zip(actual_grad, expected_grad):
        _compare(a, e, value.dtype)
