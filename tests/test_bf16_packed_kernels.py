"""Focused BF16 contract and CUDA checks for the fixed-tail implementation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).parents[1]
_BF16_TOL = {"rtol": 0.05, "atol": 0.05}


def _bf16_oracle(
    values: torch.Tensor,
    query: torch.Tensor,
    *,
    eps: float = 2**-23,
    scale: float = 1.0,
) -> torch.Tensor:
    """Test-only equation oracle; the production files do not use this path."""

    values_f32 = values.float()
    query_f32 = query.float()
    keys = values_f32[..., -query.numel() :]
    inv_rms = torch.rsqrt(keys.square().mean(dim=-1, keepdim=True) + eps)
    logits = (keys * inv_rms * query_f32).sum(dim=-1) * scale
    weights = torch.softmax(logits, dim=0)
    return (weights.unsqueeze(-1) * values_f32).sum(dim=0).to(values.dtype)


def test_target_files_have_only_the_bf16_cuda_runtime_surface():
    from attnres._kernels.fixed_tail import _validate_inputs, fused_attnres
    from attnres._kernels.fixed_tail_sources import source_attnres

    values_f32 = torch.randn(2, 3, 8)
    query_f32 = torch.randn(4)
    with pytest.raises(TypeError, match="BF16"):
        _validate_inputs(values_f32, query_f32)
    with pytest.raises(TypeError, match="BF16"):
        source_attnres((values_f32[0], values_f32[1]), query_f32)

    values = values_f32.to(torch.bfloat16)
    query = query_f32.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="CUDA BF16"):
        fused_attnres(values, query)
    with pytest.raises(RuntimeError, match="CUDA BF16"):
        source_attnres((values[0], values[1]), query)


def test_source_fallback_does_not_import_shipped_reference():
    source = (_ROOT / "src/attnres/_kernels/fixed_tail_sources.py").read_text()
    tree = ast.parse(source)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    assert "reference_attnres" not in names


def test_bf16_launch_policy_is_shared_by_packed_and_source_list_adapters():
    from attnres._kernels import fixed_tail

    assert fixed_tail.SOURCE_TILE == 2
    assert fixed_tail.FUSE_KEY_VALUE
    assert fixed_tail._should_fuse_key_value(1024, 384)
    assert not fixed_tail._should_fuse_key_value(1024, 64)
    assert not fixed_tail._should_fuse_key_value(4096, 1024)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("rank", [64, 384])
def test_packed_bf16_forward_backward_matches_oracle(rank):
    pytest.importorskip("triton")
    from attnres._kernels.fixed_tail import fused_attnres

    torch.manual_seed(20260905 + rank)
    sources, rows, width = 5, 7, 1024
    producer = torch.randn(
        sources,
        rows,
        width + 7,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    values = producer[..., :width]
    query = torch.randn(rank, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)

    actual = fused_attnres(values, query)
    expected = _bf16_oracle(values, query)
    actual_grads = torch.autograd.grad(actual, (producer, query), upstream)
    expected_grads = torch.autograd.grad(expected, (producer, query), upstream)

    assert actual.dtype == torch.bfloat16
    for got, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(got, reference, **_BF16_TOL)

    replay = fused_attnres(values, query)
    replay_grads = torch.autograd.grad(replay, (producer, query), upstream)
    torch.testing.assert_close(replay, actual, rtol=0, atol=0)
    for got, reference in zip(replay_grads, actual_grads):
        torch.testing.assert_close(got, reference, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_wide_strided_source_list_fallback_matches_oracle():
    pytest.importorskip("triton")
    from attnres._kernels.fixed_tail_sources import source_attnres

    torch.manual_seed(20260905)
    sources, rows, width, rank = 5, 3, 2304, 257
    storage = torch.randn(
        sources,
        rows,
        2 * width,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    values = storage[..., ::2]
    source_list = tuple(values.unbind(0))
    query = torch.randn(rank, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(width, rows, device="cuda", dtype=torch.bfloat16).transpose(0, 1)

    actual = source_attnres(source_list, query)
    expected = _bf16_oracle(torch.stack(source_list), query)
    actual_grads = torch.autograd.grad(actual, (storage, query), upstream)
    expected_grads = torch.autograd.grad(expected, (storage, query), upstream)

    assert actual.dtype == torch.bfloat16
    for got, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(got, reference, **_BF16_TOL)
