"""Public functional entry point for implicit-tail Attention Residuals."""

import torch
from torch import Tensor

from ._sources import _validate_scalar, validate_sources
from .reference import EPS, reference_attnres


def _validate(values: Tensor, query: Tensor, eps: float, scale: float) -> None:
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a tensor")
    if values.ndim < 2 or query.ndim != 1:
        raise ValueError("expected values [S,...,D] and query [R]")
    if not 1 <= values.shape[0] <= 129 or not 1 <= values.shape[-1] <= 8192:
        raise ValueError("supported envelope is 1<=S<=129 and 1<=D<=8192")
    if any(d < 1 for d in values.shape) or not 1 <= query.numel() <= values.shape[-1]:
        raise ValueError("dimensions must be positive and 1<=R<=D")
    if values.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("values must have BF16 or FP32 storage")
    if query.dtype not in (torch.bfloat16, torch.float32) or query.device != values.device:
        raise TypeError("query must be BF16/FP32 on the values device")
    _validate_scalar(eps, "eps", positive=True)
    _validate_scalar(scale, "scale")


def attnres(
    values: Tensor | list[Tensor] | tuple[Tensor, ...],
    query: Tensor,
    *,
    eps: float = EPS,
    scale: float = 1.0,
) -> Tensor:
    """Mix source values with a static query over implicit tail keys.

    ``values`` may be a packed ``[S, ..., D]`` tensor or an ordered
    list/tuple of ``S`` tensors shaped ``[..., D]``.  ``query`` has shape
    ``[R]``; its keys are the final ``R`` coordinates of every source, and
    the returned mixture retains the full value width ``D``.  ``R == D`` is
    standard AttnRes and ``R < D`` is sliced LR-AttnRes.

    BF16 and FP32 storage use FP32 equation math. CPU calls use the explicit
    reference implementation. CUDA packed calls use the fixed-tail Triton
    route; source-list calls use its list adapter, which routes bounded BF16
    cases through the FLA-derived source-list kernels and other cases through
    the fixed-tail fallback.
    """
    if isinstance(values, torch.Tensor):
        _validate(values, query, eps, scale)
        if not values.is_cuda:
            return reference_attnres(values, query, eps=eps, scale=scale)
        from ._kernels.fixed_tail import fused_attnres

        return fused_attnres(values, query, eps=eps, scale=scale)

    source_tuple = validate_sources(values, query, eps, scale)
    if not source_tuple[0].is_cuda:
        return reference_attnres(torch.stack(source_tuple), query, eps=eps, scale=scale)

    # Keep Triton and every CUDA kernel module lazy on CPU-only installations.
    from ._kernels.fixed_tail_sources import source_attnres

    return source_attnres(source_tuple, query, eps=eps, scale=scale)


__all__ = ["attnres"]
