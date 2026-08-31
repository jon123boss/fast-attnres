"""Shared validation and normalization for source-container APIs."""

from __future__ import annotations

import math
from numbers import Real

import torch


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)
_MAX_SOURCES = 129
_MAX_WIDTH = 8192


def _normalize_sources(values, name: str) -> tuple[torch.Tensor, ...]:
    """Normalize a packed tensor or source sequence without copying tensors."""

    if isinstance(values, torch.Tensor):
        if values.ndim < 2:
            raise ValueError(f"{name} must have shape [S,...,D]")
        return tuple(values.unbind(0))
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a tensor or a list/tuple of tensors")
    return tuple(values)


def _validate_scalar(value, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")


def validate_sources(
    values,
    query,
    eps,
    scale,
    *,
    query_ndim: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Normalize and validate packed or per-source residual containers.

    The returned tuples contain the original tensors (or views produced by
    ``unbind(0)`` for packed inputs).  No stack, concatenation, contiguity
    conversion, or other tensor materialization occurs here.  Source tensors
    may therefore retain independent physical strides.
    """

    if isinstance(query_ndim, bool) or not isinstance(query_ndim, int) or query_ndim < 1:
        raise ValueError("query_ndim must be a positive integer")
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a tensor")
    if query.ndim != query_ndim:
        raise ValueError(f"query must have {query_ndim} dimensions")
    if any(int(size) < 1 for size in query.shape):
        raise ValueError("query dimensions must be positive")
    if query.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("query must use BF16 or FP32 storage")

    source_tuple = _normalize_sources(values, "values")
    if not 1 <= len(source_tuple) <= _MAX_SOURCES:
        raise ValueError(f"supported source envelope is 1<=S<={_MAX_SOURCES}")
    if any(not isinstance(source, torch.Tensor) for source in source_tuple):
        raise TypeError("values must contain only tensors")

    first = source_tuple[0]
    if first.ndim < 1:
        raise ValueError("each source must have shape [...,D]")
    if any(int(size) < 1 for size in first.shape):
        raise ValueError("source dimensions must be positive")
    width = int(first.shape[-1])
    if not 1 <= width <= _MAX_WIDTH:
        raise ValueError(f"supported value envelope is 1<=D<={_MAX_WIDTH}")
    if first.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("values must use BF16 or FP32 storage")
    if query.device != first.device:
        raise TypeError("query must be on the values device")

    for index, source in enumerate(source_tuple):
        if source.ndim < 1:
            raise ValueError(f"source {index} must have shape [...,D]")
        if tuple(source.shape) != tuple(first.shape):
            raise ValueError("all source tensors must have the same logical shape")
        if source.dtype != first.dtype or source.device != first.device:
            raise TypeError("all source tensors must match the first source dtype/device")

    rank = int(query.shape[-1])
    if not 1 <= rank <= width:
        raise ValueError("query rank must satisfy 1<=R<=D")
    _validate_scalar(eps, "eps", positive=True)
    _validate_scalar(scale, "scale")

    return source_tuple


__all__ = ["validate_sources"]
