"""Tensor[] source-list adapter for the fixed-tail Triton core."""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
from typing import Any

import torch

from .._sources import validate_sources
from . import fixed_tail

_EPS = fixed_tail._EPS


def _validate_bf16_cuda_sources(
    sources: Sequence[torch.Tensor], query: torch.Tensor
) -> None:
    """Enforce the narrow production boundary before entering Triton."""

    if query.dtype != torch.bfloat16:
        raise TypeError("query must use BF16 storage")
    if not sources or any(source.dtype != torch.bfloat16 for source in sources):
        raise TypeError("sources must use BF16 storage")
    if any(not source.is_cuda for source in sources):
        raise RuntimeError("source_attnres requires CUDA BF16 tensors")


try:
    from torch.library import custom_op as _custom_op
    from torch.library import register_autograd as _register_autograd
    from torch.library import register_fake as _register_fake
except (ImportError, AttributeError):  # pragma: no cover - unsupported torch.
    _custom_op = None
    _register_autograd = None
    _register_fake = None


_SOURCE_CUSTOM_OP_KWARGS: dict[str, object] = {}
if _custom_op is not None:
    try:
        _custom_op_supports_tags = "tags" in inspect.signature(_custom_op).parameters
    except (TypeError, ValueError):  # pragma: no cover - unusual wrapped API.
        _custom_op_supports_tags = False
    if (
        _custom_op_supports_tags
        and hasattr(torch, "Tag")
        and hasattr(torch.Tag, "flexible_layout")
    ):
        _SOURCE_CUSTOM_OP_KWARGS = {"tags": (torch.Tag.flexible_layout,)}


def _source_rows_view(source: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    return source.reshape(rows, width)


def _source_pointer_args(
    sources: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], tuple[int, ...], int]:
    source_tuple = tuple(sources)
    if not source_tuple:
        raise ValueError("sources must be nonempty")
    length = len(source_tuple)
    l2 = fixed_tail._next_power_of_two(length)
    pointers = source_tuple + (source_tuple[0],) * (l2 - length)
    return (pointers, tuple(int(source.stride(0)) for source in pointers),
            tuple(int(source.stride(1)) for source in pointers), l2)


_SOURCE_RECORD_FIELDS = 3
_SOURCE_RECORD_ITEMSIZE = 8


def _source_record_bytes(l2: int, tables: int = 1) -> int:
    return int(tables) * int(l2) * _SOURCE_RECORD_FIELDS * _SOURCE_RECORD_ITEMSIZE


def _uniform_stride(strides: tuple[int, ...]) -> tuple[bool, int]:
    value = int(strides[0])
    return all(int(stride) == value for stride in strides), value


def _setup_source_records(
    pointers: tuple[torch.Tensor, ...],
    row_strides: tuple[int, ...],
    feature_strides: tuple[int, ...],
    l2: int,
    *,
    grad_pointers: tuple[torch.Tensor, ...] | None = None,
    grad_row_strides: tuple[int, ...] | None = None,
    grad_feature_strides: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    has_grad_values = grad_pointers is not None
    if has_grad_values:
        if grad_row_strides is None or grad_feature_strides is None:
            raise ValueError("gradient record strides are required")
        if len(grad_pointers) != l2:
            raise ValueError("gradient record tuple must match source padding")
        table_count = 2
    else:
        if grad_row_strides is not None or grad_feature_strides is not None:
            raise ValueError("gradient record strides require gradient pointers")
        grad_pointers = pointers
        grad_row_strides = row_strides
        grad_feature_strides = feature_strides
        table_count = 1

    records = torch.empty(
        (_source_record_bytes(l2, table_count) // _SOURCE_RECORD_ITEMSIZE,),
        device=pointers[0].device,
        dtype=torch.int64,
    )
    fixed_tail._source_record_setup[(1,)](
        pointers,
        grad_pointers,
        records,
        L2=l2,
        HAS_GRAD_VALUES=has_grad_values,
        ROW_STRIDES=row_strides,
        FEATURE_STRIDES=feature_strides,
        GRAD_ROW_STRIDES=grad_row_strides,
        GRAD_FEATURE_STRIDES=grad_feature_strides,
        num_warps=1,
        num_stages=1,
    )
    source_end = l2 * _SOURCE_RECORD_FIELDS
    source_records = records[:source_end]
    grad_records = records[source_end:] if has_grad_values else None
    return source_records, grad_records


def _launch_source_forward(sources: Sequence[torch.Tensor], query: torch.Tensor,
                           eps: float, scale: float):
    source_tuple = tuple(sources)
    _validate_bf16_cuda_sources(source_tuple, query)
    if fixed_tail.triton is None:
        raise RuntimeError("fixed-tail source kernels require Triton on CUDA")
    from . import fla_full_sources

    if fla_full_sources.supports(source_tuple):
        output = fla_full_sources.forward(source_tuple, query, eps, scale)
        return output[0], output[1], output[2], output[3], output[4]
    if not source_tuple or any(source.ndim != 2 for source in source_tuple):
        raise ValueError("fixed-tail source kernels require flattened [rows,D] sources")
    rows, width, rank = int(source_tuple[0].shape[0]), int(source_tuple[0].shape[1]), int(query.numel())
    pointers, row_strides, feature_strides, l2 = _source_pointer_args(source_tuple)
    row_stride_uniform, row_stride = _uniform_stride(row_strides)
    feature_stride_uniform, feature_stride = _uniform_stride(feature_strides)
    strides_uniform = row_stride_uniform and feature_stride_uniform
    first = source_tuple[0]
    source_arg = _setup_source_records(
        pointers, row_strides, feature_strides, l2
    )[0]
    output = torch.empty((rows, width), device=first.device, dtype=first.dtype)
    saved_output_fp32 = torch.empty((rows, width), device=first.device, dtype=torch.float32)
    saved_key_inv_rms = torch.empty((len(source_tuple), rows), device=first.device,
                                    dtype=torch.float32)
    saved_logit = torch.empty_like(saved_key_inv_rms)
    saved_lse = torch.empty((rows,), device=first.device, dtype=torch.float32)
    source_tile, fuse_key = fixed_tail._launch_policy(width, rank, query.device)
    fixed_tail._packed_online_forward_kernel[(rows,)](
        source_arg,
        query,
        output,
        saved_output_fp32,
        saved_key_inv_rms,
        saved_logit,
        saved_lse,
        rows,
        len(source_tuple),
        float(eps),
        float(scale),
        D=width,
        R=rank,
        BLOCK_D=fixed_tail._next_power_of_two(width),
        BLOCK_R=fixed_tail._next_power_of_two(rank),
        SOURCE_TILE=source_tile,
        QUERY_STRIDE=int(query.stride(0)),
        OUTPUT_ROW_STRIDE=int(output.stride(0)),
        OUTPUT_D_STRIDE=int(output.stride(1)),
        L2=l2,
        ROW_STRIDES=row_strides,
        FEATURE_STRIDES=feature_strides,
        LIST_SOURCES=True,
        SOURCE_RECORDS=True,
        VALUE_DTYPE=fixed_tail.tl.bfloat16,
        SOURCE_STRIDES_UNIFORM=strides_uniform,
        SOURCE_ROW_STRIDE=row_stride,
        SOURCE_FEATURE_STRIDE=feature_stride,
        FUSE_KEY_WITH_VALUE=fuse_key,
        num_warps=fixed_tail.NUM_WARPS,
        num_stages=fixed_tail.NUM_STAGES,
    )
    return output, saved_output_fp32, saved_key_inv_rms, saved_logit, saved_lse


def _launch_source_backward(sources: Sequence[torch.Tensor], query: torch.Tensor,
                            saved_output_fp32: torch.Tensor, grad_output: torch.Tensor,
                            saved_key_inv_rms: torch.Tensor, saved_logit: torch.Tensor,
                            saved_lse: torch.Tensor, scale: float) -> list[torch.Tensor]:
    source_tuple = tuple(sources)
    _validate_bf16_cuda_sources(source_tuple, query)
    if fixed_tail.triton is None:
        raise RuntimeError("fixed-tail source kernels require Triton on CUDA")
    from . import fla_full_sources

    if fla_full_sources.supports(source_tuple):
        return fla_full_sources.backward(
            source_tuple,
            query,
            saved_output_fp32,
            grad_output,
            saved_key_inv_rms,
            saved_logit,
            saved_lse,
            scale,
        )
    if not source_tuple or any(source.ndim != 2 for source in source_tuple):
        raise ValueError("fixed-tail source kernels require flattened [rows,D] sources")
    rows, width, rank = int(source_tuple[0].shape[0]), int(source_tuple[0].shape[1]), int(query.numel())
    pointers, row_strides, feature_strides, l2 = _source_pointer_args(source_tuple)
    row_stride_uniform, row_stride = _uniform_stride(row_strides)
    feature_stride_uniform, feature_stride = _uniform_stride(feature_strides)
    strides_uniform = row_stride_uniform and feature_stride_uniform
    grad_values = [
        torch.empty_like(source, memory_format=torch.contiguous_format)
        for source in source_tuple
    ]
    grad_pointers, grad_row_strides, grad_feature_strides, grad_l2 = _source_pointer_args(grad_values)
    if grad_l2 != l2:
        raise RuntimeError("source gradient pointer lengths disagree")
    grad_row_stride_uniform, grad_row_stride = _uniform_stride(grad_row_strides)
    grad_feature_stride_uniform, grad_feature_stride = _uniform_stride(grad_feature_strides)
    grad_strides_uniform = grad_row_stride_uniform and grad_feature_stride_uniform
    source_arg, grad_arg = _setup_source_records(
        pointers,
        row_strides,
        feature_strides,
        l2,
        grad_pointers=grad_pointers,
        grad_row_strides=grad_row_strides,
        grad_feature_strides=grad_feature_strides,
    )
    grad_query_token = torch.empty(
        (rows, rank), device=query.device, dtype=torch.float32
    )
    grad_query_fp32 = torch.empty((rank,), device=query.device, dtype=torch.float32)
    source_tile, fuse_key = fixed_tail._launch_policy(width, rank, query.device)
    fixed_tail._packed_online_backward_kernel[(rows,)](
        source_arg,
        query,
        saved_output_fp32,
        grad_output,
        saved_key_inv_rms,
        saved_logit,
        saved_lse,
        grad_arg,
        grad_query_token,
        rows,
        len(source_tuple),
        float(scale),
        D=width,
        R=rank,
        BLOCK_D=fixed_tail._next_power_of_two(width),
        BLOCK_R=fixed_tail._next_power_of_two(rank),
        SOURCE_TILE=source_tile,
        QUERY_STRIDE=int(query.stride(0)),
        GRAD_OUTPUT_ROW_STRIDE=int(grad_output.stride(0)),
        GRAD_OUTPUT_D_STRIDE=int(grad_output.stride(1)),
        L2=l2,
        ROW_STRIDES=row_strides,
        FEATURE_STRIDES=feature_strides,
        GRAD_ROW_STRIDES=grad_row_strides,
        GRAD_FEATURE_STRIDES=grad_feature_strides,
        LIST_SOURCES=True,
        SOURCE_RECORDS=True,
        VALUE_DTYPE=fixed_tail.tl.bfloat16,
        SOURCE_STRIDES_UNIFORM=strides_uniform,
        SOURCE_ROW_STRIDE=row_stride,
        SOURCE_FEATURE_STRIDE=feature_stride,
        GRAD_STRIDES_UNIFORM=grad_strides_uniform,
        GRAD_ROW_STRIDE=grad_row_stride,
        GRAD_FEATURE_STRIDE=grad_feature_stride,
        FUSE_KEY_WITH_VALUE=fuse_key,
    )
    fixed_tail._packed_query_reduce_kernel[(
        fixed_tail.triton.cdiv(rank, fixed_tail.QUERY_BLOCK_R),
    )](
        grad_query_token,
        grad_query_fp32,
        rows,
        R=rank,
        BLOCK_N=fixed_tail.QUERY_BLOCK_N,
        BLOCK_R=fixed_tail.QUERY_BLOCK_R,
        num_warps=fixed_tail.NUM_WARPS,
        num_stages=fixed_tail.NUM_STAGES,
    )
    return [*grad_values, grad_query_fp32.to(query.dtype)]


if _custom_op is not None:

    @_custom_op("attnres::_fixed_tail_sources_forward_with_aux", mutates_args=(),
                device_types="cuda", **_SOURCE_CUSTOM_OP_KWARGS)
    def _fixed_tail_sources_forward_with_aux_custom_op(
        sources: list[torch.Tensor], query: torch.Tensor, eps: float, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _launch_source_forward(sources, query, eps, scale)

    @_custom_op("attnres::_fixed_tail_sources_backward", mutates_args=(),
                device_types="cuda", **_SOURCE_CUSTOM_OP_KWARGS)
    def _fixed_tail_sources_backward_custom_op(
        sources: list[torch.Tensor], query: torch.Tensor,
        saved_output_fp32: torch.Tensor, grad_output: torch.Tensor,
        saved_key_inv_rms: torch.Tensor, saved_logit: torch.Tensor,
        saved_lse: torch.Tensor, scale: float,
    ) -> list[torch.Tensor]:
        return _launch_source_backward(
            sources,
            query,
            saved_output_fp32,
            grad_output,
            saved_key_inv_rms,
            saved_logit,
            saved_lse,
            scale,
        )

    def _source_forward_fake(
        sources: list[torch.Tensor], query: torch.Tensor, eps: float, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del eps, scale
        first = sources[0]
        rows = math.prod(first.shape[:-1]) or 1
        width = first.shape[-1]
        from . import fla_full_sources

        standard = (
            fla_full_sources.supports(sources, int(width))
            and fla_full_sources._standard_path(
                sources, query, int(width), int(query.numel())
            )
        )
        save_mixed = not standard or fla_full_sources._should_save_mixed(
            sources, int(rows), int(width)
        )
        # Keep the auxiliary rank stable across source-count-specialized calls
        # in one AOT graph.  The recompute checkpoint still allocates zero
        # elements, while both checkpoint policies expose a [rows, D] ABI.
        mixed_shape = (rows, width) if save_mixed else (0, width)
        stats = first.new_empty((len(sources), rows), dtype=torch.float32)
        return (
            first.new_empty(first.shape),
            first.new_empty(mixed_shape, dtype=torch.float32),
            stats,
            first.new_empty(stats.shape, dtype=torch.float32),
            first.new_empty((rows,), dtype=torch.float32),
        )

    def _source_backward_fake(
        sources: list[torch.Tensor], query: torch.Tensor,
        saved_output_fp32: torch.Tensor, grad_output: torch.Tensor,
        saved_key_inv_rms: torch.Tensor, saved_logit: torch.Tensor,
        saved_lse: torch.Tensor, scale: float
    ) -> list[torch.Tensor]:
        del saved_output_fp32, grad_output, saved_key_inv_rms, saved_logit
        del saved_lse, scale
        return [source.new_empty(source.shape) for source in sources] + [
            query.new_empty(query.shape)
        ]

    if _register_fake is not None:
        _register_fake(_fixed_tail_sources_forward_with_aux_custom_op)(
            _source_forward_fake
        )
        _register_fake(_fixed_tail_sources_backward_custom_op)(_source_backward_fake)

    def _source_setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]) -> None:
        sources, query, _eps, scale = inputs
        _output, saved_output_fp32, saved_key_inv_rms, saved_logit, saved_lse = output
        ctx.save_for_backward(*sources, query, saved_output_fp32, saved_key_inv_rms,
                              saved_logit, saved_lse)
        ctx.source_count = len(sources)
        ctx.scale = scale

    def _source_backward(ctx: Any, grad_output: torch.Tensor | None,
                         _grad_saved_output_fp32: torch.Tensor | None = None,
                         _grad_saved_key_inv_rms: torch.Tensor | None = None,
                         _grad_saved_logit: torch.Tensor | None = None,
                         _grad_saved_lse: torch.Tensor | None = None) -> tuple[Any, None, None, None]:
        if grad_output is None:
            return [None] * ctx.source_count, None, None, None
        saved = ctx.saved_tensors
        source_count = ctx.source_count
        sources = list(saved[:source_count])
        query = saved[source_count]
        saved_output_fp32 = saved[source_count + 1]
        saved_key_inv_rms = saved[source_count + 2]
        saved_logit = saved[source_count + 3]
        saved_lse = saved[source_count + 4]
        gradients = _fixed_tail_sources_backward_custom_op(
            sources, query, saved_output_fp32, grad_output, saved_key_inv_rms,
            saved_logit, saved_lse, float(ctx.scale))
        return gradients[:source_count], gradients[source_count], None, None

    if _register_autograd is not None:
        _register_autograd(
            _fixed_tail_sources_forward_with_aux_custom_op,
            _source_backward,
            setup_context=_source_setup_context,
        )
else:
    _fixed_tail_sources_forward_with_aux_custom_op = None
    _fixed_tail_sources_backward_custom_op = None


def source_attnres(sources, query: torch.Tensor, *, eps: float = _EPS,
                   scale: float = 1.0) -> torch.Tensor:
    """Mix a BF16 source sequence through the fixed-tail CUDA route."""
    source_tuple = validate_sources(sources, query, eps, scale)
    _validate_bf16_cuda_sources(source_tuple, query)
    first = source_tuple[0]
    width = int(first.shape[-1])
    rows = first.numel() // width
    if _fixed_tail_sources_forward_with_aux_custom_op is None:
        raise RuntimeError("source_attnres requires Triton on CUDA")
    from . import fla_full_sources

    if fla_full_sources.supports(source_tuple, width):
        prepared = source_tuple
        reshape_output = False
    else:
        prepared = tuple(_source_rows_view(source, rows, width) for source in source_tuple)
        reshape_output = True
    output, _saved_output, _saved_key_inv_rms, _saved_logit, _saved_lse = (
        _fixed_tail_sources_forward_with_aux_custom_op(
            list(prepared),
            query,
            float(eps),
            float(scale),
        )
    )
    return output.reshape(tuple(first.shape)) if reshape_output else output


__all__ = ["source_attnres"]
