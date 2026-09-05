"""PyTorch autograd boundary for the shared source-list kernels."""

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


def _launch_source_forward(sources: Sequence[torch.Tensor], query: torch.Tensor,
                           eps: float, scale: float):
    _validate_bf16_cuda_sources(sources, query)
    from . import fla_full_sources

    return tuple(fla_full_sources.forward(sources, query, eps, scale))


def _launch_source_backward(sources: Sequence[torch.Tensor], query: torch.Tensor,
                            saved_output_fp32: torch.Tensor, grad_output: torch.Tensor,
                            saved_key_inv_rms: torch.Tensor, saved_logit: torch.Tensor,
                            saved_lse: torch.Tensor, scale: float) -> list[torch.Tensor]:
    _validate_bf16_cuda_sources(sources, query)
    from . import fla_full_sources

    return fla_full_sources.backward(
        sources, query, saved_output_fp32, grad_output,
        saved_key_inv_rms, saved_logit, saved_lse, scale,
    )


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

        save_mixed = fla_full_sources._should_save_mixed(sources, int(rows), int(width))
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
    if _fixed_tail_sources_forward_with_aux_custom_op is None:
        raise RuntimeError("source_attnres requires Triton on CUDA")
    return _fixed_tail_sources_forward_with_aux_custom_op(
        list(source_tuple), query, float(eps), float(scale),
    )[0]


__all__ = ["source_attnres"]
