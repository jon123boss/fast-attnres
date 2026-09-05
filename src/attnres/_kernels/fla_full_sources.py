# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# This file is an isolated LR-AttnRes adaptation of
# flash-linear-attention commit 5e02dd3a7651f5f2797eb8b12bbec401826031e1,
# specifically ``fla/ops/attnres/fused.py``.  The FLA attribution and MIT
# notice above are retained; the kernels below are not the native FLA API.

"""Shared source-list kernels adapted from FLA.

The pinned FLA implementation uses one pointer per residual source and a
small source tile (``BL``) for its online softmax and value backward.  This
adaptation keeps those two structural choices while matching the LR-AttnRes
source-list ABI:

* values are full-width ``D`` vectors;
* an implicit key is the final ``R`` coordinates, with ``R <= D``;
* key RMS normalization has a parameter-free unit weight;
* the learned query is a BF16 vector;
* the saved logit is the already-scaled FP32 logit used by the existing
  source custom-op backward;
* the sliced key derivative is folded into one full-width value gradient
  before the BF16 store.

The same source kernels handle every supported width, rank, and physical
layout. ``fixed_tail_sources.py`` supplies the PyTorch autograd boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

try:  # Triton remains optional for CPU import and equation-reference tests.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised by CPU-only environments.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


_QUERY_REDUCE_MIN_BLOCK = 128
_QUERY_REDUCE_MAX_BLOCK = 1024
_QUERY_REDUCE_MAX_TILE = 32
_STANDARD_SOURCE_BLOCK_CONFIGS = None
_STANDARD_QUERY_REDUCE_CONFIGS = None

# Architecture, geometry, and physical strides separate autotune caches.
_STANDARD_AUTOTUNE_KEY = [
    "ARCH",
    "DTYPE",
    "D",
    "R",
    "L2",
    "ROW_BUCKET",
    "ROUTE",
    "CHECKPOINT",
]
_STANDARD_QUERY_AUTOTUNE_KEY = [
    "ARCH",
    "DTYPE",
    "D",
    "R",
    "N",
    "ROUTE",
    "CHECKPOINT",
]

_CONTIGUOUS_ROUTE = 1
_SAVE_MIXED_CHECKPOINT = 1
_RECOMPUTE_CHECKPOINT = 0
_AUTOTUNE_ROW_BUCKET_MAX = 8192


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (int(value) - 1).bit_length()


def _source_query_reduce_tile(rank: int) -> int:
    """Choose a bounded rank tile for the deterministic source reduction.

    The source backward writes one FP32 partial per flattened row.  A 32-lane
    reduction tile cuts the number of reduction programs for the common
    medium and large LR ranks while keeping the accumulator small enough for
    the one-program-per-rank-block kernel.  Very small ranks retain their
    natural power-of-two tile.
    """

    if rank < 1:
        raise ValueError("rank must be positive")
    return min(_QUERY_REDUCE_MAX_TILE, _next_power_of_two(rank))


def _source_query_reduce_block(count: int) -> int:
    """Use less masked work for short rows and a wider tile for long rows."""

    if count < 1:
        raise ValueError("count must be positive")
    return min(
        _QUERY_REDUCE_MAX_BLOCK,
        max(_QUERY_REDUCE_MIN_BLOCK, _next_power_of_two(count)),
    )


def _autotune_row_bucket(count: int) -> int:
    """Return a bounded power-of-two bucket for autotune cache keys."""

    if count < 1:
        raise ValueError("count must be positive")
    return min(_AUTOTUNE_ROW_BUCKET_MAX, _next_power_of_two(count))


def _architecture_id(device: torch.device | str | int) -> int:
    """Return the integer architecture token passed as a constexpr key."""

    index = device.index if isinstance(device, torch.device) else device
    major, minor = torch.cuda.get_device_capability(index)
    return int(major) * 10 + int(minor)


def _dtype_key(dtype: torch.dtype) -> int:
    """Encode the BF16-only runtime dtype as a cache-safe scalar."""

    if dtype != torch.bfloat16:
        raise TypeError("FLA source kernels require BF16 storage")
    return 0


def _should_save_mixed(
    sources: Sequence[torch.Tensor], count: int, width: int
) -> bool:
    """Choose save versus recompute from source cardinality, not result values.

    The production ABI recomputes for short lists (``S <= 3``) and saves the
    mixed value for larger lists (``S >= 4``).  This is a structural memory/
    reread trade-off that is stable across values and devices; the shape
    arguments are retained in the signature so callers can keep one policy
    hook if the storage envelope grows later.
    """

    if not sources:
        raise ValueError("sources must be nonempty")
    del count, width
    return len(sources) >= 4


def _is_row_affine(tensor: torch.Tensor) -> bool:
    """Whether flattening all batch dimensions has one row stride.

    A feature-strided tensor and a ``[..., :D]`` view of a contiguous
    ``[..., D+R]`` producer both satisfy this test.  A permuted batch layout
    does not and is compacted per source by ``_source_pointer_table``.
    """

    if tensor.ndim <= 2:
        return True
    row_stride = int(tensor.stride(-2))
    trailing = 1
    for dim in range(tensor.ndim - 3, -1, -1):
        trailing *= int(tensor.shape[dim + 1])
        if int(tensor.stride(dim)) != row_stride * trailing:
            return False
    return True


def _source_pointer_table(
    tensors: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], tuple[int, ...], int]:
    """Build an exact-length pointer tuple without packing the source list.

    Tensor pointers remain tensor arguments to Triton.  Only a source with a
    non-affine batch layout is compacted, and that copy is independent for
    that source.  The feature stride is retained for both affine and copied
    sources, including feature-strided views.
    """

    original = tuple(tensors)
    if not original:
        raise ValueError("sources must be nonempty")
    prepared = tuple(
        tensor if _is_row_affine(tensor) else tensor.contiguous()
        for tensor in original
    )
    # The selector uses a constexpr loop bound but does not require a
    # power-of-two pointer table.  Keep exactly one pointer per real source so
    # short lists and the 129-source envelope do not pay for masked selector
    # lanes that can never be addressed by a valid source id.
    length = len(prepared)
    row_strides = tuple(
        0 if tensor.ndim <= 1 else int(tensor.stride(-2)) for tensor in prepared
    )
    feature_strides = tuple(int(tensor.stride(-1)) for tensor in prepared)
    return prepared, row_strides, feature_strides, length


def _row_layout(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Return a tensor and affine row/feature strides for a flattened view."""

    prepared = tensor if _is_row_affine(tensor) else tensor.contiguous()
    row_stride = 0 if prepared.ndim <= 1 else int(prepared.stride(-2))
    return prepared, row_stride, int(prepared.stride(-1))


if triton is not None:
    # FLA's production standard/full-width route uses the complete BL x warp
    # x pipeline set on both SM90 (H100) and SM100 (B200).  Architecture is a
    # cache key, rather than a result-derived config filter, so both targets
    # see the same candidate set and tune independently.
    _STANDARD_SOURCE_BLOCK_CONFIGS = [
        triton.Config(
            {"BL": block, "PIPELINE_STAGES": stages},
            num_warps=warps,
            num_stages=stages,
        )
        for block in (1, 2, 4, 8)
        for warps in (4, 8, 16)
        for stages in (2, 3)
    ]
    _STANDARD_QUERY_REDUCE_CONFIGS = [
        triton.Config(
            {"BN": block_n, "BD": block_d, "PIPELINE_STAGES": stages},
            num_warps=warps,
            num_stages=stages,
        )
        for block_n, block_d, warps in (
            (1024, 16, 4),
            (2048, 32, 4),
            (2048, 32, 8),
            (4096, 32, 8),
            (4096, 64, 8),
        )
        for stages in (3, 4)
    ]


if triton is not None:

    @triton.jit
    def _select_source_pointer(
        values,
        source,
        row,
        row_strides: tl.constexpr,
        feature_strides: tl.constexpr,
        L2: tl.constexpr,
    ):
        """Select a source base while retaining its physical feature stride."""

        # Preserve the source tile even when the selector has one pointer.
        offsets = tl.full(source.shape, 0, tl.int64)
        pointer = values[0] + offsets
        row_stride = offsets + tl.cast(row_strides[0], tl.int64)
        feature_stride = offsets + tl.cast(feature_strides[0], tl.int64)
        for source_index in tl.static_range(1, L2):
            selected = source == source_index
            pointer = tl.where(selected, values[source_index], pointer)
            row_stride = tl.where(
                selected,
                tl.cast(row_strides[source_index], tl.int64),
                row_stride,
            )
            feature_stride = tl.where(
                selected,
                tl.cast(feature_strides[source_index], tl.int64),
                feature_stride,
            )
        return pointer + row * row_stride, feature_stride


    @triton.autotune(
        configs=_STANDARD_SOURCE_BLOCK_CONFIGS,
        key=_STANDARD_AUTOTUNE_KEY + ["ROW_STRIDES", "FEATURE_STRIDES", "QUERY_STRIDE"],
    )
    @triton.jit(do_not_specialize=["count", "sources"])
    def _fla_standard_forward_kernel(
        values,
        query,
        output,
        saved_mixed,
        saved_inv_rms,
        saved_logit,
        saved_lse,
        count,
        sources,
        eps,
        scale,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BL: tl.constexpr,
        PIPELINE_STAGES: tl.constexpr,
        ARCH: tl.constexpr,
        ROW_BUCKET: tl.constexpr,
        DTYPE: tl.constexpr,
        ROUTE: tl.constexpr,
        CHECKPOINT: tl.constexpr,
        L2: tl.constexpr,
        ROW_STRIDES: tl.constexpr,
        FEATURE_STRIDES: tl.constexpr,
        QUERY_STRIDE,
        OUTPUT_ROW_STRIDE,
        OUTPUT_D_STRIDE,
    ):
        """Full-width value reduction with an exact masked routing tail."""

        # ARCH, ROW_BUCKET, DTYPE, ROUTE, and CHECKPOINT values are constexpr
        # cache dimensions.  The
        # dispatcher supplies exact source strides and tail masks, so
        # no result-dependent or GPU-specific config filter is needed here.
        row = tl.program_id(0).to(tl.int64)
        count64 = tl.cast(count, tl.int64)
        source_count64 = tl.cast(sources, tl.int64)
        d_offsets = tl.arange(0, BLOCK_D).to(tl.int64)
        d_mask = d_offsets < D
        key_mask = d_mask & (d_offsets >= D - R)
        key_offsets = d_offsets - (D - R)
        row_valid = row < count64
        eps_f32 = tl.cast(eps, tl.float32)
        scale_f32 = tl.cast(scale, tl.float32)
        query_value = tl.load(
            query + key_offsets * tl.cast(QUERY_STRIDE, tl.int64),
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)

        running_max = tl.full([], -float("inf"), tl.float32)
        running_denom = tl.zeros([], tl.float32)
        running_output = tl.zeros((BLOCK_D,), tl.float32)
        for source_base in tl.range(
            0, sources, BL, num_stages=PIPELINE_STAGES
        ):
            source_offsets = tl.arange(0, BL).to(tl.int64)
            source_ids = tl.cast(source_base, tl.int64) + source_offsets
            source_mask = source_ids < source_count64
            value_base, value_stride = _select_source_pointer(
                values,
                source_ids,
                row,
                ROW_STRIDES,
                FEATURE_STRIDES,
                L2,
            )
            value_ptr = value_base[:, None] + d_offsets[None, :] * value_stride[:, None]
            value = tl.load(
                value_ptr,
                mask=source_mask[:, None] & row_valid & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            inverse_rms = tl.rsqrt(tl.sum(tl.where(d_offsets[None, :] >= D - R, value * value, 0.0), axis=1) / R + eps_f32)
            saved_score = (
                tl.sum(value * query_value[None, :], axis=1)
                * inverse_rms
                * scale_f32
            )
            tile_scores = tl.where(source_mask & row_valid, saved_score, -float("inf"))
            new_max = tl.maximum(running_max, tl.max(tile_scores, axis=0))
            old_scale = tl.exp(running_max - new_max)
            probability_numerator = tl.exp(tile_scores - new_max)
            probability_numerator = tl.where(
                source_mask & row_valid, probability_numerator, 0.0
            )
            running_denom = running_denom * old_scale + tl.sum(
                probability_numerator, axis=0
            )
            running_output = running_output * old_scale + tl.sum(
                probability_numerator[:, None] * value, axis=0
            )

            metadata = source_ids * count64 + row
            tl.store(saved_inv_rms + metadata, inverse_rms, mask=source_mask & row_valid)
            tl.store(saved_logit + metadata, saved_score, mask=source_mask & row_valid)
            running_max = new_max

        mixed = running_output / running_denom
        tl.store(saved_lse + row, running_max + tl.log(running_denom), mask=row_valid)
        if CHECKPOINT:
            tl.store(
                saved_mixed + row * D + d_offsets,
                mixed,
                mask=row_valid & d_mask,
            )
        output_ptr = (
            output
            + row * tl.cast(OUTPUT_ROW_STRIDE, tl.int64)
            + d_offsets * tl.cast(OUTPUT_D_STRIDE, tl.int64)
        )
        tl.store(output_ptr, mixed, mask=row_valid & d_mask)


    @triton.autotune(
        configs=_STANDARD_SOURCE_BLOCK_CONFIGS,
        key=_STANDARD_AUTOTUNE_KEY + ["VALUE_ROW_STRIDES", "VALUE_FEATURE_STRIDES",
                                      "QUERY_STRIDE", "GRAD_OUTPUT_ROW_STRIDE",
                                      "GRAD_OUTPUT_D_STRIDE"],
    )
    @triton.jit(do_not_specialize=["count", "sources"])
    def _fla_standard_backward_kernel(
        values,
        query,
        saved_mixed,
        grad_output,
        saved_inv_rms,
        saved_logit,
        saved_lse,
        grad_values,
        grad_query_partial,
        count,
        sources,
        scale,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BL: tl.constexpr,
        PIPELINE_STAGES: tl.constexpr,
        ARCH: tl.constexpr,
        ROW_BUCKET: tl.constexpr,
        DTYPE: tl.constexpr,
        ROUTE: tl.constexpr,
        CHECKPOINT: tl.constexpr,
        L2: tl.constexpr,
        VALUE_ROW_STRIDES: tl.constexpr,
        VALUE_FEATURE_STRIDES: tl.constexpr,
        GRAD_VALUE_ROW_STRIDES: tl.constexpr,
        GRAD_VALUE_FEATURE_STRIDES: tl.constexpr,
        QUERY_STRIDE,
        GRAD_OUTPUT_ROW_STRIDE,
        GRAD_OUTPUT_D_STRIDE,
    ):
        """Source and query gradients with one BF16 store per source role."""

        row = tl.program_id(0).to(tl.int64)
        count64 = tl.cast(count, tl.int64)
        source_count64 = tl.cast(sources, tl.int64)
        d_offsets = tl.arange(0, BLOCK_D).to(tl.int64)
        d_mask = d_offsets < D
        key_mask = d_mask & (d_offsets >= D - R)
        key_offsets = d_offsets - (D - R)
        row_valid = row < count64
        scale_f32 = tl.cast(scale, tl.float32)
        query_value = tl.load(
            query + key_offsets * tl.cast(QUERY_STRIDE, tl.int64),
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        if CHECKPOINT:
            grad = tl.load(
                grad_output
                + row * tl.cast(GRAD_OUTPUT_ROW_STRIDE, tl.int64)
                + d_offsets * tl.cast(GRAD_OUTPUT_D_STRIDE, tl.int64),
                mask=row_valid & d_mask,
                other=0.0,
            ).to(tl.float32)
            mixed = tl.load(
                saved_mixed + row * D + d_offsets,
                mask=row_valid & d_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            # Recompute the mixed value from saved logits.  This branch is a
            # compile-time policy choice and never depends on output values.
            mixed = tl.zeros((BLOCK_D,), tl.float32)
            for source_base in tl.range(
                0, sources, BL, num_stages=PIPELINE_STAGES
            ):
                source_offsets = tl.arange(0, BL).to(tl.int64)
                source_ids = tl.cast(source_base, tl.int64) + source_offsets
                source_mask = source_ids < source_count64
                value_base, value_stride = _select_source_pointer(
                    values,
                    source_ids,
                    row,
                    VALUE_ROW_STRIDES,
                    VALUE_FEATURE_STRIDES,
                    L2,
                )
                value_ptr = (
                    value_base[:, None]
                    + d_offsets[None, :] * value_stride[:, None]
                )
                value = tl.load(
                    value_ptr,
                    mask=source_mask[:, None] & row_valid & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                metadata = source_ids * count64 + row
                saved_score = tl.load(
                    saved_logit + metadata,
                    mask=source_mask & row_valid,
                    other=0.0,
                ).to(tl.float32)
                lse = tl.load(saved_lse + row, mask=row_valid, other=0.0).to(tl.float32)
                probability = tl.exp(saved_score - lse)
                probability = tl.where(source_mask & row_valid, probability, 0.0)
                mixed += tl.sum(probability[:, None] * value, axis=0)
            grad = tl.load(
                grad_output
                + row * tl.cast(GRAD_OUTPUT_ROW_STRIDE, tl.int64)
                + d_offsets * tl.cast(GRAD_OUTPUT_D_STRIDE, tl.int64),
                mask=row_valid & d_mask,
                other=0.0,
            ).to(tl.float32)

        delta = tl.sum(tl.where(d_mask, grad * mixed, 0.0), axis=0)
        lse = tl.load(saved_lse + row, mask=row_valid, other=0.0).to(tl.float32)
        grad_query = tl.zeros((BLOCK_D,), tl.float32)
        for source_base in tl.range(
            0, sources, BL, num_stages=PIPELINE_STAGES
        ):
            source_offsets = tl.arange(0, BL).to(tl.int64)
            source_ids = tl.cast(source_base, tl.int64) + source_offsets
            source_mask = source_ids < source_count64
            value_base, value_stride = _select_source_pointer(
                values,
                source_ids,
                row,
                VALUE_ROW_STRIDES,
                VALUE_FEATURE_STRIDES,
                L2,
            )
            grad_value_base, grad_value_stride = _select_source_pointer(
                grad_values,
                source_ids,
                row,
                GRAD_VALUE_ROW_STRIDES,
                GRAD_VALUE_FEATURE_STRIDES,
                L2,
            )
            value_ptr = value_base[:, None] + d_offsets[None, :] * value_stride[:, None]
            value = tl.load(
                value_ptr,
                mask=source_mask[:, None] & row_valid & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            metadata = source_ids * count64 + row
            inverse_rms = tl.load(
                saved_inv_rms + metadata,
                mask=source_mask & row_valid,
                other=1.0,
            ).to(tl.float32)
            saved_score = tl.load(
                saved_logit + metadata,
                mask=source_mask & row_valid,
                other=0.0,
            ).to(tl.float32)
            probability = tl.exp(saved_score - lse)
            probability = tl.where(source_mask & row_valid, probability, 0.0)
            dweight = tl.sum(value * grad[None, :], axis=1)
            dscore = probability * (dweight - delta)
            scaled_dscore = dscore * scale_f32
            normalized_key = tl.where(d_offsets[None, :] >= D - R, value * inverse_rms[:, None], 0.0)
            grad_query += tl.sum(
                scaled_dscore[:, None] * normalized_key,
                axis=0,
            )
            projection = dscore * saved_score / R
            grad_key = inverse_rms[:, None] * (
                scaled_dscore[:, None] * query_value[None, :]
                - normalized_key * projection[:, None]
            )
            grad_value = probability[:, None] * grad[None, :] + grad_key
            grad_ptr = (
                grad_value_base[:, None]
                + d_offsets[None, :] * grad_value_stride[:, None]
            )
            tl.store(
                grad_ptr,
                grad_value.to(grad_values[0].dtype.element_ty),
                mask=source_mask[:, None] & row_valid & d_mask[None, :],
            )
        tl.store(
            grad_query_partial + row * R + key_offsets,
            grad_query,
            mask=row_valid & key_mask,
        )


    @triton.autotune(
        configs=_STANDARD_QUERY_REDUCE_CONFIGS,
        key=_STANDARD_QUERY_AUTOTUNE_KEY,
    )
    @triton.jit(do_not_specialize=["count"])
    def _fla_standard_query_reduce_kernel(
        partial,
        grad_query,
        count,
        N: tl.constexpr,
        D: tl.constexpr,
        R: tl.constexpr,
        BN: tl.constexpr,
        BD: tl.constexpr,
        PIPELINE_STAGES: tl.constexpr,
        ARCH: tl.constexpr,
        DTYPE: tl.constexpr,
        L2: tl.constexpr,
        ROUTE: tl.constexpr,
        CHECKPOINT: tl.constexpr,
    ):
        """FLA-shaped tiled reduction of per-row FP32 query partials."""

        rank_block = tl.program_id(0).to(tl.int64)
        r_offsets = rank_block * BD + tl.arange(0, BD).to(tl.int64)
        r_mask = r_offsets < R
        accumulator = tl.zeros((BD,), tl.float32)
        for row_base in tl.range(
            0, N, BN, num_stages=PIPELINE_STAGES
        ):
            row_offsets = row_base.to(tl.int64) + tl.arange(0, BN).to(tl.int64)
            row_mask = row_offsets < tl.cast(count, tl.int64)
            partial_values = tl.load(
                partial + row_offsets[:, None] * R + r_offsets[None, :],
                mask=row_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(partial_values, axis=0)
        tl.store(grad_query + r_offsets, accumulator, mask=r_mask)


    @triton.jit(do_not_specialize=["count"])
    def _fla_query_reduce_kernel(
        partial,
        grad_query,
        count,
        R: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        SPLIT_N: tl.constexpr = 0,
    ):
        """Deterministically reduce one FP32 query partial per batch row."""

        rank_block = tl.program_id(0).to(tl.int64)
        r_offsets = rank_block * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
        r_mask = r_offsets < R
        accumulator = tl.zeros((BLOCK_R,), tl.float32)
        row_block = tl.program_id(1).to(tl.int64)
        begin = row_block * SPLIT_N
        finish = tl.minimum(count, begin + SPLIT_N) if SPLIT_N else count
        for row_base in tl.range(begin, finish, BLOCK_N, num_stages=2):
            row_offsets = row_base.to(tl.int64) + tl.arange(0, BLOCK_N).to(tl.int64)
            row_mask = row_offsets < tl.cast(count, tl.int64)
            partial_values = tl.load(
                partial + row_offsets[:, None] * R + r_offsets[None, :],
                mask=row_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(partial_values, axis=0)
        tl.store(grad_query + row_block * R + r_offsets, accumulator, mask=r_mask)


def _source_metadata(
    sources: Sequence[torch.Tensor], query: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], int, int, int]:
    source_tuple = tuple(sources)
    if not source_tuple:
        raise ValueError("sources must be nonempty")
    first = source_tuple[0]
    count = math.prod(int(size) for size in first.shape[:-1]) or 1
    return source_tuple, int(count), int(first.shape[-1]), int(query.numel())


def _validate_bf16_runtime(
    sources: Sequence[torch.Tensor],
    query: torch.Tensor,
    grad_output: torch.Tensor | None = None,
) -> None:
    """Enforce the CUDA operator contract at the FLA module boundary."""

    if any(source.dtype != torch.bfloat16 for source in sources):
        raise TypeError("FLA source kernels require BF16 values")
    if query.dtype != torch.bfloat16:
        raise TypeError("FLA source kernels require a BF16 query")
    if grad_output is not None and grad_output.dtype != torch.bfloat16:
        raise TypeError("FLA source kernels require BF16 operator gradients")


def _reduce_query(grad_query_partial, grad_query, count, rank):
    query_reduce_tile = _source_query_reduce_tile(rank)
    if count >= 2048:
        chunks = triton.cdiv(count, 256)
        grouped = torch.empty((chunks, rank), device=grad_query.device, dtype=torch.float32)
        _fla_query_reduce_kernel[(triton.cdiv(rank, query_reduce_tile), chunks)](
            grad_query_partial, grouped, count, R=rank, BLOCK_N=256,
            BLOCK_R=query_reduce_tile, SPLIT_N=256, num_warps=4, num_stages=2,
        )
        grad_query_partial, count = grouped, chunks
    _fla_query_reduce_kernel[(triton.cdiv(rank, query_reduce_tile),)](
        grad_query_partial,
        grad_query,
        count,
        R=rank,
        BLOCK_N=_source_query_reduce_block(count),
        BLOCK_R=query_reduce_tile,
        num_warps=8 if rank >= 1024 else 4,
        num_stages=2,
    )


def _launch_standard_forward(
    source_tuple: tuple[torch.Tensor, ...],
    query: torch.Tensor,
    count: int,
    width: int,
    rank: int,
    eps: float,
    scale: float,
) -> list[torch.Tensor]:
    pointers, row_strides, feature_strides, l2 = _source_pointer_table(source_tuple)
    first = source_tuple[0]
    save_mixed = _should_save_mixed(source_tuple, count, width)
    checkpoint = _SAVE_MIXED_CHECKPOINT if save_mixed else _RECOMPUTE_CHECKPOINT
    output = torch.empty_like(first, memory_format=torch.contiguous_format)
    saved_mixed = (
        torch.empty((count, width), device=first.device, dtype=torch.float32)
        if save_mixed
        else torch.empty((0, width), device=first.device, dtype=torch.float32)
    )
    saved_inv_rms = torch.empty(
        (len(source_tuple), count), device=first.device, dtype=torch.float32
    )
    saved_logit = torch.empty_like(saved_inv_rms)
    saved_lse = torch.empty((count,), device=first.device, dtype=torch.float32)
    _fla_standard_forward_kernel[(count,)](
        pointers,
        query,
        output,
        saved_mixed,
        saved_inv_rms,
        saved_logit,
        saved_lse,
        count,
        len(source_tuple),
        float(eps),
        float(scale),
        D=width,
        R=rank,
        BLOCK_D=_next_power_of_two(width),
        ARCH=_architecture_id(first.device),
        ROW_BUCKET=_autotune_row_bucket(count),
        DTYPE=_dtype_key(first.dtype),
        ROUTE=_CONTIGUOUS_ROUTE,
        CHECKPOINT=checkpoint,
        L2=l2,
        ROW_STRIDES=row_strides,
        FEATURE_STRIDES=feature_strides,
        QUERY_STRIDE=int(query.stride(0)),
        OUTPUT_ROW_STRIDE=0 if output.ndim <= 1 else int(output.stride(-2)),
        OUTPUT_D_STRIDE=int(output.stride(-1)),
    )
    return [output, saved_mixed, saved_inv_rms, saved_logit, saved_lse]


def _launch_standard_backward(
    source_tuple: tuple[torch.Tensor, ...],
    query: torch.Tensor,
    saved_mixed: torch.Tensor,
    grad_output: torch.Tensor,
    saved_inv_rms: torch.Tensor,
    saved_logit: torch.Tensor,
    saved_lse: torch.Tensor,
    count: int,
    width: int,
    rank: int,
    scale: float,
) -> list[torch.Tensor]:
    pointers, row_strides, feature_strides, l2 = _source_pointer_table(source_tuple)
    grad_output_prepared, grad_row_stride, grad_feature_stride = _row_layout(grad_output)
    grad_values = [
        torch.empty_like(source, memory_format=torch.contiguous_format)
        for source in source_tuple
    ]
    grad_pointers, grad_row_strides, grad_feature_strides, grad_l2 = _source_pointer_table(
        tuple(grad_values)
    )
    if grad_l2 != l2:
        raise RuntimeError("source gradient pointer tables disagree")
    save_mixed = bool(saved_mixed.numel())
    checkpoint = _SAVE_MIXED_CHECKPOINT if save_mixed else _RECOMPUTE_CHECKPOINT
    grad_query_partial = torch.empty(
        (count, rank), device=query.device, dtype=torch.float32
    )
    grad_query = torch.empty((rank,), device=query.device, dtype=query.dtype)
    _fla_standard_backward_kernel[(count,)](
        pointers,
        query,
        saved_mixed,
        grad_output_prepared,
        saved_inv_rms,
        saved_logit,
        saved_lse,
        grad_pointers,
        grad_query_partial,
        count,
        len(source_tuple),
        float(scale),
        D=width,
        R=rank,
        BLOCK_D=_next_power_of_two(width),
        ARCH=_architecture_id(query.device),
        ROW_BUCKET=_autotune_row_bucket(count),
        DTYPE=_dtype_key(source_tuple[0].dtype),
        ROUTE=_CONTIGUOUS_ROUTE,
        CHECKPOINT=checkpoint,
        L2=l2,
        VALUE_ROW_STRIDES=row_strides,
        VALUE_FEATURE_STRIDES=feature_strides,
        GRAD_VALUE_ROW_STRIDES=grad_row_strides,
        GRAD_VALUE_FEATURE_STRIDES=grad_feature_strides,
        QUERY_STRIDE=int(query.stride(0)),
        GRAD_OUTPUT_ROW_STRIDE=grad_row_stride,
        GRAD_OUTPUT_D_STRIDE=grad_feature_stride,
    )
    if rank < width:
        _reduce_query(grad_query_partial, grad_query, count, rank)
    else:
        _fla_standard_query_reduce_kernel[
            lambda meta: (triton.cdiv(rank, meta["BD"]),)
        ](
            grad_query_partial,
            grad_query,
            count,
            N=count,
            D=width,
            R=rank,
            ARCH=_architecture_id(query.device),
            DTYPE=_dtype_key(source_tuple[0].dtype),
            L2=l2,
            ROUTE=_CONTIGUOUS_ROUTE,
            CHECKPOINT=checkpoint,
        )
    return [*grad_values, grad_query]


def forward(
    sources: Sequence[torch.Tensor],
    query: torch.Tensor,
    eps: float,
    scale: float,
) -> list[torch.Tensor]:
    """Launch the FLA-derived bounded BF16 source-list forward."""

    if triton is None:
        raise RuntimeError("FLA source kernels require Triton on CUDA")
    source_tuple, count, width, rank = _source_metadata(sources, query)
    _validate_bf16_runtime(source_tuple, query)
    return _launch_standard_forward(
        source_tuple, query, count, width, rank, eps, scale
    )


def backward(
    sources: Sequence[torch.Tensor],
    query: torch.Tensor,
    saved_mixed: torch.Tensor,
    grad_output: torch.Tensor,
    saved_inv_rms: torch.Tensor,
    saved_logit: torch.Tensor,
    saved_lse: torch.Tensor,
    scale: float,
) -> list[torch.Tensor]:
    """Launch source-tiled dV/dQ and return per-source plus query gradients."""

    if triton is None:
        raise RuntimeError("FLA source kernels require Triton on CUDA")
    source_tuple, count, width, rank = _source_metadata(sources, query)
    _validate_bf16_runtime(source_tuple, query, grad_output)
    return _launch_standard_backward(
        source_tuple,
        query,
        saved_mixed,
        grad_output,
        saved_inv_rms,
        saved_logit,
        saved_lse,
        count,
        width,
        rank,
        scale,
    )


__all__ = ["backward", "forward"]
