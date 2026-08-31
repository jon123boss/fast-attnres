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

"""FLA-derived source-list Full kernels.

The pinned FLA implementation uses one pointer per residual source and a
small source tile (``BL``) for its online softmax and value backward.  This
adaptation keeps those two structural choices while matching the LR-AttnRes
source-list ABI:

* values are full-width ``D`` vectors;
* an implicit key is the final ``R`` coordinates, with ``R <= D``;
* key RMS normalization has a parameter-free unit weight;
* the learned query is an independently typed BF16 or FP32 vector;
* the saved logit is the already-scaled FP32 logit used by the existing
  source custom-op backward;
* the sliced key derivative is folded into one full-width value gradient
  before the BF16 store.

Only the bounded BF16 source-list route calls this module.  The public packed
route and the source-list FP32/wide-BF16 fallbacks stay in
``fixed_tail_sources.py``.
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


_MAX_BF16_WIDTH = 2048
_QUERY_REDUCE_BLOCK = 1024
_SOURCE_BLOCK_CONFIGS = None
_STANDARD_SOURCE_BLOCK_CONFIGS = None
_STANDARD_QUERY_REDUCE_CONFIGS = None

# These fields intentionally include all properties that can change the
# generated standard kernel or the validity of a cached timing.  The generic
# source-list kernels below retain their historical key and configuration set
# so sliced/generic dispatch stays byte-for-byte compatible with the bounded
# route that preceded this production specialization.
_STANDARD_AUTOTUNE_KEY = [
    "ARCH",
    "DTYPE",
    "D",
    "R",
    "L2",
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

_H100_ARCH = "sm90"
_B200_ARCH = "sm100"
_PRODUCTION_ARCHITECTURES = (_H100_ARCH, _B200_ARCH)
_CONTIGUOUS_ROUTE = 1
_GENERIC_ROUTE = 0
_SAVE_MIXED_CHECKPOINT = 1
_RECOMPUTE_CHECKPOINT = 0


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (int(value) - 1).bit_length()


def supports(sources: Sequence[torch.Tensor], width: int | None = None) -> bool:
    """Return whether the bounded FLA extraction owns this source case."""

    if not sources:
        return False
    first = sources[0]
    source_width = int(first.shape[-1]) if width is None else int(width)
    return first.dtype == torch.bfloat16 and source_width <= _MAX_BF16_WIDTH


def _architecture_key(device: torch.device | str | int) -> str:
    """Return a stable SM identifier for Triton autotune cache separation.

    H100 and B200 are currently the production targets (SM90 and SM100), but
    unknown CUDA devices retain their own capability key rather than sharing a
    timing cache with either target.  The helper is only called after the
    public CUDA dispatch guard, so CPU imports remain safe.
    """

    index = device.index if isinstance(device, torch.device) else device
    major, minor = torch.cuda.get_device_capability(index)
    return f"sm{int(major)}{int(minor)}"


def _architecture_id(device: torch.device | str | int) -> int:
    """Return the integer architecture token passed as a constexpr key."""

    key = _architecture_key(device)
    try:
        return int(key[2:])
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback.
        return 0


def _dtype_key(dtype: torch.dtype) -> int:
    """Encode tensor dtype as a small cache-safe scalar used by Triton."""

    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    return int(getattr(dtype, "itemsize", 0))


def _is_standard_contiguous(
    sources: Sequence[torch.Tensor], query: torch.Tensor, width: int, rank: int
) -> bool:
    """Whether the simple production specialization is ABI-safe to use."""

    return (
        bool(sources)
        and sources[0].dtype == torch.bfloat16
        and width <= _MAX_BF16_WIDTH
        and rank == width
        and query.ndim == 1
        and query.is_contiguous()
        and all(source.is_contiguous() for source in sources)
    )


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
    """Build a padded pointer tuple without packing the source list.

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
    length = max(8, _next_power_of_two(len(prepared)))
    padded = prepared + (prepared[0],) * (length - len(prepared))
    row_strides = tuple(
        0 if tensor.ndim <= 1 else int(tensor.stride(-2)) for tensor in padded
    )
    feature_strides = tuple(int(tensor.stride(-1)) for tensor in padded)
    return padded, row_strides, feature_strides, length


def _row_layout(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Return a tensor and affine row/feature strides for a flattened view."""

    prepared = tensor if _is_row_affine(tensor) else tensor.contiguous()
    row_stride = 0 if prepared.ndim <= 1 else int(prepared.stride(-2))
    return prepared, row_stride, int(prepared.stride(-1))


if triton is not None:
    _SOURCE_BLOCK_CONFIGS = [
        triton.Config(
            {"BL": block, "LAYOUT_FAMILY": layout_family},
            num_warps=warps,
            num_stages=stages,
        )
        for layout_family in (0, 1)
        for block in (1, 2, 4, 8)
        for warps, stages in ((4, 2), (8, 2))
    ]
    _BACKWARD_SOURCE_BLOCK_CONFIGS = [
        triton.Config(
            {"BL": block, "LAYOUT_FAMILY": layout_family},
            num_warps=warps,
            num_stages=stages,
        )
        for layout_family in (0, 1, 2)
        for block in ((2, 4, 8) if layout_family == 2 else (1, 2, 4, 8))
        for warps, stages in ((4, 2), (8, 2))
    ]
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

        pointer = values[0]
        row_stride = tl.cast(row_strides[0], tl.int64)
        feature_stride = tl.cast(feature_strides[0], tl.int64)
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


    @triton.jit
    def _fla_scalar_compact_source_backward(
        values,
        grad_values,
        saved_inv_rms,
        saved_logit,
        row,
        source_id,
        source_mask,
        row_valid,
        count64,
        lse,
        grad_prefix,
        grad_tail,
        query_tail,
        delta,
        p_offsets,
        prefix_p_mask,
        tail_offsets,
        r_mask,
        scale_f32,
        R: tl.constexpr,
        L2: tl.constexpr,
        VALUE_ROW_STRIDES: tl.constexpr,
        VALUE_FEATURE_STRIDES: tl.constexpr,
        GRAD_VALUE_ROW_STRIDES: tl.constexpr,
        GRAD_VALUE_FEATURE_STRIDES: tl.constexpr,
    ):
        """Process one source in compact lanes for the third layout family.

        The caller statically unrolls this helper across the configured source
        tile.  Keeping one source active at a time avoids materializing the
        ``[BL, BLOCK_PREFIX]`` and ``[BL, BLOCK_R]`` tensors used by the
        vectorized compact path while retaining the same source order and
        one-kernel, disjoint-store dataflow.
        """

        value_base, value_stride = _select_source_pointer(
            values,
            source_id,
            row,
            VALUE_ROW_STRIDES,
            VALUE_FEATURE_STRIDES,
            L2,
        )
        grad_value_base, grad_value_stride = _select_source_pointer(
            grad_values,
            source_id,
            row,
            GRAD_VALUE_ROW_STRIDES,
            GRAD_VALUE_FEATURE_STRIDES,
            L2,
        )
        value_prefix = tl.load(
            value_base + p_offsets * value_stride,
            mask=source_mask & row_valid & prefix_p_mask,
            other=0.0,
        ).to(tl.float32)
        tail = tl.load(
            value_base + tail_offsets * value_stride,
            mask=source_mask & row_valid & r_mask,
            other=0.0,
        ).to(tl.float32)
        metadata = source_id * count64 + row
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
        dweight = tl.sum(value_prefix * grad_prefix, axis=0) + tl.sum(
            tail * grad_tail, axis=0
        )
        dscore = probability * (dweight - delta)
        scaled_dscore = dscore * scale_f32
        normalized_key = tail * inverse_rms
        projection = dscore * saved_score / R
        grad_key_r = inverse_rms * (
            scaled_dscore * query_tail - normalized_key * projection
        )
        direct_prefix = probability * grad_prefix
        tl.store(
            grad_value_base + p_offsets * grad_value_stride,
            direct_prefix.to(grad_values[0].dtype.element_ty),
            mask=source_mask & row_valid & prefix_p_mask,
        )
        tl.store(
            grad_value_base + tail_offsets * grad_value_stride,
            (probability * grad_tail + grad_key_r).to(
                grad_values[0].dtype.element_ty
            ),
            mask=source_mask & row_valid & r_mask,
        )
        return scaled_dscore * normalized_key


    @triton.autotune(
        configs=_SOURCE_BLOCK_CONFIGS,
        key=["L2", "D", "R"],
    )
    @triton.jit(do_not_specialize=["count", "sources"])
    def _fla_source_forward_kernel(
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
        BLOCK_R: tl.constexpr,
        BL: tl.constexpr,
        LAYOUT_FAMILY: tl.constexpr,
        QUERY_STRIDE,
        OUTPUT_ROW_STRIDE,
        OUTPUT_D_STRIDE,
        L2: tl.constexpr,
        ROW_STRIDES: tl.constexpr,
        FEATURE_STRIDES: tl.constexpr,
    ):
        """FA-style online source reduction for one flattened batch row."""

        row = tl.program_id(0).to(tl.int64)
        count64 = tl.cast(count, tl.int64)
        source_count64 = tl.cast(sources, tl.int64)
        d_offsets = tl.arange(0, BLOCK_D).to(tl.int64)
        r_offsets = tl.arange(0, BLOCK_R).to(tl.int64)
        d_mask = d_offsets < D
        r_mask = r_offsets < R
        eps_f32 = tl.cast(eps, tl.float32)
        scale_f32 = tl.cast(scale, tl.float32)

        if LAYOUT_FAMILY == 0 and R < D and R == BLOCK_R and D % R == 0:
            # The resident family already has the full [BL, BLOCK_D] value
            # tile.  When the suffix is an aligned R-wide block, recursively
            # split that tile while keeping BL as the source axis.  The
            # no-reorder reshape preserves physical D-lane order and avoids
            # both a second tail load and a register gather.
            query_tail = tl.load(
                query
                + r_offsets * tl.cast(QUERY_STRIDE, tl.int64),
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
        elif LAYOUT_FAMILY == 1 and R < D and BLOCK_R < BLOCK_D:
            # Compact-family key math reads the physical D-R:d suffix as an
            # R-wide tile.  This second read is deliberately kept separate
            # from the resident family, whose D-wide value tile is reused.
            tail_offsets = (D - R + r_offsets).to(tl.int64)
            query_tail = tl.load(
                query
                + r_offsets * tl.cast(QUERY_STRIDE, tl.int64),
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
        elif R == D:
            query_tail = tl.load(
                query
                + r_offsets * tl.cast(QUERY_STRIDE, tl.int64),
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            # Keep the narrow query in the resident D-lane layout.  The
            # masked lanes map D-R:d back to query coordinates 0:R.
            tail_mask = (d_offsets >= (D - R)) & d_mask
            tail_indices = tl.maximum(d_offsets - (D - R), 0)
            query_d = tl.load(
                query + tail_indices * tl.cast(QUERY_STRIDE, tl.int64),
                mask=tail_mask,
                other=0.0,
            ).to(tl.float32)
        running_max = tl.full([], -float("inf"), tl.float32)
        running_denom = tl.zeros([], tl.float32)
        running_output = tl.zeros((BLOCK_D,), tl.float32)

        for source_base in tl.range(0, sources, BL, num_stages=2):
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
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if LAYOUT_FAMILY == 0 and R < D and R == BLOCK_R and D % R == 0:
                tail = value
                for split_level in tl.static_range(0, 13):
                    if (BLOCK_D >> (split_level + 1)) >= R:
                        tail_low, tail_high = tl.split(
                            tl.reshape(
                                tail,
                                (BL, 2, BLOCK_D >> (split_level + 1)),
                                can_reorder=False,
                            ).permute(0, 2, 1)
                        )
                        if (D - R) & (BLOCK_D >> (split_level + 1)):
                            tail = tail_high
                        else:
                            tail = tail_low
            elif LAYOUT_FAMILY == 1 and R < D and BLOCK_R < BLOCK_D:
                tail = tl.load(
                    value_base[:, None]
                    + tail_offsets[None, :] * value_stride[:, None],
                    mask=source_mask[:, None] & r_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            elif R == D:
                tail = value
            else:
                # The full value tile is already resident; select its tail
                # lanes without issuing a second source load.
                tail = tl.where(tail_mask[None, :], value, 0.0)

            inverse_rms = tl.rsqrt(tl.sum(tail * tail, axis=1) / R + eps_f32)
            if LAYOUT_FAMILY == 0 and R < D and R == BLOCK_R and D % R == 0:
                raw_dot = tl.sum(tail * query_tail[None, :], axis=1)
            elif LAYOUT_FAMILY == 1 and R < D and BLOCK_R < BLOCK_D:
                raw_dot = tl.sum(tail * query_tail[None, :], axis=1)
            elif R == D:
                raw_dot = tl.sum(tail * query_tail[None, :], axis=1)
            else:
                raw_dot = tl.sum(tail * query_d[None, :], axis=1)
            saved_score = raw_dot * inverse_rms * scale_f32
            tile_scores = tl.where(source_mask, saved_score, -float("inf"))
            new_max = tl.maximum(running_max, tl.max(tile_scores, axis=0))
            old_scale = tl.exp(running_max - new_max)
            probability_numerator = tl.exp(tile_scores - new_max)
            probability_numerator = tl.where(
                source_mask, probability_numerator, 0.0
            )
            running_denom = running_denom * old_scale + tl.sum(
                probability_numerator, axis=0
            )
            running_output = running_output * old_scale + tl.sum(
                probability_numerator[:, None] * value, axis=0
            )

            metadata = source_ids * count64 + row
            tl.store(saved_inv_rms + metadata, inverse_rms, mask=source_mask)
            tl.store(saved_logit + metadata, saved_score, mask=source_mask)
            running_max = new_max

        mixed = running_output / running_denom
        tl.store(saved_lse + row, running_max + tl.log(running_denom))
        output_ptr = (
            output
            + row * tl.cast(OUTPUT_ROW_STRIDE, tl.int64)
            + d_offsets * tl.cast(OUTPUT_D_STRIDE, tl.int64)
        )
        tl.store(saved_mixed + row * D + d_offsets, mixed, mask=d_mask)
        tl.store(output_ptr, mixed, mask=d_mask)


    @triton.autotune(
        configs=_BACKWARD_SOURCE_BLOCK_CONFIGS,
        key=["L2", "D", "R"],
    )
    @triton.jit(do_not_specialize=["count", "sources"])
    def _fla_source_backward_kernel(
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
        BLOCK_R: tl.constexpr,
        BLOCK_PREFIX: tl.constexpr,
        BL: tl.constexpr,
        LAYOUT_FAMILY: tl.constexpr,
        QUERY_STRIDE,
        GRAD_OUTPUT_ROW_STRIDE,
        GRAD_OUTPUT_D_STRIDE,
        L2: tl.constexpr,
        VALUE_ROW_STRIDES: tl.constexpr,
        VALUE_FEATURE_STRIDES: tl.constexpr,
        GRAD_VALUE_ROW_STRIDES: tl.constexpr,
        GRAD_VALUE_FEATURE_STRIDES: tl.constexpr,
    ):
        """FA-style source-tiled backward with folded full-width dV."""

        row = tl.program_id(0).to(tl.int64)
        count64 = tl.cast(count, tl.int64)
        source_count64 = tl.cast(sources, tl.int64)
        d_offsets = tl.arange(0, BLOCK_D).to(tl.int64)
        r_offsets = tl.arange(0, BLOCK_R).to(tl.int64)
        d_mask = d_offsets < D
        r_mask = r_offsets < R
        row_valid = row < count64
        scale_f32 = tl.cast(scale, tl.float32)

        if (
            (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
            and R < D
            and BLOCK_R < BLOCK_D
        ):
            # Compact-family key math uses the physical D-R:d suffix as its
            # native R-wide tile.  The gradient for the upstream output tail
            # is loaded once and reused for every source below.
            tail_offsets = (D - R + r_offsets).to(tl.int64)
            prefix_d_mask = d_mask & (d_offsets < (D - R))
            if BLOCK_PREFIX < BLOCK_D:
                p_offsets = tl.arange(0, BLOCK_PREFIX).to(tl.int64)
                prefix_p_mask = p_offsets < (D - R)
            query_tail = tl.load(
                query
                + r_offsets * tl.cast(QUERY_STRIDE, tl.int64),
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
        elif R == D:
            query_tail = tl.load(
                query
                + r_offsets * tl.cast(QUERY_STRIDE, tl.int64),
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
        if (
            (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
            and R < D
            and BLOCK_R < BLOCK_D
            and BLOCK_PREFIX < BLOCK_D
        ):
            grad_prefix = tl.load(
                grad_output
                + row * tl.cast(GRAD_OUTPUT_ROW_STRIDE, tl.int64)
                + p_offsets * tl.cast(GRAD_OUTPUT_D_STRIDE, tl.int64),
                mask=row_valid & prefix_p_mask,
                other=0.0,
            ).to(tl.float32)
            grad_tail = tl.load(
                grad_output
                + row * tl.cast(GRAD_OUTPUT_ROW_STRIDE, tl.int64)
                + tail_offsets * tl.cast(GRAD_OUTPUT_D_STRIDE, tl.int64),
                mask=row_valid & r_mask,
                other=0.0,
            ).to(tl.float32)
            mixed_prefix = tl.load(
                saved_mixed + row * D + p_offsets,
                mask=row_valid & prefix_p_mask,
                other=0.0,
            ).to(tl.float32)
            mixed_tail = tl.load(
                saved_mixed + row * D + tail_offsets,
                mask=row_valid & r_mask,
                other=0.0,
            ).to(tl.float32)
            delta = tl.sum(grad_prefix * mixed_prefix, axis=0) + tl.sum(
                grad_tail * mixed_tail, axis=0
            )
        else:
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
            delta = tl.sum(grad * mixed, axis=0)
        lse = tl.load(saved_lse + row, mask=row_valid, other=0.0).to(tl.float32)
        if (
            (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
            and R < D
            and BLOCK_R < BLOCK_D
        ):
            if BLOCK_PREFIX >= BLOCK_D:
                grad_tail = tl.load(
                    grad_output
                    + row * tl.cast(GRAD_OUTPUT_ROW_STRIDE, tl.int64)
                    + tail_offsets * tl.cast(GRAD_OUTPUT_D_STRIDE, tl.int64),
                    mask=row_valid & r_mask,
                    other=0.0,
                ).to(tl.float32)
        else:
            tail_mask = (d_offsets >= (D - R)) & d_mask
            tail_indices = tl.maximum(d_offsets - (D - R), 0)
            query_d = tl.load(
                query + tail_indices * tl.cast(QUERY_STRIDE, tl.int64),
                mask=tail_mask,
                other=0.0,
            ).to(tl.float32)
        if (
            (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
            and R < D
            and BLOCK_R < BLOCK_D
        ):
            grad_query = tl.zeros((BLOCK_R,), tl.float32)
        elif R == D:
            grad_query = tl.zeros((BLOCK_R,), tl.float32)
        else:
            grad_query_d = tl.zeros((BLOCK_D,), tl.float32)

        # Family 2 keeps the configured source tile but handles one source at
        # a time in a statically unrolled loop.  The vector loop is retained
        # for F0/F1 and for family-2 fallback geometries; it compiles away
        # for the narrow family-2 prefix path.
        vector_source_end = (
            0
            if (
                LAYOUT_FAMILY == 2
                and R < D
                and BLOCK_R < BLOCK_D
                and BLOCK_PREFIX < BLOCK_D
            )
            else sources
        )
        for source_base in tl.range(0, vector_source_end, BL, num_stages=2):
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
            if (
                (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
                and R < D
                and BLOCK_R < BLOCK_D
                and BLOCK_PREFIX < BLOCK_D
            ):
                value_prefix_ptr = (
                    value_base[:, None]
                    + p_offsets[None, :] * value_stride[:, None]
                )
                value_prefix = tl.load(
                    value_prefix_ptr,
                    mask=source_mask[:, None]
                    & row_valid
                    & prefix_p_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                tail = tl.load(
                    value_base[:, None]
                    + tail_offsets[None, :] * value_stride[:, None],
                    mask=source_mask[:, None]
                    & row_valid
                    & r_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            elif (
                (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
                and R < D
                and BLOCK_R < BLOCK_D
            ):
                value_ptr = (
                    value_base[:, None]
                    + d_offsets[None, :] * value_stride[:, None]
                )
                value = tl.load(
                    value_ptr,
                    mask=source_mask[:, None] & row_valid & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                tail = tl.load(
                    value_base[:, None]
                    + tail_offsets[None, :] * value_stride[:, None],
                    mask=source_mask[:, None]
                    & row_valid
                    & r_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            elif R == D:
                value_ptr = (
                    value_base[:, None]
                    + d_offsets[None, :] * value_stride[:, None]
                )
                value = tl.load(
                    value_ptr,
                    mask=source_mask[:, None] & row_valid & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                tail = value
            else:
                value_ptr = (
                    value_base[:, None]
                    + d_offsets[None, :] * value_stride[:, None]
                )
                value = tl.load(
                    value_ptr,
                    mask=source_mask[:, None] & row_valid & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                # Reuse the resident D-lane value tile for the implicit key.
                tail = tl.where(tail_mask[None, :], value, 0.0)

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
            if (
                (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
                and R < D
                and BLOCK_R < BLOCK_D
                and BLOCK_PREFIX < BLOCK_D
            ):
                dweight = tl.sum(
                    value_prefix * grad_prefix[None, :], axis=1
                ) + tl.sum(tail * grad_tail[None, :], axis=1)
            else:
                dweight = tl.sum(value * grad[None, :], axis=1)
            dscore = probability * (dweight - delta)
            scaled_dscore = dscore * scale_f32

            normalized_key = tail * inverse_rms[:, None]
            if (
                (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
                and R < D
                and BLOCK_R < BLOCK_D
            ):
                grad_query += tl.sum(
                    scaled_dscore[:, None] * normalized_key,
                    axis=0,
                )
            elif R == D:
                grad_query += tl.sum(
                    scaled_dscore[:, None] * normalized_key,
                    axis=0,
                )
            else:
                grad_query_d += tl.sum(
                    scaled_dscore[:, None] * normalized_key,
                    axis=0,
                )

            # dscore is the derivative of the already-scaled logit.  The
            # saved score therefore supplies the scale-aware projection term.
            projection = dscore * saved_score / R
            if (
                (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
                and R < D
                and BLOCK_R < BLOCK_D
                and BLOCK_PREFIX < BLOCK_D
            ):
                # The compact key derivative is stored separately from the
                # direct prefix gradient.  Prefix and tail masks are
                # disjoint, so every physical D lane is written exactly once.
                direct_prefix = probability[:, None] * grad_prefix[None, :]
                grad_key_r = inverse_rms[:, None] * (
                    scaled_dscore[:, None] * query_tail[None, :]
                    - normalized_key * projection[:, None]
                )
                grad_value_prefix_ptr = (
                    grad_value_base[:, None]
                    + p_offsets[None, :] * grad_value_stride[:, None]
                )
                grad_value_tail_ptr = (
                    grad_value_base[:, None]
                    + tail_offsets[None, :] * grad_value_stride[:, None]
                )
                tl.store(
                    grad_value_prefix_ptr,
                    direct_prefix.to(grad_values[0].dtype.element_ty),
                    mask=source_mask[:, None]
                    & row_valid
                    & prefix_p_mask[None, :],
                )
                tl.store(
                    grad_value_tail_ptr,
                    (
                        probability[:, None] * grad_tail[None, :]
                        + grad_key_r
                    ).to(grad_values[0].dtype.element_ty),
                    mask=source_mask[:, None]
                    & row_valid
                    & r_mask[None, :],
                )
            elif (
                (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
                and R < D
                and BLOCK_R < BLOCK_D
            ):
                # The parent compact path keeps the D-wide value/gradient
                # loads and masks its direct prefix store.
                direct = probability[:, None] * grad[None, :]
                grad_key_r = inverse_rms[:, None] * (
                    scaled_dscore[:, None] * query_tail[None, :]
                    - normalized_key * projection[:, None]
                )
                grad_value_prefix_ptr = (
                    grad_value_base[:, None]
                    + d_offsets[None, :] * grad_value_stride[:, None]
                )
                grad_value_tail_ptr = (
                    grad_value_base[:, None]
                    + tail_offsets[None, :] * grad_value_stride[:, None]
                )
                tl.store(
                    grad_value_prefix_ptr,
                    direct.to(grad_values[0].dtype.element_ty),
                    mask=source_mask[:, None]
                    & row_valid
                    & prefix_d_mask[None, :],
                )
                tl.store(
                    grad_value_tail_ptr,
                    (
                        probability[:, None] * grad_tail[None, :]
                        + grad_key_r
                    ).to(grad_values[0].dtype.element_ty),
                    mask=source_mask[:, None]
                    & row_valid
                    & r_mask[None, :],
                )
            else:
                direct = probability[:, None] * grad[None, :]
                normalized_value = value * inverse_rms[:, None]
                grad_key = inverse_rms[:, None] * (
                    scaled_dscore[:, None] * query_d[None, :]
                    - normalized_value * projection[:, None]
                )
                grad_value = direct + tl.where(
                    tail_mask[None, :], grad_key, 0.0
                )
                grad_ptr = (
                    grad_value_base[:, None]
                    + d_offsets[None, :] * grad_value_stride[:, None]
                )
                tl.store(
                    grad_ptr,
                    grad_value.to(grad_values[0].dtype.element_ty),
                    mask=source_mask[:, None] & row_valid & d_mask[None, :],
                )

        if (
            LAYOUT_FAMILY == 2
            and R < D
            and BLOCK_R < BLOCK_D
            and BLOCK_PREFIX < BLOCK_D
        ):
            # Keep BL source utilization without constructing BL-wide compact
            # layouts.  Each source is scalar-selected and the compact prefix
            # and tail vectors are live only for that source's computation.
            for source_base in tl.range(0, sources, BL, num_stages=2):
                for source_lane in tl.static_range(0, BL):
                    source_id = tl.cast(source_base, tl.int64) + source_lane
                    source_mask = source_id < source_count64
                    grad_query += _fla_scalar_compact_source_backward(
                        values,
                        grad_values,
                        saved_inv_rms,
                        saved_logit,
                        row,
                        source_id,
                        source_mask,
                        row_valid,
                        count64,
                        lse,
                        grad_prefix,
                        grad_tail,
                        query_tail,
                        delta,
                        p_offsets,
                        prefix_p_mask,
                        tail_offsets,
                        r_mask,
                        scale_f32,
                        R=R,
                        L2=L2,
                        VALUE_ROW_STRIDES=VALUE_ROW_STRIDES,
                        VALUE_FEATURE_STRIDES=VALUE_FEATURE_STRIDES,
                        GRAD_VALUE_ROW_STRIDES=GRAD_VALUE_ROW_STRIDES,
                        GRAD_VALUE_FEATURE_STRIDES=GRAD_VALUE_FEATURE_STRIDES,
                    )

        if (
            (LAYOUT_FAMILY == 1 or LAYOUT_FAMILY == 2)
            and R < D
            and BLOCK_R < BLOCK_D
        ):
            tl.store(
                grad_query_partial + row * R + r_offsets,
                grad_query,
                mask=row_valid & r_mask,
            )
        elif R == D:
            tl.store(
                grad_query_partial + row * R + r_offsets,
                grad_query,
                mask=row_valid & r_mask,
            )
        else:
            # Write only D-R:d into the existing [N, R] scratch layout.
            tl.store(
                grad_query_partial + row * R + tail_indices,
                grad_query_d,
                mask=row_valid & tail_mask,
            )


    @triton.autotune(
        configs=_STANDARD_SOURCE_BLOCK_CONFIGS,
        key=_STANDARD_AUTOTUNE_KEY,
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
        """Simple full-width source reduction for contiguous standard AttnRes."""

        # ARCH, DTYPE, ROUTE, and CHECKPOINT values are constexpr cache
        # dimensions.  The
        # standard dispatcher supplies only the contiguous full-rank case, so
        # no result-dependent or GPU-specific config filter is needed here.
        row = tl.program_id(0).to(tl.int64)
        count64 = tl.cast(count, tl.int64)
        source_count64 = tl.cast(sources, tl.int64)
        d_offsets = tl.arange(0, BLOCK_D).to(tl.int64)
        d_mask = d_offsets < D
        row_valid = row < count64
        eps_f32 = tl.cast(eps, tl.float32)
        scale_f32 = tl.cast(scale, tl.float32)
        query_value = tl.load(
            query + d_offsets * tl.cast(QUERY_STRIDE, tl.int64),
            mask=d_mask,
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
            inverse_rms = tl.rsqrt(tl.sum(value * value, axis=1) / R + eps_f32)
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
        key=_STANDARD_AUTOTUNE_KEY,
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
        """Full-width source dV/dQ for the contiguous standard route."""

        row = tl.program_id(0).to(tl.int64)
        count64 = tl.cast(count, tl.int64)
        source_count64 = tl.cast(sources, tl.int64)
        d_offsets = tl.arange(0, BLOCK_D).to(tl.int64)
        d_mask = d_offsets < D
        row_valid = row < count64
        scale_f32 = tl.cast(scale, tl.float32)
        query_value = tl.load(
            query + d_offsets * tl.cast(QUERY_STRIDE, tl.int64),
            mask=d_mask,
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
            normalized_key = value * inverse_rms[:, None]
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
            grad_query_partial + row * R + d_offsets,
            grad_query,
            mask=row_valid & d_mask,
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
    ):
        """Deterministically reduce one FP32 query partial per batch row."""

        rank_block = tl.program_id(0).to(tl.int64)
        r_offsets = rank_block * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
        r_mask = r_offsets < R
        accumulator = tl.zeros((BLOCK_R,), tl.float32)
        for row_base in tl.range(0, count, BLOCK_N, num_stages=2):
            row_offsets = row_base.to(tl.int64) + tl.arange(0, BLOCK_N).to(tl.int64)
            row_mask = row_offsets < tl.cast(count, tl.int64)
            partial_values = tl.load(
                partial + row_offsets[:, None] * R + r_offsets[None, :],
                mask=row_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(partial_values, axis=0)
        tl.store(grad_query + r_offsets, accumulator, mask=r_mask)


def _source_metadata(
    sources: Sequence[torch.Tensor], query: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], int, int, int]:
    source_tuple = tuple(sources)
    if not source_tuple:
        raise ValueError("sources must be nonempty")
    first = source_tuple[0]
    count = math.prod(int(size) for size in first.shape[:-1]) or 1
    return source_tuple, int(count), int(first.shape[-1]), int(query.numel())


def _standard_path(
    sources: Sequence[torch.Tensor], query: torch.Tensor, width: int, rank: int
) -> bool:
    """Return whether the production full-rank specialization owns a call."""

    return _is_standard_contiguous(sources, query, width, rank)


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
    if _standard_path(source_tuple, query, width, rank):
        return _launch_standard_forward(
            source_tuple, query, count, width, rank, eps, scale
        )
    pointers, row_strides, feature_strides, l2 = _source_pointer_table(source_tuple)
    first = source_tuple[0]
    output = torch.empty_like(first, memory_format=torch.contiguous_format)
    saved_mixed = torch.empty((count, width), device=first.device, dtype=torch.float32)
    saved_inv_rms = torch.empty(
        (len(source_tuple), count), device=first.device, dtype=torch.float32
    )
    saved_logit = torch.empty_like(saved_inv_rms)
    saved_lse = torch.empty((count,), device=first.device, dtype=torch.float32)
    _fla_source_forward_kernel[(count,)](
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
        BLOCK_R=_next_power_of_two(rank),
        QUERY_STRIDE=int(query.stride(0)),
        OUTPUT_ROW_STRIDE=0 if output.ndim <= 1 else int(output.stride(-2)),
        OUTPUT_D_STRIDE=int(output.stride(-1)),
        L2=l2,
        ROW_STRIDES=row_strides,
        FEATURE_STRIDES=feature_strides,
    )
    return [output, saved_mixed, saved_inv_rms, saved_logit, saved_lse]


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
    if _standard_path(source_tuple, query, width, rank):
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
    grad_query_partial = torch.empty(
        (count, rank), device=query.device, dtype=torch.float32
    )
    grad_query = torch.empty((rank,), device=query.device, dtype=query.dtype)
    _fla_source_backward_kernel[(count,)](
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
        BLOCK_R=_next_power_of_two(rank),
        BLOCK_PREFIX=_next_power_of_two(max(1, width - rank)),
        QUERY_STRIDE=int(query.stride(0)),
        GRAD_OUTPUT_ROW_STRIDE=grad_row_stride,
        GRAD_OUTPUT_D_STRIDE=grad_feature_stride,
        L2=l2,
        VALUE_ROW_STRIDES=row_strides,
        VALUE_FEATURE_STRIDES=feature_strides,
        GRAD_VALUE_ROW_STRIDES=grad_row_strides,
        GRAD_VALUE_FEATURE_STRIDES=grad_feature_strides,
    )
    query_reduce_tile = min(32, _next_power_of_two(rank))
    _fla_query_reduce_kernel[(triton.cdiv(rank, query_reduce_tile),)](
        grad_query_partial,
        grad_query,
        count,
        R=rank,
        BLOCK_N=_QUERY_REDUCE_BLOCK,
        BLOCK_R=query_reduce_tile,
        num_warps=8 if rank >= 1024 else 4,
        num_stages=2,
    )
    return [*grad_values, grad_query]


__all__ = ["backward", "forward", "supports"]
