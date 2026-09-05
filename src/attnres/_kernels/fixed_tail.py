"""Packed online fixed tail Attention Residuals.

This is the small BF16 CUDA core extracted from the archived one store dV
probe.  Values are packed as ``[S, ..., D]`` and the implicit key is the final
``R`` coordinates of each value.  The CUDA envelope is generalized to the
package limits, but that envelope is intentionally unverified until the root
GPU checks run.

When ``R < D`` and the physical rank and value blocks have equal width, the
online kernels use one resident D-wide value load for both mixing and the
tail key.  They shift the physical rank lanes into the final D lanes with a
masked offset.  The leading zero lanes can change reduction association, so
this branch is an unqualified GPU ablation until it passes the CUDA gate.
"""

from __future__ import annotations

from typing import Any

import torch

from .._sources import _validate_scalar

try:  # Triton is optional for local CPU development.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on CPU-only environments.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


try:  # Public on the target runtime; private names cover torch 2.5 checks.
    from torch.library import triton_op as _triton_op
    from torch.library import wrap_triton as _wrap_triton
except (ImportError, AttributeError):  # pragma: no cover - local torch 2.5.
    try:
        from torch._library import capture_triton as _wrap_triton
        from torch._library import triton_op as _triton_op
    except (ImportError, AttributeError):  # pragma: no cover - unsupported torch.
        _triton_op = None
        _wrap_triton = None


_EPS = 2**-23

# Compile-time tuning knobs.  They do not add a public runtime mode: the
# source-list and packed adapters both pass the same values to the same
# operator.  The parent GPU pass can change these constants before compiling
# a benchmark matrix.
SOURCE_TILE = 2
FUSE_KEY_VALUE = True
FUSE_KEY_VALUE_MAX_WIDTH = 2048
FUSE_KEY_VALUE_MIN_FRACTION_NUM = 1
FUSE_KEY_VALUE_MIN_FRACTION_DEN = 4
ONE_STORE_DV = True
NUM_WARPS = 4
NUM_STAGES = 2
QUERY_BLOCK_N = 1024
QUERY_BLOCK_R = 32


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (int(value) - 1).bit_length()


def _should_fuse_key_value(width: int, rank: int) -> bool:
    """Choose the bounded D-wide routing tile for a BF16 CUDA launch.

    A fused tile reuses the D-wide value load for the tail key.  Keeping it
    behind a small-width and rank-fraction gate avoids turning very low-rank,
    very wide fallback calls into D-wide key reductions.  This is a launch
    policy knob rather than a new public API mode.
    """

    return bool(
        FUSE_KEY_VALUE
        and width <= FUSE_KEY_VALUE_MAX_WIDTH
        and rank * FUSE_KEY_VALUE_MIN_FRACTION_DEN
        >= width * FUSE_KEY_VALUE_MIN_FRACTION_NUM
    )



def _launch_policy(width: int, rank: int, device: torch.device) -> tuple[int, bool]:
    # Hopper's padded value tile costs more at medium ranks on irregular
    # widths. Keep its scalar source traversal there; narrow ranks benefit
    # from two-source traversal on both architectures.
    hopper_padded = (torch.cuda.get_device_capability(device)[0] == 9
                     and width != _next_power_of_two(width) and 4 * rank >= width)
    tile = 1 if rank == width or hopper_padded else SOURCE_TILE
    return tile, _should_fuse_key_value(width, rank) and not hopper_padded

def _validate_inputs(values: torch.Tensor, query: torch.Tensor) -> tuple[int, int, int]:
    if not isinstance(values, torch.Tensor) or not isinstance(query, torch.Tensor):
        raise TypeError("values and query must be tensors")
    if values.ndim < 2:
        raise ValueError("values must have shape [S,...,D]")
    if query.ndim != 1:
        raise ValueError("query must have shape [R]")
    sources = int(values.shape[0])
    width = int(values.shape[-1])
    rank = int(query.numel())
    if not 1 <= sources <= 129:
        raise ValueError("supported source envelope is 1<=S<=129")
    if not 1 <= width <= 8192:
        raise ValueError("supported value envelope is 1<=D<=8192")
    if not 1 <= rank <= width:
        raise ValueError("query rank must satisfy 1<=R<=D")
    if any(int(size) < 1 for size in values.shape):
        raise ValueError("values dimensions must be positive")
    if values.device != query.device:
        raise ValueError("values and query must be on the same device")
    if values.dtype != torch.bfloat16:
        raise TypeError("values must use BF16 storage")
    if query.dtype != torch.bfloat16:
        raise TypeError("query must use BF16 storage")
    return sources, width, rank


def _prepare_grad_output(
    grad_output: torch.Tensor, packed_values: torch.Tensor
) -> torch.Tensor:
    """Shape an output gradient for the flattened ``[S,N,D]`` kernel input."""
    grad_output = grad_output.reshape(tuple(packed_values.shape[1:]))
    return grad_output if grad_output.is_contiguous() else grad_output.contiguous()


if triton is not None:

    _BACKWARD_STORE_FAMILY_CONFIGS = [
        triton.Config(
            {"STORE_FAMILY": store_family},
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        for store_family in (0, 1)
    ]
    _BACKWARD_STORE_FAMILY_KEY = [
        "n_tokens",
        "D",
        "R",
        "BLOCK_D",
        "BLOCK_R",
        "L2",
        "SOURCE_TILE",
        "FUSE_KEY_WITH_VALUE",
        "LIST_SOURCES",
        "SOURCE_RECORDS",
        "SOURCE_STRIDES_UNIFORM",
        "GRAD_STRIDES_UNIFORM",
    ]

    @triton.jit
    def _select_source_pointer(
        values,
        source,
        row,
        ROW_STRIDES: tl.constexpr,
        FEATURE_STRIDES: tl.constexpr,
        L2: tl.constexpr,
    ):
        """Select a typed source pointer before applying lane offsets."""
        pointer = values[0]
        row_stride = tl.cast(ROW_STRIDES[0], tl.int64)
        feature_stride = tl.cast(FEATURE_STRIDES[0], tl.int64)
        for source_index in tl.static_range(1, L2):
            selected = source == source_index
            pointer = tl.where(selected, values[source_index], pointer)
            row_stride = tl.where(
                selected,
                tl.cast(ROW_STRIDES[source_index], tl.int64),
                row_stride,
            )
            feature_stride = tl.where(
                selected,
                tl.cast(FEATURE_STRIDES[source_index], tl.int64),
                feature_stride,
            )
        return pointer + row * row_stride, feature_stride

    @triton.jit
    def _source_record_pointer(
        records,
        source,
        row,
        VALUE_DTYPE: tl.constexpr,
        STRIDES_UNIFORM: tl.constexpr,
        ROW_STRIDE: tl.constexpr,
        FEATURE_STRIDE: tl.constexpr,
    ):
        source64 = tl.cast(source, tl.int64)
        field = source64 * 3
        address = tl.load(records + field).to(tl.int64)
        if STRIDES_UNIFORM:
            row_stride = tl.cast(ROW_STRIDE, tl.int64)
            feature_stride = tl.cast(FEATURE_STRIDE, tl.int64)
        else:
            row_stride = tl.load(records + field + 1).to(tl.int64)
            feature_stride = tl.load(records + field + 2).to(tl.int64)
        pointer = address.to(tl.pointer_type(VALUE_DTYPE))
        return pointer + row * row_stride, feature_stride

    @triton.jit
    def _source_record_setup(
        values,
        grad_values,
        records,
        L2: tl.constexpr,
        HAS_GRAD_VALUES: tl.constexpr,
        ROW_STRIDES: tl.constexpr,
        FEATURE_STRIDES: tl.constexpr,
        GRAD_ROW_STRIDES: tl.constexpr,
        GRAD_FEATURE_STRIDES: tl.constexpr,
    ):
        if tl.program_id(0) == 0:
            for source_index in tl.static_range(0, L2):
                field = source_index * 3
                tl.store(records + field, tl.cast(values[source_index], tl.int64))
                tl.store(
                    records + field + 1,
                    tl.cast(ROW_STRIDES[source_index], tl.int64),
                )
                tl.store(
                    records + field + 2,
                    tl.cast(FEATURE_STRIDES[source_index], tl.int64),
                )
                if HAS_GRAD_VALUES:
                    grad_field = L2 * 3 + field
                    tl.store(
                        records + grad_field,
                        tl.cast(grad_values[source_index], tl.int64),
                    )
                    tl.store(
                        records + grad_field + 1,
                        tl.cast(GRAD_ROW_STRIDES[source_index], tl.int64),
                    )
                    tl.store(
                        records + grad_field + 2,
                        tl.cast(GRAD_FEATURE_STRIDES[source_index], tl.int64),
                    )

    @triton.jit(do_not_specialize=["n_tokens", "n_sources"])
    def _packed_online_forward_kernel(
        values,
        query,
        output,
        saved_output_fp32,
        saved_key_inv_rms,
        saved_logit,
        saved_lse,
        n_tokens,
        n_sources,
        eps,
        scale,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        SOURCE_TILE: tl.constexpr,
        QUERY_STRIDE: tl.constexpr,
        OUTPUT_ROW_STRIDE: tl.constexpr,
        OUTPUT_D_STRIDE: tl.constexpr,
        L2: tl.constexpr,
        ROW_STRIDES: tl.constexpr,
        FEATURE_STRIDES: tl.constexpr,
        LIST_SOURCES: tl.constexpr,
        SOURCE_RECORDS: tl.constexpr,
        VALUE_DTYPE: tl.constexpr,
        SOURCE_STRIDES_UNIFORM: tl.constexpr,
        SOURCE_ROW_STRIDE: tl.constexpr,
        SOURCE_FEATURE_STRIDE: tl.constexpr,
        FUSE_KEY_WITH_VALUE: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        d_offsets = tl.arange(0, BLOCK_D)
        if R < D and BLOCK_R == BLOCK_D:
            r_offsets = tl.arange(0, BLOCK_R).to(tl.int64) - (D - R)
            r_mask = (r_offsets >= 0) & (r_offsets < R)
        else:
            r_offsets = tl.arange(0, BLOCK_R)
            r_mask = r_offsets < R
        d_mask = d_offsets < D
        tail_d_mask = d_mask & (d_offsets >= (D - R))
        eps_f32 = tl.cast(eps, tl.float32)
        scale_f32 = tl.cast(scale, tl.float32)
        if FUSE_KEY_WITH_VALUE and R < D and BLOCK_R != BLOCK_D:
            tail_d_offsets = (d_offsets - (D - R)).to(tl.int64)
            if LIST_SOURCES:
                q_key = tl.load(
                    query + tail_d_offsets * tl.cast(QUERY_STRIDE, tl.int64),
                    mask=tail_d_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                q_key = tl.load(
                    query + tail_d_offsets, mask=tail_d_mask, other=0.0
                ).to(tl.float32)
        else:
            if LIST_SOURCES:
                q = tl.load(
                    query + r_offsets * tl.cast(QUERY_STRIDE, tl.int64),
                    mask=r_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                q = tl.load(query + r_offsets, mask=r_mask, other=0.0).to(tl.float32)

        running_max = tl.full([], -float("inf"), tl.float32)
        running_denom = tl.zeros([], tl.float32)
        running_output = tl.zeros((BLOCK_D,), tl.float32)

        for source_block in range(tl.cdiv(n_sources, SOURCE_TILE)):
            if LIST_SOURCES:
                source_id = (source_block * SOURCE_TILE).to(tl.int64)
                source_offsets = (source_id + tl.arange(0, SOURCE_TILE)).to(tl.int64)
                source_mask = source_offsets < n_sources
                source_lookup_offsets = tl.minimum(
                    source_offsets, tl.cast(L2 - 1, tl.int64)
                )
                if SOURCE_RECORDS:
                    value_base, value_stride = _source_record_pointer(
                        values,
                        source_lookup_offsets,
                        token,
                        VALUE_DTYPE,
                        SOURCE_STRIDES_UNIFORM,
                        SOURCE_ROW_STRIDE,
                        SOURCE_FEATURE_STRIDE,
                    )
                else:
                    value_base, value_stride = _select_source_pointer(
                        values,
                        source_lookup_offsets,
                        token,
                        ROW_STRIDES,
                        FEATURE_STRIDES,
                        L2,
                    )
                value_base = tl.broadcast_to(value_base, (SOURCE_TILE,))
                value_stride = tl.broadcast_to(value_stride, (SOURCE_TILE,))
            else:
                source_offsets = (source_block * SOURCE_TILE).to(tl.int64) + tl.arange(
                    0, SOURCE_TILE
                )
                source_mask = source_offsets < n_sources
                value_base = values + source_offsets[:, None] * n_tokens * D + token * D
                value_stride = 1
            if LIST_SOURCES:
                value_ptr = (
                    value_base[:, None]
                    + d_offsets[None, :] * value_stride[:, None]
                )
            else:
                value_ptr = value_base + d_offsets[None, :]
            if FUSE_KEY_WITH_VALUE and R < D and BLOCK_R != BLOCK_D:
                # The D-wide value tile is already needed for the mixture.
                # Reuse it for the tail key when the rank is large enough to
                # make a D-wide routing reduction a bounded trade-off.
                value = tl.load(
                    value_ptr,
                    mask=source_mask[:, None] & d_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                tail_d = tl.where(tail_d_mask[None, :], value, 0.0)
                key_inv_rms = tl.rsqrt(
                    tl.sum(tail_d * tail_d, axis=1) / R + eps_f32
                )
                key_d = tail_d * key_inv_rms[:, None]
                logit = tl.sum(key_d * q_key[None, :], axis=1) * scale_f32
            elif R == D:
                value = tl.load(
                    value_ptr,
                    mask=source_mask[:, None] & d_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                tail = value
            elif R < D and BLOCK_R == BLOCK_D:
                value = tl.load(
                    value_ptr,
                    mask=source_mask[:, None] & d_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                tail = tl.where(r_mask[None, :], value, 0.0)
            else:
                # Keep the R wide key temporaries separate from the D wide
                # value load, matching the archived source order.
                if LIST_SOURCES:
                    tail_ptr = value_base[:, None] + (
                        (D - R + r_offsets)[None, :] * value_stride[:, None]
                    )
                else:
                    tail_ptr = value_base + (D - R) + r_offsets[None, :]
                tail = tl.load(
                    tail_ptr,
                    mask=source_mask[:, None] & r_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)

            if not (FUSE_KEY_WITH_VALUE and R < D and BLOCK_R != BLOCK_D):
                key_inv_rms = tl.rsqrt(tl.sum(tail * tail, axis=1) / R + eps_f32)
                key = tail * key_inv_rms[:, None]
                logit = tl.sum(key * q[None, :], axis=1) * scale_f32
            score = tl.where(source_mask, logit, -float("inf"))

            if R < D and BLOCK_R != BLOCK_D and not FUSE_KEY_WITH_VALUE:
                value = tl.load(
                    value_ptr,
                    mask=source_mask[:, None] & d_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)

            if SOURCE_TILE == 2:
                # Keep the source-serial order exactly: lane zero updates the
                # online state before lane one, including the padded odd lane.
                value_0, value_1 = tl.split(value.permute(1, 0))
                score_0, score_1 = tl.split(score)
                source_mask_0, source_mask_1 = tl.split(source_mask)

                new_max = tl.maximum(running_max, score_0)
                old_scale = tl.exp(running_max - new_max)
                probability_numerator = tl.where(
                    source_mask_0, tl.exp(score_0 - new_max), 0.0
                )
                running_denom = running_denom * old_scale + probability_numerator
                running_output = (
                    running_output * old_scale + probability_numerator * value_0
                )
                running_max = new_max

                new_max = tl.maximum(running_max, score_1)
                old_scale = tl.exp(running_max - new_max)
                probability_numerator = tl.where(
                    source_mask_1, tl.exp(score_1 - new_max), 0.0
                )
                running_denom = running_denom * old_scale + probability_numerator
                running_output = (
                    running_output * old_scale + probability_numerator * value_1
                )
                running_max = new_max
            else:
                new_max = tl.maximum(running_max, tl.max(score, axis=0))
                old_scale = tl.exp(running_max - new_max)
                probability_numerator = tl.exp(score - new_max)
                running_denom = running_denom * old_scale + tl.sum(
                    probability_numerator, axis=0
                )
                running_output = running_output * old_scale + tl.sum(
                    probability_numerator[:, None] * value, axis=0
                )
            tl.store(
                saved_key_inv_rms + source_offsets * n_tokens + token,
                key_inv_rms,
                mask=source_mask,
            )
            tl.store(
                saved_logit + source_offsets * n_tokens + token,
                logit,
                mask=source_mask,
            )
            running_max = new_max

        lse = running_max + tl.log(running_denom)
        tl.store(saved_lse + token, lse)
        normalized_output = running_output / running_denom
        tl.store(
            saved_output_fp32 + token * D + d_offsets, normalized_output, mask=d_mask
        )
        if LIST_SOURCES:
            output_ptr = (
                output
                + token * tl.cast(OUTPUT_ROW_STRIDE, tl.int64)
                + d_offsets * tl.cast(OUTPUT_D_STRIDE, tl.int64)
            )
            tl.store(output_ptr, normalized_output, mask=d_mask)
        else:
            tl.store(output + token * D + d_offsets, normalized_output, mask=d_mask)

    @triton.autotune(
        configs=_BACKWARD_STORE_FAMILY_CONFIGS,
        key=_BACKWARD_STORE_FAMILY_KEY,
        cache_results=True,
    )
    @triton.jit(do_not_specialize=["n_tokens", "n_sources"])
    def _packed_online_backward_kernel(
        values,
        query,
        saved_output_fp32,
        grad_output,
        saved_key_inv_rms,
        saved_logit,
        saved_lse,
        grad_values,
        grad_query_token,
        n_tokens,
        n_sources,
        scale,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        SOURCE_TILE: tl.constexpr,
        QUERY_STRIDE: tl.constexpr,
        GRAD_OUTPUT_ROW_STRIDE: tl.constexpr,
        GRAD_OUTPUT_D_STRIDE: tl.constexpr,
        L2: tl.constexpr,
        ROW_STRIDES: tl.constexpr,
        FEATURE_STRIDES: tl.constexpr,
        GRAD_ROW_STRIDES: tl.constexpr,
        GRAD_FEATURE_STRIDES: tl.constexpr,
        LIST_SOURCES: tl.constexpr,
        SOURCE_RECORDS: tl.constexpr,
        VALUE_DTYPE: tl.constexpr,
        SOURCE_STRIDES_UNIFORM: tl.constexpr,
        SOURCE_ROW_STRIDE: tl.constexpr,
        SOURCE_FEATURE_STRIDE: tl.constexpr,
        GRAD_STRIDES_UNIFORM: tl.constexpr,
        GRAD_ROW_STRIDE: tl.constexpr,
        GRAD_FEATURE_STRIDE: tl.constexpr,
        FUSE_KEY_WITH_VALUE: tl.constexpr,
        STORE_FAMILY: tl.constexpr = 0,
    ):
        token = tl.program_id(0).to(tl.int64)
        d_offsets = tl.arange(0, BLOCK_D)
        if R < D and BLOCK_R == BLOCK_D:
            r_offsets = tl.arange(0, BLOCK_R).to(tl.int64) - (D - R)
            r_mask = (r_offsets >= 0) & (r_offsets < R)
        else:
            r_offsets = tl.arange(0, BLOCK_R)
            r_mask = r_offsets < R
        d_mask = d_offsets < D
        tail_d_mask = d_mask & (d_offsets >= (D - R))
        tail_d_offsets = (d_offsets - (D - R)).to(tl.int64)
        scale_f32 = tl.cast(scale, tl.float32)

        if FUSE_KEY_WITH_VALUE and R < D and BLOCK_R != BLOCK_D:
            if LIST_SOURCES:
                q_key = tl.load(
                    query + tail_d_offsets * tl.cast(QUERY_STRIDE, tl.int64),
                    mask=tail_d_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                q_key = tl.load(
                    query + tail_d_offsets, mask=tail_d_mask, other=0.0
                ).to(tl.float32)
        else:
            if LIST_SOURCES:
                q = tl.load(
                    query + r_offsets * tl.cast(QUERY_STRIDE, tl.int64),
                    mask=r_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                q = tl.load(query + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
        if FUSE_KEY_WITH_VALUE and R < D and BLOCK_R != BLOCK_D:
            grad_query_d = tl.zeros((BLOCK_D,), tl.float32)
        else:
            grad_query = tl.zeros((BLOCK_R,), tl.float32)
        if LIST_SOURCES:
            grad = tl.load(
                grad_output
                + token * tl.cast(GRAD_OUTPUT_ROW_STRIDE, tl.int64)
                + d_offsets * tl.cast(GRAD_OUTPUT_D_STRIDE, tl.int64),
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            grad = tl.load(
                grad_output + token * D + d_offsets, mask=d_mask, other=0.0
            ).to(tl.float32)
        if (
            R < D
            and BLOCK_R < BLOCK_D
            and SOURCE_STRIDES_UNIFORM
            and GRAD_STRIDES_UNIFORM
            and QUERY_STRIDE == 1
            and SOURCE_FEATURE_STRIDE == 1
            and GRAD_FEATURE_STRIDE == 1
            and not FUSE_KEY_WITH_VALUE
            and STORE_FAMILY == 1
        ):
            tail_offsets = (D - R + r_offsets).to(tl.int64)
            prefix_d_mask = d_mask & (d_offsets < (D - R))
            if LIST_SOURCES:
                grad_tail = tl.load(
                    grad_output
                    + token * tl.cast(GRAD_OUTPUT_ROW_STRIDE, tl.int64)
                    + tail_offsets * tl.cast(GRAD_OUTPUT_D_STRIDE, tl.int64),
                    mask=r_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                grad_tail = tl.load(
                    grad_output + token * D + tail_offsets,
                    mask=r_mask,
                    other=0.0,
                ).to(tl.float32)
        mixed = tl.load(
            saved_output_fp32 + token * D + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        delta = tl.sum(tl.where(d_mask, grad * mixed, 0.0), axis=0)
        lse = tl.load(saved_lse + token).to(tl.float32)

        for source_block in range(tl.cdiv(n_sources, SOURCE_TILE)):
            if LIST_SOURCES:
                source_id = (source_block * SOURCE_TILE).to(tl.int64)
                source_offsets = (source_id + tl.arange(0, SOURCE_TILE)).to(tl.int64)
                source_mask = source_offsets < n_sources
                source_lookup_offsets = tl.minimum(
                    source_offsets, tl.cast(L2 - 1, tl.int64)
                )
                if SOURCE_RECORDS:
                    value_base, value_stride = _source_record_pointer(
                        values,
                        source_lookup_offsets,
                        token,
                        VALUE_DTYPE,
                        SOURCE_STRIDES_UNIFORM,
                        SOURCE_ROW_STRIDE,
                        SOURCE_FEATURE_STRIDE,
                    )
                    grad_value_base, grad_value_stride = _source_record_pointer(
                        grad_values,
                        source_lookup_offsets,
                        token,
                        VALUE_DTYPE,
                        GRAD_STRIDES_UNIFORM,
                        GRAD_ROW_STRIDE,
                        GRAD_FEATURE_STRIDE,
                    )
                else:
                    value_base, value_stride = _select_source_pointer(
                        values,
                        source_lookup_offsets,
                        token,
                        ROW_STRIDES,
                        FEATURE_STRIDES,
                        L2,
                    )
                    grad_value_base, grad_value_stride = _select_source_pointer(
                        grad_values,
                        source_lookup_offsets,
                        token,
                        GRAD_ROW_STRIDES,
                        GRAD_FEATURE_STRIDES,
                        L2,
                    )
                value_base = tl.broadcast_to(value_base, (SOURCE_TILE,))
                value_stride = tl.broadcast_to(value_stride, (SOURCE_TILE,))
                grad_value_base = tl.broadcast_to(grad_value_base, (SOURCE_TILE,))
                grad_value_stride = tl.broadcast_to(
                    grad_value_stride, (SOURCE_TILE,)
                )
            else:
                source_offsets = (source_block * SOURCE_TILE).to(tl.int64) + tl.arange(
                    0, SOURCE_TILE
                )
                source_mask = source_offsets < n_sources
                value_base = values + source_offsets[:, None] * n_tokens * D + token * D
                grad_value_base = (
                    grad_values + source_offsets[:, None] * n_tokens * D + token * D
                )
                value_stride = 1
                grad_value_stride = 1
            if LIST_SOURCES:
                value_ptr = (
                    value_base[:, None]
                    + d_offsets[None, :] * value_stride[:, None]
                )
            else:
                value_ptr = value_base + d_offsets[None, :]
            value = tl.load(
                value_ptr,
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            dweight = tl.sum(value * grad[None, :], axis=1)
            if FUSE_KEY_WITH_VALUE and R < D and BLOCK_R != BLOCK_D:
                tail_d = tl.where(tail_d_mask[None, :], value, 0.0)
                key_inv_rms = tl.load(
                    saved_key_inv_rms + source_offsets * n_tokens + token,
                    mask=source_mask,
                    other=1.0,
                ).to(tl.float32)
                key_d = tail_d * key_inv_rms[:, None]
            elif R == D:
                tail = value
            elif R < D and BLOCK_R == BLOCK_D:
                tail = tl.where(r_mask[None, :], value, 0.0)
            else:
                if LIST_SOURCES:
                    tail_ptr = value_base[:, None] + (
                        (D - R + r_offsets)[None, :] * value_stride[:, None]
                    )
                else:
                    tail_ptr = value_base + (D - R) + r_offsets[None, :]
                tail = tl.load(
                    tail_ptr,
                    mask=source_mask[:, None] & r_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)

            logit = tl.load(
                saved_logit + source_offsets * n_tokens + token,
                mask=source_mask,
                other=0.0,
            ).to(tl.float32)
            if not (FUSE_KEY_WITH_VALUE and R < D and BLOCK_R != BLOCK_D):
                key_inv_rms = tl.load(
                    saved_key_inv_rms + source_offsets * n_tokens + token,
                    mask=source_mask,
                    other=1.0,
                ).to(tl.float32)
                key = tail * key_inv_rms[:, None]
            probability = tl.where(source_mask, tl.exp(logit - lse), 0.0)
            dlogit = probability * (dweight - delta)
            if FUSE_KEY_WITH_VALUE and R < D and BLOCK_R != BLOCK_D:
                scaled_dlogit = dlogit * scale_f32
                if SOURCE_TILE == 2:
                    scaled_dlogit_0, scaled_dlogit_1 = tl.split(scaled_dlogit)
                    key_d_0, key_d_1 = tl.split(key_d.permute(1, 0))
                    grad_query_d += scaled_dlogit_0 * key_d_0
                    grad_query_d += scaled_dlogit_1 * key_d_1
                else:
                    grad_query_d += tl.sum(
                        scaled_dlogit[:, None] * key_d, axis=0
                    )
            elif SOURCE_TILE == 2:
                # Match the serial dQ accumulation order with supported
                # permute+split operations; no tensor integer indexing.
                scaled_dlogit = dlogit * scale_f32
                scaled_dlogit_0, scaled_dlogit_1 = tl.split(scaled_dlogit)
                key_0, key_1 = tl.split(key.permute(1, 0))
                grad_query += scaled_dlogit_0 * key_0
                grad_query += scaled_dlogit_1 * key_1
            else:
                grad_query += tl.sum((dlogit * scale_f32)[:, None] * key, axis=0)

            if FUSE_KEY_WITH_VALUE and R < D and BLOCK_R != BLOCK_D:
                # Reuse the D-wide key tile and fold its derivative into the
                # same full-width BF16 gradient store.
                direct = probability[:, None] * grad[None, :]
                scaled_dlogit = dlogit * scale_f32
                grad_key_d = key_inv_rms[:, None] * (
                    scaled_dlogit[:, None] * q_key[None, :]
                    - key_d * (dlogit * logit / R)[:, None]
                )
                if LIST_SOURCES:
                    grad_value_ptr = (
                        grad_value_base[:, None]
                        + d_offsets[None, :] * grad_value_stride[:, None]
                    )
                else:
                    grad_value_ptr = grad_value_base + d_offsets[None, :]
                tl.store(
                    grad_value_ptr,
                    direct + tl.where(tail_d_mask[None, :], grad_key_d, 0.0),
                    mask=source_mask[:, None] & d_mask[None, :],
                )
            elif (
                R < D
                and BLOCK_R < BLOCK_D
                and SOURCE_STRIDES_UNIFORM
                and GRAD_STRIDES_UNIFORM
                and QUERY_STRIDE == 1
                and SOURCE_FEATURE_STRIDE == 1
                and GRAD_FEATURE_STRIDE == 1
                and not FUSE_KEY_WITH_VALUE
                and STORE_FAMILY == 1
            ):
                # The compact family keeps the key derivative in R lanes and
                # covers the physical D-wide gradient with disjoint prefix
                # and tail stores.  This writes every valid output lane once.
                direct = probability[:, None] * grad[None, :]
                scaled_dlogit = dlogit * scale_f32
                grad_key_r = key_inv_rms[:, None] * (
                    scaled_dlogit[:, None] * q[None, :]
                    - key * (dlogit * logit / R)[:, None]
                )
                if LIST_SOURCES:
                    grad_value_prefix_ptr = (
                        grad_value_base[:, None]
                        + d_offsets[None, :] * grad_value_stride[:, None]
                    )
                    grad_value_tail_ptr = grad_value_base[:, None] + (
                        tail_offsets[None, :] * grad_value_stride[:, None]
                    )
                else:
                    grad_value_prefix_ptr = grad_value_base + d_offsets[None, :]
                    grad_value_tail_ptr = grad_value_base + tail_offsets[None, :]
                tl.store(
                    grad_value_prefix_ptr,
                    direct,
                    mask=source_mask[:, None] & prefix_d_mask[None, :],
                )
                tl.store(
                    grad_value_tail_ptr,
                    probability[:, None] * grad_tail[None, :] + grad_key_r,
                    mask=source_mask[:, None] & r_mask[None, :],
                )
            elif R < D:
                # ONE_STORE_DV is fixed true: map the folded key derivative
                # into the final R lanes of the existing D wide store.
                direct = probability[:, None] * grad[None, :]
                grad_tail_d_offsets = tl.maximum(d_offsets - (D - R), 0).to(tl.int32)
                if R < D and BLOCK_R == BLOCK_D:
                    q_d = q
                elif LIST_SOURCES:
                    q_d = tl.load(
                        query + grad_tail_d_offsets * tl.cast(QUERY_STRIDE, tl.int64),
                        mask=tail_d_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    q_d = tl.load(
                        query + grad_tail_d_offsets, mask=tail_d_mask, other=0.0
                    ).to(tl.float32)
                key_d = tl.where(
                    tail_d_mask[None, :], value * key_inv_rms[:, None], 0.0
                )
                scaled_dlogit = dlogit * scale_f32
                grad_key_d = key_inv_rms[:, None] * (
                    scaled_dlogit[:, None] * q_d[None, :]
                    - key_d * (dlogit * logit / R)[:, None]
                )
                if LIST_SOURCES:
                    grad_value_ptr = (
                        grad_value_base[:, None]
                        + d_offsets[None, :] * grad_value_stride[:, None]
                    )
                else:
                    grad_value_ptr = grad_value_base + d_offsets[None, :]
                tl.store(
                    grad_value_ptr,
                    direct + tl.where(tail_d_mask[None, :], grad_key_d, 0.0),
                    mask=source_mask[:, None] & d_mask[None, :],
                )
            else:
                raw_grad_key = (dlogit * scale_f32)[:, None] * q[None, :]
                grad_key = key_inv_rms[:, None] * (
                    raw_grad_key - key * tl.sum(raw_grad_key * key, axis=1)[:, None] / R
                )
                if LIST_SOURCES:
                    grad_value_ptr = (
                        grad_value_base[:, None]
                        + d_offsets[None, :] * grad_value_stride[:, None]
                    )
                else:
                    grad_value_ptr = grad_value_base + d_offsets[None, :]
                tl.store(
                    grad_value_ptr,
                    probability[:, None] * grad[None, :] + grad_key,
                    mask=source_mask[:, None] & d_mask[None, :],
                )

        if FUSE_KEY_WITH_VALUE and R < D and BLOCK_R != BLOCK_D:
            tl.store(
                grad_query_token + token * R + tail_d_offsets,
                grad_query_d,
                mask=tail_d_mask,
            )
        else:
            tl.store(grad_query_token + token * R + r_offsets, grad_query, mask=r_mask)

    @triton.jit
    def _packed_query_reduce_kernel(
        grad_query_token,
        grad_query,
        n_tokens,
        R: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        rank_block = tl.program_id(0)
        r_offsets = rank_block * BLOCK_R + tl.arange(0, BLOCK_R)
        r_mask = r_offsets < R
        accumulator = tl.zeros((BLOCK_R,), tl.float32)
        for token_base in range(0, n_tokens, BLOCK_N):
            token_offsets = (token_base + tl.arange(0, BLOCK_N)).to(tl.int64)
            mask = (token_offsets[:, None] < n_tokens) & r_mask[None, :]
            accumulator += tl.sum(
                tl.load(
                    grad_query_token + token_offsets[:, None] * R + r_offsets[None, :],
                    mask=mask,
                    other=0.0,
                ).to(tl.float32),
                axis=0,
            )
        tl.store(grad_query + r_offsets, accumulator, mask=r_mask)


if triton is not None and _triton_op is not None and _wrap_triton is not None:

    @_triton_op("attnres::_fixed_tail_forward_with_aux", mutates_args={})
    def _fixed_tail_forward_with_aux_triton_op(
        values: torch.Tensor,
        query: torch.Tensor,
        eps: float,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sources, width, rank = _validate_inputs(values, query)
        if values.ndim != 3 or not values.is_contiguous() or not query.is_contiguous():
            raise ValueError("fixed-tail CUDA op requires contiguous packed tensors")
        count = int(values.shape[1])
        source_tile, fuse_key = _launch_policy(width, rank, values.device)
        output = torch.empty((count, width), device=values.device, dtype=values.dtype)
        saved_output_fp32 = torch.empty(
            (count, width), device=values.device, dtype=torch.float32
        )
        saved_key_inv_rms = torch.empty(
            (sources, count), device=values.device, dtype=torch.float32
        )
        saved_logit = torch.empty_like(saved_key_inv_rms)
        saved_lse = torch.empty((count,), device=values.device, dtype=torch.float32)
        block_d = _next_power_of_two(width)
        block_r = _next_power_of_two(rank)
        _wrap_triton(_packed_online_forward_kernel)[(count,)](
            values,
            query,
            output,
            saved_output_fp32,
            saved_key_inv_rms,
            saved_logit,
            saved_lse,
            count,
            sources,
            float(eps),
            float(scale),
            D=width,
            R=rank,
            BLOCK_D=block_d,
            BLOCK_R=block_r,
            SOURCE_TILE=source_tile,
            QUERY_STRIDE=1,
            OUTPUT_ROW_STRIDE=width,
            OUTPUT_D_STRIDE=1,
            L2=1,
            ROW_STRIDES=(0,),
            FEATURE_STRIDES=(1,),
            LIST_SOURCES=False,
            SOURCE_RECORDS=False,
            VALUE_DTYPE=tl.bfloat16,
            SOURCE_STRIDES_UNIFORM=True,
            SOURCE_ROW_STRIDE=0,
            SOURCE_FEATURE_STRIDE=1,
            FUSE_KEY_WITH_VALUE=fuse_key,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        return output, saved_output_fp32, saved_key_inv_rms, saved_logit, saved_lse

    @_triton_op("attnres::_fixed_tail_backward", mutates_args={})
    def _fixed_tail_backward_triton_op(
        values: torch.Tensor,
        query: torch.Tensor,
        saved_output_fp32: torch.Tensor,
        grad_output: torch.Tensor,
        saved_key_inv_rms: torch.Tensor,
        saved_logit: torch.Tensor,
        saved_lse: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sources, width, rank = _validate_inputs(values, query)
        if values.ndim != 3 or not values.is_contiguous() or not query.is_contiguous():
            raise ValueError("fixed-tail CUDA op requires contiguous packed tensors")
        if not ONE_STORE_DV:  # The module intentionally exposes one path only.
            raise RuntimeError("fixed-tail dV folding must remain enabled")
        count = int(values.shape[1])
        source_tile, fuse_key = _launch_policy(width, rank, values.device)
        grad_values = torch.empty_like(values)
        grad_query_token = torch.empty(
            (count, rank), device=values.device, dtype=torch.float32
        )
        grad_query_fp32 = torch.empty((rank,), device=query.device, dtype=torch.float32)
        block_d = _next_power_of_two(width)
        block_r = _next_power_of_two(rank)
        _wrap_triton(_packed_online_backward_kernel)[(count,)](
            values,
            query,
            saved_output_fp32,
            grad_output,
            saved_key_inv_rms,
            saved_logit,
            saved_lse,
            grad_values,
            grad_query_token,
            count,
            sources,
            float(scale),
            D=width,
            R=rank,
            BLOCK_D=block_d,
            BLOCK_R=block_r,
            SOURCE_TILE=source_tile,
            QUERY_STRIDE=1,
            GRAD_OUTPUT_ROW_STRIDE=width,
            GRAD_OUTPUT_D_STRIDE=1,
            L2=1,
            ROW_STRIDES=(0,),
            FEATURE_STRIDES=(1,),
            GRAD_ROW_STRIDES=(0,),
            GRAD_FEATURE_STRIDES=(1,),
            LIST_SOURCES=False,
            SOURCE_RECORDS=False,
            VALUE_DTYPE=tl.bfloat16,
            SOURCE_STRIDES_UNIFORM=True,
            SOURCE_ROW_STRIDE=0,
            SOURCE_FEATURE_STRIDE=1,
            GRAD_STRIDES_UNIFORM=True,
            GRAD_ROW_STRIDE=0,
            GRAD_FEATURE_STRIDE=1,
            FUSE_KEY_WITH_VALUE=fuse_key,
        )
        _wrap_triton(_packed_query_reduce_kernel)[(triton.cdiv(rank, QUERY_BLOCK_R),)](
            grad_query_token,
            grad_query_fp32,
            count,
            R=rank,
            BLOCK_N=QUERY_BLOCK_N,
            BLOCK_R=QUERY_BLOCK_R,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        return grad_values, grad_query_fp32.to(query.dtype)

    def _fixed_tail_setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        values, query, _eps, scale = inputs
        _output, saved_output_fp32, saved_key_inv_rms, saved_logit, saved_lse = output
        ctx.save_for_backward(
            values, query, saved_output_fp32, saved_key_inv_rms, saved_logit, saved_lse
        )
        ctx.scale = scale

    def _fixed_tail_backward(
        ctx: Any,
        grad_output: torch.Tensor | None,
        _grad_saved_output: torch.Tensor | None = None,
        _grad_saved_key_inv_rms: torch.Tensor | None = None,
        _grad_saved_logit: torch.Tensor | None = None,
        _grad_saved_lse: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        if grad_output is None:
            return None, None, None, None
        (
            values,
            query,
            saved_output_fp32,
            saved_key_inv_rms,
            saved_logit,
            saved_lse,
        ) = ctx.saved_tensors
        grad_output = _prepare_grad_output(grad_output, values)
        grad_values, grad_query = _fixed_tail_backward_triton_op(
            values,
            query,
            saved_output_fp32,
            grad_output,
            saved_key_inv_rms,
            saved_logit,
            saved_lse,
            float(ctx.scale),
        )
        return (
            grad_values if ctx.needs_input_grad[0] else None,
            grad_query if ctx.needs_input_grad[1] else None,
            None,
            None,
        )

    torch.library.register_autograd(
        _fixed_tail_forward_with_aux_triton_op,
        _fixed_tail_backward,
        setup_context=_fixed_tail_setup_context,
    )
else:
    _packed_online_forward_kernel = None
    _packed_online_backward_kernel = None
    _packed_query_reduce_kernel = None
    _fixed_tail_forward_with_aux_triton_op = None
    _fixed_tail_backward_triton_op = None


def fused_attnres(
    values: torch.Tensor,
    query: torch.Tensor,
    *,
    eps: float = _EPS,
    scale: float = 1.0,
) -> torch.Tensor:
    """Mix packed ``values [S,...,D]`` using the implicit final ``R`` tail.

    Values and queries use BF16 storage.  On CUDA, the adapter makes only the
    packed tensor and one dimensional query contiguous before crossing the
    Triton autograd boundary.  This module accepts packed tensors only; other
    container APIs stay outside.
    """
    sources, width, _rank = _validate_inputs(values, query)
    _validate_scalar(eps, "eps", positive=True)
    _validate_scalar(scale, "scale")
    eps_value, scale_value = float(eps), float(scale)
    if not values.is_cuda:
        raise RuntimeError("fused_attnres requires CUDA BF16 tensors")
    if triton is None or _fixed_tail_forward_with_aux_triton_op is None:
        raise RuntimeError("fused_attnres requires Triton on CUDA")

    kernel_values = values if values.is_contiguous() else values.contiguous()
    kernel_values = kernel_values.reshape(sources, -1, width)
    kernel_query = query if query.is_contiguous() else query.contiguous()
    (
        output,
        _saved_output,
        _saved_key_inv_rms,
        _saved_logit,
        _saved_lse,
    ) = _fixed_tail_forward_with_aux_triton_op(
        kernel_values,
        kernel_query,
        eps_value,
        scale_value,
    )
    return output.reshape(tuple(values.shape[1:]))


__all__ = ["fused_attnres"]
