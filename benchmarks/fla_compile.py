"""Opaque ``torch.compile`` bridge for the unmodified native FLA AttnRes kernels.

The native FLA implementation builds a Python pointer tuple and checks every
source's address.  Dynamo may execute that host code with fake tensors while
capturing a model, which makes the native call unsuitable for a fullgraph
model even when the eventual real tensors are valid.  This module keeps the
native host functions and Triton/Gluon kernels unchanged and places the host
call behind a CUDA-only ``torch.library.custom_op``.  The custom op receives a
Tensor[] source list, so the model's existing source views can be passed
without padding or a second source stack.

Only the frozen standard FLA contract is bridged here: static full-width query,
unit RMS weight, no output normalization, epsilon ``2**-23``, scale ``1``,
BF16 storage (FP32 is retained for local/reference checks), and checkpoint 1
for qualification.  Checkpoint 0 is exposed for diagnosis and deliberately
marked experimental because the native path has a known gradient failure.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, List, Literal

import torch
from torch import Tensor

from .competitors import (
    FLA_LICENSE_SHA256,
    FLA_PACKAGE_SHA256,
    FLA_REPOSITORY,
    FLA_REVISION,
    FLA_SOURCE_URL,
    FLA_SOURCE_HASHES,
    GLUON_COMPILE_ENVELOPE,
    _gluon_compile_envelope,
)
from .vendor_identity import (
    candidate_roots,
    checkout_identity,
    require_module_origins,
)
from .gluon_compat import install_gluon_barrier_compatibility


EPS = 2**-23
Implementation = Literal["triton", "gluon"]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_VENDOR_ROOT = _PROJECT_ROOT.parent / "vendor" / "fla"
_FLA_ENVIRONMENT = (
    "ATTNRES_FLA_DIR",
    "FLA_ROOT",
    "FLASH_LINEAR_ATTENTION_ROOT",
    "VENDOR_FLA_ROOT",
)

try:  # The package remains importable on the CPU development host.
    _custom_op = torch.library.custom_op
    _register_fake = torch.library.register_fake
    _register_autograd = torch.library.register_autograd
except AttributeError:  # pragma: no cover - unsupported old PyTorch only.
    _custom_op = None
    _register_fake = None
    _register_autograd = None


def resolve_vendor_root(vendor_root: str | Path | None = None) -> Path:
    """Resolve one configured FLA source tree without importing Triton.

    The resolver is intentionally path-only so release preflight tests can
    inspect a synthetic checkout. Runtime entrypoints call ``_fla_identity``
    before importing or launching native code.
    """

    candidates = candidate_roots(
        _PROJECT_ROOT,
        vendor_root,
        environment=_FLA_ENVIRONMENT,
        defaults=(
            _DEFAULT_VENDOR_ROOT,
            _PROJECT_ROOT / "vendor" / "fla",
            _PROJECT_ROOT / "vendor" / "flash-linear-attention",
            _PROJECT_ROOT.parent / "vendor" / "flash-linear-attention",
            _PROJECT_ROOT.parent.parent / "vendor" / "fla",
            _PROJECT_ROOT.parent.parent / "vendor" / "flash-linear-attention",
        ),
    )
    attempted = ", ".join(str(path) for path in candidates) or "none"
    for root in candidates:
        if (root / "fla" / "ops" / "attnres").is_dir():
            return root
    raise ImportError(f"pinned FLA AttnRes checkout was not found; tried {attempted}")


def _fla_identity(root: Path) -> dict[str, Any]:
    """Verify the exact FLA source before registering/launching custom ops."""

    return checkout_identity(
        root,
        expected_revision=FLA_REVISION,
        files={**FLA_SOURCE_HASHES, "LICENSE": FLA_LICENSE_SHA256},
        package_dir="fla",
        package_sha256=FLA_PACKAGE_SHA256,
        expected_origin=FLA_REPOSITORY,
    )


def _vendor_root(vendor_root: str | Path | None = None) -> Path:
    """Backward-compatible private alias for the resolver."""

    return resolve_vendor_root(vendor_root)


def _native_module(
    implementation: Implementation,
    vendor_root: str | Path | None = None,
) -> Any:
    if implementation not in ("triton", "gluon"):
        raise ValueError(f"implementation must be 'triton' or 'gluon', got {implementation!r}")
    root = resolve_vendor_root(vendor_root)
    _fla_identity(root)
    require_module_origins("fla", root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    module_name = (
        "fla.ops.attnres.fused"
        if implementation == "triton"
        else "fla.ops.attnres.backends.gluon"
    )
    if implementation == "gluon":
        install_gluon_barrier_compatibility()
    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve()
    expected_file = root / (
        "fla/ops/attnres/fused.py"
        if implementation == "triton"
        else "fla/ops/attnres/backends/gluon.py"
    )
    if module_file != expected_file.resolve():
        raise ImportError(
            f"loaded FLA module {module_file} does not match pinned source "
            f"{expected_file.resolve()}"
        )
    require_module_origins("fla", root)
    return module


def _native_functions(
    implementation: Implementation,
    vendor_root: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load direct native host functions without touching FLA dispatch."""

    if implementation == "triton":
        module = _native_module(implementation, vendor_root)
        return module.fused_attnres_fwd, module.fused_attnres_bwd
    if implementation == "gluon":
        module = _native_module(implementation, vendor_root)
        # These are the backend's native host launchers.  They are intentionally
        # called directly so no dispatch environment variable is changed.
        return module._fused_attnres_fwd, module._fused_attnres_bwd
    raise ValueError(f"implementation must be 'triton' or 'gluon', got {implementation!r}")


def _native_source_table(
    implementation: Implementation,
    sources: Sequence[Tensor],
    vendor_root: str | Path | None = None,
) -> tuple[Tensor, ...]:
    """Build only the native address table inside the opaque CUDA op.

    Triton needs its fixed-length pointer tuple; Gluon needs only its native
    alignment check.  The Triton table pads addresses, never tensor storage,
    and is therefore the same required host setup as the unmodified FLA call.
    Keeping it here prevents Dynamo/fake tensors from seeing ``data_ptr``.
    """

    residuals = tuple(sources)
    if implementation == "triton":
        module = _native_module(implementation, vendor_root)
        return module._build_ptr_table(residuals)
    if implementation == "gluon":
        accepted, reason, _metrics = _gluon_compile_envelope(
            len(residuals), int(residuals[0].shape[-1])
        )
        if not accepted:
            raise ValueError(reason or "Gluon compile envelope rejected the case")
        module = _native_module(implementation, vendor_root)
        return module._check_sources(residuals)
    raise ValueError(f"implementation must be 'triton' or 'gluon', got {implementation!r}")


def _as_sources(values: Tensor | Sequence[Tensor]) -> list[Tensor]:
    """Return source views while preserving the caller's existing layout."""

    if isinstance(values, Tensor):
        if values.ndim < 2:
            raise ValueError("values must have shape [S,...,D]")
        # unbind creates source views; it does not pad, stack, or copy them.
        sources = list(values.unbind(0))
    else:
        sources = list(values)
    if not sources:
        raise ValueError("at least one residual source is required")
    first = sources[0]
    if not isinstance(first, Tensor) or first.ndim < 1:
        raise TypeError("residual sources must be tensors with a feature axis")
    for source in sources:
        if not isinstance(source, Tensor):
            raise TypeError("residual sources must be tensors")
        if source.shape != first.shape:
            raise ValueError("all residual sources must have the same shape")
        if source.device != first.device or source.dtype != first.dtype:
            raise ValueError("all residual sources must share device and dtype")
        if not source.is_contiguous():
            raise ValueError("residual sources must be contiguous for native FLA strides")
    return sources


def _validate_call(
    sources: Sequence[Tensor],
    query: Tensor,
    rms_weight: Tensor,
    *,
    eps: float,
    scale: float,
    checkpoint_level: int,
) -> None:
    first = sources[0]
    if not first.is_cuda or query.device != first.device:
        raise RuntimeError("the FLA compile bridge requires CUDA tensors")
    if first.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("values must use BF16 or FP32 storage")
    if query.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("query must use BF16 or FP32 storage")
    if query.ndim != 1 or query.numel() != first.shape[-1]:
        raise ValueError("the native FLA bridge requires a static full-width query [D]")
    if rms_weight.shape != query.shape or rms_weight.dtype != query.dtype:
        raise ValueError("rms_weight must be a query-shaped tensor with query dtype")
    if not query.is_contiguous() or not rms_weight.is_contiguous():
        raise ValueError("query and rms_weight must be contiguous for native FLA strides")
    if float(eps) != EPS or float(scale) != 1.0:
        raise ValueError("the native FLA bridge uses eps=2**-23 and scale=1")
    if checkpoint_level not in (0, 1):
        raise ValueError("checkpoint_level must be 0 or 1")


def _forward_native(
    implementation: Implementation,
    sources: Sequence[Tensor],
    query: Tensor,
    rms_weight: Tensor,
    checkpoint_level: int,
    vendor_root: str | Path | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    forward, _ = _native_functions(implementation, vendor_root)
    residuals = tuple(sources)
    native_table = _native_source_table(implementation, residuals, vendor_root)
    output, output_pre, rstd, logit, lse = forward(
        q=query,
        residuals=residuals,
        res=native_table,
        w=rms_weight,
        ow=None,
        eps=EPS,
        scale=1.0,
        checkpoint_level=int(checkpoint_level),
    )
    if output_pre is None:
        # Custom-op schemas cannot return Optional tensors.  Native checkpoint
        # 1 ignores this placeholder in backward.
        output_pre = torch.empty((0,), device=output.device, dtype=output.dtype)
    return output, output_pre, rstd, logit, lse


def _backward_native(
    implementation: Implementation,
    sources: Sequence[Tensor],
    query: Tensor,
    rms_weight: Tensor,
    grad_output: Tensor,
    output_pre: Tensor,
    rstd: Tensor,
    logit: Tensor,
    lse: Tensor,
    checkpoint_level: int,
    vendor_root: str | Path | None = None,
) -> list[Tensor]:
    _, backward = _native_functions(implementation, vendor_root)
    residuals = tuple(sources)
    native_table = _native_source_table(implementation, residuals, vendor_root)
    if not grad_output.is_contiguous():
        # The direct host launcher bypasses FLA's input_guard, which normally
        # makes this copy before indexing the native contiguous kernel input.
        grad_output = grad_output.contiguous()
    dvs, grad_query, grad_weight, _ = backward(
        do=grad_output,
        q=query,
        residuals=residuals,
        res=native_table,
        w=rms_weight,
        ow=None,
        o_pre=None if int(checkpoint_level) == 1 else output_pre,
        rstd=rstd,
        logit=logit,
        lse=lse,
        eps=EPS,
        scale=1.0,
        checkpoint_level=int(checkpoint_level),
    )
    return [*dvs, grad_query, grad_weight]


def _fake_forward(
    sources: Sequence[Tensor], query: Tensor, rms_weight: Tensor, checkpoint_level: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    del rms_weight
    first = sources[0]
    output_pre = first.new_empty(first.shape if int(checkpoint_level) == 0 else (0,))
    stats_shape = (len(sources), *first.shape[:-1])
    stats = first.new_empty(stats_shape, dtype=torch.float32)
    lse = first.new_empty(first.shape[:-1], dtype=torch.float32)
    return first.new_empty(first.shape), output_pre, stats, stats.new_empty(stats.shape), lse


def _fake_backward(
    sources: Sequence[Tensor],
    query: Tensor,
    rms_weight: Tensor,
    grad_output: Tensor,
    output_pre: Tensor,
    rstd: Tensor,
    logit: Tensor,
    lse: Tensor,
    checkpoint_level: int,
) -> List[Tensor]:
    del grad_output, output_pre, rstd, logit, lse, checkpoint_level
    return [source.new_empty(source.shape) for source in sources] + [
        query.new_empty(query.shape),
        rms_weight.new_empty(rms_weight.shape),
    ]


if _custom_op is not None:

    @_custom_op("attnres_fla_compile::forward", mutates_args=(), device_types="cuda")
    def _forward_op(
        sources: Sequence[Tensor],
        query: Tensor,
        rms_weight: Tensor,
        checkpoint_level: int,
        implementation: str,
        vendor_root: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return _forward_native(
            implementation, sources, query, rms_weight, checkpoint_level, vendor_root
        )


    @_custom_op("attnres_fla_compile::backward", mutates_args=(), device_types="cuda")
    def _backward_op(
        sources: Sequence[Tensor],
        query: Tensor,
        rms_weight: Tensor,
        grad_output: Tensor,
        output_pre: Tensor,
        rstd: Tensor,
        logit: Tensor,
        lse: Tensor,
        checkpoint_level: int,
        implementation: str,
        vendor_root: str,
    ) -> List[Tensor]:
        return _backward_native(
            implementation,
            sources,
            query,
            rms_weight,
            grad_output,
            output_pre,
            rstd,
            logit,
            lse,
            checkpoint_level,
            vendor_root,
        )

else:  # pragma: no cover - only unsupported PyTorch versions.
    _forward_op = _backward_op = None


if _register_fake is not None and _custom_op is not None:

    @_register_fake(_forward_op)
    def _forward_fake(
        sources: Sequence[Tensor],
        query: Tensor,
        rms_weight: Tensor,
        checkpoint_level: int,
        implementation: str,
        vendor_root: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del implementation, vendor_root
        return _fake_forward(sources, query, rms_weight, checkpoint_level)


    @_register_fake(_backward_op)
    def _backward_fake(
        sources: Sequence[Tensor],
        query: Tensor,
        rms_weight: Tensor,
        grad_output: Tensor,
        output_pre: Tensor,
        rstd: Tensor,
        logit: Tensor,
        lse: Tensor,
        checkpoint_level: int,
        implementation: str,
        vendor_root: str,
    ) -> List[Tensor]:
        del implementation, vendor_root
        return _fake_backward(
            sources, query, rms_weight, grad_output, output_pre, rstd, logit, lse, checkpoint_level
        )


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, ...]) -> None:
    sources, query, rms_weight, checkpoint_level, implementation, vendor_root = inputs
    output_value, output_pre, rstd, logit, lse = output
    del output_value
    ctx.save_for_backward(*sources, query, rms_weight, output_pre, rstd, logit, lse)
    ctx.source_count = len(sources)
    ctx.checkpoint_level = int(checkpoint_level)
    ctx.implementation = implementation
    ctx.vendor_root = vendor_root


def _registered_backward(
    ctx: Any,
    grad_output: Tensor | None,
    _grad_output_pre: Tensor | None,
    _grad_rstd: Tensor | None,
    _grad_logit: Tensor | None,
    _grad_lse: Tensor | None,
) -> tuple[list[Tensor | None], Tensor | None, Tensor | None, None, None, None]:
    """Traceable formula that calls the opaque native backward custom op."""

    if grad_output is None:
        return [None] * ctx.source_count, None, None, None, None, None
    saved = ctx.saved_tensors
    count = int(ctx.source_count)
    sources = list(saved[:count])
    query, rms_weight, output_pre, rstd, logit, lse = saved[count:]
    gradients = _backward_op(
        sources,
        query,
        rms_weight,
        grad_output,
        output_pre,
        rstd,
        logit,
        lse,
        ctx.checkpoint_level,
        ctx.implementation,
        ctx.vendor_root,
    )
    return (
        gradients[:count],
        gradients[count],
        gradients[count + 1],
        None,
        None,
        None,
    )


if _register_autograd is not None and _custom_op is not None:
    _register_autograd(_forward_op, _registered_backward, setup_context=_setup_context)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash_metadata(
    implementation: Implementation = "triton",
    vendor_root: str | Path | None = None,
    checkpoint_level: int = 1,
) -> dict[str, Any]:
    """Return immutable source identity and the bridge's explicit contract."""

    root = resolve_vendor_root(vendor_root)
    if implementation not in ("triton", "gluon"):
        raise ValueError("implementation must be 'triton' or 'gluon'")
    if checkpoint_level not in (0, 1):
        raise ValueError("checkpoint_level must be 0 or 1")
    identity = _fla_identity(root)
    metadata = {
        "bridge": "fla_native_compile_custom_op",
        "implementation": implementation,
        "checkpoint_level": int(checkpoint_level),
        "qualification_eligible": int(checkpoint_level) == 1,
        "checkpoint0_status": "experimental_native_gradient_failure",
        "rms_weight": "ones",
        "model_rms_weight_allocation": "nonpersistent_buffer",
        "model_rms_weight_reuse": "one_buffer_per_model",
        "direct_call_fallback": "query_ones",
        "compiled_model_fill_launches_per_step": 0,
        "compiled_model_fill_launches_avoided_per_step": 1,
        "output_rms_weight": None,
        "rms_eps": EPS,
        "scale": 1.0,
        "storage": "BF16_or_FP32",
        "equation_dtype": "FP32_native_kernel_accumulation",
        "native_functions": ["fused_attnres_fwd", "fused_attnres_bwd"],
        "source_table": "native_address_only_pointer_table",
        "accepts_source_list": True,
        "model_source_argument": "sequence_of_contiguous_source_tensors",
        "model_forced_source_stack": False,
        "vendor_root": str(root),
        "vendor_source_url": FLA_SOURCE_URL,
        "expected_origin": FLA_REPOSITORY,
        "vendor_origin": identity["origin"],
        "vendor_revision": identity["revision"],
        "expected_vendor_revision": FLA_REVISION,
        "vendor_git_dirty": identity["git_dirty"],
        "vendor_package_sha256": identity.get("package_sha256"),
        "expected_vendor_package_sha256": FLA_PACKAGE_SHA256,
        "vendor_file_hashes": {
            name: identity["files"][name] for name in FLA_SOURCE_HASHES
        },
        "expected_vendor_file_hashes": dict(FLA_SOURCE_HASHES),
        "vendor_license_sha256": identity["files"].get("LICENSE"),
        "expected_vendor_license_sha256": FLA_LICENSE_SHA256,
        "qualification_oracle": "validation.oracle.oracle",
        "qualification_checks": ["output", "all_value_gradients", "query_gradient"],
        "model_qualification": "benchmarks.run._model_qualification",
        "capability_limits": {
            "rank": "R=D only",
            "values": "full-width values; implicit tail keys",
            "sources": "1<=S<=129",
            "width": "1<=D<=8192",
            "dtype": "BF16 or FP32 storage",
            "device": "CUDA only",
            "checkpoint1": "qualification eligible",
            "checkpoint0": "diagnostic only; native gradient failure known",
        },
        "adapter_file": str(Path(__file__).resolve()),
        "adapter_sha256": _file_sha256(Path(__file__).resolve()),
    }
    if implementation == "gluon":
        metadata["compile_envelope"] = dict(GLUON_COMPILE_ENVELOPE)
        metadata["dependency_compatibility"] = {
            "required_on_triton_3_7_1": "thread_barrier exact alias to barrier",
            "vendor_call_form": "zero_argument",
            "barrier_cluster": False,
            "installed_before_vendor_import": True,
            "vendor_source_modified": False,
            "runtime_mode_recorded_by": (
                "benchmarks.gluon_compat.install_gluon_barrier_compatibility"
            ),
        }
    return metadata


def make_model_backend(
    implementation: Implementation = "triton",
    *,
    checkpoint_level: int = 1,
    vendor_root: str | Path | None = None,
):
    """Build a callable accepted by ``benchmarks.model.make_model``.

    The callable keeps the model's ``(values, query)`` contract. Native FLA
    accepts standard full-width reads; sliced reads use the project's operator.
    """

    if implementation not in ("triton", "gluon"):
        raise ValueError("implementation must be 'triton' or 'gluon'")
    if checkpoint_level not in (0, 1):
        raise ValueError("checkpoint_level must be 0 or 1")
    if _custom_op is None:
        raise RuntimeError("torch.library.custom_op is required for the compile bridge")
    vendor_root_value = str(resolve_vendor_root(vendor_root))
    metadata = source_hash_metadata(implementation, vendor_root_value, checkpoint_level)

    def backend(
        values: Tensor | Sequence[Tensor],
        query: Tensor,
        *,
        eps: float = EPS,
        scale: float = 1.0,
        rms_weight: Tensor | None = None,
    ) -> Tensor:
        sources = _as_sources(values)
        if rms_weight is None:
            # Direct operator callers retain the historical convenience path.
            # Model calls pass their preallocated non-persistent buffer, which
            # keeps this allocation outside the compiled training step.
            rms_weight = torch.ones(
                query.shape,
                device=query.device,
                dtype=query.dtype,
            )
        _validate_call(
            sources,
            query,
            rms_weight,
            eps=eps,
            scale=scale,
            checkpoint_level=checkpoint_level,
        )
        output, _output_pre, _rstd, _logit, _lse = _forward_op(
            sources,
            query,
            rms_weight,
            checkpoint_level,
            implementation,
            vendor_root_value,
        )
        return output

    backend.__name__ = f"fla_{implementation}_compile_backend"
    # ``benchmarks.model`` uses this explicit capability to pass the native
    # source sequence directly.  No other callable receives this treatment.
    backend.accepts_source_list = True  # type: ignore[attr-defined]
    # ``benchmarks.model`` owns one non-persistent unit-weight buffer per model
    # and passes it through the residual read.  The optional keyword preserves
    # direct backend-call compatibility for operator-level checks.
    backend.accepts_rms_weight = True  # type: ignore[attr-defined]
    backend.source_hash_metadata = metadata  # type: ignore[attr-defined]
    backend.vendor_root = vendor_root_value  # type: ignore[attr-defined]
    return backend


# Short aliases make the bridge discoverable from benchmark configuration
# without changing the existing ``benchmarks.model`` backend contract.
model_backend = make_model_backend
fla_compile_backend = make_model_backend


__all__ = [
    "EPS",
    "fla_compile_backend",
    "make_model_backend",
    "model_backend",
    "resolve_vendor_root",
    "source_hash_metadata",
]
