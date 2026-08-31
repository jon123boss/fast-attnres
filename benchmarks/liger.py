"""Bounded adapter for the pinned Liger-Kernel AttnRes implementation.

Liger's upstream API consumes one packed ``[S, ..., D]`` tensor and a static
full-width query.  This adapter keeps that API behind CUDA-only custom ops so
the surrounding benchmark model can remain a full-graph PyTorch model.  The
vendor package is loaded lazily; importing this module does not import Triton
or require a CUDA device.

The adapter implements the common standard AttnRes contract:

* full-width values and implicit tail keys, with ``R=D`` only;
* parameter-free key RMSNorm (a tensor of ones), ``eps=2**-23``, and scale 1;
* BF16 or FP32 storage, at most 32 source blocks, and ``D<=8192``;
* Full reads and per-read Block reads only.

Sliced keys, projected keys, and the cached Block route, Gluon, and CPU
execution are explicitly unsupported.  Missing or mismatched vendor code is
reported as ``status='missing'``; it is never replaced by the project
reference implementation.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any, Callable

import torch
from torch import Tensor

from attnres._sources import validate_sources

from .vendor_identity import (
    CheckoutIdentityError,
    candidate_roots,
    checkout_identity,
    file_sha256,
    git_output,
    module_origins,
    module_origins_inside,
    require_module_origins,
)


EPS = 2**-23
LIGER_TAG = "v0.8.2"
LIGER_VERSION = "0.8.2"
LIGER_REVISION = "000be60929938fd1358e03524c6ab398b6d421bd"
LIGER_TREE = "746af1fc03014cf47cad895d01cf0d23fddf5e75"
LIGER_REPOSITORY = "https://github.com/linkedin/Liger-Kernel.git"
LIGER_SOURCE = "src/liger_kernel/ops/attn_res.py"
LIGER_SOURCE_URL = (
    f"https://github.com/linkedin/Liger-Kernel/blob/{LIGER_REVISION}/{LIGER_SOURCE}"
)
LIGER_SOURCE_SHA256 = (
    "57da6fed98f794088b2a56223e6c7ef9fc920824f0c483cb0ef0b5a343dab0b1"
)
LIGER_LICENSE = "LICENSE"
LIGER_LICENSE_SHA256 = (
    "3a1ccb0c7274b68e1af2ca1d54b10b662085ca56753400182ecf87ae33f2d1a8"
)
LIGER_NOTICE = "NOTICE"
LIGER_NOTICE_SHA256 = (
    "9e3c27a0f64b87d00df12250cf1bc218b1e2fbc5fffc0bd64737ba8e8357218f"
)
LIGER_PYPROJECT = "pyproject.toml"
LIGER_PYPROJECT_SHA256 = (
    "f55effccdecc17ca87357ed8ecd4e73a58b1a56ee275367bfe5db2827dc9ac22"
)
LIGER_MAX_SOURCES = 32
LIGER_MAX_WIDTH = 8192
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LIGER_ENVIRONMENT = (
    "ATTNRES_LIGER_DIR",
    "LIGER_ROOT",
    "LIGER_KERNEL_ROOT",
    "VENDOR_LIGER_ROOT",
)


try:  # Keep CPU imports working on hosts without a recent torch.library API.
    _custom_op = torch.library.custom_op
    _register_fake = torch.library.register_fake
    _register_autograd = torch.library.register_autograd
except AttributeError:  # pragma: no cover - unsupported old PyTorch only.
    _custom_op = None
    _register_fake = None
    _register_autograd = None


def _is_compiling() -> bool:
    """Return whether the caller is being traced by ``torch.compile``.

    The Liger model arm validates its parameter-free unit RMS buffer during
    eager qualification.  Repeating the tensor-value check while Dynamo is
    tracing would turn ``Tensor.item()`` into a data-dependent guard and make
    an otherwise static model fail fullgraph capture.  Keep this small probe
    Python-only so the compiled graph contains neither the check nor an
    allocation for it.
    """

    compiler = getattr(torch, "compiler", None)
    probe = getattr(compiler, "is_compiling", None)
    if callable(probe):
        return bool(probe())
    dynamo = getattr(torch, "_dynamo", None)
    probe = getattr(dynamo, "is_compiling", None)
    return bool(probe()) if callable(probe) else False


def _validate_rms_weight(rms_weight: Tensor, query: Tensor) -> Tensor:
    """Validate the model-owned unit weight without tracing a tensor scalar.

    Shape, device, and dtype are static contract checks and remain active in
    both eager and compiled calls.  The value check is deliberately eager
    only: ``CausalAttnResLM`` creates a nonpersistent FP32 ones buffer before
    qualification/compilation, so model use is fail-closed before tracing;
    direct eager adapter calls retain the same strict validation.
    """

    if (
        rms_weight.shape != query.shape
        or rms_weight.device != query.device
        or rms_weight.dtype != torch.float32
    ):
        raise ValueError("Liger rms_weight must match the query shape/device and use FP32")
    if not _is_compiling() and not bool(torch.all(rms_weight == 1).item()):
        raise ValueError("Liger adapter only supports parameter-free rms_weight=ones")
    return rms_weight.contiguous()


def _candidate_vendor_roots(
    project_root: Path, configured: str | os.PathLike[str] | None
) -> tuple[Path, ...]:
    return candidate_roots(
        project_root,
        configured,
        environment=_LIGER_ENVIRONMENT,
        defaults=(
            project_root / "vendor" / "Liger-Kernel",
            project_root / "vendor" / "liger-kernel",
            project_root.parent / "vendor" / "Liger-Kernel",
            project_root.parent / "vendor" / "liger-kernel",
            project_root.parent.parent / "vendor" / "Liger-Kernel",
            project_root.parent.parent / "vendor" / "liger-kernel",
        ),
    )


def _source_path(root: Path) -> Path:
    source = (root / LIGER_SOURCE).resolve()
    if not source.is_file():
        raise CheckoutIdentityError(f"pinned Liger source is missing: {source}")
    return source


def _identity(root: Path) -> dict[str, Any]:
    """Verify the exact v0.8.2 checkout before importing Triton."""

    return checkout_identity(
        root,
        expected_revision=LIGER_REVISION,
        expected_tree=LIGER_TREE,
        files={
            LIGER_SOURCE: LIGER_SOURCE_SHA256,
            LIGER_LICENSE: LIGER_LICENSE_SHA256,
            LIGER_NOTICE: LIGER_NOTICE_SHA256,
            LIGER_PYPROJECT: LIGER_PYPROJECT_SHA256,
        },
        version_file=LIGER_PYPROJECT,
        expected_version=LIGER_VERSION,
        expected_origin=LIGER_REPOSITORY,
    )


def resolve_vendor_root(
    vendor_root: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve and verify one exact Liger checkout.

    An explicit argument or configured environment variable is authoritative;
    a bad configured path never causes an automatic search elsewhere.
    """

    root = Path(project_root or _PROJECT_ROOT).expanduser().resolve()
    candidates = _candidate_vendor_roots(root, vendor_root)
    attempted = ", ".join(str(path) for path in candidates) or "none"
    for candidate in candidates:
        if not (candidate / LIGER_SOURCE).is_file():
            continue
        try:
            _identity(candidate)
        except CheckoutIdentityError as exc:
            raise ImportError(f"pinned Liger checkout verification failed: {exc}") from exc
        return candidate
    raise ImportError(f"pinned Liger-Kernel checkout was not found; tried {attempted}")


def find_vendor_root(
    project_root: str | os.PathLike[str] | None = None,
    configured: str | os.PathLike[str] | None = None,
    *,
    vendor_root: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Return the verified root, or ``None`` for optional discovery."""

    if vendor_root is not None:
        if configured is not None:
            raise TypeError("pass only one of configured and vendor_root")
        configured = vendor_root
    try:
        return resolve_vendor_root(configured, project_root)
    except (ImportError, OSError):
        return None


def _load_native(root: Path) -> Any:
    """Import Liger's native module and reject module-cache path confusion."""

    require_module_origins("liger_kernel", root / "src")
    source_root = str((root / "src").resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    importlib.invalidate_caches()
    module = importlib.import_module("liger_kernel.ops.attn_res")
    loaded = Path(getattr(module, "__file__", "")).resolve()
    expected = _source_path(root)
    if loaded != expected:
        raise ImportError(
            f"loaded Liger module {loaded} does not match pinned source {expected}"
        )
    require_module_origins("liger_kernel", root / "src")
    for name in ("attn_res_forward", "attn_res_backward"):
        if not callable(getattr(module, name, None)):
            raise ImportError(f"pinned Liger module is missing {name}")
    return module


def _all_loaded_origins_ok(root: Path) -> bool:
    """Return whether every cached Liger module belongs to ``root/src``."""

    return module_origins_inside("liger_kernel", root / "src")


def _forward_native(
    values: Tensor, query: Tensor, rms_weight: Tensor, vendor_root: str
) -> tuple[Tensor, Tensor, Tensor]:
    module = _load_native(Path(vendor_root))
    output, _values_3d, alpha, rstd = module.attn_res_forward(
        values, query, rms_weight, EPS
    )
    # Liger returns ``values.reshape(...).contiguous()`` as a saved auxiliary.
    # For already-contiguous inputs that tensor aliases ``values``, which is
    # forbidden for outputs of a PyTorch 2.13 custom op.  The autograd setup
    # can save the input directly, so keep the native computation unchanged
    # and do not expose the aliased view as an operator output.
    return output, alpha, rstd


def _backward_native(
    values_3d: Tensor,
    query: Tensor,
    rms_weight: Tensor,
    grad_output: Tensor,
    alpha: Tensor,
    rstd: Tensor,
    vendor_root: str,
) -> tuple[Tensor, Tensor, Tensor]:
    module = _load_native(Path(vendor_root))
    return module.attn_res_backward(
        grad_output,
        values_3d,
        query,
        rms_weight,
        alpha,
        rstd,
        EPS,
    )


def _fake_forward(
    values: Tensor,
    query: Tensor,
    rms_weight: Tensor,
    vendor_root: str,
) -> tuple[Tensor, Tensor, Tensor]:
    del query, rms_weight, vendor_root
    source_count = values.shape[0]
    width = values.shape[-1]
    rows = values.numel() // (source_count * width)
    stats = values.new_empty((rows, source_count), dtype=torch.float32)
    return values.new_empty(values.shape[1:]), stats, stats.new_empty(stats.shape)


def _fake_backward(
    values_3d: Tensor,
    query: Tensor,
    rms_weight: Tensor,
    grad_output: Tensor,
    alpha: Tensor,
    rstd: Tensor,
    vendor_root: str,
) -> tuple[Tensor, Tensor, Tensor]:
    del grad_output, alpha, rstd, vendor_root
    return (
        values_3d.new_empty(values_3d.shape),
        query.new_empty(query.shape),
        rms_weight.new_empty(rms_weight.shape),
    )


if _custom_op is not None:

    @_custom_op("attnres_liger_v082::forward", mutates_args=(), device_types="cuda")
    def _forward_op(
        values: Tensor,
        query: Tensor,
        rms_weight: Tensor,
        vendor_root: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _forward_native(values, query, rms_weight, vendor_root)

    @_custom_op("attnres_liger_v082::backward", mutates_args=(), device_types="cuda")
    def _backward_op(
        values_3d: Tensor,
        query: Tensor,
        rms_weight: Tensor,
        grad_output: Tensor,
        alpha: Tensor,
        rstd: Tensor,
        vendor_root: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _backward_native(
            values_3d,
            query,
            rms_weight,
            grad_output,
            alpha,
            rstd,
            vendor_root,
        )

else:  # pragma: no cover - unsupported PyTorch versions.
    _forward_op = None
    _backward_op = None


if _register_fake is not None and _custom_op is not None:

    @_register_fake(_forward_op)
    def _forward_fake(
        values: Tensor,
        query: Tensor,
        rms_weight: Tensor,
        vendor_root: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _fake_forward(values, query, rms_weight, vendor_root)

    @_register_fake(_backward_op)
    def _backward_fake(
        values_3d: Tensor,
        query: Tensor,
        rms_weight: Tensor,
        grad_output: Tensor,
        alpha: Tensor,
        rstd: Tensor,
        vendor_root: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _fake_backward(
            values_3d,
            query,
            rms_weight,
            grad_output,
            alpha,
            rstd,
            vendor_root,
        )


def _setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    values, query, rms_weight, vendor_root = inputs
    _output, alpha, rstd = output
    ctx.save_for_backward(values, query, rms_weight, alpha, rstd)
    ctx.values_shape = tuple(values.shape)
    ctx.vendor_root = vendor_root


def _registered_backward(
    ctx: Any,
    grad_output: Tensor | None,
    _grad_alpha: Tensor | None,
    _grad_rstd: Tensor | None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, None]:
    if grad_output is None:
        return None, None, None, None
    values, query, rms_weight, alpha, rstd = ctx.saved_tensors
    values_3d = values.reshape(values.shape[0], -1, values.shape[-1])
    d_values, d_query, d_rms_weight = _backward_op(
        values_3d,
        query,
        rms_weight,
        grad_output,
        alpha,
        rstd,
        ctx.vendor_root,
    )
    return d_values.view(ctx.values_shape), d_query, d_rms_weight, None


if _register_autograd is not None and _custom_op is not None:
    _register_autograd(_forward_op, _registered_backward, setup_context=_setup_context)


def _source_tuple(values: Tensor | Sequence[Tensor], query: Tensor) -> tuple[Tensor, ...]:
    """Validate the shared source contract and return original source views."""

    sources = validate_sources(values, query, EPS, 1.0)
    if len(sources) > LIGER_MAX_SOURCES:
        raise ValueError(f"pinned Liger supports at most {LIGER_MAX_SOURCES} sources")
    if query.ndim != 1 or query.shape[-1] != sources[0].shape[-1]:
        raise ValueError("Liger AttnRes is matched only for standard R=D")
    if sources[0].shape[-1] > LIGER_MAX_WIDTH:
        raise ValueError(f"pinned Liger supports at most D={LIGER_MAX_WIDTH}")
    return sources


def _validate_inputs(
    values: Tensor | Sequence[Tensor],
    query: Tensor,
    *,
    require_cuda: bool,
) -> tuple[Tensor, ...]:
    sources = _source_tuple(values, query)
    if require_cuda and not sources[0].is_cuda:
        raise RuntimeError("pinned Liger Triton AttnRes requires CUDA tensors")
    return sources


def _pack(values: Tensor | Sequence[Tensor], sources: Sequence[Tensor]) -> Tensor:
    # Liger's native entrypoint requires one packed tensor.  This copy is part
    # of the adapter boundary and is reported explicitly in metadata.
    if isinstance(values, Tensor):
        return values.contiguous()
    return torch.stack(tuple(sources), dim=0).contiguous()


def _ones_weight(query: Tensor) -> Tensor:
    # The upstream kernel casts normalized keys to ``w_norm.dtype`` before
    # scoring.  FP32 ones preserve the common FP32 equation even when values
    # and the learned query are stored in BF16.
    return torch.ones(query.shape, device=query.device, dtype=torch.float32)


def make_model_backend(
    vendor_root: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> Callable[..., Tensor]:
    """Create the callable used by ``benchmarks.model.make_model``."""

    if _custom_op is None:
        raise RuntimeError("torch.library.custom_op is required for the Liger adapter")
    root = resolve_vendor_root(vendor_root, project_root)
    metadata = source_hash_metadata(root)
    root_string = str(root)

    def backend(
        values: Tensor | Sequence[Tensor] | None = None,
        query: Tensor | None = None,
        *,
        residuals: Tensor | Sequence[Tensor] | None = None,
        rms_weight: Tensor | None = None,
        eps: float = EPS,
        scale: float = 1.0,
    ) -> Tensor:
        if values is None:
            values = residuals  # type: ignore[assignment]
        if query is None:
            raise TypeError("Liger adapter requires a query tensor")
        if values is None:
            raise TypeError("Liger adapter requires residual values")
        if float(eps) != EPS or float(scale) != 1.0:
            raise ValueError("Liger adapter uses eps=2**-23 and scale=1")
        sources = _validate_inputs(values, query, require_cuda=True)
        packed = _pack(values, sources)
        query_arg = query.contiguous()
        if rms_weight is None:
            # Direct calls retain the historical parameter-free fallback.  A
            # compiled model passes the preallocated unit buffer below; do
            # not construct/fill a fresh tensor on every residual read.
            weight = _ones_weight(query_arg)
        else:
            weight = _validate_rms_weight(rms_weight, query_arg)
        output, _alpha, _rstd = _forward_op(
            packed,
            query_arg,
            weight,
            root_string,
        )
        return output.reshape(sources[0].shape)

    backend.__name__ = "liger_attn_res_model_backend"
    backend.accepts_source_list = True  # type: ignore[attr-defined]
    backend.native_model_source_list = False  # type: ignore[attr-defined]
    # ``CausalAttnResLM`` allocates the parameter-free FP32 unit RMS weight
    # once before compilation when this capability is declared.  Without the
    # marker, the adapter would allocate/fill a new weight inside every timed
    # residual read, making a compiled training-step comparison measure an
    # avoidable adapter artifact.
    backend.accepts_rms_weight = True  # type: ignore[attr-defined]
    backend.supports_cached_block = False  # type: ignore[attr-defined]

    backend.source_hash_metadata = metadata  # type: ignore[attr-defined]
    backend.vendor_root = root_string  # type: ignore[attr-defined]
    return backend


def source_hash_metadata(
    vendor_root: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return provenance and capability limits for a verified adapter."""

    root = resolve_vendor_root(vendor_root, project_root)
    identity = _identity(root)
    return {
        "backend": "liger_native_triton_custom_op",
        "vendor_root": str(root),
        "vendor_revision": identity["revision"],
        "pinned_revision": LIGER_REVISION,
        "pinned_tree": LIGER_TREE,
        "repository": LIGER_REPOSITORY,
        "expected_origin": LIGER_REPOSITORY,
        "vendor_origin": identity["origin"],
        "origin": identity["origin"],
        "tag": LIGER_TAG,
        "version": LIGER_VERSION,
        "source": LIGER_SOURCE,
        "source_url": LIGER_SOURCE_URL,
        "source_sha256": identity["files"][LIGER_SOURCE],
        "expected_source_sha256": LIGER_SOURCE_SHA256,
        "license": "BSD-2-Clause",
        "license_sha256": identity["files"][LIGER_LICENSE],
        "expected_license_sha256": LIGER_LICENSE_SHA256,
        "notice_sha256": identity["files"][LIGER_NOTICE],
        "expected_notice_sha256": LIGER_NOTICE_SHA256,
        "pyproject_sha256": identity["files"][LIGER_PYPROJECT],
        "expected_pyproject_sha256": LIGER_PYPROJECT_SHA256,
        "module_origins": module_origins("liger_kernel"),
        "git_dirty": identity["git_dirty"],
        "rms_weight": "parameter_free_ones_fp32",
        "rms_weight_independent_of_query_dtype": True,
        "output_rms_weight": None,
        "rms_eps": EPS,
        "scale": 1.0,
        "storage": "BF16_or_FP32",
        "equation_dtype": "FP32_native_kernel_accumulation",
        "native_functions": ["attn_res_forward", "attn_res_backward"],
        "graph_boundary": "torch.library.custom_op with registered autograd",
        "accepts_source_list": True,
        "native_model_source_list": False,
        "supports_cached_block": False,
        "model_rms_weight_allocation": "nonpersistent_buffer",
        "model_rms_weight_name": "_backend_rms_weight",
        "model_rms_weight_reuse": "one_buffer_per_model",
        "compiled_model_fill_launches_per_step": 0,
        "compiled_model_fill_launches_avoided_per_step": 1,
        "model_source_argument": "sequence_stacked_inside_adapter",
        "model_forced_source_stack": True,
        "stack_cost": "torch.stack/contiguous inside model adapter boundary",
        "qualification_oracle": "validation.oracle.oracle",
        "qualification_checks": ["output", "all_value_gradients", "query_gradient"],
        "model_qualification": "benchmarks.run._model_qualification",
        "capability_limits": {
            "rank": "R=D only",
            "values": "full-width values; implicit tail keys",
            "sources": f"1<=S<={LIGER_MAX_SOURCES}",
            "width": f"1<=D<={LIGER_MAX_WIDTH}",
            "dtype": "BF16 or FP32 storage",
            "device": "CUDA only",
            "modes": "Full and Block per-read",
            "unsupported": [
                "R<D sliced keys",
                "projected keys",
                "cached Block route",
                "Gluon backend",
                "CPU execution",
            ],
        },
        "complete_training": "requires independent model output/all-parameter gradient gate",
        "adapter_file": str(Path(__file__).resolve()),
        "adapter_sha256": file_sha256(Path(__file__).resolve()),
    }


class Comparator:
    """JSON-friendly optional Liger comparator."""

    def __init__(
        self,
        call: Callable[..., Tensor] | str | None,
        maybe_call: Callable[..., Tensor] | None = None,
        *,
        status: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        vendor_root: str | os.PathLike[str] | None = None,
        vendor_revision: str | None = None,
        vendor_origin: str | None = None,
    ) -> None:
        if isinstance(call, str):
            self.name = call
            self.call = maybe_call
        else:
            if maybe_call is not None:
                raise TypeError("maybe_call is only valid with a comparator name")
            self.name = "liger"
            self.call = call
        self.status = status
        self.reason = reason
        self.metadata = dict(metadata or {})
        self.vendor_root = str(vendor_root) if vendor_root is not None else None
        self.vendor_revision = vendor_revision
        self.vendor_origin = vendor_origin
        self.accepts_source_list = True
        self.native_model_source_list = False

    @property
    def available(self) -> bool:
        return self.status == "available" and self.call is not None

    def applicable(self, values: Any, query: Any) -> tuple[bool, str | None]:
        try:
            _validate_inputs(values, query, require_cuda=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            return False, str(exc)
        return True, None

    def describe(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "kind": "triton",
            "status": self.status,
            "reason": self.reason,
            **self.metadata,
        }
        result.setdefault("pinned_revision", LIGER_REVISION)
        result.setdefault("pinned_tree", LIGER_TREE)
        result.setdefault("repository", LIGER_REPOSITORY)
        result.setdefault("expected_origin", LIGER_REPOSITORY)
        if self.vendor_origin is not None:
            result.setdefault("vendor_origin", self.vendor_origin)
            result.setdefault("origin", self.vendor_origin)
        result.setdefault("version", LIGER_VERSION)
        result.setdefault("source", LIGER_SOURCE)
        result.setdefault("source_url", LIGER_SOURCE_URL)
        result.setdefault("source_sha256", LIGER_SOURCE_SHA256)
        result.setdefault("license", "BSD-2-Clause")
        result.setdefault("license_sha256", LIGER_LICENSE_SHA256)
        result.setdefault("notice", LIGER_NOTICE)
        result.setdefault("notice_sha256", LIGER_NOTICE_SHA256)
        result.setdefault("rms_weight", "parameter_free_ones_fp32")
        result.setdefault("rms_weight_independent_of_query_dtype", True)
        result.setdefault("equation_dtype", "FP32_native_kernel_accumulation")
        result.setdefault("qualification_oracle", "validation.oracle.oracle")
        result.setdefault(
            "qualification_checks",
            ["output", "all_value_gradients", "query_gradient"],
        )
        result.setdefault(
            "capability_limits",
            {
                "rank": "R=D only",
                "sources": f"1<=S<={LIGER_MAX_SOURCES}",
                "width": f"1<=D<={LIGER_MAX_WIDTH}",
                "dtype": "BF16 or FP32 storage",
                "device": "CUDA only",
                "modes": "Full and Block per-read",
                "unsupported": [
                    "R<D sliced keys",
                    "projected keys",
                    "cached Block route",
                    "Gluon backend",
                    "CPU execution",
                ],
            },
        )
        result.setdefault("accepts_source_list", True)
        result.setdefault("native_model_source_list", False)
        if self.vendor_root is not None:
            result.setdefault("vendor_root", self.vendor_root)
        if self.vendor_revision is not None:
            result.setdefault("vendor_revision", self.vendor_revision)
        return result


def _missing(
    reason: str,
    metadata: dict[str, Any] | None = None,
    *,
    vendor_root: str | os.PathLike[str] | None = None,
    vendor_revision: str | None = None,
) -> Comparator:
    return Comparator(
        None,
        status="missing",
        reason=reason,
        metadata=metadata,
        vendor_root=vendor_root,
        vendor_revision=vendor_revision,
    )


def discover_comparator(
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> Comparator:
    """Discover Liger without hiding a setup or import failure."""

    try:
        root = resolve_vendor_root(vendor_root, project_root)
        metadata = source_hash_metadata(root)
    except Exception as exc:
        # Identity failures are intentionally represented as unavailable, so a
        # benchmark report can remain JSON serializable while retaining the
        # reason and never falling back to the reference path.
        configured_root = (
            Path(vendor_root).expanduser().resolve() if vendor_root is not None else None
        )
        return _missing(
            f"pinned Liger discovery failed: {type(exc).__name__}: {exc}",
            vendor_root=configured_root,
        )
    try:
        _load_native(root)
    except Exception as exc:
        return _missing(
            f"pinned Liger runtime import failed: {type(exc).__name__}: {exc}",
            metadata,
            vendor_root=root,
            vendor_revision=metadata.get("vendor_revision"),
        )
    return Comparator(
        make_model_backend(root),
        status="available",
        metadata=metadata,
        vendor_root=root,
        vendor_revision=metadata.get("vendor_revision"),
        vendor_origin=metadata.get("vendor_origin"),
    )


def discover_comparators(
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> dict[str, Comparator]:
    return {"liger": discover_comparator(project_root, vendor_root)}


# Compatibility aliases retained for callers of the earlier optional adapter.
discover_liger = discover_comparator
LIGER_URL = LIGER_SOURCE_URL


def comparator_inputs(
    values: Tensor | Sequence[Tensor], query: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack source inputs for callers using the common comparator interface."""

    sources = _validate_inputs(values, query, require_cuda=True)
    return query, _pack(values, sources), _ones_weight(query)


def invoke_comparator(comparator: Comparator, values: Any, query: Any) -> Tensor:
    if not comparator.available:
        raise RuntimeError(comparator.reason or "Liger comparator is unavailable")
    applicable, reason = comparator.applicable(values, query)
    if not applicable:
        raise ValueError(reason or "Liger comparator is not applicable")
    return comparator.call(values, query)  # type: ignore[misc]


def model_backend(comparator: Comparator | None = None, **kwargs: Any) -> Callable[..., Tensor]:
    if comparator is None:
        return make_model_backend(**kwargs)
    if not comparator.available:
        raise RuntimeError(comparator.reason or "Liger comparator is unavailable")
    assert comparator.call is not None
    return comparator.call


def vendor_metadata(
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return provenance without importing Triton or constructing a model."""

    root = Path(project_root or _PROJECT_ROOT).expanduser().resolve()
    candidates = _candidate_vendor_roots(root, vendor_root)
    path = next(
        (candidate for candidate in candidates if (candidate / LIGER_SOURCE).is_file()),
        None,
    )
    result: dict[str, Any] = {
        "path": str(path) if path else None,
        "git_revision": None,
        "git_dirty": None,
        "expected_revision": LIGER_REVISION,
        "expected_tree": LIGER_TREE,
        "repository": LIGER_REPOSITORY,
        "expected_origin": LIGER_REPOSITORY,
        "vendor_origin": None,
        "origin": None,
        "expected_tag": LIGER_TAG,
        "expected_version": LIGER_VERSION,
        "source": LIGER_SOURCE,
        "expected_source_sha256": LIGER_SOURCE_SHA256,
        "license": "BSD-2-Clause",
        "expected_license_sha256": LIGER_LICENSE_SHA256,
        "notice": LIGER_NOTICE,
        "expected_notice_sha256": LIGER_NOTICE_SHA256,
    }
    if path is None:
        result.update(status="missing", reason="pinned Liger-Kernel checkout was not found")
        return result
    result["git_revision"] = _git_revision(path)
    try:
        result.update(_identity(path), status="verified")
        result["vendor_origin"] = result.get("origin")
    except Exception as exc:
        result.update(
            status="missing",
            reason=f"pinned Liger checkout verification failed: {type(exc).__name__}: {exc}",
        )
    return result


def _git_revision(root: Path | None) -> str | None:
    if root is None:
        return None
    try:
        return git_output(root, "rev-parse", "HEAD") or None
    except CheckoutIdentityError:
        return None


LigerComparator = Comparator


__all__ = [
    "Comparator",
    "EPS",
    "LIGER_COMMIT",
    "LIGER_LICENSE",
    "LIGER_LICENSE_SHA256",
    "LIGER_MAX_SOURCES",
    "LIGER_MAX_WIDTH",
    "LIGER_NOTICE",
    "LIGER_NOTICE_SHA256",
    "LIGER_PYPROJECT_SHA256",
    "LIGER_REVISION",
    "LIGER_REPOSITORY",
    "LIGER_SOURCE",
    "LIGER_SOURCE_SHA256",
    "LIGER_TREE",
    "LIGER_SOURCE_URL",
    "LIGER_TAG",
    "LIGER_URL",
    "LIGER_VERSION",
    "LigerComparator",
    "comparator_inputs",
    "discover_comparator",
    "discover_comparators",
    "discover_liger",
    "find_vendor_root",
    "invoke_comparator",
    "make_model_backend",
    "model_backend",
    "resolve_vendor_root",
    "source_hash_metadata",
    "vendor_metadata",
]

# Keep the historical alias importable for callers that used the old draft.
LIGER_COMMIT = LIGER_REVISION
