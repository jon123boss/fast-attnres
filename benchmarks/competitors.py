"""Optional, directly imported FLA comparator backends.

The project benchmark owns the experiment schedule.  This module only finds
the vendored Flash Linear Attention implementation and exposes callables for
its native Triton checkpoints and Gluon kernels.  In particular, it does not
toggle FLA's dispatch environment variable: the native comparator unwraps the
dispatch decorator and the Gluon comparator calls its backend object directly.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch

from attnres._sources import validate_sources

from .comparator_registry import GLUON_COMPILE_ENVELOPE
from .vendor_identity import (
    CheckoutIdentityError,
    candidate_roots,
    checkout_identity,
    git_output,
    module_origins,
    require_module_origins,
)
from .gluon_compat import install_gluon_barrier_compatibility


EPS = 2**-23

# The FLA source used by the frozen standard baseline.  The source files are
# unchanged between several nearby upstream commits, so both the commit and
# the bytes are recorded.  A source hash alone would not identify the rest of
# the checkout; a commit alone would not detect an edited worktree.
FLA_REVISION = "5e02dd3a7651f5f2797eb8b12bbec401826031e1"
FLA_TREE = "7e4199902fb291c78b3937f223b08ae7bca82bb1"
FLA_PACKAGE_SHA256 = "2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781"
FLA_SOURCE_HASHES = {
    "fla/ops/attnres/fused.py": "0e4683ab291086a9c3919d7352e2a998112973c94f5363e58f76ea7efea114f3",
    "fla/ops/attnres/backends/gluon.py": (
        "f8f163fb7ebb8d035236674aeb668483812fb4e9a29572ed2ae937c626990190"
    ),
}
FLA_SOURCE_URL = (
    f"https://github.com/fla-org/flash-linear-attention/tree/{FLA_REVISION}/fla/ops/attnres"
)
FLA_REPOSITORY = "https://github.com/fla-org/flash-linear-attention.git"
FLA_LICENSE = "LICENSE"
FLA_LICENSE_SHA256 = "41a83c8187efc1e3ccc21909e806a9e52338e69448554d9754706c3d1cd610e7"
FLA_MAX_SOURCES = 129
FLA_MAX_WIDTH = 8192
_FLA_ENVIRONMENT = (
    "ATTNRES_FLA_DIR",
    "FLA_ROOT",
    "FLASH_LINEAR_ATTENTION_ROOT",
    "VENDOR_FLA_ROOT",
)


def _gluon_compile_envelope(
    source_count: int, width: int
) -> tuple[bool, str | None, dict[str, int]]:
    """Check the pinned Gluon checkpoint-1 constexpr work envelope.

    The same geometry rule is sealed in ``comparator_registry``.  This
    adapter-side check runs after source normalization and before allocation
    or native launch, so an oversized Gluon compile cannot become a silent
    fallback to another comparator.
    """

    if type(source_count) is not int or source_count < 1:
        return False, "Gluon compile envelope requires a positive source count", {}
    if type(width) is not int or width < 1:
        return False, "Gluon compile envelope requires a positive width", {}
    padded_width = 1 << (width - 1).bit_length()
    source_width_product = source_count * padded_width
    static_work_score = (
        int(GLUON_COMPILE_ENVELOPE["checkpoint1_static_work_multiplier"])
        * source_width_product
    )
    metrics = {
        "padded_width": padded_width,
        "source_width_product": source_width_product,
        "static_work_score": static_work_score,
    }
    if padded_width > int(GLUON_COMPILE_ENVELOPE["max_padded_width"]):
        return (
            False,
            f"Gluon compile envelope rejects BD={padded_width} for D={width}; "
            f"maximum padded width is {GLUON_COMPILE_ENVELOPE['max_padded_width']}",
            metrics,
        )
    if source_width_product > int(GLUON_COMPILE_ENVELOPE["max_source_width_product"]):
        return (
            False,
            f"Gluon compile envelope rejects S*BD={source_width_product} "
            f"(S={source_count}, BD={padded_width}); maximum is "
            f"{GLUON_COMPILE_ENVELOPE['max_source_width_product']}",
            metrics,
        )
    if static_work_score > int(GLUON_COMPILE_ENVELOPE["max_checkpoint1_static_work"]):
        return (
            False,
            f"Gluon checkpoint-1 static work rejects 33*S*BD={static_work_score}; "
            f"maximum is {GLUON_COMPILE_ENVELOPE['max_checkpoint1_static_work']}",
            metrics,
        )
    return True, None, metrics


class Comparator:
    """A lazily discovered comparator with JSON friendly metadata."""

    def __init__(
        self,
        name: str,
        call: Callable[..., Any] | None,
        *,
        status: str,
        reason: str | None = None,
        checkpoint_level: int | None = None,
        kind: str = "optional",
        vendor_root: str | None = None,
        vendor_revision: str | None = None,
        vendor_origin: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.call = call
        self.status = status
        self.reason = reason
        self.checkpoint_level = checkpoint_level
        self.kind = kind
        self.vendor_root = vendor_root
        self.vendor_revision = vendor_revision
        self.vendor_origin = vendor_origin
        self.metadata = dict(metadata or {})

    @property
    def available(self) -> bool:
        return self.status == "available" and self.call is not None

    def applicable(
        self, values: Any, query: Any, *, keys: Any = None
    ) -> tuple[bool, str | None]:
        if keys is not None:
            return False, "FLA comparator only supports implicit keys (keys=None)"
        try:
            sources = _standard_sources(
                values,
                query,
                require_cuda=self.kind in {"triton", "gluon"},
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return False, str(exc)
        if self.kind == "gluon":
            envelope_ok, envelope_reason, _metrics = _gluon_compile_envelope(
                len(sources), int(sources[0].shape[-1])
            )
            if not envelope_ok:
                return False, envelope_reason
        return True, None

    def describe(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "matched_rank_only": True,
            "autotuning": "native FLA autotune configs",
            "accepts_source_list": True,
            "model_source_argument": "sequence_of_contiguous_source_tensors",
            "model_forced_source_stack": False,
            "pinned_revision": FLA_REVISION,
            "pinned_tree": FLA_TREE,
            "repository": FLA_REPOSITORY,
            "expected_origin": FLA_REPOSITORY,
            "source_hashes": dict(FLA_SOURCE_HASHES),
            "license": FLA_LICENSE,
            "license_sha256": FLA_LICENSE_SHA256,
            "package_sha256": FLA_PACKAGE_SHA256,
            "module_origins": module_origins("fla"),
            "rms_weight": "parameter_free_ones",
            "output_rms_weight": None,
            "rms_eps": EPS,
            "scale": 1.0,
            "equation_dtype": "FP32_native_kernel_accumulation",
            "qualification_oracle": "validation.oracle.oracle",
            "qualification_checks": ["output", "all_value_gradients", "query_gradient"],
            "model_qualification": "benchmarks.run._model_qualification",
            "capability_limits": {
                "rank": "R=D only",
                "values": "full-width values; implicit tail keys",
                "sources": f"1<=S<={FLA_MAX_SOURCES}",
                "width": f"1<=D<={FLA_MAX_WIDTH}",
                "dtype": "BF16 or FP32 storage",
                "device": "CUDA only",
                "block": "per-read Full/Block native read; no projected keys",
            },
        }
        if self.kind == "gluon":
            result["compile_envelope"] = dict(GLUON_COMPILE_ENVELOPE)
        if self.checkpoint_level is not None:
            result["checkpoint_level"] = self.checkpoint_level
        if self.vendor_root is not None:
            result["vendor_root"] = self.vendor_root
        if self.vendor_revision is not None:
            result["vendor_revision"] = self.vendor_revision
        if self.vendor_origin is not None:
            result["vendor_origin"] = self.vendor_origin
            result["origin"] = self.vendor_origin
        if self.reason:
            result["reason"] = self.reason
        result.update(self.metadata)
        return result


def _candidate_vendor_roots(
    project_root: Path, configured: str | os.PathLike[str] | None
) -> list[Path]:
    return list(
        candidate_roots(
            project_root,
            configured,
            environment=_FLA_ENVIRONMENT,
            defaults=(
                project_root / "vendor" / "fla",
                project_root / "vendor" / "flash-linear-attention",
                project_root.parent / "vendor" / "fla",
                project_root.parent / "vendor" / "flash-linear-attention",
                project_root.parent.parent / "vendor" / "fla",
                project_root.parent.parent / "vendor" / "flash-linear-attention",
            ),
        )
    )


def find_vendor_root(
    project_root: str | os.PathLike[str] | None = None,
    configured: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Find a vendored FLA checkout without importing Triton."""
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    for candidate in _candidate_vendor_roots(root, configured):
        if (candidate / "fla" / "ops" / "attnres").is_dir():
            return candidate
    return None


def _git_revision(root: Path | None) -> str | None:
    if root is None:
        return None
    try:
        return git_output(root, "rev-parse", "HEAD") or None
    except CheckoutIdentityError:
        return None


def _fla_identity(root: Path) -> dict[str, Any]:
    """Verify the pinned checkout before importing Triton or Gluon."""

    return checkout_identity(
        root,
        expected_revision=FLA_REVISION,
        expected_tree=FLA_TREE,
        files={**FLA_SOURCE_HASHES, FLA_LICENSE: FLA_LICENSE_SHA256},
        package_dir="fla",
        package_sha256=FLA_PACKAGE_SHA256,
        expected_origin=FLA_REPOSITORY,
    )


def _missing(
    name: str,
    reason: str,
    *,
    kind: str,
    vendor_root: Path | None,
    vendor_revision: str | None = None,
    vendor_origin: str | None = None,
) -> Comparator:
    return Comparator(
        name,
        None,
        status="missing",
        reason=reason,
        kind=kind,
        vendor_root=str(vendor_root) if vendor_root else None,
        vendor_revision=vendor_revision,
        vendor_origin=vendor_origin,
    )


def _unwrap_dispatch(function: Callable[..., Any]) -> Callable[..., Any]:
    """Reach FLA's direct implementation beneath functools/compile wrappers."""
    current = function
    visited: set[int] = set()
    while hasattr(current, "__wrapped__") and id(current) not in visited:
        visited.add(id(current))
        current = current.__wrapped__
    return current


def _all_loaded_origins_ok(vendor: Path) -> bool:
    """Reject a previously imported FLA package from another checkout."""

    base = vendor.resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "fla" and not name.startswith("fla."):
            continue
        origins = []
        file_name = getattr(module, "__file__", None)
        if file_name:
            origins.append(file_name)
        origins.extend(getattr(module, "__path__", ()) or ())
        if not origins:
            return False
        try:
            for origin in origins:
                Path(origin).resolve().relative_to(base)
        except (OSError, ValueError, TypeError):
            return False
    return True


def _native_module(vendor: Path, module_name: str, relative_path: str) -> Any:
    """Import exactly one module from the verified checkout.

    Python caches modules by name.  Checking ``__file__`` after import keeps a
    process that has already loaded another FLA checkout from silently using
    that checkout for this comparator.
    """

    vendor_string = str(vendor)
    if not _all_loaded_origins_ok(vendor):
        raise ImportError("loaded FLA modules originate outside the pinned source")
    # Keep the shared check as the source of truth for package and namespace
    # paths as well as leaf ``__file__`` values.
    require_module_origins("fla", vendor)
    if vendor_string not in sys.path:
        sys.path.insert(0, vendor_string)
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    module_file = Path(getattr(module, "__file__", "")).resolve()
    expected = (vendor / relative_path).resolve()
    if module_file != expected:
        raise ImportError(
            f"loaded FLA module {module_file} does not match pinned source {expected}"
        )
    if not _all_loaded_origins_ok(vendor):
        raise ImportError("loaded FLA modules originate outside the pinned source")
    require_module_origins("fla", vendor)
    return module


def discover_comparators(
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> dict[str, Comparator]:
    """Discover FLA Triton checkpoint 0/1 and Gluon comparators.

    Missing optional dependencies are represented by comparator objects with a
    ``missing`` status, so a result cannot accidentally look like a complete
    three way comparison.
    """
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    vendor = find_vendor_root(root, vendor_root)
    names = {
        "fla_triton_checkpoint0": ("triton", 0),
        "fla_triton_checkpoint1": ("triton", 1),
        "fla_gluon": ("gluon", None),
    }
    if vendor is None:
        return {
            name: _missing(
                name,
                "vendored flash-linear-attention checkout was not found",
                kind=kind,
                vendor_root=None,
            )
            for name, (kind, _checkpoint) in names.items()
        }

    revision = _git_revision(vendor)
    try:
        identity = _fla_identity(vendor)
    except Exception as exc:
        reason = f"pinned FLA checkout verification failed: {type(exc).__name__}: {exc}"
        return {
            name: _missing(
                name,
                reason,
                kind=kind,
                vendor_root=vendor,
                vendor_revision=revision,
            )
            for name, (kind, _checkpoint) in names.items()
        }

    # Install the dependency compatibility bridge before importing any FLA
    # package.  Importing a leaf first also executes its parent packages, and a
    # parent is allowed to import the Gluon backend eagerly.
    barrier_compatibility = None
    barrier_compatibility_error = None
    try:
        barrier_compatibility = install_gluon_barrier_compatibility()
    except Exception as exc:
        barrier_compatibility_error = exc

    result: dict[str, Comparator] = {}
    native_function = None
    try:
        module = _native_module(
            vendor,
            "fla.ops.attnres.fused",
            "fla/ops/attnres/fused.py",
        )
        native_function = _unwrap_dispatch(getattr(module, "fused_attnres"))
    except Exception as exc:  # optional dependency or incompatible Triton build
        reason = f"native FLA Triton import failed: {type(exc).__name__}: {exc}"
        for name in ("fla_triton_checkpoint0", "fla_triton_checkpoint1"):
            result[name] = _missing(name, reason, kind="triton", vendor_root=vendor)

    if native_function is not None:
        for name, checkpoint in (("fla_triton_checkpoint0", 0), ("fla_triton_checkpoint1", 1)):
            def native_call(
                *,
                query: Any,
                residuals: Any,
                rms_weight: Any,
                checkpoint_level: int = checkpoint,
                function: Callable[..., Any] = native_function,
            ) -> Any:
                return function(
                    query=query,
                    residuals=residuals,
                    rms_weight=rms_weight,
                    output_rms_weight=None,
                    rms_eps=EPS,
                    scale=1.0,
                    return_weights=False,
                    checkpoint_level=checkpoint_level,
                )

            result[name] = Comparator(
                name,
                native_call,
                status="available",
                checkpoint_level=checkpoint,
                kind="triton",
                vendor_root=vendor,
                vendor_revision=identity["revision"],
                vendor_origin=identity["origin"],
            )

    try:
        if barrier_compatibility_error is not None:
            raise ImportError(
                "Triton Gluon barrier compatibility failed: "
                f"{type(barrier_compatibility_error).__name__}: "
                f"{barrier_compatibility_error}"
            )
        assert barrier_compatibility is not None
        gluon_module = _native_module(
            vendor,
            "fla.ops.attnres.backends.gluon",
            "fla/ops/attnres/backends/gluon.py",
        )
        backend = gluon_module.AttnResGluonBackend()

        def gluon_call(*, query: Any, residuals: Any, rms_weight: Any) -> Any:
            # Calling the backend object reaches the decorated Gluon kernels and
            # leaves its genuine FLA autotuner responsible for configuration.
            return backend.fused_attnres(
                query=query,
                residuals=residuals,
                rms_weight=rms_weight,
                output_rms_weight=None,
                rms_eps=EPS,
                scale=1.0,
                return_weights=False,
                checkpoint_level=1,
            )

        result["fla_gluon"] = Comparator(
            "fla_gluon",
            gluon_call,
            status="available",
            kind="gluon",
            vendor_root=vendor,
            vendor_revision=identity["revision"],
            vendor_origin=identity["origin"],
            metadata={
                "compatibility_shims": {
                    "triton_gluon_barrier": barrier_compatibility,
                },
            },
        )
    except Exception as exc:  # Gluon is optional in Triton builds
        result["fla_gluon"] = _missing(
            "fla_gluon",
            f"FLA Gluon import failed: {type(exc).__name__}: {exc}",
            kind="gluon",
            vendor_root=vendor,
            vendor_revision=revision,
        )

    # Keep all expected names present even if an unusual import path failed
    # between the two discovery branches.
    for name, (kind, checkpoint) in names.items():
        result.setdefault(
            name,
            _missing(name, "comparator was not registered", kind=kind, vendor_root=vendor),
        )
    for comparator in result.values():
        comparator.vendor_revision = identity["revision"]
    return result


def _standard_sources(
    values: Any,
    query: Any,
    *,
    require_cuda: bool = False,
) -> tuple[Any, ...]:
    """Validate and normalize the packed or ordered standard-R source inputs."""
    sources = validate_sources(values, query, EPS, 1.0)
    if not sources:
        raise ValueError("FLA comparator requires at least one source")
    if len(sources) > FLA_MAX_SOURCES:
        raise ValueError(
            f"FLA comparator supports at most {FLA_MAX_SOURCES} sources"
        )
    if query.shape[-1] != sources[0].shape[-1]:
        raise ValueError("FLA comparator is only matched for standard R=D")
    width = int(sources[0].shape[-1])
    if width > FLA_MAX_WIDTH:
        raise ValueError(f"FLA comparator supports at most D={FLA_MAX_WIDTH}")
    if sources[0].dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("FLA comparator values must use BF16 or FP32 storage")
    if query.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("FLA comparator query must use BF16 or FP32 storage")
    if query.device != sources[0].device:
        raise ValueError("FLA comparator values and query must share one device")
    if require_cuda and sources[0].device.type != "cuda":
        raise RuntimeError("FLA native comparator requires CUDA values and query")
    return sources


def _prepared_comparator_inputs(
    values: Any,
    query: Any,
    *,
    require_cuda: bool = False,
    check_gluon_envelope: bool = False,
) -> tuple[Any, list[Any], Any, Any]:
    sources = _standard_sources(values, query, require_cuda=require_cuda)
    d = sources[0].shape[-1]
    if check_gluon_envelope:
        envelope_ok, envelope_reason, _metrics = _gluon_compile_envelope(
            len(sources), int(d)
        )
        if not envelope_ok:
            raise ValueError(envelope_reason or "Gluon compile envelope rejected the case")
    # FLA consumes one flattened, contiguous residual per source.  Keep this
    # conversion per source so list/tuple inputs never acquire a full stack.
    residuals = [source.reshape(-1, d).contiguous() for source in sources]
    rms_weight = query.new_ones((d,))
    return query, residuals, rms_weight, sources[0]


def comparator_inputs(values: Any, query: Any) -> tuple[Any, Any, Any]:
    """Convert project packed or source-list inputs into FLA's list API."""
    query_arg, residuals, rms_weight, _first = _prepared_comparator_inputs(values, query)
    return query_arg, residuals, rms_weight


def invoke_comparator(comparator: Comparator, values: Any, query: Any) -> Any:
    """Invoke a standard-rank comparator after checking applicability."""
    if not comparator.available:
        raise RuntimeError(comparator.reason or f"{comparator.name} is {comparator.status}")
    applicable, reason = comparator.applicable(values, query)
    if not applicable:
        raise ValueError(reason or "comparator is not applicable")
    query_arg, residuals, rms_weight, _first = _prepared_comparator_inputs(
        values,
        query,
        require_cuda=comparator.kind in {"triton", "gluon"},
        check_gluon_envelope=comparator.kind == "gluon",
    )
    return comparator.call(query=query_arg, residuals=residuals, rms_weight=rms_weight)


def model_backend(comparator: Comparator) -> Callable[..., Any]:
    """Adapt a standard-rank FLA comparator to ``make_model``'s operator API."""
    if not comparator.available:
        raise RuntimeError(comparator.reason or f"{comparator.name} is {comparator.status}")

    def backend(values: Any, query: Any, *, eps: float = EPS, scale: float = 1.0) -> Any:
        if float(eps) != EPS or float(scale) != 1.0:
            raise ValueError("FLA model arms use the frozen eps and scale")
        query_arg, residuals, rms_weight, first = _prepared_comparator_inputs(
            values,
            query,
            require_cuda=comparator.kind in {"triton", "gluon"},
            check_gluon_envelope=comparator.kind == "gluon",
        )
        result = comparator.call(query=query_arg, residuals=residuals, rms_weight=rms_weight)
        # FLA consumes the source axis as a list of residuals and returns one
        # output per token.  The project model API is [N,D], so restoring the
        # source axis here would both change semantics and over-count elements.
        return result.reshape(first.shape)

    backend.__name__ = f"{comparator.name}_model_backend"
    backend.accepts_source_list = True  # type: ignore[attr-defined]
    backend.native_model_source_list = True  # type: ignore[attr-defined]
    backend.source_hash_metadata = {  # type: ignore[attr-defined]
        "backend": "fla_native_comparator",
        "pinned_revision": FLA_REVISION,
        "pinned_tree": FLA_TREE,
        "repository": FLA_REPOSITORY,
        "expected_origin": FLA_REPOSITORY,
        "vendor_origin": comparator.vendor_origin,
        "vendor_source_url": FLA_SOURCE_URL,
        "vendor_package_sha256": FLA_PACKAGE_SHA256,
        "source_hashes": dict(FLA_SOURCE_HASHES),
        "license": FLA_LICENSE,
        "license_sha256": FLA_LICENSE_SHA256,
        "rms_weight": "parameter_free_ones",
        "output_rms_weight": None,
        "rms_eps": EPS,
        "scale": 1.0,
        "equation_dtype": "FP32_native_kernel_accumulation",
        "qualification_oracle": "validation.oracle.oracle",
        "qualification_checks": ["output", "all_value_gradients", "query_gradient"],
        "model_qualification": "benchmarks.run._model_qualification",
        "accepts_source_list": True,
        "native_model_source_list": True,
        "model_source_argument": "sequence_of_contiguous_source_tensors",
        "model_forced_source_stack": False,
        "capability_limits": {
            "rank": "R=D only",
            "sources": f"1<=S<={FLA_MAX_SOURCES}",
            "width": f"1<=D<={FLA_MAX_WIDTH}",
            "dtype": "BF16 or FP32 storage",
            "device": "CUDA only",
        },
    }
    if comparator.kind == "gluon":
        backend.source_hash_metadata["compile_envelope"] = dict(GLUON_COMPILE_ENVELOPE)
    return backend


def vendor_metadata(
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return discovery metadata without importing optional comparator code."""
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    vendor = find_vendor_root(root, vendor_root)
    result: dict[str, Any] = {
        "path": str(vendor) if vendor else None,
        "git_revision": _git_revision(vendor) if vendor else None,
        "dispatch_environment": os.environ.get("FLA_ATTNRES_GLUON"),
        "expected_revision": FLA_REVISION,
        "expected_tree": FLA_TREE,
        "expected_origin": FLA_REPOSITORY,
        "repository": FLA_REPOSITORY,
        "vendor_origin": None,
        "origin": None,
        "license": FLA_LICENSE,
        "expected_license_sha256": FLA_LICENSE_SHA256,
        "expected_package_sha256": FLA_PACKAGE_SHA256,
        "gluon_compile_envelope": dict(GLUON_COMPILE_ENVELOPE),
        "source_hashes": dict(FLA_SOURCE_HASHES),
        "qualification_oracle": "validation.oracle.oracle",
        "qualification_checks": ["output", "all_value_gradients", "query_gradient"],
    }
    if vendor is None:
        result["status"] = "missing"
        result["reason"] = "vendored flash-linear-attention checkout was not found"
        return result
    try:
        identity = _fla_identity(vendor)
    except Exception as exc:
        result["status"] = "missing"
        result["reason"] = f"pinned FLA checkout verification failed: {type(exc).__name__}: {exc}"
    else:
        result["status"] = "verified"
        result.update(identity)
        result["vendor_origin"] = identity["origin"]
    return result


__all__ = [
    "Comparator",
    "EPS",
    "FLA_LICENSE_SHA256",
    "FLA_MAX_SOURCES",
    "FLA_MAX_WIDTH",
    "FLA_PACKAGE_SHA256",
    "FLA_REPOSITORY",
    "FLA_REVISION",
    "FLA_SOURCE_URL",
    "FLA_SOURCE_HASHES",
    "FLA_TREE",
    "GLUON_COMPILE_ENVELOPE",
    "comparator_inputs",
    "discover_comparators",
    "find_vendor_root",
    "invoke_comparator",
    "model_backend",
    "vendor_metadata",
]
