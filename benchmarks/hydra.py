"""Pinned Manish/Hydra-2P comparator for standard AttnRes reads.

Hydra-2P exposes an external Block-panel AttnRes operator whose public ABI is
``attnres(queries, blocks, partials, has_partial, plan)``.  The project
benchmark maps its source set and one query to that external Block-panel
interface.  This adapter maps that interface to one Hydra query row, a stacked source
tensor, and a finite all-zero dummy partial disabled by ``has_partial=False``.

Only the native Triton plan is used.  In particular, this module never calls
Hydra's ``auto`` or ``torch`` dispatch and never substitutes its portable
implementation when native discovery or execution is unavailable.  The CPU
mock below is an explicit test helper and is deliberately not returned by
``discover_comparators``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import math
import os
from numbers import Real
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from attnres._sources import validate_sources

from .vendor_identity import CheckoutIdentityError, normalize_remote_origin, verify_remote_origin


EPS = 2**-23
NAME = "hydra_2p"
PINNED_REVISION = "ea1f63eda8e31b0f10456b3b49cacd8fb66091dc"
PINNED_TREE = "b6ae55c737f9b17c9c7ea064b17bd0210510496a"
VENDOR_REVISION = PINNED_REVISION
VENDOR_TREE = PINNED_TREE
REPOSITORY = "https://github.com/manishklach/attnres-kernel-lab"
HYDRA_REPOSITORY = REPOSITORY
LICENSE = "LICENSE"
LICENSE_SHA256 = "6d7cc4b730aafd6e596d41c5cb2250c30a9ef8bcffd350fe5e3fe566936a6ebd"
NATIVE_MAX_WIDTH = 256
MAX_SOURCES = 129
MAX_WIDTH = 8192
NUM_WARPS = 4
TIMING_PREDICATE = "1 <= D <= 256"
TIMING_PREDICATE_NAME = "benchmarks.hydra._timing_eligible"
TIMING_EXCLUSION_REASON = (
    "Hydra native timing requires 1 <= D <= 256; larger widths are qualification-only"
)


# Hydra's package imports these modules as part of its public API.  Hashing
# the public package, its equation reference, and its native implementation
# makes the provenance record useful even before a CUDA call imports Triton.
_VENDOR_FILES = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "src/attnres_kernel/__init__.py",
    "src/attnres_kernel/api.py",
    "src/attnres_kernel/benchmark.py",
    "src/attnres_kernel/cadr.py",
    "src/attnres_kernel/cadr_benchmark.py",
    "src/attnres_kernel/cadr_triton.py",
    "src/attnres_kernel/hybrid.py",
    "src/attnres_kernel/kda.py",
    "src/attnres_kernel/kda_benchmark.py",
    "src/attnres_kernel/optimizer.py",
    "src/attnres_kernel/reference.py",
    "src/attnres_kernel/torch_impl.py",
    "src/attnres_kernel/triton_impl.py",
)
_VENDOR_SHA256 = {
    "LICENSE": "6d7cc4b730aafd6e596d41c5cb2250c30a9ef8bcffd350fe5e3fe566936a6ebd",
    "README.md": "c5dcd230dd640023c92be6dfa6ca16857e78c6308d506595f83c47c7845679f7",
    "pyproject.toml": "c39e8bd14ec222366a349dce7909da621ad858efeb3fc1b53ee67d89ea79881b",
    "src/attnres_kernel/__init__.py": "757f6c654267f87565efc95ada1d6e0ff85eed6443a0ed9239c5d7f8eb59d648",
    "src/attnres_kernel/api.py": "5067773602c08aaf7c44837dd15f9d95e839c788199a0385407ae9bb80d81b44",
    "src/attnres_kernel/benchmark.py": "bb44c433399de5247a422d4e175982b5d09646df139a790c77e6b9a6d9c5fa56",
    "src/attnres_kernel/cadr.py": "44c657d712b53ebe2629ae889a4b2a993e1d236e8b8db489103e5eced20230ab",
    "src/attnres_kernel/cadr_benchmark.py": "b99c3088fa9f270d02e8effbb5f50cd25658377f34fd62dfa1f15f3bb3dd898e",
    "src/attnres_kernel/cadr_triton.py": "ca55a0690702bfff7e9a14eb9a47514af795108929a58ff84692f74d2242daa6",
    "src/attnres_kernel/hybrid.py": "8180f81dced1da40f3e7f1c38628014c1a71afdaf0923b8d4c95dab64ee46079",
    "src/attnres_kernel/kda.py": "53a4ecd395c78690d53272333da5b01809b4e53c108617a7f9159520139c8228",
    "src/attnres_kernel/kda_benchmark.py": "a49835abd8d06833a06d3a117865c52585b300d5c95a5c5d9f713e95aff0b313",
    "src/attnres_kernel/optimizer.py": "2f4a8be2a53953c2406c0d83daf985200a6fd7851dc0e8d439b2108829036ca2",
    "src/attnres_kernel/reference.py": "33fb4f3e8501a5983fc0c64b71ba8a256f478ae51dc3e52ae383aa3be7868d2b",
    "src/attnres_kernel/torch_impl.py": "b48ecb99977d9ccf9a751cb9e00a24afb3dcd0eb8b4e80eb644fd8883c810196",
    "src/attnres_kernel/triton_impl.py": "520dbf69280f0b200e6b6b2f5ef9bf30863b20778795f7e1b1c23f99176b7e1c",
}
_REQUIRED_MODULES = (
    "attnres_kernel",
    "attnres_kernel.api",
    "attnres_kernel.torch_impl",
    "attnres_kernel.triton_impl",
)
_ORIGIN = REPOSITORY.removesuffix(".git").rstrip("/").lower()
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HydraProvenanceError(ImportError):
    """The explicitly pinned Hydra checkout failed an identity check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_symlink_component(path: Path) -> bool:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return any(component.is_symlink() for component in (path, *path.parents))


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _top_level(root: Path) -> Path | None:
    value = _git(root, "rev-parse", "--show-toplevel")
    if not value:
        return None
    try:
        return Path(value).resolve()
    except (OSError, RuntimeError):
        return None


def _revision(root: Path) -> str | None:
    return _git(root, "rev-parse", "HEAD")


def _tree(root: Path) -> str | None:
    return _git(root, "rev-parse", "HEAD^{tree}")


def _dirty(root: Path) -> bool | None:
    value = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if value is None:
        return None
    return bool(value)


def _origin(root: Path) -> str | None:
    return verify_remote_origin(root, REPOSITORY)


def _normal_origin(value: str | None) -> str | None:
    return normalize_remote_origin(value)


def _file_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    base = root.resolve()
    for relative in _VENDOR_FILES:
        path = root / relative
        if _contains_symlink_component(path):
            continue
        try:
            path.resolve().relative_to(base)
        except (OSError, RuntimeError, ValueError):
            continue
        if path.is_file():
            result[relative] = _sha256(path)
    return result


def _integrity(root: Path) -> tuple[str | None, dict[str, Any]]:
    """Return an identity error and the observed checkout metadata."""

    root = Path(root).expanduser().resolve()
    top = _top_level(root)
    revision = _revision(root)
    tree = _tree(root)
    dirty = _dirty(root)
    origin_error: CheckoutIdentityError | None = None
    try:
        origin = _origin(root)
    except CheckoutIdentityError as exc:
        origin = None
        origin_error = exc
    hashes = _file_hashes(root)
    observed: dict[str, Any] = {
        "vendor_root": str(root),
        "vendor_top_level": str(top) if top is not None else None,
        "vendor_revision": revision,
        "vendor_tree": tree,
        "vendor_dirty": dirty,
        "vendor_clean": dirty is False,
        "vendor_origin": origin,
        "vendor_file_sha256": hashes,
        "expected_vendor_file_sha256": dict(_VENDOR_SHA256),
    }
    if top != root:
        return "could not verify pinned Hydra checkout top-level", observed
    if revision != PINNED_REVISION:
        return f"expected pinned Hydra revision {PINNED_REVISION}, got {revision!r}", observed
    if tree != PINNED_TREE:
        return f"expected pinned Hydra tree {PINNED_TREE}, got {tree!r}", observed
    if dirty is None:
        return "could not verify pinned Hydra checkout cleanliness", observed
    if dirty:
        return "pinned Hydra checkout is dirty", observed
    if origin_error is not None:
        return f"pinned Hydra origin verification failed: {origin_error}", observed
    if _normal_origin(origin) != _ORIGIN:
        return f"expected pinned Hydra origin {_ORIGIN!r}, got {origin!r}", observed
    if hashes != _VENDOR_SHA256:
        return "pinned Hydra source/package/license hash mismatch", observed
    return None, observed


def _inside(path: str | os.PathLike[str], root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
    except (OSError, ValueError, TypeError):
        return False
    return True


def _module_origin(module: Any) -> list[str]:
    origins: list[str] = []
    file_name = getattr(module, "__file__", None)
    if file_name:
        origins.append(str(Path(file_name).resolve()))
    for item in getattr(module, "__path__", ()) or ():
        origins.append(str(Path(item).resolve()))
    return origins


def _module_origins(source_root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name in _REQUIRED_MODULES:
        module = sys.modules.get(name)
        if module is not None:
            result[name] = _module_origin(module)
    return result


def _origins_ok(source_root: Path) -> bool:
    for name in _REQUIRED_MODULES:
        module = sys.modules.get(name)
        if module is None or not _module_origin(module):
            return False
        if any(not _inside(origin, source_root) for origin in _module_origin(module)):
            return False
    return True


def _all_loaded_origins_ok(source_root: Path) -> bool:
    """Reject any already-loaded package submodule from another checkout."""

    source_root = source_root.resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "attnres_kernel" and not name.startswith("attnres_kernel."):
            continue
        origins = _module_origin(module)
        if not origins or any(not _inside(origin, source_root) for origin in origins):
            return False
    return True


def _candidate_roots(
    project_root: str | os.PathLike[str] | None,
    configured: str | os.PathLike[str] | None,
) -> list[Path]:
    anchor = Path(project_root or _PROJECT_ROOT).expanduser().resolve()
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(Path(configured).expanduser())
    if configured is None:
        for key in (
            "HYDRA_ROOT",
            "ATTNRES_HYDRA_ROOT",
            "ATTNRES_KERNEL_LAB_ROOT",
            "MANISH_ATTNRES_ROOT",
            "VENDOR_ATTNRES_KERNEL_LAB_ROOT",
        ):
            value = os.environ.get(key)
            if value:
                candidates.append(Path(value).expanduser())
                break
        if not candidates:
            candidates.extend(
                (
                    anchor / "vendor" / "attnres-kernel-lab",
                    anchor.parent / "vendor" / "attnres-kernel-lab",
                    anchor.parent.parent / "vendor" / "attnres-kernel-lab",
                    anchor.parent.parent.parent / "vendor" / "attnres-kernel-lab",
                )
            )
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if _contains_symlink_component(candidate):
            continue
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def find_vendor_root(
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Find a candidate checkout without importing optional Triton code."""

    candidates = _candidate_roots(project_root, vendor_root)
    # An explicit path is authoritative.  Falling through to an ambient
    # checkout would make a requested revision silently change identity.
    if vendor_root is not None:
        candidates = candidates[:1]
    for candidate in candidates:
        if (candidate / "src" / "attnres_kernel").is_dir():
            return candidate
    return None


def resolve_vendor_root(
    vendor_root: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve and verify the exact pinned checkout."""

    root = find_vendor_root(project_root, vendor_root)
    if root is None:
        raise HydraProvenanceError("pinned attnres-kernel-lab checkout was not found")
    reason, _observed = _integrity(root)
    if reason:
        raise HydraProvenanceError(reason)
    return root


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("hidden dimension must be positive")
    return 1 << (int(value) - 1).bit_length()


next_power_of_two = _next_power_of_two


def _fixed_scalar(value: Any, name: str, expected: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value != expected:
        raise ValueError(f"Hydra uses {name}={expected!r}")


def _source_tuple(values: Any) -> tuple[Tensor, ...]:
    if isinstance(values, Tensor):
        if values.ndim < 2:
            raise ValueError("values must have shape [S,...,D]")
        return tuple(values.unbind(0))
    if not isinstance(values, (list, tuple)):
        raise TypeError("values must be a tensor or a list/tuple of tensors")
    return tuple(values)


def _validate_sources(values: Any, query: Any) -> tuple[tuple[Tensor, ...], int]:
    # Keep this adapter on the same source/container contract as the public
    # AttnRes implementation.  The vendor ABI is standard-only, so the final
    # check intentionally rejects the otherwise-valid sliced (R<D) case.
    sources = validate_sources(values, query, EPS, 1.0)
    if not sources:
        raise ValueError("Hydra comparator requires at least one source")
    if len(sources) > MAX_SOURCES:
        raise ValueError(f"Hydra supports at most S={MAX_SOURCES} sources")
    width = int(sources[0].shape[-1])
    if width > MAX_WIDTH:
        raise ValueError(f"Hydra supports at most D={MAX_WIDTH}")
    if int(query.numel()) != width:
        raise ValueError(
            "Hydra comparator only supports standard R=D; "
            "sliced/projected keys are unsupported"
        )
    if sources[0].dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Hydra values must use BF16 or FP32 storage")
    if query.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Hydra query must use BF16 or FP32 storage")
    if query.device != sources[0].device:
        raise ValueError("Hydra values and query must share one device")
    return sources, width


def _stack_inputs(values: Any, query: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Stack and make the exact timed boundary inputs."""

    sources, width = _validate_sources(values, query)
    # The stack is intentional for Hydra's [N,...,D] block ABI.  It remains
    # differentiable, so source-list callers receive one gradient per source.
    blocks = torch.stack(sources, dim=0).contiguous()
    # The public project contract permits a source shaped only ``[D]``.  The
    # vendor ABI requires one token/batch axis between source and width, so
    # add a singleton axis and restore the logical shape at the boundary.
    if blocks.ndim == 2:
        blocks = blocks.unsqueeze(1).contiguous()
    query_row = query.contiguous().reshape(1, width)
    # A real, finite partial is required even though it is masked out.  This
    # avoids undefined 0*NaN behavior in any backend implementation.
    partial = torch.zeros_like(blocks[:1]).contiguous()
    has_partial = torch.zeros((1,), dtype=torch.bool, device=blocks.device)
    return blocks, query_row, partial, has_partial


def _timing_eligible(
    values: Any,
    query: Any | None = None,
    *,
    keys: Any = None,
) -> bool:
    """Return the bounded native timing predicate without raising.

    The predicate is intentionally independent of CUDA availability so a
    caller can label a case before allocating device inputs.  Native
    applicability still performs the explicit device check below.
    """

    if keys is not None:
        return False
    try:
        if query is None:
            if isinstance(values, bool) or not isinstance(values, int):
                return False
            width = int(values)
        else:
            _sources, width = _validate_sources(values, query)
    except (TypeError, ValueError, RuntimeError):
        return False
    return 1 <= width <= NATIVE_MAX_WIDTH


def _timing_applicability(
    values: Any,
    query: Any,
    *,
    keys: Any = None,
) -> tuple[bool, str | None]:
    """Validate standard semantics, then apply the native width boundary."""

    if keys is not None:
        return False, "Hydra comparator only supports implicit keys (keys=None)"
    try:
        _sources, width = _validate_sources(values, query)
    except (TypeError, ValueError, RuntimeError) as exc:
        return False, str(exc)
    if not 1 <= width <= NATIVE_MAX_WIDTH:
        return False, TIMING_EXCLUSION_REASON
    return True, None


timing_eligible = _timing_eligible


def _plan(plan_type: Callable[..., Any], width: int) -> Any:
    block_d = _next_power_of_two(width)
    plan = plan_type(
        backend="triton",
        block_d=block_d,
        num_warps=NUM_WARPS,
        eps=EPS,
    )
    if getattr(plan, "backend", None) != "triton":
        raise RuntimeError("Hydra native comparator requires an explicit triton plan")
    if int(getattr(plan, "block_d", -1)) != block_d:
        raise RuntimeError("Hydra plan factory changed the required block_d")
    if int(getattr(plan, "num_warps", -1)) != NUM_WARPS:
        raise RuntimeError("Hydra plan factory changed the fixed num_warps")
    if float(getattr(plan, "eps", float("nan"))) != EPS:
        raise RuntimeError("Hydra plan factory changed the frozen eps")
    return plan


class HydraBackend:
    """Callable standard-R Hydra backend with adapter-owned source stacking."""

    accepts_source_list = True
    native_model_source_list = False
    source_list_copy = True
    source_stack_owned_by_adapter = True
    supports_full = True
    supports_per_read_block = False
    timing_predicate = TIMING_PREDICATE

    def __init__(
        self,
        implementation: Callable[..., Any],
        plan_type: Callable[..., Any],
        *,
        vendor_root: Path | None = None,
        provenance: Mapping[str, Any] | None = None,
        native: bool = True,
    ) -> None:
        if not callable(implementation) or not callable(plan_type):
            raise TypeError("Hydra implementation and plan type must be callable")
        self.implementation = implementation
        self.plan_type = plan_type
        self.vendor_root = vendor_root
        self.provenance = dict(provenance or {})
        if not isinstance(native, bool):
            raise TypeError("native must be a boolean")
        self.native = native
        self.name = NAME if self.native else f"{NAME}_cpu_mock"

    @staticmethod
    def _fixed(eps: Any, scale: Any) -> None:
        _fixed_scalar(eps, "eps", EPS)
        _fixed_scalar(scale, "scale", 1.0)

    def _check(self, values: Any, query: Any) -> tuple[tuple[Tensor, ...], int]:
        sources, width = _validate_sources(values, query)
        if self.native and sources[0].device.type != "cuda":
            raise RuntimeError("Hydra native comparator requires CUDA values and query")
        return sources, width

    def timing_eligible(self, values: Any, query: Any | None = None) -> bool:
        """Whether this width is covered by the documented native envelope."""

        if not self.native:
            return False
        if query is None:
            return _timing_eligible(values)
        try:
            self._check(values, query)
        except (TypeError, ValueError, RuntimeError):
            return False
        return _timing_eligible(values, query)

    def timing_applicable(
        self,
        values: Any,
        query: Any,
        *,
        keys: Any = None,
    ) -> tuple[bool, str | None]:
        if not self.native:
            return False, "Hydra CPU mock is not a benchmark timing backend"
        if keys is not None:
            return False, "Hydra comparator only supports implicit keys (keys=None)"
        try:
            self._check(values, query)
        except (TypeError, ValueError, RuntimeError) as exc:
            return False, str(exc)
        if not _timing_eligible(values, query):
            return False, TIMING_EXCLUSION_REASON
        return True, None

    def __call__(
        self,
        values: Any = None,
        query: Tensor | None = None,
        *,
        keys: Tensor | None = None,
        eps: Any = EPS,
        scale: Any = 1.0,
        residuals: Any = None,
        **kwargs: Any,
    ) -> Tensor:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected Hydra comparator arguments: {unknown}")
        self._fixed(eps, scale)
        if keys is not None:
            raise ValueError("Hydra comparator rejects sliced/projected key inputs")
        if residuals is not None:
            if values is not None:
                raise TypeError("pass either values or residuals, not both")
            values = residuals
        if values is None or query is None:
            raise TypeError("Hydra comparator requires values and query")
        _sources, width = self._check(values, query)
        blocks, query_row, partial, has_partial = _stack_inputs(values, query)
        plan = _plan(self.plan_type, width)
        # Passing this explicit plan is the native-status guard: Hydra's API
        # cannot select its ``auto`` or portable ``torch`` path here.
        output = self.implementation(query_row, blocks, partial, has_partial, plan=plan)
        if isinstance(output, (tuple, list)):
            if not output:
                raise RuntimeError("Hydra native implementation returned an empty tuple")
            output = output[0]
        if not isinstance(output, Tensor):
            raise TypeError(f"Hydra native implementation returned {type(output).__name__}")
        expected_shape = (1, *tuple(blocks.shape[1:]))
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"Hydra native implementation returned shape {tuple(output.shape)}, expected {expected_shape}"
            )
        if output.dtype != blocks.dtype:
            raise RuntimeError(
                f"Hydra native implementation returned dtype {output.dtype}, expected {blocks.dtype}"
            )
        logical_shape = tuple(_sources[0].shape)
        return output[0].reshape(logical_shape)


@dataclass(frozen=True)
class _MockPlan:
    backend: str
    block_d: int
    num_warps: int
    eps: float


def _cpu_mock_attnres(
    queries: Tensor,
    blocks: Tensor,
    partials: Tensor,
    has_partial: Tensor,
    *,
    plan: _MockPlan,
) -> Tensor:
    """Equation implementation used only by explicit CPU tests."""

    if plan.backend != "triton":
        raise RuntimeError("CPU mock is called with the same explicit plan contract")
    q = queries.float()
    v = blocks.float()
    p = partials.float()
    keys = v * torch.rsqrt(v.square().mean(dim=-1, keepdim=True) + plan.eps)
    pkeys = p * torch.rsqrt(p.square().mean(dim=-1, keepdim=True) + plan.eps)
    block_logits = torch.einsum("sd,n...d->sn...", q, keys)
    partial_logits = torch.einsum("sd,s...d->s...", q, pkeys)
    mask = has_partial.reshape((-1,) + (1,) * (p.ndim - 2))
    partial_logits = partial_logits.masked_fill(~mask, -torch.inf)
    logits = torch.cat((block_logits, partial_logits.unsqueeze(1)), dim=1)
    values = torch.cat((v.unsqueeze(0).expand(q.shape[0], *v.shape), p.unsqueeze(1)), dim=1)
    weights = torch.softmax(logits, dim=1)
    return torch.einsum("sn...,sn...d->s...d", weights, values).to(partials.dtype)


def make_cpu_mock_backend() -> HydraBackend:
    """Return an explicit equation backend for CPU oracle/gradient tests."""

    return HydraBackend(_cpu_mock_attnres, _MockPlan, native=False)


def cpu_mock(values: Any, query: Tensor, *, eps: Any = EPS, scale: Any = 1.0) -> Tensor:
    """Run the explicit CPU mock; never used by native discovery."""

    return make_cpu_mock_backend()(values, query, eps=eps, scale=scale)


def source_hash_metadata(
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return verified checkout and adapter policy metadata."""

    root = resolve_vendor_root(vendor_root, project_root)
    reason, observed = _integrity(root)
    if reason:
        raise HydraProvenanceError(reason)
    source_root = root / "src"
    result: dict[str, Any] = {
        "backend": "hydra_2p_native_triton",
        "name": NAME,
        "repository": REPOSITORY,
        "pinned_revision": PINNED_REVISION,
        "pinned_tree": PINNED_TREE,
        "vendor_root": str(root),
        "vendor_top_level": observed["vendor_top_level"],
        "vendor_revision": observed["vendor_revision"],
        "vendor_tree": observed["vendor_tree"],
        "vendor_clean": observed["vendor_clean"],
        "vendor_origin": observed["vendor_origin"],
        "expected_origin": REPOSITORY,
        "license": "MIT",
        "license_file": LICENSE,
        "license_sha256": observed["vendor_file_sha256"].get(LICENSE),
        "expected_license_sha256": LICENSE_SHA256,
        "vendor_file_sha256": dict(observed["vendor_file_sha256"]),
        "expected_vendor_file_sha256": dict(_VENDOR_SHA256),
        "module_origins": _module_origins(source_root),
        "entrypoint": "attnres_kernel.api.attnres",
        "plan_backend": "triton",
        "block_d": "next_power_of_two(D)",
        "num_warps": NUM_WARPS,
        "eps": EPS,
        "scale": 1.0,
        "equation": "standard implicit R=D; FP32 score math; full-width value mixture",
        "supports_sliced": False,
        "supports_projected": False,
        "supports_full": True,
        "supports_per_read_block": False,
        "supports_external_block_panel": True,
        "block_scope": "external_block_panel",
        "full_schedule": "one query row over stacked source blocks",
        "block_schedule": "external Block panel over stacked source blocks",
        "accepts_source_list": True,
        "native_model_source_list": False,
        "source_list_copy": "torch.stack(sources, dim=0) inside adapter",
        "packed_copy": "torch.stack(unbind(values), dim=0), optional singleton token axis, and contiguous inside adapter",
        "source_stack": "torch.stack(sources, dim=0) inside adapter",
        "source_stack_cost": "included in caller timing boundary",
        "contiguous_cost": "stacked blocks, query row, and dummy partial use contiguous storage inside adapter",
        "dummy_partial": "finite zeros_like(blocks[:1]) with has_partial=False",
        "native_width_documented_max": MAX_WIDTH,
        "native_documented_max_width": MAX_WIDTH,
        "native_width_limit": MAX_WIDTH,
        "native_timing_width_max": NATIVE_MAX_WIDTH,
        "timing_width_max": NATIVE_MAX_WIDTH,
        "higher_width_timing": "ineligible pending independent compile gate",
        "timing_predicate": TIMING_PREDICATE,
        "timing_predicate_name": TIMING_PREDICATE_NAME,
        "timing_predicate_enforced": True,
        "timing_exclusion_reason": TIMING_EXCLUSION_REASON,
        "timing_eligibility": TIMING_PREDICATE,
        "timing_eligibility_enforced": True,
        "native_fallback": "none; auto and torch plans are rejected",
        "cuda_required": True,
        "cpu_mock": "explicit make_cpu_mock_backend only; never native discovery",
        "gradient_contract": "all source values and query vectors; live partial gradients are covered by the vendor ABI",
        "qualification": "independent Full/external Block-panel oracle, output, and every source/query gradient required; this pass ran CPU/static checks only",
    }
    return result


class Comparator:
    """JSON-friendly native discovery result."""

    def __init__(
        self,
        backend: HydraBackend | None = None,
        *,
        status: str,
        reason: str | None = None,
        root: Path | None = None,
        revision: str | None = None,
        tree: str | None = None,
        dirty: bool | None = None,
        origin: str | None = None,
        hashes: Mapping[str, str] | None = None,
        module_origins: Mapping[str, Sequence[str]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = NAME
        self.call = backend
        self.status = status
        self.reason = reason
        self.vendor_root = str(root) if root is not None else None
        self.vendor_revision = revision
        self.vendor_tree = tree
        self.vendor_dirty = dirty
        self.vendor_origin = origin
        self.vendor_file_hashes = dict(hashes or {})
        self.module_origins = {key: list(value) for key, value in (module_origins or {}).items()}
        self.kind = "triton"
        self.metadata = dict(metadata or {})

    @property
    def available(self) -> bool:
        return self.status == "available" and self.call is not None

    def applicable(
        self,
        values: Any,
        query: Any,
        *,
        keys: Any = None,
    ) -> tuple[bool, str | None]:
        if not self.available:
            return False, self.reason or "Hydra comparator is unavailable"
        if keys is not None:
            return False, "Hydra comparator only supports implicit keys (keys=None)"
        try:
            if isinstance(self.call, HydraBackend):
                self.call._check(values, query)
            else:
                _validate_sources(values, query)
        except (TypeError, ValueError, RuntimeError) as exc:
            return False, str(exc)
        return True, None

    def timing_eligible(
        self,
        values: Any,
        query: Any | None = None,
        *,
        keys: Any = None,
    ) -> bool:
        if not self.available:
            return False
        if query is None:
            return _timing_eligible(values, keys=keys)
        okay, _reason = self.timing_applicable(values, query, keys=keys)
        return okay

    def timing_applicable(
        self,
        values: Any,
        query: Any,
        *,
        keys: Any = None,
    ) -> tuple[bool, str | None]:
        if not self.available:
            return False, self.reason or "Hydra comparator is unavailable"
        if isinstance(self.call, HydraBackend):
            return self.call.timing_applicable(values, query, keys=keys)
        okay, reason = self.applicable(values, query, keys=keys)
        if not okay:
            return False, reason
        if not _timing_eligible(values, query, keys=keys):
            return False, TIMING_EXCLUSION_REASON
        return True, None

    def describe(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "vendor_root": self.vendor_root,
            "vendor_revision": self.vendor_revision,
            "vendor_tree": self.vendor_tree,
            "pinned_revision": PINNED_REVISION,
            "pinned_tree": PINNED_TREE,
            "vendor_clean": self.vendor_dirty is False,
            "vendor_dirty": self.vendor_dirty,
            "vendor_origin": self.vendor_origin,
            "expected_origin": REPOSITORY,
            "license": "MIT",
            "license_file": LICENSE,
            "license_sha256": self.vendor_file_hashes.get(LICENSE),
            "expected_license_sha256": LICENSE_SHA256,
            "vendor_file_sha256": dict(self.vendor_file_hashes),
            "expected_vendor_file_sha256": dict(_VENDOR_SHA256),
            "module_origins": dict(self.module_origins),
            "repository": REPOSITORY,
            "entrypoint": "attnres_kernel.api.attnres",
            "plan_backend": "triton",
            "block_d": "next_power_of_two(D)",
            "num_warps": NUM_WARPS,
            "eps": EPS,
            "scale": 1.0,
            "equation": "standard implicit R=D; FP32 score math; full-width value mixture",
            "supports_sliced": False,
            "supports_projected": False,
            "supports_full": True,
            "supports_per_read_block": False,
            "supports_external_block_panel": True,
            "block_scope": "external_block_panel",
            "full_schedule": "one query row over stacked source blocks",
            "block_schedule": "external Block panel over stacked source blocks",
            "matched_rank_only": True,
            "accepts_source_list": True,
            "native_model_source_list": False,
            "source_list_copy": "torch.stack(sources, dim=0) inside adapter",
            "packed_copy": "torch.stack(unbind(values), dim=0), optional singleton token axis, and contiguous inside adapter",
            "source_stack": "torch.stack(sources, dim=0) inside adapter",
            "source_stack_cost": "included in caller timing boundary",
            "contiguous_cost": "stacked blocks, query row, and dummy partial use contiguous storage inside adapter",
            "dummy_partial": "finite zeros_like(blocks[:1]) with has_partial=False",
            "native_width_documented_max": MAX_WIDTH,
            "native_documented_max_width": MAX_WIDTH,
            "native_width_limit": MAX_WIDTH,
            "native_timing_width_max": NATIVE_MAX_WIDTH,
            "timing_width_max": NATIVE_MAX_WIDTH,
            "higher_width_timing": "ineligible pending independent compile gate",
            "timing_predicate": TIMING_PREDICATE,
            "timing_predicate_name": TIMING_PREDICATE_NAME,
            "timing_predicate_enforced": True,
            "timing_exclusion_reason": TIMING_EXCLUSION_REASON,
            "timing_eligibility": TIMING_PREDICATE,
            "timing_eligibility_enforced": True,
            "native_fallback": "none; auto and torch plans are rejected",
            "cuda_required": True,
            "cpu_mock": "explicit make_cpu_mock_backend only; never native discovery",
            "gradient_contract": "all source values and query vectors; live partial gradients are covered by the vendor ABI",
            "qualification": "independent Full/external Block-panel oracle, output, and every source/query gradient required; this pass ran CPU/static checks only",
        }
        result.update(self.metadata)
        if self.reason:
            result["reason"] = self.reason
        result["adapter_sha256"] = _sha256(Path(__file__))
        return result

    as_dict = describe


def _missing(
    reason: str,
    *,
    root: Path | None = None,
    observed: Mapping[str, Any] | None = None,
) -> Comparator:
    observed = observed or {}
    source_root = root / "src" if root is not None else None
    return Comparator(
        None,
        status="missing",
        reason=reason,
        root=root,
        revision=observed.get("vendor_revision"),
        tree=observed.get("vendor_tree"),
        dirty=observed.get("vendor_dirty"),
        origin=observed.get("vendor_origin"),
        hashes=observed.get("vendor_file_sha256"),
        module_origins=_module_origins(source_root) if source_root is not None else None,
    )


def _load_native(root: Path) -> tuple[Callable[..., Any], Callable[..., Any], dict[str, list[str]]]:
    root = Path(root).expanduser().resolve()
    reason, _observed = _integrity(root)
    if reason:
        raise HydraProvenanceError(reason)
    source = root / "src"
    if not _all_loaded_origins_ok(source):
        raise HydraProvenanceError(
            "loaded attnres_kernel modules originate outside the pinned source; "
            "restart the process before rediscovery"
        )
    source_string = str(source)
    # Put the verified source first while retaining the caller's remaining
    # import search path.  Reusing an earlier path could otherwise resolve a
    # same-named package from an unrelated checkout.
    sys.path[:] = [source_string] + [item for item in sys.path if item != source_string]
    importlib.invalidate_caches()
    package = importlib.import_module("attnres_kernel")
    api = importlib.import_module("attnres_kernel.api")
    # Importing this module is an intentional native availability check.  It
    # imports Triton and never falls back to the portable implementation.
    importlib.import_module("attnres_kernel.triton_impl")
    if not _origins_ok(source) or not _all_loaded_origins_ok(source):
        origins = _module_origins(source)
        raise HydraProvenanceError(f"Hydra modules loaded outside pinned source: {origins}")
    implementation = getattr(api, "attnres", None)
    plan_type = getattr(api, "AttnResPlan", None)
    if not callable(implementation) or not callable(plan_type):
        raise ImportError("Hydra public attnres/AttnResPlan entrypoints are missing")
    # Keep references alive and make the package origin visible in metadata.
    del package
    return implementation, plan_type, _module_origins(source)


def discover_comparators(
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> dict[str, Comparator]:
    """Discover one exact, native Triton Hydra comparator."""

    root = find_vendor_root(project_root, vendor_root)
    if root is None:
        return {NAME: _missing("pinned attnres-kernel-lab checkout was not found")}
    reason, observed = _integrity(root)
    if reason:
        return {NAME: _missing(reason, root=root, observed=observed)}
    try:
        implementation, plan_type, origins = _load_native(root)
        backend = HydraBackend(
            implementation,
            plan_type,
            vendor_root=root,
            provenance=observed,
            native=True,
        )
    except Exception as exc:
        return {
            NAME: _missing(
                f"Hydra native Triton import failed: {type(exc).__name__}: {exc}",
                root=root,
                observed={**observed, "module_origins": _module_origins(root / "src")},
            )
        }
    return {
        NAME: Comparator(
            backend,
            status="available",
            root=root,
            revision=observed["vendor_revision"],
            tree=observed["vendor_tree"],
            dirty=observed["vendor_dirty"],
            origin=observed["vendor_origin"],
            hashes=observed["vendor_file_sha256"],
            module_origins=origins,
        )
    }


def discover_comparator(
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> Comparator:
    return discover_comparators(project_root, vendor_root)[NAME]


discover = discover_comparators


def invoke_comparator(
    comparator: Comparator,
    values: Any,
    query: Tensor,
    *,
    keys: Any = None,
) -> Tensor:
    """Invoke an available native comparator after shape checks."""

    if not comparator.available:
        raise RuntimeError(comparator.reason or "Hydra comparator is unavailable")
    okay, reason = comparator.applicable(values, query, keys=keys)
    if not okay:
        raise ValueError(reason or "Hydra comparator is not applicable")
    if keys is None:
        return comparator.call(values, query, eps=EPS, scale=1.0)  # type: ignore[misc]
    return comparator.call(values, query, keys=keys, eps=EPS, scale=1.0)  # type: ignore[misc]


def model_backend(comparator: Comparator) -> Callable[..., Tensor]:
    """Adapt a discovered comparator to the model's per-read operator ABI."""

    if not isinstance(comparator, Comparator) or not comparator.available:
        reason = comparator.reason if isinstance(comparator, Comparator) else None
        raise RuntimeError(reason or "Hydra comparator is unavailable")
    core = comparator.call

    def backend(
        values: Any,
        query: Tensor,
        *,
        keys: Tensor | None = None,
        eps: Any = EPS,
        scale: Any = 1.0,
    ) -> Tensor:
        if keys is None:
            return core(values, query, eps=eps, scale=scale)  # type: ignore[misc]
        return core(values, query, keys=keys, eps=eps, scale=scale)  # type: ignore[misc]

    backend.__name__ = f"{NAME}_model_backend"
    backend.accepts_source_list = True  # type: ignore[attr-defined]
    backend.native_model_source_list = False  # type: ignore[attr-defined]
    backend.source_stack_owned_by_adapter = True  # type: ignore[attr-defined]
    backend.timing_eligible = getattr(core, "timing_eligible", timing_eligible)  # type: ignore[attr-defined]
    backend.timing_applicable = getattr(  # type: ignore[attr-defined]
        core, "timing_applicable", _timing_applicability
    )
    backend.timing_predicate = TIMING_PREDICATE  # type: ignore[attr-defined]
    backend.source_hash_metadata = comparator.describe()  # type: ignore[attr-defined]
    return backend


def make_model_backend(
    comparator: Comparator | None = None,
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> Callable[..., Tensor]:
    """Return the native model callable, discovering it when needed."""

    return model_backend(comparator or discover_comparator(project_root, vendor_root))


def vendor_metadata(
    project_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Report observed vendor identity without importing Triton."""

    root = find_vendor_root(project_root, vendor_root)
    if root is None:
        return {
            "path": None,
            "git_revision": None,
            "git_tree": None,
            "vendor_file_sha256": {},
            "expected_vendor_file_sha256": dict(_VENDOR_SHA256),
            "license": "MIT",
            "license_file": LICENSE,
            "license_sha256": None,
            "expected_license_sha256": LICENSE_SHA256,
            "repository": REPOSITORY,
            "expected_origin": REPOSITORY,
            "vendor_origin": None,
            "origin": None,
            "clean": False,
            "pinned": False,
            "pinned_revision": PINNED_REVISION,
            "pinned_tree": PINNED_TREE,
            "timing_predicate": TIMING_PREDICATE,
            "timing_predicate_name": TIMING_PREDICATE_NAME,
            "timing_predicate_enforced": True,
            "timing_width_max": NATIVE_MAX_WIDTH,
            "cuda_required": True,
            "native_fallback": "none; auto and torch plans are rejected",
            "supports_per_read_block": False,
            "supports_external_block_panel": True,
            "block_scope": "external_block_panel",
            "reason": "pinned attnres-kernel-lab checkout was not found",
        }
    reason, observed = _integrity(root)
    return {
        "path": str(root),
        "git_top_level": observed["vendor_top_level"],
        "git_revision": observed["vendor_revision"],
        "git_tree": observed["vendor_tree"],
        "clean": observed["vendor_clean"],
        "origin": observed["vendor_origin"],
        "vendor_origin": observed["vendor_origin"],
        "file_sha256": observed["vendor_file_sha256"],
        "vendor_file_sha256": observed["vendor_file_sha256"],
        "expected_vendor_file_sha256": dict(_VENDOR_SHA256),
        "pinned_revision": PINNED_REVISION,
        "pinned_tree": PINNED_TREE,
        "pinned": reason is None,
        "reason": reason,
        "license": "MIT",
        "license_file": LICENSE,
        "license_sha256": observed["vendor_file_sha256"].get(LICENSE),
        "expected_license_sha256": LICENSE_SHA256,
        "repository": REPOSITORY,
        "expected_origin": REPOSITORY,
        "supports_per_read_block": False,
        "supports_external_block_panel": True,
        "block_scope": "external_block_panel",
    }


__all__ = [
    "Comparator",
    "EPS",
    "HYDRA_REPOSITORY",
    "HydraBackend",
    "HydraProvenanceError",
    "LICENSE",
    "LICENSE_SHA256",
    "MAX_SOURCES",
    "MAX_WIDTH",
    "NAME",
    "NATIVE_MAX_WIDTH",
    "NUM_WARPS",
    "TIMING_EXCLUSION_REASON",
    "TIMING_PREDICATE",
    "TIMING_PREDICATE_NAME",
    "PINNED_REVISION",
    "PINNED_TREE",
    "REPOSITORY",
    "VENDOR_REVISION",
    "VENDOR_TREE",
    "cpu_mock",
    "discover",
    "discover_comparator",
    "discover_comparators",
    "find_vendor_root",
    "invoke_comparator",
    "make_cpu_mock_backend",
    "make_model_backend",
    "model_backend",
    "next_power_of_two",
    "resolve_vendor_root",
    "source_hash_metadata",
    "timing_eligible",
    "vendor_metadata",
]
