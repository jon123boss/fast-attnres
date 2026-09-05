"""Pinned Catswe phase-1 standard-operator adapter for AttnRes.

Only the vendor's public phase-1 operator is callable here.  The repository's
cached or multi-phase implementation is deliberately outside this adapter's
surface and is never forwarded to the model runner.
"""
from __future__ import annotations
import hashlib, importlib, json, math, os, subprocess, sys
from pathlib import Path
from typing import Any
import torch
from .vendor_identity import (
    CheckoutIdentityError,
    module_origins,
    normalize_remote_origin,
    require_module_origins,
    verify_remote_origin,
)
EPS = 2**-23
PINNED_REVISION = "ff92865e4e1b18809da7a8f0c0c5252039cded7c"
PINNED_TREE = "f4f96a21dbe609044edef2fdbaf66a820c260fc0"
REPOSITORY = "https://github.com/catswe/flash-attention-residuals.git"
LICENSE = "LICENSE"
LICENSE_SHA256 = "299e72fdffa70bc47c4c6b7e60d71d698c9f0808b82275ba524538ed8233e08f"
REMOTE_MANIFEST = "provenance.json"
NAME = "catswe_phase1"
MAX_SOURCES = 129
MAX_WIDTH = 8192
MAX_PROGRAM_ELEMENTS = 1_048_576
TIMING_PREDICATE = (
    "1 <= S <= 129, 1 <= D <= 8192, D is power-of-two, "
    "nextpow2(S) * D <= 1048576"
)
TIMING_PREDICATE_NAME = "benchmarks.catswe._timing_eligible"
MODEL_TIMING_SCOPE = "compiled_training_step"
TIMING_EXCLUSION_REASON = (
    "Catswe native timing requires 1 <= S <= 129, power-of-two D with "
    "1 <= D <= 8192, and nextpow2(S) * D <= 1048576"
)
_VENDOR_FILES = ("LICENSE", "pyproject.toml", "src/flash_attn_res/__init__.py",
                 "src/flash_attn_res/kernels/__init__.py", "src/flash_attn_res/kernels/configs.py", "src/flash_attn_res/kernels/phase_1.py",
                 "src/flash_attn_res/kernels/phase_2.py", "src/flash_attn_res/kernels/reduce.py", "src/flash_attn_res/ops/__init__.py", "src/flash_attn_res/ops/phase_1.py", "src/flash_attn_res/ops/phase_2.py")
_VENDOR_SHA256 = {
    "LICENSE": "299e72fdffa70bc47c4c6b7e60d71d698c9f0808b82275ba524538ed8233e08f", "pyproject.toml": "081b3067bca515c24edb090678fad477e52ef759a9cfa571449962f0ff63f164",
    "src/flash_attn_res/__init__.py": "04d5c0eefd4d4a994f7521b26fb04d7fbe4fe29245c6f9ff64ac9a56fe224868", "src/flash_attn_res/kernels/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/flash_attn_res/kernels/configs.py": "ebec80ad42fa781e54169e69b53d154c00c1b80ac2e1bb7a63a5ec817ef9bf85", "src/flash_attn_res/kernels/phase_1.py": "bd45efeb8a69b6ff47f2caee66103a56d19494e5a0e50640b3bf4ba5b4048982",
    "src/flash_attn_res/kernels/phase_2.py": "aa2e92e5979d3093d26bbf477bf28daad834cea98b0d35e4225dd1d8c621623d", "src/flash_attn_res/kernels/reduce.py": "8dfdeb9a5031a0a4b25d9cb49e003ea39fd49157ee4a571fe91ff91e5b04ce0b",
    "src/flash_attn_res/ops/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "src/flash_attn_res/ops/phase_1.py": "251610079e6a8847c860391641c8903f250e9dd628ec4522e1039ecadb4e7060", "src/flash_attn_res/ops/phase_2.py": "b56aecf3856be87b48c7891aabb249a7b34f6a59be0af7f89eacb9ff07cd9479",
}
_REQUIRED_MODULES = tuple("flash_attn_res." + n for n in ("kernels", "kernels.configs", "kernels.phase_1", "kernels.reduce", "ops", "ops.phase_1")) + ("flash_attn_res",)
def _contains_symlink_component(path: Path) -> bool:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return any(component.is_symlink() for component in (path, *path.parents))
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def _revision(root: Path) -> str | None:
    try: return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): return None
def _tree(root: Path) -> str | None:
    try: return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): return None
def _top_level(root: Path) -> Path | None:
    try: return Path(subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True).stdout.strip()).resolve()
    except (OSError, subprocess.CalledProcessError, RuntimeError): return None
def _dirty(root: Path) -> bool | None:
    try: status = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError): return None
    return bool(status.strip())


def _origin(root: Path) -> str | None:
    try:
        return verify_remote_origin(root, REPOSITORY)
    except CheckoutIdentityError:
        return None
def _file_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    base = root.resolve()
    for name in _VENDOR_FILES:
        path = root / name
        if _contains_symlink_component(path):
            continue
        try:
            path.resolve().relative_to(base)
        except (OSError, RuntimeError, ValueError):
            continue
        if path.is_file():
            result[name] = _sha256(path)
    return result
def _remote_source_error(root: Path) -> str | None:
    expected = {name for name in _VENDOR_FILES if name.startswith("src/") and name.endswith(".py")}
    base = root.resolve()
    try:
        for current_name, directories, files in os.walk(root, followlinks=False):
            current = Path(current_name)
            for name in (*directories, *files):
                path = current / name
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    return f"remote Catswe payload contains a symlink: {relative}"
                path.resolve().relative_to(base)
                if path.suffix == ".py" and relative not in expected:
                    return f"remote Catswe payload has unexpected source file: {relative}"
    except (OSError, ValueError) as exc:
        return f"remote Catswe payload has an escaping source path: {exc}"
    return None
def _remote_integrity(root: Path):
    manifest_path = root / REMOTE_MANIFEST
    if os.path.lexists(root / ".git"): return "remote Catswe payload must not contain .git", None, None, {}, None
    try:
        if manifest_path.is_symlink():
            raise ValueError("provenance manifest is a symlink")
        manifest_path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return "remote Catswe provenance manifest is outside its payload root", None, None, {}, None
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return f"remote Catswe provenance manifest is unreadable: {type(exc).__name__}", None, None, {}, None
    if not isinstance(manifest, dict):
        return "remote Catswe provenance manifest must be an object", None, False, {}, None
    revision, tree, files = manifest.get("vendor_revision"), manifest.get("vendor_tree"), manifest.get("files")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if raw_manifest != canonical:
        return "remote Catswe provenance manifest encoding mismatch", revision, False, files if isinstance(files, dict) else {}, None
    if manifest.get("schema") != "catswe_remote_provenance_v1" or manifest.get("source_root") != "src" or revision != PINNED_REVISION or tree != PINNED_TREE:
        return "remote Catswe provenance revision/tree/schema mismatch", revision, False, files or {}, None
    if manifest.get("host_git_preflight") is not True or manifest.get("remote_git_present") is not False:
        return "remote Catswe provenance lacks host-only git preflight", revision, False, files or {}, None
    origin = manifest.get("vendor_origin")
    if normalize_remote_origin(origin) != normalize_remote_origin(REPOSITORY):
        return "remote Catswe provenance origin does not match pinned repository", revision, False, files or {}, None
    expected_manifest = os.environ.get("ATTNRES_CATSWE_MANIFEST_SHA256", "").strip().lower()
    if files != _VENDOR_SHA256: return "remote Catswe manifest file hash set mismatch", revision, False, files or {}, None
    source_error = _remote_source_error(root)
    if source_error: return source_error, revision, False, files, None
    try:
        base = root.resolve()
        for relative, expected in files.items():
            raw_path = root / relative
            if raw_path.is_symlink():
                raise ValueError(f"symlink: {relative}")
            path = raw_path.resolve(); path.relative_to(base)
            if not path.is_file() or _sha256(path) != expected: raise ValueError(relative)
        manifest_sha256 = _sha256(manifest_path)
        if expected_manifest and manifest_sha256 != expected_manifest: raise ValueError("provenance manifest digest")
    except (OSError, ValueError, TypeError) as exc:
        return f"remote Catswe payload byte mismatch: {exc}", revision, False, files, None
    transport = {"transport": "host_git_preflight+remote_bytes", "host_git_preflight": True, "remote_bytes": True, "remote_git_present": False, "vendor_revision": revision, "vendor_tree": tree, "vendor_origin": origin, "origin": origin, "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256, "files": dict(files), "manifest": manifest}
    return None, revision, False, files, transport
def _origin_ok(module: Any, source: Path) -> bool:
    if module is None: return False
    origins = [module.__file__] if getattr(module, "__file__", None) else []
    module_path = getattr(module, "__path__", ())
    if isinstance(module_path, (str, os.PathLike)):
        origins.append(module_path)
    else:
        origins.extend(module_path or ())
    try: return bool(origins) and all(Path(item).resolve().relative_to(source.resolve()) is not None for item in origins)
    except (OSError, ValueError, TypeError): return False


def _all_loaded_origins_ok(source: Path) -> bool:
    """Reject a previously imported Catswe package from another checkout."""
    source = source.resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "flash_attn_res" and not name.startswith("flash_attn_res."):
            continue
        if not _origin_ok(module, source):
            return False
    return True


def _roots(project_root: Path, configured: str | Path | None) -> list[Path]:
    candidates = [Path(configured).expanduser()] if configured is not None else []
    if configured is None:
        for key in ("CATSWE_ROOT", "ATTNRES_CATSWE_REMOTE_ROOT", "FLASH_ATTENTION_RESIDUALS_ROOT", "FLASH_ATTN_RES_ROOT"):
            value = os.environ.get(key)
            if value:
                candidates.append(Path(value).expanduser())
                break
        if not candidates:
            candidates += [project_root.parent.parent / "vendor" / "flash-attention-residuals", project_root / "vendor" / "flash-attention-residuals"]
    return list(
        dict.fromkeys(
            candidate.resolve()
            for candidate in candidates
            if not _contains_symlink_component(candidate)
        )
    )
def _remote_requested(root: Path) -> bool:
    configured = os.environ.get("ATTNRES_CATSWE_REMOTE_ROOT", "").strip()
    if not configured: return False
    try: return Path(configured).expanduser().resolve() == root.resolve()
    except (OSError, RuntimeError): return False
def find_vendor_root(project_root: str | Path | None = None, vendor_root: str | Path | None = None) -> Path | None:
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    candidates = _roots(root, vendor_root)
    # An explicit path is authoritative.  Do not silently replace a caller's
    # requested checkout with an ambient candidate from the search list.
    if vendor_root is not None:
        candidates = candidates[:1]
    return next((c for c in candidates if (c / "src" / "flash_attn_res").is_dir()), None)
def _integrity(root: Path):
    root = Path(root).expanduser().resolve()
    if (root / REMOTE_MANIFEST).is_file(): return _remote_integrity(root)
    if _remote_requested(root): return "requested remote Catswe provenance manifest is missing", None, None, {}, None
    top, revision, tree, dirty, hashes = _top_level(root), _revision(root), _tree(root), _dirty(root), _file_hashes(root)
    if top != root: return "could not verify pinned Catswe checkout top-level", revision, dirty, hashes, None
    if revision != PINNED_REVISION: return f"expected pinned revision {PINNED_REVISION}, got {revision!r}", revision, dirty, hashes, None
    if tree != PINNED_TREE: return f"expected pinned tree {PINNED_TREE}, got {tree!r}", revision, dirty, hashes, None
    if dirty is None: return "could not verify pinned checkout cleanliness", revision, dirty, hashes, None
    if dirty: return "pinned Catswe checkout is dirty", revision, dirty, hashes, None
    try:
        verify_remote_origin(root, REPOSITORY)
    except CheckoutIdentityError as exc:
        return f"pinned Catswe origin verification failed: {exc}", revision, dirty, hashes, None
    if hashes != _VENDOR_SHA256: return "pinned Catswe source/package/license hash mismatch", revision, dirty, hashes, None
    return None, revision, dirty, hashes, None


def _timing_eligible(values: Any, query: Any = None) -> bool:
    """Return whether a case is inside Catswe's bounded standard envelope."""
    try:
        if query is None:
            return (
                isinstance(values, int)
                and not isinstance(values, bool)
                and 1 <= values <= MAX_WIDTH
                and values & (values - 1) == 0
            )
        # This shape predicate is also useful before device allocation.  The
        # native backend adds its explicit CUDA applicability check below.
        CatsweBackend._check(values, query, require_cuda=False)
    except (TypeError, ValueError, RuntimeError):
        return False
    return True


timing_eligible = _timing_eligible


def make_model_backend(comparator: Any):
    """Adapt verified public phase 1 to one Full/Block model read.

    Catswe is exposed here only through its public phase 1 operator.  The
    model runner calls this adapter once for every actual Full or Block read;
    there is no prepare, merge, cache, or phase 2 route.  ``CatsweBackend``
    owns the source-list ``stack`` and ``contiguous`` operations, so those
    operations are traced inside the captured training step.
    """

    if getattr(comparator, "available", False) is not True:
        reason = getattr(comparator, "reason", None)
        raise RuntimeError(reason or "Catswe comparator is unavailable")
    call = getattr(comparator, "call", None)
    if not isinstance(call, CatsweBackend) or call.native is not True:
        raise RuntimeError(
            "Catswe compiled model arm requires the verified native phase-1 backend"
        )

    def backend(values: Any, query: torch.Tensor, *, eps=EPS, scale=1.0):
        # Keep the model route separate from invoke_comparator.  The backend
        # call itself performs all native shape/device checks and staging.
        return call(values, query, eps=eps, scale=scale)

    backend.__name__ = "catswe_phase1_compiled_model_backend"
    # CausalAttnResLM supplies an ordered tuple of sources.  The wrapped
    # backend deliberately reports that it does not consume a native pointer
    # list; CatsweBackend stacks and makes it contiguous in this call.
    backend.accepts_source_list = True  # type: ignore[attr-defined]
    backend.native_model_source_list = False  # type: ignore[attr-defined]
    backend.supports_full = True  # type: ignore[attr-defined]
    backend.supports_per_read_block = True  # type: ignore[attr-defined]
    backend.vendor_root = getattr(comparator, "vendor_root", None)  # type: ignore[attr-defined]
    backend.vendor_revision = getattr(  # type: ignore[attr-defined]
        comparator, "vendor_revision", PINNED_REVISION
    )

    provenance: dict[str, Any] = {}
    describe = getattr(comparator, "describe", None)
    if callable(describe):
        described = describe()
        if isinstance(described, dict):
            provenance.update(described)
    provenance.update(
        {
            "backend": "catswe_phase1_compiled_model_adapter",
            "model_scope": MODEL_TIMING_SCOPE,
            "model_abi": "one public phase-1 call per actual Full/Block read",
            "full_schedule": "public phase-1 per-read aggregation",
            "block_schedule": (
                "public phase-1 per-read aggregation over completed+partial sources"
            ),
            "supports_full": True,
            "supports_per_read_block": True,
            "accepts_source_list": True,
            "native_model_source_list": False,
            "model_source_argument": "ordered source-list tuple",
            "source_list_copy": "torch.stack inside captured model adapter call",
            "packed_copy": "torch.contiguous inside captured model adapter call",
            "timing_boundary": (
                "captured complete-step event includes source stack and contiguous "
                "staging in this model adapter"
            ),
            "cache_api": "none",
            "prepare_api": "none",
            "merge_api": "none",
            "phase2_api": "none",
            "native_fallback": "none; unavailable native comparator fails closed",
            "capability_limits": {
                "rank": "R=D only",
                "sources": f"1<=S<={MAX_SOURCES} per read",
                "width": f"power-of-two 1<=D<={MAX_WIDTH}",
                "program_elements": f"nextpow2(S)*D<={MAX_PROGRAM_ELEMENTS}",
                "dtype": "BF16 value storage; BF16 autocast model step",
                "device": "CUDA only",
                "modes": "Full and Block per-read",
            },
            "qualification_oracle": "validation.oracle.oracle",
            "qualification_checks": [
                "output",
                "all_value_gradients",
                "query_gradient",
            ],
            "model_qualification": "benchmarks.run._model_qualification",
            "complete_training": (
                "requires independent model output/all-parameter gradient gate"
            ),
            "adapter_file": str(Path(__file__).resolve()),
            "adapter_sha256": _sha256(Path(__file__).resolve()),
        }
    )
    backend.source_hash_metadata = provenance  # type: ignore[attr-defined]
    return backend


class CatsweBackend:
    accepts_source_list = False
    native_model_source_list = False
    supports_full = True
    supports_per_read_block = False
    timing_predicate = TIMING_PREDICATE
    def __init__(self, phase1: Any, root: Path, *, native: bool = True, metadata: dict[str, Any] | None = None):
        if not isinstance(native, bool): raise TypeError("native must be a boolean")
        self.phase1 = phase1
        self.vendor_root, self.vendor_revision = str(root), PINNED_REVISION if native else None
        self.native = native
        self.source_hash_metadata = dict(metadata or {})
        self.name = NAME if native else f"{NAME}_cpu_mock"
    @staticmethod
    def _fixed(eps: float, scale: float) -> None:
        if float(eps) != EPS or float(scale) != 1.0: raise ValueError("Catswe uses eps=2**-23 and scale=1")

    def timing_eligible(self, values: Any, query: torch.Tensor | None = None) -> bool:
        if not self.native:
            return False
        if query is None:
            return (
                isinstance(values, int)
                and not isinstance(values, bool)
                and 1 <= values <= MAX_WIDTH
                and values & (values - 1) == 0
            )
        try:
            self._check(values, query, require_cuda=self.native)
        except (TypeError, ValueError, RuntimeError):
            return False
        return True

    def timing_applicable(self, values: Any, query: torch.Tensor) -> tuple[bool, str | None]:
        if not self.native:
            return False, "Catswe CPU mock is not a benchmark timing backend"
        if not self.timing_eligible(values, query):
            return False, TIMING_EXCLUSION_REASON
        return True, None
    @staticmethod
    def _check(values: Any, query: torch.Tensor, *, require_cuda: bool = True) -> int:
        if not isinstance(query, torch.Tensor): raise TypeError("query must be a tensor")
        if isinstance(values, (list, tuple)):
            if not values or any(not isinstance(v, torch.Tensor) for v in values): raise TypeError("values must be a nonempty tensor sequence")
            first, source_count = values[0], len(values)
            if any(v.shape != first.shape or v.dtype != first.dtype or v.device != first.device for v in values): raise ValueError("all source tensors must share shape, dtype, and device")
        else:
            if not isinstance(values, torch.Tensor) or values.ndim < 2: raise ValueError("values must have shape [S,...,D]")
            first, source_count = values, int(values.shape[0])
        if first.ndim < 1 or not all(int(size) > 0 for size in first.shape) or not 1 <= source_count <= MAX_SOURCES: raise ValueError(f"Catswe supports 1<=S<={MAX_SOURCES} with positive dimensions")
        if first.dtype != torch.bfloat16: raise TypeError("Catswe requires BF16 value storage")
        if query.device != first.device: raise RuntimeError("Catswe values and query must share one device")
        if require_cuda and first.device.type != "cuda": raise RuntimeError("Catswe native comparator requires CUDA values and query")
        if query.dtype not in (torch.bfloat16, torch.float32) or query.ndim != 1: raise ValueError("query must be a BF16/FP32 vector")
        width = int(first.shape[-1])
        if not 1 <= width <= MAX_WIDTH or int(query.numel()) != width: raise ValueError("Catswe is only matched for standard R=D")
        if width & (width - 1):
            raise ValueError("Catswe native phase 1 requires power-of-two D")
        padded_sources = 1 << (source_count - 1).bit_length()
        if padded_sources * width > MAX_PROGRAM_ELEMENTS:
            raise ValueError(
                "Catswe native phase-1 block exceeds Triton's 1048576-element limit: "
                f"nextpow2(S)={padded_sources}, D={width}"
            )
        return width
    def _canonical(self, values: Any, query: torch.Tensor):
        width = self._check(values, query, require_cuda=self.native)
        if isinstance(values, (list, tuple)): values = torch.stack(tuple(values), dim=0)
        values = values.contiguous()
        output_shape = tuple(int(size) for size in values.shape[1:])
        rows = math.prod(output_shape[:-1]) or 1
        return values.reshape(int(values.shape[0]), 1, rows, width), query.reshape(1, width).contiguous(), output_shape
    def _phase1(self, packed: torch.Tensor, queries: torch.Tensor):
        result = self.phase1(packed, queries, float(EPS))
        if not isinstance(result, (tuple, list)) or len(result) != 2: raise RuntimeError("Catswe phase 1 must return (BF16 mixture, FP32 LSE)")
        mixture, lse = result
        if tuple(mixture.shape) != (int(queries.shape[0]), *packed.shape[1:]) or tuple(lse.shape) != (int(queries.shape[0]), *packed.shape[1:-1]): raise RuntimeError("Catswe phase 1 returned an unexpected shape")
        if mixture.dtype != torch.bfloat16 or lse.dtype != torch.float32: raise RuntimeError("Catswe phase 1 precision is not native BF16/FP32")
        return mixture, lse
    def __call__(self, values: Any, query: torch.Tensor, *, eps=EPS, scale=1.0):
        self._fixed(eps, scale)
        packed, query_2d, output_shape = self._canonical(values, query)
        mixture, _ = self._phase1(packed, query_2d)
        return mixture[0].reshape(output_shape)


def _cpu_phase1(packed: torch.Tensor, queries: torch.Tensor, eps: float):
    """Independent CPU equation for explicit adapter tests only."""
    if float(eps) != EPS:
        raise ValueError("Catswe CPU mock uses the frozen eps=2**-23")
    values = packed.float()
    keys = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + EPS)
    logits = torch.einsum("qd,nbtd->qnbt", queries.float(), keys)
    lse = torch.logsumexp(logits, dim=1)
    weights = torch.softmax(logits, dim=1)
    mixture = torch.einsum("qnbt,nbtd->qbtd", weights, values)
    return mixture.to(torch.bfloat16), lse.to(torch.float32)


def make_cpu_mock_backend() -> CatsweBackend:
    """Return an explicit CPU equation backend; native discovery never uses it."""
    return CatsweBackend(
        _cpu_phase1,
        Path("<catswe-cpu-mock>"),
        native=False,
        metadata={
            "name": f"{NAME}_cpu_mock",
            "status": "cpu_mock",
            "native_fallback": "not a native fallback; explicit test helper only",
            "eps": EPS,
            "gradient_contract": "all source values and query vectors",
        },
    )


def cpu_mock(values: Any, query: torch.Tensor, *, eps=EPS, scale=1.0):
    """Evaluate the explicit CPU mock without discovering or importing Triton."""
    return make_cpu_mock_backend()(values, query, eps=eps, scale=scale)
class Comparator:
    def __init__(self, backend: CatsweBackend | None, *, status: str, reason=None, root=None, revision=None, origin=None, dirty=None, hashes=None, transport=None):
        self.name, self.call, self.status = NAME, backend, status
        self.reason, self.vendor_root, self.vendor_revision = reason, root, revision
        self.vendor_origin = origin
        self.vendor_dirty, self.vendor_file_hashes, self.transport, self.kind = dirty, hashes or {}, transport or {}, "triton"
    @property
    def available(self): return self.status == "available" and self.call is not None
    def applicable(self, values, query):
        if not self.available: return False, self.reason or "Catswe comparator is unavailable"
        try: self.call._check(values, query)
        except (TypeError, ValueError, RuntimeError) as exc: return False, str(exc)
        return True, None
    def timing_eligible(self, values, query=None):
        if not self.available: return False
        return self.call.timing_eligible(values, query)
    def timing_applicable(self, values, query):
        if not self.available: return False, self.reason or "Catswe comparator is unavailable"
        okay, reason = self.applicable(values, query)
        return (okay, reason) if okay else (False, reason)
    def describe(self):
        metadata = {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "vendor_root": self.vendor_root,
            "vendor_revision": self.vendor_revision,
            "pinned_revision": PINNED_REVISION,
            "pinned_tree": PINNED_TREE,
            "vendor_clean": self.vendor_dirty is False,
            "vendor_dirty": self.vendor_dirty,
            "vendor_file_sha256": dict(self.vendor_file_hashes),
            "expected_vendor_file_sha256": dict(_VENDOR_SHA256),
            "expected_origin": REPOSITORY,
            "vendor_origin": self.vendor_origin,
            "origin": self.vendor_origin,
            "vendor_license": "Apache-2.0",
            "license": "Apache-2.0",
            "license_file": LICENSE,
            "license_sha256": LICENSE_SHA256,
            "repository": REPOSITORY,
            "entrypoints": "flash_attn_res.ops.phase_1 public operator only",
            "equation": "standard implicit R=D; FP32 score math; BF16 output",
            "requires_values_dtype": "torch.bfloat16",
            "full_schedule": "native phase 1, one query",
            "block_schedule": "not exposed; project Block remains public per-read",
            "supports_full": True,
            "supports_per_read_block": False,
            "supports_sliced": False,
            "accepts_source_list": False,
            "source_list_copy": "torch.stack inside adapter",
            "packed_copy": "contiguous inside adapter when required",
            "timing_predicate": TIMING_PREDICATE,
            "timing_predicate_name": TIMING_PREDICATE_NAME,
            "timing_predicate_enforced": True,
            "timing_exclusion_reason": TIMING_EXCLUSION_REASON,
            "max_program_elements": MAX_PROGRAM_ELEMENTS,
            "requires_power_of_two_width": True,
            "source_padding": "next_power_of_two",
            "native_program_shape": "[nextpow2(S), D]",
            "timing_boundary": "caller event around the phase-1 operator includes adapter stack/contiguous costs",
            "cuda_required": True,
            "native_fallback": "none; phase-1 operator only",
            "gradient_contract": "all source values and query vectors; independent FP32 gradient checks required",
            "qualification": "independent standard-operator phase-1 oracle required; this pass ran CPU/static checks only",
            "cpu_mock": "explicit make_cpu_mock_backend only; never native discovery",
            "module_origins": module_origins("flash_attn_res"),
        }
        metadata.update(self.transport)
        if self.reason: metadata["reason"] = self.reason
        metadata["adapter_sha256"] = _sha256(Path(__file__))
        return metadata
def _missing(reason, root=None, revision=None, dirty=None, hashes=None, transport=None, origin=None):
    return Comparator(None, status="missing", reason=reason, root=str(root) if root else None, revision=revision, origin=origin, dirty=dirty, hashes=hashes, transport=transport)
def discover_comparators(project_root=None, vendor_root=None) -> dict[str, Comparator]:
    root = find_vendor_root(project_root, vendor_root)
    if root is None: return {NAME: _missing("pinned Catswe checkout was not found")}
    reason, revision, dirty, hashes, transport = _integrity(root)
    if reason: return {NAME: _missing(reason, root, revision, dirty, hashes, transport)}
    origin = _origin(root)
    if transport:
        origin = transport.get("vendor_origin", origin)
    source = root / "src"
    if not _all_loaded_origins_ok(source):
        return {NAME: _missing("loaded Catswe modules originate outside pinned source; restart the process before rediscovery", root, revision, dirty, hashes, transport)}
    source_string = str(source.resolve())
    require_module_origins("flash_attn_res", source)
    sys.path[:] = [source_string] + [item for item in sys.path if item != source_string]
    importlib.invalidate_caches()
    try:
        phase1 = importlib.import_module("flash_attn_res.ops.phase_1")
        bad = [name for name in _REQUIRED_MODULES if not _origin_ok(sys.modules.get(name), source)]
        if bad: raise ImportError("loaded Catswe modules outside pinned source: " + ", ".join(bad))
        require_module_origins("flash_attn_res", source)
        backend = CatsweBackend(phase1.phase_1_batched_attention_triton_op, root)
        if not _all_loaded_origins_ok(source):
            raise ImportError("loaded Catswe modules outside pinned source after import")
    except Exception as exc:
        return {NAME: _missing(f"Catswe import failed: {type(exc).__name__}: {exc}", root, revision, dirty, hashes, transport)}
    return {NAME: Comparator(backend, status="available", root=str(root), revision=revision, origin=origin, dirty=dirty, hashes=hashes, transport=transport)}
def discover_comparator(project_root=None, vendor_root=None):
    return discover_comparators(project_root, vendor_root)[NAME]
def invoke_comparator(comparator: Comparator, values, query):
    if not comparator.available: raise RuntimeError(comparator.reason or "Catswe comparator is unavailable")
    applicable, reason = comparator.applicable(values, query)
    if not applicable: raise ValueError(reason or "Catswe comparator is not applicable")
    return comparator.call(values, query, eps=EPS, scale=1.0)
def vendor_metadata(project_root=None, vendor_root=None):
    root = find_vendor_root(project_root, vendor_root)
    if root is None: return {
        "path": None,
        "git_revision": None,
        "dirty": None,
        "pinned": False,
        "vendor_file_sha256": {},
        "expected_vendor_file_sha256": dict(_VENDOR_SHA256),
        "expected_origin": REPOSITORY,
        "license": "Apache-2.0",
        "license_file": LICENSE,
        "license_sha256": LICENSE_SHA256,
        "repository": REPOSITORY,
        "vendor_origin": None,
        "origin": None,
        "timing_predicate": TIMING_PREDICATE,
        "timing_predicate_name": TIMING_PREDICATE_NAME,
        "timing_predicate_enforced": True,
        "timing_exclusion_reason": TIMING_EXCLUSION_REASON,
        "timing_width_max": MAX_WIDTH,
        "max_program_elements": MAX_PROGRAM_ELEMENTS,
        "requires_power_of_two_width": True,
        "source_padding": "next_power_of_two",
        "native_program_shape": "[nextpow2(S), D]",
        "cuda_required": True,
        "native_fallback": "none; public phase operators only",
        "supports_per_read_block": False,
        "reason": "pinned Catswe checkout was not found",
    }
    reason, revision, dirty, hashes, transport = _integrity(root)
    tree = _tree(root)
    remote = (root / REMOTE_MANIFEST).is_file() or _remote_requested(root)
    origin = transport.get("vendor_origin") if transport else _origin(root)
    result = {
        "path": str(root),
        "git_revision": None if transport or remote else revision,
        "git_tree": None if transport or remote else tree,
        "dirty": dirty,
        "clean": dirty is False,
        "pinned_revision": PINNED_REVISION,
        "pinned_tree": PINNED_TREE,
        "pinned": reason is None,
        "vendor_file_sha256": hashes,
        "expected_vendor_file_sha256": dict(_VENDOR_SHA256),
        "hashes_match": hashes == _VENDOR_SHA256 and (not remote or reason is None),
        "vendor_license": "Apache-2.0",
        "license": "Apache-2.0",
        "license_file": LICENSE,
        "license_sha256": LICENSE_SHA256,
        "repository": REPOSITORY,
        "expected_origin": REPOSITORY,
        "vendor_origin": origin,
        "origin": origin,
        "timing_predicate": TIMING_PREDICATE,
        "timing_predicate_name": TIMING_PREDICATE_NAME,
        "timing_predicate_enforced": True,
        "timing_exclusion_reason": TIMING_EXCLUSION_REASON,
        "timing_width_max": MAX_WIDTH,
        "max_program_elements": MAX_PROGRAM_ELEMENTS,
        "requires_power_of_two_width": True,
        "source_padding": "next_power_of_two",
        "native_program_shape": "[nextpow2(S), D]",
        "cuda_required": True,
        "native_fallback": "none; phase-1 operator only",
        "supports_per_read_block": False,
    }
    if transport: result.update(transport)
    return result
__all__ = ["CatsweBackend", "Comparator", "EPS", "LICENSE", "LICENSE_SHA256", "MAX_PROGRAM_ELEMENTS", "MAX_SOURCES", "MAX_WIDTH", "MODEL_TIMING_SCOPE", "NAME", "PINNED_REVISION", "PINNED_TREE", "REMOTE_MANIFEST", "REPOSITORY", "TIMING_EXCLUSION_REASON", "TIMING_PREDICATE", "TIMING_PREDICATE_NAME", "cpu_mock", "discover", "discover_comparator", "discover_comparators", "find_vendor_root", "invoke_comparator", "make_cpu_mock_backend", "make_model_backend", "timing_eligible", "vendor_metadata"]; discover = discover_comparators
