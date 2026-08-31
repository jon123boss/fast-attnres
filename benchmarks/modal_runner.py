"""Modal transport; all measurements live in the independent validation/benchmark modules."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import modal

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("ATTNRES_SOURCE_DIR", str(PROJECT))).resolve()
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def _validated_sha256(value, name):
    digest = str(value or "")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA256 digest")
    return digest.lower()


_TRANSPORT_SHA256 = (
    _validated_sha256(os.environ["ATTNRES_TRANSPORT_SHA256"], "ATTNRES_TRANSPORT_SHA256")
    if "ATTNRES_TRANSPORT_SHA256" in os.environ else None
)
_LOCAL_TRANSPORT_SHA256 = (_TRANSPORT_SHA256
                         or hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
_FLA_DIR = os.environ.get("ATTNRES_FLA_DIR", "").strip()
FLA = Path(_FLA_DIR).expanduser().resolve() if _FLA_DIR else None
FLA_AVAILABLE = FLA is not None and (FLA / "fla").is_dir()
if FLA_AVAILABLE:
    from benchmarks.fla_checkout import fla_checkout_metadata

    _FLA_HOST_PREFLIGHT = fla_checkout_metadata(PROJECT, FLA)
else:
    _FLA_HOST_PREFLIGHT = None
_BASELINE_DIR = os.environ.get("ATTNRES_FROZEN_BASELINE_DIR", "").strip()
BASELINE = Path(_BASELINE_DIR).expanduser().resolve() if _BASELINE_DIR else None
if BASELINE is not None and not (BASELINE / "src/attnres/__init__.py").is_file():
    raise ValueError("frozen baseline must contain src/attnres/__init__.py")
TORCH_VERSION = os.environ.get("ATTNRES_TORCH_VERSION", "2.11.0")
TRITON_VERSION = os.environ.get("ATTNRES_TRITON_VERSION", "3.6.0")
if TORCH_VERSION != "2.11.0" or TRITON_VERSION != "3.6.0":
    raise ValueError("release evaluation requires Torch 2.11.0 and Triton 3.6.0")
_CACHE_MOUNT = "/workspace/cache"
_CACHE_VOLUME_NAME = os.environ.get("ATTNRES_CACHE_VOLUME", "").strip() or None
_CACHE_VOLUME = (modal.Volume.from_name(_CACHE_VOLUME_NAME, create_if_missing=True)
                 if _CACHE_VOLUME_NAME else None)


def _safe_component(value):
    """Keep runtime identities usable as one stable cache-path component."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def _validate_runtime(torch_module, triton_module):
    """Return measured runtime versions or raise before any model work."""

    actual_torch = str(getattr(torch_module, "__version__", ""))
    actual_triton = getattr(triton_module, "__version__", None)
    if not isinstance(actual_triton, str) or not actual_triton:
        raise RuntimeError(
            f"Triton version is unavailable; expected {TRITON_VERSION}"
        )
    actual_torch_base = actual_torch.split("+", 1)[0]
    if actual_torch_base != TORCH_VERSION or actual_triton != TRITON_VERSION:
        raise RuntimeError(
            "runtime version mismatch: "
            f"expected torch {TORCH_VERSION}/triton {TRITON_VERSION}, "
            f"got torch {actual_torch or '<unknown>'}/triton {actual_triton}"
        )
    return {
        "status": "verified",
        "expected": {"torch": TORCH_VERSION, "triton": TRITON_VERSION},
        "actual": {"torch": actual_torch, "triton": actual_triton},
    }


def _distribution_version(name, fallback):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _gpu_architecture(expected):
    if expected == "H100!":
        return "sm90"
    if expected == "B200":
        return "sm100"
    raise ValueError(f"unsupported GPU selector: {expected}")


def _source_fingerprint(source_root=Path("/workspace/project"),
                        validation_root=Path("/workspace/validation"),
                        fla_root=Path("/workspace/fla/fla"),
                        baseline_root=Path("/workspace/baseline/src/attnres"),
                        transport_sha256=None):
    """Hash all code and validation inputs that can affect a compiled graph."""
    transport_sha256 = _validated_sha256(
        transport_sha256 if transport_sha256 is not None
        else os.environ.get("ATTNRES_TRANSPORT_SHA256"),
        "ATTNRES_TRANSPORT_SHA256",
    )
    files = []

    def add_tree(root, pattern, prefix):
        if not root.is_dir():
            raise FileNotFoundError(f"fingerprint root is missing: {root}")
        for path in root.rglob(pattern):
            if path.is_file():
                files.append((f"{prefix}/{path.relative_to(root)}", path))

    add_tree(source_root / "src", "*.py", "src")
    add_tree(source_root / "benchmarks", "*.py", "benchmarks")
    add_tree(validation_root, "*.py", "validation")
    for name in ("frozen.json", "protocol.json"):
        path = validation_root / name
        if not path.is_file():
            raise FileNotFoundError(f"fingerprint input is missing: {path}")
        files.append((f"validation/{name}", path))
    # FLA is optional, but if mounted its Python source also affects generated
    # kernels. A stable marker keeps mounted and unmounted runs separate.
    if fla_root.is_dir():
        add_tree(fla_root, "*.py", "fla")
    else:
        files.append(("fla/<missing>", None))
    if baseline_root.is_dir():
        # This is the mount path used below when an external frozen baseline is enabled.
        add_tree(baseline_root, "*.py", "baseline")
    else:
        files.append(("baseline/<missing>", None))
    # Modal may relocate this module. The local entrypoint supplies this digest
    # through the image environment, so relocation cannot silently drop it.
    files.append(("transport/modal_runner.py", transport_sha256.encode("ascii")))

    digest = hashlib.sha256()
    for label, path in sorted(files):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        data = (b"<missing>" if path is None
                else path if isinstance(path, bytes) else path.read_bytes())
        digest.update(data)
        digest.update(b"\0")
    return {"algorithm": "sha256", "digest": digest.hexdigest(),
            "file_count": len(files), "transport_sha256": transport_sha256}


def _fingerprint_digest(source_fingerprint):
    if isinstance(source_fingerprint, dict):
        source_fingerprint = source_fingerprint.get("digest")
    return _validated_sha256(source_fingerprint, "source fingerprint")


def _cache_namespace(expected, torch_version=None, triton_version=None,
                    source_fingerprint=None):
    torch_version = torch_version or _distribution_version("torch", TORCH_VERSION)
    triton_version = triton_version or _distribution_version("triton", TRITON_VERSION)
    return "gpu-{gpu}-torch-{torch}-triton-{triton}".format(
        gpu=_safe_component(_gpu_architecture(expected)),
        torch=_safe_component(torch_version),
        triton=_safe_component(triton_version),
    ) + "-source-" + _fingerprint_digest(source_fingerprint)


def _cache_bytes(paths):
    total = 0
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                # A concurrently evicted cache entry does not invalidate the run.
                continue
    return total


def _prepare_cache(expected, source_fingerprint=None):
    """Configure compiler caches before importing Torch, when a volume is enabled."""
    if source_fingerprint is None:
        source_fingerprint = _source_fingerprint()
    if not isinstance(source_fingerprint, dict) or "transport_sha256" not in source_fingerprint:
        raise ValueError("source fingerprint metadata must include transport_sha256")
    fingerprint_digest = _fingerprint_digest(source_fingerprint)
    transport_sha256 = _validated_sha256(
        source_fingerprint["transport_sha256"], "transport fingerprint"
    )
    if _CACHE_VOLUME is None:
        return {"enabled": False, "volume": None, "namespace": None,
                "source_fingerprint": fingerprint_digest,
                "transport_sha256": transport_sha256,
                "bytes_before": 0, "bytes_after": 0, "bytes": 0}

    namespace = _cache_namespace(expected, source_fingerprint=source_fingerprint)
    namespace_root = Path(_CACHE_MOUNT) / namespace
    directories = {
        "torchinductor": namespace_root / "torchinductor",
        "triton": namespace_root / "triton",
    }
    # These variables must be set before importing torch: compiler modules read
    # them during import or on their first compilation.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(directories["torchinductor"])
    os.environ["TRITON_CACHE_DIR"] = str(directories["triton"])
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    bytes_before = _cache_bytes(directories.values())
    return {"enabled": True, "volume": _CACHE_VOLUME_NAME, "namespace": namespace,
            "source_fingerprint": fingerprint_digest,
            "transport_sha256": transport_sha256,
            "directories": {name: str(path) for name, path in directories.items()},
            "bytes_before": bytes_before, "bytes_after": bytes_before,
            "bytes": bytes_before}


def _align_cache_runtime(cache, expected, torch_version, triton_version):
    if not cache["enabled"]:
        return cache
    runtime_namespace = _cache_namespace(
        expected, torch_version, triton_version,
        source_fingerprint=cache["source_fingerprint"],
    )
    if runtime_namespace != cache["namespace"]:
        # Distribution metadata normally matches module versions. If a local
        # build adds a suffix, move to the runtime-specific namespace before
        # any evaluator or compiler code can run.
        cache["preimport_namespace"] = cache["namespace"]
        cache["namespace"] = runtime_namespace
        namespace_root = Path(_CACHE_MOUNT) / runtime_namespace
        directories = {
            "torchinductor": namespace_root / "torchinductor",
            "triton": namespace_root / "triton",
        }
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(directories["torchinductor"])
        os.environ["TRITON_CACHE_DIR"] = str(directories["triton"])
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        cache["directories"] = {name: str(path) for name, path in directories.items()}
    cache["runtime_namespace"] = runtime_namespace
    cache["runtime"] = {"torch": str(torch_version), "triton": str(triton_version)}
    cache["bytes_before"] = _cache_bytes(Path(path) for path in cache["directories"].values())
    cache["bytes_after"] = cache["bytes_before"]
    cache["bytes"] = cache["bytes_before"]
    return cache


def _commit_cache(result):
    """Persist compiler caches after the measured transport has finished."""
    cache = result.get("cache")
    if not cache or not cache.get("enabled") or _CACHE_VOLUME is None:
        return result
    paths = [Path(path) for path in cache.get("directories", {}).values()]
    cache["bytes_after"] = _cache_bytes(paths)
    cache["bytes"] = cache["bytes_after"]
    try:
        _CACHE_VOLUME.commit()
    except Exception as exc:
        cache["commit_status"] = "failed"
        cache["commit_error"] = f"{type(exc).__name__}: {exc}"
    else:
        cache["commit_status"] = "committed"
    return result


def _fla_source_metadata():
    vendor = Path("/workspace/fla/fla")
    requested = os.environ.get("ATTNRES_FLA_REQUESTED", "0") == "1"
    base = {
        "requested": requested,
        "mount_path": str(vendor) if vendor.is_dir() else None,
        "revision": os.environ.get("ATTNRES_FLA_REVISION", "unknown"),
        "git_dirty": os.environ.get("ATTNRES_FLA_DIRTY", "unknown"),
    }
    if not vendor.is_dir():
        return {**base, "status": "missing", "mount_available": False,
                "reason": ("ATTNRES_FLA_DIR is unset" if not requested
                           else "configured directory has no fla/ subdirectory"),
                "package_sha256": None}
    vendor_hash = hashlib.sha256()
    for path in sorted(vendor.rglob("*.py")):
        vendor_hash.update(str(path.relative_to(vendor)).encode())
        vendor_hash.update(path.read_bytes())
    return {**base, "status": "available", "mount_available": True,
            "package_sha256": vendor_hash.hexdigest()}


_IMAGE_ENV = {
    "PYTHONPATH": "/workspace:/workspace/project/src:/workspace/project",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "4",
    "ATTNRES_TORCH_VERSION": TORCH_VERSION,
    "ATTNRES_CACHE_VOLUME": _CACHE_VOLUME_NAME or "",
    "ATTNRES_TRITON_VERSION": TRITON_VERSION,
    "ATTNRES_TRANSPORT_SHA256": _TRANSPORT_SHA256 or _LOCAL_TRANSPORT_SHA256,
    "ATTNRES_FLA_REQUESTED": "1" if _FLA_DIR else "0",
    "ATTNRES_FLA_AVAILABLE": "1" if FLA_AVAILABLE else "0",
    "ATTNRES_FLA_DIR": "/workspace/fla" if FLA_AVAILABLE else "",
    "ATTNRES_FLA_REVISION": (
        str(_FLA_HOST_PREFLIGHT["revision"]) if _FLA_HOST_PREFLIGHT else "unknown"
    ),
    "ATTNRES_FLA_DIRTY": (
        "true" if _FLA_HOST_PREFLIGHT and _FLA_HOST_PREFLIGHT["git_dirty"] else "false"
    ),
    "ATTNRES_FLA_HOST_PREFLIGHT": (
        json.dumps(_FLA_HOST_PREFLIGHT, sort_keys=True) if _FLA_HOST_PREFLIGHT else ""
    ),
}
if FLA_AVAILABLE:
    _IMAGE_ENV["PYTHONPATH"] += ":/workspace/fla"
    _IMAGE_ENV["FLA_ROOT"] = "/workspace/fla"
if BASELINE is not None:
    _IMAGE_ENV["ATTNRES_FROZEN_BASELINE_DIR"] = "/workspace/baseline"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install(
        f"triton=={TRITON_VERSION}",
        "numpy==2.2.6",
        "pytest==8.4.2",
        "einops==0.8.1",
        "packaging==25.0",
    )
    .env(_IMAGE_ENV)
    .add_local_dir(SOURCE, "/workspace/project", ignore=[".git", ".DS_Store", "__pycache__", "*.pyc", "results", "dist", "build"])
    .add_local_dir(PROJECT / "validation", "/workspace/validation")
)
if FLA_AVAILABLE:
    image = image.add_local_dir(FLA / "fla", "/workspace/fla/fla", ignore=["__pycache__", "*.pyc"])
if BASELINE is not None:
    image = image.add_local_dir(BASELINE / "src/attnres", "/workspace/baseline/src/attnres",
                                ignore=["__pycache__", "*.pyc"])
app = modal.App("attnres-kernels-validation", image=image)


def _check_groups(function, config):
    if "groups" not in config:
        return function(config)
    groups = [function(group) for group in config["groups"]]
    return {"passed": sum(g["passed"] for g in groups),
            "failed": sum(g["failed"] for g in groups), "groups": groups}


def _run(payload, expected):
    started = time.time()
    source_fingerprint = None
    cache = None
    result = {
        "requested_gpu": expected,
        "task": payload["task"],
        "status": "running",
        "cache": None,
        "source_fingerprint": None,
        "runtime": {
            "status": "not_checked",
            "expected": {"torch": TORCH_VERSION, "triton": TRITON_VERSION},
            "actual": None,
        },
        "fla_checkout": {"status": "not_required"},
        "fla_source": _fla_source_metadata(),
    }
    try:
        source_fingerprint = _source_fingerprint()
        cache = _prepare_cache(expected, source_fingerprint)
        result["source_fingerprint"] = source_fingerprint
        result["cache"] = cache
        import torch
        result["software"] = {
            "python": sys.version,
            "torch": str(getattr(torch, "__version__", "<unknown>")),
            "triton": None,
            "cuda": getattr(torch.version, "cuda", None),
        }
        try:
            import triton
        except Exception as exc:
            result["runtime"] = {
                "status": "failed",
                "expected": {"torch": TORCH_VERSION, "triton": TRITON_VERSION},
                "actual": {
                    "torch": result["software"]["torch"],
                    "triton": None,
                },
                "error": {
                    "type": type(exc).__name__,
                    "message": f"Triton import failed: {exc}",
                },
            }
            raise RuntimeError(
                f"required Triton {TRITON_VERSION} is unavailable: {exc}"
            ) from exc
        result["software"]["triton"] = getattr(triton, "__version__", None)
        try:
            runtime = _validate_runtime(torch, triton)
        except Exception as exc:
            actual = {
                "torch": result["software"]["torch"],
                "triton": result["software"]["triton"],
            }
            result["runtime"] = {
                "status": "failed",
                "expected": {"torch": TORCH_VERSION, "triton": TRITON_VERSION},
                "actual": actual,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            raise
        result["runtime"] = runtime
        cache = _align_cache_runtime(cache, expected, torch.__version__, triton.__version__)
        result["cache"] = cache

        from benchmarks.fla_checkout import (
            validate_release_fla_config,
            verify_mounted_fla_checkout,
        )

        benchmark_config = payload.get("config", {})
        release_fla = (
            validate_release_fla_config(benchmark_config)
            if isinstance(benchmark_config, dict) and "production_ladder" in benchmark_config
            else None
        )
        verification = {"status": "not_required"}
        if release_fla is not None:
            try:
                host_preflight = json.loads(
                    os.environ.get("ATTNRES_FLA_HOST_PREFLIGHT", "")
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError("FLA host preflight metadata is invalid JSON") from exc
            verification = verify_mounted_fla_checkout(
                release_fla["checkout"],
                "/workspace/fla",
                host_preflight,
            )
            verification["anchor"] = release_fla["anchor"]
        result["fla_checkout"] = verification
        if verification["status"] == "failed":
            raise RuntimeError(
                "pinned FLA checkout verification failed: "
                + json.dumps(verification.get("error", {}), sort_keys=True)
            )

        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        props = torch.cuda.get_device_properties(0)
        actual = props.name
        capability = torch.cuda.get_device_capability(0)
        expected_cap = (9, 0) if expected == "H100!" else (10, 0)
        if ("H100" if expected == "H100!" else "B200") not in actual or capability != expected_cap:
            raise RuntimeError(f"hardware mismatch: {expected} -> {actual} SM{capability}")
        result["hardware"] = {"name": actual, "capability": list(capability),
                              "total_memory": props.total_memory,
                              "nvidia_smi": subprocess.check_output(
                                  ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                   "--format=csv,noheader"], text=True).strip()}
        result["software"]["cuda"] = torch.version.cuda
        source = Path("/workspace/project")
        manifest = json.loads(Path("/workspace/validation/frozen.json").read_text())
        checked_hashes = {}
        for name, digest in manifest.items():
            if name in {"tests/test_offsets.py", "tests/test_cuda.py"} and payload["task"] != "pytest":
                continue  # This final-package gate is not used by candidate checks.
            if name.startswith("benchmarks/") and payload["task"] not in {"training", "training_graph", "suite", "source_profile"}:
                continue
            path = Path("/workspace") / name if name.startswith("validation/") else source / name
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != digest:
                raise RuntimeError(f"frozen evaluator changed: {name}")
            checked_hashes[name] = digest
        result["frozen_hashes"] = checked_hashes
        result["source_hashes"] = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(list((source / "src").rglob("*.py"))
                            + list((source / "benchmarks").rglob("*.py")))
        }
        if payload["task"] == "preflight":
            from triton.experimental import gluon
            assert hasattr(torch.library, "triton_op")
            x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            fn = torch.compile(lambda v: v.float().square().sum(), fullgraph=True)
            fn(x).backward()
            torch.cuda.synchronize()
            torch.testing.assert_close(x.grad, 2 * x.detach())
            result["checks"] = {"compiled_bf16_autograd": True, "gluon_import": gluon is not None}
        elif payload["task"] == "checks":
            from validation.gpu_checks import run_checks
            result["checks"] = _check_groups(run_checks, payload.get("config", {}))
            if result["checks"]["failed"]:
                result["status"] = "failed"
        elif payload["task"] == "block_checks":
            from validation.block_checks import run_block_checks
            result["checks"] = _check_groups(run_block_checks, payload.get("config", {}))
            if result["checks"]["failed"]:
                result["status"] = "failed"
        elif payload["task"] == "pytest":
            command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            command += payload.get("config", {}).get("paths", ["tests"])
            try:
                done = subprocess.run(command, cwd=source, capture_output=True, text=True, timeout=780)
            except subprocess.TimeoutExpired as exc:
                def partial_text(value):
                    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
                result["pytest"] = {
                    "returncode": 124, "timed_out": True,
                    "stdout": partial_text(exc.stdout), "stderr": partial_text(exc.stderr),
                }
                raise
            result["pytest"] = {"returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr}
            if done.returncode:
                result["status"] = "failed"
        elif payload["task"] == "training":
            from validation.training_checks import run_training_checks
            result["checks"] = run_training_checks(payload.get("config", {}))
            if result["checks"]["failed"]:
                result["status"] = "failed"
        elif payload["task"] == "training_graph":
            from validation.training_graph_checks import run_graph_checks
            result["checks"] = run_graph_checks(payload.get("config", {}))
            if result["checks"]["failed"]:
                result["status"] = "failed"
        elif payload["task"] == "source_profile":
            from benchmarks.source_profile import run_source_profile
            result["profile"] = run_source_profile(payload.get("config", {}))
            if result["profile"]["status"] != "complete":
                result["status"] = "failed"
        elif payload["task"] == "suite":
            from benchmarks.run import run_suite
            result["measurements"] = run_suite(payload.get("config", {}))
            if result["measurements"]["status"] == "failed":
                result["status"] = "failed"
        else:
            raise ValueError("unknown task")
        if result["status"] == "running":
            result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.time() - started
    return json.loads(json.dumps(result))


def _run_and_commit(payload, expected):
    result = _run(payload, expected)
    # Volume persistence is deliberately outside _run's elapsed measurement.
    return _commit_cache(result)


_FUNCTION_OPTIONS = {
    "cpu": 4, "memory": 32768, "timeout": 900,
    "max_containers": 1, "min_containers": 0, "buffer_containers": 0,
    "scaledown_window": 2, "retries": 0,
}
if _CACHE_VOLUME is not None:
    _FUNCTION_OPTIONS["volumes"] = {_CACHE_MOUNT: _CACHE_VOLUME}


@app.function(gpu="H100!", **_FUNCTION_OPTIONS)
def h100(payload):
    return _run_and_commit(payload, "H100!")


@app.function(gpu="B200", **_FUNCTION_OPTIONS)
def b200(payload):
    return _run_and_commit(payload, "B200")


@app.local_entrypoint()
def main(gpu: str = "H100!", task: str = "preflight", config: str = "{}", output: str = ""):
    import concurrent.futures
    payload = {"task": task, "config": json.loads(config)}
    functions = [h100, b200] if gpu == "both" else [{"H100!": h100, "B200": b200}[gpu]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(functions)) as pool:
        results = list(pool.map(lambda f: f.remote(payload), functions))
    report = {"source": str(SOURCE), "results": results}
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(target)
    print(json.dumps({"output": output, "results": [
        {k: r.get(k) for k in ("requested_gpu", "status", "elapsed_seconds", "error")} for r in results
    ]}))
