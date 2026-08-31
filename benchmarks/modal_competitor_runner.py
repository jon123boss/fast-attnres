"""Modal runner for the sealed matched comparator protocol.

The release transport in :mod:`benchmarks.modal_runner` is pinned to the
Torch 2.11/Triton 3.6 environment used by the release evidence.  Matched
competitor measurements have a different sealed runtime, so they use this
separate worker.  Runtime versions and the selected GPU are checked before
the capability registry imports any optional adapter module.

The worker delegates cell planning, independent output/value-gradient/query-
gradient qualification, capability rejection, timing, and JSON result
materialization to :mod:`benchmarks.comparator_runner`.  Its vendor roots are
always explicit container paths.  A missing mount therefore produces a
visible missing route; it cannot fall through to an ambient checkout.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "configs" / "matched_competitor_benchmark.json"
REPORT_SCHEMA = "attnres.matched_competitor_benchmark.report.v1"
TORCH_VERSION = "2.13.0"
TORCH_CUDA_VERSION = "13.0"
TRITON_VERSION = "3.7.1"
_WORKER_STATUSES = frozenset(
    {"complete", "incomplete", "planned", "not_requested", "failed"}
)

_EXTERNAL_ROOTS = (
    # host environment variable, protocol family, immutable container root
    ("ATTNRES_FLA_DIR", "fla", "/workspace/vendors/fla"),
    ("ATTNRES_LIGER_DIR", "liger", "/workspace/vendors/liger"),
    ("CATSWE_ROOT", "catswe", "/workspace/vendors/catswe"),
    ("HYDRA_ROOT", "manish", "/workspace/vendors/hydra"),
)

_VENDOR_ORIGINS = {
    "fla": "https://github.com/fla-org/flash-linear-attention.git",
    "liger": "https://github.com/linkedin/Liger-Kernel.git",
    "catswe": "https://github.com/catswe/flash-attention-residuals.git",
    "manish": "https://github.com/manishklach/attnres-kernel-lab.git",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exception(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _worker_failure(selector: str, exc: BaseException) -> dict[str, Any]:
    """Return a JSON-safe record for a worker that never returned a report.

    Modal's timeout exception type is not stable across client versions, so
    timeout status is inferred from both the exception class and its message in
    addition to recognizing the built-in timeout exception.  This record is
    kept separately from successful worker reports so a partial run remains
    auditable.
    """

    exception = _exception(exc)
    text = f"{exception['type']} {exception['message']}".lower()
    timed_out = isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text
    return {
        "selector": selector,
        "status": "failed",
        "error": {**exception, "timeout": timed_out},
        "timeout": timed_out,
    }


def _require_output_path(output: str) -> str:
    """Require a durable destination before any remote worker is submitted."""

    if not isinstance(output, str) or not output.strip():
        raise ValueError(
            "an output path is required for matched comparator runs; "
            "partial worker failures must be persisted"
        )
    return output


def _validated_sha256(value: Any, name: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA256 digest")
    return digest.lower()


_LOCAL_TRANSPORT_SHA256 = _sha256_file(Path(__file__).resolve())
_TRANSPORT_SHA256 = _validated_sha256(
    os.environ.get("ATTNRES_COMPETITOR_TRANSPORT_SHA256", _LOCAL_TRANSPORT_SHA256),
    "ATTNRES_COMPETITOR_TRANSPORT_SHA256",
)


def _container_vendor_roots() -> dict[str, str]:
    """Return explicit roots for every optional family in the worker image."""

    return {family: target for _env, family, target in _EXTERNAL_ROOTS}


def _checkout_root(path: Path) -> Path:
    """Normalize a source-subdirectory setting to its checkout root."""

    candidate = path.expanduser().resolve()
    if candidate.name == "src" and (
        (candidate / "liger_kernel").is_dir()
        or (candidate / "flash_attn_res").is_dir()
        or (candidate / "attnres_kernel").is_dir()
    ):
        return candidate.parent
    if candidate.name == "fla" and (candidate / "ops" / "attnres").is_dir():
        return candidate.parent
    return candidate


def _source_fingerprint(root: Path = Path("/workspace/project")) -> dict[str, Any]:
    """Hash project and protocol inputs that can affect comparator execution."""

    files: dict[str, str] = {}
    for tree_name in ("src", "benchmarks", "validation"):
        tree = root / tree_name
        if not tree.is_dir():
            raise FileNotFoundError(f"source fingerprint root is missing: {tree}")
        for path in sorted(tree.rglob("*.py")):
            if path.is_file():
                files[str(path.relative_to(root))] = _sha256_file(path)
    for relative in (
        "configs/matched_competitor_benchmark.json",
        "configs/matched_competitor_benchmark.schema.json",
    ):
        config = root / relative
        if not config.is_file():
            raise FileNotFoundError(f"matched protocol input is missing: {config}")
        files[relative] = _sha256_file(config)
    payload = {
        "algorithm": "sha256",
        "files": files,
        "transport_sha256": _TRANSPORT_SHA256,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**payload, "digest": digest, "file_count": len(files)}


def _validate_runtime(torch_module: Any, triton_module: Any) -> dict[str, Any]:
    """Require the sealed runtime before optional adapter discovery/imports."""

    actual_torch = str(getattr(torch_module, "__version__", ""))
    actual_torch_base = actual_torch.split("+", 1)[0]
    actual_cuda = str(getattr(getattr(torch_module, "version", None), "cuda", ""))
    actual_triton = str(getattr(triton_module, "__version__", ""))
    if (
        actual_torch_base != TORCH_VERSION
        or actual_cuda != TORCH_CUDA_VERSION
        or actual_triton != TRITON_VERSION
    ):
        raise RuntimeError(
            "matched comparator runtime mismatch: "
            f"expected torch {TORCH_VERSION}+cu130/cuda {TORCH_CUDA_VERSION} "
            f"and Triton {TRITON_VERSION}, got torch {actual_torch or '<unknown>'}/"
            f"cuda {actual_cuda or '<unknown>'}/triton {actual_triton or '<unknown>'}"
        )
    return {
        "status": "verified",
        "expected": {
            "torch": TORCH_VERSION,
            "torch_cuda": TORCH_CUDA_VERSION,
            "triton": TRITON_VERSION,
        },
        "actual": {
            "torch": actual_torch,
            "torch_cuda": actual_cuda,
            "triton": actual_triton,
            "torch_module": str(getattr(torch_module, "__file__", "<unknown>")),
            "triton_module": str(getattr(triton_module, "__file__", "<unknown>")),
        },
    }


def _hardware(torch_module: Any, selector: str) -> dict[str, Any]:
    """Check the Modal function's actual device against the sealed selector."""

    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise RuntimeError("matched comparator execution requires exactly one visible CUDA device")
    properties = torch_module.cuda.get_device_properties(0)
    name = str(properties.name)
    capability = list(torch_module.cuda.get_device_capability(0))
    expected_name, expected_capability = (
        ("H100", [9, 0]) if selector == "H100!" else ("B200", [10, 0])
    )
    if selector not in {"H100!", "B200"}:
        raise ValueError(f"unsupported GPU selector {selector!r}")
    if expected_name.upper() not in name.upper() or capability != expected_capability:
        raise RuntimeError(
            f"hardware mismatch for {selector}: expected {expected_name} SM{expected_capability}, "
            f"got {name!r} SM{capability}"
        )
    return {
        "selector": selector,
        "name": name,
        "capability": capability,
        "total_memory": int(properties.total_memory),
    }


def _run(payload: Mapping[str, Any], selector: str) -> dict[str, Any]:
    """Run one matched registry job and return a JSON-safe report."""

    started = time.time()
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "running",
        "task": payload.get("task", "matched_registry"),
        "requested_gpu": selector,
        "protocol_runtime": {
            "torch": TORCH_VERSION,
            "torch_cuda": TORCH_CUDA_VERSION,
            "triton": TRITON_VERSION,
            "dependencies": {
                # FLA's pinned fused adapter imports einops at discovery time;
                # keep this exact rather than inheriting a floating image dep.
                "einops": "0.8.1",
            },
        },
        "runtime": {"status": "not_checked"},
        "hardware": None,
        "source_fingerprint": None,
        "vendor_roots": _container_vendor_roots(),
        "measurements": None,
    }
    try:
        if result["task"] != "matched_registry":
            raise ValueError(f"unknown matched comparator task {result['task']!r}")
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("payload config must be an object")

        # Protocol validation is CPU-only.  The runtime imports and version
        # check happen before comparator_runner, which in turn imports no
        # optional vendor package until explicit discovery is requested.
        from benchmarks.competitor_protocol import (
            config_digest,
            load_config,
            validate_config,
        )

        config = validate_config(config)
        supplied_digest = payload.get("config_digest")
        if supplied_digest is not None and str(supplied_digest) != config_digest(config):
            raise ValueError("matched protocol config digest does not match payload")
        sealed_config = load_config(
            Path("/workspace/project/configs/matched_competitor_benchmark.json")
        )
        if config_digest(config) != config_digest(sealed_config):
            raise ValueError("payload config differs from the checked-in sealed matched protocol")
        import torch
        import triton

        result["runtime"] = _validate_runtime(torch, triton)
        result["hardware"] = _hardware(torch, selector)
        result["source_fingerprint"] = _source_fingerprint(Path("/workspace/project"))

        # Deliberately import the dispatch layer only after the sealed runtime
        # and hardware checks above.  Adapter modules remain lazy inside it.
        from benchmarks.comparator_runner import run_matched_registry

        measurements = run_matched_registry(
            config,
            project_root=Path("/workspace/project"),
            vendor_roots=_container_vendor_roots(),
            device=torch.device("cuda"),
            execute_operator=bool(payload.get("execute_operator", True)),
            scope=str(payload.get("scope", "smoke")),
            names=payload.get("names"),
            rounds=payload.get("rounds"),
            warmup=payload.get("warmup"),
            seed=payload.get("seed"),
            gpu=selector,
        )
        result["measurements"] = measurements
        execution_status = measurements.get("execution_status", measurements.get("status"))
        if execution_status in {"complete", "planned", "not_requested", "incomplete"}:
            result["status"] = execution_status
        else:
            result["status"] = "failed"
    except Exception as exc:
        result.update(
            status="failed",
            error=_exception(exc),
            traceback=traceback.format_exc(),
        )
    result["elapsed_seconds"] = time.time() - started
    return json.loads(json.dumps(result, allow_nan=False))


def _build_modal_image(modal_module: Any) -> Any:
    """Build this worker's separate Torch 2.13/Triton 3.7.1 image."""

    source = Path(os.environ.get("ATTNRES_SOURCE_DIR", str(PROJECT))).resolve()
    image_env = {
        "PYTHONPATH": "/workspace/project/src:/workspace/project",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ATTNRES_COMPETITOR_TRANSPORT_SHA256": _TRANSPORT_SHA256,
    }
    image = (
        modal_module.Image.debian_slim(python_version="3.11")
        # The adapters verify the transported checkout revision, tree,
        # cleanliness, and sole origin inside the worker.  Debian slim does
        # not include the Git executable needed for that fail-closed check.
        .apt_install("git")
        .uv_pip_install(
            f"torch=={TORCH_VERSION}",
            index_url="https://download.pytorch.org/whl/cu130",
        )
        .uv_pip_install(
            f"triton=={TRITON_VERSION}",
            "einops==0.8.1",
            "numpy==2.2.6",
            "packaging==25.0",
        )
        .env(image_env)
    )
    for environment, _family, target in _EXTERNAL_ROOTS:
        configured = os.environ.get(environment, "").strip()
        if not configured:
            continue
        host_root = _checkout_root(Path(configured))
        if not host_root.is_dir():
            raise FileNotFoundError(f"configured comparator root is not a directory: {host_root}")
        _require_transport_provenance(host_root, family=_family)
        bundle = _create_vendor_bundle(host_root, family=_family)
        remote_bundle = f"/workspace/vendor-bundles/{_family}.bundle"
        origin = _VENDOR_ORIGINS[_family]
        image = (
            image.add_local_file(bundle, remote_bundle, copy=True)
            .run_commands(
                "git clone --quiet "
                f"{shlex.quote(remote_bundle)} {shlex.quote(target)}",
                "git -C "
                f"{shlex.quote(target)} remote set-url origin {shlex.quote(origin)}",
            )
        )
    # Runtime source is a startup mount and must remain the final image layer;
    # Modal does not permit build steps after a non-copied local mount.
    return image.add_local_dir(
        source,
        "/workspace/project",
        ignore=[".git", ".DS_Store", "__pycache__", "*.pyc", "results", "dist", "build"],
    )


def _require_transport_provenance(root: Path, *, family: str) -> None:
    """Require one clean standalone checkout at the pinned vendor origin."""

    marker = root / ".git"
    if marker.is_file():
        raise ValueError(
            f"{family} comparator root {root} is a Git worktree pointer; "
            "use a standalone pinned checkout"
        )
    if not marker.is_dir():
        raise ValueError(
            f"{family} comparator root {root} must contain a real .git directory"
        )
    top = _git_output(root, "rev-parse", "--show-toplevel")
    if Path(top).resolve() != root.resolve():
        raise ValueError(f"{family} comparator root must be the checkout top level")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError(f"{family} comparator checkout is dirty")
    origins = tuple(
        line
        for line in _git_output(root, "config", "--get-all", "remote.origin.url").splitlines()
        if line
    )
    if len(origins) != 1:
        raise ValueError(f"{family} comparator checkout needs exactly one origin URL")
    from benchmarks.vendor_identity import normalize_remote_origin

    if normalize_remote_origin(origins[0]) != normalize_remote_origin(
        _VENDOR_ORIGINS[family]
    ):
        raise ValueError(f"{family} comparator checkout origin does not match its pin")


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise ValueError(message)
    return completed.stdout.strip()


def _create_vendor_bundle(root: Path, *, family: str) -> Path:
    """Create a commit-only transport that preserves modes and symlinks.

    Modal directory mounts normalize some symlinks and can make an otherwise
    clean checkout appear dirty.  A Git bundle carries the committed object
    graph instead.  The worker clones it, assigns the pinned public origin,
    and the adapter then re-verifies revision, tree, cleanliness, hashes, and
    origin before import.
    """

    directory = Path(tempfile.mkdtemp(prefix=f"attnres-{family}-bundle-"))
    bundle = directory / f"{family}.bundle"
    completed = subprocess.run(
        ["git", "-C", str(root), "bundle", "create", str(bundle), "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not bundle.is_file() or bundle.stat().st_size < 1:
        message = completed.stderr.strip() or completed.stdout.strip() or "bundle is empty"
        raise ValueError(f"could not create {family} vendor bundle: {message}")
    return bundle


def _configured_host_roots(
    *,
    fla_root: str = "",
    liger_root: str = "",
    catswe_root: str = "",
    hydra_root: str = "",
) -> dict[str, str]:
    """Resolve CLI roots against the roots mounted into the image.

    Modal constructs the image before invoking its local entrypoint, so a root
    supplied as a flag cannot create a new mount at that point.  Requiring it
    to agree with the corresponding environment setting prevents a host path
    from being mistaken for a container path or changing the provenance
    contract.  Host paths are launch metadata only; the remote worker imports
    from fixed container roots.
    """

    requested = {
        "fla": fla_root,
        "liger": liger_root,
        "catswe": catswe_root,
        "manish": hydra_root,
    }
    environment_names = {
        family: environment for environment, family, _target in _EXTERNAL_ROOTS
    }
    resolved: dict[str, str] = {}
    for family, value in requested.items():
        configured = os.environ.get(environment_names[family], "").strip()
        supplied = str(value or "").strip()
        if supplied and configured:
            if _checkout_root(Path(supplied)) != _checkout_root(Path(configured)):
                raise ValueError(
                    f"{family} root flag does not match "
                    f"{environment_names[family]} configured before modal run"
                )
        elif supplied and not configured:
            raise ValueError(
                f"{family} root flag requires {environment_names[family]} "
                "to be set before modal run"
            )
        if configured:
            resolved[family] = str(_checkout_root(Path(configured)))
    return resolved


def _write_report(report: Mapping[str, Any], output: str) -> None:
    if not output:
        return
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    # A unique temporary file avoids clobbering another invocation's staging
    # file.  Flush and fsync before replace so a timeout or process failure
    # cannot leave a half-written JSON report at the final path.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _collect_worker_results(
    functions: Sequence[tuple[str, Any]], payload: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect Modal workers independently, preserving partial results.

    ``Future.result`` must be handled per future.  Calling it in a list
    comprehension makes the first timeout discard a worker that already
    completed and prevents the caller from materializing any report.
    """

    completed: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(functions)) as pool:
        pending: dict[concurrent.futures.Future[Any], str] = {}
        for selector, function in functions:
            try:
                pending[pool.submit(function.remote, payload)] = selector
            except Exception as exc:
                # Submission can fail before a Future exists; retain that
                # worker in the same structured failure channel.
                failures[selector] = _worker_failure(selector, exc)
        for future in concurrent.futures.as_completed(pending):
            selector = pending[future]
            try:
                result = future.result()
                if not isinstance(result, Mapping):
                    raise TypeError(
                        "worker returned "
                        f"{type(result).__name__}; expected a Mapping report"
                    )
                result = dict(result)
                if (
                    type(result.get("requested_gpu")) is not str
                    or result["requested_gpu"] != selector
                ):
                    raise ValueError(
                        "worker report requested_gpu does not match the dispatched selector"
                    )
                if (
                    type(result.get("status")) is not str
                    or result["status"] not in _WORKER_STATUSES
                ):
                    raise ValueError(
                        "worker report status is missing or outside the sealed status set"
                    )
                try:
                    json.dumps(result, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise TypeError("worker report is not JSON serializable") from exc
                completed[selector] = result
            except Exception as exc:
                failures[selector] = _worker_failure(selector, exc)

    # Keep report order tied to the requested selector order, even though
    # collection uses completion order to avoid one slow worker masking a
    # completed sibling.
    reports = [
        completed[selector]
        for selector, _function in functions
        if selector in completed
    ]
    worker_failures = [
        failures[selector]
        for selector, _function in functions
        if selector in failures
    ]
    return reports, worker_failures


def _run_worker_batch(
    functions: Sequence[tuple[str, Any]],
    payload: Mapping[str, Any],
    *,
    config_path: Path,
    config_digest: str,
    scope: str,
    configured_roots: Mapping[str, Any],
    output: str,
) -> dict[str, Any]:
    """Run, persist, and finally fail a multi-GPU batch.

    The exception is deliberately raised only after ``_write_report``.  This
    guarantees that a Modal timeout still leaves completed workers and a
    structured failed-worker record on disk, including when every worker
    fails.
    """

    _require_output_path(output)
    reports, worker_failures = _collect_worker_results(functions, payload)
    worker_return_failures = [
        report
        for report in reports
        if isinstance(report, Mapping) and report.get("status") == "failed"
    ]
    failed = bool(worker_failures or worker_return_failures)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "failed" if failed else "complete",
        "config_path": str(config_path),
        "config_digest": config_digest,
        "scope": scope,
        "host_vendor_roots": dict(configured_roots),
        "results": reports,
        "worker_failures": worker_failures,
    }
    _write_report(report, output)
    if failed:
        selectors = [
            str(item.get("selector", item.get("requested_gpu", "unknown")))
            for item in worker_failures
        ]
        selectors.extend(
            str(item.get("requested_gpu", "unknown"))
            for item in worker_return_failures
        )
        detail = ", ".join(selectors) or "unknown worker"
        destination = output or "<no output path requested>"
        raise RuntimeError(
            f"matched comparator worker failure ({detail}); report written to {destination}"
        )
    return report


# Modal discovers decorators at module scope.  Keeping the optional import and
# image construction inside this guarded block preserves CPU importability
# while avoiding the nested ``@app.function`` pattern rejected by Modal 1.5.
try:  # pragma: no cover - Modal is absent in CPU CI.
    _modal = importlib.import_module("modal")
except (ImportError, ModuleNotFoundError):  # pragma: no cover - normal CPU-only installation.
    _modal = None

if _modal is not None:  # pragma: no cover - exercised by Modal.
    image = _build_modal_image(_modal)
    app = _modal.App("attnres-matched-comparators", image=image)
    _MODAL_OPTIONS = {
        "cpu": 4,
        "memory": 32768,
        "timeout": 1800,
        "max_containers": 1,
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window": 2,
        "retries": 0,
    }

    @app.function(gpu="H100!", **_MODAL_OPTIONS)
    def h100(payload: Mapping[str, Any]) -> dict[str, Any]:
        return _run(payload, "H100!")

    @app.function(gpu="B200", **_MODAL_OPTIONS)
    def b200(payload: Mapping[str, Any]) -> dict[str, Any]:
        return _run(payload, "B200")

    @app.local_entrypoint()
    def main(
        gpu: str = "both",
        scope: str = "smoke",
        plan_only: bool = False,
        config: str = str(CONFIG_PATH),
        output: str = "",
        fla_root: str = "",
        liger_root: str = "",
        catswe_root: str = "",
        hydra_root: str = "",
    ) -> None:
        _require_output_path(output)
        from benchmarks.competitor_protocol import config_digest, load_config

        configured_roots = _configured_host_roots(
            fla_root=fla_root,
            liger_root=liger_root,
            catswe_root=catswe_root,
            hydra_root=hydra_root,
        )
        config_path = Path(config).expanduser().resolve()
        config_value = load_config(config_path)
        digest = config_digest(config_value)
        payload = {
            "task": "matched_registry",
            "config": config_value,
            "config_digest": digest,
            "scope": scope,
            "execute_operator": not plan_only,
            "host_vendor_roots": configured_roots,
        }
        if gpu == "both":
            functions = [("H100!", h100), ("B200", b200)]
        elif gpu in {"H100!", "B200"}:
            functions = [(gpu, {"H100!": h100, "B200": b200}[gpu])]
        else:
            raise ValueError("gpu must be H100!, B200, or both")
        report = _run_worker_batch(
            functions,
            payload,
            config_path=config_path,
            config_digest=digest,
            scope=scope,
            configured_roots=configured_roots,
            output=output,
        )
        reports = report["results"]
        print(
            json.dumps(
                {
                    "output": output,
                    "results": [
                        {
                            key: item.get(key)
                            for key in ("requested_gpu", "status", "elapsed_seconds", "error")
                        }
                        for item in reports
                    ],
                },
                sort_keys=True,
            )
        )
else:  # pragma: no cover - exercised only by CPU static tests.
    image = None
    app = None
    h100 = None
    b200 = None
    main = None


def _standalone_cli(argv: Sequence[str] | None = None) -> int:
    """Validate the matched protocol without importing Modal or GPU runtimes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--fla-root", default="")
    parser.add_argument("--liger-root", default="")
    parser.add_argument("--catswe-root", default="")
    parser.add_argument("--hydra-root", default="")
    args = parser.parse_args(argv)
    if not args.validate_config:
        parser.error("GPU measurement must be launched with `modal run benchmarks/modal_competitor_runner.py`")
    from benchmarks.competitor_protocol import config_digest, load_config

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    host_roots = {
        family: str(Path(value).expanduser().resolve())
        for family, value in {
            "fla": args.fla_root,
            "liger": args.liger_root,
            "catswe": args.catswe_root,
            "manish": args.hydra_root,
        }.items()
        if value
    }
    print(
        json.dumps(
            {
                "status": "valid",
                "config_path": str(config_path),
                "config_digest": config_digest(config),
                "host_vendor_roots": host_roots,
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI contract helper.
    raise SystemExit(_standalone_cli())


__all__ = [
    "CONFIG_PATH",
    "REPORT_SCHEMA",
    "TORCH_CUDA_VERSION",
    "TORCH_VERSION",
    "TRITON_VERSION",
    "_container_vendor_roots",
    "_run",
    "_source_fingerprint",
    "_standalone_cli",
    "_validate_runtime",
]
