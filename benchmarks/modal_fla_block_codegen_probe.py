#!/usr/bin/env python3
"""Modal transport for the selected FLA-derived Block codegen probe.

This module is a standalone, non-timing Modal entrypoint.  It reuses the
existing image, source fingerprint, architecture-aware cache namespace, and
optional cache Volume from benchmarks.modal_runner, then runs only
selected_fla_block_codegen_probe.run_probe.  It never imports or
calls the evaluator/timing runner.  GPU submission requires all four
arguments: --gpu, --hardware, --cache, and --output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Callable

try:  # Modal is optional for CPU/static repository checks.
    import modal as _modal
except Exception:  # pragma: no cover - exercised when Modal is unavailable.
    _modal = None

def _load_transport():
    """Load the sibling transport without importing benchmarks.__init__.

    Modal's lightweight local CLI environment intentionally has no Torch, while
    ``benchmarks.__init__`` imports the training model and therefore Torch.  The
    transport itself has no Torch import at module load time, so load that exact
    file directly for entrypoint discovery.
    """

    local_path = Path(__file__).resolve().with_name("modal_runner.py")
    remote_path = Path("/workspace/project/benchmarks/modal_runner.py")
    path = local_path if local_path.is_file() else remote_path
    spec = importlib.util.spec_from_file_location("_attnres_modal_transport", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Modal transport from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:  # Reuse the existing image and cache provenance when Modal is installed.
    _transport = _load_transport()
except Exception:  # pragma: no cover - CPU/static environments have no Modal.
    _transport = None


GPU_CHOICES = ("H100!", "B200")
HARDWARE_CHOICES = ("H100", "B200")
PROBE_HARDWARE = {"H100!": "H100", "B200": "B200"}
PROBE_MODULE = "benchmarks.selected_fla_block_codegen_probe"


def _normalize_gpu(value: str) -> str:
    normalized = str(value).strip()
    if normalized == "H100":
        normalized = "H100!"
    if normalized not in GPU_CHOICES:
        raise ValueError(f"gpu must be one of {GPU_CHOICES}, got {value!r}")
    return normalized


def _normalize_hardware(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in HARDWARE_CHOICES:
        raise ValueError(
            f"hardware must be one of {HARDWARE_CHOICES}, got {value!r}"
        )
    return normalized


def _validate_cli_args(
    gpu: str, hardware: str, cache: str, output: str
) -> tuple[str, str]:
    normalized = _normalize_gpu(gpu)
    normalized_hardware = _normalize_hardware(hardware)
    expected_hardware = PROBE_HARDWARE[normalized]
    if normalized_hardware != expected_hardware:
        raise ValueError(
            f"gpu {normalized!r} and hardware {normalized_hardware!r} disagree; "
            f"expected hardware {expected_hardware!r}"
        )
    if not str(cache).strip():
        raise ValueError("--cache is required and must be a nonempty path or name")
    if not str(output).strip():
        raise ValueError("--output is required and must be a nonempty local path")
    return normalized, normalized_hardware


def _empty_or_new(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"probe cache is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(
                f"probe cache must be new or empty; refusing to reuse {path}"
            )
    else:
        path.mkdir(parents=True)


def _resolve_probe_cache(requested: str, transport_cache: dict[str, Any]) -> Path:
    """Resolve an explicit cache path without reusing stale compiler entries."""

    requested_path = Path(str(requested).strip()).expanduser()
    if requested_path.is_absolute():
        root = requested_path.resolve()
    else:
        directories = transport_cache.get("directories", {})
        base = Path(
            str(directories.get("triton", "/tmp/attnres-codegen-probe"))
        ).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        root = (base / "codegen-probe" / requested_path).resolve()
        if base not in root.parents:
            raise ValueError("--cache must remain inside the transport cache namespace")
    _empty_or_new(root)
    return root


def _hardware_facts(torch_module: Any, expected: str) -> dict[str, Any]:
    capability = tuple(
        int(value) for value in torch_module.cuda.get_device_capability(0)
    )
    name = str(torch_module.cuda.get_device_name(0))
    expected_capability = (9, 0) if expected == "H100!" else (10, 0)
    expected_name = "H100" if expected == "H100!" else "B200"
    if (
        torch_module.cuda.device_count() != 1
        or capability != expected_capability
        or expected_name not in name
    ):
        raise RuntimeError(
            f"hardware mismatch: expected {expected} / {expected_capability}, "
            f"observed {name!r} / {capability!r}"
        )
    properties = torch_module.cuda.get_device_properties(0)
    return {
        "requested": expected,
        "name": name,
        "capability": list(capability),
        "total_memory": int(properties.total_memory),
    }


def _run_remote(payload: dict[str, Any], expected: str) -> dict[str, Any]:
    """Run the read-only six-case Block probe on one explicitly selected GPU."""

    if _transport is None:
        raise RuntimeError(
            "Modal transport is unavailable; install Modal or run the probe locally"
        )
    expected = _normalize_gpu(expected)
    expected_hardware = PROBE_HARDWARE[expected]
    payload_hardware = _normalize_hardware(payload.get("hardware", ""))
    if payload_hardware != expected_hardware:
        raise ValueError(
            f"remote hardware scope {payload_hardware!r} does not match "
            f"Modal GPU {expected!r} ({expected_hardware!r})"
        )
    source_fingerprint: dict[str, Any] | None = None
    cache: dict[str, Any] | None = None
    result: dict[str, Any] = {
        "status": "running",
        "probe": PROBE_MODULE,
        "requested_gpu": expected,
        "requested_hardware": payload_hardware,
        "requested_cache": str(payload["cache"]),
        "source_fingerprint": None,
        "cache": None,
    }
    try:
        source_fingerprint = _transport._source_fingerprint()
        cache = _transport._prepare_cache(expected, source_fingerprint)
        result["source_fingerprint"] = source_fingerprint
        result["cache"] = cache
        result["fla_source"] = _transport._fla_source_metadata()
        # Cache variables are configured by _prepare_cache before these imports.
        import torch
        import triton

        cache = _transport._align_cache_runtime(
            cache, expected, torch.__version__, triton.__version__
        )
        probe_cache = _resolve_probe_cache(str(payload["cache"]), cache)
        result["cache"] = cache
        result["probe_cache_root"] = str(probe_cache)
        result["hardware"] = _hardware_facts(torch, expected)
        result["software"] = {
            "python": os.sys.version,
            "torch": str(torch.__version__),
            "triton": str(triton.__version__),
            "cuda": str(torch.version.cuda),
        }

        from benchmarks import selected_fla_block_codegen_probe as probe

        report = probe.run_probe(
            cache_root=probe_cache,
            hardware=PROBE_HARDWARE[expected],
            source_counts=probe.SOURCE_COUNTS,
            ranks=probe.RANKS,
        )
        result["probe_report"] = report
        result["status"] = (
            "complete" if report.get("status") == "passed" else "failed"
        )
    except Exception as exc:
        result.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    finally:
        # This only records/commits compiler-cache bytes.  It does not alter
        # the selected source, evaluator, inputs, or any timing boundary.
        if cache is not None:
            result = _transport._commit_cache(result)
    return json.loads(json.dumps(result))


def _write_report(output: str, report: dict[str, Any]) -> Path:
    target = Path(str(output).strip()).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _main_impl(
    gpu: str,
    hardware: str,
    cache: str,
    output: str,
    remote: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> int:
    normalized, normalized_hardware = _validate_cli_args(
        gpu, hardware, cache, output
    )
    if remote is None:
        raise RuntimeError("Modal is required to submit this entrypoint")
    try:
        report = remote(
            {
                "cache": str(cache).strip(),
                "hardware": normalized_hardware,
            }
        )
    except Exception as exc:
        # Preserve one machine-readable report even if Modal cannot return a
        # function result (for example, an infrastructure or transport error).
        report = {
            "status": "failed",
            "probe": PROBE_MODULE,
            "requested_gpu": normalized,
            "requested_hardware": normalized_hardware,
            "requested_cache": str(cache).strip(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    target = _write_report(output, report)
    # Print the same single report that was written to --output.
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("status") != "complete":
        raise SystemExit(
            f"codegen probe failed; report written to {target}"
        )
    return 0


if _modal is not None and _transport is not None:
    image = _transport.image
    app = _modal.App("attnres-selected-fla-block-codegen-probe", image=image)
    _FUNCTION_OPTIONS = dict(_transport._FUNCTION_OPTIONS)
    _FUNCTION_OPTIONS["timeout"] = max(int(_FUNCTION_OPTIONS.get("timeout", 0)), 5400)

    @app.function(gpu="H100!", **_FUNCTION_OPTIONS)
    def h100(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote(payload, "H100!")

    @app.function(gpu="B200", **_FUNCTION_OPTIONS)
    def b200(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_remote(payload, "B200")

    @app.local_entrypoint()
    def main(
        gpu: str = "",
        hardware: str = "",
        cache: str = "",
        output: str = "",
    ) -> int:
        normalized, _ = _validate_cli_args(gpu, hardware, cache, output)
        remote = h100.remote if normalized == "H100!" else b200.remote
        return _main_impl(normalized, hardware, cache, output, remote)

else:
    # A plain function keeps CPU/static import and helper tests independent of
    # Modal.  The real Modal CLI path is defined above when Modal is present.
    def main(
        gpu: str = "",
        hardware: str = "",
        cache: str = "",
        output: str = "",
    ) -> int:
        return _main_impl(gpu, hardware, cache, output, None)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("H100!", "H100", "B200"), required=True)
    parser.add_argument("--hardware", choices=HARDWARE_CHOICES, required=True)
    parser.add_argument(
        "--cache",
        required=True,
        help=(
            "new or empty cache name/path; relative names use the "
            "namespaced Triton cache"
        ),
    )
    parser.add_argument("--output", required=True, help="one local JSON report path")
    return parser.parse_args(argv)


if __name__ == "__main__" and _modal is None:
    args = _parse_args()
    raise SystemExit(main(args.gpu, args.hardware, args.cache, args.output))
