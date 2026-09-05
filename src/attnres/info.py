"""Lightweight runtime diagnostics for an AttnRes installation.

Run ``python -m attnres.info`` to inspect the package and the local PyTorch
runtime.  The diagnostic command remains usable on CPU-only systems; the
operator itself requires CUDA BF16 tensors.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from typing import Any

import torch

from . import __version__


def _triton_info() -> tuple[bool, str | None]:
    """Return Triton availability and version without importing Triton."""

    try:
        version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        # A source checkout or test stub may expose a module without package
        # metadata.  ``find_spec`` still detects that installation and does
        # not execute its module-level code.
        try:
            available = importlib.util.find_spec("triton") is not None
        except (ImportError, ValueError):
            # ``find_spec`` raises ValueError for a partially initialized
            # module with no spec.  It is still safer to report it as absent
            # than to import Triton just to disambiguate the case.
            available = False
        return available, None
    return True, version


def collect_info() -> dict[str, Any]:
    """Collect stable package and runtime information.

    The returned mapping is intentionally JSON-serializable, making it useful
    to support scripts as well as the human-readable command output.
    """

    triton_available, triton_version = _triton_info()
    return {
        "package": "fast-attnres",
        "version": __version__,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "triton_available": triton_available,
        "triton_version": triton_version,
    }


def _format_human(info: dict[str, Any]) -> str:
    """Format :func:`collect_info` for terminal output."""

    def _bool(value: bool) -> str:
        return "yes" if value else "no"

    def _value(value: Any) -> str:
        return "unavailable" if value is None else str(value)

    return "\n".join(
        (
            f"{info['package']} {info['version']}",
            f"python: {info['python']}",
            f"torch: {info['torch']}",
            f"cuda available: {_bool(info['cuda_available'])}",
            f"cuda runtime: {_value(info['cuda_runtime'])}",
            f"triton available: {_bool(info['triton_available'])}",
            f"triton version: {_value(info['triton_version'])}",
            "attnres API: CUDA BF16 only",
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Print local runtime information and return a process status code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the diagnostic mapping as JSON",
    )
    args = parser.parse_args(argv)
    info = collect_info()
    if args.json:
        json.dump(info, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_format_human(info))
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - covered through subprocess.
    raise SystemExit(main())


__all__ = ["collect_info", "main"]
