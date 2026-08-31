"""Exact, auditable Gluon API bridge for the pinned FLA comparator."""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any


_MARKER = "__attnres_thread_barrier_compatibility__"


def install_gluon_barrier_compatibility() -> dict[str, Any]:
    """Expose FLA's pinned barrier spelling on validated Triton runtimes.

    FLA's pinned source calls the zero-argument ``gl.thread_barrier`` builtin.
    Triton 3.7.1 renamed the equivalent CTA builtin to ``gl.barrier``.  The
    alias is installed on the dependency module before any FLA import and is
    kept for the worker lifetime so JIT compilation sees it.
    """

    triton = importlib.import_module("triton")
    version = str(getattr(triton, "__version__", ""))
    language = importlib.import_module("triton.experimental.gluon.language")
    installed = getattr(language, _MARKER, None)
    if isinstance(installed, dict):
        if (
            installed.get("mode") != "thread_barrier_alias_to_barrier"
            or installed.get("triton_version") != "3.7.1"
            or version != installed.get("triton_version")
            or getattr(language, "thread_barrier", None)
            is not getattr(language, "barrier", None)
        ):
            raise ImportError("the installed Triton Gluon barrier alias was modified")
        return deepcopy(installed)
    if hasattr(language, "thread_barrier"):
        existing = getattr(language, "thread_barrier")
        if not callable(existing):
            raise ImportError("Triton Gluon thread_barrier exists but is not callable")
        if version != "3.6.0":
            raise ImportError(
                "native Gluon thread_barrier is validated only for Triton 3.6.0; "
                f"got {version!r} without an AttnRes compatibility marker"
            )
        return {
            "mode": "native_thread_barrier",
            "triton_version": version,
            "vendor_calls": "zero_argument",
            "cluster": False,
            "applied_before_vendor_import": True,
            "vendor_source_modified": False,
        }
    barrier = getattr(language, "barrier", None)
    if not callable(barrier):
        raise ImportError("Triton Gluon exposes neither thread_barrier nor barrier")
    if version != "3.7.1":
        raise ImportError(
            "the thread_barrier-to-barrier compatibility alias is validated only "
            f"for Triton 3.7.1, got {version!r}"
        )
    setattr(language, "thread_barrier", barrier)
    metadata = {
        "mode": "thread_barrier_alias_to_barrier",
        "triton_version": version,
        "vendor_calls": "zero_argument",
        "cluster": False,
        "applied_before_vendor_import": True,
        "alias_preserves_builtin_identity": language.thread_barrier is language.barrier,
        "vendor_source_modified": False,
    }
    setattr(language, _MARKER, deepcopy(metadata))
    return metadata


__all__ = ["install_gluon_barrier_compatibility"]
