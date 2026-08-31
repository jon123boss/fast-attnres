"""Small training models and helpers used by the AttnRes benchmarks.

Keep the model helpers lazy.  CPU-only launchers such as the matched-comparator
Modal transport import protocol modules through this package before the remote
worker exists; eagerly importing :mod:`benchmarks.model` would make those
launchers require a local Torch installation for no computational reason.
"""

from __future__ import annotations

from typing import Any


__all__ = ["TrainingConfig", "make_model", "training_step"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import model

    return getattr(model, name)
