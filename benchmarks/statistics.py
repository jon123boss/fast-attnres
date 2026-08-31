"""Paired latency statistics used by the benchmark runner.

The benchmark compares latencies collected on one device in an interleaved
schedule.  A ratio is always ``candidate / baseline``; values below one are
therefore faster.  Inputs are deliberately validated instead of silently
discarding bad samples because a failed timing arm changes the estimand.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _array(values: Iterable[float], name: str) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a test dependency
        raise RuntimeError("numpy is required for benchmark statistics") from exc

    result = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite sample")
    if (result <= 0).any():
        raise ValueError(f"{name} must contain positive latencies")
    return result


def _quantile(values: Any, probability: float) -> float:
    """Use NumPy's modern spelling while supporting older bundled versions."""
    try:
        return float(__import__("numpy").quantile(values, probability, method="linear"))
    except TypeError:  # pragma: no cover - NumPy < 1.22 compatibility
        return float(__import__("numpy").quantile(values, probability, interpolation="linear"))


def classify_interval(low: float, high: float, *, margin: float = 0.01) -> str:
    """Classify a simultaneous ratio interval under the frozen protocol."""
    low = float(low)
    high = float(high)
    margin = float(margin)
    if not (low <= high):
        raise ValueError("interval lower bound must not exceed upper bound")
    if not (0 <= margin < 1):
        raise ValueError("margin must satisfy 0 <= margin < 1")
    if high < 1.0:
        return "gain"
    if (1.0 - margin) <= low <= 1.0 <= high <= (1.0 + margin):
        return "plateau"
    if low > 1.0:
        return "slowdown"
    return "inconclusive"


def classify_ratio(low: float, high: float, *, margin: float = 0.01) -> str:
    """Backward compatible alias for :func:`classify_interval`."""
    return classify_interval(low, high, margin=margin)


def _summary(
    ratios: Any,
    low: float,
    high: float,
    *,
    samples: int,
    confidence: float,
    margin: float,
    simultaneous: bool,
) -> dict[str, Any]:
    point = float(ratios.mean())
    return {
        "n": int(ratios.size),
        "estimate": point,
        "ratio": point,
        "ci": [float(low), float(high)],
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
        "bootstrap_samples": int(samples),
        "simultaneous": bool(simultaneous),
        "classification": classify_interval(low, high, margin=margin),
    }


def paired_ratio_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int = 20_000,
    seed: int = 20260827,
    confidence: float = 0.95,
    margin: float = 0.01,
) -> dict[str, Any]:
    """Estimate a paired candidate/baseline latency ratio with bootstrap CI.

    Pairing is retained by resampling sample indices, rather than resampling
    each arm independently.  The point estimate is the arithmetic mean of the
    per-pair ratios, which makes each interleaved observation contribute one
    equally weighted comparison.
    """
    base = _array(baseline, "baseline")
    cand = _array(candidate, "candidate")
    if base.size != cand.size:
        raise ValueError("paired latency arms must have the same sample count")
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if not (0 < confidence < 1):
        raise ValueError("confidence must lie strictly between zero and one")

    import numpy as np

    ratios = cand / base
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, ratios.size, size=(int(samples), ratios.size))
    estimates = ratios[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low = _quantile(estimates, tail)
    high = _quantile(estimates, 1.0 - tail)
    del np
    return _summary(
        ratios,
        low,
        high,
        samples=samples,
        confidence=confidence,
        margin=margin,
        simultaneous=False,
    )


def _comparison_pair(value: Any, name: str) -> tuple[Sequence[float], Sequence[float]]:
    if isinstance(value, Mapping):
        base = value.get("baseline", value.get("control"))
        cand = value.get("candidate", value.get("treatment"))
        if base is None or cand is None:
            raise ValueError(f"comparison {name!r} needs baseline and candidate samples")
        return base, cand
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    raise TypeError(f"comparison {name!r} must be a pair or mapping")


def simultaneous_paired_ratio_bootstrap(
    comparisons: Mapping[str, Any],
    *,
    samples: int = 20_000,
    seed: int = 20260827,
    confidence: float = 0.95,
    margin: float = 0.01,
) -> dict[str, dict[str, Any]]:
    """Return max deviation simultaneous CIs for several paired comparisons.

    Every comparison must contain the same number of paired observations.  A
    common bootstrap index vector is used for all arms, and the quantile of
    the maximum absolute deviation supplies one familywise interval width.
    This is conservative and avoids reporting independently calibrated CIs as
    if they were simultaneous.
    """
    if not comparisons:
        return {}
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if not (0 < confidence < 1):
        raise ValueError("confidence must lie strictly between zero and one")

    import numpy as np

    ratio_columns = []
    names = []
    n: int | None = None
    for name, value in comparisons.items():
        baseline, candidate = _comparison_pair(value, str(name))
        base = _array(baseline, f"{name}.baseline")
        cand = _array(candidate, f"{name}.candidate")
        if base.size != cand.size:
            raise ValueError(f"comparison {name!r} has unpaired sample counts")
        if n is None:
            n = int(base.size)
        elif base.size != n:
            raise ValueError("all simultaneous comparisons need the same sample count")
        ratio_columns.append(cand / base)
        names.append(str(name))

    ratio_matrix = np.stack(ratio_columns, axis=1)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, int(n), size=(int(samples), int(n)))
    boot = ratio_matrix[indices].mean(axis=1)
    point = ratio_matrix.mean(axis=0)
    max_deviation = np.max(np.abs(boot - point[None, :]), axis=1)
    tail = (1.0 - confidence) / 2.0
    width = _quantile(max_deviation, 1.0 - tail)
    result = {}
    for column, name in enumerate(names):
        low = float(point[column] - width)
        high = float(point[column] + width)
        result[name] = _summary(
            ratio_matrix[:, column],
            low,
            high,
            samples=samples,
            confidence=confidence,
            margin=margin,
            simultaneous=True,
        )
    return result


def simultaneous_paired_ci(
    comparisons: Mapping[str, Any],
    *,
    samples: int = 20_000,
    seed: int = 20260827,
    confidence: float = 0.95,
    margin: float = 0.01,
) -> dict[str, dict[str, Any]]:
    """Alias used by callers that emphasize the CI rather than the bootstrap."""
    return simultaneous_paired_ratio_bootstrap(
        comparisons,
        samples=samples,
        seed=seed,
        confidence=confidence,
        margin=margin,
    )


def summarize_paired_comparisons(
    comparisons: Mapping[str, Any],
    *,
    samples: int = 20_000,
    seed: int = 20260827,
    confidence: float = 0.95,
    margin: float = 0.01,
) -> dict[str, dict[str, Any]]:
    """Short public name for simultaneous comparison summaries."""
    return simultaneous_paired_ratio_bootstrap(
        comparisons,
        samples=samples,
        seed=seed,
        confidence=confidence,
        margin=margin,
    )


__all__ = [
    "classify_interval",
    "classify_ratio",
    "paired_ratio_bootstrap",
    "simultaneous_paired_bootstrap",
    "simultaneous_paired_ci",
    "simultaneous_paired_ratio_bootstrap",
    "summarize_paired_comparisons",
]


simultaneous_paired_bootstrap = simultaneous_paired_ratio_bootstrap
