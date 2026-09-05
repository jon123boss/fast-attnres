"""Summarize frozen BF16 reports without dropping failed or missing results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (20260827, 20260903, 20260911)
RANKS = (1536, 768, 384, 192, 96, 64, 32, 16)


def simultaneous_intervals(contrasts, *, seed=20260905, resamples=20000):
    """Familywise max-deviation intervals; backend pairs retain round pairing.

    Ranks measured in distinct jobs are resampled independently. No timing
    outlier is removed. Seeds and GPU architectures remain separate results.
    """
    rng = np.random.default_rng(seed)
    names = list(contrasts)
    points = []
    for name in names:
        a, b, paired = contrasts[name]
        if len(a) != len(b) or len(a) < 2:
            raise ValueError("comparison requires equal, nontrivial timing counts")
        if not np.isfinite([*a, *b]).all() or min(*a, *b) <= 0:
            raise ValueError("invalid timing sample")
        points.append(float(np.mean(np.asarray(a) / b)) if paired else float(np.mean(a) / np.mean(b)))
    deviations = []
    for offset in range(0, resamples, 500):
        size = min(500, resamples - offset)
        maximum = np.zeros(size)
        draws = {}
        for i, name in enumerate(names):
            a, b, paired = contrasts[name]
            a_id, b_id = id(a), id(b)
            indices = draws.setdefault(a_id, rng.integers(0, len(a), (size, len(a))))
            if paired:
                draws[b_id] = indices
                a, b = np.asarray(a), np.asarray(b)
                estimates = (a / b)[indices].mean(1)
            else:
                other = draws.setdefault(b_id, rng.integers(0, len(b), (size, len(b))))
                a, b = np.asarray(a), np.asarray(b)
                estimates = a[indices].mean(1) / b[other].mean(1)
            maximum = np.maximum(maximum, np.abs(estimates - points[i]))
        deviations.extend(maximum.tolist())
    width = float(np.quantile(deviations, .95)) if names else 0.
    return {name: {"ratio": points[i], "ci95_simultaneous": [points[i] - width, points[i] + width],
                   "paired": bool(contrasts[name][2]), "rounds": len(contrasts[name][0])}
            for i, name in enumerate(names)}


def summarize(paths, *, candidate="candidate", required_rounds=120):
    rows, failures, inputs, identities = {}, [], [], set()
    for path in paths:
        data = Path(path).read_bytes()
        report = json.loads(data)
        inputs.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        if report.get("kind") != "training":
            raise ValueError("complete-step summary accepts training reports only")
        gpu = report["config"]["gpu"]
        identity = report.get("identities", {}).get(candidate, {}).get("content_hash")
        if identity:
            identities.add(identity)
        for result in report.get("results", []):
            case = result["case"]
            model = case["model"]
            expected = {"layers": 24, "width": 1536, "heads": 24, "ffn": 4224,
                        "vocab": 100277, "context": 2048, "block_count": 8}
            if any(model.get(k, v) != v for k, v in expected.items()) or any(
                case.get(k, v) != v for k, v in {"batch": 4, "sequence": 2048, "accumulation": 4}.items()
            ):
                failures.append({"path": str(path), "status": "coverage_only", "case": case})
                continue
            key = (gpu, model["mode"], model.get("rank", model.get("width", 1536)), result["seed"])
            if key in rows:
                raise ValueError(f"duplicate measurement cell: {key}; select immutable jobs explicitly")
            rows[key] = result
        if report.get("status") != "complete" or report.get("in_progress"):
            failures.append({"path": str(path), "status": report.get("status"),
                             "in_progress": report.get("in_progress"), "error": report.get("error")})
    if len(identities) != 1:
        raise ValueError("a final summary requires exactly one candidate source identity")

    contrasts, cells, missing = {}, [], []
    for key, result in rows.items():
        gpu, mode, rank, seed = key
        label = f"{gpu}/{mode}/r{rank}/seed{seed}"
        arms = result["arms"]
        eligible = {n: a for n, a in arms.items() if a.get("status") == "passed"
                    and len(a.get("samples_ms", [])) == required_rounds}
        failed = {n: a for n, a in arms.items() if n not in eligible}
        if failed:
            failures.append({"cell": label, "arms": failed})
        if candidate not in eligible:
            missing.append(label + "/candidate")
            continue
        alternatives = {n: a for n, a in eligible.items() if n != candidate}
        if not alternatives:
            missing.append(label + "/correct_alternative")
            continue
        strongest = min(alternatives, key=lambda n: np.mean(alternatives[n]["samples_ms"]))
        for name, arm in alternatives.items():
            contrasts[f"{label}/vs_{name}"] = (eligible[candidate]["samples_ms"], arm["samples_ms"], True)
        cells.append({"cell": label, "strongest_correct_alternative": strongest,
                      "candidate_ms": float(np.mean(eligible[candidate]["samples_ms"])),
                      "alternative_ms": float(np.mean(alternatives[strongest]["samples_ms"]))})

    adjacent = []
    for gpu in ("H100", "B200"):
        for mode in ("full", "block"):
            for seed in SEEDS:
                for rank in RANKS:
                    if (gpu, mode, rank, seed) not in rows:
                        missing.append(f"{gpu}/{mode}/r{rank}/seed{seed}")
                for higher, lower in zip(RANKS, RANKS[1:]):
                    hi = rows.get((gpu, mode, higher, seed), {}).get("arms", {}).get(candidate, {})
                    lo = rows.get((gpu, mode, lower, seed), {}).get("arms", {}).get(candidate, {})
                    if all(a.get("status") == "passed" and len(a.get("samples_ms", [])) == required_rounds
                           for a in (hi, lo)):
                        label = f"{gpu}/{mode}/seed{seed}/r{lower}_over_r{higher}"
                        contrasts[label] = (lo["samples_ms"], hi["samples_ms"], False)
                        adjacent.append(label)
    intervals = simultaneous_intervals(contrasts)
    for cell in cells:
        name = cell["cell"] + "/vs_" + cell["strongest_correct_alternative"]
        cell.update(intervals[name])
        cell["nonregression_pass"] = cell["ci95_simultaneous"][1] <= 1.005
    monotonic = [{"comparison": n, **intervals[n],
                  "pass": intervals[n]["ci95_simultaneous"][1] <= 1.005} for n in adjacent]
    gain = math.exp(np.mean([math.log(1 / cell["ratio"]) for cell in cells])) if cells else None
    return {"inputs": inputs, "candidate_identity": next(iter(identities)),
            "confidence_family": "all reported backend and adjacent-rank contrasts; seeds unpooled",
            "geometric_mean_speedup_observed_cells": gain, "cells": cells,
            "adjacent_ranks": monotonic, "failures": failures, "missing": sorted(set(missing)),
            "primary_pass": not missing and all(c["nonregression_pass"] for c in cells)
                            and all(c["pass"] for c in monotonic)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", default="candidate")
    args = parser.parse_args()
    result = summarize(args.reports, candidate=args.candidate)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"primary_pass": result["primary_pass"], "missing": len(result["missing"]),
                      "speedup": result["geometric_mean_speedup_observed_cells"]}))


if __name__ == "__main__":
    main()
