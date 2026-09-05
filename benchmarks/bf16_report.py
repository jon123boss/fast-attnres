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
BACKENDS = ("release", "candidate", "torch_compile", "legacy_uncached", "liger",
            "fla_checkpoint0", "fla_checkpoint1", "fla_gluon_checkpoint0",
            "fla_gluon_checkpoint1", "catswe_phase1", "hydra_2p", "hydra_2p8")
FAMILIES = {**{name: "fla" for name in BACKENDS if name.startswith("fla_")},
            "legacy_uncached": "legacy", "catswe_phase1": "catswe",
            "hydra_2p": "hydra", "hydra_2p8": "hydra"}
MODEL = {"layers": 24, "width": 1536, "heads": 24, "ffn": 4224,
         "vocab": 100277, "context": 2048, "block_count": 8,
         "activation_checkpointing": False, "rope_theta": 500000.,
         "norm_pos": "before", "qk_norm": True,
         "attnres_eps": 2**-23, "attnres_scale": 1.0}


def _contract_errors(report, result, required_rounds):
    """Reject plausible-looking cells that differ from the frozen experiment."""
    errors = []
    config, identities = report["config"], report.get("identities", {})
    actual_ids = {name: row.get("content_hash", row.get("sha256")) for name, row in identities.items()}
    if config.get("expected_identities") != actual_ids:
        errors.append("execution did not bind the frozen source identities")
    if result.get("model") != result["case"]["model"]:
        errors.append("result model differs from the requested case")
    if result["case"] not in config.get("cases", []):
        errors.append("result case was not requested")
    if config.get("gpu") not in ("H100", "B200"):
        errors.append("unknown GPU")
    seeds = config.get("seeds", [])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s not in SEEDS for s in seeds):
        errors.append("unknown or duplicate seeds")
    if result["seed"] not in seeds or type(result["seed"]) is not int:
        errors.append("result seed was not requested")
    if config.get("rounds") != 120 or required_rounds != 120 or config.get("warmups") != 10:
        errors.append("primary timing requires 10 warmups and 120 rounds")
    if result.get("grad_clip") != 1.0 or result.get("loss_dtype") != "bfloat16":
        errors.append("loss precision or gradient clipping changed")
    if result.get("qualification_tolerances") != {"rtol": .05, "atol": .05}:
        errors.append("BF16 tolerance changed or missing")
    requested = result.get("requested_backends", [])
    if len(requested) != len(BACKENDS) or set(requested) != set(BACKENDS):
        errors.append("frozen backend inventory changed")
    if set(result.get("arms", {})) != set(requested):
        errors.append("requested and emitted arms differ")
    if any(result.get("model", {}).get(k) != v for k, v in MODEL.items()):
        errors.append("model arithmetic changed or missing")
    runtime = report.get("runtime", {})
    if (runtime.get("torch") != "2.13.0+cu130" or runtime.get("triton") != "3.7.1" or
        runtime.get("capability") != {"H100": [9, 0], "B200": [10, 0]}.get(config.get("gpu"))):
        errors.append("runtime differs from the frozen environment")
    if config.get("cache_autotuning") is not True or runtime.get("cache_autotuning") is not True:
        errors.append("autotuning cache policy differs from the frozen environment")
    if len(result.get("input_sha256", "")) != 64:
        errors.append("missing input identity")
    order = result.get("round_order", [])
    passed = {name for name, arm in result.get("arms", {}).items() if arm.get("status") == "passed"}
    if len(order) != required_rounds or any(
        row.get("round") != i or row.get("input") != i % 8 or
        len(row.get("backends", [])) != len(passed) or set(row.get("backends", [])) != passed
        for i, row in enumerate(order)
    ):
        errors.append("missing or unequal paired rounds")
    for name in BACKENDS:
        identity = identities.get(FAMILIES.get(name, name), {})
        if not identity.get("content_hash", identity.get("sha256")):
            errors.append(f"missing frozen source identity: {name}")
    if not identities.get("training_fixture", {}).get("sha256"):
        errors.append("missing frozen training fixture identity")
    gate = result.get("operator_qualification", {})
    operator = gate.get("result") or {}
    model = result.get("model", {})
    sources = 2 * model.get("layers", 0) + 1 if model.get("mode") == "full" else model.get("block_count", 0) + 1
    expected_shape = [sources, 8192, 1536, model.get("rank")]
    if (gate.get("replays") != 8 or operator.get("case", {}).get("shape") != expected_shape or
        operator.get("case", {}).get("query_scale") != .05 or operator.get("seed") != result["seed"]):
        errors.append("missing nonzero-query operator qualification")
    for name in passed:
        arm = operator.get("arms", {}).get(name, {})
        if arm.get("status") != "passed" or len(arm.get("samples_ms", [])) != required_rounds:
            errors.append(f"missing qualified operator timings: {name}")
    optimizer = identities.get("optimizer", {})
    if not config.get("optimizer_source") or not optimizer.get("sha256") or optimizer.get("implementation") != "Muon+AdamW(configured)":
        errors.append("missing original optimizer identity")
    for name, arm in result.get("arms", {}).items():
        if arm.get("status") == "passed":
            if arm.get("round_ids") != list(range(required_rounds)):
                errors.append(f"timing samples lost their round pairing: {name}")
            qualification = arm.get("qualification", {})
            if (arm.get("optimizer") != "Muon+AdamW(configured)" or
                qualification.get("first_update", {}).get("status") not in ("baseline", "matched") or
                qualification.get("gradient_count") != 146):
                errors.append(f"missing optimizer or gradient qualification: {name}")
        elif not arm.get("error"):
            errors.append(f"missing failure evidence: {name}")
    return errors


def simultaneous_intervals(contrasts, *, seed=20260905, resamples=20000):
    """Familywise max-deviation intervals; backend pairs retain round pairing.

    Ranks measured in distinct jobs are resampled independently. No timing
    outlier is removed. Seeds and GPU architectures remain separate results.
    """
    rng = np.random.default_rng(seed)
    names = sorted(contrasts)
    points = []
    for name in names:
        a, b, paired = contrasts[name]
        if len(a) != len(b) or len(a) < 2:
            raise ValueError("comparison requires equal, nontrivial timing counts")
        if not np.isfinite([*a, *b]).all() or min(*a, *b) <= 0:
            raise ValueError("invalid timing sample")
        points.append(float(np.mean(np.asarray(a) / b)) if paired else float(np.mean(a) / np.mean(b)))
    # Pairing connects all arms from the same timing rounds. Establish the
    # components before drawing, so A/B and C/B share one round resample.
    parent = {}
    def component(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for a, b, paired in contrasts.values():
        left, right = component(id(a)), component(id(b))
        if paired:
            parent[right] = left
    deviations = []
    for offset in range(0, resamples, 500):
        size = min(500, resamples - offset)
        maximum = np.zeros(size)
        draws = {}
        for i, name in enumerate(names):
            a, b, paired = contrasts[name]
            a_id, b_id = component(id(a)), component(id(b))
            if a_id not in draws:
                draws[a_id] = rng.integers(0, len(a), (size, len(a)))
            indices = draws[a_id]
            if paired:
                a, b = np.asarray(a), np.asarray(b)
                estimates = (a / b)[indices].mean(1)
            else:
                if b_id not in draws:
                    draws[b_id] = rng.integers(0, len(b), (size, len(b)))
                other = draws[b_id]
                a, b = np.asarray(a), np.asarray(b)
                estimates = a[indices].mean(1) / b[other].mean(1)
            maximum = np.maximum(maximum, np.abs(estimates - points[i]))
        deviations.extend(maximum.tolist())
    width = float(np.quantile(deviations, .95)) if names else 0.
    return {name: {"ratio": points[i], "ci95_simultaneous": [points[i] - width, points[i] + width],
                   "paired": bool(contrasts[name][2]), "rounds": len(contrasts[name][0])}
            for i, name in enumerate(names)}


def summarize(paths, *, candidate="candidate", required_rounds=120, contract=None):
    rows, failures, inputs, identities = {}, [], [], set()
    admission_failures, frozen_identities = [], {}
    input_identities = {}
    if not contract or not contract.get("identities"):
        admission_failures.append({"reason": "missing frozen campaign identity contract"})
    for path in sorted(paths, key=str):
        data = Path(path).read_bytes()
        report = json.loads(data)
        inputs.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        if report.get("kind") != "training":
            raise ValueError("complete-step summary accepts training reports only")
        gpu = report["config"]["gpu"]
        from benchmarks.bf16_primary import contract_digest
        if contract and report["config"].get("primary_contract_sha256") != contract_digest(contract):
            admission_failures.append({"path": str(path), "reason": "execution contract digest mismatch"})
        identity = report.get("identities", {}).get(candidate, {}).get("content_hash")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"missing candidate source identity: {path}")
        identities.add(identity)
        for name, source in report.get("identities", {}).items():
            digest = source.get("content_hash", source.get("sha256"))
            if contract and contract.get("identities", {}).get(name) != digest:
                admission_failures.append({"path": str(path), "reason": f"source differs from frozen contract: {name}"})
            if name in frozen_identities and frozen_identities[name] != digest:
                admission_failures.append({"path": str(path), "reason": f"source identity changed: {name}"})
            frozen_identities[name] = digest
        if report.get("import_failures"):
            admission_failures.append({"path": str(path), "imports": report["import_failures"]})
        for result in report.get("results", []):
            case = result["case"]
            model = case["model"]
            expected = {"layers": 24, "width": 1536, "heads": 24, "ffn": 4224,
                        "vocab": 100277, "context": 2048, "block_count": 8,
                        "activation_checkpointing": False}
            if any(model.get(k) != v for k, v in expected.items()) or any(
                case.get(k) != v for k, v in {"batch": 4, "sequence": 2048, "accumulation": 4}.items()
            ):
                failures.append({"path": str(path), "status": "coverage_only", "case": case})
                continue
            key = (gpu, model["mode"], model.get("rank", model.get("width", 1536)), result["seed"])
            if model["mode"] not in ("full", "block") or type(key[2]) is not int or key[2] not in RANKS:
                admission_failures.append({"path": str(path), "reason": "unknown primary mode or rank", "case": case})
                continue
            issues = _contract_errors(report, result, required_rounds)
            token_hash = result.get("input_sha256")
            if key[3] in input_identities and input_identities[key[3]] != token_hash:
                issues.append("input identity changed at the same seed")
            input_identities[key[3]] = token_hash
            if issues:
                admission_failures.append({"path": str(path), "cell": key, "reasons": issues})
            if key in rows:
                raise ValueError(f"duplicate measurement cell: {key}; select immutable jobs explicitly")
            rows[key] = result
        if report.get("status") != "complete" or report.get("in_progress"):
            admission_failures.append({"path": str(path), "reason": "unfinished input report"})
            failures.append({"path": str(path), "status": report.get("status"),
                             "in_progress": report.get("in_progress"), "error": report.get("error")})
    if len(identities) != 1:
        raise ValueError("a final summary requires exactly one candidate source identity")

    contrasts, cells, missing = {}, [], []
    for key, result in sorted(rows.items()):
        gpu, mode, rank, seed = key
        label = f"{gpu}/{mode}/r{rank}/seed{seed}"
        arms = result["arms"]
        eligible = {n: a for n, a in arms.items() if a.get("status") == "passed"
                    and len(a.get("samples_ms", [])) == required_rounds}
        failed = {n: a for n, a in arms.items() if n not in eligible}
        if failed:
            failures.append({"cell": label, "arms": failed})
            unresolved = {n: a for n, a in failed.items()
                          if a.get("classification", a.get("status")) not in ("incorrect", "ineligible")}
            if unresolved:
                admission_failures.append({"cell": label, "arms": unresolved})
        absent = set(result.get("requested_backends") or []) - set(arms)
        if absent:
            admission_failures.append({"cell": label, "absent_arms": sorted(absent)})
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
            "contract": contract, "input_identities": input_identities,
            "confidence_family": "all reported backend and adjacent-rank contrasts; seeds unpooled",
            "geometric_mean_speedup_observed_cells": gain, "cells": cells,
            "adjacent_ranks": monotonic, "failures": failures, "missing": sorted(set(missing)),
            "admission_failures": admission_failures,
            "primary_pass": not missing and not admission_failures and all(c["nonregression_pass"] for c in cells)
                            and all(c["pass"] for c in monotonic)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", default="candidate")
    parser.add_argument("--contract", required=True, help="frozen primary configuration and source identities")
    args = parser.parse_args()
    result = summarize(args.reports, candidate=args.candidate,
                       contract=json.loads(Path(args.contract).read_text()))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"primary_pass": result["primary_pass"], "missing": len(result["missing"]),
                      "speedup": result["geometric_mean_speedup_observed_cells"]}))


if __name__ == "__main__":
    main()
