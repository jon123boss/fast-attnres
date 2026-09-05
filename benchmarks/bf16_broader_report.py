"""Summarize archived wider/irregular operator cells, including interrupted jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmarks.bf16_report import BACKENDS, RANKS, SEEDS, simultaneous_intervals


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(path):
    hashes = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*.py"))
              if "__pycache__" not in p.parts}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def validate_contract(contract):
    cases = contract["cases"]
    groups = {}
    for case in cases:
        s, n, d, r = case["shape"]
        groups.setdefault((s, n, d, case["layout"]), set()).add(r)
    primary = contract.get("scope") == "primary_operator_confirmation"
    expected = [{"shape": [s, 8192, 1536, r], "layout": "list", "query_scale": .05,
                 "backends": list(BACKENDS)} for s in (9, 49) for r in RANKS]
    geometry = (cases == expected if primary else
                len(cases) == 16 and len(groups) == 2 and
                all(len(ranks) == 8 and group[2] in ranks for group, ranks in groups.items()))
    if (contract.get("gpus") != ["H100", "B200"] or contract.get("seeds") != list(SEEDS)
        or not geometry
        or any(contract.get(k) != v for k, v in {"rounds": 120, "warmups": 10,
                "replays": 8, "monotonic_ratio_upper_bound": 1.005 if primary else 1.01}.items())):
        raise ValueError("summary requires the complete frozen two-GPU operator contract")


def admission_errors(report, job, snapshot, contract):
    errors = []
    expected = dict(job["config"])
    for key in ("sources", "competitors"):
        expected[key] = {name: "/job/" + path for name, path in expected.get(key, {}).items()}
    config = report.get("config", {})
    if report.get("kind") != "operator" or config != expected:
        errors.append("report differs from the archived operator job")
    for key in ("rounds", "warmups", "replays", "cache_autotuning", "torch_baseline"):
        if config.get(key) != contract[key]:
            errors.append("changed control: " + key)
    if any(c not in contract["cases"] for c in config.get("cases", [])):
        errors.append("undeclared case")
    seeds = config.get("seeds", [])
    if not seeds or len(seeds) != len(set(seeds)) or any(s not in SEEDS for s in seeds):
        errors.append("undeclared or duplicate seed")
    runtime = report.get("runtime", {})
    gpu = config.get("gpu")
    if (gpu not in contract["gpus"] or gpu not in runtime.get("gpu", "")
        or runtime.get("capability") != {"H100": [9, 0], "B200": [10, 0]}.get(gpu)
        or any(runtime.get(k) != v for k, v in contract["runtime"].items())):
        errors.append("runtime differs from the frozen environment")
    actual = {name: row.get("content_hash", row.get("sha256"))
              for name, row in report.get("identities", {}).items()}
    if actual != contract["identities"]:
        errors.append("source identities differ from the frozen environment")
    for name, digest in job["hashes"].items():
        if tree_hash(snapshot / name) != digest:
            errors.append("changed snapshot: " + name)
    for name, digest in contract["evaluator_files"].items():
        if file_hash(snapshot / "runner" / name) != digest:
            errors.append("changed evaluator: " + name)
    if report.get("import_failures"):
        errors.append("backend import failure")
    return errors


def summarize(work, jobs, contract):
    validate_contract(contract)
    rows, inputs, failures, admissions = {}, [], [], []
    ledger = {row["id"]: row for row in read(work / "ledger.json")["jobs"]}
    for name in jobs:
        snapshot, result = work / "snapshots" / name, work / "results" / name
        path = result / "report.json"
        report, job = read(path), read(snapshot / "job.json")
        inputs.append({"job": name, "report_sha256": file_hash(path),
                       "job_sha256": file_hash(snapshot / "job.json")})
        if report.get("status") != "complete" or ledger[name]["status"] not in ("complete", "passed"):
            failures.append({"job": name, "execution": ledger[name],
                             "report_status": report.get("status"),
                             "in_progress": report.get("in_progress")})
        issues = admission_errors(report, job, snapshot, contract)
        if issues:
            admissions.append({"job": name, "reasons": issues})
            continue
        for row in report.get("results", []):
            case, seed = row["case"], row["seed"]
            key = (report["config"]["gpu"], *case["shape"], case["layout"], seed)
            if key in rows:
                raise ValueError(f"duplicate measurement cell: {key}; select immutable jobs explicitly")
            if case not in report["config"]["cases"] or seed not in report["config"]["seeds"]:
                admissions.append({"job": name, "cell": key, "reason": "unrequested result"})
                continue
            arms = row["arms"]
            if set(arms) != set(case["backends"]):
                admissions.append({"cell": key, "reason": "missing or undeclared backend"})
            valid = {}
            for backend, arm in arms.items():
                samples = arm.get("samples_ms", [])
                if (arm.get("status") == "passed" and len(samples) == 120
                    and all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in samples)):
                    valid[backend] = arm
                else:
                    failures.append({"cell": key, "backend": backend, "arm": arm})
                    if not (arm.get("phase") in ("qualification", "changed_input")
                            and arm.get("error", "").startswith(("Ineligible:", "AssertionError:"))):
                        admissions.append({"cell": key, "backend": backend, "reason": "unresolved arm"})
            rows[key] = valid

    contrasts, cells, adjacent, missing = {}, [], [], []
    for key, arms in sorted(rows.items()):
        if "candidate" not in arms:
            continue
        label = "/".join(map(str, key))
        others = {n: a for n, a in arms.items() if n != "candidate"}
        for name, arm in others.items():
            contrasts[label + "/vs_" + name] = (arms["candidate"]["samples_ms"], arm["samples_ms"], True)
        if others:
            best = min(others, key=lambda n: sum(others[n]["samples_ms"]))
            cells.append({"cell": label, "strongest_correct_alternative": best,
                          "comparison": label + "/vs_" + best})
    for gpu in contract["gpus"]:
        for seed in SEEDS:
            groups = {}
            for case in contract["cases"]:
                s, n, d, r = case["shape"]
                key = (gpu, s, n, d, r, case["layout"], seed)
                arms = rows.get(key, {})
                if "candidate" not in arms or len(arms) < 2:
                    missing.append(key)
                groups.setdefault((s, n, d, case["layout"]), []).append((r, arms.get("candidate")))
            for group, values in groups.items():
                ordered = sorted(values, reverse=True)
                for (higher, hi), (lower, lo) in zip(ordered, ordered[1:]):
                    if hi is None or lo is None:
                        continue
                    label = "/".join(map(str, (gpu, *group, seed))) + f"/r{lower}_over_r{higher}"
                    contrasts[label] = (lo["samples_ms"], hi["samples_ms"], False)
                    adjacent.append(label)
    intervals = simultaneous_intervals(contrasts)
    for cell in cells:
        cell.update(intervals[cell.pop("comparison")])
        cell["faster_pass"] = cell["ci95_simultaneous"][1] < 1.0
    monotonic = [{"comparison": n, **intervals[n], "pass": intervals[n]["ci95_simultaneous"][1] <= contract["monotonic_ratio_upper_bound"]}
                 for n in adjacent]
    return {"scope": contract["scope"], "contract": contract, "inputs": inputs,
            "candidate_identity": contract["identities"]["candidate"], "cells": cells,
            "intervals": intervals, "adjacent_ranks": monotonic, "missing": missing,
            "failures": failures, "admission_failures": admissions,
            "fastest_pass": not missing and not admissions and bool(cells) and all(x["faster_pass"] for x in cells),
            "monotonic_pass": not missing and not admissions and all(x["pass"] for x in monotonic),
            "note": "Operator latency only; training-step requirements are evaluated separately. "
                    "Pairing follows the hash-verified evaluator's alternating order and shared inputs. "
                    "Complete rows from interrupted jobs are retained; no timing outlier is removed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="+")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path("configs/bf16_broader.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.work, args.jobs, read(args.contract))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"monotonic_pass": result["monotonic_pass"], "cells": len(result["cells"]),
                      "missing": len(result["missing"]), "admission_failures": len(result["admission_failures"])}))


if __name__ == "__main__":
    main()
