"""Audit the two-rank final sweep and reuse the existing README renderers.

Inputs are the unmodified per-device directories from the SSH runner, plus
its frozen source manifest and contract. Historical worker envelopes remain
separate. Ratios and intervals are recomputed from paired timing samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from . import plot_compiled_step_hero as hero
from . import plot_compiled_step_sweep as sweep
from .statistics import simultaneous_paired_ratio_bootstrap


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: require(False, x))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_vectors(model, seed, rounds):
    """Reject missing/duplicate pairs and retain complete qualified arms only."""
    groups = defaultdict(list)
    for row in model["raw_samples"]:
        groups[row["arm"]].append(row)
    require(groups, "no timed arms")
    active = [name for name, entry in model["compile"].items() if entry["status"] == "ok"]
    require(set(groups) == set(active), "timed arms differ from compiled arms")
    vectors = {}
    failed_at = {}
    for arm, rows in groups.items():
        require([r["sample_index"] for r in rows] == list(range(rounds)), f"unpaired {arm}")
        backend = ("fla_triton_compile" if arm.startswith("fla_triton_compile_")
                   else "catswe_phase1" if arm.startswith("catswe_phase1_")
                   else arm.split("_", 1)[0])
        for row in rows:
            require(row["backend"] == backend and row["rank"] == int(arm.rsplit("_", 1)[1]),
                    f"wrong backend/rank identity: {arm}")
            require(row["input_hash"] == sweep._logical_input_hash(
                seed, row["sample_index"], model["config"]), "wrong input identity")
        if any(r["status"] != "ok" for r in rows):
            require(not arm.startswith("kernel_"), f"candidate timing failed: {arm}")
            require(any(f.get("arm") == arm for f in model["comparator_failures"]),
                    f"unexplained timing failure: {arm}")
            failed = [r["sample_index"] for r in rows if r["status"] == "failed"]
            first = failed[0] if failed else -1
            require(len(failed) <= 1 and all(
                r["status"] == ("ok" if i < first else "failed" if i == first
                                else "skipped_due_to_failure")
                for i, r in enumerate(rows)), f"invalid failure transition: {arm}")
            require(all(r["ms"] is None for r in rows if r["status"] != "ok"),
                    f"failed arm has timings: {arm}")
            failed_at[arm] = first
            continue
        require(all(r["timing_method"] == "cuda_graph" and r["replay_count"] == 1
                    and math.isfinite(r["ms"]) and r["ms"] > 0 for r in rows),
                f"invalid timings: {arm}")
        vectors[arm] = [r["ms"] for r in rows]
    for index, scheduled in enumerate(sweep._balanced_schedule(active, rounds, seed)):
        rows = [r for r in model["raw_samples"] if r["sample_index"] == index]
        present = [name for name in scheduled if failed_at.get(name, rounds) >= index]
        skipped = [name for name in active if name not in present]
        require([r["arm"] for r in rows] == present + skipped, "incorrect paired arm schedule")
        require([r["order_index"] for r in rows] == list(range(len(present)))
                + [None] * len(skipped), "invalid interleaved arm order")
    return vectors


def recompute(model, vectors, seed):
    ranks = model["ranks"]
    pairs = {}
    for rank in ranks:
        candidate = f"kernel_rank_{rank}"
        require(candidate in vectors, f"missing candidate {rank}")
        for name in vectors:
            if name.startswith("kernel_"):
                continue
            if name.endswith(f"_{rank}") or name in model["architecture_comparisons"]:
                pairs[f"{candidate}_over_{name}"] = (vectors[name], vectors[candidate])
    small, large = sorted(ranks)
    pairs[f"kernel_rank_{small}_over_rank_{large}"] = (
        vectors[f"kernel_rank_{large}"], vectors[f"kernel_rank_{small}"])
    expected = simultaneous_paired_ratio_bootstrap(
        pairs, seed=seed + 18000, samples=20000, confidence=.95, margin=.01)
    require(set(expected) == set(model["statistics"]), "statistics comparison set changed")
    for name, result in expected.items():
        for key, value in result.items():
            actual = model["statistics"][name][key]
            if isinstance(value, float):
                require(math.isclose(actual, value, rel_tol=2e-10, abs_tol=2e-12),
                        f"statistics mismatch: {name}.{key}")
            elif key == "ci":
                require(len(actual) == 2 and all(math.isclose(a, b, rel_tol=2e-10,
                        abs_tol=2e-12) for a, b in zip(actual, value)), "interval mismatch")
            else:
                require(actual == value, f"statistics mismatch: {name}.{key}")
    return expected


def audit_report(report, cell, seed, gpu, source, source_digest, contract):
    binding = report["final_sweep_source"]
    require(binding == {"commit": source["commit"], "config": contract,
                       "manifest_sha256": source_digest}, "source binding changed")
    require(not report["failures"], "report contains core failures")
    environment = report["environment"]
    runtime = contract["runtime"]
    require(all(environment[k] == runtime[k] for k in ("torch", "triton")), "runtime mismatch")
    require(environment["python"].split()[0] == runtime["python"]
            and environment["cuda_runtime"] == runtime["cuda"], "Python/CUDA mismatch")
    require(report["device"]["capability"] == {"H100": [9, 0], "B200": [10, 0]}[gpu],
            "GPU architecture mismatch")
    require(gpu in report["device"]["name"], "GPU name mismatch")
    expected_files = {name.removeprefix("runner/") for name in source["files"]
                      if name.endswith(".py") and name.startswith(("runner/src/", "runner/benchmarks/"))}
    require(set(report["source_hashes"]["project"]) == expected_files, "source inventory changed")
    for name, sha in report["source_hashes"]["project"].items():
        require(source["files"].get("runner/" + name) == sha, f"source differs: {name}")
    for name, sha in contract["identities"].items():
        require(source["files"].get("runner/" + name) == sha, f"contract differs: {name}")
    config, model = report["config"], report["model_timings"]
    require(config["seed"] == seed and config["model_config"] == cell["model"], "geometry mismatch")
    require(config["ranks"] == model["ranks"] == cell["ranks"], "rank mismatch")
    require(model["status"] in {"complete", "incomplete"} and not model["failures"], "core model failure")
    require(model["requested_rounds"] == cell["rounds"]
            and model["effective_warmup"] == cell["warmups"], "sample count mismatch")
    require(model["accumulation"] == 1 and model["timing_method"] == "cuda_graph"
            and model["reference_timing"] is False, "timing contract mismatch")
    require(model["changed_inputs"] is True, "unchanged timing inputs")
    counters = model["timed_graph_counters"]
    require(counters["stable"] and all(counters[k] == 0 for k in
            ("graph_breaks", "recompiles", "new_unique_graphs")), "unstable timed graph")
    width = cell["model"]["width"]
    require(model["config"] == {**cell["model"], "rank": width}, "reported geometry changed")
    from .run import _catswe_model_eligibility, _liger_model_eligibility
    expected_scope = {}
    for rank in cell["ranks"]:
        expected_scope[f"liger_rank_{rank}"] = _liger_model_eligibility(model["config"], rank)
        expected_scope[f"catswe_phase1_model_rank_{rank}"] = _catswe_model_eligibility(model["config"], rank)
    require(model["model_comparator_scope"] == expected_scope, "comparator eligibility changed")
    sweep._validate_state_protocol(model["state_protocol"], model["config"], seed,
                                   cell["model"]["mode"], width)
    sweep._validate_fla_backend_metadata(model["compile_backend_metadata"]["fla_triton_compile"])
    vectors = paired_vectors(model, seed, cell["rounds"])
    allowed = {*(f"kernel_rank_{r}" for r in cell["ranks"]), *expected_scope,
               f"fla_triton_compile_standard_rank_{width}"}
    require(set(vectors) <= allowed, "unknown timing arm")
    require(model["complete_step_qualification"] == model["pre_timing_gate"], "training gate differs")
    common = model["state_protocol"]["canonical_source"]["common_fixed_state_hash"]
    for name in vectors:
        require(model["state_protocol"]["arms"][name]["common_fixed_state_hash"] == common,
                f"initial model state differs: {name}")
        qualification = (model["qualification"][name.removeprefix("kernel_")]
                         if name.startswith("kernel_") else model["comparator_qualification"][name])
        sweep._validate_model_qualification(
            {k: v for k, v in qualification.items() if k != "eligibility"}, name)
        comp = model["compile"][name]
        require(comp["status"] == "ok" and comp["fullgraph"] is True
                and comp["dynamic"] is False, f"compile not qualified: {name}")
        optimizer = model["optimizer"][name]
        require(optimizer["status"] == "ok"
                and optimizer["implementation"] == "AdamW(fused=True,capturable=True)"
                and optimizer["state_initialized_during_warmup"] is True, "optimizer mismatch")
        gate = model["complete_step_qualification"][name]
        require(gate["status"] == "qualified", f"unqualified training step: {name}")
        step = gate["compiled_step"]
        require(step["status"] == "qualified" and step["optimizer_groups_match"] is True
                and step["tolerance"] == contract["tolerance"], "training comparison incomplete")
        for key in ("model_state_max_abs", "gradient_max_abs", "optimizer_state_max_abs"):
            sweep._nonnegative_map(step[key], key)
        sweep._validate_graph_evidence(model["graph"][name], name)
    return model, vectors, recompute(model, vectors, seed)


def project_cells(model, vectors, statistics, cell, gpu, source):
    width = cell["model"]["width"]
    results = []
    for rank in cell["ranks"]:
        candidate = f"kernel_rank_{rank}"
        arms = [sweep.ArmResult("attnres", "OK", "", mean_ms=mean(vectors[candidate]))]
        names = {"fla": f"fla_triton_compile_standard_rank_{width}",
                 "liger": f"liger_rank_{rank}", "catswe": f"catswe_phase1_model_rank_{rank}"}
        for kind, name in names.items():
            if name in vectors:
                key = f"{candidate}_over_{name}"
                ratio = statistics[key]
                arms.append(sweep.ArmResult(kind, "OK", "", mean_ms=mean(vectors[name]),
                    ratio=ratio["ratio"], ci_low=ratio["ci_low"], ci_high=ratio["ci_high"],
                    n=ratio["n"], reported_key=key))
            else:
                qualification = model["comparator_qualification"].get(name, {})
                scope = model["model_comparator_scope"].get(name, {})
                status = "NA" if scope.get("eligible") is False else "FAIL"
                errors = [f for f in model["comparator_failures"] if f.get("arm") == name]
                require(status == "NA" or errors, f"missing comparator explanation: {name}")
                reason = qualification.get("reason") or scope.get("reason") or str(errors)
                arms.append(sweep.ArmResult(kind, status, reason))
        geometry = cell["model"]
        full = geometry["mode"] == "full"
        events = 2 * geometry["layers"]
        results.append(sweep.CellResult(source, gpu, "release" if cell["rounds"] == 120 else "screen",
            geometry["mode"], None if full else events // geometry["block_count"],
            events + 1 if full else geometry["block_count"] + 1, width, rank,
            "R=D" if rank == width else "R=D/4", cell["warmups"], cell["rounds"],
            "OK", "", tuple(arms)))
    return results


def audit_device(directory, gpu, source_dir):
    """Verify a completed device before releasing it; this needs no GPU."""
    directory, source_dir = Path(directory), Path(source_dir)
    source, contract = read(source_dir / "manifest.json"), read(source_dir / "contract.json")
    for name, sha in source["files"].items():
        require(digest(source_dir / name) == sha, f"snapshot differs: {name}")
    source_digest = digest(source_dir / "manifest.json")
    summary = read(directory / "summary.json")
    require(summary["status"] == "complete" and summary["gpu"] == gpu, "device sweep unfinished")
    require(summary["commit"] == source["commit"] and summary["runtime"] == contract["runtime"],
            "device source/runtime differs")
    expected = [(cell, seed) for cell in contract["cells"] for seed in cell["seeds"]]
    require(len(summary["results"]) == len(expected), "incomplete final matrix")
    cells, records = [], []
    for descriptor, (cell, seed) in zip(summary["results"], expected):
        name = f"{cell['name']}-{seed}"
        require(descriptor["name"] == name, "cell order changed")
        path = directory / "results" / (name + ".json")
        require(digest(path) == descriptor["sha256"], f"report hash mismatch: {path}")
        report = read(path)
        require(descriptor["status"] == report["status"] and report["status"] in
                {"complete", "incomplete"}, "report status mismatch")
        model, vectors, statistics = audit_report(
            report, cell, seed, gpu, source, source_digest, contract)
        projected = project_cells(model, vectors, statistics, cell, gpu, str(path))
        cells.extend(projected)
        records.append({"gpu": gpu, "cell": cell["name"], "seed": seed,
                        "sha256": descriptor["sha256"], "statistics": statistics,
                        "means_ms": {k: mean(v) for k, v in vectors.items()},
                        "comparator_failures": model["comparator_failures"]})
    return cells, records


def headline_projection(records, contract, source_digest):
    cell = contract["cells"][0]
    require(cell["name"] == "headline_full", "missing headline geometry")
    devices = {}
    candidate, baseline = "kernel_rank_1024", "fla_triton_compile_standard_rank_1024"
    for gpu, label in (("H100", "H100 SXM"), ("B200", "B200")):
        selected = [r for r in records if r["gpu"] == gpu and r["cell"] == cell["name"]]
        require([r["seed"] for r in selected] == cell["seeds"], "headline seeds incomplete")
        devices[label] = {
            "absolute_ms": {
                "attnres": median(r["means_ms"][candidate] for r in selected),
                "fla_ckpt1": median(r["means_ms"][baseline] for r in selected),
            },
            "ratios": [{"seed": str(r["seed"]), **{
                k: r["statistics"][f"{candidate}_over_{baseline}"][k]
                for k in ("ratio", "ci_low", "ci_high")}} for r in selected],
        }
    return {
        "schema": hero.PROJECTION_SCHEMA, "status": "audited",
        "provenance": {"generator": "benchmarks.final_sweep_report",
            "audit_schema": "attnres.final_sweep_audit.v1", "audit_status": "passed",
            "source_digest": source_digest},
        "campaign": {"mode": "full", "dtype": "bf16", "rank_relation": "R=D",
            "timing_method": "cuda_graph", "baseline": "native FLA Triton checkpoint 1",
            "optimizer": "AdamW(fused=True,capturable=True)", "rounds": cell["rounds"],
            "warmup": cell["warmups"], "confidence": .95,
            "seeds": [str(seed) for seed in cell["seeds"]],
            "schedule": "paired rotating arm order; three unpooled seeds",
            "model": {k: cell["model"][k] for k in hero.EXPECTED_MODEL}},
        "devices": devices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--h100", type=Path)
    parser.add_argument("--b200", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells, records = [], []
    for gpu, directory in (("H100", args.h100), ("B200", args.b200)):
        if directory:
            device_cells, device_records = audit_device(directory, gpu, args.source)
            cells.extend(device_cells)
            records.extend(device_records)
    require(records, "supply at least one device")
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {"schema": "attnres.final_sweep_audit.v1", "status": "passed", "records": records}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    sweep.write_table(sweep.table_rows(cells), args.output / "results.csv", args.output / "results.md")
    if args.h100 and args.b200:
        sweep.render_sweep([c for c in cells if c.phase == "screen"], args.output)
        projection = headline_projection(records, read(args.source / "contract.json"),
                                         digest(args.source / "manifest.json"))
        path = args.output / "hero_projection.json"
        path.write_text(json.dumps(projection, indent=2) + "\n")
        hero.render_hero(hero.load_projection(path), args.output)


if __name__ == "__main__":
    main()
