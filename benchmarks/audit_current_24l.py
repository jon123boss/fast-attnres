"""Fail-closed offline audit for the current 24-layer Modal campaign.

The performance reports measured commit ``b8837e1``.  The release repository
is intentionally squashable, so the evidence bundle carries a compact archive
of every measured source and frozen-contract file.  This auditor binds the raw
reports to that archive, the Modal transport, runtime/hardware, native FLA
checkout, qualification evidence, exact paired schedule, and independently
recomputed statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from benchmarks.statistics import simultaneous_paired_ratio_bootstrap

SCHEMA = "attnres.current_24l.audit.v1"
VERIFICATION_SCHEMA = "attnres.modal_24l_current_verification.v1"
SEEDS = (20260827, 20260903, 20260911)
ARMS = ("kernel_rank_1024", "fla_triton_compile_standard_rank_1024")
REPORT_SHA256 = {
    "H100!": "38b6bb49736fbd735553dc48daa04033c5321a78ac0d9285f53cffc202d5b093",
    "B200": "9db6cea6616ea749ff3a7724840772b10adf057fad977af7814ea51aaab393bc",
}
AUDIT_SHA256 = {
    "H100!": "a736ac2d303047da896ca30dc93a524d1080315c53b5647edd5ebab495426a73",
    "B200": "0809de2853158acdc8de47d4d3efaa152fbd93893341eb4ba0424b49b2b67a38",
}
VERIFICATION_SHA256 = "c83af3c2c4a7f50c38f161cbb0d1641d760fe66062c819347acf6dc1e1adfd68"
TRANSPORT_SHA256 = "abe14e49aed9d31fba8517ca39dfb68e212db495817c0273e030c80cf20dc60e"
SOURCE_ARCHIVE_SHA256 = "d0efbd3d70cdceb870d2c0b969a94a0444c968022b994f9203af046ef38a8648"
SOURCE_PATHS_SHA256 = "3e9654ab9993a275211dd335128d11c07e711ff3eb25405c6f3afd694726cda9"
SOURCE_REVISION = "b8837e1d74eb708a39a455840332247725a26496"
SOURCE_TREE = "6a807f2f739c45f8ec9051e83df6d7ab4df560ba"
SOURCE_FINGERPRINT = {
    "algorithm": "sha256",
    "digest": "3cd92b9dd42f55f9e2df67cc513e043bc35c25b07e81baae28268df37e1f1cd4",
    "file_count": 553,
    "transport_sha256": TRANSPORT_SHA256,
}
FLA_REVISION = "5e02dd3a7651f5f2797eb8b12bbec401826031e1"
FLA_TREE = "7e4199902fb291c78b3937f223b08ae7bca82bb1"
FLA_PACKAGE_SHA256 = "2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781"
FLA_ORIGIN = "https://github.com/fla-org/flash-linear-attention.git"
KERNEL_SHA256 = {
    "src/attnres/_kernels/fixed_tail.py":
        "2333b3034e3c0e6493855b1246280ed91e65d29a962ce1d150beff71e8bbd34e",
    "src/attnres/_kernels/fixed_tail_sources.py":
        "1373614c93d7291ad96697b1b8ff627120590b75f63f7e38bd65d50b19fcfb4a",
    "src/attnres/_kernels/fla_full_sources.py":
        "8749c72c4714145214e33e8bc7d37f57b47a79b67f2e83044205db72cda416fa",
}
MODEL = {
    "batch": 2,
    "block_count": 8,
    "ffn": 2816,
    "heads": 16,
    "layers": 24,
    "mode": "full",
    "rank": 1024,
    "sequence": 1024,
    "source_layout": "list",
    "variant": "sliced",
    "vocab": 32768,
    "width": 1024,
}
REPORT_TOP_KEYS = frozenset(
    {
        "cache",
        "campaign_results",
        "elapsed_seconds",
        "fla_checkout",
        "fla_source",
        "frozen_hashes",
        "hardware",
        "requested_gpu",
        "runtime",
        "software",
        "source_fingerprint",
        "source_hashes",
        "status",
        "task",
    }
)


class Current24LAuditError(ValueError):
    """Raised when current-campaign evidence fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Current24LAuditError(message)


def _same(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        _require(set(actual) == set(expected), f"{label}: object keys differ")
        for key in expected:
            _same(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        _require(len(actual) == len(expected), f"{label}: list lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _same(left, right, f"{label}[{index}]")
        return
    if isinstance(actual, tuple) and isinstance(expected, tuple):
        _require(len(actual) == len(expected), f"{label}: tuple lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _same(left, right, f"{label}[{index}]")
        return
    _require(type(actual) is type(expected) and actual == expected, f"{label}: {actual!r} != {expected!r}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise Current24LAuditError(f"cannot read {path}: {exc}") from exc


def _json_bytes(data: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")
            ),
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise Current24LAuditError(f"invalid JSON in {label}: {exc}") from exc
    _require(isinstance(value, Mapping), f"{label} must contain a JSON object")
    return value


def _read_json(path: Path, expected_sha256: str, label: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise Current24LAuditError(f"cannot read {label} {path}: {exc}") from exc
    _same(_sha256_bytes(data), expected_sha256, f"{label} SHA-256")
    return _json_bytes(data, label), data


def _archive_files(path: Path) -> dict[str, bytes]:
    _same(_sha256_file(path), SOURCE_ARCHIVE_SHA256, "measured-source archive SHA-256")
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            for member in archive.getmembers():
                name = PurePosixPath(member.name)
                _require(not name.is_absolute() and ".." not in name.parts, "unsafe archive path")
                if member.isdir():
                    continue
                _require(member.isfile(), f"archive member must be a regular file: {member.name}")
                _require(name.parts and name.parts[0] == "performance_source", "wrong archive prefix")
                relative = PurePosixPath(*name.parts[1:]).as_posix()
                _require(relative and relative not in files, f"duplicate archive file {relative}")
                handle = archive.extractfile(member)
                _require(handle is not None, f"cannot extract archive file {relative}")
                files[relative] = handle.read()
    except (OSError, tarfile.TarError) as exc:
        raise Current24LAuditError(f"cannot read measured-source archive: {exc}") from exc
    _require(files, "measured-source archive is empty")
    return files


def _status(value: Any, label: str, accepted: Sequence[str]) -> None:
    _require(isinstance(value, Mapping), f"{label} is missing")
    _require(value.get("status") in accepted, f"{label}.status={value.get('status')!r}")


def _fmean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "cannot take median of an empty sequence")
    return ordered[len(ordered) // 2]


def _verify_bundle(evidence_dir: Path) -> tuple[Mapping[str, Any], dict[str, bytes]]:
    verification, _ = _read_json(
        evidence_dir / "verification.json", VERIFICATION_SHA256, "verification"
    )
    _same(verification.get("schema"), VERIFICATION_SCHEMA, "verification schema")
    _same(verification.get("status"), "passed", "verification status")
    _same(
        verification.get("source"),
        {
            "checkout_clean": True,
            "history_commit_count": 1,
            "revision": SOURCE_REVISION,
            "tree": SOURCE_TREE,
        },
        "measured source identity",
    )
    _same(
        verification.get("runtime"),
        {"cuda": "13.0", "torch": "2.13.0+cu130", "triton": "3.7.1"},
        "verification runtime",
    )
    transport_record = verification.get("transport")
    _require(isinstance(transport_record, Mapping), "verification transport missing")
    _same(transport_record.get("sha256"), TRANSPORT_SHA256, "verification transport hash")
    _require(
        isinstance(transport_record.get("path"), str)
        and bool(transport_record["path"]),
        "verification transport path missing",
    )
    _same(
        verification.get("fla"),
        {
            "origin": FLA_ORIGIN,
            "package_sha256": FLA_PACKAGE_SHA256,
            "revision": FLA_REVISION,
            "tree": FLA_TREE,
        },
        "verification FLA identity",
    )
    protocol = verification.get("protocol")
    _require(isinstance(protocol, Mapping), "verification protocol missing")
    for key, expected in {
        "batch": 2,
        "dtype": "bf16",
        "ffn": 2816,
        "heads": 16,
        "layers": 24,
        "mode": "full",
        "paired_rounds_per_seed": 120,
        "pooled_gpus": False,
        "pooled_seeds": False,
        "rank": 1024,
        "read_count": 48,
        "seeds": list(SEEDS),
        "sequence": 1024,
        "source_count_range": [2, 49],
        "source_layout": "ordered list",
        "vocab": 32768,
        "warmups_per_arm": 10,
        "width": 1024,
    }.items():
        _same(protocol.get(key), expected, f"verification protocol.{key}")
    modal = verification.get("modal")
    _require(isinstance(modal, Mapping), "Modal identity missing")
    _same(modal.get("status"), "stopped", "Modal app status")
    for field in ("app_id", "h100_call_id", "b200_call_id"):
        _require(isinstance(modal.get(field), str) and modal[field], f"Modal {field} missing")

    transport = evidence_dir / "reproduction" / "modal_transport.py"
    _same(_sha256_file(transport), TRANSPORT_SHA256, "Modal transport SHA-256")
    paths_file = evidence_dir / "reproduction" / "performance_source_paths.txt"
    _same(_sha256_file(paths_file), SOURCE_PATHS_SHA256, "source path manifest SHA-256")
    archive = _archive_files(evidence_dir / "reproduction" / "performance_source.tar.gz")
    listed = tuple(line for line in paths_file.read_text(encoding="utf-8").splitlines() if line)
    _same(set(archive), set(listed), "measured-source archive paths")
    return verification, archive


def _audit_report_object(
    report: Mapping[str, Any],
    *,
    gpu: str,
    archive: Mapping[str, bytes],
    repo_root: Path,
) -> dict[str, Any]:
    _same(set(report), set(REPORT_TOP_KEYS), "report fields")
    _same(report.get("status"), "complete", "outer status")
    _same(report.get("task"), "suite_campaign", "task")
    _same(report.get("requested_gpu"), gpu, "requested GPU")
    _same(
        report.get("runtime"),
        {
            "actual": {"torch": "2.13.0+cu130", "triton": "3.7.1"},
            "expected": {"torch": "2.13.0", "triton": "3.7.1"},
            "status": "verified",
        },
        "runtime",
    )
    software = report.get("software")
    _require(isinstance(software, Mapping), "software evidence missing")
    _same(software.get("cuda"), "13.0", "CUDA runtime")
    expected_hardware = (
        ("NVIDIA H100 80GB HBM3", [9, 0])
        if gpu == "H100!"
        else ("NVIDIA B200", [10, 0])
    )
    hardware = report.get("hardware")
    _require(isinstance(hardware, Mapping), "hardware evidence missing")
    _same((hardware.get("name"), hardware.get("capability")), expected_hardware, "hardware")

    fla = report.get("fla_checkout")
    _require(isinstance(fla, Mapping), "FLA checkout evidence missing")
    _same(fla.get("status"), "verified", "FLA verification")
    fla_expected = fla.get("expected")
    fla_actual = fla.get("actual")
    fla_host = fla.get("host_preflight")
    for name, value in (("fla_checkout.expected", fla_expected), ("fla_checkout.actual", fla_actual), ("fla_checkout.host_preflight", fla_host)):
        _require(isinstance(value, Mapping), f"{name} missing")
    _same(fla_expected.get("revision"), FLA_REVISION, "FLA revision")
    _same(fla_expected.get("package_sha256"), FLA_PACKAGE_SHA256, "FLA expected package")
    _same(fla_expected.get("required_clean"), True, "FLA clean requirement")
    _same(fla_actual.get("package_sha256"), FLA_PACKAGE_SHA256, "FLA package")
    _same(fla_actual.get("package_file_count"), 506, "FLA package file count")
    _same(fla_host.get("git_dirty"), False, "FLA dirty state")
    _same(fla_host.get("revision"), FLA_REVISION, "FLA host revision")
    _same(fla_host.get("origin"), FLA_ORIGIN, "FLA origin")
    _same(fla_host.get("package_sha256"), FLA_PACKAGE_SHA256, "FLA host package")
    _same(fla_host.get("package_file_count"), 506, "FLA host package file count")
    _same(report.get("source_fingerprint"), SOURCE_FINGERPRINT, "source fingerprint")
    fla_source = report.get("fla_source")
    _require(isinstance(fla_source, Mapping), "mounted FLA source evidence missing")
    for key, expected in {
        "requested": True,
        "mount_available": True,
        "status": "available",
        "git_dirty": "false",
        "revision": FLA_REVISION,
        "package_sha256": FLA_PACKAGE_SHA256,
    }.items():
        _same(fla_source.get(key), expected, f"fla_source.{key}")

    source_hashes = report.get("source_hashes")
    frozen_hashes = report.get("frozen_hashes")
    _require(isinstance(source_hashes, Mapping) and source_hashes, "source hashes absent")
    _require(isinstance(frozen_hashes, Mapping) and frozen_hashes, "frozen hashes absent")
    expected_paths = set(source_hashes) | set(frozen_hashes) | {"validation/frozen.json"}
    _same(set(archive), expected_paths, "archive/report path set")
    for relative, digest in {**frozen_hashes, **source_hashes}.items():
        _same(_sha256_bytes(archive[relative]), digest, f"measured source {relative}")
    frozen_manifest = _json_bytes(archive["validation/frozen.json"], "archived frozen manifest")
    # The Modal transport intentionally omitted the two CUDA-only test files
    # from its mounted frozen subset.  Bind that exact, predeclared difference;
    # no other manifest entry may disappear or be added.
    _same(
        set(frozen_manifest) - set(frozen_hashes),
        {"tests/test_cuda.py", "tests/test_offsets.py"},
        "transport-omitted frozen paths",
    )
    _same(set(frozen_hashes) - set(frozen_manifest), set(), "unexpected frozen paths")
    for relative, digest in frozen_hashes.items():
        _same(frozen_manifest.get(relative), digest, f"archived frozen entry {relative}")
    for relative, digest in KERNEL_SHA256.items():
        _same(source_hashes.get(relative), digest, f"measured kernel {relative}")
        local = repo_root / relative
        _require(local.is_file(), f"release kernel missing: {relative}")
        _same(_sha256_file(local), digest, f"release kernel {relative}")

    campaigns = report.get("campaign_results")
    _require(isinstance(campaigns, list), "campaign_results must be a list")
    _same([item.get("seed") for item in campaigns], list(SEEDS), "campaign seeds")
    results = []
    for campaign in campaigns:
        seed = campaign["seed"]
        _same(campaign.get("compiled_step_execution_status"), "complete", f"{seed} execution")
        measurement = campaign.get("measurements")
        _require(isinstance(measurement, Mapping), f"{seed} measurements missing")
        _same(measurement.get("status"), "incomplete", f"{seed} model-only suite status")
        timing = measurement.get("model_timings")
        _require(isinstance(timing, Mapping), f"{seed} model_timings missing")
        _same(timing.get("status"), "complete", f"{seed} timing status")
        _same(timing.get("failures"), [], f"{seed} failures")
        _same(timing.get("comparator_failures"), [], f"{seed} comparator failures")
        _same(timing.get("config"), MODEL, f"{seed} model config")
        _same(timing.get("requested_warmup"), 10, f"{seed} warmup")
        _same(timing.get("requested_rounds"), 120, f"{seed} rounds")
        _same(timing.get("timing_method"), "cuda_graph", f"{seed} timing method")
        _same(timing.get("changed_inputs"), True, f"{seed} changed-input gate")
        _same(
            timing.get("training_step"),
            "benchmarks.training_graph.CapturedTrainingStep.replay",
            f"{seed} timed step",
        )
        boundary = timing.get("timing_boundary")
        _require(isinstance(boundary, Mapping), f"{seed} timing boundary missing")
        _same(
            boundary.get("steady_step_includes"),
            [
                "BF16 autocast",
                "zero_grad",
                "model forward",
                "cross_entropy loss",
                "backward",
                "gradient accumulation",
                "AdamW optimizer.step",
            ],
            f"{seed} timing includes",
        )
        excluded = boundary.get("excluded")
        _require(
            isinstance(excluded, list)
            and "input copies" in excluded
            and "graph capture" in excluded,
            f"{seed} timing exclusions",
        )
        identity = timing.get("timed_input_identity")
        _require(isinstance(identity, Mapping), f"{seed} timed input identity missing")
        _same(identity.get("tensor_byte_hashing"), False, f"{seed} timed hashing")
        _same(identity.get("device_to_host_copy"), False, f"{seed} timed host copy")
        _same(identity.get("shared_tensor_objects_across_arms"), True, f"{seed} shared inputs")
        counters = timing.get("timed_graph_counters")
        _require(isinstance(counters, Mapping), f"{seed} graph counters missing")
        for key, expected in {
            "stable": True,
            "delta": {},
            "graph_breaks": 0,
            "recompiles": 0,
            "new_unique_graphs": 0,
        }.items():
            _same(counters.get(key), expected, f"{seed} graph counter {key}")
        for group_name in ("complete_step_qualification", "pre_timing_gate", "graph"):
            group = timing.get(group_name)
            _require(isinstance(group, Mapping), f"{seed} {group_name} missing")
            for arm in ARMS:
                _status(group.get(arm), f"{seed} {group_name}.{arm}", ("qualified", "complete", "passed", "ok"))
        comparator_qualification = timing.get("comparator_qualification")
        _require(isinstance(comparator_qualification, Mapping), f"{seed} comparator qualification")
        _status(
            comparator_qualification.get(ARMS[1]),
            f"{seed} comparator qualification",
            ("qualified", "complete", "passed", "ok"),
        )

        raw = timing.get("raw_samples")
        _require(isinstance(raw, list) and len(raw) == 240, f"{seed} raw row count")
        by_sample: dict[int, list[Mapping[str, Any]]] = {}
        for row in raw:
            _require(isinstance(row, Mapping), f"{seed} raw row must be an object")
            _same(row.get("status"), "ok", f"{seed} raw status")
            _require(row.get("arm") in ARMS, f"{seed} unknown arm")
            _same(row.get("timing_method"), "cuda_graph", f"{seed} row timing method")
            _same(row.get("replay_count"), 1, f"{seed} replay count")
            sample = row.get("sample_index")
            order = row.get("order_index")
            _require(type(sample) is int and 0 <= sample < 120, f"{seed} sample index")
            _require(type(order) is int and order in (0, 1), f"{seed} order index")
            latency = row.get("ms")
            _require(
                type(latency) in (int, float) and math.isfinite(latency) and latency > 0,
                f"{seed} latency",
            )
            input_hash = row.get("input_hash")
            _require(
                isinstance(input_hash, str) and re.fullmatch(r"[0-9a-f]{64}", input_hash),
                f"{seed} input hash",
            )
            by_sample.setdefault(sample, []).append(row)
        _same(sorted(by_sample), list(range(120)), f"{seed} sample coverage")
        candidate: list[float] = []
        baseline: list[float] = []
        logical_hashes: list[str] = []
        first_order: list[str] | None = None
        for sample in range(120):
            pair = by_sample[sample]
            _same([row["order_index"] for row in pair], [0, 1], f"{seed} order positions {sample}")
            order = [row["arm"] for row in pair]
            _require(set(order) == set(ARMS), f"{seed} arm pair {sample}")
            if first_order is None:
                first_order = order
            else:
                expected_order = first_order if sample % 2 == 0 else list(reversed(first_order))
                _same(order, expected_order, f"{seed} ABBA order {sample}")
            _same(pair[0]["input_hash"], pair[1]["input_hash"], f"{seed} paired input {sample}")
            logical_hashes.append(pair[0]["input_hash"])
            values = {row["arm"]: float(row["ms"]) for row in pair}
            candidate.append(values[ARMS[0]])
            baseline.append(values[ARMS[1]])
        _same(len(set(logical_hashes)), 120, f"{seed} changed logical inputs")
        comparison_id = f"{ARMS[0]}_over_{ARMS[1]}"
        recomputed = simultaneous_paired_ratio_bootstrap(
            {comparison_id: (baseline, candidate)},
            samples=20_000,
            seed=seed + 18_000,
            margin=0.01,
        )[comparison_id]
        reported = timing.get("statistics", {}).get(comparison_id)
        _require(isinstance(reported, Mapping), f"{seed} reported statistics missing")
        _same(set(reported), set(recomputed), f"{seed} statistic fields")
        for key, value in recomputed.items():
            if isinstance(value, float):
                _require(
                    math.isclose(reported[key], value, rel_tol=0.0, abs_tol=1e-15),
                    f"{seed} statistic {key}",
                )
            else:
                _same(reported[key], value, f"{seed} statistic {key}")
        results.append(
            {
                "seed": seed,
                "candidate_mean_ms": _fmean(candidate),
                "fla_mean_ms": _fmean(baseline),
                **recomputed,
            }
        )
    return {
        "gpu": gpu,
        "median_advantage_pct": _median([(1.0 - row["estimate"]) * 100.0 for row in results]),
        "results": results,
    }


def audit_report(
    report_path: str | Path,
    *,
    gpu: str,
    evidence_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Audit one immutable H100 or B200 aggregate report."""

    _require(gpu in REPORT_SHA256, f"gpu must be one of {tuple(REPORT_SHA256)}")
    evidence = Path(evidence_dir).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    verification, archive = _verify_bundle(evidence)
    report, report_bytes = _read_json(Path(report_path), REPORT_SHA256[gpu], f"{gpu} report")
    artifact_key = "H100" if gpu == "H100!" else "B200"
    artifact = verification.get("artifacts", {}).get(artifact_key)
    _require(isinstance(artifact, Mapping), f"verification artifact {artifact_key} missing")
    _same(artifact.get("report_sha256"), REPORT_SHA256[gpu], f"{gpu} verification report hash")
    audit_path = evidence / "audits" / ("h100-audit.json" if gpu == "H100!" else "b200-audit.json")
    _same(_sha256_file(audit_path), AUDIT_SHA256[gpu], f"{gpu} retained audit hash")
    result = _audit_report_object(report, gpu=gpu, archive=archive, repo_root=root)
    result.update(
        {
            "bytes": len(report_bytes),
            "report_sha256": REPORT_SHA256[gpu],
            "schema": SCHEMA,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "status": "passed",
            "verification_sha256": VERIFICATION_SHA256,
        }
    )
    return result


def audit_bundle(evidence_dir: str | Path, repo_root: str | Path) -> dict[str, Any]:
    """Audit both device reports and keep their conclusions separate."""

    evidence = Path(evidence_dir).expanduser().resolve()
    reports = {
        "H100": audit_report(
            evidence / "raw" / "h100-report.json",
            gpu="H100!",
            evidence_dir=evidence,
            repo_root=repo_root,
        ),
        "B200": audit_report(
            evidence / "raw" / "b200-report.json",
            gpu="B200",
            evidence_dir=evidence,
            repo_root=repo_root,
        ),
    }
    return {"schema": SCHEMA, "status": "passed", "reports": reports}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("results/current_24l"),
        help="current 24-layer evidence directory",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="release checkout whose production kernels must match",
    )
    parser.add_argument("--output", type=Path, help="optional canonical audit JSON")
    args = parser.parse_args(argv)
    try:
        result = audit_bundle(args.evidence_dir, args.repo)
    except (Current24LAuditError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"current 24-layer audit failed: {exc}\n")
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests
    raise SystemExit(main())
