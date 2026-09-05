from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import bf16_report


GPUS = ("H100", "B200")
MODES = ("full", "block")

EXPECTED_MODEL = {
    "layers": 24,
    "width": 1536,
    "heads": 24,
    "ffn": 4224,
    "vocab": 100277,
    "context": 2048,
    "block_count": 8,
    "activation_checkpointing": False,
}


def _arm(samples, *, status="passed"):
    return {"status": status, "samples_ms": [float(value) for value in samples]}


def _training_report(
    gpu,
    mode,
    rank,
    seed,
    arms,
    *,
    identity="candidate-v1",
    status="complete",
    in_progress=False,
):
    model = {**EXPECTED_MODEL, "mode": mode, "rank": rank}
    report = {
        "kind": "training",
        "config": {"gpu": gpu},
        "results": [
            {
                "case": {
                    "model": model,
                    "batch": 4,
                    "sequence": 2048,
                    "accumulation": 4,
                },
                "seed": seed,
                "arms": arms,
            }
        ],
        "status": status,
    }
    if in_progress:
        report["in_progress"] = True
    if identity is not None:
        report["identities"] = {"candidate": {"content_hash": identity}}
    return report


def _write_report(tmp_path, name, report):
    path = Path(tmp_path) / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _default_arms(candidate_samples):
    return {
        "candidate": _arm(candidate_samples),
        "reference": _arm([2.0] * len(candidate_samples)),
    }


def _grid_paths(
    tmp_path,
    *,
    omit=(),
    candidate_for=None,
    arms_for=None,
    status_for=None,
    identity_for=None,
    rounds=2,
):
    omitted = set(omit)
    paths = []
    index = 0
    for gpu in GPUS:
        for mode in MODES:
            for seed in bf16_report.SEEDS:
                for rank in bf16_report.RANKS:
                    key = (gpu, mode, rank, seed)
                    if key in omitted:
                        continue
                    candidate_samples = (
                        list(candidate_for(key))
                        if candidate_for is not None
                        else [1.0] * rounds
                    )
                    arms = (
                        arms_for(key, candidate_samples)
                        if arms_for is not None
                        else _default_arms(candidate_samples)
                    )
                    status, in_progress = (
                        status_for(key)
                        if status_for is not None
                        else ("complete", False)
                    )
                    identity = (
                        identity_for(key)
                        if identity_for is not None
                        else "candidate-v1"
                    )
                    paths.append(
                        _write_report(
                            tmp_path,
                            f"cell-{index}.json",
                            _training_report(
                                gpu,
                                mode,
                                rank,
                                seed,
                                arms,
                                identity=identity,
                                status=status,
                                in_progress=in_progress,
                            ),
                        )
                    )
                    index += 1
    return paths


@pytest.fixture
def exact_intervals(monkeypatch):
    """Keep summary-control tests focused on selection and gate mechanics."""

    def deterministic_intervals(contrasts):
        result = {}
        for name, (a, b, paired) in contrasts.items():
            a_array = np.asarray(a, dtype=float)
            b_array = np.asarray(b, dtype=float)
            ratio = float(np.mean(a_array / b_array)) if paired else float(
                np.mean(a_array) / np.mean(b_array)
            )
            result[name] = {
                "ratio": ratio,
                "ci95_simultaneous": [ratio, ratio],
                "paired": bool(paired),
                "rounds": len(a_array),
            }
        return result

    monkeypatch.setattr(bf16_report, "simultaneous_intervals", deterministic_intervals)


def test_missing_rank_and_seed_are_explicit_and_fail_primary(tmp_path, exact_intervals):
    omitted = {
        ("H100", "full", 768, bf16_report.SEEDS[1]),
        ("B200", "block", 32, bf16_report.SEEDS[2]),
    }
    result = bf16_report.summarize(
        _grid_paths(tmp_path, omit=omitted), required_rounds=2
    )

    assert set(result["missing"]) == {
        "H100/full/r768/seed20260903",
        "B200/block/r32/seed20260911",
    }
    assert result["primary_pass"] is False


def test_changed_candidate_identities_are_rejected(tmp_path):
    first = _training_report(
        "H100", "full", 1536, bf16_report.SEEDS[0], _default_arms([1.0, 1.0]),
        identity="candidate-v1",
    )
    second = _training_report(
        "B200", "block", 768, bf16_report.SEEDS[1], _default_arms([1.0, 1.0]),
        identity="candidate-v2",
    )

    with pytest.raises(ValueError, match="exactly one candidate source identity"):
        bf16_report.summarize(
            [
                _write_report(tmp_path, "first.json", first),
                _write_report(tmp_path, "second.json", second),
            ],
            required_rounds=2,
        )


def test_candidate_identity_missing_from_one_input_is_rejected(tmp_path):
    first = _training_report(
        "H100", "full", 1536, bf16_report.SEEDS[0], _default_arms([1.0, 1.0]),
        identity="candidate-v1",
    )
    second = _training_report(
        "B200", "block", 768, bf16_report.SEEDS[1], _default_arms([1.0, 1.0]),
        identity=None,
    )

    with pytest.raises(ValueError, match="candidate source identity"):
        bf16_report.summarize(
            [
                _write_report(tmp_path, "with-identity.json", first),
                _write_report(tmp_path, "without-identity.json", second),
            ],
            required_rounds=2,
        )


def test_paired_bootstrap_preserves_round_correlation():
    a = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    b = 2.0 * a

    paired = bf16_report.simultaneous_intervals(
        {"paired": (a, b, True)}, seed=17, resamples=4000
    )["paired"]
    independent = bf16_report.simultaneous_intervals(
        {"independent": (a, b, False)}, seed=17, resamples=4000
    )["independent"]

    assert paired["ratio"] == pytest.approx(0.5)
    assert paired["ci95_simultaneous"] == pytest.approx([0.5, 0.5])
    assert independent["ratio"] == pytest.approx(0.5)
    assert independent["ci95_simultaneous"][1] > independent["ci95_simultaneous"][0]


def test_simultaneous_ci_uses_one_familywise_max_deviation():
    tight = {"tight": ([1.0] * 6, [1.0] * 6, False)}
    wide = {
        "tight": ([1.0] * 6, [1.0] * 6, False),
        "wide": ([1.0, 1.0, 1.0, 1.0, 1.0, 20.0], [1.0] * 6, False),
    }

    single = bf16_report.simultaneous_intervals(tight, seed=23, resamples=4000)[
        "tight"
    ]
    joint = bf16_report.simultaneous_intervals(wide, seed=23, resamples=4000)

    assert single["ci95_simultaneous"] == pytest.approx([1.0, 1.0])
    assert joint["tight"]["ci95_simultaneous"][1] > joint["tight"]["ratio"]
    tight_width = joint["tight"]["ci95_simultaneous"][1] - joint["tight"]["ratio"]
    wide_width = joint["wide"]["ci95_simultaneous"][1] - joint["wide"]["ratio"]
    assert tight_width == pytest.approx(wide_width)


def test_fastest_eligible_alternative_excludes_failed_and_short_arms(
    tmp_path, exact_intervals
):
    arms = {
        "candidate": _arm([4.0] * 4),
        "failed_fast": _arm([0.5] * 4, status="failed"),
        "short_fast": _arm([0.75] * 3),
        "eligible_fast": _arm([2.0] * 4),
        "eligible_slow": _arm([3.0] * 4),
    }
    result = bf16_report.summarize(
        [
            _write_report(
                tmp_path,
                "selection.json",
                _training_report(
                    "H100", "full", 1536, bf16_report.SEEDS[0], arms
                ),
            )
        ],
        required_rounds=4,
    )

    cell = result["cells"][0]
    assert cell["strongest_correct_alternative"] == "eligible_fast"
    assert cell["alternative_ms"] == pytest.approx(2.0)
    assert set(result["failures"][0]["arms"]) == {"failed_fast", "short_fast"}


def test_rank_monotonicity_gate_rejects_slower_lower_rank(tmp_path, exact_intervals):
    target = ("H100", "full", 768, bf16_report.SEEDS[0])

    def candidate_for(key):
        return [1.02, 1.02] if key == target else [1.0, 1.0]

    result = bf16_report.summarize(
        _grid_paths(tmp_path, candidate_for=candidate_for), required_rounds=2
    )

    comparison = next(
        item
        for item in result["adjacent_ranks"]
        if item["comparison"] == "H100/full/seed20260827/r768_over_r1536"
    )
    assert comparison["ratio"] == pytest.approx(1.02)
    assert comparison["pass"] is False
    assert result["missing"] == []
    assert result["primary_pass"] is False


def test_incomplete_report_cannot_primary_pass_when_cells_are_present(
    tmp_path, exact_intervals
):
    target = ("B200", "block", 64, bf16_report.SEEDS[2])

    def status_for(key):
        return ("running", True) if key == target else ("complete", False)

    result = bf16_report.summarize(
        _grid_paths(tmp_path, status_for=status_for), required_rounds=2
    )

    assert result["missing"] == []
    assert any(failure.get("in_progress") for failure in result["failures"])
    assert result["primary_pass"] is False


def test_failed_alternative_cannot_primary_pass_silently(tmp_path, exact_intervals):
    target = ("H100", "block", 1536, bf16_report.SEEDS[1])

    def arms_for(key, candidate_samples):
        arms = _default_arms(candidate_samples)
        if key == target:
            arms["failed_fast"] = _arm([0.5, 0.5], status="failed")
        return arms

    result = bf16_report.summarize(
        _grid_paths(tmp_path, arms_for=arms_for), required_rounds=2
    )

    assert result["missing"] == []
    assert any("failed_fast" in failure["arms"] for failure in result["failures"])
    assert result["primary_pass"] is False


def test_paired_contrasts_with_a_shared_arm_reuse_one_round_resample(monkeypatch):
    """A shared timing arm represents the same benchmark rounds in both pairs."""

    class ScriptedRng:
        def __init__(self):
            self.calls = 0

        def integers(self, low, high, size):
            assert low == 0
            assert high == 3
            assert size == (2, 3)
            self.calls += 1
            return np.array([[0, 1, 2], [2, 2, 2]])

    rng = ScriptedRng()
    monkeypatch.setattr(bf16_report.np.random, "default_rng", lambda seed: rng)

    shared = [1.0, 1.0, 1.0]
    result = bf16_report.simultaneous_intervals(
        {
            "a_vs_shared": ([1.0, 10.0, 100.0], shared, True),
            "c_vs_shared": ([10.0, 1.0, 100.0], shared, True),
        },
        seed=11,
        resamples=2,
    )

    # The common round index must be shared by all three arms.  A separate
    # draw for the second first-arm list breaks the cross-contrast pairing.
    assert rng.calls == 1


def _primary_report(gpu, mode, rank, seed):
    arms = {name: _arm([1.0 if name == "candidate" else 2.0] * 120)
            for name in bf16_report.BACKENDS}
    report = _training_report(gpu, mode, rank, seed, arms)
    report["config"].update(seeds=[seed], rounds=120, warmups=10,
                            optimizer_source="/frozen/optimizer", cache_autotuning=True)
    report["identities"].update({
        name: {"sha256": name + "-v1"}
        for name in ("release", "torch_compile", "fla", "liger", "legacy", "catswe", "hydra")})
    report["identities"]["optimizer"] = {
        "sha256": "optimizer-v1", "implementation": "Muon+AdamW(configured)"}
    result = report["results"][0]
    result["model"] = {**bf16_report.MODEL, "mode": mode, "rank": rank}
    result["case"]["model"] = dict(result["model"])
    report["config"]["cases"] = [result["case"]]
    report["runtime"] = {"torch": "2.13.0+cu130", "triton": "3.7.1",
                         "capability": {"H100": [9, 0], "B200": [10, 0]}[gpu], "cache_autotuning": True}
    result["input_sha256"] = "a" * 64
    result["round_order"] = [{"round": i, "input": i % 8, "backends": list(arms)} for i in range(120)]
    result.update(grad_clip=1.0, loss_dtype="bfloat16",
                  qualification_tolerances={"rtol": .05, "atol": .05},
                  requested_backends=list(bf16_report.BACKENDS))
    report["identities"]["training_fixture"] = {"sha256": "fixture-v1"}
    from benchmarks.bf16_primary import contract_digest
    expected = {name: row.get("content_hash", row.get("sha256")) for name, row in report["identities"].items()}
    report["config"]["expected_identities"] = expected
    report["config"]["primary_contract_sha256"] = contract_digest({"identities": expected})
    result["operator_qualification"] = {"replays": 8, "result": {
        "seed": seed, "case": {"shape": [49 if mode == "full" else 9, 8192, 1536, rank],
                                "query_scale": .05},
        "arms": {name: {"status": "passed", "samples_ms": [1.] * 120} for name in arms}}}
    for name, arm in arms.items():
        arm["round_ids"] = list(range(120))
        arm.update(optimizer="Muon+AdamW(configured)", qualification={
            "gradient_count": 146,
            "first_update": {"status": "baseline" if name == "release" else "matched"}})
    return report


def test_complete_frozen_campaign_passes_with_resumed_seed_subsets(tmp_path, exact_intervals):
    paths = []
    for gpu in GPUS:
        for mode in MODES:
            for rank in bf16_report.RANKS:
                for seed in bf16_report.SEEDS:
                    paths.append(_write_report(tmp_path, f"{gpu}-{mode}-{rank}-{seed}.json",
                                               _primary_report(gpu, mode, rank, seed)))
    example = json.loads(paths[0].read_text())
    contract = {"identities": {name: row.get("content_hash", row.get("sha256"))
                               for name, row in example["identities"].items()}}
    summary = bf16_report.summarize(paths, contract=contract)
    assert summary["primary_pass"]
    assert len(summary["cells"]) == 96
    assert not summary["admission_failures"]


@pytest.mark.parametrize("change", ["dtype", "clipping", "inventory", "identity", "optimizer", "seed", "rounds", "pairing", "inputs", "runtime", "model", "operator", "fixture"])
def test_primary_contract_rejects_mismatched_execution(change):
    report = _primary_report("H100", "block", 64, bf16_report.SEEDS[0])
    result = report["results"][0]
    if change == "dtype":
        result["loss_dtype"] = "float32"
    elif change == "clipping":
        result["grad_clip"] = 0.0
    elif change == "inventory":
        result["requested_backends"].remove("release")
        del result["arms"]["release"]
    elif change == "identity":
        del report["identities"]["fla"]
    elif change == "optimizer":
        result["arms"]["candidate"]["optimizer"] = "AdamW"
    elif change == "seed":
        result["seed"] = 123
    elif change == "rounds":
        report["config"]["rounds"] = 20
    elif change == "pairing":
        result["arms"]["candidate"]["round_ids"].reverse()
    elif change == "inputs":
        result["round_order"][3]["input"] = 0
    elif change == "runtime":
        report["runtime"]["capability"] = [8, 0]
    elif change == "model":
        result["model"]["rope_theta"] = 10000.
    elif change == "operator":
        result["operator_qualification"]["result"]["case"]["query_scale"] = 0.
    elif change == "fixture":
        del report["identities"]["training_fixture"]
    assert bf16_report._contract_errors(report, result, 120)


def test_intervals_are_independent_of_contrast_insertion_order():
    rng = np.random.default_rng(193)
    shared = rng.lognormal(size=120)
    rows = {f"case-{i}": (shared, rng.lognormal(size=120), True) for i in range(8)}
    expected = bf16_report.simultaneous_intervals(rows, resamples=1000)
    actual = bf16_report.simultaneous_intervals(dict(reversed(list(rows.items()))), resamples=1000)
    assert actual == expected


def test_report_cannot_relabel_a_measured_rank():
    report = _primary_report("H100", "block", 64, bf16_report.SEEDS[0])
    result = report["results"][0]
    result["model"]["rank"] = 32
    assert "result model differs from the requested case" in bf16_report._contract_errors(report, result, 120)


@pytest.mark.parametrize("location", ["config", "runtime"])
def test_primary_rejects_mixed_autotuning_cache_policy(location):
    report = _primary_report("H100", "block", 64, bf16_report.SEEDS[0])
    report[location]["cache_autotuning"] = False
    errors = bf16_report._contract_errors(report, report["results"][0], 120)
    assert "autotuning cache policy differs from the frozen environment" in errors
