from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import audit_compiled_step as auditor

TARGET_REPO = Path("/private/tmp/fast-attnres-fair-audit")
H100_REPORT = Path("/private/tmp/compiled-step-h100-full-n2048-seed20260827.json")
B200_REPORT = Path("/private/tmp/compiled-step-b200-full-n2048-seed20260827.json")
FAIR_CONFIG = Path("/private/tmp/compiled-step-full-n2048-fair-seed20260827.json")
FAIR_WRAPPER = Path("/private/tmp/run_compiled_step_fair.py")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_fair_v2(raw: dict, *, gpu: str, seed: int) -> dict:
    """Lift the retained v1 timing fixture into the wrapper's v2 envelope.

    The raw rows and qualification evidence are unchanged.  The wrapper's
    v2 additions are deterministic provenance/config fields, so this gives
    the unit tests a local v2-shaped report without fabricating timing rows.
    """

    report = copy.deepcopy(raw)
    config = json.loads(FAIR_CONFIG.read_text(encoding="utf-8"))
    assert config["seed"] == seed
    report["config"] = config
    report["compiled_step_execution_status"] = "complete"
    report["environment"]["git"]["revision"] = auditor.EXPECTED_REPO_HEAD
    project = auditor._local_project_hashes(TARGET_REPO)
    frozen = auditor._local_frozen_hashes(TARGET_REPO)
    vendor = report["source_hashes"]["vendor"]
    report["source_hashes"] = {
        "frozen": frozen,
        "project": project,
        "vendor": vendor,
        "software_hash": _digest({"frozen": frozen, "project": project, "vendor": vendor}),
    }
    report["contract"]["frozen_hashes"] = frozen
    report["protocol"]["frozen_hashes"] = frozen
    report["hashes"] = {
        "hardware": auditor._json_digest(report["device"]),
        "protocol": frozen,
        "software": report["source_hashes"]["software_hash"],
    }
    backend = report["model_timings"]["compile_backend_metadata"]["fla_triton_compile"]
    backend.update(
        {
            "adapter_sha256": auditor.EXPECTED_FLA_ADAPTER_SHA256,
            "model_rms_weight_allocation": "nonpersistent_buffer",
            "model_rms_weight_reuse": "one_buffer_per_model",
            "direct_call_fallback": "query_ones",
            "compiled_model_fill_launches_per_step": 0,
            "compiled_model_fill_launches_avoided_per_step": 1,
        }
    )
    expected_gpu = auditor.EXPECTED_GPU[gpu]
    report["compiled_step_runtime_preflight"] = {
        "schema": auditor.RUNTIME_PREFLIGHT_SCHEMA,
        "status": "passed",
        "gpu_selector": gpu,
        "gpu_name": expected_gpu["name"],
        "compute_capability": expected_gpu["capability"],
        "nvidia_smi": {
            "name": expected_gpu["name"],
            "uuid": f"GPU-{gpu}-fixture",
            "driver_version": "580.00",
            "pstate": "P0",
            "pci.bus_id": "00000000:00:00.0",
            "power.limit": "700.00 W",
            "clocks.max.sm": "2000 MHz",
            "memory.total": "81920 MiB",
        },
        **auditor.EXPECTED_RUNTIME,
        "repo_head": auditor.EXPECTED_REPO_HEAD,
        "repo_clean": True,
        "runner_sha256": auditor.EXPECTED_RUNNER_SHA256,
        "fla_adapter_sha256": auditor.EXPECTED_FLA_ADAPTER_SHA256,
        "model_sha256": auditor.EXPECTED_MODEL_SHA256,
        "kernel_sha256": auditor.EXPECTED_FAIR_KERNELS,
        "frozen_manifest_sha256": auditor.EXPECTED_FROZEN_MANIFEST_SHA256,
        "fla_revision": auditor.EXPECTED_FLA_REVISION,
        "fla_clean": True,
        "config_sha256": _sha(FAIR_CONFIG),
        "wrapper_sha256": _sha(FAIR_WRAPPER),
        "started_unix_s": 1_000.0,
        "finished_unix_s": 2_000.0,
        "timed_tensor_hashing": False,
        "timed_input_copy": False,
        "timed_qualification": False,
        "fla_unit_rms_weight_lifecycle": auditor.EXPECTED_RMS_WEIGHT_LIFECYCLE,
        "fla_fill_launches_inside_step": 0,
    }
    return report


def _require_h100() -> dict:
    if not TARGET_REPO.is_dir() or not H100_REPORT.is_file() or not FAIR_CONFIG.is_file() or not FAIR_WRAPPER.is_file():
        pytest.skip("external compiled-step v2 fixture is not available")
    if auditor._git(TARGET_REPO, "rev-parse", "HEAD") != auditor.EXPECTED_REPO_HEAD:
        pytest.skip("fair v2 source checkout is not available")
    return _as_fair_v2(auditor.read_report(H100_REPORT), gpu="H100", seed=20260827)


def _audit(report: dict, *, gpu: str = "H100", seed: int = 20260827) -> dict:
    return auditor.audit_compiled_step_report(
        report, repo_root=TARGET_REPO, gpu=gpu, seed=seed
    )


def _reject(report: dict, *, gpu: str = "H100", seed: int = 20260827) -> None:
    with pytest.raises(auditor.CompiledStepAuditError):
        _audit(report, gpu=gpu, seed=seed)


def test_h100_fixture_is_timing_verified_but_not_release_promotable():
    result = _audit(_require_h100())

    assert result["status"] == "timing_verified"
    assert result["timing_verified"] is True
    assert result["release_promotable"] is False
    assert result["timing_rows"] == 240
    assert result["timing_means_ms"] == pytest.approx(
        {"candidate": 28.697735770543417, "baseline": 30.01978848775228}
    )
    interval = result["statistics"][
        "kernel_rank_1024_over_fla_triton_compile_standard_rank_1024"
    ]
    assert interval["ratio"] == pytest.approx(0.9559610794431482)
    assert interval["ci"] == pytest.approx(
        [0.9557230529427686, 0.9561991059435279]
    )
    assert interval["bootstrap_samples"] == 20_000


def test_b200_fixture_uses_the_same_unpooled_contract():
    if not TARGET_REPO.is_dir() or not B200_REPORT.is_file():
        pytest.skip("external B200 compiled-step fixture is not available")
    if auditor._git(TARGET_REPO, "rev-parse", "HEAD") != auditor.EXPECTED_REPO_HEAD:
        pytest.skip("fair v2 source checkout is not available")
    result = _audit(_as_fair_v2(auditor.read_report(B200_REPORT), gpu="B200", seed=20260827), gpu="B200")
    assert result["gpu"] == "B200"
    assert result["timing_rows"] == 240
    assert result["statistics"][
        "kernel_rank_1024_over_fla_triton_compile_standard_rank_1024"
    ]["ratio"] == pytest.approx(0.8804387174221936)


@pytest.mark.parametrize(
    "payload",
    [
        '{"x": 1, "x": 2}',
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": -Infinity}',
        '{"x": 1e400}',
    ],
)
def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(payload: str):
    with pytest.raises(auditor.CompiledStepAuditError):
        auditor.strict_json_loads(payload)


def test_raw_input_ids_are_recomputed_from_the_logical_protocol():
    report = _require_h100()
    report = copy.deepcopy(report)
    report["model_timings"]["raw_samples"][0]["input_hash"] = "0" * 64
    _reject(report)


def test_raw_rows_must_follow_the_seeded_abba_order():
    report = _require_h100()
    report = copy.deepcopy(report)
    rows = report["model_timings"]["raw_samples"]
    rows[0], rows[1] = rows[1], rows[0]
    _reject(report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["model_timings"]["raw_samples"][0].update(ms=-1.0),
        lambda r: r["model_timings"]["raw_samples"].pop(),
        lambda r: r["model_timings"]["warmup"].pop(),
        lambda r: r["model_timings"]["timed_graph_counters"].update(
            delta={"stats": {"new_graphs": 1}}
        ),
        lambda r: r["model_timings"]["statistics"][
            "kernel_rank_1024_over_fla_triton_compile_standard_rank_1024"
        ].update(ratio=0.1),
        lambda r: r["source_hashes"]["project"].update(
            {"benchmarks/run.py": "0" * 64}
        ),
        lambda r: r["compiled_step_runtime_preflight"].update(repo_head="0" * 40),
        lambda r: r["config"]["model_config"].update(sequence=2048),
        lambda r: r["model_timings"]["complete_step_qualification"][
            "kernel_rank_1024"
        ]["compiled_step"]["model_state_max_abs"].popitem(),
    ],
)
def test_adversarial_report_mutations_fail_closed(mutate):
    report = copy.deepcopy(_require_h100())
    mutate(report)
    _reject(report)


def test_root_phase_status_cannot_be_upgraded_by_the_report():
    report = copy.deepcopy(_require_h100())
    report["status"] = "complete"
    _reject(report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("compiled_step_execution_status"),
        lambda r: r.update(compiled_step_execution_status="failed"),
        lambda r: r["compiled_step_runtime_preflight"].update(schema="attnres.compiled_step_runtime_preflight.v1"),
        lambda r: r["compiled_step_runtime_preflight"].pop("wrapper_sha256"),
        lambda r: r["compiled_step_runtime_preflight"].update(fla_fill_launches_inside_step=1),
        lambda r: r["compiled_step_runtime_preflight"].update(timed_input_copy=True),
        lambda r: r["config"]["compiled_step_campaign"].update(schema="attnres.compiled_step_campaign.v1"),
        lambda r: r["config"]["compiled_step_campaign"]["fla_unit_rms_weight"].update(fill_launches_inside_step=1),
    ],
)
def test_fair_v2_execution_and_preflight_contract_is_fail_closed(mutate):
    report = copy.deepcopy(_require_h100())
    mutate(report)
    _reject(report)


def test_nonfinite_range_sized_qualification_error_fails_closed():
    report = copy.deepcopy(_require_h100())
    report["model_timings"]["complete_step_qualification"]["kernel_rank_1024"]["compiled_step"]["loss_max_abs"] = auditor.BF16_MAX_FINITE * 2
    _reject(report)


def test_release_attestation_is_separate_and_bound_to_report_bytes():
    report = _require_h100()
    report = copy.deepcopy(report)
    report_bytes = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    device = report["device"]
    actual_vendor = report["fla_checkout"]["actual"]
    hardware = {
        "gpu": "H100",
        "name": device["name"],
        "capability": device["capability"],
        "total_memory": device["total_memory"],
        "multi_processor_count": device["multi_processor_count"],
    }
    vendor = {
        key: actual_vendor[key]
        for key in ("git_dirty", "origin", "package_file_count", "package_sha256", "revision")
    }
    digest = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True).encode("utf-8")
    ).hexdigest()
    attestation = {
        "schema": auditor.ATTESTATION_SCHEMA,
        "report_sha256": report_sha256,
        "hardware": hardware,
        "vendor": vendor,
        "hashes": {
            "hardware_sha256": digest(hardware),
            "vendor_sha256": digest(vendor),
        },
    }
    result = auditor.audit_compiled_step_report(
        report,
        repo_root=TARGET_REPO,
        gpu="H100",
        seed=20260827,
        require_release_attestation=True,
        release_attestation=attestation,
        report_sha256=report_sha256,
    )
    assert result["attestation_verified"] is True
    assert result["release_promotable"] is False

    bad_attestation = copy.deepcopy(attestation)
    bad_attestation["hashes"]["hardware_sha256"] = "0" * 64
    with pytest.raises(auditor.CompiledStepAuditError):
        auditor.audit_compiled_step_report(
            report,
            repo_root=TARGET_REPO,
            gpu="H100",
            seed=20260827,
            release_attestation=bad_attestation,
            report_sha256=report_sha256,
        )


def test_attestation_is_required_when_release_mode_requests_it():
    report = _require_h100()
    with pytest.raises(auditor.CompiledStepAuditError, match="attestation"):
        auditor.audit_compiled_step_report(
            report,
            repo_root=TARGET_REPO,
            gpu="H100",
            seed=20260827,
            require_release_attestation=True,
        )


def test_sealed_campaign_manifest_can_bind_the_exact_checkout(tmp_path: Path):
    report = _require_h100()
    report_path = tmp_path / "fair-report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    preflight = report["compiled_step_runtime_preflight"]
    manifest = {
        "schema": auditor.MANIFEST_SCHEMA,
        "repo_head": auditor.EXPECTED_REPO_HEAD,
        "project": report["source_hashes"]["project"],
        "frozen": report["source_hashes"]["frozen"],
        "runner_sha256": preflight["runner_sha256"],
        "kernel_sha256": preflight["kernel_sha256"],
    }
    path = tmp_path / "campaign-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    result = auditor.audit_path(
        report_path,
        repo_root=TARGET_REPO,
        gpu="H100",
        seed=20260827,
        campaign_manifest=path,
    )
    assert result["status"] == "timing_verified"

    bad = copy.deepcopy(manifest)
    bad["repo_head"] = "0" * 40
    bad_path = tmp_path / "bad-manifest.json"
    bad_path.write_text(json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(auditor.CompiledStepAuditError, match="manifest repo HEAD"):
        auditor.audit_path(
            report_path,
            repo_root=TARGET_REPO,
            gpu="H100",
            seed=20260827,
            campaign_manifest=bad_path,
        )


def test_hero_projection_requires_six_semantically_keyed_reports(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, int]] = []

    def fake_audit(path, *, repo_root, gpu, seed, campaign_manifest=None):
        assert campaign_manifest is None
        calls.append((gpu, seed))
        return {
            "status": "timing_verified",
            "timing_verified": True,
            "release_promotable": False,
            "report_sha256": f"{len(calls):064x}",
            "timing_means_ms": {"candidate": 10.0, "baseline": 12.0},
            "statistics": {
                "kernel_rank_1024_over_fla_triton_compile_standard_rank_1024": {
                    "ratio": 5 / 6,
                    "ci_low": 0.8,
                    "ci_high": 0.9,
                }
            },
        }

    monkeypatch.setattr(auditor, "audit_path", fake_audit)
    paths = {
        gpu: {seed: tmp_path / f"{gpu}-{seed}.json" for seed in auditor.SUPPORTED_SEEDS}
        for gpu in auditor.SUPPORTED_GPUS
    }
    projection = auditor.build_hero_projection(paths, repo_root=tmp_path)

    assert projection["schema"] == "attnres.compiled_step_hero_projection.v1"
    assert projection["status"] == "audited"
    assert tuple(projection["campaign"]["seeds"]) == (
        "20260827",
        "20260903",
        "20260911",
    )
    assert set(projection["devices"]) == {"H100 SXM", "B200"}
    assert all(len(device["ratios"]) == 3 for device in projection["devices"].values())
    assert "reports" not in projection and "raw_samples" not in json.dumps(projection)
    assert len(calls) == 6
    assert len(projection["provenance"]["source_digest"]) == 64

    with pytest.raises(auditor.CompiledStepAuditError, match="all three seeds"):
        auditor.build_hero_projection(
            {"H100": paths["H100"], "B200": {20260827: paths["B200"][20260827]}},
            repo_root=tmp_path,
        )


def test_hero_projection_can_require_and_bind_all_six_attestations(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, int, Path, bool]] = []

    def fake_audit(
        path,
        *,
        repo_root,
        gpu,
        seed,
        campaign_manifest=None,
        release_attestation_path=None,
        require_release_attestation=False,
    ):
        assert campaign_manifest == "sealed-manifest"
        attestation_path = Path(release_attestation_path)
        calls.append((gpu, seed, attestation_path, require_release_attestation))
        return {
            "status": "timing_verified",
            "timing_verified": True,
            "release_promotable": False,
            "attestation_verified": True,
            "report_sha256": f"{len(calls):064x}",
            "timing_means_ms": {"candidate": 10.0, "baseline": 12.0},
            "statistics": {
                "kernel_rank_1024_over_fla_triton_compile_standard_rank_1024": {
                    "ratio": 5 / 6,
                    "ci_low": 0.8,
                    "ci_high": 0.9,
                }
            },
        }

    monkeypatch.setattr(auditor, "audit_path", fake_audit)
    paths = {
        gpu: {seed: tmp_path / f"{gpu}-{seed}.json" for seed in auditor.SUPPORTED_SEEDS}
        for gpu in auditor.SUPPORTED_GPUS
    }
    attestations = {
        gpu: {
            seed: tmp_path / f"{gpu}-{seed}.attestation.json"
            for seed in auditor.SUPPORTED_SEEDS
        }
        for gpu in auditor.SUPPORTED_GPUS
    }
    for gpu_paths in attestations.values():
        for seed, path in gpu_paths.items():
            path.write_text(json.dumps({"seed": seed}), encoding="utf-8")

    projection = auditor.build_hero_projection(
        paths,
        repo_root=tmp_path,
        campaign_manifest="sealed-manifest",
        release_attestation_paths=attestations,
    )

    assert len(calls) == 6
    assert all(required and path.is_file() for _, _, path, required in calls)
    assert projection["provenance"]["audit_status"] == "passed"
    assert len(projection["provenance"]["source_digest"]) == 64
