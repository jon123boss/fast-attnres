"""CPU and adversarial checks for the sealed compiled-step campaign."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks import compiled_step_campaign as campaign
from benchmarks.audit_compiled_step import CompiledStepAuditError, expected_model_schedule

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "compiled_step_campaign.json"
MANIFEST = ROOT / "configs" / "compiled_step_campaign_manifest.json"


def test_sealed_matrix_and_exact_geometry_are_cpu_only():
    config, manifest, config_path, manifest_path = campaign.load_sealed_campaign(root=ROOT)
    assert config_path == CONFIG
    assert manifest_path == MANIFEST
    assert config["model_config"] == {
        "batch": 2,
        "block_count": 8,
        "ffn": 2816,
        "heads": 16,
        "layers": 24,
        "mode": "full",
        "sequence": 1024,
        "source_layout": "list",
        "variant": "sliced",
        "vocab": 32768,
        "width": 1024,
    }
    assert config["compiled_step_campaign"]["schema"] == "attnres.compiled_step_campaign.v2"
    assert config["compiled_step_campaign"]["fla_unit_rms_weight"] == campaign.EXPECTED_RMS_WEIGHT_CAMPAIGN
    resident = config["production_ladder"]["resident_candidate"]
    assert resident["identity_scope"] == "current_autotuned_release_candidate"
    assert resident["kernel_revision"] == "8ddb0bbaf184663703ded65b45839fddd1c429fc"
    assert resident["kernel_tree"] == "a91fb6d7662c36652bf648aa2e8170c90887bc1a"
    assert resident["historical_results_manifest"] == (
        "results/compiled_step/campaign_manifest.json"
    )
    assert len(manifest["jobs"]) == 6
    assert {(row["gpu"], row["seed"]) for row in manifest["jobs"]} == {
        (gpu, seed) for gpu in campaign.SUPPORTED_GPUS for seed in campaign.SUPPORTED_SEEDS
    }


def test_job_override_only_changes_seed_and_vendor_path():
    config, _, _, _ = campaign.load_sealed_campaign(root=ROOT)
    job = campaign.build_job_config(config, 20260903, vendor_root="/vendor/fla")
    assert job["seed"] == 20260903
    assert job["compiled_step_campaign"]["seed"] == 20260903
    assert job["vendor_root"] == str(Path("/vendor/fla").resolve())
    assert {key: job[key] for key in job if key not in {"seed", "compiled_step_campaign", "vendor_root"}} == {
        key: config[key] for key in config if key not in {"seed", "compiled_step_campaign", "vendor_root"}
    }


def test_schedule_is_logical_ABBA_and_has_no_tensor_hashing():
    warmup, orders = expected_model_schedule(20260827)
    assert len(warmup) == 2
    assert len(orders) == 120
    assert all(orders[index + 1] == list(reversed(orders[index])) for index in range(0, 119, 2))
    assert orders[0] == list(reversed(orders[1]))


def test_preflight_is_fail_closed_before_cuda_or_vendor_setup(tmp_path):
    output = tmp_path / "h100-full-seed20260827.json"
    report = campaign.run_job(gpu="H100", seed=20260827, output=output, root=ROOT)
    assert report["compiled_step_execution_status"] == "blocked"
    assert report["timing_subartifact"]["status"] == "blocked"
    assert report["model_timings"]["status"] == "not_run"
    assert output.is_file()
    message = json.loads(output.read_text(encoding="utf-8"))["failures"][0]["error"]["message"]
    assert "preflight" in message.lower() or "checkout" in message.lower()


def test_failed_audit_does_not_replace_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "h100-full-seed20260827.json"
    sentinel = b"sentinel\n"
    output.write_bytes(sentinel)
    monkeypatch.setattr(campaign, "runtime_preflight", lambda **kwargs: {"status": "passed"})
    monkeypatch.setattr(
        campaign,
        "_run_suite",
        lambda config: {
            "status": "incomplete",
            "config": dict(config),
            "model_timings": {"status": "complete", "failures": [], "comparator_failures": []},
        },
    )

    def reject(*args, **kwargs):
        raise campaign.CampaignError("synthetic audit rejection")

    with pytest.raises(campaign.CampaignError, match="synthetic audit rejection"):
        campaign.run_job(gpu="H100", seed=20260827, output=output, root=ROOT, audit=reject)
    assert output.read_bytes() == sentinel


def test_atomic_writer_rejects_symlink_target(tmp_path):
    destination = tmp_path / "destination.json"
    target = tmp_path / "target.json"
    target.write_text("original\n", encoding="utf-8")
    destination.symlink_to(target)
    with pytest.raises(campaign.CampaignError, match="regular file"):
        campaign.atomic_write_json(destination, {"status": "complete"})
    assert target.read_text(encoding="utf-8") == "original\n"


def test_run_job_does_not_resolve_symlink_output_on_blocked_preflight(tmp_path):
    output = tmp_path / "h100-full-seed20260827.json"
    target = tmp_path / "target.json"
    target.write_text("original\n", encoding="utf-8")
    output.symlink_to(target)
    with pytest.raises(campaign.CampaignError, match="regular file"):
        campaign.run_job(gpu="H100", seed=20260827, output=output, root=ROOT)
    assert target.read_text(encoding="utf-8") == "original\n"


def test_sealed_loader_rejects_symlink_manifest(tmp_path):
    config_link = tmp_path / "config.json"
    manifest_link = tmp_path / "manifest.json"
    config_link.symlink_to(CONFIG)
    manifest_link.symlink_to(MANIFEST)
    with pytest.raises(campaign.CampaignError, match="regular file"):
        campaign.load_sealed_campaign(root=ROOT, config_path=config_link, manifest_path=manifest_link)


def test_manifest_mutation_fails_closed():
    _, manifest, _, _ = campaign.load_sealed_campaign(root=ROOT)
    forged = copy.deepcopy(manifest)
    forged["source_sha256"]["benchmarks/model.py"] = "0" * 64
    with pytest.raises(campaign.CampaignError, match="source hash differs"):
        campaign.validate_campaign_manifest(forged, root=ROOT, config_path=CONFIG)


def test_aggregate_requires_every_manifest_job(tmp_path):
    output = tmp_path / "aggregate.json"
    with pytest.raises((campaign.CampaignError, CompiledStepAuditError), match="regular file|job report"):
        campaign.aggregate_campaign(reports_dir=tmp_path, output=output, root=ROOT)
    assert not output.exists()
