"""CPU contract tests for the remote compiled-step sweep planner."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts import compiled_step_sweep as sweep


def _result_test_payload(cell, *, run_id: str = "result-test"):
    project = sweep.make_manifest()["project_provenance"]
    parameters = {
        "seed": sweep.DEFAULT_SEED,
        "warmup": sweep.DEFAULT_WARMUP,
        "rounds": sweep.DEFAULT_ROUNDS,
        "bootstrap_samples": sweep.DEFAULT_BOOTSTRAP,
        "batch": sweep.BATCH,
        "sequence": sweep.SEQUENCE,
        "vocab": sweep.VOCAB,
    }
    roots = {
        "remote_repo": sweep.DEFAULT_REMOTE_REPO,
        "remote_fla_root": sweep.DEFAULT_REMOTE_FLA,
        "remote_liger_root": sweep.DEFAULT_REMOTE_LIGER,
        "remote_catswe_root": sweep.DEFAULT_REMOTE_CATSWE,
    }
    return {
        "schema": sweep.SCHEMA,
        "gpu": "H100",
        "cell": copy.deepcopy(dict(cell)),
        "config": sweep.make_worker_config(cell, **parameters, **roots),
        **roots,
        "triton_cache_dir": sweep.DEFAULT_CACHE_ROOT,
        "run_id": run_id,
        **parameters,
        "project_provenance": project,
    }


def _complete_result_for_test(payload):
    config = payload["config"]
    cell = payload["cell"]
    include_catswe = bool(config["include_catswe_model"])
    expected_liger = cell["competitors"]["liger"]["status"] == "model_step_arm"
    comparators = {"liger": {}}
    schedules = {"liger": "native Liger per-read aggregation"}
    model_scope = {
        f"liger_rank_{cell['rank']}": {"eligible": expected_liger},
    }
    if include_catswe:
        comparators["catswe_phase1"] = {}
        schedules["catswe_phase1"] = sweep.CATSWE_MODEL_SCHEDULE
        model_scope[f"catswe_phase1_model_rank_{cell['rank']}"] = {
            "eligible": True,
            "capability_scope": "model",
            "model_scope": "compiled_training_step",
        }
    benchmark = {
        "status": "complete",
        "config": copy.deepcopy(config),
        "comparators_enabled": True,
        "comparators": comparators,
        "coverage": {
            "scope": "custom",
            "include_liger_model": True,
            **({"include_catswe_model": True} if include_catswe else {}),
        },
        "model_timings": {
            "status": "complete",
            "failures": [],
            "comparator_failures": [],
            "include_liger_model": True,
            "include_catswe_model": include_catswe,
            "timing_method": "cuda_graph",
            "training_step": "benchmarks.training_graph.CapturedTrainingStep.replay",
            "requested_warmup": config["model_warmup"],
            "requested_rounds": config["model_rounds"],
            "timed_input_identity": {
                "tensor_byte_hashing": False,
                "device_to_host_copy": False,
            },
            "timing_boundary": {
                "steady_step_includes": ["AdamW optimizer.step"],
                "backward_orchestration": "captured complete step including optimizer update",
            },
            "execution_schedules": schedules,
            "model_comparator_scope": model_scope,
        },
    }
    project = copy.deepcopy(payload["project_provenance"])
    catswe_attestation = None
    if include_catswe:
        catswe = project.pop("catswe")
        catswe_attestation = {
            "status": "verified",
            "transport": "git_checkout",
            "revision": catswe["revision"],
            "tree": catswe["tree"],
            "clean": True,
            "origin": catswe["origin"],
            "license": catswe["license"],
            "license_file": catswe["license_file"],
            "license_sha256": catswe["license_sha256"],
            "source_hashes": copy.deepcopy(catswe["source_hashes"]),
            "vendor_file_sha256": copy.deepcopy(catswe["vendor_file_sha256"]),
        }
    else:
        project.pop("catswe", None)
    result = {
        "schema": sweep.SCHEMA,
        "status": "complete",
        "gpu": payload["gpu"],
        "cell": copy.deepcopy(cell),
        **sweep._worker_result_binding(payload),
        "runtime_preflight": {
            "status": "passed",
            "gpu": payload["gpu"],
            "name": "NVIDIA H100",
            "compute_capability": [9, 0],
            "total_memory": 80 * 2**30,
            "torch": "2.13.0+cu130",
            "cuda": "13.0",
            "triton": "3.7.1",
        },
        "provenance": {
            "project": {"status": "verified", **project},
            **({"catswe": catswe_attestation} if catswe_attestation is not None else {}),
        },
        "report_identity": sweep._report_identity(benchmark),
        "worker": {
            "run_id": payload["run_id"],
            "started_unix_s": 1.0,
            "finished_unix_s": 2.0,
            "elapsed_s": 1.0,
            "timed_tensor_hashing": False,
            "timed_input_copy": False,
            "timed_qualification": False,
        },
        "benchmark": benchmark,
        "failure": None,
    }
    return result


def _launcher_for_test(payload, output_root: Path, *, remote_output_root: str):
    run_id = payload["run_id"]
    gpu = payload["gpu"]
    cell_id = payload["cell"]["cell_id"]
    return {
        "run_id": run_id,
        "gpu": gpu,
        "cell_id": cell_id,
        "started_unix_s": 1.0,
        "finished_unix_s": 2.0,
        "elapsed_s": 1.0,
        "ssh_exit_code": 0,
        "remote_output": f"{remote_output_root.rstrip('/')}/{gpu.lower()}/failures/{cell_id}.{run_id}.json",
        "command": "ssh worker",
        "logs": {
            "stdout": str(sweep._log_path(output_root, gpu, cell_id, run_id, "stdout.log")),
            "stderr": str(sweep._log_path(output_root, gpu, cell_id, run_id, "stderr.log")),
        },
    }


def test_matrix_controls_block_source_counts_and_ranks():
    cells = sweep.build_matrix()
    assert len(cells) == 50
    assert len({cell["cell_id"] for cell in cells}) == 50
    assert {cell["width"] for cell in cells} == {1024, 1536, 2048, 3072, 4096}
    assert all(cell["head_dim"] == 64 for cell in cells)
    assert all(cell["heads"] == cell["width"] // 64 for cell in cells)
    assert all(cell["rank"] in {cell["width"] // 4, cell["width"]} for cell in cells)
    assert all(cell["rank"] != 128 for cell in cells)
    assert {
        cell["event_block_size"]: cell["max_read_sources"]
        for cell in cells
        if cell["mode"] == "block" and cell["width"] == 1024 and cell["rank"] == 256
    } == {8: 3, 4: 5, 2: 9, 1: 17}
    assert sweep.read_source_counts_for_cell(mode="full", event_block_size=None) == tuple(range(2, 18))
    for block_size, expected in {8: (2,) * 8 + (3,) * 8, 4: (2,) * 4 + (3,) * 4 + (4,) * 4 + (5,) * 4, 2: (2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9), 1: tuple(range(2, 18))}.items():
        cell = next(
            row for row in cells
            if row["mode"] == "block" and row["width"] == 1024
            and row["rank"] == 256 and row["event_block_size"] == block_size
        )
        assert tuple(cell["read_source_counts"]) == expected
        assert cell["read_source_count_histogram"] == {
            str(source_count): expected.count(source_count)
            for source_count in sorted(set(expected))
        }
        assert cell["max_read_sources"] == max(expected)
    assert all(cell["max_read_sources"] <= 17 for cell in cells)
    assert all(
        cell["event_block_size"] is None and cell["max_read_sources"] == 17
        for cell in cells
        if cell["mode"] == "full"
    )


def test_lr_rank_is_a_fixed_quarter_and_old_d1024_rank_is_historical_only():
    assert sweep.HISTORICAL_D1024_LR_RANK == 128
    assert sweep.lr_rank_for_width(1024) == 256
    assert sweep.lr_rank_for_width(4096) == 1024
    with pytest.raises(sweep.SweepError, match="D/4"):
        sweep.make_cell(mode="full", width=1024, rank=128, event_block_size=None)
    assert {
        cell["rank_relation"]
        for cell in sweep.build_matrix()
        if cell["rank"] != cell["width"]
    } == {"R=D/4"}


def test_full_l8_and_block_configs_are_unambiguous():
    full = sweep.make_cell(mode="full", width=1536, rank=384, event_block_size=None)
    block = sweep.make_cell(mode="block", width=1536, rank=1536, event_block_size=4)
    assert full["block_count"] == 16
    assert full["max_read_sources"] == 17
    assert block["block_count"] == 4
    assert block["max_read_sources"] == 5
    assert full["rank_relation"] == "R=D/4"
    assert full["head_dim"] == 64
    assert full["heads"] == 24
    assert full["read_source_counts"] == list(range(2, 18))
    assert block["read_source_counts"] == [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    assert block["rank_relation"] == "R=D"


def test_external_comparator_metadata_fails_closed_for_lr_and_unsupported_widths():
    lr = sweep.make_cell(mode="block", width=1536, rank=384, event_block_size=4)
    assert lr["competitors"]["liger"]["status"] == "not_applicable"
    assert lr["competitors"]["catswe_phase1"]["status"] == "not_applicable"
    full = sweep.make_cell(mode="full", width=1024, rank=1024, event_block_size=None)
    assert full["competitors"]["liger"]["status"] == "model_step_arm"
    assert full["competitors"]["liger"]["model_scope"] == "compiled_training_step"
    assert full["competitors"]["liger"]["adapter"] == "benchmarks.liger"
    assert full["competitors"]["catswe_phase1"]["status"] == "model_step_arm"
    assert full["competitors"]["catswe_phase1"]["model_scope"] == "compiled_training_step"
    assert full["competitors"]["catswe_phase1"]["adapter"] == "benchmarks.catswe.make_model_backend"
    assert full["competitors"]["catswe_phase1"]["operator_capability_scope"] == "standard_operator_only"
    assert full["competitors"]["catswe_phase1"]["operator_eligible"] is True
    non_power = sweep.make_cell(mode="full", width=3072, rank=3072, event_block_size=None)
    assert non_power["competitors"]["catswe_phase1"]["status"] == "not_applicable"
    assert non_power["competitors"]["catswe_phase1"]["operator_eligible"] is False
    assert "model capability rejects" in non_power["competitors"]["catswe_phase1"]["reason"]
    model_only = sweep.make_cell(mode="full", width=2048, rank=2048, event_block_size=None)
    assert model_only["model_only_admission"]["width_rank_pairs"] == [[2048, 2048]]
    model_only_lr = sweep.make_cell(mode="full", width=2048, rank=512, event_block_size=None)
    assert model_only_lr["model_only_admission"]["width_rank_pairs"] == [[2048, 2048]]


def test_worker_config_is_bf16_graph_screen_with_lr_and_standard_rank():
    cell = sweep.make_cell(mode="block", width=1024, rank=256, event_block_size=2)
    config = sweep.make_worker_config(cell)
    assert config["ranks"] == [256]
    assert config["model_config"]["block_count"] == 8
    assert config["model_config"]["layers"] == 8
    assert config["model_config"]["mode"] == "block"
    assert config["model_config"]["batch"] == 2
    assert config["model_config"]["sequence"] == 512
    assert config["model_config"]["vocab"] == 8192
    assert config["model_config"]["heads"] == 16
    assert config["model_timing"] == "cuda_graph"
    assert config["model_warmup"] == 5
    assert config["model_rounds"] == 40
    assert config["include_fla_compile"] is True
    assert config["include_fla"] is False
    assert config["fla_compile_backends"] == ["triton"]
    assert config["standard_fla_comparison"] is True
    assert config["include_fla_model"] is False
    assert config["include_liger_model"] is True
    assert config["liger_root"] == sweep.DEFAULT_REMOTE_LIGER
    assert config["include_catswe_model"] is False
    assert config["sweep_timing_contract"]["timed_tensor_hashing"] is False
    admitted = sweep.make_worker_config(
        sweep.make_cell(mode="full", width=2048, rank=512, event_block_size=None)
    )
    assert admitted["ranks"] == [512]
    assert admitted["model_only_admission"]["width_rank_pairs"] == [[2048, 2048]]
    standard = sweep.make_worker_config(
        sweep.make_cell(mode="full", width=2048, rank=2048, event_block_size=None)
    )
    assert standard["include_catswe_model"] is True


def test_manifest_is_exact_and_resume_index_is_deterministic(tmp_path: Path):
    manifest = sweep.make_manifest()
    assert sweep.validate_manifest(manifest)["cells"] == manifest["cells"]
    assert len(manifest["cells"]) == 50
    assert manifest["fixed_profile"] == {
        "layers": 8,
        "batch": 2,
        "sequence": 512,
        "vocab": 8192,
        "head_dim": 64,
        "heads_formula": "D/64",
        "ffn_formula": "11*D/4",
        "source_layout": "list",
        "timing_method": "cuda_graph",
        "dtype": "bf16_autocast",
    }
    assert manifest["ranks"] == ["D/4", "D"]
    result = sweep.run_sweep(output_dir=tmp_path, gpus=("H100",), manifest=manifest, dry_run=True)
    assert result["status"] == "dry_run"
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "index.json").is_file()
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_manifest_mutation_is_rejected(tmp_path: Path):
    manifest = sweep.make_manifest()
    forged = dict(manifest)
    forged["cells"] = list(manifest["cells"][:-1])
    with pytest.raises(sweep.SweepError, match="exactly 50"):
        sweep.validate_manifest(forged)


def test_manifest_launch_parameters_are_bound_to_supplied_manifest(tmp_path: Path):
    manifest = sweep.make_manifest()
    mismatches = {
        "seed": {"seed": manifest["seed"] + 1},
        "warmup": {"warmup": manifest["warmup"] + 1},
        "rounds": {"rounds": manifest["rounds"] + 1},
        "bootstrap": {"bootstrap_samples": manifest["bootstrap_samples"] + 1},
        "batch": {"batch": 1},
        "sequence": {"sequence": 256},
        "vocab": {"vocab": 4096},
        "remote repo": {"remote_repo": "/forged/project"},
        "FLA root": {"remote_fla_root": "/forged/fla"},
        "Liger root": {"remote_liger_root": "/forged/liger"},
        "Catswe root": {"remote_catswe_root": "/forged/catswe"},
        "remote output root": {"remote_output_root": "/forged/output"},
        "cache root": {"cache_root": "/forged/cache"},
    }
    for label, kwargs in mismatches.items():
        with pytest.raises(sweep.SweepError):
            sweep.run_sweep(
                output_dir=tmp_path / label.replace(" ", "_"),
                gpus=("H100",),
                manifest=manifest,
                dry_run=True,
                **kwargs,
            )


def test_remote_catswe_provenance_manifest_rejects_extra_keys(tmp_path: Path):
    root = tmp_path / "catswe"
    root.mkdir()
    license_bytes = b"Apache-2.0\n"
    (root / "LICENSE").write_bytes(license_bytes)
    license_sha = hashlib.sha256(license_bytes).hexdigest()
    expected_files = {"LICENSE": license_sha}
    expected = {
        "revision": "revision",
        "tree": "tree",
        "origin": "https://example.invalid/catswe.git",
        "license": "Apache-2.0",
        "license_file": "LICENSE",
        "license_sha256": license_sha,
        "source_hashes": {},
        "vendor_file_sha256": expected_files,
    }
    manifest = {
        "schema": "catswe_remote_provenance_v1",
        "source_root": "src",
        "vendor_revision": expected["revision"],
        "vendor_tree": expected["tree"],
        "vendor_origin": expected["origin"],
        "host_git_preflight": True,
        "remote_git_present": False,
        "files": expected_files,
    }
    (root / "provenance.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    assert sweep._catswe_attestation(root, expected)["status"] == "verified"

    forged = copy.deepcopy(manifest)
    forged["forged"] = 1
    (root / "provenance.json").write_text(
        json.dumps(forged, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(sweep.SweepError, match="fields are not exact"):
        sweep._catswe_attestation(root, expected)


def test_manifest_and_worker_payload_bind_project_and_catswe_provenance():
    manifest = sweep.make_manifest()
    project = manifest["project_provenance"]
    assert project["clean"] is True
    assert project["clean_required"] is True
    assert set(project["kernel_sha256"]) == set(sweep.KERNEL_PATHS)
    catswe = project["catswe"]
    assert catswe["revision"] == "ff92865e4e1b18809da7a8f0c0c5252039cded7c"
    assert catswe["tree"] == "f4f96a21dbe609044edef2fdbaf66a820c260fc0"
    assert catswe["origin"].endswith("flash-attention-residuals.git")
    assert catswe["license"] == "Apache-2.0"
    assert set(catswe["source_hashes"]).issubset(set(catswe["vendor_file_sha256"]))

    cell = sweep.make_cell(mode="full", width=1024, rank=1024, event_block_size=None)
    payload = {
        "schema": sweep.SCHEMA,
        "gpu": "H100",
        "cell": cell,
        "config": sweep.make_worker_config(cell),
        "remote_repo": sweep.DEFAULT_REMOTE_REPO,
        "remote_fla_root": sweep.DEFAULT_REMOTE_FLA,
        "remote_liger_root": sweep.DEFAULT_REMOTE_LIGER,
        "remote_catswe_root": sweep.DEFAULT_REMOTE_CATSWE,
        "triton_cache_dir": sweep.DEFAULT_CACHE_ROOT,
        "run_id": "provenance-test",
        "seed": sweep.DEFAULT_SEED,
        "warmup": sweep.DEFAULT_WARMUP,
        "rounds": sweep.DEFAULT_ROUNDS,
        "bootstrap_samples": sweep.DEFAULT_BOOTSTRAP,
        "batch": sweep.BATCH,
        "sequence": sweep.SEQUENCE,
        "vocab": sweep.VOCAB,
        "project_provenance": project,
    }
    assert sweep._validate_worker_payload(payload)["project_provenance"] == project

    forged = json.loads(json.dumps(payload))
    forged["project_provenance"]["kernel_sha256"][sweep.KERNEL_PATHS[0]] = "0" * 64
    with pytest.raises(sweep.SweepError, match="kernel hash|provenance"):
        sweep._validate_worker_payload(forged)

    forged = json.loads(json.dumps(payload))
    forged["project_provenance"]["catswe"]["source_hashes"]["src/flash_attn_res/ops/phase_1.py"] = "0" * 64
    with pytest.raises(sweep.SweepError, match="Catswe provenance"):
        sweep._validate_worker_payload(forged)


def test_resume_runner_parallelizes_devices_but_not_cells(monkeypatch, tmp_path: Path):
    active = 0
    peak = 0
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def fake_run_one_remote(**kwargs):
        nonlocal active, peak
        gpu = str(kwargs["gpu"])
        cell_id = str(kwargs["cell"]["cell_id"])
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append((gpu, cell_id))
        time.sleep(0.0005)
        with lock:
            active -= 1
        return {"status": "complete", "path": f"{gpu}/{cell_id}.json"}

    monkeypatch.setattr(sweep, "_run_one_remote", fake_run_one_remote)
    result = sweep.run_sweep(
        output_dir=tmp_path,
        gpus=("H100", "B200"),
        parallel_gpus=True,
    )
    assert result["status"] == "complete"
    assert len(calls) == 100
    assert peak == 2
    assert result["scheduling"]["gpu_parallelism"] == "one_cell_per_gpu"


def test_atomic_writer_refuses_symlink(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text("original\n", encoding="utf-8")
    output = tmp_path / "output.json"
    output.symlink_to(target)
    with pytest.raises(sweep.SweepError, match="regular file"):
        sweep.atomic_write_json(output, {"status": "complete"})
    assert target.read_text(encoding="utf-8") == "original\n"


def test_ssh_command_uses_worker_module_and_safe_payload():
    cell = sweep.make_cell(mode="full", width=1024, rank=1024, event_block_size=None)
    payload = {
        "schema": sweep.SCHEMA,
        "gpu": "H100",
        "cell": cell,
        "config": sweep.make_worker_config(cell),
        "remote_repo": "/root/project with spaces",
        "remote_fla_root": "/root/fla",
        "remote_liger_root": "/root/liger",
        "remote_catswe_root": "/root/catswe",
        "triton_cache_dir": "/root/cache",
        "run_id": "20260831T000000Z-test",
        "seed": sweep.DEFAULT_SEED,
        "warmup": sweep.DEFAULT_WARMUP,
        "rounds": sweep.DEFAULT_ROUNDS,
        "bootstrap_samples": sweep.DEFAULT_BOOTSTRAP,
        "batch": sweep.BATCH,
        "sequence": sweep.SEQUENCE,
        "vocab": sweep.VOCAB,
        "project_provenance": sweep.make_manifest()["project_provenance"],
    }
    command = sweep.ssh_command(
        gpu="H100",
        payload=payload,
        remote_output="/root/out/cell.json",
        host={"host": "example.invalid", "port": 1234, "user": "root"},
        remote_repo="/root/project with spaces",
        remote_venv="/root/venv",
    )
    text = " ".join(command)
    assert text.count("scripts.compiled_step_sweep worker") == 1
    assert "StrictHostKeyChecking=accept-new" in text
    assert "project with spaces" in text
    assert command[-1].count("--payload-b64") == 1


def test_direct_script_and_module_cli_dry_run(tmp_path: Path):
    script = Path(sweep.__file__).resolve()
    environment = dict(os.environ, PYTHONPATH="")
    direct_output = tmp_path / "direct"
    direct = subprocess.run(
        [
            sys.executable,
            str(script),
            "sweep",
            "--dry-run",
            "--gpus",
            "H100",
            "--output-dir",
            str(direct_output),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == 0, direct.stderr
    assert json.loads(direct.stdout)["status"] == "dry_run"
    assert len(json.loads((direct_output / "manifest.json").read_text())["cells"]) == 50

    module_output = tmp_path / "module"
    module = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.compiled_step_sweep",
            "sweep",
            "--dry-run",
            "--gpus",
            "H100",
            "--output-dir",
            str(module_output),
        ],
        cwd=sweep.PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert module.returncode == 0, module.stderr
    assert json.loads(module.stdout)["status"] == "dry_run"


def test_worker_payload_requires_native_liger_and_explicit_catswe_opt_in():
    cell = sweep.make_cell(mode="full", width=2048, rank=2048, event_block_size=None)
    payload = {
        "schema": sweep.SCHEMA,
        "gpu": "H100",
        "cell": cell,
        "config": sweep.make_worker_config(
            cell,
            remote_repo="/root/project",
            remote_fla_root="/root/fla",
            remote_liger_root="/root/liger",
            remote_catswe_root=sweep.DEFAULT_REMOTE_CATSWE,
        ),
        "remote_repo": "/root/project",
        "remote_fla_root": "/root/fla",
        "remote_liger_root": "/root/liger",
        "remote_catswe_root": sweep.DEFAULT_REMOTE_CATSWE,
        "triton_cache_dir": "/root/cache",
        "run_id": "run",
        "seed": sweep.DEFAULT_SEED,
        "warmup": sweep.DEFAULT_WARMUP,
        "rounds": sweep.DEFAULT_ROUNDS,
        "bootstrap_samples": sweep.DEFAULT_BOOTSTRAP,
        "batch": sweep.BATCH,
        "sequence": sweep.SEQUENCE,
        "vocab": sweep.VOCAB,
        "project_provenance": sweep.make_manifest()["project_provenance"],
    }
    assert sweep._validate_worker_payload(payload)["config"]["include_liger_model"] is True
    forged = json.loads(json.dumps(payload))
    forged["config"]["include_catswe_model"] = False
    with pytest.raises(sweep.SweepError, match="Catswe"):
        sweep._validate_worker_payload(forged)


def test_ineligible_worker_payload_does_not_require_catswe_provenance(monkeypatch, tmp_path: Path):
    cell = sweep.make_cell(mode="block", width=1536, rank=384, event_block_size=4)
    payload = _result_test_payload(cell)
    missing_catswe = str(tmp_path / "catswe-does-not-exist")
    payload["remote_catswe_root"] = missing_catswe
    payload["config"] = sweep.make_worker_config(
        cell,
        seed=payload["seed"],
        warmup=payload["warmup"],
        rounds=payload["rounds"],
        bootstrap_samples=payload["bootstrap_samples"],
        batch=payload["batch"],
        sequence=payload["sequence"],
        vocab=payload["vocab"],
        remote_repo=payload["remote_repo"],
        remote_fla_root=payload["remote_fla_root"],
        remote_liger_root=payload["remote_liger_root"],
        remote_catswe_root=missing_catswe,
    )
    payload["project_provenance"].pop("catswe")

    def catswe_contract_must_not_be_looked_up():
        pytest.fail("ineligible worker payload looked up the Catswe contract")

    monkeypatch.setattr(sweep, "_catswe_provenance_contract", catswe_contract_must_not_be_looked_up)
    validated = sweep._validate_worker_payload(payload)
    assert validated["config"]["include_catswe_model"] is False
    assert "catswe" not in validated["project_provenance"]


def _worker_run_payload(cell, tmp_path: Path):
    payload = _result_test_payload(cell)
    project_root = tmp_path / "project"
    fla_root = tmp_path / "fla"
    liger_root = tmp_path / "liger"
    catswe_root = tmp_path / "catswe"
    project_root.mkdir()
    fla_root.mkdir()
    liger_root.mkdir()
    catswe_root.mkdir()
    payload.update(
        remote_repo=str(project_root),
        remote_fla_root=str(fla_root),
        remote_liger_root=str(liger_root),
        remote_catswe_root=str(catswe_root),
        triton_cache_dir=str(tmp_path / "cache"),
    )
    payload["config"] = sweep.make_worker_config(
        cell,
        seed=payload["seed"],
        warmup=payload["warmup"],
        rounds=payload["rounds"],
        bootstrap_samples=payload["bootstrap_samples"],
        batch=payload["batch"],
        sequence=payload["sequence"],
        vocab=payload["vocab"],
        remote_repo=payload["remote_repo"],
        remote_fla_root=payload["remote_fla_root"],
        remote_liger_root=payload["remote_liger_root"],
        remote_catswe_root=payload["remote_catswe_root"],
    )
    if not payload["config"]["include_catswe_model"]:
        payload["project_provenance"].pop("catswe", None)
    return payload


def _patch_worker_runtime(monkeypatch, payload):
    base_keys = (
        "revision", "tree", "clean", "clean_required", "frozen_manifest", "kernel_sha256"
    )
    project = {
        "status": "verified",
        **{key: payload["project_provenance"][key] for key in base_keys},
    }
    monkeypatch.setattr(sweep, "_project_attestation", lambda *_args: project)
    monkeypatch.setattr(
        sweep,
        "_runtime_preflight",
        lambda gpu: {
            "status": "passed",
            "gpu": gpu,
            "name": f"NVIDIA {gpu}",
            "compute_capability": [9, 0] if gpu == "H100" else [10, 0],
            "total_memory": 192 * 2**30,
            "torch": "2.13.0+cu130",
            "cuda": "13.0",
            "triton": "3.7.1",
        },
    )
    monkeypatch.setenv("TRITON_CACHE_DIR", "test-cache-before-worker")
    monkeypatch.setenv("PYTHONPATH", "test-pythonpath-before-worker")
    return project


def test_run_worker_skips_catswe_attestation_for_ineligible_cell(monkeypatch, tmp_path: Path):
    cell = sweep.make_cell(mode="block", width=1536, rank=384, event_block_size=4)
    payload = _worker_run_payload(cell, tmp_path)
    _patch_worker_runtime(monkeypatch, payload)
    calls = []

    def catswe_must_not_be_attested(*args):
        calls.append(args)
        raise AssertionError("ineligible worker attempted Catswe attestation")

    monkeypatch.setattr(sweep, "_catswe_attestation", catswe_must_not_be_attested)
    import benchmarks.run as benchmark_run

    monkeypatch.setattr(
        benchmark_run,
        "run_suite",
        lambda config: {
            "config": copy.deepcopy(config),
            "model_timings": {"status": "incomplete"},
            "comparators": {},
        },
    )
    previous_cwd = Path.cwd()
    try:
        result = sweep.run_worker(payload, tmp_path / "worker.json")
    finally:
        os.chdir(previous_cwd)
    assert result["status"] == "failed"
    assert calls == []
    assert "catswe" not in result["provenance"]


def test_run_worker_attests_catswe_for_eligible_cell(monkeypatch, tmp_path: Path):
    cell = sweep.make_cell(mode="full", width=1024, rank=1024, event_block_size=None)
    payload = _worker_run_payload(cell, tmp_path)
    _patch_worker_runtime(monkeypatch, payload)
    calls = []
    catswe = payload["project_provenance"]["catswe"]
    attestation = {
        "status": "verified",
        "transport": "git_checkout",
        "revision": catswe["revision"],
        "tree": catswe["tree"],
        "clean": True,
        "origin": catswe["origin"],
        "license": catswe["license"],
        "license_file": catswe["license_file"],
        "license_sha256": catswe["license_sha256"],
        "source_hashes": copy.deepcopy(catswe["source_hashes"]),
        "vendor_file_sha256": copy.deepcopy(catswe["vendor_file_sha256"]),
    }

    def attest(root, expected):
        calls.append((root, expected))
        return attestation

    monkeypatch.setattr(sweep, "_catswe_attestation", attest)
    import benchmarks.run as benchmark_run

    monkeypatch.setattr(
        benchmark_run,
        "run_suite",
        lambda config: {
            "config": copy.deepcopy(config),
            "model_timings": {"status": "incomplete"},
            "comparators": {},
        },
    )
    previous_cwd = Path.cwd()
    try:
        result = sweep.run_worker(payload, tmp_path / "worker.json")
    finally:
        os.chdir(previous_cwd)
    assert result["status"] == "failed"
    assert len(calls) == 1
    assert result["provenance"]["catswe"] == attestation


def test_worker_payload_binds_every_run_suite_config_field():
    cell = sweep.make_cell(mode="block", width=2048, rank=2048, event_block_size=4)
    project = sweep.make_manifest()["project_provenance"]
    payload = {
        "schema": sweep.SCHEMA,
        "gpu": "H100",
        "cell": cell,
        "config": sweep.make_worker_config(cell),
        "remote_repo": sweep.DEFAULT_REMOTE_REPO,
        "remote_fla_root": sweep.DEFAULT_REMOTE_FLA,
        "remote_liger_root": sweep.DEFAULT_REMOTE_LIGER,
        "remote_catswe_root": sweep.DEFAULT_REMOTE_CATSWE,
        "triton_cache_dir": sweep.DEFAULT_CACHE_ROOT,
        "run_id": "config-binding-test",
        "seed": sweep.DEFAULT_SEED,
        "warmup": sweep.DEFAULT_WARMUP,
        "rounds": sweep.DEFAULT_ROUNDS,
        "bootstrap_samples": sweep.DEFAULT_BOOTSTRAP,
        "batch": sweep.BATCH,
        "sequence": sweep.SEQUENCE,
        "vocab": sweep.VOCAB,
        "project_provenance": project,
    }
    assert sweep._validate_worker_payload(payload)["config"] == payload["config"]

    mutations = {
        "project_root": lambda config: config.__setitem__("project_root", "/forged/project"),
        "vendor_root": lambda config: config.__setitem__("vendor_root", "/forged/vendor"),
        "model_config.layers": lambda config: config["model_config"].__setitem__("layers", 99),
        "model_config.ffn": lambda config: config["model_config"].__setitem__("ffn", 1),
        "model_config.block_count": lambda config: config["model_config"].__setitem__("block_count", 1),
        "model_config.mode": lambda config: config["model_config"].__setitem__("mode", "full"),
        "model_config.variant": lambda config: config["model_config"].__setitem__("variant", "unsliced"),
        "model_config.source_layout": lambda config: config["model_config"].__setitem__("source_layout", "tensor"),
        "include_fla": lambda config: config.__setitem__("include_fla", True),
        "include_fla_compile": lambda config: config.__setitem__("include_fla_compile", False),
        "standard_fla_comparison": lambda config: config.__setitem__("standard_fla_comparison", False),
        "nested extra": lambda config: config["model_config"].__setitem__("unexpected", True),
        "nested missing": lambda config: config["model_config"].pop("width"),
        "top-level config extra": lambda config: config.__setitem__("unexpected", True),
        "top-level config missing": lambda config: config.pop("scope"),
    }
    for label, mutate in mutations.items():
        forged = json.loads(json.dumps(payload))
        mutate(forged["config"])
        with pytest.raises(sweep.SweepError, match="config"):
            sweep._validate_worker_payload(forged)


def test_worker_payload_binds_explicit_custom_timing_parameters():
    cell = sweep.make_cell(mode="full", width=1024, rank=1024, event_block_size=None)
    project = sweep.make_manifest()["project_provenance"]
    parameters = {
        "seed": 123,
        "warmup": 7,
        "rounds": 41,
        "bootstrap_samples": 17,
        "batch": sweep.BATCH,
        "sequence": sweep.SEQUENCE,
        "vocab": sweep.VOCAB,
    }
    payload = {
        "schema": sweep.SCHEMA,
        "gpu": "H100",
        "cell": cell,
        "config": sweep.make_worker_config(
            cell,
            **parameters,
            remote_repo=sweep.DEFAULT_REMOTE_REPO,
            remote_fla_root=sweep.DEFAULT_REMOTE_FLA,
            remote_liger_root=sweep.DEFAULT_REMOTE_LIGER,
            remote_catswe_root=sweep.DEFAULT_REMOTE_CATSWE,
        ),
        "remote_repo": sweep.DEFAULT_REMOTE_REPO,
        "remote_fla_root": sweep.DEFAULT_REMOTE_FLA,
        "remote_liger_root": sweep.DEFAULT_REMOTE_LIGER,
        "remote_catswe_root": sweep.DEFAULT_REMOTE_CATSWE,
        "triton_cache_dir": sweep.DEFAULT_CACHE_ROOT,
        "run_id": "custom-timing-test",
        **parameters,
        "project_provenance": project,
    }
    assert sweep._validate_worker_payload(payload)["config"]["seed"] == parameters["seed"]

    forged = json.loads(json.dumps(payload))
    forged["config"]["model_rounds"] += 1
    with pytest.raises(sweep.SweepError, match="config"):
        sweep._validate_worker_payload(forged)

    forged = json.loads(json.dumps(payload))
    forged.pop("warmup")
    with pytest.raises(sweep.SweepError, match="fields"):
        sweep._validate_worker_payload(forged)


def test_worker_result_is_bound_to_payload_routes_and_report():
    payload = _result_test_payload(
        sweep.make_cell(mode="full", width=1024, rank=1024, event_block_size=None)
    )
    result = _complete_result_for_test(payload)
    assert sweep._validate_worker_result(result, payload)["status"] == "complete"

    mutations = {
        "schema": lambda forged: forged.__setitem__("schema", "forged"),
        "gpu": lambda forged: forged.__setitem__("gpu", "B200"),
        "cell": lambda forged: forged["cell"].__setitem__("mode", "block"),
        "config": lambda forged: forged["config"].__setitem__("project_root", "/forged"),
        "project provenance": lambda forged: forged["project_provenance"].__setitem__("tree", "0" * 40),
        "runtime": lambda forged: forged["runtime_preflight"].__setitem__("torch", "forged"),
        "roots": lambda forged: forged["roots"].__setitem__("remote_liger_root", "/forged"),
        "routes": lambda forged: forged["routes"]["fla"].__setitem__("compile", False),
        "eligibility": lambda forged: forged["eligibility"]["catswe_phase1"].__setitem__("model_eligible", False),
        "timing contract": lambda forged: forged["timing_contract"].__setitem__("timed_input_copy", True),
        "report config": lambda forged: forged["benchmark"]["config"].__setitem__("vendor_root", "/forged"),
        "report identity": lambda forged: forged["report_identity"].__setitem__("config_sha256", "0" * 64),
        "report route": lambda forged: forged["benchmark"]["model_timings"]["execution_schedules"].__setitem__("catswe_phase1", "phase2"),
        "report schedule substring": lambda forged: forged["benchmark"]["model_timings"]["execution_schedules"].__setitem__("catswe_phase1", "forged; no cache/prepare/merge/phase2"),
        "provenance": lambda forged: forged["provenance"]["project"].__setitem__("tree", "0" * 40),
        "nested result extra": lambda forged: forged["routes"]["model"].__setitem__("unexpected", True),
    }
    for label, mutate in mutations.items():
        forged = copy.deepcopy(result)
        mutate(forged)
        with pytest.raises(sweep.SweepError):
            sweep._validate_worker_result(forged, payload)


@pytest.mark.parametrize(
    ("location", "key"),
    [
        ("coverage", "include_liger_model"),
        ("model_timings", "include_liger_model"),
    ],
)
def test_worker_model_route_flags_are_bound_to_run_suite_locations(location, key):
    payload = _result_test_payload(
        sweep.make_cell(mode="full", width=1024, rank=1024, event_block_size=None)
    )
    result = _complete_result_for_test(payload)
    assert sweep._validate_worker_result(result, payload)["status"] == "complete"

    forged = copy.deepcopy(result)
    forged["benchmark"][location][key] = False
    with pytest.raises(sweep.SweepError, match="Liger|route"):
        sweep._validate_worker_result(forged, payload)

    forged = copy.deepcopy(result)
    forged["benchmark"][key] = True
    with pytest.raises(sweep.SweepError, match="top-level|route"):
        sweep._validate_worker_result(forged, payload)


def test_worker_runtime_binds_gpu_name_capability_and_memory_floor():
    h100 = {
        "status": "passed",
        "gpu": "H100",
        "name": "NVIDIA H100 SXM",
        "compute_capability": [9, 0],
        "total_memory": 80 * 2**30,
        "torch": "2.13.0+cu130",
        "cuda": "13.0",
        "triton": "3.7.1",
    }
    assert sweep._validate_worker_runtime(h100, "H100", allow_not_passed=False) == h100
    for key, value in {
        "name": "NVIDIA B200",
        "compute_capability": [10, 0],
        "total_memory": 1,
    }.items():
        forged = copy.deepcopy(h100)
        forged[key] = value
        with pytest.raises(sweep.SweepError):
            sweep._validate_worker_runtime(forged, "H100", allow_not_passed=False)

    b200 = copy.deepcopy(h100)
    b200.update(
        gpu="B200",
        name="NVIDIA B200 SXM",
        compute_capability=[10, 0],
        total_memory=192 * 2**30,
    )
    assert sweep._validate_worker_runtime(b200, "B200", allow_not_passed=False) == b200


def test_resume_cache_requires_full_result_and_launcher_binding(tmp_path: Path):
    cell = sweep.make_cell(mode="full", width=1024, rank=1024, event_block_size=None)
    payload = _result_test_payload(cell, run_id="cached-run")
    result = _complete_result_for_test(payload)
    result["launcher"] = _launcher_for_test(
        payload,
        tmp_path,
        remote_output_root="/root/output",
    )
    canonical = tmp_path / "h100" / f"{cell['cell_id']}.json"
    sweep.atomic_write_json(canonical, result)
    assert sweep._is_complete_cell(
        canonical,
        cell["cell_id"],
        expected_payload=payload,
        expected_output_root=tmp_path,
        expected_remote_output_root="/root/output",
    )

    minimal = {
        "schema": sweep.SCHEMA,
        "status": "complete",
        "cell": {"cell_id": cell["cell_id"]},
    }
    sweep.atomic_write_json(canonical, minimal)
    assert not sweep._is_complete_cell(
        canonical,
        cell["cell_id"],
        expected_payload=payload,
        expected_output_root=tmp_path,
        expected_remote_output_root="/root/output",
    )

    forged = copy.deepcopy(result)
    forged["runtime_preflight"]["torch"] = "forged"
    sweep.atomic_write_json(canonical, forged)
    assert not sweep._is_complete_cell(
        canonical,
        cell["cell_id"],
        expected_payload=payload,
        expected_output_root=tmp_path,
        expected_remote_output_root="/root/output",
    )


def test_remote_launcher_retains_forged_complete_result_as_failure(monkeypatch, tmp_path: Path):
    cell = sweep.make_cell(mode="full", width=1024, rank=1024, event_block_size=None)
    payload = _result_test_payload(cell, run_id="fixed-run")
    forged = _complete_result_for_test(payload)
    forged["config"]["project_root"] = "/forged/project"
    monkeypatch.setattr(sweep, "_new_run_id", lambda: "fixed-run")
    real_subprocess_run = sweep.subprocess.run

    def fake_scp(command, **kwargs):
        if command and command[0] == "scp":
            Path(command[-1]).write_text(json.dumps(forged), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        return real_subprocess_run(command, **kwargs)

    monkeypatch.setattr(sweep.subprocess, "run", fake_scp)

    def fake_ssh(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "")

    result = sweep._run_one_remote(
        cell=cell,
        gpu="H100",
        output_root=tmp_path,
        host={"host": "example.invalid", "port": 1234, "user": "root"},
        remote_repo=sweep.DEFAULT_REMOTE_REPO,
        remote_venv="/root/venv",
        remote_fla_root=sweep.DEFAULT_REMOTE_FLA,
        remote_liger_root=sweep.DEFAULT_REMOTE_LIGER,
        remote_catswe_root=sweep.DEFAULT_REMOTE_CATSWE,
        project_provenance=payload["project_provenance"],
        remote_output_root="/root/output",
        cache_root=sweep.DEFAULT_CACHE_ROOT,
        seed=sweep.DEFAULT_SEED,
        warmup=sweep.DEFAULT_WARMUP,
        rounds=sweep.DEFAULT_ROUNDS,
        bootstrap_samples=sweep.DEFAULT_BOOTSTRAP,
        batch=sweep.BATCH,
        sequence=sweep.SEQUENCE,
        vocab=sweep.VOCAB,
        timeout_s=10,
        runner=fake_ssh,
    )
    assert result["status"] == "failed"
    failure = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert failure["failure"]["type"] == "WorkerResultValidationError"
    assert not (tmp_path / "H100" / f"{cell['cell_id']}.json").exists()
