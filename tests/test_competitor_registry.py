"""CPU and static checks for the matched competitor capability surface."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from benchmarks.comparator_registry import (
    COMPETITOR_CAPABILITIES,
    GLUON_COMPILE_ENVELOPE,
    capability_for,
    eligibility_for,
)
from benchmarks.competitor_protocol import (
    HARDWARE_ORDER,
    ProtocolError,
    paired_orders,
    comparison_plan,
    competitor_capabilities,
    load_config,
    planned_comparison_cells,
    retain_failure,
    validate_raw_samples,
)
from benchmarks import catswe, liger
from benchmarks import hydra


def test_registry_is_dependency_free_and_contains_all_reviewed_adapters():
    source = Path(__file__).resolve().parents[1] / "benchmarks" / "comparator_registry.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in imported
    assert set(COMPETITOR_CAPABILITIES) == {
        "native_fla_triton_checkpoint1",
        "native_fla_gluon",
        "native_fla_triton_checkpoint0",
        "liger",
        "catswe_phase1",
        "manish_hydra_2p",
    }


def test_reviewed_vendor_identity_and_capability_limits_are_sealed():
    fla = capability_for("native_fla_triton_checkpoint1")
    assert fla["rank_scope"] == "R=D"
    assert fla["max_sources"] == 129
    assert fla["max_width"] == 8192
    assert fla["revision"] == "5e02dd3a7651f5f2797eb8b12bbec401826031e1"
    assert fla["origin"] == "https://github.com/fla-org/flash-linear-attention.git"
    assert fla["license"] == "MIT"
    assert fla["package_sha256"] == (
        "2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781"
    )
    liger = capability_for("liger")
    assert liger["revision"] == "000be60929938fd1358e03524c6ab398b6d421bd"
    assert liger["tree"] == "746af1fc03014cf47cad895d01cf0d23fddf5e75"
    assert liger["origin"] == "https://github.com/linkedin/Liger-Kernel.git"
    assert liger["max_sources"] == 32
    assert liger["source_hashes"]["src/liger_kernel/ops/attn_res.py"] == (
        "57da6fed98f794088b2a56223e6c7ef9fc920824f0c483cb0ef0b5a343dab0b1"
    )
    catswe_capability = capability_for("catswe_phase1")
    assert catswe_capability["dtypes"] == ["bf16"]
    assert catswe_capability["max_program_elements"] == 1_048_576
    assert catswe_capability["requires_power_of_two_width"] is True
    assert catswe_capability["tree"] == (
        "f4f96a21dbe609044edef2fdbaf66a820c260fc0"
    )
    hydra = capability_for("manish_hydra_2p")
    assert hydra["max_width"] == 8192
    assert hydra["timing_max_width"] == 256
    assert hydra["origin"] == "https://github.com/manishklach/attnres-kernel-lab.git"
    assert hydra["license"] == "MIT"
    assert hydra["supports_per_read_block"] is False
    assert hydra["block_scope"] == "external_block_panel"


def test_gluon_compile_envelope_is_explicit_and_matches_the_pinned_rule():
    gluon = capability_for("native_fla_gluon")
    envelope = gluon["compile_envelope"]
    assert envelope == GLUON_COMPILE_ENVELOPE
    assert envelope == {
        "padded_width_rule": "BD=next_power_of_two(D)",
        "max_padded_width": 4096,
        "source_width_product_rule": "S*BD",
        "max_source_width_product": 2**18,
        "checkpoint1_static_work_rule": "33*S*BD",
        "checkpoint1_static_work_multiplier": 33,
        "max_checkpoint1_static_work": 33 * (2**18),
    }

    for sources, width in ((1, 128), (9, 2048), (33, 2048)):
        result = eligibility_for(
            "native_fla_gluon",
            mode="standard_operator",
            rank=width,
            width=width,
            source_count=sources,
            dtype="bf16",
            timing=True,
        )
        assert result["eligible"] is True
        padded_width = 1 << (width - 1).bit_length()
        assert result["padded_width"] == padded_width
        assert result["source_width_product"] == sources * padded_width
        assert result["static_work_score"] == 33 * sources * padded_width

    oversized = eligibility_for(
        "native_fla_gluon",
        mode="standard_operator",
        rank=8192,
        width=8192,
        source_count=129,
        dtype="bf16",
        timing=True,
    )
    assert oversized["eligible"] is False
    assert "BD=8192" in oversized["reason"]
    assert "maximum padded width is 4096" in oversized["reason"]


def test_gluon_smoke_cases_are_fresh_r_equals_d_envelope_probes_and_large_case_is_na():
    config = load_config()
    smoke = config["operator_cases"]["smoke"]
    expected = {
        (1, 2, 128, 128, "bf16"),
        (9, 32, 2048, 2048, "bf16"),
        (33, 8, 2048, 2048, "bf16"),
    }
    actual = {
        (case["S"], case["N"], case["D"], case["R"], case["dtype"])
        for case in smoke
    }
    assert expected <= actual
    plan = comparison_plan(config)
    large = [
        row
        for row in plan["not_applicable"]
        if row["competitor"] == "native_fla_gluon"
        and row["scope"] == "operator"
        and row["S"] == 129
        and row["D"] == 8192
        and row["R"] == 8192
    ]
    assert large
    assert all("compile envelope" in row["eligibility_reason"] for row in large)
    assert all(not row["eligible_denominator"] for row in large)


def test_intersection_marks_liger_full_and_block_correctly():
    full = eligibility_for(
        "liger", mode="full", rank=1024, width=1024, source_count=49, dtype="bf16"
    )
    block = eligibility_for(
        "liger",
        mode="block_per_read",
        rank=1024,
        width=1024,
        read_source_count=9,
        dtype="bf16",
    )
    assert full["eligible"] is False
    assert "S<=32" in full["reason"]
    assert block["eligible"] is True


def test_external_scopes_and_denominator_gates_are_explicit():
    assert eligibility_for(
        "catswe_phase1",
        mode="standard_operator",
        rank=128,
        width=128,
        source_count=9,
        dtype="bf16",
    )["eligible"] is True
    assert eligibility_for(
        "catswe_phase1",
        mode="block_per_read",
        rank=128,
        width=128,
        read_source_count=9,
        dtype="bf16",
    )["eligible"] is False
    assert eligibility_for(
        "catswe_phase1",
        mode="standard_operator",
        rank=128,
        width=128,
        source_count=9,
        dtype="fp32",
    )["eligible"] is False
    assert eligibility_for(
        "manish_hydra_2p",
        mode="block_panel",
        rank=256,
        width=256,
        read_source_count=9,
        dtype="bf16",
    )["eligible"] is True
    assert eligibility_for(
        "manish_hydra_2p",
        mode="block_panel",
        rank=257,
        width=257,
        read_source_count=9,
        dtype="bf16",
    )["eligible"] is True
    assert eligibility_for(
        "manish_hydra_2p",
        mode="block_panel",
        rank=512,
        width=512,
        read_source_count=9,
        dtype="bf16",
        timing=True,
    )["eligible"] is False
    assert eligibility_for(
        "catswe_phase1",
        mode="standard_operator",
        rank=3000,
        width=3000,
        source_count=1,
        dtype="bf16",
        timing=True,
    )["eligible"] is False
    assert catswe.timing_eligible(8192) is True
    assert catswe.timing_eligible(4095) is False
    diagnostic = eligibility_for(
        "native_fla_triton_checkpoint0",
        mode="full",
        rank=128,
        width=128,
        source_count=4,
        dtype="bf16",
    )
    assert diagnostic["eligible"] is False
    assert diagnostic["eligible_denominator"] is False


def test_catswe_static_program_envelope_is_planned_before_cuda_execution():
    config_capability = load_config()["competitors"]["catswe_phase1"]["capability"]
    registry_capability = capability_for("catswe_phase1")
    assert catswe.MAX_PROGRAM_ELEMENTS == registry_capability["max_program_elements"]
    assert config_capability["max_program_elements"] == catswe.MAX_PROGRAM_ELEMENTS
    assert config_capability["requires_power_of_two_width"] is True

    common = {
        "mode": "standard_operator",
        "rank": 8192,
        "width": 8192,
        "dtype": "bf16",
        "timing": True,
    }
    rejected = eligibility_for("catswe_phase1", source_count=129, **common)
    assert rejected["eligible"] is False
    assert "2097152" in rejected["reason"]
    assert "1048576" in rejected["reason"]

    assert eligibility_for(
        "catswe_phase1",
        mode="standard_operator",
        rank=4096,
        width=4096,
        source_count=129,
        dtype="bf16",
        timing=True,
    )["eligible"] is True
    assert eligibility_for(
        "catswe_phase1", source_count=128, **common
    )["eligible"] is True
    assert eligibility_for(
        "catswe_phase1", source_count=65, **common
    )["eligible"] is True
    assert eligibility_for(
        "catswe_phase1",
        mode="standard_operator",
        rank=4095,
        width=4095,
        source_count=129,
        dtype="bf16",
        timing=True,
    )["eligible"] is False

    accepted_values = torch.empty((129, 1, 4096), dtype=torch.bfloat16)
    accepted_query = torch.empty((4096,), dtype=torch.bfloat16)
    assert catswe.CatsweBackend._check(
        accepted_values, accepted_query, require_cuda=False
    ) == 4096
    rejected_values = torch.empty((129, 1, 8192), dtype=torch.bfloat16)
    rejected_query = torch.empty((8192,), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="1048576-element limit"):
        catswe.CatsweBackend._check(
            rejected_values, rejected_query, require_cuda=False
        )


def test_catswe_model_capability_is_separate_from_operator_protocol():
    operator = capability_for("catswe_phase1")
    model = capability_for("catswe_phase1", scope="model")
    assert operator["model_scope"] == "standard_operator_only"
    assert model["model_scope"] == "compiled_training_step"
    assert model["adapter"] == "benchmarks.catswe.make_model_backend"
    assert model["supports_per_read_block"] is True
    assert set(model["vendor_file_sha256"]) == {
        "LICENSE",
        "pyproject.toml",
        "src/flash_attn_res/__init__.py",
        "src/flash_attn_res/kernels/__init__.py",
        "src/flash_attn_res/kernels/configs.py",
        "src/flash_attn_res/kernels/phase_1.py",
        "src/flash_attn_res/kernels/phase_2.py",
        "src/flash_attn_res/kernels/reduce.py",
        "src/flash_attn_res/ops/__init__.py",
        "src/flash_attn_res/ops/phase_1.py",
        "src/flash_attn_res/ops/phase_2.py",
    }
    common = {
        "dtype": "bf16",
        "rank": 1024,
        "width": 1024,
        "timing": True,
    }
    assert eligibility_for(
        "catswe_phase1", scope="operator", mode="block_per_read", read_source_count=9, **common
    )["eligible"] is False
    assert eligibility_for(
        "catswe_phase1", scope="model", mode="block_per_read", read_source_count=9, **common
    )["eligible"] is True
    assert eligibility_for(
        "catswe_phase1", scope="model", mode="full", source_count=17, **common
    )["eligible"] is True
    assert eligibility_for(
        "catswe_phase1", scope="model", mode="full", source_count=17,
        rank=256, width=1024, dtype="bf16", timing=True,
    )["eligible"] is False
    assert eligibility_for(
        "catswe_phase1", scope="model", mode="full", source_count=17,
        rank=1536, width=1536, dtype="bf16", timing=True,
    )["eligible"] is False


def test_catswe_model_capability_does_not_change_operator_plan():
    config = load_config()
    operator_names = set(competitor_capabilities())
    assert "catswe_phase1" in operator_names
    assert set(capability_for("catswe_phase1")) != set(capability_for("catswe_phase1", scope="model"))
    plan = comparison_plan(config)
    operator_rows = [
        row for row in (*plan["planned"], *plan["not_applicable"])
        if row["scope"] == "operator"
    ]
    assert operator_rows
    assert all(row["scope"] == "operator" for row in operator_rows)
    catswe_operator_rows = [
        row for row in operator_rows if row["competitor"] == "catswe_phase1"
    ]
    assert catswe_operator_rows
    assert all(row["arm"] == "standard_operator" for row in catswe_operator_rows)
    assert all(row["mode"] == "standard_operator" for row in catswe_operator_rows)


def test_planned_cells_are_only_applicability_intersections():
    config = load_config()
    rows = planned_comparison_cells(config)
    assert rows
    assert all("eligible" in row and "eligibility_reason" in row for row in rows)
    assert all("latency_ms" not in row for row in rows)
    assert all(row["status"] == "planned" and row["eligible"] for row in rows)
    assert all(row["eligible_denominator"] for row in rows)
    assert all(
        int(row["D"]) & (int(row["D"]) - 1) == 0
        and (1 << (int(row["S"]) - 1).bit_length()) * int(row["D"])
        <= 1_048_576
        for row in rows
        if row["competitor"] == "catswe_phase1"
    )
    plan = comparison_plan(config)
    assert plan["not_applicable"]
    liger_full = [
        row
        for row in plan["not_applicable"]
        if row["competitor"] == "liger"
        and row["arm"] == "full"
        and row["rank"] == 1024
    ]
    assert liger_full and all(not row["eligible_denominator"] for row in liger_full)
    catswe_model = [
        row
        for row in plan["not_applicable"]
        if row["competitor"] == "catswe_phase1" and row["arm"] == "block_per_read"
    ]
    assert catswe_model and all(not row["eligible"] for row in catswe_model)
    planned_block = [
        row
        for row in plan["planned"]
        if row["arm"] == "block_per_read" and row["rank"] == row["D"]
    ]
    assert planned_block
    assert {row["competitor"] for row in planned_block} == {
        "native_fla_triton_checkpoint1",
        "native_fla_gluon",
        "liger",
    }
    assert all(row["read_source_count"] == 9 for row in planned_block)
    assert not any(
        row["scope"] == "model" and row["arm"] == "standard_operator"
        for row in (*plan["planned"], *plan["not_applicable"])
    )
    assert not any(
        row["scope"] == "model" and row["competitor"] == "catswe_phase1"
        and row["eligible"]
        for row in (*plan["planned"], *plan["not_applicable"])
    )
    diagnostic = [
        row
        for row in plan["not_applicable"]
        if row["competitor"] == "native_fla_triton_checkpoint0"
    ]
    assert diagnostic and all(not row["eligible_denominator"] for row in diagnostic)
    catswe_oversized = [
        row
        for row in plan["not_applicable"]
        if row["competitor"] == "catswe_phase1"
        and row.get("S") == 129
        and row.get("D") == 8192
    ]
    assert catswe_oversized
    assert all("1048576" in row["eligibility_reason"] for row in catswe_oversized)


def test_planned_geometry_is_explicit_and_hydra_has_only_eligible_small_d_smoke():
    config = load_config()
    all_rows = planned_comparison_cells(config, include_not_applicable=True)
    assert all(
        {"scope", "S", "N", "D", "R", "dtype"}.issubset(row)
        for row in all_rows
    )
    operator_rows = [row for row in all_rows if row["scope"] == "operator"]
    assert operator_rows
    assert all(row["operator_case_id"].startswith("operator_") for row in operator_rows)
    hydra = [
        row
        for row in operator_rows
        if row["competitor"] == "manish_hydra_2p"
        and row["D"] <= 256
        and row["R"] == row["D"]
    ]
    assert hydra and all(row["status"] == "planned" for row in hydra)
    assert {
        (row["dtype"], row["S"], row["N"], row["D"])
        for row in hydra
    } >= {
        ("bf16", 1, 2, 128),
        ("bf16", 9, 3, 256),
    }
    assert all(
        not row["eligible"]
        for row in all_rows
        if row["R"] < row["D"] and row["competitor"] in COMPETITOR_CAPABILITIES
    )
    liger = [row for row in all_rows if row["competitor"] == "liger" and row["eligible"]]
    assert liger and all(
        (row["read_source_count"] or row["source_count"]) <= 32 for row in liger
    )


def test_registry_rejects_missing_or_malformed_mode_geometry_and_flags():
    valid = {
        "mode": "standard_operator",
        "dtype": "bf16",
        "rank": 128,
        "width": 128,
        "source_count": 9,
    }
    assert eligibility_for("catswe_phase1", valid)["eligible"] is True
    for field in ("mode", "dtype", "rank", "width", "source_count"):
        malformed = dict(valid)
        malformed.pop(field)
        assert eligibility_for("catswe_phase1", malformed)["eligible"] is False
    for field, value in {
        "mode": None,
        "dtype": 1,
        "rank": True,
        "width": 128.0,
        "source_count": "9",
        "rank_equals_width": 1,
        "external_route": 0,
        "timing": "yes",
    }.items():
        malformed = dict(valid)
        malformed[field] = value
        assert eligibility_for("catswe_phase1", malformed)["eligible"] is False
    assert eligibility_for(
        "liger",
        mode="block_per_read",
        dtype="bf16",
        rank=128,
        width=128,
        source_count=49,
    )["eligible"] is False


def _raw_matrix(*, statuses=("ok", "not_applicable"), seed=20260827):
    names = ("baseline", "optional")
    orders = paired_orders(names, rounds=2, seed=seed)
    rows = []
    for round_index, order in enumerate(orders):
        for order_index, arm in enumerate(order):
            status = statuses[names.index(arm)]
            rows.append(
                {
                    "seed": seed,
                    "gpu": HARDWARE_ORDER[0],
                    "round_index": round_index,
                    "order_index": order_index,
                    "input_hash": f"input-{round_index}",
                    "arm": arm,
                    "status": status,
                    "latency_ms": 1.0 if status == "ok" else None,
                    "failure_phase": None if status == "ok" else "eligibility",
                    "failure_reason": None if status == "ok" else "outside declared scope",
                    "failure_at_round": None if status == "ok" else round_index,
                    "failure_at_order": None if status == "ok" else order_index,
                }
            )
    return rows


def test_raw_validator_enforces_plan_ok_na_pairing_and_failure_provenance():
    rows = _raw_matrix()
    assert validate_raw_samples(
        rows,
        ("baseline", "optional"),
        rounds=2,
        seed=20260827,
        gpu=HARDWARE_ORDER[0],
        planned_eligibility={"baseline": True, "optional": False},
    ) == rows
    bad = _raw_matrix(statuses=("not_applicable", "not_applicable"))
    with pytest.raises(ProtocolError, match="must have status 'ok'"):
        validate_raw_samples(
            bad,
            ("baseline", "optional"),
            rounds=2,
            planned_eligibility={"baseline": True, "optional": False},
        )
    bad = _raw_matrix(statuses=("ok", "ok"))
    with pytest.raises(ProtocolError, match="predeclared ineligible"):
        validate_raw_samples(
            bad,
            ("baseline", "optional"),
            rounds=2,
            planned_eligibility={"baseline": True, "optional": False},
        )
    with pytest.raises(ProtocolError, match="require predeclared planned eligibility"):
        validate_raw_samples(
            _raw_matrix(),
            ("baseline", "optional"),
            rounds=2,
        )
    bad = _raw_matrix(statuses=("ok", "ok"))
    bad[0]["arm"], bad[1]["arm"] = bad[1]["arm"], bad[0]["arm"]
    with pytest.raises(ProtocolError, match="does not match paired_orders"):
        validate_raw_samples(bad, ("baseline", "optional"), rounds=2)
    bad = _raw_matrix()
    bad[0]["failure_reason"] = ""
    with pytest.raises(ProtocolError, match="nonempty failure provenance"):
        validate_raw_samples(
            bad,
            ("baseline", "optional"),
            rounds=2,
            planned_eligibility={"baseline": True, "optional": False},
        )
    with pytest.raises(ProtocolError, match="failure seed"):
        retain_failure(
            "optional",
            0,
            gpu=HARDWARE_ORDER[0],
            order_index=0,
            input_hash="input-0",
            failure_reason="outside declared scope",
        )


def test_protocol_config_and_registry_capabilities_have_same_names():
    config = load_config()
    assert set(config["competitors"]) == set(competitor_capabilities())
    assert config["schedule"]["timing_boundary"] == {
        "operator": "operator_forward_backward",
        "model": "complete_compiled_training_step",
    }
    json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "matched_competitor_benchmark.schema.json").read_text(
            encoding="utf-8"
        )
    )


def test_oracle_contract_has_no_impossible_key_gradient_and_excludes_inapplicable_rows():
    root = Path(__file__).resolve().parents[1]
    config_text = (root / "configs" / "matched_competitor_benchmark.json").read_text(
        encoding="utf-8"
    )
    schema_text = (root / "configs" / "matched_competitor_benchmark.schema.json").read_text(
        encoding="utf-8"
    )
    impossible_check = "keys_" + "gradient"
    assert impossible_check not in config_text
    assert impossible_check not in schema_text
    config = load_config()
    assert config["oracle"]["checks"] == [
        "output",
        "values_gradient",
        "query_gradient",
    ]
    plan = comparison_plan(config)
    assert plan["not_applicable"]
    assert all(
        row["status"] == "not_applicable"
        and row["eligible"] is False
        and row["eligible_denominator"] is False
        for row in plan["not_applicable"]
    )


def test_external_adapters_do_not_expose_cached_public_api():
    catswe_source = Path(catswe.__file__).read_text(encoding="utf-8")
    liger_source = Path(liger.__file__).read_text(encoding="utf-8")
    assert "CatsweCache" not in catswe_source
    assert "def prepare(" not in catswe_source
    assert "def merge(" not in catswe_source
    assert "def model_backend(" not in catswe_source
    assert "backend.prepare" not in liger_source
    assert "backend.merge" not in liger_source
    backend = catswe.make_cpu_mock_backend()
    assert not hasattr(backend, "prepare")
    assert not hasattr(backend, "merge")
    assert "flash_attn_res.ops.phase_2" not in catswe_source


def test_hydra_is_explicitly_external_block_panel_only(tmp_path):
    assert hydra.HydraBackend.supports_per_read_block is False
    metadata = hydra.vendor_metadata(vendor_root=tmp_path / "missing-hydra")
    assert metadata["supports_per_read_block"] is False
    assert metadata["supports_external_block_panel"] is True
    assert metadata["block_scope"] == "external_block_panel"
    assert metadata["license_file"] == hydra.LICENSE
    assert metadata["expected_license_sha256"] == hydra.LICENSE_SHA256


def test_explicit_roots_are_authoritative_and_module_origin_guards_exist(monkeypatch, tmp_path):
    catswe_source = Path(catswe.__file__).read_text(encoding="utf-8")
    liger_source = Path(liger.__file__).read_text(encoding="utf-8")
    fla_source = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "competitors.py"
    ).read_text(encoding="utf-8")
    hydra_source = Path(hydra.__file__).read_text(encoding="utf-8")
    assert "def _all_loaded_origins_ok" in catswe_source
    assert "def _all_loaded_origins_ok" in liger_source
    assert "def _all_loaded_origins_ok" in fla_source
    assert "def _all_loaded_origins_ok" in hydra_source
    monkeypatch.setenv("CATSWE_ROOT", str(tmp_path / "first"))
    monkeypatch.setenv("FLASH_ATTENTION_RESIDUALS_ROOT", str(tmp_path / "second"))
    assert catswe._roots(tmp_path, None) == [(tmp_path / "first").resolve()]
    monkeypatch.setenv("HYDRA_ROOT", str(tmp_path / "first-hydra"))
    monkeypatch.setenv("MANISH_ATTNRES_ROOT", str(tmp_path / "second-hydra"))
    assert hydra._candidate_roots(tmp_path, None) == [
        (tmp_path / "first-hydra").resolve()
    ]
    catswe_link = tmp_path / "catswe-link"
    catswe_link.symlink_to(tmp_path / "first", target_is_directory=True)
    assert catswe._roots(tmp_path, catswe_link) == []
    hydra_link = tmp_path / "hydra-link"
    hydra_link.symlink_to(tmp_path / "first-hydra", target_is_directory=True)
    assert hydra._candidate_roots(tmp_path, hydra_link) == []
