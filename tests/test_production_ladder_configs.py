"""CPU-only contract checks for the resident rank-ladder recipes."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import pytest

from benchmarks.model import CANONICAL_MAX_RANK_STATE_PROTOCOL, TrainingConfig
from benchmarks.run import _balanced_orders, _model_config, load_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
MODES = ("full", "block")
SOURCE_GATE_REVISION = "8ddb0bbaf184663703ded65b45839fddd1c429fc"
SOURCE_GATE_TREE = "a91fb6d7662c36652bf648aa2e8170c90887bc1a"
SOURCE_HASHES = {
    "src/attnres/_kernels/fixed_tail_sources.py":
        "1373614c93d7291ad96697b1b8ff627120590b75f63f7e38bd65d50b19fcfb4a",
    "src/attnres/_kernels/fla_full_sources.py":
        "8749c72c4714145214e33e8bc7d37f57b47a79b67f2e83044205db72cda416fa",
}
FLA_CHECKOUT = {
    "environment": "ATTNRES_FLA_DIR",
    "layout": "clean checkout containing fla/",
    "revision": "5e02dd3a7651f5f2797eb8b12bbec401826031e1",
    "package_sha256": "2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781",
    "required_clean": True,
}
GEOMETRY = {
    "layers": 24,
    "width": 1024,
    "heads": 16,
    "ffn": 2816,
    "batch": 2,
    "sequence": 2048,
    "vocab": 32768,
    "block_count": 8,
}


def _load(name: str) -> dict:
    parsed = json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _mapping_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _mapping_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mapping_keys(child)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_common_config(config: dict, mode: str, protocol: dict, source_root: Path) -> None:
    model = config["model_config"]
    metadata = config["production_ladder"]

    assert config["schema"] == "attnres.production_ladder_config.v1"
    assert config["scope"] == "primary"
    assert config["phases"] == ["model"]
    assert config["variant"] == model["variant"] == "sliced"
    assert config["mode"] == model["mode"] == mode
    assert config["ranks"] == RANKS
    assert all(rank in protocol["ranks"] for rank in RANKS)
    assert model["source_layout"] == metadata["source_layout"] == "list"
    assert {key: model[key] for key in GEOMETRY} == GEOMETRY

    assert config["pairwise"] is False
    assert config["reference_timing"] is False
    assert config["model_timing"] == "cuda_graph"
    assert config["model_warmup"] == protocol["warmup"] == 10
    assert config["model_rounds"] == protocol["rounds"] == 120
    assert config["accumulation"] == 1
    assert config["model_state_protocol"] == CANONICAL_MAX_RANK_STATE_PROTOCOL

    assert config["include_fla"] is False
    assert config["include_fla_model"] is False
    assert config["include_fla_compile"] is True
    assert config["fla_compile_backends"] == ["triton"]
    assert config["standard_fla_comparison"] is True
    assert config["include_baseline"] is False
    assert config["include_packed_comparison"] is False
    assert metadata["cached_block"] is False
    assert metadata["block_path"] == "per_read_public_attnres"
    assert metadata["base_revision"] == SOURCE_GATE_REVISION
    assert metadata["base_tree"] == SOURCE_GATE_TREE
    assert metadata["state_protocol"] == CANONICAL_MAX_RANK_STATE_PROTOCOL
    assert metadata["input_protocol"] == "shared_per_sample_timed_inputs_v1"
    assert metadata["priority_first_gate"]["ranks"] == [16, 64, 128, 512, 1024]
    assert metadata["priority_first_gate"]["manifest_key"] == "priority_first_gate"
    assert metadata["fla_anchor"] == {
        "implementation": "triton",
        "checkpoint_level": 1,
        "rank": 1024,
        "scope": "R=D anchor only",
    }
    assert metadata["fla_checkout"] == FLA_CHECKOUT
    assert metadata["resident_candidate"]["selection"] == (
        "active standard-source autotune candidate; evaluator and timing contract unchanged"
    )
    assert metadata["resident_candidate"]["identity_scope"] == (
        "current_autotuned_release_candidate"
    )
    for path, digest in SOURCE_HASHES.items():
        assert metadata["resident_candidate"][Path(path).name.replace(".py", "_sha256")] == digest
        assert _sha256(source_root / path) == digest

    keys = {key.lower() for key in _mapping_keys(config)}
    assert "block_execution" not in keys
    assert "include_per_read" not in keys
    assert "projected" not in keys
    assert model["source_layout"] != "packed"

    effective = _model_config(protocol, config, "primary")
    for rank in RANKS:
        TrainingConfig(**dict(effective, rank=rank))
    TrainingConfig(**dict(effective, rank=1024, variant="standard"))


@pytest.mark.parametrize("mode", MODES)
def test_mode_configs_match_the_resident_current_evaluator_contract(mode, historical_release_root):
    protocol, _ = load_protocol(ROOT)
    _assert_common_config(_load(f"production_ladder_{mode}.json"), mode, protocol, historical_release_root)


@pytest.mark.parametrize("mode", MODES)
def test_mode_schedule_is_balanced_for_every_split_shape(mode):
    config = _load(f"production_ladder_{mode}.json")
    manifest = _load("production_ladder_split_manifest.json")
    for group in manifest["split_groups"]:
        arms = [f"kernel_rank_{rank}" for rank in group["ranks"]]
        arms.append("fla_triton_compile_standard_rank_1024")
        orders = _balanced_orders(arms, config["model_rounds"], random.Random(20260827))
        assert len(orders) == 120
        assert all(Counter(order) == Counter(arms) for order in orders)
        positions = Counter(
            (arm, position)
            for order in orders
            for position, arm in enumerate(order)
        )
        assert set(positions.values()) == {120 // len(arms)}


def test_priority_first_gate_is_one_shared_input_process_per_mode_and_hardware():
    manifest = _load("production_ladder_split_manifest.json")
    first_gate = manifest["priority_first_gate"]
    assert first_gate["ranks"] == [16, 64, 128, 512, 1024]
    assert first_gate["adjacent_rank_order_pairs"] == [
        [16, 64],
        [64, 128],
        [128, 512],
        [512, 1024],
    ]
    assert first_gate["base_configs"] == {
        "full": "production_ladder_full.json",
        "block": "production_ladder_block.json",
    }
    assert first_gate["mode_order"] == ["full", "block"]
    assert first_gate["hardware_order"] == ["H100!", "B200"]
    assert first_gate["job_count"] == 4
    assert first_gate["pairwise"] is False
    assert "all five selected kernel ranks" in first_gate["shared_input_rule"]

    arms = [f"kernel_rank_{rank}" for rank in first_gate["ranks"]]
    arms.append("fla_triton_compile_standard_rank_1024")
    orders = _balanced_orders(arms, 120, random.Random(20260827))
    assert all(Counter(order) == Counter(arms) for order in orders)
    positions = Counter(
        (arm, position)
        for order in orders
        for position, arm in enumerate(order)
    )
    assert set(positions.values()) == {20}


def test_split_manifest_covers_each_requested_adjacent_edge_once():
    manifest = _load("production_ladder_split_manifest.json")
    expected_edges = [
        list(edge) for edge in zip(RANKS, RANKS[1:])
    ]
    groups = manifest["split_groups"]

    assert manifest["schema"] == "attnres.production_ladder_split_manifest.v1"
    assert manifest["source_gate_revision"] == SOURCE_GATE_REVISION
    assert manifest["source_gate_tree"] == SOURCE_GATE_TREE
    assert manifest["historical_source_gate"] == {
        "status": "historical_pre_autotune",
        "revision": "a927c8d9c3c802637a4d6cb2247378bfd6cee3bb",
        "tree": "927921eb915b6eb82c43c321574c915dba9d3a2e",
        "fixed_tail_sources_sha256":
            "20fa0206fcbf6cc6b28a2973ac280575b6e8e378b09e0903449bf423d9812196",
        "fla_full_sources_sha256":
            "2cd7ac89b15faeb13640bff4a7948e437453b69446bfc8c7922511e341843e10",
    }
    assert manifest["fla_checkout"] == FLA_CHECKOUT
    assert manifest["hardware_order"] == ["H100!", "B200"]
    assert manifest["rank_ladder"] == RANKS
    assert [group["id"] for group in groups] == [f"s{n:02d}" for n in range(1, 7)]
    assert manifest["job_matrix"]["job_count"] == 28
    assert manifest["job_matrix"]["priority_first_gate_job_count"] == 4
    assert manifest["job_matrix"]["ladder_split_job_count"] == 24

    covered_edges = []
    covered_ranks = set()
    for group in groups:
        ranks = group["ranks"]
        assert ranks == sorted(ranks)
        assert len(ranks) == group["max_kernel_ranks"]
        assert len(ranks) <= 3
        assert group["adjacent_edges"] == [
            list(edge) for edge in zip(ranks, ranks[1:])
        ]
        covered_edges.extend(group["adjacent_edges"])
        covered_ranks.update(ranks)
    assert covered_edges == expected_edges
    assert covered_ranks == set(RANKS)

    assert manifest["job_matrix"]["required_rounds"] == 120
    assert manifest["job_matrix"]["required_warmup"] == 10
    assert "one changed input hash per sample" in manifest["job_matrix"]["shared_input_rule"]
    assert "across jobs" in manifest["job_matrix"]["cross_job_rule"]
    assert "unavailable" in manifest["audit_requirements"]["edge_gate"]
    assert "never paired" in manifest["audit_requirements"]["hardware_gate"]


def test_split_effective_configs_change_only_ranks_and_remain_valid():
    protocol, _ = load_protocol(ROOT)
    manifest = _load("production_ladder_split_manifest.json")
    for mode in MODES:
        mode_metadata = manifest["modes"][mode]
        base = _load(mode_metadata["base_config"])
        assert base["mode"] == base["model_config"]["mode"] == mode
        for group in manifest["split_groups"]:
            effective = json.loads(json.dumps(base))
            effective["ranks"] = list(group["ranks"])
            unchanged = json.loads(json.dumps(base))
            unchanged.pop("ranks")
            changed_without_ranks = json.loads(json.dumps(effective))
            changed_without_ranks.pop("ranks")
            assert changed_without_ranks == unchanged
            assert effective["pairwise"] is False
            assert effective["production_ladder"]["split_policy"]["cross_job_pairing"] == "forbidden"

            model = _model_config(protocol, effective, "primary")
            for rank in group["ranks"]:
                TrainingConfig(**dict(model, rank=rank))
            TrainingConfig(**dict(model, rank=1024, variant="standard"))


def test_priority_effective_configs_change_only_ranks_and_remain_valid():
    protocol, _ = load_protocol(ROOT)
    manifest = _load("production_ladder_split_manifest.json")
    first_gate = manifest["priority_first_gate"]
    for mode in MODES:
        base = _load(first_gate["base_configs"][mode])
        effective = json.loads(json.dumps(base))
        effective["ranks"] = list(first_gate["ranks"])
        unchanged = json.loads(json.dumps(base))
        unchanged.pop("ranks")
        changed_without_ranks = json.loads(json.dumps(effective))
        changed_without_ranks.pop("ranks")
        assert changed_without_ranks == unchanged
        assert effective["pairwise"] is False
        model = _model_config(protocol, effective, "primary")
        for rank in first_gate["ranks"]:
            TrainingConfig(**dict(model, rank=rank))
        TrainingConfig(**dict(model, rank=1024, variant="standard"))
