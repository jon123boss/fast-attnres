"""CPU/static coverage for capability-scoped comparator dispatch."""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks.comparator_registry import capability_for
from benchmarks.comparator_runner import (
    ComparatorRoute,
    _DISCOVERY_SPECS,
    MATCHED_TIMING_EXCLUDED_WORK,
    discover_registered_comparators,
    materialize_comparison_plan,
    materialize_comparison_result,
    qualify_comparator,
    run_matched_comparison,
    run_matched_registry,
    run_registered_comparison,
    summarize_matched_statistics,
)
from benchmarks import comparator_runner
from benchmarks.competitor_protocol import comparison_plan, load_config, paired_orders
from benchmarks import modal_competitor_runner as modal_worker
from validation.oracle import oracle


ROOT = Path(__file__).resolve().parents[1]


def test_benchmarks_package_import_is_torch_free_for_local_modal_submission():
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(ROOT)!r}); "
                "import benchmarks; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _identity(values, query):
    packed = values if isinstance(values, torch.Tensor) else torch.stack(tuple(values), dim=0)
    return oracle(packed, query)


def _route(name: str, function=_identity) -> ComparatorRoute:
    comparator = SimpleNamespace(
        name=name,
        available=True,
        status="available",
        reason=None,
        applicable=lambda values, query: (True, None),
        describe=lambda: {"name": name, "status": "available"},
    )
    return ComparatorRoute(
        name=name,
        adapter_module="tests.fake_comparator",
        adapter_name=name,
        capability=capability_for(name),
        comparator=comparator,
        status="available",
        invoke_function=function,
    )


def _cell(
    name="native_fla_triton_checkpoint1",
    *,
    mode="standard_operator",
    rank=4,
    width=4,
    source_count=2,
    dtype="fp32",
    timing=True,
):
    return {
        "comparison_cell_id": f"test:{name}:{mode}:{width}",
        "competitor": name,
        "mode": mode,
        "rank": rank,
        "width": width,
        "source_count": source_count,
        "read_source_count": source_count,
        "dtype": dtype,
        "timing": timing,
    }


def test_default_plan_materializes_unsupported_rows_and_keeps_denominator_capability_only():
    report = materialize_comparison_plan(comparison_plan(load_config()))

    assert report["status"] == "planned"
    assert report["planned"] == 282
    assert report["not_applicable"] == 1302
    assert report["eligible_denominator"] == 282
    assert report["qualified_denominator"] == 0
    assert sum(row["eligible_denominator"] for row in report["cells"]) == 0
    unsupported = [row for row in report["cells"] if row["status"] == "not_applicable"]
    assert len(unsupported) == 1302
    assert all(not row["eligible"] for row in unsupported)
    assert all(not row["eligible_denominator"] for row in unsupported)
    assert all("latency_ms" not in row for row in unsupported)


def test_unsupported_cell_is_not_applicable_before_input_allocation_or_adapter_invoke():
    cell = _cell(
        "liger",
        mode="full",
        rank=1024,
        width=1024,
        source_count=49,
        dtype="bf16",
    )
    invoked = 0
    timed = 0

    def invoke(values, query):
        nonlocal invoked
        invoked += 1
        return _identity(values, query)

    def timer(function):
        nonlocal timed
        timed += 1
        return 1.0, function()

    result = run_registered_comparison(
        "liger",
        cell,
        load_config(),
        route=_route("liger", invoke),
        input_factory=lambda _cell: pytest.fail("inapplicable cells must not allocate inputs"),
        rounds=2,
        timing_call=timer,
    )

    assert result["status"] == "not_applicable"
    assert result["eligible"] is False
    assert result["eligible_denominator"] is False
    assert invoked == timed == 0
    assert "timing" not in result


def test_qualification_runs_output_each_value_gradient_and_query_gradient_before_timing():
    cell = _cell()
    values = torch.randn(2, 3, 4)
    query = torch.randn(4)
    calls = {"invoke": 0, "timer": 0}

    def invoke(actual_values, actual_query):
        calls["invoke"] += 1
        return _identity(actual_values, actual_query)

    def timer(function):
        calls["timer"] += 1
        return 2.5, function()

    result = run_registered_comparison(
        cell["competitor"],
        cell,
        load_config(),
        values=values,
        query=query,
        route=_route(cell["competitor"], invoke),
        rounds=2,
        warmup=1,
        timing_call=timer,
        seed=17,
        gpu="H100!",
    )

    assert result["status"] == "complete"
    assert result["qualification"]["status"] == "qualified"
    assert result["qualification"]["checks"]["output"]["status"] == "passed"
    assert result["qualification"]["checks"]["values_gradient"]["status"] == "passed"
    assert result["qualification"]["checks"]["values_gradient"]["source_count"] == 1
    assert result["qualification"]["checks"]["query_gradient"]["status"] == "passed"
    assert calls["invoke"] == 1 + 1 + 2  # qualification, warmup, timed rounds
    assert calls["timer"] == 2
    assert result["timing"]["adapter_stack_in_timing"] is True
    assert result["timing"]["raw_samples"]
    assert all(sample["status"] == "ok" for sample in result["timing"]["raw_samples"])
    assert result["eligible_denominator"] is True


def test_failed_qualification_blocks_timer_and_denominator():
    cell = _cell()
    values = torch.randn(2, 2, 4)
    query = torch.randn(4)
    timed = 0

    def wrong(actual_values, actual_query):
        return _identity(actual_values, actual_query) + 1

    def timer(function):
        nonlocal timed
        timed += 1
        return 1.0, function()

    result = run_registered_comparison(
        cell["competitor"],
        cell,
        load_config(),
        values=values,
        query=query,
        route=_route(cell["competitor"], wrong),
        rounds=2,
        timing_call=timer,
    )

    assert result["status"] == "failed"
    assert result["qualification"]["status"] == "failed"
    assert result["eligible_denominator"] is False
    assert timed == 0
    assert "timing" not in result


def test_matched_pair_qualifies_both_arms_and_times_forward_backward_abba():
    cell = {
        **_cell(source_count=2, dtype="fp32"),
        "N": 3,
        "S": 2,
        "D": 4,
        "R": 4,
        "timing_mode": "forward_backward",
    }
    values = torch.randn(2, 3, 4)
    query = torch.randn(4)
    calls = {"candidate": 0, "comparator": 0, "timers": 0}

    def candidate(actual_values, actual_query):
        calls["candidate"] += 1
        return _identity(actual_values, actual_query)

    def comparator(actual_values, actual_query):
        calls["comparator"] += 1
        return _identity(actual_values, actual_query)

    expected_orders = paired_orders(("attnres", "native_fla_triton_checkpoint1"), 2, seed=123)
    timed_arms = iter(arm for order in expected_orders for arm in order)

    def timer(function):
        calls["timers"] += 1
        output = function()
        elapsed = 1.0 if next(timed_arms) == "attnres" else 2.0
        return elapsed, output

    name = "native_fla_triton_checkpoint1"
    result = run_matched_comparison(
        name,
        cell,
        load_config(),
        values=values,
        query=query,
        route=_route(name, comparator),
        candidate=candidate,
        rounds=2,
        warmup=0,
        timing_call=timer,
        seed=123,
        gpu="H100!",
    )

    assert result["status"] == "complete"
    assert result["eligible_denominator"] is True
    assert result["candidate_qualification"]["status"] == "qualified"
    assert result["qualification"]["status"] == "qualified"
    assert calls == {"candidate": 3, "comparator": 3, "timers": 4}
    timing = result["timing"]
    assert timing["timing_mode"] == "forward_backward"
    assert timing["adapter_stack_in_timing"] is True
    assert timing["ratio"]["orientation"] == "candidate_over_baseline"
    assert timing["candidate_samples"] == [1.0, 1.0]
    assert timing["comparator_samples"] == [2.0, 2.0]
    assert all(row["status"] == "ok" for row in timing["raw_samples"])
    assert {row["input_hash"] for row in timing["raw_samples"]} == {
        timing["input_hash"]
    }
    orders = timing["orders"]
    assert len(orders) == 2
    assert tuple(orders[1]) == tuple(reversed(orders[0]))
    assert result["pair"]["same_inputs"] is True
    assert result["pair"]["ratio"]["candidate_arm"] == "attnres"


def test_matched_timing_performs_no_tensor_hashing_or_readback(monkeypatch):
    cell = {
        **_cell(source_count=2, dtype="fp32"),
        "N": 3,
        "S": 2,
        "D": 4,
        "R": 4,
        "timing_mode": "forward_backward",
    }
    values = torch.randn(2, 3, 4)
    query = torch.randn(4)
    inside_event = False
    gradient_clear_locations = []
    original_clear_gradients = comparator_runner._clear_timing_gradients

    def forbidden_tensor_hash(*_args, **_kwargs):
        raise AssertionError("speed-first matched timing must not hash tensors")

    def observed_clear_gradients(actual_values, actual_query):
        gradient_clear_locations.append(inside_event)
        return original_clear_gradients(actual_values, actual_query)

    def timer(function):
        nonlocal inside_event
        assert inside_event is False
        inside_event = True
        try:
            output = function()
        finally:
            inside_event = False
        return 1.0, output

    monkeypatch.setattr(comparator_runner, "_input_hash", forbidden_tensor_hash)
    monkeypatch.setattr(
        comparator_runner, "_clear_timing_gradients", observed_clear_gradients
    )
    name = cell["competitor"]
    result = run_matched_comparison(
        name,
        cell,
        load_config(),
        values=values,
        query=query,
        route=_route(name),
        candidate=_identity,
        rounds=2,
        warmup=0,
        timing_call=timer,
        seed=123,
        gpu="H100!",
    )

    assert result["status"] == "complete"
    assert result["timing"]["raw_validation"]["status"] == "passed"
    assert all(row["status"] == "ok" for row in result["timing"]["raw_samples"])
    assert result["timing"]["timing_excluded_work"] == list(
        MATCHED_TIMING_EXCLUDED_WORK
    )
    assert result["timing"]["integrity_mode"] == "speed_first_no_tensor_hashing"
    assert (
        result["timing"]["input_hash_kind"]
        == "logical_case_seed_id_no_tensor_readback"
    )
    assert gradient_clear_locations
    assert not any(gradient_clear_locations)


def test_matched_timing_uses_one_logical_pairing_id_without_tensor_reads(monkeypatch):
    cell = {
        **_cell(source_count=2, dtype="fp32"),
        "N": 3,
        "S": 2,
        "D": 4,
        "R": 4,
        "timing_mode": "forward",
    }
    values = torch.randn(2, 3, 4)
    query = torch.randn(4)
    inside_event = False

    def candidate(actual_values, actual_query):
        return _identity(actual_values, actual_query)

    def forbidden_tensor_hash(*_args, **_kwargs):
        raise AssertionError("speed-first matched timing must not hash tensors")

    def timer(function):
        nonlocal inside_event
        inside_event = True
        try:
            output = function()
        finally:
            inside_event = False
        return 1.0, output

    monkeypatch.setattr(comparator_runner, "_input_hash", forbidden_tensor_hash)
    name = cell["competitor"]
    result = run_matched_comparison(
        name,
        cell,
        load_config(),
        values=values,
        query=query,
        route=_route(name),
        candidate=candidate,
        rounds=2,
        warmup=0,
        timing_call=timer,
        seed=123,
        gpu="H100!",
    )

    assert result["status"] == "complete"
    logical_id = result["timing"]["input_hash"]
    assert len(logical_id) == 64
    assert set(logical_id) <= set("0123456789abcdef")
    assert {
        row["input_hash"] for row in result["timing"]["raw_samples"]
    } == {logical_id}


def test_list_inputs_keep_independent_source_gradient_checks():
    cell = _cell(source_count=3)
    values = [torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4)]
    query = torch.randn(4)
    qualification = qualify_comparator(
        cell["competitor"],
        cell,
        values,
        query,
        load_config(),
        route=_route(cell["competitor"]),
    )
    assert qualification["status"] == "qualified"
    assert qualification["checks"]["values_gradient"]["source_count"] == 3
    assert qualification["checks"]["query_gradient"]["status"] == "passed"


def test_matched_list_inputs_are_preserved_per_arm_and_use_output_shaped_upstream():
    cell = {
        **_cell(source_count=3, dtype="fp32"),
        "N": 2,
        "S": 3,
        "D": 4,
        "R": 4,
        "timing_mode": "forward_backward",
    }
    values = [torch.randn(2, 4) for _ in range(3)]
    query = torch.randn(4)
    seen_layouts = []

    def candidate(actual_values, actual_query):
        seen_layouts.append(("candidate", type(actual_values), len(actual_values)))
        return _identity(actual_values, actual_query)

    def comparator(actual_values, actual_query):
        seen_layouts.append(("comparator", type(actual_values), len(actual_values)))
        return _identity(actual_values, actual_query)

    def timer(function):
        return 1.0, function()

    result = run_matched_comparison(
        cell["competitor"],
        cell,
        load_config(),
        values=values,
        query=query,
        route=_route(cell["competitor"], comparator),
        candidate=candidate,
        rounds=2,
        warmup=0,
        timing_call=timer,
        seed=123,
        gpu="H100!",
    )

    assert result["status"] == "complete"
    assert result["timing"]["input_layout"] == "list"
    assert result["timing"]["upstream_shape"] == [2, 4]
    assert {kind for kind, container, count in seen_layouts if kind in {"candidate", "comparator"}} == {
        "candidate",
        "comparator",
    }
    assert all(container is list and count == 3 for _kind, container, count in seen_layouts)


def test_warmup_failure_retains_validator_complete_matrix_and_actual_arm_provenance():
    cell = {
        **_cell(source_count=2, dtype="fp32"),
        "N": 2,
        "S": 2,
        "D": 4,
        "R": 4,
        "timing_mode": "forward_backward",
    }
    values = [torch.randn(2, 4), torch.randn(2, 4)]
    query = torch.randn(4)
    comparator_calls = 0

    def candidate(actual_values, actual_query):
        return _identity(actual_values, actual_query)

    def comparator(actual_values, actual_query):
        nonlocal comparator_calls
        comparator_calls += 1
        if comparator_calls == 2:  # qualification succeeds; the first warmup call fails
            raise RuntimeError("mock warmup failure")
        return _identity(actual_values, actual_query)

    result = run_matched_comparison(
        cell["competitor"],
        cell,
        load_config(),
        values=values,
        query=query,
        route=_route(cell["competitor"], comparator),
        candidate=candidate,
        rounds=2,
        warmup=1,
        timing_call=lambda function: pytest.fail("warmup failure must block timing"),
        seed=123,
        gpu="H100!",
    )

    timing = result["timing"]
    assert result["status"] == "incomplete"
    assert timing["raw_validation"]["status"] == "passed"
    assert len(timing["raw_samples"]) == 4
    failed = [row for row in timing["raw_samples"] if row["status"] == "failed"]
    skipped = [row for row in timing["raw_samples"] if row["status"] == "skipped_due_to_failure"]
    assert len(failed) == 1 and len(skipped) == 3
    failure = failed[0]
    assert failure["arm"] == cell["competitor"]
    assert all(
        row["failure_at_round"] == failure["round_index"]
        and row["failure_at_order"] == failure["order_index"]
        and row["failure_reason"]
        and row["failure_phase"]
        for row in timing["raw_samples"]
    )


def test_report_statistics_use_simultaneous_common_index_family_group():
    config = load_config()
    plan = comparison_plan(config)
    planned = {
        row["competitor"]: row
        for row in plan["planned"]
        if row["comparison_family"] == "primary"
        and row["gpu"] == "H100!"
        and row["seed"] == config["seeds"][0]
        and row.get("operator_scope") == "smoke"
        and row["D"] <= 256
    }
    assert set(planned) == {"native_fla_triton_checkpoint1", "native_fla_gluon"}
    rows = []
    for competitor in ("native_fla_triton_checkpoint1", "native_fla_gluon"):
        cell = dict(planned[competitor])
        orders = paired_orders(("attnres", competitor), config["rounds"], seed=cell["seed"])
        raw_samples = []
        candidate_samples = []
        comparator_samples = []
        for round_index, order in enumerate(orders):
            for order_index, arm in enumerate(order):
                latency = 1.0 if arm == "attnres" else 2.0
                if arm == "attnres":
                    candidate_samples.append(latency)
                else:
                    comparator_samples.append(latency)
                raw_samples.append(
                    {
                        "seed": cell["seed"],
                        "gpu": cell["gpu"],
                        "round_index": round_index,
                        "order_index": order_index,
                        "input_hash": "shared-input",
                        "arm": arm,
                        "status": "ok",
                        "latency_ms": latency,
                        "failure_phase": None,
                        "failure_reason": None,
                        "failure_at_round": None,
                        "failure_at_order": None,
                        "eligible": True,
                    }
                )
        rows.append(
            {
                "comparison_cell_id": cell["comparison_cell_id"],
                "cell": cell,
                "competitor": competitor,
                "status": "complete",
                "eligible_denominator": True,
                "timing": {
                    "status": "complete",
                    "timing_mode": "forward_backward",
                    "warmup": config["warmup"],
                    "rounds": config["rounds"],
                    "arms": ["attnres", competitor],
                    "raw_samples": raw_samples,
                    "candidate_samples": candidate_samples,
                    "comparator_samples": comparator_samples,
                    "raw_validation": {"status": "passed"},
                },
            }
        )

    result = summarize_matched_statistics(rows, config)
    assert result["status"] == "complete"
    assert result["ratio"] == "candidate_over_baseline"
    assert result["common_resample_indices"] is True
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["comparison_count"] == 2
    assert all(summary["simultaneous"] for summary in group["comparisons"].values())
    assert all(
        summary["orientation"] == "candidate_over_baseline"
        for summary in group["comparisons"].values()
    )


def test_alias_dispatch_keeps_canonical_identity_in_materialized_result():
    alias_cell = _cell("hydra_2p")
    result = run_registered_comparison(
        "hydra_2p",
        alias_cell,
        load_config(),
        values=torch.randn(2, 2, 4),
        query=torch.randn(4),
        route=_route("manish_hydra_2p"),
        rounds=0,
    )
    assert result["competitor"] == "manish_hydra_2p"
    assert result["cell"]["competitor"] == "manish_hydra_2p"
    assert result["qualification"]["discovery"]["name"] == "manish_hydra_2p"


def test_discovery_uses_explicit_canonical_to_adapter_identity_without_alias_fallback():
    assert _DISCOVERY_SPECS["native_fla_triton_checkpoint0"] == (
        "benchmarks.competitors",
        "discover_comparators",
        "fla_triton_checkpoint0",
    )
    assert _DISCOVERY_SPECS["native_fla_triton_checkpoint1"][-1] == "fla_triton_checkpoint1"
    assert _DISCOVERY_SPECS["native_fla_gluon"][-1] == "fla_gluon"
    assert _DISCOVERY_SPECS["manish_hydra_2p"][-1] == "hydra_2p"
    source = (ROOT / "benchmarks" / "comparator_runner.py").read_text(encoding="utf-8")
    assert "discovered.get(name)" not in source
    assert "adapter did not return comparator" in source


def test_partial_explicit_root_map_cannot_fall_back_to_ambient_vendor_checkout(tmp_path):
    routes = discover_registered_comparators(
        project_root=tmp_path,
        vendor_roots={"fla": tmp_path / "fla"},
        names=("native_fla_triton_checkpoint1", "liger"),
    )
    assert routes["native_fla_triton_checkpoint1"].requested_root == str(
        (tmp_path / "fla").resolve()
    )
    assert routes["liger"].status == "missing"
    assert "no explicit vendor root" in routes["liger"].reason


def test_modal_worker_is_separate_and_has_sealed_runtime_before_optional_imports():
    source = (ROOT / "benchmarks" / "modal_competitor_runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in top_level_imports
    assert "triton" not in top_level_imports
    assert "modal" not in top_level_imports
    assert modal_worker.TORCH_VERSION == "2.13.0"
    assert modal_worker.TRITON_VERSION == "3.7.1"
    assert modal_worker._run.__code__ is not None
    assert '.apt_install("git")' in source
    assert '"einops": "0.8.1"' in source
    assert '"einops==0.8.1"' in source
    assert modal_worker._container_vendor_roots() == {
        "fla": "/workspace/vendors/fla",
        "liger": "/workspace/vendors/liger",
        "catswe": "/workspace/vendors/catswe",
        "manish": "/workspace/vendors/hydra",
    }
    assert modal_worker._VENDOR_ORIGINS == {
        "fla": "https://github.com/fla-org/flash-linear-attention.git",
        "liger": "https://github.com/linkedin/Liger-Kernel.git",
        "catswe": "https://github.com/catswe/flash-attention-residuals.git",
        "manish": "https://github.com/manishklach/attnres-kernel-lab.git",
    }
    release_source = (ROOT / "benchmarks" / "modal_runner.py").read_text(encoding="utf-8")
    assert 'TORCH_VERSION = os.environ.get("ATTNRES_TORCH_VERSION", "2.11.0")' in release_source
    assert 'TRITON_VERSION = os.environ.get("ATTNRES_TRITON_VERSION", "3.6.0")' in release_source


def test_modal_worker_batch_writes_partial_report_before_timeout_failure(tmp_path):
    def completed(_payload):
        return {"requested_gpu": "H100!", "status": "complete"}

    def timed_out(_payload):
        raise TimeoutError("remote worker timed out while compiling")

    functions = [
        ("H100!", SimpleNamespace(remote=completed)),
        ("B200", SimpleNamespace(remote=timed_out)),
    ]
    output = tmp_path / "matched.json"
    with pytest.raises(RuntimeError, match="report written"):
        modal_worker._run_worker_batch(
            functions,
            {},
            config_path=tmp_path / "config.json",
            config_digest="digest",
            scope="smoke",
            configured_roots={},
            output=str(output),
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["results"] == [{"requested_gpu": "H100!", "status": "complete"}]
    assert report["worker_failures"] == [
        {
            "selector": "B200",
            "status": "failed",
            "error": {
                "type": "TimeoutError",
                "message": "remote worker timed out while compiling",
                "timeout": True,
            },
            "timeout": True,
        }
    ]
    assert not list(tmp_path.glob(".*.tmp"))


def test_modal_worker_batch_writes_all_failure_report_before_raising(tmp_path):
    def failed(_payload):
        raise RuntimeError("compiler failed")

    functions = [
        ("H100!", SimpleNamespace(remote=failed)),
        ("B200", SimpleNamespace(remote=failed)),
    ]
    output = tmp_path / "all-failed.json"
    with pytest.raises(RuntimeError, match="H100!, B200"):
        modal_worker._run_worker_batch(
            functions,
            {},
            config_path=tmp_path / "config.json",
            config_digest="digest",
            scope="smoke",
            configured_roots={},
            output=str(output),
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["results"] == []
    assert [item["selector"] for item in report["worker_failures"]] == ["H100!", "B200"]
    assert all(item["timeout"] is False for item in report["worker_failures"])


def test_modal_worker_batch_requires_output_before_remote_submission(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("remote workers must not be submitted without an output path")

    monkeypatch.setattr(modal_worker, "_collect_worker_results", forbidden)
    functions = [("H100!", SimpleNamespace(remote=forbidden))]
    with pytest.raises(ValueError, match="output path is required"):
        modal_worker._run_worker_batch(
            functions,
            {},
            config_path=tmp_path / "config.json",
            config_digest="digest",
            scope="smoke",
            configured_roots={},
            output="",
        )


def test_modal_worker_batch_turns_non_mapping_result_into_worker_failure(tmp_path):
    def malformed(_payload):
        return ["not", "a", "report"]

    functions = [("H100!", SimpleNamespace(remote=malformed))]
    output = tmp_path / "malformed.json"
    with pytest.raises(RuntimeError, match="report written"):
        modal_worker._run_worker_batch(
            functions,
            {},
            config_path=tmp_path / "config.json",
            config_digest="digest",
            scope="smoke",
            configured_roots={},
            output=str(output),
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["results"] == []
    assert report["worker_failures"][0]["selector"] == "H100!"
    assert report["worker_failures"][0]["error"]["type"] == "TypeError"
    assert "expected a Mapping" in report["worker_failures"][0]["error"]["message"]


@pytest.mark.parametrize(
    "worker_result",
    [
        {"requested_gpu": "H100!"},
        {"requested_gpu": "H100!", "status": "complete", "invalid": float("nan")},
        {"requested_gpu": "B200", "status": "complete"},
    ],
)
def test_modal_worker_batch_rejects_malformed_mapping_and_persists_failure(
    tmp_path, worker_result
):
    output = tmp_path / "malformed-mapping.json"
    with pytest.raises(RuntimeError, match="report written"):
        modal_worker._run_worker_batch(
            [("H100!", SimpleNamespace(remote=lambda _payload: worker_result))],
            {},
            config_path=tmp_path / "config.json",
            config_digest="digest",
            scope="smoke",
            configured_roots={},
            output=str(output),
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["results"] == []
    assert report["worker_failures"][0]["selector"] == "H100!"


def test_vendor_bundle_transport_preserves_clean_symlinked_checkout(tmp_path):
    checkout = tmp_path / "fla"
    checkout.mkdir()
    (checkout / "target.txt").write_text("source\n", encoding="utf-8")
    (checkout / "link.txt").symlink_to("target.txt")
    for command in (
        ("init", "-q"),
        ("add", "."),
        (
            "-c",
            "user.name=Fast-AttnRes Tests",
            "-c",
            "user.email=tests@fast-attnres.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        (
            "remote",
            "add",
            "origin",
            modal_worker._VENDOR_ORIGINS["fla"],
        ),
    ):
        subprocess.run(
            ["git", "-C", str(checkout), *command],
            check=True,
            capture_output=True,
            text=True,
        )

    modal_worker._require_transport_provenance(checkout, family="fla")
    bundle = modal_worker._create_vendor_bundle(checkout, family="fla")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(bundle), str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (clone / "link.txt").is_symlink()
    assert subprocess.run(
        ["git", "-C", str(clone), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""


def test_sealed_runtime_gate_rejects_legacy_runtime_before_optional_imports():
    fake_torch = SimpleNamespace(
        __version__="2.11.0",
        version=SimpleNamespace(cuda="12.4"),
    )
    fake_triton = SimpleNamespace(__version__="3.6.0")
    from benchmarks.modal_competitor_runner import _validate_runtime

    with pytest.raises(RuntimeError, match="runtime mismatch"):
        _validate_runtime(fake_torch, fake_triton)


def test_public_dispatch_surface_has_no_cached_or_impossible_gradient_symbols():
    source = (ROOT / "benchmarks" / "comparator_runner.py").read_text(encoding="utf-8")
    assert "prepare_block" not in source
    assert "merge_block" not in source
    assert "CatsweCache" not in source
    assert "keys_gradient" not in source


def test_sealed_operator_plan_keeps_explicit_dtype_and_all_protocol_seeds():
    config = load_config()
    plan = comparison_plan(config)
    operator_rows = [
        row
        for row in (*plan["planned"], *plan["not_applicable"])
        if row["scope"] == "operator"
    ]
    assert operator_rows
    assert {row["dtype"] for row in operator_rows} == {"bf16"}
    assert {row["seed"] for row in operator_rows} == set(config["seeds"])
    assert {row["gpu"] for row in operator_rows} == set(config["hardware_order"])


def test_registry_execution_preserves_timing_only_hydra_limits(monkeypatch):
    captured = []

    def capture(cells, *_args, **_kwargs):
        captured.extend(dict(cell) for cell in cells)
        return []

    monkeypatch.setattr(comparator_runner, "run_matched_comparison_cells", capture)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA H100")
    run_matched_registry(
        load_config(),
        execute_operator=True,
        scope="smoke",
        names=("manish_hydra_2p",),
        device=SimpleNamespace(type="cuda"),
        routes={"manish_hydra_2p": _route("manish_hydra_2p")},
        gpu="H100!",
    )

    assert captured
    assert all(cell["timing"] is True for cell in captured)
    oversized = [
        cell
        for cell in captured
        if int(cell["D"]) > 256 and int(cell["R"]) == int(cell["D"])
    ]
    assert oversized
    result = run_matched_comparison(
        "manish_hydra_2p",
        oversized[0],
        load_config(),
        input_factory=lambda _cell: pytest.fail("not-applicable cell allocated inputs"),
    )
    assert result["status"] == "not_applicable"
    assert "native timing envelope" in result["reason"]
    assert result["pair"]["timing_mode"] == "forward_backward"


def test_materialization_rejects_incomplete_timing_and_untyped_eligibility():
    cell = _cell()
    incomplete = materialize_comparison_result(
        cell,
        status="complete",
        eligibility={"eligible": True, "eligible_denominator": True},
        qualification={"status": "qualified"},
        timing={
            "status": "incomplete",
            "timing_mode": "forward",
            "warmup": 0,
            "rounds": 1,
            "arms": [cell["competitor"]],
            "raw_samples": [],
        },
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["eligible_denominator"] is False

    with pytest.raises(ValueError, match=r"eligibility\.eligible must be a boolean"):
        materialize_comparison_result(
            cell,
            status="complete",
            eligibility={"eligible": "true", "eligible_denominator": True},
        )


def test_materialization_rejects_timing_with_invalid_warmup_or_cell_mode():
    cell = _cell()
    arms = ("attnres", cell["competitor"])
    raw_samples = []
    for round_index, order in enumerate(paired_orders(arms, 1, seed=17)):
        for order_index, arm in enumerate(order):
            raw_samples.append(
                {
                    "seed": 17,
                    "gpu": "H100!",
                    "round_index": round_index,
                    "order_index": order_index,
                    "input_hash": "test-input",
                    "arm": arm,
                    "status": "ok",
                    "latency_ms": 1.0,
                    "failure_phase": None,
                    "failure_reason": None,
                    "failure_at_round": None,
                    "failure_at_order": None,
                }
            )
    timing = {
        "status": "complete",
        "timing_mode": "forward",
        "warmup": 0,
        "rounds": 1,
        "arms": arms,
        "raw_samples": raw_samples,
    }
    kwargs = {
        "status": "complete",
        "eligibility": {"eligible": True, "eligible_denominator": True},
        "qualification": {"status": "qualified"},
        "timing": timing,
    }
    invalid_warmup = deepcopy(timing)
    invalid_warmup["warmup"] = -1
    result = materialize_comparison_result(cell, **{**kwargs, "timing": invalid_warmup})
    assert result["status"] == "incomplete"
    assert result["eligible_denominator"] is False

    mismatched_mode = deepcopy(timing)
    mismatched_mode["timing_mode"] = "forward_backward"
    result = materialize_comparison_result(cell, **{**kwargs, "timing": mismatched_mode})
    assert result["status"] == "incomplete"
    assert result["eligible_denominator"] is False


def test_operator_plan_carries_forward_backward_timing_mode():
    plan = comparison_plan(load_config())
    operator_rows = [row for row in plan["planned"] if row["scope"] == "operator"]
    assert operator_rows
    assert all(row["timing"] is True for row in operator_rows)
    assert all(row["timing_mode"] == "forward_backward" for row in operator_rows)


def test_direct_forward_only_comparator_cannot_time_operator_plan_as_forward():
    config = load_config()
    cell = {
        **_cell(timing=True),
        "timing_mode": "forward_backward",
    }
    result = run_registered_comparison(
        cell["competitor"],
        cell,
        config,
        values=torch.randn(2, 2, 4),
        query=torch.randn(4),
        route=_route(cell["competitor"]),
        rounds=1,
        timing_call=lambda function: pytest.fail("forward-only comparator must not be timed"),
        seed=17,
        gpu="H100!",
    )
    assert result["status"] == "incomplete"
    assert "forward-only boundary" in result["reason"]
    assert result["timing"]["timing_mode"] == "forward_backward"


def test_registry_requires_one_exact_gpu_selector_before_execution():
    with pytest.raises(ValueError, match="exactly one GPU selector"):
        run_matched_registry(
            load_config(),
            execute_operator=True,
            scope="smoke",
            device=SimpleNamespace(type="cuda"),
            gpu=None,
        )
    with pytest.raises(ValueError, match="exactly one GPU selector"):
        run_matched_registry(
            load_config(),
            execute_operator=True,
            scope="smoke",
            device=SimpleNamespace(type="cuda"),
            gpu="both",
        )


def test_registry_rejects_physical_gpu_that_does_not_match_selector(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA B200")
    with pytest.raises(RuntimeError, match="hardware mismatch"):
        run_matched_registry(
            load_config(),
            execute_operator=True,
            scope="smoke",
            device=SimpleNamespace(type="cuda"),
            gpu="H100!",
        )


def test_partial_explicit_routes_are_missing_without_ambient_rediscovery(monkeypatch):
    captured = []

    def capture(cells, *_args, **_kwargs):
        captured.extend(dict(cell) for cell in cells)
        return []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("explicit route maps must not trigger discovery")

    monkeypatch.setattr(comparator_runner, "run_matched_comparison_cells", capture)
    monkeypatch.setattr(comparator_runner, "discover_registered_comparators", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA H100")
    report = run_matched_registry(
        load_config(),
        execute_operator=True,
        scope="smoke",
        names=("native_fla_triton_checkpoint1", "native_fla_gluon"),
        device=SimpleNamespace(type="cuda"),
        routes={"native_fla_triton_checkpoint1": _route("native_fla_triton_checkpoint1")},
        gpu="H100!",
    )
    assert captured
    assert report["routes"]["native_fla_gluon"]["status"] == "missing"
    assert "ambient discovery is disabled" in report["routes"]["native_fla_gluon"]["reason"]


def test_registry_reports_zero_eligible_scope_as_not_applicable(monkeypatch):
    def materialize_na(cells, *_args, **_kwargs):
        return [
            materialize_comparison_result(
                cell,
                status="not_applicable",
                eligibility=cell["eligibility"],
            )
            for cell in cells
        ]

    monkeypatch.setattr(comparator_runner, "run_matched_comparison_cells", materialize_na)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA H100")
    report = run_matched_registry(
        load_config(),
        execute_operator=True,
        scope="primary",
        names=("manish_hydra_2p",),
        device=SimpleNamespace(type="cuda"),
        routes={"manish_hydra_2p": _route("manish_hydra_2p")},
        gpu="H100!",
    )
    assert report["operator"]["cells"]
    assert report["operator"]["eligible_denominator"] == 0
    assert report["operator"]["status"] == "not_applicable"
    assert report["status"] == "not_applicable"


def test_registered_input_factory_failure_is_retained_with_phase():
    def fail(_cell):
        raise RuntimeError("input allocation failed")

    result = run_registered_comparison(
        "native_fla_triton_checkpoint1",
        _cell(),
        load_config(),
        route=_route("native_fla_triton_checkpoint1"),
        input_factory=fail,
        rounds=0,
    )
    assert result["status"] == "failed"
    assert result["failure"]["phase"] == "input_factory"
    assert "input allocation failed" in result["reason"]


def test_route_and_adapter_applicability_require_boolean_decisions():
    route = _route("native_fla_triton_checkpoint1")
    route.comparator.available = "false"
    assert route.available is False

    route = _route("native_fla_triton_checkpoint1")
    route.comparator.applicable = lambda _values, _query: ("false", "bad type")
    assert route.applicable(None, None) == (
        False,
        "adapter applicable() must return an actual boolean decision",
    )


def test_statistics_exclude_complete_rows_without_validated_raw_matrix():
    config = load_config()
    plan = comparison_plan(config)
    cell = next(
        row
        for row in plan["planned"]
        if row["competitor"] == "native_fla_triton_checkpoint1"
        and row["comparison_family"] == "primary"
        and row["gpu"] == "H100!"
        and row["seed"] == config["seeds"][0]
        and row.get("operator_scope") == "smoke"
        and row["D"] <= 256
    )
    forged = {
        "comparison_cell_id": cell["comparison_cell_id"],
        "cell": dict(cell),
        "competitor": cell["competitor"],
        "status": "complete",
        "eligible_denominator": True,
        "timing": {
            "status": "incomplete",
            "timing_mode": "forward_backward",
            "warmup": config["warmup"],
            "rounds": config["rounds"],
            "arms": ["attnres", cell["competitor"]],
            "candidate_samples": [1.0],
            "comparator_samples": [2.0],
            "raw_samples": [],
        },
    }
    result = summarize_matched_statistics([forged], config)
    assert result["status"] == "incomplete"
    assert result["groups"] == []
    assert result["excluded"]
