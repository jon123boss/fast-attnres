from types import SimpleNamespace

import pytest

from benchmarks import bf16_resident_diagnostic as diagnostic


def test_live_memory_guard_includes_all_models_and_largest_temporary():
    gib = 2**30
    memories = [{"persistent_incremental_allocated_bytes": 6 * gib,
                 "peak_allocated_bytes_incremental": peak * gib} for peak in (25, 35, 23)]
    plan = diagnostic.memory_plan(memories, free=55 * gib, reserved=20 * gib,
                                  allocated=gib, capacity=80 * gib)
    assert plan["persistent_bytes"] == 18 * gib
    assert plan["temporary_bytes"] == 29 * gib and plan["admitted"]
    assert not diagnostic.memory_plan(memories, free=30 * gib, reserved=20 * gib,
                                      allocated=gib, capacity=80 * gib)["admitted"]


def test_hooks_restore_after_error_and_keep_initial_qualification_offload():
    calls = []
    activate = lambda arm: calls.append("activate")
    offload = lambda arm: calls.append("offload")
    training = SimpleNamespace(_activate_arm=activate, _offload_arm=offload)
    pool = diagnostic.Pool(training)
    arm = {"step": lambda: None}
    with pytest.raises(RuntimeError, match="probe failure"):
        with diagnostic.retain(training, pool):
            training._offload_arm(arm)
            assert pool.arms == [arm]
            pool.resident.add(id(arm))
            training._offload_arm(arm)
            training._activate_arm({})
            raise RuntimeError("probe failure")
    assert calls == ["offload", "activate"]
    assert training._activate_arm is activate and training._offload_arm is offload


def test_resident_diagnostic_rejects_primary_identity(tmp_path):
    with pytest.raises(ValueError, match="primary"):
        diagnostic.run({"primary_contract_sha256": ""}, tmp_path / "report.json")
