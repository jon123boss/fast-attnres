"""CPU contract tests for the optional compiled-model Liger arm."""

import inspect

import pytest
import torch


def _model_data(*, mode="full", layers=2, width=1024, block_count=2):
    return {
        "layers": layers,
        "width": width,
        "mode": mode,
        "block_count": block_count,
    }


def test_model_read_counts_match_full_and_block_scheduler():
    from benchmarks.run import _model_read_source_counts

    assert _model_read_source_counts(_model_data(mode="full", layers=2)) == (2, 3, 4, 5)
    assert _model_read_source_counts(
        _model_data(mode="block", layers=2, block_count=2)
    ) == (2, 2, 3, 3)
    assert max(
        _model_read_source_counts(_model_data(mode="block", layers=24, block_count=8))
    ) == 9


def test_liger_full_is_na_when_one_actual_read_exceeds_source_limit():
    from benchmarks.run import _liger_model_eligibility

    result = _liger_model_eligibility(_model_data(mode="full", layers=24), 1024)

    assert result["status"] == "not_applicable"
    assert result["eligible_denominator"] is False
    assert result["failed_read_index"] == 31
    assert "S<=32" in result["reason"]


def test_liger_block_uses_per_read_source_count_and_can_be_eligible():
    from benchmarks.run import _liger_model_eligibility

    result = _liger_model_eligibility(
        _model_data(mode="block", layers=24, block_count=8), 1024
    )

    assert result["status"] == "eligible"
    assert result["eligible"] is True
    assert result["max_read_source_count"] == 9
    assert len(result["per_read"]) == 48
    assert all(row["eligible"] for row in result["per_read"])


def test_catswe_model_scope_is_explicit_public_phase1_arm():
    from benchmarks.run import _catswe_model_eligibility

    result = _catswe_model_eligibility(_model_data(mode="block"), 1024)

    assert result["competitor"] == "catswe_phase1"
    assert result["status"] == "eligible"
    assert result["eligible"] is True
    assert result["eligible_denominator"] is True
    assert result["model_scope"] == "compiled_training_step"
    assert result["capability_scope"] == "model"
    assert result["read_source_counts"] == [2, 2, 3, 3]
    assert all(row["eligible"] for row in result["per_read"])
    assert "no cache/prepare/merge/phase2" in result["reason"]


def test_catswe_model_scope_rejects_lr_and_non_power_of_two_width():
    from benchmarks.run import _catswe_model_eligibility

    lr = _catswe_model_eligibility(_model_data(mode="full"), 512)
    assert lr["eligible"] is False
    assert lr["eligible_denominator"] is False
    assert "R=D" in lr["reason"]

    non_power = _catswe_model_eligibility(
        _model_data(mode="full", width=1536, block_count=2), 1536
    )
    assert non_power["eligible"] is False
    assert "power-of-two" in non_power["reason"]


def test_run_suite_skips_catswe_discovery_for_ineligible_model_cell(monkeypatch):
    from benchmarks import catswe, run

    def unexpected_discovery(*args, **kwargs):
        raise AssertionError("ineligible Catswe cells must not discover the vendor")

    monkeypatch.setattr(catswe, "discover_comparator", unexpected_discovery)
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: False)
    result = run.run_suite(
        {
            "scope": "custom",
            "phases": [],
            "include_fla": False,
            "include_liger_model": False,
            # Deliberately forge opt-in for a non-power-of-two standard rank.
            "include_catswe_model": True,
            "ranks": [1536],
            "model_config": {
                "layers": 8,
                "width": 1536,
                "heads": 24,
                "ffn": 4224,
                "batch": 2,
                "sequence": 512,
                "vocab": 8192,
                "rank": 1536,
                "mode": "full",
                "block_count": 16,
                "variant": "sliced",
                "source_layout": "list",
            },
        }
    )
    assert result["coverage"]["include_catswe_model"] is True
    assert "catswe_phase1" not in result["comparators"]


@pytest.mark.parametrize("rank", [16, 512])
def test_liger_model_lr_ranks_are_explicit_na(rank):
    from benchmarks.run import _liger_model_eligibility

    result = _liger_model_eligibility(_model_data(), rank)

    assert result["status"] == "not_applicable"
    assert result["eligible_denominator"] is False
    assert result["rank"] == rank
    assert result["width"] == 1024
    assert "R=D" in result["reason"]


def test_model_only_admission_is_narrow_sealed_and_digest_bound():
    from benchmarks.run import _model_only_rank_admission

    protocol_ranks = [1, 16, 1024]
    config = {
        "ranks": [1024, 2048],
        "model_only_admission": {
            "enabled": True,
            "sealed": True,
            "scope": "model_only",
            "width_rank_pairs": [[2048, 2048], [4096, 4096]],
        },
    }
    allowed, admission, error = _model_only_rank_admission(
        config, {"width": 2048}, protocol_ranks
    )

    assert error is None
    assert allowed == (1, 16, 1024, 2048)
    assert admission is not None
    assert admission["status"] == "sealed"
    assert admission["admitted_extra_ranks"] == [2048]
    assert len(admission["digest"]) == 64

    bad = dict(config, ranks=[3072])
    _allowed, _admission, error = _model_only_rank_admission(
        bad, {"width": 2048}, protocol_ranks
    )
    assert "outside the sealed" in error

    malformed = dict(config, ranks=[True])
    _allowed, _admission, error = _model_only_rank_admission(
        malformed, {"width": 2048}, protocol_ranks
    )
    assert "integer list" in error


def test_model_liger_backend_declares_reused_rms_weight():
    from benchmarks import liger

    source = inspect.getsource(liger.make_model_backend)
    assert "accepts_rms_weight = True" in source
    assert "one_buffer_per_model" in inspect.getsource(liger.source_hash_metadata)
    # The marker only helps if the call path actually skips the direct-call
    # fallback when a model supplies its preallocated unit vector.  Keep this
    # as a source contract because importing the pinned CUDA custom op is not
    # possible on the CPU test host.
    assert source.count("_ones_weight(query_arg)") == 1
    assert "if rms_weight is None:" in source


def test_liger_rms_weight_value_check_is_eager_only(monkeypatch):
    from benchmarks import liger

    query = torch.ones(4, dtype=torch.float32)
    invalid = torch.zeros_like(query)

    monkeypatch.setattr(liger, "_is_compiling", lambda: False)
    with pytest.raises(ValueError, match="parameter-free"):
        liger._validate_rms_weight(invalid, query)

    # Dynamo must not trace Tensor.all().item(): that creates a data-dependent
    # guard and breaks a fullgraph model capture.  The model-created
    # nonpersistent unit buffer was checked during eager qualification.
    monkeypatch.setattr(liger, "_is_compiling", lambda: True)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("tensor value validation ran while compiling")

    monkeypatch.setattr(torch, "all", fail_if_called)
    result = liger._validate_rms_weight(invalid, query)
    assert torch.equal(result, invalid)

    # Static shape/device/dtype checks remain active in the compile path.
    with pytest.raises(ValueError, match="shape/device"):
        liger._validate_rms_weight(torch.zeros(3, dtype=torch.float32), query)
