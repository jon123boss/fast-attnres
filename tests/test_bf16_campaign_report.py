from benchmarks.bf16_campaign_report import render


def test_failed_alternative_does_not_relabel_a_complete_primary_result():
    summary = {"primary_pass": True, "candidate_identity": "a" * 64,
               "geometric_mean_speedup_observed_cells": 1.02,
               "cells": [{"cell": "H100/block/r64/seed20260827", "candidate_ms": 98,
                          "alternative_ms": 100, "ratio": .98,
                          "ci95_simultaneous": [.97, .99], "nonregression_pass": True,
                          "strongest_correct_alternative": "release"}],
               "failures": [{"cell": "H100/block/r64/seed20260827", "arms": {
                   "competitor": {"classification": "incorrect"}}}]}
    result = render(summary, {})
    assert "Primary target passed: **True**" in result
    assert "admitted comparisons: 1" in result
    assert "failure records: 1" in result
    assert "incomplete campaign" not in result


def test_negative_and_unadmitted_results_are_preserved():
    summary = {"primary_pass": False, "geometric_mean_speedup_observed_cells": .9,
               "admission_failures": [{"reason": "source identity changed"}],
               "adjacent_ranks": [{"comparison": "r64_over_r96", "ratio": 1.02,
                                    "ci95_simultaneous": [1.01, 1.03], "pass": False}]}
    result = render(summary, {})
    assert "**0.9×**" in result
    assert "admitted comparisons: not established" in result
    assert result.count("r64_over_r96") == 1
    assert "source identity changed" in result


def test_missing_data_and_failed_reservations():
    assert "unknown" in render({}, {})
    ledger = {"cap_usd": 500, "jobs": [{"stage": "experiments", "reserved_usd": 10,
                                          "status": "failed"}]}
    assert "Total reserved: **$10**" in render({}, ledger)
    ledger["jobs"].append({"stage": "experiments"})
    assert "Total reserved: **$unknown**" in render({}, ledger)
