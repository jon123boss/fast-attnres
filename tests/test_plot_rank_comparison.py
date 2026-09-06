"""Quarter-rank plots retain every workload and each independent seed."""

import json
from pathlib import Path

import pytest

from benchmarks.plot_rank_comparison import rank_comparisons, render_rank_comparison


def fixture_records():
    contract = json.loads((Path(__file__).parents[1] / "configs/bf16_final_sweep.json").read_text())
    records = []
    for index, cell in enumerate(contract["cells"]):
        width = cell["model"]["width"]
        for offset, seed in enumerate(cell["seeds"]):
            ratio = .94 + index * .03 + offset * .004
            standard = 12 + index + offset
            records.append(dict(gpu="H100", cell=cell["name"], seed=seed, sha256="fixture",
                means_ms={f"kernel_rank_{width}": standard,
                          f"kernel_rank_{width // 4}": standard * ratio},
                statistics={f"kernel_rank_{width // 4}_over_rank_{width}":
                            dict(ratio=ratio, ci_low=ratio - .015, ci_high=ratio + .015)}))
    return records, contract


def test_keeps_three_headline_intervals_unpooled():
    records, contract = fixture_records()
    rows = rank_comparisons(records, contract, "H100")
    assert len(rows) == 5
    headline = rows[0]
    assert [s["seed"] for s in headline["seeds"]] == contract["cells"][0]["seeds"]
    assert headline["standard_ms"] == 13
    assert headline["lower_time_pct"] == pytest.approx(5.6)
    assert [s["paired"]["ratio"] for s in headline["seeds"]] == pytest.approx([.94, .944, .948])
    assert rows[-1]["lower_time_pct"] < 0


def test_rejects_missing_seed():
    records, contract = fixture_records()
    with pytest.raises(ValueError, match="incomplete rank comparison"):
        rank_comparisons(records[1:], contract, "H100")


def test_renders_all_workloads_and_signed_changes(tmp_path):
    pytest.importorskip("matplotlib")
    records, contract = fixture_records()
    svg, png = render_rank_comparison(rank_comparisons(records, contract, "H100"), tmp_path, "H100")
    content = svg.read_text()
    assert content.count("95% CI") == 7
    for label in ("L24", "D=2048", "bs=2", "bs=8", "R=D/4", "lower time", "higher time"):
        assert label in content
    assert png.stat().st_size > 1000
