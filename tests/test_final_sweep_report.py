"""Failures in paired evidence must never become plotted speedup claims."""

import copy

import pytest

from benchmarks.final_sweep_report import paired_vectors, recompute
from benchmarks.plot_compiled_step_sweep import _logical_input_hash
from benchmarks.statistics import simultaneous_paired_ratio_bootstrap


def model_fixture():
    config = dict(layers=8, width=1024, heads=16, ffn=2816, batch=2,
                  sequence=512, vocab=8192, block_count=16, mode="full")
    vectors = {"kernel_rank_1024": [1., 1.1, 1.2, 1.3],
               "kernel_rank_256": [.9, 1., 1.1, 1.2],
               "fla_triton_compile_standard_rank_1024": [1.2, 1.3, 1.4, 1.5]}
    raw = [{"arm": name, "sample_index": index, "order_index": order,
            "input_hash": _logical_input_hash(17, index, config), "ms": values[index],
            "status": "ok", "timing_method": "cuda_graph", "replay_count": 1}
           for index in range(4) for order, (name, values) in enumerate(vectors.items())]
    return dict(config=config, ranks=[1024, 256], raw_samples=raw,
                comparator_failures=[], architecture_comparisons={
                    "fla_triton_compile_standard_rank_1024": {}}), vectors


@pytest.mark.parametrize("field,value", [
    ("sample_index", 0), ("input_hash", "0" * 64), ("ms", float("nan")),
    ("replay_count", 2), ("order_index", 9),
])
def test_rejects_unpaired_or_invalid_samples(field, value):
    model, expected = model_fixture()
    assert paired_vectors(model, 17, 4) == expected
    model["raw_samples"][3][field] = value
    with pytest.raises(ValueError):
        paired_vectors(model, 17, 4)


def test_rejects_statistics_that_do_not_match_observations():
    model, vectors = model_fixture()
    baseline = "fla_triton_compile_standard_rank_1024"
    pairs = {f"kernel_rank_{rank}_over_{baseline}":
             (vectors[baseline], vectors[f"kernel_rank_{rank}"]) for rank in model["ranks"]}
    pairs["kernel_rank_256_over_rank_1024"] = (vectors["kernel_rank_1024"], vectors["kernel_rank_256"])
    model["statistics"] = simultaneous_paired_ratio_bootstrap(pairs, seed=18017)
    assert recompute(model, vectors, 17) == model["statistics"]
    altered = copy.deepcopy(model)
    altered["statistics"]["kernel_rank_256_over_rank_1024"]["ratio"] = .5
    with pytest.raises(ValueError, match="statistics mismatch"):
        recompute(altered, vectors, 17)
