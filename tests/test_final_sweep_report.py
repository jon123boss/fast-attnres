"""Failures in paired evidence must never become plotted speedup claims."""

import copy
import random

import pytest

from benchmarks.final_sweep_report import paired_vectors, project_cells, recompute
from benchmarks.plot_compiled_step_sweep import _balanced_schedule, _logical_input_hash, table_rows
from benchmarks.statistics import simultaneous_paired_ratio_bootstrap


def model_fixture():
    config = dict(layers=8, width=1024, heads=16, ffn=2816, batch=2,
                  sequence=512, vocab=8192, block_count=16, mode="full")
    vectors = {"kernel_rank_1024": [1., 1.1, 1.2, 1.3],
               "kernel_rank_256": [.9, 1., 1.1, 1.2],
               "fla_triton_compile_standard_rank_1024": [1.2, 1.3, 1.4, 1.5]}
    raw = [{"arm": name, "sample_index": index, "order_index": order,
            "rank": int(name.rsplit("_", 1)[1]),
            "backend": "kernel" if name.startswith("kernel_") else "fla_triton_compile",
            "input_hash": _logical_input_hash(17, index, config), "ms": vectors[name][index],
            "status": "ok", "timing_method": "cuda_graph", "replay_count": 1}
           for index, names in enumerate(_balanced_schedule(list(vectors), 4, 17))
           for order, name in enumerate(names)]
    return dict(config=config, ranks=[1024, 256], raw_samples=raw,
                compile={name: {"status": "ok"} for name in vectors},
                comparator_failures=[], architecture_comparisons={
                    "fla_triton_compile_standard_rank_1024": {}}), vectors


@pytest.mark.parametrize("field,value", [
    ("sample_index", 0), ("input_hash", "0" * 64), ("ms", float("nan")),
    ("replay_count", 2), ("order_index", 9), ("backend", "reference"), ("rank", 32),
])
def test_rejects_unpaired_or_invalid_samples(field, value):
    model, expected = model_fixture()
    assert paired_vectors(model, 17, 4) == expected
    model["raw_samples"][3][field] = value
    with pytest.raises(ValueError):
        paired_vectors(model, 17, 4)


def test_rejects_reordered_pairs_even_with_contiguous_order_indices():
    model, _ = model_fixture()
    rows = model["raw_samples"]
    rows[0], rows[1] = rows[1], rows[0]
    rows[0]["order_index"], rows[1]["order_index"] = 0, 1
    with pytest.raises(ValueError, match="paired arm schedule"):
        paired_vectors(model, 17, 4)


def test_quarter_rank_projection_preserves_different_equation_label():
    model, vectors = model_fixture()
    model.update(comparator_qualification={}, model_comparator_scope={
        f"{kind}_rank_{rank}": {"eligible": False, "reason": "unsupported"}
        for kind in ("liger", "catswe_phase1_model") for rank in model["ranks"]})
    baseline = "fla_triton_compile_standard_rank_1024"
    statistics = {f"kernel_rank_{rank}_over_{baseline}":
                  dict(ratio=.9, ci_low=.8, ci_high=1., n=4) for rank in model["ranks"]}
    cell = dict(model=model["config"], ranks=model["ranks"], warmups=5, rounds=4)
    projected = project_cells(model, vectors, statistics, cell, "H100", "fixture")
    assert projected[1].rank_relation == "R=D/4"
    row = next(r for r in table_rows(projected) if r["rank_r"] == 256)
    assert "different equation" in row["equation"]


def test_optional_failure_uses_the_producers_remaining_arm_schedule():
    from benchmarks.run import _paired_samples

    model, vectors = model_fixture()
    names = list(vectors)
    failed_arm = names[-1]
    rng = random.Random(17 + 771)
    warmup_order = list(names)
    rng.shuffle(warmup_order)

    def row_factory(name, sample, order):
        row = next(r for r in model["raw_samples"]
                   if r["arm"] == name and r["sample_index"] == sample).copy()
        for key in ("status", "timing_method", "replay_count"):
            row.pop(key)
        return {**row, "ms": None, "order_index": order}

    def measure(name, sample):
        if name == failed_arm and sample == 1:
            raise RuntimeError("fixture comparator failure")
        return dict(status="ok", ms=vectors[name][sample], timing_method="cuda_graph", replay_count=1)

    rows = _paired_samples(names, names, 4, rng, set(), row_factory, measure,
                           lambda name, row: model["comparator_failures"], "model_timing")
    model["raw_samples"] = rows
    assert paired_vectors(model, 17, 4) == {k: v for k, v in vectors.items() if k != failed_arm}
    skipped = next(row for row in rows if row["status"] == "skipped_due_to_failure")
    skipped["status"] = "failed"
    with pytest.raises(ValueError, match="failure transition"):
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


def test_phase_join_preserves_all_workloads_and_rejects_protocol_changes():
    from benchmarks.final_sweep_report import combined_contract
    headline = dict(cells=[dict(name='headline_full'), dict(name='screen_full')],
                    runtime=dict(torch='2.13.0+cu130'), identities=dict(model='old'))
    screen = copy.deepcopy(headline)
    screen['cells'] = screen['cells'][1:]
    screen['identities'] = dict(model='corrected')
    screen['candidate_package_sha256'] = 'cleaned'
    assert combined_contract(screen, headline) == headline
    screen['runtime']['torch'] = 'other'
    with pytest.raises(ValueError, match='protocols differ'):
        combined_contract(screen, headline)
    screen['runtime'] = headline['runtime']
    screen['cells'] = []
    with pytest.raises(ValueError, match='planned matrix'):
        combined_contract(screen, headline)


@pytest.mark.parametrize('change', ['exit', 'seed', 'source', 'summary'])
def test_headline_transition_rejects_incomplete_or_rebound_evidence(tmp_path, change):
    import json
    from benchmarks.final_sweep_report import phase_records
    cell = dict(name='headline_full', seeds=[1, 2, 3])
    records = [dict(name=f'headline_full-{seed}', sha256=f'report-{seed}') for seed in cell['seeds']]
    plan = dict(selected=[r['name'] for r in records], manifest_sha256='frozen')
    transition = dict(plan, signal='SIGTERM', records=records)
    summary = dict(status='running', results=copy.deepcopy(records))
    (tmp_path/'exit-code.txt').write_text('143\n')
    (tmp_path/'headline-transition-plan.json').write_text(json.dumps(plan))
    (tmp_path/'headline-complete.json').write_text(json.dumps(transition))
    assert phase_records(tmp_path, summary, dict(cells=[cell]), 'frozen', True) == ([cell], records)
    if change == 'exit':
        (tmp_path/'exit-code.txt').write_text('2\n')
    elif change == 'seed':
        transition['records'] = records[:-1]
    elif change == 'source':
        transition['manifest_sha256'] = 'different'
    else:
        summary['results'][0]['sha256'] = 'different'
    (tmp_path/'headline-complete.json').write_text(json.dumps(transition))
    with pytest.raises(ValueError):
        phase_records(tmp_path, summary, dict(cells=[cell]), 'frozen', True)
