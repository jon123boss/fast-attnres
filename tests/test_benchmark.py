import json
import random

import pytest
import torch

from benchmarks.competitors import Comparator, model_backend
from benchmarks.run import (
    _balanced_orders,
    _check_operator_graph_parity,
    _model_config,
    _operator_case,
    _operator_timings,
    assert_frozen_hashes,
    load_protocol,
    run_suite,
)
from benchmarks.statistics import (
    classify_interval,
    paired_ratio_bootstrap,
    simultaneous_paired_ratio_bootstrap,
)


def test_frozen_protocol_hashes_are_verified():
    protocol, hashes = load_protocol()
    assert protocol["version"] == 1
    assert hashes["validation/oracle.py"]
    assert hashes == assert_frozen_hashes()


def test_paired_ratio_bootstrap_and_protocol_classification():
    result = paired_ratio_bootstrap(
        [10.0, 11.0, 9.0, 10.5],
        [8.0, 8.8, 7.5, 8.4],
        samples=1500,
        seed=7,
    )
    assert result["ratio"] < 1
    assert result["ci_high"] < 1
    assert result["classification"] == "gain"
    assert classify_interval(0.995, 1.005) == "plateau"
    assert classify_interval(1.02, 1.10) == "slowdown"
    assert classify_interval(0.8, 1.02) == "inconclusive"


def test_simultaneous_ci_uses_all_comparisons():
    result = simultaneous_paired_ratio_bootstrap(
        {
            "a": ([10.0, 11.0, 9.0], [8.0, 8.8, 7.5]),
            "b": ([10.0, 11.0, 9.0], [9.8, 11.1, 8.9]),
        },
        samples=1000,
        seed=11,
    )
    assert set(result) == {"a", "b"}
    assert all(item["simultaneous"] for item in result.values())
    assert result["a"]["ci_high"] < 1


def test_statistics_do_not_drop_unpaired_or_invalid_samples():
    with pytest.raises(ValueError):
        paired_ratio_bootstrap([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_ratio_bootstrap([1.0, float("nan")], [1.0, 2.0])


def test_run_suite_is_strictly_json_serializable_without_gpu_jobs():
    result = run_suite({"phases": []})
    json.dumps(result, allow_nan=False)
    assert result["contract"]["status"] == "verified"
    assert "comparators" in result


@pytest.mark.parametrize("phases", [["operator_timing"], "typo", [{}], None])
def test_unknown_phases_fail_before_environment_setup(monkeypatch, phases):
    def unexpected_setup(*args, **kwargs):
        raise AssertionError("invalid phases must fail before environment setup")

    monkeypatch.setattr("benchmarks.run._environment", unexpected_setup)
    result = run_suite({"phases": phases})
    assert result["status"] == "failed"
    assert result["failures"][0]["phase"] == "config"
    assert "unsupported phases" in result["failures"][0]["error"]["message"]


def test_model_defaults_and_seeded_abba_schedule():
    protocol, _ = load_protocol()
    model = _model_config(protocol, {}, "smoke")
    assert model["variant"] == "standard"
    assert model["mode"] == "full"
    schedule = _balanced_orders(["a", "b"], 4, random.Random(3))
    assert schedule[1] == list(reversed(schedule[0]))
    assert schedule[2] == schedule[0]


@pytest.mark.parametrize("layout", ["packed", "list"])
def test_model_source_layout_top_level_overrides_nested(layout):
    protocol = {"smoke_model": {"width": 128, "source_layout": "packed"}}
    other = "list" if layout == "packed" else "packed"
    model = _model_config(protocol, {"model_config": {"width": 128, "source_layout": other},
                                     "source_layout": layout}, "smoke")
    assert model["source_layout"] == layout
    assert _model_config(protocol, {"model_config": {"source_layout": other}}, "smoke")["source_layout"] == other


def test_protocol_tuple_fields_are_source_rows_width_rank():
    smoke = _operator_case([1, 2, 128, 16], 0)
    primary = _operator_case([9, 4096, 768], 0)
    assert (smoke["S"], smoke["N"], smoke["D"], smoke["R"]) == (1, 2, 128, 16)
    assert (primary["S"], primary["N"], primary["D"], primary["R"]) == (9, 4096, 768, 768)


@pytest.mark.parametrize("container", ["packed", "list", "tuple"])
@pytest.mark.parametrize("strided", [False, True])
def test_model_backend_restores_token_shape_without_source_axis(monkeypatch, container, strided):
    carrier = torch.randn(2, 2, 3, 10, dtype=torch.bfloat16, requires_grad=True)
    sources = (carrier[0, ..., ::2], carrier[1, ..., ::2], carrier[0, ..., ::2])
    if not strided:
        sources = tuple(source.contiguous() for source in sources)
    values = torch.stack(sources) if container == "packed" else (list(sources) if container == "list" else sources)
    query = torch.randn(5, dtype=torch.bfloat16, requires_grad=True)
    expected = sum((i + 1) * source for i, source in enumerate(sources)) + query
    upstream = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (carrier, query), upstream, retain_graph=True)

    def fake_fla_call(*, query, residuals, rms_weight):
        assert len(residuals) == len(sources)
        torch.testing.assert_close(rms_weight, torch.ones_like(query))
        for actual, source in zip(residuals, sources):
            assert actual.is_contiguous()
            torch.testing.assert_close(actual, source.reshape(-1, 5))
        return sum((i + 1) * residual for i, residual in enumerate(residuals)) + query

    comparator = Comparator("fake", fake_fla_call, status="available")
    backend = model_backend(comparator)

    def reject_pack(*args, **kwargs):
        raise AssertionError("native comparator must not pack the source list")

    monkeypatch.setattr(torch, "stack", reject_pack)
    monkeypatch.setattr(torch, "cat", reject_pack)
    result = backend(values, query)
    assert result.shape == (2, 3, 5)
    torch.testing.assert_close(result, expected)
    for actual, wanted in zip(torch.autograd.grad(result, (carrier, query), upstream), expected_grads):
        torch.testing.assert_close(actual, wanted)
    assert backend.accepts_source_list is True


@pytest.mark.parametrize("values,query,error", [
    ([], torch.ones(4, dtype=torch.bfloat16), ValueError),
    ([torch.ones(2, 4, dtype=torch.bfloat16), "invalid"], torch.ones(4, dtype=torch.bfloat16), TypeError),
    ([torch.ones(2, 4, dtype=torch.bfloat16), torch.ones(3, 4, dtype=torch.bfloat16)], torch.ones(4, dtype=torch.bfloat16), ValueError),
    ([torch.ones(2, 4, dtype=torch.bfloat16), torch.ones(2, 4)], torch.ones(4, dtype=torch.bfloat16), TypeError),
    ([torch.ones(2, 4, dtype=torch.bfloat16)], torch.ones(3, dtype=torch.bfloat16), ValueError),
])
def test_model_backend_rejects_invalid_sources_before_native_call(values, query, error):
    def unexpected_call(**kwargs):
        pytest.fail("invalid sources reached native FLA")

    backend = model_backend(Comparator("fake", unexpected_call, status="available"))
    with pytest.raises(error):
        backend(values, query)


def test_model_backend_rejects_removed_keys_argument():
    backend = model_backend(Comparator("fake", lambda **kwargs: None, status="available"))
    with pytest.raises(TypeError):
        backend(torch.ones(2, 3, 4, dtype=torch.bfloat16), torch.ones(4, dtype=torch.bfloat16), keys=None)


def test_cuda_graph_operator_timing_options_are_strictly_validated():
    protocol, _ = load_protocol()
    invalid_method = _operator_timings(
        protocol,
        [],
        {"operator_timing": "unknown"},
        torch.device("cpu"),
        protocol["seeds"][0],
        {},
    )
    invalid_replays = _operator_timings(
        protocol,
        [],
        {"operator_timing": "cuda_graph", "graph_replays": 0},
        torch.device("cpu"),
        protocol["seeds"][0],
        {},
    )
    invalid_dtype = _operator_timings(
        protocol,
        [],
        {"operator_dtype": "fp32"},
        torch.device("cpu"),
        protocol["seeds"][0],
        {},
    )
    assert invalid_method["status"] == "failed"
    assert invalid_replays["status"] == "failed"
    assert invalid_dtype["status"] == "failed"
    assert "BF16" in invalid_dtype["failures"][0]["error"]["message"]


def test_cuda_graph_parity_uses_independent_implicit_oracle(monkeypatch):
    from validation.oracle import oracle

    values = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    query = torch.randn(4, dtype=torch.bfloat16)
    upstream = torch.randn(3, 4, dtype=torch.bfloat16)
    graph_values = values.detach().clone().requires_grad_(True)
    graph_query = query.detach().clone().requires_grad_(True)
    graph_info = {
        "values": graph_values,
        "query": graph_query,
        "upstream": upstream.detach().clone(),
        "mode": "forward_backward",
    }

    class FakeGraph:
        def replay(self):
            graph_values.grad = None
            graph_query.grad = None
            oracle(graph_values, graph_query).backward(graph_info["upstream"])

    class FakeOutput:
        def detach(self):
            return self

        def clone(self):
            return oracle(graph_values, graph_query).detach().clone()

    graph_info["graph"] = FakeGraph()
    graph_info["output"] = FakeOutput()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    samples = []
    for seed in (1, 2):
        generator = torch.Generator().manual_seed(seed)
        samples.append(
            (
                torch.randn((2, 3, 4), generator=generator, dtype=torch.bfloat16),
                torch.randn((4,), generator=generator, dtype=torch.bfloat16),
                torch.randn((3, 4), generator=generator, dtype=torch.bfloat16),
            )
        )

    def should_not_run(values, query):
        del values, query
        raise AssertionError("parity must use the independent oracle")

    result = _check_operator_graph_parity(
        should_not_run,
        graph_info,
        samples,
        {"bf16": {"rtol": 0.05, "atol": 0.05}},
        torch.bfloat16,
        torch.device("cpu"),
    )
    assert result["status"] == "qualified"
    assert all(len(errors) == 2 for errors in result["gradient_max_abs"])


def test_model_progress_disabled_never_reads_clock(monkeypatch, capsys):
    from benchmarks import run

    def unexpected_clock():
        raise AssertionError("disabled progress must not read the clock")

    monkeypatch.setattr(run.time, "monotonic", unexpected_clock)
    run._model_progress_logger({})("qualification_core_start", "kernel_rank_16")
    assert not capsys.readouterr().out


def test_model_progress_output_failure_does_not_abort(monkeypatch):
    from benchmarks import run

    def broken_output(*args, **kwargs):
        raise OSError("test stdout unavailable")

    logger = run._model_progress_logger({"model_progress": True})
    monkeypatch.setattr("builtins.print", broken_output)
    logger("warmup_start", "kernel_rank_16")


def test_model_qualification_releases_reference_graph_before_candidate(monkeypatch):
    from contextlib import nullcontext
    from benchmarks.run import _model_qualification

    events = []

    class RecordedModel(torch.nn.Linear):
        def __init__(self, name):
            super().__init__(1, 4)
            self.name = name

        def forward(self, tokens):
            events.append((self.name, "forward"))
            output = super().forward(tokens)
            output.register_hook(lambda grad: events.append((self.name, "backward")))
            return output

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    reference, candidate = RecordedModel("reference"), RecordedModel("candidate")
    candidate.load_state_dict(reference.state_dict())
    result = _model_qualification(
        reference, candidate, torch.randn(2, 3, 1), torch.zeros(2, 3, dtype=torch.long),
        {"bf16": {"rtol": 0.05, "atol": 0.05}}, torch.nn.functional.cross_entropy,
    )
    assert events == [("reference", "forward"), ("reference", "backward"),
                      ("candidate", "forward"), ("candidate", "backward")]
    assert result["status"] == "qualified"
    assert result["parameter_count"] == 2


@pytest.mark.parametrize("model_progress", [False, True])
@pytest.mark.parametrize("failure_stage", [None, "comparator", "kernel", "restore"])
def test_standard_qualification_stages_and_restores_models(monkeypatch, capsys, failure_stage, model_progress):
    import weakref
    from contextlib import nullcontext
    from benchmarks import fla_compile, model, run

    made, compiled, moves = [], [], []

    def backend(values, query):
        del values, query
    backend.source_hash_metadata = {"source": "test"}

    class RecordedModel(torch.nn.Linear):
        def __init__(self, config, backend):
            super().__init__(2, 2)
            self.config, self.backend = config, backend
            self.location = "cpu"
            self.moves = []

        def to(self, device):
            self.location = str(device)
            self.moves.append(self.location)
            moves.append((self.backend, self.location))
            if failure_stage == "restore" and self.backend == "kernel" and len(self.moves) == 3:
                raise RuntimeError("simulated restore failure")
            return self

    def make_model(config, backend):
        instance = RecordedModel(config, backend)
        made.append(weakref.ref(instance))
        return instance

    def qualify(reference, candidate, *args):
        assert reference.location == candidate.location == "cuda"
        if failure_stage == "kernel" and candidate.backend == "kernel":
            raise RuntimeError("simulated kernel qualification failure")
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
        if candidate.backend is backend:
            assert made[1]().location == "cpu"
            if failure_stage == "comparator":
                raise RuntimeError("simulated comparator qualification failure")
        return {"status": "qualified"}

    def compile_model(fn, **kwargs):
        if isinstance(fn, RecordedModel):
            assert fn.location == "cuda"
            assert made[2]() is None  # Untimed standard reference was released.
            assert made[1]().moves == ["cuda", "cpu", "cuda"]
            if failure_stage == "comparator":
                assert made[3]() is None  # Failed comparator must not survive.
            compiled.append(fn.backend)
        return fn

    def stop_before_cuda(*args, **kwargs):
        raise RuntimeError("CPU test ends before CUDA execution")

    monkeypatch.setattr(model, "make_model", make_model)
    monkeypatch.setattr(fla_compile, "make_model_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(run, "_model_inputs", lambda *args: (torch.zeros(1), torch.zeros(1)))
    monkeypatch.setattr(run, "_model_qualification", qualify)
    monkeypatch.setattr(torch, "compile", compile_model)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", stop_before_cuda)
    protocol, _ = load_protocol()
    result = run._model_timings(protocol, {
        "variant": "sliced", "mode": "full", "ranks": [16],
        "reference_timing": False, "include_fla": False,
        "include_fla_compile": True, "fla_compile_backends": ["triton"],
        "standard_fla_comparison": True, "model_progress": model_progress,
    }, "smoke", torch.device("cuda"), 7, {})
    lines = capsys.readouterr().out.splitlines()
    if model_progress:
        rows = [json.loads(line) for line in lines]
        core = [row["stage"] for row in rows if row["arm"] == "kernel_rank_16"]
        assert core[:2] == ["qualification_core_start", "qualification_core_end"]
        assert all(set(row) == {"stage", "arm", "elapsed_s"} for row in rows)
        assert [row["elapsed_s"] for row in rows] == sorted(row["elapsed_s"] for row in rows)
    else:
        assert not lines
    if failure_stage == "kernel":
        assert len(made) == 2
        assert not compiled
        assert result["failures"][0]["phase"] == "model_qualification"
    elif failure_stage == "restore":
        assert not compiled
        assert moves[-2:] == [("kernel", "cuda"), (backend, "cuda")]
        assert result["failures"][0]["phase"] == "model_qualification_restore"
    else:
        assert compiled == (["kernel"] if failure_stage == "comparator" else ["kernel", backend])
    if failure_stage == "comparator":
        assert result["comparator_failures"][0]["phase"] == "model_comparator_qualification"


@pytest.mark.parametrize("selected", [None, ["gluon"]])
def test_selected_fla_discovery_failure_remains_explicit(monkeypatch, selected):
    from benchmarks import fla_compile, run

    attempted = []

    def unavailable(name, **kwargs):
        attempted.append(name)
        raise RuntimeError("test backend unavailable")

    def no_gpu(*args, **kwargs):
        raise RuntimeError("CPU test stops before GPU execution")

    monkeypatch.setattr(fla_compile, "make_model_backend", unavailable)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch.cuda, "synchronize", no_gpu)
    monkeypatch.setattr(run, "_model_qualification", lambda *args: {"status": "qualified"})
    config = {"ranks": [128], "reference_timing": False, "include_fla": False,
              "include_fla_compile": True}
    if selected is not None:
        config["fla_compile_backends"] = selected
    protocol, _ = load_protocol()
    result = run._model_timings(protocol, config, "smoke", torch.device("cpu"), 7, {})
    assert attempted == (selected if selected is not None else ["triton", "gluon"])
    assert [row["arm"] for row in result["comparator_failures"]] == [
        f"fla_{name}_compile" for name in attempted
    ]
    assert all(row["phase"] == "model_comparator_discovery"
               for row in result["comparator_failures"])


@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("mode", ["full", "block"])
def test_fla_qualification_matches_variant_and_schedule(monkeypatch, variant, mode):
    from benchmarks import fla_compile, run

    qualified = []

    class FakeBackend:
        source_hash_metadata = {"source": "test"}

        def __call__(self, values, query):
            del values, query
            raise AssertionError("CPU test must not execute the GPU backend")

    def qualify(reference, candidate, *args):
        assert reference.config == candidate.config
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
        qualified.append((reference.variant, reference.mode, reference.rank, candidate.backend))
        return {"status": "qualified"}

    def stop_before_cuda(*args, **kwargs):
        raise RuntimeError("CPU test ends before CUDA execution")

    backend = FakeBackend()
    monkeypatch.setattr(fla_compile, "make_model_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(run, "_model_qualification", qualify)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch.cuda, "synchronize", stop_before_cuda)
    protocol, _ = load_protocol()
    ranks = [128] if variant == "standard" else [16, 128]
    result = run._model_timings(protocol, {
        "variant": variant, "mode": mode, "ranks": ranks,
        "reference_timing": False, "include_fla": False,
        "include_fla_compile": True, "fla_compile_backends": ["triton"],
        "standard_fla_comparison": variant != "standard",
    }, "smoke", torch.device("cpu"), 7, {})
    assert qualified == [(variant, mode, rank, "kernel") for rank in ranks] + [
        ("standard", mode, 128, backend)]
    arm = "fla_triton_compile_" + ("" if variant == "standard" else "standard_") + "rank_128"
    assert result["comparator_qualification"] == {
        arm: {"status": "qualified"}}


@pytest.mark.parametrize("override", [
    {"variant": "standard"}, {"mode": "invalid"}, {"include_fla_compile": False},
])
def test_standard_fla_comparison_rejects_inapplicable_requests(monkeypatch, override):
    from benchmarks import run

    def unexpected_inputs(*args, **kwargs):
        raise AssertionError("invalid comparison must fail before device allocation")

    monkeypatch.setattr(run, "_model_inputs", unexpected_inputs)
    protocol, _ = load_protocol()
    result = run._model_timings(protocol, {
        "variant": "sliced", "mode": "full", "ranks": [16],
        "include_fla_compile": True, "standard_fla_comparison": True, **override,
    }, "smoke", torch.device("cpu"), 7, {})
    assert result["status"] == "failed"
    assert result["failures"][0]["phase"] == "model_setup"


def test_architectural_fla_statistics_compare_candidate_to_standard():
    from benchmarks.run import _model_comparisons

    standard = "fla_triton_compile_standard_rank_128"
    arms = {"reference_rank_128": {"rank": 128, "backend": "reference"},
            "kernel_rank_128": {"rank": 128, "backend": "kernel"},
            standard: {"rank": 128, "backend": "fla_triton_compile"}}
    raw = [{"arm": name, "status": "ok", "ms": value}
           for name, value in zip(arms, (10.0, 4.0, 5.0))]
    comparisons = _model_comparisons(raw, arms, [128], 1, True, {standard: {}})
    assert comparisons == {
        "kernel_rank_128_over_reference": ([10.0], [4.0]),
        f"kernel_rank_128_over_{standard}": ([5.0], [4.0]),
    }


@pytest.mark.parametrize("obsolete", [
    {"include_per_read": True},
    {"model_config": {"block_execution": "unsupported"}},
])
def test_removed_block_settings_fail_before_allocating_model_inputs(monkeypatch, obsolete):
    from benchmarks import run

    def unexpected_inputs(*args, **kwargs):
        raise AssertionError("invalid comparison must fail before device allocation")

    monkeypatch.setattr(run, "_model_inputs", unexpected_inputs)
    protocol, _ = load_protocol()
    config = {"variant": "sliced", "mode": "block", "ranks": [16], **obsolete}
    result = run._model_timings(
        protocol, config, "smoke", torch.device("cpu"), 7, {}
    )
    assert result["status"] == "failed"
    assert result["failures"][0]["phase"] == "model_setup"


@pytest.mark.parametrize("include_reference", [False, True])
def test_source_layout_statistics_are_list_over_packed(include_reference):
    from benchmarks.run import _model_comparisons

    arms = {"reference_rank_16": {"rank": 16, "backend": "reference"},
            "kernel_rank_16": {"rank": 16, "backend": "kernel"},
            "packed_rank_16": {"rank": 16, "backend": "kernel",
                               "comparison": "source_layout"}}
    raw = [{"arm": name, "status": "ok", "ms": value}
           for name, value in zip(arms, (10.0, 4.0, 5.0))]
    comparisons = _model_comparisons(raw, arms, [16], 1, include_reference, {})
    assert comparisons["kernel_rank_16_over_packed_rank_16"] == ([5.0], [4.0])
    assert "packed_rank_16_over_reference" not in comparisons


def test_source_layout_comparison_requires_list_before_input_allocation(monkeypatch):
    from benchmarks import run

    def unexpected_inputs(*args, **kwargs):
        raise AssertionError("invalid layout must fail before input allocation")

    monkeypatch.setattr(run, "_model_inputs", unexpected_inputs)
    protocol, _ = load_protocol()
    result = run._model_timings(protocol, {
        "variant": "sliced", "mode": "full", "ranks": [16],
        "include_packed_comparison": True,
    }, "smoke", torch.device("cpu"), 7, {})
    assert result["status"] == "failed"
    assert result["failures"][0]["phase"] == "model_setup"


@pytest.mark.parametrize("include_reference", [False, True])
def test_source_layout_report_records_actual_staging_and_arm_layout(monkeypatch,
                                                                   include_reference):
    from benchmarks import run

    def fake_step(model, *args, **kwargs):
        if callable(model.backend):
            return torch.tensor(10.)
        return torch.tensor(4. if model.config.source_layout == "list" else 5.)

    monkeypatch.setattr(run, "_model_qualification", lambda *args: {"status": "qualified"})
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(run, "_adamw", lambda *args, **kwargs: (None, "CPU test double"))
    monkeypatch.setattr(run, "_compiled_training_step", fake_step)
    monkeypatch.setattr(run, "_check_model_gradients", lambda *args: None)
    monkeypatch.setattr(run, "_cuda_event_call", lambda fn, device: (float(fn()), fn()))
    protocol, _ = load_protocol()
    result = run._model_timings(protocol, {
        "variant": "sliced", "mode": "full", "ranks": [16],
        "model_config": dict(protocol["smoke_model"], source_layout="list"),
        "reference_timing": include_reference, "include_fla": False,
        "include_packed_comparison": True, "model_rounds": 4,
        "model_warmup": 1, "bootstrap_samples": 100,
    }, "smoke", torch.device("cpu"), 7, {})
    assert result["status"] == "complete"
    assert result["qualification_staging"].startswith("CPU during source-layout")
    layouts = {row["arm"]: row["source_layout"] for row in result["raw_samples"]}
    expected = {"kernel_rank_16": "list", "packed_rank_16": "packed"}
    if include_reference:
        expected["reference_rank_16"] = "reference_stack"
    assert layouts == expected
    metadata = result["schedule_comparisons"]["kernel_rank_16_over_packed_rank_16"]
    assert metadata["packed_config"] == dict(metadata["list_config"], source_layout="packed")


@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("mode", ["full", "block"])
@pytest.mark.parametrize("failure_stage", [None, "packed", "kernel", "restore"])
def test_source_layout_qualification_state_and_lifetime(monkeypatch, variant, mode,
                                                       failure_stage):
    import weakref
    from contextlib import nullcontext
    from benchmarks import model, run

    made, compiled, qualified = [], [], []

    class RecordedModel(torch.nn.Linear):
        def __init__(self, config, backend):
            super().__init__(2, 2)
            self.config, self.backend = config, backend
            self.location, self.moves = "cpu", []

        def to(self, device):
            self.location = str(device)
            self.moves.append(self.location)
            if (failure_stage == "restore" and self.backend == "kernel"
                    and self.config.source_layout == "list" and len(self.moves) == 3):
                raise RuntimeError("simulated list restoration failure")
            return self

    def make_model(config, backend):
        instance = RecordedModel(config, backend)
        made.append(weakref.ref(instance))
        return instance

    def qualify(reference, candidate, *args):
        assert reference.config == candidate.config
        assert reference.location == candidate.location == "cuda"
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
        layout = candidate.config.source_layout
        qualified.append(layout)
        if layout == "packed":
            assert made[0]().location == made[1]().location == "cpu"
            for name, value in made[1]().state_dict().items():
                torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
        if failure_stage == ("packed" if layout == "packed" else "kernel"):
            raise RuntimeError("simulated layout qualification failure")
        return {"status": "qualified"}

    def compile_model(fn, **kwargs):
        if isinstance(fn, RecordedModel):
            assert fn.location == "cuda"
            assert made[0]() is None and made[2]() is None
            assert made[1]().moves == ["cuda", "cpu", "cuda"]
            if failure_stage == "packed":
                assert made[3]() is None
            compiled.append(fn.config.source_layout)
        return fn

    def stop_before_cuda(*args, **kwargs):
        raise RuntimeError("CPU test ends before CUDA execution")

    monkeypatch.setattr(model, "make_model", make_model)
    monkeypatch.setattr(run, "_model_inputs", lambda *args: (torch.zeros(1), torch.zeros(1)))
    monkeypatch.setattr(run, "_model_qualification", qualify)
    monkeypatch.setattr(torch, "compile", compile_model)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", stop_before_cuda)
    protocol, _ = load_protocol()
    rank = 128 if variant == "standard" else 16
    result = run._model_timings(protocol, {
        "variant": variant, "mode": mode, "ranks": [rank],
        "model_config": dict(protocol["smoke_model"], source_layout="list"),
        "reference_timing": False, "include_fla": False,
        "include_packed_comparison": True,
    }, "smoke", torch.device("cuda"), 7, {})
    if failure_stage == "kernel":
        assert len(made) == 2 and not compiled and qualified == ["list"]
        assert result["failures"][0]["phase"] == "model_qualification"
    elif failure_stage == "restore":
        assert not compiled
        assert result["failures"][0]["phase"] == "model_qualification_restore"
    else:
        assert compiled == (["list"] if failure_stage == "packed" else ["list", "packed"])
        assert qualified == ["list", "packed"]
        from dataclasses import asdict
        list_config = dict(protocol["smoke_model"], variant=variant, mode=mode,
                           rank=rank, source_layout="list")
        list_config = asdict(model.TrainingConfig(**list_config))
        expected_packed = dict(list_config, source_layout="packed")
        metadata = run._source_layout_metadata(list_config, expected_packed,
                                               variant=variant, mode=mode, rank=rank)
        assert metadata["kind"] == "same_equation_source_layout_only"
        assert metadata["ratio"] == "list/packed"
        assert metadata["packed_config"] == expected_packed
    if failure_stage == "packed":
        assert result["comparator_qualification"][f"packed_rank_{rank}"]["status"] == "failed"
        assert result["qualification"][f"rank_{rank}"]["status"] == "qualified"


@pytest.mark.parametrize("failed_cleanup_call", [1, 2])
def test_cleanup_failure_does_not_drop_selected_comparators(monkeypatch, failed_cleanup_call):
    from benchmarks import fla_compile, run

    qualified, cleanups = [], []

    def backend(values, query):
        del values, query
        raise AssertionError("CPU test must not execute the GPU backend")

    backend.source_hash_metadata = {"source": "test"}

    def qualify(reference, candidate, *args):
        qualified.append(candidate.backend)
        return {"status": "qualified"}

    def cleanup(device):
        cleanups.append(str(device))
        if len(cleanups) == failed_cleanup_call:
            raise RuntimeError("simulated cleanup failure")

    def stop_before_cuda(*args, **kwargs):
        raise RuntimeError("CPU test ends before CUDA execution")

    monkeypatch.setattr(fla_compile, "make_model_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(run, "_model_qualification", qualify)
    monkeypatch.setattr(run, "_release_qualification_memory", cleanup)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch.cuda, "synchronize", stop_before_cuda)
    protocol, _ = load_protocol()
    result = run._model_timings(protocol, {
        "variant": "sliced", "mode": "full", "ranks": [16],
        "reference_timing": False, "include_fla": False,
        "include_fla_compile": True, "fla_compile_backends": ["triton", "gluon"],
        "standard_fla_comparison": True,
    }, "smoke", torch.device("cpu"), 7, {})
    assert qualified == ["kernel", backend, backend]
    assert all(row["status"] == "qualified"
               for row in result["comparator_qualification"].values())
    assert any(row["phase"] == "model_comparator_cleanup"
               for row in result["comparator_failures"])


@pytest.mark.parametrize("mode", ["full", "block"])
@pytest.mark.parametrize("variant", ["standard", "sliced"])
def test_canonical_source_runner_qualifies_all_arm_states(monkeypatch, mode, variant):
    from contextlib import nullcontext
    from benchmarks import fla_compile, model, run
    from validation.oracle import oracle

    def backend(values, query, *, eps=2**-23, scale=1.0):
        if isinstance(values, (list, tuple)):
            values = torch.stack(values)
        return oracle(values, query, eps=eps, scale=scale)

    backend.accepts_source_list = True
    backend.source_hash_metadata = {"source": "CPU oracle test double"}

    def stop_before_gpu(*args, **kwargs):
        raise RuntimeError("CPU state test stops after independent qualifications")

    monkeypatch.setattr(fla_compile, "make_model_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(model, "attnres", backend)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", stop_before_gpu)
    protocol = json.loads((run.PROJECT_ROOT / "validation/protocol.json").read_text())
    ranks = [16] if variant == "standard" else [4, 16]
    result = run._model_timings(protocol, {
        "variant": variant, "mode": mode, "ranks": ranks,
        "model_config": {"layers": 2, "width": 16, "heads": 4, "ffn": 32,
                         "batch": 2, "sequence": 5, "vocab": 31, "block_count": 2,
                         "source_layout": "list"},
        "model_state_protocol": "canonical_implicit_max_rank_v1",
        "reference_timing": False, "include_fla": False,
        "include_fla_compile": True, "fla_compile_backends": ["triton"],
        "standard_fla_comparison": variant != "standard",
        "include_packed_comparison": True,
    }, "smoke", torch.device("cpu"), 20260827, {})
    assert result["failures"] and all(row["phase"] == "model_compile" for row in result["failures"])
    assert all(row["status"] == "qualified" for row in result["qualification"].values())
    fla_arm = ("fla_triton_compile_rank_16" if variant == "standard"
               else "fla_triton_compile_standard_rank_16")
    assert result["comparator_qualification"][fla_arm]["status"] == "qualified"
    state, records = result["state_protocol"], result["state_protocol"]["arms"]
    expected = {f"{prefix}_rank_{rank}" for rank in ranks
                for prefix in ("reference", "kernel", "packed_reference", "packed")}
    expected.add(fla_arm)
    if variant != "standard":
        expected.add("standard_reference_rank_16")
    assert set(records) == expected
    for name in ("common_fixed_state_hash",):
        assert {row[name] for row in records.values()} == {state["canonical_source"][name]}
    for rank in ranks:
        hashes = {records[f"{prefix}_rank_{rank}"]["initial_state_hash"]
                  for prefix in ("reference", "kernel", "packed_reference", "packed")}
        assert len(hashes) == 1
        assert result["comparator_qualification"][f"packed_rank_{rank}"]["status"] == "qualified"
        assert records[f"kernel_rank_{rank}"]["shape_metadata"]["queries.0"] == [rank]
    standard_ref = "reference_rank_16" if variant == "standard" else "standard_reference_rank_16"
    assert records[standard_ref]["initial_state_hash"] == records[fla_arm]["initial_state_hash"]


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("capture_failure_layout", [None, "list", "packed"])
def test_post_capture_device_error_preserves_diagnostics(
    monkeypatch, canonical, capture_failure_layout
):
    from benchmarks import run, training_graph

    captures = []
    sync_failures = []

    def capture(model, *args, **kwargs):
        layout = model.config.source_layout
        captures.append(layout)
        if layout == capture_failure_layout:
            raise RuntimeError("injected arm capture failure")
        return object()

    def synchronize(*args):
        if len(captures) == 2:
            sync_failures.append("asynchronous device fault")
            raise RuntimeError(sync_failures[-1])

    def no_timing(*args, **kwargs):
        pytest.fail("a device fault must return before any timing or statistics")

    monkeypatch.setattr(run, "_model_qualification", lambda *args: {"status": "qualified"})
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(run, "_adamw", lambda *args, **kwargs: (None, "CPU test double"))
    monkeypatch.setattr(run, "_compiled_training_step", lambda *args: torch.tensor(1.))
    monkeypatch.setattr(run, "_check_model_gradients", lambda *args: None)
    monkeypatch.setattr(training_graph, "capture_training_step", capture)
    monkeypatch.setattr(run, "_paired_samples", no_timing)
    monkeypatch.setattr(run, "simultaneous_paired_ratio_bootstrap", no_timing)
    protocol, _ = load_protocol()
    config = {
        "variant": "sliced", "mode": "block", "ranks": [16],
        "model_config": dict(protocol["smoke_model"], source_layout="list"),
        "reference_timing": False, "include_fla": False,
        "include_packed_comparison": True, "model_timing": "cuda_graph",
        "model_rounds": 4, "model_warmup": 1,
    }
    if canonical:
        config["model_state_protocol"] = "canonical_implicit_max_rank_v1"
    result = run._model_timings(protocol, config, "smoke", torch.device("cpu"), 7, {})
    assert result["status"] == "failed" and sync_failures == ["asynchronous device fault"]
    assert result["qualification"]["rank_16"]["status"] == "qualified"
    assert result["comparator_qualification"]["packed_rank_16"]["status"] == "qualified"
    assert ("state_protocol" in result) == canonical
    assert set(result["compile"]) == set(result["optimizer"]) == set(result["graph"])
    assert set(result["graph_counters"]) == {"kernel_rank_16", "packed_rank_16"}
    assert len(result["warmup"]) == 2
    assert all(row["status"] == "ok" for row in result["warmup"])
    sync_error = result["failures"][-1]
    assert sync_error["phase"] == "model_graph_sync" and "arm" not in sync_error
    assert "asynchronous device fault" in sync_error["error"]["message"]
    if capture_failure_layout is not None:
        name = "kernel_rank_16" if capture_failure_layout == "list" else "packed_rank_16"
        errors = result["failures"] if capture_failure_layout == "list" else result["comparator_failures"]
        assert any(row.get("arm") == name and row["phase"] == "model_graph_capture" for row in errors)
    assert "statistics" not in result and "raw_samples" not in result
