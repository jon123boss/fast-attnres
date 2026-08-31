from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import benchmarks.run as runner
from benchmarks.run import _compiled_training_step, _operator_graph_step


class _TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.embedding = nn.Embedding(7, 4)
        self.head = nn.Linear(4, 7)

    def forward(self, tokens):
        return self.head(self.embedding(tokens))


def _tiny_step(model, optimizer, tokens, targets):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
    loss.backward()
    optimizer.step()
    return loss.detach()


def _tiny_protocol():
    return {"bf16": {"rtol": 0.05, "atol": 0.05}}


def _tiny_inputs():
    return torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.tensor([[1, 2, 3], [4, 5, 6]])


def _tiny_reference_factory(_config, _device):
    return _TinyLanguageModel()


def _tiny_warmed_state():
    torch.manual_seed(17)
    model = _TinyLanguageModel()
    optimizer, _ = runner._adamw(model.parameters(), {"lr": 1e-2}, cuda_graph=False)
    tokens, targets = _tiny_inputs()
    _tiny_step(model, optimizer, tokens, targets)
    return model, optimizer, tokens, targets


def test_complete_step_qualification_checks_update_and_restores_state():
    model, optimizer, tokens, targets = _tiny_warmed_state()
    before_model = runner._clone_model_checkpoint(model)
    before_optimizer = runner._clone_optimizer_checkpoint(optimizer)
    before_named_optimizer = runner._named_optimizer_state(model, optimizer)
    before_gradients = runner._clone_named_gradients(model)

    report = runner._complete_step_qualification(
        candidate_model=model,
        candidate_optimizer=optimizer,
        candidate_step=lambda step_tokens, step_targets: _tiny_step(
            model, optimizer, step_tokens, step_targets
        ),
        reference_factory=_tiny_reference_factory,
        optimizer_config={"lr": 1e-2},
        tokens=tokens,
        targets=targets,
        accumulation=1,
        protocol=_tiny_protocol(),
        device=torch.device("cpu"),
        cuda_graph=False,
        label="tiny complete step",
    )

    assert report["status"] == "qualified"
    assert set(report["gradient_max_abs"]) == {
        name for name, _ in model.named_parameters()
    }
    assert report["candidate_parameter_updates"]
    assert report["candidate_optimizer_updates"]
    assert runner._value_equal(runner._clone_model_checkpoint(model), before_model)
    assert runner._value_equal(runner._clone_optimizer_checkpoint(optimizer), before_optimizer)
    assert runner._value_equal(
        runner._named_optimizer_state(model, optimizer), before_named_optimizer
    )
    assert runner._value_equal(runner._clone_named_gradients(model), before_gradients)


def test_complete_step_qualification_rejects_noop_candidate_and_restores_state():
    model, optimizer, tokens, targets = _tiny_warmed_state()
    before_model = runner._clone_model_checkpoint(model)
    before_optimizer = runner._clone_optimizer_checkpoint(optimizer)

    def noop_step(step_tokens, step_targets):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(step_tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), step_targets.reshape(-1)
            )
        loss.backward()
        return loss.detach()

    with pytest.raises(RuntimeError, match="no model parameter update"):
        runner._complete_step_qualification(
            candidate_model=model,
            candidate_optimizer=optimizer,
            candidate_step=noop_step,
            reference_factory=_tiny_reference_factory,
            optimizer_config={"lr": 1e-2},
            tokens=tokens,
            targets=targets,
            accumulation=1,
            protocol=_tiny_protocol(),
            device=torch.device("cpu"),
            cuda_graph=False,
            label="tiny no-op step",
        )

    assert runner._value_equal(runner._clone_model_checkpoint(model), before_model)
    assert runner._value_equal(runner._clone_optimizer_checkpoint(optimizer), before_optimizer)


def test_graph_replay_qualification_checks_changed_inputs_and_restores_state():
    model, optimizer, tokens, targets = _tiny_warmed_state()

    class FakeGraphStep:
        def __init__(self):
            self.inputs = (tokens, targets)

        def copy_inputs(self, new_tokens, new_targets):
            self.inputs = (new_tokens, new_targets)

        def replay(self):
            return _tiny_step(model, optimizer, *self.inputs)

    before_model = runner._clone_model_checkpoint(model)
    before_optimizer = runner._clone_optimizer_checkpoint(optimizer)
    report = runner._graph_replay_qualification(
        candidate_model=model,
        candidate_optimizer=optimizer,
        graph_step=FakeGraphStep(),
        reference_factory=_tiny_reference_factory,
        optimizer_config={"lr": 1e-2},
        tokens=tokens,
        targets=targets,
        accumulation=1,
        vocab=7,
        protocol=_tiny_protocol(),
        device=torch.device("cpu"),
        capture_inputs=(tokens, targets),
    )

    assert report["status"] == "qualified"
    assert report["replay_count"] == 2
    assert len(set(report["replay_input_hashes"])) == 2
    assert report["capture_input_hash"] not in report["replay_input_hashes"]
    assert runner._value_equal(runner._clone_model_checkpoint(model), before_model)
    assert runner._value_equal(runner._clone_optimizer_checkpoint(optimizer), before_optimizer)


def test_paired_order_preserves_small_arm_schedule_and_rng():
    for count in range(3):
        for rounds in (0, 1, 7, 120):
            expected_rng = random.Random(123)
            actual_rng = random.Random(123)
            arms = list("abc"[:count])
            first = arms.copy()
            expected_rng.shuffle(first)
            expected = [first.copy() if i % 2 == 0 else first[::-1] for i in range(rounds)]
            assert runner._balanced_orders(arms, rounds, actual_rng) == expected
            assert actual_rng.getstate() == expected_rng.getstate()
            assert arms == list("abc"[:count])


@pytest.mark.parametrize("profile_error", [False, True])
def test_model_trace_aggregates_device_events_after_one_replay(monkeypatch, profile_error):
    events = [
        SimpleNamespace(device_type="DeviceType.CPU", name="host", device_time_total=999., count=1),
        SimpleNamespace(device_type="DeviceType.CUDA", name="kernel", device_time_total=2., count=1),
        SimpleNamespace(device_type="DeviceType.CUDA", name="kernel", device_time_total=3., count=1),
    ]
    seen = []

    class Trace:
        def __enter__(self):
            seen.append("profile_enter")
            return self

        def __exit__(self, *args):
            seen.append("profile_exit")

        def events(self):
            if profile_error:
                raise RuntimeError("diagnostic failure")
            return events

    profiler = SimpleNamespace(ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
                               profile=lambda **kwargs: Trace())
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: profiler)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: seen.append("sync"))
    step = SimpleNamespace(replay=lambda: seen.append("replay"))
    result = runner._profile_cuda_graph_replay(step, torch.device("cpu"))
    assert seen == ["sync", "profile_enter", "replay", "profile_exit", "sync"]
    assert result["replay_count"] == 1
    if profile_error:
        assert result["status"] == "failed" and result["error"]["message"] == "diagnostic failure"
    else:
        assert result["status"] == "complete"
        assert result["cuda_kernels"] == [{"name": "kernel", "count": 2, "device_time_us": 5.}]
        assert result["cuda_launches"] == []
        assert result["trace_summary"]["status"] == "unavailable"


def test_model_trace_skips_non_graph_and_failed_or_partial_arms(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "_profile_cuda_graph_replay", lambda step, device: seen.append(step) or {"status": "complete"})
    names = ["okay", "failed", "partial", "no_capture"]
    rows = [{"arm": name, "status": "ok"} for name in names for _ in range(1 if name == "partial" else 2)]
    steps = {name: name for name in names if name != "no_capture"}
    skipped = runner._model_profile_report(names, {"failed"}, rows, 2, steps, "eager", torch.device("cpu"))
    assert skipped["status"] == "skipped" and not seen
    result = runner._model_profile_report(names, {"failed"}, rows, 2, steps, "cuda_graph", torch.device("cpu"))
    assert result["profiled_arms"] == seen == ["okay"]


def test_model_trace_rejects_unstable_timing_before_profiler_import(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("unstable timing must not invoke the profiler")

    monkeypatch.setattr(runner, "_profile_cuda_graph_replay", unexpected)
    stability = {"stable": False, "delta": {"graph_break": 1}}
    result = runner._model_profile_report(
        ["kernel"], set(), [{"arm": "kernel", "status": "ok"}], 1,
        {"kernel": object()}, "cuda_graph", torch.device("cuda:2"), stability,
    )
    assert result["status"] == "skipped" and result["qualification"] == "unqualified"
    assert result["timed_graph_counters"] == stability


def test_launch_trace_groups_resources_without_splitting_correlation_ids(tmp_path):
    args = {"grid": [512, 1, 1], "block": [128, 1, 1],
            "registers per thread": 64, "shared memory": 0}
    events = [{"cat": "kernel", "ph": "X", "name": "backward", "dur": 2.,
               "args": {**args, "correlation": index}} for index in range(35)]
    events += [{"cat": "kernel", "ph": "X", "name": "backward", "dur": 3.,
                "args": {**args, "grid": [1024, 1, 1], "correlation": 99}},
               {"cat": "not_a_kernel", "ph": "X", "name": "host", "dur": 1000.}]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    rows, summary = runner._profile_trace_kernel_rows(path)
    assert summary["kernel_event_count"] == 36
    assert summary["kernel_group_count"] == len(rows) == 2
    assert rows[0]["count"] == 35 and rows[0]["device_time_us"] == 70.
    assert rows[0]["correlation_ids"] == list(range(32))
    assert rows[0]["correlation_ids_omitted"] == 3
    assert summary["correlation_ids_omitted"] == 3
    assert summary["correlation_ids_truncated"]
    assert rows[0]["grid"] == args["grid"]
    retained, limited = runner._profile_trace_kernel_rows(path, limit=1)
    assert len(retained) == 1 and limited["truncated"]
    assert limited["dropped_kernel_group_count"] == 1


@pytest.mark.parametrize("metadata,expected", [
    ({}, "unavailable"),
    ({"grid": None, "block": None, "registers per thread": None, "shared memory": None}, "unknown"),
    ({"grid": "512,1,1", "block": [128, 1], "registers per thread": True, "shared memory": -1}, "unknown"),
])
def test_launch_trace_does_not_invent_missing_or_invalid_resources(tmp_path, metadata, expected):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [
        {"cat": "kernel", "ph": "X", "name": "kernel", "dur": 1., "args": metadata},
        {"cat": "kernel", "name": "missing_phase", "dur": 1., "args": {}},
    ]}))
    rows, summary = runner._profile_trace_kernel_rows(path)
    assert summary["kernel_event_count"] == 1 and rows[0]["name"] == "kernel"
    for field in ("grid", "block", "registers per thread", "shared memory"):
        assert rows[0]["metadata_status"][field] == expected


@pytest.mark.parametrize("invalid", [
    {"name": True}, {"name": ["kernel"]}, {"dur": "2.0"}, {"dur": True}, {"dur": -1.},
])
def test_launch_trace_rejects_malformed_kernel_events(tmp_path, invalid):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [
        {"cat": "kernel", "ph": "X", "name": "kernel", "dur": 2., **invalid},
    ]}))
    with pytest.raises(RuntimeError, match="exported CUDA kernel event"):
        runner._profile_trace_kernel_rows(path)


@pytest.mark.parametrize("bad_export", [False, True])
def test_launch_trace_export_preserves_events_and_cleans_up(monkeypatch, bad_export):
    paths, replays = [], []

    class Trace:
        def events(self):
            return [SimpleNamespace(device_type="DeviceType.CUDA", name="kernel",
                                    device_time_total=2., count=1)]

        def export_chrome_trace(self, path):
            paths.append(Path(path))
            if bad_export:
                Path(path).write_text("invalid JSON")
            else:
                Path(path).write_text(json.dumps({"traceEvents": [
                    {"cat": "kernel", "ph": "X", "name": "kernel", "dur": 2.,
                     "args": {"grid": [1, 1, 1], "block": [128, 1, 1]}},
                ]}))

    profiler = SimpleNamespace(ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
                               profile=lambda **kwargs: nullcontext(Trace()))
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: profiler)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    result = runner._profile_cuda_graph_replay(
        SimpleNamespace(replay=lambda: replays.append(1)), torch.device("cpu"))
    assert replays == [1] and len(paths) == 1 and not paths[0].exists()
    assert result["cuda_kernels"] == [{"name": "kernel", "count": 1, "device_time_us": 2.}]
    assert result["cuda_event_count"] == 1
    if bad_export:
        assert result["status"] == "failed"
        assert result["trace_summary"]["status"] == "failed"
    else:
        assert result["cuda_launches"][0]["grid"] == [1, 1, 1]


def test_model_trace_replay_enters_explicit_device(monkeypatch):
    from contextlib import contextmanager

    seen = []

    @contextmanager
    def device_context(device):
        assert device == torch.device("cuda:2")
        seen.append("enter")
        yield
        seen.append("exit")

    trace = SimpleNamespace(events=lambda: [SimpleNamespace(
        device_type="DeviceType.CUDA", name="kernel", device_time_total=3., count=1)])
    profiler = SimpleNamespace(ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
                               profile=lambda **kwargs: nullcontext(trace))
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: profiler)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    result = runner._profile_cuda_graph_replay(
        SimpleNamespace(replay=lambda: seen.append("replay")), torch.device("cuda:2"))
    assert seen == ["enter", "replay", "exit"]
    assert result["status"] == "complete"


@pytest.mark.parametrize("enabled", [False, True])
def test_model_trace_hook_follows_timing_and_stats_without_changing_status(monkeypatch, enabled):
    from benchmarks import training_graph

    phases = []
    monkeypatch.setattr(runner, "_model_qualification", lambda *args: {"status": "qualified"})
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(runner, "_adamw", lambda *args, **kwargs: (None, "CPU test double"))
    monkeypatch.setattr(runner, "_compiled_training_step", lambda *args: torch.tensor(1.))
    monkeypatch.setattr(runner, "_check_model_gradients", lambda *args: None)
    monkeypatch.setattr(runner, "_dynamo_counters", lambda: {})
    monkeypatch.setattr(training_graph, "capture_training_step", lambda *args, **kwargs: SimpleNamespace(
        copy_inputs=lambda *args: None, replay=lambda: torch.tensor(1.)))
    monkeypatch.setattr(runner, "_cuda_event_call", lambda fn, device: (phases.append("timed") or 1., fn()))
    estimator = runner.simultaneous_paired_ratio_bootstrap

    def statistics(*args, **kwargs):
        phases.append("statistics")
        return estimator(*args, **kwargs)

    def diagnostic(*args):
        assert phases[-1] == "statistics"
        phases.append("profile")
        raise RuntimeError("optional profiler failure")

    monkeypatch.setattr(runner, "simultaneous_paired_ratio_bootstrap", statistics)
    monkeypatch.setattr(runner, "_model_profile_report", diagnostic)
    protocol = {"smoke_model": {"layers": 1, "width": 16, "heads": 2, "ffn": 32,
                               "batch": 1, "sequence": 4, "vocab": 32, "block_count": 1},
                "ranks": [16], "warmup": 1, "smoke_rounds": 2, "rounds": 2,
                "bootstrap_samples": 32, "plateau_margin": .01}
    result = runner._model_timings(protocol, {
        "variant": "standard", "mode": "full", "ranks": [16], "reference_timing": True,
        "include_fla": False, "include_fla_compile": False, "model_timing": "cuda_graph",
        "model_rounds": 2, "model_warmup": 1, "model_profile": enabled,
    }, "smoke", torch.device("cpu"), 7, {})
    assert result["status"] == "complete" and not result["failures"]
    assert len(result["raw_samples"]) == 4 and all(row["ms"] == 1. for row in result["raw_samples"])
    assert phases == ["timed"] * 4 + ["statistics"] + (["profile"] if enabled else [])
    assert result["model_profile"]["status"] == ("failed" if enabled else "disabled")


@pytest.mark.parametrize("count", [3, 4, 5])
def test_paired_order_balances_every_position_without_extra_rng(count):
    arms = list("abcde"[:count])
    rng, expected_rng = random.Random(71), random.Random(71)
    first = arms.copy()
    expected_rng.shuffle(first)
    orders = runner._balanced_orders(arms, 120, rng)
    assert rng.getstate() == expected_rng.getstate()
    for index, order in enumerate(orders):
        assert sorted(order) == arms
        if index % 2:
            assert order == orders[index - 1][::-1]
    for position in range(count):
        assert Counter(order[position] for order in orders) == {arm: 120 // count for arm in arms}
    if count == 3:
        assert sorted(Counter(tuple(order) for order in orders).values()) == [20] * 6


def test_rotated_pairs_preserve_failed_and_skipped_rows():
    failures, failed = [], set()

    def row(name, sample, order):
        return {"arm": name, "sample_index": sample, "order_index": order,
                "input_hash": f"input-{sample}"}

    def measure(name, sample):
        if name == "b" and sample == 1:
            raise RuntimeError("controlled failure")
        return {"status": "ok", "ms": 1.0}

    rows = runner._paired_samples(
        list("abc"), list("abc"), 6, random.Random(71), failed, row,
        measure, lambda name, result: failures, "test",
    )
    assert len(rows) == 18 and len(failures) == 1 and failed == {"b"}
    for sample in range(6):
        group = [item for item in rows if item["sample_index"] == sample]
        assert sorted(item["arm"] for item in group) == list("abc")
        assert {item["input_hash"] for item in group} == {f"input-{sample}"}
        present = [item for item in group if item["order_index"] is not None]
        assert [item["order_index"] for item in present] == list(range(len(present)))
    b_rows = [item for item in rows if item["arm"] == "b"]
    assert [item["status"] for item in b_rows] == ["ok", "failed"] + ["skipped_due_to_failure"] * 4


def test_compiled_step_splits_distinct_microbatches(monkeypatch):
    """Accumulation must consume each batch slice once, not repeat the full batch."""

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(1, 2)
            self.calls = []

        def forward(self, tokens):
            self.calls.append(tokens.detach().clone())
            return self.projection(tokens.unsqueeze(-1))

    monkeypatch.setattr(torch, "autocast", lambda **_: nullcontext())
    model = Recorder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tokens = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    targets = torch.zeros_like(tokens, dtype=torch.long)

    result = _compiled_training_step(
        model,
        optimizer,
        torch.nn.functional.cross_entropy,
        tokens,
        targets,
        accumulation=2,
    )

    assert [call.flatten().tolist() for call in model.calls] == [[0.0, 1.0], [2.0, 3.0]]
    assert result.ndim == 0
    assert not result.requires_grad


def test_operator_warmup_passes_mode_and_upstream_in_order(monkeypatch):
    """The warmup call must match ``_operator_step``'s mode/upstream order."""

    calls = []

    def fake_factory(name, comparator=None):
        del comparator

        def operator(values, query):
            del query
            calls.append(name)
            return values[0]

        return operator

    seen = []
    original_step = runner._operator_step

    def checked_step(function, values, query, mode, upstream):
        seen.append((mode, upstream))
        return original_step(function, values, query, mode, upstream)

    monkeypatch.setattr(runner, "_operator_function", fake_factory)
    monkeypatch.setattr(runner, "_qualify_operator", lambda *args: {"status": "qualified"})
    monkeypatch.setattr(runner, "_operator_step", checked_step)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(runner, "_cuda_event_call", lambda function, device: (1.0, function()))

    protocol = {
        "smoke_rounds": 1,
        "rounds": 1,
        "warmup": 1,
        "bootstrap_samples": 32,
        "plateau_margin": 0.01,
        "bf16": {"rtol": 0.05, "atol": 0.05},
        "fp32": {"rtol": 0.001, "atol": 0.0001},
    }
    case = {"id": "operator_0", "S": 1, "N": 2, "D": 4, "R": 4}
    result = runner._operator_timings(
        protocol,
        [case],
        {"operator_rounds": 1, "operator_warmup": 1, "operator_modes": ("forward_backward",)},
        torch.device("cpu"),
        7,
        {},
    )

    assert result["status"] == "complete"
    assert calls
    assert len(seen) == len(calls)
    assert all(mode == "forward_backward" for mode, _ in seen)
    assert all(isinstance(upstream, torch.Tensor) for _, upstream in seen)


def test_graph_step_reuses_preallocated_gradients():
    values = torch.randn(1, 2, 3, requires_grad=True)
    query = torch.randn(3, requires_grad=True)
    upstream = torch.ones_like(values)

    def operator(values, query):
        return values + query

    _operator_graph_step(operator, values, query, "forward_backward", upstream)
    gradient_ids = [id(tensor.grad) for tensor in (values, query)]
    _operator_graph_step(operator, values, query, "forward_backward", upstream)

    assert all(tensor.grad is not None for tensor in (values, query))
    assert [id(tensor.grad) for tensor in (values, query)] == gradient_ids


def test_operator_statistics_use_kernel_as_baseline(monkeypatch):
    last_arm = {"name": None}

    def fake_factory(name, comparator=None):
        del comparator

        def operator(values, query):
            del query
            last_arm["name"] = name
            return values[0]

        return operator

    monkeypatch.setattr(runner, "_operator_function", fake_factory)
    monkeypatch.setattr(runner, "_qualify_operator", lambda *args: {"status": "qualified"})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)

    def fake_event(function, device):
        del device
        output = function()
        return (2.0 if last_arm["name"] == "kernel" else 1.0, output)

    monkeypatch.setattr(runner, "_cuda_event_call", fake_event)
    protocol = {
        "smoke_rounds": 2,
        "rounds": 2,
        "warmup": 1,
        "bootstrap_samples": 32,
        "plateau_margin": 0.01,
        "bf16": {"rtol": 0.05, "atol": 0.05},
        "fp32": {"rtol": 0.001, "atol": 0.0001},
    }
    case = {"id": "operator_0", "S": 1, "N": 2, "D": 4, "R": 4}
    result = runner._operator_timings(
        protocol,
        [case],
        {"operator_rounds": 2, "operator_warmup": 1, "operator_modes": ("forward",)},
        torch.device("cpu"),
        11,
        {},
    )

    assert result["status"] == "complete"
    assert result["cases"][0]["statistics"]["forward"]["reference"]["ratio"] == 0.5


def test_invalid_scope_fails_before_cuda_setup(monkeypatch):
    monkeypatch.setattr(runner, "_environment", lambda root: {})
    monkeypatch.setattr(runner, "_device_info", lambda device=None: (_ for _ in ()).throw(AssertionError("CUDA setup should be skipped")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CUDA setup should be skipped")))

    result = runner.run_suite({"scope": "invalid"})

    assert result["status"] == "failed"
    assert result["coverage"]["scope"] == "invalid"
    assert result["failures"][0]["phase"] == "config"


@pytest.mark.parametrize("variant,rank", [("standard", 17), ("sliced", 5)])
def test_source_profile_checks_all_gradients_without_accumulation(variant, rank):
    from attnres import attnres
    from benchmarks import source_profile as profile

    case = {"variant": variant, "shape": [3, 5, 17, rank]}
    arms = profile._make_arms(case, torch.device("cpu"), 20260827, True)
    protocol = {"bf16": {"rtol": .05, "atol": .05}}
    for name, arm in arms.items():
        result = profile._qualify(attnres, arm, protocol)
        tensors = profile._tensors(arm)
        assert len(result["gradient_max_abs"]) == len(tensors)
        assert all(t.is_leaf and t.grad is None for t in tensors)
        assert arm["query"].dtype == torch.float32
        for tensor in tensors:
            tensor.grad = torch.full_like(tensor, 7)
        _, gradients = profile._step(attnres, arm, "forward_backward")
        assert len(gradients) == len(tensors)
        assert all(torch.equal(t.grad, torch.full_like(t, 7)) for t in tensors)
        if name != "packed":
            torch.testing.assert_close(torch.stack(arm["values"]), arms["packed"]["values"])


@pytest.mark.parametrize("variant,rank", [("standard", 8), ("sliced", 3)])
def test_source_profile_entrypoint_hashes_implicit_inputs(monkeypatch, variant, rank):
    from benchmarks import source_profile as profile

    make_arms = profile._make_arms
    arms_seen = []

    def cpu_arms(case, device, seed, baseline):
        arms = make_arms(case, torch.device("cpu"), seed, baseline)
        arms_seen.append(arms)
        return arms

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profile, "load_protocol", lambda root: (
        {"bf16": {"rtol": .05, "atol": .05}, "warmup": 0, "smoke_rounds": 1}, {}))
    monkeypatch.setattr(profile, "_make_arms", cpu_arms)
    monkeypatch.setattr(profile, "_metric", lambda *args: ({"status": "not_timed_cpu_control"}, []))
    monkeypatch.setattr(profile, "_resource_metadata", lambda variants: {"status": "unavailable"})
    result = profile.run_source_profile({"cases": [{"variant": variant, "shape": [3, 2, 8, rank]}]})
    assert result["status"] == "complete"
    assert result["failures"] == []
    packed = arms_seen[0]["packed"]
    assert result["cases"][0]["input_hash"] == profile._operator_digest(
        packed["values"], packed["query"], packed["upstream"])
    assert result["input_contract"]["input_hash_schema"] == "values-query-upstream-v1"
    assert "packing and producer-gradient assembly excluded" in result["input_contract"]["comparison_boundary"]


def test_source_profile_reads_predicated_ptx_and_tuple_constants(monkeypatch):
    from benchmarks import source_profile as profile

    compiled = type("CompiledKernel", (), {})()
    compiled.n_regs, compiled.n_spills = 64, 0
    compiled.metadata = SimpleNamespace(shared=256, num_warps=4, num_stages=3)
    compiled.src = SimpleNamespace(constants={(0,): True, (1,): 1024},
                                   fn=SimpleNamespace(arg_names=["LIST_SOURCES", "D"]))
    compiled.asm = {"ptx": "@%p1 ld.global.v4.b32 {%r1}, [%rd1];\n"
                            "ld.local.b32 %r2, [%rd2];\n"
                            "@!%p2 st.global.b16 [%rd3], %rs1;\n"}
    jit = type("JITFunction", (), {})()
    jit.device_caches = {0: ({"key": compiled}, {}, None, None, None)}
    module = SimpleNamespace(_test_kernel=jit)
    monkeypatch.setitem(profile.sys.modules, "attnres._kernels.fixed_tail", module)
    result = profile._resource_metadata(["standard", "sliced"])
    assert len(result["kernels"]) == 1
    resources = result["kernels"][0]["resources"]
    assert resources["constants"] == {"LIST_SOURCES": True, "D": 1024}
    assert resources["num_warps"] == 4
    assert resources["ptx_instruction_counts"] == {
        "ld.global.v4.b32": 1, "ld.local.b32": 1, "st.global.b16": 1,
    }


@pytest.mark.parametrize("variant,helper", [
    ("standard", "attnres._kernels.fixed_tail"),
    ("sliced", "_attnres_frozen_fixture._kernels.fixed_tail"),
])
def test_source_profile_reads_loaded_fixed_tail_helpers(monkeypatch, variant, helper):
    import importlib
    from benchmarks import source_profile as profile

    imports = []
    modules = {}

    def load(name):
        imports.append(name)
        raise ImportError("resource inspection must not load a kernel module")

    monkeypatch.setattr(importlib, "import_module", load)
    monkeypatch.delitem(profile.sys.modules, helper, raising=False)
    assert profile._resource_metadata([variant])["status"] == "unavailable"
    assert not imports

    compiled = type("CompiledKernel", (), {})()
    compiled.n_regs, compiled.n_spills = 80, 0
    compiled.src = SimpleNamespace(constants={(0,): 2},
                                   fn=SimpleNamespace(arg_names=["BL"]))
    jit = type("JITFunction", (), {})()
    jit.device_caches = {0: ({"key": compiled}, {}, None, None, None)}
    modules[helper] = SimpleNamespace(_native_kernel=jit)
    monkeypatch.setitem(profile.sys.modules, helper, modules[helper])
    result = profile._resource_metadata([variant, variant])
    assert len(result["kernels"]) == 1
    assert result["kernels"][0]["kernel"] == f"{helper}._native_kernel"
    assert result["kernels"][0]["resources"]["constants"] == {"BL": 2}
    assert "not a runtime trace" in result["resource_scope"]


def test_source_profile_groups_replays_inside_one_event(monkeypatch):
    from benchmarks import source_profile as profile

    calls = {"source_list": 0, "packed": 0}
    events = []

    def graph(function, *args):
        def replay():
            calls[function] += 1
        return {"graph": SimpleNamespace(replay=replay), "capture_host_ms": 0}

    def event(function, device):
        events.append(1)
        return 6., function()

    monkeypatch.setattr(profile, "_graph", graph)
    monkeypatch.setattr(profile, "_verify_graph", lambda *args: {"initial_and_changed_inputs": "qualified"})
    monkeypatch.setattr(profile, "_event", event)
    names = list(calls)
    result, failures = profile._metric(
        {name: name for name in names}, {name: {} for name in names}, names, names,
        {"id": "fixture", "variant": "sliced", "shape": [3, 5, 17, 5]}, "same-input",
        "forward_backward", "cuda_graph", torch.device("cpu"), 4, 2, 3, 7,
        {"bootstrap_samples": 32, "plateau_margin": .01},
    )
    assert not failures
    assert calls == {"source_list": 12, "packed": 12}
    assert len(events) == 8
    assert all(row["elapsed_ms"] == 6. and row["ms"] == 2. for row in result["raw_samples"])
    assert result["statistics"]["comparisons"]["source_over_packed"]["ratio"] == 1.


@pytest.mark.parametrize("variant,layout", [("sliced", "independent")])
@pytest.mark.parametrize("arm_name", ["source_list", "packed"])
@pytest.mark.parametrize("mode", ["forward", "forward_backward"])
def test_source_profile_checks_replay_outputs_and_restores_inputs(variant, layout, arm_name, mode):
    from benchmarks import source_profile as profile
    from validation.oracle import oracle

    case = {"variant": variant, "shape": [3, 5, 17, 5], "layout": layout}
    original = profile._make_arms(case, torch.device("cpu"), 19, False)[arm_name]
    static = profile._make_arms(case, torch.device("cpu"), 19, False)[arm_name]
    state = {"static": static}
    replays = []

    def function(values, query, **kwargs):
        if isinstance(values, tuple):
            values = torch.stack(values)
        return oracle(values, query, **kwargs)

    def replay():
        result = profile._step(function, static, mode)
        state["output"], state["grads"] = result if mode == "forward_backward" else (result, ())
        replays.append(state["output"].detach().clone())

    state["graph"] = SimpleNamespace(replay=replay)
    protocol = {"bf16": {"rtol": .05, "atol": .05}}
    rng = torch.random.get_rng_state()
    result = profile._verify_graph(state, original, mode, protocol)
    torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)
    assert result["initial_and_changed_inputs"] == "qualified"
    assert len(replays) == 3 and not torch.equal(replays[0], replays[1])
    torch.testing.assert_close(replays[0], replays[2], rtol=0, atol=0)
    for actual, expected in zip(profile._tensors(static), profile._tensors(original)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if mode == "forward_backward":
        state["grads"] = (state["grads"][0] + 1, *state["grads"][1:])
    else:
        state["output"] = state["output"] + 1
    with pytest.raises(AssertionError):
        profile._check_captured(state, mode, protocol)
