from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from benchmarks import run


PROTOCOL = {
    "fp32": {"rtol": 1e-3, "atol": 1e-4},
    "bf16": {"rtol": 0.05, "atol": 0.05},
}


class TinyTrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.projection = nn.Linear(3, 5)

    def forward(self, tokens):
        logits = self.projection(tokens.float())
        return logits.unsqueeze(1).expand(-1, tokens.shape[1], -1)


def _inputs():
    tokens = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4]])
    return tokens, targets


def _optimizer(model):
    optimizer, _ = run._adamw(
        model.parameters(),
        {"lr": 1e-2, "betas": (0.9, 0.95), "weight_decay": 0.1},
    )
    return optimizer


def _step(model, optimizer, tokens, targets):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
    loss.backward()
    optimizer.step()
    return loss.detach()


def _reference_factory(_config, _device):
    return TinyTrainingModel()


def _walk_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_tensors(item)


def test_model_qualification_requires_loss_parity(monkeypatch):
    calls = 0

    def loss_function(logits, targets):
        nonlocal calls
        calls += 1
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss + (0.25 if calls == 2 else 0.0)

    monkeypatch.setattr(torch, "autocast", lambda **_: nullcontext())
    reference = TinyTrainingModel()
    candidate = TinyTrainingModel()
    candidate.load_state_dict(reference.state_dict())
    tokens, targets = _inputs()
    with pytest.raises(AssertionError):
        run._model_qualification(
            reference, candidate, tokens, targets, PROTOCOL, loss_function
        )


def test_graph_reference_evidence_is_cpu_owned_and_exactly_two_steps(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda **_: nullcontext())
    candidate, reference = TinyTrainingModel(), TinyTrainingModel()
    reference.load_state_dict(candidate.state_dict())
    optimizer = _optimizer(candidate)
    tokens, targets = _inputs()
    before = run._clone_model_checkpoint(candidate)

    evidence = run._precompute_graph_reference_evidence(
        candidate_model=candidate,
        candidate_optimizer=optimizer,
        reference=reference,
        optimizer_config={"lr": 1e-2, "betas": (0.9, 0.95), "weight_decay": 0.1},
        tokens=tokens,
        targets=targets,
        accumulation=1,
        vocab=5,
        device=torch.device("cpu"),
    )

    assert len(evidence) == 2
    assert all(
        tensor.device.type == "cpu"
        for record in evidence
        for tensor in _walk_tensors(record)
    )
    for name, value in before.items():
        torch.testing.assert_close(candidate.state_dict()[name], value)


def test_complete_step_rejects_nonfinite_optimizer_update_and_restores(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda **_: nullcontext())
    candidate = TinyTrainingModel()
    optimizer = _optimizer(candidate)
    tokens, targets = _inputs()
    before_model = run._clone_model_checkpoint(candidate)
    before_optimizer = run._clone_optimizer_checkpoint(optimizer)

    def bad_step(step_tokens, step_targets):
        loss = _step(candidate, optimizer, step_tokens, step_targets)
        next(iter(optimizer.state.values()))["exp_avg"].fill_(float("nan"))
        return loss

    with pytest.raises(FloatingPointError, match="non-finite"):
        run._complete_step_qualification(
            candidate_model=candidate,
            candidate_optimizer=optimizer,
            candidate_step=bad_step,
            reference_factory=_reference_factory,
            optimizer_config={"lr": 1e-2},
            tokens=tokens,
            targets=targets,
            accumulation=1,
            protocol=PROTOCOL,
            device=torch.device("cpu"),
            cuda_graph=False,
            label="nonfinite update",
        )

    assert run._value_equal(run._clone_model_checkpoint(candidate), before_model)
    assert run._value_equal(run._clone_optimizer_checkpoint(optimizer), before_optimizer)


def test_cuda_graph_replay_requires_precomputed_reference_on_cuda():
    candidate, reference = TinyTrainingModel(), TinyTrainingModel()
    optimizer = _optimizer(candidate)
    tokens, targets = _inputs()

    with pytest.raises(RuntimeError, match="precomputed before capture"):
        run._graph_replay_qualification(
            candidate_model=candidate,
            candidate_optimizer=optimizer,
            graph_step=object(),
            reference_factory=_reference_factory,
            optimizer_config={"lr": 1e-2},
            tokens=tokens,
            targets=targets,
            accumulation=1,
            vocab=5,
            protocol=PROTOCOL,
            device=torch.device("cuda"),
            capture_inputs=(tokens, targets),
        )


def test_bf16_gate_accepts_small_nonassociative_difference():
    tolerance = run._tolerance(PROTOCOL, torch.bfloat16)
    metric = run._compare_state_values(
        torch.tensor([1.0, 100.0]),
        torch.tensor([1.04, 100.0]),
        tolerance,
        label="bf16 witness",
    )
    assert metric == pytest.approx(0.04)
    with pytest.raises(AssertionError):
        run._compare_state_values(
            torch.tensor([1.0]),
            torch.tensor([1.2]),
            tolerance,
            label="bf16 outlier",
        )


def test_model_timings_fails_closed_before_cuda_event_samples(monkeypatch):
    from benchmarks import model as model_module

    class FakeModel(nn.Linear):
        def __init__(self, config, backend):
            super().__init__(3, 5)
            self.config = config
            self.backend = backend

        def to(self, _device):
            return self

    gate_calls = []

    def blocked_gate(**kwargs):
        gate_calls.append(kwargs["candidate_model"])
        raise RuntimeError("complete-step mismatch")

    monkeypatch.setattr(model_module, "make_model", FakeModel)
    monkeypatch.setattr(run, "_model_inputs", lambda *args: _inputs())
    monkeypatch.setattr(run, "_model_qualification", lambda *args: {"status": "qualified"})
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(run, "_adamw", lambda parameters, config, **kwargs: (
        torch.optim.SGD(list(parameters), lr=0.01),
        "CPU fake optimizer",
    ))
    monkeypatch.setattr(run, "_compiled_training_step", lambda *args: torch.tensor(1.0))
    monkeypatch.setattr(run, "_check_model_gradients", lambda *args: None)
    monkeypatch.setattr(run, "_complete_step_qualification", blocked_gate)
    monkeypatch.setattr(
        run,
        "_cuda_event_call",
        lambda *args, **kwargs: pytest.fail("timing started before the gate passed"),
    )

    protocol = {
        "smoke_model": {
            "layers": 1,
            "width": 3,
            "heads": 1,
            "ffn": 5,
            "batch": 2,
            "sequence": 3,
            "vocab": 5,
            "block_count": 1,
        },
        "ranks": [3],
        "warmup": 1,
        "smoke_rounds": 1,
        "rounds": 1,
        "bootstrap_samples": 16,
        "plateau_margin": 0.01,
    }
    result = run._model_timings(
        protocol,
        {
            "variant": "standard",
            "mode": "full",
            "ranks": [3],
            "reference_timing": False,
            "include_fla": False,
            "model_rounds": 1,
            "model_warmup": 1,
        },
        "smoke",
        torch.device("cuda"),
        7,
        {},
    )

    assert len(gate_calls) == 1
    assert result["status"] == "failed"
    assert result["failures"][0]["phase"] == "model_complete_step_qualification"
    assert "raw_samples" not in result
