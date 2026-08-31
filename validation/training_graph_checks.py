"""Independent complete-step graph execution gate; no timing or kernel tuning."""
import copy
import gc
import traceback

import torch

from benchmarks.model import TrainingConfig, make_model, _microbatches
from .training_checks import PROTOCOL


def _cross_entropy_loss(logits, targets):
    return torch.nn.functional.cross_entropy(logits, targets)


def _ordinary_step(model, optimizer, loss_function, tokens, targets, accumulation):
    optimizer.zero_grad(set_to_none=True)
    batches = _microbatches(tokens, targets, accumulation)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in batches:
            logits = model(x)
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            (loss / len(batches)).backward()
    optimizer.step()
    return loss.detach()


def _state(model, optimizer):
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def _same_state(actual, expected, *, exact=False):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _same_state(actual[key], expected[key], exact=exact or key == "step")
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _same_state(a, e, exact=exact)
    elif isinstance(actual, torch.Tensor):
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=0 if exact else 1e-5,
                                   atol=0 if exact else 1e-6)
    else:
        assert actual == expected


def _case(config, variant, mode, accumulation):
    from benchmarks.training_graph import capture_training_step

    torch._dynamo.reset()
    torch.manual_seed(config.get("seed", PROTOCOL["seeds"][0]))
    shape = dict(PROTOCOL["smoke_model"])
    shape.update(config.get("model", {}))
    cfg = TrainingConfig(**shape, variant=variant, mode=mode,
                         rank=shape["width"] if variant == "standard" else 16)
    graph_model = make_model(cfg, backend="kernel").cuda()
    ordinary_model = make_model(cfg, backend="kernel").cuda()
    ordinary_model.load_state_dict(graph_model.state_dict())
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True,
                                   capturable=True) for m in (graph_model, ordinary_model)]
    compiled = [torch.compile(m, fullgraph=True, dynamic=False)
                for m in (graph_model, ordinary_model)]
    loss_fn = torch.compile(_cross_entropy_loss, fullgraph=True, dynamic=False)
    tokens = torch.randint(cfg.vocab, (cfg.batch, cfg.sequence), device="cuda")
    targets = torch.randint_like(tokens, cfg.vocab)
    # Exercise capture of an already compiled model with populated AdamW state.
    for model, optimizer in zip(compiled, optimizers):
        _ordinary_step(model, optimizer, loss_fn, tokens, targets, accumulation)
    torch.cuda.synchronize()
    _same_state(_state(graph_model, optimizers[0]), _state(ordinary_model, optimizers[1]))
    before_capture = _state(graph_model, optimizers[0])
    gc.collect()
    graph = capture_training_step(graph_model, optimizers[0], tokens, targets,
                                  accumulation=accumulation)
    _same_state(_state(graph_model, optimizers[0]), before_capture, exact=True)
    before_counters = dict(torch._dynamo.utils.counters["stats"])
    losses = []
    for index in range(2):
        changed_tokens = (tokens + index + 1) % cfg.vocab
        changed_targets = (targets + index + 3) % cfg.vocab
        graph.copy_inputs(changed_tokens, changed_targets)
        actual = graph.replay().clone()
        expected = _ordinary_step(compiled[1], optimizers[1], loss_fn,
                                           changed_tokens, changed_targets, accumulation)
        torch.cuda.synchronize()
        _same_state(actual, expected)
        _same_state(_state(graph_model, optimizers[0]), _state(ordinary_model, optimizers[1]))
        for a, e in zip(graph_model.parameters(), ordinary_model.parameters()):
            assert a.grad is not None and e.grad is not None
            _same_state(a.grad, e.grad)
        losses.append(float(actual))
    assert before_counters == dict(torch._dynamo.utils.counters["stats"])
    return {"two_changed_input_updates": True, "warmup_state_restored_exactly": True,
            "optimizer_steps_exact": True, "state_rtol": 1e-5, "state_atol": 1e-6,
            "accumulation": accumulation, "total_batch": cfg.batch, "losses": losses}


def run_graph_checks(config):
    cases = []
    for variant in config.get("variants", ["standard", "sliced"]):
        for mode in config.get("modes", ["full", "block"]):
            for accumulation in config.get("accumulations", [1, 2]):
                item = {"variant": variant, "mode": mode, "accumulation": accumulation}
                try:
                    item.update(status="passed", metrics=_case(config, variant, mode, accumulation))
                except Exception as exc:
                    item.update(status="failed", error=str(exc), traceback=traceback.format_exc())
                cases.append(item)
                gc.collect()
                torch.cuda.empty_cache()
    return {"passed": sum(c["status"] == "passed" for c in cases),
            "failed": sum(c["status"] == "failed" for c in cases), "cases": cases}
