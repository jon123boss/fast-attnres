"""Root-owned complete-training correctness gate; no timing or kernel selection."""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
import traceback

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from benchmarks.model import TrainingConfig, make_model, training_step
from .oracle import oracle

PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())


def _oracle_backend(values, query, *, eps=2**-23, scale=1.0):
    """Run the frozen BF16 test oracle without using a package reference path."""

    if isinstance(values, (list, tuple)):
        values = torch.stack(tuple(values), dim=0)
    return oracle(values, query, eps=eps, scale=scale)


def _close(actual, expected, name):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"nonfinite {name}")
    torch.testing.assert_close(actual, expected, **PROTOCOL["bf16"], msg=name)
    return float((actual.float() - expected.float()).abs().max())


def _loss(model, tokens, targets, use_checkpoint=False):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = checkpoint(model, tokens, use_reentrant=False) if use_checkpoint else model(tokens)
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
    return logits, loss


def _output_and_grads(function, model, tokens, targets):
    logits, loss = function(tokens, targets)
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, tuple(named.values()))
    return logits.detach(), loss.detach(), dict(zip(named, gradients))


def _parity(actual, expected):
    logits, loss, grads = actual
    ref_logits, ref_loss, ref_grads = expected
    assert grads.keys() == ref_grads.keys()
    return {
        "logits_max_abs": _close(logits, ref_logits, "logits"),
        "loss_max_abs": _close(loss, ref_loss, "loss"),
        "gradient_max_abs": {n: _close(g, ref_grads[n], n) for n, g in grads.items()},
    }


def _case(config, variant, mode):
    torch._dynamo.reset()
    torch.manual_seed(config.get("seed", PROTOCOL["seeds"][0]))
    shape = dict(PROTOCOL[config.get("scope", "smoke") + "_model"])
    shape.update(config.get("model", {}))
    rank = shape["width"] if variant == "standard" else config.get("rank", 16)
    cfg = TrainingConfig(**shape, variant=variant, mode=mode, rank=rank)
    model = make_model(cfg, backend="kernel").cuda()
    oracle_model = make_model(cfg, backend=_oracle_backend).cuda()
    oracle_model.load_state_dict(model.state_dict())
    assert all(torch.count_nonzero(q).item() for q in model.queries)
    tokens = torch.randint(cfg.vocab, (cfg.batch, cfg.sequence), device="cuda")
    targets = torch.randint_like(tokens, cfg.vocab)
    eager = lambda x, y: _loss(model, x, y)
    oracle_loss = lambda x, y: _loss(oracle_model, x, y)
    expected = _output_and_grads(oracle_loss, oracle_model, tokens, targets)
    metrics = {"eager": _parity(_output_and_grads(eager, model, tokens, targets), expected)}
    compiled = torch.compile(eager, fullgraph=True, dynamic=False)
    metrics["compiled"] = _parity(_output_and_grads(compiled, model, tokens, targets), expected)
    counters = torch._dynamo.utils.counters
    before = dict(counters["stats"])
    tokens.copy_(tokens.roll(1, dims=1))
    targets.copy_(targets.roll(1, dims=1))
    expected = _output_and_grads(oracle_loss, oracle_model, tokens, targets)
    metrics["changed_input"] = _parity(_output_and_grads(compiled, model, tokens, targets), expected)
    if dict(counters["stats"]) != before:
        raise AssertionError("unexpected recompilation on changed input")
    checked = torch.compile(lambda x, y: _loss(model, x, y, True), fullgraph=True, dynamic=False)
    metrics["activation_checkpoint"] = _parity(_output_and_grads(checked, model, tokens, targets), expected)

    # Compare complete accumulated optimizer steps, including every parameter.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    oracle_optimizer = torch.optim.AdamW(oracle_model.parameters(), lr=1e-3, fused=True)
    micro_tokens = torch.stack((tokens, tokens.roll(1, dims=0)))
    micro_targets = torch.stack((targets, targets.roll(1, dims=0)))
    for current, opt in ((model, optimizer), (oracle_model, oracle_optimizer)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            training_step(current, opt, micro_tokens, micro_targets, accumulation=2)
    metrics["optimizer_max_abs"] = {
        n: _close(p, dict(oracle_model.named_parameters())[n], "updated " + n)
        for n, p in model.named_parameters()
    }
    # Save/resume must reproduce the next step with identical data and state.
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        next_loss = training_step(model, optimizer, tokens, targets)
    next_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(saved_model)
    optimizer.load_state_dict(saved_optimizer)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        resumed_loss = training_step(model, optimizer, tokens, targets)
    torch.testing.assert_close(next_loss, resumed_loss, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, next_state[name], rtol=0, atol=0, msg=name)
    metrics["resume_exact"] = True
    return metrics


def run_training_checks(config):
    cases = []
    for variant in config.get("variants", ["standard", "sliced"]):
        for mode in config.get("modes", ["full", "block"]):
            item = {"variant": variant, "mode": mode}
            try:
                item.update(status="passed", metrics=_case(config, variant, mode))
            except Exception as exc:
                item.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc())
            cases.append(item)
            gc.collect()
            torch.cuda.empty_cache()
    return {"passed": sum(c["status"] == "passed" for c in cases),
            "failed": sum(c["status"] == "failed" for c in cases), "cases": cases}
