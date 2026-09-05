"""Complete-step BF16 measurements with ordinary source assembly and optimizers."""
from __future__ import annotations

from dataclasses import asdict
import gc
import importlib
import sys
import time
import traceback

import torch
from torch.nn import functional as F

from benchmarks.baseline import load_baseline
from benchmarks.bf16_competitors import load_all
from benchmarks.bf16_device import bf16_torch, compare, metadata, source_digest
from benchmarks.bf16_model import Config, Model


def _optimizers(model, config):
    if config.get("optimizer_source"):
        sys.path.insert(0, config["optimizer_source"])
        module = importlib.import_module("optimizer")
        opts = module.configure_optimizers(model, module.OptimizerConfig(
            muon_lr=.001, adamw_lr=.0003, muon_weight_decay=.1,
            adamw_weight_decay=0., cautious=True, beta1=.9, beta2=.95,
            muon_momentum=.95, verbose=False))
        return opts
    return [torch.optim.AdamW(model.parameters(), lr=.0003, betas=(.9, .95),
                             weight_decay=0., fused=True, capturable=True)]


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def training_case(case, backends, config, seed, checkpoint):
    model_config = Config(**case["model"])
    torch.manual_seed(seed)
    initial_model = Model(model_config, bf16_torch)
    initial = _cpu_state(initial_model)
    del initial_model
    batch = case.get("batch", 4)
    sequence = case.get("sequence", model_config.context)
    accumulation = case.get("accumulation", 4)
    generator = torch.Generator().manual_seed(seed + 17)
    host_tokens = torch.randint(model_config.vocab, (8, accumulation, batch, sequence + 1),
                               generator=generator).pin_memory()
    tokens = torch.empty((accumulation, batch, sequence), device="cuda", dtype=torch.long)
    targets = torch.empty_like(tokens)
    arms, failures = {}, {}
    record = {"case": case, "seed": seed, "model": asdict(model_config),
              "measurement": "complete optimizer step; input H2D copies, source assembly, "
                             "forward, loss, backward, accumulation, Muon+AdamW included",
              "arms": failures}

    for name, op in backends.items():
        if case.get("backends") is not None and name not in case["backends"]:
            continue
        started = time.monotonic()
        try:
            model = Model(model_config, op).cuda().train()
            model.load_state_dict(initial)
            optimizers = _optimizers(model, config)
            compiled = torch.compile(model, fullgraph=True, dynamic=False,
                                     options={"triton.cudagraphs": False})

            def loss_forward(x, y, compiled=compiled):
                logits = compiled(x)
                return F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
            compiled_loss = torch.compile(loss_forward, fullgraph=True, dynamic=False,
                                          options={"triton.cudagraphs": False})

            def step(input_index, *, update=True, optimizers=optimizers, loss_fn=compiled_loss):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                source = host_tokens[input_index % len(host_tokens)]
                tokens.copy_(source[..., :-1], non_blocking=True)
                targets.copy_(source[..., 1:], non_blocking=True)
                losses = []
                for micro in range(accumulation):
                    loss = loss_fn(tokens[micro], targets[micro]) / accumulation
                    loss.backward()
                    losses.append(loss.detach())
                if update:
                    for optimizer in optimizers:
                        optimizer.step()
                return torch.stack(losses).sum()

            # First-step comparisons retain every parameter gradient. Copies
            # are outside timing and freed before the next training arm runs.
            loss = step(0, update=False)
            gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                         if p.grad is not None}
            qualification = {"loss": float(loss), "gradient_count": len(gradients)}
            if arms:
                other = next(iter(arms.values()))
                compare(loss, loss.new_tensor(other["qualification"]["loss"]))
                if gradients.keys() != other["first_gradients"].keys():
                    raise AssertionError("gradient parameter sets differ")
                metrics = [compare(v, other["first_gradients"][n]) for n, v in gradients.items()]
                qualification["max_gradient_abs"] = max(x["max_abs"] for x in metrics)
            torch.cuda.reset_peak_memory_stats()
            for iteration in range(config.get("warmups", 10)):
                step(iteration)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            arms[name] = {"model": model, "optimizers": optimizers, "step": step,
                          "first_gradients": gradients, "qualification": qualification,
                          "compile_warmup_s": time.monotonic() - started,
                          "peak_allocated_bytes": peak, "samples_ms": [], "wall_ms": []}
        except Exception as exc:
            failures[name] = {"status": "failed", "phase": "training_qualification",
                              "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()}
        record["arms"] = {**failures, **{n: {"status": "qualified",
            "compile_warmup_s": a["compile_warmup_s"], "qualification": a["qualification"]}
            for n, a in arms.items()}}
        checkpoint(record)

    del initial
    for arm in arms.values():
        arm.pop("first_gradients")
    gc.collect()
    names = list(arms)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for iteration in range(config.get("rounds", 120)):
        # Rotate the starting backend and reverse every other round. Each arm
        # sees identical minibatches and optimizer-update counts.
        pivot = iteration % max(1, len(names))
        order = names[pivot:] + names[:pivot]
        if iteration % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            begin.record()
            arms[name]["step"](iteration)
            end.record()
            end.synchronize()
            arms[name]["samples_ms"].append(begin.elapsed_time(end))
            arms[name]["wall_ms"].append((time.perf_counter() - started) * 1000)
        if iteration % 10 == 9:
            record["arms"] = {**failures, **{n: {"status": "running", "samples_ms": a["samples_ms"],
                                               "wall_ms": a["wall_ms"]} for n, a in arms.items()}}
            checkpoint(record)
    record["arms"] = {**failures, **{n: {"status": "passed", **{k: a[k] for k in (
        "samples_ms", "wall_ms", "compile_warmup_s", "peak_allocated_bytes", "qualification")}}
        for n, a in arms.items()}}
    del arms
    gc.collect()
    torch.cuda.empty_cache()
    return record


def run_training(config, checkpoint):
    backends, identities = {}, {}
    for name, root in config["sources"].items():
        loaded = load_baseline(root)
        backends[name], identities[name] = loaded.attnres, loaded.metadata
    external, external_ids, import_failures = load_all(config.get("competitors", {}))
    backends.update(external)
    identities.update(external_ids)
    if config.get("torch_baseline", True):
        backends["torch_compile"] = bf16_torch
    if config.get("optimizer_source"):
        identities["optimizer"] = source_digest(config["optimizer_source"])
    report = {"kind": "training", "status": "running", "config": config,
              "runtime": metadata(), "identities": identities,
              "import_failures": import_failures, "results": []}
    checkpoint(report)
    for case in config["cases"]:
        for seed in config["seeds"]:
            report["in_progress"] = {"case": case, "seed": seed}
            def partial(record):
                report["in_progress"] = record
                checkpoint(report)
            result = training_case(case, backends, config, seed, partial)
            report["results"].append(result)
            report.pop("in_progress", None)
            checkpoint(report)
    report["status"] = "complete"
    checkpoint(report)
    return report
