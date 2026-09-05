"""Complete-step BF16 measurements with ordinary source assembly and optimizers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import gc
import importlib
import io
import sys
import time
import traceback

import torch
from torch.nn import functional as F

from benchmarks.baseline import load_baseline
from benchmarks.bf16_competitors import load_all, model_ineligibility, Ineligible
from benchmarks.bf16_device import bf16_torch, compare, metadata, source_digest, operator_case
from benchmarks.bf16_model import Config, Model


_EXPECTED_GPU_CAPABILITIES = {"H100": (9, 0), "B200": (10, 0)}
_QUALIFICATION_RTOL = 0.05
_QUALIFICATION_ATOL = 0.05
_DEFAULT_DYNAMO_CACHE_SIZE_LIMIT = 64
_DEFAULT_DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT = 4096
_SAVE_RESUME_AUTO = "auto"


def _validate_runtime(config):
    """Validate the actual CUDA device before allocating benchmark tensors."""

    gpu = config.get("gpu")
    if gpu not in _EXPECTED_GPU_CAPABILITIES:
        expected = ", ".join(_EXPECTED_GPU_CAPABILITIES)
        raise ValueError(f"training benchmark requires gpu to be one of {expected}; got {gpu!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("training benchmark requires a CUDA device")

    actual = metadata()
    try:
        capability = tuple(int(value) for value in actual["capability"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"runtime metadata has no valid GPU capability: {actual!r}") from exc
    expected_capability = _EXPECTED_GPU_CAPABILITIES[gpu]
    actual_name = str(actual.get("gpu", ""))
    if capability != expected_capability or gpu not in actual_name:
        raise RuntimeError(
            f"GPU substitution: requested {gpu} {expected_capability}, "
            f"observed {actual_name!r} {capability}"
        )

    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(checker) or not checker():
        raise RuntimeError(f"runtime GPU does not advertise BF16 support: {actual!r}")
    return actual


def _bounded_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dynamo_limits(config):
    """Resolve a finite specialization budget for a multi-cell campaign."""

    nested = config.get("dynamo", {})
    if not isinstance(nested, Mapping):
        raise TypeError("dynamo configuration must be a mapping")
    cache = config.get(
        "dynamo_cache_size_limit",
        nested.get("cache_size_limit", _DEFAULT_DYNAMO_CACHE_SIZE_LIMIT),
    )
    accumulated = config.get(
        "dynamo_accumulated_cache_size_limit",
        nested.get(
            "accumulated_cache_size_limit",
            _DEFAULT_DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT,
        ),
    )
    cache = _bounded_int("dynamo cache_size_limit", cache)
    accumulated = _bounded_int(
        "dynamo accumulated_cache_size_limit", accumulated
    )
    if accumulated < cache:
        raise ValueError(
            "dynamo accumulated_cache_size_limit must be at least cache_size_limit"
        )
    return {
        "cache_size_limit": cache,
        "accumulated_cache_size_limit": accumulated,
        "fullgraph": True,
        "dynamic": False,
    }


def _configure_dynamo(config):
    """Apply and return the campaign's bounded Torch Dynamo specialization budget."""

    import torch._dynamo.config as dynamo_config

    limits = _dynamo_limits(config)
    previous = {}
    # PyTorch versions use either cache_size_limit or the newer recompile_limit
    # spelling. Prefer the canonical cache names; on versions exposing only the
    # newer names, use those instead. Some releases expose aliases backed by the
    # same setting, so setting both would make restoration ambiguous.
    fields = {
        (
            "cache_size_limit"
            if hasattr(dynamo_config, "cache_size_limit")
            else "recompile_limit"
        ): limits["cache_size_limit"],
        (
            "accumulated_cache_size_limit"
            if hasattr(dynamo_config, "accumulated_cache_size_limit")
            else "accumulated_recompile_limit"
        ): limits["accumulated_cache_size_limit"],
    }
    for field, value in fields.items():
        if hasattr(dynamo_config, field):
            previous[field] = getattr(dynamo_config, field)
            setattr(dynamo_config, field, value)
    if not previous:
        raise RuntimeError("Torch Dynamo exposes no specialization limit settings")
    return {"limits": limits, "previous": previous}


def _restore_dynamo(configuration):
    if not configuration:
        return
    import torch._dynamo.config as dynamo_config

    for field, value in configuration["previous"].items():
        setattr(dynamo_config, field, value)


def _mark_optimizer_implementation(optimizers, implementation):
    for optimizer in optimizers:
        # Optimizer instances are ordinary Python objects. Keeping the resolved
        # implementation beside the object avoids claiming a fused variant when
        # construction had to fall back.
        optimizer._benchmark_implementation = implementation
    return optimizers


def _optimizers(model, config):
    """Build the configured Muon+AdamW stack or a labelled AdamW fallback."""

    if config.get("optimizer_source"):
        source = str(config["optimizer_source"])
        if source not in sys.path:
            sys.path.insert(0, source)
        module = importlib.import_module("optimizer")
        optimizers = list(module.configure_optimizers(model, module.OptimizerConfig(
            muon_lr=.001, adamw_lr=.0003, muon_weight_decay=.1,
            adamw_weight_decay=0., cautious=True, beta1=.9, beta2=.95,
            muon_momentum=.95, verbose=False)))
        if not optimizers:
            raise RuntimeError("configured optimizer returned no optimizer instances")
        return _mark_optimizer_implementation(optimizers, "Muon+AdamW(configured)")

    options = {"lr": .0003, "betas": (.9, .95), "weight_decay": 0.}
    attempts = (
        ({"fused": True, "capturable": True}, "AdamW(fused=True,capturable=True)"),
        ({"fused": True}, "AdamW(fused=True)"),
        ({"foreach": True}, "AdamW(foreach=True)"),
        ({}, "AdamW(default)"),
    )
    last_error = None
    for extra, implementation in attempts:
        try:
            optimizer = torch.optim.AdamW(model.parameters(), **options, **extra)
        except (TypeError, RuntimeError) as exc:
            last_error = exc
            continue
        return _mark_optimizer_implementation([optimizer], implementation)
    raise RuntimeError("could not construct an AdamW optimizer fallback") from last_error


def _optimizer_label(optimizers):
    """Return the implementation label recorded by ``_optimizers``."""

    labels = {getattr(optimizer, "_benchmark_implementation", None)
              for optimizer in optimizers}
    labels.discard(None)
    if len(labels) == 1:
        return next(iter(labels))
    if labels:
        return "+".join(sorted(labels))
    return "optimizer implementation unavailable"


def _cpu_clone(value):
    """Clone arbitrary model/optimizer state to CPU without retaining CUDA storage."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return value


def _cpu_state(model):
    return {name: _cpu_clone(value) for name, value in model.state_dict().items()}


def _cpu_optimizer_state(optimizers):
    return [_cpu_clone(optimizer.state_dict()) for optimizer in optimizers]


def _combine_metrics(total, metrics):
    total["max_abs"] = max(total["max_abs"], metrics["max_abs"])
    total["relative_l2"] = max(total["relative_l2"], metrics["relative_l2"])


def _compare_state_tree(actual, expected, *, path="state", strict=False):
    """Compare a nested state dict, including optimizer groups and tensors."""

    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            raise AssertionError(f"{path} changed between arms")
        if strict:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            difference = (actual.float() - expected.float()).abs()
            return {
                "max_abs": float(difference.max()) if difference.numel() else 0.0,
                "relative_l2": float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(expected.float()).clamp_min(1e-20)
                ),
            }
        return compare(actual, expected)

    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise AssertionError(f"{path} changed between arms")
        if set(actual) != set(expected):
            raise AssertionError(
                f"{path} keys changed between arms: "
                f"actual={sorted(actual, key=str)!r}, expected={sorted(expected, key=str)!r}"
            )
        total = {"max_abs": 0.0, "relative_l2": 0.0}
        for key in expected:
            _combine_metrics(
                total,
                _compare_state_tree(actual[key], expected[key],
                                    path=f"{path}[{key!r}]", strict=strict),
            )
        return total

    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"{path} sequence changed between arms")
        total = {"max_abs": 0.0, "relative_l2": 0.0}
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _combine_metrics(
                total,
                _compare_state_tree(actual_item, expected_item,
                                    path=f"{path}[{index}]", strict=strict),
            )
        return total

    if actual != expected:
        raise AssertionError(f"{path} changed between arms: {actual!r} != {expected!r}")
    return {"max_abs": 0.0, "relative_l2": 0.0}


def _memory_record(baseline, peak, current, *, model_incremental=None,
                   model_optimizer_incremental=None):
    """Describe allocated memory relative to the arm's pre-construction baseline."""

    record = {
        "baseline_allocated_bytes": int(baseline),
        "persistent_incremental_allocated_bytes": max(0, int(current) - int(baseline)),
        "current_allocated_bytes_global_total": int(current),
        "peak_allocated_bytes_incremental": max(0, int(peak) - int(baseline)),
        "peak_allocated_bytes_global_total": int(peak),
    }
    if model_incremental is not None:
        record["model_incremental_allocated_bytes"] = max(0, int(model_incremental))
    if model_optimizer_incremental is not None:
        record["model_optimizer_incremental_allocated_bytes"] = int(
            max(0, int(model_optimizer_incremental))
        )
    return record


def _safe_memory_record(baseline, *, model_incremental=None,
                        model_optimizer_incremental=None):
    if baseline is None:
        return None
    try:
        return _memory_record(
            baseline,
            torch.cuda.max_memory_allocated(),
            torch.cuda.memory_allocated(),
            model_incremental=model_incremental,
            model_optimizer_incremental=model_optimizer_incremental,
        )
    except Exception:
        return None


def _release_arm_references(*objects):
    """Release a failed arm's closures, model, optimizer, and compiler references."""

    del objects
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        # The original exception is the useful qualification/timing failure.
        pass


def _discard_qualified_arm(arm):
    references = [arm.pop(key, None) for key in ("model", "optimizers", "step")]
    _release_arm_references(*references)


def _case_backend_items(case, backends):
    """Select exactly the backend names requested by one case."""

    requested = case.get("backends")
    if requested is None:
        return list(backends.items()), []
    if isinstance(requested, str):
        raise TypeError("case backends must be a sequence of backend names")
    requested = list(requested)
    requested_set = set(requested)
    selected = [(name, op) for name, op in backends.items() if name in requested_set]
    missing = []
    for name in requested:
        if name not in backends and name not in missing:
            missing.append(name)
    return selected, missing


def _save_resume_smoke(model, optimizers):
    """Round-trip model and optimizer state through the normal torch serializer."""

    model_state = _cpu_state(model)
    optimizer_state = _cpu_optimizer_state(optimizers)
    stream = io.BytesIO()
    torch.save({"model": model_state, "optimizers": optimizer_state}, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu")
    model.load_state_dict(restored["model"])
    restored_optimizers = restored["optimizers"]
    if len(restored_optimizers) != len(optimizers):
        raise AssertionError("save/resume changed the optimizer count")
    for optimizer, state in zip(optimizers, restored_optimizers):
        optimizer.load_state_dict(state)

    return {
        "status": "passed",
        "model_state": _compare_state_tree(
            _cpu_state(model), model_state, path="model", strict=True
        ),
        "optimizer_state": _compare_state_tree(
            _cpu_optimizer_state(optimizers), optimizer_state,
            path="optimizers", strict=True
        ),
    }


def _save_resume_enabled(case, config, model_config):
    requested = case.get("save_resume_smoke", config.get("save_resume_smoke", _SAVE_RESUME_AUTO))
    if requested is True:
        return True
    if requested is False:
        return False
    if requested != _SAVE_RESUME_AUTO:
        raise ValueError("save_resume_smoke must be true, false, or 'auto'")
    # The smoke exercises serialization without adding a second resident model.
    # Keep it automatic for the tiny structural configurations used in local
    # checks, while leaving the production geometry's timed path unchanged.
    return (
        model_config.layers <= 4
        and model_config.width <= 256
        and model_config.ffn <= 1024
        and model_config.vocab <= 1024
        and model_config.context <= 256
    )


def _failure_record(phase, exc, *, samples_ms=(), wall_ms=(), compile_warmup_s=None,
                    optimizer=None, memory=None, qualification=None, classification=None):
    error_traceback = traceback.format_exc() if isinstance(exc, Exception) else ""
    if error_traceback == "NoneType: None\n":
        error_traceback = ""
    if classification is None:
        classification = getattr(exc, "classification", None)
    if classification not in {"incorrect", "ineligible", "unresolved"}:
        classification = (
            "ineligible" if type(exc).__name__ == "Ineligible"
            else "incorrect" if (
                isinstance(exc, AssertionError)
                and phase in {"qualification", "first_optimizer_update"}
            )
            else "unresolved"
        )
    record = {
        "status": "failed",
        "phase": phase,
        "classification": classification,
        "error": f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc),
        "traceback": error_traceback,
        "samples_ms": list(samples_ms),
        "wall_ms": list(wall_ms),
    }
    if compile_warmup_s is not None:
        record["compile_warmup_s"] = compile_warmup_s
    if optimizer is not None:
        record["optimizer"] = optimizer
    if memory is not None:
        record["memory"] = memory
        record["peak_allocated_bytes"] = memory["peak_allocated_bytes_incremental"]
        record["peak_allocated_bytes_incremental"] = memory["peak_allocated_bytes_incremental"]
        record["peak_allocated_bytes_global_total"] = memory[
            "peak_allocated_bytes_global_total"
        ]
    if qualification is not None:
        record["qualification"] = qualification
    return record


def _public_arm(arm, status):
    record = {
        "status": status,
        "samples_ms": list(arm.get("samples_ms", ())),
        "round_ids": list(range(len(arm.get("samples_ms", ())))),
        "wall_ms": list(arm.get("wall_ms", ())),
        "compile_warmup_s": arm["compile_warmup_s"],
        "qualification": arm["qualification"],
        "optimizer": arm["optimizer"],
    }
    if arm.get("memory") is not None:
        memory = arm["memory"]
        record.update({
            "memory": memory,
            # Keep the historical field, but make it the per-arm incremental
            # value. The explicit global-total name prevents ambiguity.
            "peak_allocated_bytes": memory["peak_allocated_bytes_incremental"],
            "peak_allocated_bytes_incremental": memory["peak_allocated_bytes_incremental"],
            "peak_allocated_bytes_global_total": memory[
                "peak_allocated_bytes_global_total"
            ],
        })
    if "save_resume" in arm:
        record["save_resume"] = arm["save_resume"]
    return record


def _arm_rows(arms, failures, status):
    return {
        **failures,
        **{name: _public_arm(arm, status) for name, arm in arms.items()},
    }


def training_case(case, backends, config, seed, checkpoint, runtime=None):
    model_config = Config(**case["model"])
    runtime = _validate_runtime(config) if runtime is None else runtime
    del runtime  # Validation is intentionally complete before CUDA allocation.
    dynamo = _dynamo_limits(config)

    selected_backends, missing_backends = _case_backend_items(case, backends)
    requested_backends = (
        [name for name, _ in selected_backends]
        if case.get("backends") is None else list(case["backends"])
    )
    failures = {
        name: _failure_record(
            "case_filtering",
            RuntimeError(f"backend {name!r} is not available for this case"),
        )
        for name in missing_backends
    }
    record = {
        "case": case,
        "seed": seed,
        "requested_backends": requested_backends,
        "grad_clip": 1.0, "loss_dtype": "bfloat16",
        "model": asdict(model_config),
        "measurement": (
            "complete optimizer step; input H2D copies, source assembly, forward, loss, "
            "backward, accumulation, Muon+AdamW included"
        ),
        "qualification_tolerances": {
            "rtol": _QUALIFICATION_RTOL,
            "atol": _QUALIFICATION_ATOL,
        },
        "dynamo": dynamo,
        "memory_measurement": (
            "peak_allocated_bytes is per-arm incremental allocation from the pre-arm "
            "global baseline and includes persistent model/optimizer allocation; "
            "peak_allocated_bytes_global_total is the global allocator total"
        ),
        "arms": failures,
    }
    checkpoint(record)
    if not selected_backends:
        return record

    # Zero-initialized model queries cannot expose every routing derivative
    # defect. Qualify nonzero queries at the largest read before model setup.
    gate_ops = {}
    for name, op in selected_backends:
        reason = model_ineligibility(name, model_config)
        if reason:
            failures[name] = _failure_record("operator_qualification", Ineligible(reason))
        else:
            gate_ops[name] = (torch.compile(op, fullgraph=True, dynamic=False)
                              if name == "torch_compile" else op)
    sources = (2 * model_config.layers + 1 if model_config.mode == "full"
               else min(2 * model_config.layers, model_config.block_count) + 1)
    gate_case = {"shape": [sources, case.get("batch", 4) * case.get("sequence", model_config.context),
                           model_config.width, model_config.rank], "layout": "list",
                 "backends": list(gate_ops), "query_scale": .05}
    gate = operator_case(gate_case, gate_ops, seed=seed, warmups=1,
                         rounds=config.get("rounds", 120), replays=8) if gate_ops else None
    record["operator_qualification"] = {"replays": 8, "result": gate}
    for name, result in (gate or {}).get("arms", {}).items():
        if result["status"] != "passed":
            classification = "incorrect" if result.get("error", "").startswith("AssertionError:") else "unresolved"
            failures[name] = _failure_record("operator_qualification", result.get("error", "operator failed"),
                                            classification=classification)
    selected_backends = [(name, op) for name, op in selected_backends if name not in failures]
    record["arms"] = dict(failures)
    checkpoint(record)
    if not selected_backends:
        return record
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(seed)
    initial_model = Model(model_config, bf16_torch)
    initial = _cpu_state(initial_model)
    del initial_model
    batch = case.get("batch", 4)
    sequence = case.get("sequence", model_config.context)
    accumulation = case.get("accumulation", 4)
    generator = torch.Generator().manual_seed(seed + 17)
    host_tokens = torch.randint(
        model_config.vocab, (8, accumulation, batch, sequence + 1), generator=generator
    ).pin_memory()
    import hashlib
    record["input_sha256"] = hashlib.sha256(host_tokens.numpy().tobytes()).hexdigest()
    record["round_order"] = []
    tokens = torch.empty((accumulation, batch, sequence), device="cuda", dtype=torch.long)
    targets = torch.empty_like(tokens)
    arms = {}

    # Exactly one CPU gradient dictionary and one CPU post-update state form the
    # qualification reference. Successful arms are compared against those
    # snapshots and do not retain their own CPU copies.
    baseline_loss = None
    baseline_gradients = None
    baseline_model_update = None
    baseline_optimizer_update = None

    smoke_enabled = _save_resume_enabled(case, config, model_config)
    for name, op in selected_backends:
        started = time.monotonic()
        model = optimizers = compiled = compiled_loss = step = None
        gradients = None
        qualification = None
        arm_baseline = None
        model_incremental = None
        model_optimizer_incremental = None
        memory = None
        optimizer_label = None
        phase = "allocation"
        committed = False
        try:
            reason = model_ineligibility(name, model_config)
            if reason:
                raise Ineligible(reason)
            print(f"training arm {name} mode={model_config.mode} rank={model_config.rank} seed={seed}", flush=True)
            # Reset after the previous arm remains resident. This makes the
            # subsequent peak a delta above the resident-arm global baseline.
            torch.cuda.synchronize()
            arm_baseline = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()

            model = Model(model_config, op).cuda().train()
            model.load_state_dict(initial)
            model_incremental = int(torch.cuda.memory_allocated()) - arm_baseline
            phase = "optimizer"
            optimizers = _optimizers(model, config)
            optimizer_label = _optimizer_label(optimizers)
            model_optimizer_incremental = int(torch.cuda.memory_allocated()) - arm_baseline
            phase = "compile"
            compiled = torch.compile(
                model,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )

            def loss_forward(x, y, compiled=compiled):
                logits = compiled(x)
                return F.cross_entropy(logits.flatten(0, 1), y.flatten())

            compiled_loss = torch.compile(
                loss_forward,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )

            def step(input_index, *, update=True, optimizers=optimizers, loss_fn=compiled_loss, model=model):
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    for optimizer in optimizers:
                        optimizer.step()
                return torch.stack(losses).sum()

            # First compare the complete pre-update loss/gradient result.
            phase = "qualification"
            loss = step(0, update=False)
            loss_cpu = loss.detach().cpu()
            gradients = {
                parameter_name: parameter.grad.detach().cpu().clone()
                for parameter_name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            if not gradients:
                raise AssertionError("no parameter gradients were produced")
            qualification = {
                "loss": float(loss_cpu),
                "gradient_count": len(gradients),
            }
            if baseline_loss is not None:
                loss_metrics = compare(
                    loss_cpu,
                    loss_cpu.new_tensor(baseline_loss),
                )
                if set(gradients) != set(baseline_gradients):
                    raise AssertionError("gradient parameter sets differ")
                gradient_metrics = [
                    compare(gradient, baseline_gradients[parameter_name])
                    for parameter_name, gradient in gradients.items()
                ]
                qualification["loss_comparison"] = loss_metrics
                qualification["max_gradient_abs"] = max(
                    metric["max_abs"] for metric in gradient_metrics
                )

            # Take the first optimizer update solely for a complete-state gate.
            # Restore the pre-update model/optimizer state before warmup so the
            # timed training geometry retains its original update count.
            phase = "first_optimizer_update"
            pre_update_optimizer = _cpu_optimizer_state(optimizers)
            step(0, update=True)
            torch.cuda.synchronize()
            model_update = _cpu_state(model)
            optimizer_update = _cpu_optimizer_state(optimizers)
            if baseline_model_update is None:
                first_update = {
                    "status": "baseline",
                    "model_state": {"max_abs": 0.0, "relative_l2": 0.0},
                    "optimizer_state": {"max_abs": 0.0, "relative_l2": 0.0},
                }
            else:
                first_update = {
                    "status": "matched",
                    "model_state": _compare_state_tree(
                        model_update,
                        baseline_model_update,
                        path="first_update.model_state",
                    ),
                    "optimizer_state": _compare_state_tree(
                        optimizer_update,
                        baseline_optimizer_update,
                        path="first_update.optimizer_state",
                    ),
                }
            qualification["first_update"] = first_update

            phase = "save_resume"
            if smoke_enabled:
                save_resume = _save_resume_smoke(model, optimizers)
            else:
                save_resume = {
                    "status": "skipped",
                    "reason": "auto-disabled for production-sized configuration",
                }

            model.load_state_dict(initial)
            for optimizer, state in zip(optimizers, pre_update_optimizer):
                optimizer.load_state_dict(state)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            phase = "warmup"
            for iteration in range(config.get("warmups", 10)):
                step(iteration)
            torch.cuda.synchronize()
            memory = _safe_memory_record(
                arm_baseline,
                model_incremental=model_incremental,
                model_optimizer_incremental=model_optimizer_incremental,
            )
            arms[name] = {
                "model": model,
                "optimizers": optimizers,
                "step": step,
                "qualification": qualification,
                "optimizer": optimizer_label,
                "compile_warmup_s": time.monotonic() - started,
                "memory": memory,
                "save_resume": save_resume,
                "samples_ms": [],
                "wall_ms": [],
            }
            if baseline_model_update is None:
                baseline_loss = qualification["loss"]
                baseline_gradients = gradients
                baseline_model_update = model_update
                baseline_optimizer_update = optimizer_update
            committed = True
        except Exception as exc:
            failures[name] = _failure_record(
                phase,
                exc,
                compile_warmup_s=time.monotonic() - started,
                optimizer=optimizer_label,
                memory=(memory or _safe_memory_record(
                    arm_baseline,
                    model_incremental=model_incremental,
                    model_optimizer_incremental=model_optimizer_incremental,
                )),
                qualification=qualification,
            )
        finally:
            if not committed:
                _release_arm_references(model, optimizers, compiled, compiled_loss, step)
            model = optimizers = compiled = compiled_loss = step = None
            gradients = None
            model_update = optimizer_update = pre_update_optimizer = None
        record["arms"] = _arm_rows(arms, failures, "qualified")
        checkpoint(record)

    del initial
    baseline_loss = baseline_gradients = baseline_model_update = baseline_optimizer_update = None
    gc.collect()

    if not arms:
        record["arms"] = failures
        gc.collect()
        torch.cuda.empty_cache()
        return record

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for iteration in range(config.get("rounds", 120)):
        names = list(arms)
        if not names:
            break
        # Rotate the starting backend and reverse every other round. Each arm
        # sees identical minibatches and optimizer-update counts.
        pivot = iteration % len(names)
        order = names[pivot:] + names[:pivot]
        if iteration % 2:
            order.reverse()
        record["round_order"].append({"round": iteration, "input": iteration % 8, "backends": order})
        for name in order:
            arm = arms.get(name)
            if arm is None:
                continue
            started = time.perf_counter()
            try:
                begin.record()
                loss = arm["step"](iteration)
                end.record()
                end.synchronize()
                if not torch.isfinite(loss).item():
                    raise RuntimeError("nonfinite training loss")
                arm["samples_ms"].append(begin.elapsed_time(end))
                arm["wall_ms"].append((time.perf_counter() - started) * 1000)
            except Exception as exc:
                failed_arm = arms.pop(name)
                failures[name] = _failure_record(
                    "timing",
                    exc,
                    samples_ms=failed_arm["samples_ms"],
                    wall_ms=failed_arm["wall_ms"],
                    compile_warmup_s=failed_arm["compile_warmup_s"],
                    optimizer=failed_arm["optimizer"],
                    memory=failed_arm.get("memory"),
                    qualification=failed_arm["qualification"],
                )
                _discard_qualified_arm(failed_arm)
                record["arms"] = _arm_rows(arms, failures, "running")
                checkpoint(record)
        if iteration % 10 == 9:
            record["arms"] = _arm_rows(arms, failures, "running")
            checkpoint(record)

    record["arms"] = _arm_rows(arms, failures, "passed")
    for arm in list(arms.values()):
        _discard_qualified_arm(arm)
    del arms
    gc.collect()
    torch.cuda.empty_cache()
    return record


def _case_failure(case, seed, exc):
    return {
        "status": "failed",
        "case": case,
        "seed": seed,
        "requested_backends": case.get("backends"),
        "arms": {"__case__": _failure_record("case", exc)},
    }


def run_training(config, checkpoint):
    runtime = _validate_runtime(config)
    dynamo = _configure_dynamo(config)
    try:
        backends, identities = {}, {}
        for name, root in config["sources"].items():
            loaded = load_baseline(root)
            backends[name], identities[name] = loaded.attnres, loaded.metadata
        external, external_ids, import_failures = load_all(config.get("competitors", {}))
        backends.update(external)
        identities.update(external_ids)
        if config.get("torch_baseline", True):
            import hashlib
            from pathlib import Path
            fixture_files = (Path(__file__).with_name("bf16_device.py"),
                             Path(__file__).parents[1] / "validation/oracle.py")
            backends["torch_compile"] = bf16_torch
            identities["torch_compile"] = {
                "implementation": "bf16_torch oracle; outer model torch.compile(fullgraph=True,dynamic=False)",
                "source": "benchmarks.bf16_device.bf16_torch",
                "sha256": hashlib.sha256(b"".join(path.read_bytes() for path in fixture_files)).hexdigest(),
            }
        from benchmarks.bf16_primary import fixture_digest
        from pathlib import Path
        identities["training_fixture"] = {"sha256": fixture_digest(Path(__file__).parents[1])}
        if config.get("optimizer_source"):
            optimizer_identity = source_digest(config["optimizer_source"])
            optimizer_identity["implementation"] = "Muon+AdamW(configured)"
            identities["optimizer"] = optimizer_identity
        else:
            identities["optimizer"] = {
                "implementation": "resolved per-arm torch.optim.AdamW",
                "fallback_labels": [
                    "AdamW(fused=True,capturable=True)",
                    "AdamW(fused=True)",
                    "AdamW(foreach=True)",
                    "AdamW(default)",
                ],
            }
        if config.get("expected_identities"):
            actual_ids = {name: row.get("content_hash", row.get("sha256")) for name, row in identities.items()}
            if actual_ids != config["expected_identities"]:
                raise RuntimeError("training inputs differ from frozen primary source identities")
        report = {
            "kind": "training",
            "status": "running",
            "config": config,
            "runtime": runtime,
            "dynamo": dynamo["limits"],
            "identities": identities,
            "import_failures": import_failures,
            "results": [],
        }
        checkpoint(report)
        for case in config["cases"]:
            for seed in config["seeds"]:
                report["in_progress"] = {"case": case, "seed": seed}

                def partial(record):
                    report["in_progress"] = record
                    checkpoint(report)

                try:
                    result = training_case(
                        case, backends, config, seed, partial, runtime=runtime
                    )
                except Exception as exc:
                    result = _case_failure(case, seed, exc)
                report["results"].append(result)
                report.pop("in_progress", None)
                checkpoint(report)
        report["status"] = "complete"
        checkpoint(report)
        return report
    finally:
        _restore_dynamo(dynamo)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def save(report):
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temporary.replace(args.output)
    run_training(json.loads(args.config.read_text()), save)
