"""Small, resumable CUDA BF16 qualification for the public ``attnres`` API.

This module is a correctness runner, not a timing or launch wrapper.  It uses
the repository's frozen source-list helper for the operator matrix and keeps
the result JSON-safe after every completed case.
"""

from __future__ import annotations

import argparse
import copy
import gc
import io
import json
import math
import os
import platform
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

BF16_TOLERANCE = {"rtol": 0.05, "atol": 0.05}
GRAPH_REPLAYS = 8
DEFAULT_SEED = 20260827

DEFAULT_OPERATOR_CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "block_list_packed_D513",
        "shape": [5, 7, 513, 257],
        "mode": "block",
        "shared": True,
        "graph": True,
    },
    {
        "name": "full_list_packed_D3072",
        "shape": [5, 2, 3072, 257],
        "mode": "full",
        "shared": True,
        "graph": False,
    },
    {
        "name": "full_list_packed_D4096",
        "shape": [3, 2, 4096, 4096],
        "mode": "full",
        "shared": True,
        "graph": False,
    },
)

DEFAULT_TRAINING: dict[str, Any] = {
    "enabled": True,
    "seed": DEFAULT_SEED,
    "layers": 2,
    "width": 32,
    "heads": 4,
    "ffn": 64,
    "vocab": 97,
    "context": 16,
    "rank": 16,
    "mode": "block",
    "block_count": 2,
    "batch": 2,
    "accumulation": 2,
    "activation_checkpointing": True,
    "lr": 1e-3,
}

DEFAULT_CONFIG: dict[str, Any] = {
    "device": "cuda",
    "seed": DEFAULT_SEED,
    "operator_cases": [copy.deepcopy(case) for case in DEFAULT_OPERATOR_CASES],
    "training": copy.deepcopy(DEFAULT_TRAINING),
}


class _QualificationSkip(RuntimeError):
    """A target-device preflight condition that is not a code failure."""


def _jsonable(value: Any) -> Any:
    """Convert report/config values to strict-JSON-compatible objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(_jsonable(value), allow_nan=False))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint(checkpoint: Any, report: Mapping[str, Any]) -> None:
    """Persist a report through either a callback or a JSON path."""

    if checkpoint is None:
        return
    payload = _json_copy(report)
    if callable(checkpoint):
        checkpoint(payload)
    else:
        _atomic_json(Path(checkpoint), payload)


def _merge_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping or None")
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in config.items():
        if key == "training" and isinstance(value, Mapping):
            merged["training"].update(copy.deepcopy(dict(value)))
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _cuda_device(config: Mapping[str, Any]) -> torch.device:
    if not torch.cuda.is_available():
        raise _QualificationSkip("CUDA is not available")
    device = torch.device(config.get("device", "cuda"))
    if device.type != "cuda":
        raise _QualificationSkip(f"GPU qualification requires CUDA, got {device}")
    if device.index is not None:
        torch.cuda.set_device(device)
    return torch.device("cuda", torch.cuda.current_device())


def _runtime(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "device": str(device),
        "gpu": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "memory_bytes": int(properties.total_memory),
        "sm_count": int(properties.multi_processor_count),
        "dtype": "torch.bfloat16",
    }


def _shape(case: Mapping[str, Any]) -> tuple[int, int, int, int]:
    raw = case.get("shape")
    if raw is None:
        raw = [case[name] for name in ("sources", "batch", "width", "rank")]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("operator case shape must be [sources, batch, width, rank]")
    result = tuple(int(value) for value in raw)
    sources, _batch, width, rank = result
    if min(result) < 1 or sources > 129 or width > 8192 or rank > width:
        raise ValueError(f"unsupported operator shape {result}")
    return result


def _operator_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Run one frozen source-list case, including its packed control."""

    from validation.source_checks import source_case

    shape = _shape(case)
    mode = str(case.get("mode", "full")).lower()
    if mode not in {"full", "block"}:
        raise ValueError("operator case mode must be 'full' or 'block'")
    graph = bool(case.get("graph", False))
    shared = bool(case.get("shared", True))
    metrics = source_case(
        shape,
        mode,
        torch.bfloat16,
        graph=graph,
        shared=shared,
        device="cuda",
    )
    replay_count = len(metrics.get("graph_changed_input", ())) if graph else 0
    if graph and replay_count != GRAPH_REPLAYS:
        raise AssertionError(
            f"expected {GRAPH_REPLAYS} changed-input graph replays, got {replay_count}"
        )
    return {
        "shape": list(shape),
        "mode": mode,
        "layouts": ["ordered_source_list", "packed_control"],
        "duplicate_sources": True,
        "shared_views": shared,
        "noncontiguous_values_query_upstream": True,
        "compiled_fullgraph": graph,
        "changed_input_graph_replays": replay_count,
        "metrics": metrics,
    }


def _oracle_operator(values, query, *, eps: float, scale: float):
    from validation.oracle import oracle

    if isinstance(values, (list, tuple)):
        values = torch.stack(tuple(values), dim=0)
    return oracle(values, query, eps=eps, scale=scale)


def _compare(actual: torch.Tensor, expected: torch.Tensor, name: str) -> float:
    from validation.gpu_checks import _compare as frozen_compare

    if actual.dtype is not torch.bfloat16 or expected.dtype is not torch.bfloat16:
        raise AssertionError(f"{name} must use BF16 storage")
    return float(frozen_compare(actual, expected, torch.bfloat16)["max_abs"])


def _clone_tree(value: Any, *, cpu: bool = False) -> Any:
    if isinstance(value, torch.Tensor):
        result = value.detach().clone()
        return result.cpu() if cpu else result
    if isinstance(value, Mapping):
        return {key: _clone_tree(item, cpu=cpu) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item, cpu=cpu) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item, cpu=cpu) for item in value)
    return copy.deepcopy(value)


def _tree_max_abs(actual: Any, expected: Any) -> float:
    if isinstance(actual, Mapping):
        if set(actual) != set(expected):
            raise AssertionError("checkpoint mappings have different keys")
        return max((_tree_max_abs(actual[key], expected[key]) for key in actual), default=0.0)
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError("checkpoint sequences have different lengths")
        return max((_tree_max_abs(a, e) for a, e in zip(actual, expected)), default=0.0)
    if isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor) or actual.shape != expected.shape:
            raise AssertionError("checkpoint tensor shapes differ")
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError("checkpoint contains a nonfinite tensor")
        return float((actual.float() - expected.float()).abs().max().item())
    if actual != expected:
        raise AssertionError(f"checkpoint values differ: {actual!r} != {expected!r}")
    return 0.0


def _tree_exact(actual: Any, expected: Any) -> None:
    if isinstance(actual, Mapping):
        if set(actual) != set(expected):
            raise AssertionError("checkpoint mappings have different keys")
        for key in actual:
            _tree_exact(actual[key], expected[key])
        return
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError("checkpoint sequences have different lengths")
        for a, e in zip(actual, expected):
            _tree_exact(a, e)
        return
    if isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor):
            raise TypeError("checkpoint tensor types differ")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if actual != expected:
        raise AssertionError(f"checkpoint values differ: {actual!r} != {expected!r}")


def _make_training_config(spec: Mapping[str, Any]):
    from benchmarks.bf16_model import Config

    fields = {
        key: value
        for key, value in spec.items()
        if key
        in {
            "layers",
            "width",
            "heads",
            "ffn",
            "vocab",
            "context",
            "rank",
            "mode",
            "block_count",
            "activation_checkpointing",
            "rope_theta",
            "norm_pos",
            "qk_norm",
            "attnres_eps",
            "attnres_scale",
        }
    }
    if "sequence" in spec and "context" not in fields:
        fields["context"] = spec["sequence"]
    return Config(**fields)


def _optimizer(model: torch.nn.Module, spec: Mapping[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(spec.get("lr", 1e-3)),
        betas=(float(spec.get("beta1", 0.9)), float(spec.get("beta2", 0.95))),
        weight_decay=float(spec.get("weight_decay", 0.0)),
        fused=bool(spec.get("fused", True)),
        capturable=bool(spec.get("capturable", True)),
    )


def _training_batch(
    config: Any,
    spec: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(spec.get("batch", 2))
    accumulation = int(spec.get("accumulation", 2))
    if batch < 1 or accumulation < 1:
        raise ValueError("training batch and accumulation must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    host = torch.randint(
        config.vocab,
        (accumulation, batch, config.context + 1),
        generator=generator,
        dtype=torch.int64,
    )
    return host[..., :-1].to(device), host[..., 1:].to(device)


def _backward_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    from torch.nn import functional as F

    optimizer.zero_grad(set_to_none=False)
    losses = []
    last_logits = None
    accumulation = tokens.shape[0]
    for micro_tokens, micro_targets in zip(tokens.unbind(0), targets.unbind(0)):
        logits = model(micro_tokens)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), micro_targets.reshape(-1)
        )
        (loss / accumulation).backward()
        losses.append(loss.detach())
        last_logits = logits.detach()
    if last_logits is None:
        raise RuntimeError("training step had no microbatches")
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return torch.stack(losses).mean().detach(), last_logits, gradients



def _analytic_zero_query(device):
    from attnres import attnres
    sources, rows, width, rank = 7, 3, 513, 96
    torch.manual_seed(20260905)
    values = torch.randn(sources, rows, width, dtype=torch.bfloat16,
                         device=device, requires_grad=True)
    query = torch.zeros(rank, dtype=torch.bfloat16, device=device, requires_grad=True)
    upstream = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    output = attnres(values, query, scale=.7)
    dv, dq = torch.autograd.grad(output, (values, query), upstream)
    # At q=0 every source has weight 1/S, and routing contributes no dV.
    x, dy = values.detach().float(), upstream.float()
    key = x[..., -rank:]
    key = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + 2**-23)
    dlogit = ((x - x.mean(0)) * dy).sum(-1) / sources
    expected_dq = (.7 * dlogit[..., None] * key).sum((0, 1)).bfloat16()
    expected_dv = (dy[None].expand_as(x) / sources).bfloat16()
    return {"output_max_abs": _compare(output, x.mean(0).bfloat16(), "analytic output"),
            "source_gradient_max_abs": _compare(dv, expected_dv, "analytic dV"),
            "query_gradient_max_abs": _compare(dq, expected_dq, "analytic dQ"),
            "scale": .7, "shape": [sources, rows, width, rank]}

def _training_case(spec: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    from attnres import attnres
    from benchmarks.bf16_model import Model

    config = _make_training_config(spec)
    seed = int(spec.get("seed", DEFAULT_SEED))
    torch.manual_seed(seed)
    candidate = Model(config, op=attnres).to(device).train()
    reference = Model(config, op=_oracle_operator).to(device).train()
    reference.load_state_dict(candidate.state_dict())
    candidate_optimizer = _optimizer(candidate, spec)
    reference_optimizer = _optimizer(reference, spec)
    tokens, targets = _training_batch(config, spec, seed + 17, device)

    candidate_loss, candidate_logits, candidate_grads = _backward_step(
        candidate, candidate_optimizer, tokens, targets
    )
    reference_loss, reference_logits, reference_grads = _backward_step(
        reference, reference_optimizer, tokens, targets
    )
    max_gradient_abs = 0.0
    for name, gradient in candidate_grads.items():
        max_gradient_abs = max(
            max_gradient_abs,
            _compare(gradient, reference_grads[name], f"gradient {name}"),
        )
    max_logits_abs = _compare(candidate_logits, reference_logits, "training logits")
    torch.testing.assert_close(
        candidate_loss, reference_loss, **BF16_TOLERANCE, msg="training loss"
    )
    candidate_optimizer.step()
    reference_optimizer.step()
    torch.cuda.synchronize(device)

    max_update_abs = _tree_max_abs(candidate.state_dict(), reference.state_dict())
    for name, value in candidate.state_dict().items():
        _compare(value, reference.state_dict()[name], f"updated parameter {name}")

    saved = {
        "model": _clone_tree(candidate.state_dict(), cpu=True),
        "optimizer": _clone_tree(candidate_optimizer.state_dict(), cpu=True),
    }
    encoded = io.BytesIO()
    torch.save(saved, encoded)

    uninterrupted_loss, _, _ = _backward_step(candidate, candidate_optimizer, tokens, targets)
    candidate_optimizer.step()
    uninterrupted_model = _clone_tree(candidate.state_dict(), cpu=True)
    uninterrupted_optimizer = _clone_tree(candidate_optimizer.state_dict(), cpu=True)

    encoded.seek(0)
    resumed = torch.load(encoded, map_location=device)
    candidate.load_state_dict(resumed["model"], strict=True)
    candidate_optimizer.load_state_dict(resumed["optimizer"])
    resumed_loss, _, _ = _backward_step(candidate, candidate_optimizer, tokens, targets)
    candidate_optimizer.step()
    resumed_model = _clone_tree(candidate.state_dict(), cpu=True)
    resumed_optimizer = _clone_tree(candidate_optimizer.state_dict(), cpu=True)
    _tree_exact(resumed_model, uninterrupted_model)
    _tree_exact(resumed_optimizer, uninterrupted_optimizer)
    torch.testing.assert_close(uninterrupted_loss, resumed_loss, rtol=0, atol=0)

    return {
        "model": {
            "layers": config.layers,
            "width": config.width,
            "heads": config.heads,
            "ffn": config.ffn,
            "vocab": config.vocab,
            "context": config.context,
            "rank": config.rank,
            "mode": config.mode,
            "block_count": config.block_count,
        },
        "activation_checkpointing": config.activation_checkpointing,
        "batch": int(spec.get("batch", 2)),
        "accumulation": int(spec.get("accumulation", 2)),
        "oracle_logits_max_abs": max_logits_abs,
        "oracle_gradient_max_abs": max_gradient_abs,
        "optimizer_update_max_abs": max_update_abs,
        "save_resume": "exact",
        "same_inputs": True,
    }


def _failure(name: str, phase: str, error: BaseException) -> dict[str, Any]:
    return {
        "name": name,
        "phase": phase,
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }


def run_qualification(
    config: Mapping[str, Any] | None = None,
    checkpoint: Callable[[Mapping[str, Any]], Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Run the CUDA BF16 qualification and return a durable JSON report.

    ``checkpoint`` may be a callback, matching the other benchmark runners, or
    a path.  It is called/written at start, after every case, and at completion.
    Case failures are retained in the report so a remote worker can resume or
    diagnose the run without parsing logs.
    """

    effective = _merge_config(config)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "bf16_qualification",
        "status": "running",
        "config": _json_copy(effective),
        "tolerance": dict(BF16_TOLERANCE),
        "graph_replays_required": GRAPH_REPLAYS,
        "cases": [],
        "failures": [],
        "passed": 0,
        "failed": 0,
    }
    _checkpoint(checkpoint, report)

    try:
        device = _cuda_device(effective)
        report["runtime"] = _runtime(device)
    except _QualificationSkip as exc:
        report["status"] = "skipped"
        report["failures"] = [{"phase": "preflight", "status": "skipped", "error": str(exc)}]
        _checkpoint(checkpoint, report)
        return _json_copy(report)
    except Exception as exc:  # noqa: BLE001 - persist every qualification failure
        report["status"] = "failed"
        report["failed"] = 1
        report["failures"].append(_failure("preflight", "preflight", exc))
        _checkpoint(checkpoint, report)
        return _json_copy(report)

    for index, case in enumerate(effective.get("operator_cases", ())):
        name = (
            str(case.get("name", f"operator_{index}"))
            if isinstance(case, Mapping)
            else f"operator_{index}"
        )
        try:
            metrics = _operator_case(case)
            row = {"name": name, "phase": "operator", "status": "passed", "metrics": metrics}
            report["passed"] += 1
        except Exception as exc:  # noqa: BLE001 - persist every qualification failure
            row = _failure(name, "operator", exc)
            report["failed"] += 1
            report["failures"].append(row)
        report["cases"].append(row)
        _checkpoint(checkpoint, report)
        gc.collect()
        torch.cuda.empty_cache()

    try:
        row = {"name": "zero_query_analytic_gradient", "phase": "analytic", "status": "passed",
               "metrics": _analytic_zero_query(device)}
        report["passed"] += 1
    except Exception as exc:
        row = _failure("zero_query_analytic_gradient", "analytic", exc)
        report["failed"] += 1
        report["failures"].append(row)
    report["cases"].append(row)
    _checkpoint(checkpoint, report)

    training = effective.get("training", {})
    if isinstance(training, Mapping) and bool(training.get("enabled", True)):
        try:
            metrics = _training_case(training, device)
            row = {
                "name": "activation_checkpoint_optimizer_resume",
                "phase": "training",
                "status": "passed",
                "metrics": metrics,
            }
            report["passed"] += 1
        except Exception as exc:  # noqa: BLE001 - persist every qualification failure
            row = _failure("activation_checkpoint_optimizer_resume", "training", exc)
            report["failed"] += 1
            report["failures"].append(row)
        report["cases"].append(row)
        _checkpoint(checkpoint, report)
        gc.collect()
        torch.cuda.empty_cache()

    if effective.get("pytest"):
        import subprocess
        import sys
        import re
        try:
            checked = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--strict-markers", *effective["pytest"]],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=1200)
            output = checked.stdout + checked.stderr
            row = {"name": "additional_cuda_tests", "phase": "pytest", "exit_code": checked.returncode,
                   "output": output, "status": "passed"}
            if checked.returncode or re.search(r"[1-9][0-9]* skipped", output):
                row["status"] = "failed"
                row["error"] = "CUDA tests failed or required coverage was skipped"
        except Exception as exc:
            row = _failure("additional_cuda_tests", "pytest", exc)
        report["cases"].append(row)
        report["passed" if row["status"] == "passed" else "failed"] += 1
        if row["status"] != "passed":
            report["failures"].append(row)
        _checkpoint(checkpoint, report)

    report["status"] = "failed" if report["failures"] else "passed"
    report["complete"] = True
    _checkpoint(checkpoint, report)
    return _json_copy(report)


def _read_config(value: str) -> Mapping[str, Any]:
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="{}", help="JSON object or JSON file path")
    parser.add_argument("--output", default="", help="durable JSON report path")
    args = parser.parse_args(argv)
    try:
        report = run_qualification(_read_config(args.config), args.output or None)
    except Exception as exc:  # noqa: BLE001 - convert CLI errors to JSON
        report = {
            "kind": "bf16_qualification",
            "status": "failed",
            "complete": True,
            "failures": [_failure("runner", "runner", exc)],
        }
        if args.output:
            _atomic_json(Path(args.output), report)
    print(json.dumps(_jsonable(report), sort_keys=True, allow_nan=False))
    return 0 if report.get("status") in {"passed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BF16_TOLERANCE",
    "DEFAULT_CONFIG",
    "DEFAULT_OPERATOR_CASES",
    "GRAPH_REPLAYS",
    "run_qualification",
]
