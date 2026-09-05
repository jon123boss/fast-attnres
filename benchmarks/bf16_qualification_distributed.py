"""Executable NCCL/DDP BF16 resume qualification.

Launch this module with ``torchrun``.  It intentionally keeps one local
uncached public-operator model per rank: the check is collective-gradient and
save/load/continue consistency, rather than a comparison against another
operator implementation.
"""

from __future__ import annotations

import argparse
import io
from contextlib import nullcontext
import gc
import json
import os
import platform
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

if __package__ in {None, ""}:  # Allow ``python benchmarks/bf16_qualification_distributed.py``.
    _SOURCE_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_SOURCE_ROOT))
    sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from attnres import attnres
from benchmarks.bf16_model import Config, Model

TOLERANCE = {"rtol": 0.05, "atol": 0.05}
SMALL_DEFAULT = {
    "layers": 2,
    "width": 64,
    "heads": 4,
    "ffn": 128,
    "vocab": 257,
    "context": 16,
    "block_count": 2,
    "rank": 16,
    "mode": "block",
}
PRIMARY_DEFAULT = {
    "layers": 24,
    "width": 1536,
    "heads": 24,
    "ffn": 4224,
    "vocab": 100277,
    "context": 2048,
    "block_count": 8,
    "mode": "block",
    "activation_checkpointing": False,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return str(value)


def _save_checkpoint(checkpoint: Callable[[dict[str, Any]], Any] | None,
                     report: dict[str, Any]) -> None:
    if checkpoint is not None and _rank() == 0:
        checkpoint(_jsonable(report))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _config_from_spec(spec: Mapping[str, Any], *, require_rank: bool = False) -> Config:
    raw = dict(spec)
    allowed = {field.name for field in fields(Config)}
    architecture = {key: value for key, value in raw.items() if key in allowed}
    if require_rank and "rank" not in architecture and "primary_rank" not in raw:
        raise ValueError("primary distributed qualification requires a supplied rank")
    if "rank" not in architecture and "primary_rank" in raw:
        architecture["rank"] = raw["primary_rank"]
    return Config(**architecture)


def _model_spec(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    if name == "small":
        raw = config.get("small", config.get("small_model", SMALL_DEFAULT))
        defaults = SMALL_DEFAULT
    else:
        raw = config.get("primary", config.get("primary_model", PRIMARY_DEFAULT))
        defaults = PRIMARY_DEFAULT
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} model configuration must be a mapping")
    result = {**defaults, **dict(raw)}
    if name == "primary" and "rank" not in result:
        for key in ("primary_rank", "rank"):
            if key in config:
                result["rank"] = config[key]
                break
    return result


def _device_metadata(device: torch.device) -> dict[str, Any]:
    import triton
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "device": str(device),
        "gpu": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "memory_bytes": int(properties.total_memory),
        "sms": int(properties.multi_processor_count),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _clone_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return value


def _unwrap_model(model):
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def _state_snapshot(model: DistributedDataParallel,
                    optimizers: list[torch.optim.Optimizer]) -> dict[str, Any]:
    return {
        "model": _clone_cpu(_unwrap_model(model).state_dict()),
        "optimizers": [_clone_cpu(optimizer.state_dict()) for optimizer in optimizers],
    }


def _tree_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return actual.keys() == expected.keys() and all(
            _tree_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _tree_equal(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(actual, Tensor) and isinstance(expected, Tensor):
        return torch.equal(actual, expected)
    return actual == expected


def _compare_tree(actual: Any, expected: Any, name: str) -> dict[str, Any]:
    maximum = 0.0
    tensor_count = 0

    def visit(left: Any, right: Any, path: str) -> None:
        nonlocal maximum, tensor_count
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if left.keys() != right.keys():
                raise AssertionError(f"{name}:{path} keys differ")
            for key in left:
                visit(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                raise AssertionError(f"{name}:{path} lengths differ")
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left, Tensor) and isinstance(right, Tensor):
            if not torch.isfinite(left).all().item() or not torch.isfinite(right).all().item():
                raise AssertionError(f"non-finite {name}:{path}")
            torch.testing.assert_close(left, right, **TOLERANCE, msg=f"{name}:{path}")
            tensor_count += 1
            maximum = max(maximum, float((left.float() - right.float()).abs().max().item()))
            return
        if left != right:
            raise AssertionError(f"{name}:{path} values differ")

    visit(actual, expected, "state")
    return {"tensor_count": tensor_count, "max_abs": maximum}



def _collective_gradients(model: DistributedDataParallel,
                          device: torch.device) -> dict[str, Any]:
    parameters = tuple(model.module.named_parameters())
    finite = torch.ones((), device=device, dtype=torch.int32)
    for _, parameter in parameters:
        if (parameter.grad is None or parameter.grad.dtype != torch.bfloat16 or
            not torch.isfinite(parameter.grad).all().item()):
            finite.zero_()
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if int(finite.item()) != 1:
        raise AssertionError("collective gradient contains a non-finite or missing value")

    maximum = 0.0
    mismatch = torch.zeros((), device=device, dtype=torch.int32)
    for name, parameter in parameters:
        reduced = parameter.grad.detach().float().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        mean = reduced / dist.get_world_size()
        difference = (parameter.grad.float() - mean).abs()
        mismatch.bitwise_or_((difference > TOLERANCE["atol"] + TOLERANCE["rtol"] * mean.abs()).any().int())
        maximum = max(maximum, float(difference.max().item()))
        del reduced, mean
    dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
    maximum_tensor = torch.tensor(maximum, device=device, dtype=torch.float32)
    dist.all_reduce(maximum_tensor, op=dist.ReduceOp.MAX)
    if mismatch.item():
        raise AssertionError(f"collective gradient mismatch: max_abs={maximum_tensor.item()}")
    return {
        "parameters": len(parameters),
        "finite": True,
        "max_rank_mean_abs": float(maximum_tensor.item()),
        "tolerance": dict(TOLERANCE),
    }


def _step(model: DistributedDataParallel, optimizers: list[torch.optim.Optimizer],
          tokens: Tensor, targets: Tensor, accumulation: int,
          device: torch.device) -> tuple[Tensor, dict[str, Any]]:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    losses = []
    for micro, (micro_tokens, micro_targets) in enumerate(zip(tokens.unbind(0), targets.unbind(0))):
        with (model.no_sync() if micro + 1 < accumulation else nullcontext()):
            logits = model(micro_tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), micro_targets.reshape(-1)
            )
            finite = torch.isfinite(loss).all().int()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise AssertionError("nonfinite distributed BF16 loss")
            (loss / accumulation).backward()
        losses.append(loss.detach())
    collective = _collective_gradients(model, device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for optimizer in optimizers:
        optimizer.step()
    return torch.stack(losses).mean().detach(), collective


def _case(config: Mapping[str, Any], name: str, seed: int,
          device: torch.device) -> dict[str, Any]:
    spec = _model_spec(config, name)
    model_config = _config_from_spec(spec, require_rank=name == "primary")
    batch = int(spec.get("batch", 2 if name == "small" else 4))
    accumulation = int(spec.get("accumulation", 2 if name == "small" else 4))
    if batch < 1 or accumulation < 1:
        raise ValueError("batch and accumulation must be positive")
    if name == "primary":
        expected = {key: PRIMARY_DEFAULT[key] for key in (
            "layers", "width", "heads", "ffn", "vocab", "context", "block_count",
            "mode", "activation_checkpointing"
        )}
        actual = {key: getattr(model_config, key) for key in expected}
        if actual != expected:
            raise ValueError(f"primary configuration changed: expected {expected}, got {actual}")
        if (batch != 4 or accumulation != 4 or model_config.rank not in (1536, 64) or
            model_config.rope_theta != 500000. or model_config.attnres_eps != 2**-23 or
            model_config.attnres_scale != 1.):
            raise ValueError("primary qualification requires R1536/R64, batch4/accum4 and original arithmetic")
    if not config.get("optimizer_source"):
        raise ValueError("distributed qualification requires the original Muon+AdamW source")

    torch.manual_seed(seed)
    model = Model(model_config, op=attnres).to(device).train()
    compiled = torch.compile(model, fullgraph=True, dynamic=False,
                             options={"triton.cudagraphs": False})
    ddp = DistributedDataParallel(
        compiled,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=True,
    )
    from benchmarks.bf16_training import _optimizers
    optimizers = _optimizers(ddp, config)
    expected_parameters = {id(p) for p in model.parameters()}
    covered = [id(p) for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]]
    if (sorted(type(o).__name__ for o in optimizers) != ["AdamW", "Muon"] or
        len(covered) != len(expected_parameters) or set(covered) != expected_parameters):
        raise RuntimeError("Muon+AdamW must cover every model parameter exactly once")
    torch.manual_seed(seed + 17 + 1009 * dist.get_rank())
    step_tokens = torch.randint(
        model_config.vocab,
        (2, accumulation, batch, model_config.context),
        device=device,
    )
    step_targets = torch.randint(
        model_config.vocab,
        (2, accumulation, batch, model_config.context),
        device=device,
    )

    first_loss, first_collective = _step(
        ddp, optimizers, step_tokens[0], step_targets[0], accumulation, device
    )
    saved = _state_snapshot(ddp, optimizers)
    uninterrupted_loss, uninterrupted_collective = _step(
        ddp, optimizers, step_tokens[1], step_targets[1], accumulation, device
    )
    uninterrupted = _state_snapshot(ddp, optimizers)

    encoded = io.BytesIO()
    torch.save(saved, encoded)
    encoded.seek(0)
    saved = torch.load(encoded, map_location="cpu", weights_only=True)
    _unwrap_model(ddp).load_state_dict(saved["model"], strict=True)
    for optimizer, state in zip(optimizers, saved["optimizers"]):
        optimizer.load_state_dict(state)
    del encoded
    dist.barrier()
    resumed_loss, resumed_collective = _step(
        ddp, optimizers, step_tokens[1], step_targets[1], accumulation, device
    )
    resumed = _state_snapshot(ddp, optimizers)
    resume_metrics = {
        "same_inputs": True,
        "exact": (
            _tree_equal(resumed, uninterrupted)
            and torch.equal(resumed_loss.cpu(), uninterrupted_loss.cpu())
        ),
        "loss_max_abs": float(
            (resumed_loss.cpu() - uninterrupted_loss.cpu()).abs().max().item()
        ),
        "state": _compare_tree(resumed, uninterrupted, "resume"),
    }
    if not resume_metrics["exact"] and bool(spec.get("require_exact_resume", True)):
        raise AssertionError("save/load/continue diverged for identical inputs")

    result = {
        "name": name,
        "status": "passed",
        "operator": "attnres",
        "cache": "uncached",
        "schedule": model_config.mode,
        "architecture": asdict(model_config),
        "batch": batch,
        "accumulation": accumulation,
        "steps": 2,
        "losses": [float(first_loss.cpu()), float(uninterrupted_loss.cpu())],
        "collective_gradients": [
            first_collective, uninterrupted_collective, resumed_collective
        ],
        "optimizer_update_and_resume": resume_metrics,
        "bf16_tolerance": dict(TOLERANCE),
    }
    del ddp, model, compiled, optimizers
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _initial_report(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "bf16_distributed_qualification",
        "status": "running",
        "complete": False,
        "passed": False,
        "config": _jsonable(dict(config)),
        "world_size": _world_size(),
        "tolerance": dict(TOLERANCE),
        "dtype": str(torch.bfloat16),
        "operator": "attnres",
        "cache": "uncached",
        "cases": [],
        "failures": [],
    }


def run_distributed_qualification(
    config: Mapping[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run small DDP and optional primary Block cases on an NCCL process group."""

    config = dict(config or {})
    report = _initial_report(config)
    if not dist.is_available():
        report.update(status="failed", complete=True)
        report["failures"].append({"phase": "environment", "error": "torch.distributed is unavailable"})
        _save_checkpoint(checkpoint, report)
        return report
    if not dist.is_initialized():
        required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
        if not all(name in os.environ for name in required):
            report.update(status="failed", complete=True)
            report["failures"].append({
                "phase": "environment",
                "error": "launch with torchrun and NCCL (RANK, WORLD_SIZE and LOCAL_RANK are required)",
            })
            _save_checkpoint(checkpoint, report)
            return report
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    report["world_size"] = world_size
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if not torch.cuda.is_available():
        local_failure = {"phase": "environment", "error": "NCCL qualification requires CUDA"}
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, local_failure)
        if rank == 0:
            report["failures"].extend(gathered)
        report.update(status="failed", complete=True)
        _save_checkpoint(checkpoint, report)
        return report if rank == 0 else {"rank": rank, "status": "failed"}
    expected_world_size = int(config.get("world_size", 8))
    if world_size != expected_world_size:
        local_failure = {
            "phase": "environment",
            "error": f"expected {expected_world_size} ranks, got {world_size}",
        }
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_failure)
        if rank == 0:
            report["failures"].extend(gathered)
        report.update(status="failed", complete=True)
        _save_checkpoint(checkpoint, report)
        return report if rank == 0 else {"rank": rank, "status": "failed"}

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    runtime = {**_device_metadata(device), "rank": rank, "local_rank": local_rank}
    runtimes = [None] * world_size
    dist.all_gather_object(runtimes, runtime)
    report["runtime"] = runtimes[0]
    report["rank_runtimes"] = runtimes
    expected_capability = {"H100": [9, 0], "B200": [10, 0]}.get(config.get("gpu"))
    invalid = [row for row in runtimes if
               row["capability"] != expected_capability or not row["bf16_supported"] or
               row["torch"] != "2.13.0+cu130" or row["triton"] != "3.7.1" or row["cuda"] != "13.0"]
    if invalid or sorted(row["local_rank"] for row in runtimes) != list(range(8)):
        report["failures"].append({"phase": "environment", "error": "distributed runtime substitution", "runtimes": runtimes})
        report.update(status="failed", complete=True)
        _save_checkpoint(checkpoint, report)
        return report if rank == 0 else {"rank": rank, "status": "failed"}
    names = ["small"]
    if bool(config.get("include_primary", False)):
        names.append("primary")

    local_rows = []
    for index, name in enumerate(names):
        try:
            row = _case(config, name, int(config.get("seed", 20260827)) + index, device)
        except Exception as error:  # noqa: BLE001 - record every rank's case failure.
            row = {
                "name": name,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        local_rows.append(row)
        # All ranks execute the same case order.  This barrier keeps the
        # single-model memory release synchronized before the next case.
        dist.barrier()

    gathered_rows: list[Any] = [None] * world_size
    dist.all_gather_object(gathered_rows, local_rows)
    if rank == 0:
        for index, name in enumerate(names):
            rows = [items[index] for items in gathered_rows]
            failed = [row for row in rows if row.get("status") != "passed"]
            if failed:
                row = {"name": name, "status": "failed", "rank_results": rows}
                report["failures"].extend({
                    "case": name,
                    "phase": "distributed",
                    "rank": rank_result,
                    "error": rank_result.get("error", "distributed case failed"),
                    "traceback": rank_result.get("traceback"),
                } for rank_result in failed)
            else:
                row = dict(rows[0])
                row["rank_results"] = rows
            report["cases"].append(row)
            _save_checkpoint(checkpoint, report)
        report["status"] = "passed" if not report["failures"] else "failed"
        report["complete"] = True
        report["passed"] = not bool(report["failures"])
        _save_checkpoint(checkpoint, report)
        return _jsonable(report)

    return {"rank": rank, "status": "complete", "cases": local_rows}


def _file_checkpoint(path: Path) -> Callable[[dict[str, Any]], None]:
    def save(report: dict[str, Any]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    return save


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-primary", action="store_true")
    parser.add_argument("--primary-rank", "--rank", dest="primary_rank", type=int)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text()) if args.config else {}
    if args.include_primary:
        config["include_primary"] = True
    if args.primary_rank is not None:
        config["primary_rank"] = args.primary_rank
        config["include_primary"] = True
    result = run_distributed_qualification(config, _file_checkpoint(args.output))
    if _rank() == 0:
        print(json.dumps({"status": result["status"], "failures": len(result["failures"])}))
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if result.get("passed", result.get("status") == "complete") else 1


if __name__ == "__main__":  # pragma: no cover - exercised by torchrun remotely.
    raise SystemExit(main())
