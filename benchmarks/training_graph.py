"""Optional complete compiled-training CUDA Graph support.

The graph owns the complete step boundary: zeroing gradients, compiled model
and cross entropy, BF16 autocast, all accumulated backwards, and a capturable
fused AdamW update. Timed callers copy inputs before their timing interval and
call ``replay()`` without arguments.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


def _cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits, targets)


def _compiled_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    loss_function: Any,
    tokens: Tensor,
    targets: Tensor,
    accumulation: int,
) -> Tensor:
    """The fixed complete-training contract captured and replayed by the graph.

    A 2D batch is partitioned along its batch axis; a 3D input already carries
    one explicit microbatch per leading slice.
    """

    # ``set_to_none=False`` is required for replay: the captured zero_ kernels
    # clear the same gradient buffers on every launch.  set_to_none=True only
    # changes Python attributes and would leave replayed gradients accumulated.
    optimizer.zero_grad(set_to_none=False)
    result: Tensor | None = None
    if tokens.ndim == 3:
        token_batches = tokens.unbind(0)
        target_batches = targets.unbind(0)
    else:
        token_batches = tokens.chunk(accumulation, dim=0)
        target_batches = targets.chunk(accumulation, dim=0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for micro_tokens, micro_targets in zip(token_batches, target_batches):
            logits = model(micro_tokens)
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), micro_targets.reshape(-1))
            result = loss
            (loss / accumulation).backward()
    optimizer.step()
    if result is None:  # guarded by public validation, retained for type checkers
        raise RuntimeError("compiled step did not produce a loss")
    return result


def _clone_module_state(model: torch.nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_module_state(model: torch.nn.Module, state: Mapping[str, Tensor]) -> None:
    with torch.no_grad():
        current = model.state_dict()
        if current.keys() != state.keys():
            raise RuntimeError("model state topology changed during CUDA Graph warmup")
        for name, value in state.items():
            current[name].copy_(value)


def _clone_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[Any, dict[str, Any]]:
    snapshot: dict[Any, dict[str, Any]] = {}
    for parameter, values in optimizer.state.items():
        snapshot[parameter] = {
            key: value.detach().clone() if isinstance(value, Tensor) else copy.deepcopy(value)
            for key, value in values.items()
        }
    return snapshot


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    snapshot: Mapping[Any, Mapping[str, Any]],
) -> None:
    """Restore values while retaining newly allocated AdamW state tensors.

    AdamW creates ``step``, ``exp_avg`` and ``exp_avg_sq`` during the first
    warmup update.  Deleting those tensors after warmup would make capture
    allocate state, which CUDA Graphs prohibit.  Existing values are restored;
    newly created state tensors are reset to their ordinary initial zeros.
    """

    for parameter, values in optimizer.state.items():
        before = snapshot.get(parameter, {})
        for key, value in values.items():
            original = before.get(key)
            if isinstance(value, Tensor):
                if isinstance(original, Tensor):
                    value.copy_(original)
                else:
                    value.zero_()
            elif key in before:
                values[key] = copy.deepcopy(original)
        for key, original in before.items():
            if key not in values:
                values[key] = (
                    original.detach().clone()
                    if isinstance(original, Tensor)
                    else copy.deepcopy(original)
                )


def _validate_optimizer(optimizer: torch.optim.Optimizer) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("CUDA Graph training requires torch.optim.AdamW")
    for group in optimizer.param_groups:
        if group.get("fused") is not True or group.get("capturable") is not True:
            raise ValueError("optimizer must use fused=True and capturable=True")
        if group.get("differentiable") is True:
            raise ValueError("differentiable AdamW is incompatible with this capture path")


class CapturedTrainingStep:
    """A fixed-shape CUDA Graph containing one complete optimizer step."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        tokens: Tensor,
        targets: Tensor,
        accumulation: int = 1,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph training requires a CUDA device")
        if accumulation < 1 or isinstance(accumulation, bool) or not isinstance(accumulation, int):
            raise ValueError("accumulation must be a positive integer")
        _validate_optimizer(optimizer)
        if tokens.device.type != "cuda" or targets.device != tokens.device:
            raise ValueError("tokens and targets must be on the same CUDA device")
        if tokens.ndim not in (2, 3) or targets.shape != tokens.shape:
            raise ValueError("tokens and targets must have matching 2D or 3D shapes")
        if tokens.ndim == 3 and tokens.shape[0] != accumulation:
            raise ValueError("3D inputs must have leading dimension equal to accumulation")
        if tokens.ndim == 2 and tokens.shape[0] % accumulation:
            raise ValueError("2D batch size must be divisible by accumulation for graph capture")

        self.model = model
        self.optimizer = optimizer
        self.accumulation = accumulation
        self.device = tokens.device
        self.shape = tuple(tokens.shape)
        self.compiled_model = torch.compile(model, fullgraph=True, dynamic=False)
        self.compiled_loss = torch.compile(_cross_entropy, fullgraph=True, dynamic=False)

        # All warmup allocations, including fresh static leaves, occur on the
        # eventual capture stream.  A model/optimizer snapshot makes warmup
        # observationally neutral: first replay starts at the caller checkpoint.
        model_state = _clone_module_state(model)
        optimizer_state = _clone_optimizer_state(optimizer)
        default_stream = torch.cuda.current_stream(device=self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(default_stream)
        with torch.cuda.stream(self.stream):
            self.static_tokens = tokens.detach().clone()
            self.static_targets = targets.detach().clone()
            for _ in range(2):
                _compiled_step(
                    self.compiled_model,
                    optimizer,
                    self.compiled_loss,
                    self.static_tokens,
                    self.static_targets,
                    accumulation,
                )
            _restore_module_state(model, model_state)
            _restore_optimizer_state(optimizer, optimizer_state)

        # Wait only for the side stream's own work.  Capture starts on that
        # stream below, preserving the CUDA stream dependency contract.
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss = _compiled_step(
                self.compiled_model,
                optimizer,
                self.compiled_loss,
                self.static_tokens,
                self.static_targets,
                accumulation,
            )
        # Capture executes one real step while recording the graph.  Restore a
        # second time so the first replay is the caller's first optimizer step,
        # while the captured parameter/state pointers remain valid.
        with torch.cuda.stream(self.stream):
            _restore_module_state(model, model_state)
            _restore_optimizer_state(optimizer, optimizer_state)
        self.stream.synchronize()
        self.loss = self.loss.detach()

    def _validate_inputs(self, tokens: Tensor, targets: Tensor) -> None:
        if tokens.device != self.device or targets.device != self.device:
            raise ValueError("replay inputs must use the capture CUDA device")
        if tuple(tokens.shape) != self.shape or tuple(targets.shape) != self.shape:
            raise ValueError("replay inputs must match the captured shape")
        if tokens.dtype != self.static_tokens.dtype or targets.dtype != self.static_targets.dtype:
            raise TypeError("replay inputs must match captured dtypes")

    def copy_inputs(self, tokens: Tensor, targets: Tensor) -> None:
        """Copy a new batch into graph storage; call this outside timing."""

        self._validate_inputs(tokens, targets)
        self.static_tokens.copy_(tokens)
        self.static_targets.copy_(targets)

    def replay(self, tokens: Tensor | None = None, targets: Tensor | None = None) -> Tensor:
        """Replay the graph and return its loss.

        Supplying inputs is a convenience that calls :meth:`copy_inputs` first.
        Timed callers should invoke ``copy_inputs`` before their interval and
        then call ``replay()`` without arguments.
        """

        if (tokens is None) != (targets is None):
            raise ValueError("tokens and targets must be supplied together")
        if tokens is not None and targets is not None:
            self.copy_inputs(tokens, targets)
        self.graph.replay()
        return self.loss.detach()

    __call__ = replay


def capture_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: Tensor,
    targets: Tensor,
    *,
    accumulation: int = 1,
) -> CapturedTrainingStep:
    """Compile and capture a fixed-shape complete training step.

    The supplied optimizer's hyperparameters and parameter groups are retained;
    callers must construct fused, capturable AdamW themselves.  Warmup runs two
    complete steps solely to initialize compiler/optimizer state and restores
    ordinary model and optimizer state tensors both before and after capture.
    For a 2D input, its batch size must be divisible by ``accumulation`` so all
    captured microbatches have one static shape.
    """

    return CapturedTrainingStep(model, optimizer, tokens, targets, accumulation)


__all__ = ["CapturedTrainingStep", "capture_training_step"]
