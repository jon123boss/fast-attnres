"""Check comparison-state transfers without changing the timed training step."""
from __future__ import annotations

import time

import torch
from torch.nn import functional as F


def qualify(op, config):
    from benchmarks.bf16_model import Config, Model
    from benchmarks.bf16_training import (
        _activate_arm, _compare_state_tree, _cpu_optimizer_state, _cpu_state,
        _offload_arm, _optimizers, _state_tensors,
    )

    started = time.monotonic()
    torch.manual_seed(20260905)
    spec = Config(layers=2, width=64, heads=4, ffn=128, vocab=257,
                  context=16, rank=16, mode="block", block_count=2)
    control, transferred = [Model(spec, op).cuda().train() for _ in range(2)]
    transferred.load_state_dict(control.state_dict())
    control_opts, transferred_opts = [_optimizers(model, config) for model in (control, transferred)]
    compiled = [torch.compile(model, fullgraph=True, dynamic=False,
                              options={"triton.cudagraphs": False}) for model in (control, transferred)]
    def compile_loss(forward):
        def loss_forward(tokens, targets):
            return F.cross_entropy(forward(tokens).flatten(0, 1), targets.flatten())
        return torch.compile(loss_forward, fullgraph=True, dynamic=False,
                             options={"triton.cudagraphs": False})

    compiled = [compile_loss(forward) for forward in compiled]
    # The frozen fixture uses independent dense parameter/buffer storage.
    # Gradient views are intentionally replaced; every step clears them.
    for model in (control, transferred):
        tensors = [*model.parameters(), *model.buffers()]
        storage = [t.untyped_storage().data_ptr() for t in tensors]
        if len(storage) != len(set(storage)):
            raise AssertionError("fixture has aliased model storage")

    def metadata(model, optimizers):
        tensors = [*model.parameters(), *model.buffers(),
                   *(p.grad for p in model.parameters() if p.grad is not None),
                   *_state_tensors([o.state_dict() for o in optimizers])]
        return [(str(t.dtype), tuple(t.shape), tuple(t.stride()), str(t.device)) for t in tensors]

    arm = {"model": transferred, "optimizers": transferred_opts}
    parameters = tuple(transferred.parameters())
    graph_counts = []
    for index in range(8):
        tokens = torch.randint(spec.vocab, (2, spec.context), device="cuda")
        targets = torch.randint(spec.vocab, tokens.shape, device="cuda")
        losses = []
        for model, forward, optimizers in zip((control, transferred), compiled,
                                               (control_opts, transferred_opts)):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss = forward(tokens, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            for optimizer in optimizers:
                optimizer.step()
            losses.append(loss.detach())
        graph_counts.append(int(torch._dynamo.utils.counters["stats"]["unique_graphs"]))
        if index > 1 and graph_counts[-1] != graph_counts[-2]:
            raise AssertionError("state transfers trigger repeated model recompilation")
        torch.testing.assert_close(*losses, rtol=0, atol=0)
        _compare_state_tree(_cpu_state(control), _cpu_state(transferred), strict=True)
        _compare_state_tree(_cpu_optimizer_state(control_opts),
                            _cpu_optimizer_state(transferred_opts), strict=True)
        before_metadata = metadata(transferred, transferred_opts)
        _offload_arm(arm)
        inactive = [*transferred.parameters(), *transferred.buffers(),
                    *(p.grad for p in transferred.parameters() if p.grad is not None),
                    *_state_tensors(arm["optimizer_states_cpu"])]
        if any(tensor.is_cuda for tensor in inactive) or any(o.state for o in transferred_opts):
            raise AssertionError("inactive comparison state remains on CUDA")
        _activate_arm(arm)
        if before_metadata != metadata(transferred, transferred_opts):
            raise AssertionError("transfer changed tensor dtype, shape, stride or device")
        _compare_state_tree(_cpu_state(control), _cpu_state(transferred), strict=True)
        _compare_state_tree(_cpu_optimizer_state(control_opts),
                            _cpu_optimizer_state(transferred_opts), strict=True)
        if not all(a is b for a, b in zip(parameters, transferred.parameters())):
            raise AssertionError("parameter identities changed during transfer")
    result = {"status": "passed", "updates": 8, "exact": True,
              "parameter_identities_preserved": True, "tensor_metadata_preserved": True, "unique_graph_counts": graph_counts,
              "optimizer": [type(optimizer).__name__ for optimizer in transferred_opts],
              "elapsed_s": time.monotonic() - started}
    # CPU parking releases the proof models' GPU state even if Dynamo retains a closure.
    _offload_arm(arm)
    _offload_arm({"model": control, "optimizers": control_opts})
    return result
