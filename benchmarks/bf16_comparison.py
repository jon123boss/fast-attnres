"""Keep independent comparison models resident when measured memory permits."""
from __future__ import annotations

import torch


def memory_plan(memories, *, free, reserved, allocated, capacity):
    persistent = sum(m["persistent_incremental_allocated_bytes"] for m in memories)
    scratch = max(m["peak_allocated_bytes_incremental"] - m["persistent_incremental_allocated_bytes"]
                  for m in memories)
    available = free + reserved - allocated
    margin = max(8 * 2**30, capacity // 10)
    return {"persistent_bytes": persistent, "temporary_bytes": scratch,
            "available_bytes": available, "margin_bytes": margin,
            "admitted": persistent + scratch + margin <= available}


def activate_all(arms, activate, state_tensors):
    """Activate qualified arms outside timing, or retain the single-arm policy.

    Capacity rejection leaves every arm on CPU. A transfer error is fatal;
    silently timing a partially activated pool would invalidate comparisons.
    """
    free, capacity = torch.cuda.mem_get_info()
    plan = memory_plan([a["memory"] for a in arms], free=free, capacity=capacity,
                       reserved=torch.cuda.memory_reserved(), allocated=torch.cuda.memory_allocated())
    if not plan["admitted"]:
        return plan
    owners = {}
    for index, arm in enumerate(arms):
        parameters = tuple(arm["model"].parameters())
        activate(arm)
        if not all(a is b for a, b in zip(parameters, arm["model"].parameters())):
            raise AssertionError("activation replaced Parameter identities")
        tensors = [*arm["model"].parameters(), *arm["model"].buffers(),
                   *(p.grad for p in parameters if p.grad is not None),
                   *state_tensors([o.state_dict() for o in arm["optimizers"]])]
        for tensor in tensors:
            if not tensor.is_cuda:
                if tensor.numel() != 1:  # Optimizer scalar counters may stay on CPU.
                    raise AssertionError("non-scalar comparison state remains on CPU")
                continue
            pointer = tensor.untyped_storage().data_ptr()
            if pointer in owners and owners[pointer] != index:
                raise AssertionError("different comparison models share GPU storage")
            owners[pointer] = index
    plan.update(arms=len(arms), disjoint_gpu_storages=len(owners),
                allocated_after_activation=torch.cuda.memory_allocated())
    return plan
