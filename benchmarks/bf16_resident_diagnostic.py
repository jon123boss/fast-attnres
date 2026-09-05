"""Diagnostic: retain independently qualified comparison models on one GPU."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

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


class Pool:
    def __init__(self, training):
        self.training = training
        self.activate_original, self.offload_original = training._activate_arm, training._offload_arm
        self.arms, self.resident, self.plan = [], set(), None

    def offload(self, arm):
        if id(arm) in self.resident:
            return
        self.offload_original(arm)
        if "step" in arm and not any(arm is known for known in self.arms):
            self.arms.append(arm)

    def activate(self, arm):
        if "step" not in arm:
            return self.activate_original(arm)
        if self.plan is None:
            arms = [a for a in self.arms if "step" in a]
            if not arms or not any(arm is a for a in arms):
                raise RuntimeError("resident diagnostic missed a qualified arm")
            free, capacity = torch.cuda.mem_get_info()
            self.plan = memory_plan([a["memory"] for a in arms], free=free, capacity=capacity,
                                    reserved=torch.cuda.memory_reserved(), allocated=torch.cuda.memory_allocated())
            print(json.dumps({"resident_admission": self.plan}), flush=True)
            if not self.plan["admitted"]:
                raise RuntimeError("resident diagnostic exceeds guarded live GPU capacity")
            owners = {}
            for index, known in enumerate(arms):
                parameters = tuple(known["model"].parameters())
                self.activate_original(known)
                if not all(a is b for a, b in zip(parameters, known["model"].parameters())):
                    raise AssertionError("activation replaced Parameter identities")
                tensors = [*known["model"].parameters(), *known["model"].buffers(),
                           *(p.grad for p in parameters if p.grad is not None),
                           *self.training._state_tensors([o.state_dict() for o in known["optimizers"]])]
                for tensor in tensors:
                    if not tensor.is_cuda:
                        # Optimizer scalar counters can intentionally remain on CPU.
                        if tensor.numel() != 1:
                            raise AssertionError("non-scalar comparison state remains on CPU")
                        continue
                    pointer = tensor.untyped_storage().data_ptr()
                    if pointer in owners and owners[pointer] != index:
                        raise AssertionError("different comparison models share GPU storage")
                    owners[pointer] = index
                self.resident.add(id(known))
            self.plan.update(arms=len(arms), disjoint_gpu_storages=len(owners),
                             allocated_after_activation=torch.cuda.memory_allocated())
        if id(arm) not in self.resident:
            raise RuntimeError("unregistered resident comparison arm")


@contextmanager
def retain(training, pool):
    training._activate_arm, training._offload_arm = pool.activate, pool.offload
    try:
        yield
    finally:
        training._activate_arm, training._offload_arm = pool.activate_original, pool.offload_original


def run(config, output):
    if "primary_contract_sha256" in config:
        raise ValueError("resident diagnostics cannot replace primary measurements")
    from benchmarks import bf16_training as training
    original = training.training_case

    def case(spec, backends, controls, seed, checkpoint, runtime=None):
        pool = Pool(training)
        def save(row):
            row["resident_admission"] = pool.plan
            row["arm_residency"] = "all_gpu_arms_diagnostic" if pool.resident else "one_gpu_arm"
            checkpoint(row)
        with retain(training, pool):
            result = original(spec, backends, controls, seed, save, runtime=runtime)
        result["resident_admission"] = pool.plan
        result["arm_residency"] = "all_gpu_arms_diagnostic" if pool.resident else "one_gpu_arm"
        return result

    def checkpoint(report):
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temporary.replace(output)

    training.training_case = case
    try:
        return training.run_training(config, checkpoint)
    finally:
        training.training_case = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
