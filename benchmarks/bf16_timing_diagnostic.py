"""Observe GC and host step intervals without changing the training evaluator."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import gc
import json
from pathlib import Path
import time


class Observer:
    def __init__(self):
        self.collections, self.steps, self.names, self.open = [], [], {}, {}

    def callback(self, phase, info):
        now, generation = time.perf_counter_ns(), info["generation"]
        if phase == "start":
            row = {"generation": generation, "start_ns": now, "stop_ns": None}
            self.collections.append(row)
            self.open[generation] = row
        elif phase == "stop" and generation in self.open:
            row = self.open.pop(generation)
            row.update(stop_ns=now, duration_ns=now - row["start_ns"],
                       collected=info.get("collected"), uncollectable=info.get("uncollectable"))

    def wrap(self, arm):
        step = arm.get("step")
        if step is None or getattr(step, "_gc_observed", False):
            return
        backend = self.names.get(id(arm["model"].op), "unknown")

        @functools.wraps(step)
        def observed(index, *args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return step(index, *args, **kwargs)
            finally:
                self.steps.append({"backend": backend, "input_index": index,
                                   "start_ns": start, "stop_ns": time.perf_counter_ns()})
        observed._gc_observed = True
        arm["step"] = observed

    def payload(self, status):
        return {"kind": "gc_timing_diagnostic", "status": status,
                "clock": "time.perf_counter_ns", "policy": "observation only; GC unchanged",
                "steps": self.steps, "gc_intervals": self.collections}


@contextmanager
def observe(training, observer):
    activate, select = training._activate_arm, training._case_backend_items

    def select_observed(case, backends):
        selected, missing = select(case, backends)
        observer.names.update((id(op), name) for name, op in selected)
        return selected, missing

    def activate_observed(arm):
        result = activate(arm)
        observer.wrap(arm)
        return result

    training._activate_arm, training._case_backend_items = activate_observed, select_observed
    callback = observer.callback
    gc.callbacks.append(callback)
    try:
        yield
    finally:
        training._activate_arm, training._case_backend_items = activate, select
        gc.callbacks.remove(callback)


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def run(config, output):
    if "primary_contract_sha256" in config:
        raise ValueError("GC observations are diagnostic, not primary measurements")
    from benchmarks import bf16_training as training
    observer, status = Observer(), "failed"
    sidecar = output.with_suffix(".gc.json")

    def checkpoint(report):
        write(output, report)
        write(sidecar, observer.payload("running"))

    try:
        with observe(training, observer):
            result = training.run_training(config, checkpoint)
        status = "complete"
        return result
    finally:
        write(sidecar, observer.payload(status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
