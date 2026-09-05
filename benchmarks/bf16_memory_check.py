"""Diagnostic entry point for a bounded compiler save/recompute policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    budget = config["activation_memory_budget"]
    if type(budget) not in (int, float) or budget not in (0., .25, .5):
        raise ValueError("diagnostic memory budget must be 0, 0.25 or 0.5")
    if config.get("primary_contract_sha256"):
        raise ValueError("a compiler diagnostic cannot masquerade as a primary run")
    import torch._functorch.config as functorch_config
    with functorch_config.patch(activation_memory_budget=budget):
        print(json.dumps({"activation_memory_budget": functorch_config.activation_memory_budget,
                          "scope": "diagnostic_only"}), flush=True)
        runpy.run_module("benchmarks.bf16_training", run_name="__main__")


if __name__ == "__main__":
    main()
