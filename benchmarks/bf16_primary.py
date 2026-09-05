"""Create a bounded slice of the frozen primary matrix without renting GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[1] / "configs/bf16_primary.json"


def configuration(contract, modes, ranks, seeds):
    for name, selected in (("modes", modes), ("ranks", ranks), ("seeds", seeds)):
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(contract[name]):
            raise ValueError(f"invalid or duplicate primary {name}")
    cases = [{"model": {**contract["model"], "mode": mode, "rank": rank},
              "batch": contract["batch"], "sequence": contract["sequence"],
              "accumulation": contract["accumulation"], "backends": contract["backends"]}
             for mode in modes for rank in ranks]
    return {"kind": "training", "cases": cases, "seeds": seeds,
            "rounds": contract["rounds"], "warmups": contract["warmups"],
            "torch_baseline": True, "reuse_compiler_cache": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--modes", nargs="+", choices=("full", "block"), required=True)
    parser.add_argument("--ranks", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    config = configuration(contract, args.modes, args.ranks, args.seeds or contract["seeds"])
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
