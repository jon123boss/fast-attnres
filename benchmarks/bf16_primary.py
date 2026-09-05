"""Create a bounded slice of the frozen primary matrix without renting GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[1] / "configs/bf16_primary.json"

FIXTURE_FILES = tuple("benchmarks/" + name for name in (
    "baseline.py", "bf16_training.py", "bf16_model.py", "bf16_competitors.py",
    "bf16_device.py", "bf16_primary.py", "gluon_compat.py",
)) + ("validation/oracle.py",)


def contract_digest(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def fixture_digest(root):
    hashes = {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
              for name in FIXTURE_FILES}
    return contract_digest(hashes)


def package_digest(package):
    digest = hashlib.sha256()
    for path in sorted(Path(package).rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
            "torch_baseline": True, "reuse_compiler_cache": True,
            "cache_autotuning": contract["runtime"]["cache_autotuning"],
            "expected_identities": contract["identities"],
            "primary_contract_sha256": contract_digest(contract)}


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
