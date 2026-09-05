"""Prove disk autotuning reuse in fresh processes with unchanged BF16 checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def _worker(output):
    import torch
    from triton.runtime.autotuner import Autotuner
    from attnres import attnres
    from attnres._kernels import fla_full_sources
    from benchmarks.bf16_device import operator_case

    started = time.monotonic()
    result = operator_case({"shape": [9, 257, 1536, 384], "query_scale": .05},
                           {"candidate": attnres}, seed=20260827,
                           warmups=2, rounds=2, replays=8)
    torch.cuda.synchronize()
    if result["arms"]["candidate"]["status"] != "passed":
        raise AssertionError(result)
    tuners = {name: value for name, value in vars(fla_full_sources).items()
              if isinstance(value, Autotuner) and value.cache}
    if not tuners or not all(tuner.cache_results for tuner in tuners.values()):
        raise AssertionError("no enabled autotuners exercised")
    report = {"setup_and_check_s": time.monotonic() - started,
              "operator": result,
              "configs": {name: sorted((repr(key), str(config)) for key, config in tuner.cache.items())
                          for name, tuner in tuners.items()},
              "retuned": {name: hasattr(tuner, "bench_time") for name, tuner in tuners.items()},
              "metadata": {str(path.relative_to(os.environ["TRITON_CACHE_DIR"])):
                           hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in Path(os.environ["TRITON_CACHE_DIR"]).rglob("*.autotune.json")}}
    Path(output).write_text(json.dumps(report, indent=2) + "\n")


def check_reuse():
    with tempfile.TemporaryDirectory(prefix="attnres-cache-check-") as directory:
        root = Path(directory)
        env = {**os.environ, "TRITON_CACHE_AUTOTUNING": "1",
               "TRITON_CACHE_DIR": str(root / "triton"),
               "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor")}
        results = []
        for phase in ("cold", "warm"):
            output = root / f"{phase}.json"
            child = subprocess.run([sys.executable, "-m", "benchmarks.bf16_cache_check", str(output)],
                                   env=env, capture_output=True, text=True, timeout=600)
            if child.returncode:
                raise RuntimeError(f"{phase} cache check failed: {child.stdout}\n{child.stderr}")
            results.append(json.loads(output.read_text()))
        cold, warm = results
        if (not cold["metadata"] or cold["metadata"] != warm["metadata"] or
            cold["configs"] != warm["configs"] or not all(cold["retuned"].values()) or
            any(warm["retuned"].values())):
            raise AssertionError({"cache_reuse_not_demonstrated": results})
        return {"cold": cold, "warm": warm, "fresh_processes": 2,
                "changed_input_graph_replays_per_process": 8}


if __name__ == "__main__":
    _worker(sys.argv[1])
