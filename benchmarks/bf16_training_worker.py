"""Persist training-arm compilations at synchronous, untimed checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks.bf16_cache import compiler_cache_stamp, save_compiler_cache


def training_checkpoint(output, cache_archive=None, commit=None, cache_store=None):
    """Keep the evaluator unchanged and finish backups before it resumes work."""
    roots = [Path(os.environ[name]) for name in
             ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")] if cache_archive else []
    stamp = compiler_cache_stamp(roots)
    saved_arms = set()

    def save(report):
        nonlocal stamp
        current = report.get("in_progress", {})
        qualified = {(current.get("seed"), json.dumps(current.get("case"), sort_keys=True), name)
                     for name, arm in current.get("arms", {}).items()
                     if arm.get("status") == "qualified"}
        if (cache_archive or cache_store) and qualified - saved_arms:
            try:
                if cache_store:
                    report["compiler_cache_checkpoint"] = cache_store.save()
                else:
                    updated = save_compiler_cache(roots, cache_archive, stamp)
                    if updated != stamp and commit is not None:
                        commit()
                    stamp = updated
                saved_arms.update(qualified)
            except Exception as exc:
                # A backup failure does not turn a valid measurement into a
                # correctness failure, but it must remain visible in evidence.
                report["compiler_backup_error"] = f"{type(exc).__name__}: {exc}"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temporary.replace(output)

    return save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache-archive", type=Path)
    cache.add_argument("--cache-store", type=Path)
    parser.add_argument("--cache-volume")
    args = parser.parse_args()
    commit = None
    store = None
    if args.cache_archive or args.cache_store:
        if not args.cache_volume:
            parser.error("compiler cache requires --cache-volume")
        import modal
        commit = modal.Volume.from_name(args.cache_volume).commit
        if args.cache_store:
            from benchmarks.bf16_cache_store import CompilerCache
            store = CompilerCache(**json.loads(args.cache_store.read_text()), commit=commit)
    from benchmarks.bf16_training import run_training
    run_training(json.loads(args.config.read_text()),
                 training_checkpoint(args.output, args.cache_archive, commit, store))


if __name__ == "__main__":
    main()
