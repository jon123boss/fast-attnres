"""Atomic backups of completed compiler files; no CUDA or cloud imports."""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path


def completed_cache_path(path):
    return not any(part in ("lock", "locks", "__pycache__") or part.endswith((".lock", ".tmp"))
                   or part.startswith(("tmp.", ".pending-")) for part in Path(path).parts)


def compiler_cache_stamp(roots):
    """Detect backup changes; the compilers still validate their own cache keys."""
    entries = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and completed_cache_path(path):
                stat = path.stat()
                entries.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(entries)


def save_compiler_cache(roots, destination, previous=None):
    """Replace the last good archive only after a complete, changed backup."""
    stamp = compiler_cache_stamp(roots)
    if stamp == previous:
        return previous
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f".{destination.name}.{os.getpid()}.pending")

    def completed_file(info):
        if completed_cache_path(info.name) and (info.isfile() or info.isdir()):
            return info
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="attnres-compiler-backup-") as temporary:
            local = Path(temporary) / "artifacts.tar.gz"
            with tarfile.open(local, "w:gz", compresslevel=1) as archive:
                for root in roots:
                    if root.exists():
                        archive.add(root, arcname=root.name, filter=completed_file)
            shutil.copyfile(local, pending)
            pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    return stamp
