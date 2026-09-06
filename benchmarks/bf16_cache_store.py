"""Incremental compiler artifacts, published only at synchronous checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zlib

from benchmarks.bf16_cache import completed_cache_path


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _replace(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class CompilerCache:
    """Keep compiler keys and absolute paths; cache manifests never retune them.

    Call save only after compilation has joined, outside timing/capture. Triton
    kernel groups are indivisible; Inductor publishes its files atomically.
    Local roots must match the roots that created serialized compiler records.
    """

    def __init__(self, directory, roots, identity, commit=None):
        self.roots = {name: Path(path).absolute() for name, path in roots.items()}
        self.identity = {"runtime": identity,
                         "roots": {name: str(path) for name, path in self.roots.items()}}
        self.directory = Path(directory) / _digest(_json(self.identity))
        self.commit = commit or (lambda: None)
        self.previous = {}
        self.last_groups = None

    def _path(self, name):
        parts = Path(name).parts
        if len(parts) < 2 or parts[0] not in self.roots or ".." in parts:
            raise ValueError("invalid compiler cache path")
        return self.roots[parts[0]].joinpath(*parts[1:])

    def _groups(self):
        groups = {}
        for name, root in self.roots.items():
            for path in sorted(root.rglob("*")):
                if path.is_symlink() or not path.is_file() or not completed_cache_path(path):
                    continue
                relative = path.relative_to(root)
                # A Triton directory is one compiler key; every __grp__ child
                # must be present before any of that key can be published.
                group = f"{name}/{relative.parts[0]}" if name == "triton" else f"{name}/{relative}"
                groups.setdefault(group, []).append((f"{name}/{relative}", path))
        complete = {}
        for key, files in groups.items():
            valid = True
            present = {str(path) for _, path in files}
            for _, path in files:
                if not path.name.startswith("__grp__"):
                    continue
                try:
                    children = json.loads(path.read_text())["child_paths"]
                    valid &= (isinstance(children, dict) and bool(children)
                              and all(isinstance(child, str) and child in present
                                      for child in children.values()))
                except (KeyError, TypeError, ValueError):
                    valid = False
            if valid:
                complete[key] = files
        return complete

    def save(self, *, triton_only=False):
        """Copy only changed bytes, commit them, then publish a complete manifest."""
        groups, pending = {}, {}
        stored_bytes = stored_files = 0
        for key, files in self._groups().items():
            if triton_only and not key.startswith("triton/"):
                continue
            records = {}
            for name, path in files:
                before = path.stat()
                stamp = (before.st_size, before.st_mtime_ns, before.st_ino)
                old = self.previous.get(name)
                if old is not None and old[0] == stamp:
                    record = old[1]
                else:
                    data = path.read_bytes()
                    after = path.stat()
                    if stamp != (after.st_size, after.st_mtime_ns, after.st_ino):
                        raise RuntimeError("compiler cache changed during synchronous checkpoint")
                    record = {"sha256": _digest(data), "bytes": len(data),
                              "mode": before.st_mode & 0o777}
                    target = self.directory / "objects" / record["sha256"]
                    if not target.exists():
                        compressed = zlib.compress(data, level=1)
                        _replace(target, compressed)
                        stored_bytes += len(compressed)
                        stored_files += 1
                records[name] = record
                pending[name] = (stamp, record)
            groups[key] = records
        if groups == self.last_groups:
            return {"changed": False, "groups": len(groups), "files": len(pending)}
        self.commit()
        manifest = _json({"version": 1, "identity": self.identity,
                          "created_ns": time.time_ns(), "groups": groups})
        path = self.directory / "manifests" / (_digest(manifest) + ".json")
        _replace(path, manifest)
        self.commit()
        self.previous, self.last_groups = pending, groups
        return {"changed": True, "manifest": path.name, "groups": len(groups),
                "files": len(pending), "manifest_sha256": _digest(manifest),
                "stored_files": stored_files, "stored_bytes": stored_bytes}

    def restore(self):
        """Restore the union of complete compatible groups, newest version per key."""
        manifests, groups = [], {}
        for path in self.directory.glob("manifests/*.json"):
            data = path.read_bytes()
            if path.stem != _digest(data):
                raise ValueError("compiler manifest checksum mismatch")
            manifest = json.loads(data)
            if manifest["version"] != 1 or manifest["identity"] != self.identity:
                raise ValueError("compiler manifest identity mismatch")
            manifests.append((manifest["created_ns"], path.name, manifest))
        for _, _, manifest in sorted(manifests):
            groups.update(manifest["groups"])
        # Stage verified bytes locally before publishing any compiler records.
        records = {name: record for group in groups.values() for name, record in group.items()}
        with tempfile.TemporaryDirectory(prefix="attnres-cache-restore-") as temporary:
            for name, record in records.items():
                self._path(name)
                digest = record["sha256"]
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid compiler object digest")
                try:
                    data = zlib.decompress((self.directory / "objects" / digest).read_bytes())
                except zlib.error as exc:
                    raise ValueError("compiler object checksum mismatch") from exc
                if len(data) != record["bytes"] or _digest(data) != digest:
                    raise ValueError("compiler object checksum mismatch")
                _replace(Path(temporary) / name, data)
            for name in sorted(records, key=lambda name: Path(name).name.startswith("__grp__")):
                record, path = records[name], self._path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(Path(temporary) / name, path)
                path.chmod(record["mode"])
                stat = path.stat()
                self.previous[name] = ((stat.st_size, stat.st_mtime_ns, stat.st_ino), record)
        self.last_groups = groups if manifests else None
        return {"loaded": bool(manifests), "manifests": [name for _, name, _ in sorted(manifests)],
                "groups": len(groups), "files": len(records)}
