#!/usr/bin/env python3
"""Verify final evidence and bind a release to its measured runtime, offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_runtime(package, measured):
    """Accept a version-only metadata change; all other bytes must match."""
    package, measured = Path(package), Path(measured)
    inventory = lambda root: {p.relative_to(root).as_posix(): p for p in root.rglob("*")
                              if p.is_file() and p.suffix != ".pyc"}
    current, original = inventory(package), inventory(measured)
    require(current.keys() == original.keys(), "runtime file inventory differs from measured source")
    version = re.compile(rb'^__version__ = "([0-9]+\.[0-9]+\.[0-9]+)"$', re.M)
    changed = []
    for name in sorted(current):
        actual, expected = current[name].read_bytes(), original[name].read_bytes()
        if actual == expected:
            continue
        require(name == "__init__.py", f"unmeasured runtime change: {name}")
        require(len(version.findall(actual)) == len(version.findall(expected)) == 1,
                "runtime version must be a single literal")
        require(version.sub(b'__version__ = "VERSION"', actual)
                == version.sub(b'__version__ = "VERSION"', expected),
                "unmeasured runtime change in __init__.py")
        changed.append(name)
    return {"status": "passed", "version_only_changes": changed,
            "runtime_files": {name: sha256(path) for name, path in sorted(current.items())}}


def extract_evidence(root, destination):
    """Check the archive and every member before exposing source or reports."""
    root, destination = Path(root), Path(destination)
    evidence = root / "results/final_sweep"
    manifest = json.loads((evidence / "manifest.json").read_text())
    require(manifest["schema"] == "attnres.final_sweep_bundle.v2"
            and manifest["status"] == "audited", "unexpected final evidence manifest")
    archive = evidence / "evidence.tar.gz"
    require(archive.stat().st_size == manifest["archive_bytes"]
            and sha256(archive) == manifest["archive_sha256"], "final evidence archive hash differs")
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        require(len(members) == len(manifest["files"])
                and {m.name for m in members} == set(manifest["files"]), "archive inventory differs")
        for member in members:
            path = Path(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts,
                    f"unsafe archive member: {member.name}")
            data = stream.extractfile(member).read()
            require(hashlib.sha256(data).hexdigest() == manifest["files"][member.name],
                    f"archive member hash differs: {member.name}")
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return destination / "evidence"


def verify_release(root, work):
    root, work = Path(root).resolve(), Path(work).resolve()
    evidence = extract_evidence(root, work)
    identity = verify_runtime(root / "src/attnres", evidence / "sources/final78/runner/src/attnres")
    sys.path.insert(0, str(root))
    from benchmarks.source_identity import package_digest
    manifest = json.loads((root / "results/final_sweep/manifest.json").read_text())
    require(package_digest(evidence / "sources/final78/runner/src/attnres")
            == manifest["candidate_package_sha256"], "measured package identity differs")
    command = [sys.executable, "-m", "benchmarks.final_sweep_report"]
    for prefix, phase in (("", "final78"), ("headline-", "final74"), ("completion-", "final80")):
        command += [f"--{prefix}source", str(evidence / "sources" / phase)]
        for gpu in ("H100", "B200"):
            command += [f"--{prefix}{gpu.lower()}", str(evidence / "runs" / gpu / phase)]
    output = work / "reproduced"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    subprocess.run(command + ["--output", str(output)], cwd=root, env=environment, check=True)
    for name in ("audit.json", "rank_comparison.json", "hero_projection.json"):
        require((root / "results/final_sweep" / name).read_bytes() == (output / name).read_bytes(),
                f"final evidence does not reproduce: {name}")
    identity["measured_package_sha256"] = manifest["candidate_package_sha256"]
    identity["release_package_sha256"] = package_digest(root / "src/attnres")
    identity["evidence_archive_sha256"] = manifest["archive_sha256"]
    (work / "release-audit.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if args.work:
        result = verify_release(args.root, args.work)
    else:
        with tempfile.TemporaryDirectory(prefix="final-release-") as temporary:
            result = verify_release(args.root, Path(temporary))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
