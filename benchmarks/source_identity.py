"""Stable source and contract digests shared by evidence readers."""
import hashlib
import json
from pathlib import Path


def contract_digest(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def package_digest(package):
    digest = hashlib.sha256()
    for path in sorted(Path(package).rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
