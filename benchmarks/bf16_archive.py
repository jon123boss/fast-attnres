"""Deduplicate immutable campaign sources and restore any recorded job offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def archive(work: Path, output: Path):
    ledger = json.loads((work / "ledger.json").read_text())
    if any(row["status"] in ("running", "reserved") for row in ledger["jobs"]):
        raise ValueError("finish or reconcile active jobs before sealing the campaign")
    manifest = {"ledger": ledger, "jobs": {}}
    seen = set()
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for job in ledger["jobs"]:
            files = {}
            for section in ("snapshots", "results"):
                root = work / section / job["id"]
                if not root.exists():
                    continue  # Accounting reserves can exist without an execution.
                for path in sorted(root.rglob("*")):
                    if not path.is_file() or "__pycache__" in path.parts:
                        continue
                    data = path.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    if digest not in seen:
                        bundle.writestr("objects/" + digest, data)
                        seen.add(digest)
                    files[f"{section}/{path.relative_to(root)}"] = digest
            manifest["jobs"][job["id"]] = files
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
    return {"jobs": len(manifest["jobs"]), "unique_files": len(seen),
            "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def restore(bundle_path: Path, job: str, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        for relative, digest in manifest["jobs"][job].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("invalid archived path")
            data = bundle.read("objects/" + digest)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"corrupt archived source: {relative}")
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create")
    create.add_argument("work", type=Path)
    create.add_argument("output", type=Path)
    extract = actions.add_parser("restore")
    extract.add_argument("bundle", type=Path)
    extract.add_argument("job")
    extract.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "create":
        print(json.dumps(archive(args.work, args.output)))
    else:
        print(restore(args.bundle, args.job, args.output))


if __name__ == "__main__":
    main()
