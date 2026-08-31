"""Verify the compiled-step release evidence and its external FLA checkout.

This verifier is intentionally CPU-only.  It re-audits all six report files
against a clean checkout of the measured source commit, checks that the
release tree still ships the measured production kernel bytes, and hashes a
separately supplied FLA checkout instead of trusting adapter metadata alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from benchmarks.fla_checkout import verify_release_fla_config
from scripts.build_release import ReleaseError, audit_compiled_step_evidence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_release(
    *,
    repo: str | Path,
    performance_source: str | Path,
    fla_root: str | Path,
    evidence_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return a bounded JSON verification result or raise on any mismatch."""

    root = Path(repo).expanduser().resolve()
    source = Path(performance_source).expanduser().resolve()
    evidence = (
        root / "results" / "compiled_step"
        if evidence_dir is None
        else Path(evidence_dir).expanduser().resolve()
    )
    manifest_path = evidence / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernels = manifest.get("kernel_sha256")
    if not isinstance(kernels, dict) or not kernels:
        raise ReleaseError("compiled-step manifest kernel identity is missing")
    actual_kernels: dict[str, str] = {}
    for relative, expected in sorted(kernels.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ReleaseError("compiled-step manifest kernel identity is malformed")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ReleaseError(f"unsafe compiled-step kernel path: {relative}") from exc
        if not path.is_file() or path.is_symlink():
            raise ReleaseError(f"compiled-step kernel is missing or not regular: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ReleaseError(f"release kernel hash differs for {relative}")
        actual_kernels[relative] = actual

    evidence_result = audit_compiled_step_evidence(
        evidence,
        performance_source=source,
        campaign_manifest=manifest_path,
    )
    if evidence_result.get("status") != "verified":
        raise ReleaseError("compiled-step evidence audit did not verify")

    config = json.loads(
        (root / "configs" / "compiled_step_campaign.json").read_text(encoding="utf-8")
    )
    fla_result = verify_release_fla_config(
        config,
        project_root=root,
        configured=Path(fla_root).expanduser().resolve(),
    )
    if fla_result.get("status") != "verified":
        error = fla_result.get("error", fla_result)
        raise ReleaseError(f"external FLA checkout did not verify: {error}")

    return {
        "schema": "attnres.compiled_step_release_verification.v1",
        "status": "verified",
        "performance_source": evidence_result.get("source_commit"),
        "reports": len(evidence_result.get("reports", ())),
        "kernel_sha256": actual_kernels,
        "fla": {
            key: fla_result["actual"][key]
            for key in (
                "revision",
                "origin",
                "package_sha256",
                "package_file_count",
                "git_dirty",
            )
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--performance-source", required=True)
    parser.add_argument("--fla-root", required=True)
    parser.add_argument("--evidence-dir")
    args = parser.parse_args(argv)
    try:
        result = verify_release(
            repo=args.repo,
            performance_source=args.performance_source,
            fla_root=args.fla_root,
            evidence_dir=args.evidence_dir,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": "attnres.compiled_step_release_verification.v1",
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI.
    raise SystemExit(main())

