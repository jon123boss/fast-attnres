from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import verify_compiled_step_release as verifier
from scripts.build_release import ReleaseError


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    evidence = root / "results" / "compiled_step"
    kernel = root / "src" / "attnres" / "_kernels" / "fixed_tail.py"
    config = root / "configs" / "compiled_step_campaign.json"
    kernel.parent.mkdir(parents=True)
    evidence.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    kernel.write_text("kernel\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    (evidence / "campaign_manifest.json").write_text(
        json.dumps(
            {"kernel_sha256": {str(kernel.relative_to(root)): hashlib.sha256(kernel.read_bytes()).hexdigest()}}
        )
        + "\n",
        encoding="utf-8",
    )
    return root, evidence


def test_verifier_binds_release_kernel_evidence_and_external_fla(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence = _fixture(tmp_path)
    monkeypatch.setattr(
        verifier,
        "audit_compiled_step_evidence",
        lambda *args, **kwargs: {"status": "verified", "source_commit": "a" * 40, "reports": [1] * 6},
    )
    monkeypatch.setattr(
        verifier,
        "verify_release_fla_config",
        lambda *args, **kwargs: {
            "status": "verified",
            "actual": {
                "revision": "b" * 40,
                "origin": "https://github.com/fla-org/flash-linear-attention.git",
                "package_sha256": "c" * 64,
                "package_file_count": 506,
                "git_dirty": False,
            },
        },
    )

    result = verifier.verify_release(
        repo=root,
        performance_source=tmp_path / "source",
        fla_root=tmp_path / "fla",
        evidence_dir=evidence,
    )

    assert result["status"] == "verified"
    assert result["reports"] == 6
    assert result["fla"]["git_dirty"] is False


def test_verifier_rejects_changed_release_kernel(tmp_path: Path) -> None:
    root, evidence = _fixture(tmp_path)
    (root / "src/attnres/_kernels/fixed_tail.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ReleaseError, match="kernel hash differs"):
        verifier.verify_release(
            repo=root,
            performance_source=tmp_path / "source",
            fla_root=tmp_path / "fla",
            evidence_dir=evidence,
        )

