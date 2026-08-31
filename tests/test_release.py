"""Tests for reproducible release asset construction."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

from scripts import build_release as release


ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PERFORMANCE_BUNDLE = (
    ROOT
    / "validation"
    / "performance_sources"
    / "compiled-step-81dffbfeb0f84470513e846e3df8080e8ffb563d.bundle"
)


def _write_source_tar(path: Path) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        directory = tarfile.TarInfo("demo-0.1.0/")
        directory.mode = 0o755
        directory.mtime = int(time.time())
        archive.addfile(directory)
        for name, payload in (("z.txt", b"z"), ("a.txt", b"a")):
            member = tarfile.TarInfo(f"demo-0.1.0/{name}")
            member.mode = 0o644
            member.mtime = int(time.time()) + 100
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def _write_source_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("z.txt", "z")
        archive.writestr("a.txt", "a")


def test_wheel_normalization_is_stable_and_canonical(tmp_path: Path) -> None:
    wheel = tmp_path / "demo.whl"
    _write_source_wheel(wheel)

    release.normalize_wheel(wheel, epoch=0)
    first = wheel.read_bytes()
    release.normalize_wheel(wheel, epoch=0)
    assert wheel.read_bytes() == first

    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
        assert [member.filename for member in members] == ["a.txt", "z.txt"]
        assert all(member.date_time == (1980, 1, 1, 0, 0, 0) for member in members)
        assert all(member.create_system == 3 for member in members)


def test_wheel_normalization_rejects_traversal(tmp_path: Path) -> None:
    wheel = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("../outside.txt", "x")
    with pytest.raises(release.ReleaseError, match="unsafe archive member"):
        release.normalize_wheel(wheel, epoch=0)


def test_sdist_normalization_is_stable_and_canonical(tmp_path: Path) -> None:
    sdist = tmp_path / "demo.tar.gz"
    _write_source_tar(sdist)

    release.normalize_sdist(sdist, epoch=1700000000)
    first = sdist.read_bytes()
    release.normalize_sdist(sdist, epoch=1700000000)
    assert sdist.read_bytes() == first

    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == [
            "demo-0.1.0/",
            "demo-0.1.0/a.txt",
            "demo-0.1.0/z.txt",
        ]
        assert all(member.mtime == 1700000000 for member in members)
        assert all(member.uid == 0 and member.gid == 0 for member in members)
        assert all(member.uname == "" and member.gname == "" for member in members)


def test_evidence_archive_preserves_tree_and_symlinks_without_dereferencing(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "results" / "compiled_step"
    evidence.mkdir(parents=True)
    (evidence / "z.json").write_text("z", encoding="utf-8")
    (evidence / "a.json").write_text("a", encoding="utf-8")
    (evidence / "link.json").symlink_to("a.json")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    first_path = tmp_path / "first.tar.gz"
    second_path = tmp_path / "second.tar.gz"
    release.create_evidence_archive(
        evidence,
        first_path,
        epoch=0,
        extra_files=((manifest, "configs/compiled_step_campaign_manifest.json"),),
    )
    release.create_evidence_archive(
        evidence,
        second_path,
        epoch=0,
        extra_files=((manifest, "configs/compiled_step_campaign_manifest.json"),),
    )
    assert first_path.read_bytes() == second_path.read_bytes()

    with tarfile.open(first_path, mode="r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == [
            "configs/compiled_step_campaign_manifest.json",
            "results/compiled_step",
            "results/compiled_step/a.json",
            "results/compiled_step/link.json",
            "results/compiled_step/z.json",
        ]
        link = archive.getmember("results/compiled_step/link.json")
        assert link.issym()
        assert link.linkname == "a.json"


def test_evidence_archive_rejects_unsafe_prefix(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="unsafe archive member"):
        release.create_evidence_archive(
            evidence, tmp_path / "out.tar.gz", 0, archive_prefix="../evidence"
        )


@pytest.mark.parametrize("target", ("../outside", "/tmp/outside", "a/../outside"))
def test_evidence_archive_rejects_escaping_symlinks(tmp_path: Path, target: str) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "escape").symlink_to(target)

    with pytest.raises(release.ReleaseError, match="symlink target"):
        release.create_evidence_archive(evidence, tmp_path / "out.tar.gz", 0)


def test_sha256sums_is_sorted_and_uses_standard_format(tmp_path: Path) -> None:
    beta = tmp_path / "b.bin"
    alpha = tmp_path / "a.bin"
    alpha.write_bytes(b"alpha")
    beta.write_bytes(b"beta")
    checksums = tmp_path / "SHA256SUMS"

    release.write_sha256sums((beta, alpha), checksums)

    expected = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in (alpha, beta)
    )
    assert checksums.read_text(encoding="ascii") == expected


def test_build_release_returns_four_assets_and_excludes_evidence_from_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'demo-release'\nversion = '0.1.0'\n", encoding="utf-8"
    )
    evidence = root / "results" / "compiled_step"
    evidence.mkdir(parents=True)
    (evidence / "report.json").write_text("{}", encoding="utf-8")
    manifest = evidence / "campaign_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    support_files = (
        "LICENSE",
        "NOTICE",
        "PROVENANCE.md",
        "docs/compiled_step_results.md",
        "configs/compiled_step_campaign.json",
        "configs/compiled_step_campaign_manifest.json",
    )
    for relative in support_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"support:{relative}\n".encode())

    def fake_build(_root: Path, output: Path, _epoch: int) -> tuple[Path, Path]:
        wheel = output / "demo_release-0.1.0-py3-none-any.whl"
        _write_source_wheel(wheel)
        sdist = output / "demo-release-0.1.0.tar.gz"
        _write_source_tar(sdist)
        return wheel, sdist

    monkeypatch.setattr(release, "_build_distributions", fake_build)
    artifacts = release.build_release(
        root,
        root / "dist",
        evidence,
        epoch=0,
        manifest=manifest,
        verify_evidence=False,
    )

    assert [path.name for path in artifacts.all] == [
        "demo_release-0.1.0-py3-none-any.whl",
        "demo_release-0.1.0.tar.gz",
        "demo-release-0.1.0-evidence.tar.gz",
        "SHA256SUMS",
    ]
    with zipfile.ZipFile(artifacts.wheel) as archive:
        assert not any("results/compiled_step" in member for member in archive.namelist())
    with tarfile.open(artifacts.sdist, mode="r:gz") as archive:
        assert not any("results/compiled_step" in member.name for member in archive.getmembers())
    with tarfile.open(artifacts.evidence, mode="r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
        assert "results/compiled_step/report.json" in names
        assert "results/compiled_step/campaign_manifest.json" in names
        assert set(support_files) <= names
    assert artifacts.checksums.read_text(encoding="ascii").count("\n") == 3


def test_release_preserves_pep_625_backend_sdist_name(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo-release"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    evidence = root / "results" / "compiled_step"
    evidence.mkdir(parents=True)
    (evidence / "report.json").write_text("{}", encoding="utf-8")
    support_files = (
        "LICENSE",
        "NOTICE",
        "PROVENANCE.md",
        "docs/compiled_step_results.md",
        "configs/compiled_step_campaign.json",
        "configs/compiled_step_campaign_manifest.json",
    )
    for relative in support_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    def fake_build(_root: Path, output: Path, _epoch: int) -> tuple[Path, Path]:
        wheel = output / "demo_release-0.1.0-py3-none-any.whl"
        _write_source_wheel(wheel)
        sdist = output / "demo_release-0.1.0.tar.gz"
        _write_source_tar(sdist)
        return wheel, sdist

    monkeypatch.setattr(release, "_build_distributions", fake_build)
    artifacts = release.build_release(
        root,
        root / "dist",
        evidence,
        epoch=1700000000,
        verify_evidence=False,
    )

    assert artifacts.sdist.name == "demo_release-0.1.0.tar.gz"
    assert artifacts.sdist.is_file()
    assert not (artifacts.sdist.parent / "demo-release-0.1.0.tar.gz").exists()


def test_sdist_manifest_includes_provenance_and_readme_hero() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "PROVENANCE.md" in manifest
    assert "recursive-include docs *.md *.svg *.png" in manifest


def _fake_compiled_step_evidence(root: Path) -> tuple[Path, Path, Path]:
    evidence = root / "results" / "compiled_step"
    for directory in (
        evidence / "raw",
        evidence / "audits",
        evidence / "attestations",
        evidence / "configs",
        evidence / "reproduction",
    ):
        directory.mkdir(parents=True)
    wrapper = evidence / "reproduction" / "run_exact_fair_campaign.py"
    wrapper.write_text("fake wrapper\n", encoding="utf-8")
    wrapper_digest = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    manifest = evidence / "campaign_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frozen": {},
                "kernel_sha256": {},
                "project": {},
                "repo_head": "a" * 40,
                "runner_sha256": "b" * 64,
                "schema": "attnres.compiled_step_campaign.manifest.v1",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (evidence / "hero_projection.json").write_text("{}\n", encoding="utf-8")
    for gpu in ("H100", "B200"):
        for seed in (20260827, 20260903, 20260911):
            stem = f"{gpu.lower()}_seed_{seed}"
            config = {"seed": seed}
            config_path = evidence / "configs" / f"seed_{seed}.json"
            config_path.write_text(
                json.dumps(config, sort_keys=True) + "\n", encoding="utf-8"
            )
            config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
            raw = evidence / "raw" / f"{stem}.json"
            raw.write_text(
                json.dumps(
                    {
                        "compiled_step_runtime_preflight": {
                            "config_sha256": config_digest,
                            "wrapper_sha256": wrapper_digest,
                        },
                        "config": config,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            sidecar = {
                "attestation_verified": True,
                "gpu": gpu,
                "release_promotable": False,
                "report_sha256": digest,
                "schema": "attnres.compiled_step_campaign.audit.v1",
                "seed": seed,
                "status": "timing_verified",
                "timing_verified": True,
            }
            (evidence / "audits" / f"{stem}.json").write_text(
                json.dumps(sidecar, sort_keys=True) + "\n", encoding="utf-8"
            )
            (evidence / "attestations" / f"{stem}.json").write_text(
                json.dumps({"schema": "attnres.compiled_step_campaign.attestation.v1"}, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
    source = root / "performance-source"
    source.mkdir()
    return evidence, manifest, source


def test_compiled_step_audit_requires_all_six_reports_and_attestations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, manifest, source = _fake_compiled_step_evidence(tmp_path)
    monkeypatch.setattr(release, "COMPILED_STEP_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(release, "_git_output", lambda *_args: "a" * 40 if _args[-1] == "HEAD" else "")

    def fake_audit(path: Path, **kwargs: object) -> dict[str, object]:
        assert kwargs["repo_root"] == source
        assert kwargs["require_release_attestation"] is True
        assert kwargs["campaign_manifest"] == manifest
        name = path.stem
        gpu = "H100" if name.startswith("h100") else "B200"
        seed = int(name.rsplit("_", 1)[-1])
        return {
            "attestation_verified": True,
            "gpu": gpu,
            "release_promotable": False,
            "report_sha256": release.sha256_file(path),
            "schema": "attnres.compiled_step_campaign.audit.v1",
            "seed": seed,
            "status": "timing_verified",
            "timing_verified": True,
        }

    import benchmarks.audit_compiled_step as auditor

    monkeypatch.setattr(
        auditor,
        "EXPECTED_WRAPPER_SHA256",
        hashlib.sha256(
            (evidence / "reproduction" / "run_exact_fair_campaign.py").read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(auditor, "audit_path", fake_audit)
    monkeypatch.setattr(auditor, "build_hero_projection", lambda *_args, **_kwargs: {})
    result = release.audit_compiled_step_evidence(
        evidence, performance_source=source, campaign_manifest=manifest
    )
    assert result["status"] == "verified"
    assert result["source_commit"] == "a" * 40
    assert len(result["reports"]) == 6

    (evidence / "raw" / "b200_seed_20260911.json").unlink()
    with pytest.raises(release.ReleaseError, match="missing|canonical|exactly six"):
        release.audit_compiled_step_evidence(evidence, performance_source=source)


def test_build_release_rejects_noncanonical_evidence_when_verification_requested(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "results" / "release"
    evidence.mkdir(parents=True)
    (evidence / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="results/compiled_step"):
        release.build_release(
            tmp_path,
            tmp_path / "dist",
            evidence,
            verify_evidence=True,
            performance_source=tmp_path,
        )


def test_release_workflow_is_pinned_protected_and_least_privilege() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert 'tags:\n      - "v*"' in workflow
    assert "github.ref_protected == true" in workflow
    assert "python scripts/build_release.py" in workflow
    assert 'SOURCE_DATE_EPOCH="$epoch"' in workflow
    assert "validation/frozen.json" in workflow
    assert "results/compiled_step" in workflow
    assert "--campaign-manifest results/compiled_step/campaign_manifest.json" in workflow
    assert "--performance-source \"$PERFORMANCE_SOURCE_DIR\"" in workflow
    assert "git bundle verify" in workflow
    assert "git clone --no-local" in workflow
    assert "git fetch --no-tags origin" not in workflow
    assert "fetch-depth: 1" in workflow
    assert "subject-path: ${{ runner.temp }}/fast-attnres-release/*" in workflow
    assert "SHA256SUMS" in workflow
    assert "expected_names =" in workflow
    assert "if-no-files-found: error" in workflow

    action_refs = re.findall(r"^[ \t]*- uses:[ \t]+([^ \t#]+)", workflow, flags=re.MULTILINE)
    assert action_refs
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", ref) for ref in action_refs), action_refs
    assert "actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd" in workflow
    assert "softprops/action-gh-release" not in workflow

    # Only the GitHub-release job receives contents write. PyPI gets OIDC but
    # remains read-only for repository contents.
    assert workflow.count("contents: write") == 1
    assert "contents: read" in workflow
    assert "id-token: write" in workflow
    assert "packages-dir: ${{ runner.temp }}/fast-attnres-pypi/" in workflow
    assert "skip-existing: true" in workflow
    assert 're.sub(r"[-_.]+", "_", name)' in workflow
    assert "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33" in workflow


def test_sealed_performance_bundle_has_exact_source_identity(tmp_path: Path) -> None:
    assert hashlib.sha256(PERFORMANCE_BUNDLE.read_bytes()).hexdigest() == (
        "09547604f0a9630ed8769cf55479f255754dce2431d325ddcf250af8bafdde17"
    )
    checkout = tmp_path / "performance-source"
    result = subprocess.run(
        ["git", "clone", "--no-local", str(PERFORMANCE_BUNDLE), str(checkout)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    assert head == release.COMPILED_STEP_SOURCE_COMMIT
    assert status == ""


def test_release_workflow_separates_evidence_from_pypi_payloads() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "Attest every release asset" in workflow
    assert "if not path.name.endswith(\"-evidence.tar.gz\")" in workflow
    assert "shutil.copy2(path, pypi_dir / path.name)" in workflow
    assert "Publish only wheel and source distribution" in workflow
    assert "python -m attnres.verify_release" not in workflow
    assert "configs/release_evidence.json" not in workflow


def test_ci_evidence_job_validates_current_published_screen() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "Published compiled-step screen" in workflow
    assert "Verify current compiled-step evidence and populated chart" in workflow
    assert "tests/test_plot_compiled_step_sweep.py" in workflow
    assert "audit_compiled_step_evidence" not in workflow
    assert "compiled_step_hero.py" not in workflow
    assert "python -m attnres.verify_release" not in workflow
    assert "configs/release_evidence.json" not in workflow
    assert "plot_release_hero.py" not in workflow
