#!/usr/bin/env python3
"""Build reproducible release artifacts for ``fast-attnres``.

The installable wheel intentionally contains runtime code only.  The audited
compiled-step reports live in ``results/compiled_step`` and are shipped as a
separately named evidence archive, together with the source distribution and a
SHA256 manifest.  No other results tree is accepted as release evidence.
All archive metadata is normalized after the PEP 517 build backend runs so the
same source tree and ``SOURCE_DATE_EPOCH`` produce byte-identical files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, Sequence
import zipfile


class ReleaseError(RuntimeError):
    """Raised when a release cannot be built without guessing."""


# Keep the evidence path and archive prefix in one place so a release can never
# silently substitute another results tree for the compiled-step campaign.
COMPILED_STEP_EVIDENCE_DIRNAME = "compiled_step"
COMPILED_STEP_ARCHIVE_PREFIX = "results/compiled_step"
COMPILED_STEP_MANIFEST_NAME = "campaign_manifest.json"
COMPILED_STEP_MANIFEST_SCHEMA = "attnres.compiled_step_campaign.manifest.v1"
COMPILED_STEP_SOURCE_COMMIT = "81dffbfeb0f84470513e846e3df8080e8ffb563d"


@dataclass(frozen=True)
class ProjectMetadata:
    """The project name and version used in release asset names."""

    name: str
    version: str

    @property
    def normalized_name(self) -> str:
        """Return the PEP 503 spelling used by wheel filenames."""

        return re.sub(r"[-_.]+", "_", self.name)

    @property
    def artifact_stem(self) -> str:
        return f"{self.name}-{self.version}"


@dataclass(frozen=True)
class ReleaseArtifacts:
    """Paths to the four files produced for a release."""

    wheel: Path
    sdist: Path
    evidence: Path
    checksums: Path

    @property
    def all(self) -> tuple[Path, ...]:
        return (self.wheel, self.sdist, self.evidence, self.checksums)


def _read_project_metadata(root: Path) -> ProjectMetadata:
    """Read ``[project]`` metadata without requiring an extra TOML package."""

    pyproject = root / "pyproject.toml"
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # Python 3.10 with no optional tomli install.
        tomllib = None

    if tomllib is not None:
        try:
            document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project = document["project"]
            name = str(project["name"])
            version = str(project["version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReleaseError(f"invalid project metadata in {pyproject}") from exc
    else:
        # The release script only needs two scalar keys.  Keep a small fallback
        # rather than adding a runtime dependency to the installed package.
        text = pyproject.read_text(encoding="utf-8")
        match = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
        if match is None:
            raise ReleaseError(f"missing [project] section in {pyproject}")
        section = match.group(1)
        name_match = re.search(r"(?m)^\s*name\s*=\s*['\"]([^'\"]+)['\"]", section)
        version_match = re.search(
            r"(?m)^\s*version\s*=\s*['\"]([^'\"]+)['\"]", section
        )
        if name_match is None or version_match is None:
            raise ReleaseError(f"project name/version missing from {pyproject}")
        name, version = name_match.group(1), version_match.group(1)

    if not name or not version or "/" in name or "/" in version:
        raise ReleaseError(f"invalid project name/version: {name!r}, {version!r}")
    return ProjectMetadata(name=name, version=version)


def source_date_epoch(value: int | str | None = None) -> int:
    """Resolve and validate the timestamp used for all archive metadata."""

    raw = os.environ.get("SOURCE_DATE_EPOCH", "0") if value is None else str(value)
    try:
        epoch = int(raw)
    except (TypeError, ValueError) as exc:
        raise ReleaseError(f"SOURCE_DATE_EPOCH must be an integer, got {raw!r}") from exc
    if epoch < 0:
        raise ReleaseError("SOURCE_DATE_EPOCH must be non-negative")
    return epoch


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading a potentially large report into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    """Return ``path`` only when it is a non-symlink regular file."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReleaseError(f"cannot inspect {label}: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ReleaseError(f"{label} must be a regular file: {path}")
    return path


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    """Read a finite JSON object from a regular file."""

    _regular_file(path, label)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot parse {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must contain a JSON object: {path}")
    return value


def _git_output(root: Path, *args: str) -> str:
    """Run a read-only Git query against a source checkout."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseError(
            f"cannot inspect performance source checkout {root}: git {' '.join(args)}"
        ) from exc
    return completed.stdout.strip()


def _absolute_leaf(path: str | Path) -> Path:
    """Resolve a path's parent while preserving a possibly symlinked leaf."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent.resolve() / candidate.name


def _compiled_step_source_commit(manifest: dict[str, object]) -> str:
    """Extract the exact measured source commit from the compact manifest."""

    expected_keys = {
        "frozen",
        "kernel_sha256",
        "project",
        "repo_head",
        "runner_sha256",
        "schema",
    }
    if set(manifest) != expected_keys:
        raise ReleaseError(
            "compiled-step campaign manifest must be the compact fair manifest"
        )
    if manifest.get("schema") != COMPILED_STEP_MANIFEST_SCHEMA:
        raise ReleaseError("compiled-step campaign manifest schema differs")
    value = manifest.get("repo_head")
    if value != COMPILED_STEP_SOURCE_COMMIT:
        raise ReleaseError(
            "compiled-step campaign manifest does not name the exact measured "
            f"source commit {COMPILED_STEP_SOURCE_COMMIT}"
        )
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ReleaseError("compiled-step campaign manifest repo_head is not a Git SHA")
    return value


def _compiled_step_job_paths(
    evidence_dir: Path,
) -> tuple[tuple[str, int, Path, Path, Path], ...]:
    """Resolve the six canonical report, audit, and attestation paths."""

    try:
        from benchmarks.audit_compiled_step import SUPPORTED_GPUS, SUPPORTED_SEEDS
    except (ImportError, ModuleNotFoundError) as exc:
        raise ReleaseError("compiled-step auditor is unavailable") from exc

    raw_dir = evidence_dir / "raw"
    audit_dir = evidence_dir / "audits"
    attestation_dir = evidence_dir / "attestations"
    result: list[tuple[str, int, Path, Path, Path]] = []
    for gpu in SUPPORTED_GPUS:
        for seed in SUPPORTED_SEEDS:
            stem = f"{gpu.lower()}_seed_{seed}"
            raw = raw_dir / f"{stem}.json"
            if not raw.exists() and not raw.is_symlink():
                raise ReleaseError(
                    f"compiled-step raw report for {gpu}/{seed} is missing: {raw}"
                )
            raw = _regular_file(raw, f"compiled-step raw report {gpu}/{seed}")
            audit = audit_dir / f"{stem}.json"
            attestation = attestation_dir / f"{stem}.json"
            audit = _regular_file(audit, f"compiled-step audit {gpu}/{seed}")
            attestation = _regular_file(
                attestation, f"compiled-step attestation {gpu}/{seed}"
            )
            result.append((gpu, seed, raw, audit, attestation))

    expected_names_by_dir = {
        raw_dir: {raw.name for _, _, raw, _, _ in result},
        audit_dir: {audit.name for _, _, _, audit, _ in result},
        attestation_dir: {attestation.name for _, _, _, _, attestation in result},
    }
    for directory, expected_names in expected_names_by_dir.items():
        if not directory.is_dir() or directory.is_symlink():
            raise ReleaseError(
                f"compiled-step evidence directory is missing: {directory}"
            )
        actual_names = {path.name for path in directory.iterdir()}
        if actual_names != expected_names:
            raise ReleaseError(
                f"compiled-step {directory.name} contains unexpected report files: "
                f"expected {sorted(expected_names)!r}, found {sorted(actual_names)!r}"
            )
    return tuple(sorted(result, key=lambda item: (item[0], item[1])))


def _compiled_step_reproduction_paths(
    evidence_dir: Path,
    jobs: Sequence[tuple[str, int, Path, Path, Path]],
) -> tuple[Path, dict[int, Path], str]:
    """Resolve the sealed wrapper/config inputs used by the fair reports."""

    try:
        from benchmarks.audit_compiled_step import EXPECTED_WRAPPER_SHA256
    except (ImportError, ModuleNotFoundError) as exc:
        raise ReleaseError("compiled-step auditor is unavailable") from exc

    reproduction_dir = evidence_dir / "reproduction"
    if not reproduction_dir.is_dir() or reproduction_dir.is_symlink():
        raise ReleaseError(
            f"compiled-step reproduction directory is missing: {reproduction_dir}"
        )
    wrapper = _regular_file(
        reproduction_dir / "run_exact_fair_campaign.py",
        "compiled-step reproduction wrapper",
    )
    wrapper_digest = sha256_file(wrapper)
    if wrapper_digest != EXPECTED_WRAPPER_SHA256:
        raise ReleaseError(
            "compiled-step reproduction wrapper differs from the sealed fair wrapper"
        )
    if {path.name for path in reproduction_dir.iterdir()} != {wrapper.name}:
        raise ReleaseError(
            "compiled-step reproduction directory contains unexpected files"
        )
    config_dir = evidence_dir / "configs"
    if not config_dir.is_dir() or config_dir.is_symlink():
        raise ReleaseError(f"compiled-step config directory is missing: {config_dir}")
    configs = {
        seed: _regular_file(
            config_dir / f"seed_{seed}.json",
            f"compiled-step config {seed}",
        )
        for _, seed, _, _, _ in jobs
    }
    actual_names = {path.name for path in config_dir.iterdir()}
    expected_names = {path.name for path in configs.values()}
    if actual_names != expected_names:
        raise ReleaseError(
            "compiled-step config directory contains unexpected files: "
            f"expected {sorted(expected_names)!r}, found {sorted(actual_names)!r}"
        )
    return wrapper, configs, wrapper_digest


def audit_compiled_step_evidence(
    evidence_dir: str | Path,
    *,
    performance_source: str | Path,
    campaign_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Audit all six compiled-step reports before creating a release archive.

    Every report is checked against a clean checkout of the exact measured
    source commit and a separate report-byte-bound hardware/vendor
    attestation.  The pre-existing audit JSON is checked for the same identity
    and status, but the raw report and attestation are always re-audited so a
    stale sidecar cannot promote changed bytes.
    """

    evidence = _absolute_leaf(evidence_dir)
    if evidence.name != COMPILED_STEP_EVIDENCE_DIRNAME:
        raise ReleaseError(
            "release evidence must come from results/compiled_step"
        )
    if not evidence.is_dir() or evidence.is_symlink():
        raise ReleaseError(f"compiled-step evidence directory is missing: {evidence}")
    manifest_path = (
        _absolute_leaf(campaign_manifest)
        if campaign_manifest is not None
        else evidence / COMPILED_STEP_MANIFEST_NAME
    )
    manifest = _read_json_object(manifest_path, "compiled-step campaign manifest")
    in_tree_manifest = evidence / COMPILED_STEP_MANIFEST_NAME
    _regular_file(in_tree_manifest, "compiled-step campaign manifest in evidence")
    if manifest_path != in_tree_manifest:
        if manifest_path.read_bytes() != in_tree_manifest.read_bytes():
            raise ReleaseError("external campaign manifest differs from archived evidence manifest")
    source = _absolute_leaf(performance_source)
    if not source.is_dir() or source.is_symlink():
        raise ReleaseError(f"performance source checkout is missing: {source}")
    source_commit = _compiled_step_source_commit(manifest)
    actual_commit = _git_output(source, "rev-parse", "HEAD")
    if actual_commit != source_commit:
        raise ReleaseError(
            f"performance source checkout is {actual_commit}, expected exact measured commit {source_commit}"
        )
    if _git_output(source, "status", "--porcelain", "--untracked-files=all"):
        raise ReleaseError(f"performance source checkout is dirty: {source}")

    try:
        from benchmarks.audit_compiled_step import audit_path, build_hero_projection
    except (ImportError, ModuleNotFoundError) as exc:
        raise ReleaseError("compiled-step auditor is unavailable") from exc

    jobs = _compiled_step_job_paths(evidence)
    wrapper, configs, wrapper_digest = _compiled_step_reproduction_paths(evidence, jobs)
    audited: list[dict[str, object]] = []
    for gpu, seed, raw, audit, attestation in jobs:
        report_digest = sha256_file(raw)
        report = _read_json_object(raw, f"compiled-step raw report {gpu}/{seed}")
        preflight = report.get("compiled_step_runtime_preflight")
        if not isinstance(preflight, dict):
            raise ReleaseError(
                f"compiled-step raw report {gpu}/{seed} has no runtime preflight"
            )
        config_path = configs[seed]
        config_digest = sha256_file(config_path)
        if preflight.get("wrapper_sha256") != wrapper_digest:
            raise ReleaseError(
                f"compiled-step raw report {gpu}/{seed} is not bound to the sealed reproduction wrapper"
            )
        if preflight.get("config_sha256") != config_digest:
            raise ReleaseError(
                f"compiled-step raw report {gpu}/{seed} is not bound to its archived config"
            )
        config = _read_json_object(config_path, f"compiled-step config {seed}")
        if report.get("config") != config:
            raise ReleaseError(
                f"compiled-step raw report {gpu}/{seed} config differs from its archived config"
            )
        try:
            result = audit_path(
                raw,
                repo_root=source,
                gpu=gpu,
                seed=seed,
                release_attestation_path=attestation,
                require_release_attestation=True,
                campaign_manifest=manifest_path,
            )
        except Exception as exc:  # auditor uses its own fail-closed exception
            raise ReleaseError(f"compiled-step audit failed for {gpu}/{seed}: {exc}") from exc
        expected = {
            "schema": "attnres.compiled_step_campaign.audit.v1",
            "status": "timing_verified",
            "timing_verified": True,
            "release_promotable": False,
            "attestation_verified": True,
            "gpu": gpu,
            "seed": seed,
            "report_sha256": report_digest,
        }
        for key, value in expected.items():
            if result.get(key) != value:
                raise ReleaseError(f"compiled-step audit result for {gpu}/{seed} has {key}={result.get(key)!r}")
        sidecar = _read_json_object(audit, f"compiled-step audit {gpu}/{seed}")
        for key, value in expected.items():
            if sidecar.get(key) != value:
                raise ReleaseError(f"compiled-step audit sidecar for {gpu}/{seed} has {key}={sidecar.get(key)!r}")
        # The sidecar is part of the archive for human inspection.  Bind its
        # numerical summary to the fresh raw-row audit as well, while allowing
        # harmless historical extra fields such as compiled_step_execution_status.
        for key in (
            "statistics",
            "timing_means_ms",
            "timing_rows",
            "rounds",
            "warmup",
            "sequence",
            "mode",
            "release_blockers",
        ):
            if key in result and sidecar.get(key) != result[key]:
                raise ReleaseError(
                    f"compiled-step audit sidecar for {gpu}/{seed} has stale {key}"
                )
        audited.append({
            "gpu": gpu,
            "seed": seed,
            "report": raw.name,
            "report_sha256": report_digest,
            "audit": audit.name,
            "attestation": attestation.name,
            "config": config_path.name,
            "wrapper": wrapper.name,
        })

    report_paths: dict[str, dict[int, Path]] = {"H100": {}, "B200": {}}
    attestation_paths: dict[str, dict[int, Path]] = {"H100": {}, "B200": {}}
    for gpu, seed, raw, _audit, attestation in jobs:
        report_paths[gpu][seed] = raw
        attestation_paths[gpu][seed] = attestation
    try:
        projection = build_hero_projection(
            report_paths,
            repo_root=source,
            campaign_manifest=manifest_path,
            release_attestation_paths=attestation_paths,
        )
    except Exception as exc:
        raise ReleaseError(f"compiled-step projection recomputation failed: {exc}") from exc
    archived_projection = _read_json_object(
        evidence / "hero_projection.json", "compiled-step hero projection"
    )
    if projection != archived_projection:
        raise ReleaseError("compiled-step hero projection differs from the six audited reports")
    return {
        "schema": "attnres.compiled_step_release_audit.v1",
        "status": "verified",
        "source_commit": source_commit,
        "campaign_manifest": manifest_path.name,
        "hero_projection": "hero_projection.json",
        "reports": audited,
    }


# Keep the verb used by workflow callers explicit while retaining the shorter
# audit name for Python callers and tests.
verify_compiled_step_evidence = audit_compiled_step_evidence


def _safe_archive_name(name: str) -> str:
    """Validate and normalize a path written into a tar or zip archive."""

    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ReleaseError(f"unsafe archive member path: {name!r}")
    return normalized


def _safe_link_target(target: str) -> str:
    """Reject symlink targets that could escape an extracted archive."""

    normalized = target.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ReleaseError(f"unsafe archive symlink target: {target!r}")
    return normalized


def _zip_datetime(epoch: int) -> tuple[int, int, int, int, int, int]:
    """Convert an epoch to ZIP's two-second, 1980-limited timestamp."""

    # ZIP cannot represent dates before 1980.  The default epoch of zero is
    # therefore represented by the earliest legal date.
    minimum = 315532800  # 1980-01-01T00:00:00Z
    maximum = 4354819198  # 2107-12-31T23:59:58Z, ZIP's latest representable date.
    if epoch > maximum:
        raise ReleaseError("SOURCE_DATE_EPOCH is too large for ZIP metadata")
    fields = time.gmtime(max(epoch, minimum))
    return (
        fields.tm_year,
        fields.tm_mon,
        fields.tm_mday,
        fields.tm_hour,
        fields.tm_min,
        fields.tm_sec - fields.tm_sec % 2,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes next to ``path`` and replace it as one filesystem action."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def normalize_wheel(path: Path, epoch: int) -> Path:
    """Canonicalize ZIP ordering, timestamps, permissions, and compression."""

    path = Path(path)
    with zipfile.ZipFile(path, "r") as source:
        infos = source.infolist()
        names = [_safe_archive_name(info.filename) for info in infos]
        if len(names) != len(set(names)):
            raise ReleaseError(f"wheel contains duplicate members: {path}")
        members = sorted(zip(names, infos), key=lambda item: item[0])
        payloads: list[tuple[str, zipfile.ZipInfo, bytes]] = []
        for name, info in members:
            if info.flag_bits & 0x1:
                raise ReleaseError(f"encrypted wheel members are unsupported: {name}")
            payloads.append((name, info, source.read(info)))

    output = io.BytesIO()
    timestamp = _zip_datetime(epoch)
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True
    ) as archive:
        for name, original, data in payloads:
            info = zipfile.ZipInfo(filename=name, date_time=timestamp)
            info.create_system = 3  # Unix; avoids host-specific DOS metadata.
            info.create_version = 20
            info.extract_version = 20
            info.flag_bits = 0x800 if any(ord(char) > 127 for char in name) else 0
            info.comment = b""
            info.extra = b""
            if original.is_dir() or name.endswith("/"):
                info.external_attr = (0o755 << 16) | 0x10
                info.compress_type = zipfile.ZIP_STORED
            else:
                original_mode = (original.external_attr >> 16) & 0o777
                mode = 0o755 if original_mode & 0o111 else 0o644
                info.external_attr = mode << 16
                info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compress_type=info.compress_type, compresslevel=9)

    _atomic_write(path, output.getvalue())
    return path


@dataclass(frozen=True)
class _TarEntry:
    name: str
    kind: str
    mode: int
    data: bytes = b""
    linkname: str = ""


def _tar_entry_from_path(path: Path, name: str, epoch: int) -> _TarEntry:
    """Read one tree entry without dereferencing symlinks."""

    name = _safe_archive_name(name)
    metadata = path.lstat()
    mode = metadata.st_mode
    if stat.S_ISDIR(mode):
        return _TarEntry(name=name.rstrip("/") + "/", kind="dir", mode=0o755)
    if stat.S_ISREG(mode):
        permissions = 0o755 if mode & 0o111 else 0o644
        return _TarEntry(name=name, kind="file", mode=permissions, data=path.read_bytes())
    if stat.S_ISLNK(mode):
        return _TarEntry(
            name=name,
            kind="symlink",
            mode=0o777,
            linkname=_safe_link_target(os.readlink(path)),
        )
    raise ReleaseError(f"unsupported evidence entry type: {path}")


def _tree_entries(root: Path, archive_prefix: str) -> list[_TarEntry]:
    """Collect a deterministic, symlink-preserving tree listing."""

    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ReleaseError(f"evidence directory must be a real directory: {root}")
    prefix = _safe_archive_name(archive_prefix).rstrip("/")
    entries = [_tar_entry_from_path(root, prefix + "/", 0)]

    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        directories.sort()
        files.sort()
        # os.walk lists symlinked directories in ``directories`` but will not
        # descend into them with followlinks=False.  Record the link itself and
        # remove it from the traversal list, so no external target is read.
        real_directories: list[str] = []
        for child in directories:
            child_path = directory_path / child
            relative = child_path.relative_to(root).as_posix()
            name = f"{prefix}/{relative}/"
            entries.append(_tar_entry_from_path(child_path, name, 0))
            if not child_path.is_symlink():
                real_directories.append(child)
        directories[:] = real_directories
        for child in files:
            child_path = directory_path / child
            relative = child_path.relative_to(root).as_posix()
            entries.append(_tar_entry_from_path(child_path, f"{prefix}/{relative}", 0))
    if len(entries) == 1:
        raise ReleaseError(f"evidence directory is empty: {root}")
    return sorted(entries, key=lambda entry: entry.name)


def _tar_entry_from_archive_member(member: tarfile.TarInfo, data: bytes = b"") -> _TarEntry:
    """Convert a source distribution member into canonical metadata."""

    name = _safe_archive_name(member.name)
    if member.isdir():
        return _TarEntry(name=name.rstrip("/") + "/", kind="dir", mode=0o755)
    if member.isreg():
        mode = 0o755 if member.mode & 0o111 else 0o644
        return _TarEntry(name=name, kind="file", mode=mode, data=data)
    if member.issym():
        return _TarEntry(
            name=name,
            kind="symlink",
            mode=0o777,
            linkname=_safe_link_target(member.linkname),
        )
    if member.islnk():
        return _TarEntry(
            name=name,
            kind="hardlink",
            mode=0o644,
            linkname=_safe_archive_name(member.linkname),
        )
    raise ReleaseError(f"unsupported source distribution member type: {member.name}")


def _write_tar_gz(path: Path, entries: Iterable[_TarEntry], epoch: int) -> Path:
    """Write a deterministic gzip-compressed POSIX tar archive."""

    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output, mode="wb", filename="", mtime=epoch, compresslevel=9
    ) as stream:
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for entry in sorted(entries, key=lambda item: item.name):
                info = tarfile.TarInfo(entry.name)
                info.mode = entry.mode
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = epoch
                info.pax_headers = {}
                if entry.kind == "dir":
                    info.type = tarfile.DIRTYPE
                    info.size = 0
                elif entry.kind == "file":
                    info.type = tarfile.REGTYPE
                    info.size = len(entry.data)
                elif entry.kind == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.size = 0
                    info.linkname = entry.linkname
                elif entry.kind == "hardlink":
                    info.type = tarfile.LNKTYPE
                    info.size = 0
                    info.linkname = entry.linkname
                else:  # pragma: no cover - _TarEntry is private and validated above.
                    raise ReleaseError(f"unknown tar entry kind: {entry.kind}")
                archive.addfile(info, io.BytesIO(entry.data) if entry.kind == "file" else None)
    _atomic_write(path, output.getvalue())
    return path


def normalize_sdist(path: Path, epoch: int) -> Path:
    """Canonicalize a backend-generated source distribution."""

    entries: list[_TarEntry] = []
    names: set[str] = set()
    with tarfile.open(path, mode="r:gz") as source:
        for member in source.getmembers():
            data = source.extractfile(member).read() if member.isreg() else b""
            entry = _tar_entry_from_archive_member(member, data)
            if entry.name in names:
                raise ReleaseError(f"source distribution contains duplicate member: {entry.name}")
            entries.append(entry)
            names.add(entry.name)
    return _write_tar_gz(path, entries, epoch)


def _build_distributions(root: Path, output_dir: Path, epoch: int) -> tuple[Path, Path]:
    """Run the standard PEP 517 builder, with a local fallback for CI smoke tests."""

    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = str(epoch)
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    build_available = subprocess.run(
        [sys.executable, "-m", "build", "--version"],
        cwd=root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if build_available:
        command = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(output_dir),
        ]
        subprocess.run(command, cwd=root, env=environment, check=True)
    else:
        # ``build`` is a dev dependency, but this fallback keeps the script
        # useful in a minimal checkout and does not install anything globally.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(root),
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(output_dir),
            ],
            cwd=root,
            env=environment,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from setuptools.build_meta import build_sdist; "
                    "build_sdist(sys.argv[1])"
                ),
                str(output_dir),
            ],
            cwd=root,
            env=environment,
            check=True,
        )

    wheels = sorted(output_dir.glob("*.whl"))
    sdists = sorted(output_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseError(
            f"expected one wheel and one sdist, found {len(wheels)} wheels and {len(sdists)} sdists"
        )
    return wheels[0], sdists[0]


def create_evidence_archive(
    evidence_dir: Path,
    output_path: Path,
    epoch: int,
    *,
    archive_prefix: str = COMPILED_STEP_ARCHIVE_PREFIX,
    extra_files: Sequence[tuple[Path, str]] = (),
) -> Path:
    """Archive raw evidence and optional repository files deterministically.

    ``extra_files`` is a sequence of ``(source_path, archive_name)`` pairs.
    Paths are read exactly once and validated; symlinks are represented as
    symlink entries and are never dereferenced.
    """

    entries = _tree_entries(Path(evidence_dir), archive_prefix)
    names = {entry.name for entry in entries}
    for source, archive_name in extra_files:
        source = Path(source)
        archive_name = _safe_archive_name(archive_name)
        if not stat.S_ISREG(source.lstat().st_mode):
            raise ReleaseError(f"evidence extra file is missing or not regular: {source}")
        entry = _tar_entry_from_path(source, archive_name, epoch)
        if entry.name in names:
            raise ReleaseError(f"duplicate evidence archive member: {entry.name}")
        entries.append(entry)
        names.add(entry.name)
    return _write_tar_gz(Path(output_path), entries, epoch)


def write_sha256sums(paths: Sequence[Path], output_path: Path) -> Path:
    """Write a stable GNU-compatible checksum file for the given artifacts."""

    records: list[tuple[str, str]] = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise ReleaseError(f"cannot checksum missing artifact: {path}")
        records.append((path.name, sha256_file(path)))
    if len({name for name, _ in records}) != len(records):
        raise ReleaseError("checksum artifact names must be unique")
    records.sort(key=lambda item: item[0])
    text = "".join(f"{digest}  {name}\n" for name, digest in records)
    _atomic_write(Path(output_path), text.encode("ascii"))
    return Path(output_path)


def build_release(
    root: Path,
    output_dir: Path,
    evidence_dir: Path,
    *,
    epoch: int | str | None = None,
    manifest: Path | None = None,
    campaign_manifest: Path | None = None,
    performance_source: Path | None = None,
    verify_evidence: bool | None = None,
) -> ReleaseArtifacts:
    """Build and return the complete release asset set.

    The default CLI evidence directory is the compiled-step campaign.  When
    that directory is selected, evidence verification is mandatory unless the
    caller explicitly opts into a local smoke build with
    ``verify_evidence=False``.  Other results trees are rejected.
    """

    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    evidence_dir = _absolute_leaf(evidence_dir)
    expected_evidence_dir = root / "results" / COMPILED_STEP_EVIDENCE_DIRNAME
    if campaign_manifest is not None:
        if manifest is not None:
            raise ReleaseError("pass either manifest or campaign_manifest, not both")
        manifest = campaign_manifest
    authoritative = evidence_dir.name == COMPILED_STEP_EVIDENCE_DIRNAME
    if evidence_dir.name == "release" and evidence_dir.parent.name == "results":
        raise ReleaseError("release evidence must come from results/compiled_step")
    if verify_evidence is None:
        verify_evidence = authoritative
    if authoritative and verify_evidence and evidence_dir != expected_evidence_dir:
        raise ReleaseError(
            "authoritative compiled-step evidence must be ROOT/results/compiled_step"
        )
    if verify_evidence and not authoritative:
        raise ReleaseError(
            "release evidence must come from results/compiled_step"
        )
    if manifest is not None and not authoritative:
        raise ReleaseError("campaign manifest requires the results/compiled_step evidence directory")
    if verify_evidence:
        performance_root = root if performance_source is None else performance_source
        audit_compiled_step_evidence(
            evidence_dir,
            performance_source=performance_root,
            campaign_manifest=manifest,
        )
    epoch_value = source_date_epoch(epoch)
    metadata = _read_project_metadata(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.relative_to(evidence_dir)
    except ValueError:
        pass
    else:
        raise ReleaseError("output directory must not be inside the evidence directory")

    with tempfile.TemporaryDirectory(prefix="release-build-", dir=output_dir) as temporary:
        staging = Path(temporary)
        wheel, sdist = _build_distributions(root, staging, epoch_value)
        normalize_wheel(wheel, epoch_value)
        normalize_sdist(sdist, epoch_value)
        canonical_sdist = staging / f"{metadata.normalized_name}-{metadata.version}.tar.gz"
        if sdist != canonical_sdist:
            if canonical_sdist.exists():
                raise ReleaseError(
                    f"canonical source distribution already exists: {canonical_sdist.name}"
                )
            os.replace(sdist, canonical_sdist)
            sdist = canonical_sdist

        evidence_name = f"{metadata.artifact_stem}-evidence.tar.gz"
        evidence = staging / evidence_name
        # The evidence tarball is a standalone release asset.  Keep its
        # attribution, provenance, result narrative, and sealed launch
        # configs beside the raw reports so extraction does not depend on the
        # wheel, sdist, or a live repository checkout.
        evidence_support_files = (
            "LICENSE",
            "NOTICE",
            "PROVENANCE.md",
            "docs/compiled_step_results.md",
            "configs/compiled_step_campaign.json",
            "configs/compiled_step_campaign_manifest.json",
        )
        extra_files = [(root / relative, relative) for relative in evidence_support_files]
        archive_prefix = COMPILED_STEP_ARCHIVE_PREFIX
        create_evidence_archive(
            evidence_dir,
            evidence,
            epoch_value,
            archive_prefix=archive_prefix,
            extra_files=extra_files,
        )

        checksums = staging / "SHA256SUMS"
        write_sha256sums((wheel, sdist, evidence), checksums)

        final_paths = {
            wheel: output_dir / wheel.name,
            sdist: output_dir / sdist.name,
            evidence: output_dir / evidence.name,
            checksums: output_dir / checksums.name,
        }
        for source, destination in final_paths.items():
            # copyfile plus replace keeps a previous complete artifact visible
            # until its replacement is ready.
            temporary_destination = output_dir / f".{destination.name}.tmp"
            shutil.copyfile(source, temporary_destination)
            os.replace(temporary_destination, destination)

    return ReleaseArtifacts(
        wheel=output_dir / wheel.name,
        sdist=output_dir / sdist.name,
        evidence=output_dir / evidence.name,
        checksums=output_dir / "SHA256SUMS",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: the parent of scripts/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for release assets (default: ROOT/dist/release)",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="compiled-step evidence directory (default: ROOT/results/compiled_step)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="compiled-step campaign manifest (default: EVIDENCE_DIR/campaign_manifest.json)",
    )
    parser.add_argument(
        "--campaign-manifest",
        dest="campaign_manifest",
        type=Path,
        default=None,
        help="alias for --manifest",
    )
    parser.add_argument(
        "--performance-source",
        "--performance-source-root",
        dest="performance_source",
        type=Path,
        default=None,
        help="clean checkout of the exact measured performance source commit",
    )
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        default=None,
        help="archive timestamp; defaults to SOURCE_DATE_EPOCH or zero",
    )
    parser.add_argument(
        "--without-manifest",
        action="store_true",
        help="omit the legacy manifest in a non-authoritative local smoke build",
    )
    parser.add_argument(
        "--skip-evidence-audit",
        action="store_true",
        help="skip compiled-step evidence audit (local smoke build only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "dist" / "release").resolve()
    evidence_dir = _absolute_leaf(args.evidence_dir or root / "results" / "compiled_step")
    selected_manifest = args.campaign_manifest or args.manifest
    if args.campaign_manifest is not None and args.manifest is not None:
        parser.error("pass either --manifest or --campaign-manifest, not both")
    manifest = None if args.without_manifest else selected_manifest
    if manifest is None and not args.without_manifest and evidence_dir.name == COMPILED_STEP_EVIDENCE_DIRNAME:
        manifest = evidence_dir / COMPILED_STEP_MANIFEST_NAME
    try:
        artifacts = build_release(
            root,
            output_dir,
            evidence_dir,
            epoch=args.source_date_epoch,
            manifest=manifest,
            performance_source=args.performance_source,
            verify_evidence=not args.skip_evidence_audit,
        )
    except (OSError, ReleaseError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    for artifact in artifacts.all:
        print(f"{artifact.name}  {sha256_file(artifact)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI workflow.
    raise SystemExit(main())
