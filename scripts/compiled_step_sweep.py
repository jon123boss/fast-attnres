"""Screen a Full/Block compiled-training-step matrix on remote GPUs.

This module is deliberately separate from the sealed six-job release campaign.
It is a fast, resume-safe screen for choosing the next release matrix.  The
worker runs the existing ``benchmarks.run.run_suite`` model phase, whose CUDA
event surrounds only one captured complete training step.  No tensor hashing,
report auditing, or per-round numerical check is added to that event.

The compact matrix fixes ``L=8``, ``B=2, T=512, V=8192``, and
``head_dim=64`` (therefore ``heads=D/64``).  Full has 17 maximum visible
sources.  Block's ``event_block_size`` is the number of Transformer sublayer
events per block; with 16 events at L8, block sizes 8/4/2/1 map to two/four/
eight/sixteen blocks and maximum per-read source counts 3/5/9/17.  The LR
arm uses the fixed compression ratio ``R=D/4``.  The former D=1024,R=128
screen is historical and is not part of this matrix.

The command has two roles:

``worker``
    Runs one cell on a host that already contains this checkout and atomically
    writes a worker result.  It performs only a small runtime/GPU preflight.

``sweep``
    Runs cells one at a time over SSH, retaining every failure and atomically
    updating a local index after each cell.  A complete canonical cell result
    is never overwritten, so an interrupted run can be resumed safely.

Liger is wired as a native compiled model-step comparator when the cell's
actual per-read source schedule satisfies its envelope.  Its source lists are
stacked and made contiguous inside the adapter call, and that work remains
inside the captured training-step boundary.  Catswe is an explicit opt-in
model-step comparator through its pinned public phase-1 adapter.  It runs one
public phase-1 call per actual Full or Block read, with source-list staging
inside the captured call; it has no prepare, merge, cache, or phase-2 route
and never falls back to an operator-only path.
"""

from __future__ import annotations

import argparse
import base64
import copy
import datetime as _datetime
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# ``python -m scripts.compiled_step_sweep`` starts with the repository root on
# ``sys.path``.  A direct ``python scripts/compiled_step_sweep.py`` invocation
# starts with only ``scripts/`` there, so add the resolved checkout root before
# any phase-local benchmark imports are reached.  Keep module invocation's
# package resolution unchanged.
if __package__ in {None, ""}:
    _SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_SCRIPT_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

SCHEMA = "attnres.compiled_step_sweep.cell.v1"
INDEX_SCHEMA = "attnres.compiled_step_sweep.index.v1"
MANIFEST_SCHEMA = "attnres.compiled_step_sweep.manifest.v1"
SUPPORTED_GPUS = ("H100", "B200")
DEFAULT_HOSTS = {
    "H100": {"host": "103.207.149.54", "port": 12785, "user": "root"},
    "B200": {"host": "38.80.152.146", "port": 31283, "user": "root"},
}
DEFAULT_REMOTE_REPO = "/root/fast-attnres-bf16-final"
DEFAULT_REMOTE_VENV = {
    "H100": "/root/fast-attnres-py213-sxm-v1",
    "B200": "/root/fast-attnres-py213-b200-v1",
}
DEFAULT_REMOTE_FLA = "/root/fast-attnres-vendors-v7/fla"
DEFAULT_REMOTE_LIGER = "/root/fast-attnres-vendors-v7/liger"
DEFAULT_REMOTE_CATSWE = "/root/fast-attnres-vendors-v7/catswe"
DEFAULT_REMOTE_OUTPUT_ROOT = "/root/attnres-compiled-step-sweep"
DEFAULT_CACHE_ROOT = "/root/.triton/cache-attnres-compiled-step-sweep"
DEFAULT_SEED = 20260827
DEFAULT_WARMUP = 5
DEFAULT_ROUNDS = 40
DEFAULT_BOOTSTRAP = 2000
LAYERS = 8
EVENTS = 2 * LAYERS
HEAD_DIM = 64
BATCH = 2
SEQUENCE = 512
VOCAB = 8192
WIDTHS = (1024, 1536, 2048, 3072, 4096)
BLOCK_SIZES = (8, 4, 2, 1)
RANK_LABELS = ("rd4", "rd")
MODEL_ONLY_STANDARD_WIDTHS = (2048, 4096)
HISTORICAL_D1024_LR_RANK = 128
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST_PATH = "validation/frozen.json"
KERNEL_PATHS = (
    "src/attnres/_kernels/fixed_tail.py",
    "src/attnres/_kernels/fixed_tail_sources.py",
    "src/attnres/_kernels/fla_full_sources.py",
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MIN_GPU_MEMORY_BYTES = {
    # Conservative SKU floors: real H100 SXM and B200 devices exceed these;
    # the floor prevents a forged tiny-memory runtime row from qualifying.
    "H100": 64 * 2**30,
    "B200": 128 * 2**30,
}
CATSWE_MODEL_SCHEDULE = (
    "native Catswe public phase-1 per-read aggregation for eligible "
    "Full/Block models; source lists are stacked and made contiguous "
    "inside the adapter call, with no cache/prepare/merge/phase2"
)


class SweepError(ValueError):
    """Raised for malformed matrix, transport, or worker data."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SweepError(message)


def _sha256_file(path: Path, label: str) -> str:
    """Hash one regular, non-symlink file for provenance attestation."""

    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    try:
        path.resolve().relative_to(path.parent.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise SweepError(f"{label} path is invalid: {path}") from exc
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SweepError(f"cannot read {label}: {path}: {exc}") from exc
    return digest.hexdigest()


def _git(root: Path, *args: str, allow_empty: bool = False) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SweepError(f"git is unavailable for project provenance: {exc}") from exc
    _require(
        completed.returncode == 0,
        f"git {' '.join(args)} failed for project provenance: {completed.stderr.strip()}",
    )
    value = completed.stdout.strip()
    _require(allow_empty or value, f"git {' '.join(args)} returned no project provenance")
    return value


def _project_provenance(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return a clean, payload-bound project identity for one sweep."""

    root = Path(root).expanduser().resolve()
    _require((root / "benchmarks" / "run.py").is_file(), "project checkout is missing benchmarks/run.py")
    revision = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    _require(_HEX40.fullmatch(revision) is not None, "project revision is not a full git object id")
    _require(_HEX40.fullmatch(tree) is not None, "project tree is not a full git object id")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all", allow_empty=True)
    _require(dirty == "", "project checkout is dirty; seal the sweep from a clean commit")
    frozen = root / FROZEN_MANIFEST_PATH
    kernels = {
        relative: _sha256_file(root / relative, f"project kernel {relative}")
        for relative in KERNEL_PATHS
    }
    frozen_digest = _sha256_file(frozen, "frozen manifest")
    _require(_HEX64.fullmatch(frozen_digest) is not None, "frozen manifest digest is malformed")
    return {
        "revision": revision,
        "tree": tree,
        "clean": True,
        "clean_required": True,
        "frozen_manifest": {
            "path": FROZEN_MANIFEST_PATH,
            "sha256": frozen_digest,
        },
        "kernel_sha256": kernels,
    }


def _catswe_provenance_contract() -> dict[str, Any]:
    """Return the exact pinned vendor identity required by the model arm."""

    from benchmarks.comparator_registry import capability_for

    capability = capability_for("catswe_phase1", scope="model")
    vendor_hashes = capability.get("vendor_file_sha256")
    _require(isinstance(vendor_hashes, Mapping) and vendor_hashes, "Catswe vendor hash contract is missing")
    return {
        "revision": capability["revision"],
        "tree": capability["tree"],
        "origin": capability["origin"],
        "license": capability["license"],
        "license_file": "LICENSE",
        "license_sha256": capability["license_sha256"],
        "source_hashes": dict(capability["source_hashes"]),
        "vendor_file_sha256": dict(vendor_hashes),
        "clean_required": True,
        "root_field": "remote_catswe_root",
    }


def _root_file(root: Path, relative: str, label: str) -> Path:
    """Resolve a required file while rejecting symlink escapes."""

    base = root.expanduser().resolve()
    path = base / relative
    _require(not path.is_symlink(), f"{label} is a symlink: {relative}")
    try:
        for parent in (path, *path.parents):
            if parent == base.parent:
                break
            if parent.is_symlink():
                raise SweepError(f"{label} has a symlink component: {relative}")
        path.resolve().relative_to(base)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SweepError(f"{label} escapes its root: {relative}") from exc
    return path


def _project_attestation(
    root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the worker checkout against the manifest before imports."""

    root = root.expanduser().resolve()
    # Worker payloads may carry the optional Catswe contract for an eligible
    # model arm, but this attestation covers only the project checkout.  The
    # separate Catswe attestation below owns vendor identity and bytes.
    expected = dict(expected)
    expected.pop("catswe", None)
    _require(root.is_dir(), f"remote project checkout is missing: {root}")
    _require(
        _root_file(root, "benchmarks/run.py", "project runner").is_file(),
        "remote project checkout is missing benchmarks/run.py",
    )
    top = Path(_git(root, "rev-parse", "--show-toplevel")).expanduser().resolve()
    _require(top == root, "remote project checkout is not the git top-level")
    revision = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all", allow_empty=True)
    clean = dirty == ""
    _require(
        set(expected)
        == {"revision", "tree", "clean", "clean_required", "frozen_manifest", "kernel_sha256"},
        "worker project provenance fields are not exact",
    )
    _require(expected.get("clean_required") is True, "worker project clean state is not required")
    _require(clean, "remote project checkout is dirty")
    _require(expected.get("clean") is True, "worker project provenance is not sealed clean")
    _require(revision == expected.get("revision"), "remote project revision differs from manifest")
    _require(tree == expected.get("tree"), "remote project tree differs from manifest")
    frozen_expected = _json_object(expected.get("frozen_manifest"), "expected frozen manifest")
    frozen_path = _root_file(root, str(frozen_expected.get("path")), "frozen manifest")
    frozen_sha = _sha256_file(frozen_path, "frozen manifest")
    _require(frozen_sha == frozen_expected.get("sha256"), "remote frozen manifest digest differs")
    kernel_expected = expected.get("kernel_sha256")
    _require(isinstance(kernel_expected, Mapping), "expected kernel hashes are missing")
    kernel_sha = {
        relative: _sha256_file(
            _root_file(root, relative, f"project kernel {relative}"),
            f"project kernel {relative}",
        )
        for relative in KERNEL_PATHS
    }
    _require(dict(kernel_expected) == kernel_sha, "remote project kernel hashes differ")
    return {
        "status": "verified",
        "revision": revision,
        "tree": tree,
        "clean": clean,
        "clean_required": True,
        "frozen_manifest": {
            "path": str(frozen_expected["path"]),
            "sha256": frozen_sha,
        },
        "kernel_sha256": kernel_sha,
    }


def _normalise_origin(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    result = value.strip().lower().rstrip("/")
    return result.removesuffix(".git")


def _catswe_attestation(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Verify an immutable pinned Catswe checkout or remote byte payload."""

    root = root.expanduser().resolve()
    _require(root.is_dir(), f"remote Catswe checkout is missing: {root}")
    expected_files = expected.get("vendor_file_sha256")
    _require(isinstance(expected_files, Mapping) and expected_files, "expected Catswe vendor hashes are missing")
    expected_sources = expected.get("source_hashes")
    _require(
        isinstance(expected_sources, Mapping)
        and set(expected_sources).issubset(set(expected_files))
        and all(
            isinstance(digest, str) and _HEX64.fullmatch(digest) is not None
            for digest in expected_sources.values()
        ),
        "expected Catswe source hashes are malformed",
    )
    expected_revision = expected.get("revision")
    expected_tree = expected.get("tree")
    expected_origin = expected.get("origin")
    manifest_path = root / "provenance.json"
    transport = "git_checkout"
    git_marker = root / ".git"
    _require(not git_marker.is_symlink(), "Catswe checkout .git marker is a symlink")
    if git_marker.exists():
        top = Path(_git(root, "rev-parse", "--show-toplevel")).expanduser().resolve()
        _require(top == root, "Catswe checkout is not the git top-level")
        revision = _git(root, "rev-parse", "HEAD")
        tree = _git(root, "rev-parse", "HEAD^{tree}")
        _require(revision == expected_revision, "Catswe revision differs from manifest")
        _require(tree == expected_tree, "Catswe tree differs from manifest")
        dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all", allow_empty=True)
        _require(dirty == "", "Catswe checkout is dirty")
        origin = _git(root, "config", "--get", "remote.origin.url")
        _require(
            _normalise_origin(origin) == _normalise_origin(expected_origin),
            "Catswe origin differs from manifest",
        )
    else:
        transport = "host_git_preflight+remote_bytes"
        _require(manifest_path.is_file() and not manifest_path.is_symlink(), "Catswe provenance manifest is missing")
        try:
            raw = manifest_path.read_bytes()
            manifest = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SweepError(f"Catswe provenance manifest is unreadable: {exc}") from exc
        _require(isinstance(manifest, Mapping), "Catswe provenance manifest must be an object")
        _require(
            set(manifest)
            == {
                "schema",
                "source_root",
                "vendor_revision",
                "vendor_tree",
                "vendor_origin",
                "host_git_preflight",
                "remote_git_present",
                "files",
            },
            "Catswe provenance manifest fields are not exact",
        )
        canonical = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode("utf-8")
        _require(raw == canonical, "Catswe provenance manifest encoding differs")
        _require(manifest.get("schema") == "catswe_remote_provenance_v1", "Catswe provenance schema differs")
        _require(manifest.get("source_root") == "src", "Catswe provenance source root differs")
        _require(manifest.get("host_git_preflight") is True, "Catswe host git preflight is missing")
        _require(manifest.get("remote_git_present") is False, "Catswe remote payload must not contain git metadata")
        revision = manifest.get("vendor_revision")
        tree = manifest.get("vendor_tree")
        origin = manifest.get("vendor_origin")
        _require(revision == expected_revision and tree == expected_tree, "Catswe remote revision/tree differs")
        _require(_normalise_origin(origin) == _normalise_origin(expected_origin), "Catswe remote origin differs")
        _require(manifest.get("files") == dict(expected_files), "Catswe remote file hash manifest differs")
        _require(not git_marker.exists(), "Catswe remote payload unexpectedly contains git metadata")
        expected_payload_files = set(expected_files) | {"provenance.json"}
        actual_payload_files: set[str] = set()
        for directory, directories, files in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            for name in directories:
                child = directory_path / name
                _require(not child.is_symlink(), f"Catswe remote payload has a symlink directory: {child.relative_to(root)}")
            for name in files:
                child = directory_path / name
                _require(not child.is_symlink(), f"Catswe remote payload has a symlink file: {child.relative_to(root)}")
                _require(child.is_file(), f"Catswe remote payload member is not a regular file: {child.relative_to(root)}")
                actual_payload_files.add(child.relative_to(root).as_posix())
        _require(
            actual_payload_files == expected_payload_files,
            "Catswe remote payload file set differs from the pinned manifest",
        )

    actual_files = {
        relative: _sha256_file(
            _root_file(root, relative, f"Catswe vendor file {relative}"),
            f"Catswe vendor file {relative}",
        )
        for relative in expected_files
    }
    _require(actual_files == dict(expected_files), "Catswe vendor file hashes differ")
    license_file = str(expected.get("license_file"))
    _require(license_file in actual_files, "Catswe license file is not covered by vendor hashes")
    _require(actual_files[license_file] == expected.get("license_sha256"), "Catswe license hash differs")
    return {
        "status": "verified",
        "transport": transport,
        "revision": revision,
        "tree": tree,
        "clean": True,
        "origin": origin,
        "license": expected.get("license"),
        "license_file": license_file,
        "license_sha256": actual_files[license_file],
        "source_hashes": {
            relative: actual_files[relative]
            for relative in expected_sources
        },
        "vendor_file_sha256": actual_files,
    }


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _json_object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _absolute_leaf(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent.resolve() / candidate.name


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Write one JSON object atomically, refusing a symlink target."""

    target = _absolute_leaf(path)
    _require(target.suffix.lower() == ".json", "JSON output must have a .json suffix")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _require(not target.is_symlink() and target.is_file(), f"output is not a regular file: {target}")
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
    return target


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SweepError(f"cannot read {label}: {path}: {exc}") from exc
    return _json_object(value, label)


def block_count_for_event_size(event_block_size: int) -> int:
    """Translate an event block size to the model's ``block_count`` field."""

    _require(type(event_block_size) is int and event_block_size in BLOCK_SIZES, "unsupported event block size")
    _require(EVENTS % event_block_size == 0, "event block size must divide 2*layers")
    return EVENTS // event_block_size


def _block_ends_for_events(source_events: int, block_count: int) -> tuple[int, ...]:
    """Mirror ``benchmarks.model._block_ends`` without importing torch."""

    _require(type(source_events) is int and source_events > 0, "source event count must be positive")
    _require(type(block_count) is int and block_count > 0, "block count must be positive")
    count = min(source_events, block_count)
    return tuple(math.ceil(source_events * i / count) for i in range(1, count + 1))


def read_source_counts_for_cell(*, mode: str, event_block_size: int | None) -> tuple[int, ...]:
    """Return the exact source count for every residual read in the cell.

    ``Full`` reads the embedding plus each completed event, giving counts
    ``2..2*L+1``.  ``Block`` follows the implementation's event loop: an
    event read sees completed block sources plus a currently accumulated
    partial source, and the terminal read sees only completed block sources.
    Keeping this sequence in the manifest prevents a headline ``Smax`` from
    hiding a different per-read schedule.
    """

    _require(mode in {"full", "block"}, "mode must be full or block")
    if mode == "full":
        _require(event_block_size is None, "Full cells do not have an event block size")
        return tuple(range(2, EVENTS + 2))
    _require(event_block_size in BLOCK_SIZES, "Block cells require an event block size")
    block_count = block_count_for_event_size(int(event_block_size))
    ends = _block_ends_for_events(EVENTS, block_count)
    completed_count = 1
    partial_exists = False
    previous_end = 0
    read_counts: list[int] = []
    for end in ends:
        for event_index in range(previous_end, end):
            if event_index != 0:
                read_counts.append(completed_count + int(partial_exists))
            partial_exists = True
        completed_count += 1
        partial_exists = False
        previous_end = end
    read_counts.append(completed_count)
    return tuple(read_counts)


def _source_count_histogram(read_counts: Sequence[int]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for source_count in read_counts:
        key = str(int(source_count))
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def max_sources_for_cell(*, mode: str, event_block_size: int | None) -> int:
    """Return the largest source count visible to one model read."""

    return max(read_source_counts_for_cell(mode=mode, event_block_size=event_block_size))


def lr_rank_for_width(width: int) -> int:
    """Return the fixed one-quarter rank used by the current LR screen."""

    _require(type(width) is int and width in WIDTHS, "unsupported sweep width")
    _require(width % 4 == 0, "LR rank requires a width divisible by four")
    return width // 4


def _model_only_admission_for_width(width: int) -> dict[str, Any] | None:
    """Return the sealed model-only admission needed for new standard widths."""

    if width not in MODEL_ONLY_STANDARD_WIDTHS:
        return None
    return {
        "enabled": True,
        "sealed": True,
        "scope": "model_only",
        "width_rank_pairs": [[width, width]],
        "purpose": "compiled-step standard R=D width screen; never an operator/protocol rank",
    }


def _cell_id(*, mode: str, width: int, rank: int, event_block_size: int | None) -> str:
    rank_label = "rd" if rank == width else "rd4"
    if mode == "full":
        return f"full-l{LAYERS}-d{width}-{rank_label}"
    return f"block-bs{event_block_size}-l{LAYERS}-d{width}-{rank_label}"


def _operator_comparator_metadata(
    *, mode: str, width: int, rank: int, source_count: int
) -> dict[str, dict[str, Any]]:
    """Describe model comparators using the same per-read geometry gates."""

    rank_is_full = rank == width
    lr_rank = lr_rank_for_width(width)
    liger_envelope = rank_is_full and source_count <= 32 and width <= 8192
    from benchmarks.comparator_registry import eligibility_for

    catswe_fields = {
        "mode": "full" if mode == "full" else "block_per_read",
        "rank": rank,
        "width": width,
        "dtype": "bf16",
        "timing": True,
        "source_count" if mode == "full" else "read_source_count": source_count,
    }
    catswe_decision = eligibility_for(
        "catswe_phase1", scope="model", **catswe_fields
    )
    operator_decision = eligibility_for(
        "catswe_phase1",
        scope="operator",
        mode="standard_operator",
        rank=rank,
        width=width,
        dtype="bf16",
        timing=True,
        source_count=source_count,
    )
    catswe_envelope = bool(catswe_decision.get("eligible", False))
    operator_envelope = bool(operator_decision.get("eligible", False))
    padded_sources = 1 << (source_count - 1).bit_length()
    return {
        "fla_triton_checkpoint1": {
            "status": "model_step_arm",
            "relation": (
                "R=D standard arm"
                if rank_is_full
                else f"LR R=D/4 (R={lr_rank}) is compared architecturally against the standard R=D arm"
            ),
            "mode": mode,
        },
        "liger": {
            "status": "model_step_arm" if liger_envelope else "not_applicable",
            "model_scope": "compiled_training_step" if liger_envelope else "not_applicable",
            "adapter": "benchmarks.liger",
            "operator_envelope": "R=D, S<=32, D<=8192",
            "source_list_timing": "torch.stack(...).contiguous() inside captured step",
            "reason": (
                "native Liger compiled model arm; source-list stack/contiguous is timed"
                if liger_envelope
                else "operator envelope rejects this cell: "
                f"R={rank}, D={width}, S={source_count}"
            ),
        },
        "catswe_phase1": {
            # Keep the operator capability visible as context, while routing
            # the actual model decision through comparator_registry's explicit
            # model scope.  The model arm is public phase 1 per read, never a
            # cached or operator-only substitution.
            "status": "model_step_arm" if catswe_envelope else "not_applicable",
            "model_scope": "compiled_training_step" if catswe_envelope else "not_applicable",
            "adapter": "benchmarks.catswe.make_model_backend",
            "operator_capability_scope": "standard_operator_only",
            "operator_eligible": operator_envelope,
            "operator_geometry_eligible": operator_envelope,
            "model_eligible": catswe_envelope,
            "operator_envelope": "BF16, standard operator, R=D, power-of-two D, nextpow2(S)*D<=1048576",
            "model_schedule": "public_phase1_per_read",
            "source_list_timing": "torch.stack(...).contiguous() inside captured step",
            "cache_prepare_merge_phase2": "forbidden",
            "reason": (
                "native Catswe public phase-1 model arm; source-list stack/contiguous is timed"
                if catswe_envelope
                else "Catswe model capability rejects this cell: "
                f"{catswe_decision.get('reason', f'R={rank}, D={width}, padded_S={padded_sources}')}"
            ),
            "operator_reason": operator_decision.get(
                "reason", "Catswe operator geometry is outside its declared scope"
            ),
        },
        "hydra_2p": {
            "status": "not_applicable",
            "reason": "native timing envelope is D<=256 and this sweep requires D>768; external panel is not a model-step route",
        },
    }


def make_cell(*, mode: str, width: int, rank: int, event_block_size: int | None) -> dict[str, Any]:
    """Build and validate one deterministic sweep cell."""

    _require(mode in {"full", "block"}, "invalid cell mode")
    _require(type(width) is int and width in WIDTHS and width > 768, "invalid sweep width")
    lr_rank = lr_rank_for_width(width)
    _require(
        type(rank) is int and rank in {lr_rank, width} and 1 <= rank <= width,
        f"invalid sweep rank: expected R=D/4 ({lr_rank}) or R=D ({width})",
    )
    _require(width % HEAD_DIM == 0, "sweep widths must be divisible by head_dim=64")
    read_source_counts = read_source_counts_for_cell(
        mode=mode, event_block_size=event_block_size
    )
    max_sources = max(read_source_counts)
    block_count = EVENTS if mode == "full" else block_count_for_event_size(int(event_block_size))
    cell_id = _cell_id(mode=mode, width=width, rank=rank, event_block_size=event_block_size)
    return {
        "cell_id": cell_id,
        "mode": mode,
        "layers": LAYERS,
        "width": width,
        "head_dim": HEAD_DIM,
        "heads": width // HEAD_DIM,
        "rank": rank,
        "rank_relation": "R=D" if rank == width else "R=D/4",
        "rank_ratio": "1" if rank == width else "1/4",
        "event_block_size": event_block_size,
        "block_count": block_count,
        "read_source_counts": list(read_source_counts),
        "read_source_count_histogram": _source_count_histogram(read_source_counts),
        "max_read_sources": max_sources,
        "dtype": "bf16_autocast",
        "source_layout": "list",
        "timing_method": "cuda_graph",
        "model_only_admission": (
            _model_only_admission_for_width(width)
            if width in MODEL_ONLY_STANDARD_WIDTHS
            else None
        ),
        "competitors": _operator_comparator_metadata(
            mode=mode, width=width, rank=rank, source_count=max_sources
        ),
    }


def build_matrix(
    *, widths: Sequence[int] = WIDTHS, block_sizes: Sequence[int] = BLOCK_SIZES
) -> list[dict[str, Any]]:
    """Return the 50-cell Full/Block × D × rank matrix.

    Each width contributes exactly two ranks: the standard ``R=D`` arm and
    the fixed LR arm ``R=D/4``.  Full contributes one cell per rank and Block
    contributes one cell per rank and event block size.  In particular, Block
    source counts are controlled by ``event_block_size`` while Full keeps the
    model's natural all-completed-source schedule.
    """

    selected_widths = tuple(widths)
    selected_blocks = tuple(block_sizes)
    _require(selected_widths, "matrix width set cannot be empty")
    _require(selected_blocks, "matrix block-size set cannot be empty")
    cells: list[dict[str, Any]] = []
    for width in selected_widths:
        _require(type(width) is int and width in WIDTHS, f"unsupported matrix width: {width!r}")
        ranks = (lr_rank_for_width(width), width)
        for rank in ranks:
            cells.append(make_cell(mode="full", width=width, rank=rank, event_block_size=None))
        for block_size in selected_blocks:
            for rank in ranks:
                cells.append(
                    make_cell(
                        mode="block",
                        width=width,
                        rank=rank,
                        event_block_size=block_size,
                    )
                )
    _require(len({cell["cell_id"] for cell in cells}) == len(cells), "matrix contains duplicate cell IDs")
    return cells


def make_model_config(
    cell: Mapping[str, Any], *, batch: int = BATCH, sequence: int = SEQUENCE, vocab: int = VOCAB
) -> dict[str, Any]:
    """Return the fixed surrounding Transformer profile for a cell."""

    width = int(cell["width"])
    _require(width % HEAD_DIM == 0, "sweep widths must be divisible by head_dim=64")
    _require(type(batch) is int and batch == BATCH, f"compact sweep requires batch={BATCH}")
    _require(type(sequence) is int and sequence == SEQUENCE, f"compact sweep requires sequence={SEQUENCE}")
    _require(type(vocab) is int and vocab == VOCAB, f"compact sweep requires vocab={VOCAB}")
    return {
        "batch": int(batch),
        "block_count": int(cell["block_count"]),
        "ffn": int(width * 11 // 4),
        "heads": width // HEAD_DIM,
        "layers": LAYERS,
        "mode": str(cell["mode"]),
        "sequence": int(sequence),
        "source_layout": "list",
        "variant": "sliced",
        "vocab": int(vocab),
        "width": width,
    }


def make_worker_config(
    cell: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    warmup: int = DEFAULT_WARMUP,
    rounds: int = DEFAULT_ROUNDS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP,
    batch: int = BATCH,
    sequence: int = SEQUENCE,
    vocab: int = VOCAB,
    remote_repo: str = DEFAULT_REMOTE_REPO,
    remote_fla_root: str = DEFAULT_REMOTE_FLA,
    remote_liger_root: str = DEFAULT_REMOTE_LIGER,
    remote_catswe_root: str = DEFAULT_REMOTE_CATSWE,
) -> dict[str, Any]:
    """Build the unsealed custom config consumed by ``run_suite``.

    The worker measures the cell's one candidate rank.  The configured FLA
    standard comparison is a separate R=D arm created by ``run_suite``; it is
    deliberately not added to ``ranks`` here.
    """

    _require(type(seed) is int and seed > 0, "seed must be a positive integer")
    _require(type(warmup) is int and warmup >= 0, "warmup must be a non-negative integer")
    _require(type(rounds) is int and rounds > 0, "rounds must be a positive integer")
    _require(type(bootstrap_samples) is int and bootstrap_samples > 0, "bootstrap_samples must be positive")
    model_config = make_model_config(cell, batch=batch, sequence=sequence, vocab=vocab)
    config = {
        "scope": "custom",
        "phases": ["model"],
        "device": "cuda:0",
        "seed": seed,
        "model_config": model_config,
        "mode": cell["mode"],
        "variant": "sliced",
        # One worker cell measures exactly the rank named by its cell.  The
        # standard_fla_comparison option constructs the separate R=D FLA arm,
        # so putting both candidate ranks in every job would duplicate work
        # and make the cell's rank label ambiguous.
        "ranks": [int(cell["rank"])],
        "model_timing": "cuda_graph",
        "model_warmup": warmup,
        "model_rounds": rounds,
        "bootstrap_samples": bootstrap_samples,
        "pairwise": False,
        "reference_timing": False,
        "include_baseline": False,
        "include_packed_comparison": False,
        # The complete-step FLA arm is created by ``include_fla_compile``.
        # Keep the optional operator-discovery registry disabled here so the
        # screen does not import Gluon or any unrelated comparator route.
        "include_fla": False,
        "include_fla_compile": True,
        "fla_compile_backends": ["triton"],
        "standard_fla_comparison": True,
        "include_fla_model": False,
        # Liger is a native compiled model arm for standard R=D cells whose
        # actual read schedule satisfies S<=32.  Its adapter's source-list
        # stack/contiguous work stays inside the captured step.
        "include_liger_model": True,
        "liger_root": str(remote_liger_root),
        # Catswe discovery is enabled only for cells admitted by the shared
        # model-scope capability metadata.  Ineligible LR/non-power-of-two
        # cells remain explicit model NA rows and must not import the vendor.
        "include_catswe_model": bool(
            cell["competitors"]["catswe_phase1"].get("model_eligible", False)
        ),
        "catswe_root": str(remote_catswe_root),
        "model_state_protocol": "canonical_implicit_max_rank_v1",
        "model_progress": True,
        "model_profile": False,
        "project_root": str(remote_repo),
        "vendor_root": str(remote_fla_root),
        "sweep_cell": copy.deepcopy(dict(cell)),
        "sweep_timing_contract": {
            "event": "one CapturedTrainingStep.replay CUDA event",
            "inside": [
                "zero_grad",
                "BF16 autocast model forward",
                "cross_entropy",
                "backward and gradient accumulation",
                "capturable AdamW step",
            ],
            "outside": [
                "input copies",
                "tensor hashing",
                "torch.compile",
                "qualification",
                "warmup",
                "graph capture",
                "report serialization",
            ],
            "timed_tensor_hashing": False,
            "timed_input_copy": False,
            "per_round_numerical_checks": False,
        },
    }
    model_only_admission = _model_only_admission_for_width(int(cell["width"]))
    if model_only_admission is not None:
        # D=2048 and D=4096 are absent from the frozen protocol rank ladder.
        # This explicitly admits only their standard model rows; it does not
        # expand the operator protocol or LR rank set.
        config["model_only_admission"] = model_only_admission
    return config


def make_manifest(
    *,
    seed: int = DEFAULT_SEED,
    warmup: int = DEFAULT_WARMUP,
    rounds: int = DEFAULT_ROUNDS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP,
    batch: int = BATCH,
    sequence: int = SEQUENCE,
    vocab: int = VOCAB,
    widths: Sequence[int] = WIDTHS,
    block_sizes: Sequence[int] = BLOCK_SIZES,
    remote_repo: str = DEFAULT_REMOTE_REPO,
    remote_fla_root: str = DEFAULT_REMOTE_FLA,
    remote_liger_root: str = DEFAULT_REMOTE_LIGER,
    remote_catswe_root: str = DEFAULT_REMOTE_CATSWE,
    remote_output_root: str = DEFAULT_REMOTE_OUTPUT_ROOT,
    cache_root: str = DEFAULT_CACHE_ROOT,
) -> dict[str, Any]:
    _require(type(batch) is int and batch == BATCH, f"compact sweep requires batch={BATCH}")
    _require(type(sequence) is int and sequence == SEQUENCE, f"compact sweep requires sequence={SEQUENCE}")
    _require(type(vocab) is int and vocab == VOCAB, f"compact sweep requires vocab={VOCAB}")
    _require(tuple(widths) == WIDTHS, "the sealed compact matrix requires the default width set")
    _require(tuple(block_sizes) == BLOCK_SIZES, "the sealed compact matrix requires the default block-size set")
    launch = {
        "remote_repo": remote_repo,
        "remote_fla_root": remote_fla_root,
        "remote_liger_root": remote_liger_root,
        "remote_catswe_root": remote_catswe_root,
        "remote_output_root": remote_output_root,
        "cache_root": cache_root,
    }
    _require(
        all(isinstance(value, str) and value for value in launch.values()),
        "sweep launch paths must be non-empty strings",
    )
    cells = build_matrix(widths=widths, block_sizes=block_sizes)
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "planned",
        "scope": "BF16 compiled complete training step screen",
        "runtime": {"torch": "2.13.0+cu130", "cuda": "13.0", "triton": "3.7.1"},
        "fixed_profile": {
            "layers": LAYERS,
            "batch": batch,
            "sequence": sequence,
            "vocab": vocab,
            "head_dim": HEAD_DIM,
            "heads_formula": "D/64",
            "ffn_formula": "11*D/4",
            "source_layout": "list",
            "timing_method": "cuda_graph",
            "dtype": "bf16_autocast",
        },
        "seed": seed,
        "warmup": warmup,
        "rounds": rounds,
        "bootstrap_samples": bootstrap_samples,
        "source_count_control": {
            "definition": "Block event block size is Transformer sublayer events per block",
            "events_at_L8": EVENTS,
            "block_size_to_block_count": {
                str(size): block_count_for_event_size(size) for size in block_sizes
            },
            "block_size_to_max_read_S": {
                str(size): max_sources_for_cell(mode="block", event_block_size=size)
                for size in block_sizes
            },
            "block_size_to_read_source_counts": {
                str(size): list(
                    read_source_counts_for_cell(mode="block", event_block_size=size)
                )
                for size in block_sizes
            },
            "full_read_source_counts": list(
                read_source_counts_for_cell(mode="full", event_block_size=None)
            ),
            "full_max_read_S": max_sources_for_cell(mode="full", event_block_size=None),
        },
        "launch": launch,
        "ranks": ["D/4", "D"],
        "model_only_admission": {
            "scope": "model_only",
            "standard_widths": list(MODEL_ONLY_STANDARD_WIDTHS),
            "pairs": [[width, width] for width in MODEL_ONLY_STANDARD_WIDTHS],
            "reason": "2048/4096 are absent from the frozen operator rank ladder",
        },
        "project_provenance": {
            **_project_provenance(),
            "catswe": _catswe_provenance_contract(),
        },
        "cells": cells,
        "model_comparator_contract": {
            "liger": {
                "status": "native_model_arm_when_eligible",
                "scope": "compiled_training_step",
                "capability": "BF16 autocast; R=D; every actual read S<=32; D<=8192",
                "source_list_boundary": "torch.stack(...).contiguous() is inside the captured step",
                "root_field": "remote_liger_root",
            },
            "catswe_phase1": {
                "status": "native_model_arm_when_eligible",
                "scope": "compiled_training_step",
                "capability": "BF16; R=D; each actual read S<=129; power-of-two D<=8192; nextpow2(S)*D<=1048576",
                "execution": "public phase1 per actual Full/Block read",
                "source_list_boundary": "torch.stack(...).contiguous() inside the captured step",
                "cache_prepare_merge_phase2": "forbidden",
                "root_field": "remote_catswe_root",
            },
        },
        "comparison_notes": {
            "fla": "FLA Triton checkpoint-1 is the only complete model-step comparator; R=D/4 is an LR candidate versus standard FLA R=D and must be labeled cross-equation.",
            "liger": "Native compiled model-step arm when every actual read has S<=32 and R=D; source-list stack/contiguous cost remains inside the captured step.",
            "catswe": "Explicit opt-in native compiled model-step arm when every actual read satisfies the public phase-1 envelope; stack/contiguous cost remains inside the captured call and no cache/prepare/merge/phase2 fallback is permitted.",
            "hydra": "D>768 is outside native timing envelope; external two-phase panel is not a model-step comparator.",
            "historical": "The former D=1024,R=128 screen is retained only as historical context and is excluded from this matrix.",
        },
    }


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact screen plan before any remote process is started."""

    value = _json_object(manifest, "sweep manifest")
    required = {
        "schema", "status", "scope", "runtime", "fixed_profile", "seed", "warmup",
        "rounds", "bootstrap_samples", "source_count_control", "ranks", "cells",
        "model_only_admission", "project_provenance", "model_comparator_contract", "comparison_notes", "launch",
    }
    _require(set(value) == required, "sweep manifest fields are not exact")
    _require(value["schema"] == MANIFEST_SCHEMA and value["status"] == "planned", "sweep manifest schema/status differs")
    _require(value["runtime"] == {"torch": "2.13.0+cu130", "cuda": "13.0", "triton": "3.7.1"}, "sweep runtime differs")
    launch = _json_object(value["launch"], "sweep launch binding")
    _require(
        set(launch)
        == {
            "remote_repo", "remote_fla_root", "remote_liger_root", "remote_catswe_root",
            "remote_output_root", "cache_root",
        }
        and all(isinstance(path, str) and path for path in launch.values()),
        "sweep launch binding is malformed",
    )
    project = _json_object(value["project_provenance"], "sweep project provenance")
    _require(
        set(project) == {"revision", "tree", "clean", "clean_required", "frozen_manifest", "kernel_sha256", "catswe"},
        "sweep project provenance fields are not exact",
    )
    expected_project = _project_provenance()
    _require(
        _same(
            {key: project[key] for key in expected_project},
            expected_project,
        ),
        "sweep project revision/tree/frozen/kernel provenance differs from this checkout",
    )
    _require(project["clean_required"] is True, "sweep project clean state must be required")
    frozen = _json_object(project["frozen_manifest"], "sweep frozen manifest provenance")
    _require(
        set(frozen) == {"path", "sha256"}
        and frozen["path"] == FROZEN_MANIFEST_PATH
        and _HEX64.fullmatch(frozen["sha256"]) is not None,
        "sweep frozen manifest provenance is malformed",
    )
    kernels = project["kernel_sha256"]
    _require(
        isinstance(kernels, Mapping)
        and set(kernels) == set(KERNEL_PATHS)
        and all(isinstance(digest, str) and _HEX64.fullmatch(digest) for digest in kernels.values()),
        "sweep kernel provenance is malformed",
    )
    _require(
        _same(project["catswe"], _catswe_provenance_contract()),
        "sweep Catswe provenance contract differs",
    )
    _require(
        value["fixed_profile"] == {
            "layers": LAYERS,
            "batch": BATCH,
            "sequence": SEQUENCE,
            "vocab": VOCAB,
            "head_dim": HEAD_DIM,
            "heads_formula": "D/64",
            "ffn_formula": "11*D/4",
            "source_layout": "list",
            "timing_method": "cuda_graph",
            "dtype": "bf16_autocast",
        },
        "sweep fixed profile differs",
    )
    _require(type(value["seed"]) is int and value["seed"] > 0, "sweep seed is invalid")
    _require(type(value["warmup"]) is int and value["warmup"] >= 0, "sweep warmup is invalid")
    _require(type(value["rounds"]) is int and value["rounds"] > 0, "sweep rounds is invalid")
    _require(type(value["bootstrap_samples"]) is int and value["bootstrap_samples"] > 0, "sweep bootstrap count is invalid")
    _require(value["ranks"] == ["D/4", "D"], "sweep rank matrix differs")
    _require(
        value["model_only_admission"] == {
            "scope": "model_only",
            "standard_widths": list(MODEL_ONLY_STANDARD_WIDTHS),
            "pairs": [[width, width] for width in MODEL_ONLY_STANDARD_WIDTHS],
            "reason": "2048/4096 are absent from the frozen operator rank ladder",
        },
        "sweep model-only admission differs",
    )
    _require(
        value["model_comparator_contract"] == {
            "liger": {
                "status": "native_model_arm_when_eligible",
                "scope": "compiled_training_step",
                "capability": "BF16 autocast; R=D; every actual read S<=32; D<=8192",
                "source_list_boundary": "torch.stack(...).contiguous() is inside the captured step",
                "root_field": "remote_liger_root",
            },
            "catswe_phase1": {
                "status": "native_model_arm_when_eligible",
                "scope": "compiled_training_step",
                "capability": "BF16; R=D; each actual read S<=129; power-of-two D<=8192; nextpow2(S)*D<=1048576",
                "execution": "public phase1 per actual Full/Block read",
                "source_list_boundary": "torch.stack(...).contiguous() inside the captured step",
                "cache_prepare_merge_phase2": "forbidden",
                "root_field": "remote_catswe_root",
            },
        },
        "sweep model comparator contract differs",
    )
    _require(
        value["source_count_control"] == {
            "definition": "Block event block size is Transformer sublayer events per block",
            "events_at_L8": EVENTS,
            "block_size_to_block_count": {
                str(size): block_count_for_event_size(size) for size in BLOCK_SIZES
            },
            "block_size_to_max_read_S": {
                str(size): max_sources_for_cell(mode="block", event_block_size=size)
                for size in BLOCK_SIZES
            },
            "block_size_to_read_source_counts": {
                str(size): list(
                    read_source_counts_for_cell(mode="block", event_block_size=size)
                )
                for size in BLOCK_SIZES
            },
            "full_read_source_counts": list(
                read_source_counts_for_cell(mode="full", event_block_size=None)
            ),
            "full_max_read_S": max_sources_for_cell(mode="full", event_block_size=None),
        },
        "sweep source-count control differs",
    )
    _require(
        value["comparison_notes"] == {
            "fla": "FLA Triton checkpoint-1 is the only complete model-step comparator; R=D/4 is an LR candidate versus standard FLA R=D and must be labeled cross-equation.",
            "liger": "Native compiled model-step arm when every actual read has S<=32 and R=D; source-list stack/contiguous cost remains inside the captured step.",
            "catswe": "Explicit opt-in native compiled model-step arm when every actual read satisfies the public phase-1 envelope; stack/contiguous cost remains inside the captured call and no cache/prepare/merge/phase2 fallback is permitted.",
            "hydra": "D>768 is outside native timing envelope; external two-phase panel is not a model-step comparator.",
            "historical": "The former D=1024,R=128 screen is retained only as historical context and is excluded from this matrix.",
        },
        "sweep comparison notes differ",
    )
    cells = value["cells"]
    _require(isinstance(cells, list) and len(cells) == 50, "sweep must contain exactly 50 cells")
    for cell in cells:
        _validate_cell(cell)
    _require(len({cell["cell_id"] for cell in cells}) == len(cells), "sweep manifest has duplicate cells")
    return value


def _validate_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_object(cell, "cell")
    required = {
        "cell_id", "mode", "layers", "width", "rank", "rank_relation",
        "head_dim", "heads", "rank_ratio", "event_block_size", "block_count",
        "read_source_counts", "read_source_count_histogram", "max_read_sources",
        "dtype", "source_layout", "timing_method", "model_only_admission",
        "competitors",
    }
    _require(set(value) == required, "cell fields are not exact")
    expected = make_cell(
        mode=value["mode"],
        width=value["width"],
        rank=value["rank"],
        event_block_size=value["event_block_size"],
    )
    _require(_same(value, expected), f"cell metadata differs from derived matrix: {value.get('cell_id')!r}")
    return value


def _validate_worker_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_object(payload, "worker payload")
    required = {
        "schema", "gpu", "cell", "config", "remote_repo", "remote_fla_root",
        "remote_liger_root", "remote_catswe_root", "triton_cache_dir", "run_id",
        "project_provenance", "seed", "warmup", "rounds", "bootstrap_samples",
        "batch", "sequence", "vocab",
    }
    _require(set(value) == required, "worker payload fields are not exact")
    _require(value["schema"] == SCHEMA, "worker payload schema differs")
    _require(value["gpu"] in SUPPORTED_GPUS, "worker payload GPU is unsupported")
    cell = _validate_cell(value["cell"])
    for key in ("remote_repo", "remote_fla_root", "remote_liger_root", "remote_catswe_root", "triton_cache_dir", "run_id"):
        _require(isinstance(value[key], str) and value[key], f"worker payload {key} is required")
    _require(type(value["seed"]) is int and value["seed"] > 0, "worker payload seed is invalid")
    _require(type(value["warmup"]) is int and value["warmup"] >= 0, "worker payload warmup is invalid")
    _require(type(value["rounds"]) is int and value["rounds"] > 0, "worker payload rounds are invalid")
    _require(
        type(value["bootstrap_samples"]) is int and value["bootstrap_samples"] > 0,
        "worker payload bootstrap_samples is invalid",
    )
    _require(type(value["batch"]) is int and value["batch"] == BATCH, f"worker payload batch must be {BATCH}")
    _require(
        type(value["sequence"]) is int and value["sequence"] == SEQUENCE,
        f"worker payload sequence must be {SEQUENCE}",
    )
    _require(type(value["vocab"]) is int and value["vocab"] == VOCAB, f"worker payload vocab must be {VOCAB}")
    config = _json_object(value["config"], "worker config")
    _require(_same(config.get("sweep_cell"), cell), "worker config/cell metadata differs")
    _require(config.get("include_liger_model") is True, "worker must request the native Liger model arm")
    expected_catswe_model = bool(
        cell["competitors"]["catswe_phase1"].get("model_eligible", False)
    )
    _require(
        config.get("include_catswe_model") is expected_catswe_model,
        "worker Catswe model opt-in does not match the cell's shared model eligibility",
    )
    _require(config.get("liger_root") == value["remote_liger_root"], "worker Liger root differs from payload root")
    _require(config.get("catswe_root") == value["remote_catswe_root"], "worker Catswe root differs from payload root")
    expected_config = make_worker_config(
        cell,
        seed=value["seed"],
        warmup=value["warmup"],
        rounds=value["rounds"],
        bootstrap_samples=value["bootstrap_samples"],
        batch=value["batch"],
        sequence=value["sequence"],
        vocab=value["vocab"],
        remote_repo=value["remote_repo"],
        remote_fla_root=value["remote_fla_root"],
        remote_liger_root=value["remote_liger_root"],
        remote_catswe_root=value["remote_catswe_root"],
    )
    _require(
        _same(config, expected_config),
        "worker config differs from the payload-bound expected config",
    )
    include_catswe = bool(config["include_catswe_model"])
    _validate_worker_project_provenance(
        value["project_provenance"], catswe_required=include_catswe
    )
    return {**value, "cell": cell, "config": config}


def _validate_worker_project_provenance(
    project_value: Any, *, catswe_required: bool
) -> dict[str, Any]:
    """Validate worker project identity, with Catswe gated by the model arm.

    The sweep manifest carries the pinned Catswe contract globally, but an
    ineligible worker does not import or use that vendor.  Such a payload may
    omit the optional ``catswe`` metadata entirely (legacy callers may still
    carry it); only eligible payloads require and compare that contract.
    """

    project = _json_object(project_value, "worker project provenance")
    base_fields = {
        "revision", "tree", "clean", "clean_required", "frozen_manifest", "kernel_sha256"
    }
    fields_with_catswe = base_fields | {"catswe"}
    if catswe_required:
        _require(
            set(project) == fields_with_catswe,
            "worker project provenance fields are not exact",
        )
    else:
        _require(
            set(project) == base_fields or set(project) == fields_with_catswe,
            "worker project provenance fields are not exact",
        )
    _require(
        isinstance(project["revision"], str)
        and _HEX40.fullmatch(project["revision"]) is not None
        and isinstance(project["tree"], str)
        and _HEX40.fullmatch(project["tree"]) is not None
        and project["clean"] is True
        and project["clean_required"] is True,
        "worker project revision/tree/clean contract is malformed",
    )
    frozen = _json_object(project["frozen_manifest"], "worker frozen manifest provenance")
    _require(
        set(frozen) == {"path", "sha256"}
        and frozen["path"] == FROZEN_MANIFEST_PATH
        and isinstance(frozen["sha256"], str)
        and _HEX64.fullmatch(frozen["sha256"]) is not None,
        "worker frozen manifest contract is malformed",
    )
    kernels = project["kernel_sha256"]
    _require(
        isinstance(kernels, Mapping)
        and set(kernels) == set(KERNEL_PATHS)
        and all(
            isinstance(digest, str) and _HEX64.fullmatch(digest) is not None
            for digest in kernels.values()
        ),
        "worker kernel hash contract is malformed",
    )
    expected_project = _project_provenance()
    _require(
        _same(
            {key: project[key] for key in expected_project},
            expected_project,
        ),
        "worker project provenance differs from this checkout",
    )
    if catswe_required:
        _require(
            _same(project["catswe"], _catswe_provenance_contract()),
            "worker Catswe provenance contract differs from the pinned registry",
        )
    return project


def _worker_result_routes(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the route switches that a worker result must echo exactly."""

    return {
        "phases": copy.deepcopy(config["phases"]),
        "fla": {
            "include": config["include_fla"],
            "compile": config["include_fla_compile"],
            "compile_backends": copy.deepcopy(config["fla_compile_backends"]),
            "standard_comparison": config["standard_fla_comparison"],
            "model": config["include_fla_model"],
        },
        "liger": {"include": config["include_liger_model"]},
        "catswe_phase1": {"include": config["include_catswe_model"]},
        "model": {
            "timing": config["model_timing"],
            "mode": config["mode"],
            "variant": config["variant"],
            "source_layout": config["model_config"]["source_layout"],
        },
    }


def _worker_result_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable payload identity copied into every worker result."""

    config = _json_object(payload["config"], "worker config")
    cell = _json_object(payload["cell"], "worker cell")
    roots = {
        key: payload[key]
        for key in (
            "remote_repo",
            "remote_fla_root",
            "remote_liger_root",
            "remote_catswe_root",
            "triton_cache_dir",
        )
    }
    run_parameters = {
        key: payload[key]
        for key in (
            "seed",
            "warmup",
            "rounds",
            "bootstrap_samples",
            "batch",
            "sequence",
            "vocab",
        )
    }
    return {
        "config": copy.deepcopy(config),
        "project_provenance": copy.deepcopy(dict(payload["project_provenance"])),
        "roots": roots,
        "run_parameters": run_parameters,
        "routes": _worker_result_routes(config),
        "eligibility": copy.deepcopy(dict(cell["competitors"])),
        "timing_contract": copy.deepcopy(dict(config["sweep_timing_contract"])),
    }


def _json_sha256(value: Any, label: str) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise SweepError(f"{label} is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _report_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize report identity using only stable, non-tensor metadata."""

    report_config = _json_object(report.get("config"), "worker benchmark config")
    model = _json_object(report.get("model_timings"), "worker benchmark model timings")
    comparators = _json_object(report.get("comparators"), "worker benchmark comparators")
    for key in ("include_liger_model", "include_catswe_model"):
        flag = report.get(key, False)
        _require(type(flag) is bool, f"worker benchmark {key} must be boolean")
    return {
        "status": report.get("status"),
        "scope": report_config.get("scope"),
        "phases": copy.deepcopy(report_config.get("phases")),
        "cell_id": _json_object(report_config.get("sweep_cell"), "worker benchmark sweep cell").get("cell_id"),
        "config_sha256": _json_sha256(report_config, "worker benchmark config"),
        "model_status": model.get("status"),
        "timing_method": model.get("timing_method"),
        "comparators": sorted(comparators),
        "include_liger_model": bool(report.get("include_liger_model", False)),
        "include_catswe_model": bool(report.get("include_catswe_model", False)),
    }


def _unavailable_report_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "cell_id": payload["cell"]["cell_id"],
        "config_sha256": _json_sha256(payload["config"], "worker config"),
    }


def _validate_worker_runtime(runtime: Any, gpu: str, *, allow_not_passed: bool) -> dict[str, Any]:
    value = _json_object(runtime, "worker runtime preflight")
    status = value.get("status")
    if status == "not_passed":
        _require(allow_not_passed and set(value) == {"status"}, "worker runtime preflight is not exact")
        return value
    _require(
        set(value)
        == {"status", "gpu", "name", "compute_capability", "total_memory", "torch", "cuda", "triton"},
        "worker runtime preflight fields are not exact",
    )
    _require(status == "passed", "worker runtime preflight did not pass")
    _require(value["gpu"] == gpu, "worker runtime GPU differs from payload")
    _require(isinstance(value["name"], str) and value["name"], "worker runtime GPU name is missing")
    expected_name = gpu
    _require(expected_name in value["name"], "worker runtime GPU name does not match payload GPU")
    capability = value["compute_capability"]
    _require(
        isinstance(capability, list)
        and len(capability) == 2
        and all(type(component) is int and component >= 0 for component in capability),
        "worker runtime compute capability is malformed",
    )
    expected_capability = [9, 0] if gpu == "H100" else [10, 0]
    _require(capability == expected_capability, "worker runtime compute capability does not match payload GPU")
    _require(
        type(value["total_memory"]) is int
        and value["total_memory"] >= _MIN_GPU_MEMORY_BYTES[gpu],
        "worker runtime memory is below the conservative payload GPU floor",
    )
    _require(value["torch"] == "2.13.0+cu130", "worker runtime Torch version differs")
    _require(value["cuda"] == "13.0", "worker runtime CUDA version differs")
    _require(value["triton"] == "3.7.1", "worker runtime Triton version differs")
    return value


def _validate_worker_provenance(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    required: bool,
    catswe_required: bool | None = None,
) -> dict[str, Any] | None:
    if actual is None:
        _require(not required, "complete worker result is missing provenance")
        return None
    value = _json_object(actual, "worker result provenance")
    if catswe_required is None:
        catswe_required = "catswe" in expected
    expected_fields = {"project", "catswe"} if catswe_required else {"project"}
    _require(set(value) == expected_fields, "worker result provenance fields are not exact")
    project = _json_object(value["project"], "worker result project attestation")
    expected_project = dict(expected)
    catswe_expected = expected_project.pop("catswe", None)
    expected_project = {"status": "verified", **expected_project}
    _require(_same(project, expected_project), "worker result project attestation differs")
    if catswe_required:
        catswe_expected = _json_object(catswe_expected, "expected Catswe provenance")
        catswe = _json_object(value["catswe"], "worker result Catswe attestation")
        _require(
            set(catswe)
            == {
                "status", "transport", "revision", "tree", "clean", "origin", "license",
                "license_file", "license_sha256", "source_hashes", "vendor_file_sha256",
            },
            "worker result Catswe attestation fields are not exact",
        )
        _require(catswe["status"] == "verified", "worker result Catswe attestation did not pass")
        _require(
            catswe["transport"] in {"git_checkout", "host_git_preflight+remote_bytes"},
            "worker result Catswe transport is invalid",
        )
        _require(catswe["revision"] == catswe_expected["revision"], "worker result Catswe revision differs")
        _require(catswe["tree"] == catswe_expected["tree"], "worker result Catswe tree differs")
        _require(
            catswe["clean"] is True and catswe_expected["clean_required"] is True,
            "worker result Catswe clean contract is invalid",
        )
        _require(
            _normalise_origin(catswe["origin"]) == _normalise_origin(catswe_expected["origin"]),
            "worker result Catswe origin differs",
        )
        _require(catswe["license"] == catswe_expected["license"], "worker result Catswe license differs")
        _require(
            catswe["license_file"] == catswe_expected["license_file"],
            "worker result Catswe license file differs",
        )
        _require(
            catswe["license_sha256"] == catswe_expected["license_sha256"],
            "worker result Catswe license hash differs",
        )
        _require(
            _same(catswe["source_hashes"], catswe_expected["source_hashes"]),
            "worker result Catswe source hashes differ",
        )
        _require(
            _same(catswe["vendor_file_sha256"], catswe_expected["vendor_file_sha256"]),
            "worker result Catswe vendor hashes differ",
        )
    return value


def _validate_worker_model_report(report: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the returned model report's identity and comparator routes."""

    config = _json_object(report.get("config"), "worker benchmark config")
    _require(_same(config, payload["config"]), "worker benchmark config differs from payload")
    _require(report.get("comparators_enabled") is True, "worker benchmark comparator route is disabled")
    comparators = _json_object(report.get("comparators"), "worker benchmark comparators")
    expected_comparators = {"liger"}
    include_catswe = bool(payload["config"]["include_catswe_model"])
    if include_catswe:
        expected_comparators.add("catswe_phase1")
    _require(set(comparators) == expected_comparators, "worker benchmark comparator routes differ")
    coverage = _json_object(report.get("coverage"), "worker benchmark coverage")
    _require(coverage.get("scope") == "custom", "worker benchmark scope differs")
    model = _json_object(report.get("model_timings"), "worker benchmark model timings")
    # ``run_suite`` emits the optional model-arm switches in ``coverage`` and
    # ``model_timings``.  They are deliberately absent at the benchmark
    # report's top level.  The old validator checked the latter as if it were
    # a producer field, so every genuine ``run_worker`` result failed offline
    # validation while a forged top-level flag could look authoritative.
    for key, expected in (("include_liger_model", True), ("include_catswe_model", include_catswe)):
        if expected:
            _require(coverage.get(key) is True, f"worker benchmark coverage {key} route is missing")
            _require(model.get(key) is True, f"worker benchmark model {key} route is missing")
        else:
            _require(key not in coverage, f"worker benchmark coverage {key} route was unexpectedly enabled")
            _require(model.get(key) is False, f"worker benchmark model {key} route was unexpectedly enabled")
        _require(key not in report, f"worker benchmark top-level {key} route field is not producer output")
    _require(model.get("status") == "complete", "worker model timings are incomplete")
    _require(model.get("failures") == [] and model.get("comparator_failures") == [], "worker model report contains failures")
    _require(model.get("timing_method") == payload["config"]["model_timing"], "worker model timing method differs")
    _require(model.get("training_step") == "benchmarks.training_graph.CapturedTrainingStep.replay", "worker training step route differs")
    _require(model.get("requested_warmup") == payload["config"]["model_warmup"], "worker warmup differs")
    _require(model.get("requested_rounds") == payload["config"]["model_rounds"], "worker rounds differ")
    timed_identity = _json_object(model.get("timed_input_identity"), "worker timed input identity")
    _require(timed_identity.get("tensor_byte_hashing") is False, "worker timed tensor hashing is enabled")
    _require(timed_identity.get("device_to_host_copy") is False, "worker timed device-to-host copying is enabled")
    timing_boundary = _json_object(model.get("timing_boundary"), "worker model timing boundary")
    _require("AdamW optimizer.step" in timing_boundary.get("steady_step_includes", []), "worker timing omits optimizer step")
    _require(timing_boundary.get("backward_orchestration") == "captured complete step including optimizer update", "worker timing boundary differs")
    schedules = _json_object(model.get("execution_schedules"), "worker model execution schedules")
    _require("liger" in schedules, "worker Liger schedule is missing")
    liger_arm = f"liger_rank_{payload['cell']['rank']}"
    model_scope = _json_object(model.get("model_comparator_scope"), "worker model comparator eligibility")
    expected_liger = payload["cell"]["competitors"]["liger"]["status"] == "model_step_arm"
    liger_row = _json_object(model_scope.get(liger_arm), "worker Liger eligibility")
    _require(liger_row.get("eligible") is expected_liger, "worker Liger eligibility differs")
    if include_catswe:
        _require("catswe_phase1" in comparators and "catswe_phase1" in schedules, "worker Catswe schedule is missing")
        _require(
            schedules["catswe_phase1"] == CATSWE_MODEL_SCHEDULE,
            "worker Catswe schedule differs from the canonical public phase1 route",
        )
        catswe_arm = f"catswe_phase1_model_rank_{payload['cell']['rank']}"
        catswe_row = _json_object(model_scope.get(catswe_arm), "worker Catswe eligibility")
        _require(catswe_row.get("eligible") is True, "worker Catswe eligibility is not true")
        _require(catswe_row.get("capability_scope") == "model", "worker Catswe capability scope differs")
        _require(catswe_row.get("model_scope") == "compiled_training_step", "worker Catswe model scope differs")
    else:
        _require("catswe_phase1" not in schedules, "worker Catswe schedule appeared on an ineligible cell")
        _require("catswe_phase1" not in model_scope and not any(str(name).startswith("catswe_phase1_") for name in model_scope), "worker Catswe eligibility appeared on an ineligible cell")
    return dict(report)


def _validate_worker_result(result: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one remote result before the launcher promotes it."""

    payload_value = _validate_worker_payload(payload)
    value = _json_object(result, "worker result")
    required = {
        "schema", "status", "gpu", "cell", "config", "project_provenance", "roots",
        "run_parameters", "routes", "eligibility", "timing_contract", "runtime_preflight",
        "provenance", "report_identity", "worker", "benchmark", "failure",
    }
    _require(set(value) == required, "worker result fields are not exact")
    _require(value["schema"] == SCHEMA, "worker result schema differs")
    _require(value["status"] in {"complete", "failed"}, "worker result status is invalid")
    _require(value["gpu"] == payload_value["gpu"], "worker result GPU differs from payload")
    _require(_same(value["cell"], payload_value["cell"]), "worker result cell differs from payload")
    binding = _worker_result_binding(payload_value)
    for key in ("config", "project_provenance", "roots", "run_parameters", "routes", "eligibility", "timing_contract"):
        _require(_same(value[key], binding[key]), f"worker result {key} differs from payload")
    complete = value["status"] == "complete"
    _validate_worker_runtime(value["runtime_preflight"], str(payload_value["gpu"]), allow_not_passed=not complete)
    _validate_worker_provenance(
        value["provenance"],
        payload_value["project_provenance"],
        required=complete,
        catswe_required=bool(payload_value["config"]["include_catswe_model"]),
    )
    if complete:
        _require(value["failure"] is None, "complete worker result contains a failure")
        worker = _json_object(value["worker"], "complete worker identity")
        _require(
            set(worker)
            == {
                "run_id", "started_unix_s", "finished_unix_s", "elapsed_s",
                "timed_tensor_hashing", "timed_input_copy", "timed_qualification",
            },
            "complete worker identity fields are not exact",
        )
        _require(worker["run_id"] == payload_value["run_id"], "worker result run ID differs")
        for key in ("started_unix_s", "finished_unix_s", "elapsed_s"):
            _require(type(worker[key]) in {int, float} and not isinstance(worker[key], bool) and math.isfinite(float(worker[key])), f"worker result {key} is malformed")
        _require(worker["finished_unix_s"] >= worker["started_unix_s"] and worker["elapsed_s"] >= 0, "worker result timing is malformed")
        _require(worker["timed_tensor_hashing"] is False, "worker result timed tensor hashing is enabled")
        _require(worker["timed_input_copy"] is False, "worker result timed input copy is enabled")
        _require(worker["timed_qualification"] is False, "worker result timed qualification is enabled")
        benchmark = _json_object(value["benchmark"], "complete worker benchmark")
        _validate_worker_model_report(benchmark, payload_value)
        _require(_same(value["report_identity"], _report_identity(benchmark)), "worker result report identity is forged")
        identity = _report_identity(benchmark)
        _require(identity["cell_id"] == payload_value["cell"]["cell_id"], "worker benchmark cell identity differs")
        _require(identity["config_sha256"] == _json_sha256(payload_value["config"], "worker config"), "worker benchmark config identity differs")
        _require(identity["scope"] == "custom" and identity["phases"] == ["model"], "worker benchmark phase route differs")
    else:
        _require(isinstance(value["failure"], Mapping), "failed worker result is missing failure details")
        failure = _json_object(value["failure"], "failed worker result failure")
        _require(set(failure) == {"type", "message"}, "worker failure fields are not exact")
        _require(isinstance(failure["type"], str) and failure["type"], "worker failure type is missing")
        _require(isinstance(failure["message"], str) and failure["message"], "worker failure message is missing")
        if value["benchmark"] is None:
            _require(value["worker"] is None, "failed worker without a benchmark has worker timing")
            _require(_same(value["report_identity"], _unavailable_report_identity(payload_value)), "failed worker report identity is forged")
        else:
            benchmark = _json_object(value["benchmark"], "failed worker benchmark")
            _validate_worker_model_report(benchmark, payload_value)
            _require(_same(value["report_identity"], _report_identity(benchmark)), "failed worker report identity is forged")
    return value


def _runtime_preflight(gpu: str) -> dict[str, Any]:
    """Check exact runtime/device before importing the benchmark stack."""

    try:
        import torch
        import triton
    except Exception as exc:  # pragma: no cover - CUDA host only
        raise SweepError(f"runtime import failed: {type(exc).__name__}: {exc}") from exc
    expected = {"H100": ((9, 0), "H100"), "B200": ((10, 0), "B200")}[gpu]
    _require(str(torch.__version__) == "2.13.0+cu130", f"Torch must be 2.13.0+cu130, got {torch.__version__}")
    _require(str(torch.version.cuda) == "13.0", f"CUDA runtime must be 13.0, got {torch.version.cuda}")
    _require(str(triton.__version__) == "3.7.1", f"Triton must be 3.7.1, got {triton.__version__}")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one CUDA GPU is required")
    name = str(torch.cuda.get_device_name(0))
    capability = tuple(torch.cuda.get_device_capability(0))
    _require(expected[1] in name and capability == expected[0], f"GPU does not match {gpu}: {name} cc={capability}")
    properties = torch.cuda.get_device_properties(0)
    _require(
        int(properties.total_memory) >= _MIN_GPU_MEMORY_BYTES[gpu],
        f"GPU memory is below the conservative {gpu} floor: {properties.total_memory}",
    )
    return {
        "status": "passed",
        "gpu": gpu,
        "name": name,
        "compute_capability": list(capability),
        "total_memory": int(properties.total_memory),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "triton": str(triton.__version__),
    }


def _worker_failure(
    payload: Mapping[str, Any],
    exc: BaseException,
    *,
    preflight: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding = _worker_result_binding(payload)
    result = {
        "schema": SCHEMA,
        "status": "failed",
        "cell": copy.deepcopy(dict(payload.get("cell", {}))),
        "gpu": payload.get("gpu"),
        **binding,
        "runtime_preflight": dict(preflight or {"status": "not_passed"}),
        "provenance": copy.deepcopy(dict(provenance)) if provenance is not None else None,
        "report_identity": _unavailable_report_identity(payload),
        "worker": None,
        "benchmark": None,
        "failure": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }
    if provenance is not None:
        result["provenance"] = copy.deepcopy(dict(provenance))
    return result


def run_worker(payload: Mapping[str, Any], output: str | os.PathLike[str]) -> dict[str, Any]:
    """Run one remote cell and atomically retain both success and failure."""

    value = _validate_worker_payload(payload)
    output_path = _absolute_leaf(output)
    preflight: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    try:
        root = Path(str(value["remote_repo"])).expanduser().resolve()
        fla_root = Path(str(value["remote_fla_root"])).expanduser().resolve()
        liger_root = Path(str(value["remote_liger_root"])).expanduser().resolve()
        provenance = {"project": _project_attestation(root, value["project_provenance"])}
        if value["config"]["include_catswe_model"]:
            catswe_root = Path(str(value["remote_catswe_root"])).expanduser().resolve()
            provenance["catswe"] = _catswe_attestation(
                catswe_root, value["project_provenance"]["catswe"]
            )
        preflight = _runtime_preflight(str(value["gpu"]))
        _require(fla_root.is_dir(), "remote FLA checkout is missing")
        _require(liger_root.is_dir(), "remote Liger checkout is missing")
        os.environ["TRITON_CACHE_DIR"] = str(value["triton_cache_dir"])
        os.environ["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), os.environ.get("PYTHONPATH", "")))
        os.chdir(root)
        from benchmarks.run import run_suite

        started = time.time()
        report = run_suite(dict(value["config"]))
        finished = time.time()
        model = report.get("model_timings") if isinstance(report, Mapping) else None
        complete = (
            isinstance(model, Mapping)
            and model.get("status") == "complete"
            and model.get("failures") == []
            and model.get("comparator_failures") == []
        )
        result = {
            "schema": SCHEMA,
            "status": "complete" if complete else "failed",
            "gpu": value["gpu"],
            "cell": copy.deepcopy(value["cell"]),
            **_worker_result_binding(value),
            "runtime_preflight": preflight,
            "provenance": provenance,
            "worker": {
                "run_id": value["run_id"],
                "started_unix_s": started,
                "finished_unix_s": finished,
                "elapsed_s": finished - started,
                "timed_tensor_hashing": False,
                "timed_input_copy": False,
                "timed_qualification": False,
            },
            "benchmark": report,
            "report_identity": _report_identity(report),
            "failure": None,
        }
        if not complete:
            result["failure"] = {
                "type": "IncompleteModelStep",
                "message": "model_timings did not complete with zero failures",
            }
    except Exception as exc:  # noqa: BLE001 - retain remote worker failure
        result = _worker_failure(
            value, exc, preflight=preflight, provenance=provenance
        )
    atomic_write_json(output_path, result)
    return result


def _encode_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return base64.urlsafe_b64encode(encoded).decode("ascii")


def ssh_command(
    *,
    gpu: str,
    payload: Mapping[str, Any],
    remote_output: str,
    host: Mapping[str, Any],
    remote_repo: str,
    remote_venv: str,
) -> list[str]:
    """Build a shell-safe SSH command for one worker."""

    _require(gpu in SUPPORTED_GPUS, "unsupported SSH GPU")
    for key in ("host", "port", "user"):
        _require(key in host, f"SSH host mapping lacks {key}")
    payload_b64 = _encode_payload(payload)
    remote = " ".join(
        (
            f"cd {shlex.quote(remote_repo)}",
            "&&",
            f"export PYTHONPATH={shlex.quote(remote_repo + '/src:' + remote_repo)}",
            "&&",
            f"{shlex.quote(remote_venv + '/bin/python')} -m scripts.compiled_step_sweep worker",
            f"--payload-b64 {shlex.quote(payload_b64)}",
            f"--output {shlex.quote(remote_output)}",
        )
    )
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(int(host["port"])),
        f"{host['user']}@{host['host']}",
        remote,
    ]


def scp_command(*, host: Mapping[str, Any], remote_path: str, local_path: Path) -> list[str]:
    return [
        "scp",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-P",
        str(int(host["port"])),
        f"{host['user']}@{host['host']}:{remote_path}",
        str(local_path),
    ]


def _new_run_id() -> str:
    now = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{now}Z-{uuid.uuid4().hex[:10]}"


def _success_path(output_root: Path, gpu: str, cell_id: str) -> Path:
    return output_root / gpu.lower() / f"{cell_id}.json"


def _failure_path(output_root: Path, gpu: str, cell_id: str, run_id: str) -> Path:
    return output_root / gpu.lower() / "failures" / f"{cell_id}.{run_id}.json"


def _log_path(output_root: Path, gpu: str, cell_id: str, run_id: str, suffix: str) -> Path:
    return output_root / gpu.lower() / "logs" / f"{cell_id}.{run_id}.{suffix}"


def _is_complete_cell(
    path: Path,
    cell_id: str,
    *,
    expected_payload: Mapping[str, Any],
    expected_output_root: Path | None = None,
    expected_remote_output_root: str | None = None,
) -> bool:
    """Return whether a cached result is fully bound to this campaign cell."""

    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = _read_json(path, "existing cell result")
        _require(value.get("status") == "complete", "existing cell result is not complete")
        cached_cell = _json_object(value.get("cell"), "existing cell metadata")
        _require(cached_cell.get("cell_id") == cell_id, "existing cell result belongs to another cell")
        worker = _json_object(value.get("worker"), "existing worker identity")
        cached_run_id = worker.get("run_id")
        _require(isinstance(cached_run_id, str) and cached_run_id, "existing worker run ID is missing")
        candidate = copy.deepcopy(value)
        launcher = _json_object(candidate.pop("launcher", None), "existing launcher identity")
        _require(
            set(launcher)
            == {
                "run_id", "gpu", "cell_id", "started_unix_s", "finished_unix_s", "elapsed_s",
                "ssh_exit_code", "remote_output", "command", "logs",
            },
            "existing launcher identity fields are not exact",
        )
        _require(launcher["run_id"] == cached_run_id, "existing launcher run ID differs")
        _require(launcher["gpu"] == expected_payload["gpu"], "existing launcher GPU differs")
        _require(launcher["cell_id"] == cell_id, "existing launcher cell differs")
        _require(type(launcher["ssh_exit_code"]) is int and launcher["ssh_exit_code"] == 0, "existing launcher did not succeed")
        for key in ("started_unix_s", "finished_unix_s", "elapsed_s"):
            _require(
                type(launcher[key]) in {int, float}
                and not isinstance(launcher[key], bool)
                and math.isfinite(float(launcher[key])),
                f"existing launcher {key} is malformed",
            )
        _require(
            launcher["finished_unix_s"] >= launcher["started_unix_s"]
            and launcher["elapsed_s"] >= 0,
            "existing launcher timing is malformed",
        )
        for key in ("remote_output", "command"):
            _require(isinstance(launcher[key], str) and launcher[key], f"existing launcher {key} is missing")
        if expected_remote_output_root is not None:
            expected_remote_output = (
                f"{expected_remote_output_root.rstrip('/')}/{expected_payload['gpu'].lower()}"
                f"/failures/{cell_id}.{cached_run_id}.json"
            )
            _require(
                launcher["remote_output"] == expected_remote_output,
                "existing launcher remote output differs",
            )
        logs = _json_object(launcher["logs"], "existing launcher logs")
        _require(
            set(logs) == {"stdout", "stderr"}
            and all(isinstance(path_value, str) and path_value for path_value in logs.values()),
            "existing launcher logs are malformed",
        )
        if expected_output_root is not None:
            expected_logs = {
                suffix: str(_log_path(expected_output_root, expected_payload["gpu"], cell_id, cached_run_id, f"{suffix}.log"))
                for suffix in ("stdout", "stderr")
            }
            _require(logs == expected_logs, "existing launcher logs differ")
        expected = copy.deepcopy(dict(expected_payload))
        expected["run_id"] = cached_run_id
        _validate_worker_result(candidate, expected)
    except (SweepError, AttributeError, KeyError, TypeError):
        return False
    return True


def _load_index(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": INDEX_SCHEMA,
            "status": "running",
            "manifest": copy.deepcopy(dict(manifest)),
            "results": {},
        }
    value = _read_json(path, "sweep index")
    _require(value.get("schema") == INDEX_SCHEMA, "sweep index schema differs")
    _require(_same(value.get("manifest"), manifest), "existing sweep index manifest differs")
    _require(isinstance(value.get("results"), Mapping), "sweep index results must be an object")
    return value


def _write_failure_logs(output_root: Path, gpu: str, cell_id: str, run_id: str, stdout: str, stderr: str) -> dict[str, str]:
    stdout_path = _log_path(output_root, gpu, cell_id, run_id, "stdout.log")
    stderr_path = _log_path(output_root, gpu, cell_id, run_id, "stderr.log")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {"stdout": str(stdout_path), "stderr": str(stderr_path)}


def _make_worker_payload(
    *,
    cell: Mapping[str, Any],
    gpu: str,
    remote_repo: str,
    remote_fla_root: str,
    remote_liger_root: str,
    remote_catswe_root: str,
    cache_root: str,
    seed: int,
    warmup: int,
    rounds: int,
    bootstrap_samples: int,
    batch: int,
    sequence: int,
    vocab: int,
    run_id: str,
    project_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct the one-cell payload used both for launch and resume checks."""

    config = make_worker_config(
        cell,
        seed=seed,
        warmup=warmup,
        rounds=rounds,
        bootstrap_samples=bootstrap_samples,
        batch=batch,
        sequence=sequence,
        vocab=vocab,
        remote_repo=remote_repo,
        remote_fla_root=remote_fla_root,
        remote_liger_root=remote_liger_root,
        remote_catswe_root=remote_catswe_root,
    )
    project = copy.deepcopy(dict(project_provenance))
    if not config["include_catswe_model"]:
        # Keep ineligible payloads independent of the optional vendor.  The
        # manifest still carries its global Catswe contract, while this cell
        # transports no Catswe provenance that it cannot use.
        project.pop("catswe", None)
    return {
        "schema": SCHEMA,
        "gpu": gpu,
        "cell": copy.deepcopy(dict(cell)),
        "config": config,
        "remote_repo": remote_repo,
        "remote_fla_root": remote_fla_root,
        "remote_liger_root": remote_liger_root,
        "remote_catswe_root": remote_catswe_root,
        "triton_cache_dir": f"{cache_root.rstrip('/')}/{gpu.lower()}",
        "run_id": run_id,
        "seed": seed,
        "warmup": warmup,
        "rounds": rounds,
        "bootstrap_samples": bootstrap_samples,
        "batch": batch,
        "sequence": sequence,
        "vocab": vocab,
        "project_provenance": project,
    }


def _run_one_remote(
    *,
    cell: Mapping[str, Any],
    gpu: str,
    output_root: Path,
    host: Mapping[str, Any],
    remote_repo: str,
    remote_venv: str,
    remote_fla_root: str,
    remote_liger_root: str,
    remote_catswe_root: str,
    project_provenance: Mapping[str, Any],
    remote_output_root: str,
    cache_root: str,
    seed: int,
    warmup: int,
    rounds: int,
    bootstrap_samples: int,
    batch: int,
    sequence: int,
    vocab: int,
    timeout_s: float | None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    run_id = _new_run_id()
    cell_id = str(cell["cell_id"])
    remote_output = f"{remote_output_root.rstrip('/')}/{gpu.lower()}/failures/{cell_id}.{run_id}.json"
    payload = _make_worker_payload(
        cell=cell,
        gpu=gpu,
        remote_repo=remote_repo,
        remote_fla_root=remote_fla_root,
        remote_liger_root=remote_liger_root,
        remote_catswe_root=remote_catswe_root,
        cache_root=cache_root,
        seed=seed,
        warmup=warmup,
        rounds=rounds,
        bootstrap_samples=bootstrap_samples,
        batch=batch,
        sequence=sequence,
        vocab=vocab,
        run_id=run_id,
        project_provenance=project_provenance,
    )
    command = ssh_command(
        gpu=gpu,
        payload=payload,
        remote_output=remote_output,
        host=host,
        remote_repo=remote_repo,
        remote_venv=remote_venv,
    )
    command_text = shlex.join(command)
    started = time.time()
    try:
        completed = runner(command, capture_output=True, text=True, timeout=timeout_s, check=False)
        return_code = int(completed.returncode)
        stdout = str(completed.stdout or "")
        stderr = str(completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        return_code = None
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "") + "\nworker timed out"
    except Exception as exc:  # noqa: BLE001 - retain SSH transport failure
        return_code = None
        stdout = ""
        stderr = f"launcher exception: {type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    # Pull only after SSH has returned.  A worker always writes an atomic JSON
    # before its final summary, so a missing remote file is retained as a
    # launcher failure rather than treated as a successful empty result.
    attempt_local = _failure_path(output_root, gpu, cell_id, run_id)
    attempt_local.parent.mkdir(parents=True, exist_ok=True)
    pulled_temp = attempt_local.with_name(f".{attempt_local.name}.pull.tmp")
    if pulled_temp.exists() or pulled_temp.is_symlink():
        pulled_temp.unlink(missing_ok=True)
    scp = scp_command(host=host, remote_path=remote_output, local_path=pulled_temp)
    try:
        pulled = subprocess.run(scp, capture_output=True, text=True, timeout=120, check=False)
    except Exception as exc:  # noqa: BLE001 - retain SCP transport failure
        pulled = None
        stderr += f"\nscp exception: {type(exc).__name__}: {exc}"
    worker_result: dict[str, Any] | None = None
    if pulled is not None:
        stdout += str(pulled.stdout or "")
        stderr += str(pulled.stderr or "")
        if pulled.returncode == 0 and pulled_temp.is_file() and not pulled_temp.is_symlink():
            try:
                worker_result = _read_json(pulled_temp, "pulled worker result")
            except SweepError as exc:
                stderr += f"\ninvalid worker result: {exc}"
    launcher = {
        "run_id": run_id,
        "gpu": gpu,
        "cell_id": cell_id,
        "started_unix_s": started,
        "finished_unix_s": time.time(),
        "elapsed_s": elapsed,
        "ssh_exit_code": return_code,
        "remote_output": remote_output,
        "command": command_text,
    }
    log_paths = _write_failure_logs(output_root, gpu, cell_id, run_id, stdout, stderr)
    launcher["logs"] = log_paths
    try:
        pulled_temp.unlink(missing_ok=True)
    except OSError:
        pass
    result_validation_error: SweepError | None = None
    if worker_result is not None:
        try:
            _validate_worker_result(worker_result, payload)
        except SweepError as exc:
            result_validation_error = exc
            stderr += f"\ninvalid worker result: {exc}"
    if worker_result is None or result_validation_error is not None:
        observed = None
        if worker_result is not None:
            observed = {
                "status": worker_result.get("status"),
                "cell_id": worker_result.get("cell", {}).get("cell_id")
                if isinstance(worker_result.get("cell"), Mapping)
                else None,
            }
        failure = {
            "schema": SCHEMA,
            "status": "failed",
            "gpu": gpu,
            "cell": copy.deepcopy(dict(cell)),
            "failure": {
                "type": "WorkerResultValidationError" if result_validation_error else "RemoteWorkerUnavailable",
                "message": str(result_validation_error)
                if result_validation_error
                else "worker output was missing, malformed, or belonged to another cell",
            },
            "observed_worker": observed,
            "launcher": launcher,
        }
        atomic_write_json(attempt_local, failure)
        return {"status": "failed", "path": str(attempt_local), "launcher": launcher}
    worker_result["launcher"] = launcher
    if worker_result.get("status") == "complete" and return_code == 0:
        canonical = _success_path(output_root, gpu, cell_id)
        _require(
            not canonical.exists()
            or _is_complete_cell(
                canonical,
                cell_id,
                expected_payload=payload,
                expected_output_root=output_root,
                expected_remote_output_root=remote_output_root,
            ),
            "refusing to replace a complete cell result",
        )
        atomic_write_json(canonical, worker_result)
        return {"status": "complete", "path": str(canonical), "launcher": launcher}
    if worker_result.get("status") == "complete":
        failure = {
            "schema": SCHEMA,
            "status": "failed",
            "gpu": gpu,
            "cell": copy.deepcopy(dict(cell)),
            "failure": {
                "type": "RemoteWorkerExit",
                "message": f"worker returned a complete result with SSH exit code {return_code}",
            },
            "observed_worker": worker_result,
            "launcher": launcher,
        }
        atomic_write_json(attempt_local, failure)
        return {"status": "failed", "path": str(attempt_local), "launcher": launcher}
    atomic_write_json(attempt_local, worker_result)
    return {"status": "failed", "path": str(attempt_local), "launcher": launcher}


def run_sweep(
    *,
    output_dir: str | os.PathLike[str],
    gpus: Sequence[str] = SUPPORTED_GPUS,
    hosts: Mapping[str, Mapping[str, Any]] = DEFAULT_HOSTS,
    remote_repo: str = DEFAULT_REMOTE_REPO,
    remote_venvs: Mapping[str, str] = DEFAULT_REMOTE_VENV,
    remote_fla_root: str = DEFAULT_REMOTE_FLA,
    remote_liger_root: str = DEFAULT_REMOTE_LIGER,
    remote_catswe_root: str = DEFAULT_REMOTE_CATSWE,
    remote_output_root: str = DEFAULT_REMOTE_OUTPUT_ROOT,
    cache_root: str = DEFAULT_CACHE_ROOT,
    seed: int = DEFAULT_SEED,
    warmup: int = DEFAULT_WARMUP,
    rounds: int = DEFAULT_ROUNDS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP,
    batch: int = BATCH,
    sequence: int = SEQUENCE,
    vocab: int = VOCAB,
    timeout_s: float | None = None,
    manifest: Mapping[str, Any] | None = None,
    parallel_gpus: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the matrix with resume support and one active cell per GPU.

    Independent H100 and B200 workers are launched concurrently by default,
    but cells remain serialized on each individual GPU.  The cell barrier
    makes the ordering deterministic and prevents two large model processes
    from competing for one device's memory.  Set ``parallel_gpus=False`` for
    a strictly serial diagnostic run.
    """

    selected_gpus = tuple(gpus)
    _require(selected_gpus and all(gpu in SUPPORTED_GPUS for gpu in selected_gpus), "unsupported GPU selection")
    _require(len(set(selected_gpus)) == len(selected_gpus), "duplicate GPU selection")
    selected_manifest = validate_manifest(
        manifest
        or make_manifest(
            seed=seed,
            warmup=warmup,
            rounds=rounds,
            bootstrap_samples=bootstrap_samples,
            batch=batch,
            sequence=sequence,
            vocab=vocab,
            remote_repo=remote_repo,
            remote_fla_root=remote_fla_root,
            remote_liger_root=remote_liger_root,
            remote_catswe_root=remote_catswe_root,
            remote_output_root=remote_output_root,
            cache_root=cache_root,
        )
    )
    _require(
        selected_manifest["seed"] == seed
        and selected_manifest["warmup"] == warmup
        and selected_manifest["rounds"] == rounds
        and selected_manifest["bootstrap_samples"] == bootstrap_samples,
        "launch timing parameters differ from the persisted sweep manifest",
    )
    _require(
        selected_manifest["fixed_profile"]["batch"] == batch
        and selected_manifest["fixed_profile"]["sequence"] == sequence
        and selected_manifest["fixed_profile"]["vocab"] == vocab,
        "launch model dimensions differ from the persisted sweep manifest",
    )
    expected_launch = {
        "remote_repo": remote_repo,
        "remote_fla_root": remote_fla_root,
        "remote_liger_root": remote_liger_root,
        "remote_catswe_root": remote_catswe_root,
        "remote_output_root": remote_output_root,
        "cache_root": cache_root,
    }
    _require(
        _same(selected_manifest["launch"], expected_launch),
        "launch roots differ from the persisted sweep manifest",
    )
    # The manifest is a screen plan, not the sealed six-job release manifest.
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not _same(_read_json(manifest_path, "sweep manifest"), selected_manifest):
        raise SweepError("existing sweep manifest differs")
    atomic_write_json(manifest_path, selected_manifest)
    index_path = output_root / "index.json"
    index = _load_index(index_path, selected_manifest)
    index["status"] = "dry_run" if dry_run else "running"
    index["scheduling"] = {
        "gpu_parallelism": "one_cell_per_gpu" if parallel_gpus and len(selected_gpus) > 1 else "serial",
        "max_active_cells_per_gpu": 1,
        "cell_barrier": True,
    }
    atomic_write_json(index_path, index)
    if dry_run:
        return index
    for gpu in selected_gpus:
        _require(gpu in hosts and gpu in remote_venvs, f"missing remote configuration for {gpu}")

    def run_cell(gpu: str, cell: Mapping[str, Any]) -> dict[str, Any]:
        return _run_one_remote(
            cell=cell,
            gpu=gpu,
            output_root=output_root,
            host=hosts[gpu],
            remote_repo=remote_repo,
            remote_venv=remote_venvs[gpu],
            remote_fla_root=remote_fla_root,
            remote_liger_root=remote_liger_root,
            remote_catswe_root=remote_catswe_root,
            project_provenance=selected_manifest["project_provenance"],
            remote_output_root=remote_output_root,
            cache_root=cache_root,
            seed=seed,
            warmup=warmup,
            rounds=rounds,
            bootstrap_samples=bootstrap_samples,
            batch=batch,
            sequence=sequence,
            vocab=vocab,
            timeout_s=timeout_s,
        )

    executor = (
        ThreadPoolExecutor(max_workers=len(selected_gpus), thread_name_prefix="sweep-gpu")
        if parallel_gpus and len(selected_gpus) > 1
        else None
    )
    try:
        # Geometry-major ordering provides a barrier between cells while using
        # both devices concurrently.  Within each GPU, the order is the
        # manifest order: Full first, then increasing source count through the
        # Block event sizes 8/4/2/1.
        for cell in selected_manifest["cells"]:
            cell_id = str(cell["cell_id"])
            pending: list[str] = []
            for gpu in selected_gpus:
                canonical = _success_path(output_root, gpu, cell_id)
                resume_payload = _make_worker_payload(
                    cell=cell,
                    gpu=gpu,
                    remote_repo=remote_repo,
                    remote_fla_root=remote_fla_root,
                    remote_liger_root=remote_liger_root,
                    remote_catswe_root=remote_catswe_root,
                    cache_root=cache_root,
                    seed=seed,
                    warmup=warmup,
                    rounds=rounds,
                    bootstrap_samples=bootstrap_samples,
                    batch=batch,
                    sequence=sequence,
                    vocab=vocab,
                    run_id="resume-check",
                    project_provenance=selected_manifest["project_provenance"],
                )
                if _is_complete_cell(
                    canonical,
                    cell_id,
                    expected_payload=resume_payload,
                    expected_output_root=output_root,
                    expected_remote_output_root=remote_output_root,
                ):
                    index["results"][f"{gpu}:{cell_id}"] = {
                        "status": "skipped_complete",
                        "path": str(canonical),
                    }
                else:
                    pending.append(gpu)
            if pending:
                if executor is None:
                    completed = {gpu: run_cell(gpu, cell) for gpu in pending}
                else:
                    futures = {gpu: executor.submit(run_cell, gpu, cell) for gpu in pending}
                    # Record in manifest GPU order even if the faster device
                    # completes first, keeping index files deterministic.
                    completed = {gpu: futures[gpu].result() for gpu in pending}
                for gpu in pending:
                    index["results"][f"{gpu}:{cell_id}"] = completed[gpu]
                    index["last_completed_or_attempted"] = f"{gpu}:{cell_id}"
            atomic_write_json(index_path, index)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    expected_keys = {
        f"{gpu}:{cell['cell_id']}"
        for gpu in selected_gpus
        for cell in selected_manifest["cells"]
    }
    selected_results = {
        key: index["results"].get(key)
        for key in expected_keys
    }
    index["status"] = "complete" if (
        all(
            isinstance(value, Mapping)
            and value.get("status") in {"complete", "skipped_complete"}
            for value in selected_results.values()
        )
        and all(value is not None for value in selected_results.values())
    ) else "incomplete"
    atomic_write_json(index_path, index)
    return index


def _worker_cli(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(base64.urlsafe_b64decode(args.payload_b64.encode("ascii")).decode("utf-8"))
        result = run_worker(payload, args.output)
    except Exception as exc:  # noqa: BLE001 - retain worker validation failure
        # Validation can fail before a complete payload exists.  Keep the
        # process nonzero and print machine-readable diagnostics; the local
        # launcher will retain the SSH stderr and create its own failure row.
        print(json.dumps({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}}, sort_keys=True), flush=True)
        return 1
    print(json.dumps({"status": result.get("status"), "cell_id": result.get("cell", {}).get("cell_id"), "output": str(args.output)}, sort_keys=True), flush=True)
    return 0 if result.get("status") == "complete" else 1


def _parse_gpu_list(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    _require(items and all(item in SUPPORTED_GPUS for item in items), "--gpus must contain H100 and/or B200")
    _require(len(set(items)) == len(items), "--gpus contains duplicates")
    return items


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep_parser = subparsers.add_parser("sweep", help="run or dry-run the remote matrix")
    sweep_parser.add_argument("--output-dir", type=Path, required=True)
    sweep_parser.add_argument("--gpus", default=",".join(SUPPORTED_GPUS))
    sweep_parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    sweep_parser.add_argument("--h100-host", default=DEFAULT_HOSTS["H100"]["host"])
    sweep_parser.add_argument("--h100-port", type=int, default=DEFAULT_HOSTS["H100"]["port"])
    sweep_parser.add_argument("--b200-host", default=DEFAULT_HOSTS["B200"]["host"])
    sweep_parser.add_argument("--b200-port", type=int, default=DEFAULT_HOSTS["B200"]["port"])
    sweep_parser.add_argument("--h100-venv", default=DEFAULT_REMOTE_VENV["H100"])
    sweep_parser.add_argument("--b200-venv", default=DEFAULT_REMOTE_VENV["B200"])
    sweep_parser.add_argument("--fla-root", default=DEFAULT_REMOTE_FLA)
    sweep_parser.add_argument("--liger-root", default=DEFAULT_REMOTE_LIGER)
    sweep_parser.add_argument("--catswe-root", default=DEFAULT_REMOTE_CATSWE)
    sweep_parser.add_argument("--remote-output-root", default=DEFAULT_REMOTE_OUTPUT_ROOT)
    sweep_parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    sweep_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sweep_parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    sweep_parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    sweep_parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP)
    sweep_parser.add_argument("--batch", type=int, default=BATCH)
    sweep_parser.add_argument("--sequence", type=int, default=SEQUENCE)
    sweep_parser.add_argument("--vocab", type=int, default=VOCAB)
    sweep_parser.add_argument("--timeout-s", type=float)
    sweep_parser.add_argument(
        "--serial-gpus",
        action="store_true",
        help="run H100 and B200 one after the other (default uses one cell per GPU concurrently)",
    )
    sweep_parser.add_argument("--dry-run", action="store_true")
    worker_parser = subparsers.add_parser("worker", help="run one cell on a remote host")
    worker_parser.add_argument("--payload-b64", required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            return _worker_cli(args)
        gpus = _parse_gpu_list(args.gpus)
        hosts = {
            "H100": {"host": args.h100_host, "port": args.h100_port, "user": "root"},
            "B200": {"host": args.b200_host, "port": args.b200_port, "user": "root"},
        }
        result = run_sweep(
            output_dir=args.output_dir,
            gpus=gpus,
            hosts=hosts,
            remote_repo=args.remote_repo,
            remote_venvs={"H100": args.h100_venv, "B200": args.b200_venv},
            remote_fla_root=args.fla_root,
            remote_liger_root=args.liger_root,
            remote_catswe_root=args.catswe_root,
            remote_output_root=args.remote_output_root,
            cache_root=args.cache_root,
            seed=args.seed,
            warmup=args.warmup,
            rounds=args.rounds,
            bootstrap_samples=args.bootstrap_samples,
            batch=args.batch,
            sequence=args.sequence,
            vocab=args.vocab,
            timeout_s=args.timeout_s,
            parallel_gpus=not args.serial_gpus,
            dry_run=args.dry_run,
        )
        planned = len(result.get("manifest", {}).get("cells", ())) * len(gpus)
        print(json.dumps({"status": result.get("status"), "cells": len(result.get("results", {})) or planned, "index": str(args.output_dir.resolve() / "index.json")}, sort_keys=True))
        return 0 if result.get("status") in {"dry_run", "complete"} else 1
    except (SweepError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())


__all__ = [
    "BATCH",
    "BLOCK_SIZES",
    "DEFAULT_HOSTS",
    "DEFAULT_REMOTE_CATSWE",
    "DEFAULT_REMOTE_LIGER",
    "HEAD_DIM",
    "HISTORICAL_D1024_LR_RANK",
    "INDEX_SCHEMA",
    "KERNEL_PATHS",
    "LAYERS",
    "MANIFEST_SCHEMA",
    "MODEL_ONLY_STANDARD_WIDTHS",
    "SCHEMA",
    "SEQUENCE",
    "SUPPORTED_GPUS",
    "VOCAB",
    "WIDTHS",
    "atomic_write_json",
    "block_count_for_event_size",
    "build_matrix",
    "lr_rank_for_width",
    "main",
    "make_cell",
    "make_manifest",
    "make_model_config",
    "make_worker_config",
    "max_sources_for_cell",
    "read_source_counts_for_cell",
    "run_sweep",
    "run_worker",
    "scp_command",
    "ssh_command",
    "validate_manifest",
]
