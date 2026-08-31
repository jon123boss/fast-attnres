"""Strict identity checks for the external Flash Linear Attention checkout.

The production ladder uses one native FLA anchor.  This module keeps its
checkout identity check independent of Torch, Triton, and Modal so it can run
before any model is constructed.  The package digest intentionally matches
the transport hash documented by the ladder: sorted Python files below
``fla/``, with each relative path followed by its bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from .vendor_identity import normalize_remote_origin, verify_remote_origin


FLA_CHECKOUT_KEYS = frozenset(
    {"environment", "layout", "revision", "package_sha256", "required_clean"}
)
FLA_CHECKOUT_ENVIRONMENT = "ATTNRES_FLA_DIR"
FLA_CHECKOUT_LAYOUT = "clean checkout containing fla/"
FLA_REPOSITORY = "https://github.com/fla-org/flash-linear-attention.git"
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _report_value(value: Any) -> Any:
    """Convert malformed input to bounded JSON-friendly diagnostic data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _report_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(child) for child in value]
    return repr(value)


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def validate_fla_checkout_spec(value: Any) -> dict[str, Any]:
    """Validate and normalize the exact ``fla_checkout`` config schema.

    The release metadata is deliberately strict.  In particular, booleans
    are not accepted as integers, hexadecimal values must be lowercase, and
    unknown keys are rejected instead of being ignored.
    """

    if not isinstance(value, Mapping):
        raise TypeError("production_ladder.fla_checkout must be an object")
    keys = set(value)
    if keys != set(FLA_CHECKOUT_KEYS):
        missing = sorted(FLA_CHECKOUT_KEYS - keys, key=str)
        extra = sorted(keys - FLA_CHECKOUT_KEYS, key=str)
        raise ValueError(
            "production_ladder.fla_checkout keys must be exactly "
            f"{sorted(FLA_CHECKOUT_KEYS)!r}; missing={missing!r}, extra={extra!r}"
        )

    environment = value["environment"]
    if type(environment) is not str:
        raise TypeError("fla_checkout.environment must be a string")
    if environment != FLA_CHECKOUT_ENVIRONMENT:
        raise ValueError(
            f"fla_checkout.environment must be {FLA_CHECKOUT_ENVIRONMENT!r}"
        )

    layout = value["layout"]
    if type(layout) is not str:
        raise TypeError("fla_checkout.layout must be a string")
    if layout != FLA_CHECKOUT_LAYOUT:
        raise ValueError(f"fla_checkout.layout must be {FLA_CHECKOUT_LAYOUT!r}")

    revision = value["revision"]
    if type(revision) is not str or not _REVISION_RE.fullmatch(revision):
        raise ValueError("fla_checkout.revision must be a lowercase 40-character git SHA")

    package_sha256 = value["package_sha256"]
    if type(package_sha256) is not str or not _SHA256_RE.fullmatch(package_sha256):
        raise ValueError(
            "fla_checkout.package_sha256 must be a lowercase 64-character SHA256 digest"
        )

    required_clean = value["required_clean"]
    if type(required_clean) is not bool:
        raise TypeError("fla_checkout.required_clean must be a boolean")

    return {
        "environment": environment,
        "layout": layout,
        "revision": revision,
        "package_sha256": package_sha256,
        "required_clean": required_clean,
    }


def validate_release_fla_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate active release FLA claims and return the checkout spec.

    ``None`` means the config does not request a native FLA model/compile arm.
    A requested arm must carry an explicit production ladder object and must
    select only the Triton checkpoint-1 anchor.  This prevents a missing or
    ambiguous checkout from being treated as an optional comparator.
    """

    if not isinstance(config, Mapping):
        raise TypeError("benchmark config must be an object")
    for name in ("include_fla_compile", "standard_fla_comparison", "include_fla_model"):
        if name in config and type(config[name]) is not bool:
            raise TypeError(f"{name} must be a boolean")
    requested = any(
        bool(config.get(name, False))
        for name in ("include_fla_compile", "standard_fla_comparison", "include_fla_model")
    )
    if not requested:
        return None

    production = config.get("production_ladder")
    if not isinstance(production, Mapping):
        raise TypeError(
            "native FLA release arms require a production_ladder object with pinned metadata"
        )
    checkout = validate_fla_checkout_spec(production.get("fla_checkout"))

    anchor = production.get("fla_anchor")
    if not isinstance(anchor, Mapping):
        raise TypeError("production_ladder.fla_anchor must be an object")
    anchor_keys = {"implementation", "checkpoint_level", "rank", "scope"}
    if set(anchor) != anchor_keys:
        raise ValueError(
            "production_ladder.fla_anchor keys must be exactly "
            f"{sorted(anchor_keys)!r}"
        )
    if anchor["implementation"] != "triton":
        raise ValueError("the active release FLA anchor must use Triton")
    if type(anchor["checkpoint_level"]) is not int or anchor["checkpoint_level"] != 1:
        raise ValueError("the active release FLA anchor must use checkpoint level 1")
    if type(anchor["rank"]) is not int or anchor["rank"] < 1:
        raise ValueError("production_ladder.fla_anchor.rank must be a positive integer")
    if anchor["scope"] != "R=D anchor only":
        raise ValueError("production_ladder.fla_anchor.scope must be 'R=D anchor only'")

    if bool(config.get("include_fla_compile", False)):
        backends = config.get("fla_compile_backends")
        if type(backends) is not list or backends != ["triton"]:
            raise ValueError(
                "the active release FLA compile arm must set fla_compile_backends to ['triton']"
            )
    if bool(config.get("include_fla_model", False)):
        raise ValueError(
            "the active release must disable optional FLA model discovery; "
            "use the explicit Triton checkpoint-1 anchor"
        )

    return {"checkout": checkout, "anchor": dict(anchor)}


def _candidate_roots(project_root: Path, configured: str | os.PathLike[str] | None) -> list[Path]:
    if configured is not None:
        return [Path(configured).expanduser()]
    candidates: list[Path] = []
    for variable in (
        "ATTNRES_FLA_DIR",
        "FLA_ROOT",
        "FLASH_LINEAR_ATTENTION_ROOT",
        "VENDOR_FLA_ROOT",
    ):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value).expanduser())
            break
    if not candidates:
        candidates.extend(
            [
                project_root / "vendor" / "fla",
                project_root.parent / "vendor" / "fla",
                project_root.parent / "vendor" / "flash-linear-attention",
            ]
        )
    seen: set[Path] = set()
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def resolve_fla_checkout(
    project_root: str | os.PathLike[str] | None = None,
    configured: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a checkout containing the expected native FLA package tree."""

    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    candidates = _candidate_roots(root, configured)
    for candidate in candidates:
        if (candidate / "fla" / "ops" / "attnres").is_dir():
            return candidate
    attempted = ", ".join(str(candidate.resolve()) for candidate in candidates) or "none"
    raise FileNotFoundError(
        "pinned FLA checkout with fla/ops/attnres was not found; tried " + attempted
    )


def _git_output(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot read FLA checkout git metadata: {' '.join(args)}") from exc
    return completed.stdout.strip()


def _package_metadata(root: Path) -> dict[str, Any]:
    package = root / "fla"
    paths = sorted(path for path in package.rglob("*.py") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"FLA package contains no Python files: {package}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(package)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {"package_sha256": digest.hexdigest(), "package_file_count": len(paths)}


def fla_checkout_metadata(
    project_root: str | os.PathLike[str] | None = None,
    configured: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Read revision, cleanliness, and package digest from an FLA checkout."""

    root = resolve_fla_checkout(project_root, configured)
    top = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise RuntimeError(f"configured FLA root {root} is inside checkout {top}")
    origin = verify_remote_origin(root, FLA_REPOSITORY)
    return {
        "path": str(root),
        "revision": _git_output(root, "rev-parse", "HEAD"),
        **_package_metadata(root),
        "origin": origin,
        "git_dirty": bool(
            _git_output(root, "status", "--porcelain", "--untracked-files=all")
        ),
    }


def verify_fla_checkout(
    expected: Any,
    project_root: str | os.PathLike[str] | None = None,
    configured: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a structured verification result; never silently downgrade it."""

    result: dict[str, Any] = {
        "status": "failed",
        "expected": _report_value(expected),
        "actual": None,
    }
    try:
        spec = validate_fla_checkout_spec(expected)
        result["expected"] = spec
        actual = fla_checkout_metadata(project_root, configured)
        result["actual"] = actual
        mismatches: dict[str, Any] = {}
        if actual["revision"] != spec["revision"]:
            mismatches["revision"] = {
                "expected": spec["revision"],
                "actual": actual["revision"],
            }
        if actual["package_sha256"] != spec["package_sha256"]:
            mismatches["package_sha256"] = {
                "expected": spec["package_sha256"],
                "actual": actual["package_sha256"],
            }
        if normalize_remote_origin(actual.get("origin")) != normalize_remote_origin(
            FLA_REPOSITORY
        ):
            mismatches["origin"] = {
                "expected": FLA_REPOSITORY,
                "actual": actual.get("origin"),
            }
        if spec["required_clean"] and actual["git_dirty"]:
            mismatches["git_dirty"] = {"expected": False, "actual": True}
        if mismatches:
            result["error"] = {
                "type": "FLACheckoutMismatch",
                "message": "pinned FLA checkout metadata does not match config",
                "mismatches": mismatches,
            }
        else:
            result["status"] = "verified"
    except Exception as exc:
        result["error"] = _error(exc)
    return result


def verify_release_fla_config(
    config: Any,
    project_root: str | os.PathLike[str] | None = None,
    configured: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Verify an active release config without raising or silently downgrading.

    The returned object is suitable for embedding in a benchmark report.  A
    ``failed`` status always carries the schema or checkout error, allowing a
    caller to stop before constructing a model while preserving diagnostics.
    """

    result: dict[str, Any] = {"status": "not_required"}
    try:
        release = validate_release_fla_config(config)
    except Exception as exc:
        result.update(status="failed", error=_error(exc))
        return result
    if release is None:
        return result
    verification = verify_fla_checkout(
        release["checkout"], project_root=project_root, configured=configured
    )
    verification["anchor"] = release["anchor"]
    return verification


def verify_runtime_fla_config(
    config: Any,
    project_root: str | os.PathLike[str] | None = None,
    configured: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Verify the exact FLA root that the current runtime can consume.

    Direct runs require a clean Git checkout. Modal transports only the
    verified ``fla/`` package bytes, so the host preflight embedded in the
    image is checked against the canonical mounted root instead.
    """

    host_preflight = os.environ.get("ATTNRES_FLA_HOST_PREFLIGHT", "")
    if not host_preflight:
        return verify_release_fla_config(
            config, project_root=project_root, configured=configured
        )

    result: dict[str, Any] = {"status": "failed"}
    try:
        release = validate_release_fla_config(config)
        if release is None:
            return {"status": "not_required"}
        mounted_root = os.environ.get("ATTNRES_FLA_DIR", "")
        if not mounted_root:
            raise RuntimeError("mounted FLA verification requires ATTNRES_FLA_DIR")
        verification = verify_mounted_fla_checkout(
            release["checkout"], mounted_root, json.loads(host_preflight)
        )
        verification["anchor"] = release["anchor"]
        return verification
    except Exception as exc:
        result["error"] = _error(exc)
        return result


def verify_mounted_fla_checkout(
    expected: Any,
    mounted_root: str | os.PathLike[str],
    host_preflight: Any,
) -> dict[str, Any]:
    """Verify transported bytes against a host-verified clean Git checkout."""

    result: dict[str, Any] = {
        "status": "failed",
        "expected": _report_value(expected),
        "host_preflight": _report_value(host_preflight),
        "actual": None,
    }
    try:
        spec = validate_fla_checkout_spec(expected)
        if not isinstance(host_preflight, Mapping):
            raise TypeError("FLA host preflight metadata must be an object")
        required_host_keys = {
            "path", "revision", "package_sha256", "package_file_count", "origin", "git_dirty"
        }
        if set(host_preflight) != required_host_keys:
            raise ValueError("FLA host preflight metadata has an unexpected schema")
        if type(host_preflight["path"]) is not str or not host_preflight["path"]:
            raise TypeError("FLA host preflight path must be a non-empty string")
        if (
            type(host_preflight["revision"]) is not str
            or not _REVISION_RE.fullmatch(host_preflight["revision"])
        ):
            raise ValueError("FLA host preflight revision must be a lowercase Git SHA")
        if (
            type(host_preflight["package_sha256"]) is not str
            or not _SHA256_RE.fullmatch(host_preflight["package_sha256"])
        ):
            raise ValueError("FLA host preflight package digest must be lowercase SHA256")
        if (
            type(host_preflight["package_file_count"]) is not int
            or host_preflight["package_file_count"] < 1
        ):
            raise TypeError("FLA host preflight package_file_count must be positive")
        if type(host_preflight["git_dirty"]) is not bool:
            raise TypeError("FLA host preflight git_dirty must be a boolean")
        mounted = Path(mounted_root).resolve()
        actual = {"path": str(mounted), **_package_metadata(mounted)}
        result["expected"] = spec
        result["actual"] = actual
        mismatches: dict[str, Any] = {}
        if host_preflight["revision"] != spec["revision"]:
            mismatches["revision"] = {
                "expected": spec["revision"], "actual": host_preflight["revision"]
            }
        if normalize_remote_origin(host_preflight["origin"]) != normalize_remote_origin(
            FLA_REPOSITORY
        ):
            mismatches["origin"] = {
                "expected": FLA_REPOSITORY,
                "actual": host_preflight["origin"],
            }
        if spec["required_clean"] and host_preflight["git_dirty"] is not False:
            mismatches["git_dirty"] = {
                "expected": False, "actual": host_preflight["git_dirty"]
            }
        for name in ("package_sha256", "package_file_count"):
            if actual[name] != host_preflight[name]:
                mismatches[f"transport_{name}"] = {
                    "expected": host_preflight[name], "actual": actual[name]
                }
        if actual["package_sha256"] != spec["package_sha256"]:
            mismatches["package_sha256"] = {
                "expected": spec["package_sha256"],
                "actual": actual["package_sha256"],
            }
        if mismatches:
            result["error"] = {
                "type": "FLACheckoutMismatch",
                "message": "mounted FLA bytes or host provenance do not match config",
                "mismatches": mismatches,
            }
        else:
            result["status"] = "verified"
    except Exception as exc:
        result["error"] = _error(exc)
    return result


__all__ = [
    "FLA_CHECKOUT_ENVIRONMENT",
    "FLA_CHECKOUT_KEYS",
    "FLA_CHECKOUT_LAYOUT",
    "FLA_REPOSITORY",
    "fla_checkout_metadata",
    "resolve_fla_checkout",
    "validate_fla_checkout_spec",
    "validate_release_fla_config",
    "verify_fla_checkout",
    "verify_mounted_fla_checkout",
    "verify_release_fla_config",
    "verify_runtime_fla_config",
]
