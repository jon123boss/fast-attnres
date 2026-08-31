"""Small, dependency-free helpers for strict external checkout identity.

The benchmark adapters load code from checkouts that are intentionally kept
outside this repository.  A path that merely contains a module is not enough
to identify an experiment: a different commit, an edited source file, or a
dirty checkout can change the measured kernel.  This module centralizes the
read-only checks used by the optional FLA and Liger adapters.

There is no fallback from an explicitly configured path.  If a caller sets a
path or an environment variable, only that path is considered.  Automatic
discovery is used only when no path was configured at all.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping


class CheckoutIdentityError(ImportError):
    """Raised when an external checkout is not the pinned source."""


_SHA_RE = re.compile(r"[0-9a-f]{40}")


def file_sha256(path: Path) -> str:
    """Hash a file in bounded chunks without importing optional runtimes."""

    if path.is_symlink():
        raise CheckoutIdentityError(f"refusing to hash symlinked vendor file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_symlink_component(path: Path) -> bool:
    """Return whether a lexical path component is a symlink."""

    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return any(component.is_symlink() for component in (path, *path.parents))


def _safe_relative_path(checkout: Path, relative: str | os.PathLike[str]) -> Path:
    """Resolve one checkout-relative path without following an escape link."""

    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CheckoutIdentityError(
            f"vendor identity path must remain checkout-relative: {relative!r}"
        )
    candidate = checkout.joinpath(relative_path)
    current = checkout
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise CheckoutIdentityError(
                f"vendor identity path contains a symlink: {relative!r}"
            )
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(checkout)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckoutIdentityError(
            f"vendor identity path escapes checkout: {relative!r}"
        ) from exc
    if candidate.is_symlink():
        raise CheckoutIdentityError(
            f"vendor identity path contains a symlink: {relative!r}"
        )
    return candidate


def _safe_package_files(checkout: Path, package_dir: str) -> list[Path]:
    """List package Python files while rejecting symlinked tree entries."""

    package = _safe_relative_path(checkout, package_dir)
    if not package.is_dir():
        return []
    package_files: list[Path] = []
    for current_name, directories, files in os.walk(package, followlinks=False):
        current = Path(current_name)
        _safe_relative_path(checkout, current.relative_to(checkout))
        for name in (*directories, *files):
            child = current / name
            relative = child.relative_to(checkout)
            _safe_relative_path(checkout, relative)
            if child.is_symlink():
                raise CheckoutIdentityError(
                    f"vendor package contains a symlink: {relative.as_posix()}"
                )
            if child.is_file() and child.suffix == ".py":
                package_files.append(child)
    return sorted(package_files)


def git_output(root: Path, *arguments: str) -> str:
    """Return one Git value, converting setup failures to identity errors."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        command = "git -C " + str(root) + " " + " ".join(arguments)
        raise CheckoutIdentityError(f"cannot read external checkout metadata: {command}") from exc
    return completed.stdout.strip()


def normalize_remote_origin(value: str | None) -> str | None:
    """Return a stable comparison form for one Git remote URL.

    The pinned metadata uses HTTPS URLs, while Git may spell the same URL with
    a case difference, a trailing slash, or the conventional ``.git`` suffix.
    Those harmless spellings are normalized.  Empty, whitespace-containing, or
    malformed URL values are rejected by returning ``None``; callers must not
    treat an unparseable origin as provenance.
    """

    if not isinstance(value, str):
        return None
    if (
        not value
        or value != value.strip()
        or any(character.isspace() for character in value)
    ):
        return None
    # Keep the comparison deliberately conservative.  The expected origins
    # are HTTPS repository URLs; an scp-style URL or a local path cannot match
    # one merely because its final path component happens to be the same.
    if "://" not in value:
        return None
    scheme, separator, remainder = value.partition("://")
    if not separator or not scheme or not remainder:
        return None
    if "?" in remainder or "#" in remainder:
        return None
    remainder = remainder.rstrip("/").lower().removesuffix(".git").rstrip("/")
    if not remainder:
        return None
    return f"{scheme.lower()}://{remainder}"


def remote_origins(root: Path, remote: str = "origin") -> tuple[str, ...]:
    """Return every configured fetch URL for one Git remote.

    ``git remote get-url origin`` returns only the first URL by default.  The
    config query preserves all values so a checkout with multiple ``origin``
    URLs cannot pass identity verification accidentally.
    """

    if not isinstance(remote, str) or not remote or any(
        character in remote for character in "\r\n"
    ):
        raise CheckoutIdentityError("Git remote name must be a non-empty single-line string")
    try:
        output = git_output(root, "config", "--get-all", f"remote.{remote}.url")
    except CheckoutIdentityError as exc:
        raise CheckoutIdentityError(
            f"Git remote {remote!r} has no configured origin URL"
        ) from exc
    values = tuple(output.splitlines())
    if not values or any(not value.strip() for value in values):
        raise CheckoutIdentityError(
            f"Git remote {remote!r} has a missing or empty origin URL"
        )
    if any(normalize_remote_origin(value) is None for value in values):
        raise CheckoutIdentityError(
            f"Git remote {remote!r} has an invalid origin URL"
        )
    return values


def remote_origin(root: Path, remote: str = "origin") -> str:
    """Return the sole configured URL for ``remote`` or fail closed."""

    values = remote_origins(root, remote)
    if len(values) != 1:
        raise CheckoutIdentityError(
            f"Git remote {remote!r} has ambiguous origin URLs ({len(values)})"
        )
    return values[0]


def verify_remote_origin(
    root: Path,
    expected_origin: str,
    remote: str = "origin",
) -> str:
    """Require one origin URL that matches the pinned repository."""

    expected = normalize_remote_origin(expected_origin)
    if expected is None:
        raise CheckoutIdentityError("expected vendor origin must be a valid repository URL")
    actual = remote_origin(root, remote)
    if normalize_remote_origin(actual) != expected:
        raise CheckoutIdentityError(
            f"vendor origin {actual!r} does not match pinned origin {expected_origin!r}"
        )
    return actual


def candidate_roots(
    project_root: str | os.PathLike[str],
    configured: str | os.PathLike[str] | None,
    *,
    environment: Iterable[str],
    defaults: Iterable[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    """Return deterministic roots, preserving explicit-path fail-closed rules.

    ``configured`` wins over every environment variable.  If it is absent,
    the first configured environment variable in ``environment`` wins.  This
    prevents a typo in a requested vendor path from silently selecting another
    checkout elsewhere on disk.
    """

    if configured is not None:
        values: list[str | os.PathLike[str]] = [configured]
    else:
        values = []
        for variable in environment:
            value = os.environ.get(variable)
            if value:
                values = [value]
                break
        if not values:
            values = list(defaults)

    result: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        raw_root = Path(value).expanduser()
        if _contains_symlink_component(raw_root):
            continue
        root = raw_root.resolve()
        # A caller may point at a source directory for a source-mounted
        # package.  Normalize that form before checking the Git root.
        if root.name == "src" and (root / "liger_kernel").is_dir():
            root = root.parent
        if root not in seen:
            seen.add(root)
            result.append(root)
    return tuple(result)


def checkout_identity(
    root: str | os.PathLike[str],
    *,
    expected_revision: str,
    expected_tree: str | None = None,
    expected_origin: str | None = None,
    files: Mapping[str, str],
    package_dir: str | None = None,
    package_sha256: str | None = None,
    version_file: str | None = None,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Verify a clean Git checkout and return JSON-friendly provenance.

    ``files`` maps checkout-relative paths to their expected SHA256 digests.
    A missing Git repository, a nested checkout, a detached source edit, and
    a source/license/hash mismatch all fail explicitly.
    """

    raw_checkout = Path(root).expanduser()
    if _contains_symlink_component(raw_checkout):
        raise CheckoutIdentityError(
            f"configured checkout root must not contain a symlink: {raw_checkout}"
        )
    checkout = raw_checkout.resolve()
    if not checkout.is_dir():
        raise CheckoutIdentityError(f"external checkout does not exist: {checkout}")
    if not _SHA_RE.fullmatch(expected_revision):
        raise CheckoutIdentityError(
            "expected vendor revision must be a 40-character lowercase Git SHA"
        )
    if expected_tree is not None and not _SHA_RE.fullmatch(expected_tree):
        raise CheckoutIdentityError(
            "expected vendor tree must be a 40-character lowercase Git SHA"
        )

    top = Path(git_output(checkout, "rev-parse", "--show-toplevel")).resolve()
    if top != checkout:
        raise CheckoutIdentityError(f"configured checkout {checkout} is inside Git root {top}")
    revision = git_output(checkout, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise CheckoutIdentityError(
            f"checkout revision {revision!r} is not pinned {expected_revision}"
        )
    tree = git_output(checkout, "rev-parse", "HEAD^{tree}")
    if expected_tree is not None and tree != expected_tree:
        raise CheckoutIdentityError(
            f"checkout tree {tree!r} is not pinned {expected_tree}"
        )
    dirty = git_output(checkout, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise CheckoutIdentityError("pinned external checkout has uncommitted changes")

    origin: str | None = None
    if expected_origin is not None:
        origin = verify_remote_origin(checkout, expected_origin)

    actual_files: dict[str, str] = {}
    mismatches: dict[str, dict[str, str | None]] = {}
    for relative, expected in files.items():
        path = _safe_relative_path(checkout, relative)
        actual = file_sha256(path) if path.is_file() else None
        actual_files[relative] = actual or ""
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}

    package_metadata: dict[str, Any] = {}
    if package_dir is not None:
        package = _safe_relative_path(checkout, package_dir)
        package_files = _safe_package_files(checkout, package_dir)
        if not package_files:
            mismatches[package_dir] = {"expected": package_sha256, "actual": None}
        else:
            digest = hashlib.sha256()
            for path in package_files:
                digest.update(str(path.relative_to(package)).encode("utf-8"))
                digest.update(path.read_bytes())
            actual_package = digest.hexdigest()
            package_metadata = {
                "package_sha256": actual_package,
                "package_file_count": len(package_files),
            }
            if package_sha256 is not None and actual_package != package_sha256:
                mismatches["package_sha256"] = {
                    "expected": package_sha256,
                    "actual": actual_package,
                }

    actual_version: str | None = None
    if version_file is not None and expected_version is not None:
        version_path = _safe_relative_path(checkout, version_file)
        if version_path.is_file():
            text = version_path.read_text(encoding="utf-8")
            match = re.search(r"(?m)^version\s*=\s*[\"']([^\"']+)[\"']", text)
            actual_version = match.group(1) if match else None
        if actual_version != expected_version:
            mismatches["version"] = {
                "expected": expected_version,
                "actual": actual_version,
            }

    result: dict[str, Any] = {
        "path": str(checkout),
        "revision": revision,
        "tree": tree,
        "git_dirty": False,
        "files": actual_files,
        **package_metadata,
    }
    if expected_origin is not None:
        result["origin"] = origin
    if version_file is not None:
        result["version"] = actual_version
    if mismatches:
        details = "; ".join(
            f"{name}: expected={item['expected']!r}, actual={item['actual']!r}"
            for name, item in mismatches.items()
        )
        raise CheckoutIdentityError(f"external checkout identity mismatch ({details})")
    return result


def module_origins(prefix: str) -> dict[str, list[str]]:
    """Return every loaded module path below ``prefix``.

    Optional backends are imported from user supplied checkouts.  Python's
    module cache is process global, so checking only the requested leaf module
    is insufficient: a package or one of its already loaded submodules can
    still come from a different checkout.  This helper intentionally imports
    nothing and is therefore safe during CPU protocol validation.
    """

    import sys

    result: dict[str, list[str]] = {}
    for name, module in tuple(sys.modules.items()):
        if name != prefix and not name.startswith(prefix + "."):
            continue
        origins: list[str] = []
        file_name = getattr(module, "__file__", None)
        if file_name:
            origins.append(str(Path(file_name).resolve()))
        for item in getattr(module, "__path__", ()) or ():
            origins.append(str(Path(item).resolve()))
        result[name] = origins
    return result


def module_origins_inside(prefix: str, root: str | os.PathLike[str]) -> bool:
    """Return whether all loaded ``prefix`` modules originate below ``root``.

    An unloaded package is considered valid.  Once any module under the
    prefix is present, every recorded file and package path must be inside the
    requested source root; an empty origin is rejected because it cannot be
    attributed to the pinned checkout.
    """

    checkout = Path(root).expanduser().resolve()
    for origins in module_origins(prefix).values():
        if not origins:
            return False
        for origin in origins:
            try:
                Path(origin).resolve().relative_to(checkout)
            except (OSError, RuntimeError, ValueError):
                return False
    return True


def require_module_origins(prefix: str, root: str | os.PathLike[str]) -> None:
    """Raise an identity error when loaded package modules mix checkouts."""

    if not module_origins_inside(prefix, root):
        raise CheckoutIdentityError(
            f"loaded {prefix} modules originate outside pinned source {Path(root).resolve()}; "
            "restart the process before rediscovery"
        )


__all__ = [
    "CheckoutIdentityError",
    "candidate_roots",
    "checkout_identity",
    "file_sha256",
    "git_output",
    "normalize_remote_origin",
    "remote_origin",
    "remote_origins",
    "verify_remote_origin",
    "module_origins",
    "module_origins_inside",
    "require_module_origins",
]
