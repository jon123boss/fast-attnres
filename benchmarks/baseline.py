"""Isolated loader for the retained public AttnRes baseline package."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EPS = 2**-23
OPERATOR_PREFIX = "frozen_baseline"
SOURCE_ENV = "ATTNRES_FROZEN_BASELINE_DIR"
_CACHE: dict[str, "FrozenBaseline"] = {}
_NAMESPACE_LITERAL = re.compile(rb"(?P<quote>['\"])attnres::")
_ABSOLUTE_IMPORT = re.compile(
    rb"(?m)^\s*(?:from\s+attnres(?:\s|\.|$)|import\s+attnres(?:\s|,|$))"
)


class FrozenBaselineError(RuntimeError):
    """The explicit retained baseline cannot be imported safely."""


@dataclass(frozen=True)
class FrozenBaseline:
    """Public AttnRes callables loaded from one frozen source hash."""

    attnres: Callable[..., Any]
    metadata: dict[str, Any]
    _temporary_directory: tempfile.TemporaryDirectory[str] = field(repr=False)

    def __call__(
        self,
        values: Any,
        query: Any,
        *,
        eps: float = EPS,
        scale: float = 1.0,
    ) -> Any:
        return self.attnres(values, query, eps=eps, scale=scale)


def _source_files(package: Path) -> list[tuple[str, bytes]]:
    files = sorted(path for path in package.rglob("*.py") if path.is_file())
    if not files or len(files) > 128:
        raise FrozenBaselineError("baseline src/attnres has an invalid Python file count")
    rows = [(path.relative_to(package).as_posix(), path.read_bytes()) for path in files]
    if sum(len(data) for _relative, data in rows) > 8 * 1024 * 1024:
        raise FrozenBaselineError("baseline src/attnres is too large to isolate")
    if any(_ABSOLUTE_IMPORT.search(data) for _relative, data in rows):
        raise FrozenBaselineError("baseline must use relative imports, not absolute attnres imports")
    return rows


def _source_root(source_dir: str | os.PathLike[str] | None) -> tuple[Path, Path]:
    if source_dir is None:
        source_dir = os.environ.get(SOURCE_ENV)
    if not source_dir:
        raise FrozenBaselineError(
            f"an external source checkout is required; pass source_dir or set {SOURCE_ENV}"
        )
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise FrozenBaselineError(f"source checkout does not exist: {root}")
    package = root / "src" / "attnres"
    if not package.is_dir() or not (package / "__init__.py").is_file():
        raise FrozenBaselineError(f"source checkout must expose {root / 'src' / 'attnres'}")
    current = (Path(__file__).resolve().parents[1] / "src" / "attnres").resolve()
    if package.resolve() == current:
        raise FrozenBaselineError("the current package cannot serve as the frozen baseline")
    return root, package


def _adapt(data: bytes, namespace: str) -> tuple[bytes, int]:
    replacement = namespace.encode("ascii") + b"::"
    return _NAMESPACE_LITERAL.subn(
        lambda match: match.group("quote") + replacement,
        data,
    )


def _copy_package(
    rows: list[tuple[str, bytes]], destination: Path, namespace: str
) -> tuple[dict[str, str], dict[str, str], int]:
    original: dict[str, str] = {}
    adapted: dict[str, str] = {}
    rewrites = 0
    for relative, data in rows:
        transformed, count = _adapt(data, namespace)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(transformed)
        original[relative] = hashlib.sha256(data).hexdigest()
        adapted[relative] = hashlib.sha256(transformed).hexdigest()
        rewrites += count
    return original, adapted, rewrites


def load_baseline(source_dir: str | os.PathLike[str] | None = None) -> FrozenBaseline:
    """Load one explicit checkout's public AttnRes API in an isolated namespace."""

    root, package = _source_root(source_dir)
    rows = _source_files(package)
    digest = hashlib.sha256()
    for relative, data in rows:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(data)
    content_hash = digest.hexdigest()
    if content_hash in _CACHE:
        return _CACHE[content_hash]

    module_name = f"_attnres_frozen_{content_hash[:16]}"
    operator_namespace = f"{OPERATOR_PREFIX}_{content_hash[:16]}"
    temporary_directory = tempfile.TemporaryDirectory(prefix="attnres_frozen_")
    adapted_package = Path(temporary_directory.name) / module_name
    try:
        original, adapted, rewrites = _copy_package(rows, adapted_package, operator_namespace)
        sys.path.insert(0, temporary_directory.name)
        package_module = importlib.import_module(module_name)
        attnres = getattr(package_module, "attnres", None)
        if not callable(attnres):
            raise FrozenBaselineError("baseline public API must expose callable attnres")
    except Exception as exc:
        temporary_directory.cleanup()
        if isinstance(exc, FrozenBaselineError):
            raise
        raise FrozenBaselineError(
            f"failed to import frozen baseline from {root}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.path.remove(temporary_directory.name)

    metadata = {
        "source_dir": str(root),
        "content_hash": content_hash,
        "module_namespace": module_name,
        "operator_namespace": operator_namespace,
        "namespace_rewrites": rewrites,
        "original_hashes": original,
        "adapted_hashes": adapted,
    }
    result = FrozenBaseline(attnres, metadata, temporary_directory)
    _CACHE[content_hash] = result
    return result


load_frozen_baseline = load_baseline
