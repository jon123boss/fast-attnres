"""CPU and static regressions for the optional FLA/Liger adapters.

These tests deliberately never call a native kernel.  They cover identity
resolution, fail-closed discovery, input envelopes, and metadata that records
the common standard AttnRes qualification boundary.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks import competitors, liger
from benchmarks import gluon_compat
from benchmarks.vendor_identity import (
    CheckoutIdentityError,
    candidate_roots,
    checkout_identity,
    normalize_remote_origin,
    remote_origin,
    verify_remote_origin,
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "AttnRes adapter test",
            "GIT_AUTHOR_EMAIL": "attnres@example.invalid",
            "GIT_COMMITTER_NAME": "AttnRes adapter test",
            "GIT_COMMITTER_EMAIL": "attnres@example.invalid",
        },
    )
    return result.stdout.strip()


def test_external_identity_helpers_are_dependency_free_and_fail_closed(tmp_path, monkeypatch):
    root = tmp_path / "vendor"
    root.mkdir()
    (root / "marker.txt").write_text("source\n", encoding="utf-8")
    package = root / "fixture_package"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    revision = _git(root, "rev-parse", "HEAD")

    identity = checkout_identity(
        root,
        expected_revision=revision,
        files={"marker.txt": hashlib.sha256(b"source\n").hexdigest()},
    )
    assert identity["revision"] == revision
    assert identity["git_dirty"] is False

    package_digest = hashlib.sha256()
    package_digest.update(b"__init__.py")
    package_digest.update(b"VALUE = 1\n")
    package_identity = checkout_identity(
        root,
        expected_revision=revision,
        files={"marker.txt": hashlib.sha256(b"source\n").hexdigest()},
        package_dir="fixture_package",
        package_sha256=package_digest.hexdigest(),
    )
    assert package_identity["package_file_count"] == 1
    assert package_identity["package_sha256"] == package_digest.hexdigest()

    monkeypatch.setenv("TEST_VENDOR_A", str(tmp_path / "wrong"))
    monkeypatch.setenv("TEST_VENDOR_B", str(root))
    assert candidate_roots(
        tmp_path,
        None,
        environment=("TEST_VENDOR_A", "TEST_VENDOR_B"),
        defaults=(root,),
    ) == ((tmp_path / "wrong").resolve(),)

    with pytest.raises(CheckoutIdentityError, match="identity mismatch"):
        checkout_identity(
            root,
            expected_revision=revision,
            files={"marker.txt": "0" * 64},
        )

    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    symlink_root = tmp_path / "symlink-vendor"
    symlink_root.mkdir()
    (symlink_root / "marker.txt").symlink_to(outside)
    _git(symlink_root, "init", "-q")
    _git(symlink_root, "add", ".")
    _git(symlink_root, "commit", "-qm", "symlink fixture")
    symlink_revision = _git(symlink_root, "rev-parse", "HEAD")
    with pytest.raises(CheckoutIdentityError, match="symlink"):
        checkout_identity(
            symlink_root,
            expected_revision=symlink_revision,
            files={"marker.txt": hashlib.sha256(b"outside\n").hexdigest()},
        )

    linked_root = tmp_path / "linked-vendor"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(CheckoutIdentityError, match="symlink"):
        checkout_identity(
            linked_root,
            expected_revision=revision,
            files={"marker.txt": "0" * 64},
        )


def test_expected_origin_rejects_missing_wrong_and_ambiguous_remotes(tmp_path):
    root = tmp_path / "origin-vendor"
    root.mkdir()
    (root / "marker.txt").write_text("source\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    revision = _git(root, "rev-parse", "HEAD")
    expected = "https://github.com/example/vendor.git"
    _git(root, "remote", "add", "origin", expected)

    identity = checkout_identity(
        root,
        expected_revision=revision,
        expected_origin=expected,
        files={"marker.txt": hashlib.sha256(b"source\n").hexdigest()},
    )
    assert identity["origin"] == expected
    assert remote_origin(root) == expected

    with pytest.raises(CheckoutIdentityError, match="does not match pinned origin"):
        verify_remote_origin(root, "https://github.com/example/other.git")

    _git(root, "remote", "set-url", "--add", "origin", "https://github.com/example/other.git")
    with pytest.raises(CheckoutIdentityError, match="ambiguous origin URLs"):
        checkout_identity(
            root,
            expected_revision=revision,
            expected_origin=expected,
            files={"marker.txt": hashlib.sha256(b"source\n").hexdigest()},
        )

    missing = tmp_path / "missing-origin"
    missing.mkdir()
    (missing / "marker.txt").write_text("source\n", encoding="utf-8")
    _git(missing, "init", "-q")
    _git(missing, "add", ".")
    _git(missing, "commit", "-qm", "fixture")
    missing_revision = _git(missing, "rev-parse", "HEAD")
    with pytest.raises(CheckoutIdentityError, match="no configured origin URL"):
        checkout_identity(
            missing,
            expected_revision=missing_revision,
            expected_origin=expected,
            files={"marker.txt": hashlib.sha256(b"source\n").hexdigest()},
        )


def test_remote_origin_normalization_is_conservative():
    assert normalize_remote_origin("HTTPS://GitHub.com/example/vendor.GIT/") == (
        "https://github.com/example/vendor"
    )
    assert normalize_remote_origin(" https://github.com/example/vendor.git") is None
    assert normalize_remote_origin("git@github.com:example/vendor.git") is None


def test_liger_import_does_not_import_triton():
    source = Path(liger.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "triton" not in imported_modules
    assert "triton" not in source.split("def _load_native", 1)[0]
    assert "validation.oracle.oracle" in source


def test_liger_revision_and_source_identity_are_exact():
    assert liger.LIGER_TAG == "v0.8.2"
    assert liger.LIGER_VERSION == "0.8.2"
    assert liger.LIGER_REVISION == "000be60929938fd1358e03524c6ab398b6d421bd"
    assert liger.LIGER_TREE == "746af1fc03014cf47cad895d01cf0d23fddf5e75"
    assert liger.LIGER_REPOSITORY == "https://github.com/linkedin/Liger-Kernel.git"
    assert liger.LIGER_SOURCE_SHA256 == (
        "57da6fed98f794088b2a56223e6c7ef9fc920824f0c483cb0ef0b5a343dab0b1"
    )
    assert liger.LIGER_LICENSE_SHA256 == (
        "3a1ccb0c7274b68e1af2ca1d54b10b662085ca56753400182ecf87ae33f2d1a8"
    )
    assert liger.LIGER_NOTICE_SHA256 == (
        "9e3c27a0f64b87d00df12250cf1bc218b1e2fbc5fffc0bd64737ba8e8357218f"
    )
    assert liger.LIGER_PYPROJECT_SHA256 == (
        "f55effccdecc17ca87357ed8ecd4e73a58b1a56ee275367bfe5db2827dc9ac22"
    )
    assert liger.LIGER_SOURCE == "src/liger_kernel/ops/attn_res.py"
    assert competitors.FLA_REVISION == "5e02dd3a7651f5f2797eb8b12bbec401826031e1"
    assert competitors.FLA_TREE == "7e4199902fb291c78b3937f223b08ae7bca82bb1"
    assert competitors.FLA_REPOSITORY == "https://github.com/fla-org/flash-linear-attention.git"
    assert competitors.FLA_PACKAGE_SHA256 == (
        "2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781"
    )


def test_liger_parameter_free_key_weight_stays_fp32_for_bf16_storage():
    query = torch.ones(8, dtype=torch.bfloat16)
    weight = liger._ones_weight(query)
    assert weight.dtype == torch.float32
    assert weight.device == query.device
    torch.testing.assert_close(weight, torch.ones_like(weight))


def test_liger_custom_op_does_not_return_vendor_input_alias(monkeypatch):
    values = torch.randn(2, 3, 8)
    query = torch.randn(8)
    weight = torch.ones(8)
    output = torch.randn(3, 8)
    alpha = torch.randn(3, 2)
    rstd = torch.randn(3, 2)

    module = SimpleNamespace(
        attn_res_forward=lambda actual, *_args: (
            output,
            actual.reshape(2, 3, 8),
            alpha,
            rstd,
        )
    )
    monkeypatch.setattr(liger, "_load_native", lambda _root: module)
    result = liger._forward_native(values, query, weight, "/fixture")

    assert result == (output, alpha, rstd)
    assert len(result) == 3
    assert len(__import__("inspect").signature(liger._registered_backward).parameters) == 4


def test_gluon_barrier_compatibility_is_explicit_and_fail_closed(monkeypatch):
    barrier = lambda: None
    language = SimpleNamespace(barrier=barrier)

    def import_module(name):
        if name == "triton":
            return SimpleNamespace(__version__="3.7.1")
        if name == "triton.experimental.gluon.language":
            return language
        pytest.fail(f"unexpected import {name}")

    monkeypatch.setattr(
        gluon_compat.importlib,
        "import_module",
        import_module,
    )
    metadata = gluon_compat.install_gluon_barrier_compatibility()
    assert metadata["mode"] == "thread_barrier_alias_to_barrier"
    assert metadata["alias_preserves_builtin_identity"] is True
    assert language.thread_barrier is barrier
    assert gluon_compat.install_gluon_barrier_compatibility()["mode"] == (
        "thread_barrier_alias_to_barrier"
    )

    native = SimpleNamespace(thread_barrier=lambda: None)
    monkeypatch.setattr(
        gluon_compat.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="3.6.0")
        if name == "triton"
        else native,
    )
    assert gluon_compat.install_gluon_barrier_compatibility()["mode"] == (
        "native_thread_barrier"
    )

    missing = SimpleNamespace()
    monkeypatch.setattr(
        gluon_compat.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="3.7.1")
        if name == "triton"
        else missing,
    )
    with pytest.raises(ImportError, match="neither thread_barrier nor barrier"):
        gluon_compat.install_gluon_barrier_compatibility()

    unknown_language = SimpleNamespace(barrier=barrier)
    monkeypatch.setattr(
        gluon_compat.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="3.8.0")
        if name == "triton"
        else unknown_language,
    )
    with pytest.raises(ImportError, match="validated only for Triton 3.7.1"):
        gluon_compat.install_gluon_barrier_compatibility()

    unknown_native = SimpleNamespace(thread_barrier=lambda: None)
    monkeypatch.setattr(
        gluon_compat.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="3.8.0")
        if name == "triton"
        else unknown_native,
    )
    with pytest.raises(ImportError, match="validated only for Triton 3.6.0"):
        gluon_compat.install_gluon_barrier_compatibility()

    marked = SimpleNamespace(
        barrier=barrier,
        thread_barrier=barrier,
        __attnres_thread_barrier_compatibility__={
            "mode": "thread_barrier_alias_to_barrier",
            "triton_version": "3.7.1",
        },
    )
    monkeypatch.setattr(
        gluon_compat.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="3.8.0")
        if name == "triton"
        else marked,
    )
    with pytest.raises(ImportError, match="alias was modified"):
        gluon_compat.install_gluon_barrier_compatibility()


def test_gluon_discovery_installs_barrier_shim_before_any_fla_import(monkeypatch, tmp_path):
    """The Triton 3.7.1 bridge must exist before FLA parent packages execute."""

    events = []
    vendor = tmp_path / "fla-vendor"
    identity = {"revision": competitors.FLA_REVISION, "origin": competitors.FLA_REPOSITORY}

    monkeypatch.setattr(competitors, "find_vendor_root", lambda *_args: vendor)
    monkeypatch.setattr(competitors, "_git_revision", lambda _root: competitors.FLA_REVISION)
    monkeypatch.setattr(competitors, "_fla_identity", lambda _root: dict(identity))
    monkeypatch.setattr(
        competitors,
        "install_gluon_barrier_compatibility",
        lambda: events.append("shim")
        or {
            "mode": "thread_barrier_alias_to_barrier",
            "triton_version": "3.7.1",
            "applied_before_vendor_import": True,
        },
    )

    fused = SimpleNamespace(fused_attnres=lambda **_kwargs: None)
    gluon = SimpleNamespace(
        AttnResGluonBackend=lambda: SimpleNamespace(fused_attnres=lambda **_kwargs: None)
    )

    def fake_native(_vendor, module_name, _relative_path):
        events.append(f"import:{module_name}")
        return fused if module_name.endswith(".fused") else gluon

    monkeypatch.setattr(competitors, "_native_module", fake_native)
    result = competitors.discover_comparators(tmp_path, vendor)

    assert events[0] == "shim"
    assert events.index("shim") < events.index("import:fla.ops.attnres.fused")
    assert events.index("shim") < events.index("import:fla.ops.attnres.backends.gluon")
    assert result["fla_gluon"].status == "available"


def test_gluon_adapter_metadata_and_envelope_helper_are_explicit():
    comparator = competitors.Comparator(
        "fla_gluon",
        lambda **_kwargs: None,
        status="available",
        kind="gluon",
    )
    metadata = comparator.describe()
    assert metadata["compile_envelope"] == competitors.GLUON_COMPILE_ENVELOPE
    assert metadata["compile_envelope"]["max_padded_width"] == 4096
    assert competitors._gluon_compile_envelope(33, 2048)[0] is True
    accepted, reason, metrics = competitors._gluon_compile_envelope(129, 8192)
    assert accepted is False
    assert "BD=8192" in (reason or "")
    assert metrics["padded_width"] == 8192


def test_configured_liger_path_never_falls_back(tmp_path, monkeypatch):
    # A missing explicitly configured root must remain missing even if another
    # candidate is placed under the project root.
    fallback = tmp_path / "vendor" / "Liger-Kernel"
    (fallback / "src/liger_kernel/ops").mkdir(parents=True)
    (fallback / liger.LIGER_SOURCE).write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setenv("LIGER_ROOT", str(tmp_path / "configured-missing"))
    result = liger.discover_comparator(project_root=tmp_path)
    assert result.status == "missing"
    assert "not found" in (result.reason or "")
    assert result.call is None


def test_liger_discovery_records_missing_vendor_without_reference_fallback(tmp_path):
    result = liger.discover_comparator(project_root=tmp_path, vendor_root=tmp_path / "missing")
    assert result.status == "missing"
    assert result.call is None
    assert "pinned Liger" in (result.reason or "")
    description = result.describe()
    assert description["name"] == "liger"
    assert description["status"] == "missing"


def test_liger_capabilities_reject_sliced_and_cpu_inputs():
    values = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    query = torch.randn(4, dtype=torch.bfloat16)
    comparator = liger.Comparator(lambda *_args: None, status="available")
    okay, reason = comparator.applicable(values, query)
    assert not okay
    assert "standard R=D" in (reason or "")

    full_query = torch.randn(8, dtype=torch.bfloat16)
    okay, reason = comparator.applicable(values, full_query)
    assert not okay
    assert "CUDA" in (reason or "")


def test_fla_capabilities_reject_reduced_rank_and_overflow_sources():
    reduced_values = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    reduced_query = torch.randn(4, dtype=torch.bfloat16)
    comparator = competitors.Comparator("fake", lambda **_kwargs: None, status="available")
    okay, reason = comparator.applicable(reduced_values, reduced_query)
    assert not okay
    assert "standard R=D" in (reason or "")

    overflow = torch.randn(competitors.FLA_MAX_SOURCES + 1, 1, 8, dtype=torch.bfloat16)
    full_query = torch.randn(8, dtype=torch.bfloat16)
    okay, reason = comparator.applicable(overflow, full_query)
    assert not okay
    assert "129" in (reason or "")


def test_metadata_exposes_common_oracle_and_limits_for_missing_adapters(tmp_path):
    fla = competitors.vendor_metadata(project_root=tmp_path, vendor_root=tmp_path / "missing")
    liger_metadata = liger.vendor_metadata(project_root=tmp_path, vendor_root=tmp_path / "missing")
    assert fla["status"] == "missing"
    assert liger_metadata["status"] == "missing"
    assert fla["qualification_oracle"] == "validation.oracle.oracle"
    assert liger_metadata["expected_revision"] == liger.LIGER_REVISION
    assert "expected_source_sha256" in liger_metadata


def test_cpu_import_has_no_native_runtime_side_effects():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.liger; print('triton' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{project_root / 'src'}:{project_root}",
        },
    )
    assert result.stdout.strip() == "False"
