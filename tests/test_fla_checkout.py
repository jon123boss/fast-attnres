"""CPU-safe strict identity checks for the release FLA checkout contract."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import json

import pytest

from benchmarks.fla_checkout import (
    FLA_CHECKOUT_ENVIRONMENT,
    FLA_CHECKOUT_LAYOUT,
    fla_checkout_metadata,
    validate_fla_checkout_spec,
    validate_release_fla_config,
    verify_fla_checkout,
    verify_mounted_fla_checkout,
    verify_release_fla_config,
    verify_runtime_fla_config,
)
from benchmarks.fla_compile import resolve_vendor_root


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "AttnRes Test",
            "GIT_AUTHOR_EMAIL": "attnres@example.invalid",
            "GIT_COMMITTER_NAME": "AttnRes Test",
            "GIT_COMMITTER_EMAIL": "attnres@example.invalid",
        },
    )
    return completed.stdout.strip()


def _fake_checkout(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "flash-linear-attention"
    package = root / "fla"
    (package / "ops/attnres").mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "ops/__init__.py").write_text("\n", encoding="utf-8")
    (package / "ops/attnres/__init__.py").write_text("\n", encoding="utf-8")
    (package / "ops/attnres/fused.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    _git(root, "remote", "add", "origin", "https://github.com/fla-org/flash-linear-attention.git")
    metadata = fla_checkout_metadata(configured=root)
    expected = {
        "environment": FLA_CHECKOUT_ENVIRONMENT,
        "layout": FLA_CHECKOUT_LAYOUT,
        "revision": metadata["revision"],
        "package_sha256": metadata["package_sha256"],
        "required_clean": True,
    }
    return root, expected


def test_clean_checkout_revision_and_package_digest_are_verified(tmp_path):
    root, expected = _fake_checkout(tmp_path)
    result = verify_fla_checkout(expected, configured=root)
    assert result["status"] == "verified"
    assert result["expected"] == expected
    assert result["actual"]["revision"] == expected["revision"]
    assert result["actual"]["package_sha256"] == expected["package_sha256"]
    assert result["actual"]["git_dirty"] is False


def test_configured_subdirectory_is_not_accepted_as_checkout_identity(tmp_path):
    root, _ = _fake_checkout(tmp_path)
    nested = root / "nested"
    shutil.copytree(root / "fla", nested / "fla")
    with pytest.raises(RuntimeError, match="is inside checkout"):
        fla_checkout_metadata(configured=nested)


def test_native_compile_resolver_honors_the_release_checkout_environment(
    tmp_path, monkeypatch
):
    root, _ = _fake_checkout(tmp_path)
    for name in ("FLA_ROOT", "FLASH_LINEAR_ATTENTION_ROOT", "VENDOR_FLA_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ATTNRES_FLA_DIR", str(root))
    assert resolve_vendor_root() == root.resolve()


def test_missing_release_checkout_environment_does_not_fall_back_to_defaults(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ATTNRES_FLA_DIR", str(tmp_path / "configured-missing"))
    monkeypatch.delenv("FLA_ROOT", raising=False)
    monkeypatch.delenv("FLASH_LINEAR_ATTENTION_ROOT", raising=False)
    monkeypatch.delenv("VENDOR_FLA_ROOT", raising=False)
    with pytest.raises(ImportError, match="not found"):
        resolve_vendor_root()


def test_mounted_package_bytes_are_bound_to_clean_host_preflight(tmp_path):
    root, expected = _fake_checkout(tmp_path)
    host = fla_checkout_metadata(configured=root)
    mounted = tmp_path / "mounted"
    shutil.copytree(root / "fla", mounted / "fla")

    result = verify_mounted_fla_checkout(expected, mounted, host)
    assert result["status"] == "verified"
    assert result["actual"]["package_file_count"] == host["package_file_count"]

    (mounted / "fla/ops/attnres/fused.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = verify_mounted_fla_checkout(expected, mounted, host)
    assert changed["status"] == "failed"
    assert "transport_package_sha256" in changed["error"]["mismatches"]

    clean_mounted = tmp_path / "clean-mounted"
    shutil.copytree(root / "fla", clean_mounted / "fla")
    dirty = verify_mounted_fla_checkout(expected, clean_mounted, {**host, "git_dirty": True})
    assert dirty["status"] == "failed"
    assert "git_dirty" in dirty["error"]["mismatches"]

    wrong_origin = verify_mounted_fla_checkout(
        expected,
        clean_mounted,
        {**host, "origin": "https://github.com/example/not-fla.git"},
    )
    assert wrong_origin["status"] == "failed"
    assert "origin" in wrong_origin["error"]["mismatches"]

    missing_origin = verify_mounted_fla_checkout(
        expected,
        clean_mounted,
        {key: value for key, value in host.items() if key != "origin"},
    )
    assert missing_origin["status"] == "failed"
    assert "unexpected schema" in missing_origin["error"]["message"]


def test_runtime_verifier_accepts_only_the_host_bound_mounted_bytes(tmp_path, monkeypatch):
    root, expected = _fake_checkout(tmp_path)
    host = fla_checkout_metadata(configured=root)
    mounted = tmp_path / "runtime-mounted"
    shutil.copytree(root / "fla", mounted / "fla")
    config = {
        "include_fla_compile": True,
        "fla_compile_backends": ["triton"],
        "production_ladder": {
            "fla_checkout": expected,
            "fla_anchor": {
                "implementation": "triton",
                "checkpoint_level": 1,
                "rank": 1024,
                "scope": "R=D anchor only",
            },
        },
    }
    monkeypatch.setenv("ATTNRES_FLA_DIR", str(mounted))
    monkeypatch.setenv("ATTNRES_FLA_HOST_PREFLIGHT", json.dumps(host))
    result = verify_runtime_fla_config(config, configured=root / "wrong-host-path")
    assert result["status"] == "verified"
    assert result["actual"]["path"] == str(mounted.resolve())

    (mounted / "fla/ops/attnres/fused.py").write_text("VALUE = 7\n", encoding="utf-8")
    changed = verify_runtime_fla_config(config)
    assert changed["status"] == "failed"
    assert "transport_package_sha256" in changed["error"]["mismatches"]


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda spec: spec.update(required_clean=1), "required_clean"),
        (lambda spec: spec.update(package_sha256="A" * 64), "package_sha256"),
        (lambda spec: spec.update(extra="rejected"), "keys must be exactly"),
    ],
)
def test_checkout_schema_rejects_wrong_types_and_unknown_keys(mutator, message):
    spec = {
        "environment": FLA_CHECKOUT_ENVIRONMENT,
        "layout": FLA_CHECKOUT_LAYOUT,
        "revision": "a" * 40,
        "package_sha256": "b" * 64,
        "required_clean": True,
    }
    mutator(spec)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_fla_checkout_spec(spec)


def test_revision_and_digest_mismatch_fail_with_structured_diagnostics(tmp_path):
    root, expected = _fake_checkout(tmp_path)
    wrong_revision = dict(expected, revision="0" * 40)
    result = verify_fla_checkout(wrong_revision, configured=root)
    assert result["status"] == "failed"
    assert result["error"]["type"] == "FLACheckoutMismatch"
    assert "revision" in result["error"]["mismatches"]

    (root / "fla/ops/attnres/fused.py").write_text("VALUE = 2\n", encoding="utf-8")
    digest_result = verify_fla_checkout(dict(expected, required_clean=False), configured=root)
    assert digest_result["status"] == "failed"
    assert "package_sha256" in digest_result["error"]["mismatches"]

    dirty_result = verify_fla_checkout(expected, configured=root)
    assert dirty_result["status"] == "failed"
    assert dirty_result["actual"]["git_dirty"] is True
    assert "git_dirty" in dirty_result["error"]["mismatches"]


def test_active_release_claims_are_triton_checkpoint_one_only():
    checkout = {
        "environment": FLA_CHECKOUT_ENVIRONMENT,
        "layout": FLA_CHECKOUT_LAYOUT,
        "revision": "a" * 40,
        "package_sha256": "b" * 64,
        "required_clean": True,
    }
    config = {
        "include_fla_compile": True,
        "include_fla_model": False,
        "fla_compile_backends": ["triton"],
        "production_ladder": {
            "fla_checkout": checkout,
            "fla_anchor": {
                "implementation": "triton",
                "checkpoint_level": 1,
                "rank": 1024,
                "scope": "R=D anchor only",
            },
        },
    }
    release = validate_release_fla_config(config)
    assert release["checkout"] == checkout
    assert release["anchor"]["checkpoint_level"] == 1

    for invalid in (
        {**config, "fla_compile_backends": ["gluon"]},
        {**config, "production_ladder": {
            **config["production_ladder"],
            "fla_anchor": {**config["production_ladder"]["fla_anchor"], "checkpoint_level": 0},
        }},
        {**config, "include_fla_model": True},
    ):
        with pytest.raises((TypeError, ValueError)):
            validate_release_fla_config(invalid)


def test_release_verification_is_fail_closed_and_machine_readable(tmp_path):
    config = {
        "include_fla_compile": True,
        "fla_compile_backends": ["triton"],
        "production_ladder": {
            "fla_checkout": {
                "environment": FLA_CHECKOUT_ENVIRONMENT,
                "layout": FLA_CHECKOUT_LAYOUT,
                "revision": "a" * 40,
                "package_sha256": "b" * 64,
                "required_clean": True,
            },
            "fla_anchor": {
                "implementation": "triton",
                "checkpoint_level": 1,
                "rank": 1024,
                "scope": "R=D anchor only",
            },
        },
    }
    result = verify_release_fla_config(config, project_root=tmp_path, configured=tmp_path / "missing")
    assert result["status"] == "failed"
    assert result["actual"] is None
    assert result["error"]["type"] == "FileNotFoundError"


def test_release_run_stops_before_device_work_when_checkout_is_missing(tmp_path, monkeypatch):
    from benchmarks import run as benchmark_run

    config = {
        "scope": "custom",
        "project_root": str(tmp_path),
        "include_fla_compile": True,
        "fla_compile_backends": ["triton"],
        "production_ladder": {
            "fla_checkout": {
                "environment": FLA_CHECKOUT_ENVIRONMENT,
                "layout": FLA_CHECKOUT_LAYOUT,
                "revision": "a" * 40,
                "package_sha256": "b" * 64,
                "required_clean": True,
            },
            "fla_anchor": {
                "implementation": "triton",
                "checkpoint_level": 1,
                "rank": 1024,
                "scope": "R=D anchor only",
            },
        },
    }
    monkeypatch.setattr(benchmark_run, "_environment", lambda root: {"project": str(root)})
    monkeypatch.setattr(
        benchmark_run,
        "load_protocol",
        lambda root: ({"version": "test", "seeds": [1], "ranks": [1]}, {}),
    )

    def unexpected_device_work(*args, **kwargs):
        raise AssertionError("device inspection must not run before FLA preflight")

    monkeypatch.setattr(benchmark_run, "_device_info", unexpected_device_work)
    result = benchmark_run.run_suite(config)
    assert result["status"] == "failed"
    assert result["fla_checkout"]["status"] == "failed"
    assert result["failures"][0]["phase"] == "fla_checkout_preflight"


def test_release_run_passes_the_verified_root_to_the_compile_path(tmp_path, monkeypatch):
    from benchmarks import run as benchmark_run

    root, expected = _fake_checkout(tmp_path)
    config = {
        "scope": "custom",
        "phases": [],
        "project_root": str(tmp_path),
        "vendor_root": str(root),
        "include_fla": False,
        "include_fla_compile": True,
        "fla_compile_backends": ["triton"],
        "production_ladder": {
            "fla_checkout": expected,
            "fla_anchor": {
                "implementation": "triton",
                "checkpoint_level": 1,
                "rank": 1024,
                "scope": "R=D anchor only",
            },
        },
    }
    monkeypatch.setattr(benchmark_run, "_environment", lambda path: {"project": str(path)})
    monkeypatch.setattr(
        benchmark_run,
        "load_protocol",
        lambda path: ({"version": "test", "seeds": [1], "ranks": [1]}, {}),
    )
    monkeypatch.setattr(benchmark_run, "_device_info", lambda *args: {
        "requested": "cuda", "type": "cuda", "available": False, "count": 0
    })
    monkeypatch.setattr(benchmark_run, "_operator_cases", lambda *args: ([], []))
    monkeypatch.setattr(benchmark_run, "_model_config", lambda *args: {})
    monkeypatch.setattr(
        benchmark_run,
        "_source_hashes",
        lambda *args: {"software_hash": "test-software-hash"},
    )
    monkeypatch.setattr(benchmark_run, "_hardware_hash", lambda *args: "test-hardware-hash")

    result = benchmark_run.run_suite(config)
    assert result["fla_checkout"]["status"] == "verified"
    assert result["config"]["vendor_root"] == str(root.resolve())
