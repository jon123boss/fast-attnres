"""Cache identity tests without importing Modal or provisioning resources."""
import ast
import hashlib
import os
from pathlib import Path
import re

import pytest


@pytest.fixture
def cache_helpers(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "benchmarks/modal_runner.py"
    names = {"_validated_sha256", "_source_fingerprint", "_fingerprint_digest",
             "_cache_namespace", "_safe_component", "_gpu_architecture"}
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in names]
    namespace = {"Path": Path, "hashlib": hashlib, "os": os,
                 "_SHA256_RE": re.compile(r"[0-9a-fA-F]{64}"), "re": re}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    files = ["src/kernel.py", "benchmarks/run.py", "validation/oracle.py",
             "validation/protocol.json", "validation/frozen.json", "fla/fused.py",
             "baseline/kernel.py"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if name.endswith(".json") else "value = 1\n")
    monkeypatch.setenv("ATTNRES_TRANSPORT_SHA256", "a" * 64)

    def fingerprint():
        return namespace["_source_fingerprint"](
            tmp_path, tmp_path / "validation", tmp_path / "fla", tmp_path / "baseline")

    return namespace, fingerprint, tmp_path


@pytest.mark.parametrize("name", [
    "src/kernel.py", "benchmarks/run.py", "validation/oracle.py",
    "validation/protocol.json", "validation/frozen.json", "fla/fused.py",
    "baseline/kernel.py",
])
def test_cache_digest_changes_with_every_compiled_source(cache_helpers, name):
    _, fingerprint, root = cache_helpers
    before = fingerprint()
    assert fingerprint() == before
    (root / name).write_text("changed\n")
    assert fingerprint()["digest"] != before["digest"]


def test_cache_transport_identity_is_read_from_remote_environment(cache_helpers, monkeypatch):
    _, fingerprint, _ = cache_helpers
    before = fingerprint()
    monkeypatch.setenv("ATTNRES_TRANSPORT_SHA256", "b" * 64)
    after = fingerprint()
    assert after["transport_sha256"] == "b" * 64
    assert after["digest"] != before["digest"]
    monkeypatch.delenv("ATTNRES_TRANSPORT_SHA256")
    with pytest.raises(ValueError, match="ATTNRES_TRANSPORT_SHA256"):
        fingerprint()


def test_cache_namespaces_separate_hardware_runtime_and_source(cache_helpers):
    helpers, fingerprint, _ = cache_helpers
    digest = fingerprint()["digest"]
    namespace = helpers["_cache_namespace"]
    first = namespace("H100!", "2.11.0", "3.6.0", digest)
    assert first == namespace("H100!", "2.11.0", "3.6.0", digest)
    assert len({first, namespace("B200", "2.11.0", "3.6.0", digest),
                namespace("H100!", "2.12.1", "3.7.1", digest),
                namespace("H100!", "2.11.0", "3.6.0", "b" * 64)}) == 4
    with pytest.raises(ValueError, match="source fingerprint"):
        namespace("H100!", "2.11.0", "3.6.0", "unknown")


def test_modal_release_runtime_defaults_match_measured_environment():
    source_path = Path(__file__).parents[1] / "benchmarks/modal_runner.py"
    source = source_path.read_text()
    assert 'os.environ.get("ATTNRES_TORCH_VERSION", "2.11.0")' in source
    assert 'os.environ.get("ATTNRES_TRITON_VERSION", "3.6.0")' in source
    assert 'TORCH_VERSION != "2.11.0" or TRITON_VERSION != "3.6.0"' in source
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_runtime"
    )
    namespace = {"TORCH_VERSION": "2.11.0", "TRITON_VERSION": "3.6.0"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)

    class Module:
        def __init__(self, version=None):
            if version is not None:
                self.__version__ = version

    report = namespace["_validate_runtime"](
        Module("2.11.0+cu130"), Module("3.6.0")
    )
    assert report["status"] == "verified"
    assert report["actual"] == {"torch": "2.11.0+cu130", "triton": "3.6.0"}
    with pytest.raises(RuntimeError, match="version is unavailable"):
        namespace["_validate_runtime"](Module("2.11.0+cu130"), Module())
    with pytest.raises(RuntimeError, match="version mismatch"):
        namespace["_validate_runtime"](Module("2.10.0"), Module("3.6.0"))


def test_modal_fla_transport_uses_host_preflight_and_canonical_remote_root():
    source = (Path(__file__).parents[1] / "benchmarks/modal_runner.py").read_text()
    assert "_FLA_HOST_PREFLIGHT = fla_checkout_metadata(PROJECT, FLA)" in source
    assert 'json.dumps(_FLA_HOST_PREFLIGHT, sort_keys=True)' in source
    assert 'image.add_local_dir(FLA / "fla", "/workspace/fla/fla"' in source
    assert 'image.add_local_dir(FLA / ".git"' not in source
    assert 'verify_mounted_fla_checkout(\n                release_fla["checkout"],\n                "/workspace/fla",' in source
