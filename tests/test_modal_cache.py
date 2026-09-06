"""Cache identity tests without importing Modal or provisioning resources."""
import ast
import hashlib
import os
from pathlib import Path
import re
import tarfile

import pytest


def test_parallel_bf16_cache_writers_use_distinct_paths(tmp_path):
    source = Path(__file__).parents[1] / "benchmarks/bf16_modal.py"
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_compiler_cache_paths")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    paths = namespace["_compiler_cache_paths"]
    assert paths(tmp_path, "first")[0] is None
    old = tmp_path / "artifacts.tar.gz"
    old.write_bytes(b"legacy cache")
    first_in, first_out = paths(tmp_path, "first")
    second_in, second_out = paths(tmp_path, "second")
    assert first_in == second_in == old
    assert first_out != second_out
    first_out.parent.mkdir()
    first_out.with_suffix(".pending").write_bytes(b"incomplete")
    assert paths(tmp_path, "third")[0] == old
    first_out.write_bytes(b"completed cache")
    os.utime(first_out, ns=(old.stat().st_mtime_ns + 1_000_000,) * 2)
    assert paths(tmp_path, "third")[0] == first_out


def test_bf16_cache_backup_detects_completed_changes_and_ignores_temporary_files(tmp_path):
    from benchmarks.bf16_cache import compiler_cache_stamp
    stamp = lambda: compiler_cache_stamp([tmp_path])
    artifact = tmp_path / "kernel.cubin"
    artifact.write_bytes(b"first")
    before = stamp()
    for name in ("locks/compile", "tmp.build/kernel.cubin", "kernel.tmp", "__pycache__/x.pyc"):
        temporary = tmp_path / name
        temporary.parent.mkdir(exist_ok=True)
        temporary.write_bytes(b"in progress")
    assert stamp() == before
    original = artifact.stat()
    artifact.write_bytes(b"other")
    os.utime(artifact, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
    assert stamp() != before
    before = stamp()
    artifact.unlink()
    assert stamp() != before


def test_bf16_archive_survives_interrupted_replacement(tmp_path, monkeypatch):
    from benchmarks import bf16_cache
    root = tmp_path / "triton"
    root.mkdir()
    (root / "kernel.cubin").write_bytes(b"compiled")
    (root / "kernel.tmp").write_bytes(b"unfinished")
    destination = tmp_path / "saved/artifacts.tar.gz"
    stamp = bf16_cache.save_compiler_cache([root], destination)
    with tarfile.open(destination) as archive:
        assert archive.extractfile("triton/kernel.cubin").read() == b"compiled"
        assert "triton/kernel.tmp" not in archive.getnames()
    previous = destination.read_bytes()
    modified = destination.stat().st_mtime_ns
    assert bf16_cache.save_compiler_cache([root], destination, stamp) == stamp
    assert destination.stat().st_mtime_ns == modified
    (root / "kernel.cubin").write_bytes(b"new compiled version")

    def interrupted_copy(source, target):
        target.write_bytes(b"partial upload")
        raise OSError("interrupted")

    monkeypatch.setattr(bf16_cache.shutil, "copyfile", interrupted_copy)
    with pytest.raises(OSError, match="interrupted"):
        bf16_cache.save_compiler_cache([root], destination, stamp)
    assert destination.read_bytes() == previous
    assert not list(destination.parent.glob("*.pending"))


def test_training_cache_checkpoints_precede_timing_and_ignore_timing_updates(tmp_path, monkeypatch):
    import json
    from benchmarks.bf16_training_worker import training_checkpoint
    roots = [tmp_path / name for name in ("triton", "inductor")]
    for root, variable in zip(roots, ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")):
        root.mkdir()
        monkeypatch.setenv(variable, str(root))
    output, archive = tmp_path / "report.json", tmp_path / "cache/artifacts.tar.gz"
    commits = []
    checkpoint = training_checkpoint(output, archive, lambda: commits.append(archive.read_bytes()))
    report = {"in_progress": {"seed": 7, "case": {"rank": 384}, "arms": {}}}
    checkpoint(report)
    assert not archive.exists()
    (roots[0] / "kernel.cubin").write_bytes(b"qualified arm")
    report["in_progress"]["arms"]["candidate"] = {"status": "qualified"}
    checkpoint(report)
    assert len(commits) == 1 and json.loads(output.read_text()) == report
    previous = archive.read_bytes()
    # Even a compiler-file change must not trigger a backup from timed updates.
    (roots[0] / "kernel.cubin").write_bytes(b"later change")
    report["in_progress"]["arms"]["candidate"] = {"status": "passed", "samples_ms": [1.]}
    checkpoint(report)
    assert len(commits) == 1 and archive.read_bytes() == previous
    assert json.loads(output.read_text()) == report


def test_training_backup_errors_are_retained_without_losing_results(tmp_path, monkeypatch):
    import json
    from benchmarks.bf16_training_worker import training_checkpoint
    for variable in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.setenv(variable, str(tmp_path / variable))
        Path(os.environ[variable]).mkdir()
    output = tmp_path / "report.json"

    def failed_commit():
        raise OSError("commit unavailable")

    checkpoint = training_checkpoint(output, tmp_path / "cache/artifacts.tar.gz", failed_commit)
    (Path(os.environ["TRITON_CACHE_DIR"]) / "kernel.cubin").write_bytes(b"compiled")
    report = {"in_progress": {"seed": 7, "case": {},
                              "arms": {"candidate": {"status": "qualified"}}}}
    checkpoint(report)
    saved = json.loads(output.read_text())
    assert saved["compiler_backup_error"] == "OSError: commit unavailable"
    assert saved["in_progress"] == report["in_progress"]


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
