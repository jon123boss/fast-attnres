"""Cache publication, interruption and union-restore behavior without a GPU."""
import json
from pathlib import Path
import shutil

import pytest

from benchmarks.bf16_cache_store import CompilerCache


@pytest.fixture
def cache(tmp_path):
    roots = {name: tmp_path / name for name in ("triton", "inductor")}
    for root in roots.values():
        root.mkdir()
    return CompilerCache(tmp_path / "store", roots, {"sm": 100, "toolchain": "pinned"})


def kernel(cache, key, contents=b"compiled"):
    directory = cache.roots["triton"] / key
    directory.mkdir(parents=True, exist_ok=True)
    child = directory / "kernel.cubin"
    child.write_bytes(contents)
    (directory / "__grp__kernel.json").write_text(json.dumps({"child_paths": {"kernel.cubin": str(child)}}))
    return child


def clear_roots(cache):
    for root in cache.roots.values():
        shutil.rmtree(root)
        root.mkdir()


def test_union_restore_keeps_independent_jobs_and_complete_newest_groups(cache):
    kernel(cache, "a", b"a")
    kernel(cache, "same", b"old")
    cache.save()
    clear_roots(cache)
    second = CompilerCache(cache.directory.parent, cache.roots, cache.identity["runtime"])
    kernel(second, "b", b"b")
    kernel(second, "same", b"new")
    source = second.roots["inductor"] / "compiled.py"
    source.write_text("compiled = True\n")
    source.chmod(0o755)
    second.save()
    clear_roots(cache)
    restored = cache.restore()
    assert restored["loaded"] and len(restored["manifests"]) == 2
    assert [(cache.roots["triton"] / key / "kernel.cubin").read_bytes()
            for key in ("a", "b", "same")] == [b"a", b"b", b"new"]
    assert source.stat().st_mode & 0o777 == 0o755
    for marker in cache.roots["triton"].rglob("__grp__*"):
        assert all(Path(p).is_file()
                   for p in json.loads(marker.read_text())["child_paths"].values())


def test_incomplete_groups_and_transient_files_never_publish(cache):
    child = kernel(cache, "partial")
    child.unlink()
    malformed = kernel(cache, "malformed")
    (malformed.parent / "__grp__kernel.json").write_text('{"child_paths": []}')
    complete = kernel(cache, "complete")
    for name in ("lock", "x.tmp", "tmp.pid_1/x", ".pending-123"):
        path = complete.parent / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"unfinished")
    (complete.parent / "linked").symlink_to(complete)
    saved = cache.save()
    assert saved["groups"] == 1 and saved["files"] == 2
    clear_roots(cache)
    cache.restore()
    assert complete.read_bytes() == b"compiled"
    assert not (cache.roots["triton"] / "partial").exists()


def test_unchanged_files_are_not_read_or_republished(cache, monkeypatch):
    child = kernel(cache, "a")
    calls = []
    cache.commit = lambda: calls.append("commit")
    cache.save()
    original = type(child).read_bytes
    def read(path):
        assert path != child, "unchanged artifact was read again"
        return original(path)
    monkeypatch.setattr(type(child), "read_bytes", read)
    assert not cache.save()["changed"]
    assert calls == ["commit", "commit"]
    kernel(cache, "b")
    assert cache.save()["changed"]
    assert len(calls) == 4


def test_object_commit_failure_cannot_publish_new_manifest(cache):
    child = kernel(cache, "a", b"old")
    cache.save()
    old_manifests = list(cache.directory.glob("manifests/*"))
    child.write_bytes(b"new")
    def fail():
        raise OSError("durability unavailable")
    cache.commit = fail
    with pytest.raises(OSError, match="durability"):
        cache.save()
    assert list(cache.directory.glob("manifests/*")) == old_manifests
    clear_roots(cache)
    cache.restore()
    assert child.read_bytes() == b"old"


def test_publication_commits_objects_before_manifest(cache):
    kernel(cache, "a")
    observed = []
    cache.commit = lambda: observed.append((len(list(cache.directory.glob("objects/*"))),
                                           len(list(cache.directory.glob("manifests/*")))))
    cache.save()
    assert observed == [(2, 0), (2, 1)]


def test_corrupt_object_fails_before_restoring_any_file(cache):
    kernel(cache, "a")
    cache.save()
    next(cache.directory.glob("objects/*")).write_bytes(b"corrupt")
    clear_roots(cache)
    with pytest.raises(ValueError, match="checksum"):
        cache.restore()
    assert not list(cache.roots["triton"].rglob("*"))


def test_namespace_separates_runtime_and_absolute_cache_roots(cache, tmp_path):
    kernel(cache, "a")
    cache.save()
    other = CompilerCache(cache.directory.parent, cache.roots, {"sm": 90, "toolchain": "pinned"})
    assert not other.restore()["loaded"]
    moved = CompilerCache(cache.directory.parent, {"triton": tmp_path / "moved"}, cache.identity["runtime"])
    assert not moved.restore()["loaded"]


def test_failed_compile_checkpoint_preserves_only_atomic_triton_entries(cache):
    kernel(cache, "old")
    compiled = cache.roots["inductor"] / "compiled.so"
    compiled.write_bytes(b"complete old binary")
    cache.save()
    kernel(cache, "new")
    compiled.write_bytes(b"unfinished overwrite")
    partial = cache.roots["inductor"] / "new.so"
    partial.write_bytes(b"unfinished new binary")
    cache.save(triton_only=True)
    clear_roots(cache)
    cache.restore()
    assert compiled.read_bytes() == b"complete old binary"
    assert not partial.exists()
    assert (cache.roots["triton"] / "new/kernel.cubin").exists()


def test_training_worker_saves_incrementally_only_at_qualified_arms(cache, tmp_path):
    from benchmarks.bf16_training_worker import training_checkpoint
    commits = []
    cache.commit = lambda: commits.append("committed")
    output = tmp_path / "report.json"
    checkpoint = training_checkpoint(output, cache_store=cache)
    report = {"in_progress": {"seed": 7, "case": {}, "arms": {}}}
    checkpoint(report)
    kernel(cache, "first")
    report["in_progress"]["arms"]["candidate"] = {"status": "qualified"}
    checkpoint(report)
    assert len(commits) == 2
    assert report["compiler_cache_checkpoint"]["stored_files"] == 2
    kernel(cache, "later")
    report["in_progress"]["arms"]["candidate"]["status"] = "passed"
    checkpoint(report)
    assert len(commits) == 2
    assert json.loads(output.read_text()) == report
