import pytest
import torch
from pathlib import Path
import shutil


@pytest.fixture(autouse=True)
def isolate_cuda_compilation(request):
    """Keep unrelated CUDA tests from exhausting one code object's guard cache."""
    if request.node.get_closest_marker("cuda") is not None:
        torch.compiler.reset()


@pytest.fixture(scope="session")
def historical_release_root(tmp_path_factory):
    """Audit v1 evidence against its measured bytes, never the new candidate."""
    from benchmarks.audit_current_24l import _archive_files
    project = Path(__file__).resolve().parents[1]
    root = tmp_path_factory.mktemp("historical-release") / "checkout"
    shutil.copytree(project, root, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".pytest_cache", "build", "dist", "*.egg-info"))
    archive = project / "results/current_24l/reproduction/performance_source.tar.gz"
    for relative, data in _archive_files(archive).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root
