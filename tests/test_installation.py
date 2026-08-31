import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


_INSTALLED_SMOKE = r"""
import os
from pathlib import Path
import torch
import attnres
from attnres import attnres as call, reference_attnres
assert attnres.__version__ == "1.0.0"
assert Path(attnres.__file__).resolve().is_relative_to(Path(os.environ["ATTNRES_INSTALL_DIR"]).resolve())
package_dir = Path(attnres.__file__).resolve().parent
assert not hasattr(attnres, "prepare_block")
assert not hasattr(attnres, "merge_block")
assert not hasattr(attnres, "BlockCache")
assert not (package_dir / "block.py").exists()
assert not (package_dir / "_types.py").exists()
assert not (package_dir / "_kernels" / "block.py").exists()
device = os.environ["ATTNRES_TEST_DEVICE"]
torch.set_default_device(device)
torch.manual_seed(42019)
options = dict(eps=2**-20, scale=0.75)
def check(values, query, function=call):
    leaves = (values,) if isinstance(values, torch.Tensor) else tuple(values)
    output = function(values, query, **options)
    expected = reference_attnres(values if isinstance(values, torch.Tensor) else torch.stack(leaves), query, **options)
    tol = (0.05, 0.05) if output.dtype == torch.bfloat16 else (0.001, 0.0001)
    torch.testing.assert_close(output, expected, rtol=tol[0], atol=tol[1])
    gradients = torch.autograd.grad(output.float().square().mean(), (*leaves, query))
    reference_gradients = torch.autograd.grad(expected.float().square().mean(), (*leaves, query))
    for actual, reference in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, reference, rtol=tol[0], atol=tol[1])
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
packed = torch.randn(3, 2, 8, requires_grad=True)
check(packed, torch.randn(4, requires_grad=True))
sources = [torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
check(sources, torch.randn(3, requires_grad=True))
if device == "cuda":
    compiled = torch.compile(call, fullgraph=True, dynamic=False)
    for shift in (0.0, 0.2):
        sources = tuple((torch.randn(5, 17, dtype=torch.bfloat16) + shift).requires_grad_() for _ in range(3))
        query = torch.randn(5, requires_grad=True)
        check(sources, query, compiled)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = compiled(sources, query, **options)
            gradients = torch.autograd.grad(output.sum(), (*sources, query))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = compiled(sources, query, **options)
        gradients = torch.autograd.grad(output.sum(), (*sources, query))
    with torch.no_grad():
        for value in sources:
            value.add_(0.125)
        query.add_(0.125)
    graph.replay()
    expected = reference_attnres(torch.stack(sources), query, **options)
    reference_gradients = torch.autograd.grad(expected.sum(), (*sources, query))
    torch.testing.assert_close(output, expected, rtol=0.05, atol=0.05)
    for actual, reference in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0.05, atol=0.05)
"""


def _run(command, cwd):
    return subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)


def _copy_build_source(repository, destination):
    destination.mkdir()
    for filename in (
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "NOTICE",
        "CITATION.cff",
    ):
        source = repository / filename
        if source.is_file():
            shutil.copy2(source, destination / filename)
    shutil.copytree(
        repository / "src",
        destination / "src",
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"),
    )


def _build_wheel(source, wheel_dir, cwd):
    wheel_dir.mkdir()
    result = _run(
        [sys.executable, "-m", "pip", "wheel", str(source), "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheel_dir)],
        cwd,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    wheels = sorted(wheel_dir.glob("fast_attnres-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_wheel_builds_and_imports_from_clean_target(tmp_path, device):
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA device")
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    _copy_build_source(repository, source)
    expected = {"attnres/" + path.relative_to(source / "src/attnres").as_posix()
                for path in (source / "src/attnres").rglob("*.py")}

    direct = _build_wheel(source, tmp_path / "direct", source)
    sdist_dir = tmp_path / "sdist"
    sdist_dir.mkdir()
    sdist = _run(
        [sys.executable, "-c", "import sys; from setuptools.build_meta import build_sdist; build_sdist(sys.argv[1])", str(sdist_dir)],
        source,
    )
    assert sdist.returncode == 0, sdist.stdout + "\n" + sdist.stderr
    # Distribution filenames are normalized by the active setuptools version
    # (hyphens may become underscores), so validate the single built sdist
    # rather than duplicating setuptools' filename normalization policy here.
    archives = sorted(sdist_dir.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        assert any(Path(member.name).name == "CITATION.cff" for member in archive.getmembers())
    rebuilt = _build_wheel(archives[0], tmp_path / "rebuilt", tmp_path)

    for index, wheel in enumerate((direct, rebuilt)):
        with zipfile.ZipFile(wheel) as package:
            names = set(package.namelist())
            packaged = {name for name in names if name.startswith("attnres/") and name.endswith(".py")}
            assert packaged == expected
            metadata = package.read(next(name for name in names if name.endswith("/METADATA"))).decode()
            assert "Name: fast-attnres" in metadata and "Requires-Dist: torch" in metadata
            entry_points = package.read(
                next(name for name in names if name.endswith("/entry_points.txt"))
            ).decode()
            assert "fast-attnres-info = attnres.info:main" in entry_points
            for filename in ("LICENSE", "NOTICE"):
                bundled = [name for name in names if name.endswith("/" + filename)]
                assert len(bundled) == 1
                assert package.read(bundled[0]) == (source / filename).read_bytes()
        install_dir = tmp_path / ("install-" + str(index))
        install_dir.mkdir()
        install = _run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(install_dir), str(wheel)], tmp_path)
        assert install.returncode == 0, install.stdout + "\n" + install.stderr
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(install_dir), ATTNRES_INSTALL_DIR=str(install_dir), ATTNRES_TEST_DEVICE=device)
        check = subprocess.run([sys.executable, "-c", _INSTALLED_SMOKE], cwd=tmp_path, env=environment, check=False, capture_output=True, text=True)
        assert check.returncode == 0, check.stdout + "\n" + check.stderr
