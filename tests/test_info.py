import json
import os
import subprocess
import sys
from pathlib import Path

import torch

import attnres as package
from attnres import LearnedQuery, __version__, attnres
from attnres.modules import LearnedQuery as ModuleLearnedQuery


def test_root_exports_learned_query_and_version_without_shadowing_function():
    assert LearnedQuery is ModuleLearnedQuery
    assert package.LearnedQuery is LearnedQuery
    assert __version__ == "1.0.0"
    assert package.__version__ == __version__
    assert callable(attnres)
    assert attnres is package.attnres
    assert "LearnedQuery" in package.__all__
    assert "__version__" in package.__all__
    assert "reference_attnres" not in package.__all__
    assert not hasattr(package, "reference_attnres")


def test_learned_query_defaults_to_bf16_and_casts_differentiably():
    learned = LearnedQuery(7)
    assert learned.query.dtype == torch.bfloat16
    assert learned().dtype == torch.bfloat16

    learned.float()
    query = learned()
    assert learned.query.dtype == torch.float32
    assert query.dtype == torch.bfloat16
    query.float().square().sum().backward()
    assert learned.query.grad is not None
    assert learned.query.grad.dtype == torch.float32


def _subprocess_environment(repository: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source = str(repository / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def test_info_module_runs_without_triton_on_cpu():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "attnres.info"],
        cwd=repository,
        env=_subprocess_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "fast-attnres 1.0.0" in result.stdout
    assert "torch:" in result.stdout
    assert "cuda available:" in result.stdout
    assert "triton available:" in result.stdout
    assert "attnres API: CUDA BF16 only" in result.stdout


def test_info_module_supports_json_for_scripts():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "attnres.info", "--json"],
        cwd=repository,
        env=_subprocess_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    info = json.loads(result.stdout)
    assert info["package"] == "fast-attnres"
    assert info["version"] == __version__
    assert info["torch"] == torch.__version__
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["triton_available"], bool)
