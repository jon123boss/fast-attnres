from pathlib import Path

import pytest
import torch

from benchmarks.baseline import FrozenBaselineError, load_frozen_baseline


def _write_frozen_fixture(root: Path, legacy_optional_keys: bool) -> Path:
    package = root / "src" / "attnres"
    kernels = package / "_kernels"
    kernels.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from .api import attnres\n",
        encoding="utf-8",
    )
    (package / "api.py").write_text(
        f"def attnres(values, query, *, {'keys=None, ' if legacy_optional_keys else ''}eps=2**-23, scale=1.0):\n"
        "    return fused_attnres(values, query, eps=eps, scale=scale)\n"
        "\n"
        "from ._kernels.full import fused_attnres\n",
        encoding="utf-8",
    )
    (kernels / "__init__.py").write_text("", encoding="utf-8")
    (kernels / "full.py").write_text(
        "import torch\n"
        "REGISTRATION = 'attnres::fixture_full'\n"
        "def fused_attnres(values, query, *, eps=2**-23, scale=1.0):\n"
        "    del query, eps, scale\n"
        "    return values.sum(dim=0)\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("legacy_optional_keys", [False, True])
def test_frozen_baseline_is_explicit_isolated_and_hashes_adaptation(tmp_path, legacy_optional_keys):
    with pytest.raises(FrozenBaselineError, match="does not exist"):
        load_frozen_baseline(tmp_path / "missing")

    source = _write_frozen_fixture(tmp_path / "retained", legacy_optional_keys)
    source_bytes = {
        path: path.read_bytes() for path in (source / "src" / "attnres").rglob("*.py")
    }
    baseline = load_frozen_baseline(source)
    metadata = baseline.metadata

    assert metadata["operator_namespace"].startswith("frozen_baseline_")
    assert metadata["namespace_rewrites"] == 1
    assert (
        metadata["original_hashes"]["_kernels/full.py"]
        != metadata["adapted_hashes"]["_kernels/full.py"]
    )
    assert source_bytes == {
        path: path.read_bytes() for path in (source / "src" / "attnres").rglob("*.py")
    }
    assert load_frozen_baseline(source) is baseline
    assert baseline.attnres.__module__.startswith(baseline.metadata["module_namespace"])

    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    query = torch.ones(4)
    torch.testing.assert_close(baseline(values, query), values.sum(dim=0))
