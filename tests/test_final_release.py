"""A release may change version metadata, never unmeasured runtime behavior."""
from pathlib import Path
import shutil

import pytest

from scripts.verify_final_release import verify_runtime


def test_actual_release_matches_measured_runtime(final_measured_source):
    root = Path(__file__).resolve().parents[1]
    result = verify_runtime(root / "src/attnres", final_measured_source / "src/attnres")
    assert result["version_only_changes"] == ["__init__.py"]


@pytest.mark.parametrize("mutation", ["kernel", "extra", "missing", "initializer"])
def test_release_rejects_unmeasured_changes(tmp_path, mutation):
    measured, current = tmp_path / "measured", tmp_path / "current"
    measured.mkdir()
    (measured / "__init__.py").write_text('__version__ = "1.0.0"\nfrom .kernel import call\n')
    (measured / "kernel.py").write_text('def call(): return 1\n')
    shutil.copytree(measured, current)
    (current / "__init__.py").write_text('__version__ = "2.0.0"\nfrom .kernel import call\n')
    assert verify_runtime(current, measured)["status"] == "passed"
    if mutation == "missing":
        (current / "kernel.py").unlink()
    elif mutation == "extra":
        (current / "extra.py").write_text("# new runtime\n")
    else:
        target = current / ("kernel.py" if mutation == "kernel" else "__init__.py")
        target.write_text(target.read_text() + 'raise RuntimeError("changed")\n')
    with pytest.raises(ValueError, match="runtime"):
        verify_runtime(current, measured)
