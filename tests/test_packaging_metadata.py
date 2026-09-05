import ast
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _project_metadata():
    with (ROOT / "pyproject.toml").open("rb") as metadata_file:
        return tomllib.load(metadata_file)["project"]


def test_distribution_name_and_import_surface_are_distinct():
    project = _project_metadata()

    assert project["name"] == "fast-attnres"
    assert project["dependencies"] == ["torch>=2.9"]
    assert project["scripts"] == {"fast-attnres-info": "attnres.info:main"}
    assert (ROOT / "src" / "attnres" / "__init__.py").is_file()


def test_typing_stub_explicitly_exports_the_root_api():
    stub = (ROOT / "src" / "attnres" / "__init__.pyi").read_text(encoding="utf-8")

    assert "from .modules import LearnedQuery as LearnedQuery" in stub
    assert "reference_attnres" not in stub
    tree = ast.parse(stub)
    exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    assert set(exports) == {"attnres", "LearnedQuery", "__version__"}


def test_cuda_extra_pins_the_supported_pytorch_triton_stack():
    extras = _project_metadata()["optional-dependencies"]

    assert extras["cuda"] == [
        "torch==2.13.0; platform_system == 'Linux' and platform_machine == 'x86_64'",
        "triton==3.7.1; platform_system == 'Linux' and platform_machine == 'x86_64'",
    ]
    assert "tomli>=2; python_version < '3.11'" in extras["test"]


def test_readme_explains_the_latest_compatible_stable_runtime_pair():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "newest mutually compatible stable PyTorch/Triton pair" in readme
    assert "PyTorch 2.13.0" in readme
    assert "triton==3.7.1" in readme
    assert "Triton 3.8.0" in readme


def test_project_provenance_and_citation_are_discoverable():
    project = _project_metadata()
    urls = project["urls"]

    assert project["authors"] == [{"name": "Jonathan Su"}]
    assert urls["Repository"] == "https://github.com/jon123boss/fast-attnres"
    assert urls["Citation"].endswith("/CITATION.cff")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "title: Fast Attention Residuals" in citation
    assert "family-names: Su" in citation
    assert "given-names: Jonathan" in citation
    assert "https://arxiv.org/abs/2607.09694" in citation
