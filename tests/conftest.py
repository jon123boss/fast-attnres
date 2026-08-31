import pytest
import torch


@pytest.fixture(autouse=True)
def isolate_cuda_compilation(request):
    """Keep unrelated CUDA tests from exhausting one code object's guard cache."""
    if request.node.get_closest_marker("cuda") is not None:
        torch.compiler.reset()
