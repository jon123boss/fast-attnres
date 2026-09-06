import pytest
import torch

from attnres._kernels.fla_full_sources import _row_layout, _source_pointer_table


@pytest.mark.parametrize("shape,strides", [
    ((3, 1, 4), (4, 8, 1)),
    ((1, 3, 4), (99, 4, 1)),
    ((2, 1, 3, 1, 4), (12, 99, 4, 77, 1)),
    ((1, 1, 4), (64, 12, 1)),
    ((3, 1, 4), (8, 99, 2)),
    ((3, 1, 4), (7, 99, 1)),
    ((3, 1, 4), (0, 99, 1)),
    ((3, 2, 4), (4, 12, 1)),
    ((3, 4), (4, 1)),
    ((4,), (2,)),
])
def test_flattened_addresses_match_logical_rows(shape, strides):
    storage = torch.arange(100)
    values = storage.as_strided(shape, strides, storage_offset=1)
    prepared, row_stride, feature_stride = _row_layout(values)
    flattened = prepared.as_strided((values.numel() // shape[-1], shape[-1]),
                                   (row_stride, feature_stride))
    torch.testing.assert_close(flattened, values.reshape(-1, shape[-1]), rtol=0, atol=0)
    sources, rows, features, count = _source_pointer_table((values, values))
    assert count == 2 and rows == (row_stride, row_stride)
    assert features == (feature_stride, feature_stride)
    if values.is_contiguous():
        assert prepared is values and all(source is values for source in sources)


def test_singleton_layout_compiles_without_copying_or_losing_leaf_gradients():
    leaf = torch.arange(12., requires_grad=True)
    values = leaf.as_strided((3, 1, 4), (4, 8, 1))
    compiled = torch.compile(lambda x: _row_layout(x), fullgraph=True,
                             dynamic=False, backend="eager")
    prepared, row_stride, feature_stride = compiled(values)
    assert prepared is values and (row_stride, feature_stride) == (4, 1)
    gradient, = torch.autograd.grad(prepared.square().sum(), (leaf,))
    torch.testing.assert_close(gradient, 2 * leaf, rtol=0, atol=0)


@pytest.mark.cuda
def test_row_layout_leaf_gradients_and_changed_input_graphs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from attnres import attnres
    from validation.layout_checks import run_layout_checks
    assert len(run_layout_checks(attnres)) == 4
