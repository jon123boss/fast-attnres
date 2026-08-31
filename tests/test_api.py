import importlib.abc
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from attnres import attnres, reference_attnres


class _RejectTritonImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise AssertionError(f"CPU source-list path attempted to import {fullname}")


def test_cpu_source_list_does_not_import_triton(monkeypatch):
    for name in tuple(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    guard = _RejectTritonImports()
    sys.meta_path.insert(0, guard)
    try:
        values = [torch.randn(2, 8, requires_grad=True) for _ in range(3)]
        query = torch.randn(4, requires_grad=True)
        output = attnres(values, query)
        output.sum().backward()
    finally:
        sys.meta_path.remove(guard)
    assert output.shape == (2, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_tail_key_api_matches_reference(dtype):
    torch.manual_seed(7)
    values = torch.randn(5, 2, 13, dtype=dtype)
    query = torch.randn(4, dtype=torch.float32)

    actual = attnres(values, query)
    expected = reference_attnres(values, query)

    assert actual.shape == (2, 13)
    assert actual.dtype == dtype
    tolerance = {"rtol": 0.05, "atol": 0.05} if dtype == torch.bfloat16 else {
        "rtol": 0.001,
        "atol": 0.0001,
    }
    torch.testing.assert_close(actual, expected, **tolerance)


def test_cpu_options_match_reference():
    torch.manual_seed(11)
    values = torch.randn(4, 3, 9, dtype=torch.float32)
    query = torch.randn(3, dtype=torch.bfloat16)
    options = {"eps": 2**-20, "scale": 0.75}
    torch.testing.assert_close(attnres(values, query, **options),
                               reference_attnres(values, query, **options),
                               rtol=.001, atol=.0001)










def test_cpu_noncontiguous_values_are_supported():
    torch.manual_seed(13)
    storage = torch.randn(4, 3, 14, dtype=torch.float32)
    values = storage[..., ::2]
    query = torch.randn(5, dtype=torch.float32)

    assert not values.is_contiguous()
    torch.testing.assert_close(
        attnres(values, query),
        reference_attnres(values, query),
        rtol=0.001,
        atol=0.0001,
    )


@pytest.mark.parametrize("values,query,error", [
    (torch.randn(4), torch.randn(2), ValueError),
    (torch.randn(4, 8), torch.randn(2, 1), ValueError),
    (torch.randn(4, 8, dtype=torch.float16), torch.randn(2), TypeError),
    (torch.randn(4, 8), torch.randn(2, dtype=torch.float16), TypeError),
])
def test_api_rejects_invalid_shapes_and_storage(values, query, error):
    with pytest.raises(error):
        attnres(values, query)


@pytest.mark.parametrize(
    "kwargs", [{"eps": 0.0}, {"eps": float("nan")}, {"scale": float("inf")}]
)
@pytest.mark.parametrize("sequence", [False, True])
def test_api_rejects_invalid_numeric_options(kwargs, sequence):
    values = torch.randn(3, 8)
    query = torch.randn(4)
    with pytest.raises(ValueError):
        attnres(list(values.unbind(0)) if sequence else values, query, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("container", ["list", "tuple", "packed_values"])
def test_source_containers_preserve_producer_and_query_gradients(dtype, container):
    from validation.oracle import oracle

    torch.manual_seed(29)
    width, rank = 13, 5
    leaves = [torch.randn(2, 3, 2 * width, dtype=dtype, requires_grad=True),
              torch.randn(3, 2, width, dtype=dtype, requires_grad=True),
              torch.randn(2, 3, width + 7, dtype=dtype, requires_grad=True)]
    values = [leaves[0][..., ::2], leaves[1].transpose(0, 1), leaves[2][..., :width]]
    values.append(values[0])
    query = (torch.randn(2 * rank) * .25).requires_grad_()
    q = query[::2]
    expected = oracle(torch.stack(values), q)
    if container == "tuple":
        values = tuple(values)
    elif container == "packed_values":
        values = torch.stack(values)
    actual = attnres(values, q)
    upstream = torch.randn(width, 3, 2, dtype=dtype).permute(2, 1, 0)
    ga = torch.autograd.grad(actual, (*leaves, query), upstream)
    ge = torch.autograd.grad(expected, (*leaves, query), upstream)
    tolerance = {"rtol": .05, "atol": .05} if dtype == torch.bfloat16 else {
        "rtol": .001, "atol": .0001}
    for a, e in zip((actual, *ga), (expected, *ge)):
        torch.testing.assert_close(a, e, **tolerance)


@pytest.mark.parametrize("values,query,error", [
    ([], torch.randn(2), ValueError),
    ([torch.randn(2, 5), torch.randn(3, 5)], torch.randn(2), ValueError),
    ([torch.randn(2, 5), 3], torch.randn(2), TypeError),
    ([torch.randn(2, 5)], torch.randn(6), ValueError),
    ({"source": torch.randn(2, 5)}, torch.randn(2), TypeError),
    ([torch.randn(2, 5)], torch.randn(2, device="meta"), TypeError),
    ([torch.randn(2, 5)] * 130, torch.randn(2), ValueError),
])
def test_source_containers_reject_malformed_inputs(values, query, error):
    with pytest.raises(error):
        attnres(values, query)


@pytest.mark.parametrize("compile_call", [False, True])
def test_source_stack_guard_rejects_packing_without_rejecting_arithmetic(compile_call):
    from validation.source_checks import _NoSourceStack

    values = (torch.randn(2, 4), torch.randn(2, 4))
    for operation, packs in ((lambda v: v[0] + v[1], False),
                             (lambda v: torch.stack(v).sum(0), True),
                             (lambda v: torch.cat(v, 0), True)):
        if compile_call:
            call_targets = []

            def record_graph(graph_module, _example_inputs):
                call_targets.extend(
                    node.target
                    for node in graph_module.graph.nodes
                    if node.op == "call_function"
                )
                return graph_module.forward

            function = torch.compile(operation, fullgraph=True, backend=record_graph)
            # PyTorch 2.13 masks modes that opt into compile internals while
            # tracing, then restores the mode for the compiled artifact.  The
            # first call deliberately happens inside the guard so this test
            # covers both halves of that contract.
            with _NoSourceStack(((2, 4),), (2,)):
                if packs:
                    with pytest.raises(AssertionError, match="source stack"):
                        function(values)
                else:
                    function(values)
            saw_pack = any(target in (torch.stack, torch.cat) for target in call_targets)
            if hasattr(TorchDispatchMode, "ignore_compile_internals"):
                assert saw_pack is packs
            else:
                # Older, out-of-profile Torch releases run the guard during
                # tracing and reject the packed graph before the backend sees
                # it.  The runtime prohibition still holds.
                assert not saw_pack
        else:
            with _NoSourceStack(((2, 4),), (2,)):
                if packs:
                    with pytest.raises(AssertionError, match="source stack"):
                        operation(values)
                else:
                    operation(values)


@pytest.mark.cuda
def test_cuda_api_smoke_and_first_order_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from validation.oracle import oracle

    torch.manual_seed(19)
    values = torch.randn(3, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    query = torch.randn(4, device="cuda", dtype=torch.float32, requires_grad=True)
    output = attnres(values, query)
    expected = oracle(values, query)
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad(output, (values, query), upstream)
    reference_gradients = torch.autograd.grad(expected, (values, query), upstream)
    for a, e in zip((output, *gradients), (expected, *reference_gradients)):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e, rtol=.05, atol=.05)


@pytest.mark.parametrize("container", [None, list, tuple])
@pytest.mark.parametrize("query", [None, 2, [1.0, 2.0]])
def test_query_type_errors_are_consistent_across_source_containers(container, query):
    values = torch.randn(3, 8)
    with pytest.raises(TypeError, match="query"):
        attnres(container(values.unbind(0)) if container else values, query)


@pytest.mark.parametrize("container", [None, list, tuple])
@pytest.mark.parametrize("name", ["eps", "scale"])
@pytest.mark.parametrize("value", [True, "0.5", torch.tensor(0.5)])
def test_scalar_type_errors_are_consistent_across_source_containers(container, name, value):
    values, query = torch.randn(3, 8), torch.randn(2)
    with pytest.raises(TypeError):
        attnres(container(values.unbind(0)) if container else values,
                query, **{name: value})


@pytest.mark.parametrize("container", [None, list, tuple])
def test_real_integer_scalar_options_preserve_equations(container):
    values, query = torch.randn(3, 8), torch.randn(2)
    source = container(values.unbind(0)) if container else values
    actual = attnres(source, query, eps=1, scale=2)
    expected = reference_attnres(values, query, eps=1.0, scale=2.0)
    torch.testing.assert_close(actual, expected)


def test_public_api_removes_explicit_keys_and_carriers():
    import inspect
    import attnres as package
    from attnres import modules

    assert not hasattr(package, "carrier_attnres")
    assert not hasattr(modules, "ProjectedOutput")
    assert not hasattr(modules, "split_projected_output")
    assert not hasattr(package, "prepare_block")
    assert not hasattr(package, "merge_block")
    assert not hasattr(package, "BlockCache")
    for function in (attnres, reference_attnres):
        assert "keys" not in inspect.signature(function).parameters
        assert "partial_key" not in inspect.signature(function).parameters
    with pytest.raises(TypeError, match="keys"):
        attnres(torch.randn(3, 8), torch.randn(2), keys=torch.randn(3, 2))
