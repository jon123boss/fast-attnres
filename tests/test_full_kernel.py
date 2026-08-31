import ast
from pathlib import Path

import pytest
import torch

from attnres import attnres, reference_attnres
from attnres._kernels.fixed_tail_sources import _source_rows_view


def _check_launch_arguments(tree, launch_tree=None):
    definitions = {node.name: node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)}
    kernels = {name for name in definitions if name.endswith("_kernel")}
    launch_options = {"num_warps", "num_stages", "maxnreg", "enable_fp_fusion"}
    checked = set()
    for call in ast.walk(tree if launch_tree is None else launch_tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Subscript):
            continue
        target = call.func.value
        if (isinstance(target, ast.Call) and isinstance(target.func, ast.Name)
                and target.func.id == "_wrap_triton"):
            target = target.args[0]
        if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == "fixed_tail"):
            target = ast.Name(id=target.attr)
        if not isinstance(target, ast.Name) or target.id not in definitions:
            continue
        # Setup launches need the same ABI check even without a _kernel suffix.
        kernels.add(target.id)
        signature = definitions[target.id].args
        positional = [arg.arg for arg in signature.posonlyargs + signature.args]
        parameters = set(positional) | {arg.arg for arg in signature.kwonlyargs}
        required = set(positional[:len(positional) - len(signature.defaults)])
        required.update(arg.arg for arg, default in
                        zip(signature.kwonlyargs, signature.kw_defaults) if default is None)
        assert not any(isinstance(arg, ast.Starred) for arg in call.args)
        keywords = {keyword.arg for keyword in call.keywords}
        assert None not in keywords, f"cannot statically check {target.id} keyword expansion"
        assert len(call.args) <= len(positional), target.id
        supplied = set(positional[:len(call.args)]) | keywords
        assert not required - supplied, (target.id, "missing", required - supplied)
        assert not keywords - parameters - launch_options, (target.id, "unknown", keywords)
        checked.add(target.id)
    assert checked == kernels, "every Full kernel must have a checked launch"


@pytest.mark.parametrize("values_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("query_dtype", [torch.float32, torch.bfloat16])
def test_cpu_public_api_still_uses_equation_reference(values_dtype, query_dtype):
    torch.manual_seed(17)
    values = torch.randn(3, 2, 7, dtype=values_dtype)
    query = torch.randn(3, dtype=query_dtype)
    actual = attnres(values, query)
    expected = reference_attnres(values, query)
    torch.testing.assert_close(actual, expected)


def test_affine_source_view_preserves_storage_and_strides():
    producer = torch.randn(2, 4, 11)
    values = producer[..., :7]
    rows = _source_rows_view(values, 8, 7)
    assert rows.stride() == (11, 1)
    assert rows.untyped_storage().data_ptr() == producer.untyped_storage().data_ptr()
    torch.testing.assert_close(rows, values.flatten(0, -2), rtol=0, atol=0)


def test_non_affine_batch_layout_compacts_under_fullgraph():
    values = torch.randn(2, 3, 4, 7).transpose(1, 2).requires_grad_()
    compiled = torch.compile(_source_rows_view, fullgraph=True, dynamic=False, backend="eager")
    rows = compiled(values, 24, 7)
    assert rows.is_contiguous()
    torch.testing.assert_close(rows, values.flatten(0, -2), rtol=0, atol=0)
    gradient, = torch.autograd.grad(rows.sum(), (values,))
    torch.testing.assert_close(gradient, torch.ones_like(values), rtol=0, atol=0)


def test_output_compaction_handles_non_affine_leading_batches():
    from attnres._kernels.fixed_tail import _prepare_grad_output

    grad = torch.randn(3, 2, 7).transpose(0, 1)
    prepared = _prepare_grad_output(grad, torch.empty(4, 6, 7))
    assert prepared.shape == (6, 7) and prepared.is_contiguous()
    torch.testing.assert_close(prepared, grad.flatten(0, -2), rtol=0, atol=0)


def test_fixed_tail_kernel_launch_signatures_without_cuda():
    from attnres._kernels import fixed_tail, fixed_tail_sources

    core = ast.parse(Path(fixed_tail.__file__).read_text())
    adapter = ast.parse(Path(fixed_tail_sources.__file__).read_text())
    _check_launch_arguments(core)
    _check_launch_arguments(core, adapter)
    _check_launch_arguments(adapter)


@pytest.mark.parametrize("arguments", ["x", "x, WIDTH=8, unknown=True"])
def test_source_setup_launch_guard_rejects_missing_or_unknown_arguments(arguments):
    tree = ast.parse(f"def setup(x, WIDTH): pass\nsetup[(1,)]({arguments})")
    with pytest.raises(AssertionError):
        _check_launch_arguments(tree)


def test_fixed_tail_backward_flattens_output_not_source_dimensions():
    from attnres._kernels.fixed_tail import _prepare_grad_output

    values = torch.empty(3, 6, 17)
    upstream = torch.randn(17, 6).t()
    prepared = _prepare_grad_output(upstream, values)
    assert prepared.shape == (6, 17)
    assert prepared.is_contiguous()
    torch.testing.assert_close(prepared, upstream, rtol=0, atol=0)


@pytest.mark.parametrize("value_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("query_dtype", [torch.float32, torch.bfloat16])
def test_fixed_tail_cpu_reference_and_strided_gradients(value_dtype, query_dtype):
    from attnres._kernels.fixed_tail import fused_attnres as fixed_tail
    from validation.oracle import oracle

    torch.manual_seed(20260827)
    producer = torch.randn(3, 2, 3, 34, dtype=value_dtype, requires_grad=True)
    query = (torch.randn(10, dtype=query_dtype) * .25).requires_grad_()
    values, q = producer[..., ::2], query[::2]
    actual, expected = fixed_tail(values, q), oracle(values, q)
    upstream = torch.randn(17, 3, 2, dtype=value_dtype).permute(2, 1, 0)
    ga = torch.autograd.grad(actual, (producer, query), upstream)
    ge = torch.autograd.grad(expected, (producer, query), upstream)
    tol = dict(rtol=.05, atol=.05) if torch.bfloat16 in (value_dtype, query_dtype) else dict(rtol=.001, atol=.0001)
    for a, e in zip((actual, *ga), (expected, *ge)):
        torch.testing.assert_close(a, e, **tol)


@pytest.mark.cuda
@pytest.mark.parametrize("source_layout", ["packed", "list"])
@pytest.mark.parametrize("shape", [
    (1, 1, 1, 1), (3, 5, 17, 5), (9, 257, 768, 64),
    (9, 17, 1024, 512), (9, 17, 1024, 1024), (5, 7, 3000, 257),
    (9, 8, 3072, 3072), (5, 3, 4096, 32), (129, 1, 8192, 8192),
    (2, 5, 17, 5), (4, 5, 63, 33), (3, 7, 64, 33),
    (9, 17, 1024, 768), (129, 1, 8192, 6144),
])
def test_fixed_tail_bf16_envelope(shape, source_layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from attnres._kernels.fixed_tail import fused_attnres
    from attnres._kernels.fixed_tail_sources import source_attnres
    from validation.gpu_checks import _compare
    from validation.oracle import oracle

    torch.manual_seed(20260827)
    s, n, d, r = shape
    source = torch.randn(s, n, d + 7, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    query = (torch.randn(2 * r, device="cuda") * .25).requires_grad_()
    values, q = source[..., :d], query[::2]
    upstream = torch.randn(d, n, device="cuda", dtype=source.dtype).t()
    actual = (fused_attnres(values, q) if source_layout == "packed"
              else source_attnres(tuple(values.unbind(0)), q))
    expected = oracle(values, q)
    ga = torch.autograd.grad(actual, (source, query), upstream)
    ge = torch.autograd.grad(expected, (source, query), upstream)
    for a, e in zip((actual, *ga), (expected, *ge)):
        _compare(a, e, source.dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("source_layout", ["packed", "list", "list_nonaffine"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("query_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width,rank", [(17, 5), (63, 33)])
def test_fixed_tail_compiled_changed_inputs_and_graph(dtype, query_dtype, source_layout, width, rank):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from attnres._kernels.fixed_tail import fused_attnres
    from attnres._kernels.fixed_tail_sources import source_attnres
    from validation.gpu_checks import _compare
    from validation.oracle import oracle

    torch.manual_seed(20260827)
    source = torch.randn(3, 2, 5, 2 * width, device="cuda", dtype=dtype, requires_grad=True)
    query = (torch.randn(2 * rank, device="cuda", dtype=query_dtype) * .25).requires_grad_()
    upstream = torch.randn(width, 5, 2, device="cuda", dtype=dtype).permute(2, 1, 0)
    if source_layout == "list_nonaffine":
        upstream = upstream.transpose(0, 1)
    replay_source, replay_query = torch.randn_like(source), torch.randn_like(query) * .25
    replay_upstream = torch.randn_like(upstream)
    def views(v):
        return (v.transpose(1, 2) if source_layout == "list_nonaffine" else v)[..., ::2]

    def forward(v, q):
        v = views(v)
        return (fused_attnres(v, q[::2]) if source_layout == "packed"
                else source_attnres(tuple(v.unbind(0)), q[::2]))

    compiled = torch.compile(forward, fullgraph=True, dynamic=False)

    def step():
        out = compiled(source, query)
        return (out, *torch.autograd.grad(out, (source, query), upstream))

    def check(actual):
        v, q = source.detach().clone().requires_grad_(), query.detach().clone().requires_grad_()
        expected = oracle(views(v), q[::2])
        grads = torch.autograd.grad(expected, (v, q), upstream)
        for a, e in zip(actual, (expected, *grads)):
            _compare(a, e, dtype)

    check(step())
    counters = dict(torch._dynamo.utils.counters["stats"])
    with torch.no_grad():
        source.copy_(replay_source)
        query.copy_(replay_query)
        upstream.copy_(replay_upstream)
    check(step())
    assert dict(torch._dynamo.utils.counters["stats"]) == counters
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = step()
    with torch.no_grad():
        source.copy_(replay_source + .1)
        query.copy_(replay_query + .01)
        upstream.copy_(replay_upstream * .75)
    graph.replay()
    torch.cuda.synchronize()
    check(captured)
    assert dict(torch._dynamo.utils.counters["stats"]) == counters


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rank", [5, 17])
def test_fixed_tail_source_views_aliases_and_graph(monkeypatch, device, dtype, rank):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    import attnres
    from attnres._kernels.fixed_tail import fused_attnres
    from attnres._kernels.fixed_tail_sources import source_attnres
    from validation.source_checks import source_case

    def fixed_tail(values, query, *, eps=2**-23, scale=1.0):
        function = fused_attnres if isinstance(values, torch.Tensor) else source_attnres
        return function(values, query, eps=eps, scale=scale)

    monkeypatch.setattr(attnres, "attnres", fixed_tail)
    source_case((5, 7, 17, rank), "full", dtype, graph=device == "cuda",
                shared=True, device=device)
