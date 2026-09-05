"""Target-device gates; skipped by ordinary CPU CI."""
import pytest
import torch

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device")]


def test_operator_equations_and_replay():
    from validation.gpu_checks import run_checks
    result = run_checks({"dtype": "bf16"})
    assert result["failed"] == 0, result


def test_block_envelope():
    from validation.block_checks import run_block_checks
    result = run_block_checks({"dtype": "bf16"})
    assert result["failed"] == 0, result


def test_block_compiled_and_replay():
    from validation.block_checks import run_block_checks
    result = run_block_checks({"cases": [[9, 11, 256, 127]], "graph": True})
    assert result["failed"] == 0, result


@pytest.mark.parametrize("shape", [(9, 65, 1024, 512), (9, 19, 1024, 1023),
                                   (33, 17, 1536, 769), (129, 17, 2048, 1024),
                                   (33, 19, 2047, 1024), (9, 65, 2049, 1025)])
def test_compiled_bf16_sliced_envelope(shape):
    from validation.gpu_checks import _compiled_and_graph, _single, PROTOCOL
    torch.manual_seed(PROTOCOL["seeds"][0])
    _single(shape, torch.bfloat16, strided=True)
    _compiled_and_graph(dtype=torch.bfloat16, shape=shape)


def test_compiled_bf16_sliced_feature_and_query_strides():
    from attnres import attnres
    from validation.gpu_checks import _compare, PROTOCOL
    from validation.oracle import oracle
    torch.manual_seed(PROTOCOL["seeds"][0])
    s, n, d, r = 9, 7, 1536, 769
    values = torch.randn(s, n, 2*d, device="cuda", dtype=torch.bfloat16,
                         requires_grad=True)
    query = (torch.randn(2*r, device="cuda", dtype=torch.bfloat16) * .25).requires_grad_()
    upstream = torch.randn(d, n, device="cuda", dtype=torch.bfloat16).T
    compiled = torch.compile(lambda v, q: attnres(v[..., ::2], q[::2]),
                             fullgraph=True, dynamic=False)
    actual = compiled(values, query)
    expected = oracle(values[..., ::2], query[::2])
    ga = torch.autograd.grad(actual, (values, query), upstream)
    ge = torch.autograd.grad(expected, (values, query), upstream)
    for a, e in zip((actual, *ga), (expected, *ge)):
        _compare(a, e, torch.bfloat16)


@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("mode", ["full", "block"])
def test_compiled_training(variant, mode):
    from validation.training_checks import run_training_checks
    result = run_training_checks({"variants": [variant], "modes": [mode]})
    assert result["failed"] == 0, result


@pytest.mark.parametrize("mode", ["full", "block"])
@pytest.mark.parametrize("variant", ["standard", "sliced"])
def test_source_list_strides_shared_gradients_and_graph(mode, variant):
    from validation.source_checks import source_case
    source_case((5, 7, 128, 128 if variant == "standard" else 16),
                mode, torch.bfloat16, graph=True, shared=True)


@pytest.mark.parametrize("variant", ["standard", "sliced"])
def test_source_lists_keep_fresh_storage_after_graph_capture(variant):
    """Keep two input sets alive and check replay under unrelated allocations."""
    from attnres import attnres
    from validation.gpu_checks import _compare, PROTOCOL
    from validation.oracle import oracle
    from validation.source_checks import _NoSourceStack

    torch.manual_seed(PROTOCOL["seeds"][0])
    rank = 17 if variant == "standard" else 5

    def fixture():
        values = tuple(torch.randn(5, 34, device="cuda", dtype=torch.bfloat16,
                                   requires_grad=True) for _ in range(3))
        query = torch.randn(2 * rank, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(17, 5, device="cuda", dtype=torch.bfloat16).T
        return (*values, query), weight

    # Create both sets before any candidate call. Keeping both alive prevents
    # an allocator from making a stale address accidentally refer to new inputs.
    first, first_weight = fixture()
    second, second_weight = fixture()

    def inputs(args):
        values = tuple(t[..., ::2] for t in args[:3])
        return (*values, values[0]), args[3][::2]

    def forward(args):
        values, query = inputs(args)
        return attnres(values, query)

    def step(function, args, weight):
        with _NoSourceStack({(5, 17), (5, rank)}, (3, 4)):
            output = function(args)
            gradients = torch.autograd.grad(output, args, weight)
        return output, gradients

    def check(output, gradients, args, weight):
        values, query = inputs(args)
        expected = oracle(torch.stack(values), query)
        expected_gradients = torch.autograd.grad(expected, args, weight)
        assert len(gradients) == len(expected_gradients)
        for actual, reference in zip((output, *gradients), (expected, *expected_gradients)):
            _compare(actual, reference, torch.bfloat16)

    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
    for function in (forward, compiled):
        for args, weight in ((first, first_weight), (second, second_weight),
                             (first, first_weight)):
            check(*step(function, args, weight), args, weight)
    before = dict(torch._dynamo.utils.counters["stats"])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step(compiled, first, first_weight)
    torch.cuda.current_stream().wait_stream(stream)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        output, gradients = step(compiled, first, first_weight)

    check(*step(compiled, second, second_weight), second, second_weight)
    pressure = [torch.full((65536 + i,), 17 + i, device="cuda", dtype=torch.int64)
                for i in range(4)]
    with torch.no_grad():
        for a, b in zip(first, second):
            a.copy_(b * .5)
        first_weight.copy_(second_weight * .5)
    capture.replay()
    torch.cuda.synchronize()
    check(output, gradients, first, first_weight)
    for i, tensor in enumerate(pressure):
        assert torch.all(tensor == 17 + i), "unrelated allocation was overwritten"
    assert dict(torch._dynamo.utils.counters["stats"]) == before


@pytest.mark.parametrize("variant", ["standard", "sliced"])
def test_source_list_shared_storage_keeps_distinct_gradient_edges(variant):
    from attnres import attnres
    from validation.gpu_checks import _compare
    from validation.oracle import oracle

    torch.manual_seed(20260827)
    rank = 17 if variant == "standard" else 5
    shared = torch.randn(5, 18, device="cuda", dtype=torch.bfloat16)
    values = (shared[:, :17].detach().requires_grad_(),
              shared[:, 1:].detach().requires_grad_(),
              torch.randn(5, 17, device="cuda", dtype=torch.bfloat16, requires_grad=True))
    query = torch.randn(rank, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    parameters = (*values, query)
    actual = attnres(values, query)
    expected = oracle(torch.stack(values), query)
    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, parameters, upstream)
    expected_grad = torch.autograd.grad(expected, parameters, upstream)
    assert not torch.equal(expected_grad[0], expected_grad[1])
    assert len(actual_grad) == len(parameters)
    _compare(actual, expected, torch.bfloat16)
    for a, e in zip(actual_grad, expected_grad):
        _compare(a, e, torch.bfloat16)


def _source_lifetime_refs(variant, mode, sequence, device, dtype):
    import gc
    import weakref
    from attnres import attnres

    rank = 17 if variant == "standard" else 5
    values = [torch.randn(5, 17, device=device, dtype=dtype, requires_grad=True) for _ in range(3)]
    query = torch.randn(rank, device=device, dtype=torch.bfloat16, requires_grad=True)
    refs = [weakref.ref(tensor) for tensor in [*values, query]]
    value_arg = tuple(values) if sequence else torch.stack(values)
    if mode == "full":
        loss = attnres(value_arg, query).square().sum()
    else:
        partial = torch.randn_like(values[0], requires_grad=True)
        refs.append(weakref.ref(partial))
        block_values = (
            (*value_arg, partial)
            if sequence
            else torch.cat((value_arg, partial.unsqueeze(0)), dim=0)
        )
        output = attnres(block_values, query)
        loss = output.square().sum()
    # Only autograd may retain arguments across this boundary.
    del values, query, value_arg
    if mode == "block":
        del partial, block_values, output
    gc.collect()
    loss.backward()
    return refs


@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("mode", ["full", "block"])
@pytest.mark.parametrize("sequence", [False, True])
def test_source_calls_release_tensor_arguments(variant, mode, sequence):
    import gc

    torch.manual_seed(20260827)
    for _ in range(2):
        refs = _source_lifetime_refs(variant, mode, sequence, "cuda", torch.bfloat16)
        torch.cuda.synchronize()
        gc.collect()
        assert all(ref() is None for ref in refs), "completed call retained input tensors"


def _compiled_source_lifetime_refs(compiled, variant, mode, graph, device="cuda"):
    """Check Python Tensor ownership, not CUDA allocator/graph-pool reservation."""
    import weakref

    rank = 17 if variant == "standard" else 5
    values = tuple(torch.randn(5, 17, device=device, dtype=torch.bfloat16,
                               requires_grad=True) for _ in range(3))
    query = torch.randn(rank, device=device, dtype=torch.bfloat16, requires_grad=True)
    partial = torch.randn_like(values[0], requires_grad=True) if mode == "block" else None
    leaves = (*values, query,
              *((partial,) if partial is not None else ()),
              )
    refs = [weakref.ref(tensor) for tensor in leaves]

    def step():
        for tensor in leaves:
            if tensor.grad is not None:
                tensor.grad.zero_()
        compiled(values, query, partial).backward()

    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
            step()
        stream.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            step()
        for _ in range(8):
            with torch.no_grad():
                for tensor in leaves:
                    tensor.add_(0.01)
            captured.replay()
        torch.cuda.synchronize()
        del captured, stream
    else:
        import gc

        loss = compiled(values, query, partial)
        del step, leaves, values, query, partial
        gc.collect()
        loss.backward()
    return refs


@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("mode", ["full", "block"])
@pytest.mark.parametrize("graph", [False, True])
def test_compiled_source_calls_release_tensor_arguments(variant, mode, graph):
    import gc
    from torch._dynamo.utils import counters
    from attnres import attnres

    def forward(values, query, partial):
        if mode == "full":
            return attnres(values, query).square().sum()
        return attnres((*values, partial), query).square().sum()

    torch.manual_seed(20260827)
    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
    graph_count = None
    graph_breaks = dict(counters["graph_break"])
    for _ in range(2):
        refs = _compiled_source_lifetime_refs(compiled, variant, mode, graph)
        torch.cuda.synchronize()
        gc.collect()
        assert all(ref() is None for ref in refs), "compiled call retained input tensors"
        assert dict(counters["graph_break"]) == graph_breaks
        current_count = counters["stats"]["unique_graphs"]
        if graph_count is not None:
            assert current_count == graph_count, "fresh inputs caused recompilation"
        graph_count = current_count


@pytest.mark.parametrize("mode,shape", [
    *(("full", shape) for shape in (
        (1, 1, 1, 1), (9, 19, 1024, 32), (33, 7, 2047, 257),
        (129, 8, 7168, 1024), (129, 8, 8192, 8192),
    )),
    *(("block", shape) for shape in (
        (1, 1, 1, 1), (9, 19, 1024, 32), (33, 7, 2047, 257),
        (128, 8, 7168, 1024), (128, 8, 8192, 8192),
    )),
])
def test_source_list_bf16_envelope(mode, shape):
    from validation.source_checks import source_case
    source_case(shape, mode, torch.bfloat16)


@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("mode", ["full", "block"])
def test_source_list_compiled_training(variant, mode):
    from validation.training_checks import run_training_checks
    result = run_training_checks({"variants": [variant], "modes": [mode],
                                  "model": {"source_layout": "list"}})
    assert result["failed"] == 0, result


@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("sequence", [False, True])
def test_non_affine_output_batches_preserve_all_gradients(variant, sequence):
    from attnres import attnres
    from validation.gpu_checks import _compare
    from validation.oracle import oracle

    torch.manual_seed(20260827)
    rank = 17 if variant == "standard" else 5
    values = torch.randn(3, 2, 3, 17, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    query = torch.randn(rank, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(3, 2, 17, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    params = (values, query)

    def forward(v, q):
        return attnres(tuple(v.unbind(0)) if sequence else v, q)

    for function in (forward, torch.compile(forward, fullgraph=True, dynamic=False)):
        actual = function(values, query)
        expected = oracle(values, query)
        ga = torch.autograd.grad(actual, params, upstream)
        ge = torch.autograd.grad(expected, params, upstream)
        for a, e in zip((actual, *ga), (expected, *ge)):
            _compare(a, e, torch.bfloat16)


@pytest.mark.parametrize("shape", [(7, 9, 513, 257), (33, 7, 2048, 2048),
                                   (129, 8, 7168, 1024), (129, 8, 8192, 8192)])
def test_full_source_bf16_tile_boundaries_and_graph(shape):
    from validation.source_checks import source_case
    source_case(shape, "full", torch.bfloat16, graph=True)


@pytest.mark.parametrize("shape", [(7, 9, 513, 257), (33, 7, 2048, 2048)])
def test_block_source_bf16_fusion_boundaries_and_graph(shape):
    from validation.source_checks import source_case
    source_case(shape, "block", torch.bfloat16, graph=True)


@pytest.mark.parametrize("shape", [(128, 8, 7168, 1024), (128, 8, 8192, 8192)])
def test_block_source_bf16_wide_compiled_graph(shape):
    from validation.source_checks import source_case
    source_case(shape, "block", torch.bfloat16, graph=True)


@pytest.mark.cuda
def test_source_bf16_bounded_maximum_compiled_graph():
    from validation.source_checks import source_case

    source_case((129, 7, 2048, 2048), "full", torch.bfloat16, graph=True)
