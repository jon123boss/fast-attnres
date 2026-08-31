"""GPU-only validation for the explicit native-FLA compile bridge."""

from __future__ import annotations

import ast
import copy
import os

import pytest
import torch

from benchmarks.fla_compile import (
    EPS,
    _native_functions,
    make_model_backend,
    resolve_vendor_root,
    source_hash_metadata,
)
from benchmarks.model import TrainingConfig, make_model
from validation.oracle import oracle


_VENDOR_ENV_VARS = (
    "ATTNRES_FLA_DIR",
    "FLA_ROOT",
    "FLASH_LINEAR_ATTENTION_ROOT",
    "VENDOR_FLA_ROOT",
)


def _configured_vendor_root():
    """Use the same env/default resolution as the backend under test."""

    try:
        return resolve_vendor_root()
    except ImportError as error:
        # An explicitly configured checkout is a test setup error, not an
        # optional dependency that should silently turn GPU tests into skips.
        if any(os.environ.get(variable) for variable in _VENDOR_ENV_VARS):
            raise
        pytest.skip(str(error))


def test_bridge_import_and_source_identity_without_cuda():
    metadata = source_hash_metadata(vendor_root=_configured_vendor_root())
    assert metadata["bridge"] == "fla_native_compile_custom_op"
    assert metadata["checkpoint_level"] == 1
    assert metadata["qualification_eligible"] is True
    assert metadata["rms_weight"] == "ones"
    assert metadata["model_rms_weight_allocation"] == "nonpersistent_buffer"
    assert metadata["model_rms_weight_reuse"] == "one_buffer_per_model"
    assert metadata["direct_call_fallback"] == "query_ones"
    assert metadata["compiled_model_fill_launches_per_step"] == 0
    assert metadata["compiled_model_fill_launches_avoided_per_step"] == 1
    assert metadata["output_rms_weight"] is None
    assert metadata["rms_eps"] == EPS
    assert metadata["accepts_source_list"] is True
    assert metadata["model_forced_source_stack"] is False
    assert set(metadata["vendor_file_hashes"]) == {
        "fla/ops/attnres/fused.py",
        "fla/ops/attnres/backends/gluon.py",
    }


def test_checkpoint_zero_is_explicitly_non_qualification_eligible():
    metadata = source_hash_metadata(
        implementation="triton", vendor_root=_configured_vendor_root(), checkpoint_level=0
    )
    assert metadata["checkpoint_level"] == 0
    assert metadata["qualification_eligible"] is False
    assert metadata["checkpoint0_status"] == "experimental_native_gradient_failure"


def test_gluon_source_metadata_discloses_dependency_compatibility_shim():
    metadata = source_hash_metadata(
        implementation="gluon", vendor_root=_configured_vendor_root()
    )
    compatibility = metadata["dependency_compatibility"]
    assert compatibility["required_on_triton_3_7_1"] == (
        "thread_barrier exact alias to barrier"
    )
    assert compatibility["vendor_call_form"] == "zero_argument"
    assert compatibility["barrier_cluster"] is False
    assert compatibility["vendor_source_modified"] is False
    assert metadata["compile_envelope"] == {
        "padded_width_rule": "BD=next_power_of_two(D)",
        "max_padded_width": 4096,
        "source_width_product_rule": "S*BD",
        "max_source_width_product": 262144,
        "checkpoint1_static_work_rule": "33*S*BD",
        "checkpoint1_static_work_multiplier": 33,
        "max_checkpoint1_static_work": 8650752,
    }

    source = (
        _configured_vendor_root() / "fla" / "ops" / "attnres" / "backends" / "gluon.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    barrier_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "thread_barrier"
    ]
    assert barrier_calls
    assert all(not call.args and not call.keywords for call in barrier_calls)


def _available_backend(implementation: str):
    vendor_root = _configured_vendor_root()
    try:
        _native_functions(implementation, vendor_root)
    except ImportError as error:
        if any(os.environ.get(variable) for variable in _VENDOR_ENV_VARS):
            raise
        pytest.skip(str(error))
    return make_model_backend(implementation, vendor_root=vendor_root)


def _assert_operator_case(backend, values, query, expected_values, expected_query):
    output = backend(values, query)
    expected = oracle(expected_values, expected_query)
    torch.testing.assert_close(output, expected, rtol=0.05, atol=0.05)
    upstream = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, (values, query), upstream)
    expected_grads = torch.autograd.grad(expected, (expected_values, expected_query), upstream)
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("implementation", ["triton", "gluon"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fullgraph_changed_inputs_and_value_query_gradients(implementation):
    backend = _available_backend(implementation)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    def run(values, query):
        return backend(values, query)

    compiled = torch.compile(run, fullgraph=True, dynamic=False)
    values = torch.randn(3, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    query = torch.randn(64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected_values = values.detach().clone().requires_grad_()
    expected_query = query.detach().clone().requires_grad_()
    _assert_operator_case(compiled, values, query, expected_values, expected_query)
    graph_count = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)

    changed = torch.randn_like(values, requires_grad=True)
    changed_query = torch.randn_like(query, requires_grad=True)
    expected_changed = changed.detach().clone().requires_grad_()
    expected_changed_query = changed_query.detach().clone().requires_grad_()
    _assert_operator_case(
        compiled, changed, changed_query, expected_changed, expected_changed_query
    )
    assert torch._dynamo.utils.counters["stats"].get("unique_graphs", 0) == graph_count


@pytest.mark.parametrize("implementation", ["triton", "gluon"])
@pytest.mark.parametrize("mode", ["full", "block"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fullgraph_model_changed_inputs_all_parameter_gradients(implementation, mode):
    backend = _available_backend(implementation)
    assert backend.accepts_source_list is True
    config = TrainingConfig(
        layers=1,
        width=64,
        heads=4,
        ffn=128,
        batch=2,
        sequence=8,
        vocab=37,
        block_count=1,
        variant="standard",
        mode=mode,
    )
    torch.manual_seed(41)
    reference = make_model(config, backend="reference").cuda()
    kernel = make_model(config, backend=backend).cuda()
    kernel.load_state_dict(copy.deepcopy(reference.state_dict()))
    compiled = torch.compile(kernel, fullgraph=True, dynamic=False)

    def check(tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = compiled(tokens)
            expected = reference(tokens)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)
        actual_loss = actual.float().mean()
        expected_loss = expected.float().mean()
        actual_grads = torch.autograd.grad(actual_loss, tuple(kernel.parameters()))
        expected_grads = torch.autograd.grad(expected_loss, tuple(reference.parameters()))
        assert len(actual_grads) == len(expected_grads)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0.05, atol=0.05)

    tokens = torch.randint(config.vocab, (config.batch, config.sequence), device="cuda")
    check(tokens)
    check(tokens.roll(1, dims=1))
