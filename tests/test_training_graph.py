from __future__ import annotations

import copy
from dataclasses import replace
import os

import pytest
import torch
from torch.nn import functional as F

from benchmarks.model import TrainingConfig, make_model
from benchmarks.training_graph import capture_training_step


def _ordinary_step(model, optimizer, compiled_loss, tokens, targets, accumulation=1):
    optimizer.zero_grad(set_to_none=True)
    result = None
    if tokens.ndim == 3:
        token_batches = tokens.unbind(0)
        target_batches = targets.unbind(0)
    else:
        token_batches = tokens.chunk(accumulation, dim=0)
        target_batches = targets.chunk(accumulation, dim=0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for micro_tokens, micro_targets in zip(token_batches, target_batches):
            logits = model(micro_tokens)
            loss = compiled_loss(logits.reshape(-1, logits.shape[-1]), micro_targets.reshape(-1))
            result = loss
            (loss / accumulation).backward()
    optimizer.step()
    return result.detach()


def test_training_graph_module_imports_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CPU import guard is only exercised without CUDA")
    config = TrainingConfig(layers=1, width=8, heads=2, ffn=16, batch=1, sequence=3, vocab=17)
    model = make_model(config, backend="reference")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(config.vocab, (config.batch, config.sequence))
    with pytest.raises(RuntimeError, match="CUDA"):
        capture_training_step(model, optimizer, tokens, tokens.clone())


def _assert_close_state(graph_model, ordinary_model, graph_optimizer, ordinary_optimizer):
    for name, value in graph_model.state_dict().items():
        torch.testing.assert_close(
            value,
            ordinary_model.state_dict()[name],
            rtol=0.05,
            atol=0.05,
            msg=name,
        )
    ordinary_parameters = list(ordinary_model.parameters())
    for graph_parameter, ordinary_parameter in zip(graph_model.parameters(), ordinary_parameters):
        assert graph_parameter.grad is not None
        assert ordinary_parameter.grad is not None
        torch.testing.assert_close(
            graph_parameter.grad,
            ordinary_parameter.grad,
            rtol=0.05,
            atol=0.05,
        )
    for graph_parameter, ordinary_parameter in zip(graph_model.parameters(), ordinary_parameters):
        graph_state = graph_optimizer.state[graph_parameter]
        ordinary_state = ordinary_optimizer.state[ordinary_parameter]
        assert graph_state.keys() == ordinary_state.keys()
        for key in graph_state:
            if isinstance(graph_state[key], torch.Tensor):
                torch.testing.assert_close(
                    graph_state[key], ordinary_state[key], rtol=0.05, atol=0.05, msg=key
                )
            else:
                assert graph_state[key] == ordinary_state[key]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_changed_input_two_step_graph_replay_matches_compiled_training():
    config = TrainingConfig(
        layers=1,
        width=16,
        heads=4,
        ffn=32,
        batch=2,
        sequence=5,
        vocab=31,
        block_count=1,
        variant="sliced",
        mode="full",
        rank=4,
    )
    _check_complete_graph(config)


def _check_complete_graph(config, graph_backend="kernel", ordinary_backend="kernel",
                          ordinary_layout=None):
    torch.manual_seed(23)
    graph_model = make_model(config, backend=graph_backend).cuda()
    ordinary_config = replace(config, source_layout=ordinary_layout) if ordinary_layout else config
    ordinary_model = make_model(ordinary_config, backend=ordinary_backend).cuda()
    ordinary_model.load_state_dict(copy.deepcopy(graph_model.state_dict()))
    graph_optimizer = torch.optim.AdamW(
        graph_model.parameters(), lr=1e-3, fused=True, capturable=True
    )
    ordinary_optimizer = torch.optim.AdamW(
        ordinary_model.parameters(), lr=1e-3, fused=True, capturable=True
    )
    ordinary_compiled = torch.compile(ordinary_model, fullgraph=True, dynamic=False)
    ordinary_loss = torch.compile(lambda x, y: F.cross_entropy(x, y), fullgraph=True, dynamic=False)
    initial_tokens = torch.randint(config.vocab, (config.batch, config.sequence), device="cuda")
    initial_targets = torch.randint_like(initial_tokens, config.vocab)
    graph_step = capture_training_step(
        graph_model,
        graph_optimizer,
        initial_tokens,
        initial_targets,
        accumulation=2,
    )
    for shift in (0, 1):
        tokens = initial_tokens.roll(shift, dims=1)
        targets = initial_targets.roll(shift, dims=1)
        expected = _ordinary_step(
            ordinary_compiled,
            ordinary_optimizer,
            ordinary_loss,
            tokens,
            targets,
            accumulation=2,
        )
        actual = graph_step.replay(tokens, targets)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)
        _assert_close_state(
            graph_model,
            ordinary_model,
            graph_optimizer,
            ordinary_optimizer,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("mode", ["full", "block"])
def test_source_list_complete_graph_matches_packed_training(variant, mode):
    config = TrainingConfig(layers=1, width=64, heads=4, ffn=128, batch=2,
                            sequence=8, vocab=37, block_count=1, variant=variant,
                            mode=mode, rank=64 if variant == "standard" else 16,
                            source_layout="list")
    _check_complete_graph(config, ordinary_layout="packed")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("block_count", [1, 3])
def test_per_read_block_complete_graph_matches_reference_training(variant, block_count):
    config = TrainingConfig(layers=2, width=128, heads=4, ffn=256, batch=2,
                            sequence=8, vocab=37, block_count=block_count, variant=variant,
                            mode="block", rank=128 if variant == "standard" else 16,
                            source_layout="list")
    _check_complete_graph(config, ordinary_backend="reference")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("variant,rank", [("standard", 64), ("sliced", 16), ("sliced", 48)])
@pytest.mark.parametrize("mode", ["full", "block"])
def test_fixed_tail_source_complete_graph_matches_reference(variant, rank, mode):
    from attnres._kernels.fixed_tail_sources import source_attnres

    config = TrainingConfig(layers=2, width=64, heads=4, ffn=128, batch=2,
                            sequence=8, vocab=37, block_count=2, variant=variant,
                            mode=mode, rank=rank, source_layout="list")
    _check_complete_graph(config, graph_backend=source_attnres,
                          ordinary_backend="reference")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("implementation", ["triton", "gluon"])
@pytest.mark.parametrize("mode", ["full", "block"])
def test_native_fla_complete_graph_matches_reference_training(implementation, mode):
    from benchmarks.fla_compile import make_model_backend, resolve_vendor_root, _native_functions

    try:
        root = resolve_vendor_root()
        _native_functions(implementation, root)
    except ImportError as error:
        if any(os.environ.get(name) for name in
               ("FLA_ROOT", "FLASH_LINEAR_ATTENTION_ROOT", "VENDOR_FLA_ROOT")):
            raise
        pytest.skip(str(error))
    backend = make_model_backend(implementation, vendor_root=root)
    config = TrainingConfig(layers=1, width=64, heads=4, ffn=128, batch=2,
                            sequence=8, vocab=37, block_count=1,
                            variant="standard", mode=mode, source_layout="list")
    _check_complete_graph(config, graph_backend=backend, ordinary_backend="reference")
