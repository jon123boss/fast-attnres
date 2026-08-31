from __future__ import annotations

import copy

import pytest
import torch

from benchmarks.model import TrainingConfig, make_model, training_step


def _tiny_config(**kwargs) -> TrainingConfig:
    values = dict(
        layers=2,
        width=16,
        heads=4,
        ffn=32,
        batch=2,
        sequence=5,
        vocab=31,
        block_count=2,
    )
    values.update(kwargs)
    return TrainingConfig(**values)


def test_training_example_requires_an_explicit_sliced_rank(monkeypatch, capsys):
    from examples.train import _parse_args

    monkeypatch.setattr("sys.argv", ["train.py", "--variant", "sliced"])
    with pytest.raises(SystemExit) as error:
        _parse_args()
    assert error.value.code == 2
    assert "--rank is required" in capsys.readouterr().err
    monkeypatch.setattr("sys.argv", ["train.py", "--variant", "sliced", "--rank", "16"])
    assert _parse_args().rank == 16
    monkeypatch.setattr("sys.argv", ["train.py"])
    args = _parse_args()
    assert args.variant == "standard" and args.rank is None


def test_standard_variant_requires_full_rank():
    with pytest.raises(ValueError, match="rank == width"):
        TrainingConfig(width=16, heads=4, rank=4, variant="standard")
    assert TrainingConfig(width=16, heads=4, variant="standard").rank == 16


@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("mode", ["full", "block"])
def test_model_shape_and_nonzero_queries(variant, mode):
    torch.manual_seed(5)
    rank = 16 if variant == "standard" else 4
    model = make_model(_tiny_config(variant=variant, mode=mode, rank=rank), backend="reference")
    tokens = torch.randint(31, (2, 5))
    logits = model(tokens)
    assert logits.shape == (2, 5, 31)
    assert all(torch.count_nonzero(query).item() for query in model.queries)


@pytest.mark.parametrize("source_layout", ["packed", "list"])
@pytest.mark.parametrize("variant", ["standard", "sliced"])
@pytest.mark.parametrize("mode", ["full", "block"])
def test_reference_and_kernel_state_and_gradient_parity_on_cpu(variant, mode, source_layout):
    config = _tiny_config(variant=variant, mode=mode, rank=16 if variant == "standard" else 4,
                          source_layout=source_layout)
    torch.manual_seed(7)
    reference = make_model(config, backend="reference")
    kernel = make_model(config, backend="kernel")
    kernel.load_state_dict(copy.deepcopy(reference.state_dict()))
    tokens = torch.randint(config.vocab, (config.batch, config.sequence))
    targets = torch.randint(config.vocab, (config.batch, config.sequence))
    logits_reference = reference(tokens)
    logits_kernel = kernel(tokens)
    torch.testing.assert_close(logits_reference, logits_kernel, rtol=1e-5, atol=1e-6)
    loss_reference = logits_reference.float().mean()
    loss_kernel = logits_kernel.float().mean()
    loss_reference.backward()
    loss_kernel.backward()
    reference_parameters = dict(reference.named_parameters())
    kernel_parameters = dict(kernel.named_parameters())
    assert reference_parameters.keys() == kernel_parameters.keys()
    for name in reference_parameters:
        torch.testing.assert_close(
            reference_parameters[name].grad,
            kernel_parameters[name].grad,
            rtol=1e-5,
            atol=1e-6,
            msg=name,
        )


def test_training_step_updates_weights_and_supports_accumulation():
    config = _tiny_config(layers=1, block_count=1, variant="sliced", mode="full", rank=3)
    model = make_model(config, backend="reference")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tokens = torch.randint(config.vocab, (2, config.batch, config.sequence))
    targets = torch.randint(config.vocab, (2, config.batch, config.sequence))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss = training_step(model, optimizer, tokens, targets, accumulation=2)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())


@pytest.mark.parametrize("variant", ["standard", "sliced"])
def test_block_callable_keeps_source_schedule_and_all_gradients(variant):
    from validation.oracle import oracle

    sources_seen = []

    def backend(values, query):
        sources_seen.append(values.shape[0])
        return oracle(values, query)

    config = _tiny_config(variant=variant, mode="block", rank=16 if variant == "standard" else 4)
    torch.manual_seed(13)
    reference = make_model(config, backend="reference")
    candidate = make_model(config, backend=backend)
    candidate.load_state_dict(reference.state_dict())
    tokens = torch.randint(config.vocab, (config.batch, config.sequence))
    expected, actual = reference(tokens), candidate(tokens)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.sum().backward()
    actual.sum().backward()
    for left, right in zip(candidate.parameters(), reference.parameters()):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
    assert sources_seen == [2, 2, 3, 3]


@pytest.mark.parametrize("mode", ["full", "block"])
@pytest.mark.parametrize("variant,rank,opt_in", [
    ("standard", 16, True), ("sliced", 16, True),
    ("sliced", 4, True), ("standard", 16, False),
])
def test_native_source_list_preserves_order_and_all_gradients(mode, variant, rank, opt_in):
    from validation.oracle import oracle

    recorded = [[], []]
    native_sources = []
    expects_list = opt_in and rank == 16

    def stacked(values, query):
        assert isinstance(values, torch.Tensor)
        recorded[0].append(values.detach())
        return oracle(values, query)

    def native(values, query):
        assert isinstance(values, tuple) == expects_list
        if expects_list:
            native_sources.append(values)
            values = torch.stack(values)
        recorded[1].append(values.detach())
        return oracle(values, query)

    if opt_in:
        native.accepts_source_list = True
    config = _tiny_config(variant=variant, mode=mode, rank=rank)
    torch.manual_seed(17)
    reference = make_model(config, backend=stacked)
    candidate = make_model(config, backend=native)
    candidate.load_state_dict(reference.state_dict())
    tokens = torch.randint(config.vocab, (config.batch, config.sequence))
    expected, actual = reference(tokens), candidate(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    actual.sum().backward()
    for left, right in zip(candidate.parameters(), reference.parameters()):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    assert [v.shape[0] for v in recorded[1]] == (
        [2, 3, 4, 5] if mode == "full" else [2, 2, 3, 3]
    )
    for left, right in zip(recorded[0], recorded[1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    if expects_list:
        # Preserve original source objects, not newly stacked-and-unbound views.
        assert all(v[0] is native_sources[0][0] for v in native_sources)


def test_checkpoint_restores_model_and_optimizer_state():
    config = _tiny_config(layers=1, block_count=1, variant="sliced", mode="block", rank=3)
    torch.manual_seed(9)
    model = make_model(config, backend="reference")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(config.vocab, (config.batch, config.sequence))
    targets = torch.randint(config.vocab, (config.batch, config.sequence))
    training_step(model, optimizer, tokens, targets)
    model_copy = make_model(config, backend="reference")
    optimizer_copy = torch.optim.AdamW(model_copy.parameters(), lr=1e-3)
    model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
    optimizer_copy.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    torch.testing.assert_close(model(tokens), model_copy(tokens), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["full", "block"])
@pytest.mark.parametrize("seed", [20260827, 99123])
def test_canonical_rank_states_preserve_common_parameters_and_coordinate_mapping(mode, seed):
    from benchmarks.model import canonical_max_rank_state, make_model_with_canonical_state

    config = _tiny_config(variant="sliced", mode=mode, rank=16)
    before = torch.random.get_rng_state()
    try:
        torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        independent = make_model(config, backend="reference").state_dict()
    finally:
        torch.random.set_rng_state(before)
    canonical = canonical_max_rank_state(config, seed)
    torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
    assert set(canonical) == set(independent)
    for name in canonical:
        torch.testing.assert_close(canonical[name], independent[name], rtol=0, atol=0)

    for variant, rank in [("standard", 16), ("sliced", 1), ("sliced", 7),
                          ("sliced", 16)]:
        target_config = _tiny_config(variant=variant, mode=mode, rank=rank)
        target = make_model_with_canonical_state(target_config, "reference", canonical, seed + 1)
        torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
        for name, actual in target.state_dict().items():
            expected = independent[name]
            if name.startswith("queries."):
                expected = expected[-rank:] if variant == "sliced" else expected[:rank]
                assert torch.count_nonzero(actual) == actual.numel()
            assert actual.device.type == "cpu"
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # A loaded target must never mutate the canonical source through aliases.
        first = next(target.parameters())
        with torch.no_grad():
            first.add_(1)
        for name in canonical:
            torch.testing.assert_close(canonical[name], independent[name], rtol=0, atol=0)
