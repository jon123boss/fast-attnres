"""CUDA correctness gate for the scalar compact FLA source route."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from attnres import attnres
from attnres._kernels import fla_full_sources
from validation.oracle import oracle


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "fla_scalar_compact_gate_v1.json"
SOURCE_PATH = ROOT / "src" / "attnres" / "_kernels" / "fla_full_sources.py"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CASES = tuple(
    (source_count, rank)
    for source_count in CONFIG["protocol"]["source_counts"]
    for rank in CONFIG["protocol"]["ranks"]
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_value(config, name: str) -> int:
    return int(config.all_kwargs()[name])


def _select_config(kernel, family: int, block: int):
    selected = [
        config
        for config in kernel.configs
        if _config_value(config, "LAYOUT_FAMILY") == family
        and _config_value(config, "BL") == block
        and int(config.num_warps) == 4
        and int(config.num_stages) == 2
    ]
    assert len(selected) == 1, (family, block, selected)
    return selected[0]


@contextmanager
def _forced_config(kernel, family: int, block: int):
    original_configs = kernel.configs
    original_cache = dict(kernel.cache)
    had_best = hasattr(kernel, "best_config")
    original_best = getattr(kernel, "best_config", None)
    kernel.configs = [_select_config(kernel, family, block)]
    kernel.cache.clear()
    if had_best:
        kernel.best_config = None
    try:
        yield kernel
    finally:
        kernel.configs = original_configs
        kernel.cache.clear()
        kernel.cache.update(original_cache)
        if had_best:
            kernel.best_config = original_best


@contextmanager
def _forced_forward_and_backward(backward_family: int, block: int):
    with _forced_config(fla_full_sources._fla_source_forward_kernel, 0, 1):
        with _forced_config(
            fla_full_sources._fla_source_backward_kernel, backward_family, block
        ) as backward:
            yield backward


def test_gate_identity_and_structural_keys_are_frozen():
    candidate = CONFIG["candidate"]
    assert CONFIG["schema"] == "attnres.fla_scalar_compact_gate.v1"
    assert candidate["base_commit"] == "95811f9de6749186c11166f7ab37197084684d79"
    assert candidate["implementation_commit"] == (
        "25a85a9b99985ac90d69ce636d6b42b5f636a129"
    )
    assert candidate["source_sha256"] == (
        "2cd7ac89b15faeb13640bff4a7948e437453b69446bfc8c7922511e341843e10"
    )
    assert candidate["source_sha256"] != _sha256(SOURCE_PATH)
    assert CONFIG["protocol"]["forward_key"] == ["L2", "D", "R"]
    assert CONFIG["protocol"]["backward_key"] == ["L2", "D", "R"]
    assert CONFIG["protocol"]["backward_families"] == [0, 1, 2]
    if hasattr(fla_full_sources, "_fla_source_forward_kernel"):
        assert len(fla_full_sources._fla_source_forward_kernel.configs) == 16
        backward_configs = fla_full_sources._fla_source_backward_kernel.configs
        assert len(backward_configs) == 22
        assert sum(
            _config_value(config, "LAYOUT_FAMILY") == 2
            for config in backward_configs
        ) == 6
    else:
        source = SOURCE_PATH.read_text(encoding="utf-8")
        assert "_BACKWARD_SOURCE_BLOCK_CONFIGS" in source
        assert "for layout_family in (0, 1, 2)" in source
        assert "if layout_family == 2 else (1, 2, 4, 8)" in source
    assert set(CASES) == {
        (source_count, rank)
        for source_count in (2, 9)
        for rank in (128, 512, 1024)
    }


@pytest.mark.cuda
@pytest.mark.skipif(
    CONFIG["candidate"]["source_sha256"] != _sha256(SOURCE_PATH),
    reason="historical scalar-compact gate does not qualify the active source",
)
@pytest.mark.parametrize("backward_family", (0, 1, 2))
@pytest.mark.parametrize("case_index,case", tuple(enumerate(CASES)))
def test_public_source_route_matches_bf16_oracle(
    case_index: int, case: tuple[int, int], backward_family: int
):
    source_count, rank = case
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")

    torch.manual_seed(20260830 + source_count * 100 + rank)
    rows = int(CONFIG["protocol"]["rows"])
    width = int(CONFIG["protocol"]["width"])
    values = tuple(
        torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(source_count)
    )
    reference_values = tuple(
        value.detach().clone().requires_grad_(True) for value in values
    )
    query = torch.randn(rank, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_query = query.detach().clone().requires_grad_(True)

    block = (
        (2, 4, 8)[case_index % 3]
        if backward_family == 2
        else (1, 2, 4, 8)[case_index % 4]
    )
    with _forced_forward_and_backward(backward_family, block) as backward:
        actual = attnres(values, query)
        expected = oracle(torch.stack(reference_values), reference_query)
        torch.manual_seed(20270830 + source_count * 100 + rank)
        upstream = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(actual, (*values, query), upstream)
        expected_gradients = torch.autograd.grad(
            expected, (*reference_values, reference_query), upstream
        )
        assert len(backward.configs) == 1
        assert _config_value(backward.configs[0], "LAYOUT_FAMILY") == backward_family

    for observed, reference in zip(
        (actual, *actual_gradients), (expected, *expected_gradients)
    ):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(
            observed,
            reference,
            rtol=float(CONFIG["protocol"]["rtol"]),
            atol=float(CONFIG["protocol"]["atol"]),
        )
