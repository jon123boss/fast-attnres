"""CPU and static contracts for the shared source-list kernels."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from attnres._kernels import fixed_tail_sources, fla_full_sources

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "attnres" / "_kernels" / "fla_full_sources.py"


def test_save_mixed_policy_is_structural_and_value_independent():
    one = (torch.empty(2, 4, dtype=torch.bfloat16),)
    three = one + tuple(torch.empty(2, 4, dtype=torch.bfloat16) for _ in range(2))
    four = three + (torch.empty(2, 4, dtype=torch.bfloat16),)
    assert not fla_full_sources._should_save_mixed(one, 2, 4)
    assert not fla_full_sources._should_save_mixed(three, 2, 4)
    assert fla_full_sources._should_save_mixed(four, 2, 4)


def test_production_configs_and_cache_dimensions_are_static():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "_fla_standard_forward_kernel",
        "_fla_standard_backward_kernel",
        "_fla_standard_query_reduce_kernel",
    } <= functions
    assert "_STANDARD_AUTOTUNE_KEY = [" in source
    for field in (
        '"ARCH"',
        '"DTYPE"',
        '"D"',
        '"R"',
        '"L2"',
        '"ROUTE"',
        '"CHECKPOINT"',
    ):
        assert field in source
    assert "for block in (1, 2, 4, 8)" in source
    assert "for warps in (4, 8, 16)" in source
    assert "for stages in (2, 3)" in source
    assert "(1024, 16, 4)" in source
    assert "(2048, 32, 4)" in source
    assert "(2048, 32, 8)" in source
    assert "(4096, 32, 8)" in source
    assert "(4096, 64, 8)" in source
    assert "for stages in (3, 4)" in source
    assert '"N"' in source
    assert '"PIPELINE_STAGES": stages' in source
    assert source.count("num_stages=PIPELINE_STAGES") == 4
    for field in ('"ROW_STRIDES"', '"FEATURE_STRIDES"', '"QUERY_STRIDE"'):
        assert field in source


def test_fake_aux_shape_matches_saved_and_recomputed_reads():
    sources = [torch.empty(2, 4, dtype=torch.bfloat16)]
    low_rank_recompute = fixed_tail_sources._source_forward_fake(
        sources, torch.empty(2), 1e-6, 1.0
    )
    standard_recompute = fixed_tail_sources._source_forward_fake(
        sources, torch.empty(4), 1e-6, 1.0
    )
    assert low_rank_recompute[1].shape == (0, 4)
    assert standard_recompute[1].shape == (0, 4)

    saved = fixed_tail_sources._source_forward_fake(
        sources * 4, torch.empty(4), 1e-6, 1.0
    )
    assert saved[1].shape == (2, 4)


def test_fake_saved_mixed_rank_is_stable_across_block_source_counts():
    rows, width = 8, 1536
    query = torch.empty(width, dtype=torch.float32)
    shapes = []
    for source_count in range(2, 10):
        sources = [
            torch.empty(rows, width, dtype=torch.bfloat16)
            for _ in range(source_count)
        ]
        outputs = fixed_tail_sources._source_forward_fake(
            sources, query, 1e-6, 1.0
        )
        shapes.append(outputs[1].shape)
        assert outputs[0].shape == (rows, width)
        assert outputs[2].shape == (source_count, rows)
        assert outputs[3].shape == (source_count, rows)
        assert outputs[4].shape == (rows,)

    assert all(len(shape) == 2 for shape in shapes)
    assert shapes[:2] == [(0, width), (0, width)]
    assert shapes[2:] == [(rows, width)] * 6


def test_fake_backward_abi_returns_one_gradient_per_source_and_query():
    rows, width = 3, 16
    query = torch.empty(width, dtype=torch.float32)
    for source_count in (2, 4, 9):
        sources = [
            torch.empty(rows, width, dtype=torch.bfloat16)
            for _ in range(source_count)
        ]
        aux = fixed_tail_sources._source_forward_fake(
            sources, query, 1e-6, 1.0
        )
        gradients = fixed_tail_sources._source_backward_fake(
            sources,
            query,
            aux[1],
            torch.empty(rows, width, dtype=torch.bfloat16),
            aux[2],
            aux[3],
            aux[4],
            1.0,
        )
        assert len(gradients) == source_count + 1
        assert [gradient.shape for gradient in gradients[:-1]] == [
            source.shape for source in sources
        ]
        assert gradients[-1].shape == query.shape
