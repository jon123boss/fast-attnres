# Fast Attention Residuals

[![CI](https://github.com/jon123boss/fast-attnres/actions/workflows/ci.yml/badge.svg)](https://github.com/jon123boss/fast-attnres/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Tested PyTorch 2.13](https://img.shields.io/badge/tested-PyTorch_2.13-EE4C2C.svg)](https://pytorch.org/)
[![Tested Triton 3.7.1](https://img.shields.io/badge/tested-Triton_3.7.1-654FF0.svg)](https://github.com/triton-lang/triton/releases/tag/v3.7.1)
[![License: MIT](https://img.shields.io/badge/license-MIT-2E7D32.svg)](https://github.com/jon123boss/fast-attnres/blob/main/LICENSE)

**Fast Attention Residuals** (`Fast-AttnRes`) provides a CUDA BF16 PyTorch
operator for Attention Residuals. Pass ordered full-width residual values and
one query; receive one full-width residual. The same
`attnres(values, query)` call is used for standard Full reads and sequential
Block reads.

This branch is undergoing H100/B200 qualification. Both GPUs have passed the
current BF16 output/gradient, alias, compilation, changed-input CUDA Graph,
checkpoint, and save/resume checks. Complete-step comparisons and distributed
qualification are still in progress; this is not yet a final performance claim.

## Runtime contract

The public call is:

```text
attnres(values, query, *, eps=2**-23, scale=1.0)
```

- `values` is either a packed `[S, ..., D]` tensor or an ordered list/tuple of
  `[..., D]` tensors.
- `values` and `query` are CUDA `torch.bfloat16` tensors. The query has shape
  `[R]`, with `1 <= R <= D`.
- The output retains value width `D` and is `torch.bfloat16`.
- First-order gradients of values, query, and operator output are
  `torch.bfloat16` when their inputs are BF16.
- Implementations may use FP32 accumulators for normalization, logits, softmax,
  and value/gradient reductions. This internal arithmetic does not widen the
  public storage or gradient contract.
- Standard AttnRes uses `R == D`. Sliced LR-AttnRes uses `R < D` and takes its
  routing key from the final `R` coordinates of each full-width value.

The public surface has one operator and ordinary source assembly. Block reuses
that operator at each read and does not add a stateful read object or a second
model path.

## Install

Install this draft from its checkout with the tested CUDA runtime:

```bash
git clone --branch codex/h100-b200-optimization https://github.com/jon123boss/fast-attnres.git
cd fast-attnres
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0
python -m pip install -e ".[cuda,test,benchmark]"
```

The campaign pins PyTorch 2.13.0, CUDA 13.0, Triton 3.7.1, and Python 3.11.
Triton compiles kernels for the selected GPU. This branch has not been published
as a package release.

## Quickstart: standard AttnRes

Standard AttnRes uses a full-width query (`R == D`):

```python
import torch

from attnres import attnres

device = torch.device("cuda")
dtype = torch.bfloat16
batch, width, source_count = 2, 1024, 8

values = torch.randn(
    source_count, batch, width, device=device, dtype=dtype, requires_grad=True
)
query = torch.nn.Parameter(torch.randn(width, device=device, dtype=dtype))

output = attnres(values, query)  # [batch, width], BF16
loss = output.square().mean()
loss.backward()
assert output.dtype == dtype
assert values.grad is not None and values.grad.dtype == dtype
assert query.grad is not None and query.grad.dtype == dtype
```

The operator accepts an ordered source list as well:

```python
source_list = tuple(values.unbind(0))
output_from_list = attnres(source_list, query)
torch.testing.assert_close(output_from_list, output, rtol=0.05, atol=0.05)
```

## Equation

For source `s`, let `v_s` be its full-width value and let `t_s` be its final
`R` coordinates. The implicit-tail routing equation is:

```text
t_s       = v_s[..., D-R:D]
r_s       = sqrt(mean(t_s ** 2) + eps)
k_s       = t_s / r_s
score_s   = scale * dot(k_s, query)
p_s       = softmax(score, axis=source)_s
output    = sum_s p_s * v_s
```

Normalization and softmax are applied independently at every carried batch or
token position. There is no output normalization, source-count prior, or
source averaging. The detailed contract is in
[`docs/equation.md`](docs/equation.md).

## Full and Block schedules

Full supplies the embedding and all preceding writer outputs to one read. A
sequential Block schedule supplies the embedding, completed block sums, and a
currently accumulated partial block as ordinary sources. The caller performs
the block sums and calls the same public function:

```python
# Full: embedding plus every writer remains visible.
full_sources = (embedding, *writers)
full_output = attnres(full_sources, query)

# Block: each completed block is one ordinary source.
completed = (embedding, first_block, second_block)
partial = current_partial  # one BF16 tensor, or None at a block boundary
block_sources = completed if partial is None else completed + (partial,)
block_output = attnres(block_sources, query)
```

A partial block is summed before the call and is not averaged. Full and Block
therefore share source ordering, query, `eps`, `scale`, and the public
operator. Each Block read receives ordinary source tensors directly.

## Sliced LR-AttnRes

Sliced LR-AttnRes keeps full-width values and output while using a shorter
implicit key and query:

```python
import torch

from attnres import attnres

rank = 64
query = torch.nn.Parameter(torch.randn(rank, device="cuda", dtype=torch.bfloat16))
output = attnres(source_list, query)  # [..., D], BF16
```

For a trainable static query, `attnres.modules.LearnedQuery(rank)` can be moved
to CUDA BF16:

```python
from attnres import LearnedQuery, attnres

learned_query = LearnedQuery(rank).to(device="cuda", dtype=torch.bfloat16)
output = attnres(source_list, learned_query())
```

The [Low-Rank Attention Residuals paper](https://arxiv.org/abs/2607.09694)
describes the sliced construction.

## PyTorch integration

`attnres` is an ordinary PyTorch call and can be wrapped by `torch.compile` or
checkpointing in a CUDA BF16 graph:

```python
import torch
from torch.utils.checkpoint import checkpoint

from attnres import attnres

values = tuple(
    torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for _ in range(4)
)
query = torch.nn.Parameter(torch.randn(1024, device="cuda", dtype=torch.bfloat16))

def residual(source_values, source_query):
    return attnres(source_values, source_query)

compiled_residual = torch.compile(residual)
compiled_output = compiled_residual(values, query)
checkpointed_output = checkpoint(residual, values, query, use_reentrant=False)
```

The examples cover packed and list sources, backward gradients, Block source
assembly, sliced routing, and compiled execution:

```bash
python examples/standard_attnres.py --device cuda
python examples/packed_and_list.py --device cuda
python examples/block_schedules.py --device cuda
python examples/lr_attnres.py --device cuda
python examples/backward.py --device cuda
python examples/torch_compile.py --device cuda
python examples/train.py --device cuda
```

Every example rejects a non-CUDA device and constructs values, queries, outputs,
and operator gradients in BF16. The examples are smoke checks, not timing
benchmarks.

## Validation

Candidate validation must use `rtol=0.05` and `atol=0.05` for BF16 output and
first-order gradient comparisons. It should cover packed and ordered sources,
standard and sliced ranks, repeated reads, completed plus partial Block source
sets, changed inputs, non-contiguous layouts, duplicate/shared sources, and
compiled replay. Correctness must be established before any timing is
considered. See [`docs/validation.md`](docs/validation.md) for the protocol.

## Historical v1.0.0 performance evidence

The following artifacts were measured for the tagged v1.0.0 implementation.
They are retained for provenance and comparison, and do not qualify this
checkout or any later candidate:

| Archive | Workload | Historical measured result |
| --- | --- | --- |
| [`docs/current_24l_results.md`](docs/current_24l_results.md) | 24-layer Full, BF16, `B=2`, `T=1024`, three seeds | 5.20% lower complete-step time on H100 SXM and 15.52% lower on B200 versus the pinned FLA route |
| [`docs/compiled_step_results.md`](docs/compiled_step_results.md) | 8-layer compiled complete-step screen, BF16, named H100/B200 cells | Per-cell paired results and intervals in the archived report |

Those are GPU measurements with their named configurations and archived
reports. No new timing number is inferred from this documentation patch.

## Citation and license

Please cite [Attention Residuals](https://arxiv.org/abs/2603.15031) for the
base method and [Low-Rank Attention Residuals](https://arxiv.org/abs/2607.09694)
for the sliced extension. For this implementation:

```bibtex
@misc{su2026attnreskernels,
  author  = {Jonathan Su},
  title   = {Fast Attention Residuals},
  year    = {2026},
  url     = {https://github.com/jon123boss/fast-attnres}
}
```

The package is released under the [MIT License](https://github.com/jon123boss/fast-attnres/blob/main/LICENSE).
The FLA-derived source-list attribution and license notice remain in
[`NOTICE`](NOTICE).
