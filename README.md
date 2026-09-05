# Fast Attention Residuals

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/tested-PyTorch_2.13-EE4C2C.svg)](https://pytorch.org/)
[![Triton 3.7.1](https://img.shields.io/badge/tested-Triton_3.7.1-654FF0.svg)](https://github.com/jon123boss/fast-attnres)
[![License: MIT](https://img.shields.io/badge/license-MIT-2E7D32.svg)](LICENSE)

Fast-AttnRes provides a CUDA BF16 PyTorch operator for standard and sliced
low-rank Attention Residuals. Give it ordered full-width residual values and
one query; it returns one full-width residual. Full and sequential Block reads
use the same public call.

The H100/B200 BF16 campaign is still pending. This README makes no final timing
claim. The reproducible campaign commands and reporting layer are in
[`docs/bf16_campaign.md`](docs/bf16_campaign.md); the final Markdown report is
not available until the primary measurements finish.

## Install

Install the checkout with the pinned CUDA runtime:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0
python -m pip install -e ".[cuda,test,benchmark]"
```

The campaign uses Python 3.11, PyTorch 2.13.0 with CUDA 13.0, and Triton 3.7.1.
This branch has not been published as a package release.

## One call

```python
import torch
from attnres import attnres

values = torch.randn(8, 2, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
source_list = tuple(values.unbind(0))
query = torch.randn(1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
output = attnres(values, query, eps=2**-23, scale=1.0)
output.square().mean().backward()
compiled = torch.compile(attnres, fullgraph=True)
```

The public signature is `attnres(values, query, *, eps=2**-23, scale=1.0)`.
`values` is either packed `[S, ..., D]` or an ordered list/tuple of `[ ..., D ]`
tensors. The query is `[R]`, with `1 <= R <= D`; the output retains width `D`.
Values, query, output, and first-order operator gradients are CUDA BF16. Internal
FP32 accumulators may be used for normalization, logits, softmax, and reductions.
There is no shipped CPU, FP32, or reference execution product.

## Equation

For source value `v_s`, take its final `R` coordinates as the implicit key:

```text
t_s       = v_s[..., D-R:D]
r_s       = sqrt(mean(t_s ** 2) + eps)
k_s       = t_s / r_s
score_s   = scale * dot(k_s, query)
p_s       = softmax(score, axis=source)_s
output    = sum_s p_s * v_s
```

Normalization and softmax run independently at each carried batch or token
position. The output is not normalized, source-count weighted, or source averaged.
See [`docs/equation.md`](docs/equation.md) for the complete contract.

## Full and Block

Full supplies the embedding and every preceding writer output. Block supplies
the embedding, completed block sums, and an optional current partial sum. The
caller owns block boundaries and sums a partial block before the read; it is
passed as one ordinary source and is not averaged.

```python
full_output = attnres((embedding, *writers), query)

completed = (embedding, first_block, second_block)
block_sources = completed if partial is None else completed + (partial,)
block_output = attnres(block_sources, query)
```

There is no stateful prepared Block object, phase cache, cross-read reuse, or
second public model path. Both schedules use the same source ordering, query,
`eps`, `scale`, and operator.

## Sliced LR-AttnRes

Sliced LR-AttnRes keeps full-width values and output while using a shorter
implicit key and query:

```python
rank = 64
query = torch.randn(rank, device="cuda", dtype=torch.bfloat16)
output = attnres(source_list, query)  # [..., D], BF16
```

For a trainable static query, use `LearnedQuery(rank)` from `attnres.modules`:

```python
from attnres import LearnedQuery
learned_query = LearnedQuery(rank).to(device="cuda", dtype=torch.bfloat16)
output = attnres(source_list, learned_query())
```

Standard AttnRes is the `R == D` case; sliced routing uses `R < D` and the final
`R` value coordinates. Projected keys, routing priors, and architectural changes
are outside this package contract.

## Validation scope

Correctness precedes timing. The BF16 protocol uses `rtol=0.05` and `atol=0.05`
for outputs and first-order gradients, and covers packed/list sources, repeated
reads, partial Blocks, changed inputs, non-contiguous layouts, shared sources,
compiled replay, optimizer updates, and save/resume. See
[`docs/validation.md`](docs/validation.md).

The primary timing campaign uses L24/D1536/H24/FFN4224, vocabulary 100277,
context 2048, batch 4, accumulation 4, and eight Blocks. It uses BF16
cross-entropy, gradient clipping 1.0, the original Muon plus AdamW implementation,
no activation checkpointing, three seeds, and 120 paired timing rounds. Reports
retain failed, incomplete, and inconclusive results alongside verified gains.

## License

Fast-AttnRes is released under the [MIT License](LICENSE). FLA-derived source
list attribution remains in [`NOTICE`](NOTICE).
