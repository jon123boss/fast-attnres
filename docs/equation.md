# Equation and API contract

This document defines the CUDA BF16 implicit-tail operator in Fast-AttnRes.
The terminology follows the [Attention Residuals paper](https://arxiv.org/abs/2603.15031)
and the [Low-Rank Attention Residuals paper](https://arxiv.org/abs/2607.09694).

## Public call and source containers

The public functional entry point is:

```text
attnres(values, query, *, eps=2**-23, scale=1.0)
```

`values` may be a packed source-major tensor or an ordered source container:

| Input | Packed form | List/tuple form |
| --- | --- | --- |
| `values` | `[S, ..., D]` | `S` tensors shaped `[..., D]` |

The source axis `S` is reduced. In list/tuple form, each element is one source
and has no source axis of its own. Every source must have the same logical
shape, BF16 dtype, and CUDA device; source order is preserved. The query has
shape `[R]`, and the result has shape `[..., D]`.

The documented runtime contract is CUDA BF16: values, query, output, and
first-order operator gradients are BF16. `eps` must be finite and positive,
with default `2**-23`; `scale` must be finite, with default `1.0`.

The implementation may use FP32 accumulators for RMS normalization, routing
logits, softmax, and value/gradient reductions. That is internal arithmetic;
it does not change the BF16 input, output, or gradient contract.

## Implicit-tail equation

For source `s`, let `v_s` be the full-width value and let `t_s` be its final
`R` coordinates. The operator computes:

```text
t_s       = v_s[..., D-R:D]
r_s       = sqrt(mean(t_s ** 2) + eps)
k_s       = t_s / r_s
score_s   = scale * dot(k_s, query)
p_s       = softmax(score, axis=source)_s
output    = sum_s p_s * v_s
```

Normalization and softmax are applied independently at every carried batch or
token position. The output is full width `D`; there is no output normalization,
source-count prior, or source averaging.

## Standard AttnRes

Standard AttnRes uses `R == D`, so each full-width value is also its implicit
routing key. The source axis is reduced with one softmax. Packed values and
ordered source lists use the same equation and public call.

## Sliced LR-AttnRes

Sliced LR-AttnRes chooses `R < D`. Values and output remain full width, while
the query and every implicit routing key use only the final `R` value
coordinates:

```python
import torch
from attnres import attnres

rank = 64
query = torch.nn.Parameter(
    torch.randn(rank, device="cuda", dtype=torch.bfloat16)
)
output = attnres(sources, query)  # full-width BF16 output, [..., D]
```

`attnres.modules.LearnedQuery(rank)` supplies a trainable static query. Move
the module to CUDA BF16 before calling it:

```python
from attnres import LearnedQuery, attnres

learned_query = LearnedQuery(rank).to(device="cuda", dtype=torch.bfloat16)
output = attnres(sources, learned_query())
```

## Execution and gradients

The public execution path is CUDA BF16. Values and query cross the operator
boundary in BF16, and the output and first-order value/query gradients remain
BF16. Internal FP32 accumulation is permitted for numerical stability and does
not expose an additional public storage mode.

The first-order backward differentiates values and the query. For an implicit
key, the routing gradient through the RMS-normalized value tail is combined
with the direct value gradient before the BF16 storage boundary. Lists,
repeated sources, and shared views remain ordinary autograd edges. Second-order
behavior is outside this contract.

The [`examples/backward.py`](../examples/backward.py) script checks value and
query gradients on CUDA BF16 inputs.

## Full and per-read Block schedules

Full and sequential Block schedules use the same public `attnres` call. Full
supplies the embedding and all preceding writer outputs. Block changes only
when reads happen and which ordinary source tensors are supplied:

```python
# Full source assembly.
full_sources = (embedding, *writers)
full_output = attnres(full_sources, query)

# Block source assembly.
completed = (embedding, first_block, second_block)
partial = current_partial  # BF16 tensor, or None at a block boundary
read_sources = completed if partial is None else completed + (partial,)
block_output = attnres(read_sources, query)
```

The caller owns block boundaries, partial sums, source order, and learned
queries. A partial block is summed before the read and passed as one ordinary
source; it is not averaged. Each Block read receives ordinary tensors directly
through the same public function.

The [`examples/block_schedules.py`](../examples/block_schedules.py) script
constructs completed block sums and partial sources, then calls `attnres` for
each read.
