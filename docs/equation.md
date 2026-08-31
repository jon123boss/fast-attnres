# Equation and API contract

This document describes the public implicit-tail operator in Fast-AttnRes.
The terminology follows the [Attention Residuals paper](https://arxiv.org/abs/2603.15031)
and the [Low-Rank Attention Residuals paper](https://arxiv.org/abs/2607.09694).

## Public call and source containers

The public functional entry point is:

```python
attnres(values, query, *, eps=2**-23, scale=1.0)
```

`values` may be a packed source-major tensor or an ordered source container:

| Input | Packed form | List/tuple form |
| --- | --- | --- |
| `values` | `[S, ..., D]` | `S` tensors shaped `[..., D]` |

The source axis `S` is reduced. In the list/tuple form, each element is one
source and has no source axis of its own. Every source must have the same
logical shape, storage dtype, and device; source order is preserved. The query
has shape `[R]`, and the result has shape `[..., D]`.

The supported envelope is `1 <= S <= 129`, `1 <= D <= 8192`, and `1 <= R <= D`.
Values and queries use BF16 or FP32 storage. Equation math is evaluated in
FP32 and the result is cast back to the values' storage dtype. `eps` must be
finite and positive, with default `2**-23`; `scale` must be finite, with
default `1.0`.

BF16 storage and autocast are the performance target for CUDA training. FP32
storage remains available for the explicit equation/reference path and for
debugging. Passing an FP32 correctness check does not make the route an FP32
performance target, and the published training timings should be read as
BF16-targeted measurements.

## Standard AttnRes

Standard AttnRes uses `R == D`, so each full-width value is also its routing
key. The source axis is reduced with one softmax and the output retains all
`D` value coordinates. This route can use either packed values or an ordered
list/tuple of source tensors.

For source `s`, let `v_s` be the full-width value and let `t_s` be its final
`R` coordinates. The implicit key and source mixture are:

```text
t_s       = v_s[..., D-R:D]
r_s       = sqrt(mean(t_s ** 2) + eps)
k_s       = t_s / r_s
score_s   = scale * dot(k_s, query)
p_s       = softmax(score, axis=source)_s
output    = sum_s p_s * v_s
```

For standard AttnRes, `R == D` and therefore `t_s == v_s`. Normalization and
softmax are applied independently at every carried batch or token position.
There is no output normalization, source-count prior, or averaging.

## Sliced LR-AttnRes

Sliced LR-AttnRes chooses `R < D`. Values remain full width, while the query
and every implicit routing key use only the final `R` value coordinates. The
output is still `[..., D]`:

```python
rank = 64
query = torch.nn.Parameter(torch.randn(rank, device=device))
output = attnres(sources, query)  # full-width output, [..., D]
```

`attnres.modules.LearnedQuery(rank)` supplies one trainable static query
parameter. It does not add a routing prior, dynamic query policy, or model
architecture. Explicit projected-key and carrier APIs are outside this public
release surface.

## Execution and gradients

CPU execution uses the explicit PyTorch equation reference and does not import
Triton. CUDA packed calls use the fixed-tail Triton implementation. Ordered
source-list calls use the fixed-tail source adapter: bounded BF16 lists with
`D <= 2048` use the FLA-derived source-list route, while FP32 and wider BF16
lists use the fixed-tail fallback.

Source-list calls keep individual producer tensors when their layout is usable.
A non-affine source or incoming gradient may receive its own contiguous copy;
source lists do not promise universal zero-copy behavior.

The first-order backward differentiates values and the query. For an implicit
key, the routing gradient through the RMS-normalized value tail is combined
with the direct value gradient before the final storage-dtype boundary. Lists,
repeated sources, and shared views remain ordinary autograd edges. Second-order
behavior is not part of the public contract.

When several reads share a BF16 source or compose a BF16 intermediate, the
order in which those storage-dtype additions occur can change a few gradient
elements even when the real-valued equation is the same. That is a useful
diagnostic for a multi-read training graph, but it is not a relaxed correctness
rule: strict gates retain the mismatch and report the route as failing until
the underlying graph or kernel behavior is fixed.

The [`examples/backward.py`](../examples/backward.py) script checks value and
query gradients for both standard and sliced routes.

## Full and per-read Block schedules

Full and sequential Block schedules use the same public `attnres` call. Full
supplies all preceding residual sources at one read. Block changes only when
the reads happen and which completed or partial source tensors are supplied:

```python
completed = (embedding, first_block, second_block)
partial = current_partial
read_sources = completed if partial is None else completed + (partial,)
output = attnres(read_sources, query)
```

The caller owns block boundaries, partial sums, and learned queries. A partial
block is summed before the read and is passed as one ordinary source; it is not
averaged. The benchmark recipes use the per-read public operator directly and
do not expose a separate Block operator or prepared-state/cache API.

The [`examples/block_schedules.py`](../examples/block_schedules.py) script
checks a Full reduction against the same per-read schedule on small inputs.
