# BF16 H100/B200 evaluation contract

## Operator and reference

`attnres(values, query, *, eps=2**-23, scale=1)` accepts CUDA BF16 values
`[S, ..., D]` or ordered sources `[..., D]`, and a BF16 query `[R]`.
Keys are the last `R` value coordinates; values and outputs retain width `D`.
Full and Block call the same operator. The caller supplies the embedding,
writer outputs or block sums, and an optional partial block sum.

The independent reference in `validation/oracle.py` normalizes each key before
its query dot product, then applies source softmax and the full-width mixture.
It uses BF16 inputs/outputs and FP32 internal accumulation with autocast disabled.
Outputs and first-order input gradients must be finite and satisfy
`rtol=0.05, atol=0.05`. BF16 addition and casting are not assumed associative.
The public operator has no alternative reference or precision mode.

## Correctness

Coverage includes packed/list sources, odd dimensions, strides, duplicate
sources, shared views, repeated reads, partial blocks, analytic gradients,
activation recomputation, fullgraph compilation, and changed-input CUDA Graph
replay. Training checks include gradients and optimizer updates. Model
checkpoint serialization and resumed training are outside this stateless
operator's scope. See [the validation protocol](docs/validation.md).

## Performance

The [final sweep](configs/bf16_final_sweep.json) defines five workloads on
H100 and B200, each at `R=D` and `R=D/4`. The pinned runtime is PyTorch
2.13.0+cu130 and Triton 3.7.1. `benchmarks.run` captures complete CUDA Graph
steps with fused AdamW, using paired inputs and rotating arm order. Compilation,
qualification, input copies, and graph capture remain outside timed events.

Compare the two candidate ranks within each workload. LR comparisons against
standard-only competitors use different routing equations and must be labelled.
Record exact sources, hardware, runtime, tolerances, raw paired samples,
confidence intervals, and unsupported/failed arms. Failed qualification does
not produce an eligible timing result. See [benchmark scope](docs/benchmark_results.md).

Earlier `bf16_primary*.json`, operator, and broader matrices remain historical
contracts. Their reports and immutable source archives retain their original
identities and do not qualify a changed kernel or evaluator. Claims remain
limited to the workloads, ranks, and eligible alternatives actually measured.
