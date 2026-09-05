# BF16 H100/B200 evaluation contract

The root owns the oracle, tests, timing harness, source manifests, budget ledger,
and final selection. Independent candidate authors may change only their
assigned isolated implementation. Historical release reports retain their own
source archives and do not qualify the current package.

## Operator

`attnres(values, query, *, eps=2**-23, scale=1)` accepts CUDA BF16 values
`[S, ..., D]` or ordered sources `[..., D]`, and a BF16 query `[R]`.
Keys are the last R value coordinates. RMS normalization, query dot product,
source softmax, and the full-width value mixture use stable internal FP32
accumulation. Outputs and input gradients are BF16. There is no shipped CPU,
FP32, or reference execution product.

Full passes the embedding and previous writer outputs. Block passes the
embedding, completed block sums, and current partial sum. Both use the same
function and dispatch. Prepared Block state, phase caches, cross-read reuse,
projected keys, priors, and architectural changes are outside this campaign.

## Correctness

The BF16 test oracle in `validation/oracle.py` is the independent comparison.
Every output and source/query gradient must be finite and satisfy
`rtol=atol=0.05`. Direct and routing derivatives combine before the BF16 input
boundary; BF16 addition and casting are not assumed associative. The user's
BF16 nonlinearity clarification does not change the tolerance.

Cover packed and source-list layouts, odd dimensions, strides, duplicate
sources and shared views, repeated reads, partial blocks, analytic gradients,
activation checkpointing, fullgraph compilation, eight changed-input CUDA
Graph replays, optimizer updates, exact checkpoint restoration, and BF16
continuation after resume. Compare equal ranks.

## Performance

`configs/bf16_primary.json` defines the current primary model, rank ladder,
seeds, runtime, competitor inventory, and immutable source identity contract.
The model has 24 layers, width 1536, 24 heads, MLP width 4224, vocabulary
100277, context 2048, batch four, accumulation four, and eight blocks. It uses
ordinary source assembly, BF16 cross-entropy, gradient clipping at 1.0, and the
original Muon plus AdamW implementation. Activation checkpointing is qualified
separately and is disabled in the primary model.

Measure three seeds and 120 balanced paired rounds after ten warmups. Include
input copies, source preparation, forward, loss, backward, accumulation,
zeroing gradients, and optimizer work. Record compilation/warmup, operator
latency, complete-step latency, and memory separately. This controlled
synthetic-data fixture excludes dataset I/O, logging, and scheduler host work;
it does not reproduce historical training throughput.

Compare each cell with its fastest correct eligible alternative. Preserve all
failures and incomplete measurements. Use simultaneous 95% confidence
intervals and require adjacent lower/higher-rank latency ratio upper bounds
at most 1.005 for primary coverage, or 1.01 for broader coverage. Never slow
higher ranks, relax correctness tolerances, omit regressions, or select the
fastest retry. Missing or inconclusive coverage is an unmet target.

## Resources and delivery

The Modal cap is US$500: baseline/profiling $80, experiments $220,
confirmation/distributed $140, and infrastructure/retry reserve $60. Reserve
each job's full timeout and startup maximum before launch. Run at most one
single-GPU job per architecture; eight-GPU qualification runs exclusively.
Reservations remain charged to the cap after failures. Resume only missing
work and retain incremental evidence.

Deliver a clean BF16 package, shared model integration, reproducible commands,
raw results, confidence intervals, source identities, failure records, and a
draft GitHub PR. Do not merge or publish a release. Unverified fastest-kernel,
monotonicity, or production-qualification claims are not deliverables.
