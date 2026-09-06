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
accumulation. Outputs and input gradients are BF16.

Full passes the embedding and previous writer outputs. Block passes the
embedding, completed block sums, and current partial sum. Both use the same
function and dispatch. Every read computes routing and the mixture from its
current inputs. Projected keys, priors, and architectural changes are outside
this campaign.

## Correctness

The independent reference in `validation/oracle.py` uses BF16 PyTorch
operations throughout normalization, scoring, softmax, mixing, and autograd.
It has no higher-precision reference mode.
Every output and source/query gradient must be finite and satisfy
`rtol=atol=0.05`. BF16 addition and casting are not assumed associative. The reference precision
was changed at the user's explicit request; historical reference results remain
historical and do not qualify the new evaluator.

Cover packed and source-list layouts, odd dimensions, strides, duplicate
sources and shared views, repeated reads, partial blocks, analytic gradients,
activation checkpointing, fullgraph compilation, eight changed-input CUDA
Graph replays, clipped gradients, and optimizer updates. Compare equal ranks.
Model checkpoint serialization and bitwise resumed training are outside this
stateless kernel's qualification scope.

## Performance

The current final sweep is defined in `configs/bf16_final_sweep.json`. It reuses
the existing headline and competitor-plot workloads on H100 and B200 with
`R=D` and `R=D/4`, as explicitly requested by the user. The existing
`benchmarks.run` and `training_graph` harnesses retain their complete CUDA Graph
step boundary, paired schedule, and optimizer. Compare the two candidate ranks
within each workload; comparisons of LR with standard-only competitors must
be labelled as different equations. Raw failures remain visible.

The earlier `bf16_primary*.json`, operator and broader matrices remain
historical contracts and do not expand this final refresh. The former 1B
fixture remains available for reproducing its recorded experiments.

## Resources and delivery

The Modal cap is US$600 following the additional $100 authorized on
2026-09-06: baseline/profiling $80, experiments $220, confirmation $240,
and infrastructure/retry reserve $60. The ledger records this increase;
earlier job reservations remain unchanged. Reserve
each job's full timeout and startup maximum before launch. Development uses
one GPU across both architectures. Final qualification and frozen matrices
may use up to eight independent single-GPU jobs. Distributed work is disabled.
Reservations remain charged to the cap after failures. Resume only missing
work and retain incremental evidence.

Training jobs back up completed compiler artifacts after each arm qualifies,
before timing resumes. Backups replace the previous archive atomically and
skip unchanged caches; backup failures remain in the report. A timeout during
an unfinished arm can still discard that arm's new compilations.

Deliver a clean BF16 package, shared model integration, reproducible commands,
raw results, confidence intervals, source identities, failure records, and a
draft GitHub PR. Do not merge or publish a release. Unverified fastest-kernel,
monotonicity, or production-qualification claims are not deliverables.
