# Current 24-layer Full AttnRes results

This is the current headline performance campaign for Fast-AttnRes. It uses
the release kernel bytes and compares the standard `R=D=1024` equation with a
clean, pinned native FLA Triton checkpoint-1 checkout. H100 and B200 were run
as separate Modal jobs and the three protocol seeds remain separate.

## Workload

| Field | Value |
| --- | --- |
| Schedule | Full ordered source list; 48 reads with `S=2..49` |
| Model | L24 / D1024 / H16 / FFN2816 / vocab32768 |
| Batch and sequence | B2 / T1024 (`N=2048` flattened rows per source) |
| Equation | Standard AttnRes, `R=D=1024` |
| Storage | BF16 autocast |
| Timed step | zero gradients, forward, cross-entropy, backward, accumulation=1, fused capturable AdamW |
| Timing | one captured CUDA Graph replay, measured with CUDA events |
| Runtime | PyTorch 2.13.0+cu130 / CUDA 13.0 / Triton 3.7.1 |
| Devices | H100 SXM SM90 and B200 SM100 |
| Per seed | 10 warmups per arm; 120 paired ABBA rounds |
| Statistics | mean of paired Fast-AttnRes/FLA ratios; 20,000-resample simultaneous 95% bootstrap |
| Seeds | 20260827, 20260903, 20260911; never pooled |

## Results

A ratio below 1 means Fast-AttnRes has lower step latency. The per-seed
advantage is `(1-ratio) x 100%`. Absolute arm means are descriptive; inference
uses the paired ratio and its per-seed interval.

| GPU | Seed | Fast-AttnRes ms/step | FLA ms/step | Fast-AttnRes / FLA [95% CI] | Advantage | Pairs | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H100 SXM | 20260827 | 28.4012 | 29.9716 | 0.947604 [0.947420, 0.947788] | 5.2396% | 120 | passed |
| H100 SXM | 20260903 | 28.4223 | 29.9794 | 0.948062 [0.947871, 0.948253] | 5.1938% | 120 | passed |
| H100 SXM | 20260911 | 28.4147 | 29.9719 | 0.948043 [0.947863, 0.948223] | 5.1957% | 120 | passed |
| B200 | 20260827 | 16.5283 | 19.5360 | 0.846043 [0.845939, 0.846148] | 15.3957% | 120 | passed |
| B200 | 20260903 | 16.5301 | 19.5668 | 0.844802 [0.844677, 0.844928] | 15.5198% | 120 | passed |
| B200 | 20260911 | 16.5248 | 19.5612 | 0.844772 [0.844628, 0.844916] | 15.5228% | 120 | passed |

| GPU | Descriptive summary across three unpooled seeds |
| --- | --- |
| H100 SXM | Median per-seed advantage **5.20%**; descriptive mean arm latencies 28.413 vs 29.974 ms/step |
| B200 | Median per-seed advantage **15.52%**; descriptive mean arm latencies 16.528 vs 19.555 ms/step |

The summaries do not create a pooled confidence interval. Every one of the six
separately evaluated per-seed intervals is below parity.

## Exact timing boundary

The start event is recorded immediately before one `CUDAGraph.replay()` and
the end event immediately after it. The captured graph contains optimizer
zeroing, BF16-autocast model forward, cross-entropy, backward, gradient
accumulation, and the fused capturable AdamW update.

Input creation and logical input identity, graph input copies, compilation,
optimizer construction, independent qualification, warmup, graph capture,
source or tensor hashing, report serialization, and all CPU work are outside
the timed events. This is captured complete-training-step device time, not an
isolated AttnRes kernel timer and not host wall-clock latency.

## Qualification and provenance

Every arm passed before timing:

- model output, loss, every named parameter gradient, model state, optimizer
  state, and parameter-update checks;
- fullgraph static compilation;
- changed-input CUDA Graph replay checks with stable graph counters;
- exact 120-pair ABBA schedules with the same logical input identity in both
  arms;
- finite positive timing samples and independent statistical recomputation.

The measured performance source is commit
`b8837e1d74eb708a39a455840332247725a26496`, tree
`6a807f2f739c45f8ec9051e83df6d7ab4df560ba`. The release keeps the exact
measured kernel bytes:

| Kernel | SHA-256 |
| --- | --- |
| `fixed_tail.py` | `2333b3034e3c0e6493855b1246280ed91e65d29a962ce1d150beff71e8bbd34e` |
| `fixed_tail_sources.py` | `1373614c93d7291ad96697b1b8ff627120590b75f63f7e38bd65d50b19fcfb4a` |
| `fla_full_sources.py` | `8749c72c4714145214e33e8bc7d37f57b47a79b67f2e83044205db72cda416fa` |

FLA is pinned to revision
`5e02dd3a7651f5f2797eb8b12bbec401826031e1`, tree
`7e4199902fb291c78b3937f223b08ae7bca82bb1`, package SHA-256
`2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781`,
from `https://github.com/fla-org/flash-linear-attention.git`.

The exact reports, their SHA-256s, the Modal call IDs, transport source, compact
projection, and CPU-only audit command are in
[`results/current_24l`](../results/current_24l/README.md).

## Scope

- This is one large Full standard-AttnRes model at `D=R=1024`.
- It is not a rank, width, sequence, or depth sweep.
- The smaller adoption screen provides broader Full, Block, LR, Liger, and
  Catswe coverage; it is not pooled with this campaign.
- H100 and B200 are separate replications, not a cross-device ranking.
- The archived earlier campaign remains valid only for its named historical
  source bytes and is not used by the current headline.
