# Provenance

This file records the implementation, benchmark, and third-party identities
behind the current Fast-AttnRes release. It is an integrity record; the scoped
performance claim and its limitations live in the current
[`compiled-step adoption evidence`](results/adoption/README.md).

## Repository lineage

| Field | Value |
| --- | --- |
| Private adoption repository | `https://github.com/jon123boss/fast-attnres` |
| Initial private seed | `76669dde5b2b34ac73772f456cba978c264a9ac5` |
| Current adoption-screen source | `79a5ad623fd223f93dc00933b9885831977712d3` |
| Current adoption-screen tree | `37786d60fdb27f2c9071db61b608224105d56b4b` |
| Historical three-seed source | `81dffbfeb0f84470513e846e3df8080e8ffb563d` |
| Historical three-seed tree | `1cceb5e0a37330015ca2945312da29aa7566aaeb` |
| Package | `fast-attnres` 1.0.0 |
| Project license | MIT |

The compiled-step reports hash their exact performance-source checkout. Later
documentation, audit, and plotting commits do not alter or relabel those
measured source bytes.

## Production kernel identity

| Role | Path | SHA-256 |
| --- | --- | --- |
| Packed kernel | `src/attnres/_kernels/fixed_tail.py` | `2333b3034e3c0e6493855b1246280ed91e65d29a962ce1d150beff71e8bbd34e` |
| Source-list fallback | `src/attnres/_kernels/fixed_tail_sources.py` | `1373614c93d7291ad96697b1b8ff627120590b75f63f7e38bd65d50b19fcfb4a` |
| Bounded BF16 source-list route | `src/attnres/_kernels/fla_full_sources.py` | `8749c72c4714145214e33e8bc7d37f57b47a79b67f2e83044205db72cda416fa` |

The source-list hashes above identify the current standard-source autotuned
implementation introduced at `8ddb0bbaf184663703ded65b45839fddd1c429fc`
(tree `a91fb6d7662c36652bf648aa2e8170c90887bc1a`). They are the bytes
sealed by `validation/frozen.json` and the active campaign recipes.

The FLA-derived source-list route retains its upstream attribution and MIT
notice in [`NOTICE`](NOTICE) and in the source header. The public API, packed
kernel, and fallback kernel are project code under the top-level MIT license.

## Historical compiled-step campaign

The preserved campaign contains exactly six BF16 Full reports: three seeds on
H100 SXM and three on B200. Every report records 120 matched AttnRes/FLA pairs
under PyTorch 2.13.0+cu130, CUDA 13.0, and Triton 3.7.1. Seeds and devices are
kept separate.

Those reports predate the current autotuned source-list kernels. Their measured
candidate hashes are intentionally retained in the result manifest:

| Historical measured role | SHA-256 |
| --- | --- |
| `fixed_tail.py` | `2333b3034e3c0e6493855b1246280ed91e65d29a962ce1d150beff71e8bbd34e` |
| `fixed_tail_sources.py` | `20fa0206fcbf6cc6b28a2973ac280575b6e8e378b09e0903449bf423d9812196` |
| `fla_full_sources.py` | `2cd7ac89b15faeb13640bff4a7948e437453b69446bfc8c7922511e341843e10` |

They remain reproducibility evidence for that exact historical implementation;
they are not relabelled as measurements of the current kernel. New performance
claims require reports whose runtime preflight records the current hashes.

The campaign's immutable bindings are stored in
[`results/compiled_step/campaign_manifest.json`](https://github.com/jon123boss/fast-attnres/blob/main/results/compiled_step/campaign_manifest.json).
That directory also contains the raw rows, per-report attestations, independent
audit outputs, exact seed configs, reproduction wrapper, compact hero
projection, and deterministic Matplotlib inputs.

The measured model is L24/D1024/H16/FFN2816/B2/T1024/V32768. Both arms execute
the same standard AttnRes equation (`R=D=1024`), ordered Full source schedule,
loss, backward, and fused capturable AdamW update. The event interval contains
one complete CUDA Graph replay. Compilation, capture, warmup, qualification,
hashing, input copies, and CPU work are outside the event interval.

## External FLA denominator

| Field | Value |
| --- | --- |
| Repository | `https://github.com/fla-org/flash-linear-attention.git` |
| Revision | `5e02dd3a7651f5f2797eb8b12bbec401826031e1` |
| Package SHA-256 | `2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781` |
| Package file count | 506 |
| Route | Native Triton AttnRes checkpoint 1 |
| Checkout state | Clean |

FLA receives one preallocated, nonpersistent FP32 unit RMS-weight buffer owned
by its model wrapper. Generated-code inspection found no RMS allocation or
unit-fill launch inside the measured graph. The FLA buffer is not optimized,
serialized, or recreated per read.

## Other external implementations

FLA Gluon, Liger-Kernel, Catswe phase 1, Hydra 2P, and FLA checkpoint 0 are
represented by optional adapters and a fail-closed capability registry. They
do not have an accepted complete Full-model result in the current campaign,
so they never enter its denominator. Their constraints and exclusion reasons
are listed in the README and
[`docs/matched_competitor_protocol.md`](https://github.com/jon123boss/fast-attnres/blob/main/docs/matched_competitor_protocol.md).

No external comparator kernel source is redistributed in this repository.
Optional adapters import separately installed, pinned checkouts. Their
upstream license and notice files remain authoritative; [`NOTICE`](NOTICE)
records source URLs and license identities at the project boundary.

## Current compiled-step adoption screen

The README chart and table are derived only from the eight raw reports under
[`results/adoption/compiled_step_screen`](results/adoption/compiled_step_screen/).
Their manifest binds every report, the plotting code, PNG/SVG, CSV, and
Markdown table. The screen uses one predeclared seed, 5 warmups, 40 paired
rounds, and a 20,000-resample simultaneous paired bootstrap per cell on H100
SXM and B200 with PyTorch 2.13.0+cu130 and Triton 3.7.1. It covers standard
Full, standard per-read Block, and explicitly labeled sliced LR architectural
cells. Numeric, failed, and not-applicable arms remain separate; GPUs and cells
are never pooled into a global ranking.

The measured kernel hashes equal the current production identities above. The
historical `validation/frozen.json` digest recorded at measurement time is
`9286b3b5b7cbe3a8fb7c062ce5a795b2b1fe3c0d03dc2cf2b6848483b9ed1a31`.

## Scholarly references

- Kimi Team, *Attention Residuals*, arXiv:2603.15031,
  <https://arxiv.org/abs/2603.15031>.
- Jonathan Su, *Low-Rank Attention Residuals*, arXiv:2607.09694,
  <https://arxiv.org/abs/2607.09694>.
