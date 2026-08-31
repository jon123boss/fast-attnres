# BF16 compiled-step adoption evidence

This directory is the publication boundary for the current BF16 adoption
screen. Only independently complete comparator arms that pass the final audit
can contribute a number to the README figure or table. A report may still
retain a different failed or not-applicable arm without suppressing its
independently qualified arms.

## Published screen

The current screen uses one seed (`20260827`), 5 warmups, 40 paired rounds per
arm, and a 20,000-resample simultaneous paired bootstrap. Ratios are
`Fast-AttnRes / comparator`; lower is faster. The intervals are simultaneous
within each cell's comparator family. GPUs and shapes are never pooled. The table is deliberately long-form: append
one row per fully audited GPU x cell x comparator, and never create a numeric
placeholder for an unfinished arm.

The rendered [SVG](../../docs/assets/compiled_step_screen.svg) and
[PNG](../../docs/assets/compiled_step_screen.png) are derived from eight raw
worker reports. The [long-form Markdown table](compiled_step_screen/results.md)
and [CSV](compiled_step_screen/results.csv) retain every numeric, failed, and
not-applicable comparator arm. The [manifest](compiled_step_screen/manifest.json)
binds report and derived-artifact hashes.

Headline same-equation results range from **15.64% faster than Liger** and
**12.91% faster than Catswe** in the B200 Full `D=1024` cell to **1.32% slower
than FLA** in the B200 Block `D=1536, Smax=3` cell. The B200 `D=2048` FLA row
is parity because its simultaneous interval crosses 1; its Liger arm remains
an explicit strict-numerical-gate failure. LR-AttnRes `D=1536, R=384` is
**0.26% faster on H100** and **0.66% faster on B200** than standard `R=D` FLA,
and is labeled as a different-equation architectural comparison.

All rows are 8-layer BF16-autocast training with `N=1024` flattened rows per
source (`batch 2 x sequence 512`) on PyTorch 2.13.0+cu130 / Triton 3.7.1. The
timed CUDA Graph contains optimizer zeroing, model forward, cross-entropy,
backward, gradient accumulation, and fused AdamW. Input copy, compilation,
qualification, warmup, graph capture, hashing, and JSON work remain outside
the event interval.

## Matrix vocabulary

- `N` is the flattened token-row count per source: `B*T`.
- `D` is the full residual value and output width.
- `R` is the implicit-key/query rank. Standard AttnRes uses `R=D`; sliced
  LR-AttnRes uses `R<D`.
- `S` is the source count visible to one read. Full grows through the complete
  history. Block controls the maximum with the event block size; for the
  displayed 8-layer Block cell, event block size 2 yields `S=2..9`.

## Competitor envelope

| Comparator | Admission rule |
| --- | --- |
| FLA Triton checkpoint 1 | Native source list, standard `R=D`, checkpoint 1. |
| Liger 0.8.2 | `R=D`, `S<=32`; stack and contiguous staging are timed. |
| Catswe phase 1 | BF16, `R=D`, power-of-two `D`, `nextpow2(S)*D<=2^20`; staging is timed. Phase 2, prepare/merge, and cached Block are forbidden. |
| Hydra 2P | Timing requires `D<=256`, so all `D>768` screen cells are explicit `not_applicable`. |
| Sliced LR-AttnRes vs FLA | Different routing equation: candidate `R<D` versus standard FLA `R=D`; report as architectural, never same-equation. |

## Publication gate

A row is added to the table and `docs/assets/compiled_step_screen.svg` only
when all of the following are true:

1. the worker report and route provenance bind the intended GPU, runtime,
   kernel hashes, vendor revision, and exact cell;
2. the candidate and comparator pass independent output, every-source value
   gradient, and query-gradient qualification;
3. fullgraph compilation and changed-input CUDA Graph replay pass with no new
   graph, graph break, or recompile during timing;
4. the raw paired schedule is complete, finite, same-input, and outside-event
   hashing is verified;
5. the ratio and simultaneous interval are recomputed from the raw 40-round
   arms by the final auditor.

`not_applicable`, failed, missing, incomplete, and pending rows stay visible in
the retained machine-readable evidence but receive no plotted ratio. This
one-seed screen is an adoption signal, not a release-wide ranking or a pooled
hardware claim.
