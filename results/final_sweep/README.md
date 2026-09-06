# H100 and B200 final measurements

The [results table](results.md), [CSV](results.csv), and [audit](audit.json)
cover 14 admitted timing reports: six final74 headline reports, six final78
screen reports, and two final80 replacement reports. The two original final78
D2048/S9 failures remain unchanged in the archive and manifest.

The headline uses three seeds and 120 paired rounds per seed. Each smaller
workload uses one seed and 40 paired rounds. Values and queries are BF16;
the runtime is PyTorch 2.13.0+cu130 and Triton 3.7.1. Timing covers a complete
CUDA Graph training step with fused, capturable AdamW. Compilation,
qualification, input copy, and capture occur outside timing.

## Explicit normalization-rounding acceptance

The D2048/S9 completion accepts and records initial model-logit differences
from normalization order. These are **accepted discrepancies,
not strict output-tolerance passes**. Finite-value checks, loss, every parameter
gradient, optimizer/model state, and complete-step/replay checks remain strict
at `rtol=0.05`, `atol=0.05`. Each exception remains in its raw report and the
manifest's per-record `normalization_rounding` field:

| GPU | Arm | Mismatched / total logits | Maximum absolute difference |
| --- | --- | ---: | ---: |
| H100 | rank_2048 | 2 / 8388608 | 0.0625 |
| H100 | liger_rank_2048 | 2 / 8388608 | 0.05859375 |
| B200 | rank_2048 | 1 / 8388608 | 0.06640625 |
| B200 | fla_triton_compile_standard_rank_2048 | 7 / 8388608 | 0.07421875 |

Both original failed reports retain their original hashes and failure status;
they are not admitted timing reports. FLA Triton checkpoint 1, Liger 0.8.2,
and Catswe phase 1 remain native comparators. Unsupported/failed arms stay
visible. Standard-only FLA versus quarter-rank routing compares different equations.

## Exact source and supporting evidence

Headline: `48f3caf8165b7a8fab5fbdfed164a546f3418e85`. Original screens:
`fc0a2445d5ca24dd61dd5ac4a8179b0855a67c18`. D2048/S9 completion:
`67315dfcfa1f33049d7aba16a6f5e302acb17d59`. The production package remains `2a8838d1471c8eb45899c423f7bfae0c334c922d151a6c0de7f680e226900bb2`.
No normalization experiment replaces the measured production package.

The archive preserves final74/78/80 worker archives, all four source snapshots
(final71 for CPU compiler preparation), both 12-record headline supplements
with prior-attempt/script bindings, terminal compiler-helper records, and the
H100 frozen-gate and normalize-first diagnostic proofs. The unused final79
experiment is not a measurement source. All files have SHA-256 entries in the
[manifest](manifest.json). Compiler-cache evidence includes persisted autotuning
records and a content digest of the complete cache; regenerable binaries are omitted.

## Reproduce the audit and figures

From the repository root with the plot dependencies installed:

```bash
tar -xzf results/final_sweep/evidence.tar.gz
python -m benchmarks.final_sweep_report \
  --source evidence/sources/final78 \
  --h100 evidence/runs/H100/final78 --b200 evidence/runs/B200/final78 \
  --headline-source evidence/sources/final74 \
  --headline-h100 evidence/runs/H100/final74 \
  --headline-b200 evidence/runs/B200/final74 \
  --completion-source evidence/sources/final80 \
  --completion-h100 evidence/runs/H100/final80 \
  --completion-b200 evidence/runs/B200/final80 \
  --output reproduced/final_sweep
```

The auditor verifies source/report hashes, explicit acceptance scope, strict
remaining qualification, paired samples, and recomputed statistics. Historical
results retain their original identities.
