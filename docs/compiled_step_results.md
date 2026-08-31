# Audited BF16 Full compiled-step results

> **Historical implementation.** These reports measured the pre-autotune
> source-list hashes recorded in `results/compiled_step/campaign_manifest.json`.
> They remain valid for those exact bytes, but do not measure the current
> autotuned production kernel. Current-kernel performance must be established
> by a fresh, separately sealed campaign.

This page documents the historical six-report large-model campaign. Its
separately labeled summary is the top README hero because it is the strongest
audited Full-workload evidence; it is never pooled with the current adoption
screen or described as a measurement of the current autotuned source bytes. It
compares the audited Fast-AttnRes route with pinned native FLA Triton
checkpoint 1 on one Full `R=D=1024` same-equation AttnRes workload. Current
broader-screen evidence is documented in
[`results/adoption`](../results/adoption/README.md).

## Workload

| Field | Value |
| --- | --- |
| Equation | Same-equation standard AttnRes, implicit keys, `R=D=1024` |
| Schedule | Full ordered source-list schedule; 48 residual reads with `S=2…49` across the L24 model |
| Model | L24 / D1024 / H16 / FFN2816 / vocab32768 |
| Batch and sequence | B2 / T1024 (`N=2048` flattened token rows) |
| Storage | BF16 autocast training |
| Step | zero gradients, forward, cross-entropy, backward, accumulation=1, fused capturable AdamW (`lr=3e-4`, `betas=(0.9,0.95)`, `weight_decay=0.1`) |
| Runtime | PyTorch 2.13.0+cu130 / CUDA 13.0 / Triton 3.7.1 |
| Devices | H100 SXM SM90 and B200 SM100, kept separate |
| Per seed | 10 warmups and 120 paired ABBA measured rounds |
| Statistics | mean of paired candidate/FLA ratios; common-index 20,000-resample simultaneous 95% bootstrap |
| Seeds | 20260827, 20260903, 20260911; never pooled |

## Results

A ratio below 1 means lower AttnRes latency. “Advantage” is `(1-ratio)×100%`. Absolute arm means are descriptive; inference uses the paired ratio.

| GPU | Seed | AttnRes mean ms/step | FLA mean ms/step | AttnRes / FLA [95% CI] | Per-seed advantage [95% CI] | Rows | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H100 SXM | 20260827 | 28.6996 | 29.9294 | 0.958913 [0.958730, 0.959096] | 4.11% [4.09%, 4.13%] | 120 pairs / 240 rows | audited |
| H100 SXM | 20260903 | 28.7316 | 29.9625 | 0.958921 [0.958725, 0.959116] | 4.11% [4.09%, 4.13%] | 120 pairs / 240 rows | audited |
| H100 SXM | 20260911 | 28.0623 | 29.2748 | 0.958583 [0.958408, 0.958759] | 4.14% [4.12%, 4.16%] | 120 pairs / 240 rows | audited |
| B200 | 20260827 | 17.2723 | 19.5251 | 0.884620 [0.884510, 0.884730] | 11.54% [11.53%, 11.55%] | 120 pairs / 240 rows | audited |
| B200 | 20260903 | 17.2231 | 19.5055 | 0.882988 [0.882847, 0.883129] | 11.70% [11.69%, 11.72%] | 120 pairs / 240 rows | audited |
| B200 | 20260911 | 17.2280 | 19.5188 | 0.882634 [0.882470, 0.882799] | 11.74% [11.72%, 11.75%] | 120 pairs / 240 rows | audited |

| GPU | Descriptive summary across three unpooled seeds |
| --- | --- |
| H100 SXM | Median advantage **4.11% faster**; seed range 4.11–4.14% faster |
| B200 | Median advantage **11.70% faster**; seed range 11.54–11.74% faster |

The descriptive medians above do not create a pooled confidence interval. Each seed’s paired interval remains the inferential result.

## Exact timed boundary

The start CUDA event is recorded immediately before one `CUDAGraph.replay()` and the end event immediately after it. The captured replay includes:

1. `zero_grad(set_to_none=False)`;
2. BF16-autocast model forward;
3. cross-entropy loss;
4. backward and gradient accumulation;
5. fused capturable AdamW update.

Input generation and logical input hashing, graph input copies, `torch.compile`, optimizer construction, warmup, independent qualification, graph capture, report serialization, and all CPU work are outside the event interval. The reported metric is therefore **captured complete-step device time**, not host-observed end-to-end wall-clock latency.

## Matched arms and fairness

Both arms receive the same initialized model state, tokens, targets, loss, optimizer settings, source order, and static Full schedule. They implement the same `R=D=1024` standard AttnRes equation. The only intended difference is the AttnRes execution route.

The candidate selects the sliced rank-1024 code path at `R=D`; at this endpoint it is the standard full-width equation. The FLA denominator uses its separate native standard `R=D` path.

The FLA model owns one preallocated, nonpersistent FP32 unit RMS-weight buffer. It follows model device moves, is excluded from optimizer/state serialization, and is reused by every FLA read. The final reports require `fla_fill_launches_inside_step=0`; the pinned vendor checkout remains separate from the bounded FLA-derived BF16 source-list candidate.

FLA is a separately installed clean checkout pinned to revision `5e02dd3a7651f5f2797eb8b12bbec401826031e1` and package SHA-256 `2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781`.

## Qualification before timing

Each arm must pass before its timing rows exist:

- independent model-output, loss, every parameter-gradient, and optimizer-state comparison;
- complete compiled-step state update checks;
- fullgraph compile with static shapes;
- changed-input CUDA-Graph replay parity;
- state restoration and stable data pointers;
- zero timed graph breaks, recompiles, or new graphs;
- 120 finite timing rows with the exact deterministic ABBA schedule and one shared logical input identity per pair.

A failed, incomplete, missing, or unsupported arm has no ratio and cannot enter the denominator.

## Offline audit

Every raw report binds the exact performance-source Git revision, three production-kernel hashes, runner, FLA bridge, model, frozen manifest, runtime, hardware selector, and external FLA identity. The CPU-only auditor hashes a clean checkout, reconstructs every sample pair and ABBA order, verifies shared input identities, and recomputes the paired estimates and intervals.

```bash
python -m benchmarks.audit_compiled_step \
  results/compiled_step/raw/b200_seed_20260827.json \
  --repo /path/to/clean/performance-source-checkout \
  --gpu B200 --seed 20260827 \
  --campaign-manifest results/compiled_step/campaign_manifest.json \
  --release-attestation results/compiled_step/attestations/b200_seed_20260827.json \
  --require-release-attestation
```

The raw suite intentionally leaves non-timed validation phases `not_run`; its root status is therefore `incomplete`, the auditor reports `release_promotable=false`, and the compiled model sub-artifact alone is independently `complete` and `timing_verified`. Claims on this page are scoped only to that named timed sub-artifact.

## Reproduce and render

The three seed configs and campaign manifest are in [`results/compiled_step`](https://github.com/jon123boss/fast-attnres/tree/main/results/compiled_step). The final campaign runner performs exact runtime/source/device preflight and writes each report atomically. After all six reports audit, generate the compact projection and Matplotlib figure:

```bash
python - <<'PY'
import json
from pathlib import Path

from benchmarks.audit_compiled_step import build_hero_projection

projection = build_hero_projection(
    {
        "H100": {
            20260827: "results/compiled_step/raw/h100_seed_20260827.json",
            20260903: "results/compiled_step/raw/h100_seed_20260903.json",
            20260911: "results/compiled_step/raw/h100_seed_20260911.json",
        },
        "B200": {
            20260827: "results/compiled_step/raw/b200_seed_20260827.json",
            20260903: "results/compiled_step/raw/b200_seed_20260903.json",
            20260911: "results/compiled_step/raw/b200_seed_20260911.json",
        },
    },
    repo_root="/path/to/clean/performance-source-checkout",
    campaign_manifest="results/compiled_step/campaign_manifest.json",
    release_attestation_paths={
        gpu: {
            seed: f"results/compiled_step/attestations/{gpu.lower()}_seed_{seed}.json"
            for seed in (20260827, 20260903, 20260911)
        }
        for gpu in ("H100", "B200")
    },
)
Path("results/compiled_step/hero_projection.json").write_text(
    json.dumps(projection, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

python -m benchmarks.plot_compiled_step_hero \
  --projection results/compiled_step/hero_projection.json \
  --output-dir docs/assets \
  --svg-name compiled_step_hero.svg \
  --png-name compiled_step_hero.png
```

The Matplotlib renderer reads only the compact audited projection. Raw-report parsing, provenance checks, schedule validation, and statistical recomputation belong to the auditor.

## Limits

- This is one Full standard-AttnRes model at `D=R=1024`; it is not a rank, width, sequence, depth, or model-size sweep.
- It is not a Full-versus-Block or packed-versus-list comparison.
- It does not compare LR-AttnRes (`R<D`) with FLA; the pinned external routes require `R=D`.
- CUDA Graph input-copy and host launch time are excluded.
- H100 and B200 results are separate replications, not a cross-device ranking.
- FLA Gluon, Liger, Catswe, Hydra, and FLA checkpoint 0 do not have an accepted complete Full-model result in this campaign; their exact stipulations are listed in the README.
