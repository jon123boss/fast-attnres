# Compiled-step evidence

This directory contains the immutable evidence behind the historical BF16 Full
training-step figure. These six reports do not feed the current README; the
current screen is under `results/adoption/compiled_step_screen`. H100 and B200
are separate replications, and the three seeds on each GPU are never pooled
into one confidence interval.

## Contents

- `raw/`: one atomic report per GPU and seed, including all 240 timing rows.
- `audits/`: CPU-only audit outputs after recomputing ABBA pairing, shared input
  identities, means, ratios, and 20,000-resample paired intervals.
- `attestations/`: report-byte-bound hardware and pinned-FLA identity records.
- `configs/`: the exact three seed configurations used by the final campaign.
- `campaign_manifest.json`: immutable measured-source revision, repository,
  frozen-file, production-kernel, and runner hash binding. Runtime, seed-config,
  device, and external-FLA identities are bound by each report and attestation.
- `hero_projection.json`: the six-report compact projection accepted by the
  Matplotlib renderer.
- `reproduction/run_exact_fair_campaign.py`: the exact remote wrapper used for
  these reports.

## Reports

| GPU | Seed | Raw report SHA-256 | Audit |
| --- | ---: | --- | --- |
| H100 SXM | 20260827 | `d8bbc9c0757e7e10ea18c9d47580b320e76e126731a04be10d3bac34863c1d21` | [`h100_seed_20260827.json`](audits/h100_seed_20260827.json) |
| H100 SXM | 20260903 | `532c9f4ad46a5dca2396c3c1c86a3c02f62278e8ce2e6b1824ecb19624c96140` | [`h100_seed_20260903.json`](audits/h100_seed_20260903.json) |
| H100 SXM | 20260911 | `73e40bb71e64140bcdbd1731581e1513ce25378dca5afd158e8ed0e0d5837552` | [`h100_seed_20260911.json`](audits/h100_seed_20260911.json) |
| B200 | 20260827 | `509295d898f1d1fbc0fa45ae0266025ab3391ea90805094e5829d1ac61dbf129` | [`b200_seed_20260827.json`](audits/b200_seed_20260827.json) |
| B200 | 20260903 | `ac51374ea002e3b5cc4a3dddab401b5cf066f070505679efdc0704937ac43954` | [`b200_seed_20260903.json`](audits/b200_seed_20260903.json) |
| B200 | 20260911 | `065a037bcb19877671baa7e52ca07ea7e9d73d20d50aee5cdf351a19e3d1c9b6` | [`b200_seed_20260911.json`](audits/b200_seed_20260911.json) |

The performance source is Git commit
`81dffbfeb0f84470513e846e3df8080e8ffb563d`, with production kernel hashes:

```text
fixed_tail.py         2333b3034e3c0e6493855b1246280ed91e65d29a962ce1d150beff71e8bbd34e
fixed_tail_sources.py 20fa0206fcbf6cc6b28a2973ac280575b6e8e378b09e0903449bf423d9812196
fla_full_sources.py   2cd7ac89b15faeb13640bff4a7948e437453b69446bfc8c7922511e341843e10
```

The external denominator is a clean FLA checkout at revision
`5e02dd3a7651f5f2797eb8b12bbec401826031e1`, package SHA-256
`2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781`.

## Verify one report

Use a clean checkout of the exact performance-source commit:

```bash
python -m benchmarks.audit_compiled_step \
  results/compiled_step/raw/b200_seed_20260827.json \
  --repo /path/to/clean/81dffbf-checkout \
  --gpu B200 --seed 20260827 \
  --campaign-manifest results/compiled_step/campaign_manifest.json \
  --release-attestation results/compiled_step/attestations/b200_seed_20260827.json \
  --require-release-attestation
```

The suite root remains `incomplete` because non-timed validation phases were
deliberately not executed, so
`release_promotable=false`. The audited `model_timings` sub-artifact is
complete. This directory supports only the scoped compiled Full training-step
claim documented in [`docs/compiled_step_results.md`](../../docs/compiled_step_results.md).
