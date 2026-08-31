# Resident complete-training rank ladder

This directory contains recipes for the active autotuned source-list candidate
at `8ddb0bbaf184663703ded65b45839fddd1c429fc` (tree
`a91fb6d7662c36652bf648aa2e8170c90887bc1a`). The candidate source files are
recorded in each recipe by SHA256:

* `fixed_tail_sources.py`: `1373614c93d7291ad96697b1b8ff627120590b75f63f7e38bd65d50b19fcfb4a`
* `fla_full_sources.py`: `8749c72c4714145214e33e8bc7d37f57b47a79b67f2e83044205db72cda416fa`

The earlier scalar-compact gate at `a927c8d9c3c802637a4d6cb2247378bfd6cee3bb`
and its `20fa0206…` / `2cd7ac89…` hashes are historical. They remain recorded
in the split manifest and historical reports, but are not the source identity
of a new ladder run.

## Pinned native FLA checkout

The native FLA compile anchor is external to this repository. Every ladder
invocation must set `ATTNRES_FLA_DIR` to a clean flash-linear-attention
checkout containing `fla/`. The required checkout is pinned to revision
`5e02dd3a7651f5f2797eb8b12bbec401826031e1` and its Python package tree must
have SHA-256
`2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781`.
The package digest is computed over sorted `*.py` files below `fla/`, hashing
each relative path followed by its file bytes, matching the Modal transport.
The adapted in-repository kernel hash above is separate from this vendor hash.

Provision the checkout once, using a path outside this repository:

```bash
FLA_DIR=/absolute/path/to/flash-linear-attention
FLA_REVISION=5e02dd3a7651f5f2797eb8b12bbec401826031e1
git clone https://github.com/fla-org/flash-linear-attention.git "$FLA_DIR"
git -C "$FLA_DIR" fetch --depth=1 origin "$FLA_REVISION"
git -C "$FLA_DIR" checkout --detach "$FLA_REVISION"
```

Before each run, verify the revision and package digest, then export the
transport variables required by `benchmarks/modal_runner.py`:

```bash
FLA_DIR=/absolute/path/to/flash-linear-attention
FLA_REVISION=5e02dd3a7651f5f2797eb8b12bbec401826031e1
FLA_PACKAGE_SHA256=2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781
test "$(git -C "$FLA_DIR" rev-parse HEAD)" = "$FLA_REVISION"
test -z "$(git -C "$FLA_DIR" status --porcelain)"
test -d "$FLA_DIR/fla/ops/attnres"
python - "$FLA_DIR" "$FLA_PACKAGE_SHA256" <<'PY'
import hashlib
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve() / "fla"
digest = hashlib.sha256()
for path in sorted(root.rglob("*.py")):
    digest.update(str(path.relative_to(root)).encode())
    digest.update(path.read_bytes())
actual = digest.hexdigest()
if actual != sys.argv[2]:
    raise SystemExit(f"unexpected FLA package hash: {actual}")
PY
export ATTNRES_FLA_DIR="$FLA_DIR"
export ATTNRES_FLA_REVISION="$FLA_REVISION"
export ATTNRES_FLA_DIRTY=0
```

The recipes call `benchmarks.run.run_suite`; they do not change the evaluator,
the reference, or the frozen validation contract. They are intended for the
primary model on H100! and B200. No GPU run is part of this change.

## Fixed recipe

[`production_ladder_full.json`](../configs/production_ladder_full.json) and
[`production_ladder_block.json`](../configs/production_ladder_block.json) use
the same primary geometry:

```text
L24 / D1024 / H16 / FFN2816 / B2 / T2048 / V32768 / blocks8
```

Both recipes select the explicit sliced rank ladder
`[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]`, ordered source-list
inputs, BF16 autocast, 10 warmup rounds, and 120 timed rounds. The timing
method is `cuda_graph`, and the captured step includes zeroing gradients,
model forward, compiled cross-entropy, backward, gradient accumulation, and
the fused capturable AdamW update.

`model_state_protocol` is
`canonical_implicit_max_rank_v1`. Each rank is constructed from the same
canonical standard `R=D=1024` state; a sliced query receives the corresponding
value-tail query coordinates. Timed inputs are changed per sample and are
shared by every arm in one job. The evaluator's balanced forward/reverse
schedule keeps the arms paired.

`reference_timing: false` removes reference from timed arms only. The current
evaluator still constructs an independent reference and qualifies every
kernel rank before compilation and timing. Thus a complete report must retain
qualification for every selected rank.

The Full recipe uses the complete public Full schedule. The Block recipe uses
the same public `attnres` primitive for every residual read; its difference is
the per-read Block schedule. There is no cached Block execution and no
`block_execution` or `include_per_read` setting. `source_layout` is `list`,
`include_packed_comparison` is false, and no packed arm is scheduled. There is
also no projected candidate.

The only release FLA arm is the native Triton compile bridge at checkpoint
level 1. FLA Triton checkpoint-0 and Gluon diagnostics are outside this
release scope and must not be substituted for the anchor.
With `standard_fla_comparison: true`, the evaluator creates one separately
qualified standard FLA `R=D=1024` anchor for each job. It is an architecture
anchor for the sliced ranks, not an additional rank-ladder edge. Optional FLA
model discovery and extra backends are disabled.

## Runtime-safe split matrix

The 12-rank recipes are the current complete-ladder definition. Because a
12-rank complete-training job also qualifies, compiles, captures, and warms
the FLA anchor, the runtime-safe launch matrix is in
[`production_ladder_split_manifest.json`](../configs/production_ladder_split_manifest.json).
It derives each effective job by loading the appropriate mode recipe and
replacing **only** its `ranks` field:

### Priority first gate

Run one shared-input job for each mode and hardware with the bounded key-rank
set `[16, 64, 128, 512, 1024]`. The adjacent pairs in this first-gate order
are `16→64`, `64→128`, `128→512`, and `512→1024`. Both Full and Block use the
same five ranks and the same per-job shared-input rule, with the native FLA
anchor added by the base recipe. These pairs are direct paired evidence for
the first gate; they are not relabeled as adjacent edges of the full requested
ladder, whose authoritative edge jobs are listed below.

| Split | Kernel ranks | Edges evaluated in that job |
| --- | --- | --- |
| `s01` | 1, 2, 4 | 1→2; 2→4 |
| `s02` | 4, 8, 16 | 4→8; 8→16 |
| `s03` | 16, 32, 64 | 16→32; 32→64 |
| `s04` | 64, 128, 256 | 64→128; 128→256 |
| `s05` | 256, 512, 768 | 256→512; 512→768 |
| `s06` | 768, 1024 | 768→1024 |

Every requested adjacent edge is present in exactly one split. In particular,
`256→512` is adjacent in this requested ladder even though the broader frozen
protocol also contains `R=384`; split jobs therefore keep `pairwise: false`
and use the evaluator's normal balanced scheduler. Each job has at most three
kernel ranks plus its one FLA anchor. The priority first gate is the exception:
it intentionally uses five kernel ranks; ladder split jobs use at most three.
Together they give 28 jobs total:

```text
2 modes × 2 hardware targets × (1 priority gate + 6 split groups) = 28 jobs
```

Run the priority Full and Block jobs first on `H100!`, then the six H100 ladder
split groups in `s01` through `s06`. Repeat that sequence on B200. The hardware
arms are independent replications; do not pair their observations. A launch
command can pass the resulting effective JSON to the existing Modal runner,
for example:

```bash
ATTNRES_FLA_DIR="$FLA_DIR" ATTNRES_FLA_REVISION="$FLA_REVISION" \
ATTNRES_FLA_DIRTY=0 modal run benchmarks/modal_runner.py \
  --gpu H100! --task suite --config '<effective split JSON>'
```

The launcher GPU selection is deliberately outside the evaluator JSON. The
manifest is the source of the job order and rank replacement; do not alter
timing, optimizer, state, source-layout, FLA, or candidate fields when making
an effective config.

## Evidence and acceptance rules

For each split job, retain the full evaluator report and require:

* top-level and `model_timings` status `complete`;
* reference qualification for every kernel rank;
* successful compile and complete-step CUDA Graph capture for every timed arm;
* 120 finite positive rows per timed arm, with changed inputs and one shared
  input hash per sample;
* zero timed graph breaks, recompiles, or new unique graphs.

Apply the same checks to each priority first-gate job and evaluate only its
four explicitly listed adjacent-rank-order pairs.
Compute each listed edge only from the two ranks in its split job, using the
shared per-sample inputs emitted by that job. Do not join raw rows, ratios,
bootstrap pools, or confidence intervals across split jobs. A familywise
all-12-rank result is therefore unavailable from the split matrix alone; the
manifest records this explicitly. The native FLA anchor may be compared with
each candidate within its job, but it does not fill a missing rank edge.

The JSON-only checks are CPU-safe:

```bash
/opt/anaconda3/bin/python -m json.tool configs/production_ladder_full.json >/dev/null
/opt/anaconda3/bin/python -m json.tool configs/production_ladder_block.json >/dev/null
/opt/anaconda3/bin/python -m json.tool configs/production_ladder_split_manifest.json >/dev/null
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -m pytest -q tests/test_production_ladder_configs.py
```
