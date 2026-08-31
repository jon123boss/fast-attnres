# Matched competitor protocol and runners

> **Development protocol, not current performance evidence.** This document
> specifies the optional eager operator-compatibility harness. No timing from
> that harness feeds the README or release claims. Current performance evidence
> is the audited compiled complete-training-step screen described in
> [`docs/benchmark_results.md`](benchmark_results.md).

[`configs/matched_competitor_benchmark.json`](../configs/matched_competitor_benchmark.json)
is the sealed selection and timing contract. The Modal transport in
[`benchmarks/modal_competitor_runner.py`](../benchmarks/modal_competitor_runner.py)
loads that contract, checks the runtime and device, and delegates capability
planning, qualification, paired operator timing, and report construction to
`benchmarks.comparator_runner`. It is separate from the release transport:
the latter uses the release evidence runtime, while this worker uses the
matched-comparator runtime below.

## Product dtype scope

BF16 storage/autocast is the product and performance target for this release.
The sealed capability matrix records FP32 where an adapter accepts it so that
reference and compatibility coverage stays explicit, but the matched timing
claim is BF16-only. FP32 correctness evidence is not an FP32 speed, ranking, or
no-regression claim.

## Runtime and device gate

The worker image is Python 3.11 with these pinned packages:

| Component | Pin | Check or role |
| --- | --- | --- |
| PyTorch | `2.13.0` from the `cu130` index | `torch.__version__` base is `2.13.0` and `torch.version.cuda` is `13.0` |
| Triton | `3.7.1` | exact `triton.__version__` |
| einops | `0.8.1` | required by the pinned FLA adapter |
| NumPy / packaging | `2.2.6` / `25.0` | image dependencies |

The runtime check runs before optional adapter discovery or import. Each remote
function must expose exactly one CUDA device: `H100!` means an H100 with SM90
(`compute_capability [9, 0]`), and `B200` means a B200 with SM100
(`[10, 0]`). A mismatch produces a failed per-GPU report. The source
fingerprint records SHA256 hashes for the project Python trees, the sealed
configuration and schema, and the transport file.

## Standalone vendor roots

Set the host paths before `modal run`. For a complete comparator matrix, set
all four variables; an unset family is deliberately mounted nowhere and is
reported as a missing route.

| Host variable | Family | Fixed container mount |
| --- | --- | --- |
| `ATTNRES_FLA_DIR` | FLA | `/workspace/vendors/fla` |
| `ATTNRES_LIGER_DIR` | Liger | `/workspace/vendors/liger` |
| `CATSWE_ROOT` | Catswe | `/workspace/vendors/catswe` |
| `HYDRA_ROOT` | Hydra/Manish | `/workspace/vendors/hydra` |

For example:

```bash
export ATTNRES_FLA_DIR=/absolute/path/to/standalone/flash-linear-attention
export ATTNRES_LIGER_DIR=/absolute/path/to/standalone/Liger-Kernel
export CATSWE_ROOT=/absolute/path/to/standalone/flash-attn-res
export HYDRA_ROOT=/absolute/path/to/standalone/attnres-kernel-lab
```

The paths may point at a package subdirectory; the launcher normalizes them to
the checkout root. Every family requires a clean standalone pinned checkout
with a real `.git` directory and exactly one matching public origin. A Git
worktree `.git` pointer file is rejected. The launcher creates a commit-only
Git bundle, copies that bundle into the image, clones it at the fixed container
path, and assigns the pinned public origin. This preserves symlinks and file
modes that ordinary directory mounts can normalize. Adapter discovery then
re-verifies the pinned revision, tree, cleanliness, source/license hashes,
origin, and module origins before import. The worker never falls back to an
ambient checkout. The optional `--fla-root`, `--liger-root`, `--catswe-root`,
and `--hydra-root` flags are consistency metadata and must agree with their
environment variables; they do not create a new transport.

## Scopes and launch

The `scope` argument is one of `smoke`, `primary`, or `heldout` and selects
only the corresponding explicit operator cases in the sealed configuration.
The current matrices contain eight smoke, five primary, and four held-out
cases. Every case carries explicit `S`, `N`, `D`, `R`, and `dtype` values; the
primary cases use `R = D`, while smoke is a quick mixed geometry gate. The
current held-out cases all use `R < D`, so they remain explicit
`not_applicable` audit rows under the external and candidate eligibility gates
and produce no timed operator rows. All three sets are fixed before results
are observed. The three protocol seeds (`20260827`, `20260903`, and
`20260911`) are always run; seed, warmup, and timed-round overrides are
rejected.

Run both devices concurrently and save the local report with:

```bash
modal run benchmarks/modal_competitor_runner.py \
  --gpu both \
  --scope smoke \
  --config configs/matched_competitor_benchmark.json \
  --output /absolute/path/matched-smoke-both.json
```

Use `--scope primary` or `--scope heldout` for the other predeclared sets.
For one device, use `--gpu 'H100!'` or `--gpu B200`. `both` submits one
H100 function and one B200 function through a local thread pool; each
function is still restricted to one visible GPU. `--plan-only` materializes
the plan without operator execution, but a Modal invocation still performs
the remote runtime and hardware checks.

The CPU-only config check does not import Modal, Torch, or Triton:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /opt/anaconda3/bin/python benchmarks/modal_competitor_runner.py \
  --validate-config
```

## Report artifact and cost boundaries

`--output` is a local JSON path. The file is written through a temporary path
and replacement, and has schema
`attnres.matched_competitor_benchmark.report.v1`. Its top-level metadata
includes `config_path`, `config_digest`, `scope`, `host_vendor_roots`, and a
`results` entry for each selected GPU. Each entry records the requested GPU,
status, exact actual runtime/module paths, hardware, source fingerprint, the
fixed `vendor_roots`, and `measurements`. The measurements contain the full
capability plan (including `not_applicable` audit rows), the selected operator
rows, route metadata, and statistics. A remote exception is retained with
`status = "failed"`, an exception object, and a traceback. Without `--output`,
only the short JSON status summary is printed and no report artifact is
persisted.

The Modal function requests four CPUs, 32,768 MB, a 1,800-second timeout, at
most one container per GPU function, and zero retries. `both` can therefore
run two billable GPU containers concurrently. The runner records resource
settings but does not estimate or enforce a dollar budget; check current
Modal rates and obtain launch approval before submitting a GPU run.

If this development harness is run, its operator timing boundary is a
CUDA-event measured forward plus backward
invocation. It includes adapter-owned source stacking and contiguous
preparation, so those costs are part of the comparison. The development
operator timing surface uses BF16 inputs and autocast. Ten warmup rounds are
excluded and 120 paired timed rounds are required. The model contract names a
complete compiled training-step boundary, but this worker leaves model cells,
including all LR-rank cells, planned until a model runner supplies their
complete-step inputs; it does not claim model timings.

## Eligibility and comparison surface

Before timing, both the public `attnres` candidate and the comparator must pass
the independent equation oracle evaluated in FP32 for the output, every
source-value gradient, and the query gradient. A capability-eligible row
enters the qualified denominator only after that gate and a complete set of
120 `ok` rows. Missing routes,
failed qualification, and incomplete timing remain visible and do not enter a
denominator.

Multi-read BF16 training can expose a small strict mismatch when an
intermediate is stored in BF16 and later additions associate differently from
the flattened reference graph. That accumulation-order explanation is a
diagnostic hypothesis supported by the graph structure, not a waiver. The
unchanged strict gate retains the row as failed until the route passes; no
tolerance is widened and no failed cell is timed as a win.

The declared native comparator surface is:

| Comparator family | Eligible scope |
| --- | --- |
| FLA Triton checkpoint 1 | standard `R = D` operator, Full, and public per-read Block; `S <= 129`, `D <= 8192`; BF16 or FP32 |
| FLA Gluon (conditional) | the same `R = D` scopes and storage dtypes, subject to the pinned compile envelope below |
| Liger v0.8.2 | standard `R = D` operator, Full, and public per-read Block; relevant source count `S <= 32`, `D <= 8192`; BF16 or FP32 |
| Catswe phase 1 | BF16 standard operator only, `R = D`, `S <= 129`, power-of-two `D <= 8192`, and `nextpow2(S) * D <= 1,048,576` |
| Hydra 2P | standard operator and declared Block panel; native timing is limited to `D <= 256` |

FLA Triton checkpoint 0 is diagnostic-only and is never an eligible
denominator. Catswe has no cached Block or model route, and the public AttnRes
Block path is per-read; no cached Block method is benchmarked here.

All external comparator capabilities require `R = D`. The `lr_ranks` arm is a
candidate model arm with ranks `[16, 64, 128, 512, 1024]`; its `R < D` rows
are therefore predeclared `not_applicable` for every external comparator.
There are no LR-versus-external-comparator measurements. The config's LR
architectural comparisons (`lr_rank_over_standard_operator` and adjacent LR
edges) are candidate-only model comparisons and are not executed by this
operator worker.

### Conditional Gluon compile envelope

The pinned FLA Gluon checkpoint-1 source statically unrolls the source loop,
so the sealed protocol applies one transparent geometry rule before native
allocation or compilation. Let `BD = next_power_of_two(D)` for the feature
tile visible to Gluon. A conditional Gluon row must satisfy
`BD <= 4096`, `S * BD <= 2^18` (262,144), and the documented checkpoint-1
static-work score `33 * S * BD <= 33 * 2^18 = 8,650,752`. For Block, `S` is the
number of sources supplied to that individual read, not the model's total
history. These are compile-safety bounds, not performance claims or a shape
dispatch table. The existing `S=129, D=8192, R=D` smoke row remains an explicit
`not_applicable` audit row because `BD=8192`; it is never silently routed to a
different comparator. Gluon remains conditional and development-only until
independent H100 and B200 correctness, graph, compile, and timing gates pass.

## Dependency compatibility without vendor edits

The pinned FLA Gluon source spells its zero-argument CTA barrier as
`gl.thread_barrier()`. Triton 3.7.1 exposes the equivalent builtin as
`gl.barrier(cluster=False)`. Before any FLA package import, the adapter places
the exact Triton builtin object at the old module attribute. It does not wrap
the function or edit the pinned vendor checkout, and its report records the
Triton version, alias mode, builtin identity, call form, and
`vendor_source_modified = false`. The bridge is fail-closed outside the
explicitly handled Triton versions.

Liger's native forward returns a reshaped value tensor solely as backward
state. On a contiguous input that auxiliary aliases the custom-op input, which
PyTorch 2.13 correctly rejects as an operator output. The adapter keeps the
native forward computation, exposes only the mixture and scalar statistics,
and saves the input through `setup_context`; backward recreates the same view.
No clone or source-stack copy is added to the timed path by this repair.

Catswe's source phase materializes a Triton block of shape
`[nextpow2(S), D]` and uses `tl.arange(0, D)`. Its power-of-two width and
1,048,576-element compiler envelope are therefore checked in the sealed plan
and registry before input allocation, then checked again by the adapter before
native launch or timing. An unsupported cell is retained as
`not_applicable`; it is not a failed benchmark and never enters a denominator.

## Pairing, failures, and statistics

Each pair uses the same device, initial state, ordered source-list inputs, and
upstream. One deterministic arm permutation is used for even rounds and its
exact reverse for odd rounds. The reported ratio is candidate over baseline.

Every raw row retains `seed`, `gpu`, `round_index`, `order_index`,
`input_hash`, `arm`, `status`, `latency_ms`, and failure provenance. Failure
and audit statuses are `failed`, `skipped_due_to_failure`, and
`not_applicable`; skipped rounds are recorded with the originating failure.
Failures and raw samples are never dropped, missing samples count as failures,
there is no interpolation, and an unchanged failure is not retried. A failed
warmup or timing event produces a complete retained failed/skipped matrix, but
the cell remains incomplete and cannot enter the statistics denominator.

The promotion estimator is a simultaneous paired-ratio bootstrap with 20,000
resamples, 95% confidence, common resample indices, and a 0.01 plateau margin.
Familywise intervals are computed over all planned comparisons within each
`(comparison_family, GPU, seed)` group. Seeds and GPUs are never pooled, the
per-seed gate remains active, and comparison selection never occurs after
results are observed. Incomplete, missing, diagnostic, and inapplicable rows
remain in the report but are excluded from these complete eligible groups.
