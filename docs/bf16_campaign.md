# BF16 campaign runbook

This runbook describes the frozen BF16 campaign around `bf16_primary.py`,
`bf16_modal.py`, `bf16_report.py`, and `bf16_archive.py`. The campaign renderer
[`benchmarks/bf16_campaign_report.py`](../benchmarks/bf16_campaign_report.py)
only consumes JSON produced by those tools; it does not alter evaluation or
recompute timing statistics.

## Frozen scope

The primary matrix covers H100 and B200, both `full` and `block` modes, ranks
`1536 1024 768 640 512 384 256 128 64 32 16`, and seeds
`20260827 20260903 20260911`. This targets powers of two and common intermediate
widths; earlier development sweeps of unusual ranks remain historical evidence.
The original model geometry and controls are:

| Field | Primary value |
| --- | --- |
| Layers / width / heads | 24 / 1536 / 24 |
| MLP width | 4224 |
| Vocabulary / context | 100277 / 2048 |
| Batch / accumulation | 4 / 4 |
| Block count | 8 |
| Attention residual | `eps=2**-23`, `scale=1.0`, ordinary source assembly |
| Precision | BF16 cross-entropy and BF16 operator inputs/outputs/first-order gradients |
| Optimizer | original `Muon+AdamW(configured)` implementation |
| Gradient clipping | `1.0` |
| Activation checkpointing | disabled in primary; qualified separately |
| Timing | 10 warmups, 120 balanced paired rounds |
| Runtime | Python 3.11, PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1 |

Each timed step includes pinned input copies, source preparation, forward, loss,
backward, accumulation, gradient zeroing, and optimizer work. Dataset I/O,
logging, and scheduler host updates are excluded. The primary source inventory
is the one in `configs/bf16_primary.json`; the evaluator and
`validation/oracle.py` are immutable inputs.

Comparison models are qualified individually. For timing, the current contract
keeps all qualified models and optimizer states on the GPU when their measured
persistent allocations plus the largest measured temporary allocation fit with
at least 8 GiB or 10% device capacity in reserve. Otherwise it parks inactive
comparison models on CPU. These transfers occur outside timed steps. The memory
decision, storage independence, and actual timing residency are recorded.
Eight changed-input updates compare an uninterrupted resident control with a
transferred control exactly. Garbage collection keeps its ordinary policy.
Historical measurements using only one resident model have a different fixture
identity and remain separate; none of their samples are discarded.

Correctness uses `rtol=0.05` and `atol=0.05` for outputs and first-order
gradients. It covers packed and ordered source lists, repeated reads, partial
Blocks, changed inputs, non-contiguous layouts, duplicate/shared sources,
compiled replay, optimizer updates, and save/resume. Nonzero-query operator
qualification uses eight replays at query scale `0.05`. The report compares each
cell with its strongest correct eligible alternative and uses simultaneous 95%
confidence intervals. Adjacent lower/higher-rank gates use the recorded ratio
upper bound; missing or inconclusive coverage is an unmet target.

The package contract is the CUDA BF16 call
`attnres(values, query, *, eps=2**-23, scale=1.0)`. Full and sequential Block
reads use the same call. Block sums are ordinary caller-owned source tensors;
each read evaluates its current inputs. Projected keys, priors, and
architectural changes are outside this campaign.

## Configuration

Run these commands from the repository root. Set `CAMPAIGN` to a new directory
for this campaign; the command writes a configuration only and does not rent a
GPU.

```bash
export PYTHONPATH=src:.
CAMPAIGN=/absolute/path/to/bf16-campaign
mkdir -p "$CAMPAIGN"

python -m benchmarks.bf16_primary \
  --modes block \
  --ranks 64 32 16 \
  --seeds 20260827 \
  --output "$CAMPAIGN/primary-config.json"
```

The example creates a bounded slice; repeat for the remaining modes, ranks, and seeds.
`bf16_primary.py` accepts `--contract PATH` when the frozen contract is stored
elsewhere. Its output contains the selected cases, rounds, warmups, backend
inventory, expected source identities, and the contract digest.

Initialize the ledger once. `init` creates the ledger and does not rent a GPU:

```bash
ATTNRES_CAMPAIGN_WORK="$CAMPAIGN" \
  python -m benchmarks.bf16_modal init --cap 500
```

The cap is US$500. The launcher reserves at most US$80 for `baseline`, US$220
for `experiments`, US$140 for `confirmation`, and US$60 for `reserve`.

## Prepare

`prepare` copies the runner, each named source, each named competitor, and the
optional optimizer into a timestamped snapshot. For a primary training job,
the names below cover the identities required by the frozen contract. Replace
the placeholder paths with clean, immutable checkouts that match the contract.

```bash
CANDIDATE=/absolute/path/to/candidate-checkout
RELEASE=/absolute/path/to/release-checkout
FLA=/absolute/path/to/fla-checkout
LIGER=/absolute/path/to/liger-checkout
LEGACY=/absolute/path/to/legacy-checkout
CATSWE=/absolute/path/to/catswe-checkout
HYDRA=/absolute/path/to/hydra-checkout
HILDA=/absolute/path/to/hilda-kernel-directory
OPTIMIZER=/absolute/path/to/original-optimizer-root

ATTNRES_CAMPAIGN_WORK="$CAMPAIGN" \
  python -m benchmarks.bf16_modal prepare \
    --config "$CAMPAIGN/primary-config.json" \
    --source "release=$RELEASE" \
    --source "candidate=$CANDIDATE" \
    --competitor "fla=$FLA" \
    --competitor "liger=$LIGER" \
    --competitor "legacy=$LEGACY" \
    --competitor "catswe=$CATSWE" \
    --competitor "hydra=$HYDRA" \
    --competitor "hilda=$HILDA" \
    --optimizer-source "$OPTIMIZER" \
    --gpus 1 \
    --timeout 2400 \
    --gpu H100 \
    --name h100-primary \
    --stage baseline
```

Use `--gpu B200` and a distinct `--name` for the B200 job. This continuation
permits one GPU at a time, including across architectures. Intermediate work
uses B200; final H100 and B200 qualification runs sequentially. The launcher
rejects distributed jobs and overlapping reservations. `--timeout` accepts 600 through 10,800
seconds. `prepare` prints the snapshot path; set it explicitly for the next
command:

```bash
SNAPSHOT=/absolute/path/to/bf16-campaign/snapshots/JOB_ID
```

The source paths are read at preparation time and copied into the snapshot.
The snapshot records byte digests and Git origins where available. The remote
runner verifies those digests before execution and verifies the selected
candidate against the runner for qualification jobs. Do not edit a source,
the evaluator, the oracle, or a prepared snapshot after this point; prepare a
new job when source bytes or controls change.

## Run

`run` is the only shipped command in this layer that can admit a paid job. It
reserves the full worst-case timeout, a 300-second startup allowance, requested
CPU/memory, GPU multiplicity, and the launcher's 10% margin before submission:

```bash
ATTNRES_CAMPAIGN_WORK="$CAMPAIGN" \
  python -m benchmarks.bf16_modal run "$SNAPSHOT"
```

At most one single-GPU job per architecture is active at a time. Reservations
remain commitments after failures; they are not actual bills and are not
automatically refunded. Reconcile an interrupted client by confirming that its app is stopped with zero
running containers. Resume only missing cells in a new job; an admitted job ID
cannot be rerun.

A completed `run` writes `$CAMPAIGN/results/JOB_ID/report.json`. Retrieve durable
reports, child-process logs, and incremental history after completion or an
interrupted client without renting another GPU:

```bash
ATTNRES_CAMPAIGN_WORK="$CAMPAIGN" \
  python -m benchmarks.bf16_modal fetch JOB_ID --history
```


## Summarize and render

Pass the completed training report JSON files explicitly to the existing
summarizer. Explicit paths keep the input set reproducible and let the
summarizer reject duplicate cells or mismatched source identities:

```bash
python -m benchmarks.bf16_report \
  "$CAMPAIGN/results/<job-id-1>/report.json" \
  "$CAMPAIGN/results/<job-id-2>/report.json" \
  --candidate candidate \
  --contract configs/bf16_primary.json \
  --output "$CAMPAIGN/primary-summary.json"
```

Add every immutable completed report needed for the intended matrix. The
summarizer preserves missing, failed, and admission-failure records; it does
not treat a partial input set as a passing primary campaign.

Render the delivery Markdown from the primary summary and the ledger:

```bash
python -m benchmarks.bf16_campaign_report \
  "$CAMPAIGN/primary-summary.json" \
  "$CAMPAIGN/ledger.json" \
  --output "$CAMPAIGN/bf16_campaign.md"
```

The renderer reports the candidate source hash, the recorded `primary_pass`,
recorded/admitted comparisons, missing entries, and failure/admission records.
It shows each configuration and failed adjacent-rank gate, with all intervals
retained in the linked summary. Latencies, ratios, intervals, and geometric
mean speedup are copied from that summary. The speedup is always labelled as
an observed-cell result; no negative result is removed or global winner claimed.

Wider and irregular operator measurements have a separate full contract. Supply
job IDs whose snapshots and results are present in the campaign directory:

```bash
python -m benchmarks.bf16_broader_report \
  --work "$CAMPAIGN" --contract configs/bf16_broader.json \
  --output "$CAMPAIGN/broader-summary.json" JOB_ID_1 JOB_ID_2
```

The summary verifies the archived evaluator and source identities, accepts
completed rows from interrupted jobs, and retains unresolved arms and missing
coverage. Operator pairing follows the verified evaluator's alternating order;
the eight changed-input correctness replays precede timing. These measurements
do not establish complete-step speedups.

## Archive and restore

Seal only after all jobs are terminal or explicitly reconciled:

```bash
python -m benchmarks.bf16_archive create \
  "$CAMPAIGN" \
  "${CAMPAIGN%/*}/bf16-campaign.zip"
```

`create` stores a content-addressed copy of snapshot and result files plus the
ledger manifest. It refuses active `reserved` or `running` jobs. Restore one
job into a directory that does not already exist:

```bash
ARCHIVE=/absolute/path/to/bf16-campaign.zip
RESTORE_DIR=/absolute/path/to/restored-JOB_ID

python -m benchmarks.bf16_archive restore \
  "$ARCHIVE" \
  JOB_ID \
  "$RESTORE_DIR"
```

The restore command verifies each content hash before writing the recorded
relative paths. Keep the archive, primary summary, ledger, and rendered report
together when delivering the campaign record.

Compiler artifacts and Triton autotuning metadata may persist between fresh
benchmark processes. The primary contract enables `TRITON_CACHE_AUTOTUNING=1`
uniformly for every arm; the launcher sets it before backend imports and records
the input archive hash. No source-dependent tensors or Block state persist.
Qualification checks cold and warm fresh processes against the same BF16 oracle
and eight changed-input CUDA Graph replays, then verifies identical selected
configurations without retuning in the warm process. `compile_warmup_s` includes
compilation, correctness checks and warmup; it is not pure cold compilation time.
The metadata policy follows the pinned [Triton 3.7.1 autotuner](https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/runtime/autotuner.py).

Longer model jobs reserve their full bound before admission and retain the
same stage and total caps; retries remain off.
Completed cell reports from a timed-out parent remain usable, while unfinished
cells remain incomplete. Select the first complete attempt, never the fastest
retry. Modal supports these bounds via its [function timeout](https://modal.com/docs/guide/timeouts).

Training uses each cyclic backend ordering followed immediately by its exact
reverse, giving every backend pair 60 first/second exposures over 120 rounds.
Unused allocator blocks are released before constructing each model arm,
outside timing. Allocated, reserved, and driver-free memory are recorded
separately. A resource failure remains unresolved and cannot remove an eligible
alternative from a strongest-alternative claim.

The final comparison keeps only one model arm's parameters, gradients, and
optimizer state on GPU. Inactive comparison state resides on CPU; no residual
source state is reused. Transfers between comparison arms are bookkeeping
outside the CUDA-event update and are reported as `residency_transfer_s`. Each
timed update still includes its input copies and all training work listed above.
Before model measurements, an eight-update check compares the transferred model
and original Muon+AdamW state against an uninterrupted control, preserving
Parameter identities and checking that transfers do not repeatedly recompile
the model. These transfer costs are not production training latency.

### Verified billing reconciliation

The launcher retains every original full-timeout reservation. A stopped app can
be reconciled explicitly using two identical hourly Modal billing readings at
least ten minutes apart, covering every GPU/CPU/memory row and at least one hour
after shutdown. Hash-bound proof lives in that job's results directory. The
accounting bound retains 150% of the metered charge plus $0.25, capped by the
original reservation. Active, unmatched, changing, or incomplete metering keeps
the full original bound. Admission verifies the retained evidence again; a
client error or short elapsed time alone never releases budget. The report
shows historical reservations and current accounting bounds separately.

Distributed save/resume verifies the serialized checkpoint and restored model
and optimizer state exactly before the next update. Same-input continuation
uses the unchanged BF16 oracle for state and loss; bitwise continuation equality
is retained as a diagnostic because floating-point collective reduction order
can differ. Earlier exact-continuation failures remain in the raw records.

### Separate diagnostics

Configurations with `gc_diagnostic: true` record Python collection intervals
and host step windows in `report.gc.json`, alongside the ordinary measurements.
They leave collection policy and training arithmetic unchanged. Compiler memory
diagnostics instead set `activation_memory_budget` to `0`, `0.25`, or `0.5`
before compilation. Both entry points reject primary-contract configurations;
their observations cannot replace primary samples or justify dropping outliers.
