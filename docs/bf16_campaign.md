# Historical BF16 campaign reproduction

The active refresh uses the [final sweep contract](../configs/bf16_final_sweep.json)
and the harnesses in [benchmark results](benchmark_results.md). Public operator,
changed-input replay, training, and cache-reuse qualification remain available;
see the [validation guide](validation.md).

The earlier 24-layer, width-1536 campaign used a different model, rank ladder,
optimizer, and budgeted launch system. Its launchers, training evaluator, and
standalone diagnostics have been removed from the active checkout. Their exact
sources, contracts, raw reports, and failures remain in recorded snapshots.
Historical measurements continue to describe those recorded bytes.

## Restore the recorded source

Use the source snapshot associated with the report being reproduced. The
`final78` snapshot records commit `fc0a2445d5ca24dd61dd5ac4a8179b0855a67c18`
and retains the retired campaign modules and tests. Its `manifest.json` binds
the files under `runner/`; preserve those bytes and the original contract.
Earlier reports may require their own earlier snapshots.

The older v1 performance-source tarball predates this BF16 campaign and does
not contain its evaluator. Keep the relevant campaign snapshots with the final
delivery bundle, including the recorded vendor and optimizer identities and
all raw failure records.

For a ledger-based campaign archive, restore one job with the retained tool:

```bash
python -m benchmarks.bf16_archive restore \
  "$ARCHIVE" "$JOB_ID" "$RESTORE_DIR"
```

`RESTORE_DIR` must not already exist. The tool verifies each restored file's
content hash. Ledger-based archives restore the runner under
`$RESTORE_DIR/snapshots/runner`; standalone source snapshots such as `final78`
place it under their own `runner/` directory. Public `fast-attnres-public-v1`
archives are also supported. External comparator and optimizer checkouts must
match the identities recorded for the original job.

Run historical launch/preparation commands from the restored runner, using its
original runbook and pinned environment. External `kernel-work` preparation
and billing scripts that hard-code the active checkout must instead target the
archived snapshot. Do not modify recorded sources or rebind old measurements
to the cleaned checkout.

## Summarize historical evidence

The active checkout retains the report readers, archive tool, and plotting
code. Supply explicit immutable inputs and the contract recorded for those
inputs. In these examples, `PRIMARY_CONTRACT` and `OPERATOR_CONTRACT` name the
original contract files, and `CAMPAIGN` contains the matching results/snapshots.

```bash
python -m benchmarks.bf16_report \
  "$CAMPAIGN/results/<job-id-1>/report.json" \
  "$CAMPAIGN/results/<job-id-2>/report.json" \
  --candidate candidate --contract "$PRIMARY_CONTRACT" \
  --output "$CAMPAIGN/primary-summary.json"

python -m benchmarks.bf16_campaign_report \
  "$CAMPAIGN/primary-summary.json" "$CAMPAIGN/ledger.json" \
  --output "$CAMPAIGN/bf16_campaign.md"

python -m benchmarks.bf16_broader_report \
  --work "$CAMPAIGN" --contract "$OPERATOR_CONTRACT" \
  --output "$CAMPAIGN/operator-summary.json" JOB_ID_1 JOB_ID_2
```

The training summarizer retains missing, failed, and admission-failure records.
The renderer copies recorded statistics and accounting bounds; it does not
reconcile billing, change samples, or turn partial coverage into a passing
campaign. The operator summarizer verifies archived evaluator/source identities
and keeps completed rows from interrupted jobs alongside unresolved coverage.

For a complete, admitted common-rank operator confirmation summary, the retained
plotter can regenerate its figures:

```bash
python -m benchmarks.plot_bf16 \
  "$CAMPAIGN/operator-summary.json" --output "$FIGURE_DIR"
```

The plotter rejects incomplete or broader/irregular-only summaries. Operator
latency and complete training-step latency remain separate measurements.

## Seal a campaign archive

The archive tool remains available for an already completed or reconciled
ledger-based campaign:

```bash
python -m benchmarks.bf16_archive create "$CAMPAIGN" "$ARCHIVE"
```

It refuses jobs still marked `reserved` or `running` and stores a
content-addressed copy of snapshot/result files plus the ledger. This is a
private archive that can include billing records and external checkouts; a
public evidence bundle is a separately reviewed subset. Keep the archive,
original contracts, summary, ledger, and rendered report together.
