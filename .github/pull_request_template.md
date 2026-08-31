## What changed

<!-- Describe the smallest coherent change and why it is needed. Link the
issue or design note when one exists. -->

## Scope and risk

- Affected files/routes:
- Changed numerical, public API, layout, packaging, or installation contract:
- Rollout or compatibility risk:

## Correctness evidence (required before performance evidence)

<!-- A performance claim cannot be reviewed without a correctness result. -->

- [ ] I named the independent FP32 equation oracle and the candidate and
      baseline routes.
- [ ] Forward output and loss behavior pass at the declared dtype and
      tolerance; outputs and gradients are finite.
- [ ] All relevant gradients pass, including every source-value gradient, the
      query gradient, and all model parameters touched by the route.
- [ ] I checked route identity, source order, dimensions, layouts, aliases,
      launch metadata, and kernel-selection flags; no silent fallback is
      included in the result.
- [ ] I tested the declared shape/rank/source-count matrix, including standard
      (`R == D`) and sliced (`R < D`) cases and held-out cases where in scope.
- [ ] I preserved failed, skipped, partial, fallback, and untimed cells
      instead of presenting them as accepted results.
- [ ] I ran the relevant checks below and recorded any checks that could not
      run.

Reference, tolerances, shape matrix, and results:

```text
<!-- Paste a concise summary or link to a durable artifact. -->
```

## Performance evidence (complete only if this PR makes a performance claim)

- [ ] Correctness passed before timing began.
- [ ] Candidate and baseline use matched source revision, hardware, software
      and compiler, dtype, shapes, model state, optimizer, compilation mode,
      warmups, repetitions, synchronization, and timing method.
- [ ] The timing boundary is stated; compilation, warmup, and profiling are
      excluded or reported separately.
- [ ] I included raw timing samples and a durable JSON/CSV artifact with
      source/software/hardware provenance.
- [ ] I state whether the result is measured, diagnostic, provisional, rejected,
      historical, or accepted; `passed` alone does not mean faster.
- [ ] I do not use “faster,” “beats,” “production-ready,” or “fastest” beyond
      the conditions actually tested.

Benchmark command and environment:

```text
<!-- Include commit/hash, device, dtype, B/T/D/R/S, baseline, timing boundary,
     warmups, repetitions, and artifact path. -->
```

## Tests

```text
# Commands run and their results
```

## Documentation and review notes

- [ ] I updated the README/design note when behavior, commands, or supported
      boundaries changed.
- [ ] I removed secrets and private data from logs and artifacts.
- [ ] I did not modify the frozen evaluator, equation reference, validation
      contract, or root-owned tests outside an explicitly assigned migration.
- [ ] I added or updated focused tests where coverage was needed.
- [ ] I called out known limitations and follow-up work.
