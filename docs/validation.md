# Validation protocol

This repository is an experimental rebuild whose CUDA training target is BF16
storage/autocast. Correctness is checked against the explicit FP32 equation
before any timing is considered, and the results below are scoped to the named
checks rather than presented as blanket release qualification. FP32 remains a
reference and debugging mode; it is not a separate performance target.

## Current scope

The public `attnres` route uses `src/attnres/_kernels/fixed_tail.py` for packed
calls and `fixed_tail_sources.py` as the ordered source-list adapter. Within
that adapter, bounded BF16 lists (`D <= 2048`) use the FLA-derived
`fla_full_sources.py` kernels; FP32 and wider BF16 lists use the fixed-tail
source-list fallback. The selected direct fixed-tail gate at `de96c7b` passed
44/44 checks on H100 and 44/44 on B200. The public-route gate frozen at
`6beb157` subsequently passed 75/75 selected checks on each GPU with PyTorch
2.11.0+cu130 and Triton 3.6.0. This includes source layouts, alias gradients,
compiled replay, and small complete-training checks for Full, per-read Block,
and native FLA comparators. It is not the full CUDA suite or a performance
qualification.

The active release evidence uses only the native FLA Triton checkpoint-1 arm
described in the production ladder. The historical correctness wording above
does not promote checkpoint-0 or Gluon diagnostics to release comparators.

The current standard source-list implementation is identified by commit
`8ddb0bbaf184663703ded65b45839fddd1c429fc` and hashes `1373614c93d7…`
(`fixed_tail_sources.py`) and `8749c72c4714…` (`fla_full_sources.py`). The
scalar-compact config and selected-codegen probe retain older hashes as
historical diagnostics; their tests do not relabel those bytes as current.

The timed and promoted training surface is BF16. FP32 cases remain useful for
the equation oracle, API compatibility, and debugging, but an FP32 pass does
not support a speed, ranking, or no-regression claim. Where the existing
correctness matrix checks query gradients, that check is correctness coverage
for the selected storage dtype; it does not turn FP32 checks into a performance
workload.

Older operator and model measurements belong to earlier implementations and
remain historical. They must not be used as measurements of this rebuild or
combined with its correctness gate. The full CUDA suite, broader shape
coverage, and performance behavior still require independent qualification.

## Reproducible local checks

Run these commands from the repository root with the checkout's dependencies
already configured. CPU import and reference execution do not require Triton.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -m "not cuda" -q
PYTHONDONTWRITEBYTECODE=1 python -m pytest -m cuda -q
```

The second command requires a configured CUDA device and Triton; it does not
provision hardware. To inspect the training example's current interface:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python examples/train.py --help
```

The sliced example requires an explicit `--rank`; standard is the default.

The correctness-only benchmark harness writes its JSON outside the checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m benchmarks.run \
  --scope smoke --phase correctness --no-fla \
  --out /tmp/attnres-correctness.json
```

A report should retain the JSON, source revision, software versions, device,
dtype, shapes, and whether inputs were packed or supplied as a source list.
Failed or incomplete checks remain visible; they are not silently promoted to
timing data.

## Required correctness coverage

The equation comparison should cover both standard (`R == D`) and sliced
(`R < D`) reads in Full and Block schedules, BF16 and FP32 storage where
available, and first-order gradients for values and queries. Block checks
must exercise repeated reads, completed plus partial source sets, changed
inputs, non-contiguous layouts, duplicate/shared source aliases, and source
lists.

Source-list tests may observe a separate contiguous copy for an individual
non-affine producer. That implementation detail does not imply universal
zero-copy behavior or a performance result. Full and Block route checks must
also prove that both schedules invoke the same public operator for a fixed
source layout.

In a multi-read BF16 graph, an intermediate BF16 store can round a partial sum
before a later read, and the resulting addition order can differ from a
flattened FP32 equation reference. A sparse difference may therefore identify
accumulation order rather than a different real-valued equation. The
diagnosis does not waive the gate: strict BF16 mismatches remain failures, with
their raw shape, route, and gradient evidence retained.

No timing claim follows from passing these checks. Any future performance
comparison needs a fixed source revision, named hardware and software stack,
matched inputs and model state, an explicit end-to-end timing boundary, and
paired samples with failed arms retained.
