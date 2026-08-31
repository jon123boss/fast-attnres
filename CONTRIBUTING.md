# Contributing to Fast-AttnRes

Thanks for taking an interest in `fast-attnres`. I built this repository as
a small research implementation of standard and sliced implicit-tail Attention
Residuals. A useful contribution makes the operator, the evidence, or the
installation surface easier for another person to inspect and reproduce. A
focused, tested change is more valuable here than a large change with an
unmeasured claim attached to it.

Please read the [README](README.md), [equation and API contract](docs/equation.md),
and [evaluation contract](EVALUATION.md) before changing the public operator,
validation, or benchmark paths. The repository contains an explicit equation
reference and optimized routes. Keep that distinction visible in code,
documentation, and result reports.

## Before you start

For a bug, search the existing issues and check whether the current checkout
already contains a fix. For a new idea, open a feature issue first when the
change would alter the model equation, public API, checkpoint behavior,
benchmark protocol, or supported environment.

The frozen evaluation contract, `reference.py`, `api.py`, `_sources.py`, `validation/`,
`EVALUATION.md`, and release tests are part of the evidence boundary. Do not
change them incidentally in an implementation pull request. A deliberate
contract change must explain the scope, update the manifest, and regenerate
the affected evidence.

Never commit credentials, tokens, private data, model access keys, or raw logs
that contain them. Replace secrets with `[REDACTED_SECRET]` before sharing a
command or log. If a real credential was exposed, revoke or rotate it before
opening an issue.

## Development workflow

1. Create a focused branch from the current default branch.
2. Make the smallest change that addresses the issue. Keep unrelated cleanup
   out of the same pull request.
3. Preserve the independent FP32 equation reference and the existing control
   conditions when adding or changing a CUDA, Triton, or source-list route.
   New routes should remain opt-in until their gates pass.
4. Keep the public contract intact: full-width values, implicit tail keys,
   parameter-free RMS key normalization, one static learned query, FP32
   equation math, and the declared BF16/FP32 storage behavior. Retired
   projected-key, carrier, dynamic-query, output-normalization, and source
   count-prior paths are outside the active interface.
5. Add or update a focused test for changed behavior. Update the design note or
   README when a command, flag, equation, or supported behavior changes.
6. Run the checks possible in your environment and record what you ran. CUDA
   or B200/H100 checks that you did not run must be marked as not run.
7. Open a pull request using the repository template. Include the exact
   revision, command, hardware, software versions, and raw artifact for any
   benchmark claim.

## Local checks

Use the checkout's interpreter; do not install dependencies globally. From the
repository root, the CPU/static checks are:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -m "not cuda" -q
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  src/attnres/*.py src/attnres/_kernels/*.py examples/train.py
```

The small training harness is a CPU/reference smoke test:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python examples/train.py \
  --backend reference --device cpu --mode full --variant standard --steps 2
```

On a configured CUDA device, the GPU-marked checks are separate:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -m cuda -q
```

The correctness-only benchmark harness is also a useful local check when its
optional dependencies are installed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python -m benchmarks.run \
  --scope smoke --phase correctness --no-fla \
  --out /tmp/attnres-correctness.json
```

If a dependency, device, or dataset prevents a check, report that limitation
instead of weakening the check or presenting a partial run as a full result.

## Correctness comes before performance

This is the project rule for every optimized path:

> A route is not eligible for timing, promotion, or a speed claim until its
> correctness gate has passed.

Before looking at timing numbers, the contributor and reviewer should be able
to answer all of these questions:

- What is the authoritative reference? Name the independent FP32 equation
  oracle and the exact candidate and baseline routes. Equivalent real-valued
  equations can still produce different low-precision gradients.
- Does the candidate match the reference in forward output and loss behavior at
  the declared dtype and tolerance? The frozen gates are BF16 `rtol=atol=0.05`
  and FP32 `rtol=0.001, atol=0.0001`, with finite outputs and gradients.
- Do all relevant gradients pass, including every source-value gradient, the
  query gradient, and all model parameters touched by a complete training
  graph? Include FP64 gradcheck where that contract applies.
- Does the candidate select the intended route? Check source order, source
  identity, dimensions, layouts, copies, launch metadata, and kernel-selection
  flags. A silent fallback or a different route is not a successful
  optimization.
- Does the gate cover the declared shape matrix, including standard (`R == D`)
  and sliced (`R < D`) cases, packed and source-list inputs, aliases/repeated
  reads, non-contiguous layouts, nonzero queries, and held-out dimensions where
  those cases are in scope?
- Are changed-input CUDA Graph replay, compilation, checkpointing, and
  complete-training all-parameter gradients checked when the route claims to
  support them?
- Are failed, skipped, untimed, fallback, or partial cells preserved as such?
  A harness status such as `passed` does not mean that a candidate was faster
  or accepted.

Only after every required correctness check passes should timing begin. A
performance comparison must keep the relevant conditions matched: source
revision, hardware, software and compiler, dtype, batch and sequence shape,
model state, optimizer, compilation mode, warmups, repetitions,
synchronization, and timing method. Include source/software/hardware hashes and
raw samples in a durable artifact.

For release ladder results, a pair is a gain only when the simultaneous paired
95% confidence interval for the latency ratio is entirely below `1`. A plateau
has an interval containing `1` and lying wholly in `[0.99, 1.01]`; an interval
entirely above `1` is a slowdown, and other outcomes are inconclusive. Every
required adjacent rank pair must pass its declared gain-or-plateau rule and
show the required gain over the larger rank. A correctness failure cannot be
traded for speed, and a correctness pass alone does not establish a
performance win.

Do not use words such as “faster,” “beats,” “production-ready,” or “fastest”
without a matched comparison and enough information for someone else to repeat
it. State the scope: a result may be a measured win for one named device,
dtype, shape, and route while remaining unproven elsewhere.

## Benchmark reports

Every benchmark report should include, at minimum:

- source commit or source hash, including the pinned FLA revision when FLA is
  a comparator;
- candidate route and baseline route;
- GPU or CPU model, driver/runtime, framework, compiler, and dtype;
- batch size, sequence length, hidden/value width, rank, source count, model
  configuration, source container, and relevant flags;
- complete timing boundary, warmup count, measured repetition count,
  synchronization, timing method, and whether compilation/profiling were
  excluded;
- correctness reference, tolerances, complete gate result, and evidence that
  correctness ran before timing;
- raw timing samples or a path to the saved JSON/CSV artifact; and
- whether the result is measured, diagnostic, provisional, rejected, or
  historical.

The primary timing boundary is a complete compiled training step: projection,
source assembly, loss, backward, accumulation, zeroing gradients, optimizer,
and any scheduler included by the declared configuration. Do not compare
reference-only or kernel-only timing with that metric. Do not report a smoke
test, skipped task, interrupted sweep, or untimed correctness run as a full
benchmark. Keep failed cells in the evidence ledger so later readers can see
what was tried and why it was not promoted.

## Pull requests

Keep the pull request focused and explain what changed, why it was necessary,
and how it was verified. For implementation changes, identify any changed
numerical, public API, layout, or installation contract. For benchmark changes,
explain how the evaluator is protected from accidental changes to inputs,
timing, or correctness logic.

I may ask for a smaller reproducer, an oracle comparison, or a held-out shape
before accepting an optimization. That preserves a useful research record and
keeps a local result from being mistaken for a universal guarantee.

By contributing, you agree that your contribution is provided under the
repository's [MIT License](LICENSE). Please follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
