# Changelog

I keep this file for user-visible changes to `fast-attnres`. Research
results and experimental benchmark outcomes should remain traceable to their
raw artifact, source commit, and stated evaluation conditions; a changelog
entry is not a replacement for that evidence.

## [Unreleased]

## [1.0.0] - 2026-08-31

### Added

- The `attnres(values, query)` public operator for standard AttnRes (`R == D`)
  and sliced LR-AttnRes (`R < D`), with packed tensors or ordered source lists.
- Native first-order backward for every source value and the learned query,
  plus CPU equation fallback, `torch.compile`, and CUDA Graph support.
- Full and per-read Block examples that share the same public operator; Block
  changes the source schedule rather than introducing a second kernel API.
- Runnable examples for standard AttnRes, LR-AttnRes, packed and list inputs,
  backward, Full/Block schedules, and `torch.compile`.
- Deterministic Matplotlib benchmark figures, eight current H100/B200 adoption-
  screen reports, and a current three-seed 24-layer Full campaign on both GPUs
  with exact raw reports and a fail-closed offline auditor. The earlier
  campaign remains separately archived against its measured source identity.
- A bounded PyTorch 2.13.0+cu130 and Triton 3.7.1 adoption profile, qualified
  on H100 and B200, plus three-seed BF16 Full complete CUDA Graph training-step
  measurements against pinned native FLA checkpoint 1 on the same runtime.
- A fail-closed external capability registry that keeps FLA Gluon, Liger,
  Catswe, Hydra, checkpoint-0, and sliced-rank exclusions explicit instead of
  treating unsupported routes as benchmark wins.
- Reproducible wheel, source distribution, evidence archive, checksums, and a
  trusted-publishing release workflow.
- Contributor guidance for preserving the equation reference and reporting
  reproducible experiments.
- Security reporting and secret-handling guidance.
- A Code of Conduct and GitHub issue forms for bugs, feature requests,
  support questions, and benchmark/correctness reports.
- A pull request checklist that makes correctness evidence explicit before
  performance evidence.

### Changed

- Established **Fast Attention Residuals** (`fast-attnres`) as the project and
  installable distribution name while retaining the concise `attnres` Python
  import.
- Standard Attention Residuals now lead the documentation before the optional
  sliced LR-AttnRes optimization.
- Cached Block preparation and merge APIs were removed. Full and Block both
  use direct calls to the same public AttnRes primitive.
- CI now checks the frozen release contract once before the test matrix, so a
  stale manifest produces one precise failure instead of dozens of cascaded
  benchmark-test failures.
- Documented the project rule that correctness must pass before a route is
  timed, promoted, or described as a performance improvement.
- Clarified that BF16 storage/autocast is the optimized training and timing
  target, while FP32 remains available for the equation reference,
  compatibility checks, and debugging.
- Preallocated FLA's nonpersistent unit RMS-weight model buffer before compile
  and capture, removing its allocation/fill launch from the matched step.
- Replaced the historical headline with the current-release PyTorch 2.13 /
  Triton 3.7.1 three-seed compiled-step campaign. Seeds and GPUs remain
  unpooled.
- Documented multi-read BF16 accumulation-order mismatches as retained strict
  failures rather than silently relaxed correctness results.
