# Changelog

User-visible changes to `fast-attnres`. Performance results remain tied to
their raw artifacts, source commits, and measurement conditions.

## [2.0.1] - 2026-09-06

This is the first published 2.x package. The 2.0.0 tag remains an unpublished
release attempt: its archive guard incorrectly rejected the evidence directory
entry. Version 2.0.1 corrects that packaging check; measured kernel code is unchanged.

- Narrow the public operator to CUDA BF16 values, queries, and first-order
  gradients, with FP32 internal math and an independent BF16 validation reference.
- Share one per-read backend across Full/Block standard and sliced LR AttnRes.
- Reduce unused auxiliary-gradient work and coalesce transposed upstream
  gradients; tune source kernels with CUDA Graph timing on each architecture.
- Preserve benchmark provenance and refresh the existing H100/B200 headline
  and competitor workloads at `R=D` and `R=D/4`. The headline step is 5.95%
  faster on H100 SXM and 22.74% faster on B200 than native FLA checkpoint 1;
  H100 D2048 is 0.28% slower, within the declared 1% parity band.
- Publish the complete final evidence, including original failures and explicit
  initial-logit normalization-rounding exceptions. Other training gates pass.
- Keep package runtime bytes identical to the final measured source except for
  the version literal; audit that relationship during release builds.
- Clarify the API and benchmark documentation, retaining earlier results as
  historical evidence.

## [1.0.0] - 2026-08-31

Historical release notes below describe that release and its measured source
identities. Its CPU/FP32 compatibility and performance statements do not
describe the current CUDA BF16 contract.

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
