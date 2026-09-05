# CUDA BF16 validation protocol

This page specifies validation for the CUDA BF16 candidate surface. It is a
protocol, not a new result report: this documentation patch does not claim a
new GPU qualification or timing result.

## Contract gate

Every candidate check must use:

- CUDA tensors with `torch.bfloat16` values and queries;
- BF16 operator output and BF16 first-order value/query gradients;
- the public call `attnres(values, query, *, eps, scale)`;
- ordinary packed or ordered source tensors;
- `rtol=0.05` and `atol=0.05` for BF16 output and gradient comparisons.

The implementation may accumulate normalization, logits, softmax, and value or
gradient reductions in FP32 internally. The gate checks the BF16 tensors at the
operator boundary and does not treat internal accumulator dtype as a public
mode.

The only allowed oracle in a correctness test is a test-local PyTorch/autograd
calculation over BF16 inputs. It is test machinery, not an importable runtime
entry point and not a second public operator.

## Required correctness coverage

The CUDA gate should exercise:

1. standard Full reads with `R == D` and sliced reads with `R < D`;
2. packed tensors and ordered list/tuple sources;
3. Full source assembly from an embedding plus writer outputs;
4. per-read Block assembly from an embedding, completed block sums, and a
   current partial sum;
5. repeated reads, changed inputs, non-contiguous layouts, duplicate sources,
   and shared source views;
6. finite output, value-gradient, and query-gradient checks;
7. compiled replay where the surrounding training graph requires it.

Full and Block must call the same public function with ordinary source
containers. A stateful Block read path is outside the candidate contract. A
failed strict comparison stays failed; the tolerance is not widened and the
case is not converted into a timing row.

## Static and GPU checks

The examples are intentionally GPU-only. A no-write syntax check can run with
the configured Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -c \
  'import ast; from pathlib import Path; [ast.parse(p.read_text()) for p in Path("examples").glob("*.py")]'
```

On H100 or B200, run the repository's CUDA-marked checks with the checkout's
CUDA and Triton dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -m cuda -q
```

Run the example smoke checks only on a configured CUDA device:

```bash
python examples/standard_attnres.py --device cuda
python examples/block_schedules.py --device cuda
python examples/backward.py --device cuda
python examples/torch_compile.py --device cuda
```

These checks establish behavior for the named shapes and routes. They do not
establish throughput, end-to-end speed, or release qualification.

## Timing and evidence

Timing may begin only after the BF16 output and gradient gates pass for the
exact source revision, hardware, software stack, shapes, source layout, and
training graph being timed. A timing report must retain the source revision,
device, runtime versions, dtype, shapes, timing boundary, warmup policy, paired
samples, and failed or incomplete arms.

The existing performance artifacts are historical v1.0.0 evidence:

- [`docs/current_24l_results.md`](current_24l_results.md) contains the archived
  24-layer Full report and its named H100/B200 measurements.
- [`docs/compiled_step_results.md`](compiled_step_results.md) contains the
  archived 8-layer complete-step screen and per-cell reports.

Those reports remain linked for provenance. Their numbers do not qualify this
candidate, and no new timing number should be inferred from this page.
