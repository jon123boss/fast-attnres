# CUDA BF16 validation protocol

This protocol covers the CUDA BF16 operator. Measured qualification results
belong to the exact source and runtime recorded in each report.

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

The independent PyTorch/autograd reference uses BF16 inputs, outputs, and
input gradients, with FP32 internal accumulation matching the kernel's
precision contract. It disables ambient autocast so its arithmetic is the same
inside and outside compiled training checks. It rejects other input dtypes and
is validation machinery, not a second public operator.

The calculation preserves the normalize-then-dot order of the training
reference on main (`ce0881dc`), with explicit BF16 boundaries for this release.

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
PYTHONDONTWRITEBYTECODE=1 python -c \
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
- [`results/adoption/compiled_step_screen/results.md`](../results/adoption/compiled_step_screen/results.md)
  contains the archived 8-layer complete-step screen.

Those reports remain linked for provenance. Their numbers do not qualify this
candidate, and no new timing number should be inferred from this page.
