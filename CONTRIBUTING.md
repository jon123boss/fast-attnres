# Contributing to Fast-AttnRes

A useful contribution makes the operator, its evidence, or its installation
simpler to inspect and reproduce. Keep changes focused and support performance
claims with measurements.

Read the [public contract](docs/equation.md), [evaluation contract](EVALUATION.md),
and [benchmark scope](docs/benchmark_results.md) before changing kernel or benchmark
behavior. The public operator uses CUDA BF16 storage and may use FP32
accumulators internally. Correctness checks use an independent BF16 PyTorch
reference, separate from the installed package.

## Development

1. Create a focused branch and preserve full-width values, implicit tail keys,
   parameter-free RMS normalization, and the learned static query.
2. Keep Full and Block on the same per-read operator. The caller owns source
   assembly and block sums.
3. Preserve the independent oracle and benchmark controls. Deliberate contract
   changes need an explanation, updated hashes, and new affected measurements.
4. Verify changed behavior and update documentation when the API, installation,
   or reproduction commands change.
5. Record checks that passed, failed, or could not run. Keep credentials and
   private data out of commits and shared logs.

## Local checks

Use an isolated environment and the installation instructions in the
[README](README.md#install). CPU checks do not require Triton or a GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -m "not cuda" -q
python -m py_compile src/attnres/*.py src/attnres/_kernels/*.py
ruff check src tests benchmarks validation --select E9,F63,F7,F82
```

For final CUDA qualification, run the GPU checks on one device at a time:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -m cuda -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python examples/train.py \
  --device cuda --mode full --variant standard --steps 2
```

The training example is a smoke check. Follow the [validation protocol](docs/validation.md)
for changed-input graph replay and shared or strided sources. Record H100 and
B200 results separately.

## Performance evidence

Check each workload's outputs and first-order gradients before timing it.
The BF16 gate uses `rtol=0.05` and `atol=0.05`. Keep all failed, ineligible,
incomplete, and inconclusive results in the evidence record.

Every claim needs exact source and evaluator identities, hardware and runtime,
workload and layout, timing boundaries, warmups and rounds, correctness results,
and raw timing samples. Operator latency and complete training-step latency
are separate measurements. Historical results keep their original identities.

Reuse the existing benchmark harness and report tools. Keep workload geometry,
timing boundaries, comparison eligibility, and statistical methods explicit.
The current refresh covers the existing headline and competitor plot workloads
at `R=D` and `R=D/4`; see [benchmark results](docs/benchmark_results.md).
A passing correctness check alone does not establish a speedup.

## Pull requests

Explain the problem, resulting behavior, and validation. Identify changes to
numerical semantics, the public API, layouts, installation, or the evaluator.
Link the exact evidence supporting any speed claim and state its limits.

Contributions use the repository's [MIT License](LICENSE). Please follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
