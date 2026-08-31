# Benchmark results

The README's current performance evidence is the BF16 compiled
complete-training-step screen in
[`results/adoption/compiled_step_screen`](../results/adoption/compiled_step_screen/).
It contains eight independently audited H100/B200 reports, the exact
[long-form table](../results/adoption/compiled_step_screen/results.md), a
[machine-readable CSV](../results/adoption/compiled_step_screen/results.csv),
and a hash-bound [manifest](../results/adoption/compiled_step_screen/manifest.json).
The screen uses PyTorch 2.13.0+cu130 and Triton 3.7.1 and is the sole source for
the bars and values at the top of the README.

The older three-seed Full campaign in
[`results/compiled_step`](../results/compiled_step/) remains an immutable
historical replication and packaging-evidence bundle. It does not feed the
current README chart or table.
