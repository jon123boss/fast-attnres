# Benchmark results

The H100/B200 refresh uses the existing README headline and competitor plot
workloads with standard (`R=D`) and sliced (`R=D/4`) routing. Fresh results
are pending; archived measurements retain their original source identities.

| Plot | Existing workload and harness |
| --- | --- |
| Headline | L24 / D1024 / H16 / FFN2816 / B2 / T1024 / vocabulary 32768; [compiled-step campaign](../benchmarks/compiled_step_campaign.py) and [hero renderer](../benchmarks/plot_compiled_step_hero.py). |
| Competitor plots | L8 / B2 / T512 / vocabulary 8192: D1024 Full, D1536 Block with event sizes 8 or 2, and D2048 Block with event size 2; [compiled-step sweep](../scripts/compiled_step_sweep.py) and [sweep renderer](../benchmarks/plot_compiled_step_sweep.py). |

Both workload groups use `R=D` and `R=D/4`. Block event size counts Transformer
sublayer events per block. The final run configuration is
[`configs/bf16_final_sweep.json`](../configs/bf16_final_sweep.json); the earlier `bf16_primary_v3.json` matrix is historical. The harness links above identify the existing measurement paths;
they do not imply that the archived configurations cover the new rank scope.

The headline uses three seeds (20260827, 20260903, 20260911), 120 paired
rounds, and 10 warmup rounds. Each smaller workload uses seed 20260827,
40 paired rounds, and 5 warmup rounds. Timed arms share model weights,
logical inputs, and the rotating execution order.

The runtime is Python 3.11.13, PyTorch 2.13.0+cu130, and Triton 3.7.1.
The timing boundary includes forward, backward, and fused capturable AdamW;
compilation, input copies, qualification, and CUDA Graph capture are excluded.

The reference accepts BF16 inputs, accumulates internally in FP32, normalizes
keys before the query dot product, and returns BF16 outputs. The same
`rtol=0.05, atol=0.05` applies to outputs and first-order gradients.

Each report must identify its rank, eligible comparators, BF16 correctness result, timing
boundary, paired samples, and exact source/runtime. Unsupported, failed, and
incomplete comparisons stay visible. A comparison of sliced `R=D/4` routing
with a standard-only `R=D` backend must identify the different equations.
H100 and B200 results remain separate.

## Historical measurements

- [24-layer Full results](current_24l_results.md): the previous headline and
  its exact reports in [results/current_24l](../results/current_24l/README.md).
- [Competitor screen](../results/adoption/compiled_step_screen/results.md):
  eight archived H100/B200 reports, with a [CSV](../results/adoption/compiled_step_screen/results.csv)
  and [manifest](../results/adoption/compiled_step_screen/manifest.json).
- [Earlier compiled-step results](compiled_step_results.md): the six-report
  Full campaign preserved in the historical packaging evidence bundle.

The existing plot assets remain unchanged until fresh reports have been
checked and rendered through the corresponding report tools.
