# Provenance

Version 2.0.1 contains the CUDA BF16 operator merged in PR #1. Measurements
apply to the exact source bytes recorded in each report. The release auditor
compares every runtime source file with the final measured snapshot, allowing
only the `__version__` literal to change from 1.0.0 to 2.0.1. Historical source
manifests are preserved; a release version is not a new GPU measurement.

## Source identity

[`validation/frozen.json`](validation/frozen.json) records the selected source,
validation, and packaging files. The final sweep binds the package and evaluator
in [`configs/bf16_final_sweep.json`](configs/bf16_final_sweep.json). Each worker
verifies its complete source snapshot before qualification. The headline and
smaller-workload phases retain their own exact source identities; the latter
corrects Liger's constant dtype and removes two unused private helpers.
The earlier [`configs/bf16_primary.json`](configs/bf16_primary.json) and its
versioned successors retain their historical source identities.

The shared source-list kernels adapt FLA's
[`fused.py` at 5e02dd3a](https://github.com/fla-org/flash-linear-attention/blob/5e02dd3a7651f5f2797eb8b12bbec401826031e1/fla/ops/attnres/fused.py).
Their upstream MIT attribution remains in the source header and
[`NOTICE`](NOTICE). The public API and packed kernels are project code under
[`LICENSE`](LICENSE).

Full and Block execute the same per-read operator. Sliced routing uses the
last `R` coordinates as keys while retaining width-`D` values and outputs.
The independent [BF16 oracle](validation/oracle.py) defines the tested
normalization, softmax, mixture, and first-order gradient contract. Its active
implementation normalizes keys before the query dot product, using BF16
inputs/outputs and FP32 internal accumulation with autocast disabled.

## External comparisons

The final sweep calls separately supplied native FLA Triton checkpoint 1,
Liger 0.8.2, and Catswe phase 1 implementations. Earlier
campaign adapters for additional alternatives remain in the original source
snapshots; the [historical campaign guide](docs/bf16_campaign.md) describes
restoration and reporting.
Each report records exact source hashes, adapter
changes, runtime, shape restrictions, and correctness failures. A faster
incorrect result does not enter the timing comparison.

Source preparation and native gradient work remain inside the measured call.
Constant unit RMS-weight buffers contain no source-dependent data. Adapters
preserve the native arithmetic; they do not repair a comparator's gradients.
Their custom autograd boundaries disable unused auxiliary gradients, avoiding
zero-fill work introduced by the benchmark wrapper.

[Hilda](https://github.com/kirsten-1/hilda-kernel/tree/c0b4d8a587c5fd06e85d7c057c7224d68ddc35cf)
is pinned at `c0b4d8a587c5fd06e85d7c057c7224d68ddc35cf`. Its native wrapper
caps source count at 32, so the adapter explicitly rejects larger reads. Its
source is supplied externally and is not bundled with this repository.

No external comparator kernel source is redistributed in the package. Each
upstream checkout's own license and notices remain authoritative.

## Measurements

The [final sweep contract](configs/bf16_final_sweep.json) defines five workloads,
the standard and quarter ranks, precision, seeds, and paired timing rounds.
Raw results retain failed and interrupted runs.
All published figures must identify their measured source and workload;
operator latency and complete training-step latency remain separate.

An earlier diagnostic contract used the 1B research architecture: 24 layers, width 1536,
batch 4, accumulation 4, and the original Muon plus AdamW implementation.
Context was reduced to 1024 for every arm on both GPUs to lower control memory
use. Earlier context-2048 reports retain their recorded workload. Synthetic
inputs exclude dataset I/O, logging, and scheduler host work. These measurements
do not reproduce historical training throughput. The final README refresh
instead uses the existing 24-layer headline and 8-layer competitor workloads.

## Historical evidence

The [v1.0.0 release](https://github.com/jon123boss/fast-attnres/tree/v1.0.0)
retains its original source and measurements. Its
[three-seed Full campaign](https://github.com/jon123boss/fast-attnres/tree/v1.0.0/results/compiled_step)
and [adoption screen](https://github.com/jon123boss/fast-attnres/tree/v1.0.0/results/adoption)
use different source bytes and workloads from this continuation. Their raw
reports and manifests remain historical evidence; their speedups are not
relabelled as results for the new kernel.
