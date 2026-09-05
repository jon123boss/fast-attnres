# Provenance

This branch develops the CUDA BF16 operator in the existing draft PR. It has
not been merged or released. Measurements apply to the exact source bytes
recorded in each report, rather than to a branch name or package version.

## Source identity

[`validation/frozen.json`](validation/frozen.json) records the selected source,
validation, and packaging files. The primary experiment additionally binds
package, evaluator, optimizer, and comparator identities in
[`configs/bf16_primary.json`](configs/bf16_primary.json). The launcher verifies
those identities before GPU admission and again inside the immutable snapshot.

The shared source-list kernels adapt FLA's
[`fused.py` at 5e02dd3a](https://github.com/fla-org/flash-linear-attention/blob/5e02dd3a7651f5f2797eb8b12bbec401826031e1/fla/ops/attnres/fused.py).
Their upstream MIT attribution remains in the source header and
[`NOTICE`](NOTICE). The public API and packed kernels are project code under
[`LICENSE`](LICENSE).

Full and Block execute the same per-read operator. Sliced routing uses the
last `R` coordinates as keys while retaining width-`D` values and outputs.
The independent [BF16 oracle](validation/oracle.py) defines the tested
normalization, softmax, mixture, and first-order gradient contract.

## External comparisons

The [campaign adapters](benchmarks/bf16_competitors.py) call separately supplied
native implementations: the released Fast-AttnRes package, PyTorch compilation,
the research repository's per-read operators, FLA Triton and Gluon, Liger,
Catswe, Hydra, and Hilda. The report records exact source hashes, adapter
changes, runtime, shape restrictions, and correctness failures. A faster
incorrect result does not enter the timing comparison.

Source preparation and native gradient work remain inside the measured call.
Constant unit RMS-weight buffers contain no source-dependent data. Adapters
preserve the native arithmetic; they do not repair a comparator's gradients.

[Hilda](https://github.com/kirsten-1/hilda-kernel/tree/c0b4d8a587c5fd06e85d7c057c7224d68ddc35cf)
is pinned at `c0b4d8a587c5fd06e85d7c057c7224d68ddc35cf`. Its native wrapper
caps source count at 32, so the adapter explicitly rejects larger reads. Its
source is supplied externally and is not bundled with this repository.

No external comparator kernel source is redistributed in the package. Each
upstream checkout's own license and notices remain authoritative.

## Measurements

The [evaluation contract](EVALUATION.md) and [runbook](docs/bf16_campaign.md)
define the model, rank ladder, precision, optimizer, timing interval, confidence
intervals, and resource limits. Raw results retain failed and interrupted runs.
All published figures must identify their measured source and workload;
operator latency and complete training-step latency remain separate.

The primary model matches the 1B research geometry: 24 layers, width 1536,
batch 4, context 2048, accumulation 4, and the original Muon plus AdamW
implementation. Synthetic inputs exclude dataset I/O, logging, and scheduler
host work. These measurements do not reproduce historical training throughput.

## Historical evidence

The [v1.0.0 release](https://github.com/jon123boss/fast-attnres/tree/v1.0.0)
retains its original source and measurements. Its
[three-seed Full campaign](https://github.com/jon123boss/fast-attnres/tree/v1.0.0/results/compiled_step)
and [adoption screen](https://github.com/jon123boss/fast-attnres/tree/v1.0.0/results/adoption)
use different source bytes and workloads from this continuation. Their raw
reports and manifests remain historical evidence; their speedups are not
relabelled as results for the new kernel.
