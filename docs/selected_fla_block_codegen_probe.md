# Scalar compact FLA-derived Block selected-config probe

benchmarks/selected_fla_block_codegen_probe.py is a historical, read-only,
non-timing diagnostic for the frozen scalar source-tile compact-prefix candidate.

It is pinned to:

- implementation commit 25a85a9b99985ac90d69ce636d6b42b5f636a129;
- implementation tree 74f0b86eac24c2ff85ad01d7a77039dcaf84044c;
- src/attnres/_kernels/fla_full_sources.py SHA-256
  2cd7ac89b15faeb13640bff4a7948e437453b69446bfc8c7922511e341843e10.

The six cases are D=1024, R in {128,512,1024}, S in {2,9},
with source pointer-table widths L2=8 for S=2 and L2=16 for S=9.
Each GPU subprocess uses a fresh Triton cache directory. It observes the
exact dtype-bearing Triton 3.6 autotuner key for both
_fla_source_forward_kernel and _fla_source_backward_kernel, records
the selected Config and CompiledKernel hash/resource metadata, including
backward LAYOUT_FAMILY 0/1/2 and the selected BL/warp/stage values, warms
the route, captures and replays it in a CUDA Graph, and then hashes the
post-capture TTIR, TTGIR, LLIR, and PTX artifacts. Artifact analyses report
counts and flags for layout conversion, shared/local memory, barriers,
shuffles, local loads, and global loads. R=128 is the diagnostic compact
case; R=512 is the main compact production case; R=1024 is the resident
R=D control.

The probe does not import the evaluator, alter timing inputs, wrap
do_bench, collect timing samples, or select a configuration. It refuses
GPU work unless --allow-gpu is supplied.

Example (GPU only):

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
      /opt/anaconda3/bin/python -m benchmarks.selected_fla_block_codegen_probe \
      --source-count 2 --rank 512 --cache-dir /tmp/attnres-codegen-S2-R512 \
      --allow-gpu --hardware H100 --output /tmp/attnres-codegen-S2-R512.json

Run the whole six-case matrix from Python with run_probe(cache_root=...).
This diagnostic has no local GPU result in the candidate branch; CPU/static
coverage is provided by tests/test_selected_fla_block_codegen_probe.py.

## Modal entrypoint

`benchmarks/modal_fla_block_codegen_probe.py` is a separate, non-timing Modal transport
for the complete six-case matrix. It reuses `benchmarks.modal_runner`'s pinned
image, source fingerprint, architecture/runtime cache namespace, and optional
cache Volume, but imports only this probe on the remote GPU. The GPU selector
and probe hardware scope are both required and must agree. `--cache` must name
a new or empty directory; relative names are placed below the namespaced Triton
cache. `--output` is the single local JSON report path.

H100/SM90:

```bash
modal run benchmarks/modal_fla_block_codegen_probe.py \
  --gpu 'H100!' --hardware H100 --cache codegen-h100-run-01 \
  --output /absolute/path/codegen-h100-run-01.json
```

B200/SM100:

```bash
modal run benchmarks/modal_fla_block_codegen_probe.py \
  --gpu B200 --hardware B200 --cache codegen-b200-run-01 \
  --output /absolute/path/codegen-b200-run-01.json
```

Set `ATTNRES_CACHE_VOLUME` before `modal run` when a persistent Modal Volume
is desired; the transport keeps H100 and B200 plus source/runtime versions in
separate namespaces. The launcher does not import the evaluator, alter timing
inputs, benchmark candidates, or select a configuration. It writes a failed
report and exits nonzero if compilation, artifact capture, or any case fails.
