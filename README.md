# Fast Attention Residuals

[![CI](https://github.com/jon123boss/fast-attnres/actions/workflows/ci.yml/badge.svg)](https://github.com/jon123boss/fast-attnres/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Tested PyTorch 2.13](https://img.shields.io/badge/tested-PyTorch_2.13-EE4C2C.svg)](https://pytorch.org/)
[![Tested Triton 3.7.1](https://img.shields.io/badge/tested-Triton_3.7.1-654FF0.svg)](https://github.com/triton-lang/triton/releases/tag/v3.7.1)
[![License: MIT](https://img.shields.io/badge/license-MIT-2E7D32.svg)](https://github.com/jon123boss/fast-attnres/blob/main/LICENSE)

[![Current large-model Full AttnRes complete-training-step comparison](https://raw.githubusercontent.com/jon123boss/fast-attnres/main/docs/assets/compiled_step_hero.png)](https://github.com/jon123boss/fast-attnres/blob/main/docs/assets/compiled_step_hero.png)

**Fast Attention Residuals** (`Fast-AttnRes`) makes [Attention Residuals](https://arxiv.org/abs/2603.15031) a small, ordinary PyTorch operation: pass an ordered set of full-width residual sources and one learned query, get one full-width residual back. The same `attnres(values, query)` call handles standard AttnRes in Full and per-read Block schedules with either packed tensors or ordered source lists.

Start with the [standard quickstart](#quickstart-standard-attnres), then choose a
[Full or Block schedule](#full-and-block-schedules), and try [sliced
LR-AttnRes](#sliced-lr-attnres) when you want a smaller routing query.

## Why use Fast-AttnRes

- **One PyTorch call:** `attnres(values, query)` returns a full-width residual
  and participates in ordinary first-order autograd.
- **One schedule primitive:** Full and per-read Block call the same public
  operator; the caller controls source order, block sums, and learned queries.
- **BF16-ready training path:** the optimized target is BF16 storage/autocast in
  compiled CUDA training graphs, with FP32 equation math.

## Large-model Full AttnRes: the advantage expands with scale

The current release was verified through Modal on a 24-layer model, `batch 2 x
sequence 1024`, and a Full ordered source-list schedule with 48 AttnRes reads
growing from `S=2` to `S=49`. Across three separately analyzed seeds,
Fast-AttnRes measured **5.20% lower complete-step latency on H100 SXM** and
**15.52% lower on B200** than pinned native FLA Triton checkpoint 1. Each seed
used 10 warmups and 120 paired CUDA Graph rounds.

Each captured step contains optimizer zeroing, model forward, cross-entropy, backward,
gradient accumulation, and fused capturable AdamW. Input copies, hashing,
compilation, qualification, warmup, graph capture, and report serialization are
outside the timed CUDA events.

The measured advantage is materially wider on this larger Full workload than
on the smaller eight-layer coverage screen below. That is the observed scaling
result for these two named configurations; their samples are never pooled
because model depth, token count, source schedule, and campaign size differ.

| Evidence | Configuration | H100 vs FLA | B200 vs FLA |
| --- | --- | ---: | ---: |
| Small adoption screen | 8 layers, `B2 x T512`, Full `S=2..17`, 1 seed x 40 rounds | **1.12% lower** | **1.74% lower** |
| Current large-model Full | 24 layers, `B2 x T1024`, Full `S=2..49`, 3 seeds x 120 rounds | **5.20% lower** | **15.52% lower** |

See the [per-seed ratios, confidence intervals, and complete configuration](https://github.com/jon123boss/fast-attnres/blob/main/docs/current_24l_results.md)
and the [exact current raw reports](https://github.com/jon123boss/fast-attnres/blob/main/results/current_24l/README.md). The
[earlier campaign](https://github.com/jon123boss/fast-attnres/blob/main/docs/compiled_step_results.md) remains archived against its
own historical source identity and is not used by this headline.

## Compiled BF16 complete-training-step screen

[![H100 BF16 compiled complete-training-step adoption screen](https://raw.githubusercontent.com/jon123boss/fast-attnres/main/docs/assets/compiled_step_screen_h100.png)](https://github.com/jon123boss/fast-attnres/blob/main/docs/assets/compiled_step_screen_h100.png)

[![B200 BF16 compiled complete-training-step adoption screen](https://raw.githubusercontent.com/jon123boss/fast-attnres/main/docs/assets/compiled_step_screen_b200.png)](https://github.com/jon123boss/fast-attnres/blob/main/docs/assets/compiled_step_screen_b200.png)

This screen times the work people actually train with: a captured, compiled
BF16 step containing optimizer zeroing, model forward, cross-entropy, backward,
gradient accumulation, and fused AdamW. Input copies, compilation,
qualification, warmup, graph capture, hashing, and report serialization are
outside the CUDA-event interval.

Each numeric comparator arm below is a completed, independently audited row
from one predeclared seed, with 5 warmups, 40 paired rounds, and a 20,000-resample simultaneous
paired bootstrap. The ratio is `Fast-AttnRes / comparator`; lower is faster.
The 95% intervals are simultaneous within each cell's comparator family. GPUs,
shapes, and seeds are not pooled. This is an adoption screen, not a universal
fastest claim.

### Standard AttnRes (`R=D`, same equation)

Values are mean complete-step latency in milliseconds; the percentage in each
comparator column is the Fast-AttnRes advantage.

| GPU | Schedule | `D` | Read sources | Fast-AttnRes | FLA ckpt-1 | Liger 0.8.2 | Catswe phase 1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| H100 SXM | Full | 1024 | `S=2..17` | **5.686 ms** | 5.750 ms (**+1.12%**) | 6.511 ms (**+12.67%**) | 6.452 ms (**+11.88%**) |
| H100 SXM | Block, event block size 2 | 1536 | `S=2..9` | **9.455 ms** | 9.483 ms (**+0.30%**) | 10.236 ms (**+7.63%**) | NA: `D` is not a power of two |
| B200 | Full | 1024 | `S=2..17` | **3.683 ms** | 3.749 ms (**+1.74%**) | 4.366 ms (**+15.64%**) | 4.229 ms (**+12.91%**) |
| B200 | Block, event block size 8 | 1536 | `S=2..3` | 5.304 ms | **5.235 ms (-1.32%)** | 5.677 ms (**+6.57%**) | NA: `D` is not a power of two |
| B200 | Block, event block size 2 | 1536 | `S=2..9` | **5.430 ms** | 5.456 ms (**+0.48%**) | 6.101 ms (**+11.00%**) | NA: `D` is not a power of two |
| B200 | Block, event block size 2 | 2048 | `S=2..9` | **7.844 ms** | 7.844 ms (parity) | FAIL: strict numerical gate | 8.298 ms (**+5.48%**) |

### Sliced LR-AttnRes (`R=D/4`, architectural comparison)

| GPU | Schedule | `D / R` | Read sources | LR Fast-AttnRes | standard FLA `R=D` |
| --- | --- | ---: | --- | ---: | ---: |
| H100 SXM | Block, event block size 2 | `1536 / 384` | `S=2..9` | **9.655 ms** | 9.681 ms (**+0.26%**) |
| B200 | Block, event block size 2 | `1536 / 384` | `S=2..9` | **5.417 ms** | 5.452 ms (**+0.66%**) |

All displayed models use 8 layers, BF16 autocast, and `N=1024` flattened token
rows per source (`batch 2 x sequence 512`) on PyTorch 2.13.0+cu130 and Triton
3.7.1. `S` is the number of residual sources visible to each AttnRes read. Full
exposes the growing `2..17` history; Block source count is controlled by the
event block size, so event block size 2 produces two reads at each count
`2..9`.

Only arms that pass native discovery, independent output/value-gradient/query-
gradient qualification, fullgraph compilation, changed-input CUDA Graph checks,
and all paired samples receive a ratio. Unsupported, failed, and missing arms
stay visible without a number. See the [exact long-form table](https://github.com/jon123boss/fast-attnres/blob/main/results/adoption/compiled_step_screen/results.md),
[machine-readable CSV](https://github.com/jon123boss/fast-attnres/blob/main/results/adoption/compiled_step_screen/results.csv), and
[evidence manifest](https://github.com/jon123boss/fast-attnres/blob/main/results/adoption/compiled_step_screen/manifest.json) for the
unrounded paired ratios and simultaneous 95% intervals.

## Product target and dtype roles

I treat BF16 training as the optimized product target. CUDA implementation
and qualification work prioritize BF16 storage/autocast in complete training
graphs, under the named hardware, runtime, and shape gates. The existing
performance evidence is a BF16 training measurement, and the bounded adoption
matrix is the current qualification surface for that target.

FP32 remains supported for the explicit equation/reference path, correctness
oracles, and debugging. FP32 support is not an FP32 performance target: I do
not infer FP32 timing, ranking, or speedup from API support or from passing an
FP32 oracle check.

## Install

The distribution is named `fast-attnres` and the Python import is the concise
`attnres`. For the CPU/reference path, install the stable release from PyPI:

```bash
python -m pip install "fast-attnres==1.0.0"
```

For the qualified Linux x86-64 CUDA profile, install PyTorch's CUDA 13.0 wheel
first, then the package's exact CUDA dependency set:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.13.0
python -m pip install "fast-attnres[cuda]==1.0.0"
```

The CPU/reference path only needs PyTorch and does not import Triton. Contributors
can instead install a checkout with development and test dependencies:

```bash
git clone https://github.com/jon123boss/fast-attnres.git
cd fast-attnres
python -m pip install -e ".[cuda,dev,test]"
```

This installs the repository-qualified CUDA profile: PyTorch 2.13.0+cu130 with the Triton 3.7.1 dependency used by that release. The compiled-step screen above was measured on this exact pair.

The wheel contains ordinary Python and Triton source. There is no compiled
CUDA extension in the distribution: Triton JIT-compiles the selected kernel
for the local GPU at runtime. Confirm an installation without importing
Triton or requiring a GPU:

```bash
fast-attnres-info
# or: python -m attnres.info
```

**Why Triton 3.7.1 rather than the standalone 3.8.0 release?** At the v1.0.0
release cutoff (2026-08-31), PyTorch 2.13.0 was the newest stable PyTorch
release and its Linux wheel declared
`triton==3.7.1`. Triton 3.8.0 is newer in isolation, but installing it beside
stable PyTorch 2.13.0 creates a dependency conflict. Fast-AttnRes therefore
uses that cutoff's newest mutually compatible stable PyTorch/Triton pair instead of
advertising an unqualified combination.

## Quickstart: standard AttnRes

Standard AttnRes uses a full-width query (`R == D`). The source axis is reduced
with a softmax, and the output retains the full value width:

```python
import torch

from attnres import attnres

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
D = 1024

sources = tuple(
    torch.randn(2, D, device=device, dtype=dtype, requires_grad=True)
    for _ in range(8)
)
query = torch.nn.Parameter(torch.randn(D, device=device, dtype=torch.float32))

output = attnres(sources, query)  # [2, D]
loss = output.float().square().mean()
loss.backward()
```

Verify the installed package directly, without relying on repository-only examples:

```bash
fast-attnres-info
python -c "import torch; from attnres import attnres; print(attnres(torch.randn(3,2,8), torch.randn(4)).shape)"
```

## API

`values` can also be one packed tensor with shape `[S, ..., D]`. The public
signature is:

```python
attnres(values, query, *, eps=2**-23, scale=1.0)
```

For source `s`, the operator takes the final `R` value coordinates as its
implicit key and computes, in FP32,

```text
t_s      = v_s[..., D-R:D]
k_s      = t_s / sqrt(mean(t_s ** 2) + eps)
p_s      = softmax(scale * dot(k_s, query), over sources)
output   = sum_s p_s * v_s
```

There is no output normalization or source-count prior. BF16 and FP32 values
and queries are supported; the result is returned in the values' storage dtype.
BF16 is the optimized training storage target. FP32 storage remains available
for equation checks and debugging, not as a separate performance target.

## Full and Block schedules

Full AttnRes gives each read all preceding residual sources. In the sequential
Block schedule, the caller sums each completed block and passes that block
output, plus the current partial block output when one exists, as ordinary
sources to the same public operator:

```python
completed = (embedding, first_block, second_block)
partial = current_partial
read_sources = completed if partial is None else completed + (partial,)
block_output = attnres(read_sources, query)
```

Full and per-read Block therefore share one `attnres` primitive. The caller
owns block boundaries, partial sums, and learned queries. Block values are
sums, not averages. There is no separate Block kernel, prepared-state API, or
cached Block API in the public release surface.

### Multi-read BF16 diagnostics

A multi-read Full or Block model can compose an intermediate source before a
later read. When that intermediate is stored in BF16, the runtime's sequence
of partial sums can round at different points or associate additions in a
different order from a flattened equation reference. These paths are
mathematically equivalent over real numbers, but BF16 storage can expose a
small strict elementwise difference.

I treat such a difference as a composition-order diagnostic, not a relaxed
correctness criterion. A strict elementwise mismatch remains a test failure:
the protocol does not change tolerances, delete the case, or convert the row
into a passing qualification result.

[`examples/block_schedules.py`](https://github.com/jon123boss/fast-attnres/blob/main/examples/block_schedules.py) runs the Full
reduction and the same per-read schedule on small CPU or CUDA inputs:

```bash
python examples/block_schedules.py
```

## Sliced LR-AttnRes

Once standard AttnRes is working, choose a shorter query (`R < D`) to use
sliced LR-AttnRes, the extension developed in the
[Low-Rank Attention Residuals paper](https://arxiv.org/abs/2607.09694). The
big idea is to keep the residual stream at full width while learning the
source-selection query in a smaller routing space. Its implicit key is the
final `R` coordinates of each full-width value, so routing shrinks while the
output remains full width:

```python
rank = 64
query = torch.nn.Parameter(torch.randn(rank, device=device, dtype=torch.float32))
output = attnres(sources, query)  # still [2, D]
```

The [`examples/lr_attnres.py`](https://github.com/jon123boss/fast-attnres/blob/main/examples/lr_attnres.py) script uses the
`LearnedQuery` helper and checks the full-width output:

```bash
python examples/lr_attnres.py
```

`R == D` is standard AttnRes; `R < D` is sliced LR-AttnRes. The small
`attnres.modules.LearnedQuery(rank)` module is available when a trainable
static query parameter is convenient. Explicit projected-key and carrier APIs
are outside this release.

## Support and API boundaries

| Area | Supported contract |
| --- | --- |
| Sources | Packed `[S, ..., D]` tensor or nonempty ordered `list`/`tuple` of `[..., D]` tensors |
| Dimensions | `1 <= S <= 129`, `1 <= D <= 8192`, `1 <= R <= D` |
| Storage | BF16 or FP32 values and queries; FP32 equation math; output in the values' dtype |
| Devices | CPU equation reference; CUDA Triton kernels |
| CUDA source lists | Bounded BF16 lists (`D <= 2048`) use the FLA-derived source-list route; FP32 and wider BF16 lists use the fixed-tail fallback |
| Autograd | Ordinary first-order gradients for values and queries, including repeated sources and shared views |

These dimensions and dtypes describe the equation and API domain. CUDA
qualification is narrower: only the bounded adoption matrix has passed under
PyTorch 2.13/Triton 3.7.1, so the entire `S <= 129`, `D <= 8192` API envelope
must not be treated as CUDA-qualified.

The [`examples/packed_and_list.py`](https://github.com/jon123boss/fast-attnres/blob/main/examples/packed_and_list.py) script compares
packed tensors with ordered lists and tuples for both standard and sliced
calls:

```bash
python examples/packed_and_list.py
```

CUDA source-list calls keep individual producer tensors when their layout is
usable. A non-affine producer or incoming gradient may receive its own
contiguous copy, so source lists do not promise universal zero-copy behavior.
Second-order behavior is not part of the public contract.

The package exports `attnres` for the runtime route, `reference_attnres` for
the explicit equation reference, `LearnedQuery`, and `__version__`.

For a compact equation and detailed layout/gradient rules, see the
[equation and API contract](https://github.com/jon123boss/fast-attnres/blob/main/docs/equation.md). For support, please open a
[GitHub issue](https://github.com/jon123boss/fast-attnres/issues) with your
OS, GPU, PyTorch/Triton/CUDA versions, source container, shapes, dtypes, and a
minimal reproducer.

## PyTorch integration

The public operator is an ordinary PyTorch call, so it can be wrapped by
`torch.compile` or checkpointing:

```python
import torch

from attnres import attnres
from torch.utils.checkpoint import checkpoint

sources = tuple(
    torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for _ in range(4)
)
query = torch.nn.Parameter(torch.randn(1024, device="cuda", dtype=torch.float32))

def residual(values, query):
    return attnres(values, query)

compiled_residual = torch.compile(residual)
compiled_output = compiled_residual(sources, query)
checkpointed = checkpoint(residual, sources, query, use_reentrant=False)
```

The [`examples/backward.py`](https://github.com/jon123boss/fast-attnres/blob/main/examples/backward.py) script checks first-order
value and query gradients. The [`examples/torch_compile.py`](https://github.com/jon123boss/fast-attnres/blob/main/examples/torch_compile.py)
script compares eager and compiled outputs; it uses the lightweight `eager`
compile backend by default and accepts `--backend inductor` for an optimized
backend smoke test:

```bash
python examples/backward.py
python examples/torch_compile.py
python examples/torch_compile.py --backend inductor
```

## Competitor eligibility and interpretation

The measured rows above are complete training-step comparisons at the stated
shape. The same public AttnRes schedule, model state, inputs, optimizer, warmup,
and paired order are used for every eligible arm. Adapter staging is not hidden
outside the captured step.

| Route | What the comparison means | Stipulations |
| --- | --- | --- |
| [FLA Triton checkpoint 1](https://github.com/fla-org/flash-linear-attention) | Same-equation standard AttnRes | Native source-list route; `R=D`; checkpoint 1 only. |
| [Liger-Kernel 0.8.2](https://github.com/linkedin/Liger-Kernel) | Same-equation standard AttnRes | `R=D`, every read `S<=32`; required `torch.stack(...).contiguous()` is inside the timed captured step. |
| [Catswe phase 1](https://github.com/catswe/flash-attention-residuals) | Same-equation standard AttnRes where eligible | BF16 values, `R=D`, power-of-two `D`, and `nextpow2(S)*D <= 2^20`; stack/contiguous is timed. Phase 1 only: no phase-2, prepare/merge, or cached Block path. |
| Hydra 2P | Not applicable to this width screen | Native timing envelope is `D<=256`, so Hydra is `not_applicable` for every displayed `D>768` row. Its external panel is not presented as the public per-read Block equation. |
| Sliced LR-AttnRes (`R<D`) vs FLA | Architectural comparison, when shown | LR-AttnRes changes the routing equation while retaining full-width values/output; FLA remains standard `R=D`. It is never labelled a same-equation kernel speedup. |

The public API envelope is broader than this measured screen. FP32 remains an
equation/reference and debugging path, but no FP32 performance claim is made.
See the [matched competitor protocol](https://github.com/jon123boss/fast-attnres/blob/main/docs/matched_competitor_protocol.md) for
capability gates, adapter costs, provenance, licenses, and explicit
`not_applicable` handling.

## Verify locally

Run the CPU and static checks from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -m "not cuda" -q
```

On a configured CUDA device, run the GPU-marked checks separately:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -m cuda -q
```

The headline 24-layer campaign is independently auditable on CPU from the
checked-in raw reports and exact measured-source archive:

```bash
python -m benchmarks.audit_current_24l \
  --evidence-dir results/current_24l --repo .
```

To regenerate the hero after installing `.[plot]`:

```bash
python -m benchmarks.plot_compiled_step_hero \
  --projection results/current_24l/hero_projection.json \
  --output-dir docs/assets
```

The small training harness exercises standard/sliced and Full/Block routes:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python examples/train.py --backend reference --device cpu \
  --mode full --variant standard --steps 2
```

Use `--backend kernel --device cuda` on a configured CUDA environment. The
example trains on random next-token data and is a smoke test, not a benchmark.

## Citation and license

Please cite [Attention Residuals](https://arxiv.org/abs/2603.15031) for the
base method and [Low-Rank Attention Residuals](https://arxiv.org/abs/2607.09694)
for the low-rank extension. For the implementation itself:

```bibtex
@misc{su2026attnreskernels,
  author  = {Jonathan Su},
  title   = {Fast Attention Residuals},
  year    = {2026},
  url     = {https://github.com/jon123boss/fast-attnres}
}
```

The package is released under the [MIT License](https://github.com/jon123boss/fast-attnres/blob/main/LICENSE). The FLA-derived
source-list kernel attribution and license notice are recorded in [NOTICE](https://github.com/jon123boss/fast-attnres/blob/main/NOTICE).
