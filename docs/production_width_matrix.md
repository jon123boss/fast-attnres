# BF16 production-width matched matrix

I froze the ten primary cells in
[`configs/matched_competitor_benchmark_d_gt_768.json`](../configs/matched_competitor_benchmark_d_gt_768.json)
before collecting results. This page explains what I mean by each dimension,
why the cells are shaped this way, and which comparisons can honestly be made.
I want someone reading a report later to reconstruct why each row exists
without guessing. It is a measurement plan, not a results page: I am not
claiming a latency, speedup, ranking, or adoption result for any cell here.
No operator latency table from this plan is published in the current release;
the public performance claim uses the separately audited complete compiled
training-step campaign.

## The four dimensions

The operator receives an ordered list of `S` source tensors, each shaped
`[N, D]`, and a query shaped `[R]`. The output is `[N, D]`.

| Symbol | Meaning in this matrix |
| --- | --- |
| `S` | Source depth: the number of residual sources reduced by one operator call. |
| `N` | Flattened rows per source: normally `microbatch × sequence length` (`B × T`) for a training call. |
| `D` | Full value width: the feature count in every source row and output row. |
| `R` | Routing/query width: the query length and the width of the implicit key tail. |

Every primary row is BF16 and uses `R = D`. The top-level config also records
FP32 because some adapters and the equation oracle support it; that does not
create an FP32 production timing lane. In this profile, a cell's logical value
payload is `[S, N, D]`, while its routing query is `[R]`.

## The ten primary cells

The table below is a transcription of `operator_cases.primary`. It has ten
rows, all with `dtype = "bf16"`.

| Cell | `S` | `N` | `D` | `R` | `N × D` per source | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `D1024_S3` | 3 | 4096 | 1024 | 1024 | 4,194,304 | Matched width/depth pair |
| `D1024_S9` | 9 | 4096 | 1024 | 1024 | 4,194,304 | Matched width/depth pair |
| `D2048_S3` | 3 | 2048 | 2048 | 2048 | 4,194,304 | Matched width/depth pair |
| `D2048_S9` | 9 | 2048 | 2048 | 2048 | 4,194,304 | Matched width/depth pair |
| `D4096_S3` | 3 | 1024 | 4096 | 4096 | 4,194,304 | Matched width/depth pair |
| `D4096_S9` | 9 | 1024 | 4096 | 4096 | 4,194,304 | Matched width/depth pair |
| `D4096_S33` | 33 | 1024 | 4096 | 4096 | 4,194,304 | Longer source-depth cell |
| `D7168_S9` | 9 | 512 | 7168 | 7168 | 3,670,016 | Non-power-of-two generalization |
| `D8192_S3` | 3 | 512 | 8192 | 8192 | 4,194,304 | Matched width/depth pair |
| `D8192_S9` | 9 | 512 | 8192 | 8192 | 4,194,304 | Matched width/depth pair |

### Why hold `N × D` constant?

For `D = 1024, 2048, 4096,` and `8192`, I choose `N` so that

```text
N × D = 4,194,304 = 2²² elements per source.
```

That keeps the payload of one source near 4.2 million elements while the
feature width changes. It does not keep the whole cell constant: changing
`S` from 3 to 9 triples the number of source elements, and `S = 33` is an
explicit deeper reduction. The constant tier makes the width pairs easier to
read; it is not a claim that all kernels perform the same amount of work or
that these cells identify a universal scaling law.

`D = 7168` is intentionally not rounded to `8192` in the matrix. Its declared
shape is `S = 9, N = 512, D = R = 7168`, which is `3,670,016` elements per
source (about 3.67 million). I keep this real non-power-of-two width visible
as a generalization check. Any implementation-specific tile padding is an
implementation detail; it does not turn the input into a `D = 8192` cell, and
the one `D7168` result must not be presented as a constant-payload comparison
with the `D8192` pair.

### Why `S = 33`?

`D4096_S33` keeps `N`, `D`, `R`, and the per-source payload equal to the
`D4096` pair, then raises the source depth to 33. It tests a longer source
reduction at a width already represented by `S = 3` and `S = 9`; I will not
interpolate a result for 33 sources from those shorter cases. It is a primary
cell, not a license to move an unsupported comparator into the denominator.

## Timing and pairing contract

I run the primary cells independently on each named device: `H100!` (SM90)
and `B200` (SM100). The device reports stay separate. I do not pool GPUs or
pair an H100 sample with a B200 sample.

For each GPU, cell, and fixed protocol seed, I use the three sealed seeds
`20260827`, `20260903`, and `20260911`, then run 10 warmup rounds that are
excluded from timing and 120 measured rounds. The timed boundary is operator
forward plus backward (`F+B`), including adapter-owned source stacking and
contiguous preparation; it is not a complete model training step.

The arm order is balanced in an ABBA pattern. I choose one deterministic arm
permutation for a cell, use it on even rounds, and use its exact reverse on
odd rounds. Candidate and comparator see the same device, initial state,
ordered source list, inputs, and upstream gradient for each pair. This keeps
the comparison paired while exposing order effects rather than silently
averaging them away.

Before timing, the candidate and any comparator must pass the independent FP32
equation oracle for the output, every source-value gradient, and the query
gradient. A missing route, failed gate, failed round, or incomplete 120-round
cell remains visible in the raw report and cannot be promoted by dropping,
interpolating, or retrying the unchanged failure.

## Comparator boundaries

The primary family names native FLA Triton checkpoint 1 and conditional FLA
Gluon. All external routes in this config require `R = D`; none is a valid
external comparator for sliced `R < D` LR-AttnRes. The candidate's primary
operator cells already use `R = D`, so that architectural distinction is kept
out of this headline.

The route-specific limits still matter for individual rows:

| Route | Stipulation for this matrix |
| --- | --- |
| FLA Triton checkpoint 1 | Standard `R = D`, `S ≤ 129`, `D ≤ 8192`; eligible only after the same oracle and complete timed-row gate. |
| FLA Gluon checkpoint 1 | Conditional standard `R = D`; it also requires `BD = next_power_of_two(D) ≤ 4096`, `S × BD ≤ 262,144`, and the pinned `33 × S × BD` static-work bound. Therefore `D7168` and `D8192` are outside this Gluon envelope because `BD = 8192`. No Gluon promotion occurs without independent H100 and B200 gates. |
| Liger | Conditional `R = D` route with `S ≤ 32` for the relevant Full/Block cells; it cannot cover the `S = 33` cell. |
| Catswe phase 1 | BF16 standard operator only, `R = D`, power-of-two `D`, and `nextpow2(S) × D ≤ 1,048,576`; it has no accepted Full, Block, or model route here, and `D7168` is non-applicable. |
| Hydra 2P | `R = D` standard/Block panel, but native timing is limited to `D ≤ 256`; it is not a comparator for this production-width primary set. |
| FLA Triton checkpoint 0 | Diagnostic only; it is never an eligible denominator even when a shape is otherwise applicable. |

These stipulations describe capability, not results. An ineligible smoke or
primary row remains an audit row; it is never silently substituted with a
different width, schedule, or vendor route.

## Headline scope and smoke scope

The production-width headline is the sealed `primary` scope, whose ten cells
all satisfy `D > 768`, use BF16, and use `R = D`. The `D > 768` wording is a
headline boundary, not a blanket promotion rule: a row that happens to have
`D > 768` but belongs to `smoke` remains diagnostic.

The smaller smoke cases in the same config are there to exercise routing,
shape, and adapter behavior quickly. They are diagnostic only and are
excluded from adoption claims and the primary performance denominator. I will
not combine their timings with these ten cells, use them to fill a missing
primary result, or present them as evidence for a production-width ranking.

For the launch mechanics, capability plan, raw-row schema, and statistical
denominator, see the [matched competitor protocol](matched_competitor_protocol.md).
