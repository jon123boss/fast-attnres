# Evaluator revisions

2026-08-27: GPU checks revised from `76a04b33909a1d91c495bb510a85c3e6f31ec657f37e4ce6c7c5529ee256d74d` to `53186d885cd88d3d6b2565733bef2bca30cbc9b4d2907ff5f22b6d98fb166240`. CUDA Graph warmup and fresh static autograd leaves now use the capture side stream, as required by PyTorch 2.12 CUDA semantics. The prior default-stream warmup caused a capture-stream dependency failure after compiled parity passed. Equations, seeds, shapes, tolerances, and changed-input replay checks are unchanged. Prior reports are retained; all candidates must pass this revision. No performance evaluation had started.

Reference: https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-graphs

2026-08-27: Benchmark shape parser corrected from `[S,R,D,N]` to frozen protocol `[S,N,D,R]`, and `[S,N,D]` for standard primary cases. Runner hash changed from `cc73c24296e14e1ef3ff8e89ed7f2743c3212184d49e0e6eb47e88c34a980307` to `bb290b54c3576212721d362b8fcf2a7e5b198d6d1ec3c9f60c6c84579587f32c`. The first operator smoke report `native_fla_smoke_b200` therefore describes its recorded LR shape, not the intended standard/FLA case; it is excluded from that comparison and retained. Timing boundaries, samples, equations, and tolerances are unchanged. Added independently frozen `block_checks.py` for wide widths/ranks, direct mixture/LSE adjoints, exact repeat, and graph replay before running those gates.

2026-08-27: Corrected the FLA model adapter to remove the source axis in its returned shape. Added a separately labeled opt-in CUDA Graph operator timing mode, with changed-input independent-oracle parity and fixed replay counts. Existing eager-event operator timing and complete compiled training timing are unchanged, including all raw samples and the estimator. No graph timing results existed before this freeze. `benchmarks/run.py`: `bb290b54c3576212721d362b8fcf2a7e5b198d6d1ec3c9f60c6c84579587f32c` -> `953be0cb659cbb6f947fb3e47a2cf80058d9a8eb376d6b6b19b1a0555c4d61dc`; `benchmarks/competitors.py`: `ff33dd539a752fa5a401f18222b5fbf3f65f60183543bb3c90d04af7df6486cb` -> `8a84da32f4d83471ddd589d445024801e64c3edea5135ac510b5c48b96b9ad68`. Native FLA Gluon cannot compile under Triton3.7.1 because thread_barrier was renamed; a separately pinned Torch2.11.0/Triton3.6.0 environment will test the unmodified backend, with both sides on the same software stack.
# Additional development diagnostics

The first complete-step CUDA Graph gate incorrectly reused the old benchmark's
accumulation loop, which repeated the full batch instead of splitting it. Its
accumulation-2 comparisons are invalid. The revised independent gate uses the
frozen model's `_microbatches` helper, holds total batch fixed, checks restored
state exactly, and compares replayed weights, gradients, and AdamW buffers at
rtol=1e-5/atol=1e-6 with exact step counters. Primary timing used accumulation=1
and is unaffected. The old accumulation-2 smoke timing is not a valid test of
microbatch accumulation. Both implementations must pass the corrected gate.

`tests/test_offsets.py` adds a real allocation with source strides above signed
32-bit element offsets. Both implicit and separate-key paths use the unchanged
equation oracle and tolerances. `diagnose_fp32.py` compares a development case to
FP64 only to locate numerical error; FP64 is not a replacement promotion oracle.
Both additions are hashed before their first GPU execution.

2026-08-27: Added an explicit external frozen-baseline adapter, namespaced by source hash, and optional Modal source mount. Model Block execution now uses that adapter's prepare/merge hooks when present; existing kernel/reference/ordinary callable paths are unchanged. The external baseline therefore retains the identical cache schedule instead of silently recomputing Full reads. CPU output/all-parameter gradient parity and cache-call counts pass. No timing, oracle, tolerance, inputs, or heldout cases changed. The baseline source is retained outside the package at revision daf2226.

2026-08-27: Selected independently reviewed concise runner, corrected its warmup mode/upstream argument order before any GPU timing, and reused canonical microbatch splitting. Added separately labeled model_timing=cuda_graph using the complete-step helper that passed all 12 variant/mode/accumulation cases on both H100 and B200. Input copies and capture remain outside event intervals; each measured replay includes zero-grad, projection/source assembly, forward/loss/backward/accumulation and fused capturable AdamW. Existing event timing remains available. Every raw paired sample is retained, and seeds, bootstrap, equivalence band, and tolerances are unchanged. No heldout execution has occurred.

2026-08-27: Independent compact-runner review found operator-only regressions before that runner had timed any operator: captured gradients needed stable zeroed buffers, comparator/kernel ratio orientation and per-mode graph failure handling needed restoration, and invalid scopes needed immediate rejection. These are corrected with CPU fake regressions; the model-only graph smoke and primary runs do not execute these paths and remain valid. The event and graph operator modes require fresh GPU confirmation. Native FLA compile bridge tests passed six checks on each H100 and B200 under Torch2.11/Triton3.6; the bridge remains explicitly labeled, with source hashes, and does not modify vendor kernels.

2026-08-27: The first primary H100 graph job exhausted 79GiB during simultaneous reference/candidate qualification at rank1024, before any timed sample. Qualification now completes and releases the reference autograd graph before candidate forward, retaining detached logits and every parameter gradient for the unchanged fixed comparison. Model, total batch, ranks, seeds, loss, timings, and tolerances are unchanged. CPU regression checks execution order and full gradient comparison. The original failed job is retained.

2026-08-27: Operator input byte hashes are now computed once per fixed leaf tuple, not repeatedly copied from GPU to CPU for every sample. Operator primals are unchanged during timing; only gradient buffers change. Raw rows retain the same digest, full timing samples, paired order, graph replay count, and CUDA event boundary. This removes substantial untimed host work on large sources. Changed-input graph qualification still hashes and checks two distinct inputs. Model timing is unchanged.

2026-08-27: Added a final-package compiled FP32 envelope gate for singleton rank, D=7168/R=1024, and D=R=8192, both implicit and projected. It reuses the existing fullgraph-forward/backward and changed-input CUDA Graph checker with optional dtype/shape arguments; defaults and every existing candidate check are unchanged. Fixed seeds and tolerances are unchanged; these are development envelope cases, not heldouts. New test inputs are frozen before first execution. GPU checker hash: 53186d885cd88d3d6b2565733bef2bca30cbc9b4d2907ff5f22b6d98fb166240 -> 392152a223b0fd5e0798278cd9520b0fe19794f88cf9818552c5b5510fb7e860.

2026-08-27: Added optional explicit selection of Triton and/or Gluon compile-bridge comparator arms. Default remains both. Selecting one allows a two-model paired primary comparison within H100 graph-pool memory; it changes neither kernel/model configuration nor the per-arm timing, inputs, qualification, or estimator. Failed selected comparators remain failures. Native sources remain unmodified.

2026-08-27: Reject unknown or malformed phase names before environment setup. The first regression-direct submission misspelled operator as operator_timing and produced no samples; its report is retained and is not performance evidence. Existing phase aliases, empty metadata-only configuration, equations, inputs, timing, and statistics are unchanged. Four CPU regression cases cover invalid names/types. Runner hash e18dc1843e7a4b925d817dac37808a87bfdae246e779903a8713812b6174d0a3 -> 2cb474c805743c5184a98d4488e59b7f235e8e02ae1ec7d28e076502b2ccfcc4.

2026-08-27: Added opt-in standard_fla_comparison for the requested projected-key rank sweep. The projected model retains its D+R learned output projections and independent embedding key; a separate standard R=D FLA model is qualified against its own standard equation reference, then timed in the same balanced rounds. Cross-architecture comparisons are explicitly labeled and include every projected parameter, projection backward, and optimizer work. No production kernel, equation oracle, tolerances, model recipe, timing boundary, seeds, or statistical estimator changed. Existing comparisons are unchanged when the flag is absent. Runner hash 2cb474c805743c5184a98d4488e59b7f235e8e02ae1ec7d28e076502b2ccfcc4 -> a5cf087d50adb0edcee3eaf584a09d57e377032c78912a3b54627117b9127735.

Before the first projected-versus-FLA GPU run, independent paired reviews caught an unlabelled standard-versus-projected-reference statistic when reference timing was enabled. Architectural arms now bypass the same-architecture comparison loop. A pure statistics regression verifies this at projected R=D with reference timing enabled. Metadata names both architectures explicitly. Final pre-run runner SHA256: d6e452b5cc737a60680995a9f71913081e29a0f3c12b3a8e97cf55ad038dfd1b.

2026-08-27: Extended runner eligibility to all six requested paths: standard/sliced/projected in Full/Block. Block FLA uses the existing callable model path over completed and partial sources, while the candidate retains the existing prepare/merge cache. Same-equation R=D schedule comparisons are distinguished from LR-versus-standard architectural comparisons. Explicit standard baselines suppress duplicate sliced R=D FLA arms. Model, kernels, math, primary recipe, timing, and estimator are unchanged. CPU coverage verifies every variant/mode, callable Block source order and all parameter gradients. No Block performance claims precede fresh GPU qualification. Runner SHA256: 14179510aafd06dd0ff932def019142c969c05e830fd6b77f5a77308ad5e928d.

2026-08-27: The projected R16 H100 comparison exhausted memory during the standard reference's untimed backward while an already-qualified projected arm remained resident. Qualified arms now reside on CPU during the separate standard qualification and return to CUDA before compilation or optimizer creation. Failed model locals are released, failed core qualification skips further comparator allocation, and restoration failures are reported. State, batch, sequence length, gradients, tolerances, timing, and random inputs are unchanged. CPU lifecycle regressions cover success, comparator failure, core failure, and restoration failure. Runner SHA256: 55cb8805cb336bd524c0c98d44d6e0b5e14c42156b37f238ec4da696268c1a9a.

2026-08-27: Block graph checks previously generated replay inputs only after the initial candidate passed. A failed early case therefore changed later cases' inputs despite an unchanged seed. Replay inputs are now generated before invoking the candidate, retaining the original successful-path draw order; initial and replay input hashes are recorded on passing and failing rows. Equations, shapes, seed, dtype, tolerances, direct cache adjoints, exact repeats, and replay comparisons are unchanged. The H100 r2 implicit wide failure and B200 r2 projected wide failure remain failures, not evidence of a scalar-cast regression. Previous reports are retained; all Block candidates require this corrected gate. Checker SHA256: 07c7a32b18e21b1f2206abdca9e56caae68cc2b2b773aa08d422382262335e0d -> 0b79dbc6512f29a5d37ed3a91ac4935ccc9059addcfbc603715d01352d3c52be.

2026-08-27: Added opt-in include_per_read for matched Block cache ablations. The additional arm uses the existing public attnres function over the unchanged completed/partial source schedule, at the same variant, rank, model state and inputs. Both arms qualify independently against the same equation reference; direct cache/per-read paired statistics are produced regardless of reference timing. Full requests fail before input allocation. Comparator cleanup failures are recorded without silently dropping later selected backends, and staging metadata records actual execution. No kernel, model, oracle, tolerance, timing boundary, seed, or estimator changed. Root regressions cover all three variants, failed comparators, mode rejection, statistics orientation and cleanup continuation. Runner SHA256: 55cb8805cb336bd524c0c98d44d6e0b5e14c42156b37f238ec4da696268c1a9a -> 4f0034a17435b14c8b0ff56977c76527c7b2f3abf346161808d5dfb8b3dc9fad.

2026-08-27: Added BF16 sliced development checks at mid and near-full ranks, ragged D/R, the D=2048 resource boundary, and an outside-boundary control. Each reuses the unchanged strided-producer equation check and fullgraph/changed-input graph checker, with FP32 learned queries. No oracle, tolerance, kernel, timing or heldout changed. Also checks non-unit feature/query strides through compiled producer gradients. Frozen before first GPU execution. CUDA test SHA256: 1d2a620d4e0462b95e813e785a4d8a0a6d989b0a0b7cf1c606d69c952a7010cc -> 67e622f0fa9f813ab6f2c5ee3a7464ad8cb079c831e38a6f7fed299a0fe3a1b3.

2026-08-27: Native FLA now receives its ordered source tuple directly for standard/sliced R=D Full and Block reads via an explicit backend capability. Other callables, cached kernel/reference paths and reduced/projected keys retain their previous assembly. This removes avoidable model-side stacking from the comparator; previous FLA results remain packed-input integration measurements, not strongest-native comparisons. Vendor kernels, native autotuning, equations, model parameters, inputs, timing boundary, statistics and tolerances are unchanged. CPU source identity/order and all-gradient tests cover both schedules and opt-out paths; the native fullgraph model GPU gate now covers both Full and Block. New GPU qualification is required before timing. Frozen source changes: {"benchmarks/fla_compile.py": {"after": "4d5d2cf8b34e040f9a6e8e5eee042fe53b1e44975195d54bd5deb4b7506a68ee", "before": "dc2b4ac26cf06b3787e2c16b851232bcfb910ff3050357b66034f9c0857f6bb4"}, "benchmarks/model.py": {"after": "281faf4b49af8b0ed4045fc992ebaa830098af279d1cbeef1f5fe7ae40a59faa", "before": "fcbf10321532489e28714beba71dc5e9b03934636f809b7aa810a1e1dea89578"}}. Root-owned test hashes before first GPU execution: {"tests/test_fla_compile.py": "cce53fa89b25b9e0fab43b77aa59591456b4258ceb950dfa82e55569118d88fb", "tests/test_training.py": "502583c5b802f8a94a8a0d6d84ee813c4cd759bf95f0451528cc1bde4346d7b8"}.

## Source-list interface gates v1 (2026-08-27, before candidate GPU evaluation)

Root-owned extension for direct source containers across Full/Block and standard/sliced/projected. Fixed oracle, numerical tolerance, seeds and timing protocol are unchanged. Each source case resets the seed and draws replay inputs before candidate execution. Gates compare all source/producer, query and partial gradients against the independent equation and same-input packed control, test shared tensors and both FP32 cache adjoints, check compile and compiled CUDA Graph replay, and extend complete optimizer-step checks. The no-stack check is a shape/count heuristic for stack/cat, complemented by source review; it is not proof of all memory traffic. Mixed fixtures permit only their explicitly packed side. Optional FLA tests skip only an unconfigured missing vendor, not a failed configured comparator.

API A and model B are integrated for local CPU checks; kernel candidates remain separate. Known packed FP32 failures are not waived. No source-list CUDA qualification or timing claim exists at this freeze.

```json
{
  "benchmarks/model.py": {
    "old": "281faf4b49af8b0ed4045fc992ebaa830098af279d1cbeef1f5fe7ae40a59faa",
    "new": "54a9d6b95cfb67d95141de9fd33cde3eae0cd632a8bb875f97877b2c5df036af"
  },
  "tests/test_cuda.py": {
    "old": "67e622f0fa9f813ab6f2c5ee3a7464ad8cb079c831e38a6f7fed299a0fe3a1b3",
    "new": "df3c02ffdd039db699937d1d113cef8949b29af2ea78f71d1ecc9e2ed576028c"
  },
  "validation/source_checks.py": {
    "old": null,
    "new": "4b17092417c88ff4f38a70e03dff432c7b83061ba289b125f303e718cf6b00da"
  },
  "tests/test_api.py": {
    "old": null,
    "new": "14bf825b3716cc50fef2cf8e727559e2300bed655f923dda69fdbea11b4cd931"
  },
  "tests/test_block.py": {
    "old": null,
    "new": "9016dda1fcdf6b6e5ffe302b1e1d8236719d2b9403f1a742c3a064181dd0148f"
  },
  "tests/test_training.py": {
    "old": null,
    "new": "9335329baea12c7c74c261f5862ed75403fd1070be3577255d4b1e9d8caef0f2"
  },
  "tests/test_training_graph.py": {
    "old": null,
    "new": "db39940fec71d4a4a5a7e3abcc51007a34e90dd66f49eeae46595283edbfb32f"
  },
  "tests/test_projected_kernel.py": {
    "old": null,
    "new": "8ccc2ad11d09ce0b31e0003cf49c71e3b3b453e4535ab13ca7e71eb230855ff9"
  }
}
```

## Source-layout timing comparison v1 (2026-08-27, before first timing)

Root selected independently reviewed runner A. Opt-in include_packed_comparison creates an independently qualified packed kernel/reference pair with identical model state and configurations except source_layout. List models are staged on CPU during qualification and restored before compile/optimizer creation; no timed work is moved. The direct paired statistic is list/packed with and without reference timing. Raw rows distinguish list, packed, and reference_stack; metadata records actual staging and all projected parameter costs. Root regression tests cover all six variants/schedules, state identity, reference lifetime, failed core/comparator/restoration, ratio orientation, default paths and output metadata. These are CPU control-flow tests, not GPU timing evidence. Oracle, numerical tolerances, model recipe, seeds, timing boundary, statistical estimator and heldouts are unchanged.

{
  "benchmarks/run.py": {
    "old": "db5dfe835e80884b891988ffe3f465f5448cbf7c65c413abef8c0330bfdf5a05",
    "new": "18864fbd1963f3eddd56c03a981853bc748e723bd2f8222be9505b45c195085e"
  },
  "tests/test_benchmark.py": {
    "old": "f448429d8d236f9b76a9c20954c97570c7ff3ac1c1dcced094de30fad23332d1",
    "new": "f448429d8d236f9b76a9c20954c97570c7ff3ac1c1dcced094de30fad23332d1"
  }
}

## Independent CUDA test compilation state

2026-08-27: Added an autouse fixture that calls the public `torch.compiler.reset()` before each CUDA-marked test. The unchanged six-mode batch exceeded Dynamo's default per-code-object guard cache after four variants on both GPUs; all remaining variants passed unchanged in fresh processes. This fixture isolates unrelated tests without raising compiler limits, suppressing errors, resetting within a test or timed step, changing inputs, tolerances, or assertions, or altering any benchmark/model/kernel code. Original batch failures remain archived.

New frozen file: `tests/conftest.py` SHA256 `a0be140a426cfaadc41c7cb2fb06f717317db1f56ec574ea65424be4fcd2fc19`. PyTorch 2.11 reset documentation: https://docs.pytorch.org/docs/2.11/generated/torch.compiler.reset.html

## Source argument lifetime regression

2026-08-28: Added a root-owned weak-reference gate for completed eager forward/backward calls across standard/sliced/projected x Full/Block x packed/list x BF16/FP32. Every case creates fresh leaf values, keys/query and partials, returns only weak references, and requires their Python Tensor objects to be released after the call. The test never clears compiler caches or side tables inside its two-call loop; the existing fixture resets before each independent test only. CPU equation controls pass. Existing oracle, inputs, tolerances, compile/graph assertions and timing are unchanged. GPU control and corrected candidates remain untested at this freeze.

tests/test_cuda.py: `df3c02ffdd039db699937d1d113cef8949b29af2ea78f71d1ecc9e2ed576028c` -> `83a2b436f5126815712f7b189289a2a78d5f820965ec21df24370b21a965fd2e`.

## Full kernel launch signature guard

2026-08-28: Added a root-owned CPU AST check in the existing Full test module. Every packed and source Triton launch must supply its kernel parameters and must not supply unknown kernel keywords. This catches the observed FP32_VALUES port mistake (keyword sent to forward and omitted from backward) before GPU admission. The guard rejects immutable candidate 0653416 and accepts corrected independent candidate 3614133. It does not evaluate CUDA math or change any numerical/timing gate. Existing kernels, oracle, inputs and tolerances are unchanged.

New frozen file: `tests/test_full_kernel.py` SHA256 `a525d430c23714b84923c24a1455a32e93acd102c0eb5fce4c7d4f1b08cad41c`.

## Compiled source argument lifetime

2026-08-28: Added a root-owned BF16 tuple-source weakref gate across all six modes, each under fullgraph compiled autograd and compiled CUDA Graph capture/replay (12 cases). Each case calls the same compiled function with two fresh input sets; all Python Tensor input objects must be released after backward and explicit graph/stream destruction. Graph cases replay twice with changed leaf values. Both Block cache adjoints are nonzero. Graph breaks and new graphs on the second input set fail. This checks Python ownership, not CUDA allocator/graph-pool reservation; existing numerical and timing gates remain independent and unchanged. The old 24-case eager lifetime gate is unchanged. CPU backend=eager helper controls pass for all six modes; CUDA remains untested at this freeze.

tests/test_cuda.py: `83a2b436f5126815712f7b189289a2a78d5f820965ec21df24370b21a965fd2e` -> `5d7015ea7ecfcfafbbe0c7174c5b49564001591e6dc09f5f8ac56d75c3c0a97c`.

## API and output-stride qualification assembly

Reuses the existing strict scalar tests from source_api_contract_gates and the existing leading output batch stride regressions from b894944 unchanged. Oracle, tolerances, ordinary and held-out inputs, and timing are unchanged. This separate frozen snapshot evaluates the API/stride pair on direct opaque source launches; earlier reports and failures remain unchanged.


2026-08-28: Root extended the existing scalar-container API tests to packed and list Block preparation/merge, reusing the shared scalar helper. New checks cover invalid scalar types and valid integer scalar equation parity; no kernel math, numerical tolerances or GPU evaluator changes. tests/test_api.py a61e02724e002ab597b67e7b9db1cff3dc1587e9dae1c7c87f60b503348a9e76 -> 8b90e3397f917f8dcf2a5fd1be6f6e3eb3b9b5ad2b2dde62480187dd1cbca40e.


2026-08-28: Root froze exact independently identical source runtime composition A/B plus canonical source-layout evaluator. Preserves base strict scalar and non-affine stride fixes. Reuses unchanged Full pair/resident boundaries, Block narrow/wide graph gates, and source API/weakref/training checks. Adds one development maximum-resource shape S129,N7,D=R2048 for implicit/projected BF16 compiled/replay; existing cases, oracle, tolerances and timing unchanged. No FP32 precision candidate or projected M1/BL2 backward selected. Source release remains unqualified pending fresh all-six-mode GPU results and complete-training comparison.

2026-08-28: Root froze16 CPU producer-control tests before candidate integration: projected-only strict bool validation, unchanged views/storage/strides and all-parameter gradients for Full/Block and packed/list, slice/split paired-stat orientation, standalone config pass-through, and staged-reference release including comparator failure. The pre-feature4b6774a fails all16 as expected. Existing oracle, seeds, tolerances, timing estimator and CUDA gates are unchanged.

2026-08-28: Integrated isolated producer-controlB c2a61e4 and profiler-safetyB95ca283. All16 pre-frozen CPU producer tests pass without hash bypass. Added two root profiler device/stability tests and six compiled BF16 Full/Block complete-training controls at projectedR16/32/64: torch.split producer versus ordinary slices, fixed state/input, two changed-input steps, accumulation2 and fused capturable AdamW; all parameter gradients and optimizer state use existing tolerances. Existing CUDA helper default behavior and all runtime kernel files remain unchanged.


2026-08-28: Root froze additive launch-trace diagnostic tests before candidate follow-up: resource grouping independent of correlation IDs, bounded identifiers and group truncation, explicit missing versus invalid metadata, strict complete kernel events, one replay, old event aggregation preserved, and temp-file cleanup on success/error. Candidate20ca7f2 passes3 and fails3 of these6 new checks (missing phase accepted); existing40 compact tests unchanged. No kernel, oracle, inputs, numerical tolerances, or primary timing/statistics changes. Only benchmarks/run.py and tests/test_benchmark_compact.py manifest entries updated.

Root pre-follow-up gate also extends the existing missing-exporter mock to require an explicit unavailable launch summary while preserving name/time events.

2026-08-28: Root integrated A schema follow-up7a2d245, then hardened non-string names/non-numeric durations and retained existing event aggregation on export failure. Correlation IDs are explicitly bounded with aggregate omission signal; complete launch-group coverage is distinct from exhaustive IDs. Existing compact suite plus11 independent trace cases passes51/51. Static comparison with13a confirms all runtime source bytes, model, timing and statistics functions unchanged; only after-timing profiler helper changes, plus three parser helpers. Frozen runner/test hashes updated; oracle, inputs and tolerances unchanged.

2026-08-28: Added root-owned tests for the optional public carrier wrapper: strict numeric/query/sequence validation, explicit value dimension, independent CPU equation and all-gradient parity, and BF16 public-symbol fullgraph plus changed-input CUDA Graph replay at R16/32/64. The existing carrier fixture now also exercises non-affine multidimensional batches, shared sources and noncontiguous upstream/query; individual-source copies are allowed, whole-source packing remains guarded. Existing private/packed tests, oracle, tolerances, inputs, training timing and estimator are unchanged. This freeze precedes public-wrapper GPU qualification and is not release promotion.

2026-08-28: Source-isolated carrier-forward screen adds no oracle or tolerance changes. Both Full and per-read Block use the same candidate carrier forward; backward retains addressA token2/stage3. Corrected carrier report scope to use the actual Full/Block mode; timing, inputs, qualification and estimator are unchanged. Existing public non-affine/alias/graph and private layout/gradient gates are reused.

2026-08-29: Root starts a separate implicit-only rebuild at the user's request. Projected/carrier public/runtime paths and their dedicated test file are removed; standard/sliced tests retain their equations, seeds and tolerances, with a new independent Block cache-adjoint check. B d43e866 supplies the three-kernel archived fixed-tail core; root removes its unused grad_tail load. The public CUDA dispatch remains unchanged pending qualification. Direct core gates cover BF16 ragged/wide shapes, mixed query dtypes, noncontiguous inputs, fullgraph changed inputs and CUDA Graph replay. These are new development gates, not a passed release or a replacement of the immutable 261ee206 holdout. Numerical oracle and protocol files remain byte-identical. Benchmark/model migration and remaining test migration are still pending in this partial integration.

2026-08-29: Root completes the implicit-only model, benchmark, test and documentation migration in this separate rebuild. The new canonical initializer is explicitly named `canonical_implicit_max_rank_v1`; it constructs a standard R=D model and takes tail query slices for lower ranks. It does not reuse historical projected-max state hashes. Projected-specific tests and the obsolete `diagnose_fp32` task are retired; the latter also depended on a removed fused module. Historical checkouts, diagnostics, failures and this revision history remain unchanged. Common implicit tests, finite/elementwise tolerances, paired statistics and training boundaries are retained. The evaluation document reflects the current implicit-only scope and existing US$300 total authorization; the root ledger is not reset.

Root selects source adapter A `00b9ea0` together with its generalized `fixed_tail.py`. Only addressing changes between packed and Tensor[] routes: source order, fixed source tile/warps/stages, physical-R tail arithmetic, FP32 saved state and one folded D-wide dV store remain shared. No public dispatch change or performance promotion is made here. New direct gates extend the packed cases to lists, mixed value/query dtypes, non-affine batches, shared source edges, no-stack compiled/capture paths, changed-input replay and four complete standard/sliced Full/per-read Block training steps against the reference. The static launch guard now checks both modules. Local full suite: 268 passed, 151 CUDA/optional skips. The frozen 44-case source gate has not yet run on CUDA at this revision. `validation/oracle.py` and `validation/protocol.json` remain byte-identical to the prior packed freeze and the historical holdout.

2026-08-29: The first source gate (77c278b) completed on H100 with 32 passes and 12 failures, all at the generator-based math.prod row-count expression under Torch2.11 fullgraph tracing. The immutable report and root failure verification are retained. Independent review also reproduced a fake-tracing exception escaping the non-affine view fallback. Root selects adapter repair A599e1d4: derive rows from numel/width and use reshape outside the opaque operator (a view where possible, otherwise an individual-source copy). Kernels, launches, schemas, arithmetic, all 44 tests, oracle and protocol remain unchanged. Re-freeze before rerunning; no B200 run or public dispatch change on the failed candidate.

2026-08-29: The unchanged repaired direct gate at de96c7b passed44/44 on H100 and B200, with source/config/frozen hashes and terminal nodes verified. Root now wires the shared fixed-tail primitive into the public API via thin route B e97ed1c. Public/cached-Block complementary qualification and all performance claims remain pending; historical failures stay retained.

Root corrected two benchmark-only migration defects before timing: the implicit profiler input digest now hashes values/query/upstream without the removed None key slot, explicitly tagged values-query-upstream-v1; top-level source_layout overrides nested configuration consistently. The same tensors, seeds, oracle, numerical tolerances, timing and paired estimator are unchanged. The loaded-only fixed-tail resource probe includes normal/frozen modules but makes no selected-binary claim. Root extends existing tests for the loaded probe and native eager comparator's source sequences, order, shape, shared gradients, no whole-stack dispatch, malformed inputs and removed keys argument. Before comparator repair these12 selected adapter cases fail; four resource tests pass. These are correctness regression tests, not performance evidence.

2026-08-29: Root selected native comparator A f9f981e after independent A/B implementation and code review (37 additions versus B48; A avoids repeated model validation and duplicate metadata fields). All12 frozen native-adapter regressions now pass, plus the four loaded-resource tests. Native FLA receives its ordered per-source tensors directly; permitted individual reshape/contiguous conversions remain explicit. This repairs the comparison interface and does not alter the vendor kernels or establish performance.

2026-08-29: Root migrated tests for the active fixed-tail implementation before deleting unreachable Full/FLA-derived modules. Existing private affinity checks now exercise actual source/output adapters, including CPU fullgraph non-affine gradients; obsolete private CUDA-only import/old launcher checks are removed (new fixed-tail launch checks retained). Resource probes require already-loaded normal/frozen fixed-tail modules and forbid imports; both cases fail against the old eager-import probe. Added one training CLI regression for explicit sliced rank. No oracle, tolerance, input generator, timing, or active kernel change.

tests/test_full_kernel.py: 4f912a6988c521454e076e16fa23b3261667b2fc53d590b38d2ce4c84be72b90 -> 12d5675c31438f0a68188e5d379531fdb98b8a2f471ee15167a5ff4b644e7cdd
tests/test_benchmark_compact.py: 8337c1a790d8219b4423ca14a875cea058a55451081de615f6223c19318845f8 -> 60dd9bded9e54d520bd07083e0ad5c44ccd7e7ba8a97a2a701850d6da58003f1
tests/test_training.py: ca29b826dfd9ab13ffb6a1448a082728b6d43deb893a2266ebfeef1317101b50 -> 9fafae9b084b4c49f4f113dbd92eda6bd043c6806f48e44fa56bcd8b68acb5c0

2026-08-29: Selected isolated prune A1d4916d: delete unreachable _kernels/full.py and fla_full_sources.py (2400 lines); inspect only already-loaded fixed-tail resource modules. All eleven remaining runtime Python files are byte-identical to public GPU-qualified6beb157. New source/profile snapshot is distinct; historical frozen files/reports preserved externally. Source-profile hash 208ea4e775eca96a4f9a3ce1bd8abe406d28101f3f4b09e6f13f7fe567706518 -> 2604c701cb00aad9f6784c237b2434274c781e0d23ced6ab81f733b766ecc2c9.

## Installed artifact gate

Add a frozen installed-wheel test: clean current-source copy, direct wheel and sdist roundtrip, exact runtime inventory/license checks, CPU gradients, and optional CUDA fullgraph plus changed-input graph replay. No runtime, equation, input protocol, timing, or tolerance changes. CUDA execution pending at this freeze.

## Fixed-tail resource experiments

Root extends the existing envelope with even source counts, equal padded D/R widths, nonpower dimensions and a wide BF16 near-full-rank case. Existing compiled/changed-input graph checks also run D63/R33 for both storage/query dtypes and packed/list/non-affine layouts. Original cases, seeds, comparisons and tolerances are retained. This gate is frozen on the unchanged baseline before separate source-tile and resident-tail candidates; it does not combine those changes or establish performance.

tests/test_full_kernel.py: 12d5675c31438f0a68188e5d379531fdb98b8a2f471ee15167a5ff4b644e7cdd -> e71770f2f6b7f45653503665f303a75a1d2a581b0ff0f86c90eddf2800fb0764

Extend the existing fixed-tail Full/per-read Block complete-step reference gate with sliced R48/D64, retaining the R16 and R=D configurations. This exercises the equal-padded-width branch through all parameter gradients and the optimizer boundary, without tolerance/state/timing changes.
tests/test_training_graph.py: 82306de454dc4756d8518f5277dd483e1f07b9581a085f0e55f708f7871ca228 -> e0873b212a1e412f2ea8bed7f656ad09fb253752913b2196c57a407f0d3ee1e3

## Ephemeral source metadata gates

Root extends the launch ABI check to source-adapter setup kernels and setup functions without a naming suffix. Negative missing/unknown argument checks cover the checker itself. Existing input-release and compiled/capture lifetime cases will be included in the selected GPU set; no runtime, oracle, input recipe, timing, or tolerance changes.
tests/test_full_kernel.py: e71770f2f6b7f45653503665f303a75a1d2a581b0ff0f86c90eddf2800fb0764 -> 7bc74596a78df273e73c4b494ecbf624e537a8605a0d712f1b60856fe2ffb540

Select isolated pointer-record A37df0da for BF16 source lists only; FP32 and packed routes retain tuple/base addressing. Ephemeral setup and input/gradient records are included in calls; no persistent address cache. Source tile1, arithmetic, query reduction, tolerances and timing are unchanged. Existing lifetime tests are selected for GPU screening. No equal-pad, tile2 or epsilon combination.
src/attnres/_kernels/fixed_tail.py: 782e649dff14e036a226fd0c37b2924afa1d0ad4e02fed2b21485e7df7242d91 -> fb59c581f8ee6d516b24e901aa38570ec62880fefc6fe5f18bd2984413fb6436
src/attnres/_kernels/fixed_tail_sources.py: 78a0273ecb0c35530d36ce0b36164ce1a8d9146ee3b05da9999f2a78b9c52c72 -> 536dcbe4248e254068e9dcf8e113de77b7073d87a958ca7e2778bbf41b3bce61

2026-08-29: Separate development composition gate combines previously isolated ephemeral BF16 pointer records with source tile two, selecting A c488dd2 after independent A/B static reviews. Source traversal remains lane0 then lane1; S=1 record lookup is bounded; FP32 tuple fallback and packed source tile one remain unchanged. No equal-pad reuse, split stores, epsilon specialization, shape table, oracle, tolerance, model or timing change. Reuse all 116 metadata/setup/alias/lifetime/graph tests before performance. CPU-only approval is not CUDA qualification or promotion.
{
  "src/attnres/_kernels/fixed_tail.py": {
    "old": "fb59c581f8ee6d516b24e901aa38570ec62880fefc6fe5f18bd2984413fb6436",
    "new": "bc3970c7cdd16e78623ca3f6d0c51bb8c15724ea5776593a6bd983ac7052b046"
  },
  "src/attnres/_kernels/fixed_tail_sources.py": {
    "old": "536dcbe4248e254068e9dcf8e113de77b7073d87a958ca7e2778bbf41b3bce61",
    "new": "987c5bccd21b1e818bcfc163f43528e0a42132a3d6f64f892a86a2c4a347a7d5"
  }
}

2026-08-29: Separate equal-padded reuse composition screen from71387bf. Independent candidates016a3cf/f61f2b5 have byte-identical kernel files and apply the previously screened fe3b45db patch exactly: when R<D and BLOCK_R==BLOCK_D, reuse loaded D values for tail keys. Source tiling, pointer records, packed tile1, schemas, launch settings, reducer and all other paths remain unchanged. Leading masked zeros may change reduction association, so reuse all116 CUDA correctness/lifetime gates. No reference/tolerance/timing or exact-shape dispatch changes; no promotion.
src/attnres/_kernels/fixed_tail.py: bc3970c7cdd16e78623ca3f6d0c51bb8c15724ea5776593a6bd983ac7052b046 -> f9ee20683aab094ea7ff72b1318e361acdd286cdce41a012f3238fcfeac05d5b

2026-08-29: Root freezes independently identical A7f738af/Bf82991e four-line query reuse atop equal-padded/records/tile2. Only equal-padded R<D backward uses the resident shifted query instead of the identical per-source strided load; all masks, equations, source order, stores, launches, default non-equal and R=D routes remain unchanged. Existing116 correctness/lifetime cases unchanged; no oracle/tolerance/timing edits. No performance or promotion claim.

2026-08-29: Root removes the cached Block runtime, public helpers, cache type,
benchmark arms, and cache-specific tests. Full and Block now share one read
helper and the public `attnres` primitive; Block differs only in its source
schedule and sequential partial sums. Direct per-read Block validation retains
standard and sliced equations, all gradients, aliases, compilation, and
changed-input CUDA Graph replay. Historical cache results above remain intact.
