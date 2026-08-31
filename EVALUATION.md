# Frozen evaluation contract

Candidate authors must not modify this document, validation/oracle.py, validation/protocol.json, validation/frozen.json, reference.py, or root-owned tests. Root verifies their hashes.

## Equations

values [S,...,D] or a sequence of S tensors [...,D], query [R]. Keys are the implicit tail values[...,D-R:]. R=D is standard AttnRes. Parameter-free RMS key norm, query dot, scale, source softmax, full-width value mixture. eps=2^-23; scale=1. FP32 compute, BF16/FP32 storage and queries; FP64 reference for gradcheck. S=1..129,D=1..8192,R=1..D. Support packed inputs and strided source views; include source assembly and any required copies in training timing. Separate projected-key and carrier APIs are outside the active scope.

Block passes completed values and the sequential current partial as ordinary sources to the same `attnres` primitive used by Full. Block values are sums, not averages; their keys are the implicit tail. There is no separate Block attention kernel or prepared-state API. No private GraphTask APIs, mutable tickets, alpha/beta/count priors, dynamic queries, output norm, or split-block extensions.

## Correctness

Independent FP32 equation oracle. Strict output and all value/query gradient gates: BF16 rtol=atol=.05; FP32 rtol=.001,atol=.0001; require finite outputs/gradients. No normwise rescue. Test FP64 gradcheck, aliases, repeated reads and backward, checkpointing, noncontiguous upstreams, nonzero queries, changed-input CUDA Graph replay and compiled model all-parameter gradients. Compare identical architectures/weights/data for parity.

## Timing

Primary metric is complete compiled training: projection, source assembly, loss, backward, accumulation, zero_grad, optimizer and any scheduler. Exclude compilation/warmup/profiling; record them separately. Same-device balanced paired runs; fixed model/batch/sequence/optimizer/checkpoint settings across ranks. Preserve every raw sample and source/software/hardware hash. The active native baseline is FLA Triton checkpoint 1 from the exact clean checkout pinned in the production config. Same-R LR comparisons are separate from architectural LR-versus-standard comparisons.

Primary: layers24,D1024,heads16,MLP2816,B2,T2048,vocab32768,Block count8,AdamW,BF16 autocast,static queries. R=1,2,4,8,16,32,64,128,256,512,1024. Accumulation1 primary; accumulation2/checkpointing separate integration checks. Three seeds for final confirmation; incomplete budget-limited subsets are not promotion.

Gain: simultaneous paired95% CI for smaller/larger latency entirely below1. Plateau: CI contains1 and lies wholly in[.99,1.01]. Entirely above1 is slowdown; other outcomes inconclusive. All adjacent ranks must pass gain/plateau and show gains over larger ranks. No dropped failures or hold-out tuning.

## Budget

Root alone submits Modal jobs; exact H100! and B200. User-authorized total US$400. The root budget ledger accounts for prior jobs; this rebuild does not reset spending. Workspace limits remain hard stops. Bounded resumable jobs; no paid retries of unchanged compiler/correctness failures.
