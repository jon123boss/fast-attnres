# Current 24-layer Modal campaign

This directory contains the exact evidence behind the README hero chart. The
campaign measures the current release kernel bytes in a compiled, captured,
complete BF16 training step against pinned native FLA Triton checkpoint 1.

## Audit locally

No GPU is required. From the repository root:

```bash
python -m benchmarks.audit_current_24l \
  --evidence-dir results/current_24l \
  --repo . \
  --output /tmp/current-24l-audit.json
```

The command validates the immutable artifact hashes, measured-source archive,
release kernel hashes, runtime and hardware records, Modal transport and call
identity, FLA checkout identity, complete-step and graph qualifications,
timing exclusions, every ABBA pair and shared input identity, then independently
recomputes all six paired estimates and 95% intervals. It exits nonzero on any
disagreement.

## Immutable artifacts

| Artifact | SHA-256 |
| --- | --- |
| `raw/h100-report.json` | `38b6bb49736fbd735553dc48daa04033c5321a78ac0d9285f53cffc202d5b093` |
| `raw/b200-report.json` | `9db6cea6616ea749ff3a7724840772b10adf057fad977af7814ea51aaab393bc` |
| `audits/h100-audit.json` | `a736ac2d303047da896ca30dc93a524d1080315c53b5647edd5ebab495426a73` |
| `audits/b200-audit.json` | `0809de2853158acdc8de47d4d3efaa152fbd93893341eb4ba0424b49b2b67a38` |
| `verification.json` | `c83af3c2c4a7f50c38f161cbb0d1641d760fe66062c819347acf6dc1e1adfd68` |
| `reproduction/modal_transport.py` | `abe14e49aed9d31fba8517ca39dfb68e212db495817c0273e030c80cf20dc60e` |
| `reproduction/performance_source.tar.gz` | `d0efbd3d70cdceb870d2c0b969a94a0444c968022b994f9203af046ef38a8648` |
| `reproduction/performance_source_paths.txt` | `3e9654ab9993a275211dd335128d11c07e711ff3eb25405c6f3afd694726cda9` |
| `hero_projection.json` | `0a70524ff038d55b57ff8727f85d85460e97724f3cc583de59684b4e9c81b243` |

`audits/bundle-audit.json` is a deterministic convenience output produced by
the checked-in auditor. Re-running the command above reproduces it.

The compact measured-source archive contains only the exact paths named by the
reports plus their frozen manifest. It keeps the audit self-contained even
though this marketing repository is intentionally distributed as a clean,
squashed release history.

## Modal identity

| Field | Value |
| --- | --- |
| App | `ap-oODWwtqD2TumRub2B3fQxA` |
| H100 call | `fc-01M1B2R1F8237VFV3AXRFQFW5W` |
| B200 call | `fc-01M1B2R1R9KAJP4DYV4J1KDDJX` |
| Final app state | stopped |
| Performance source | `b8837e1d74eb708a39a455840332247725a26496` / tree `6a807f2f739c45f8ec9051e83df6d7ab4df560ba` |
| Runtime | PyTorch 2.13.0+cu130 / CUDA 13.0 / Triton 3.7.1 |

The first attempted app, `ap-I4iNFEVkkZq3dPvBv3eR0x`, was invalidated before
any ratio was accepted because its FLA clone did not preserve Git metadata.
Only the corrected app and reports above enter the evidence.

## Results

See [`docs/current_24l_results.md`](../../docs/current_24l_results.md) for the
six per-seed rows, exact workload, timing boundary, and interpretation. The
Matplotlib renderer consumes only `hero_projection.json`; it never treats a
plot as an audit.
