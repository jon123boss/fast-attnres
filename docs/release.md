# Releasing Fast-AttnRes

Version 2.0.0 ships the CUDA BF16 operator and the final H100/B200 sweep.
Version 1's CPU/FP32 execution and exported reference are removed. The
[original release procedure](https://github.com/jon123boss/fast-attnres/blob/v1.0.0/docs/release.md)
remains available with its historical evidence.

## Build and verify

From a clean release checkout, install the development dependencies and run:

```bash
python -m pip install -e ".[dev,test,plot]" "setuptools>=68" wheel "twine>=5" "matplotlib==3.8.0"
python -m pytest -m "not cuda" -q
python scripts/verify_final_release.py --work /tmp/fast-attnres-release-audit
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)" \
  python scripts/build_release.py --output-dir dist/release
python -m twine check dist/release/*.whl dist/release/fast_attnres-2.0.0.tar.gz
(cd dist/release && shasum -a 256 -c SHA256SUMS)
```

The auditor checks the final archive and every member hash, reconstructs all
14 admitted timing reports and the original failures, reproduces the audit and
both chart projections, and compares every runtime source file with the final
measured snapshot. Only the package `__version__` literal may differ. This is
explicit metadata equivalence, not a new GPU measurement. All historical source
manifests and the documented normalization-rounding exceptions remain intact.

The deterministic builder produces exactly four assets:

- `fast_attnres-2.0.0-py3-none-any.whl`: runtime package.
- `fast_attnres-2.0.0.tar.gz`: source distribution.
- `fast-attnres-2.0.0-evidence.tar.gz`: the final evidence directory, immutable
  raw source/report bundle, release audit, attribution and reproduction guidance.
- `SHA256SUMS`: hashes of those three payloads.

The large evidence bundle stays outside the installable distributions. README
images and documentation links use the release tag and render on PyPI as well
as GitHub. Compare two builds with the same source and `SOURCE_DATE_EPOCH` to
check reproducibility. `--skip-evidence-audit` is for local packaging diagnostics
only and must never be used for a published release.

## Publish

After the exact release commit passes CI on `main`, push its matching protected
`v2.0.0` tag. The release workflow verifies the frozen contract and evidence,
builds and checks the assets, and creates build-provenance attestations. Only
the wheel and source distribution are sent to PyPI through the existing `pypi`
environment and OIDC trusted publisher. All four assets are attached to the
GitHub release. Tag protection and job-scoped permissions remain required.

Verify the GitHub asset checksums, PyPI version and distribution hashes, then
install the published wheel in an isolated location and check package metadata,
exports and CUDA-only input validation. Do not overwrite an existing published
version; fix any release issue in a new version.
