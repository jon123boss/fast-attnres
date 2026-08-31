# Release artifacts

The release workflow builds four assets from the tagged source tree:

* the installable wheel;
* the source distribution;
* a standalone evidence archive; and
* `SHA256SUMS`, containing the SHA256 digest of the other three files.

The wheel and source distribution are produced through the PEP 517 backend.
The builder then canonicalizes ZIP and tar/gzip ordering, timestamps, owner
metadata, permissions, and compression. Set `SOURCE_DATE_EPOCH` to the commit
timestamp (the GitHub workflow does this) to make repeated builds of the same
tree byte-identical. The default is epoch zero, which is also deterministic.

The installable artifact's historical evidence bundle is the six-report Full
compiled-step campaign under `results/compiled_step`. It preserves its compact campaign
manifest, raw reports, audited sidecars, hardware/vendor attestations, the
exact reproduction wrapper and per-seed configs, and the projection JSON
freshly recomputed from all six reports. The current-release README hero is a
separate campaign and is deliberately not copied into this historical archive.
The `MANIFEST.in` `prune
results` rule keeps this evidence out of both installable package formats.
No results tree is embedded in the wheel or source distribution.

The current GitHub adoption screen is repository-hosted under
`results/adoption/compiled_step_screen`; its chart, table, CSV, raw reports,
and manifest are independently hash-bound and feed the README. It is
intentionally excluded from installable packages to avoid shipping benchmark
reports to library users. It is distinct from the historical standalone
evidence archive produced by `scripts/build_release.py`.

The bundle measures its own pinned pre-autotune kernel hashes. It must not be
used to claim performance for a release whose current kernel hashes differ;
such a release needs a new sealed campaign. The old raw reports and manifest
remain byte-for-byte historical evidence.

## Local build

From the repository root:

```bash
SOURCE_DATE_EPOCH=0 PYTHONPATH=src:. python scripts/build_release.py \
  --performance-source /path/to/clean/81dffbf-checkout
```

Use `--output-dir` and `--evidence-dir` to select different output and
evidence locations. Published builds require `results/compiled_step`, its
compact `campaign_manifest.json`, all six raw/audit/attestation triplets, and a
clean checkout of measured commit `81dffbfeb0f84470513e846e3df8080e8ffb563d`.
`--skip-evidence-audit` is available only for a local packaging smoke test and
must not be used for a published release.

Verify an asset set with:

```bash
cd dist/release
sha256sum -c SHA256SUMS
```

## Tag workflow

Tags matching `v*` invoke `.github/workflows/release.yml`. The tag must be
protected and must equal `v` followed by the version in `pyproject.toml`.
Before building, the workflow verifies every file in `validation/frozen.json`,
audits the current 24-layer H100/B200 campaign against the exact production
kernel bytes in the tagged checkout,
reconstructs a clean checkout of the exact measured source commit from the
sealed repository bundle, and checks the compact compiled-step manifest, all
six reports and attestations, and the deterministic historical projection. It
then builds and checks exactly the four
documented assets, verifies `SHA256SUMS`, and uploads all four files to the
GitHub release. GitHub build provenance attestations are created for all four
files.

Only the wheel and source distribution are copied into the PyPI upload
directory. The final publish step uses PyPI Trusted Publishing through the
`pypi` GitHub environment and never sends the evidence archive or checksum
file to PyPI. The workflow's write-capable GitHub token is limited to the
GitHub release job; build and attestation jobs retain read-only contents
access.

Repository settings must provide a protected `v*` tag rule. Availability of
tag protection can depend on the GitHub plan, including private repositories.
The `pypi` environment must exist and its Trusted Publisher configuration
must match this repository, workflow, and environment; otherwise the OIDC
publish step intentionally fails closed.
