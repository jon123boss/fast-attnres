from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HASH_RE = re.compile(r"\b[0-9a-f]{64}\b")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_citation_file_has_verified_public_identity():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "cff-version: 1.2.0" in citation
    assert "title: Fast Attention Residuals" in citation
    assert "type: software" in citation
    assert "license: MIT" in citation
    assert "version: 1.0.0" in citation
    assert "family-names: Su" in citation
    assert "given-names: Jonathan" in citation
    assert "2607.09694" in citation
    assert "2603.15031" in citation
    assert "email:" not in citation
    assert "orcid:" not in citation


def test_provenance_covers_kernel_and_license_identities():
    provenance = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    campaign = json.loads(
        (ROOT / "results/compiled_step/campaign_manifest.json").read_text(encoding="utf-8")
    )
    historical_runtime_hashes = {
        "src/attnres/_kernels/fla_full_sources.py":
            "2cd7ac89b15faeb13640bff4a7948e437453b69446bfc8c7922511e341843e10",
        "src/attnres/_kernels/fixed_tail.py":
            "2333b3034e3c0e6493855b1246280ed91e65d29a962ce1d150beff71e8bbd34e",
        "src/attnres/_kernels/fixed_tail_sources.py":
            "20fa0206fcbf6cc6b28a2973ac280575b6e8e378b09e0903449bf423d9812196",
    }
    current_runtime_hashes = {
        "src/attnres/_kernels/fla_full_sources.py":
            "8749c72c4714145214e33e8bc7d37f57b47a79b67f2e83044205db72cda416fa",
        "src/attnres/_kernels/fixed_tail.py":
            "2333b3034e3c0e6493855b1246280ed91e65d29a962ce1d150beff71e8bbd34e",
        "src/attnres/_kernels/fixed_tail_sources.py":
            "1373614c93d7291ad96697b1b8ff627120590b75f63f7e38bd65d50b19fcfb4a",
    }
    found_hashes = set(HASH_RE.findall(provenance))

    assert campaign["schema"] == "attnres.compiled_step_campaign.manifest.v1"
    assert campaign["repo_head"] == "81dffbfeb0f84470513e846e3df8080e8ffb563d"
    assert campaign["kernel_sha256"] == historical_runtime_hashes
    assert campaign["repo_head"] in provenance
    assert "1cceb5e0a37330015ca2945312da29aa7566aaeb" in provenance
    assert "results/compiled_step/campaign_manifest.json" in provenance
    assert set(historical_runtime_hashes.values()) <= found_hashes
    assert set(current_runtime_hashes.values()) <= found_hashes
    assert "5e02dd3a7651f5f2797eb8b12bbec401826031e1" in provenance
    assert "2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781" in found_hashes

    for path, digest in current_runtime_hashes.items():
        assert _sha256(ROOT / path) == digest
    assert "Historical compiled-step campaign" in provenance
    assert "not relabelled as measurements of the current kernel" in provenance
    assert _sha256(ROOT / "LICENSE") == (
        "1f373b38f897df1fffb9e5747f44b1a1f3249fffc7da687c96ee6f46a251901d"
    )
    assert _sha256(ROOT / "NOTICE") == (
        "e41d502c2ea57057bd1f603cbb0eca4330df6341511d95e554305e9ff44b8561"
    )
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Songlin Yang, Yu Zhang, Zhiyuan Li" in notice
    assert "5e02dd3a7651f5f2797eb8b12bbec401826031e1" in notice
    assert "MIT License" in notice
