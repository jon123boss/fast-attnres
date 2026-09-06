"""Content verification for private and public evidence bundles."""
import hashlib
import json
import zipfile

import pytest

from benchmarks.bf16_archive import restore


@pytest.mark.parametrize('public', [False, True])
def test_restore_verifies_both_evidence_formats(tmp_path, public):
    data = b'kernel = "retained source"\n'
    digest = hashlib.sha256(data).hexdigest()
    files = {'snapshots/runner/kernel.py': digest}
    manifest = {'jobs': {'job': files}}
    if public:
        manifest.update(format='fast-attnres-public-v1',
                        jobs={'job': {'execution': {'status': 'complete'}, 'files': files}})
    path = tmp_path / 'evidence.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('objects/' + digest, data)
    output = restore(path, 'job', tmp_path / 'restored')
    assert (output / 'snapshots/runner/kernel.py').read_bytes() == data


@pytest.mark.parametrize('relative,data,message', [
    ('snapshots/kernel.py', b'tampered', 'corrupt archived source'),
    ('../outside.py', b'original', 'invalid archived path'),
])
def test_restore_rejects_corruption_and_parent_paths(tmp_path, relative, data, message):
    digest = hashlib.sha256(b'original').hexdigest()
    path = tmp_path / 'invalid.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps({'jobs': {'job': {relative: digest}}}))
        archive.writestr('objects/' + digest, data)
    with pytest.raises(ValueError, match=message):
        restore(path, 'job', tmp_path / 'restored')
    assert not (tmp_path / 'outside.py').exists()
