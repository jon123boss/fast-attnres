"""Development failures stop promptly without changing complete campaign collection."""
from types import SimpleNamespace

import pytest

from benchmarks import bf16_competitors, bf16_device


@pytest.mark.parametrize("stop_on_failure", [False, True])
def test_development_stop_retains_failure_and_default_collects_all(monkeypatch, stop_on_failure):
    monkeypatch.setattr(bf16_device, "metadata", lambda: {
        "torch": "2.13.0", "triton": "3.7.1", "gpu": "NVIDIA B200", "capability": [10, 0],
    })
    monkeypatch.setattr(bf16_device, "load_baseline", lambda _: SimpleNamespace(
        attnres=None, metadata={}))
    monkeypatch.setattr(bf16_competitors, "load_all", lambda _: ({}, {}, {}))
    monkeypatch.setattr(bf16_device, "operator_case", lambda case, *args, **kwargs: {
        "case": case, "arms": {"candidate": {"status": "failed", "error": "compile failed"}},
    })
    monkeypatch.setattr(bf16_device.torch.cuda, "empty_cache", lambda: None)
    resets = []
    monkeypatch.setattr(bf16_device.torch.compiler, "reset", lambda: resets.append(True))
    cases = [{"shape": [5, 7, 1536, 16]}, {"shape": [9, 8192, 1536, 768]}]
    checkpoints = []
    report = bf16_device.run_operator({
        "gpu": "B200", "sources": {"candidate": "unused"}, "cases": cases,
        "seeds": [1], "stop_on_failure": stop_on_failure,
    }, lambda report: checkpoints.append((report["status"], len(report["results"]))))
    assert report["results"][0]["arms"]["candidate"]["error"] == "compile failed"
    assert "in_progress" not in report
    if stop_on_failure:
        assert len(resets) == 1
        assert checkpoints[-1] == ("failed", 1)
        assert report["stopped_after"] == {"case": cases[0], "seed": 1}
    else:
        assert len(resets) == len(cases)
        assert checkpoints[-1] == ("complete", 2)
        assert "stopped_after" not in report
