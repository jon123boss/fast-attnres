from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.plot_compiled_step_hero import (
    PROJECTION_SCHEMA,
    ProjectionError,
    load_projection,
    render_hero,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "compiled_step_hero_projection.json"
CURRENT_PROJECTION = ROOT / "results" / "current_24l" / "hero_projection.json"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _write_payload(tmp_path: Path, payload: dict, name: str = "projection.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def test_fixture_is_small_and_has_the_complete_two_gpu_three_seed_contract():
    payload = _payload()
    assert FIXTURE.stat().st_size < 10_000
    assert payload["schema"] == PROJECTION_SCHEMA
    assert payload["campaign"]["mode"] == "full"
    assert payload["campaign"]["dtype"] == "bf16"
    assert payload["campaign"]["model"] == {
        "batch": 2,
        "block_count": 8,
        "ffn": 2816,
        "heads": 16,
        "layers": 24,
        "sequence": 1024,
        "vocab": 32768,
        "width": 1024,
    }
    assert tuple(payload["devices"]) == ("B200", "H100 SXM")
    assert tuple(payload["campaign"]["seeds"]) == ("20260827", "20260903", "20260911")
    assert all(len(device["ratios"]) == 3 for device in payload["devices"].values())
    assert "raw_samples" not in json.dumps(payload)
    assert "reports" not in payload


def test_load_projection_normalises_device_order_and_keeps_unpooled_rows():
    projection = load_projection(FIXTURE)

    assert projection.status == "fixture"
    assert tuple(device.key for device in projection.devices) == ("H100 SXM", "B200")
    assert tuple(item.seed for item in projection.devices[0].ratios) == (
        "20260827",
        "20260903",
        "20260911",
    )
    assert projection.devices[0].advantage_pct == pytest.approx(
        100 * (1 - 0.918)
    )
    assert projection.devices[0].advantage_range_pct == pytest.approx((7.7, 9.3))


def test_load_projection_rejects_raw_report_or_pooled_fields(tmp_path: Path):
    raw_payload = _payload()
    raw_payload["reports"] = {"raw.json": {"status": "complete"}}
    with pytest.raises(ProjectionError, match="unexpected"):
        load_projection(_write_payload(tmp_path, raw_payload, "raw.json"))

    pooled_payload = _payload()
    pooled_payload["devices"]["B200"]["pooled_ci"] = [0.9, 0.95]
    with pytest.raises(ProjectionError, match="unexpected"):
        load_projection(_write_payload(tmp_path, pooled_payload, "pooled.json"))


def test_load_projection_rejects_wrong_geometry_or_seed_count(tmp_path: Path):
    geometry_payload = _payload()
    geometry_payload["campaign"]["model"]["sequence"] = 2048
    with pytest.raises(ProjectionError, match="sequence"):
        load_projection(_write_payload(tmp_path, geometry_payload, "geometry.json"))

    seed_payload = _payload()
    seed_payload["campaign"]["seeds"] = ["20260827", "20260903"]
    with pytest.raises(ProjectionError, match="exactly three"):
        load_projection(_write_payload(tmp_path, seed_payload, "seeds.json"))


def test_render_hero_is_byte_deterministic_searchable_and_honest(tmp_path: Path):
    pytest.importorskip("matplotlib")
    projection = load_projection(FIXTURE)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    svg_path, png_path = render_hero(projection, first_dir)
    svg_copy, png_copy = render_hero(projection, second_dir)

    assert svg_path.read_bytes() == svg_copy.read_bytes()
    assert png_path.read_bytes() == png_copy.read_bytes()
    assert hashlib.sha256(svg_path.read_bytes()).hexdigest() == hashlib.sha256(
        svg_copy.read_bytes()
    ).hexdigest()
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "Faster Full AttnRes training" in svg_text
    assert "24 layers · BF16 complete CUDA Graph step · lower is faster" in svg_text
    assert "H100 SXM" in svg_text
    assert "B200" in svg_text
    assert "8.20% faster" in svg_text
    assert "7.60% faster" in svg_text
    assert "Bars are descriptive means of the three per-seed arm means" in svg_text
    assert "seeds and GPUs are not pooled" in svg_text
    assert "Fixture projection" in svg_text
    assert "<text" in svg_text

    width, height = struct.unpack(">II", png_path.read_bytes()[16:24])
    assert (width, height) == (2800, 1600)


def test_current_projection_keeps_exact_campaign_labels_and_scope(tmp_path: Path):
    pytest.importorskip("matplotlib")
    projection = load_projection(CURRENT_PROJECTION)
    assert projection.status == "audited"
    assert projection.devices[0].attnres_ms == pytest.approx(28.4127466)
    assert projection.devices[0].fla_ckpt1_ms == pytest.approx(29.9743266)
    assert projection.devices[1].attnres_ms == pytest.approx(16.5277154)
    assert projection.devices[1].fla_ckpt1_ms == pytest.approx(19.5546755)
    svg_path, _ = render_hero(projection, tmp_path)
    svg_text = svg_path.read_text(encoding="utf-8")

    assert "5.20% faster" in svg_text
    assert "15.52% faster" in svg_text
    assert "Verified current-release campaign" in svg_text
    for label in ("28.41 ms", "29.97 ms", "16.53 ms", "19.55 ms"):
        assert label in svg_text
    assert "ms / step" in svg_text
    assert "native FLA ckpt1" in svg_text
    assert "Gluon" not in svg_text
    assert "Catswe" not in svg_text


def test_current_projection_headlines_keep_two_decimal_precision(tmp_path: Path):
    payload = _payload()
    current_values = {
        "H100 SXM": {
            "attnres": 28.4127466,
            "fla_ckpt1": 29.9743260,
            "ratios": (0.9476044537, 0.9480618251, 0.9480432328),
        },
        "B200": {
            "attnres": 16.5277154,
            "fla_ckpt1": 19.5547499,
            "ratios": (0.8460434028, 0.8448021851, 0.8447718995),
        },
    }
    for device_key, values in current_values.items():
        payload["devices"][device_key]["absolute_ms"] = {
            "attnres": values["attnres"],
            "fla_ckpt1": values["fla_ckpt1"],
        }
        for item, ratio in zip(payload["devices"][device_key]["ratios"], values["ratios"]):
            item["ratio"] = ratio
            item["ci_low"] = ratio - 0.0001
            item["ci_high"] = ratio + 0.0001

    svg_path, _ = render_hero(load_projection(_write_payload(tmp_path, payload)), tmp_path)
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "5.20% faster" in svg_text
    assert "15.52% faster" in svg_text


@pytest.mark.parametrize(
    "entry_point",
    [
        ("benchmarks/plot_compiled_step_hero.py",),
        ("-m", "benchmarks.plot_compiled_step_hero"),
    ],
    ids=["direct-script", "module"],
)
def test_cli_entry_points_render_the_same_projection(tmp_path: Path, entry_point: tuple[str, ...]):
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "rendered"
    result = subprocess.run(
        [
            sys.executable,
            *entry_point,
            "--projection",
            str(FIXTURE.resolve()),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "compiled_step_hero.svg").is_file()
    assert (output_dir / "compiled_step_hero.png").is_file()
