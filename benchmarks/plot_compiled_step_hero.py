"""Render the audited compiled-step README hero.

The plotter intentionally accepts one small, auditor-generated projection
JSON.  It does not know the raw report layout and never opens a report,
manifest, source checkout, or timing log.  The auditor is responsible for
binding the projection to the raw evidence and for recomputing its statistics.

The projection has one Full, BF16, complete CUDA Graph training step at
``L24/D1024/B2/T1024`` and exactly three unpooled seeds on each of H100 SXM and
B200.  Absolute bars are device summaries supplied by the auditor.  Ratio
intervals remain one interval per seed; this figure intentionally does not
construct a pooled interval or a cross-device ranking.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECTION_SCHEMA = "attnres.compiled_step_hero_projection.v1"
DEFAULT_OUTPUT_DIR = Path("docs/assets")
DEFAULT_SVG_NAME = "compiled_step_hero.svg"
DEFAULT_PNG_NAME = "compiled_step_hero.png"
PNG_DPI = 200
EXPECTED_DEVICES = ("H100 SXM", "B200")
EXPECTED_SEED_COUNT = 3
EXPECTED_MODEL = {
    "layers": 24,
    "width": 1024,
    "heads": 16,
    "ffn": 2816,
    "batch": 2,
    "sequence": 1024,
    "vocab": 32768,
    "block_count": 8,
}


class ProjectionError(ValueError):
    """Raised when the compact hero projection is missing or inconsistent."""


def _median(values: Iterable[float]) -> float:
    """Return a finite sequence median without importing ``statistics``.

    Direct execution puts ``benchmarks/`` ahead of the standard-library path,
    where this repository's ``benchmarks/statistics.py`` shadows the stdlib
    module.  The hero only needs a tiny deterministic median for its three
    already-validated seed estimates, so keeping this helper local makes both
    the script and ``python -m`` entry points behave identically.
    """

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("median requires at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


@dataclass(frozen=True)
class RatioInterval:
    """One unpooled AttnRes/FLA interval for one protocol seed."""

    seed: str
    ratio: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class DeviceMeasurement:
    """The compact, already-audited summary for one physical device."""

    key: str
    attnres_ms: float
    fla_ckpt1_ms: float
    ratios: tuple[RatioInterval, ...]

    @property
    def advantage_pct(self) -> float:
        """Median unpooled seed estimate; positive means lower latency."""

        return 100.0 * (1.0 - _median(item.ratio for item in self.ratios))

    @property
    def advantage_range_pct(self) -> tuple[float, float]:
        """Observed range of the independent seed point estimates."""

        values = tuple(100.0 * (1.0 - item.ratio) for item in self.ratios)
        return min(values), max(values)


@dataclass(frozen=True)
class HeroProjection:
    """Validated projection data consumed by :func:`render_hero`."""

    status: str
    provenance: Mapping[str, Any]
    campaign: Mapping[str, Any]
    devices: tuple[DeviceMeasurement, ...]


def _fail(message: str) -> None:
    raise ProjectionError(message)


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing {missing!r}")
        if extra:
            detail.append(f"unexpected {extra!r}")
        _fail(f"{path} has " + "; ".join(detail))


def _string(value: Any, *, path: str, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        _fail(f"{path} must be a non-empty string")
    return value


def _positive_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        _fail(f"{path} must be finite and positive")
    return number


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{path} must be a positive integer")
    return int(value)


def _read_json(path: Path | str) -> Mapping[str, Any]:
    """Read only the compact projection object.

    ``parse_constant`` rejects NaN and Infinity so a malformed numeric value
    cannot reach Matplotlib as an accidental axis limit.
    """

    path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectionError(f"could not read projection {path}: {exc}") from exc
    return _mapping(payload, path=str(path))


def load_projection(path: Path | str) -> HeroProjection:
    """Load and validate the small auditor projection used by the plotter.

    The exact-key checks are deliberate: raw reports, raw samples, pooled
    intervals, and hidden ranking fields cannot be silently accepted as a
    second input format.
    """

    root = _read_json(path)
    _exact_keys(
        root,
        {"schema", "status", "provenance", "campaign", "devices"},
        path="projection",
    )
    if root.get("schema") != PROJECTION_SCHEMA:
        _fail(f"projection.schema must be {PROJECTION_SCHEMA!r}")
    status = _string(root.get("status"), path="projection.status")
    if status not in {"audited", "fixture"}:
        _fail("projection.status must be 'audited' or 'fixture'")

    provenance = _mapping(root.get("provenance"), path="projection.provenance")
    _exact_keys(
        provenance,
        {"generator", "audit_schema", "audit_status", "source_digest"},
        path="projection.provenance",
    )
    _string(provenance.get("generator"), path="projection.provenance.generator")
    _string(provenance.get("audit_schema"), path="projection.provenance.audit_schema")
    audit_status = _string(
        provenance.get("audit_status"), path="projection.provenance.audit_status"
    )
    _string(provenance.get("source_digest"), path="projection.provenance.source_digest")
    if status == "audited" and audit_status != "passed":
        _fail("audited projection must carry provenance.audit_status='passed'")
    if status == "fixture" and audit_status != "fixture":
        _fail("fixture projection must carry provenance.audit_status='fixture'")

    campaign = _mapping(root.get("campaign"), path="projection.campaign")
    _exact_keys(
        campaign,
        {
            "mode",
            "dtype",
            "rank_relation",
            "timing_method",
            "baseline",
            "optimizer",
            "rounds",
            "warmup",
            "confidence",
            "seeds",
            "schedule",
            "model",
        },
        path="projection.campaign",
    )
    if campaign.get("mode") != "full":
        _fail("projection.campaign.mode must be 'full'")
    if str(campaign.get("dtype", "")).lower().replace("torch.", "") not in {
        "bf16",
        "bfloat16",
    }:
        _fail("projection.campaign.dtype must be BF16")
    if campaign.get("rank_relation") != "R=D":
        _fail("projection.campaign.rank_relation must be 'R=D'")
    if campaign.get("timing_method") != "cuda_graph":
        _fail("projection.campaign.timing_method must be 'cuda_graph'")
    if campaign.get("baseline") != "native FLA Triton checkpoint 1":
        _fail("projection.campaign.baseline must name native FLA Triton checkpoint 1")
    _string(campaign.get("optimizer"), path="projection.campaign.optimizer")
    _positive_int(campaign.get("rounds"), path="projection.campaign.rounds")
    _positive_int(campaign.get("warmup"), path="projection.campaign.warmup")
    if not 0 < float(campaign.get("confidence", 0)) <= 1:
        _fail("projection.campaign.confidence must be in (0, 1]")
    confidence = float(campaign["confidence"])
    if confidence != 0.95:
        _fail("projection.campaign.confidence must be 0.95")
    _string(campaign.get("schedule"), path="projection.campaign.schedule")
    if "raw" in str(campaign["schedule"]).lower():
        _fail("projection.campaign.schedule cannot describe raw report data")

    seeds_value = campaign.get("seeds")
    if not isinstance(seeds_value, Sequence) or isinstance(seeds_value, (str, bytes)):
        _fail("projection.campaign.seeds must be a list")
    if len(seeds_value) != EXPECTED_SEED_COUNT:
        _fail("projection.campaign.seeds must contain exactly three unpooled seeds")
    seeds = tuple(_string(seed, path=f"projection.campaign.seeds[{index}]") for index, seed in enumerate(seeds_value))
    if len(set(seeds)) != EXPECTED_SEED_COUNT:
        _fail("projection.campaign.seeds must be unique")

    model = _mapping(campaign.get("model"), path="projection.campaign.model")
    _exact_keys(model, set(EXPECTED_MODEL), path="projection.campaign.model")
    for field, expected in EXPECTED_MODEL.items():
        observed = _positive_int(model.get(field), path=f"projection.campaign.model.{field}")
        if observed != expected:
            _fail(
                f"projection.campaign.model.{field} must be {expected}, got {observed}"
            )

    devices_value = root.get("devices")
    devices = _mapping(devices_value, path="projection.devices")
    if set(devices) != set(EXPECTED_DEVICES):
        _fail(
            "projection.devices must contain exactly H100 SXM and B200; "
            f"got {tuple(devices)!r}"
        )
    parsed_devices: list[DeviceMeasurement] = []
    for device_key in EXPECTED_DEVICES:
        device = _mapping(devices.get(device_key), path=f"projection.devices.{device_key}")
        _exact_keys(
            device,
            {"absolute_ms", "ratios"},
            path=f"projection.devices.{device_key}",
        )
        absolute = _mapping(
            device.get("absolute_ms"), path=f"projection.devices.{device_key}.absolute_ms"
        )
        _exact_keys(
            absolute,
            {"attnres", "fla_ckpt1"},
            path=f"projection.devices.{device_key}.absolute_ms",
        )
        attnres_ms = _positive_number(
            absolute.get("attnres"),
            path=f"projection.devices.{device_key}.absolute_ms.attnres",
        )
        fla_ckpt1_ms = _positive_number(
            absolute.get("fla_ckpt1"),
            path=f"projection.devices.{device_key}.absolute_ms.fla_ckpt1",
        )
        ratio_values = device.get("ratios")
        if not isinstance(ratio_values, Sequence) or isinstance(ratio_values, (str, bytes)):
            _fail(f"projection.devices.{device_key}.ratios must be a list")
        if len(ratio_values) != EXPECTED_SEED_COUNT:
            _fail(f"projection.devices.{device_key}.ratios must contain three intervals")
        ratios: list[RatioInterval] = []
        for index, ratio_value in enumerate(ratio_values):
            ratio = _mapping(
                ratio_value,
                path=f"projection.devices.{device_key}.ratios[{index}]",
            )
            _exact_keys(
                ratio,
                {"seed", "ratio", "ci_low", "ci_high"},
                path=f"projection.devices.{device_key}.ratios[{index}]",
            )
            seed = _string(
                ratio.get("seed"), path=f"projection.devices.{device_key}.ratios[{index}].seed"
            )
            if seed != seeds[index]:
                _fail(
                    f"projection.devices.{device_key}.ratios[{index}].seed must be {seeds[index]!r}"
                )
            estimate = _positive_number(
                ratio.get("ratio"),
                path=f"projection.devices.{device_key}.ratios[{index}].ratio",
            )
            ci_low = _positive_number(
                ratio.get("ci_low"),
                path=f"projection.devices.{device_key}.ratios[{index}].ci_low",
            )
            ci_high = _positive_number(
                ratio.get("ci_high"),
                path=f"projection.devices.{device_key}.ratios[{index}].ci_high",
            )
            if ci_low > estimate or estimate > ci_high:
                _fail(
                    f"projection.devices.{device_key}.ratios[{index}] must satisfy "
                    "ci_low <= ratio <= ci_high"
                )
            ratios.append(
                RatioInterval(
                    seed=seed,
                    ratio=estimate,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
            )
        parsed_devices.append(
            DeviceMeasurement(
                key=device_key,
                attnres_ms=attnres_ms,
                fla_ckpt1_ms=fla_ckpt1_ms,
                ratios=tuple(ratios),
            )
        )

    return HeroProjection(
        status=status,
        provenance=provenance,
        campaign=campaign,
        devices=tuple(parsed_devices),
    )


def _output_name(value: str, *, suffix: str) -> str:
    name = _string(value, path="output name")
    path = Path(name)
    if path.name != name or path.suffix.lower() != suffix:
        _fail(f"output name must be a plain {suffix} filename")
    return name


def render_hero(
    projection: HeroProjection,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    svg_name: str = DEFAULT_SVG_NAME,
    png_name: str = DEFAULT_PNG_NAME,
) -> tuple[Path, Path]:
    """Render a deterministic, searchable SVG and matching 2x PNG."""

    if not isinstance(projection, HeroProjection):
        raise ProjectionError("render_hero expects a HeroProjection from load_projection")
    svg_name = _output_name(svg_name, suffix=".svg")
    png_name = _output_name(png_name, suffix=".png")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / svg_name
    png_path = output_dir / png_name

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ProjectionError("Matplotlib is required to render the compiled-step hero") from exc

    colors = {
        "background": "#0B1220",
        "candidate": "#2AB7F6",
        "baseline": "#FF8B2C",
        "gain": "#42D6A4",
        "text": "#F7FAFC",
        "muted": "#A8B6C2",
        "grid": "#28384A",
        "spine": "#33465A",
    }
    campaign = projection.campaign
    model = campaign["model"]
    seeds = tuple(campaign["seeds"])
    rounds = int(campaign["rounds"])
    status_line = (
        "Verified current-release campaign"
        if projection.status == "audited"
        else "Fixture projection"
    )

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Avenir", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "attnres-compiled-step-hero-v2",
            "axes.axisbelow": True,
            "figure.facecolor": colors["background"],
            "axes.facecolor": colors["background"],
            "savefig.facecolor": colors["background"],
        }
    ):
        fig, ax = plt.subplots(figsize=(14.0, 8.0), facecolor=colors["background"])
        fig.subplots_adjust(left=0.085, right=0.97, top=0.75, bottom=0.17)

        fig.text(
            0.065,
            0.94,
            "Faster Full AttnRes training",
            ha="left",
            va="top",
            fontsize=30,
            fontweight="bold",
            color=colors["text"],
        )
        fig.text(
            0.065,
            0.865,
            f"{model['layers']} layers · BF16 complete CUDA Graph step · lower is faster",
            ha="left",
            va="top",
            fontsize=15,
            color=colors["muted"],
        )

        centers = [0.0, 1.5]
        width = 0.45
        candidate_values = [item.attnres_ms for item in projection.devices]
        baseline_values = [item.fla_ckpt1_ms for item in projection.devices]
        candidate_bars = ax.bar(
            [x - width / 2 for x in centers],
            candidate_values,
            width,
            label="Fast-AttnRes",
            color=colors["candidate"],
            zorder=3,
        )
        baseline_bars = ax.bar(
            [x + width / 2 for x in centers],
            baseline_values,
            width,
            label="native FLA ckpt1",
            color=colors["baseline"],
            zorder=3,
        )

        ymax = 36.0
        ax.set_ylim(0.0, ymax)
        ax.set_xlim(-0.62, centers[-1] + 0.62)
        ax.set_ylabel("ms / step", fontsize=14, fontweight="bold", color=colors["muted"])
        ax.set_xticks(centers, [item.key for item in projection.devices])
        ax.tick_params(axis="x", labelsize=17, colors=colors["text"], pad=10)
        ax.tick_params(axis="y", labelsize=12, colors=colors["muted"])
        ax.yaxis.grid(True, color=colors["grid"], linewidth=1.0, zorder=0)
        ax.xaxis.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(colors["spine"])

        for bars, values in ((candidate_bars, candidate_values), (baseline_bars, baseline_values)):
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value - ymax * 0.03,
                    f"{value:.2f} ms",
                    ha="center",
                    va="top",
                    fontsize=14,
                    fontweight="bold",
                    color=colors["text"],
                )

        for center, device in zip(centers, projection.devices):
            top = max(device.attnres_ms, device.fla_ckpt1_ms)
            ax.text(
                center,
                top + ymax * 0.075,
                f"{device.advantage_pct:.2f}% faster",
                ha="center",
                va="center",
                fontsize=21,
                fontweight="bold",
                color=colors["gain"],
            )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=2,
            frameon=False,
            fontsize=14,
            handlelength=1.8,
            columnspacing=2.2,
            labelcolor=colors["text"],
        )

        footer = (
            f"{status_line} · L{model['layers']} · B{model['batch']}×T{model['sequence']} · "
            f"D=R={model['width']} · Full S=2…49 · {len(seeds)} seeds × {rounds} paired rounds"
        )
        fig.text(0.065, 0.055, footer, ha="left", va="bottom", fontsize=11.5, color=colors["muted"])

        metadata = {
            "Title": "Current-release Full AttnRes complete-training-step latency",
            "Creator": "benchmarks/plot_compiled_step_hero.py",
            "Description": (
                "Dark grouped bar chart of current Full BF16 captured CUDA Graph "
                "complete-step device time for Fast-AttnRes and native FLA Triton "
                "checkpoint 1 on H100 SXM and B200. Bars are descriptive means of "
                "the three per-seed arm means; seeds and GPUs are not pooled."
            ),
            "Date": None,
        }
        fig.savefig(svg_path, format="svg", metadata=metadata)
        # Keep SVG text searchable and regeneration byte-stable across runs.
        svg_payload = svg_path.read_bytes()
        svg_path.write_bytes(b"\n".join(line.rstrip() for line in svg_payload.splitlines()) + b"\n")
        fig.savefig(
            png_path,
            format="png",
            dpi=PNG_DPI,
            metadata={"Software": "Fast-AttnRes compiled-step evidence"},
        )
        plt.close(fig)
    return svg_path, png_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection",
        required=True,
        type=Path,
        help="small auditor-generated projection JSON; raw reports are not accepted",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for SVG and PNG"
    )
    parser.add_argument("--svg-name", default=DEFAULT_SVG_NAME)
    parser.add_argument("--png-name", default=DEFAULT_PNG_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        projection = load_projection(args.projection)
        svg_path, png_path = render_hero(
            projection,
            args.output_dir,
            svg_name=args.svg_name,
            png_name=args.png_name,
        )
    except ProjectionError as exc:
        print(f"error: {exc}")
        return 2
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PNG_NAME",
    "DEFAULT_SVG_NAME",
    "PROJECTION_SCHEMA",
    "DeviceMeasurement",
    "HeroProjection",
    "ProjectionError",
    "RatioInterval",
    "load_projection",
    "render_hero",
]
