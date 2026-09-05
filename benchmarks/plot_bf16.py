"""Render the source-verified common-rank operator summary for the README."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

from benchmarks.bf16_broader_report import validate_contract
from benchmarks.bf16_report import RANKS, SEEDS

BG, FG, MUTED = "#0b1320", "#edf2f7", "#a9b5c1"
BLUE, ORANGE, GRID = "#22b5e8", "#ff892b", "#263544"


def render(summary, output):
    validate_contract(summary["contract"])
    if (summary["scope"] != "primary_operator_confirmation" or summary["missing"]
            or summary["admission_failures"]):
        raise ValueError("figure requires complete, admitted common-rank operator evidence")
    cells = summary["cells"]
    groups = {}
    for cell in cells:
        groups.setdefault((cell["gpu"], cell["sources"], cell["rank"]), []).append(cell)
    expected = {(gpu, sources, rank) for gpu in ("H100", "B200")
                for sources in (9, 49) for rank in RANKS}
    if (set(groups) != expected or any(len(rows) != len(SEEDS)
            or {row["seed"] for row in rows} != set(SEEDS) for rows in groups.values())):
        raise ValueError("figure requires each seed exactly once in every geometry")

    output.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "figure.facecolor": BG, "axes.facecolor": BG,
             "text.color": FG, "axes.labelcolor": MUTED, "xtick.color": MUTED,
             "ytick.color": MUTED, "axes.edgecolor": GRID, "font.size": 11,
             "svg.hashsalt": "fast-attnres-bf16"}
    source = summary["candidate_identity"][:12]

    def decorate(ax):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)

    def save(fig, name):
        for suffix in ("svg", "png"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180,
                        metadata={"Description": f"Operator evidence; source {source}"}
                        if suffix == "svg" else {"Description": f"Source {source}"})
        plt.close(fig)

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.subplots_adjust(left=.10, right=.96, bottom=.18, top=.74)
        fig.text(.08, .90, "BF16 Attention Residuals", fontsize=26, weight="bold")
        fig.text(.08, .84, "Full read + backward · 49 sources · D=R=1536 · lower is faster",
                 color=MUTED, fontsize=13, weight="bold")
        for index, gpu in enumerate(("H100", "B200")):
            rows = groups[gpu, 49, 1536]
            heights = [1000 * median(row[key] for row in rows)
                       for key in ("candidate_ms", "alternative_ms")]
            for offset, height, color, label in zip((-.18, .18), heights, (BLUE, ORANGE),
                    ("Fast-AttnRes", "Strongest correct alternative")):
                ax.bar(index + offset, height, .36, color=color, label=label if index == 0 else None)
                ax.text(index + offset, height * .94, f"{height:.1f} µs", ha="center",
                        va="top", weight="bold", color="white", fontsize=12)
            ratio = median(row["ratio"] for row in rows)
            annotation = (f"{100 * (1-ratio):.2f}% lower latency"
                          if all(row["faster_pass"] for row in rows)
                          else "Speedup not established")
            ax.text(index, max(heights) * 1.06, annotation, ha="center", weight="bold",
                    color="#35d2a5" if all(row["faster_pass"] for row in rows) else MUTED)
        ax.set_xticks([0, 1], ["H100 SXM", "B200"], fontsize=14, weight="bold")
        ax.set_ylabel("µs / read + backward")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.12), ncol=2, frameon=False)
        decorate(ax)
        fig.text(.08, .08, f"8192 tokens · 3 seeds × 120 paired rounds · source {source}",
                 color=MUTED, fontsize=10, weight="bold")
        fig.text(.08, .045, "Bars: median seed means. Operator timing; complete training steps are reported separately.",
                 color=MUTED, fontsize=9)
        save(fig, "bf16_hero")

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        fig.subplots_adjust(left=.08, right=.97, bottom=.13, top=.86, hspace=.30, wspace=.22)
        fig.suptitle("Common-rank BF16 latency", fontsize=25, weight="bold", y=.96)
        for column, gpu in enumerate(("H100", "B200")):
            for row_index, sources in enumerate((9, 49)):
                ax = axes[row_index, column]
                for key, color, label in (("candidate_ms", BLUE, "Fast-AttnRes"),
                        ("alternative_ms", ORANGE, "Strongest correct alternative")):
                    values = [[1000 * row[key] for row in groups[gpu, sources, rank]] for rank in RANKS]
                    ax.plot(range(len(RANKS)), list(map(median, values)), color=color,
                            marker="o", markersize=4, label=label)
                    ax.fill_between(range(len(RANKS)), list(map(min, values)), list(map(max, values)),
                                    color=color, alpha=.15)
                ax.set_title(f"{gpu} · {sources} sources", weight="bold")
                ax.set_ylabel("µs / read + backward")
                ax.set_xticks(range(len(RANKS)), RANKS, rotation=45, fontsize=9)
                decorate(ax)
                if row_index == 1:
                    ax.set_xlabel("Routing rank R (decreasing →)")
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.text(.08, .04, f"D=1536 · 8192 tokens · median and range across 3 seeds · 120 rounds per seed · source {source}",
                 color=MUTED, fontsize=10)
        save(fig, "bf16_ranks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    render(json.loads(args.summary.read_text()), args.output)


if __name__ == "__main__":
    main()
