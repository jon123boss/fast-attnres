"""Plot the audited quarter-rank and full-rank Fast-AttnRes measurements."""

from pathlib import Path
from statistics import median

from .plot_compiled_step_sweep import DARK_ARM_COLORS, _style


def rank_comparisons(records, contract, gpu):
    """Keep each seed's paired interval separate from the displayed bar summary."""
    rows = []
    for cell in contract["cells"]:
        width = cell["model"]["width"]
        if cell["ranks"] != [width, width // 4] or width % 4:
            raise ValueError("quarter-rank plots require exactly R=D and R=D/4")
        selected = [r for r in records if r["gpu"] == gpu and r["cell"] == cell["name"]]
        if [r["seed"] for r in selected] != cell["seeds"]:
            raise ValueError(f"incomplete rank comparison: {gpu}/{cell['name']}")
        key = f"kernel_rank_{width // 4}_over_rank_{width}"
        seeds = [{"seed": r["seed"], "source_sha256": r["sha256"],
                  "standard_ms": r["means_ms"][f"kernel_rank_{width}"],
                  "quarter_ms": r["means_ms"][f"kernel_rank_{width // 4}"],
                  "paired": r["statistics"][key]} for r in selected]
        rows.append({"gpu": gpu, "cell": cell["name"], "model": cell["model"],
                     "rounds": cell["rounds"], "seeds": seeds,
                     "standard_ms": median(r["standard_ms"] for r in seeds),
                     "quarter_ms": median(r["quarter_ms"] for r in seeds),
                     "lower_time_pct": 100 * (1 - median(r["paired"]["ratio"] for r in seeds))})
    return rows


def render_rank_comparison(rows, output_dir, gpu):
    """Use the README's dark chart style with one matched workload per panel."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not rows or any(r["gpu"] != gpu for r in rows):
        raise ValueError("supply rank comparisons from one device")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    face, text, muted = "#0B1220", "#F7FAFC", "#A8B6C2"
    colors = [DARK_ARM_COLORS["attnres"], "#A78BFA"]
    height = (len(rows) + 1 + 2) // 3
    with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif":
            ["Avenir Next", "Avenir", "Helvetica", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none", "svg.hashsalt": "attnres-quarter-rank-v1",
            "figure.facecolor": face, "savefig.facecolor": face}):
        fig, axes = plt.subplots(height, 3, figsize=(17, 1.6 + 4.6 * height), squeeze=False)
        fig.subplots_adjust(left=.06, right=.98, top=.79, bottom=.1, hspace=.6, wspace=.27)
        fig.text(.055, .95, f"{gpu} · quarter-rank comparison", color=text,
                 fontsize=26, fontweight="bold", va="top")
        fig.text(.055, .89, "Fast-AttnRes R=D/4 versus R=D · BF16 complete CUDA Graph step",
                 color=muted, fontsize=15, va="top")
        for ax, row in zip(axes.flat, rows):
            _style(ax, dark=True)
            model = row["model"]
            width = model["width"]
            full = model["mode"] == "full"
            sources = 2 * model["layers"] + 1 if full else model["block_count"] + 1
            schedule = "Full" if full else "Block"
            block = "" if full else f" · bs={2 * model['layers'] // model['block_count']}"
            ax.set_title(f"{schedule} · L{model['layers']} · D={width}\n"
                         f"B{model['batch']}×T{model['sequence']}{block} · Smax={sources}",
                         loc="left", color=text, fontsize=14, fontweight="bold", pad=14)
            values = [row["standard_ms"], row["quarter_ms"]]
            ceiling = max(values) * 1.75
            ax.bar([0, 1], values, width=.6, color=colors, zorder=2)
            for index, value in enumerate(values):
                ax.text(index, value + .025 * ceiling, f"{value:.3f} ms", ha="center",
                        color=text, fontsize=13, fontweight="bold")
            intervals = [(100 * (1 - s["paired"]["ci_high"]),
                          100 * (1 - s["paired"]["ci_low"])) for s in row["seeds"]]
            color = (DARK_ARM_COLORS["liger"] if min(lo for lo, hi in intervals) > 0
                     else DARK_ARM_COLORS["fail"] if max(hi for lo, hi in intervals) < 0 else muted)
            change = row["lower_time_pct"]
            label = ("0.0% change" if abs(change) < .05
                     else f"{abs(change):.1f}% {'lower' if change >= 0 else 'higher'} time")
            ax.text(.5, .96, label, transform=ax.transAxes, ha="center", va="top",
                    color=color, fontsize=14, fontweight="bold")
            for index, (seed, (lo, hi)) in enumerate(zip(row["seeds"], intervals)):
                ax.text(.5, .87 - .055 * index,
                        f"{seed['seed']} · 95% CI {lo:+.1f}% to {hi:+.1f}%",
                        transform=ax.transAxes, ha="center", va="top", fontsize=10, color=muted)
            ax.set_ylim(0, ceiling)
            ax.set_xticks([0, 1], [f"R=D\n{width}", f"R=D/4\n{width // 4}"],
                          color=text, fontsize=12, fontweight="bold")
            ax.set_ylabel("ms / complete step", color=text, fontsize=11)
        notes = list(axes.flat)[len(rows)]
        notes.axis("off")
        notes.text(.02, .96, "Same Fast-AttnRes backend", color=text, fontsize=16,
                   fontweight="bold", va="top")
        notes.text(.02, .84,
                   "Full-width values and output.\nOnly the routing rank changes.\n\n"
                   "Bars: median per-seed mean time.\n"
                   "Change: median paired estimate.\n"
                   "Each seed retains its 95% interval.\n"
                   "Intervals are simultaneous within a seed.\n"
                   "Positive percentages mean less time.\n\n"
                   "L24 Full: 3 seeds × 120 paired rounds.\n"
                   "Other workloads: 1 seed × 40 paired rounds.",
                   color=muted, fontsize=12, linespacing=1.5, va="top")
        for ax in list(axes.flat)[len(rows) + 1:]:
            ax.axis("off")
        fig.legend([Patch(color=c) for c in colors], ["Fast-AttnRes · R=D", "Fast-AttnRes · R=D/4"],
                   loc="lower center", bbox_to_anchor=(.5, .025), ncol=2,
                   frameon=False, labelcolor=text, prop={"size": 14, "weight": "bold"})
        paths = tuple(out / f"rank_comparison_{gpu.lower()}.{ext}" for ext in ("svg", "png"))
        for path in paths:
            fig.savefig(path, dpi=160, facecolor=face, metadata={"Creator": __name__})
        plt.close(fig)
    return paths
