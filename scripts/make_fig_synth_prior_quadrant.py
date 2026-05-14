#!/usr/bin/env python3
"""Synthesizability-prior 2x2 quadrant figure.

Two axes: in-basin/frontier (rows), formula-match/no-match (cols). Each
external source contributes one bubble per quadrant; bubble area ∝
fraction of that source's proposals in that quadrant. Loaded directly
from notes/formula_synth_prior_summary.json.

Note: in-basin classification here is the FULL-MAP classification carried
in the source's frontier-records CSV, NOT the cutoff-calibrated historical
classification used in @tbl:held-out. That distinction must be made
explicit in the caption.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Resolve repo-root anchored paths so this script can be run from any
# working directory (consistent with make_fig_renaissance_validation.py
# and the other make_fig_*.py producers).
REPO = Path(__file__).resolve().parent.parent

SOURCE_ORDER = ["GNoME", "MatterGen", "MP", "JARVIS", "Alexandria"]
SOURCE_LABELS = {
    "GNoME": "GNoME",
    "MatterGen": "MatterGen",
    "MP": "MP-theoretical",
    "JARVIS": "JARVIS-DFT",
    "Alexandria": "Alexandria off-hull",
}
SOURCE_COLORS = {
    "GNoME": "#ff7e2a",
    "MatterGen": "#7a2cad",
    "MP": "#0f6d61",
    "JARVIS": "#1f77b4",
    "Alexandria": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default=str(REPO / "notes" / "formula_synth_prior_summary.json"),
        help="Path to formula_synth_prior_summary.json.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO / "resources" / "figures" / "icsd_densification" / "synth_prior_quadrant.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text())
    sources = summary["sources"]

    # Quadrant grid layout: rows are in-basin / frontier; cols are formula
    # match / no-match. Within each quadrant, sources are placed in a
    # horizontal row so they don't overlap; bubble area ∝ fraction of the
    # source's proposals in that quadrant.
    fig, ax = plt.subplots(figsize=(10.0, 7.0), dpi=180)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.05, 2.05)
    ax.set_aspect("equal")

    # Quadrant background squares
    quad_bg = "#f7f7f7"
    quad_alt = "#efefef"
    rects = [
        # (x, y, w, h, color)
        (0, 1, 1, 1, quad_bg),     # top-left: in-basin & match
        (1, 1, 1, 1, quad_alt),    # top-right: in-basin & no-match
        (0, 0, 1, 1, quad_alt),    # bottom-left: frontier & match
        (1, 0, 1, 1, quad_bg),     # bottom-right: frontier & no-match
    ]
    for x, y, w, h, c in rects:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="white", linewidth=2))

    # Quadrant labels (corners)
    label_kw = dict(fontsize=9, color="#5b6672", style="italic", ha="center")
    ax.text(0.5, 1.93, "in-basin & formula match\n(strongest synth prior)", **label_kw)
    ax.text(1.5, 1.93, "in-basin, no formula match\n(novel chemistry,\nknown structural basin)", **label_kw)
    ax.text(0.5, 0.07, "frontier & formula match\n(known composition,\nnovel structural variant)", **label_kw)
    ax.text(1.5, 0.07, "frontier, no formula match\n(weakest synth prior;\nmost exploratory)", **label_kw)

    # Within each quadrant, place 5 source bubbles in a row at fixed y.
    # Bubble area scaling: pick a max-area constant so the largest bubble
    # is comfortable within a quadrant.
    # Bubble-area scaling. Documented in paper/supporting_information.md §S1.5.
    #
    # max_area_pts = the area in matplotlib points^2 that corresponds to a
    # fraction of 1.0 (i.e. 100% of the source's proposals in this quadrant).
    # The largest actual fraction in the data is 75.68% (GNoME ib&no-match);
    # at 6500 × 0.7568 ≈ 4900 pt^2 this bubble sits comfortably inside its
    # quadrant without overlapping the next source. 6500 was chosen as the
    # smallest value for which all middle-magnitude fractions (MP, JARVIS,
    # Alexandria) sit at resolvable sizes without crowding.
    max_area_pts = 6500.0
    label_offset_y = -0.08

    for src_i, src in enumerate(SOURCE_ORDER):
        rec = sources[src]
        n = rec["n"]
        q = rec["quadrants"]
        # in-basin & match : (0.5, 1.5)
        # in-basin & no-match : (1.5, 1.5)
        # frontier & match : (0.5, 0.5)
        # frontier & no-match : (1.5, 0.5)
        fracs = {
            "ib_match": q["in_basin_and_formula_match"] / n,
            "ib_nomatch": q["in_basin_and_no_formula_match"] / n,
            "fr_match": q["frontier_and_formula_match"] / n,
            "fr_nomatch": q["frontier_and_no_formula_match"] / n,
        }
        # Position within each quadrant: distribute sources horizontally.
        # x within the quadrant (0..1) for source i out of N: (i+1)/(N+1).
        x_off = (src_i + 1) / (len(SOURCE_ORDER) + 1)  # 0..1
        positions = {
            "ib_match": (0.0 + x_off, 1.5),
            "ib_nomatch": (1.0 + x_off, 1.5),
            "fr_match": (0.0 + x_off, 0.5),
            "fr_nomatch": (1.0 + x_off, 0.5),
        }
        color = SOURCE_COLORS[src]
        for key, (x, y) in positions.items():
            frac = fracs[key]
            if frac <= 0:
                # Draw a tiny X marker for "0 in this quadrant"
                ax.plot(x, y, marker="x", color=color, markersize=6, alpha=0.45)
                continue
            # 18.0 pt^2 minimum bubble area: floor below which a bubble
            # would shrink to a single pixel and become unreadable. Zeros
            # are handled separately above (X marker), so this floor only
            # affects positive fractions below ~0.28%. Trade-off discussed
            # in TODO.md §B10. The current data does not trigger it.
            area = max(frac * max_area_pts, 18.0)
            ax.scatter([x], [y], s=area, color=color, alpha=0.72,
                       edgecolors="white", linewidths=0.8, zorder=5)
            # Annotate fraction. For small bubbles (frac < 5%) the label
            # doesn't fit inside the circle, so place it above the bubble
            # in the source's color and a small font; for larger bubbles
            # keep the label centered inside in white for contrast.
            if frac < 0.05:
                # Distance above the bubble scales with bubble radius
                radius = (area / 3.14159) ** 0.5  # in points
                # Convert points to data coords approximately: the axis spans
                # ~1 data unit per ~0.4× quadrant width. Use a small fixed
                # offset that works for our axis range (0..2 in data coords).
                offset_y = 0.03 + radius / 700.0
                ax.text(x, y + offset_y, f"{frac*100:.1f}%",
                        ha="center", va="bottom", fontsize=7.5,
                        color=color, zorder=6, fontweight="bold")
            else:
                ax.text(x, y, f"{frac*100:.1f}%",
                        ha="center", va="center", fontsize=7,
                        color="white", zorder=6, fontweight="bold")

    # Axis labels (the actual quadrant axes)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["formula ∈ post-1980 ICSD", "formula ∉ post-1980 ICSD"], fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["frontier", "in-basin"], fontsize=10, rotation=90, va="center")
    ax.tick_params(left=False, bottom=False)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)

    # Legend: source bubbles
    legend_handles = []
    for src in SOURCE_ORDER:
        legend_handles.append(plt.scatter([], [], s=120, color=SOURCE_COLORS[src], alpha=0.72,
                                          edgecolors="white", linewidths=0.8,
                                          label=SOURCE_LABELS[src]))
    ax.legend(handles=legend_handles, loc="center", bbox_to_anchor=(0.5, -0.06),
              ncol=5, frameon=False, fontsize=8.5, handletextpad=0.5)

    ax.set_title(
        "Synthesizability prior: in-basin × formula-match quadrant",
        fontsize=12, fontweight="bold", pad=14,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
