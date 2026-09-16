#!/usr/bin/env python3
"""Make the "temporal cliff" stacked-area figure that combines what used to
be Fig 1 (HDBSCAN densification by decade) and Fig 2 (graph-time ratios).

Reads the per-decade graph-time summary JSON and emits a 100% stacked-area
chart over decades. Each band is a category of the discovery event:

  community_birth     structures that open a brand-new community
  same_community      attach to a community by joining its same-community core
  cross_community     attach across a community boundary
  bridge              attach by bridging two previously separate communities
  outlier             classified as outlier (no community)

Reading the figure: the "community_birth" band collapses across the century.
That visible cliff is the headline of the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time-json", required=True, help="graph_time_summary.json")
    p.add_argument("--output", required=True, help="output png path")
    p.add_argument("--start-decade", default="1910s")
    p.add_argument("--end-decade", default="2020s")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(Path(args.time_json).read_text())
    by_decade = summary["by_decade"]

    decades = sorted(d for d in by_decade if args.start_decade <= d <= args.end_decade)
    decade_year = [int(d[:-1]) for d in decades]

    # category → per-decade ratio. Ordered bottom→top so birth ends on top
    # and is the last band the eye sees.
    cats = [
        ("outlier_ratio", "outlier", "#cccccc"),
        ("bridge_attachment_ratio", "bridge attachment", "#b56200"),
        ("cross_community_attachment_ratio", "cross-community attachment", "#9b6db6"),
        ("same_community_attachment_ratio", "same-community attachment", "#0f6d61"),
        ("cluster_birth_point_ratio", "community birth", "#8a2f2f"),
    ]

    rows = np.array([
        [float(by_decade[d].get(field, 0.0)) for d in decades]
        for field, _, _ in cats
    ])
    # renormalise so each decade-column sums to 1; the JSON ratios are derived
    # from disjoint event counts but rounding leaves tiny gaps.
    col_sums = rows.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    rows = rows / col_sums[None, :]

    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=180)
    ax.stackplot(
        decade_year, rows,
        labels=[label for _, label, _ in cats],
        colors=[color for _, _, color in cats],
        alpha=0.92,
    )

    ax.set_xlim(decade_year[0], decade_year[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Publication decade")
    ax.set_ylabel("Share of decade's structures")
    ax.set_title(
        "The end of structural exploration: what changes from one decade to the next",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xticks(decade_year)
    ax.set_xticklabels([f"{y}s" for y in decade_year], rotation=0)
    ax.grid(True, axis="y", color="white", linewidth=0.7, alpha=0.6)

    # Inline annotations so the eye can read the cliff without consulting
    # the legend. Place birth-share label on the topmost band at its
    # widest point.
    birth_row = rows[-1]  # last row in stack = top band
    if len(birth_row) > 0:
        i_max = int(np.argmax(birth_row))
        x_anno = decade_year[i_max]
        # vertical position is mid-band: 1 - half of birth_row[i_max]
        y_anno = 1.0 - 0.5 * birth_row[i_max]
        ax.annotate(
            f"community-birth share peaks at {birth_row[i_max] * 100:.0f}% in the {x_anno}s",
            xy=(x_anno, y_anno),
            xytext=(decade_year[len(decade_year) // 4], 0.55),
            arrowprops={"arrowstyle": "-", "color": "#444", "lw": 0.8},
            fontsize=9, color="#222",
        )
        x_last = decade_year[-1]
        y_last_birth = 1.0 - 0.5 * birth_row[-1]
        ax.annotate(
            f"→ {birth_row[-1] * 100:.1f}% by the {x_last}s",
            xy=(x_last, y_last_birth),
            xytext=(x_last - 24, 0.97),
            arrowprops={"arrowstyle": "-", "color": "#444", "lw": 0.8},
            fontsize=9, color="#222",
        )

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        ncol=5, frameon=False, fontsize=9,
    )
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size} bytes; {len(decades)} decades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
