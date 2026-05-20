#!/usr/bin/env python3
"""Render the year-shuffle null vs observed community-birth ratio.

For each decade, the observed birth ratio is compared against a
distribution of birth ratios obtained by randomly permuting publication
years within the featurized ICSD set while keeping the final community
labels fixed. The figure shows observed (red), the null mean (grey
line), and the null 5–95% band (grey shading).

The interpretation flips around the 1960s/1970s: in the early decades
the observed birth ratio is *below* the null (more densification than
chance under a year-shuffle), while in the 1970s-2010s the observed
ratio is *above* the null (the literature still birthed more new basins
than random year-assignment, even as the absolute share collapsed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from frontier_common import require_zenodo_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary-json", required=True, help="temporal_null_summary.json")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = require_zenodo_file(
        args.summary_json,
        what="year-shuffle null summary driving the temporal-null figure",
    )
    summary = json.loads(summary_path.read_text())
    by_decade = summary["by_decade"]
    decades = list(by_decade.keys())
    decade_year = [int(d[:-1]) for d in decades]

    obs = np.array([by_decade[d]["observed_birth_ratio"] for d in decades])
    null_mean = np.array([by_decade[d]["shuffle_mean_birth_ratio"] for d in decades])
    null_p05 = np.array([by_decade[d]["shuffle_p05_birth_ratio"] for d in decades])
    null_p95 = np.array([by_decade[d]["shuffle_p95_birth_ratio"] for d in decades])

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=180)
    ax.fill_between(decade_year, null_p05, null_p95, color="#cccccc", alpha=0.7,
                    label="year-shuffle null (5–95% over 200 shuffles)")
    ax.plot(decade_year, null_mean, color="#666", linestyle="--", linewidth=1.0, label="null mean")
    ax.plot(decade_year, obs, color="#8a2f2f", linewidth=2.4, label="observed")

    # Shade where observed is OUTSIDE the null band, on either side, so the
    # eye sees both regimes (densification before ~1960s, frontier-pushing
    # after ~1970s).
    below = obs < null_p05
    above = obs > null_p95
    for i, x in enumerate(decade_year):
        if below[i]:
            ax.scatter([x], [obs[i]], s=80, color="#8a2f2f", marker="v", zorder=5,
                       label="observed below null (densification)" if i == np.where(below)[0][0] else None)
        if above[i]:
            ax.scatter([x], [obs[i]], s=80, color="#0f6d61", marker="^", zorder=5,
                       label="observed above null (more births than chance)" if i == np.where(above)[0][0] else None)

    ax.set_xlabel("Publication decade")
    ax.set_ylabel("community-birth ratio")
    ax.set_xticks(decade_year)
    ax.set_xticklabels([f"{y}s" for y in decade_year])
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", color="#eee", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "Observed community-birth ratio vs year-shuffle null",
        fontsize=12, fontweight="bold", pad=8,
    )
    ax.text(
        0.01, -0.18,
        "Densification claim is statistically supported in the early decades "
        "(observed below null), and replaced by frontier-pushing in the late decades "
        "(observed above null) once absolute birth share has collapsed. "
        "Null preserves community sizes and decade counts; only publication years are shuffled.",
        transform=ax.transAxes, fontsize=8.5, color="#5b6672", wrap=True,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=9, frameon=False)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
