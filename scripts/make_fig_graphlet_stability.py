#!/usr/bin/env python3
"""SI Figure: densification under an independent structural representation.

Two-panel overlay of per-decade temporal-replay statistics computed by the
SAME production pipeline (icsd_graph_time_evolution.compute_temporal_metrics)
on (a) the manuscript's matminer+WL+PCA-32 embedding, and (b) graphlet
histograms (Lesser et al. 2025) on the same ICSD with the same mutual-kNN
k=16 and Louvain resolution=1.0. Only the feature representation differs.

Inputs:
  --prod      notes/graph_time_summary.json (manuscript's Fig.1 source)
  --graphlet  experiments/graphlet_compare/results/graphlet_graph_time_summary.json
  --output    SI figure PNG path
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DECADES = ["1920s", "1930s", "1940s", "1950s", "1960s",
           "1970s", "1980s", "1990s", "2000s", "2010s"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prod", required=True)
    p.add_argument("--graphlet", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def series(d: dict, field: str) -> list[float]:
    return [float(d["by_decade"].get(dec, {}).get(field, float("nan")))
            for dec in DECADES]


def main() -> int:
    a = parse_args()
    prod = json.loads(Path(a.prod).read_text())
    grpl = json.loads(Path(a.graphlet).read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=170, sharex=True)
    x = list(range(len(DECADES)))

    # --- Panel A: community-birth ratio
    ax = axes[0]
    ax.plot(x, series(prod, "cluster_birth_point_ratio"),
            "o-", color="#0f6d61", lw=2.2, ms=7,
            label="manuscript embedding (matminer + WL + PCA-32)")
    ax.plot(x, series(grpl, "cluster_birth_point_ratio"),
            "s--", color="#ff7e2a", lw=2.0, ms=6,
            label="graphlet histograms (Lesser et al. 2025)")
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("community-birth ratio", fontsize=10.5)
    ax.set_title(
        "(a) Per-decade share of entries that open a new structural community",
        fontsize=10.5, fontweight="bold", pad=8)
    ax.grid(True, axis="y", color="#eee", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # annotate the endpoint values
    pb = prod["by_decade"]; gb = grpl["by_decade"]
    ax.annotate(f"prod {pb['1930s']['cluster_birth_point_ratio']*100:.1f}%",
                xy=(1, pb['1930s']['cluster_birth_point_ratio']),
                xytext=(2.0, 0.50), fontsize=8.5, color="#0f6d61",
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#0f6d61"})
    ax.annotate(f"grpl {gb['1930s']['cluster_birth_point_ratio']*100:.1f}%",
                xy=(1, gb['1930s']['cluster_birth_point_ratio']),
                xytext=(2.0, 0.43), fontsize=8.5, color="#ff7e2a",
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#ff7e2a"})
    ax.annotate(
        f"both converge: prod {pb['2010s']['cluster_birth_point_ratio']*100:.1f}% / "
        f"grpl {gb['2010s']['cluster_birth_point_ratio']*100:.1f}%",
        xy=(9, pb['2010s']['cluster_birth_point_ratio']),
        xytext=(5.0, 0.16), fontsize=8.5, color="#333",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#333"})

    # --- Panel B: same-community attachment ratio
    ax = axes[1]
    ax.plot(x, series(prod, "same_community_attachment_ratio"),
            "o-", color="#0f6d61", lw=2.2, ms=7,
            label="manuscript embedding")
    ax.plot(x, series(grpl, "same_community_attachment_ratio"),
            "s--", color="#ff7e2a", lw=2.0, ms=6,
            label="graphlet histograms")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("same-community attachment ratio", fontsize=10.5)
    ax.set_title(
        "(b) Per-decade share attaching to an existing community of the same kind",
        fontsize=10.5, fontweight="bold", pad=8)
    ax.grid(True, axis="y", color="#eee", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    pa, ga = pb["2010s"]["same_community_attachment_ratio"], gb["2010s"]["same_community_attachment_ratio"]
    ax.annotate(f"2010s: prod {pa*100:.1f}% / grpl {ga*100:.1f}%",
                xy=(9, pa), xytext=(5.5, 0.45), fontsize=8.5, color="#333",
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#333"})

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(DECADES, rotation=0)
        ax.set_xlabel("publication decade", fontsize=10.5)

    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(
        "Densification is robust to the structural representation: the temporal "
        "cliff reproduces under an independent graphlet featurization",
        fontsize=12, fontweight="bold", y=1.00)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
