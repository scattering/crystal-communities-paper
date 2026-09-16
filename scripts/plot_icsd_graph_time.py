#!/usr/bin/env python3
"""Time-evolution plots for ICSD graph community summaries.

Visual fixes vs the previous version:
  * Two-tier line weighting -- the dominant ratios (existing-community,
    same-community) are drawn thick; secondary ratios (cross, bridge, birth,
    outlier) are drawn thin so the eye reads the headline first.
  * Legend moves to the right margin instead of floating over the data.
  * `recent_decades` companion plot now reuses the same styling.
  * Fonts and grid colors match the rest of the manuscript figures.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


# ----------------------------------------------------------------- arg parse
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--recent-only",
        action="store_true",
        help="Also emit `icsd_graph_recent_decades.png` covering 1980s onward.",
    )
    return parser.parse_args()


def decade_sort_key(decade: str) -> tuple[int, str]:
    if decade == "unknown":
        return (10**9, decade)
    try:
        return (int(decade[:-1]), decade)
    except Exception:
        return (10**8, decade)


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    color: str
    weight: str  # "primary" | "secondary"


PRIMARY_COLOR = "#1f3a68"
SECONDARY_COLORS = {
    "cluster_birth_point_ratio": "#c44e52",
    "outlier_ratio": "#937860",
    "bridge_attachment_ratio": "#dd8452",
    "cross_community_attachment_ratio": "#8172b2",
}


def _setup_axes(ax) -> None:
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of structures in decade")
    ax.set_xlabel("publication decade")
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_series(ax, decades, by_decade, series: list[Series]) -> None:
    for s in series:
        values = [by_decade[d].get(s.key, 0.0) for d in decades]
        if s.weight == "primary":
            ax.plot(decades, values, marker="o", linewidth=2.8, markersize=6.5, color=s.color, label=s.label, zorder=4)
        else:
            ax.plot(decades, values, marker="o", linewidth=1.4, markersize=4.5, color=s.color, label=s.label, alpha=0.9, zorder=3)


def _legend_right(ax, fig) -> None:
    fig.subplots_adjust(right=0.78)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
        handlelength=2.2,
    )


def plot_ratios(decades: list[str], by_decade: dict[str, dict[str, float]], out: Path, title: str) -> None:
    series = [
        Series("existing_cluster_ratio", "existing-community attachment", PRIMARY_COLOR, "primary"),
        Series("same_community_attachment_ratio", "same-community attachment", "#3f8f6c", "primary"),
        Series("cross_community_attachment_ratio", "cross-community attachment", SECONDARY_COLORS["cross_community_attachment_ratio"], "secondary"),
        Series("bridge_attachment_ratio", "bridge attachment", SECONDARY_COLORS["bridge_attachment_ratio"], "secondary"),
        Series("cluster_birth_point_ratio", "community birth", SECONDARY_COLORS["cluster_birth_point_ratio"], "secondary"),
        Series("outlier_ratio", "outlier", SECONDARY_COLORS["outlier_ratio"], "secondary"),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.4), dpi=180)
    _setup_axes(ax)
    _plot_series(ax, decades, by_decade, series)
    ax.set_title(title, fontsize=12, pad=8)
    _legend_right(ax, fig)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_core_periphery(decades: list[str], by_decade: dict[str, dict[str, float]], out: Path) -> None:
    series = [
        Series("same_community_attachment_ratio", "same-community attachment", PRIMARY_COLOR, "primary"),
        Series("core_attachment_ratio", "core attachment", "#3f8f6c", "secondary"),
        Series("periphery_attachment_ratio", "periphery attachment", "#dd8452", "secondary"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=180)
    _setup_axes(ax)
    _plot_series(ax, decades, by_decade, series)
    ax.set_title("ICSD graph attachment geometry by decade", fontsize=12, pad=8)
    _legend_right(ax, fig)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_stacked(decades: list[str], by_decade: dict[str, dict[str, float]], out: Path) -> None:
    birth = [by_decade[d].get("n_cluster_birth_point", 0.0) for d in decades]
    existing = [by_decade[d].get("n_existing_cluster", 0.0) for d in decades]
    outlier = [by_decade[d].get("n_outlier", 0.0) for d in decades]

    x = range(len(decades))
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=180)
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar(x, existing, color=PRIMARY_COLOR, label="existing community")
    ax.bar(x, birth, bottom=existing, color=SECONDARY_COLORS["cluster_birth_point_ratio"], label="community birth")
    ax.bar(x, outlier, bottom=[a + b for a, b in zip(existing, birth)], color=SECONDARY_COLORS["outlier_ratio"], label="outlier")
    ax.set_xticks(list(x))
    ax.set_xticklabels(decades)
    ax.set_xlabel("publication decade")
    ax.set_ylabel("number of structures")
    ax.set_title("ICSD graph community outcome counts by decade", fontsize=12, pad=8)
    _legend_right(ax, fig)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_decade = summary["by_decade"]
    decades = sorted([d for d in by_decade if d != "unknown"], key=decade_sort_key)

    plot_ratios(decades, by_decade, out_dir / "graph_time_ratios.png", "ICSD graph community time evolution by decade")
    plot_core_periphery(decades, by_decade, out_dir / "graph_time_core_periphery.png")
    plot_stacked(decades, by_decade, out_dir / "graph_time_stacked.png")
    if args.recent_only:
        recent = [d for d in decades if d >= "1980s"]
        if recent:
            plot_ratios(recent, by_decade, out_dir / "icsd_graph_recent_decades.png", "Recent ICSD graph dynamics (1980s-2010s)")
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
