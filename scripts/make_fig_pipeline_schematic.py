#!/usr/bin/env python3
"""Pipeline-overview schematic for Extended Data Figure 1.

Trunk: licensed ICSD CIFs → per-site features → Weisfeiler–Lehman → pool
→ PCA frozen basis. The frozen basis forks into two analysis tracks:
ICSD self-comparison (Louvain → temporal replay) and external-source
projection (in-basin classification). Same geometry feeds both halves.

Each box has two lines: a technical/method title (bold) and a short
plain-English subtitle (italic, same color). The subtitle is what makes
the schematic legible to a reader who hasn't read Methods. Style mirrors
the other paper figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


DPI = 180
FIG_W = 11.5
FIG_H = 9.6

BOX_FACE = "#fff8ef"
BOX_EDGE = "#5b6672"
BOX_LW = 1.4

TITLE_COLOR = "#1f2530"

LEFT_COLOR = "#0f6d61"      # ICSD self-comparison branch
RIGHT_COLOR = "#b56200"     # external-projection branch
HIGHLIGHT = "#8a2f2f"       # FROZEN BASIS — the fork point


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="resources/figures/icsd_densification/pipeline_schematic.png",
    )
    return p.parse_args()


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    *,
    face: str = BOX_FACE,
    edge: str = BOX_EDGE,
    lw: float = BOX_LW,
    title_color: str = TITLE_COLOR,
    title_fontsize: float = 12.0,
    subtitle_fontsize: float = 9.5,
) -> None:
    """Two-line box: bold title, italic subtitle in the box edge color."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.20",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x, y + h * 0.20, title,
        ha="center", va="center",
        fontsize=title_fontsize, fontweight="bold",
        color=title_color, zorder=3,
    )
    ax.text(
        x, y - h * 0.22, subtitle,
        ha="center", va="center",
        fontsize=subtitle_fontsize,
        color=title_color, alpha=0.78, style="italic",
        zorder=3,
    )


def add_arrow(
    ax,
    p_from: tuple[float, float],
    p_to: tuple[float, float],
    *,
    color: str = BOX_EDGE,
    lw: float = 1.7,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    arrow = FancyArrowPatch(
        p_from, p_to,
        arrowstyle="-|>",
        mutation_scale=16,
        color=color,
        linewidth=lw,
        connectionstyle=connectionstyle,
        zorder=1,
    )
    ax.add_patch(arrow)


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 13.5)
    ax.set_aspect("equal")
    ax.axis("off")

    BX = 7.0
    BW = 5.4
    BH = 1.20
    GAP = 1.55  # center-to-center

    Y_ICSD = 12.55
    Y_FEAT = Y_ICSD - GAP
    Y_WL = Y_FEAT - GAP
    Y_POOL = Y_WL - GAP
    Y_PCA = Y_POOL - GAP

    add_box(ax, BX, Y_ICSD, BW, BH,
            "ICSD CIFs",
            "167,500 inorganic crystals, dated by publication year")

    add_box(ax, BX, Y_FEAT, BW, BH,
            "Per-site features  (68-d)",
            "chemistry + local geometry, one vector per atomic site")

    add_box(ax, BX, Y_WL, BW, BH,
            "Weisfeiler–Lehman  (×3)",
            "three rounds of neighbor-info mixing on the bond graph")

    add_box(ax, BX, Y_POOL, BW, BH,
            "Structure pool  (213-d)",
            "aggregate sites + lattice into one vector per crystal")

    add_box(ax, BX, Y_PCA, BW + 0.6, BH + 0.10,
            "PCA(32)   —   frozen basis",
            "32-d compression, fixed once and used for every comparison",
            face="#fbecec", edge=HIGHLIGHT, lw=2.0, title_color=HIGHLIGHT)

    for y_from, y_to in [
        (Y_ICSD - BH / 2, Y_FEAT + BH / 2),
        (Y_FEAT - BH / 2, Y_WL + BH / 2),
        (Y_WL - BH / 2, Y_POOL + BH / 2),
        (Y_POOL - BH / 2, Y_PCA + (BH + 0.10) / 2 + 0.04),
    ]:
        add_arrow(ax, (BX, y_from - 0.04), (BX, y_to + 0.04))

    # ── Fork ──
    LX = 3.4
    RX = 10.6

    Y_BR1 = Y_PCA - 1.65
    Y_BR2 = Y_BR1 - 1.55
    Y_BR3 = Y_BR2 - 1.55

    BR_W = 5.0
    BR_H = 1.15

    add_arrow(
        ax,
        (BX - 1.6, Y_PCA - (BH + 0.10) / 2 - 0.02),
        (LX, Y_BR1 + BR_H / 2 + 0.04),
        color=LEFT_COLOR, lw=1.9,
        connectionstyle="angle,angleA=-90,angleB=180,rad=0",
    )
    add_arrow(
        ax,
        (BX + 1.6, Y_PCA - (BH + 0.10) / 2 - 0.02),
        (RX, Y_BR1 + BR_H / 2 + 0.04),
        color=RIGHT_COLOR, lw=1.9,
        connectionstyle="angle,angleA=-90,angleB=0,rad=0",
    )

    # Left branch — ICSD self-comparison
    add_box(ax, LX, Y_BR1, BR_W, BR_H,
            "Mutual k-NN  +  Louvain",
            "build similarity graph, detect communities",
            edge=LEFT_COLOR, lw=1.7, title_color=LEFT_COLOR)
    add_box(ax, LX, Y_BR2, BR_W, BR_H,
            "Temporal replay",
            "reorder communities by publication year, track growth",
            edge=LEFT_COLOR, lw=1.7, title_color=LEFT_COLOR)
    add_box(ax, LX, Y_BR3, BR_W, BR_H,
            "Figs.  1  &  2",
            "structural memory of experimental discovery",
            face="#eef6f4", edge=LEFT_COLOR, lw=1.7, title_color=LEFT_COLOR)

    add_arrow(ax, (LX, Y_BR1 - BR_H / 2 - 0.04), (LX, Y_BR2 + BR_H / 2 + 0.04), color=LEFT_COLOR)
    add_arrow(ax, (LX, Y_BR2 - BR_H / 2 - 0.04), (LX, Y_BR3 + BR_H / 2 + 0.04), color=LEFT_COLOR)

    # Right branch — external-source projection
    add_box(ax, RX, Y_BR1, BR_W, BR_H,
            "Project 5 external samples",
            "GNoME, MatterGen, MP, JARVIS, Alexandria",
            edge=RIGHT_COLOR, lw=1.7, title_color=RIGHT_COLOR)
    add_box(ax, RX, Y_BR2, BR_W, BR_H,
            "In-basin classification",
            "is each computed structure inside a known basin?",
            edge=RIGHT_COLOR, lw=1.7, title_color=RIGHT_COLOR)
    add_box(ax, RX, Y_BR3, BR_W, BR_H,
            "Figs.  3  &  4",
            "calibrated comparison of computed proposals",
            face="#fbf2e7", edge=RIGHT_COLOR, lw=1.7, title_color=RIGHT_COLOR)

    add_arrow(ax, (RX, Y_BR1 - BR_H / 2 - 0.04), (RX, Y_BR2 + BR_H / 2 + 0.04), color=RIGHT_COLOR)
    add_arrow(ax, (RX, Y_BR2 - BR_H / 2 - 0.04), (RX, Y_BR3 + BR_H / 2 + 0.04), color=RIGHT_COLOR)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
