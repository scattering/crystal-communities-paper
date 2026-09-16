#!/usr/bin/env python3
"""REVISED (npj review 2026-08) "temporal cliff" stacked-area figure (Fig. 1).

Derived from scripts/make_fig_temporal_cliff.py (companion-repo producer).
Differences from the original:

  * Reads the MUTUALLY EXCLUSIVE per-decade partition in
    notes/review_2026_08/fig1_exclusive_by_decade.csv (built by
    notes/review_2026_08/compute_fig1_exclusive.py from
    notes/node_temporal_events.csv) instead of the overlapping
    *_ratio fields of graph_time_summary.json.  The five bands are now a
    true partition of each decade's entries, so NO renormalisation is done
    (the script asserts each decade column sums to 1).
  * Font sizes >= 11 pt throughout.
  * Annotates the community-birth share and the within-community share in
    the 1930s and 2010s with the exact values from the CSV.
  * Same colours, legend labels and stacking order as the original
    (outlier bottom -> community birth top).

Bands (bottom -> top):
  outlier             graph-filtered outlier (community < 0)
  bridge entry        enters an existing community; active mutual-kNN
                      neighbours in >= 2 other communities
  cross-community     enters an existing community; active neighbours in
                      exactly 1 other community
  within-community    enters an existing community; no active neighbour in
                      another community. This includes entries with no active
                      graph neighbour.
  community birth     entry's year == earliest year of its community
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exclusive-csv", required=True, help="fig1_exclusive_by_decade.csv")
    p.add_argument("--output", required=True, help="output png path")
    p.add_argument("--start-decade", default="1910s")
    p.add_argument("--end-decade", default="2010s")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.exclusive_csv).open(newline="") as fh:
        table = {row["decade"]: row for row in csv.DictReader(fh)}

    decades = sorted(
        d for d in table
        if d != "unknown" and args.start_decade <= d <= args.end_decade
    )
    decade_year = [int(d[:-1]) for d in decades]

    # category -> per-decade EXCLUSIVE share. Ordered bottom->top so birth
    # ends on top and is the last band the eye sees. Colours/labels are
    # identical to the original producer.
    cats = [
        ("share_outlier", "outlier", "#cccccc"),
        ("share_bridge", "bridge entry", "#b56200"),
        ("share_cross", "cross-community entry", "#9b6db6"),
        ("share_same", "within-community entry", "#0f6d61"),
        ("share_birth", "community birth", "#8a2f2f"),
    ]
    rows = np.array([[float(table[d][field]) for d in decades] for field, _, _ in cats])

    # The five classes are mutually exclusive and exhaustive; refuse to
    # renormalise -- if this fails the CSV is wrong.
    col_sums = rows.sum(axis=0)
    if not np.allclose(col_sums, 1.0, atol=1e-6):
        raise SystemExit(f"exclusive shares do not sum to 1 per decade: {col_sums}")

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    fig, ax = plt.subplots(figsize=(11.0, 5.8), dpi=args.dpi)
    ax.stackplot(
        decade_year, rows,
        labels=[label for _, label, _ in cats],
        colors=[color for _, _, color in cats],
        alpha=0.92,
    )

    ax.set_xlim(decade_year[0], decade_year[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Publication decade")
    ax.set_ylabel("Share of decade's ICSD entries")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(
        "Historical growth of structural communities",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xticks(decade_year)
    ax.set_xticklabels([f"{y}s" for y in decade_year], rotation=0)
    ax.grid(True, axis="y", color="white", linewidth=0.7, alpha=0.6)

    # ---- annotations: exact exclusive values for 1930s and 2010s ----
    cum = np.cumsum(rows, axis=0)            # top edge of each band
    bottom = cum - rows                      # bottom edge of each band
    idx = {field: i for i, (field, _, _) in enumerate(cats)}
    di = {d: i for i, d in enumerate(decades)}

    def band_mid(field: str, decade: str) -> float:
        i, j = idx[field], di[decade]
        return 0.5 * (bottom[i, j] + cum[i, j])

    def share(field: str, decade: str) -> float:
        return 100.0 * rows[idx[field], di[decade]]

    bbox = {"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#444", "lw": 0.6, "alpha": 0.95}
    arrow = {"arrowstyle": "-|>", "color": "#222", "lw": 0.9, "shrinkA": 0, "shrinkB": 2}
    anno_kw = {"fontsize": 11, "color": "#111", "bbox": bbox, "arrowprops": arrow}

    for dec, x_txt, y_birth_txt, y_same_txt, ha in (
        ("1930s", 1942, 0.86, 0.30, "left"),
        ("2010s", 2007, 0.905, 0.56, "right"),
    ):
        if dec not in di:
            continue
        x = int(dec[:-1])
        ax.annotate(
            f"community birth {share('share_birth', dec):.1f}% ({dec})",
            xy=(x, band_mid("share_birth", dec)), xytext=(x_txt, y_birth_txt),
            ha=ha, va="center", **anno_kw,
        )
        ax.annotate(
            f"within-community {share('share_same', dec):.1f}% ({dec})",
            xy=(x, band_mid("share_same", dec)), xytext=(x_txt, y_same_txt),
            ha=ha, va="center", **anno_kw,
        )

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.11),
        ncol=5, frameon=False,
    )
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size} bytes; {len(decades)} decades)")
    for dec in ("1930s", "2010s"):
        if dec in di:
            print(f"  {dec}: birth {share('share_birth', dec):.1f}%  same {share('share_same', dec):.1f}%  "
                  f"cross {share('share_cross', dec):.1f}%  bridge {share('share_bridge', dec):.1f}%  "
                  f"outlier {share('share_outlier', dec):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
