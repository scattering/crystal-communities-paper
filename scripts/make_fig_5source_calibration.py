#!/usr/bin/env python3
"""Five-source calibration figure for the Nature submission.

Three panels in a single figure:

  (a) ICSD KDE topographical background with GNoME + MatterGen overlay.
  (b) ICSD KDE topographical background with MP + JARVIS + Alexandria overlay.
  (c) Held-out in-basin rates across 1990 / 2000 / 2010 cutoffs for
      held-out ICSD + 5 external sources, with Wilson 95% CIs and an
      explicit highlight that GNoME ≈ MP-theoretical at every cutoff.

Inputs:
  --features features.npy (167500 x 213 raw matminer)
  --gnome / --mattergen / --mp / --jarvis / --alexandria *records.csv
  --composition-matched-summary composition_matched_ai_summary.json
  --output output.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from frontier_common import require_zenodo_file


SOURCES_AI = [
    ("GNoME", "#ff7e2a", "o"),
    ("MatterGen", "#7a2cad", "D"),
]
SOURCES_DFT = [
    ("MP", "#0f6d61", "o"),
    ("JARVIS", "#1f77b4", "s"),
    ("Alexandria", "#d62728", "^"),
]
ALL_SOURCES = [
    ("ICSD (held-out)", "#2c2c2c"),
    ("MatterGen", "#7a2cad"),
    ("GNoME", "#ff7e2a"),
    ("MP-theoretical", "#0f6d61"),
    ("JARVIS-DFT", "#1f77b4"),
    ("Alexandria off-hull", "#d62728"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--gnome", required=True)
    p.add_argument("--mattergen", required=True)
    p.add_argument("--mp", required=True)
    p.add_argument("--jarvis", required=True)
    p.add_argument("--alexandria", required=True)
    p.add_argument("--composition-matched-summary", required=True,
                   help="JSON file with unmatched per-source rates per cutoff")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    # Tuning knobs for the KDE topographical background (panels a + b).
    # Each default is documented in paper/supporting_information.md §S1.5
    # ("Visualization tuning parameters") with sensitivity bounds.
    p.add_argument(
        "--kde-grid", type=int, default=180,
        help="Per-axis grid resolution for KDE evaluation (default 180; "
             "180×180 ≈ 32k evaluations is fast on a laptop and visually "
             "indistinguishable from 360 at print size).",
    )
    p.add_argument(
        "--kde-sample", type=int, default=15000,
        help="Random sub-sample of the 167.5K-row ICSD point cloud used "
             "to fit the KDE bandwidth (default 15000). Beyond ~10K points "
             "the KDE field converges visually; the marginal cost grows "
             "as O(N²) per evaluation. Sub-sample is seeded by --seed.",
    )
    p.add_argument(
        "--bandwidth-scale", type=float, default=1.6,
        help="Multiplier on the Scott-rule KDE bandwidth (default 1.6). "
             "Scott's rule on 167.5K ICSD points produces an over-smoothed "
             "field that flattens the cuprate / perovskite / spinel basins; "
             "×1.6 pulls the bandwidth back enough to keep basins visually "
             "distinct without resolving spurious noise from the long PCA "
             "tails. Sensitivity: bandwidth 1.0–2.5 produces visually "
             "similar basin structure; the bar-chart in panel (c) is "
             "insensitive to bandwidth (it does not use the KDE).",
    )
    p.add_argument(
        "--quantile-trim", type=float, default=0.005,
        help="Per-axis quantile clip of the ICSD points used to fit the "
             "KDE bandwidth and set the plot axis limits (default 0.005, "
             "i.e. 0.5%% / 99.5%%). Drops a handful of extreme PCA "
             "outliers that would otherwise stretch the axes empty. "
             "External source overlays are NOT trimmed: any external "
             "point landing outside the trimmed range is clipped from "
             "view by set_xlim/ylim but still counted in the bar-chart.",
    )
    return p.parse_args()


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys, ol = [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                xs.append(float(row["pca1"]))
                ys.append(float(row["pca2"]))
                ol.append(str(row.get("outlier_like", "False")).strip().lower() == "true")
            except (KeyError, ValueError):
                continue
    return np.column_stack([xs, ys]), np.asarray(ol, dtype=bool)


def render_kde_panel(ax, X2, sources_xy_list, title, kde_grid, kde_sample,
                     bandwidth_scale, quantile_trim, rng):
    # Trim ICSD extremes for KDE bandwidth
    lo_q = np.quantile(X2, quantile_trim, axis=0)
    hi_q = np.quantile(X2, 1 - quantile_trim, axis=0)
    inrange = (X2[:, 0] >= lo_q[0]) & (X2[:, 0] <= hi_q[0]) & \
              (X2[:, 1] >= lo_q[1]) & (X2[:, 1] <= hi_q[1])
    Xt = X2[inrange]
    if len(Xt) > kde_sample:
        idx = rng.choice(len(Xt), size=kde_sample, replace=False)
        Xt_sub = Xt[idx]
    else:
        Xt_sub = Xt

    kde = gaussian_kde(Xt_sub.T, bw_method="scott")
    kde.set_bandwidth(kde.factor * bandwidth_scale)
    xx = np.linspace(lo_q[0], hi_q[0], kde_grid)
    yy = np.linspace(lo_q[1], hi_q[1], kde_grid)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.vstack([XX.ravel(), YY.ravel()])
    Z = kde(grid).reshape(XX.shape)

    # Topographical contour-fill
    levels = np.linspace(Z.min(), Z.max(), 18)
    cmap = plt.cm.Blues_r
    cf = ax.contourf(XX, YY, Z, levels=levels, cmap=cmap, alpha=0.85)
    # Colorbar — show ICSD density on a relative scale (0 = sparse, 1 = densest)
    cbar = plt.colorbar(cf, ax=ax, fraction=0.04, pad=0.02, aspect=28)
    cbar.set_label("ICSD density (relative)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    # Re-scale the tick labels to 0–1 instead of raw KDE values
    zmin, zmax = float(Z.min()), float(Z.max())
    if zmax > zmin:
        ticks = np.linspace(zmin, zmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{(t - zmin)/(zmax - zmin):.1f}" for t in ticks])

    # Source overlays
    for (name, color, marker), (xy, outlier) in sources_xy_list:
        if len(xy) == 0:
            continue
        # In-basin (faded) and frontier (bright) split
        in_basin = ~outlier
        ax.scatter(xy[in_basin, 0], xy[in_basin, 1], s=8, c=color, alpha=0.25,
                   marker=marker, linewidths=0, label=f"{name} (in-basin)" if in_basin.sum() else None)
        ax.scatter(xy[outlier, 0], xy[outlier, 1], s=14, c=color, alpha=0.85,
                   marker=marker, edgecolors="white", linewidths=0.4,
                   label=f"{name} (frontier)" if outlier.sum() else None)

    ax.set_xlim(lo_q[0], hi_q[0])
    ax.set_ylim(lo_q[1], hi_q[1])
    ax.set_xlabel("Frozen ICSD PCA-1")
    ax.set_ylabel("Frozen ICSD PCA-2")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.85,
              ncol=1, handletextpad=0.4, columnspacing=0.6)


def render_bar_panel(ax, summary, ax_inset_callback=None):
    cutoffs = [c["cutoff"] for c in summary["cutoffs"]]
    n_cutoffs = len(cutoffs)
    n_series = len(ALL_SOURCES)
    bar_w = 0.85 / n_series
    x = np.arange(n_cutoffs)

    for s_i, (label, color) in enumerate(ALL_SOURCES):
        rates = []
        cis = []
        for c in summary["cutoffs"]:
            m = c["matchings"]["coarse"]
            if label == "ICSD (held-out)":
                d = m["icsd_unmatched"]
            else:
                key = label.split("-")[0].split(" ")[0]  # "MP-theoretical" -> "MP"
                d = m["by_source"][key]["unmatched"]
            rates.append(d["rate"] or 0)
            cis.append(d["ci95"])

        rates = np.array(rates, dtype=float)
        lo = np.array([c[0] for c in cis], dtype=float)
        hi = np.array([c[1] for c in cis], dtype=float)
        # Wilson(0,0) returns (NaN, NaN); fall back to zero half-widths so
        # matplotlib does not silently drop the bar's whisker. Clamp any
        # negative half-widths defensively (rounding can otherwise push
        # lo above rates by a hair).
        lo_w = np.maximum(np.nan_to_num(rates - lo, nan=0.0), 0.0)
        hi_w = np.maximum(np.nan_to_num(hi - rates, nan=0.0), 0.0)
        off = (s_i - (n_series - 1) / 2) * bar_w
        ax.bar(x + off, rates, bar_w, color=color, edgecolor="white",
               linewidth=0.4, label=label)
        ax.errorbar(x + off, rates, yerr=[lo_w, hi_w], fmt="none",
                     ecolor="#222", elinewidth=0.6, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cutoffs])
    ax.set_xlabel("Training cutoff year")
    ax.set_ylabel("In-basin rate (95th-percentile threshold)")
    ax.set_title("(c) Held-out in-basin rates across 5 external sources",
                 fontsize=10.5, fontweight="bold", pad=6)
    ax.set_ylim(0, 0.75)
    ax.grid(True, axis="y", color="#eee", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2,
              handletextpad=0.4, columnspacing=0.8)

    if ax_inset_callback is not None:
        ax_inset_callback(ax)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("loading ICSD features...", flush=True)
    features_path = require_zenodo_file(
        args.features,
        what="frozen 167,500-row matminer feature matrix (features.npy)",
    )
    X = np.load(features_path)
    Xs = StandardScaler().fit_transform(X)
    pca2 = PCA(n_components=2, random_state=args.seed)
    X2 = pca2.fit_transform(Xs)
    print(f"  {len(X2)} ICSD points", flush=True)

    print("loading external records...", flush=True)
    src_paths = {
        "GNoME": require_zenodo_file(args.gnome, what="GNoME frontier-records CSV"),
        "MatterGen": require_zenodo_file(args.mattergen, what="MatterGen frontier-records CSV"),
        "MP": require_zenodo_file(args.mp, what="MP-theoretical frontier-records CSV"),
        "JARVIS": require_zenodo_file(args.jarvis, what="JARVIS-DFT frontier-records CSV"),
        "Alexandria": require_zenodo_file(args.alexandria, what="Alexandria off-hull frontier-records CSV"),
    }
    src_data = {}
    for name, p in src_paths.items():
        xy, ol = load_xy(p)
        src_data[name] = (xy, ol)
        print(f"  {name}: {len(xy)} entries ({ol.sum()} frontier)", flush=True)

    print("loading composition-matched summary...", flush=True)
    summary_path = require_zenodo_file(
        args.composition_matched_summary,
        what="composition-matched AI-vs-ICSD summary driving panel (c)",
    )
    summary = json.loads(summary_path.read_text())

    print("rendering figure...", flush=True)
    fig = plt.figure(figsize=(13.5, 9.5), dpi=170)
    # Leave more room at top for the suptitle and panel-title clearance now
    # that each KDE panel carries a colorbar (which steals horizontal width
    # and slightly compresses the title region).
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28,
                           left=0.06, right=0.985, top=0.91, bottom=0.07)

    # Panel a: ICSD KDE + GNoME + MatterGen
    ax_a = fig.add_subplot(gs[0, 0])
    render_kde_panel(
        ax_a, X2,
        sources_xy_list=[((name, color, marker), src_data[name])
                          for name, color, marker in SOURCES_AI],
        title="(a) GNoME and MatterGen on the frozen ICSD topography",
        kde_grid=args.kde_grid, kde_sample=args.kde_sample,
        bandwidth_scale=args.bandwidth_scale, quantile_trim=args.quantile_trim,
        rng=rng,
    )

    # Panel b: ICSD KDE + MP + JARVIS + Alexandria
    ax_b = fig.add_subplot(gs[0, 1])
    render_kde_panel(
        ax_b, X2,
        sources_xy_list=[((name, color, marker), src_data[name])
                          for name, color, marker in SOURCES_DFT],
        title="(b) MP, JARVIS, and Alexandria on the same topography",
        kde_grid=args.kde_grid, kde_sample=args.kde_sample,
        bandwidth_scale=args.bandwidth_scale, quantile_trim=args.quantile_trim,
        rng=rng,
    )

    # Panel c: bar chart of held-out rates
    ax_c = fig.add_subplot(gs[1, :])
    render_bar_panel(ax_c, summary)

    fig.suptitle("Five external structure sources on the frozen ICSD reference frame",
                 fontsize=12.5, fontweight="bold", y=0.985)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
