#!/usr/bin/env python3
"""KDE topographical map of the human ICSD structural landscape with GNoME +
MatterGen proposals overlaid as bright dots.

The human ICSD background is rendered as a 2D Gaussian-KDE contour plot in
PCA-1 / PCA-2 space — the "topography" of the human-densified region. AI
proposals are dropped on top, classified in-basin (faded teal) vs frontier
(bright orange/red), so the eye can immediately read who is sitting in the
gravity wells and who is climbing out into the high-elevation frontier.

Inputs (all live under the same frozen PCA used by analyze_gnome_frontier):
  --features features_pca.npy
  --gnome    gnome_frontier_records.csv  (columns: pca1, pca2, outlier_like)
  --mattergen mattergen_frontier_records.csv (same schema)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--gnome", required=True)
    p.add_argument("--mattergen", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kde-grid", type=int, default=240)
    p.add_argument("--kde-sample", type=int, default=20000,
                   help="random subsample of ICSD points used for the KDE estimator (full 167K is too slow and adds little)")
    p.add_argument("--bandwidth-scale", type=float, default=1.6,
                   help="multiplier on Scott's rule bandwidth; >1 spreads the contours so the topography reads")
    p.add_argument("--quantile-trim", type=float, default=0.005,
                   help="quantile to trim from each end when fitting KDE; 167K-point ICSD has extreme outliers that compress the topography")
    return p.parse_args()


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xy, outlier_like) from a *_frontier_records.csv."""
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


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("loading ICSD features...", flush=True)
    X = np.load(args.features)
    Xs = StandardScaler().fit_transform(X)
    pca2 = PCA(n_components=2, random_state=args.seed)
    X2 = pca2.fit_transform(Xs)
    print(f"  {len(X2)} ICSD points in PCA-2", flush=True)

    print("loading AI records...", flush=True)
    gn_xy, gn_outlier = load_xy(Path(args.gnome))
    mg_xy, mg_outlier = load_xy(Path(args.mattergen))
    print(f"  {len(gn_xy)} GNoME ({gn_outlier.sum()} frontier), "
          f"{len(mg_xy)} MatterGen ({mg_outlier.sum()} frontier)", flush=True)

    # Trim ICSD extremes BEFORE fitting the KDE so the bandwidth isn't pulled
    # out by a handful of distant points; the trimmed quantile is also the
    # display window. AI points are still allowed to fall outside the trimmed
    # window — they're shown as overlays on the same axes.
    q_lo = np.quantile(X2, args.quantile_trim, axis=0)
    q_hi = np.quantile(X2, 1.0 - args.quantile_trim, axis=0)
    in_window = (
        (X2[:, 0] >= q_lo[0]) & (X2[:, 0] <= q_hi[0])
        & (X2[:, 1] >= q_lo[1]) & (X2[:, 1] <= q_hi[1])
    )
    X2_kde = X2[in_window]
    print(f"  trimmed to {len(X2_kde)} ICSD points (dropped {len(X2) - len(X2_kde)} extreme outliers)", flush=True)

    ai_xy = np.vstack([gn_xy, mg_xy])
    x_lo = float(min(q_lo[0], np.quantile(ai_xy[:, 0], 0.01)))
    x_hi = float(max(q_hi[0], np.quantile(ai_xy[:, 0], 0.99)))
    y_lo = float(min(q_lo[1], np.quantile(ai_xy[:, 1], 0.01)))
    y_hi = float(max(q_hi[1], np.quantile(ai_xy[:, 1], 0.99)))

    print(f"fitting KDE on ICSD background (bandwidth scale = {args.bandwidth_scale})...", flush=True)
    sample_idx = rng.choice(len(X2_kde), size=min(args.kde_sample, len(X2_kde)), replace=False)
    sample = X2_kde[sample_idx]
    kde = gaussian_kde(sample.T)
    # Scott's rule produces a peaky estimator on tightly-clustered data;
    # widening the bandwidth makes the basin topography visible at the scale
    # where AI proposals actually sit.
    kde.set_bandwidth(bw_method=kde.factor * args.bandwidth_scale)
    xx, yy = np.meshgrid(
        np.linspace(x_lo, x_hi, args.kde_grid),
        np.linspace(y_lo, y_hi, args.kde_grid),
    )
    grid = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(grid).reshape(xx.shape)
    print(f"  KDE on {len(sample)} sample points evaluated on a {args.kde_grid}x{args.kde_grid} grid", flush=True)

    fig, ax = plt.subplots(figsize=(10.0, 7.5), dpi=180)

    # Render density on a log scale so both the gravity wells AND the
    # surrounding low-elevation territory are visible. Pure linear scaling
    # (used in the first pass) collapses everything except the central peak
    # into the lowest contour band.
    log_density = np.log10(density + 1e-12)
    log_density -= log_density.max()  # 0 = peak, more negative = quieter

    # Filled topography (the "human gravity wells"). Use a sequential
    # green/teal palette so the AI orange/red overlays read as foreign.
    levels = np.linspace(log_density.min(), 0.0, 14)
    ax.contourf(xx, yy, log_density, levels=levels, cmap="YlGnBu", alpha=0.92)
    # Crisp contour lines on top to pull the topography out of the fill
    ax.contour(xx, yy, log_density, levels=levels[::2], colors="#2c4a55", linewidths=0.5, alpha=0.6)

    # AI overlays — small dots, low alpha for in-basin, brighter for frontier.
    # Plot in-basin first so frontier dots sit on top.
    ax.scatter(
        gn_xy[~gn_outlier, 0], gn_xy[~gn_outlier, 1],
        s=4, c="#b56200", alpha=0.18, edgecolors="none",
        label=f"GNoME, in-basin (n = {(~gn_outlier).sum()})", zorder=3,
    )
    ax.scatter(
        mg_xy[~mg_outlier, 0], mg_xy[~mg_outlier, 1],
        s=6, c="#0f6d61", alpha=0.40, edgecolors="none",
        label=f"MatterGen, in-basin (n = {(~mg_outlier).sum()})", zorder=4,
    )
    ax.scatter(
        gn_xy[gn_outlier, 0], gn_xy[gn_outlier, 1],
        s=10, c="#ff5f1f", alpha=0.85, edgecolors="#4a1d00", linewidths=0.3,
        label=f"GNoME, frontier (n = {gn_outlier.sum()})", zorder=5,
    )
    ax.scatter(
        mg_xy[mg_outlier, 0], mg_xy[mg_outlier, 1],
        s=18, c="#7a2cad", alpha=0.95, edgecolors="white", linewidths=0.5,
        marker="D",
        label=f"MatterGen, frontier (n = {mg_outlier.sum()})", zorder=6,
    )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Frozen ICSD PCA-1")
    ax.set_ylabel("Frozen ICSD PCA-2")
    ax.set_title(
        "Human gravity wells (KDE topography) with AI proposals overlaid",
        fontsize=12, fontweight="bold", pad=10,
    )

    # Legend in the bottom-right with white background to keep it readable on
    # the dark KDE region.
    leg = ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.92, edgecolor="#888")
    leg.get_frame().set_linewidth(0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#fdfcfa")

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
