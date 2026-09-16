#!/usr/bin/env python3
"""Render the revised five-cohort comparison from repaired analysis outputs.

Panels (a) and (b) show the first two saved full-production ICSD PCA
coordinates with equal-size, seed-42 external overlays (386 records per
source by default). Quantitative basin assignments use all 32 dimensions.
Panel (c) reads the independently cutoff-trained retrospective summary,
uses every successfully projected record, and plots the source rates and
Wilson intervals without recomputing their statistics. The fixed display
order groups the experimental reference and source types; it is not a
ranking of in-basin rates.

The white end of the Blues_r background denotes high ICSD density; dark
blue denotes low density. Legends sit outside the overlay panels, and all
outputs are written as both PNG and SVG. --dump-rates records the exact
bar heights and interval endpoints for comparison with the source table.

Preferred inputs:
  --features-pca saved full-production PCA coordinates (167392 x 32)
  --gnome / --mattergen / --mp / --jarvis / --alexandria repaired records.csv
  --composition-matched-summary composition_matched_ai_summary.json
  --rates-json cutoff_trained_retrospective_summary.json
  --output output.png

--features accepts raw features only as a legacy fallback when saved PCA
coordinates are unavailable; current production figures use --features-pca.
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
    ("GNoME", "#ff7e2a"),
    ("MatterGen", "#7a2cad"),
    ("MP-theoretical", "#0f6d61"),
    ("JARVIS-DFT", "#1f77b4"),
    ("Alexandria off-hull", "#d62728"),
]
# Fixed order in which the per-source subsamples are drawn from the
# dedicated subsample rng (so the draw is reproducible independent of
# dict ordering).
SUBSAMPLE_ORDER = ["GNoME", "MatterGen", "MP", "JARVIS", "Alexandria"]

# Font sizes (points) -- reviewer request: everything legible at print size.
FS_SUPTITLE = 15
FS_TITLE = 13
FS_LABEL = 12
FS_TICK = 11
FS_LEGEND = 10
FS_CBAR_LABEL = 11
FS_CBAR_TICK = 10
FS_FOOTNOTE = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", help="Raw ICSD features (legacy fallback when saved PCA is unavailable).")
    p.add_argument("--features-pca", help="Saved production PCA coordinates; plot the first two columns directly.")
    p.add_argument("--gnome", required=True)
    p.add_argument("--mattergen", required=True)
    p.add_argument("--mp", required=True)
    p.add_argument("--jarvis", required=True)
    p.add_argument("--alexandria", required=True)
    p.add_argument("--composition-matched-summary", required=True,
                   help="JSON file with unmatched per-source rates per cutoff")
    p.add_argument(
        "--rates-json", default=None,
        help="Optional JSON that drives panel (c). Accepts either the legacy "
             "cutoffs[].matchings layout or the cutoff-trained retrospective "
             "summary produced by analyze_cutoff_trained_retrospective.py.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the KDE-fit ICSD sub-sample rng (unchanged "
                        "from the original script; keeps the background identical).")
    p.add_argument("--dpi", type=int, default=200)
    # --- revision knobs -------------------------------------------------
    p.add_argument(
        "--subsample-n", type=int, default=386,
        help="Equal per-source random subsample size for the overlay in "
             "panels (a),(b). Default 386 = the full public MatterGen set, "
             "so MatterGen is shown in full and every other source is "
             "thinned to the same count. 0 disables subsampling (plots "
             "every record, as the original figure did).",
    )
    p.add_argument(
        "--subsample-seed", type=int, default=42,
        help="Seed for the DEDICATED subsample rng (separate from --seed so "
             "the KDE background is not perturbed).",
    )
    p.add_argument(
        "--dump-rates", default=None,
        help="Optional path; writes a JSON list of the exact bar heights and "
             "CI whisker ends that panel (c) draws.",
    )
    p.add_argument(
        "--no-footnote", action="store_true",
        help="Suppress the one-line subsample-rule footnote under the figure.",
    )
    # Tuning knobs for the KDE topographical background (panels a + b).
    # Each default is documented in the Supporting Information
    # §S1.9 ("Visualization and survey parameters").
    # UNCHANGED from the original script.
    p.add_argument(
        "--kde-grid", type=int, default=180,
        help="Per-axis grid resolution for KDE evaluation (default 180; "
             "180×180 ≈ 32k evaluations is fast on a laptop and visually "
             "indistinguishable from 360 at print size).",
    )
    p.add_argument(
        "--kde-sample", type=int, default=15000,
        help="Random sub-sample of the ICSD point cloud used "
             "to fit the KDE bandwidth (default 15000). Beyond ~10K points "
             "the KDE field converges visually; the marginal cost grows "
             "as O(N²) per evaluation. Sub-sample is seeded by --seed.",
    )
    p.add_argument(
        "--bandwidth-scale", type=float, default=1.6,
        help="Multiplier on the Scott-rule KDE bandwidth (default 1.6). "
             "Values above one smooth the density field more strongly. "
             "Sensitivity: bandwidth 1.0–2.5 produces visually "
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
    args = p.parse_args()
    if not args.features_pca and not args.features:
        p.error("provide --features-pca (preferred) or --features")
    return args


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


def subsample_sources(src_data: dict, n: int, seed: int) -> dict:
    """Equal-size random subsample per source for the overlay panels.

    Rule: for each source in SUBSAMPLE_ORDER, if it has more than `n`
    records, draw `n` row indices without replacement from a dedicated
    np.random.default_rng(seed); otherwise keep every record. Plain
    (unstratified) sampling. Returns {name: (xy, outlier)} with the same
    structure as the input. n <= 0 returns the input unchanged.
    """
    if n <= 0:
        print("subsample: disabled (plotting all records)", flush=True)
        return src_data
    rng = np.random.default_rng(seed)
    out = {}
    print(f"subsample rule: n = {n} per source, seed = {seed}, "
          f"draw order = {SUBSAMPLE_ORDER}", flush=True)
    for name in SUBSAMPLE_ORDER:
        xy, ol = src_data[name]
        total = len(xy)
        if total > n:
            idx = np.sort(rng.choice(total, size=n, replace=False))
            xy_s, ol_s = xy[idx], ol[idx]
            how = "random subsample"
        else:
            xy_s, ol_s = xy, ol
            how = "all records (<= n)"
        out[name] = (xy_s, ol_s)
        full_frac = ol.sum() / max(total, 1)
        sub_frac = ol_s.sum() / max(len(xy_s), 1)
        print(f"  {name:<11s} {len(xy_s):>4d}/{total:<5d} shown ({how}); "
              f"frontier fraction full={full_frac:.3f} shown={sub_frac:.3f}",
              flush=True)
    return out


def render_kde_panel(ax, X2, sources_xy_list, title, kde_grid, kde_sample,
                     bandwidth_scale, quantile_trim, rng, legend_ncol):
    # ---- KDE background: IDENTICAL to the original script -------------
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

    # Topographical contour-fill (Blues_r: white = densest, deep blue = sparse)
    levels = np.linspace(Z.min(), Z.max(), 18)
    cmap = plt.cm.Blues_r
    cf = ax.contourf(XX, YY, Z, levels=levels, cmap=cmap, alpha=0.85)
    # Colorbar — show ICSD density on a relative scale (0 = sparse, 1 = densest)
    cbar = plt.colorbar(cf, ax=ax, fraction=0.04, pad=0.02, aspect=28)
    cbar.set_label("ICSD density (relative)", fontsize=FS_CBAR_LABEL)
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK)
    # Re-scale the tick labels to 0–1 instead of raw KDE values
    zmin, zmax = float(Z.min()), float(Z.max())
    if zmax > zmin:
        ticks = np.linspace(zmin, zmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{(t - zmin)/(zmax - zmin):.1f}" for t in ticks])

    # ---- Source overlays (equal-size subsample, larger edged markers) --
    for (name, color, marker), (xy, outlier) in sources_xy_list:
        if len(xy) == 0:
            continue
        # In-basin (softer) and frontier (bright) split. Both carry a white
        # edge so blue/green fills stay visible on the dark low-density
        # ground; the frontier edge is heavier.
        in_basin = ~outlier
        ax.scatter(xy[in_basin, 0], xy[in_basin, 1], s=22, c=color, alpha=0.55,
                   marker=marker, edgecolors="white", linewidths=0.4,
                   label=f"{name} (in-basin)" if in_basin.sum() else None)
        ax.scatter(xy[outlier, 0], xy[outlier, 1], s=34, c=color, alpha=0.92,
                   marker=marker, edgecolors="white", linewidths=0.8,
                   label=f"{name} (frontier)" if outlier.sum() else None)

    ax.set_xlim(lo_q[0], hi_q[0])
    ax.set_ylim(lo_q[1], hi_q[1])
    ax.set_xlabel("Frozen ICSD PCA-1", fontsize=FS_LABEL)
    ax.set_ylabel("Frozen ICSD PCA-2", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=8)
    # Legend OUTSIDE the axes, centred below the x-label, so it can never
    # occlude data (the original upper-right in-axes legend sat on top of
    # the densest part of the overlay).
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    fontsize=FS_LEGEND, frameon=False, ncol=legend_ncol,
                    handletextpad=0.5, columnspacing=1.4, markerscale=1.4,
                    borderaxespad=0.0)
    for h in leg.legend_handles:
        try:
            h.set_alpha(1.0)
        except Exception:
            pass


def render_bar_panel(ax, summary, ax_inset_callback=None):
    """Render panel (c) and return every plotted rate and interval."""
    cutoffs = [c["cutoff"] for c in summary["cutoffs"]]
    n_cutoffs = len(cutoffs)
    n_series = len(ALL_SOURCES)
    bar_w = 0.85 / n_series
    x = np.arange(n_cutoffs)
    drawn = []

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
               linewidth=0.5, label=label)
        ax.errorbar(x + off, rates, yerr=[lo_w, hi_w], fmt="none",
                     ecolor="#222", elinewidth=0.9, capsize=3)
        for cutoff, r, l_, h_ in zip(cutoffs, rates, rates - lo_w, rates + hi_w):
            drawn.append({"source": label, "cutoff": cutoff, "rate": float(r),
                          "ci_lo": float(l_), "ci_hi": float(h_)})

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cutoffs])
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel("Historical cutoff year", fontsize=FS_LABEL)
    ax.set_ylabel("In-basin rate (95th-percentile threshold)", fontsize=FS_LABEL)
    ax.set_title("(c) In-basin rates in independently trained historical maps",
                 fontsize=FS_TITLE, fontweight="bold", pad=8)
    # Leave headroom above the source rates for the one-row legend.
    ax.set_ylim(0, 0.85)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Single row across the top of the axes, above the tallest bar.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=FS_LEGEND,
              frameon=False, ncol=n_series, handletextpad=0.5,
              columnspacing=1.6, borderaxespad=0.2)

    if ax_inset_callback is not None:
        ax_inset_callback(ax)
    return drawn


def normalize_rate_summary(summary: dict) -> dict:
    """Convert the cutoff-trained output to the legacy rendering layout."""
    if isinstance(summary.get("cutoffs"), list):
        return summary
    converted = {"cutoffs": []}
    for cutoff_text, result in sorted(
        summary["cutoffs"].items(), key=lambda item: int(item[0])
    ):
        held = result["heldout_rate"]
        by_source = {}
        for source, source_result in result["external_source_rates"].items():
            rate = source_result["all_records"]
            by_source[source] = {
                "unmatched": {
                    "k": rate["k"],
                    "n": rate["n"],
                    "rate": rate["rate"],
                    "ci95": rate["wilson95"],
                }
            }
        converted["cutoffs"].append(
            {
                "cutoff": int(cutoff_text),
                "matchings": {
                    "coarse": {
                        "icsd_unmatched": {
                            "k": held["k"],
                            "n": held["n"],
                            "rate": held["rate"],
                            "ci95": held["wilson95"],
                        },
                        "by_source": by_source,
                    }
                },
            }
        )
    return converted


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("loading ICSD features...", flush=True)
    if args.features_pca:
        X2 = np.load(args.features_pca)[:, :2]
    else:
        X = np.load(args.features)
        Xs = StandardScaler().fit_transform(X)
        pca2 = PCA(n_components=2, random_state=args.seed)
        X2 = pca2.fit_transform(Xs)
    print(f"  {len(X2)} ICSD points", flush=True)

    print("loading external records...", flush=True)
    src_paths = {
        "GNoME": Path(args.gnome),
        "MatterGen": Path(args.mattergen),
        "MP": Path(args.mp),
        "JARVIS": Path(args.jarvis),
        "Alexandria": Path(args.alexandria),
    }
    src_data = {}
    for name, p in src_paths.items():
        xy, ol = load_xy(p)
        src_data[name] = (xy, ol)
        print(f"  {name}: {len(xy)} entries ({ol.sum()} frontier)", flush=True)

    # Equal-size per-source subsample for the overlay panels only.
    src_plot = subsample_sources(src_data, args.subsample_n, args.subsample_seed)

    if args.rates_json:
        print(f"loading panel (c) rates from --rates-json {args.rates_json} ...", flush=True)
        summary = normalize_rate_summary(json.loads(Path(args.rates_json).read_text()))
    else:
        print("loading composition-matched summary...", flush=True)
        summary = normalize_rate_summary(
            json.loads(Path(args.composition_matched_summary).read_text())
        )

    print("rendering figure...", flush=True)
    fig = plt.figure(figsize=(14.5, 12.0), dpi=args.dpi)
    # Taller canvas than the original (13.5 x 9.5): the larger fonts and the
    # below-axes legends of (a),(b) need the extra vertical room (hspace).
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], hspace=0.50, wspace=0.30,
                           left=0.06, right=0.985, top=0.925, bottom=0.075)

    # Panel a: ICSD KDE + GNoME + MatterGen
    ax_a = fig.add_subplot(gs[0, 0])
    render_kde_panel(
        ax_a, X2,
        sources_xy_list=[((name, color, marker), src_plot[name])
                          for name, color, marker in SOURCES_AI],
        title="(a) GNoME and MatterGen on the frozen ICSD topography",
        kde_grid=args.kde_grid, kde_sample=args.kde_sample,
        bandwidth_scale=args.bandwidth_scale, quantile_trim=args.quantile_trim,
        rng=rng, legend_ncol=2,
    )

    # Panel b: ICSD KDE + MP + JARVIS + Alexandria
    ax_b = fig.add_subplot(gs[0, 1])
    render_kde_panel(
        ax_b, X2,
        sources_xy_list=[((name, color, marker), src_plot[name])
                          for name, color, marker in SOURCES_DFT],
        title="(b) MP, JARVIS, and Alexandria on the same topography",
        kde_grid=args.kde_grid, kde_sample=args.kde_sample,
        bandwidth_scale=args.bandwidth_scale, quantile_trim=args.quantile_trim,
        rng=rng, legend_ncol=3,
    )

    # Panel c: bar chart of held-out rates (ALL records; not subsampled)
    ax_c = fig.add_subplot(gs[1, :])
    drawn = render_bar_panel(ax_c, summary)
    if args.dump_rates:
        Path(args.dump_rates).write_text(json.dumps(drawn, indent=1))
        print(f"wrote panel (c) rates -> {args.dump_rates}")

    fig.suptitle("Five computed structure cohorts in ICSD-derived reference frames",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=0.985)

    if not args.no_footnote and args.subsample_n > 0:
        n_mg = len(src_data["MatterGen"][0])
        fig.text(
            0.5, 0.004,
            f"Panels a,b: equal-size random subsample of each source "
            f"(n = {args.subsample_n} per source, seed {args.subsample_seed}; "
            f"MatterGen n = {n_mg} shown in full). Panel c uses all records.\n"
            f"Bright (white) = historically densified ICSD core; "
            f"deep blue = low ICSD density.",
            ha="center", va="bottom", fontsize=FS_FOOTNOTE, color="#444444",
            style="italic", linespacing=1.5,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    if out.suffix.lower() != ".svg":
        fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
