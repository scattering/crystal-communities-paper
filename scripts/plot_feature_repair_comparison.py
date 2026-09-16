#!/usr/bin/env python3
"""Plot comparison.json from compare_feature_repair.py without interpreting it.

Writes PNG and SVG figures for 1930s–2010s birth shares. If available, a second
figure shows the common-ID supplement with fixed labels and birth years
recomputed within the common cohort; it does not represent reclustered data.
Missing values remain gaps, and failed/unavailable representations are labelled.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


DECADES = tuple(f"{year}s" for year in range(1930, 2020, 10))
TITLES = {
    "production": "Production representation",
    "magpie": "Magpie ablation",
    "graphlets": "Graphlet representation",
}
STYLES = {
    "old": {"color": "#666666", "linestyle": "--", "marker": "o", "label": "Historical"},
    "new": {"color": "#0072B2", "linestyle": "-", "marker": "s", "label": "Repaired"},
}


def read_curve(curve, context):
    """Read recorded fractions only; null/absent decades are plotting gaps."""
    if not isinstance(curve, dict):
        raise ValueError(f"{context}: expected birth_curve_comparison object")
    values = {name: [] for name in STYLES}
    for decade in DECADES:
        row = curve.get(decade, {})
        if not isinstance(row, dict):
            raise ValueError(f"{context}/{decade}: expected an object")
        for name in STYLES:
            value = row.get(name)
            if value is None:
                values[name].append(float("nan"))
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{context}/{decade}/{name}: birth share must be a finite fraction in [0, 1] or null")
            values[name].append(float(value))
    return values


def panels_from_report(report, common_cohort=False):
    representations = report.get("representations")
    if not isinstance(representations, dict) or not representations:
        raise ValueError("comparison.json must contain a nonempty representations object")
    panels = []
    for name, result in representations.items():
        if not isinstance(result, dict):
            raise ValueError(f"{name}: expected a representation object")
        panel = {"title": TITLES.get(name, name), "values": None, "note": ""}
        if "error" in result:
            panel["note"] = "Unavailable / validation failed\n" + str(result["error"])
        elif common_cohort:
            shared = result.get("common_cohort_birth_shares", {})
            status = shared.get("status")
            if status == "available":
                method = shared["common_cohort_birth_year"]
                panel["values"] = read_curve(method["birth_curve_comparison"], f"{name}/common_cohort_birth_year")
                count = shared.get("n_common")
                if count is not None:
                    panel["title"] += f"\n{count:,} shared entries"
            elif status == "not_needed_identical_id_sets":
                panel["note"] = "Identical successful-ID sets;\nno restriction needed."
            else:
                panel["note"] = "Supplement unavailable\n" + str(shared.get("reason", "No common-ID supplement recorded."))
        else:
            panel["values"] = read_curve(result["birth_curve_comparison"], name)
        if panel["values"] is not None and not any(
            math.isfinite(value) for series in panel["values"].values() for value in series
        ):
            panel["note"] = "No recorded birth shares\nfor 1930s–2010s."
        panels.append(panel)
    return panels


def draw_figure(panels, title, caption):
    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(9.5, 4.0 * ncols), 3.3 * nrows + 1.4),
                             sharex=True, sharey=True, squeeze=False)
    finite_values = [value for panel in panels if panel["values"] is not None
                     for series in panel["values"].values() for value in series
                     if math.isfinite(value)]
    ymax = min(1.0, max(0.05, math.ceil(max(finite_values) * 1.1 / 0.05) * 0.05)) if finite_values else 1.0
    for index, panel in enumerate(panels):
        ax = axes.flat[index]
        ax.set_title(panel["title"], fontsize=11, pad=10)
        if panel["values"] is not None:
            for name, style in STYLES.items():
                ax.plot(range(len(DECADES)), panel["values"][name], linewidth=1.7,
                        markersize=4.5, **style)
        if panel["note"]:
            note = "\n".join(textwrap.fill(line, width=38) for line in panel["note"].splitlines())
            ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#555555", wrap=True)
        ax.set_xticks(range(len(DECADES)), DECADES, rotation=45, ha="right", fontsize=9)
        ax.set_xlim(-0.25, len(DECADES) - 0.75)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if index % ncols == 0:
            ax.set_ylabel("Community birth share", fontsize=10)
        if index // ncols == nrows - 1:
            ax.set_xlabel("Publication decade", fontsize=10)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14, y=0.98)
    handles = [Line2D([], [], linewidth=1.7, markersize=4.5, **style) for style in STYLES.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=2, frameon=False, fontsize=10)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.13, 1, 0.85), h_pad=2, w_pad=1.5)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="comparison.json produced by compare_feature_repair.py")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for PNG and SVG figures")
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.input.read_text())
        if not isinstance(report, dict):
            raise ValueError("comparison.json must contain a JSON object")
        main_panels = panels_from_report(report)
        common_panels = panels_from_report(report, common_cohort=True)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.error(str(exc))
    figures = [("birth_share_comparison", main_panels, "Historical and repaired community birth shares",
                "Each curve uses the cohort recorded in comparison.json. Outliers remain in denominators.\n"
                "Decades outside 1930s–2010s and unknown years are not shown; missing values remain gaps.")]
    if any(panel["values"] is not None for panel in common_panels):
        figures.append(("common_cohort_birth_year", common_panels, "Common-ID supplement: fixed-label restriction",
                        "Shared ICSD IDs only; labels and noise membership fixed, birth years recomputed within shared IDs.\n"
                        "No refitting or reclustering; partitions retain dependence on their full fitting populations."))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none"}):
        for stem, panels, title, caption in figures:
            fig = draw_figure(panels, title, caption)
            try:
                for suffix in ("png", "svg"):
                    path = args.out_dir / f"{stem}.{suffix}"
                    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
                    print(f"Wrote {path}")
            finally:
                plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
