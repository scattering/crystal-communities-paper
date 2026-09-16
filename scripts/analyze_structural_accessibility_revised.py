#!/usr/bin/env python3
"""Compute the structural accessibility score 𝒜ᵢ from the frozen ICSD map.

REVISED COPY (2026-09, npj review) of ``analyze_structural_accessibility.py``.
The only change to the *analysis* is the GNoME in-basin / frontier split:
the original used the ``outlier_like`` column of the GNoME records file,
which is a POOLED-threshold flag (distance > pooled 95th percentile,
τ = 4.756), whereas the manuscript classifies in-basin per community
(d ≤ τ_c, the community's own 95th-percentile member distance). The
revised script takes the per-community flag from
``notes/review_2026_08/quadrant_assignments.csv`` (``in_basin`` column,
``source == GNoME``, joined on ``material_id``). Everything else — the
score, α = β = 0.5, the 2019 age reference for external proposals, the
ICSD role assignment and the z-scoring — is byte-identical to the
original. Figure cosmetics (200 dpi, ≥ 11 pt fonts, per-box n / mean /
median annotations) and a ``--dump-stats`` option were added so the
figure can be checked against the printed statistics.

Producer for the score 𝒜ᵢ used throughout the manuscript. For each
ICSD entry the *raw* accessibility is

    raw = log1p(distance / core_threshold)
          - alpha * log1p(community_size)
          - beta  * log1p(age_within_community)

with alpha and beta fixed at 0.5 (production value; the (alpha, beta)
sensitivity sweep is in ``analyze_accessibility_sensitivity.py``). The
raw scores over the
historical ICSD record set the (μ, σ) used to z-score every
subsequent value, so 𝒜ᵢ is in standard-deviation units relative to
the historical record. Higher values are more frontier-like.

For every ICSD entry the score is grouped by the structural-graph
*event type* attached to it (community_birth / core / periphery /
bridge_attachment) and aggregated. The same scorer is then applied to
the GNoME public bundle (using the GNoME records emitted by
``analyze_gnome_frontier.py`` and an age relative to the end of the
ICSD observation window, 2019), split into in-basin vs. frontier using
the per-community 95th-percentile rule.

Inputs:
  --community-assignments  community_assignments.csv (icsd_id, year,
                            community).
  --node-events            Per-entry node-event table (event_type,
                            core_periphery, is_bridge_attachment,
                            distance_to_centroid).
  --gnome-records          gnome_frontier_records.csv.
  --quadrant-assignments   quadrant_assignments.csv (source, material_id,
                            in_basin, ...); only source == GNoME rows are
                            used. Supplies the per-community split.
  --figure-path            Optional explicit path for the boxplot PNG
                            (default: <output-dir>/
                            structural_accessibility_boxplot_revised.png).
  --dump-stats             Print n / mean / median for the six groups and
                            the resulting ordering to stdout.

Outputs (under ``--output-dir``):
  structural_accessibility_summary.json    group means + ICSD-p90
                                            anchor used for the
                                            "GNoME above ICSD-p90"
                                            statistic in the text.
  gnome_accessibility_records.csv          per-GNoME 𝒜ᵢ + both flags
                                            (pooled outlier_like and the
                                            per-community in_basin).
  structural_accessibility_boxplot_revised.png  six-group boxplot.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a structural accessibility score from the frozen ICSD map.")
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--node-events", required=True)
    parser.add_argument("--gnome-records", required=True)
    parser.add_argument("--quadrant-assignments", required=True, help="notes/review_2026_08/quadrant_assignments.csv (per-community in_basin flags)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-path", default=None, help="Explicit output path for the boxplot PNG")
    parser.add_argument("--dump-stats", action="store_true", help="Print n / mean / median per group and the group ordering")
    return parser.parse_args()


def parse_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def parse_float(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var) or 1.0


def zscore(v: float, m: float, s: float) -> float:
    return (v - m) / s if s else 0.0


def median(values: list[float]) -> float:
    return percentile(sorted(values), 0.5) if values else 0.0


def load_gnome_in_basin(quadrant_path: Path) -> dict[str, bool]:
    """material_id -> in_basin (per-community 95th-percentile rule) for the
    ``source == GNoME`` rows of quadrant_assignments.csv."""
    flags: dict[str, bool] = {}
    with quadrant_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("source") or "").strip() != "GNoME":
                continue
            material_id = (row.get("material_id") or "").strip()
            flag = (row.get("in_basin") or "").strip().lower()
            if not material_id or flag not in ("true", "false"):
                continue
            flags[material_id] = flag == "true"
    return flags


GROUP_ORDER = ["ICSD core", "ICSD periphery", "ICSD bridge", "ICSD birth", "GNoME in-basin", "GNoME frontier"]


def dump_stats(groups: dict[str, list[float]]) -> None:
    print(f"{'group':<16}{'n':>8}{'mean':>10}{'median':>10}")
    for name in GROUP_ORDER:
        vals = groups.get(name, [])
        print(f"{name:<16}{len(vals):>8}{mean(vals):>+10.3f}{median(vals):>+10.3f}")
    by_mean = sorted(GROUP_ORDER, key=lambda k: mean(groups.get(k, [])))
    print("ordering by mean  : " + " < ".join(by_mean))
    by_median = sorted(GROUP_ORDER, key=lambda k: median(groups.get(k, [])))
    print("ordering by median: " + " < ".join(by_median))
    m = {k: mean(groups.get(k, [])) for k in GROUP_ORDER}
    chain = ["ICSD core", "ICSD periphery", "ICSD bridge", "GNoME in-basin", "GNoME frontier"]
    strict = all(m[a] < m[b] for a, b in zip(chain, chain[1:]))
    print(f"caption chain core < periphery < bridge < GNoME in-basin < GNoME frontier (means): {'HOLDS' if strict else 'FAILS'}")
    print(f"GNoME frontier vs ICSD birth (means): {m['GNoME frontier']:+.3f} vs {m['ICSD birth']:+.3f} (diff {m['GNoME frontier'] - m['ICSD birth']:+.3f})")


def load_community_metadata(assign_path: Path, node_events_path: Path) -> dict[int, dict[str, float]]:
    sizes = Counter()
    births: dict[int, int] = {}
    core_thresholds: dict[int, list[float]] = defaultdict(list)

    with assign_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            comm = parse_int(row.get("community", ""))
            year = parse_int(row.get("year", ""))
            if comm is None or comm < 0:
                continue
            sizes[comm] += 1
            if year is not None:
                births[comm] = min(year, births.get(comm, year))

    with node_events_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            comm = parse_int(row.get("community", ""))
            thresh = parse_float(row.get("core_threshold", ""))
            if comm is None or comm < 0 or thresh is None:
                continue
            core_thresholds[comm].append(thresh)

    meta = {}
    for comm, size in sizes.items():
        thr = mean(core_thresholds.get(comm, [])) or 1.0
        meta[comm] = {
            "size": float(size),
            "birth_year": float(births.get(comm, 2010)),
            "core_threshold": float(thr),
        }
    return meta


def raw_accessibility(distance: float, core_threshold: float, size: float, community_age: float) -> float:
    norm_dist = distance / max(core_threshold, 1e-6)
    return math.log1p(norm_dist) - 0.5 * math.log1p(size) - 0.5 * math.log1p(max(community_age, 0.0))


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = q * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def boxplot(groups: dict[str, list[float]], out_path: Path, dpi: int = 200) -> None:
    """Six-group boxplot (whiskers at 1.5 IQR, fliers hidden). Revised
    cosmetics only: 200 dpi, every font >= 11 pt, white-diamond mean
    marker, and an n / mean / median annotation above each box so the
    rendered figure can be checked against ``--dump-stats``."""
    labels = list(groups.keys())
    values = [groups[k] for k in labels]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    tick_labels = [lab.replace(" ", "\n", 1) for lab in labels]
    ax.boxplot(
        values,
        tick_labels=tick_labels,
        showfliers=False,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
        medianprops={"color": "C3", "linewidth": 1.5},
    )
    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
    # Headroom for annotations: find the highest upper whisker (1.5 IQR rule).
    tops = []
    for vals in values:
        if not vals:
            continue
        s = sorted(vals)
        q1, q3 = percentile(s, 0.25), percentile(s, 0.75)
        cap = q3 + 1.5 * (q3 - q1)
        tops.append(max(v for v in s if v <= cap))
    y_top = max(tops) if tops else 1.0
    lo, _ = ax.get_ylim()
    ax.set_ylim(lo, y_top + 0.30 * (y_top - lo))
    for i, (lab, vals) in enumerate(zip(labels, values), start=1):
        ax.text(
            i,
            0.985,
            f"n = {len(vals):,}\nmean {mean(vals):+.2f}\nmedian {median(vals):+.2f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
        )
    ax.set_ylabel(r"Structural accessibility score $\mathcal{A}_i$" + "\n(z-units; higher = more frontier-like)")
    ax.set_title("Historical ICSD roles versus public GNoME proposals (per-community in-basin split)")
    ax.text(0.99, 0.02, "diamond = mean, red line = median; whiskers 1.5 IQR, fliers hidden", transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="0.35")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    community_meta = load_community_metadata(Path(args.community_assignments), Path(args.node_events))

    icsd_rows = []
    raw_values = []
    with Path(args.node_events).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            comm = parse_int(row.get("community", ""))
            year = parse_int(row.get("year", ""))
            dist = parse_float(row.get("distance_to_centroid", ""))
            if comm is None or comm < 0 or year is None or dist is None:
                continue
            meta = community_meta.get(comm)
            if meta is None:
                continue
            age = year - meta["birth_year"]
            raw = raw_accessibility(dist, meta["core_threshold"], meta["size"], age)
            raw_values.append(raw)
            icsd_rows.append(
                {
                    "icsd_id": parse_int(row.get("icsd_id", "")),
                    "community": comm,
                    "year": year,
                    "event_type": row.get("event_type", ""),
                    "core_periphery": row.get("core_periphery", ""),
                    "is_bridge_attachment": row.get("is_bridge_attachment", ""),
                    "raw_score": raw,
                }
            )

    mu = mean(raw_values)
    sigma = stdev(raw_values)

    groups: dict[str, list[float]] = defaultdict(list)
    for row in icsd_rows:
        score = zscore(float(row["raw_score"]), mu, sigma)
        row["accessibility_score"] = score
        et = str(row["event_type"])
        cp = str(row["core_periphery"]).lower()
        bridge = str(row["is_bridge_attachment"]).lower() == "true"
        if et == "community_birth":
            groups["ICSD birth"].append(score)
        if bridge:
            groups["ICSD bridge"].append(score)
        elif cp == "core":
            groups["ICSD core"].append(score)
        elif cp == "periphery":
            groups["ICSD periphery"].append(score)

    # REVISED: the in-basin / frontier split comes from the per-community
    # 95th-percentile rule (quadrant_assignments.csv, in_basin column), not
    # from the pooled-threshold ``outlier_like`` flag of the records file.
    in_basin_by_id = load_gnome_in_basin(Path(args.quadrant_assignments))
    gnome_scores = []
    n_gnome_unmatched = 0
    with Path(args.gnome_records).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            comm = parse_int(row.get("assigned_community", ""))
            dist = parse_float(row.get("nearest_centroid_distance", ""))
            if comm is None or dist is None:
                continue
            meta = community_meta.get(comm)
            if meta is None:
                continue
            material_id = (row.get("material_id") or "").strip()
            in_basin = in_basin_by_id.get(material_id)
            if in_basin is None:
                n_gnome_unmatched += 1
                continue
            # Use age relative to the end of the ICSD observation window.
            age = 2019 - meta["birth_year"]
            raw = raw_accessibility(dist, meta["core_threshold"], meta["size"], age)
            score = zscore(raw, mu, sigma)
            outlier_like = str(row.get("outlier_like", "")).lower() == "true"
            gnome_scores.append(
                {
                    "material_id": material_id,
                    "community": comm,
                    "outlier_like": outlier_like,
                    "in_basin_per_community": in_basin,
                    "accessibility_score": score,
                }
            )
            groups["GNoME in-basin" if in_basin else "GNoME frontier"].append(score)

    summary = {
        "alpha": 0.5,
        "beta": 0.5,
        "external_age_reference_year": 2019,
        "gnome_split": "per-community 95th-percentile threshold (in_basin column of quadrant_assignments.csv)",
        "n_icsd_scored": len(icsd_rows),
        "icsd_raw_mu": mu,
        "icsd_raw_sigma": sigma,
        "n_gnome_scored": len(gnome_scores),
        "n_gnome_unmatched_in_quadrant_file": n_gnome_unmatched,
        "icsd_group_means": {k: mean(v) for k, v in groups.items() if k.startswith("ICSD")},
        "gnome_group_means": {k: mean(v) for k, v in groups.items() if k.startswith("GNoME")},
        "group_stats": {k: {"n": len(groups.get(k, [])), "mean": mean(groups.get(k, [])), "median": median(groups.get(k, []))} for k in GROUP_ORDER},
        "gnome_frontier_fraction_above_icsd_p90": None,
        "gnome_in_basin_fraction_below_icsd_p90": None,
    }

    icsd_sorted = sorted([float(r["accessibility_score"]) for r in icsd_rows])
    p90 = percentile(icsd_sorted, 0.90)
    frontier_vals = [r["accessibility_score"] for r in gnome_scores if not r["in_basin_per_community"]]
    in_basin_vals = [r["accessibility_score"] for r in gnome_scores if r["in_basin_per_community"]]
    if frontier_vals:
        summary["gnome_frontier_fraction_above_icsd_p90"] = sum(v > p90 for v in frontier_vals) / len(frontier_vals)
    if in_basin_vals:
        summary["gnome_in_basin_fraction_below_icsd_p90"] = sum(v <= p90 for v in in_basin_vals) / len(in_basin_vals)
    summary["icsd_p90_accessibility"] = p90

    (out_dir / "structural_accessibility_summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "gnome_accessibility_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["material_id", "community", "outlier_like", "in_basin_per_community", "accessibility_score"])
        writer.writeheader()
        writer.writerows(gnome_scores)

    ordered = {
        "ICSD core": groups.get("ICSD core", []),
        "ICSD periphery": groups.get("ICSD periphery", []),
        "ICSD bridge": groups.get("ICSD bridge", []),
        "ICSD birth": groups.get("ICSD birth", []),
        "GNoME in-basin": groups.get("GNoME in-basin", []),
        "GNoME frontier": groups.get("GNoME frontier", []),
    }
    figure_path = Path(args.figure_path) if args.figure_path else out_dir / "structural_accessibility_boxplot_revised.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    boxplot(ordered, figure_path)
    if args.dump_stats:
        print(f"ICSD scored: {len(icsd_rows):,}; raw mu = {mu:.4f}, sigma = {sigma:.4f}; ICSD p90 = {p90:.3f}")
        print(f"GNoME scored: {len(gnome_scores):,} (unmatched in quadrant file: {n_gnome_unmatched})")
        dump_stats(ordered)
        print(f"GNoME frontier above ICSD p90: {summary['gnome_frontier_fraction_above_icsd_p90']:.3f}; in-basin at/below p90: {summary['gnome_in_basin_fraction_below_icsd_p90']:.3f}")
        print(f"figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
