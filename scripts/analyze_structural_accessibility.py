#!/usr/bin/env python3
"""Compute the structural accessibility score 𝒜ᵢ from the frozen ICSD map.

Producer for the score 𝒜ᵢ used throughout the manuscript. For each
ICSD entry the *raw* accessibility is

    raw = log1p(distance / core_threshold)
          - alpha * log1p(community_size)
          - beta  * log1p(age_within_community)

with alpha and beta fixed at 1.0 (sensitivity sweep is in
``analyze_accessibility_sensitivity.py``). The raw scores over the
historical ICSD record set the (μ, σ) used to z-score every
subsequent value, so 𝒜ᵢ is in standard-deviation units relative to
the historical record. Higher values are more frontier-like.

For every ICSD entry the score is grouped by the structural-graph
*event type* attached to it (community_birth / core / periphery /
bridge_attachment) and aggregated. The same scorer is then applied to
the GNoME public bundle (using the GNoME records emitted by
``analyze_gnome_frontier.py`` and an age relative to the end of the
ICSD observation window, 2019), split into in-basin vs. frontier-like.

Inputs:
  --community-assignments  community_assignments.csv (icsd_id, year,
                            community).
  --node-events            Per-entry node-event table (event_type,
                            core_periphery, is_bridge_attachment,
                            distance_to_centroid).
  --gnome-records          gnome_frontier_records.csv.

Outputs (under ``--output-dir``):
  structural_accessibility_summary.json    group means + ICSD-p90
                                            anchor used for the
                                            "GNoME above ICSD-p90"
                                            statistic in the text.
  gnome_accessibility_records.csv          per-GNoME 𝒜ᵢ + outlier flag.
  structural_accessibility_boxplot.png     six-group QA boxplot.
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
    parser.add_argument("--output-dir", required=True)
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


def boxplot(groups: dict[str, list[float]], out_path: Path) -> None:
    labels = list(groups.keys())
    values = [groups[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_ylabel("Structural accessibility score (higher = more frontier-like)")
    ax.set_title("Historical ICSD versus public GNoME structural accessibility")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
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

    gnome_scores = []
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
            # Use age relative to the end of the ICSD observation window.
            age = 2019 - meta["birth_year"]
            raw = raw_accessibility(dist, meta["core_threshold"], meta["size"], age)
            score = zscore(raw, mu, sigma)
            outlier_like = str(row.get("outlier_like", "")).lower() == "true"
            gnome_scores.append(
                {
                    "material_id": row.get("material_id", ""),
                    "community": comm,
                    "outlier_like": outlier_like,
                    "accessibility_score": score,
                }
            )
            groups["GNoME in-basin" if not outlier_like else "GNoME frontier"].append(score)

    summary = {
        "icsd_group_means": {k: mean(v) for k, v in groups.items() if k.startswith("ICSD")},
        "gnome_group_means": {k: mean(v) for k, v in groups.items() if k.startswith("GNoME")},
        "gnome_frontier_fraction_above_icsd_p90": None,
        "gnome_in_basin_fraction_below_icsd_p90": None,
    }

    icsd_sorted = sorted([float(r["accessibility_score"]) for r in icsd_rows])
    p90 = percentile(icsd_sorted, 0.90)
    frontier_vals = [r["accessibility_score"] for r in gnome_scores if r["outlier_like"]]
    in_basin_vals = [r["accessibility_score"] for r in gnome_scores if not r["outlier_like"]]
    if frontier_vals:
        summary["gnome_frontier_fraction_above_icsd_p90"] = sum(v > p90 for v in frontier_vals) / len(frontier_vals)
    if in_basin_vals:
        summary["gnome_in_basin_fraction_below_icsd_p90"] = sum(v <= p90 for v in in_basin_vals) / len(in_basin_vals)
    summary["icsd_p90_accessibility"] = p90

    (out_dir / "structural_accessibility_summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "gnome_accessibility_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["material_id", "community", "outlier_like", "accessibility_score"])
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
    boxplot(ordered, out_dir / "structural_accessibility_boxplot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
