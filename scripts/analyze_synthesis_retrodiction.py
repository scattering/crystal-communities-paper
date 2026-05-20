#!/usr/bin/env python3
"""Held-out historical retrodiction for the structural accessibility score.

Cited from Methods. The premise: if the per-community structural-
accessibility score 𝒜ᵢ captures *synthesizability priority* and not
just nearest-centroid distance, then on entries reported *after* an
artificial cutoff, the *first* observed reduced-formula instance
should arrive earlier the lower 𝒜ᵢ is, even though those entries
were unseen at training time. This script trains the entire scoring
pipeline on ICSD entries with year ≤ ``--holdout-year`` (default 2000;
the manuscript also reports 1990 / 2010) and scores everything that
came later.

Pipeline:
  1. Load frozen ICSD raw matminer features and the production graph
     community labels; PCA-reduce to 32 components on the *full* set
     (the same projection used by every figure script).
  2. Restrict to ``year <= --holdout-year``: recompute per-community
     centroids and the within-community ``--threshold-percentile``
     (default 95) centroid-distance threshold; recompute the
     normalization moments (μ, σ) of the raw accessibility score on
     the same training subset.
  3. Score every post-cutoff ICSD entry by nearest-centroid distance,
     in-basin flag, and standardized 𝒜ᵢ.
  4. For each reduced formula appearing post-cutoff, take the
     *earliest* entry (its year-of-first-report); compute Spearman ρ
     of 𝒜ᵢ vs. first-report year and of in_basin vs. first-report
     year. Build a year-permutation null over ``--null-repeats``
     shuffles (default 1000) for both correlations.
  5. For polymorph "races" (multiple distinct years per formula), test
     whether the *first*-reported polymorph has lower 𝒜ᵢ than later
     ones (i.e. whether the easier polymorph arrives first).

Outputs (under ``--output-dir``):
  synthesis_retrodiction_summary.json        ρ, null bands, race tally.
  post_cutoff_accessibility_records.csv      Per-entry 𝒜ᵢ + in_basin.
                                              Consumed by every external
                                              validation script
                                              (Kononova, A-Lab, etc.)
                                              and by the synthesis
                                              pivot analysis.
  first_report_formulas.csv                   Earliest entry per formula.
  polymorph_sibling_results.csv               Per-formula race outcomes.
  retrodiction_first_report_scatter.png       In-script preview only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Composition
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal synthesis retrodiction on post-cutoff ICSD.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--sample-assignments", required=True)
    parser.add_argument("--icsd-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-year", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null-repeats", type=int, default=1000)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    return parser.parse_args()


def parse_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
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


def reduce_formula(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return Composition(text).reduced_formula
    except Exception:
        return None


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    num = sum(a * b for a, b in zip(dx, dy))
    denx = sum(a * a for a in dx) ** 0.5
    deny = sum(b * b for b in dy) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return num / denx / deny


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def load_comm_rows(path: Path) -> list[dict[str, int | None]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "icsd_id": parse_int(row.get("icsd_id", "")),
                    "year": parse_int(row.get("year", "")),
                    "community": parse_int(row.get("community", "")),
                }
            )
    return rows


def load_sample_rows(path: Path) -> list[dict[str, int | None]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({"icsd_id": parse_int(row.get("icsd_id", "")), "year": parse_int(row.get("year", ""))})
    return rows


def load_icsd_index(path: Path) -> dict[int, dict[str, object]]:
    out = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            cif_id = parse_int(row.get("cif_names", ""))
            if cif_id is None:
                continue
            year = parse_int(row.get("publication_year", ""))
            out[cif_id] = {
                "reduced_formula": reduce_formula(row.get("name", "")),
                "year": year,
            }
    return out


def centroid_thresholds(X: np.ndarray, labels: np.ndarray, percentile: float = 95.0) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float], dict[int, float]]:
    centroids: dict[int, np.ndarray] = {}
    thresholds: dict[int, float] = {}
    sizes: dict[int, float] = {}
    for comm in sorted({int(v) for v in labels if int(v) >= 0}):
        members = np.flatnonzero(labels == comm)
        if len(members) == 0:
            continue
        center = X[members].mean(axis=0)
        centroids[comm] = center
        dists = np.linalg.norm(X[members] - center, axis=1)
        thresholds[comm] = float(np.percentile(dists, percentile))
        sizes[comm] = float(len(members))
    return centroids, thresholds, sizes, {c: thresholds[c] for c in thresholds}


def raw_accessibility(distance: float, core_threshold: float, size: float, community_age: float) -> float:
    norm_dist = distance / max(core_threshold, 1e-6)
    return math.log1p(norm_dist) - 0.5 * math.log1p(size) - 0.5 * math.log1p(max(community_age, 0.0))


def scatter_plot(first_reports: list[dict[str, object]], rho: float | None, out_path: Path) -> None:
    x = [float(row["A_i"]) for row in first_reports]
    y = [float(row["year"]) for row in first_reports]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(x, y, s=10, alpha=0.35, c="#1b9e77", linewidths=0)
    ax.set_xlabel("Pre-2000 structural accessibility score $A_i$")
    ax.set_ylabel("First post-2000 report year")
    ax.set_title("Post-2000 ICSD retrodiction")
    if rho is not None:
        ax.text(0.03, 0.97, f"Spearman rho = {rho:.3f}", transform=ax.transAxes, va="top", ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.features)
    comm_rows = load_comm_rows(Path(args.community_assignments))
    sample_rows = load_sample_rows(Path(args.sample_assignments))
    index = load_icsd_index(Path(args.icsd_index))

    community_by_id = {row["icsd_id"]: row["community"] for row in comm_rows if row["icsd_id"] is not None}
    aligned_rows = list(sample_rows)
    labels = np.asarray([community_by_id.get(row["icsd_id"], -1) if row["icsd_id"] is not None else -1 for row in aligned_rows], dtype=int)

    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    Xp = PCA(n_components=min(32, Xs.shape[0], Xs.shape[1]), random_state=args.seed).fit_transform(Xs)
    if Xp.shape[0] != len(aligned_rows):
        raise ValueError(f"Feature rows ({Xp.shape[0]}) != aligned assignment rows ({len(aligned_rows)})")

    valid_year = np.asarray([row["year"] is not None for row in aligned_rows], dtype=bool)
    years = np.asarray([int(row["year"]) for row in aligned_rows if row["year"] is not None], dtype=int)
    icsd_ids = np.asarray([int(row["icsd_id"]) for row in aligned_rows if row["year"] is not None and row["icsd_id"] is not None], dtype=int)
    Xp = Xp[valid_year]
    labels = labels[valid_year]

    train_mask = years <= args.holdout_year
    holdout_mask = years > args.holdout_year
    train_X = Xp[train_mask]
    train_labels = labels[train_mask]
    train_years = years[train_mask]

    centroids, thresholds, sizes, _ = centroid_thresholds(train_X, train_labels, percentile=float(args.threshold_percentile))
    centroid_ids = np.array(sorted(centroids))
    centroid_arr = np.vstack([centroids[c] for c in centroid_ids])

    births: dict[int, int] = {}
    for comm, year in zip(train_labels, train_years):
        if int(comm) < 0:
            continue
        births[int(comm)] = min(int(year), births.get(int(comm), int(year)))

    train_raws = []
    for x, comm, year in zip(train_X, train_labels, train_years):
        if int(comm) < 0:
            continue
        dist = float(np.linalg.norm(x - centroids[int(comm)]))
        age = float(int(year) - births[int(comm)])
        train_raws.append(raw_accessibility(dist, thresholds[int(comm)], sizes[int(comm)], age))
    mu = mean(train_raws)
    sigma = stdev(train_raws)

    scored_rows = []
    for x, icsd_id, year in zip(Xp[holdout_mask], icsd_ids[holdout_mask], years[holdout_mask]):
        dists = np.linalg.norm(centroid_arr - x, axis=1)
        idx = int(np.argmin(dists))
        comm = int(centroid_ids[idx])
        dist = float(dists[idx])
        age = float(int(year) - births[comm])
        raw = raw_accessibility(dist, thresholds[comm], sizes[comm], age)
        score = zscore(raw, mu, sigma)
        in_basin = dist <= thresholds[comm]
        meta = index.get(int(icsd_id), {})
        scored_rows.append(
            {
                "cif_id": int(icsd_id),
                "reduced_formula": meta.get("reduced_formula"),
                "year": int(year),
                "assigned_community": comm,
                "nearest_centroid_distance": dist,
                "is_in_basin": int(in_basin),
                "A_i": score,
            }
        )

    scored_rows = [row for row in scored_rows if row["reduced_formula"]]
    first_by_formula: dict[str, dict[str, object]] = {}
    for row in scored_rows:
        formula = str(row["reduced_formula"])
        prev = first_by_formula.get(formula)
        if prev is None or int(row["year"]) < int(prev["year"]):
            first_by_formula[formula] = row
    first_reports = list(first_by_formula.values())

    rho = spearman([float(r["A_i"]) for r in first_reports], [float(r["year"]) for r in first_reports])
    rho_in_basin = spearman([float(r["is_in_basin"]) for r in first_reports], [float(r["year"]) for r in first_reports])
    rng = random.Random(args.seed)
    years_first = [int(r["year"]) for r in first_reports]
    a_vals = [float(r["A_i"]) for r in first_reports]
    basin_vals = [float(r["is_in_basin"]) for r in first_reports]
    null_rhos = []
    null_rhos_basin = []
    for _ in range(args.null_repeats):
        ys = years_first[:]
        rng.shuffle(ys)
        r = spearman(a_vals, ys)
        if r is not None:
            null_rhos.append(float(r))
        rb = spearman(basin_vals, ys)
        if rb is not None:
            null_rhos_basin.append(float(rb))

    by_formula: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in scored_rows:
        by_formula[str(row["reduced_formula"])].append(row)
    polymorph_formulas = {f: rows for f, rows in by_formula.items() if len({int(r["year"]) for r in rows}) > 1}

    first_easier_count = 0
    total_races = 0
    sibling_rows = []
    for formula, rows in polymorph_formulas.items():
        first_year = min(int(r["year"]) for r in rows)
        first_vals = [float(r["A_i"]) for r in rows if int(r["year"]) == first_year]
        later_vals = [float(r["A_i"]) for r in rows if int(r["year"]) > first_year]
        if not later_vals:
            continue
        first_A = mean(first_vals)
        later_A = mean(later_vals)
        first_easier = first_A < later_A
        if first_easier:
            first_easier_count += 1
        total_races += 1
        sibling_rows.append(
            {
                "reduced_formula": formula,
                "first_year": first_year,
                "first_A_i_mean": first_A,
                "later_A_i_mean": later_A,
                "first_easier": first_easier,
            }
        )

    summary = {
        "holdout_year": int(args.holdout_year),
        "threshold_percentile": float(args.threshold_percentile),
        "n_train_icsd": int(np.sum(train_mask)),
        "n_post_cutoff_icsd": int(np.sum(holdout_mask)),
        "n_first_report_formulas": int(len(first_reports)),
        "spearman_Ai_vs_first_report_year": rho,
        "spearman_in_basin_vs_first_report_year": rho_in_basin,
        "null_mean_rho": mean(null_rhos) if null_rhos else None,
        "null_std_rho": stdev(null_rhos) if len(null_rhos) > 1 else None,
        "null_abs_rho_gt_3sigma": None
        if rho is None or len(null_rhos) < 2
        else abs(float(rho) - mean(null_rhos)) > 3.0 * stdev(null_rhos),
        "null_mean_rho_in_basin": mean(null_rhos_basin) if null_rhos_basin else None,
        "null_std_rho_in_basin": stdev(null_rhos_basin) if len(null_rhos_basin) > 1 else None,
        "null_abs_rho_in_basin_gt_3sigma": None
        if rho_in_basin is None or len(null_rhos_basin) < 2
        else abs(float(rho_in_basin) - mean(null_rhos_basin)) > 3.0 * stdev(null_rhos_basin),
        "post_cutoff_in_basin_fraction": mean([float(r["is_in_basin"]) for r in scored_rows]) if scored_rows else None,
        "first_report_in_basin_fraction": mean([float(r["is_in_basin"]) for r in first_reports]) if first_reports else None,
        "n_temporal_polymorph_formulas": int(total_races),
        "polymorph_first_easier_fraction": (first_easier_count / total_races) if total_races else None,
    }

    (out_dir / "synthesis_retrodiction_summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "post_cutoff_accessibility_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0].keys()) if scored_rows else [])
        writer.writeheader()
        writer.writerows(scored_rows)

    with (out_dir / "first_report_formulas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first_reports[0].keys()) if first_reports else [])
        writer.writeheader()
        writer.writerows(first_reports)

    with (out_dir / "polymorph_sibling_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sibling_rows[0].keys()) if sibling_rows else [])
        writer.writeheader()
        writer.writerows(sibling_rows)

    if first_reports:
        scatter_plot(first_reports, rho, out_dir / "retrodiction_first_report_scatter.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
