#!/usr/bin/env python3
"""Temporal sweep of GNoME in-basin rate across multiple held-out cutoffs.

For each split year in ``--split-years`` (the manuscript uses
1990 / 2000 / 2010), recomputes per-community 95th-percentile centroid-
distance thresholds from the *training* ICSD subset (year ≤ split),
then reuses the precomputed GNoME records' nearest_centroid_distance
and assigned_community to classify each public GNoME entry under that
cutoff's thresholds. Reports the held-out ICSD in-basin rate and the
GNoME in-basin rate for each cutoff with Wilson 95% CIs. Same
projection-once / threshold-many-times approximation used by the
composition-matched control.

Inputs:
  --features, --community-assignments, --sample-assignments,
  --gnome-records, --split-years.

Outputs (under ``--output-dir``):
  gnome_temporal_sweep_summary.json   Per-cutoff held-out and GNoME
                                       in-basin rates with 95% CIs.
  gnome_temporal_sweep_rows.csv       Same, tabular.
  gnome_temporal_sweep_p95.png        Cutoff-vs-rate plot.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple held-out ICSD/GNoME baseline cutoffs.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--sample-assignments", required=True)
    parser.add_argument("--gnome-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-years", nargs="+", type=int, required=True)
    return parser.parse_args()


def load_sample_rows(path: Path) -> list[dict[str, int | None]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                icsd_id = int(row["icsd_id"])
            except Exception:
                continue
            try:
                year = int(row["year"])
            except Exception:
                year = None
            rows.append({"icsd_id": icsd_id, "year": year})
    return rows


def load_community_rows(path: Path) -> dict[int, int]:
    out = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                out[int(row["icsd_id"])] = int(row["community"])
            except Exception:
                continue
    return out


def load_gnome_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "material_id": row["material_id"],
                    "assigned_community": int(row["assigned_community"]),
                    "nearest_centroid_distance": float(row["nearest_centroid_distance"]),
                }
            )
    return rows


def centroid_thresholds(X: np.ndarray, labels: np.ndarray, percentile: float) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    centroids: dict[int, np.ndarray] = {}
    thresholds: dict[int, float] = {}
    for comm in sorted({int(v) for v in labels if int(v) >= 0}):
        members = np.flatnonzero(labels == comm)
        if len(members) == 0:
            continue
        center = X[members].mean(axis=0)
        centroids[comm] = center
        dists = np.linalg.norm(X[members] - center, axis=1)
        thresholds[comm] = float(np.percentile(dists, percentile))
    return centroids, thresholds


def classify_external(distances: list[float], communities: list[int], thresholds: dict[int, float]) -> float:
    keep = 0
    total = 0
    for dist, comm in zip(distances, communities):
        thr = thresholds.get(comm)
        if thr is None:
            continue
        total += 1
        if dist <= thr:
            keep += 1
    return float(keep / total) if total else 0.0


def plot_summary(summary_rows: list[dict[str, object]], out_path: Path) -> None:
    split_years = [int(r["split_year"]) for r in summary_rows]
    held = [float(r["heldout_in_basin_fraction_p95"]) for r in summary_rows]
    gnome = [float(r["gnome_in_basin_fraction_p95"]) for r in summary_rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(split_years, held, marker="o", lw=2, label="Held-out ICSD")
    ax.plot(split_years, gnome, marker="s", lw=2, label="GNoME")
    ax.set_xlabel("Training cutoff year")
    ax.set_ylabel("In-basin fraction at 95th percentile threshold")
    ax.set_title("Held-out ICSD remains less frontier-like than GNoME")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.features)
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    Xp = PCA(n_components=min(32, Xs.shape[0], Xs.shape[1]), random_state=42).fit_transform(Xs)

    sample_rows = load_sample_rows(Path(args.sample_assignments))
    community_by_id = load_community_rows(Path(args.community_assignments))
    aligned_rows = list(sample_rows)
    labels = np.asarray([community_by_id.get(int(row["icsd_id"]), -1) for row in aligned_rows], dtype=int)
    if Xp.shape[0] != len(aligned_rows):
        raise ValueError(f"Feature rows ({Xp.shape[0]}) != aligned assignment rows ({len(aligned_rows)})")

    valid_year = np.asarray([row["year"] is not None for row in aligned_rows], dtype=bool)
    Xp = Xp[valid_year]
    labels = labels[valid_year]
    years = np.asarray([int(row["year"]) for row in aligned_rows if row["year"] is not None], dtype=int)

    gnome_rows = load_gnome_rows(Path(args.gnome_records))
    percentiles = list(range(50, 100, 5)) + [99]
    sweep_rows = []

    for split_year in args.split_years:
        train_mask = years <= split_year
        holdout_mask = years > split_year
        train_X = Xp[train_mask]
        train_labels = labels[train_mask]
        holdout_X = Xp[holdout_mask]
        holdout_labels = labels[holdout_mask]
        valid_holdout = holdout_labels >= 0

        threshold_sensitivity = []
        for percentile in percentiles:
            centroids, thresholds = centroid_thresholds(train_X, train_labels, percentile)
            train_centroid_arr = np.vstack([centroids[c] for c in sorted(centroids)])
            centroid_ids = np.array(sorted(centroids))
            holdout_distances = []
            holdout_comms = []
            for x in holdout_X[valid_holdout]:
                dists = np.linalg.norm(train_centroid_arr - x, axis=1)
                idx = int(np.argmin(dists))
                holdout_distances.append(float(dists[idx]))
                holdout_comms.append(int(centroid_ids[idx]))

            held_frac = classify_external(holdout_distances, holdout_comms, thresholds)
            gnome_frac = classify_external(
                [float(r["nearest_centroid_distance"]) for r in gnome_rows],
                [int(r["assigned_community"]) for r in gnome_rows],
                thresholds,
            )
            threshold_sensitivity.append(
                {
                    "percentile": int(percentile),
                    "gnome_in_basin_fraction": float(gnome_frac),
                    "heldout_icsd_in_basin_fraction": float(held_frac),
                }
            )

        row = {
            "split_year": int(split_year),
            "n_train_icsd": int(np.sum(train_mask)),
            "n_holdout_icsd": int(np.sum(holdout_mask)),
            "gnome_in_basin_fraction_p95": float(
                next(item["gnome_in_basin_fraction"] for item in threshold_sensitivity if item["percentile"] == 95)
            ),
            "heldout_in_basin_fraction_p95": float(
                next(item["heldout_icsd_in_basin_fraction"] for item in threshold_sensitivity if item["percentile"] == 95)
            ),
            "threshold_sensitivity": threshold_sensitivity,
        }
        sweep_rows.append(row)

    summary = {"rows": sweep_rows}
    (out_dir / "gnome_temporal_sweep_summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "gnome_temporal_sweep_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split_year", "n_train_icsd", "n_holdout_icsd", "gnome_in_basin_fraction_p95", "heldout_in_basin_fraction_p95"],
        )
        writer.writeheader()
        for row in sweep_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    plot_summary(sweep_rows, out_dir / "gnome_temporal_sweep_p95.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
