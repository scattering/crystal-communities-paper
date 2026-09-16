#!/usr/bin/env python3
"""Earlier single-source threshold-sensitivity sweep for GNoME.

Predecessor of ``analyze_external_baselines.py`` (which generalizes
this code to any external source). Sweeps the within-community
centroid-distance percentile from low to high, recomputes both the
GNoME in-basin fraction and the held-out post-cutoff ICSD in-basin
fraction at each threshold, and writes a single sensitivity figure.
Retained for reproducibility of the earlier draft analyses; the main-
text Figure 3 sensitivity numbers come from
``analyze_external_baselines.py``.

Inputs:
  --features, --community-assignments, --sample-assignments,
  --gnome-records, --holdout-year (default 2015).

Outputs (under ``--output-dir``):
  gnome_baselines_summary.json            Per-percentile in-basin rates.
  gnome_threshold_sensitivity.png         GNoME-vs-held-out ICSD curve.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sensitivity and held-out ICSD baseline for GNoME projection.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--sample-assignments", required=True)
    parser.add_argument("--gnome-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-year", type=int, default=2015)
    return parser.parse_args()


def load_comm_rows(path: Path) -> list[dict[str, int]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                year = int(row["year"])
                community = int(row["community"])
                icsd_id = int(row["icsd_id"])
            except Exception:
                continue
            rows.append({"icsd_id": icsd_id, "year": year, "community": community})
    return rows


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


def load_gnome_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "material_id": row["material_id"],
                    "assigned_community": int(row["assigned_community"]),
                    "nearest_centroid_distance": float(row["nearest_centroid_distance"]),
                    "outlier_like": row["outlier_like"] == "True",
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


def nearest_classification(distances: list[float], communities: list[int], thresholds: dict[int, float], percentile: float) -> float:
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


def plot(percentiles: list[int], gnome_vals: list[float], heldout_vals: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percentiles, gnome_vals, marker="o", label="GNoME sample")
    ax.plot(percentiles, heldout_vals, marker="s", label="Held-out ICSD (> split year)")
    ax.set_xlabel("Centroid-distance threshold percentile")
    ax.set_ylabel("In-basin fraction")
    ax.set_title("Sensitivity of in-basin classification threshold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.features)
    rows = load_comm_rows(Path(args.community_assignments))
    sample_rows = load_sample_rows(Path(args.sample_assignments))
    gnome_rows = load_gnome_rows(Path(args.gnome_records))
    community_by_id = {row["icsd_id"]: row["community"] for row in rows}
    aligned_rows = list(sample_rows)
    labels = np.asarray([community_by_id.get(int(row["icsd_id"]), -1) for row in aligned_rows], dtype=int)

    # Reproduce the production PCA frame.
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    Xp = PCA(n_components=min(32, Xs.shape[0], Xs.shape[1]), random_state=42).fit_transform(Xs)
    if Xp.shape[0] != len(aligned_rows):
        raise ValueError(f"Feature rows ({Xp.shape[0]}) != aligned assignment rows ({len(aligned_rows)})")
    valid_year = np.asarray([row["year"] is not None for row in aligned_rows], dtype=bool)
    years = np.asarray([int(row["year"]) for row in aligned_rows if row["year"] is not None], dtype=int)
    Xp = Xp[valid_year]
    labels = labels[valid_year]

    percentiles = list(range(50, 100, 5)) + [99]
    gnome_curve = []
    heldout_curve = []

    train_mask = years <= args.holdout_year
    holdout_mask = years > args.holdout_year
    train_X = Xp[train_mask]
    train_labels = labels[train_mask]
    holdout_X = Xp[holdout_mask]
    holdout_labels = labels[holdout_mask]
    valid_holdout = holdout_labels >= 0

    for percentile in percentiles:
        centroids, thresholds = centroid_thresholds(train_X, train_labels, percentile)
        gnome_curve.append(
            nearest_classification(
                [float(r["nearest_centroid_distance"]) for r in gnome_rows],
                [int(r["assigned_community"]) for r in gnome_rows],
                thresholds,
                percentile,
            )
        )

        # Held-out ICSD baseline using training centroids/thresholds only.
        holdout_distances = []
        holdout_comms = []
        train_centroid_arr = np.vstack([centroids[c] for c in sorted(centroids)])
        centroid_ids = np.array(sorted(centroids))
        for x in holdout_X[valid_holdout]:
            dists = np.linalg.norm(train_centroid_arr - x, axis=1)
            idx = int(np.argmin(dists))
            holdout_distances.append(float(dists[idx]))
            holdout_comms.append(int(centroid_ids[idx]))
        heldout_curve.append(nearest_classification(holdout_distances, holdout_comms, thresholds, percentile))

    summary = {
        "holdout_year": int(args.holdout_year),
        "n_gnome": int(len(gnome_rows)),
        "n_train_icsd": int(np.sum(train_mask)),
        "n_holdout_icsd": int(np.sum(holdout_mask)),
        "threshold_sensitivity": [
            {"percentile": int(p), "gnome_in_basin_fraction": float(g), "heldout_icsd_in_basin_fraction": float(h)}
            for p, g, h in zip(percentiles, gnome_curve, heldout_curve)
        ],
        "gnome_in_basin_fraction_p95": float(gnome_curve[percentiles.index(95)]),
        "heldout_icsd_in_basin_fraction_p95": float(heldout_curve[percentiles.index(95)]),
    }
    (out_dir / "gnome_baselines_summary.json").write_text(json.dumps(summary, indent=2))
    plot(percentiles, gnome_curve, heldout_curve, out_dir / "gnome_threshold_sensitivity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
