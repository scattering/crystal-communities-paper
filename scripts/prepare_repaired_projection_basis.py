#!/usr/bin/env python3
"""Reconstruct and verify the repaired production basis for external projections.

Run on the same environment as production. The reference coordinates remain the
saved PCA matrix; a freshly fitted StandardScaler/PCA must reproduce that matrix
before any basis or radius artifact is written. No pickle or credentials are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS
from feature_provenance import require_feature_version


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assignment_rows(path, label):
    with Path(path).open(newline="") as handle:
        rows = [(int(r["icsd_id"]), int(r["year"]) if r["year"].strip() else None,
                 int(r[label])) for r in csv.DictReader(handle)]
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate ICSD IDs in {path}")
    return rows


def community_reference(coordinates, rows):
    labels = np.array([r[2] for r in rows])
    communities = np.unique(labels[labels >= 0])
    if not len(communities):
        raise ValueError("Reference has no assigned communities")
    centroids, p95, p50, counts, births, within = [], [], [], [], [], []
    for community in communities:
        indices = np.flatnonzero(labels == community)
        values = coordinates[indices]
        center = values.mean(axis=0)
        distances = np.linalg.norm(values - center, axis=1)
        centroids.append(center)
        p95.append(np.quantile(distances, .95))
        p50.append(np.quantile(distances, .5))
        counts.append(len(indices))
        years = [rows[i][1] for i in indices if rows[i][1] is not None]
        births.append(min(years) if years else -1)
        within.append(distances)
    arrays = {"communities": communities, "centroids": np.array(centroids),
              "p95": np.array(p95), "p50": np.array(p50),
              "counts": np.array(counts), "birth_years": np.array(births)}
    return arrays, float(np.quantile(np.concatenate(within), .95))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("icsd-features", "icsd-pca", "sample-assignments", "community-assignments", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    metadata = require_feature_version(args.icsd_features)
    if metadata.get("neighbor_settings") != NEIGHBOR_SETTINGS or metadata.get("local_mode") != "matminer_ops":
        raise ValueError("Expected repaired production matminer_ops neighbour settings")
    samples = assignment_rows(args.sample_assignments, "cluster")
    rows = assignment_rows(args.community_assignments, "community")
    if [(i, y) for i, y, _ in samples] != [(i, y) for i, y, _ in rows]:
        raise ValueError("Sample/community ID or year order differs")
    raw = np.load(args.icsd_features, mmap_mode="r", allow_pickle=False)
    saved = np.load(args.icsd_pca, mmap_mode="r", allow_pickle=False)
    if raw.shape != (len(rows), 213) or saved.ndim != 2 or len(saved) != len(rows):
        raise ValueError("Raw/PCA/assignment dimensions disagree; expected production 213-D rows")
    if not np.isfinite(raw).all() or not np.isfinite(saved).all():
        raise ValueError("Nonfinite reference features")
    if metadata.get("sample_size_featurized") != len(rows):
        raise ValueError("Reference summary success count disagrees")
    print(f"Reconstructing StandardScaler/PCA for {raw.shape}, seed {args.seed}", flush=True)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(raw)
    pca = PCA(n_components=saved.shape[1], random_state=args.seed)
    reconstructed = pca.fit_transform(standardized)
    max_error = float(np.max(np.abs(reconstructed - saved)))
    if not np.allclose(reconstructed, saved, rtol=1e-7, atol=1e-8):
        raise ValueError(f"Reconstructed PCA differs from the saved map (max absolute error {max_error}); use the original production environment/settings")
    # Randomized PCA fit_transform can differ from transform; record this
    # separately without changing the canonical saved reference coordinates.
    transform_error = float(np.max(np.abs(pca.transform(standardized) - saved)))
    arrays, pooled = community_reference(saved, rows)
    worker_hashes = {name: sha256_file(Path(__file__).parent / name)
                     for name in ("icsd_densify_worker.py", "crystal_neighbors.py")}
    provenance = {
        "feature_version": FEATURE_VERSION, "neighbor_settings": dict(NEIGHBOR_SETTINGS),
        "local_mode": "matminer_ops", "wl_iters": int(metadata["wl_iters"]),
        "n_reference_rows": len(rows), "n_features": raw.shape[1], "pca_dim": saved.shape[1],
        "seed": args.seed, "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "pca_solver_used": pca._fit_svd_solver, "worker_sha256": worker_hashes,
        "reference_files": {name: {"path": str(getattr(args, name).resolve()),
                                   "sha256": sha256_file(getattr(args, name))}
                            for name in ("icsd_features", "icsd_pca", "sample_assignments", "community_assignments")},
        "row_ids_and_years_verified": True, "fit_transform_reproduces_saved_pca": True,
        "fit_transform_max_abs_error": max_error,
        "transform_vs_saved_max_abs_error": transform_error,
        "transform_vs_saved_allclose": bool(np.allclose(pca.transform(standardized), saved, rtol=1e-7, atol=1e-8)),
        "reference_coordinate_rule": "Centroids/radii use the saved production PCA coordinates; external projection uses the reconstructed PCA transform.",
        "radius_rule": "Per-community Euclidean distance to centroid, numpy.quantile p95; in-basin includes equality. Noise is excluded when estimating centroids/radii.",
    }
    arrays.update(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                  pca_mean=pca.mean_, pca_components=pca.components_,
                  metadata=np.array(json.dumps(provenance, sort_keys=True)))
    thresholds = {
        "feature_version": FEATURE_VERSION, "pooled_p95_threshold": pooled,
        "per_community_p95_threshold": {str(c): float(v) for c, v in zip(arrays["communities"], arrays["p95"])},
        "per_community_median_distance": {str(c): float(v) for c, v in zip(arrays["communities"], arrays["p50"])},
        "per_community_size": {str(c): int(v) for c, v in zip(arrays["communities"], arrays["counts"])},
        "per_community_birth_year": {str(c): int(v) if v >= 0 else None for c, v in zip(arrays["communities"], arrays["birth_years"])},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "projection_basis.npz", **arrays)
    (args.out_dir / "projection_basis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.out_dir / "community_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "n_communities": len(arrays["communities"]),
                      "fit_transform_max_abs_error": max_error, "transform_vs_saved_max_abs_error": transform_error}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
