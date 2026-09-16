#!/usr/bin/env python3
"""Compare stable-Louvain gain tolerances on one saved neighbor graph.

The audit deliberately reuses a saved NNDescent graph so that only the
Louvain local-move tolerance changes.  It writes the assignments for every
tolerance and a strict JSON comparison, including temporal endpoint metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_representations as analysis  # noqa: E402
from icsd_graph_time_evolution import compute_temporal_metrics  # noqa: E402


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tolerance_key(value: float) -> str:
    return f"{value:.0e}".replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-ids", type=Path, required=True)
    parser.add_argument("--neighbor-ids", type=Path, required=True)
    parser.add_argument("--neighbors", type=Path, required=True)
    parser.add_argument("--production-labels", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tolerances = sorted(set(args.tolerance), reverse=True)
    if any(not np.isfinite(value) or value <= 0 for value in tolerances):
        parser.error("Tolerances must be positive finite numbers")

    feature_ids = analysis.read_ids(args.feature_ids)
    neighbor_ids = analysis.read_ids(args.neighbor_ids)
    if len(feature_ids) != len(set(feature_ids.tolist())):
        raise ValueError("Feature IDs are not unique")
    feature_rows = {int(iid): row for row, iid in enumerate(feature_ids)}
    if any(int(iid) not in feature_rows for iid in neighbor_ids):
        raise ValueError("Neighbor cohort extends outside the feature rows")
    selected_rows = [feature_rows[int(iid)] for iid in neighbor_ids]
    features = np.load(args.features, mmap_mode="r", allow_pickle=False)
    if features.shape[0] != len(feature_ids) or not np.isfinite(features).all():
        raise ValueError("Feature array and feature IDs do not align")
    features = np.asarray(features[selected_rows])

    saved = np.load(args.neighbors, allow_pickle=False)
    indices = saved["indices"]
    distances = saved["distances"]
    if indices.shape != distances.shape or indices.shape[0] != len(neighbor_ids):
        raise ValueError("Saved neighbor arrays do not match the cohort")
    if not np.isfinite(distances).all():
        raise ValueError("Saved neighbor distances are not finite")
    graph, sigma = analysis.mutual_graph(
        indices, distances, "historical_graphlet_temporal",
    )

    labels = analysis.read_labels(args.production_labels)
    years = [labels[int(iid)]["year"] for iid in neighbor_ids]
    reference = np.asarray(
        [labels[int(iid)]["community"] for iid in neighbor_ids], dtype=np.int64,
    )
    rows = [
        {"icsd_id": int(iid), "year": year, "community": int(community)}
        for iid, year, community in zip(neighbor_ids, years, reference)
    ]
    if any(year is None for year in years):
        raise ValueError("Tolerance audit requires a dated cohort")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "purpose": "isolate stable-Louvain node-move gain tolerance on one fixed saved NNDescent graph",
        "inputs": {
            "features_sha256": digest(args.features),
            "feature_ids_sha256": digest(args.feature_ids),
            "neighbor_ids_sha256": digest(args.neighbor_ids),
            "neighbors_sha256": digest(args.neighbors),
            "production_labels_sha256": digest(args.production_labels),
            "analyzer_sha256": digest(Path(analysis.__file__)),
        },
        "fixed_protocol": {
            "n_entries": len(neighbor_ids),
            "n_edges": graph.number_of_edges(),
            "knn_k": indices.shape[1],
            "metric": "euclidean",
            "mutual_knn": True,
            "sigma": sigma,
            "weight": "exp(-distance^2/sigma^2)",
            "louvain_seed": 42,
            "louvain_level_modularity_threshold": analysis.LOUVAIN_LEVEL_MODULARITY_THRESHOLD,
            "louvain_max_local_move_sweeps": analysis.LOUVAIN_MAX_LOCAL_MOVE_SWEEPS,
            "minimum_community_size": 10,
        },
        "tolerances": {},
        "comparisons": {},
    }
    assignments = {}
    import networkx as nx
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    for tolerance in tolerances:
        key = tolerance_key(tolerance)
        communities, diagnostics = analysis.stable_louvain_communities(
            graph,
            weight="weight",
            resolution=1.0,
            seed=42,
            move_gain_tolerance=tolerance,
        )
        alternate = np.full(len(neighbor_ids), -1, dtype=np.int64)
        next_label = 0
        for members in communities:
            if len(members) >= 10:
                alternate[list(members)] = next_label
                next_label += 1
        assignments[key] = alternate
        assignment_path = args.output_dir / f"assignments_{key}.npy"
        np.save(assignment_path, alternate, allow_pickle=False)
        temporal, _, _, _ = compute_temporal_metrics(
            features, rows, alternate, graph, 25,
        )
        accepted_gains = [
            level["minimum_accepted_gain"]
            for level in diagnostics["level_diagnostics"]
            if level["minimum_accepted_gain"] is not None
        ]
        result["tolerances"][key] = {
            "node_move_gain_tolerance": tolerance,
            "converged": diagnostics["termination"] in {
                "no_local_improvement", "level_modularity_threshold", "empty_graph",
            },
            "louvain_diagnostics": diagnostics,
            "minimum_accepted_gain": min(accepted_gains) if accepted_gains else None,
            "modularity_before_small_community_filter": float(
                nx.algorithms.community.modularity(
                    graph, communities, weight="weight", resolution=1.0,
                )
            ),
            "n_communities": next_label,
            "n_noise": int(np.sum(alternate < 0)),
            "comparison_to_production": analysis.partition_metrics(reference, alternate),
            "temporal_endpoints": {
                decade: temporal["by_decade"][decade]
                for decade in ("1930s", "2010s")
            },
            "assignments_sha256": digest(assignment_path),
        }

    for left_index, left in enumerate(tolerances):
        for right in tolerances[left_index + 1:]:
            left_key = tolerance_key(left)
            right_key = tolerance_key(right)
            a = assignments[left_key]
            b = assignments[right_key]
            result["comparisons"][f"{left_key}_vs_{right_key}"] = {
                "exact_label_array_equal": bool(np.array_equal(a, b)),
                "n_numeric_labels_different": int(np.sum(a != b)),
                "adjusted_rand_index": float(adjusted_rand_score(a, b)),
                "normalized_mutual_information_arithmetic": float(
                    normalized_mutual_info_score(a, b)
                ),
                "noise_transition_counts": {
                    "nonnoise_to_nonnoise": int(np.sum((a >= 0) & (b >= 0))),
                    "nonnoise_to_noise": int(np.sum((a >= 0) & (b < 0))),
                    "noise_to_nonnoise": int(np.sum((a < 0) & (b >= 0))),
                    "noise_to_noise": int(np.sum((a < 0) & (b < 0))),
                },
            }

    output = args.output_dir / "tolerance_audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
