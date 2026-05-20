#!/usr/bin/env python3
"""Reconstruct the structural graph year-by-year and track community growth.

Produces the time-evolution view of the ICSD structural network used
in Figure 2 and in the densification narrative: at each year, only
nodes published up to that year are retained, the same weighted k-NN
graph (matching the ``icsd_graph_community_postprocess.py`` settings
``--k`` / ``--mutual-knn``) is reconstructed on the partial node set,
and per-community size, edge counts, and cumulative-growth trajectories
are recorded for the ``--top-n-communities`` largest communities.

Inputs:
  --run-dir              Densification run directory (provides
                          features_pca.npy / community labels).
  --community-dir        Output directory of
                          ``icsd_graph_community_postprocess.py``.
  --features-file        Default ``features_pca.npy``.
  --assignments-file     Default ``community_assignments.csv``.
  --prototype-labels-file Default ``community_prototype_labels.json``;
                          attaches a human-readable name when present.

Outputs (under ``--output-dir``):
  graph_time_summary.json    per-year and per-decade community stats.
  top_communities.json       cumulative-growth trajectories for the
                              top-N communities (by terminal size),
                              labeled with prototype names where known.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-evolution analysis for ICSD graph communities.")
    parser.add_argument("--run-dir", required=True, help="Finished densification run directory")
    parser.add_argument("--community-dir", required=True, help="Graph community output directory")
    parser.add_argument("--output-dir", required=True, help="Directory for time-evolution outputs")
    parser.add_argument("--features-file", default="features_pca.npy", help="Feature array filename under run-dir")
    parser.add_argument("--assignments-file", default="community_assignments.csv", help="Community assignments filename")
    parser.add_argument("--prototype-labels-file", default="community_prototype_labels.json")
    parser.add_argument("--k", type=int, default=16, help="k for k-NN graph reconstruction")
    parser.add_argument("--mutual-knn", action="store_true", help="Keep only mutual k-NN edges")
    parser.add_argument("--top-n-communities", type=int, default=25, help="Top communities for cumulative growth output")
    return parser.parse_args()


def decade_from_year(year: int | None) -> str:
    if year is None:
        return "unknown"
    return f"{(year // 10) * 10}s"


def load_assignments(path: Path) -> list[dict[str, int | None]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_year = row.get("year", "").strip()
            rows.append(
                {
                    "icsd_id": int(row["icsd_id"]),
                    "year": int(raw_year) if raw_year else None,
                    "community": int(row["community"]),
                }
            )
    return rows


def load_prototype_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    labels: dict[int, str] = {}
    for entry in data:
        community = int(entry["community"])
        matches = entry.get("prototype_matches") or []
        if matches:
            tags = matches[0].get("tags", {})
            label = tags.get("aflow") or tags.get("mineral") or tags.get("strukturbericht")
            if label:
                labels[community] = str(label)
                continue
        labels[community] = f"ICSD {entry['centroid_icsd_id']}"
    return labels


def build_weighted_graph(X: np.ndarray, k: int, mutual_knn: bool) -> nx.Graph:
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    neighbor_sets = [set(row[1:]) for row in indices]

    positive = distances[:, 1:]
    sigma = float(np.median(positive[positive > 0])) if np.any(positive > 0) else 1.0
    sigma = max(sigma, 1e-8)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(X)))
    for i in range(len(X)):
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):
            j = int(j)
            if mutual_knn and i not in neighbor_sets[j]:
                continue
            weight = math.exp(-((float(dist) / sigma) ** 2))
            if graph.has_edge(i, j):
                if weight > graph[i][j]["weight"]:
                    graph[i][j]["weight"] = weight
            else:
                graph.add_edge(i, j, weight=weight)
    return graph


def compute_core_thresholds(X: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for community in sorted({int(v) for v in labels if int(v) >= 0}):
        members = np.flatnonzero(labels == community)
        if len(members) == 0:
            continue
        centroid = X[members].mean(axis=0)
        dists = np.linalg.norm(X[members] - centroid, axis=1)
        thresholds[community] = float(np.median(dists))
    return thresholds


def compute_temporal_metrics(
    X: np.ndarray,
    rows: list[dict[str, int | None]],
    labels: np.ndarray,
    graph: nx.Graph,
    top_n_communities: int,
) -> tuple[dict, list[dict[str, object]], list[dict[str, object]], list[tuple[int, int]]]:
    community_birth_year: dict[int, int] = {}
    for row, label in zip(rows, labels):
        year = row["year"]
        if label < 0 or year is None:
            continue
        community_birth_year[label] = min(year, community_birth_year.get(label, year))

    thresholds = compute_core_thresholds(X, labels)

    order = sorted(range(len(rows)), key=lambda idx: (rows[idx]["year"] is None, rows[idx]["year"] or 10**9, rows[idx]["icsd_id"]))
    active_nodes: set[int] = set()
    by_decade: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    node_events: list[dict[str, object]] = []

    for idx in order:
        row = rows[idx]
        year = row["year"]
        decade = decade_from_year(year)
        label = int(labels[idx])
        by_decade[decade]["n_total"] += 1

        active_neighbors = [nbr for nbr in graph.neighbors(idx) if nbr in active_nodes]
        active_same = [nbr for nbr in active_neighbors if int(labels[nbr]) == label and label >= 0]
        active_other_labels = sorted({int(labels[nbr]) for nbr in active_neighbors if int(labels[nbr]) >= 0 and int(labels[nbr]) != label})
        bridge = len(active_other_labels) >= 2

        event = {
            "icsd_id": int(row["icsd_id"]),
            "year": year,
            "decade": decade,
            "community": label,
            "n_active_neighbors": int(len(active_neighbors)),
            "n_active_same_community_neighbors": int(len(active_same)),
            "n_active_other_communities": int(len(active_other_labels)),
            "is_bridge_attachment": bool(bridge),
        }

        if label < 0:
            by_decade[decade]["n_outlier"] += 1
            event["event_type"] = "outlier"
        else:
            birth = community_birth_year.get(label)
            if birth is not None and year == birth:
                by_decade[decade]["n_cluster_birth_point"] += 1
                event["event_type"] = "community_birth"
            else:
                by_decade[decade]["n_existing_cluster"] += 1
                event["event_type"] = "existing_community"

            if active_same:
                by_decade[decade]["n_same_community_attachment"] += 1
            if active_other_labels:
                by_decade[decade]["n_cross_community_attachment"] += 1
            if bridge:
                by_decade[decade]["n_bridge_attachment"] += 1

            members = np.flatnonzero(labels == label)
            centroid = X[members].mean(axis=0)
            dist = float(np.linalg.norm(X[idx] - centroid))
            threshold = thresholds.get(label, dist)
            event["distance_to_centroid"] = dist
            event["core_threshold"] = threshold
            event["core_periphery"] = "core" if dist <= threshold else "periphery"
            if dist <= threshold:
                by_decade[decade]["n_core_attachment"] += 1
            else:
                by_decade[decade]["n_periphery_attachment"] += 1

        node_events.append(event)
        active_nodes.add(idx)

    for decade, stats in by_decade.items():
        total = stats["n_total"] or 1.0
        for key in [
            "n_outlier",
            "n_cluster_birth_point",
            "n_existing_cluster",
            "n_same_community_attachment",
            "n_cross_community_attachment",
            "n_bridge_attachment",
            "n_core_attachment",
            "n_periphery_attachment",
        ]:
            ratio_key = key.replace("n_", "") + "_ratio"
            stats[ratio_key] = stats[key] / total

    top_counts = Counter(int(v) for v in labels if int(v) >= 0)
    cumulative_rows: list[dict[str, object]] = []
    decades_in_order = sorted({decade_from_year(row["year"]) for row in rows if row["year"] is not None})
    top_communities = top_counts.most_common(top_n_communities)
    top_community_ids = {community for community, _ in top_communities}
    rows_by_decade_community: dict[tuple[str, int], int] = Counter()
    for row, label in zip(rows, labels):
        label = int(label)
        if label < 0 or label not in top_community_ids:
            continue
        rows_by_decade_community[(decade_from_year(row["year"]), label)] += 1

    for community, _ in top_communities:
        running = 0
        for decade in decades_in_order:
            running += rows_by_decade_community.get((decade, community), 0)
            cumulative_rows.append({"community": int(community), "decade": decade, "cumulative_size": running})

    summary = {
        "n_points": int(len(rows)),
        "n_outliers": int(np.sum(labels < 0)),
        "outlier_ratio": float(np.mean(labels < 0)),
        "community_birth_year": {str(k): int(v) for k, v in sorted(community_birth_year.items())},
        "by_decade": {k: dict(v) for k, v in sorted(by_decade.items())},
    }
    return summary, node_events, cumulative_rows, top_communities


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    community_dir = Path(args.community_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(run_dir / args.features_file)
    rows = load_assignments(community_dir / args.assignments_file)
    labels = np.asarray([int(row["community"]) for row in rows], dtype=int)
    if len(rows) != len(X):
        raise ValueError(f"Assignments/features length mismatch: {len(rows)} vs {len(X)}")

    graph = build_weighted_graph(X, args.k, args.mutual_knn)
    summary, node_events, cumulative_rows, top_counts = compute_temporal_metrics(
        X, rows, labels, graph, args.top_n_communities
    )
    prototype_names = load_prototype_names(community_dir / args.prototype_labels_file)

    top_communities = []
    for community, size in top_counts:
        top_communities.append(
            {
                "community": int(community),
                "size": int(size),
                "label": prototype_names.get(int(community), f"community_{int(community)}"),
                "birth_year": summary["community_birth_year"].get(str(int(community))),
            }
        )

    (out_dir / "graph_time_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "top_communities.json").write_text(json.dumps(top_communities, indent=2))

    with (out_dir / "community_growth_by_decade.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["community", "decade", "cumulative_size"])
        writer.writeheader()
        for row in cumulative_rows:
            writer.writerow(row)

    with (out_dir / "node_temporal_events.csv").open("w", newline="") as handle:
        fieldnames = [
            "icsd_id",
            "year",
            "decade",
            "community",
            "event_type",
            "n_active_neighbors",
            "n_active_same_community_neighbors",
            "n_active_other_communities",
            "is_bridge_attachment",
            "distance_to_centroid",
            "core_threshold",
            "core_periphery",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in node_events:
            writer.writerow(row)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
