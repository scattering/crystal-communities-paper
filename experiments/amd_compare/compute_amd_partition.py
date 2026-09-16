#!/usr/bin/env python3
"""Build mutual-kNN graph on AMD vectors + run Louvain to produce an
AMD-based community partition for ARI/NMI comparison against labels3, the
June 2026 partition superseded by the CrystalWeave partition.

Uses the same graph-construction parameters as the production labels3 partition
(k=16 mutual kNN, gamma=1.0 Louvain) so the comparison is apples-to-apples
under §S1.8.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

K_NN = 16
GAMMA = 1.0
RNG = 42


def build_mutual_knn_graph(X: np.ndarray, k: int) -> nx.Graph:
    nbrs = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    dists, idxs = nbrs.kneighbors(X)
    n = X.shape[0]
    # Gaussian edge weights with sigma = median k-NN distance (excluding self)
    sigma = float(np.median(dists[:, 1:]))
    print(f"  median k-NN distance: {sigma:.4f}", flush=True)
    edges: dict[tuple[int, int], float] = {}
    nbr_sets = [set(idxs[i, 1:]) for i in range(n)]
    for i in range(n):
        for j_pos, j in enumerate(idxs[i, 1:], start=1):
            if i in nbr_sets[j]:  # mutual
                d = dists[i, j_pos]
                w = float(np.exp(-(d ** 2) / (2 * sigma ** 2)))
                key = (i, int(j)) if i < int(j) else (int(j), i)
                if key not in edges:
                    edges[key] = w
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from((u, v, w) for (u, v), w in edges.items())
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, mean-deg={2*G.number_of_edges()/G.number_of_nodes():.2f}", flush=True)
    return G


def louvain(G: nx.Graph, gamma: float = 1.0, seed: int = RNG) -> dict[int, int]:
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, weight="weight", resolution=gamma, seed=seed)
    label_of: dict[int, int] = {}
    for cid, members in enumerate(communities):
        for n in members:
            label_of[n] = cid
    print(f"  louvain: {len(communities)} communities", flush=True)
    return label_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="amd_features.npy")
    ap.add_argument("--ids", required=True, help="amd_features.npy.ids.json")
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X = np.load(args.features).astype(np.float64)
    with open(args.ids) as f:
        ids = json.load(f)
    print(f"  features: {X.shape},  ids: {len(ids)}", flush=True)

    # Standardize so Euclidean k-NN is scale-balanced
    X_std = StandardScaler().fit_transform(X)

    G = build_mutual_knn_graph(X_std, K_NN)
    labels = louvain(G, gamma=GAMMA)

    with (out / "amd_community_assignments.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["icsd_id", "community"])
        for i, iid in enumerate(ids):
            w.writerow([iid, labels.get(i, -1)])
    print(f"  wrote {out / 'amd_community_assignments.csv'}", flush=True)

    with (out / "amd_partition_meta.json").open("w") as f:
        json.dump({
            "k_nn": K_NN, "gamma": GAMMA, "n_features": X.shape[1],
            "n_nodes": X.shape[0], "n_edges": G.number_of_edges(),
            "n_communities": len(set(labels.values())),
        }, f, indent=2)


if __name__ == "__main__":
    main()
