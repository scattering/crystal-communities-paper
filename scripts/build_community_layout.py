#!/usr/bin/env python3
"""Compute a graph-aware 2-D layout of structural communities.

The PCA scatter that drives the placement-figure dashboard collapses the
top communities into a tiny region — they overlap visually even though
they are well-separated in the underlying k-NN graph used for community
detection.  This script re-lays out the *community-contracted* graph:

  1. build a k-NN graph in the frozen structural-feature space
  2. count edges that cross community boundaries  ->  inter-community
     edge weights
  3. run a force-directed (NetworkX `spring_layout`, Fruchterman-Reingold)
     layout on the community-contracted graph
  4. emit a CSV the dashboard / viz scripts can load instead of the raw
     centroid PCA positions

The output schema is the smallest thing the dashboard needs:

    community,x,y,size,intercommunity_edge_count,top_neighbor

`top_neighbor` is the community id with the largest inter-community edge
count (useful for drawing a single-strongest-link spanning view if the
viewer is bandwidth-limited).

Usage:
    python scripts/build_community_layout.py \
        --features /path/to/features_pca.npy \
        --community-assignments /path/to/community_assignments.csv \
        --output /path/to/community_layout.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="features_pca.npy")
    p.add_argument("--community-assignments", required=True, help="community_assignments.csv")
    p.add_argument("--output", required=True, help="destination CSV")
    p.add_argument("--k", type=int, default=15, help="neighbours per node in the structural k-NN graph")
    p.add_argument(
        "--min-community-size",
        type=int,
        default=20,
        help="drop communities smaller than this from the layout (they tend to clutter the force layout)",
    )
    p.add_argument(
        "--top-communities",
        type=int,
        default=0,
        help="if >0, keep only the N largest communities",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="spring_layout iterations; bump for smoother layouts on dense graphs",
    )
    return p.parse_args()


def load_assignments(path: Path) -> list[int]:
    out: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.append(int(row["community"]))
            except (KeyError, TypeError, ValueError):
                out.append(-1)
    return out


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    X = np.load(args.features)
    labels = load_assignments(Path(args.community_assignments))
    if len(X) != len(labels):
        raise SystemExit(
            f"row mismatch: features has {len(X)} rows, community_assignments has {len(labels)}"
        )

    labels_arr = np.asarray(labels, dtype=int)
    keep_mask = labels_arr >= 0
    if not keep_mask.any():
        raise SystemExit("no community labels >= 0 in the assignments file")

    sizes = Counter(labels_arr[keep_mask].tolist())

    if args.top_communities > 0:
        top = {c for c, _ in sizes.most_common(args.top_communities)}
    else:
        top = {c for c, n in sizes.items() if n >= args.min_community_size}
    if not top:
        raise SystemExit(
            f"no communities pass min-size {args.min_community_size} or top-N {args.top_communities}"
        )
    keep_mask &= np.isin(labels_arr, list(top))

    Xk = X[keep_mask]
    Lk = labels_arr[keep_mask]

    n_neighbors = min(args.k + 1, len(Xk))
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(Xk)
    _, knn_idx = nn.kneighbors(Xk)

    # Inter-community edge counts. Skip self-edges (knn_idx[:, 0] == i).
    inter: dict[tuple[int, int], int] = {}
    for i in range(len(Xk)):
        ci = int(Lk[i])
        for j in knn_idx[i, 1:]:
            cj = int(Lk[j])
            if cj == ci:
                continue
            key = (ci, cj) if ci < cj else (cj, ci)
            inter[key] = inter.get(key, 0) + 1

    G = nx.Graph()
    for c in top:
        G.add_node(int(c), size=int(sizes[c]))
    for (a, b), w in inter.items():
        # Symmetrise the count in undirected graph: each (i, j) and (j, i)
        # cross-edge is counted once because we always store (min, max).
        G.add_edge(int(a), int(b), weight=int(w))

    # Spring layout uses edge weights as the spring constant: heavier weight =
    # stronger pull. Two communities with many cross-edges end up closer.
    pos = nx.spring_layout(
        G,
        weight="weight",
        seed=args.seed,
        iterations=args.iterations,
        k=None,  # auto: 1/sqrt(n)
    )

    rows = []
    for c in sorted(top):
        x, y = pos.get(int(c), (rng.normal(), rng.normal()))
        nbrs = [(int(b if a == c else a), int(d["weight"])) for a, b, d in G.edges(c, data=True)]
        nbrs.sort(key=lambda t: -t[1])
        top_neighbor = nbrs[0][0] if nbrs else ""
        ic_total = sum(d["weight"] for _, _, d in G.edges(c, data=True))
        rows.append({
            "community": int(c),
            "x": f"{float(x):.6f}",
            "y": f"{float(y):.6f}",
            "size": int(sizes[c]),
            "intercommunity_edge_count": int(ic_total),
            "top_neighbor": top_neighbor,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["community", "x", "y", "size", "intercommunity_edge_count", "top_neighbor"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(
        f"wrote {len(rows)} community layout rows to {out_path} "
        f"(k={args.k}, n_inter_edges={sum(d['weight'] for _,_,d in G.edges(data=True))})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
