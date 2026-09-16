#!/usr/bin/env python3
"""Build a formula-level structural graph and compare it to the TRI network.

Supports SI §S2. Collapses the per-entry structural k-NN graph to a
formula-level graph: each node is a normalized nominal composition appearing at least
``--min-formula-count`` times in ICSD; an edge exists between two
formulas if at least one structural-graph edge connects entries with
those formulas. Reports per-formula structural-graph degree,
clustering, and k-core number, then aligns to TRI per-formula scalars
to test whether high-TRI-degree formulas are also high-structural-
graph-degree formulas.

Inputs:
  --features-pca         Frozen ICSD PCA features.
  --sample-assignments   icsd_id (+ year) per row matching features.
  --icsd-index           ICSD index CSV with formulas.
  --tri-dir              TRI network directory.
  --k, --mutual-knn      k-NN graph reconstruction parameters.
  --min-formula-count    Minimum formula count to include (default 3).

Outputs (under ``--output-dir``):
  formula_graph_shared.csv     Per-shared-formula joined record.
  formula_graph_summary.json   Spearman correlations between TRI and
                                structural formula-graph metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from pymatgen.core import Composition
from sklearn.neighbors import NearestNeighbors

from formula_conventions import normalized_fraction_key


FormulaIdentity = tuple[tuple[str, ...], tuple[float, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a formula-collapsed structural graph and compare it to TRI.")
    parser.add_argument("--features-pca", required=True)
    parser.add_argument("--sample-assignments", required=True)
    parser.add_argument("--icsd-index", required=True)
    parser.add_argument("--tri-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--mutual-knn", action="store_true")
    parser.add_argument("--min-formula-count", type=int, default=3)
    return parser.parse_args()


def reduce_formula(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return Composition(text).reduced_formula
    except Exception:
        return None


def formula_identity(text: str) -> FormulaIdentity | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return normalized_fraction_key(text, decimals=12)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return num / (denx * deny)


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def load_icsd_index(path: Path) -> dict[int, dict[str, object]]:
    out = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                icsd_id = int(row["cif_names"])
            except Exception:
                continue
            try:
                year = int(row["publication_year"]) if row.get("publication_year", "").strip() else None
            except Exception:
                year = None
            out[icsd_id] = {
                "formula": reduce_formula(row.get("name", "")),
                "formula_identity": formula_identity(row.get("name", "")),
                "year": year,
            }
    return out


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


def load_tri_existing(path: Path) -> dict[FormulaIdentity, dict[str, object]]:
    data = json.loads(path.read_text())
    out = {}
    for formula, attrs in data.items():
        reduced = reduce_formula(formula)
        identity = formula_identity(formula)
        if reduced is None or identity is None:
            continue
        if identity in out:
            raise ValueError(f"TRI formulas collide under normalized identity: {formula!r}")
        def scalarize(value):
            if isinstance(value, list):
                value = value[-1] if value else None
            if value is None:
                return None
            try:
                return float(value)
            except Exception:
                return None
        out[identity] = {
            "formula": reduced,
            "tri_formula_raw": formula,
            "tri_deg": float(scalarize(attrs.get("deg")) or 0.0),
            "tri_eigen_cent": float(scalarize(attrs.get("eigen_cent")) or 0.0),
            "tri_clus_coeff": float(scalarize(attrs.get("clus_coeff")) or 0.0),
            "tri_discovery": int(scalarize(attrs.get("discovery")) or 0),
        }
    return out


def build_graph(
    X: np.ndarray, labels: list[FormulaIdentity], k: int, mutual_knn: bool
) -> nx.Graph:
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    # A duplicate vector may precede the query in a zero-distance tie.
    neighbors = [[(int(j), float(d)) for j, d in zip(js, ds) if int(j) != i][:k]
                 for i, (js, ds) in enumerate(zip(indices, distances))]
    neighbor_sets = [{j for j, _ in row} for row in neighbors]
    positive = np.asarray([d for row in neighbors for _, d in row])
    sigma = float(np.median(positive[positive > 0])) if np.any(positive > 0) else 1.0
    sigma = max(sigma, 1e-8)

    G = nx.Graph()
    for label in labels:
        G.add_node(label)
    for i, label_i in enumerate(labels):
        for j, dist in neighbors[i]:
            j = int(j)
            if i == j:
                continue
            if mutual_knn and i not in neighbor_sets[j]:
                continue
            label_j = labels[j]
            weight = math.exp(-((float(dist) / sigma) ** 2))
            if G.has_edge(label_i, label_j):
                if weight > G[label_i][label_j]["weight"]:
                    G[label_i][label_j]["weight"] = weight
            else:
                G.add_edge(label_i, label_j, weight=weight)
    return G


def eigenvector_centrality_by_component(graph: nx.Graph) -> dict[FormulaIdentity, float]:
    if graph.number_of_nodes() == 0:
        return {}
    out: dict[FormulaIdentity, float] = {}
    for nodes in nx.connected_components(graph):
        sub = graph.subgraph(nodes).copy()
        if sub.number_of_nodes() == 1:
            node = next(iter(sub.nodes()))
            out[node] = 1.0
            continue
        try:
            vals = nx.eigenvector_centrality_numpy(sub, weight="weight")
        except Exception:
            vals = nx.eigenvector_centrality(sub, weight="weight", max_iter=500, tol=1e-06)
        out.update({k: float(v) for k, v in vals.items()})
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.features_pca)
    sample_rows = load_sample_rows(Path(args.sample_assignments))
    if X.shape[0] != len(sample_rows):
        raise ValueError(f"Feature rows ({X.shape[0]}) != sample rows ({len(sample_rows)})")
    index = load_icsd_index(Path(args.icsd_index))
    tri = load_tri_existing(Path(args.tri_dir) / "data" / "NetworkParams_ExistingMaterials_v1.1.json")

    formula_members: dict[FormulaIdentity, list[int]] = defaultdict(list)
    formula_years: dict[FormulaIdentity, list[int]] = defaultdict(list)
    for i, row in enumerate(sample_rows):
        meta = index.get(int(row["icsd_id"]))
        if not meta:
            continue
        identity = meta["formula_identity"]
        if identity is None:
            continue
        formula_members[identity].append(i)
        year = meta["year"]
        if year is not None:
            formula_years[identity].append(int(year))

    kept = {f: idxs for f, idxs in formula_members.items() if len(idxs) >= args.min_formula_count}
    formulas = sorted(kept)
    centroids = np.vstack([X[kept[f]].mean(axis=0) for f in formulas])
    graph = build_graph(centroids, formulas, args.k, args.mutual_knn)

    clustering = nx.clustering(graph, weight="weight")
    core = nx.core_number(graph) if graph.number_of_nodes() else {}
    degree = dict(graph.degree())
    eig = eigenvector_centrality_by_component(graph)
    between = nx.betweenness_centrality(graph, weight="weight") if graph.number_of_nodes() else {}

    shared = sorted(set(formulas) & set(tri))
    rows = []
    for identity in shared:
        rows.append(
            {
                "formula": tri[identity]["formula"],
                "n_entries": int(len(kept[identity])),
                "first_year": min(formula_years[identity]) if formula_years[identity] else None,
                "struct_deg": int(degree.get(identity, 0)),
                "struct_core": int(core.get(identity, 0)),
                "struct_clustering": float(clustering.get(identity, 0.0)),
                "struct_eigen": float(eig.get(identity, 0.0)),
                "struct_betweenness": float(between.get(identity, 0.0)),
                "tri_deg": float(tri[identity]["tri_deg"]),
                "tri_eigen_cent": float(tri[identity]["tri_eigen_cent"]),
                "tri_clus_coeff": float(tri[identity]["tri_clus_coeff"]),
                "tri_discovery": int(tri[identity]["tri_discovery"]),
            }
        )

    with (out_dir / "formula_graph_shared.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "formula_identity": "element-sorted normalized atomic fractions rounded to 12 decimal places",
        "input_sha256": {
            "features_pca": sha256_file(Path(args.features_pca)),
            "sample_assignments": sha256_file(Path(args.sample_assignments)),
            "icsd_index": sha256_file(Path(args.icsd_index)),
            "tri_existing_materials": sha256_file(
                Path(args.tri_dir) / "data" / "NetworkParams_ExistingMaterials_v1.1.json"
            ),
        },
        "n_formula_nodes": int(len(formulas)),
        "n_formula_edges": int(graph.number_of_edges()),
        "n_shared_formulas": int(len(rows)),
        "spearman_tri_deg_vs_struct_deg": spearman([r["tri_deg"] for r in rows], [r["struct_deg"] for r in rows]),
        "spearman_tri_deg_vs_struct_core": spearman([r["tri_deg"] for r in rows], [r["struct_core"] for r in rows]),
        "spearman_tri_eigen_vs_struct_eigen": spearman([r["tri_eigen_cent"] for r in rows], [r["struct_eigen"] for r in rows]),
        "spearman_tri_clustering_vs_struct_clustering": spearman([r["tri_clus_coeff"] for r in rows], [r["struct_clustering"] for r in rows]),
        "spearman_tri_discovery_vs_struct_first_year": spearman(
            [r["tri_discovery"] for r in rows if r["first_year"] is not None],
            [r["first_year"] for r in rows if r["first_year"] is not None],
        ),
        "top_structural_hubs": sorted(rows, key=lambda r: r["struct_deg"], reverse=True)[:25],
        "top_tri_hubs": sorted(rows, key=lambda r: r["tri_deg"], reverse=True)[:25],
    }
    (out_dir / "formula_graph_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
