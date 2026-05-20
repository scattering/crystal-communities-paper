#!/usr/bin/env python3
"""Cross-check the TRI thermodynamic network against ICSD structural roles.

Supports SI §S2 TRI comparison. Loads the Aykol-et-al. Thermodynamic
Reachability Index (TRI) network from ``--tri-dir``, finds reduced
formulas shared between TRI and the ICSD index, and asks whether
formulas that are *hubs* in the chemistry-only TRI graph (high
``tri_deg`` / eigenvector centrality) are also *bridges* in the
structural community graph: nodes whose nearest neighbors span
multiple Louvain communities.

Pipeline:
  1. Load TRI per-formula metrics (degree, eigenvector centrality,
     discovery year).
  2. Reconstruct the weighted k-NN structural graph from
     ``--features-file`` with the same ``--k`` / ``--mutual-knn`` as
     the production community detection.
  3. Collapse to a *formula graph*: edges between two formulas if at
     least one structural-graph edge connects entries with those
     formulas. Compute degree, clustering, k-core number, and a
     per-formula bridge rate (fraction of entries whose neighborhood
     spans ≥2 communities) and cross-community rate (≥1 cross edge).
  4. For each shared formula build a per-row record with TRI metrics,
     ICSD community statistics (dominant community, fragmentation
     entropy, outlier fraction), birth-year lags between TRI discovery
     and dominant-community birth, and a stepping-stone class
     (joins_existing / co_birth / community_after_formula).
  5. Spearman correlations of TRI degree against structural metrics
     and a hub-quartile breakdown of stepping-stone classes.

Inputs:
  --tri-dir, --icsd-index, --community-assignments, --features-file,
  --k (default 16), --mutual-knn, --output-dir.

Outputs:
  tri_structural_role_records.csv   per-formula joined record.
  tri_structural_role_summary.json  Spearman ρ table + hub-quartile
                                     stepping-stone counts.
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
from pymatgen.core import Composition
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TRI hubness against ICSD structural roles.")
    parser.add_argument("--tri-dir", required=True)
    parser.add_argument("--icsd-index", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--features-file", required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--mutual-knn", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def reduce_formula(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return Composition(text).reduced_formula
    except Exception:
        return None


def scalarize(value: object) -> float | int | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[-1]
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        return float(value)
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
    return num / (denx * deny)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for n in counts.values():
        p = n / total
        ent -= p * math.log(p)
    return ent


def load_tri_existing(path: Path) -> dict[str, dict[str, float | int]]:
    data = json.loads(path.read_text())
    out: dict[str, dict[str, float | int]] = {}
    for formula, attrs in data.items():
        reduced = reduce_formula(formula)
        if reduced is None:
            continue
        out[reduced] = {
            "tri_formula_raw": formula,
            "tri_deg": float(scalarize(attrs.get("deg")) or 0.0),
            "tri_deg_cent": float(scalarize(attrs.get("deg_cent")) or 0.0),
            "tri_eigen_cent": float(scalarize(attrs.get("eigen_cent")) or 0.0),
            "tri_clus_coeff": float(scalarize(attrs.get("clus_coeff")) or 0.0),
            "tri_deg_neigh": float(scalarize(attrs.get("deg_neigh")) or 0.0),
            "tri_shortest_path": float(scalarize(attrs.get("shortest_path")) or 0.0),
            "tri_discovery": int(scalarize(attrs.get("discovery")) or 0),
        }
    return out


def load_icsd_index(path: Path) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                icsd_id = int(row["cif_names"])
            except Exception:
                continue
            formula = reduce_formula(row.get("name", ""))
            if formula is None:
                continue
            year = None
            try:
                year = int(row["publication_year"])
            except Exception:
                pass
            out[icsd_id] = {
                "formula": formula,
                "raw_formula": row.get("name", ""),
                "year": year,
            }
    return out


def load_assignment_rows(path: Path) -> list[dict[str, int | None]]:
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


def build_neighbors(X: np.ndarray, k: int, mutual_knn: bool) -> list[list[int]]:
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean", algorithm="auto")
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)
    neighbor_sets = [set(row[1:]) for row in indices]
    out: list[list[int]] = []
    for i in range(len(X)):
        row = []
        for j in indices[i, 1:]:
            j = int(j)
            if mutual_knn and i not in neighbor_sets[j]:
                continue
            row.append(j)
        out.append(row)
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tri = load_tri_existing(Path(args.tri_dir) / "data" / "NetworkParams_ExistingMaterials_v1.1.json")
    icsd_index = load_icsd_index(Path(args.icsd_index))
    rows = load_assignment_rows(Path(args.community_assignments))
    X = np.load(args.features_file)
    if len(rows) != len(X):
        raise ValueError(f"assignment rows ({len(rows)}) != feature rows ({len(X)})")

    community_birth: dict[int, int] = {}
    for row in rows:
        community = int(row["community"])
        year = row["year"]
        if community < 0 or year is None:
            continue
        community_birth[community] = min(year, community_birth.get(community, year))

    formula_nodes: dict[str, list[int]] = defaultdict(list)
    formula_years: dict[str, list[int]] = defaultdict(list)
    formula_communities: dict[str, list[int]] = defaultdict(list)

    formula_for_node: list[str | None] = [None] * len(rows)
    community_for_node: list[int] = [int(r["community"]) for r in rows]
    for idx, row in enumerate(rows):
        meta = icsd_index.get(int(row["icsd_id"]))
        if meta is None:
            continue
        formula = str(meta["formula"])
        formula_for_node[idx] = formula
        formula_nodes[formula].append(idx)
        if meta["year"] is not None:
            formula_years[formula].append(int(meta["year"]))
        formula_communities[formula].append(int(row["community"]))

    shared_formulas = sorted(set(tri) & set(formula_nodes))

    neighbors = build_neighbors(X, args.k, args.mutual_knn)

    formula_adj: dict[str, set[str]] = defaultdict(set)
    formula_bridge_hits: Counter[str] = Counter()
    formula_bridge_total: Counter[str] = Counter()
    formula_cross_hits: Counter[str] = Counter()

    for i, nbrs in enumerate(neighbors):
        fi = formula_for_node[i]
        if fi is None or fi not in shared_formulas:
            continue
        ci = community_for_node[i]
        formula_bridge_total[fi] += 1
        nbr_communities = set()
        for j in nbrs:
            fj = formula_for_node[j]
            if fj is None or fj not in shared_formulas:
                continue
            if fi != fj:
                formula_adj[fi].add(fj)
                formula_adj[fj].add(fi)
            cj = community_for_node[j]
            if cj >= 0 and cj != ci:
                nbr_communities.add(cj)
        if nbr_communities:
            formula_cross_hits[fi] += 1
        if len(nbr_communities) >= 2:
            formula_bridge_hits[fi] += 1

    G = nx.Graph()
    G.add_nodes_from(shared_formulas)
    for f, nbrs in formula_adj.items():
        for g in nbrs:
            if f < g:
                G.add_edge(f, g)

    clustering = nx.clustering(G)
    core_num = nx.core_number(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}

    records: list[dict[str, object]] = []
    for formula in shared_formulas:
        comm_counts = Counter(c for c in formula_communities[formula] if c >= 0)
        dominant_comm = comm_counts.most_common(1)[0][0] if comm_counts else None
        dominant_birth = community_birth.get(dominant_comm) if dominant_comm is not None else None
        first_comm_birth = min((community_birth[c] for c in comm_counts if c in community_birth), default=None)
        first_year = min(formula_years[formula]) if formula_years[formula] else None
        outlier_count = sum(1 for c in formula_communities[formula] if c < 0)
        total_count = len(formula_nodes[formula])
        dominant_fraction = (comm_counts[dominant_comm] / total_count) if dominant_comm is not None and total_count else 0.0
        stepping = None
        if first_year is not None and first_comm_birth is not None:
            if first_comm_birth < first_year:
                stepping = "joins_existing"
            elif first_comm_birth == first_year:
                stepping = "co_birth"
            else:
                stepping = "community_after_formula"

        records.append(
            {
                "formula": formula,
                **tri[formula],
                "icsd_n_entries": total_count,
                "icsd_first_year": first_year,
                "icsd_n_communities": len(set(c for c in formula_communities[formula] if c >= 0)),
                "icsd_fragmentation_entropy": entropy_from_counts(comm_counts),
                "icsd_dominant_fraction": dominant_fraction,
                "icsd_dominant_community": dominant_comm,
                "icsd_dominant_community_size": sum(1 for r in rows if int(r["community"]) == dominant_comm) if dominant_comm is not None else 0,
                "icsd_dominant_community_birth": dominant_birth,
                "icsd_first_associated_community_birth": first_comm_birth,
                "icsd_outlier_fraction": outlier_count / total_count if total_count else 0.0,
                "icsd_formula_graph_degree": int(G.degree(formula)),
                "icsd_formula_graph_clustering": float(clustering.get(formula, 0.0)),
                "icsd_formula_graph_core_number": int(core_num.get(formula, 0)),
                "icsd_formula_cross_community_rate": formula_cross_hits[formula] / formula_bridge_total[formula] if formula_bridge_total[formula] else 0.0,
                "icsd_formula_bridge_rate": formula_bridge_hits[formula] / formula_bridge_total[formula] if formula_bridge_total[formula] else 0.0,
                "tri_to_icsd_first_year_lag": (first_year - int(tri[formula]["tri_discovery"])) if first_year is not None and tri[formula]["tri_discovery"] else None,
                "tri_to_dominant_community_birth_lag": (dominant_birth - int(tri[formula]["tri_discovery"])) if dominant_birth is not None and tri[formula]["tri_discovery"] else None,
                "stepping_stone_class": stepping,
            }
        )

    with (out_dir / "tri_structural_role_records.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(records[0].keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    def valid_xy(key_x: str, key_y: str) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for row in records:
            x = row.get(key_x)
            y = row.get(key_y)
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        return xs, ys

    summary = {
        "n_shared_formulas": len(records),
        "corr_tri_deg_vs_formula_graph_degree": spearman(*valid_xy("tri_deg", "icsd_formula_graph_degree")),
        "corr_tri_deg_vs_formula_graph_clustering": spearman(*valid_xy("tri_deg", "icsd_formula_graph_clustering")),
        "corr_tri_deg_vs_formula_graph_core_number": spearman(*valid_xy("tri_deg", "icsd_formula_graph_core_number")),
        "corr_tri_deg_vs_bridge_rate": spearman(*valid_xy("tri_deg", "icsd_formula_bridge_rate")),
        "corr_tri_deg_vs_cross_community_rate": spearman(*valid_xy("tri_deg", "icsd_formula_cross_community_rate")),
        "corr_tri_deg_vs_dominant_community_size": spearman(*valid_xy("tri_deg", "icsd_dominant_community_size")),
        "corr_tri_deg_vs_outlier_fraction": spearman(*valid_xy("tri_deg", "icsd_outlier_fraction")),
        "corr_tri_deg_vs_fragmentation_entropy": spearman(*valid_xy("tri_deg", "icsd_fragmentation_entropy")),
        "corr_tri_deg_vs_dominant_fraction": spearman(*valid_xy("tri_deg", "icsd_dominant_fraction")),
        "corr_tri_eigen_vs_formula_graph_degree": spearman(*valid_xy("tri_eigen_cent", "icsd_formula_graph_degree")),
        "corr_tri_discovery_vs_icsd_first_year": spearman(*valid_xy("tri_discovery", "icsd_first_year")),
        "corr_tri_discovery_vs_dominant_community_birth": spearman(*valid_xy("tri_discovery", "icsd_dominant_community_birth")),
        "corr_tri_discovery_vs_tri_to_dominant_community_birth_lag": spearman(*valid_xy("tri_discovery", "tri_to_dominant_community_birth_lag")),
        "stepping_stone_counts": Counter(row["stepping_stone_class"] for row in records if row["stepping_stone_class"] is not None),
        "top_by_tri_degree": sorted(records, key=lambda r: float(r["tri_deg"]), reverse=True)[:25],
    }

    # Hub quartile stepping-stone view
    sorted_by_deg = sorted(records, key=lambda r: float(r["tri_deg"]))
    qcut = max(1, len(sorted_by_deg) // 4)
    quartiles = {
        "low": sorted_by_deg[:qcut],
        "mid_low": sorted_by_deg[qcut : 2 * qcut],
        "mid_high": sorted_by_deg[2 * qcut : 3 * qcut],
        "high": sorted_by_deg[3 * qcut :],
    }
    hub_quartile = {}
    for name, rows_q in quartiles.items():
        hub_quartile[name] = Counter(r["stepping_stone_class"] for r in rows_q if r["stepping_stone_class"] is not None)
    summary["stepping_stone_by_tri_degree_quartile"] = hub_quartile

    (out_dir / "tri_structural_role_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
