#!/usr/bin/env python3
"""Sister script to ``analyze_tri_structural_roles.py``: TRI vs. ICSD partition agreement.

Supports SI §S2 TRI comparison. For the formulas shared between the
Aykol et al. TRI network and the ICSD index, this script reports the
agreement between (a) TRI per-formula scalar features, (b) the
graph-Louvain community label, and (c) the HDBSCAN cluster label
(both produced by the production pipeline), via Spearman correlations
and per-formula joined records.

Inputs:
  --tri-dir, --icsd-index, --community-assignments,
  --hdbscan-assignments, --output-dir.

Outputs:
  shared_formula_comparison.csv   Per-formula joined record.
  comparison_summary.json         Spearman ρ between TRI scalars and
                                   structural community / HDBSCAN
                                   labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pymatgen.core import Composition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TRI thermodynamic network metrics with ICSD structural communities.")
    parser.add_argument("--tri-dir", required=True, help="Directory containing unpacked TRI network data")
    parser.add_argument("--icsd-index", required=True, help="ICSD index CSV")
    parser.add_argument("--community-assignments", required=True, help="Graph community assignments CSV")
    parser.add_argument("--hdbscan-assignments", required=True, help="HDBSCAN assignments CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    return parser.parse_args()


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
    return num / (denx * deny)


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def load_tri_existing(path: Path) -> dict[str, dict[str, float | int]]:
    data = json.loads(path.read_text())
    out: dict[str, dict[str, float | int]] = {}

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

    for formula, attrs in data.items():
        reduced = reduce_formula(formula)
        if reduced is None:
            continue
        deg = scalarize(attrs.get("deg", 0.0))
        deg_cent = scalarize(attrs.get("deg_cent", 0.0))
        eigen_cent = scalarize(attrs.get("eigen_cent", 0.0))
        clus_coeff = scalarize(attrs.get("clus_coeff", 0.0))
        deg_neigh = scalarize(attrs.get("deg_neigh", 0.0))
        shortest_path = scalarize(attrs.get("shortest_path", 0.0))
        discovery = scalarize(attrs.get("discovery", 0))
        out[reduced] = {
            "tri_formula_raw": formula,
            "deg": float(deg) if deg is not None else 0.0,
            "deg_cent": float(deg_cent) if deg_cent is not None else 0.0,
            "eigen_cent": float(eigen_cent) if eigen_cent is not None else 0.0,
            "clus_coeff": float(clus_coeff) if clus_coeff is not None else 0.0,
            "deg_neigh": float(deg_neigh) if deg_neigh is not None else 0.0,
            "shortest_path": float(shortest_path) if shortest_path is not None else 0.0,
            "tri_discovery": int(discovery) if discovery is not None else 0,
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
            out[icsd_id] = {"formula": formula, "raw_formula": row.get("name", ""), "year": year}
    return out


def load_assignment_map(path: Path, key_name: str) -> dict[int, int]:
    out: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[int(row["icsd_id"])] = int(row[key_name])
    return out


def main() -> int:
    args = parse_args()
    tri_dir = Path(args.tri_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tri = load_tri_existing(tri_dir / "data" / "NetworkParams_ExistingMaterials_v1.1.json")
    icsd = load_icsd_index(Path(args.icsd_index))
    community_map = load_assignment_map(Path(args.community_assignments), "community")
    hdbscan_map = load_assignment_map(Path(args.hdbscan_assignments), "cluster")

    formula_entries: dict[str, list[int]] = defaultdict(list)
    formula_years: dict[str, list[int]] = defaultdict(list)
    formula_communities: dict[str, list[int]] = defaultdict(list)
    formula_hdbscan: dict[str, list[int]] = defaultdict(list)

    for icsd_id, meta in icsd.items():
        formula = str(meta["formula"])
        formula_entries[formula].append(icsd_id)
        if meta["year"] is not None:
            formula_years[formula].append(int(meta["year"]))
        if icsd_id in community_map:
            formula_communities[formula].append(community_map[icsd_id])
        if icsd_id in hdbscan_map:
            formula_hdbscan[formula].append(hdbscan_map[icsd_id])

    shared_formulas = sorted(set(tri) & set(formula_entries))
    community_sizes = Counter(community_map.values())
    hdbscan_sizes = Counter(hdbscan_map.values())

    rows: list[dict[str, object]] = []
    for formula in shared_formulas:
        communities = formula_communities.get(formula, [])
        clusters = formula_hdbscan.get(formula, [])
        dominant_community = Counter(communities).most_common(1)[0][0] if communities else None
        dominant_cluster = Counter(clusters).most_common(1)[0][0] if clusters else None
        row = {
            "formula": formula,
            "tri_formula_raw": tri[formula]["tri_formula_raw"],
            "tri_discovery": tri[formula]["tri_discovery"],
            "tri_deg": tri[formula]["deg"],
            "tri_deg_cent": tri[formula]["deg_cent"],
            "tri_eigen_cent": tri[formula]["eigen_cent"],
            "tri_clus_coeff": tri[formula]["clus_coeff"],
            "icsd_n_entries": len(formula_entries[formula]),
            "icsd_first_year": min(formula_years[formula]) if formula_years[formula] else None,
            "icsd_last_year": max(formula_years[formula]) if formula_years[formula] else None,
            "icsd_n_communities": len(set(communities)),
            "icsd_n_hdbscan_clusters": len(set(clusters)),
            "dominant_community": dominant_community,
            "dominant_community_size": community_sizes.get(dominant_community, 0) if dominant_community is not None else 0,
            "dominant_hdbscan_cluster": dominant_cluster,
            "dominant_hdbscan_cluster_size": hdbscan_sizes.get(dominant_cluster, 0) if dominant_cluster is not None else 0,
        }
        rows.append(row)

    with (out_dir / "shared_formula_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    x_deg = [float(r["tri_deg"]) for r in rows]
    x_eig = [float(r["tri_eigen_cent"]) for r in rows]
    y_count = [float(r["icsd_n_entries"]) for r in rows]
    y_comm = [float(r["icsd_n_communities"]) for r in rows]
    y_hdb = [float(r["icsd_n_hdbscan_clusters"]) for r in rows]
    rows_with_year = [r for r in rows if r["icsd_first_year"] is not None and r["tri_discovery"]]
    deg_year = spearman([float(r["tri_deg"]) for r in rows_with_year], [float(r["icsd_first_year"]) for r in rows_with_year])
    tri_year_align = spearman([float(r["tri_discovery"]) for r in rows_with_year], [float(r["icsd_first_year"]) for r in rows_with_year])

    summary = {
        "n_tri_existing_formulas": len(tri),
        "n_icsd_formulas_total": len(formula_entries),
        "n_shared_formulas": len(rows),
        "shared_fraction_of_tri_existing": len(rows) / len(tri) if tri else None,
        "spearman_tri_deg_vs_icsd_entry_count": spearman(x_deg, y_count),
        "spearman_tri_deg_vs_icsd_n_communities": spearman(x_deg, y_comm),
        "spearman_tri_deg_vs_icsd_n_hdbscan_clusters": spearman(x_deg, y_hdb),
        "spearman_tri_eigen_cent_vs_icsd_entry_count": spearman(x_eig, y_count),
        "spearman_tri_discovery_vs_icsd_first_year": tri_year_align,
        "spearman_tri_deg_vs_icsd_first_year": deg_year,
        "top_shared_by_tri_degree": sorted(rows, key=lambda r: float(r["tri_deg"]), reverse=True)[:25],
        "top_shared_by_icsd_entry_count": sorted(rows, key=lambda r: int(r["icsd_n_entries"]), reverse=True)[:25],
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
