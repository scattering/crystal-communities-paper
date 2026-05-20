#!/usr/bin/env python3
"""Post-process a finished densification run into named structural communities.

Producer for ``community_assignments.csv`` and the prototype-labeled
community catalogues consumed by every downstream analysis. Reads a
finished densification run directory (``--run-dir``), builds a weighted
k-NN graph (Gaussian-decayed Euclidean distances on the PCA features,
``--k`` neighbors, optionally ``--mutual-knn``), runs Louvain at
``--resolution``, drops components / communities below
``--min-component-size`` / ``--min-community-size``, and labels the
``--prototype-top-n`` largest communities by attempting an
AflowPrototypeMatcher fit and falling back to a StructureMatcher
representative-pair check.

Pipeline:
  1. Build weighted k-NN graph on ``features_pca.npy`` (Gaussian kernel
     with sigma = median k-NN distance).
  2. Drop nodes in connected components below ``--min-component-size``.
  3. Louvain partition at ``--resolution``; drop communities below
     ``--min-community-size``.
  4. Densification summary by decade (outlier / existing-cluster /
     cluster-birth ratios).
  5. Prototype labels for the largest N communities via AFLOW (initial
     ltol/stol/angle-tol thresholds in the CLI) with StructureMatcher
     fallback. Same pass is applied to the HDBSCAN clusters from
     ``sample_assignments.csv`` for the SI comparison.

Inputs (under ``--run-dir``):
  features_pca.npy, sample_assignments.csv (HDBSCAN labels), and
  optionally a local ICSD CIF source (requires a valid ICSD license)
  for prototype labeling.

Outputs (under ``--output-dir``):
  community_assignments.csv             icsd_id, year, community
  community_summary.json                per-decade densification stats
  community_prototype_labels.json       AFLOW/SM matches for the top N
  hdbscan_prototype_labels.json         same for HDBSCAN clusters
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import copy
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from pymatgen.analysis.prototypes import AflowPrototypeMatcher
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph community postprocessing for ICSD densification runs.")
    parser.add_argument("--run-dir", required=True, help="Finished densification run directory")
    parser.add_argument("--output-dir", required=True, help="Directory for graph community outputs")
    parser.add_argument("--features-file", default="features_pca.npy", help="Feature array filename under run-dir")
    parser.add_argument("--assignments-file", default="sample_assignments.csv", help="Assignments CSV filename under run-dir")
    parser.add_argument("--k", type=int, default=16, help="k for k-NN graph construction")
    parser.add_argument("--mutual-knn", action="store_true", help="Keep only mutual k-NN edges")
    parser.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution")
    parser.add_argument("--min-component-size", type=int, default=8, help="Connected components smaller than this become outliers")
    parser.add_argument("--min-community-size", type=int, default=4, help="Communities smaller than this become outliers")
    parser.add_argument("--prototype-top-n", type=int, default=25, help="Prototype-label the N largest communities")
    parser.add_argument(
        "--label-hdbscan-top-n",
        type=int,
        default=25,
        help="Prototype-label the N largest HDBSCAN clusters from sample_assignments.csv",
    )
    parser.add_argument("--aflow-initial-ltol", type=float, default=0.25, help="Initial AflowPrototypeMatcher ltol")
    parser.add_argument("--aflow-initial-stol", type=float, default=0.35, help="Initial AflowPrototypeMatcher stol")
    parser.add_argument(
        "--aflow-initial-angle-tol",
        type=float,
        default=8.0,
        help="Initial AflowPrototypeMatcher angle tolerance",
    )
    parser.add_argument("--structure-matcher-ltol", type=float, default=0.25, help="Fallback StructureMatcher ltol")
    parser.add_argument("--structure-matcher-stol", type=float, default=0.35, help="Fallback StructureMatcher stol")
    parser.add_argument(
        "--structure-matcher-angle-tol",
        type=float,
        default=8.0,
        help="Fallback StructureMatcher angle tolerance",
    )
    parser.add_argument("--icsd-zip", help="Optional local CIF archive path for prototype labeling")
    parser.add_argument("--zip-password", help="Optional password for protected CIF archive members")
    parser.add_argument("--cif-root", help="Optional extracted CIF directory for prototype labeling")
    return parser.parse_args()


def load_assignments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def decade_from_year(year: int | None) -> str:
    if year is None:
        return "unknown"
    return f"{(year // 10) * 10}s"


def summarize_densification(records: list[dict[str, int | None]], labels: np.ndarray) -> dict:
    community_birth_year: dict[int, int] = {}
    for rec, label in zip(records, labels):
        if label < 0 or rec["year"] is None:
            continue
        if label not in community_birth_year:
            community_birth_year[label] = rec["year"]
        else:
            community_birth_year[label] = min(community_birth_year[label], rec["year"])

    by_decade: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    community_sizes = Counter(int(v) for v in labels if int(v) >= 0)

    for rec, label in zip(records, labels):
        decade = decade_from_year(rec["year"])
        by_decade[decade]["n_total"] += 1
        if int(label) < 0:
            by_decade[decade]["n_outlier"] += 1
            continue
        birth = community_birth_year.get(int(label))
        if birth is not None and rec["year"] is not None and rec["year"] > birth:
            by_decade[decade]["n_existing_cluster"] += 1
        elif birth is not None and rec["year"] == birth:
            by_decade[decade]["n_cluster_birth_point"] += 1

    for decade, stats in by_decade.items():
        total = stats["n_total"] or 1.0
        stats["outlier_ratio"] = stats["n_outlier"] / total
        stats["existing_cluster_ratio"] = stats["n_existing_cluster"] / total
        stats["cluster_birth_point_ratio"] = stats["n_cluster_birth_point"] / total

    return {
        "n_communities": int(len(community_sizes)),
        "largest_communities": community_sizes.most_common(20),
        "community_birth_year": {str(k): int(v) for k, v in sorted(community_birth_year.items())},
        "by_decade": {k: dict(v) for k, v in sorted(by_decade.items())},
    }


def build_weighted_graph(
    X: np.ndarray,
    k: int,
    mutual_knn: bool,
    min_component_size: int,
) -> tuple[nx.Graph, np.ndarray]:
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
            if i == j:
                continue
            if mutual_knn and i not in neighbor_sets[j]:
                continue
            weight = math.exp(-((float(dist) / sigma) ** 2))
            if graph.has_edge(i, j):
                if weight > graph[i][j]["weight"]:
                    graph[i][j]["weight"] = weight
            else:
                graph.add_edge(i, j, weight=weight)

    labels = np.full(len(X), -1, dtype=int)
    keep_nodes: list[int] = []
    for component in nx.connected_components(graph):
        if len(component) < min_component_size:
            continue
        keep_nodes.extend(component)
    return graph.subgraph(keep_nodes).copy(), labels


def run_louvain(
    graph: nx.Graph,
    n_points: int,
    resolution: float,
    min_community_size: int,
    base_labels: np.ndarray,
) -> np.ndarray:
    labels = base_labels.copy()
    if graph.number_of_nodes() == 0:
        return labels

    communities = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=42)
    community_id = 0
    for community in communities:
        if len(community) < min_community_size:
            continue
        for node in community:
            labels[node] = community_id
        community_id += 1
    return labels


def read_structure(icsd_id: int, icsd_zip: str | None, zip_password: str | None, cif_root: str | None) -> Structure:
    text = read_cif_text(icsd_id, icsd_zip, zip_password, cif_root)
    return Structure.from_str(text, fmt="cif")


def read_cif_text(icsd_id: int, icsd_zip: str | None, zip_password: str | None, cif_root: str | None) -> str:
    member = f"FindIt_CIFs/icsd_{icsd_id:06d}.cif"
    if icsd_zip:
        with zipfile.ZipFile(icsd_zip) as zf:
            pwd = zip_password.encode("utf-8") if zip_password else None
            with zf.open(member, pwd=pwd) as handle:
                return handle.read().decode("utf-8", errors="replace")
    if cif_root:
        return Path(cif_root, f"icsd_{icsd_id:06d}.cif").read_text(encoding="utf-8", errors="replace")
    raise ValueError("Prototype labeling requires --icsd-zip or --cif-root")


def extract_cif_metadata_labels(cif_text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    tag_map = {
        "_chemical_name_mineral": "mineral_name",
        "_chemical_name_common": "common_name",
        "_chemical_name_systematic": "systematic_name",
        "_chemical_formula_structural": "structural_formula",
        "_chemical_formula_sum": "formula_sum",
    }
    for cif_tag, key in tag_map.items():
        pattern = rf"(?im)^{re.escape(cif_tag)}\s+(.+?)\s*$"
        match = re.search(pattern, cif_text)
        if not match:
            continue
        value = match.group(1).strip().strip("'").strip('"')
        if value and value not in {"?", "."}:
            labels[key] = value
    return labels


def _normalize_label_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        values = [str(item).strip() for item in value if str(item).strip()]
        text = ", ".join(values)
    else:
        text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "unknown", "n/a", "na", "?"}:
        return None
    if re.fullmatch(r"[\s;:.,_\-+/\\]+", text):
        return None
    return text


def _label_priority(label_source: str | None) -> int:
    order = {"mineral": 0, "strukturbericht": 1, "aflow": 2}
    return order.get(label_source or "", 3)


def _preferred_tag(tags: dict[str, object]) -> tuple[str | None, str | None]:
    for key in ("mineral", "strukturbericht", "aflow"):
        label = _normalize_label_text(tags.get(key))
        if label:
            return key, label
    for key, value in tags.items():
        label = _normalize_label_text(value)
        if label:
            return str(key), label
    return None, None


def _extract_centroid_member(X: np.ndarray, members: np.ndarray) -> int:
    centroid = X[members].mean(axis=0)
    member_vectors = X[members]
    return int(members[np.argmin(np.linalg.norm(member_vectors - centroid, axis=1))])


def _build_aflow_matcher(initial_ltol: float, initial_stol: float, initial_angle_tol: float) -> AflowPrototypeMatcher:
    attempts = (
        {
            "initial_ltol": initial_ltol,
            "initial_stol": initial_stol,
            "initial_angle_tol": initial_angle_tol,
        },
        {
            "ltol": initial_ltol,
            "stol": initial_stol,
            "angle_tol": initial_angle_tol,
        },
    )
    for kwargs in attempts:
        try:
            return AflowPrototypeMatcher(**kwargs)
        except TypeError:
            continue
    return AflowPrototypeMatcher()


def _build_structure_matcher(ltol: float, stol: float, angle_tol: float) -> StructureMatcher:
    try:
        return StructureMatcher(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            primitive_cell=True,
            scale=True,
            attempt_supercell=False,
            allow_subset=False,
        )
    except TypeError:
        return StructureMatcher()


def sanitize_prototype_matches(prototypes: list[dict] | None) -> list[dict] | None:
    if not prototypes:
        return None
    cleaned: list[dict] = []
    for match in prototypes:
        tags = dict(match.get("tags", {}))
        preferred_key, preferred_label = _preferred_tag(tags)
        if preferred_key is None or preferred_label is None:
            continue
        snl = match.get("snl")
        entry = {
            "tags": {preferred_key: preferred_label},
            "label": preferred_label,
            "label_source": preferred_key,
        }
        if snl is not None:
            try:
                entry["formula"] = snl.structure.composition.reduced_formula
                entry["nsites"] = int(len(snl.structure))
            except Exception:
                pass
        cleaned.append(entry)
    cleaned.sort(key=lambda entry: (_label_priority(entry.get("label_source")), entry.get("nsites", 10**9), entry.get("label", "")))
    return cleaned


def _apply_structure_matcher_fallback(entries: list[dict], matcher: StructureMatcher | None) -> None:
    if matcher is None:
        return

    resolved = [entry for entry in entries if entry.get("prototype_matches")]
    candidates = [
        entry
        for entry in entries
        if entry.get("_structure") is not None and entry.get("label_source") not in {"mineral", "strukturbericht"}
    ]
    if not resolved or not candidates:
        return

    donors = sorted(
        resolved,
        key=lambda entry: (_label_priority(entry.get("label_source")), -int(entry.get("size", 0)), str(entry.get("label", ""))),
    )

    for entry in candidates:
        struct = entry.get("_structure")
        if struct is None:
            continue
        current_priority = _label_priority(entry.get("label_source"))
        for donor in donors:
            donor_struct = donor.get("_structure")
            if donor_struct is None:
                continue
            donor_priority = _label_priority(donor.get("label_source"))
            if donor_priority >= current_priority:
                continue
            try:
                if matcher.fit(struct, donor_struct):
                    entry["prototype_matches"] = copy.deepcopy(donor["prototype_matches"])
                    entry["label"] = donor.get("label")
                    entry["label_source"] = "structure_matcher"
                    entry["prototype_fallback_source"] = f"matched_{donor.get('label_source', 'unknown')}"
                    break
            except Exception:
                continue


def _apply_metadata_fallback(entries: list[dict]) -> None:
    for entry in entries:
        if entry.get("label_source") in {"mineral_name", "common_name", "systematic_name"}:
            continue
        metadata = entry.get("centroid_metadata") or {}
        for key in ("mineral_name", "common_name", "systematic_name"):
            label = _normalize_label_text(metadata.get(key))
            if label:
                entry["label"] = label
                entry["label_source"] = key
                break
        else:
            for key in ("formula", "formula_sum", "structural_formula"):
                label = _normalize_label_text(metadata.get(key))
                if label:
                    entry["label"] = label
                    entry["label_source"] = key
                    break


def _finalize_label_entries(entries: list[dict]) -> list[dict]:
    finalized: list[dict] = []
    for entry in entries:
        row = {k: v for k, v in entry.items() if not k.startswith("_")}
        row.setdefault("prototype_matches", [])
        row.setdefault("label", f"ICSD {row['centroid_icsd_id']}")
        finalized.append(row)
    return finalized


def _prototype_label_records(
    X: np.ndarray,
    records: list[dict[str, int | None]],
    labels: np.ndarray,
    label_field: str,
    icsd_zip: str | None,
    zip_password: str | None,
    cif_root: str | None,
    top_n: int,
    aflow_initial_ltol: float,
    aflow_initial_stol: float,
    aflow_initial_angle_tol: float,
    structure_matcher_ltol: float,
    structure_matcher_stol: float,
    structure_matcher_angle_tol: float,
) -> list[dict]:
    if not icsd_zip and not cif_root:
        return []

    matcher = _build_aflow_matcher(aflow_initial_ltol, aflow_initial_stol, aflow_initial_angle_tol)
    fallback_matcher = _build_structure_matcher(structure_matcher_ltol, structure_matcher_stol, structure_matcher_angle_tol)
    counts = Counter(int(v) for v in labels if int(v) >= 0)
    entries: list[dict] = []

    for cluster_id, size in counts.most_common(top_n):
        members = np.flatnonzero(labels == cluster_id)
        centroid_idx = _extract_centroid_member(X, members)
        rec = records[centroid_idx]
        entry = {
            label_field: int(cluster_id),
            "size": int(size),
            "centroid_icsd_id": int(rec["icsd_id"]),
            "centroid_year": rec["year"],
        }
        try:
            cif_text = read_cif_text(int(rec["icsd_id"]), icsd_zip, zip_password, cif_root)
            entry["centroid_metadata"] = extract_cif_metadata_labels(cif_text)
            struct = read_structure(int(rec["icsd_id"]), icsd_zip, zip_password, cif_root)
            entry["_structure"] = struct
            entry["centroid_metadata"]["formula"] = struct.composition.reduced_formula
            prototypes = matcher.get_prototypes(struct)
            entry["prototype_matches"] = sanitize_prototype_matches(prototypes) or []
        except Exception as exc:
            entry["prototype_matches"] = []
            entry["prototype_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        if entry["prototype_matches"]:
            entry["label"] = entry["prototype_matches"][0]["label"]
            entry["label_source"] = entry["prototype_matches"][0]["label_source"]
        else:
            entry["label"] = f"ICSD {rec['icsd_id']}"
            entry["label_source"] = "centroid_icsd_id"
        entries.append(entry)

    # Only compare the small top-N shortlist, and only for entries that AFLOW did not label well.
    _apply_structure_matcher_fallback(entries, fallback_matcher)
    _apply_metadata_fallback(entries)
    return _finalize_label_entries(entries)


def prototype_label_communities(
    X: np.ndarray,
    records: list[dict[str, int | None]],
    labels: np.ndarray,
    icsd_zip: str | None,
    zip_password: str | None,
    cif_root: str | None,
    top_n: int,
    aflow_initial_ltol: float,
    aflow_initial_stol: float,
    aflow_initial_angle_tol: float,
    structure_matcher_ltol: float,
    structure_matcher_stol: float,
    structure_matcher_angle_tol: float,
) -> list[dict]:
    return _prototype_label_records(
        X,
        records,
        labels,
        "community",
        icsd_zip,
        zip_password,
        cif_root,
        top_n,
        aflow_initial_ltol,
        aflow_initial_stol,
        aflow_initial_angle_tol,
        structure_matcher_ltol,
        structure_matcher_stol,
        structure_matcher_angle_tol,
    )


def prototype_label_existing_labels(
    X: np.ndarray,
    records: list[dict[str, int | None]],
    labels: np.ndarray,
    label_name: str,
    icsd_zip: str | None,
    zip_password: str | None,
    cif_root: str | None,
    top_n: int,
    aflow_initial_ltol: float,
    aflow_initial_stol: float,
    aflow_initial_angle_tol: float,
    structure_matcher_ltol: float,
    structure_matcher_stol: float,
    structure_matcher_angle_tol: float,
) -> list[dict]:
    return _prototype_label_records(
        X,
        records,
        labels,
        label_name,
        icsd_zip,
        zip_password,
        cif_root,
        top_n,
        aflow_initial_ltol,
        aflow_initial_stol,
        aflow_initial_angle_tol,
        structure_matcher_ltol,
        structure_matcher_stol,
        structure_matcher_angle_tol,
    )


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(run_dir / args.features_file)
    rows = load_assignments(run_dir / args.assignments_file)
    records = []
    hdb_labels = []
    for row in rows:
        raw_year = row.get("year", "").strip()
        year = int(raw_year) if raw_year else None
        records.append({"icsd_id": int(row["icsd_id"]), "year": year})
        hdb_labels.append(int(row["cluster"]))
    hdb_labels_arr = np.asarray(hdb_labels, dtype=int)

    if len(records) != len(X):
        raise ValueError(f"Assignments/features length mismatch: {len(records)} vs {len(X)}")

    graph, base_labels = build_weighted_graph(X, args.k, args.mutual_knn, args.min_component_size)
    labels = run_louvain(graph, len(X), args.resolution, args.min_community_size, base_labels)

    summary = {
        "n_points": int(len(X)),
        "graph": {
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
            "k": int(args.k),
            "mutual_knn": bool(args.mutual_knn),
            "resolution": float(args.resolution),
            "min_component_size": int(args.min_component_size),
            "min_community_size": int(args.min_community_size),
        },
        "n_outliers": int(np.sum(labels < 0)),
        "outlier_ratio": float(np.mean(labels < 0)),
        "densification": summarize_densification(records, labels),
    }

    community_prototype_labels = prototype_label_communities(
        X,
        records,
        labels,
        args.icsd_zip,
        args.zip_password,
        args.cif_root,
        args.prototype_top_n,
        args.aflow_initial_ltol,
        args.aflow_initial_stol,
        args.aflow_initial_angle_tol,
        args.structure_matcher_ltol,
        args.structure_matcher_stol,
        args.structure_matcher_angle_tol,
    )
    hdbscan_prototype_labels = prototype_label_existing_labels(
        X,
        records,
        hdb_labels_arr,
        "hdbscan_cluster",
        args.icsd_zip,
        args.zip_password,
        args.cif_root,
        args.label_hdbscan_top_n,
        args.aflow_initial_ltol,
        args.aflow_initial_stol,
        args.aflow_initial_angle_tol,
        args.structure_matcher_ltol,
        args.structure_matcher_stol,
        args.structure_matcher_angle_tol,
    )

    with (out_dir / "community_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (out_dir / "community_prototype_labels.json").open("w") as handle:
        json.dump(community_prototype_labels, handle, indent=2)
    with (out_dir / "hdbscan_prototype_labels.json").open("w") as handle:
        json.dump(hdbscan_prototype_labels, handle, indent=2)
    with (out_dir / "community_assignments.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["icsd_id", "year", "community"])
        for rec, label in zip(records, labels):
            writer.writerow([rec["icsd_id"], rec["year"] if rec["year"] is not None else "", int(label)])

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
