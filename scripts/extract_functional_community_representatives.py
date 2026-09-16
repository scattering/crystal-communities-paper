#!/usr/bin/env python3
"""Extract top-k central representatives per community for functional labeling.

Helper: standardizes + PCAs the frozen ICSD features, computes
per-community centroids, and for each community of size ≥
``--min-size`` outputs the ``--top-k`` closest entries to the centroid.
Joined to the ICSD index for human-readable formula / mineral name and
to the community-labels CSV for any existing tag, this is the source
of the representatives table that drives manual functional-class
labeling.

Inputs:
  --features, --community-assignments, --icsd-index,
  --community-labels, --top-k, --min-size.

Outputs:
  --output  CSV with one row per representative entry.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_assignments(path: Path) -> list[dict[str, int | None]]:
    rows = []
    for row in read_csv_rows(path):
        rows.append(
            {
                "icsd_id": parse_int(row.get("icsd_id", "")),
                "community": parse_int(row.get("community", "")),
                "year": parse_int(row.get("year", "")),
            }
        )
    return rows


def load_icsd_index(path: Path) -> dict[int, dict[str, str]]:
    rows = {}
    for row in read_csv_rows(path):
        icsd_id = parse_int(row.get("cif_names", ""))
        if icsd_id is None:
            continue
        rows[icsd_id] = row
    return rows


def load_labels(path: Path) -> dict[int, dict[str, str]]:
    labels = {}
    for row in read_csv_rows(path):
        community = parse_int(row.get("community", ""))
        if community is None:
            continue
        labels[community] = row
    return labels


def compute_centroids(Xp: np.ndarray, communities: list[int]) -> dict[int, np.ndarray]:
    grouped: dict[int, list[int]] = {}
    for idx, community in enumerate(communities):
        grouped.setdefault(community, []).append(idx)
    centroids = {}
    for community, indices in grouped.items():
        if community < 0:
            continue
        centroids[community] = Xp[indices].mean(axis=0)
    return centroids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract top-k central representatives for structural communities to support functional labeling."
    )
    parser.add_argument("--features", required=True, help="Path to frozen ICSD features.npy")
    parser.add_argument("--community-assignments", required=True, help="Path to community_assignments.csv")
    parser.add_argument("--icsd-index", required=True, help="Path to ICSD_index.csv")
    parser.add_argument("--community-labels", required=True, help="Path to top_graph_communities_labels3.csv")
    parser.add_argument("--top-k", type=int, default=20, help="Number of central representatives per community")
    parser.add_argument("--min-size", type=int, default=25, help="Minimum community size to keep")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    X = np.load(args.features)
    assignments = load_assignments(Path(args.community_assignments))
    if len(X) != len(assignments):
        raise ValueError(f"Feature rows ({len(X)}) do not match assignment rows ({len(assignments)})")

    communities = [a["community"] if a["community"] is not None else -1 for a in assignments]
    counts = Counter(c for c in communities if c is not None and c >= 0)
    labels = load_labels(Path(args.community_labels))
    icsd_index = load_icsd_index(Path(args.icsd_index))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(32, Xs.shape[0], Xs.shape[1]), random_state=42)
    Xp = pca.fit_transform(Xs)
    centroids = compute_centroids(Xp, communities)

    rows_out: list[dict[str, object]] = []
    grouped_indices: dict[int, list[int]] = {}
    for idx, community in enumerate(communities):
        if community is None or community < 0 or counts[community] < args.min_size:
            continue
        grouped_indices.setdefault(community, []).append(idx)

    for community, indices in sorted(grouped_indices.items(), key=lambda kv: (-counts[kv[0]], kv[0])):
        centroid = centroids[community]
        ranked = sorted(
            indices,
            key=lambda idx: float(np.linalg.norm(Xp[idx] - centroid)),
        )[: args.top_k]
        label_row = labels.get(community, {})
        for rank, idx in enumerate(ranked, start=1):
            assignment = assignments[idx]
            icsd_id = assignment["icsd_id"]
            meta = icsd_index.get(icsd_id or -1, {})
            distance = float(np.linalg.norm(Xp[idx] - centroid))
            rows_out.append(
                {
                    "community": community,
                    "community_size": counts[community],
                    "community_birth_year": label_row.get("birth_year", assignment["year"] or ""),
                    "community_label": label_row.get("label", ""),
                    "label_source": label_row.get("label_source", ""),
                    "rank_by_centroid_distance": rank,
                    "centroid_distance": f"{distance:.6f}",
                    "icsd_id": icsd_id or "",
                    "name": meta.get("name", ""),
                    "cif_names": meta.get("cif_names", ""),
                    "publication_year": meta.get("publication_year", ""),
                    "Bravais": meta.get("Bravais", ""),
                    "sym_group": meta.get("sym_group", ""),
                    "a": meta.get("a", ""),
                    "b": meta.get("b", ""),
                    "c": meta.get("c", ""),
                    "alpha": meta.get("alpha", ""),
                    "beta": meta.get("beta", ""),
                    "gamma": meta.get("gamma", ""),
                    "V": meta.get("V", ""),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "community",
                "community_size",
                "community_birth_year",
                "community_label",
                "label_source",
                "rank_by_centroid_distance",
                "centroid_distance",
                "icsd_id",
                "name",
                "cif_names",
                "publication_year",
                "Bravais",
                "sym_group",
                "a",
                "b",
                "c",
                "alpha",
                "beta",
                "gamma",
                "V",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} representative rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
