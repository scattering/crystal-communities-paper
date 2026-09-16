#!/usr/bin/env python3
"""Generate small synthetic inputs that mimic the production visualization
artifacts (features_pca.npy, community_assignments.csv, top_communities.json,
graph_time_summary.json, community_prototype_labels.json) so the rewritten
visualization scripts can be exercised locally without the full TACC run.

Usage:
    python scripts/synthesize_viz_inputs.py --out-root /tmp/viz_synth
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--n-communities", type=int, default=12)
    parser.add_argument("--n-per-community", type=int, default=200)
    parser.add_argument("--n-outliers", type=int, default=400)
    parser.add_argument("--n-pca", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out_root)
    run_dir = out_root / "run"
    community_dir = out_root / "community"
    time_dir = out_root / "time"
    for d in (run_dir, community_dir, time_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_comm = args.n_communities
    n_per = args.n_per_community
    n_pca = args.n_pca

    # Place community centers on a ring, with a few "interior" basins.
    centers = []
    for i in range(n_comm):
        angle = 2 * math.pi * i / n_comm
        radius = 6.0 if i % 4 != 0 else 3.5
        c = np.zeros(n_pca)
        c[0] = radius * math.cos(angle)
        c[1] = radius * math.sin(angle)
        # add a tiny structured offset on a third axis so 3D viewer has motion
        c[2] = (i % 3) * 0.6 - 0.6
        centers.append(c)
    centers = np.stack(centers)

    feats = []
    rows = []
    icsd_id = 100000
    # Communities are born on different decades 1930s..2000s
    birth_decades = sorted(np.linspace(1930, 2000, n_comm).astype(int).tolist())
    rng.shuffle(birth_decades)

    community_size_target = [max(60, int(n_per * (0.4 + 0.6 * rng.random()))) for _ in range(n_comm)]

    for ci in range(n_comm):
        n = community_size_target[ci]
        scale = 0.7 + 0.4 * rng.random()
        member_feats = rng.normal(loc=centers[ci], scale=scale, size=(n, n_pca))
        # year: weighted toward birth_decade with a long tail to present
        year_floor = birth_decades[ci]
        years = year_floor + rng.integers(0, 95 - (year_floor - 1930) + 1, size=n)
        years = np.clip(years, year_floor, 2024)
        # ensure at least one structure published in birth year
        years[0] = year_floor
        for yr, fv in zip(years, member_feats):
            feats.append(fv)
            rows.append({"icsd_id": icsd_id, "year": int(yr), "community": ci})
            icsd_id += 1

    # outliers (community = -1)
    out_feats = rng.normal(loc=0.0, scale=10.0, size=(args.n_outliers, n_pca))
    for fv in out_feats:
        feats.append(fv)
        rows.append({"icsd_id": icsd_id, "year": int(rng.integers(1900, 2025)), "community": -1})
        icsd_id += 1

    feats = np.stack(feats).astype(np.float32)
    np.save(run_dir / "features_pca.npy", feats)

    with (community_dir / "community_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["icsd_id", "year", "community"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Top communities, sorted by size desc
    sample_labels = [
        "Heavy-RE oxide family",
        "Ti15.512Sn9.92O41.272F11.368",  # intentionally long to test the shortener
        "JBW zeolite family",
        "MgZn(FeO2)4",
        "Lead Antimony Dioxide Bromide",
        "Beryllium Copper Silicon (4/1/1)",
        "Y0.844Cu1.5Se2",
        "Vanadium Gallide (1/1)",
        "Aluminium Tin Niobium (0.5/0.5/3)",
        "Copper Tin Selenide (2.67/1.33/4)",
        "Silver Lead Lithium (1/1/2)",
        "Holmium Scandium Silicide (1/1/1)",
    ]
    top_communities = sorted(
        [
            {
                "community": ci,
                "label": sample_labels[ci % len(sample_labels)],
                "birth_year": int(birth_decades[ci]),
                "size": int(community_size_target[ci]),
            }
            for ci in range(n_comm)
        ],
        key=lambda item: -item["size"],
    )
    (time_dir / "top_communities.json").write_text(json.dumps(top_communities, indent=2))

    # graph_time_summary.json with the numeric fields plot_icsd_graph_time expects.
    decades = [f"{d}s" for d in range(1900, 2030, 10)]
    by_decade: dict[str, dict[str, float]] = {}
    for d in decades:
        decade_year = int(d[:-1])
        ratio = min(1.0, max(0.0, (decade_year - 1900) / 130.0))
        existing = 0.5 + 0.4 * ratio
        same = 0.45 + 0.4 * ratio
        cross = 0.05 + 0.2 * ratio
        bridge = 0.02 + 0.1 * ratio
        birth = max(0.02, 0.45 - 0.4 * ratio)
        outlier = max(0.05, 0.18 - 0.05 * ratio)
        n_total = int(50 + 1500 * ratio)
        by_decade[d] = {
            "existing_cluster_ratio": existing,
            "same_community_attachment_ratio": same,
            "cross_community_attachment_ratio": cross,
            "bridge_attachment_ratio": bridge,
            "cluster_birth_point_ratio": birth,
            "outlier_ratio": outlier,
            "core_attachment_ratio": 0.55 + 0.05 * ratio,
            "periphery_attachment_ratio": 0.45 - 0.05 * ratio,
            "n_existing_cluster": int(n_total * existing),
            "n_cluster_birth_point": int(n_total * birth),
            "n_outlier": int(n_total * outlier),
        }
    (time_dir / "graph_time_summary.json").write_text(
        json.dumps({"by_decade": by_decade}, indent=2)
    )

    # node_events.csv mirrors what the production graph-time pipeline writes:
    # one row per (community, year, distance_to_centroid) attachment event with
    # the community's running core_threshold. The dashboard's
    # load_community_metadata reads these to compute the accessibility z-score.
    feats_arr = feats
    rows_arr = rows
    # Per-community member feature mean (synthetic centroid) for distance calc
    per_comm_idx: dict[int, list[int]] = {}
    for i, r in enumerate(rows_arr):
        c = r["community"]
        if c < 0:
            continue
        per_comm_idx.setdefault(c, []).append(i)
    centroids_synth = {
        c: feats_arr[idx].mean(axis=0)
        for c, idx in per_comm_idx.items()
    }
    node_events_path = community_dir / "node_events.csv"
    with node_events_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["icsd_id", "community", "year", "distance_to_centroid", "core_threshold"],
        )
        writer.writeheader()
        # Pre-compute the synthetic core_threshold per community as the 95th
        # percentile of in-community centroid distances, matching the production
        # convention used downstream by analyze_alab_validation.
        core_thresh_by_comm: dict[int, float] = {}
        all_dists: dict[int, list[float]] = {}
        for c, idx in per_comm_idx.items():
            dists = [float(np.linalg.norm(feats_arr[i] - centroids_synth[c])) for i in idx]
            all_dists[c] = dists
            core_thresh_by_comm[c] = float(np.quantile(dists, 0.95)) if dists else 1.0
        for c, idx in per_comm_idx.items():
            thresh = core_thresh_by_comm[c]
            for j, i in enumerate(idx):
                writer.writerow({
                    "icsd_id": rows_arr[i]["icsd_id"],
                    "community": c,
                    "year": rows_arr[i]["year"],
                    "distance_to_centroid": f"{all_dists[c][j]:.6f}",
                    "core_threshold": f"{thresh:.6f}",
                })

    # community_prototype_labels.json with a couple of "better" labels mirroring
    # what the production pipeline emits.
    proto_labels = [
        {"community": 0, "label": "Heavy-RE oxide family"},
        {"community": 2, "label": "JBW zeolite family"},
        {"community": 5, "label": "Beryllium silicide family"},
    ]
    (community_dir / "community_prototype_labels.json").write_text(json.dumps(proto_labels, indent=2))

    # Tiny canonical labels CSV in the same shape as notes/canonical_family_names_labels3.csv
    canonical_csv = community_dir / "canonical_family_names.csv"
    with canonical_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["kind", "id", "size", "birth_year", "centroid_icsd_id", "raw_label", "canonical_family_name", "evidence", "confidence", "notes"],
        )
        writer.writeheader()
        for ci in range(n_comm):
            writer.writerow({
                "kind": "graph_community",
                "id": ci,
                "size": community_size_target[ci],
                "birth_year": birth_decades[ci],
                "centroid_icsd_id": 0,
                "raw_label": sample_labels[ci % len(sample_labels)],
                "canonical_family_name": sample_labels[ci % len(sample_labels)] if ci < 6 else "",
                "evidence": "",
                "confidence": "high" if ci < 6 else "low",
                "notes": "",
            })

    print(json.dumps({
        "run_dir": str(run_dir),
        "community_dir": str(community_dir),
        "time_dir": str(time_dir),
        "canonical_labels_csv": str(canonical_csv),
        "node_events_csv": str(node_events_path),
        "n_structures": int(feats.shape[0]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
