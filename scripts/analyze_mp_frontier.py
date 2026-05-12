#!/usr/bin/env python3
"""Project a Materials Project sample into the frozen ICSD structural map.

Companion to analyze_gnome_frontier.py / analyze_mattergen_frontier.py. Tests
whether MP-predicted (theoretical-only) structures look frontier-like the
same way generative AI samples do, controlling for the fact that MP heavily
overlaps with ICSD by filtering to entries with theoretical=True AND no
ICSD database_IDs.

Output schema matches gnome_frontier_records.csv exactly so downstream
analyses (analyze_gnome_temporal_sweep.py, analyze_composition_matched_ai.py)
can read it without modification.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from icsd_densify_worker import build_structure_embedding


@dataclass
class MPRecord:
    material_id: str
    reduced_formula: str
    energy_above_hull: float | None
    spg: str | None
    structure_dict: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icsd-features", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--candidate-cache", required=True,
                        help="JSONL cache of theoretical-only MP entries (one entry per line). "
                        "Created on first run; reused on subsequent runs to avoid re-querying MP.")
    parser.add_argument("--candidate-pool-size", type=int, default=20000,
                        help="Size of theoretical-only candidate pool to fetch from MP. "
                        "Sample will be drawn from this pool.")
    parser.add_argument("--max-e-above-hull", type=float, default=0.2,
                        help="Filter candidates with energy_above_hull <= this value (eV/atom). "
                        "Default 0.2 keeps only metastable-or-better predictions.")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sites", type=int, default=256)
    parser.add_argument("--wl-iters", type=int, default=3)
    parser.add_argument("--local-mode", choices=["matminer_ops", "fast_local"], default="matminer_ops")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


from frontier_common import (
    parse_int,
    load_community_rows,
    centroid_thresholds,
)


def fetch_mp_candidates(cache_path: Path, pool_size: int, max_e_above_hull: float) -> list[MPRecord]:
    """Query MP for theoretical-only entries with no ICSD provenance, cache to JSONL."""
    if cache_path.exists():
        records = []
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                d = json.loads(line)
                records.append(MPRecord(
                    material_id=d["material_id"],
                    reduced_formula=d["reduced_formula"],
                    energy_above_hull=d.get("energy_above_hull"),
                    spg=d.get("spg"),
                    structure_dict=d["structure"],
                ))
        print(f"loaded {len(records)} candidates from cache {cache_path}", flush=True)
        return records

    import yaml
    with open(os.path.expanduser("~/.pmgrc.yaml")) as f:
        cfg = yaml.safe_load(f)
    api_key = cfg["PMG_MAPI_KEY"]

    from mp_api.client import MPRester

    fields = ["material_id", "formula_pretty", "theoretical", "database_IDs",
              "energy_above_hull", "symmetry", "structure", "nsites"]
    print(f"querying MP for theoretical=True entries (target pool {pool_size})...", flush=True)
    with MPRester(api_key) as mpr:
        # MP's query is per-document; we ask for a generous pool, filter, then sample.
        docs = mpr.materials.summary.search(
            theoretical=True,
            energy_above_hull=(0.0, max_e_above_hull),
            fields=fields,
            chunk_size=1000,
            num_chunks=max(1, pool_size // 1000 + 1),
        )

    print(f"fetched {len(docs)} docs from MP", flush=True)
    records = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for d in docs:
            db_ids = d.database_IDs or {}
            # Belt-and-suspenders: theoretical=True should already exclude ICSD,
            # but check directly for any 'icsd' key to be sure.
            if db_ids.get("icsd"):
                continue
            spg = d.symmetry.symbol if d.symmetry else None
            row = {
                "material_id": d.material_id,
                "reduced_formula": d.formula_pretty,
                "energy_above_hull": d.energy_above_hull,
                "spg": spg,
                "structure": d.structure.as_dict() if d.structure else None,
            }
            if row["structure"] is None:
                continue
            handle.write(json.dumps(row) + "\n")
            records.append(MPRecord(
                material_id=d.material_id,
                reduced_formula=d.formula_pretty,
                energy_above_hull=d.energy_above_hull,
                spg=spg,
                structure_dict=row["structure"],
            ))
    print(f"cached {len(records)} candidates to {cache_path}", flush=True)
    return records


def featurize_mp_record(args_tuple) -> tuple[bool, MPRecord, np.ndarray | None, str | None]:
    rec, wl_iters, local_mode = args_tuple
    try:
        from pymatgen.core import Structure
        structure = Structure.from_dict(rec.structure_dict)
        emb = build_structure_embedding(structure, wl_iters=wl_iters, local_mode=local_mode)
        return True, rec, emb, None
    except Exception as exc:
        return False, rec, None, str(exc)[:200]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.icsd_features)
    community_rows = load_community_rows(Path(args.community_assignments))
    if len(X) != len(community_rows):
        raise ValueError(f"ICSD features rows ({len(X)}) != community rows ({len(community_rows)})")
    community_labels = [int(r["community"]) if r["community"] is not None else -1 for r in community_rows]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca32 = PCA(n_components=32, random_state=args.seed)
    Xp = pca32.fit_transform(Xs)
    pca2 = PCA(n_components=2, random_state=args.seed)
    pca2.fit(Xs)

    communities, centroids, threshold = centroid_thresholds(Xp, community_labels)
    print(f"trained {len(communities)} community thresholds; p95 = {threshold:.4f}", flush=True)

    candidates = fetch_mp_candidates(
        Path(args.candidate_cache),
        pool_size=args.candidate_pool_size,
        max_e_above_hull=args.max_e_above_hull,
    )
    candidates = [r for r in candidates if (r.structure_dict.get("sites") and len(r.structure_dict["sites"]) <= args.max_sites)]
    print(f"after max_sites filter: {len(candidates)} candidates", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = candidates[:args.sample_size]
    print(f"sampling {len(sample)} candidates for featurization", flush=True)

    import concurrent.futures
    results = []
    work = [(rec, args.wl_iters, args.local_mode) for rec in sample]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        for item in ex.map(featurize_mp_record, work, chunksize=16):
            results.append(item)

    kept = [(rec, emb) for ok, rec, emb, _ in results if ok and emb is not None]
    failures = [{"material_id": rec.material_id, "formula": rec.reduced_formula, "detail": err}
                for ok, rec, _, err in results if not ok]
    print(f"featurized {len(kept)}; failures {len(failures)}", flush=True)
    if not kept:
        raise RuntimeError("No MP structures featurized successfully")

    recs = [rec for rec, _ in kept]
    Y = np.vstack([emb for _, emb in kept])
    # Drop rows with NaN features. A handful of MP structures contain noble
    # gases (He) with no Pauling electronegativity; pymatgen returns NaN for
    # those, and `el.X or 0.0` in the worker doesn't catch it because NaN is
    # truthy. The lost structures are only the noble-gas-containing ones.
    nan_mask = np.isnan(Y).any(axis=1)
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        print(f"dropping {n_dropped} structures with NaN features (likely noble-gas-containing)", flush=True)
        Y = Y[~nan_mask]
        recs = [r for r, m in zip(recs, nan_mask) if not m]
    Ys = scaler.transform(Y)
    Yp = pca32.transform(Ys)
    Y2 = pca2.transform(Ys)

    dmat = np.linalg.norm(Yp[:, None, :] - centroids[None, :, :], axis=2)
    nearest_idx = np.argmin(dmat, axis=1)
    nearest_comm = communities[nearest_idx]
    nearest_dist = dmat[np.arange(len(Yp)), nearest_idx]
    outlier_like = nearest_dist > threshold

    top_communities = Counter(int(c) for c in nearest_comm[~outlier_like]).most_common(15)
    summary = {
        "source": "materials_project_theoretical",
        "candidate_pool_size": len(candidates),
        "sample_size_requested": args.sample_size,
        "n_featurized": len(recs),
        "n_failures": len(failures),
        "icsd_centroid_distance_threshold_p95": threshold,
        "n_outlier_like": int(np.count_nonzero(outlier_like)),
        "outlier_like_ratio": float(np.mean(outlier_like)),
        "top_existing_communities": top_communities,
        "mean_nearest_centroid_distance": float(np.mean(nearest_dist)),
        "max_e_above_hull_filter": args.max_e_above_hull,
    }
    (out_dir / "mp_frontier_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "mp_frontier_failures.json").write_text(json.dumps(failures[:500], indent=2))

    with (out_dir / "mp_frontier_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "material_id", "reduced_formula", "energy_above_hull",
                "assigned_community", "nearest_centroid_distance", "outlier_like",
                "pca1", "pca2",
            ],
        )
        writer.writeheader()
        for rec, comm, dist, is_out, xy in zip(recs, nearest_comm, nearest_dist, outlier_like, Y2):
            writer.writerow({
                "material_id": rec.material_id,
                "reduced_formula": rec.reduced_formula,
                "energy_above_hull": rec.energy_above_hull,
                "assigned_community": int(comm),
                "nearest_centroid_distance": float(dist),
                "outlier_like": bool(is_out),
                "pca1": float(xy[0]),
                "pca2": float(xy[1]),
            })

    print(f"wrote {len(recs)} records to {out_dir}/mp_frontier_records.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
