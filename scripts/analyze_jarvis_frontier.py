#!/usr/bin/env python3
"""Project a JARVIS-DFT 3D sample into the frozen ICSD structural map.

Companion to analyze_gnome_frontier.py / analyze_mp_frontier.py /
analyze_alexandria_frontier.py. JARVIS-DFT (NIST) publishes ~76K DFT-relaxed
3D bulk crystals with a labeled ICSD-provenance field, so we can filter
out ICSD-derived entries directly. We additionally restrict to off-hull
predictions (ehull >= 0.05 eV/atom) for parity with the Alexandria filter.

The JARVIS structure dict uses its own schema (lattice_mat, elements,
coords, cartesian); we convert to pymatgen.Structure on the worker.

Output schema matches gnome_frontier_records.csv exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
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
class JarvisRecord:
    material_id: str        # e.g. JVASP-90856
    reduced_formula: str
    ehull: float | None
    formation_energy: float | None
    icsd_field: str
    reference: str
    atoms_dict: dict        # JARVIS-flavor: lattice_mat, elements, coords, cartesian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icsd-features", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--jarvis-json", required=True,
                        help="Path to jdft_3d-*.json (the unzipped figshare release).")
    parser.add_argument("--min-ehull", type=float, default=0.05,
                        help="Filter candidates with ehull >= this value (eV/atom). "
                        "Default 0.05 matches the Alexandria filter.")
    parser.add_argument("--max-ehull", type=float, default=0.5)
    parser.add_argument("--require-empty-icsd", action="store_true", default=True,
                        help="Drop entries with non-empty 'icsd' field (i.e., ICSD-derived).")
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


def jarvis_atoms_to_structure(atoms: dict):
    """Convert JARVIS atoms dict to pymatgen Structure. JARVIS uses
    lattice_mat (3x3), elements (list[str]), coords (list[3]), cartesian (bool)."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice(np.asarray(atoms["lattice_mat"]))
    coords = np.asarray(atoms["coords"])
    species = atoms["elements"]
    coords_are_cartesian = bool(atoms.get("cartesian", False))
    return Structure(lattice, species, coords, coords_are_cartesian=coords_are_cartesian)


def reduced_formula_from_atoms(atoms: dict) -> str:
    try:
        s = jarvis_atoms_to_structure(atoms)
        return s.composition.reduced_formula
    except Exception:
        return ""


def featurize_jarvis_record(args_tuple) -> tuple[bool, JarvisRecord, np.ndarray | None, str | None]:
    rec, wl_iters, local_mode = args_tuple
    try:
        structure = jarvis_atoms_to_structure(rec.atoms_dict)
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

    print(f"loading JARVIS-DFT JSON {args.jarvis_json} ...", flush=True)
    with open(args.jarvis_json) as f:
        raw = json.load(f)
    print(f"  {len(raw)} JARVIS entries loaded", flush=True)

    candidates: list[JarvisRecord] = []
    n_drop_icsd = 0
    n_drop_ehull = 0
    n_drop_size = 0
    n_drop_other = 0
    for e in raw:
        atoms = e.get("atoms")
        if not atoms or not atoms.get("elements"):
            n_drop_other += 1
            continue
        if args.require_empty_icsd and (e.get("icsd") or "").strip():
            n_drop_icsd += 1
            continue
        ehull = e.get("ehull")
        try:
            ehull_f = float(ehull) if ehull is not None else None
        except (TypeError, ValueError):
            ehull_f = None
        if ehull_f is None or not (args.min_ehull <= ehull_f <= args.max_ehull):
            n_drop_ehull += 1
            continue
        if len(atoms["elements"]) > args.max_sites:
            n_drop_size += 1
            continue
        formula = reduced_formula_from_atoms(atoms)
        if not formula:
            n_drop_other += 1
            continue
        candidates.append(JarvisRecord(
            material_id=e.get("jid", ""),
            reduced_formula=formula,
            ehull=ehull_f,
            formation_energy=e.get("formation_energy_peratom"),
            icsd_field=e.get("icsd", ""),
            reference=e.get("reference", ""),
            atoms_dict=atoms,
        ))
    print(f"  dropped: icsd={n_drop_icsd}, ehull-filter={n_drop_ehull}, "
          f"size={n_drop_size}, other={n_drop_other}", flush=True)
    print(f"  kept {len(candidates)} candidates", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = candidates[:args.sample_size]
    print(f"sampling {len(sample)} for featurization", flush=True)

    import concurrent.futures
    results = []
    work = [(rec, args.wl_iters, args.local_mode) for rec in sample]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        for item in ex.map(featurize_jarvis_record, work, chunksize=16):
            results.append(item)

    kept = [(rec, emb) for ok, rec, emb, _ in results if ok and emb is not None]
    failures = [{"material_id": rec.material_id, "formula": rec.reduced_formula, "detail": err}
                for ok, rec, _, err in results if not ok]
    print(f"featurized {len(kept)}; failures {len(failures)}", flush=True)
    if not kept:
        raise RuntimeError("No JARVIS structures featurized successfully")

    recs = [rec for rec, _ in kept]
    Y = np.vstack([emb for _, emb in kept])
    nan_mask = np.isnan(Y).any(axis=1)
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        print(f"dropping {n_dropped} structures with NaN features", flush=True)
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
        "source": "jarvis_dft_3d_2022_12_12",
        "candidate_pool_size": len(candidates),
        "sample_size_requested": args.sample_size,
        "n_featurized": len(recs),
        "n_failures": len(failures),
        "icsd_centroid_distance_threshold_p95": threshold,
        "n_outlier_like": int(np.count_nonzero(outlier_like)),
        "outlier_like_ratio": float(np.mean(outlier_like)),
        "top_existing_communities": top_communities,
        "mean_nearest_centroid_distance": float(np.mean(nearest_dist)),
        "min_ehull_filter": args.min_ehull,
        "max_ehull_filter": args.max_ehull,
        "require_empty_icsd": args.require_empty_icsd,
    }
    (out_dir / "jarvis_frontier_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "jarvis_frontier_failures.json").write_text(json.dumps(failures[:500], indent=2))

    with (out_dir / "jarvis_frontier_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "material_id", "reduced_formula", "ehull", "formation_energy",
                "assigned_community", "nearest_centroid_distance", "outlier_like",
                "pca1", "pca2",
            ],
        )
        writer.writeheader()
        for rec, comm, dist, is_out, xy in zip(recs, nearest_comm, nearest_dist, outlier_like, Y2):
            writer.writerow({
                "material_id": rec.material_id,
                "reduced_formula": rec.reduced_formula,
                "ehull": rec.ehull,
                "formation_energy": rec.formation_energy,
                "assigned_community": int(comm),
                "nearest_centroid_distance": float(dist),
                "outlier_like": bool(is_out),
                "pca1": float(xy[0]),
                "pca2": float(xy[1]),
            })

    print(f"wrote {len(recs)} records to {out_dir}/jarvis_frontier_records.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
