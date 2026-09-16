#!/usr/bin/env python3
"""Re-embed exact historical external cohorts and project onto the repaired map.

All source inputs must already exist locally. No API calls or downloads occur.
Raw feature caches are isolated by encoder/settings/code/source hashes. Output
outlier_like and in_basin use per-community p95; outlier_like_pooled preserves
the explicitly named historical pooled statistic. Every attempted ID is saved.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from scipy.spatial.distance import cdist

from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS, geometry_crystalnn
from external_frozen_cohorts import load_cohort, public_fields, record_key
from icsd_densify_worker import CrystalNNFingerprint, build_structure_embedding
from prepare_repaired_projection_basis import sha256_file

WORKER_ZIP = None
WORKER_PASSWORD = None
WORKER_WL = 3
WORKER_CNN = None
WORKER_FP = None


def valid_embedding(vector):
    if vector.shape != (213,) or not np.isfinite(vector).all():
        return False
    pooled = vector[:-9].reshape(3, 68)
    return bool(np.any(pooled[:, :7]) and np.any(pooled[:, 7:]))


def initialize_worker(source, source_path, wl_iters, password):
    global WORKER_ZIP, WORKER_PASSWORD, WORKER_WL, WORKER_CNN, WORKER_FP
    WORKER_PASSWORD = password
    WORKER_WL = wl_iters
    WORKER_CNN = geometry_crystalnn()
    WORKER_FP = CrystalNNFingerprint.from_preset("ops")
    if source in ("gnome", "mattergen"):
        path = Path(source_path) / "by_id.zip" if source == "gnome" else Path(source_path)
        WORKER_ZIP = zipfile.ZipFile(path)


def featurize(record):
    public = public_fields(record)
    try:
        kind = record["_kind"]
        if kind == "pymatgen":
            structure = Structure.from_dict(record["_structure"])
        elif kind == "jarvis":
            from analyze_jarvis_frontier import jarvis_atoms_to_structure
            structure = jarvis_atoms_to_structure(record["_structure"])
        else:
            mid = record["material_id"]
            names = [record["zip_member"]] if kind == "mattergen" else [
                f"{mid}.cif", f"{mid}.CIF", f"by_id/{mid}.cif", f"by_id/{mid}.CIF", f"{mid}.vasp.cif"]
            for name in names:
                try:
                    data = WORKER_ZIP.read(name, pwd=WORKER_PASSWORD.encode() if WORKER_PASSWORD else None)
                    break
                except KeyError:
                    continue
            else:
                raise KeyError(f"Missing CIF for {mid}")
            structure = Structure.from_str(data.decode("utf-8", errors="replace"), fmt="cif")
        diagnostics = {}
        embedding = build_structure_embedding(structure, WORKER_WL, cnn=WORKER_CNN,
                                              site_fp=WORKER_FP, local_mode="matminer_ops",
                                              diagnostics=diagnostics)
        if not valid_embedding(embedding):
            raise ValueError("Invalid repaired production embedding")
        public["reduced_formula"] = public.get("reduced_formula") or structure.composition.reduced_formula
        public.update(n_sites=len(structure), n_elements=len(structure.composition.elements))
        return True, public, embedding, diagnostics
    except Exception as exc:
        # An optional archive password remains in memory; protected failures
        # persist only the exception class, never potentially sensitive text.
        failure = {"reason": type(exc).__name__}
        if not WORKER_PASSWORD:
            failure["detail"] = str(exc)[:300]
        return False, public, None, failure


def cache_file(directory, signature, key):
    name = hashlib.sha256((signature + "\n" + key).encode()).hexdigest()
    return directory / (name + ".npz")


def load_cached(path, signature, key):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["signature"]) != signature or str(saved["record_key"]) != key:
                return None
            vector = saved["embedding"]
            if not valid_embedding(vector):
                return None
            public = json.loads(str(saved["public_record"]))
            info = json.loads(str(saved["diagnostics"]))
            if record_key(public) != key or not all(k in info for k in
                    ("n_sites", "ordered", "sites_with_unrepresented_cn_mass", "max_unrepresented_cn_mass")):
                return None
            return True, public, vector, info
    except (OSError, ValueError, KeyError, TypeError):
        return None


def store_cached(path, signature, result):
    _, public, vector, info = result
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, signature=signature, record_key=record_key(public),
                            public_record=json.dumps(public), embedding=vector,
                            diagnostics=json.dumps(info))
    temporary.replace(path)


def write_csv(path, records):
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def project_to_communities(vectors, basis):
    standardized = (vectors - basis["scaler_mean"]) / basis["scaler_scale"]
    projected = (standardized - basis["pca_mean"]) @ basis["pca_components"].T
    indices, distances = [], []
    for start in range(0, len(projected), 128):
        matrix = cdist(projected[start:start + 128], basis["centroids"], metric="euclidean")
        nearest = matrix.argmin(axis=1)
        indices.extend(nearest.tolist())
        distances.extend(matrix[np.arange(len(nearest)), nearest].tolist())
    return projected, np.array(indices, dtype=int), np.array(distances)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("gnome", "mp", "jarvis", "alexandria", "mattergen"), required=True)
    parser.add_argument("--source-path", type=Path, required=True,
                        help="GNoME/Alexandria directory; otherwise the frozen source JSONL/JSON/zip file")
    parser.add_argument("--historical-records", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--zip-password-stdin", action="store_true")
    args = parser.parse_args(argv)
    password = None
    if args.zip_password_stdin:
        if args.source not in ("gnome", "mattergen"):
            parser.error("--zip-password-stdin applies only to archive sources")
        password = sys.stdin.readline().rstrip("\r\n")
        if not password:
            parser.error("--zip-password-stdin requires a non-empty input line")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    with np.load(args.basis, allow_pickle=False) as saved:
        basis = {name: saved[name] for name in saved.files if name != "metadata"}
        reference = json.loads(str(saved["metadata"]))
    worker_hashes = {name: sha256_file(Path(__file__).parent / name)
                     for name in ("icsd_densify_worker.py", "crystal_neighbors.py")}
    if (reference.get("feature_version") != FEATURE_VERSION or
            reference.get("neighbor_settings") != NEIGHBOR_SETTINGS or
            reference.get("worker_sha256") != worker_hashes):
        raise ValueError("Projection basis and current repaired encoder provenance differ")
    if not reference.get("fit_transform_reproduces_saved_pca") or not reference.get("row_ids_and_years_verified"):
        raise ValueError("Projection basis lacks successful reference checks")
    print(f"Replaying exact historical {args.source} candidate/sample rules", flush=True)
    selected, historical_keys, source_files, cohort = load_cohort(args.source, args.source_path, args.historical_records)
    sources = [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in source_files]
    encoder = {"feature_version": FEATURE_VERSION, "neighbor_settings": dict(NEIGHBOR_SETTINGS),
               "worker_sha256": worker_hashes, "wl_iters": reference["wl_iters"], "local_mode": "matminer_ops",
               "source": args.source, "source_sha256": [r["sha256"] for r in sources]}
    signature = json.dumps(encoder, sort_keys=True, separators=(",", ":"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or args.out_dir / "feature_cache"
    cache.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "attempted_cohort.csv", [public_fields(r) for r in selected])
    results, pending = {}, []
    for record in selected:
        key = record_key(record)
        result = load_cached(cache_file(cache, signature, key), signature, key)
        if result is None:
            pending.append(record)
        else:
            results[key] = result
    n_cached = len(results)
    print(f"{len(selected)} attempted IDs; {n_cached} valid cached features; {len(pending)} to compute", flush=True)
    initargs = (args.source, str(args.source_path), reference["wl_iters"], password)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs, initializer=initialize_worker, initargs=initargs) as pool:
        for index, result in enumerate(pool.map(featurize, pending, chunksize=8), 1):
            key = record_key(result[1])
            results[key] = result
            if result[0]:
                store_cached(cache_file(cache, signature, key), signature, result)
            if index % 250 == 0 or index == len(pending):
                print(f"Computed {index}/{len(pending)}", flush=True)
    ordered = [results[record_key(r)] for r in selected]
    successful = [r for r in ordered if r[0]]
    failures = [{**public, **info} for ok, public, _, info in ordered if not ok]
    slug = "mattergen-public" if args.source == "mattergen" else args.source
    (args.out_dir / f"{slug}_frontier_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if not successful:
        raise ValueError("No external structures featurized successfully; full failures have been saved")
    vectors = np.vstack([r[2] for r in successful])
    projected, nearest, distances = project_to_communities(vectors, basis)
    thresholds_path = args.basis.parent / "community_thresholds.json"
    thresholds = json.loads(thresholds_path.read_text())
    expected_radii = {str(c): float(v) for c, v in zip(basis["communities"], basis["p95"])}
    if thresholds.get("feature_version") != FEATURE_VERSION or thresholds.get("per_community_p95_threshold") != expected_radii:
        raise ValueError("Threshold JSON does not match the numeric projection basis")
    pooled = thresholds["pooled_p95_threshold"]
    rows = []
    for i, ((_, public, _, _), index, distance) in enumerate(zip(successful, nearest, distances)):
        radius = float(basis["p95"][index])
        rows.append({**public, "assigned_community": int(basis["communities"][index]),
                     "nearest_centroid_distance": float(distance), "community_threshold_p95": radius,
                     "in_basin": bool(distance <= radius), "outlier_like": bool(distance > radius),
                     "outlier_like_pooled": bool(distance > pooled),
                     "pca1": float(projected[i, 0]), "pca2": float(projected[i, 1]), "feature_row": i})
    keys = [record_key(r) for r in rows]
    old_set, new_set = set(historical_keys), set(keys)
    infos = [r[3] for r in successful]
    summary = {
        **encoder, "projection_basis": {"path": str(args.basis.resolve()), "sha256": sha256_file(args.basis)},
        "threshold_file_sha256": sha256_file(thresholds_path),
        "source_files": sources, "historical_records_sha256": sha256_file(args.historical_records),
        "cohort": cohort, "sample_size_requested": 386 if args.source == "mattergen" else 5000,
        "n_records_loaded": len(selected), "n_featurized": len(rows), "n_failures": len(failures),
        "n_valid_cache_hits": n_cached, "threshold_mode": "per_community_p95",
        "outlier_like_definition": "distance > assigned community p95; equality is in-basin",
        "pooled_flag_definition": "outlier_like_pooled retains the separate historical pooled-p95 statistic",
        "pca_plot_coordinates": "First two coordinates of the verified 32-D reference basis; no separate PCA-2 fit",
        "n_in_basin": sum(r["in_basin"] for r in rows), "in_basin_ratio": float(np.mean([r["in_basin"] for r in rows])),
        "n_outlier_like": sum(r["outlier_like"] for r in rows), "outlier_like_ratio": float(np.mean([r["outlier_like"] for r in rows])),
        "outlier_like_pooled_ratio": float(np.mean([r["outlier_like_pooled"] for r in rows])),
        "successful_id_changes": {"shared_count": len(old_set & new_set),
                                  "only_historical_successful": sorted(old_set - new_set),
                                  "newly_successful": sorted(new_set - old_set)},
        "feature_diagnostics": {"n_structures": len(infos), "n_disordered": sum(not r["ordered"] for r in infos),
                                "n_structures_with_unrepresented_cn_mass": sum(r["sites_with_unrepresented_cn_mass"] > 0 for r in infos),
                                "n_sites_with_unrepresented_cn_mass": sum(r["sites_with_unrepresented_cn_mass"] for r in infos),
                                "max_unrepresented_cn_mass": max(r["max_unrepresented_cn_mass"] for r in infos)},
    }
    write_csv(args.out_dir / f"{slug}_frontier_records.csv", rows)
    np.save(args.out_dir / "features.npy", vectors)
    np.save(args.out_dir / "features_pca.npy", projected)
    (args.out_dir / "feature_ids.json").write_text(json.dumps(keys) + "\n")
    (args.out_dir / "feature_metadata.json").write_text(json.dumps({**encoder, "n_features": 213,
                                                                   "n_rows": len(rows)}, indent=2) + "\n")
    (args.out_dir / f"{slug}_frontier_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"source": args.source, "n_attempted": len(selected), "n_featurized": len(rows),
                      "n_failures": len(failures), "in_basin_ratio": summary["in_basin_ratio"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
