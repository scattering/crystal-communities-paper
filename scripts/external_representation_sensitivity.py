#!/usr/bin/env python3
"""Project the frozen public cohorts into repaired Magpie/graphlet ICSD maps.

This is a full-map representation sensitivity, never an independently trained
historical validation. Full-map ICSD rates are explicitly in-sample calibration
descriptives. Additional T1990/T2000/T2010 comparisons estimate centroids/radii
only from pre-cutoff members and evaluate later ICSD entries, while retaining
the full-fitted representation and partition. No download or API fallback exists.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/graphlet_compare"))
from external_frozen_cohorts import load_cohort, public_fields, record_key

SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")
SEMANTICS = {
    "full": "Full-map calibration sensitivity; the ICSD reference helped fit its own representation, partition, centroids and radii.",
    "retrospective": "Full-map retrospective sensitivity; centroids/radii use only year <= T, but representation and partition were fitted to all map entries. This is not cutoff-trained validation.",
}
WORKER_KIND = WORKER_EDGES = WORKER_ARCHIVE = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def encoder_provenance(kind):
    from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS
    files = [ROOT / "scripts/crystal_neighbors.py", ROOT / "scripts/icsd_densify_worker.py"]
    if kind == "magpie":
        import icsd_ablation_paper_text_worker as worker
        version = worker.FEATURE_VERSION
        files.append(ROOT / "scripts/icsd_ablation_paper_text_worker.py")
        missing = "Retains the repaired Magpie ablation's declared zero filling of unavailable Magpie table values. Featurization exceptions and nonfinite final vectors are explicit failures."
    else:
        import graphlet_features as gf
        version = gf.GRAPHLET_FEATURE_VERSIONS["crystalnn"]
        files.append(ROOT / "experiments/graphlet_compare/graphlet_features.py")
        missing = "Retains the repaired graphlet channel definitions; unavailable channel observations are excluded, and zero-mass/missing histograms fail explicitly."
    return {"kind": kind, "feature_version": version, "production_feature_version": FEATURE_VERSION,
            "neighbor_settings": NEIGHBOR_SETTINGS, "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "missing_property_policy": missing}


def aligned_reference(feature_ids, assignments):
    """Join by identity; exclude rows outside this representation's map."""
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("Duplicate feature IDs")
    labels = {int(row["icsd_id"]): row for row in assignments}
    if len(labels) != len(assignments):
        raise ValueError("Duplicate partition IDs")
    feature_index = {iid: i for i, iid in enumerate(feature_ids)}
    absent = set(labels) - set(feature_index)
    if absent:
        raise ValueError(f"Partition contains {len(absent)} unrepresented IDs")
    ids = [iid for iid in feature_ids if iid in labels]
    rows = [labels[iid] for iid in ids]
    years = [int(r["year"]) if r["year"].strip() else -1 for r in rows]
    return np.array([feature_index[i] for i in ids]), np.array(ids), np.array(years), np.array([int(r["community"]) for r in rows])


def community_basis(coordinates, labels, training_mask):
    """Arithmetic centroids and member-distance p95; noise never defines a basin."""
    labels = np.asarray(labels)
    mask = np.asarray(training_mask, dtype=bool)
    if coordinates.ndim != 2 or len(coordinates) != len(labels) or mask.shape != labels.shape:
        raise ValueError("Reference dimensions disagree")
    communities = np.unique(labels[mask & (labels >= 0)])
    if not len(communities):
        raise ValueError("No training communities")
    centers, radii, counts = [], [], []
    for c in communities:
        values = np.asarray(coordinates[mask & (labels == c)], dtype=np.float64)
        center = values.mean(axis=0)
        centers.append(center)
        radii.append(np.quantile(np.linalg.norm(values - center, axis=1), .95))
        counts.append(len(values))
    return {"communities": communities, "centroids": np.array(centers), "p95": np.array(radii), "counts": np.array(counts)}


def classify(coordinates, basis, batch_size=256):
    """Assign nearest Euclidean centroid, then test its own p95 (equality included).

    The minimum normalized distance is deliberately not used: a distant large
    basin cannot replace the nearest centroid under the declared rule.
    """
    centers = np.asarray(basis["centroids"], dtype=np.float64)
    if (coordinates.ndim != 2 or centers.ndim != 2 or coordinates.shape[1] != centers.shape[1]
            or not len(centers) or not np.isfinite(coordinates).all() or not np.isfinite(centers).all()
            or len(basis["p95"]) != len(centers) or np.any(basis["p95"] < 0)):
        raise ValueError("Invalid projection coordinates or basis")
    center_norm = np.einsum("ij,ij->i", centers, centers)
    nearest = np.empty(len(coordinates), dtype=int)
    distances = np.empty(len(coordinates), dtype=float)
    for start in range(0, len(coordinates), batch_size):
        block = np.asarray(coordinates[start:start + batch_size], dtype=np.float64)
        squared = np.einsum("ij,ij->i", block, block)[:, None] + center_norm - 2 * block @ centers.T
        chosen = np.argmin(squared, axis=1)
        nearest[start:start + len(block)] = chosen
        # Compute the selected distance directly to preserve p95 boundary tests
        # and avoid cancellation for exact centroid matches.
        distances[start:start + len(block)] = np.linalg.norm(block - centers[chosen], axis=1)
    return nearest, distances, distances <= np.asarray(basis["p95"])[nearest]


def rate(flags):
    flags = np.asarray(flags, dtype=bool)
    interval = None
    if len(flags):
        # Nominal binomial uncertainty only; community/member dependencies and
        # fixed-cohort selection are not captured by this interval.
        n, p, z = len(flags), float(flags.mean()), 1.959963984540054
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        halfwidth = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        interval = [max(0., float(center - halfwidth)), min(1., float(center + halfwidth))]
    return {"n": len(flags), "n_in_basin": int(flags.sum()),
            "in_basin_fraction": float(flags.mean()) if len(flags) else None,
            "wilson_95_ci": interval,
            "interval_interpretation": "Nominal binomial interval; ignores within-community dependence and does not convert calibration into held-out validation."}


def projection_rows(keys, coordinates, basis):
    nearest, distances, inside = classify(coordinates, basis)
    rows = [{"record_key": str(key), "assigned_community": int(basis["communities"][j]),
             "nearest_centroid_distance": float(d), "community_threshold_p95": float(basis["p95"][j]),
             "in_basin": bool(ok)} for key, j, d, ok in zip(keys, nearest, distances, inside)]
    return rows, rate(inside)


def make_basis(args):
    from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.metadata.read_text())
    encoder = encoder_provenance(args.kind)
    if meta.get("feature_version") != (encoder["feature_version"] if args.kind == "magpie" else FEATURE_VERSION):
        raise ValueError("Reference metadata feature version is incompatible")
    if meta.get("neighbor_settings") != NEIGHBOR_SETTINGS:
        raise ValueError("Reference neighbor settings are incompatible")
    assignments = csv_rows(args.assignments)
    raw = np.load(args.features, mmap_mode="r", allow_pickle=False)
    if args.ids.suffix == ".csv":
        feature_ids = [int(r["icsd_id"]) for r in csv_rows(args.ids)]
    else:
        feature_ids = json.loads(args.ids.read_text())
    if raw.ndim != 2 or len(raw) != len(feature_ids) or not np.isfinite(raw).all():
        raise ValueError("Invalid reference matrix or ID alignment")
    indices, ids, years, labels = aligned_reference(feature_ids, assignments)
    reference = {name: {"path": str(getattr(args, name)), "sha256": sha(getattr(args, name))}
                 for name in ("features", "ids", "metadata", "assignments")}
    extra = {}
    if args.kind == "magpie":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        if args.pca is None or raw.shape[1] != 4491 or meta.get("wl_iters") != 3:
            raise ValueError("Magpie requires its saved PCA and WL3 4491-D raw features")
        saved = np.load(args.pca, allow_pickle=False)
        if saved.shape != (len(raw), 32):
            raise ValueError("Expected a 32-D saved Magpie map")
        print(f"Reconstructing Magpie scaler/PCA for {raw.shape}", flush=True)
        scaler = StandardScaler()
        standardized = scaler.fit_transform(raw)
        pca = PCA(n_components=32, random_state=42)
        reconstructed = pca.fit_transform(standardized)
        if not np.allclose(saved, reconstructed, rtol=1e-7, atol=1e-8):
            raise ValueError(f"Magpie PCA reconstruction differs: {np.max(np.abs(saved-reconstructed))}")
        transformed = pca.transform(standardized)
        extra = {"fit_transform_reproduces_saved_pca": True,
                 "fit_transform_max_abs_error": float(np.max(np.abs(saved - reconstructed))),
                 "transform_vs_saved_max_abs_error": float(np.max(np.abs(saved - transformed))),
                 "pca_solver": pca._fit_svd_solver,
                 "coordinate_rule": "Saved full-map PCA coordinates define ICSD centers/radii; external features use reconstructed transform."}
        np.savez_compressed(args.out_dir / "transform.npz", scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                            pca_mean=pca.mean_, pca_components=pca.components_)
        coordinates = saved[indices]
        del standardized, reconstructed, transformed
        reference["pca"] = {"path": str(args.pca), "sha256": sha(args.pca)}
    else:
        import graphlet_features as gf
        if raw.shape[1] != 1280 or args.bin_edges is None or (meta.get("neighbor_method") or "crystalnn") != "crystalnn":
            raise ValueError("Expected repaired CrystalNN graphlet CDF and its frozen bin edges")
        edges = json.loads(args.bin_edges.read_text())
        if set(edges) != set(gf.REGISTRY.all) or any(len(v) != 2 or not np.isfinite(v).all() or v[1] <= v[0] for v in edges.values()):
            raise ValueError("Invalid graphlet bin ranges")
        for start in range(0, len(raw), 4096):
            chunk = raw[start:start + 4096].reshape(-1, 64, 20)
            if (np.any(chunk < -2e-6) or np.any(chunk > 1 + 2e-6) or np.any(np.diff(chunk, axis=2) < -2e-6)
                    or np.any(np.abs(chunk[:, :, -1] - 1) > 2e-6)):
                raise ValueError("Reference vectors are not normalized graphlet CDFs")
        coordinates = raw[indices]
        shutil.copyfile(args.bin_edges, args.out_dir / "bin_edges.json")
        reference["bin_edges"] = {"path": str(args.bin_edges), "sha256": sha(args.bin_edges)}
        extra = {"metric": "Euclidean on unscaled 1280-D CDF, matching the independent graphlet temporal partition",
                 "coordinate_rule": "Exact frozen CrystalNN graphlet CDF channels and bins; no PCA or new bin fit."}
    reports = {}
    for cutoff in [None] + args.cutoffs:
        key = "full" if cutoff is None else f"T{cutoff}"
        training = np.ones(len(ids), dtype=bool) if cutoff is None else (years >= 0) & (years <= cutoff)
        evaluation = np.ones(len(ids), dtype=bool) if cutoff is None else years > cutoff
        basis = community_basis(coordinates, labels, training)
        np.savez_compressed(args.out_dir / f"basis_{key}.npz", **basis)
        rows, stats = projection_rows(ids[evaluation], coordinates[evaluation], basis)
        write_csv(args.out_dir / f"icsd_{key}.csv", rows)
        reports[key] = {**stats, "cutoff": cutoff, "n_training_rows_including_noise": int(training.sum()),
                        "n_training_nonnoise": int((training & (labels >= 0)).sum()),
                        "n_communities": len(basis["communities"]),
                        "interpretation": SEMANTICS["full" if cutoff is None else "retrospective"]}
        print(json.dumps({"reference": key, **stats}), flush=True)
    provenance = {**encoder, **extra, "reference_files": reference, "n_feature_rows": len(raw),
                  "n_map_rows": len(ids), "n_map_noise": int((labels < 0).sum()), "n_map_undated": int((years < 0).sum()),
                  "n_feature_rows_outside_map": len(raw) - len(ids), "reference_rates": reports,
                  "radius_rule": "Per-community arithmetic centroid; numpy.quantile(member Euclidean distance, 0.95); nearest-centroid assignment; distance <= radius is in basin; no pooled threshold.",
                  "historical_radius_rule": "Every community with at least one pre-cutoff member defines a historical radius; singleton radius is zero. Noise never defines a centroid.",
                  "script_sha256": sha(__file__), "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "pymatgen", "matminer")}}
    dump(args.out_dir / "basis_manifest.json", provenance)


def initialize_worker(kind, source, source_path, edges):
    global WORKER_KIND, WORKER_EDGES, WORKER_ARCHIVE
    WORKER_KIND, WORKER_EDGES = kind, edges
    if kind == "magpie":
        import icsd_ablation_paper_text_worker as worker
        worker.init_worker(None, None, None, 3, 256)
    WORKER_ARCHIVE = None
    if source in ("gnome", "mattergen"):
        path = Path(source_path) / "by_id.zip" if source == "gnome" else Path(source_path)
        WORKER_ARCHIVE = zipfile.ZipFile(path)


def featurize(record):
    from pymatgen.core import Structure
    public = public_fields(record)
    stage = "parse_structure"
    try:
        kind = record["_kind"]
        if kind == "pymatgen":
            structure = Structure.from_dict(record["_structure"])
        elif kind == "jarvis":
            from analyze_jarvis_frontier import jarvis_atoms_to_structure
            structure = jarvis_atoms_to_structure(record["_structure"])
        else:
            mid = record["material_id"]
            names = [record["zip_member"]] if kind == "mattergen" else [f"{mid}.cif", f"{mid}.CIF", f"by_id/{mid}.cif", f"by_id/{mid}.CIF", f"{mid}.vasp.cif"]
            for name in names:
                try:
                    data = WORKER_ARCHIVE.read(name)
                    break
                except KeyError:
                    continue
            else:
                raise KeyError("Missing public CIF")
            structure = Structure.from_str(data.decode("utf-8", errors="replace"), fmt="cif")
        stage = "site_count"
        if not 0 < len(structure) <= 256:
            raise ValueError("Unsupported site count")
        public["reduced_formula"] = public.get("reduced_formula") or structure.composition.reduced_formula
        public["n_sites"] = len(structure)
        stage = WORKER_KIND + "_features"
        if WORKER_KIND == "magpie":
            import icsd_ablation_paper_text_worker as worker
            vector = worker.build_structure_embedding_paper(structure, 3, cnn=worker.WORKER_CNN,
                       site_fp=worker.WORKER_SITE_FP, geom_dim=worker.WORKER_GEOM_DIM)
            expected = 4491
        else:
            import graphlet_features as gf
            raw = gf.all_raw_features(structure, neighbor_method="crystalnn")
            vector = np.cumsum(gf.histogram_tensor(raw, WORKER_EDGES), axis=1).reshape(-1).astype(np.float32)
            expected = 1280
        stage = "validate_vector"
        if vector.shape != (expected,) or not np.isfinite(vector).all():
            raise ValueError("Invalid representation vector")
        return public, vector, None
    except Exception as exc:
        return public, None, {"stage": stage, "exception_type": type(exc).__name__}


def save_cache(path, signature, record, vector):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, signature=signature, record=json.dumps(record), vector=vector)
    temporary.replace(path)


def load_cache(path, signature, key, width):
    try:
        with np.load(path, allow_pickle=False) as saved:
            record = json.loads(str(saved["record"]))
            vector = saved["vector"]
            if (str(saved["signature"]) != signature or record_key(record) != key
                    or vector.shape != (width,) or not np.isfinite(vector).all()):
                return None
            return record, vector, None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def project_coordinates(raw, basis_dir, kind):
    if kind == "graphlet":
        return raw
    with np.load(Path(basis_dir) / "transform.npz", allow_pickle=False) as t:
        return ((raw - t["scaler_mean"]) / t["scaler_scale"] - t["pca_mean"]) @ t["pca_components"].T


def external_features(args):
    basis_meta = json.loads((args.basis_dir / "basis_manifest.json").read_text())
    encoder = encoder_provenance(args.kind)
    for key, value in encoder.items():
        if basis_meta.get(key) != value:
            raise ValueError(f"External encoder and reference differ: {key}")
    selected, old_keys, files, cohort = load_cohort(args.source, args.source_path, args.historical_records)
    keys = [record_key(r) for r in selected]
    if args.attempted_cohort:
        expected = [record_key(r) for r in csv_rows(args.attempted_cohort)]
        if keys != expected:
            raise ValueError("Attempted IDs/order differ from the frozen repaired cohort")
    sources = [{"path": str(p), "sha256": sha(p)} for p in files]
    edges = json.loads((args.basis_dir / "bin_edges.json").read_text()) if args.kind == "graphlet" else None
    signature = hashlib.sha256(json.dumps({**encoder, "sources": sources, "edges": edges}, sort_keys=True).encode()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir / "feature_cache"
    cache.mkdir(exist_ok=True)
    write_csv(args.out_dir / "attempted_cohort.csv", [public_fields(r) for r in selected])
    dump(args.out_dir / "attempted_ids.json", keys)
    def cache_path(key):
        return cache / (hashlib.sha256((signature + key).encode()).hexdigest() + ".npz")
    width = 4491 if args.kind == "magpie" else 1280
    pending, results = [], {}
    for record in selected:
        key = record_key(record)
        result = load_cache(cache_path(key), signature, key, width)
        if result is None:
            pending.append(record)
        else:
            results[key] = result
    print(f"{args.kind}/{args.source}: {len(selected)} frozen attempts, {len(results)} cached, {len(pending)} pending", flush=True)
    n_cached = len(results)
    init = (args.kind, args.source, str(args.source_path), edges)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker, initargs=init) as pool:
        for index, result in enumerate(pool.map(featurize, pending, chunksize=4), 1):
            record, vector, error = result
            key = record_key(record)
            results[key] = result
            if error is None:
                save_cache(cache_path(key), signature, record, vector)
            if index % 250 == 0 or index == len(pending):
                print(f"{args.kind}/{args.source}: computed {index}/{len(pending)}", flush=True)
    ordered = [results[key] for key in keys]
    success = [r for r in ordered if r[2] is None]
    failures = [{**record, **error} for record, vector, error in ordered if error is not None]
    dump(args.out_dir / "failures.json", failures)
    if len(success) + len(failures) != len(keys) or not success:
        raise ValueError("No successful features or incomplete attempt accounting")
    raw = np.stack([r[1] for r in success])
    success_keys = [record_key(r[0]) for r in success]
    coordinates = project_coordinates(raw, args.basis_dir, args.kind)
    np.save(args.out_dir / "features.npy", raw)
    np.save(args.out_dir / "coordinates.npy", coordinates)
    dump(args.out_dir / "feature_ids.json", success_keys)
    write_csv(args.out_dir / "successful_records.csv", [r[0] for r in success])
    reports = {}
    for key, ref in basis_meta["reference_rates"].items():
        with np.load(args.basis_dir / f"basis_{key}.npz", allow_pickle=False) as saved:
            basis = dict(saved)
        rows, stats = projection_rows(success_keys, coordinates, basis)
        write_csv(args.out_dir / f"projection_{key}.csv", rows)
        reports[key] = {**stats, "icsd_reference": ref, "icsd_minus_external_percentage_points": 100 * (ref["in_basin_fraction"] - stats["in_basin_fraction"]),
                        "external_fraction_all_attempts_lower_bound": stats["n_in_basin"] / len(keys),
                        "external_fraction_all_attempts_upper_bound": (stats["n_in_basin"] + len(failures)) / len(keys)}
    dump(args.out_dir / "summary.json", {**encoder, "source": args.source, "signature": signature,
         "source_files": sources, "cohort": cohort, "n_attempted": len(keys), "n_successful": len(success),
         "n_failures": len(failures), "n_cached": n_cached, "successful_denominator_rule": "Only explicit successful feature vectors; failure bounds additionally use every frozen attempted ID.",
         "basis_manifest_sha256": sha(args.basis_dir / "basis_manifest.json"), "attempted_ids_sha256": sha(args.out_dir / "attempted_ids.json"),
         "feature_ids_sha256": sha(args.out_dir / "feature_ids.json"), "historical_records_sha256": sha(args.historical_records),
         "attempted_cohort_sha256": sha(args.attempted_cohort) if args.attempted_cohort else None,
         "rates": reports, "script_sha256": sha(__file__)})
    print(json.dumps({"source": args.source, "n_successful": len(success), "n_failures": len(failures), "rates": reports}), flush=True)


def summarize(args):
    rows = []
    for source in SOURCES:
        summary = json.loads((args.run_dir / "external" / source / "summary.json").read_text())
        for key, result in summary["rates"].items():
            rows.append({"representation": summary["kind"], "map": key, "source": source,
                         "n_attempted": summary["n_attempted"], "n_successful": summary["n_successful"], "n_failures": summary["n_failures"],
                         "external_in_basin_fraction": result["in_basin_fraction"],
                         "icsd_n": result["icsd_reference"]["n"], "icsd_in_basin_fraction": result["icsd_reference"]["in_basin_fraction"],
                         "icsd_minus_external_percentage_points": result["icsd_minus_external_percentage_points"],
                         "all_attempts_lower": result["external_fraction_all_attempts_lower_bound"],
                         "all_attempts_upper": result["external_fraction_all_attempts_upper_bound"]})
    write_csv(args.run_dir / "comparison.csv", rows)
    dump(args.run_dir / "comparison.json", {"interpretation": SEMANTICS, "rows": rows,
         "full_map_direction_all_five": all(r["icsd_minus_external_percentage_points"] > 0 for r in rows if r["map"] == "full"),
         "retrospective_full_map_direction_all_five_all_cutoffs": all(r["icsd_minus_external_percentage_points"] > 0 for r in rows if r["map"] != "full")})
    print(json.dumps(rows, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    basis = sub.add_parser("basis")
    basis.add_argument("--kind", choices=("magpie", "graphlet"), required=True)
    for name in ("features", "ids", "metadata", "assignments", "out-dir"):
        basis.add_argument("--" + name, type=Path, required=True)
    basis.add_argument("--pca", type=Path)
    basis.add_argument("--bin-edges", type=Path)
    basis.add_argument("--cutoffs", nargs="*", type=int, default=[1990, 2000, 2010])
    basis.set_defaults(function=make_basis)
    feature = sub.add_parser("features")
    feature.add_argument("--kind", choices=("magpie", "graphlet"), required=True)
    feature.add_argument("--source", choices=SOURCES, required=True)
    for name in ("source-path", "historical-records", "basis-dir", "out-dir"):
        feature.add_argument("--" + name, type=Path, required=True)
    feature.add_argument("--attempted-cohort", type=Path)
    feature.add_argument("--n-jobs", type=int, default=16)
    feature.set_defaults(function=external_features)
    summary = sub.add_parser("summarize")
    summary.add_argument("--run-dir", type=Path, required=True)
    summary.set_defaults(function=summarize)
    args = parser.parse_args(argv)
    if getattr(args, "n_jobs", 1) < 1:
        parser.error("n-jobs must be positive")
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
