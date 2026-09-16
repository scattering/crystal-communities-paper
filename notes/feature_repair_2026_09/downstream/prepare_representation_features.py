#!/usr/bin/env python3
"""Extend validated raw AMD or repaired graphlet CDFs to current non-noise IDs.

Graphlet extension freezes the supplied bin edges; it never mixes CDFs made
with independently fitted bins. Reused rows are numerically checked against
fresh CIF featurization before combining old and new rows. Password input is
stdin only. Failures retain IDs, stages and exception types, never messages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments/graphlet_compare")]
from crystal_neighbors import FEATURE_VERSION

_ZIP = _PASSWORD = _KIND = _EDGES = _ZIP_NAMES = _GRAPHLET_NEIGHBOR_METHOD = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_labels(path):
    rows = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            iid = int(row["icsd_id"])
            if iid in rows:
                raise ValueError(f"Duplicate ICSD ID {iid}")
            year = row["year"].strip()
            rows[iid] = {"icsd_id": iid, "year": int(float(year)) if year else None,
                         "community": int(row["community"])}
    return rows


def initialize(zip_path, password, kind, edges, graphlet_neighbor_method):
    global _ZIP, _PASSWORD, _KIND, _EDGES, _ZIP_NAMES, _GRAPHLET_NEIGHBOR_METHOD
    warnings.filterwarnings("ignore")
    _ZIP = zipfile.ZipFile(zip_path)
    _PASSWORD, _KIND, _EDGES = password, kind, edges
    _GRAPHLET_NEIGHBOR_METHOD = graphlet_neighbor_method
    _ZIP_NAMES = {}
    if kind == "amd":
        for name in _ZIP.namelist():
            leaf = name.rsplit("/", 1)[-1]
            if leaf.lower().endswith(".cif"):
                stem = leaf[:-4]
                if stem.lower().startswith("icsd_"):
                    stem = stem[5:]
                if stem.isdigit():
                    _ZIP_NAMES[int(stem)] = name


def featurize(iid):
    stage = "read_cif"
    try:
        if _KIND == "graphlet":
            from icsd_densify_worker import read_structure_from_zip
            import graphlet_features as gf
            structure = read_structure_from_zip(_ZIP, iid, _PASSWORD)
            stage = "graphlet_raw"
            raw = gf.all_raw_features(
                structure, neighbor_method=_GRAPHLET_NEIGHBOR_METHOD,
            )
            stage = "graphlet_histogram"
            vector = np.cumsum(gf.histogram_tensor(raw, _EDGES), axis=1).reshape(-1)
        else:
            import amd
            from amd.io import periodicset_from_pymatgen_structure
            from pymatgen.core import Structure
            raw = _ZIP.read(_ZIP_NAMES[iid], pwd=_PASSWORD.encode() if _PASSWORD else None)
            stage = "parse_cif"
            structure = Structure.from_str(raw.decode("utf-8", errors="replace"), fmt="cif")
            stage = "amd_periodicset_adapter"
            periodic_set = periodicset_from_pymatgen_structure(structure)
            stage = "amd_100"
            vector = amd.AMD(periodic_set, 100)
        vector = np.asarray(vector, dtype=np.float32)
        stage = "validate_feature_vector"
        width = 1280 if _KIND == "graphlet" else 100
        if vector.shape != (width,) or not np.isfinite(vector).all():
            raise ValueError("Invalid feature vector")
        return iid, vector, None
    except Exception as exc:
        return iid, None, {"icsd_id": int(iid), "stage": stage, "exception_type": type(exc).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("graphlet", "amd"), required=True)
    parser.add_argument("--production-labels", required=True, type=Path)
    parser.add_argument("--production-metadata", required=True, type=Path)
    parser.add_argument("--existing-features", required=True, type=Path)
    parser.add_argument("--existing-ids", required=True, type=Path)
    parser.add_argument("--existing-metadata", type=Path)
    parser.add_argument("--bin-edges", type=Path)
    parser.add_argument("--graphlet-neighbor-method", choices=("crystalnn", "voronoinn"),
                        default="crystalnn")
    parser.add_argument("--icsd-zip", required=True, type=Path)
    parser.add_argument("--zip-password-stdin", required=True, action="store_true")
    parser.add_argument("--procs", type=int, default=16)
    parser.add_argument("--reuse-check-count", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.procs < 1 or args.reuse_check_count < 1:
        parser.error("Positive process and reuse-check counts are required")
    password = sys.stdin.readline().rstrip("\r\n")
    if not password:
        parser.error("A nonempty password line is required on stdin")
    production_metadata = json.loads(args.production_metadata.read_text())
    if production_metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Production labels must be associated with repaired features")
    labels = read_labels(args.production_labels)
    targets = sorted(iid for iid, row in labels.items() if row["community"] >= 0)
    existing_ids = json.loads(args.existing_ids.read_text())
    if not all(isinstance(iid, int) and iid > 0 for iid in existing_ids) or len(set(existing_ids)) != len(existing_ids):
        raise ValueError("Existing row IDs must be unique positive integers")
    existing = np.load(args.existing_features, mmap_mode="r", allow_pickle=False)
    width = 1280 if args.kind == "graphlet" else 100
    if existing.shape != (len(existing_ids), width) or not np.isfinite(existing).all():
        raise ValueError("Existing matrix shape/finite-value check failed")
    existing_row = {iid: row for row, iid in enumerate(existing_ids)}
    reused = [iid for iid in targets if iid in existing_row]
    missing = [iid for iid in targets if iid not in existing_row]
    source_metadata = json.loads(args.existing_metadata.read_text()) if args.existing_metadata else None
    edges = None
    if args.kind == "graphlet":
        import graphlet_features as gf
        expected_settings = gf.neighbor_settings(args.graphlet_neighbor_method)
        expected_graphlet_version = gf.GRAPHLET_FEATURE_VERSIONS[args.graphlet_neighbor_method]
        source_method = (source_metadata.get("neighbor_method") or "crystalnn") if source_metadata else None
        source_graphlet_version = source_metadata.get("graphlet_feature_version") if source_metadata else None
        version_matches = (
            source_graphlet_version == expected_graphlet_version
            or (args.graphlet_neighbor_method == "crystalnn" and source_graphlet_version is None)
        )
        if (not source_metadata or source_metadata.get("feature_version") != FEATURE_VERSION
                or source_method != args.graphlet_neighbor_method
                or not version_matches
                or source_metadata.get("neighbor_settings") != expected_settings
                or source_metadata.get("representation") != gf.GRAPHLET_REPRESENTATION):
            raise ValueError("Reused CDF lacks matching repaired version/settings/representation")
        if args.bin_edges is None:
            parser.error("Graphlet extension requires the original CDF bin edges")
        edges = json.loads(args.bin_edges.read_text())
        if set(edges) != set(gf.REGISTRY.all):
            raise ValueError("Bin-edge feature names do not match the registry")
        if any(len(v) != 2 or not np.isfinite(v).all() or v[1] <= v[0] for v in edges.values()):
            raise ValueError("Invalid frozen bin edges")
        for start in range(0, len(existing), 4096):
            chunk = existing[start:start + 4096].reshape(-1, 64, 20)
            if (np.any(chunk < -2e-6) or np.any(chunk > 1 + 2e-6)
                    or np.any(np.diff(chunk, axis=2) < -2e-6)
                    or np.any(np.abs(chunk[:, :, -1] - 1) > 2e-6)):
                raise ValueError("Reused matrix is not a normalized CDF")
    elif np.any(existing < 0) or np.any(np.diff(existing, axis=1) < -1e-6):
        raise ValueError("Raw AMD distances must be nonnegative and nondecreasing")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"kind": args.kind, "production_feature_version": FEATURE_VERSION,
        "target_definition": "all current production non-noise IDs, including undated entries",
        "n_requested": len(targets), "n_reused_candidates": len(reused), "n_missing_to_attempt": len(missing),
        "requested_ids": targets, "missing_ids_to_attempt": missing,
        "n_requested_undated": sum(labels[iid]["year"] is None for iid in targets),
        "production_labels": str(args.production_labels), "production_labels_sha256": digest(args.production_labels),
        "existing_features": str(args.existing_features), "existing_features_sha256": digest(args.existing_features),
        "existing_ids": str(args.existing_ids), "existing_ids_sha256": digest(args.existing_ids),
        "existing_metadata": source_metadata,
        "bin_edges_sha256": digest(args.bin_edges) if args.bin_edges else None,
        "representation": gf.GRAPHLET_REPRESENTATION if args.kind == "graphlet" else "AMD-100-raw-periodic-point-set",
        "feature_version": FEATURE_VERSION if args.kind == "graphlet" else "amd-100-existing-cache-verified-extension",
        "neighbor_method": args.graphlet_neighbor_method if args.kind == "graphlet" else None,
        "neighbor_settings": gf.neighbor_settings(args.graphlet_neighbor_method) if args.kind == "graphlet" else None,
        "graphlet_feature_version": gf.GRAPHLET_FEATURE_VERSIONS[args.graphlet_neighbor_method] if args.kind == "graphlet" else None,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__), ROOT / "experiments/graphlet_compare/graphlet_features.py", ROOT / "scripts/crystal_neighbors.py"]},
        "packages": {}, "reuse_validation": [], "failures": []}
    for package in ("numpy", "pymatgen", "average-minimum-distance"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report_path = args.out_dir / "feature_preparation.json"
    report_path.write_text(json.dumps(report, indent=2))
    rng = np.random.default_rng(42)
    check_ids = sorted(rng.choice(reused, min(args.reuse_check_count, len(reused)), replace=False).tolist())
    matrix = np.empty((len(targets), width), dtype=np.float32)
    rows, successes = {iid: n for n, iid in enumerate(targets)}, set(reused)
    for iid in reused:
        matrix[rows[iid]] = existing[existing_row[iid]]
    started = time.time()
    with mp.get_context("spawn").Pool(args.procs, initializer=initialize,
            initargs=(str(args.icsd_zip), password, args.kind, edges,
                      args.graphlet_neighbor_method)) as pool:
        for iid, vector, error in pool.imap_unordered(featurize, check_ids, chunksize=1):
            reference = existing[existing_row[iid]]
            matches = error is None and np.allclose(vector, reference, rtol=1e-6, atol=2e-6)
            report["reuse_validation"].append({"icsd_id": iid, "matches": bool(matches),
                "max_abs_difference": float(np.max(np.abs(vector-reference))) if error is None else None,
                "failure": error})
        report_path.write_text(json.dumps(report, indent=2))
        if not all(row["matches"] for row in report["reuse_validation"]):
            raise ValueError("Fresh-CIF reuse checks failed; do not combine incompatible feature caches")
        for n, (iid, vector, error) in enumerate(pool.imap_unordered(featurize, missing, chunksize=1), 1):
            if error:
                report["failures"].append(error)
            else:
                matrix[rows[iid]] = vector
                successes.add(iid)
            if n % 1000 == 0 or n == len(missing):
                report.update(n_new_attempted=n, n_successful_so_far=len(successes), runtime_seconds=time.time()-started)
                report_path.write_text(json.dumps(report, indent=2))
                print(f"{args.kind}: attempted missing {n}/{len(missing)}, successful total {len(successes)}", flush=True)
    ordered_ids = [iid for iid in targets if iid in successes]
    matrix = matrix[[rows[iid] for iid in ordered_ids]]
    np.save(args.out_dir / "features.npy", matrix)
    (args.out_dir / "features.ids.json").write_text(json.dumps(ordered_ids))
    if edges is not None:
        (args.out_dir / "bin_edges.json").write_text(json.dumps(edges, indent=2))
    with (args.out_dir / "production_assignments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["icsd_id", "year", "community"])
        writer.writeheader()
        writer.writerows(labels[iid] for iid in ordered_ids)
    report.update(n_successful=len(ordered_ids), n_failed=len(report["failures"]), n_reused=len(reused),
        n_new_successful=len(successes)-len(reused), n_successful_undated=sum(labels[iid]["year"] is None for iid in ordered_ids),
        runtime_seconds=time.time()-started, ids_sha256=digest(args.out_dir/"features.ids.json"),
        features_sha256=digest(args.out_dir/"features.npy"), feature_shape=list(matrix.shape), complete=True)
    if report["n_successful"] + report["n_failed"] != report["n_requested"]:
        raise ValueError("Feature coverage counts do not sum")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("n_requested", "n_successful", "n_failed", "n_reused", "n_new_successful")}))


if __name__ == "__main__":
    main()
