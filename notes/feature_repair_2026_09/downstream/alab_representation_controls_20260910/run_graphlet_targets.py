#!/usr/bin/env python3
"""Project the 57 frozen pre-experiment A-Lab targets into existing Graphlet maps.

No download, refitting, structure standardization or post-experiment substitution.
The original repaired encoder supplies weighted observations, normalized histograms
and float32 CDF coordinates. Existing nearest-centroid/member-p95 bases are frozen.
"""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parent
DOWNSTREAM = PACKAGE.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments/graphlet_compare")]

import numpy as np
from pymatgen.core import Composition, Structure
import graphlet_features as gf
from external_representation_sensitivity import classify

SOURCE = DOWNSTREAM / "alab_mp_targets/source"
BASIS_DIRS = {
    "crystalnn": DOWNSTREAM / "external_representation/graphlet/basis",
    "voronoinn": DOWNSTREAM / "representation_mechanism_diagnostics/voronoi_external/results/basis",
}
WORKER_METHOD = WORKER_EDGES = WORKER_OUT = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def same_composition(left, right, *, normalized):
    if normalized:
        left, right = left.fractional_composition, right.fractional_composition
    a, b = left.get_el_amt_dict(), right.get_el_amt_dict()
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= 1e-10 for k in a)


def load_targets():
    docs_path = SOURCE / "mp_2022_10_28_summary_target_docs.json"
    outcomes_path = SOURCE / "alab_targets.csv"
    docs = json.loads(docs_path.read_text())
    with outcomes_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["mp_id"] for row in rows]
    assert len(rows) == len(set(ids)) == len(docs) == 57
    assert set(ids) == set(docs)
    attempted, tasks, identity = [], [], []
    for i, row in enumerate(rows):
        mid = row["mp_id"]
        doc = docs[mid]
        structure = Structure.from_dict(doc["structure"])
        checks = {
            "snapshot_id_matches": str(doc["material_id"]) == mid,
            "snapshot_nsites_matches": int(doc["nsites"]) == len(structure),
            "snapshot_composition_matches": same_composition(
                structure.composition, Composition(doc["composition"]), normalized=False),
            "target_formula_matches_normalized_composition": same_composition(
                structure.composition, Composition(row["formula"]), normalized=True),
        }
        if not all(checks.values()):
            raise ValueError(f"Pre-experiment target identity/composition mismatch: {mid}: {checks}")
        public = {"target_row": i, "mp_id": mid, "formula": row["formula"],
                  "corrected_outcome": row["corrected_outcome"], "n_sites": len(structure),
                  "structure_formula": structure.composition.formula,
                  "snapshot_structure_sha256": hashlib.sha256(json.dumps(
                      doc["structure"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        attempted.append(public)
        tasks.append((public, doc["structure"]))
        identity.append({**public, **checks})
    return attempted, tasks, identity, [docs_path, outcomes_path]


def initialize_worker(method, edges, output):
    global WORKER_METHOD, WORKER_EDGES, WORKER_OUT
    WORKER_METHOD, WORKER_EDGES, WORKER_OUT = method, edges, Path(output)


def encode_target(task):
    public, serialized = task
    started = time.perf_counter()
    stage = "parse_structure"
    try:
        structure = Structure.from_dict(serialized)
        stage = "site_count"
        if not 0 < len(structure) <= 256:
            raise ValueError("Unsupported site count under the frozen external-cohort protocol")
        stage = "graphlet_observations"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw = gf.all_raw_features(structure, neighbor_method=WORKER_METHOD)
            stage = "frozen_histograms"
            hist = gf.histogram_tensor(raw, WORKER_EDGES)
        coordinates = np.cumsum(hist, axis=1).reshape(-1).astype(np.float32)
        tensor = coordinates.reshape(64, 20)
        if (hist.shape != (64, 20) or not np.isfinite(hist).all()
                or np.any(hist < 0) or not np.allclose(hist.sum(axis=1), 1, atol=2e-6, rtol=0)
                or not np.isfinite(coordinates).all() or np.any(np.diff(tensor, axis=1) < -2e-6)
                or not np.allclose(tensor[:, -1], 1, atol=2e-6, rtol=0)):
            raise ValueError("Invalid normalized histogram/CDF coordinates")
        stage = "save_raw_observations"
        raw_path = WORKER_OUT / "raw_features" / f"{public['mp_id']}.npz"
        np.savez_compressed(raw_path, **{k: np.asarray(raw[k], dtype=np.float64).reshape(-1, 2)
                                      for k in gf.REGISTRY.all})
        warning_counts = {}
        for item in caught:
            key = f"{item.category.__name__}: {item.message}"
            warning_counts[key] = warning_counts.get(key, 0) + 1
        detail = {**public, "elapsed_seconds": time.perf_counter() - started,
                  "raw_features_path": str(raw_path.relative_to(PACKAGE)),
                  "raw_features_sha256": sha(raw_path),
                  "channel_observation_counts": {k: len(raw[k]) for k in gf.REGISTRY.all},
                  "warnings": warning_counts}
        return detail, hist, coordinates, None
    except Exception as exc:
        return public, None, None, {**public, "stage": stage,
                                   "exception_type": type(exc).__name__, "detail": str(exc),
                                   "elapsed_seconds": time.perf_counter() - started}


def run_method(method, workers, attempted, tasks, identity, sources):
    start = time.perf_counter()
    output = PACKAGE / f"graphlet_{method}"
    output.mkdir(exist_ok=True)
    (output / "raw_features").mkdir(exist_ok=True)
    basis_dir = BASIS_DIRS[method]
    manifest_path, edges_path, basis_path = (basis_dir / x for x in
                                            ("basis_manifest.json", "bin_edges.json", "basis_full.npz"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["feature_version"] == gf.GRAPHLET_FEATURE_VERSIONS[method]
    assert manifest["neighbor_settings"] == gf.neighbor_settings(method)
    for name, digest in manifest["code_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Encoder source differs from frozen basis: {name}")
    edges = json.loads(edges_path.read_text())
    assert set(edges) == set(gf.REGISTRY.all)
    if method == "crystalnn":
        expected_edges_hash = manifest["reference_files"]["bin_edges"]["sha256"]
    else:
        expected_edges_hash, = [h for name, h in manifest["input_sha256"].items()
                                if Path(name).name == "bin_edges.json"]
    assert sha(edges_path) == expected_edges_hash
    with np.load(basis_path, allow_pickle=False) as saved:
        basis = dict(saved)
    assert basis["centroids"].shape[1] == 1280
    assert len(basis["communities"]) == manifest["reference_rates"]["full"]["n_communities"]
    packages = {p: importlib.metadata.version(p) for p in
                ("numpy", "scipy", "scikit-learn", "pymatgen", "pymatgen-core", "matminer")}
    version_differences = {p: {"frozen_producer": version, "local": packages[p]}
                           for p, version in manifest["packages"].items() if packages[p] != version}
    # Project conda differs only in the NumPy patch release; preserve this fact.
    if set(version_differences) - {"numpy"}:
        raise ValueError(f"Unexpected frozen-runtime differences: {version_differences}")
    dump(output / "attempted_ids.json", [r["mp_id"] for r in attempted])
    write_csv(output / "attempted_records.csv", attempted)
    dump(output / "identity_verification.json", {"status": "PASS", "n_targets": 57,
         "scope": "Exact 2022-10-28 public snapshot structures with corrected outcomes; no refinements",
         "formula_rule": "Identical element sets and normalized element amounts within 1e-10; snapshot composition also checked without normalization",
         "records": identity})
    encoded = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker,
            initargs=(method, edges, str(output))) as pool:
        for i, result in enumerate(pool.map(encode_target, tasks, chunksize=1), 1):
            encoded.append(result)
            if i % 10 == 0 or i == len(tasks):
                print(f"{method}: encoded {i}/{len(tasks)}", flush=True)
    successful = [r for r in encoded if r[3] is None]
    failures = [r[3] for r in encoded if r[3] is not None]
    dump(output / "failures.json", failures)
    assert len(successful) + len(failures) == 57
    if not successful:
        raise ValueError(f"No successful {method} targets")
    features = np.stack([r[1] for r in successful])
    coordinates = np.stack([r[2] for r in successful])
    ids = [r[0]["mp_id"] for r in successful]
    np.save(output / "features.npy", features)
    np.save(output / "coordinates.npy", coordinates)
    dump(output / "feature_ids.json", ids)
    dump(output / "feature_details.json", [r[0] for r in successful])
    nearest, distances, inside = classify(coordinates, basis)
    rows = []
    for i, (record, j, distance, flag) in enumerate(zip(successful, nearest, distances, inside)):
        threshold = float(basis["p95"][j])
        rows.append({**{k: record[0][k] for k in attempted[0]}, "feature_row": i,
                     "assigned_center_index": int(j), "assigned_community": int(basis["communities"][j]),
                     "nearest_centroid_distance": float(distance), "community_threshold_p95": threshold,
                     "distance_minus_p95": float(distance - threshold),
                     "distance_over_p95": float(distance / threshold) if threshold > 0 else "",
                     "assigned_center_member_count": int(basis["counts"][j]), "in_basin": bool(flag)})
    write_csv(output / "projection_records.csv", rows)
    provenance_files = sources + [manifest_path, edges_path, basis_path, Path(__file__),
        ROOT / "experiments/graphlet_compare/graphlet_features.py", ROOT / "scripts/crystal_neighbors.py",
        ROOT / "scripts/external_representation_sensitivity.py"]
    zero_assigned = [r["mp_id"] for r in rows if r["community_threshold_p95"] == 0]
    summary = {"status": "COMPLETE", "representation": f"Graphlet ({'CrystalNN' if method == 'crystalnn' else 'VoronoiNN'})",
        "neighbor_method": method, "feature_version": gf.GRAPHLET_FEATURE_VERSIONS[method],
        "neighbor_settings": gf.neighbor_settings(method), "n_attempted": 57,
        "n_successful": len(successful), "n_failures": len(failures), "n_in_basin": int(inside.sum()),
        "basis_community_count": len(basis["communities"]),
        "basis_zero_radius_center_count": int(np.sum(basis["p95"] == 0)),
        "target_zero_radius_assignment_count": len(zero_assigned), "target_zero_radius_assignment_ids": zero_assigned,
        "smallest_absolute_distance_to_threshold": min(abs(r["distance_minus_p95"]) for r in rows),
        "feature_schema": {"raw_features/*.npz": "64 named arrays of weighted (value, weight) observations, float64",
                           "features.npy": "N x 64 x 20 normalized probability histograms, float32",
                           "coordinates.npy": "N x 1280 unscaled within-channel CDF coordinates, float32",
                           "feature_ids.json": "mp_id order shared by both matrices and feature_row"},
        "fitting": "None; original bins, full-map community centroids and member-p95 radii frozen",
        "classification": "Euclidean nearest centroid, then direct norm <= that centroid's member-p95 radius; equality accepted; original zero-radius centers preserved",
        "outcome_testing": "Deferred to parent analysis; all 57 corrected outcomes retained",
        "source_and_code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in provenance_files},
        "basis_manifest_sha256": sha(manifest_path), "basis_sha256": sha(basis_path),
        "bin_edges_sha256": sha(edges_path), "encoder_hashes_match_frozen_basis": True,
        "packages": packages, "frozen_producer_packages": manifest["packages"],
        "runtime_version_differences": version_differences, "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(), "workers": workers,
        "blas_thread_limit": 1, "elapsed_seconds": time.perf_counter() - start,
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    dump(output / "summary.json", summary)
    payload = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(output)}\n" for p in payload))
    print(json.dumps({k: summary[k] for k in ("representation", "n_successful", "n_failures",
        "n_in_basin", "target_zero_radius_assignment_count", "elapsed_seconds")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("both", "crystalnn", "voronoinn"), default="both")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    attempted, tasks, identity, sources = load_targets()
    methods = ("crystalnn", "voronoinn") if args.method == "both" else (args.method,)
    for method in methods:
        run_method(method, args.workers, attempted, tasks, identity, sources)


if __name__ == "__main__":
    main()
