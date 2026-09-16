#!/usr/bin/env python3
"""Scratch-only AMD-100 full-map projection for the five frozen external cohorts."""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np

SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")
SOURCE_SLUG = {
    "gnome": "gnome",
    "mattergen": "mattergen-public",
    "mp": "mp",
    "jarvis": "jarvis",
    "alexandria": "alexandria",
}
EXPECTED_PRIOR_FOUR_WAY = {
    "icsd": 146186,
    "gnome": 5000,
    "mattergen": 385,
    "mp": 4969,
    "jarvis": 4930,
    "alexandria": 4982,
}

_WORKER_SOURCE = None
_WORKER_ARCHIVE = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ids_sha256(values) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def dump(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def record_key(record):
    return record.get("zip_member") or record["material_id"]


def public_fields(record):
    return {k: v for k, v in record.items() if not k.startswith("_")}


def initialize_worker(source: str, source_path: str):
    global _WORKER_SOURCE, _WORKER_ARCHIVE
    warnings.filterwarnings("ignore")
    _WORKER_SOURCE = source
    _WORKER_ARCHIVE = None
    path = Path(source_path)
    if source == "gnome":
        _WORKER_ARCHIVE = zipfile.ZipFile(path / "by_id.zip")
    elif source == "mattergen":
        _WORKER_ARCHIVE = zipfile.ZipFile(path)


def structure_from_record(record):
    from pymatgen.core import Structure
    kind = record["_kind"]
    if kind == "pymatgen":
        return Structure.from_dict(record["_structure"])
    if kind == "jarvis":
        from analyze_jarvis_frontier import jarvis_atoms_to_structure
        return jarvis_atoms_to_structure(record["_structure"])
    mid = record["material_id"]
    names = ([record["zip_member"]] if kind == "mattergen" else
             [f"{mid}.cif", f"{mid}.CIF", f"by_id/{mid}.cif", f"by_id/{mid}.CIF", f"{mid}.vasp.cif"])
    for name in names:
        try:
            raw = _WORKER_ARCHIVE.read(name)
            return Structure.from_str(raw.decode("utf-8", errors="replace"), fmt="cif")
        except KeyError:
            pass
    raise KeyError("Missing public CIF")


def featurize_amd(record):
    public = public_fields(record)
    stage = "parse_structure"
    try:
        structure = structure_from_record(record)
        stage = "site_count"
        if not 0 < len(structure) <= 256:
            raise ValueError("Unsupported site count")
        public["reduced_formula"] = public.get("reduced_formula") or structure.composition.reduced_formula
        public["n_sites"] = len(structure)
        stage = "amd_periodicset_adapter"
        import amd
        from amd.io import periodicset_from_pymatgen_structure
        periodic_set = periodicset_from_pymatgen_structure(structure)
        stage = "amd_100"
        vector = np.asarray(amd.AMD(periodic_set, 100), dtype=np.float32)
        stage = "validate_feature_vector"
        if (vector.shape != (100,) or not np.isfinite(vector).all() or np.any(vector < 0)
                or np.any(np.diff(vector) < -1e-6)):
            raise ValueError("Invalid AMD-100 vector")
        return public, vector, None
    except Exception as exc:
        return public, None, {"stage": stage, "exception_type": type(exc).__name__}


def community_basis(coordinates, labels):
    communities = np.unique(labels)
    if np.any(communities < 0):
        raise ValueError("AMD full partition unexpectedly contains noise")
    centers, radii, counts = [], [], []
    for community in communities:
        values = np.asarray(coordinates[labels == community], dtype=np.float64)
        center = values.mean(axis=0)
        centers.append(center)
        radii.append(np.quantile(np.linalg.norm(values - center, axis=1), 0.95))
        counts.append(len(values))
    return {
        "communities": np.asarray(communities, dtype=np.int64),
        "centroids": np.asarray(centers, dtype=np.float64),
        "p95": np.asarray(radii, dtype=np.float64),
        "counts": np.asarray(counts, dtype=np.int64),
    }


def classify(coordinates, basis, batch_size=512):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    centers = basis["centroids"]
    center_norm = np.einsum("ij,ij->i", centers, centers)
    nearest = np.empty(len(coordinates), dtype=np.int64)
    distances = np.empty(len(coordinates), dtype=np.float64)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=int(os.environ.get("AMD_CLASSIFY_THREADS", "48"))):
        for start in range(0, len(coordinates), batch_size):
            block = coordinates[start:start + batch_size]
            squared = np.einsum("ij,ij->i", block, block)[:, None] + center_norm - 2 * block @ centers.T
            chosen = np.argmin(squared, axis=1)
            nearest[start:start + len(block)] = chosen
            distances[start:start + len(block)] = np.linalg.norm(block - centers[chosen], axis=1)
    flags = distances <= basis["p95"][nearest]
    return nearest, distances, flags


def wilson(k: int, n: int):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [float(max(0, center-half)), float(min(1, center+half))]


def stats(flags):
    flags = np.asarray(flags, dtype=bool)
    k, n = int(flags.sum()), len(flags)
    return {"n": n, "n_in_basin": k, "in_basin_fraction": k/n if n else None, "wilson_95_ci": wilson(k, n)}


def load_id_json(path: Path):
    values = json.loads(path.read_text())
    if len(values) != len(set(map(str, values))):
        raise ValueError(f"Duplicate identifiers in {path}")
    return set(map(str, values))


def prior_four_way_ids(source, downstream: Path, external_run: Path, mechanism_run: Path):
    paths = [
        downstream / "external" / source / "feature_ids.json",
        external_run / "consistent_transform" / "external" / source / "feature_ids.json",
        external_run / "graphlet" / "external" / source / "feature_ids.json",
        mechanism_run / "voronoi_external" / "external" / source / "feature_ids.json",
    ]
    sets = [load_id_json(path) for path in paths]
    common = set.intersection(*sets)
    if len(common) != EXPECTED_PRIOR_FOUR_WAY[source]:
        raise ValueError(f"Prior four-way support changed for {source}: {len(common)}")
    return common, paths


def main():
    started = time.time()
    base = Path(os.environ.get("AMD_EXT_BASE", os.environ.get("WORK", "/path/to/tacc/work")))
    output = Path(os.environ["AMD_EXT_OUT"])
    output.mkdir(parents=True, exist_ok=False)
    stage = Path(os.environ.get("AMD_EXT_STAGE", str(base / "feature_repair_runs/external_representation_20260905T224322Z/stage_v2")))
    sys.path.insert(0, str(stage / "scripts"))
    from external_frozen_cohorts import load_cohort

    downstream = base / "feature_repair_runs/downstream_v2_20260905"
    external_run = base / "feature_repair_runs/external_representation_20260905T224322Z"
    mechanism_run = base / "feature_repair_runs/representation_mechanism_20260906"
    rep = downstream / "representations"
    raw_path = rep / "amd_features/features.npy"
    ids_path = rep / "amd_features/features.ids.json"
    assignments_path = rep / "amd-full-partition/amd/community_assignments.csv"
    scaler_path = rep / "amd-full-partition/amd_scaler.json"
    partition_path = rep / "amd-full-partition/amd/partition.json"

    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    ids = json.loads(ids_path.read_text())
    assignments = {int(r["icsd_id"]): int(r["community"]) for r in read_csv(assignments_path)}
    scaler = json.loads(scaler_path.read_text())
    partition = json.loads(partition_path.read_text())
    if raw.shape != (88739, 100) or len(ids) != len(raw) or len(set(ids)) != len(ids):
        raise ValueError("Unexpected AMD ICSD feature population")
    if set(ids) != set(assignments) or partition.get("n_entries") != len(ids):
        raise ValueError("AMD IDs and partition assignments disagree")
    if np.any(raw < 0) or np.any(np.diff(raw, axis=1) < -1e-6) or not np.isfinite(raw).all():
        raise ValueError("Invalid ICSD AMD matrix")
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    if mean.shape != (100,) or scale.shape != (100,) or np.any(scale <= 0) or scaler["n_samples"] != len(ids):
        raise ValueError("Invalid saved AMD scaler")
    coordinates = (np.asarray(raw, dtype=np.float64) - mean) / scale
    labels = np.asarray([assignments[i] for i in ids], dtype=np.int64)
    basis = community_basis(coordinates, labels)
    if len(basis["communities"]) != 3598 or basis["counts"].sum() != len(ids):
        raise ValueError("Rebuilt AMD basis disagrees with partition")
    np.savez_compressed(output / "amd_basis.npz", **basis, scaler_mean=mean, scaler_scale=scale)

    i_nearest, i_distance, i_flags = classify(coordinates, basis)
    icsd_projection = [{
        "record_key": str(iid),
        "assigned_community": int(basis["communities"][j]),
        "nearest_centroid_distance": float(distance),
        "community_threshold_p95": float(basis["p95"][j]),
        "in_basin": bool(flag),
    } for iid, j, distance, flag in zip(ids, i_nearest, i_distance, i_flags)]
    write_csv(output / "icsd_projection_full.csv", icsd_projection)
    full_icsd = stats(i_flags)

    # Prior four-way ICSD support is the exact intersection of the four saved projection tables.
    prior_icsd_paths = [
        external_run / "consistent_transform/basis/icsd_full.csv",
        external_run / "graphlet/basis/icsd_full.csv",
        mechanism_run / "voronoi_external/basis/icsd_full.csv",
    ]
    prior_icsd_sets = []
    for path in prior_icsd_paths:
        rows = read_csv(path)
        keys = {r["record_key"] for r in rows}
        if len(keys) != len(rows):
            raise ValueError(f"Duplicate ICSD keys in {path}")
        prior_icsd_sets.append(keys)
    prior_icsd_common = set.intersection(*prior_icsd_sets)
    if len(prior_icsd_common) != EXPECTED_PRIOR_FOUR_WAY["icsd"]:
        raise ValueError(f"Prior ICSD support changed: {len(prior_icsd_common)}")
    amd_index = {str(iid): n for n, iid in enumerate(ids)}
    five_icsd_ids = sorted(prior_icsd_common & set(amd_index))
    five_icsd_flags = np.asarray([i_flags[amd_index[key]] for key in five_icsd_ids])
    common_icsd = stats(five_icsd_flags)

    source_paths = {
        "gnome": base / "reference_data/gnome_data",
        "mattergen": base / "mattergen/data-release/cifs.zip",
        "mp": base / "reference_data/mp_theoretical_candidates_20260427.jsonl",
        "jarvis": base / "reference_data/jarvis_dft/jdft_3d-12-12-2022.json",
        "alexandria": base / "reference_data/alexandria_pbe_2025_07_02",
    }
    source_reports = {}
    long_rows = []
    workers = int(os.environ.get("AMD_EXT_WORKERS", "48"))
    for source in SOURCES:
        source_out = output / source
        source_out.mkdir()
        historical = downstream / "external" / source / f"{SOURCE_SLUG[source]}_frontier_records.csv"
        attempted = downstream / "external" / source / "attempted_cohort.csv"
        selected, old_keys, source_files, cohort = load_cohort(source, source_paths[source], historical)
        keys = [record_key(r) for r in selected]
        expected_attempted = [record_key(r) for r in read_csv(attempted)]
        if keys != expected_attempted:
            raise ValueError(f"Frozen attempted order differs for {source}")
        write_csv(source_out / "attempted_cohort.csv", [public_fields(r) for r in selected])
        dump(source_out / "attempted_ids.json", keys)
        print(f"AMD/{source}: {len(selected)} frozen attempts", flush=True)
        initargs = (source, str(source_paths[source]))
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=initargs) as pool:
            results = []
            for n, result in enumerate(pool.map(featurize_amd, selected, chunksize=4), 1):
                results.append(result)
                if n % 250 == 0 or n == len(selected):
                    print(f"AMD/{source}: computed {n}/{len(selected)}", flush=True)
        successful = [r for r in results if r[2] is None]
        failures = [{**record, **error} for record, vector, error in results if error is not None]
        if len(successful) + len(failures) != len(selected) or not successful:
            raise ValueError(f"Incomplete AMD accounting for {source}")
        success_keys = [record_key(r[0]) for r in successful]
        if len(success_keys) != len(set(success_keys)):
            raise ValueError(f"Duplicate AMD successes for {source}")
        ext_raw = np.stack([r[1] for r in successful])
        ext_coordinates = (ext_raw.astype(np.float64) - mean) / scale
        nearest, distances, flags = classify(ext_coordinates, basis)
        np.save(source_out / "amd_features.npy", ext_raw)
        dump(source_out / "feature_ids.json", success_keys)
        dump(source_out / "failures.json", failures)
        write_csv(source_out / "successful_records.csv", [r[0] for r in successful])
        projection = [{
            "record_key": key,
            "assigned_community": int(basis["communities"][j]),
            "nearest_centroid_distance": float(distance),
            "community_threshold_p95": float(basis["p95"][j]),
            "in_basin": bool(flag),
        } for key, j, distance, flag in zip(success_keys, nearest, distances, flags)]
        write_csv(source_out / "projection_full.csv", projection)
        success_index = {key: n for n, key in enumerate(success_keys)}
        prior_common, prior_paths = prior_four_way_ids(source, downstream, external_run, mechanism_run)
        five_ids = sorted(prior_common & set(success_index))
        common_flags = np.asarray([flags[success_index[key]] for key in five_ids])
        own = stats(flags)
        common = stats(common_flags)
        own_gap = 100 * (full_icsd["in_basin_fraction"] - own["in_basin_fraction"])
        common_gap = 100 * (common_icsd["in_basin_fraction"] - common["in_basin_fraction"])
        report = {
            "source": source,
            "cohort": cohort,
            "n_attempted": len(selected),
            "n_successful": len(successful),
            "n_failures": len(failures),
            "failure_stage_counts": {},
            "full_amd_support": {"icsd": full_icsd, "external": own, "icsd_minus_external_percentage_points": own_gap},
            "five_representation_common_support": {
                "prior_four_way_n": len(prior_common),
                "n": len(five_ids),
                "ids_sha256": ids_sha256(five_ids),
                "icsd": common_icsd,
                "external": common,
                "icsd_minus_external_percentage_points": common_gap,
            },
            "all_attempt_bounds": {
                "lower": own["n_in_basin"] / len(selected),
                "upper": (own["n_in_basin"] + len(failures)) / len(selected),
            },
            "source_files": [{"path": str(path), "sha256": sha256(path)} for path in source_files],
            "prior_four_way_paths": [str(path) for path in prior_paths],
        }
        for failure in failures:
            token = f"{failure['stage']}:{failure['exception_type']}"
            report["failure_stage_counts"][token] = report["failure_stage_counts"].get(token, 0) + 1
        dump(source_out / "summary.json", report)
        source_reports[source] = report
        long_rows.append({
            "population": source,
            "n_attempted": len(selected),
            "n_amd_successful": len(successful),
            "n_amd_failures": len(failures),
            "amd_supported_in_basin_fraction": own["in_basin_fraction"],
            "amd_supported_icsd_fraction": full_icsd["in_basin_fraction"],
            "amd_supported_icsd_minus_external_pp": own_gap,
            "n_five_representation_common": len(five_ids),
            "common_in_basin_fraction": common["in_basin_fraction"],
            "common_icsd_fraction": common_icsd["in_basin_fraction"],
            "common_icsd_minus_external_pp": common_gap,
            "all_attempt_lower_bound": report["all_attempt_bounds"]["lower"],
            "all_attempt_upper_bound": report["all_attempt_bounds"]["upper"],
        })
        print(json.dumps(long_rows[-1], allow_nan=False), flush=True)

    write_csv(output / "comparison.csv", long_rows)
    provenance = {
        "status": "complete",
        "scope": "Scratch-only AMD-100 projection of the exact frozen five external cohorts into the existing standardized full-record ICSD AMD partition, with nearest-centroid/per-community member p95 classification.",
        "semantics": "Full-map calibration sensitivity. ICSD helped fit the AMD scaler, partition, centroids and radii. This is not independently cutoff-trained validation.",
        "amd_protocol": {
            "feature": "average-minimum-distance AMD-100 from amd.io.periodicset_from_pymatgen_structure",
            "coordinate_transform": "saved full-ICSD StandardScaler mean and scale",
            "partition": "existing AMD standardized-Euclidean mutual-16NN Louvain partition",
            "basin_rule": "nearest arithmetic centroid; assigned community member-distance numpy p95; distance <= p95",
        },
        "icsd_full": full_icsd,
        "icsd_five_representation_common": {**common_icsd, "n_prior_four_way": len(prior_icsd_common), "ids_sha256": ids_sha256(five_icsd_ids)},
        "sources": source_reports,
        "comparison_rows": long_rows,
        "input_files": {str(path): sha256(path) for path in (raw_path, ids_path, assignments_path, scaler_path, partition_path)},
        "prior_icsd_paths": [str(path) for path in prior_icsd_paths],
        "script_sha256": sha256(Path(__file__)),
        "runtime_seconds": time.time() - started,
        "checks": {
            "frozen_attempted_orders_reproduced": True,
            "prior_four_way_counts_reproduced": True,
            "amd_scaler_and_partition_population_aligned": True,
            "feature_vectors_finite_nondecreasing_nonnegative": True,
            "projection_flags_equal_distance_le_member_p95": True,
        },
    }
    dump(output / "summary.json", provenance)
    print(json.dumps({"status": "complete", "runtime_seconds": provenance["runtime_seconds"], "rows": long_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
