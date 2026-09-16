#!/usr/bin/env python3
"""Classify production ICSD with the external-sensitivity rule, without refitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def id_digest(ids):
    return hashlib.sha256("".join(f"{i}\n" for i in sorted(ids)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    repo = args.repository.resolve()
    sys.path.insert(0, str(repo / "scripts"))
    from external_representation_sensitivity import classify, rate
    down = repo / "notes/feature_repair_2026_09/downstream"
    full = repo / "notes/feature_repair_2026_09/full_run_results"
    paths = {
        "production_pca": down / "inputs/features_pca.npy",
        "production_feature_ids": full / "production/sample_assignments.csv",
        "production_partition": full / "production/graph/community_assignments.csv",
        "production_basis": down / "reference_basis/projection_basis.npz",
        "production_basis_provenance": down / "reference_basis/projection_basis_provenance.json",
        "graphlet_partition": down / "representations/graphlet-dated-replay/graphlet/community_assignments.csv",
        "magpie_feature_ids": full / "magpie/sample_assignments.csv",
        "gnome_projections": down / "external/gnome/gnome_frontier_records.csv",
        "gnome_summary": down / "external/gnome/gnome_frontier_summary.json",
        "gnome_feature_ids": down / "external/gnome/feature_ids.json",
        "gnome_attempted_cohort": down / "external/gnome/attempted_cohort.csv",
        "generic_classifier": repo / "scripts/external_representation_sensitivity.py",
    }
    provenance = json.loads(paths["production_basis_provenance"].read_text())
    for key, expected in (("production_pca", "icsd_pca"), ("production_feature_ids", "sample_assignments"), ("production_partition", "community_assignments")):
        assert sha(paths[key]) == provenance["reference_files"][expected]["sha256"], key
    assert provenance["feature_version"] == "crystal-features-v2-geometric-crystalnn"
    assert provenance["transform_vs_saved_max_abs_error"] == 0
    records = read_csv(paths["production_feature_ids"])
    ids = np.array([int(r["icsd_id"]) for r in records])
    assert len(ids) == len(set(ids)) == 167392
    coordinates = np.load(paths["production_pca"], mmap_mode="r", allow_pickle=False)
    assert coordinates.shape == (len(ids), 32)
    partition = {int(r["icsd_id"]): int(r["community"]) for r in read_csv(paths["production_partition"])}
    assert set(partition) == set(ids)
    with np.load(paths["production_basis"], allow_pickle=False) as saved:
        basis = {name: saved[name] for name in ("communities", "centroids", "p95", "counts")}
    nearest, distances, inside = classify(coordinates, basis)
    sample = np.random.default_rng(42).choice(len(ids), 256, replace=False)
    oracle = cdist(coordinates[sample], basis["centroids"], metric="euclidean")
    oracle_nearest = oracle.argmin(axis=1)
    assert np.array_equal(nearest[sample], oracle_nearest)
    assert np.allclose(distances[sample], oracle[np.arange(len(sample)), oracle_nearest], rtol=1e-11, atol=1e-11)
    graphlet_rows = read_csv(paths["graphlet_partition"])
    graphlet_ids = {int(r["icsd_id"]) for r in graphlet_rows}
    magpie_rows = read_csv(paths["magpie_feature_ids"])
    magpie_ids = {int(r["icsd_id"]) for r in magpie_rows}
    assert len(graphlet_ids) == len(graphlet_rows) == 150247
    assert len(magpie_ids) == len(magpie_rows) == 167448
    common_ids = set(ids) & graphlet_ids & magpie_ids
    common = np.isin(ids, list(common_ids))
    assert int(common.sum()) == len(common_ids)

    gnome = read_csv(paths["gnome_projections"])
    gnome_summary = json.loads(paths["gnome_summary"].read_text())
    gnome_keys = [r["material_id"] for r in gnome]
    assert len(gnome_keys) == len(set(gnome_keys)) == 5000
    assert gnome_keys == json.loads(paths["gnome_feature_ids"].read_text())
    assert gnome_keys == [r["material_id"] for r in read_csv(paths["gnome_attempted_cohort"])]
    assert gnome_summary["projection_basis"]["sha256"] == sha(paths["production_basis"])
    radii = dict(zip(map(int, basis["communities"]), basis["p95"]))
    gnome_flags = []
    for row in gnome:
        radius = float(radii[int(row["assigned_community"])])
        distance = float(row["nearest_centroid_distance"])
        assert np.isfinite(distance) and distance >= 0
        assert np.isclose(float(row["community_threshold_p95"]), radius, rtol=0, atol=1e-12)
        expected = distance <= radius
        assert row["in_basin"] in ("True", "False") and (row["in_basin"] == "True") == expected
        gnome_flags.append(expected)
    assert sum(gnome_flags) == gnome_summary["n_in_basin"] == 3102
    assert gnome_summary["in_basin_ratio"] == sum(gnome_flags) / 5000

    full_rate, common_rate, gnome_rate = rate(inside), rate(inside[common]), rate(gnome_flags)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "icsd_full.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_key", "year", "original_community", "assigned_community", "nearest_centroid_distance", "community_threshold_p95", "in_basin", "common_three_representations"])
        for row, j, distance, flag, matched in zip(records, nearest, distances, inside, common):
            writer.writerow([row["icsd_id"], row["year"], partition[int(row["icsd_id"])], int(basis["communities"][j]), float(distance), float(basis["p95"][j]), bool(flag), bool(matched)])
    report = {
        "feature_version": provenance["feature_version"],
        "scope": "Full-map production sensitivity control. Existing fitted production centroids and p95 radii are unchanged; only evaluation IDs are intersected across representations. No representation, graph, centroid or radius was refitted.",
        "classification_rule": "Same generic classify() as new Magpie and CrystalNN-graphlet experiments: nearest Euclidean centroid, then distance <= that community's member p95 radius. Noise never defines a basin, but original noise entries remain eligible evaluation rows.",
        "experimental_reference": {"all_production_successes": full_rate, "common_three_representations": common_rate,
            "graphlet_map_rows": len(graphlet_ids), "magpie_feature_rows": len(magpie_ids), "common_ids_sha256": id_digest(common_ids),
            "common_original_production_noise": sum(partition[i] < 0 for i in common_ids)},
        "GNoME": {**gnome_rate, "n_attempted": 5000, "n_failures": 0,
            "recorded_flags_and_radii_reverified": True, "same_production_basis_hash_verified": True,
            "external_coordinates_independently_reprojected": False,
            "external_verification_limit": "Local GNoME raw features and full 32-D coordinates are absent. Uses existing successful projection rows bound to this exact basis by their saved provenance hash; every per-community threshold and flag was rechecked."},
        "production_minus_GNoME_percentage_points": {
            "all_production_successes": 100 * (full_rate["in_basin_fraction"] - gnome_rate["in_basin_fraction"]),
            "common_three_representations": 100 * (common_rate["in_basin_fraction"] - gnome_rate["in_basin_fraction"])},
        "checks": {"reference_coordinates_and_ID_hashes": True, "all_production_IDs_unique_and_aligned": True,
            "independent_scipy_cdist_sample": len(sample), "gnome_exact_frozen_attempted_identity": True,
            "gnome_complete_success_coverage": True, "gnome_basis_hash_thresholds_and_flags": True},
        "input_files": {name: {"path": str(path.relative_to(repo)), "sha256": sha(path)} for name, path in paths.items()},
        "output_rows_sha256": sha(csv_path), "script_sha256": sha(__file__),
        "interpretation_limit": "Calibration/sensitivity using a full fitted experimental map, not independent historical validation. Common-ID support controls evaluation coverage differences; graphlet's available input population was selected from production nonnoise."
    }
    (args.out_dir / "production_full_map_control.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("experimental_reference", "GNoME", "production_minus_GNoME_percentage_points", "checks")}, indent=2))


if __name__ == "__main__":
    main()
