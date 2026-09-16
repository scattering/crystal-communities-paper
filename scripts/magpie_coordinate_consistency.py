#!/usr/bin/env python3
"""Recompute Magpie basins with one frozen PCA transform for every population.

The original independent Magpie partition remains fixed. Both ICSD and external
rows use ((X - scaler_mean) / scaler_scale - pca_mean) @ components.T. No PCA,
graph, partition, feature extraction, or source selection is fitted again.
Original saved-score results remain in their source directory. A separate run
reports changed assignments/rates and displacement relative to original p95.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from external_representation_sensitivity import (
    SOURCES, aligned_reference, community_basis, csv_rows, dump, projection_rows,
    sha, summarize, write_csv,
)


def transform_rows(raw, transform, chunk_size=1024):
    """Apply the frozen external transform in bounded-memory row chunks."""
    width = len(transform["scaler_mean"])
    components = transform["pca_components"]
    if (raw.ndim != 2 or raw.shape[1] != width or components.ndim != 2 or components.shape[1] != width
            or any(np.shape(transform[k]) != (width,) for k in ("scaler_mean", "scaler_scale", "pca_mean"))
            or np.any(np.asarray(transform["scaler_scale"]) <= 0)
            or not all(np.isfinite(value).all() for value in transform.values())):
        raise ValueError("Invalid frozen PCA transform")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    output = np.empty((len(raw), len(components)), dtype=np.float64)
    for start in range(0, len(raw), chunk_size):
        values = np.asarray(raw[start:start + chunk_size])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite ICSD features")
        standardized = (values - transform["scaler_mean"]) / transform["scaler_scale"]
        output[start:start + len(values)] = (standardized - transform["pca_mean"]) @ components.T
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite projected ICSD coordinates")
    return output


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic values")
    if not len(values):
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "maximum": None}
    return {"n": len(values), "mean": float(values.mean()),
            **{key: float(np.quantile(values, q)) for key, q in [("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("maximum", 1.)]}}


def displacement_report(original, consistent, labels, original_basis):
    if original.shape != consistent.shape or len(labels) != len(original):
        raise ValueError("Displacement identity/dimension mismatch")
    displacement = np.linalg.norm(consistent - original, axis=1)
    radii = dict(zip(map(int, original_basis["communities"]), map(float, original_basis["p95"])))
    member_mask = np.asarray(labels) >= 0
    member_radii = np.array([radii[int(label)] for label in np.asarray(labels)[member_mask]])
    member_displacement = displacement[member_mask]
    positive = member_radii > 0
    zero = member_radii == 0
    return {"euclidean_coordinate_displacement_all_map_rows": quantiles(displacement),
            "euclidean_coordinate_displacement_nonnoise_members": quantiles(member_displacement),
            "displacement_divided_by_original_own_community_p95_positive_radius": quantiles(member_displacement[positive] / member_radii[positive]),
            "n_nonnoise_zero_radius": int(zero.sum()),
            "n_zero_radius_members_with_nonzero_displacement": int(np.sum(zero & (member_displacement > 0))),
            "maximum_component_absolute_displacement": float(np.max(np.abs(consistent - original))),
            "ratio_definition": "Each nonnoise ICSD member's Euclidean coordinate displacement divided by its original partition community's saved-score full-map p95. Zero radii are reported separately; noise has no own-community radius."}


def as_boolean(value):
    if value in (True, "True"):
        return True
    if value in (False, "False"):
        return False
    raise ValueError("Invalid projection boolean")


def paired_changes(original_rows, consistent_rows):
    """Compare paired decisions only after checking exact identities and order."""
    if [r["record_key"] for r in original_rows] != [r["record_key"] for r in consistent_rows]:
        raise ValueError("Projection identities or order changed")
    before = np.array([as_boolean(r["in_basin"]) for r in original_rows], dtype=bool)
    after = np.array([as_boolean(r["in_basin"]) for r in consistent_rows], dtype=bool)
    original_community = np.array([int(r["assigned_community"]) for r in original_rows])
    new_community = np.array([int(r["assigned_community"]) for r in consistent_rows])
    n = len(before)
    return {"n": n, "n_original_in_basin": int(before.sum()), "n_consistent_in_basin": int(after.sum()),
            "original_in_basin_fraction": float(before.mean()) if n else None,
            "consistent_in_basin_fraction": float(after.mean()) if n else None,
            "rate_change_percentage_points": float(100 * (after.mean() - before.mean())) if n else None,
            "n_membership_flags_changed": int(np.sum(before != after)),
            "n_in_basin_to_frontier": int(np.sum(before & ~after)),
            "n_frontier_to_in_basin": int(np.sum(~before & after)),
            "n_nearest_community_changed": int(np.sum(original_community != new_community))}


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Completed original Magpie external-representation run")
    parser.add_argument("--out-dir", required=True, type=Path, help="Separate sibling output, e.g. consistent_transform")
    parser.add_argument("--raw-features", type=Path)
    parser.add_argument("--saved-pca", type=Path)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.out_dir.resolve() == args.run_dir.resolve() or args.run_dir.resolve() in args.out_dir.resolve().parents:
        raise ValueError("Consistent-transform output must be separate from the original run")
    original_manifest_path = args.run_dir / "basis/basis_manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text())
    if original_manifest["kind"] != "magpie":
        raise ValueError("This sensitivity applies to the Magpie PCA map")
    # Require all original external results before creating an apparent partial run.
    original_external = {}
    for source in SOURCES:
        source_dir = args.run_dir / "external" / source
        original_external[source] = json.loads((source_dir / "summary.json").read_text())
        if original_external[source]["basis_manifest_sha256"] != sha(original_manifest_path):
            raise ValueError(f"Original external basis provenance differs: {source}")
        if not (source_dir / "coordinates.npy").is_file():
            raise FileNotFoundError(source_dir / "coordinates.npy")
    files = original_manifest["reference_files"]
    resolved = {key: override or Path(files[key]["path"]) for key, override in
                [("features", args.raw_features), ("pca", args.saved_pca), ("ids", args.ids), ("assignments", args.assignments)]}
    for key, path in resolved.items():
        if sha(path) != files[key]["sha256"]:
            raise ValueError(f"Frozen reference file hash mismatch: {key}")
    transform_path = args.run_dir / "basis/transform.npz"
    transform = load_npz(transform_path)
    raw = np.load(resolved["features"], mmap_mode="r", allow_pickle=False)
    saved = np.load(resolved["pca"], mmap_mode="r", allow_pickle=False)
    ids_path = resolved["ids"]
    feature_ids = [int(r["icsd_id"]) for r in csv_rows(ids_path)] if ids_path.suffix == ".csv" else json.loads(ids_path.read_text())
    indices, ids, years, labels = aligned_reference(feature_ids, csv_rows(resolved["assignments"]))
    if len(raw) != len(feature_ids) or len(saved) != len(feature_ids):
        raise ValueError("Raw/PCA/ID alignment mismatch")
    print(f"Applying frozen external transform to {len(raw)} Magpie ICSD rows", flush=True)
    coordinates = transform_rows(raw, transform, args.chunk_size)[indices]
    old_coordinates = np.asarray(saved[indices])
    old_full_basis = load_npz(args.run_dir / "basis/basis_full.npz")
    diagnostics = displacement_report(old_coordinates, coordinates, labels, old_full_basis)
    print(json.dumps({"coordinate_displacement": diagnostics}), flush=True)
    basis_dir = args.out_dir / "basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transform_path, basis_dir / "transform.npz")
    np.save(basis_dir / "reference_coordinates.npy", coordinates)
    dump(basis_dir / "reference_ids.json", ids.tolist())
    reports, changes, community_changes = {}, [], []
    basis_by_key = {}
    for key, original_rate in original_manifest["reference_rates"].items():
        cutoff = original_rate["cutoff"]
        training = np.ones(len(ids), dtype=bool) if cutoff is None else (years >= 0) & (years <= cutoff)
        evaluation = np.ones(len(ids), dtype=bool) if cutoff is None else years > cutoff
        consistent_basis = community_basis(coordinates, labels, training)
        original_basis = load_npz(args.run_dir / "basis" / f"basis_{key}.npz")
        if not np.array_equal(original_basis["communities"], consistent_basis["communities"]) or not np.array_equal(original_basis["counts"], consistent_basis["counts"]):
            raise ValueError("Coordinate sensitivity changed partition membership/training counts")
        basis_by_key[key] = consistent_basis
        np.savez_compressed(basis_dir / f"basis_{key}.npz", **consistent_basis)
        rows, stats = projection_rows(ids[evaluation], coordinates[evaluation], consistent_basis)
        write_csv(basis_dir / f"icsd_{key}.csv", rows)
        original_rows = csv_rows(args.run_dir / "basis" / f"icsd_{key}.csv")
        changes.append({"population": "ICSD", "map": key, **paired_changes(original_rows, rows)})
        reports[key] = {**original_rate, **stats}
        center_displacement = np.linalg.norm(consistent_basis["centroids"] - original_basis["centroids"], axis=1)
        radius_difference = consistent_basis["p95"] - original_basis["p95"]
        for c, n, center_shift, before, after in zip(consistent_basis["communities"], consistent_basis["counts"], center_displacement, original_basis["p95"], consistent_basis["p95"]):
            community_changes.append({"map": key, "community": int(c), "n_training_members": int(n),
                                      "centroid_displacement": float(center_shift), "original_p95": float(before),
                                      "consistent_p95": float(after), "p95_difference": float(after - before),
                                      "centroid_displacement_divided_by_original_p95": float(center_shift / before) if before > 0 else None})
        print(json.dumps(changes[-1]), flush=True)
    basis_manifest = {**original_manifest, "coordinate_variant": "consistent_transform",
                      "coordinate_rule": "ICSD and external coordinates use the same frozen scaler/PCA transform. The original saved-full-map partition is retained without refitting.",
                      "reference_rates": reports, "original_basis_manifest_sha256": sha(original_manifest_path),
                      "frozen_transform_sha256": sha(transform_path), "script_sha256": sha(__file__),
                      "displacement_diagnostics": diagnostics,
                      "scope": "Coordinate-consistency sensitivity of a fixed full-fitted representation and partition. Historical centers/radii use pre-cutoff members, but this is not cutoff-trained validation."}
    dump(basis_dir / "basis_manifest.json", basis_manifest)
    for source in SOURCES:
        source_dir = args.run_dir / "external" / source
        target_dir = args.out_dir / "external" / source
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("attempted_cohort.csv", "attempted_ids.json", "feature_ids.json", "failures.json", "successful_records.csv", "coordinates.npy"):
            shutil.copyfile(source_dir / filename, target_dir / filename)
        external_coordinates = np.load(target_dir / "coordinates.npy", mmap_mode="r", allow_pickle=False)
        external_ids = json.loads((target_dir / "feature_ids.json").read_text())
        if len(external_coordinates) != len(external_ids):
            raise ValueError("External coordinate/ID mismatch")
        original_summary = original_external[source]
        if len(external_ids) != original_summary["n_successful"]:
            raise ValueError("External success denominator changed")
        rates = {}
        for key, basis in basis_by_key.items():
            rows, stats = projection_rows(external_ids, external_coordinates, basis)
            write_csv(target_dir / f"projection_{key}.csv", rows)
            original_rows = csv_rows(source_dir / f"projection_{key}.csv")
            changes.append({"population": source, "map": key, **paired_changes(original_rows, rows)})
            ref = reports[key]
            rates[key] = {**stats, "icsd_reference": ref,
                          "icsd_minus_external_percentage_points": 100 * (ref["in_basin_fraction"] - stats["in_basin_fraction"]),
                          "external_fraction_all_attempts_lower_bound": stats["n_in_basin"] / original_summary["n_attempted"],
                          "external_fraction_all_attempts_upper_bound": (stats["n_in_basin"] + original_summary["n_failures"]) / original_summary["n_attempted"]}
            print(json.dumps(changes[-1]), flush=True)
        dump(target_dir / "summary.json", {**original_summary, "coordinate_variant": "consistent_transform", "rates": rates,
             "basis_manifest_sha256": sha(basis_dir / "basis_manifest.json"), "original_summary_sha256": sha(source_dir / "summary.json"),
             "external_coordinate_sha256": sha(source_dir / "coordinates.npy"),
             "feature_reuse": "Exact existing successful external coordinates and frozen identities; no external featurization or transform was repeated.", "script_sha256": sha(__file__)})
    write_csv(args.out_dir / "paired_membership_changes.csv", changes)
    write_csv(args.out_dir / "community_coordinate_changes.csv", community_changes)
    dump(args.out_dir / "coordinate_consistency.json", {"original_run": str(args.run_dir), "consistent_run": str(args.out_dir),
         "original_basis_manifest_sha256": sha(original_manifest_path), "frozen_transform_sha256": sha(transform_path),
         "coordinate_displacement": diagnostics, "paired_membership_changes": changes,
         "same_external_feature_successes_and_coordinates": True, "same_partition_and_training_member_counts": True,
         "script_sha256": sha(__file__), "decision_rule": "Report observed displacement and paired classification changes without imposing an arbitrary acceptability threshold."})
    summarize(argparse.Namespace(run_dir=args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
