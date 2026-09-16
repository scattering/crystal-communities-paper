#!/usr/bin/env python3
"""Audit saved public-cohort projections and compare common represented IDs.

Requires the small basis NPZ/manifest, projection CSV, identity, failure and
summary artifacts. If coordinates.npy is available, additionally verify 64
seed-42 external assignments against scipy.cdist. No source CIFs are required.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from external_frozen_cohorts import record_key
from external_representation_sensitivity import SOURCES, csv_rows, dump, rate, sha, write_csv


def boolean(value):
    if value not in ("True", "False", True, False):
        raise ValueError(f"Invalid boolean: {value}")
    return value in ("True", True)


def audit_projection(path, expected_keys, basis):
    rows = csv_rows(path)
    keys = [r["record_key"] for r in rows]
    if keys != list(map(str, expected_keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"Projection ID identity/order mismatch: {path}")
    radii = dict(zip(map(int, basis["communities"]), map(float, basis["p95"])))
    flags = []
    for row in rows:
        community = int(row["assigned_community"])
        distance, saved_radius = float(row["nearest_centroid_distance"]), float(row["community_threshold_p95"])
        if (community not in radii or not np.isfinite([distance, saved_radius]).all()
                or distance < 0 or saved_radius != radii[community]):
            raise ValueError(f"Invalid radius/distance/community: {path}")
        flag = boolean(row["in_basin"])
        if flag != (distance <= saved_radius):
            raise ValueError(f"In-basin flag disagrees with nearest-community rule: {path}")
        flags.append(flag)
    return rows, dict(zip(keys, flags)), rate(flags)


def validate_report(saved, actual, where):
    for key in ("n", "n_in_basin"):
        if saved[key] != actual[key]:
            raise ValueError(f"Saved summary count mismatch: {where}/{key}")
    if saved["in_basin_fraction"] != actual["in_basin_fraction"]:
        raise ValueError(f"Saved summary fraction mismatch: {where}")


def audit_run(run_dir, frozen_root):
    manifest_path = run_dir / "basis" / "basis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model, source_flags, reference_flags, checks = manifest["kind"], {}, {}, []
    basis_by_key = {}
    for key, report in manifest["reference_rates"].items():
        with np.load(run_dir / "basis" / f"basis_{key}.npz", allow_pickle=False) as saved:
            basis = dict(saved)
        basis_by_key[key] = basis
        ref_path = run_dir / "basis" / f"icsd_{key}.csv"
        rows = csv_rows(ref_path)
        _, flags, actual = audit_projection(ref_path, [r["record_key"] for r in rows], basis)
        validate_report(report, actual, f"{model}/ICSD/{key}")
        reference_flags[key] = flags
        checks.append({"check": f"{model}/ICSD/{key}/radii_flags_counts", "passed": True, "n": len(flags)})
    for source in SOURCES:
        directory = run_dir / "external" / source
        summary = json.loads((directory / "summary.json").read_text())
        if summary["kind"] != model or summary["source"] != source or summary["basis_manifest_sha256"] != sha(manifest_path):
            raise ValueError(f"Provenance mismatch: {model}/{source}")
        attempts = json.loads((directory / "attempted_ids.json").read_text())
        ids = json.loads((directory / "feature_ids.json").read_text())
        failures = json.loads((directory / "failures.json").read_text())
        failure_ids = [record_key(r) for r in failures]
        success_set, failure_set = set(ids), set(failure_ids)
        if (len(attempts) != len(set(attempts)) or len(ids) != len(set(ids)) or len(failure_ids) != len(set(failure_ids))
                or success_set & failure_set or success_set | failure_set != set(attempts)
                or [key for key in attempts if key in success_set] != ids):
            raise ValueError(f"Success/failure identity coverage mismatch: {model}/{source}")
        if len(attempts) != (386 if source == "mattergen" else 5000):
            raise ValueError(f"Wrong frozen attempt count: {model}/{source}")
        if (summary["n_attempted"], summary["n_successful"], summary["n_failures"]) != (len(attempts), len(ids), len(failures)):
            raise ValueError(f"Failure coverage count mismatch: {model}/{source}")
        for name in ("attempted_ids", "feature_ids"):
            if summary[name + "_sha256"] != sha(directory / f"{name}.json"):
                raise ValueError(f"Saved identity hash mismatch: {model}/{source}/{name}")
        if frozen_root:
            expected = [record_key(r) for r in csv_rows(frozen_root / source / "attempted_cohort.csv")]
            if attempts != expected:
                raise ValueError(f"Attempted identities differ from original frozen cohort: {model}/{source}")
        checks.append({"check": f"{model}/{source}/exact_attempt_identity_and_failure_coverage", "passed": True,
                       "n_attempted": len(attempts), "n_successful": len(ids), "n_failures": len(failures)})
        source_flags[source] = {}
        coordinates_path = directory / "coordinates.npy"
        coordinates = np.load(coordinates_path, mmap_mode="r", allow_pickle=False) if coordinates_path.exists() else None
        if coordinates is not None and (len(coordinates) != len(ids) or not np.isfinite(coordinates).all()):
            raise ValueError(f"Invalid saved coordinates: {model}/{source}")
        for key, basis in basis_by_key.items():
            rows, flags, actual = audit_projection(directory / f"projection_{key}.csv", ids, basis)
            validate_report(summary["rates"][key], actual, f"{model}/{source}/{key}")
            source_flags[source][key] = flags
            checks.append({"check": f"{model}/{source}/{key}/radii_flags_counts", "passed": True, "n": len(flags)})
            if coordinates is not None:
                selected = np.random.default_rng(42).choice(len(ids), min(64, len(ids)), replace=False)
                matrix = cdist(coordinates[selected], basis["centroids"], metric="euclidean")
                nearest = matrix.argmin(axis=1)
                for sample, chosen, distance in zip(selected, nearest, matrix[np.arange(len(selected)), nearest]):
                    row = rows[sample]
                    if int(row["assigned_community"]) != int(basis["communities"][chosen]) or not np.isclose(float(row["nearest_centroid_distance"]), distance, rtol=1e-11, atol=1e-11):
                        raise ValueError(f"Independent nearest-centroid audit failed: {model}/{source}/{key}/{ids[sample]}")
                checks.append({"check": f"{model}/{source}/{key}/independent_scipy_cdist", "passed": True, "n": len(selected)})
    return model, reference_flags, source_flags, checks


def common_support(references, external):
    """Same successful experimental/external ID denominators for each map pair."""
    rows = []
    for left, right in itertools.combinations(sorted(references), 2):
        for key in sorted(set(references[left]) & set(references[right])):
            common_ref = sorted(set(references[left][key]) & set(references[right][key]))
            for source in SOURCES:
                common_ext = sorted(set(external[left][source][key]) & set(external[right][source][key]))
                for model in (left, right):
                    ref_rate = rate([references[model][key][iid] for iid in common_ref])
                    ext_rate = rate([external[model][source][key][iid] for iid in common_ext])
                    gap = None if not common_ref or not common_ext else 100 * (ref_rate["in_basin_fraction"] - ext_rate["in_basin_fraction"])
                    rows.append({"pair": f"{left}/{right}", "representation": model, "map": key, "source": source,
                                 "common_icsd_n": len(common_ref), "common_external_n": len(common_ext),
                                 "icsd_in_basin_fraction": ref_rate["in_basin_fraction"],
                                 "external_in_basin_fraction": ext_rate["in_basin_fraction"],
                                 "icsd_minus_external_percentage_points": gap})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--frozen-external-root", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    checks, references, external = [], {}, {}
    for directory in args.run_dir:
        model, ref, ext, result = audit_run(directory, args.frozen_external_root)
        if model in references:
            raise ValueError("Repeated representation")
        references[model], external[model] = ref, ext
        checks.extend(result)
    matched = common_support(references, external)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "common_support_comparison.csv", matched)
    dump(args.out_dir / "verification.json", {"n_checks": len(checks), "n_passed": len(checks), "n_failed": 0,
         "checks": checks, "common_support": matched, "script_sha256": sha(__file__),
         "input_manifest_sha256": {str(p): sha(p / "basis/basis_manifest.json") for p in args.run_dir},
         "limitation": "Common successful-ID support controls representation coverage differences; it does not remove full-map fitting or source selection differences."})
    print(json.dumps({"n_checks": len(checks), "n_passed": len(checks), "n_common_support_rows": len(matched)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
