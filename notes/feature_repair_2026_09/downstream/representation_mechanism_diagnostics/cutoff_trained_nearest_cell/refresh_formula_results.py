#!/usr/bin/env python3
"""Refresh formula-dependent nearest-cell results from a corrected cutoff package.

The nearest-cell calibration itself depends only on the frozen structural maps,
training coordinates, and saved distances.  This utility first requires exact
row-for-row agreement in every structural field between a final cutoff package
and the retrieved calibration outputs.  It then replaces only formula metadata
and precedent flags, reconstructs the per-formula sampling unit, and regenerates
all quadrant and enrichment blocks.  Calibration thresholds and structural
in-basin flags are never recomputed or changed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from run_cutoff_nearest_cell import paired_enrichment_bootstrap, quadrant


FORMULA_COLUMNS = (
    "formula",
    "formula_identity",
    "stratum_coarse",
    "stratum_anon",
    "formula_match_all_year_le_T",
    "formula_match_post1980_le_T",
    "formula_match_all_year_le_T_index",
    "formula_match_post1980_le_T_index",
)
STRUCTURAL_COLUMNS = (
    "assigned_community",
    "nearest_centroid_distance",
    "community_threshold_p95",
    "in_basin",
)
CALIBRATION_COLUMNS = (
    "nearest_cell_threshold_p95",
    "nearest_cell_calibrated",
    "recalibrated_in_basin",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_same_rows(final: pd.DataFrame, calibrated: pd.DataFrame, keys: tuple[str, ...]) -> None:
    if len(final) != len(calibrated):
        raise AssertionError(f"row-count mismatch: {len(final)} != {len(calibrated)}")
    if not np.array_equal(
        final[list(keys)].astype(str).to_numpy(), calibrated[list(keys)].astype(str).to_numpy()
    ):
        raise AssertionError(f"row order/key mismatch for {keys}")
    for column in STRUCTURAL_COLUMNS:
        left, right = final[column], calibrated[column]
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(left.to_numpy(), right.to_numpy(), rtol=0, atol=1e-12, equal_nan=True):
                raise AssertionError(f"structural field changed: {column}")
        elif not left.astype(str).equals(right.astype(str)):
            raise AssertionError(f"structural field changed: {column}")


def refreshed_table(final: pd.DataFrame, calibrated: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    assert_same_rows(final, calibrated, keys)
    if not final.formula_identity.ne("").all():
        raise AssertionError("final cutoff package contains an empty formula identity")
    out = final.copy()
    for column in CALIBRATION_COLUMNS:
        out[column] = calibrated[column].to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[1990, 2000, 2010])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    started = time.time()
    results = args.diagnostic_root / "results"
    summary_path = args.cutoff_root / "cutoff_trained_retrospective_summary.json"
    summary = json.loads(summary_path.read_text())
    old_joint_path = results / "quadrants_and_enrichment.json"
    old_joint_sha = sha256(old_joint_path)
    structural_checks: dict[str, object] = {}
    joint: list[dict[str, object]] = []

    for cutoff in args.cutoffs:
        saved = summary["cutoffs"][str(cutoff)]
        final_held = pd.read_csv(args.cutoff_root / f"T{cutoff}/heldout_entries.csv", keep_default_na=False)
        final_external = pd.read_csv(
            args.cutoff_root / f"T{cutoff}/external_classifications.csv", keep_default_na=False
        )
        cutoff_audit: dict[str, object] = {
            "heldout_rows": len(final_held),
            "external_rows": len(final_external),
            "formula_identities": int(final_held.formula_identity.nunique()),
            "policies": {},
        }
        for policy in ("all_training", "original_nonnoise_only"):
            leaf = results / f"T{cutoff}/{policy}"
            calibrated_held = pd.read_csv(leaf / "heldout_entries.csv", keep_default_na=False)
            calibrated_external = pd.read_csv(
                leaf / "external_classifications.csv", keep_default_na=False
            )
            held = refreshed_table(final_held, calibrated_held, ("icsd_id",))
            external = refreshed_table(
                final_external, calibrated_external, ("source", "material_id")
            )
            held.to_csv(leaf / "heldout_entries.csv", index=False)
            external.to_csv(leaf / "external_classifications.csv", index=False)
            cutoff_audit["policies"][policy] = {
                "heldout_structural_rows_equal": True,
                "external_structural_rows_equal": True,
                "heldout_output_sha256": sha256(leaf / "heldout_entries.csv"),
                "external_output_sha256": sha256(leaf / "external_classifications.csv"),
            }

            rng = np.random.default_rng(42 + cutoff + (0 if policy == "all_training" else 10000))
            parseable = held[held.formula != ""].copy()
            first_formula = (
                parseable.sort_values(["year", "icsd_id"], kind="stable")
                .drop_duplicates("formula_identity", keep="first")
            )
            for unit_name, unit in (("per_entry", parseable), ("per_formula", first_formula)):
                valid = unit[unit.nearest_cell_calibrated].copy()
                for reference in saved["analyses"]:
                    match = valid[f"formula_match_{reference}"].to_numpy(bool)
                    old = valid.in_basin.to_numpy(bool)
                    new = valid.recalibrated_in_basin.to_numpy(bool)
                    joint.append(
                        {
                            "cutoff": cutoff,
                            "policy": policy,
                            "unit": unit_name,
                            "formula_reference": reference,
                            "n_original_unit": len(unit),
                            "n_calibrated_unit": len(valid),
                            "original_same_support": quadrant(old, match),
                            "recalibrated": quadrant(new, match),
                            "bootstrap": paired_enrichment_bootstrap(
                                old, new, match, rng, args.n_boot
                            ),
                        }
                    )
            for source, unit in external.groupby("source", sort=True):
                valid = unit[unit.nearest_cell_calibrated]
                for reference in saved["analyses"]:
                    match = valid[f"formula_match_{reference}"].to_numpy(bool)
                    joint.append(
                        {
                            "cutoff": cutoff,
                            "policy": policy,
                            "unit": "external_" + source,
                            "formula_reference": reference,
                            "n_original_unit": len(unit),
                            "n_calibrated_unit": len(valid),
                            "original_same_support": quadrant(valid.in_basin, match),
                            "recalibrated": quadrant(valid.recalibrated_in_basin, match),
                        }
                    )
        structural_checks[str(cutoff)] = cutoff_audit

    old_report_path = results / "report.json"
    report = json.loads(old_report_path.read_text())
    old_report_sha = sha256(old_report_path)
    old_script_sha = report["script_sha256"]
    old_joint_path.write_text(json.dumps(joint, indent=2, allow_nan=False) + "\n")
    report["protocol"]["formula_policy"] = (
        "Formula-precedent flags come from the final occupancy-aware cutoff package. "
        "Per-formula units use one element-sorted normalized atomic-fraction identity "
        "rounded to 12 decimals for every heldout record and retain the earliest "
        "postcutoff record, selected before exclusion."
    )
    report["formula_refresh"] = {
        "status": "complete",
        "scope": (
            "Formula metadata, precedent flags, per-formula selection, quadrants and "
            "enrichment were refreshed. Nearest-cell thresholds, assignments, distances, "
            "structural in-basin flags and rates were retained after exact equality checks."
        ),
        "cutoff_package": str(args.cutoff_root),
        "cutoff_summary_sha256": sha256(summary_path),
        "cutoff_reporting_manifest_sha256": sha256(
            args.cutoff_root / "reporting/reporting_manifest.json"
        ),
        "refresh_script_sha256": sha256(Path(__file__)),
        "structural_producer_script_sha256": old_script_sha,
        "pre_refresh_report_sha256": old_report_sha,
        "pre_refresh_quadrants_sha256": old_joint_sha,
        "formula_identity": "element-sorted normalized atomic fractions rounded to 12 decimal places",
        "n_boot": args.n_boot,
        "structural_equality": structural_checks,
        "runtime_seconds": time.time() - started,
    }
    old_report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "quadrant_blocks": len(joint),
                "cutoffs": args.cutoffs,
                "report": str(old_report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
