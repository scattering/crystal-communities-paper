#!/usr/bin/env python3
"""Audit saved production/Magpie matrices without refitting or modifying inputs.

Reads .npy arrays with memory mapping, scans bounded row blocks, and writes only
feature_matrix_audit.json to --out-dir. Exact all-zero pooled rows diagnose the
original failure; individual zero channels (especially variance/std) are valid.
These checks do not establish scientific robustness or per-site correctness.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def feature_groups(representation, summary, width):
    """Match the two workers' [chemistry || OPS61] ordering through WL/pooling."""
    magpie = representation == "magpie"
    chemistry, pools = (22, 2) if magpie else (7, 3)
    rounds = int(summary["wl_iters"])
    if rounds < 0 or summary["local_mode"] != ("paper_text" if magpie else "matminer_ops"):
        raise ValueError("Unsupported WL count or local mode for this audit")
    base = chemistry + 61
    local_width = pools * base * (3 ** rounds if magpie else 1)
    if width != local_width + 9:
        raise ValueError(f"Expected {local_width + 9} {representation} columns; found {width}")
    columns = np.arange(local_width)
    return {
        "all_features": slice(None), "pooled_local": slice(0, local_width),
        "pooled_chemistry": columns[columns % base < chemistry],
        "pooled_geometry": columns[columns % base >= chemistry],
        "global_lattice": slice(local_width, width),
    }


def scan_matrix(matrix, groups, chunk_rows):
    if matrix.ndim != 2 or matrix.dtype.kind not in "fiu":
        raise ValueError("Expected a two-dimensional real numeric feature matrix")
    result = {"shape": list(matrix.shape), "dtype": str(matrix.dtype),
              "n_nonfinite_values": 0, "n_rows_with_nonfinite_values": 0,
              "groups": {name: {"n_columns": len(range(*index.indices(matrix.shape[1])))
                                  if isinstance(index, slice) else len(index),
                                "n_exact_zero_rows": 0} for name, index in groups.items()}}
    for start in range(0, len(matrix), chunk_rows):
        block = matrix[start:start + chunk_rows]
        finite = np.isfinite(block)
        result["n_nonfinite_values"] += int(finite.size - finite.sum())
        result["n_rows_with_nonfinite_values"] += int(np.count_nonzero(~finite.all(axis=1)))
        for name, index in groups.items():
            result["groups"][name]["n_exact_zero_rows"] += int(np.count_nonzero((block[:, index] == 0).all(axis=1)))
    result["all_finite"] = result["n_nonfinite_values"] == 0
    return result


def assignment_rows(path, label_column):
    records, noise = [], 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            records.append((int(row["icsd_id"]), int(row["year"]) if row["year"].strip() else None))
            noise += int(int(row[label_column]) < 0)
    return records, {"n_rows": len(records), "n_noise": noise,
                     "unique_ids": len({i for i, _ in records}) == len(records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=("production", "magpie"), required=True)
    for name in ("features", "pca-features", "summary", "sample-assignments", "community-assignments", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    report = {"representation": args.representation,
              "scope": "Saved matrix checks only; no refitting, reclustering, per-site verification, or scientific robustness claim. Zero counts use exact equality; zero variance/std channels and zero PCA rows may be legitimate.",
              "sources": {name: str(getattr(args, name)) for name in
                          ("features", "pca_features", "summary", "sample_assignments", "community_assignments")},
              "chunk_rows": args.chunk_rows}
    try:
        summary = json.loads(args.summary.read_text())
        raw = np.load(args.features, mmap_mode="r", allow_pickle=False)
        pca = np.load(args.pca_features, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 2:
            raise ValueError("features.npy must be two-dimensional")
        groups = feature_groups(args.representation, summary, raw.shape[1])
        report["features"] = scan_matrix(raw, groups, args.chunk_rows)
        report["features_pca"] = scan_matrix(pca, {"all_features": slice(None)}, args.chunk_rows)
        sample_ids, sample = assignment_rows(args.sample_assignments, "cluster")
        graph_ids, graph = assignment_rows(args.community_assignments, "community")
        report["assignments"] = {"sample": sample, "community": graph}
        report["feature_version"] = summary.get("feature_version")
        report["neighbor_settings"] = {k: summary.get("neighbor_settings", {}).get(k) for k in
                                       ("weighted_cn", "cation_anion", "distance_cutoffs", "x_diff_weight", "porous_adjustment")}
        report["wl_iters"] = summary["wl_iters"]
        report["feature_layout"] = (
            "Two pools [mean, std]; each has 3**wl_iters repeated [Magpie22, OPS61] blocks; then 9 globals."
            if args.representation == "magpie" else
            "Three pools [mean, max, variance], each [chemistry7, OPS61]; then 9 globals.")
        diagnostics = summary.get("feature_diagnostics", {})
        report["feature_diagnostics"] = {k: diagnostics.get(k) for k in
                                         ("n_structures", "n_disordered", "n_structures_with_unrepresented_cn_mass",
                                          "n_sites_with_unrepresented_cn_mass", "max_unrepresented_cn_mass")}
        report["diagnostics_note"] = "Summary aggregates, not recomputed here. Unrepresented high-CN probability mass may coexist with a nonzero pooled OPS vector."
        expected_version = "crystal-features-v2-geometric-crystalnn"
        if args.representation == "magpie":
            expected_version += "-magpie22-concat-mean-std"
        n = len(raw)
        report["checks"] = {
            "repaired_feature_version": summary.get("feature_version") == expected_version,
            "nonempty_features": n > 0,
            "features_finite": report["features"]["all_finite"],
            "pca_finite": report["features_pca"]["all_finite"],
            "pca_row_count_matches_features": len(pca) == n,
            "summary_successful_count_matches_features": summary.get("sample_size_featurized") == n,
            "sample_label_count_matches_features": sample["n_rows"] == n,
            "community_label_count_matches_features": graph["n_rows"] == n,
            "sample_ids_unique": sample["unique_ids"], "community_ids_unique": graph["unique_ids"],
            "sample_community_ids_and_years_same_order": sample_ids == graph_ids,
            "summary_hdbscan_noise_matches_sample": summary.get("n_outliers") == sample["n_noise"],
            "diagnostic_structure_count_matches_features": diagnostics.get("n_structures") == n,
            **{name + "_has_no_zero_rows": report["features"]["groups"][name]["n_exact_zero_rows"] == 0
               for name in ("pooled_local", "pooled_chemistry", "pooled_geometry")},
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["error"] = str(exc)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "feature_matrix_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "checks": report.get("checks"), "error": report.get("error")}, indent=2))
    return int("error" in report or not all(report["checks"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
