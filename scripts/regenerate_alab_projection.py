#!/usr/bin/env python3
"""Re-embed the original A-Lab targets using the verified repaired basis.

Preserves the existing loader's corrected outcomes and CIF selection. Reports
every missing/failed target, raw features, and projections; accessibility scores
must be computed separately from the repaired reference population/moments.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from analyze_alab_validation import load_alab_targets
from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS
from prepare_repaired_projection_basis import sha256_file
from regenerate_external_projection import featurize, initialize_worker, project_to_communities, write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("alab-zip", "basis", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    with np.load(args.basis, allow_pickle=False) as saved:
        basis = {name: saved[name] for name in saved.files if name != "metadata"}
        reference = json.loads(str(saved["metadata"]))
    hashes = {name: sha256_file(Path(__file__).parent / name)
              for name in ("icsd_densify_worker.py", "crystal_neighbors.py")}
    if (reference.get("feature_version") != FEATURE_VERSION or reference.get("neighbor_settings") != NEIGHBOR_SETTINGS
            or reference.get("worker_sha256") != hashes or not reference.get("fit_transform_reproduces_saved_pca")
            or not reference.get("row_ids_and_years_verified")):
        raise ValueError("A-Lab requires a verified basis matching the current repaired encoder")
    targets = load_alab_targets(args.alab_zip)
    if len(targets) != 57 or len({r.formula for r in targets}) != 57:
        raise ValueError("Expected the original 57 distinct A-Lab targets")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "alab_targets.csv", [asdict(r) for r in targets])
    initialize_worker("mattergen", str(args.alab_zip), reference["wl_iters"], None)
    successful, failures = [], []
    for target in targets:
        if not target.cif_member:
            failures.append({**asdict(target), "reason": "missing_cif"})
            continue
        record = {**asdict(target), "material_id": target.formula, "reduced_formula": None,
                  "zip_member": target.cif_member, "_kind": "mattergen"}
        result = featurize(record)
        if result[0]:
            successful.append(result)
        else:
            failures.append({**result[1], **result[3]})
    (args.out_dir / "alab_projection_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if not successful:
        raise ValueError("No A-Lab structures projected; all failures saved")
    vectors = np.vstack([r[2] for r in successful])
    projected, nearest, distances = project_to_communities(vectors, basis)
    rows = []
    for i, ((_, public, _, _), index, distance) in enumerate(zip(successful, nearest, distances)):
        birth = int(basis["birth_years"][index])
        rows.append({**public, "assigned_community": int(basis["communities"][index]),
                     "nearest_centroid_distance": float(distance),
                     "community_threshold_p95": float(basis["p95"][index]),
                     "community_median_distance": float(basis["p50"][index]),
                     "community_size": int(basis["counts"][index]),
                     "community_birth_year": birth if birth >= 0 else None,
                     "in_basin": bool(distance <= basis["p95"][index]),
                     "outlier_like": bool(distance > basis["p95"][index]), "feature_row": i})
    write_csv(args.out_dir / "alab_projection_records.csv", rows)
    np.save(args.out_dir / "features.npy", vectors)
    np.save(args.out_dir / "features_pca.npy", projected)
    (args.out_dir / "feature_ids.json").write_text(json.dumps([r["formula"] for r in rows]) + "\n")
    metadata = {"feature_version": FEATURE_VERSION, "neighbor_settings": dict(NEIGHBOR_SETTINGS),
                "worker_sha256": hashes, "wl_iters": reference["wl_iters"], "n_features": 213,
                "n_rows": len(rows), "basis_sha256": sha256_file(args.basis),
                "source_zip_sha256": sha256_file(args.alab_zip),
                "corrected_outcome_loader_sha256": sha256_file(Path(__file__).parent / "analyze_alab_validation.py")}
    (args.out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    info = [r[3] for r in successful]
    summary = {**metadata, "n_targets": len(targets), "n_with_cif": sum(bool(t.cif_member) for t in targets),
               "n_scored": len(rows), "n_failures": len(failures),
               "failure_reasons": dict(Counter(r["reason"] for r in failures)),
               "all_target_corrected_outcomes": dict(Counter(t.corrected_outcome for t in targets)),
               "scored_corrected_outcomes": dict(Counter(r["corrected_outcome"] for r in rows)),
               "n_scored_without_community_birth_year": sum(r["community_birth_year"] is None for r in rows),
               "feature_diagnostics": {"n_disordered": sum(not r["ordered"] for r in info),
                                       "n_structures_with_unrepresented_cn_mass": sum(r["sites_with_unrepresented_cn_mass"] > 0 for r in info),
                                       "max_unrepresented_cn_mass": max(r["max_unrepresented_cn_mass"] for r in info)},
               "scope": "Projection only; corrected outcomes and CIF selection come from the original loader. Accessibility requires repaired reference moments and an explicit observation year."}
    (args.out_dir / "alab_projection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_targets", "n_with_cif", "n_scored", "n_failures", "scored_corrected_outcomes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
