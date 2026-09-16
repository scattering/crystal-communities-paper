#!/usr/bin/env python3
"""Inventory a repaired generation without including licensed scratch inputs.

Arrays, record tables over 1 MiB and other files over 5 MiB are listed by hash for archive retrieval,
but are not selected for git. No file contents are copied by this command.
"""
from __future__ import annotations

import os
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

VERSION = "crystal-features-v2-geometric-crystalnn"
BULK_LIMIT = 5 * 1024 * 1024
# Provenance roots recorded in the manifest. The producing runs lived under the
# TACC $WORK and project-archive roots; set these variables to reproduce the
# exact strings of an archived manifest off-site.
TACC_WORK = os.environ.get("CRYSTAL_COMMUNITIES_TACC_WORK", "$WORK")
TACC_ARCHIVE = os.environ.get("CRYSTAL_COMMUNITIES_TACC_ARCHIVE", "$PROJECT_ARCHIVE")
ALAB_MP_TARGET_ROOT = Path("alab_mp_targets")
ALAB_MP_TARGET_SHA256SUMS_SHA256 = (
    "916cc6b3abed7f8e077bb41960e06863673997130422209efdde4a639b91e703"
)
ALAB_MP_TARGET_VERIFICATION_SHA256 = (
    "79237aa46b7a9f735b12620a70c27aa41d5e67ff961c993e87a8dded0931534e"
)
ALAB_MP_TARGET_REQUIRED_FILES = {
    "alab_mp_targets/.gitignore",
    "alab_mp_targets/README.md",
    "alab_mp_targets/SHA256SUMS",
    "alab_mp_targets/code/check_primary_geometry_equivalence.py",
    "alab_mp_targets/code/fetch_mp_snapshot_targets.py",
    "alab_mp_targets/code/outcome_sensitivity.py",
    "alab_mp_targets/code/run_target_projection.py",
    "alab_mp_targets/code/verify_packaged_results.py",
    "alab_mp_targets/inputs/refined_features_pca.npy",
    "alab_mp_targets/output/failures.json",
    "alab_mp_targets/output/features.npy",
    "alab_mp_targets/output/features_pca.npy",
    "alab_mp_targets/output/target_projection_records.csv",
    "alab_mp_targets/output/target_projection_summary.json",
    "alab_mp_targets/output/target_vs_refinement_records.csv",
    "alab_mp_targets/provenance/original_extracted_structures_manifest.json",
    "alab_mp_targets/provenance/original_output_SHA256SUMS",
    "alab_mp_targets/provenance/sacct_job3471430.psv",
    "alab_mp_targets/provenance/slurm-3471430.err",
    "alab_mp_targets/provenance/slurm-3471430.out",
    "alab_mp_targets/provenance/tacc_run_job3471430.sbatch",
    "alab_mp_targets/source/alab_targets.csv",
    "alab_mp_targets/source/mp_2022_10_28_materials_source_manifest.json",
    "alab_mp_targets/source/mp_2022_10_28_materials_target_docs.json",
    "alab_mp_targets/source/mp_2022_10_28_summary_source_manifest.json",
    "alab_mp_targets/source/mp_2022_10_28_summary_target_docs.json",
    "alab_mp_targets/verification/independent_verification.json",
    "alab_mp_targets/verification/outcome_sensitivity.json",
    "alab_mp_targets/verification/package_verification.json",
    "alab_mp_targets/verification/primary_materials_geometry_equivalence.json",
}
CUTOFF_TRAINED_ROOT = Path("cutoff_trained")
# Canonical occupancy-aware package from independently verified job 3472690.
CUTOFF_TRAINED_SLURM_PATH: str | None = "cutoff_trained/slurm-3472690.out"
CUTOFF_TRAINED_SUMMARY_SHA256: str | None = (
    "2cbb8cf5c69b09766e75037734723d71d4db4f64df2490c25265e1bfc84d67d9"
)
CUTOFF_TRAINED_REPORTING_MANIFEST_SHA256: str | None = (
    "37f63dfeaf2d438f87043e31282c9352659be56d72213c57e26586690a5debb7"
)
CUTOFF_TRAINED_SLURM_SHA256: str | None = (
    "0f953d8ec1cb385cd7dc7b41bc63a470ded25ed3f9983f28977946ab221b99c3"
)
CUTOFF_TRAINED_INDEPENDENT_VERIFICATION_SHA256: str | None = (
    "09a5b32d7ac7f7bae2a0b49dc51df65e1c20567dcc5b3acc3b12acab83e47995"
)

CUTOFF_TRAINED_STATIC_REQUIRED_FILES = {
    *(f"cutoff_trained/T{cutoff}/{name}"
      for cutoff in (1990, 2000, 2010)
      for name in (
          "cutoff_map.npz",
          "external_classifications.csv",
          "heldout_entries.csv",
          "training_partition.csv",
      )),
    "cutoff_trained/cutoff_trained_retrospective_summary.json",
    "cutoff_trained/production_filter_audit_assignments.csv",
    "cutoff_trained/reporting/cutoff_trained_composition_summary.json",
    "cutoff_trained/reporting/cutoff_trained_joint.csv",
    "cutoff_trained/reporting/cutoff_trained_rates.csv",
    "cutoff_trained/reporting/cutoff_trained_shared_strata.csv",
    "cutoff_trained/reporting/reporting_manifest.json",
    "cutoff_trained/verification/independent_verification.json",
}
CUTOFF_TRAINED_REQUIRED_FILES = CUTOFF_TRAINED_STATIC_REQUIRED_FILES | (
    {CUTOFF_TRAINED_SLURM_PATH} if CUTOFF_TRAINED_SLURM_PATH else set()
)
CUTOFF_TRAINED_ARCHIVE_FILES = {
    *(f"cutoff_trained/T{cutoff}/{name}"
      for cutoff in (1990, 2000, 2010)
      for name in (
          "cutoff_map.npz",
          "external_classifications.csv",
          "heldout_entries.csv",
      )),
    "cutoff_trained/T2000/training_partition.csv",
    "cutoff_trained/T2010/training_partition.csv",
    "cutoff_trained/production_filter_audit_assignments.csv",
}
CUTOFF_TRAINED_SELECTED_FILES = (
    CUTOFF_TRAINED_REQUIRED_FILES - CUTOFF_TRAINED_ARCHIVE_FILES
)
CUTOFF_TRAINED_REQUIRED_SUMMARY_INPUTS = {
    "occupancy_flags",
    "formula_conventions",
    "index_loader",
    "wrapper",
}
CUTOFF_TRAINED_REQUIRED_CUTOFF_FIELDS = {
    "reference_composition_counts",
    "earliest_year_tie_audit",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_integrated_cutoff_trained_package(root: Path) -> tuple[dict, str]:
    """Reject legacy, partial or not-yet-pinned cutoff-trained packages."""
    summary_path = root / CUTOFF_TRAINED_ROOT / "cutoff_trained_retrospective_summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            "FINAL-OUTPUT INTEGRATION POINT: the occupancy-aware cutoff-trained "
            "summary has not been installed"
        )
    summary = json.loads(summary_path.read_text())
    expected_cutoffs = {"1990", "2000", "2010"}
    protocol = summary.get("protocol", {})
    required_protocol_fields = {
        "cutoffs",
        "max_year",
        "scaler",
        "pca",
        "pca_components",
        "knn_k",
        "louvain_resolution",
        "min_component_size",
        "min_community_size",
        "seed",
        "partition",
        "future_assignment",
        "partial_occupancy_fraction_tolerance",
        "formula_rule",
        "per_formula_identity",
        "per_formula_selection",
        "formula_references",
        "n_boot",
        "n_perm",
        "n_jobs",
    }
    required_software_fields = {
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "networkx",
        "scipy",
        "pymatgen",
    }
    cutoffs = summary.get("cutoffs", {})
    current_schema = (
        CUTOFF_TRAINED_REQUIRED_SUMMARY_INPUTS <= set(summary.get("inputs", {}))
        and required_protocol_fields <= set(protocol)
        and set(map(str, protocol.get("cutoffs", []))) == expected_cutoffs
        and required_software_fields <= set(summary.get("software", {}))
        and {"argv", "slurm_job_id", "hostname"}
        <= set(summary.get("execution", {}))
        and set(cutoffs) == expected_cutoffs
        and all(
            CUTOFF_TRAINED_REQUIRED_CUTOFF_FIELDS <= set(saved)
            for saved in cutoffs.values()
        )
    )
    if not current_schema:
        raise RuntimeError(
            "FINAL-OUTPUT INTEGRATION POINT: replace the legacy or partial "
            "cutoff-trained package with the completed occupancy-aware package"
        )

    canonical_hashes = {
        "summary": CUTOFF_TRAINED_SUMMARY_SHA256,
        "reporting_manifest": CUTOFF_TRAINED_REPORTING_MANIFEST_SHA256,
        "slurm_log": CUTOFF_TRAINED_SLURM_SHA256,
        "independent_verification": CUTOFF_TRAINED_INDEPENDENT_VERIFICATION_SHA256,
    }
    incomplete = [name for name, value in canonical_hashes.items() if value is None]
    malformed = [
        name
        for name, value in canonical_hashes.items()
        if value is not None
        and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        )
    ]
    if CUTOFF_TRAINED_SLURM_PATH is None or incomplete or malformed:
        raise RuntimeError(
            "FINAL-OUTPUT INTEGRATION POINT: set the final cutoff-trained Slurm "
            "path and canonical hashes after independent verification; "
            f"unset_hashes={incomplete}, malformed_hashes={malformed}"
        )
    slurm_job_id = str(summary["execution"]["slurm_job_id"])
    expected_slurm_path = f"cutoff_trained/slurm-{slurm_job_id}.out"
    if CUTOFF_TRAINED_SLURM_PATH != expected_slurm_path:
        raise RuntimeError(
            "Configured cutoff-trained Slurm path does not match the producer's "
            f"recorded job: configured={CUTOFF_TRAINED_SLURM_PATH}, "
            f"expected={expected_slurm_path}"
        )
    return summary, CUTOFF_TRAINED_SLURM_PATH


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("notes/feature_repair_2026_09/downstream"))
    p.add_argument("--write-gitignore", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    basis = json.loads((root / "reference_basis/projection_basis_provenance.json").read_text())
    if basis["feature_version"] != VERSION:
        raise ValueError("Projection basis belongs to a different feature generation")
    records = []
    bulk = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        packaged_alab_input = rel == ALAB_MP_TARGET_ROOT / "inputs/refined_features_pca.npy"
        if (any(part in {"inputs", "cache", "__pycache__", ".pytest_cache"}
                for part in rel.parts) and not packaged_alab_input):
            continue
        if str(rel) in {"manifest.json", ".gitignore"} or path.suffix in {".pyc", ".log", ".tmp"}:
            continue
        size = path.stat().st_size
        packaged_alab_file = rel.parts and rel.parts[0] == str(ALAB_MP_TARGET_ROOT)
        is_bulk = not packaged_alab_file and (
            size > BULK_LIMIT or path.suffix in {".npy", ".npz", ".gz"}
            or (path.suffix in {".csv", ".json"} and size > 1024 * 1024)
        )
        category = "derived_archive" if is_bulk else "git_selected"
        if category == "derived_archive":
            bulk.append(str(rel))
        records.append({"path": str(rel), "bytes": size, "sha256": digest(path), "storage": category})
    recorded_paths = {record["path"] for record in records}
    missing_alab_files = sorted(ALAB_MP_TARGET_REQUIRED_FILES - recorded_paths)
    if missing_alab_files:
        raise FileNotFoundError(
            "Incomplete packaged A-Lab target validation: " + ", ".join(missing_alab_files)
        )
    if digest(root / ALAB_MP_TARGET_ROOT / "SHA256SUMS") != ALAB_MP_TARGET_SHA256SUMS_SHA256:
        raise ValueError("Packaged A-Lab SHA256SUMS is not the canonical release manifest")
    if (digest(root / ALAB_MP_TARGET_ROOT / "verification/package_verification.json")
            != ALAB_MP_TARGET_VERIFICATION_SHA256):
        raise ValueError("Packaged A-Lab verification report is not the canonical release report")
    _, cutoff_slurm_path = require_integrated_cutoff_trained_package(root)
    cutoff_records = {
        record["path"]: record
        for record in records
        if Path(record["path"]).parts
        and Path(record["path"]).parts[0] == str(CUTOFF_TRAINED_ROOT)
    }
    if set(cutoff_records) != CUTOFF_TRAINED_REQUIRED_FILES:
        missing = sorted(CUTOFF_TRAINED_REQUIRED_FILES - set(cutoff_records))
        unexpected = sorted(set(cutoff_records) - CUTOFF_TRAINED_REQUIRED_FILES)
        raise ValueError(
            "Cutoff-trained package inventory differs from the canonical release: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if digest(root / CUTOFF_TRAINED_ROOT / "cutoff_trained_retrospective_summary.json") \
            != CUTOFF_TRAINED_SUMMARY_SHA256:
        raise ValueError("Cutoff-trained summary is not the canonical release artifact")
    if digest(root / CUTOFF_TRAINED_ROOT / "reporting/reporting_manifest.json") \
            != CUTOFF_TRAINED_REPORTING_MANIFEST_SHA256:
        raise ValueError("Cutoff-trained reporting manifest is not the canonical release artifact")
    if digest(root / cutoff_slurm_path) \
            != CUTOFF_TRAINED_SLURM_SHA256:
        raise ValueError("Cutoff-trained Slurm provenance is not the canonical release artifact")
    if digest(root / CUTOFF_TRAINED_ROOT / "verification/independent_verification.json") \
            != CUTOFF_TRAINED_INDEPENDENT_VERIFICATION_SHA256:
        raise ValueError("Cutoff-trained independent verification is not the canonical release artifact")
    archive_paths = {
        path for path, record in cutoff_records.items()
        if record["storage"] == "derived_archive"
    }
    selected_paths = {
        path for path, record in cutoff_records.items()
        if record["storage"] == "git_selected"
    }
    if (archive_paths != CUTOFF_TRAINED_ARCHIVE_FILES
            or selected_paths != CUTOFF_TRAINED_SELECTED_FILES):
        raise ValueError(
            "Cutoff-trained selected/archive policy changed: "
            f"selected={sorted(selected_paths)}, archive={sorted(archive_paths)}"
        )
    report = {
        "feature_version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "remote_root": f"{TACC_WORK}/feature_repair_runs/downstream_v2_20260905",
        "remote_roots": {
            "primary_downstream": f"{TACC_WORK}/feature_repair_runs/downstream_v2_20260905",
            "graphlet_voronoi_sensitivity": f"{TACC_WORK}/feature_repair_runs/downstream_voronoi_20260905",
            "factor_ablations": f"{TACC_WORK}/feature_repair_runs/factor_ablations_20260906",
        },
        "archive_root": f"{TACC_ARCHIVE}/crystal-communities-feature-repair-20260905",
        "excluded": [
            "inputs/ (licensed index and local input copies; the packaged nonlicensed A-Lab PCA input is included)",
            "cache/", "logs", "temporary files", "Python bytecode",
        ],
        "counts": dict(Counter(r["storage"] for r in records)),
        "bytes": sum(r["bytes"] for r in records),
        "artifacts": records,
    }
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.write_gitignore:
        lines = ["# Scratch and bulk derived artifacts are retained outside git.", "/inputs/", "cache/", "*.npy", "*.npz"]
        lines += ["/" + name for name in bulk if Path(name).suffix not in {".npy", ".npz"}]
        (root / ".gitignore").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: report[k] for k in ("feature_version", "counts", "bytes")}, indent=2))


if __name__ == "__main__":
    main()
