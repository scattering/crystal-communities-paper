#!/usr/bin/env python3
"""Check repaired downstream artifacts by identities and independent count joins.

This reads saved artifacts, not remote services or CIFs, and does not repeat
feature extraction or nearest-neighbor searches. Optional --array-root checks
the archived production matrices against their hashes and scans for finiteness.
Only the JSON report is written. Run from any directory with numpy and pandas.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from formula_conventions import scale_invariant_formula_key

VERSION = "crystal-features-v2-geometric-crystalnn"
SOURCES = {
    "GNoME": ("gnome", "gnome", 5000),
    "MatterGen": ("mattergen", "mattergen-public", 386),
    "MP": ("mp", "mp", 4999),
    "JARVIS": ("jarvis", "jarvis", 4964),
    "Alexandria": ("alexandria", "alexandria", 5000),
}

GRAPHLET_REPRESENTATION = "graphlet-64x20-periodic-images-short-long-arms"
GRAPHLET_ANALYSIS_N_JOBS = 48
GRAPHLET_EXACT_AUDIT_MAX_WORKERS = 16
GRAPHLET_FEATURE_VERSIONS = {
    "crystalnn": "graphlet-64x20-geometric-crystalnn-v1",
    "voronoinn": "graphlet-64x20-voronoinn-slater15-uniform-v1",
}
GRAPHLET_NEIGHBOR_SETTINGS = {
    "crystalnn": {
        "weighted_cn": True,
        "cation_anion": False,
        "distance_cutoffs": None,
        "x_diff_weight": 0.0,
        "porous_adjustment": False,
    },
    "voronoinn": {
        "constructor": {
            "tol": 0.0,
            "targets": None,
            "cutoff": 13.0,
            "allow_pathological": False,
            "weight": "solid_angle",
            "extra_nn_info": False,
            "compute_adj_neighbors": False,
        },
        "distance_screen": "d <= 1.5 * (occupancy-normalized Slater radius_i + radius_j)",
        "distance_screen_multiplier": 1.5,
        "retained_neighbor_weight": "uniform",
    },
}
GRAPHLET_PARTITION_PROTOCOL = {
    "protocol": "historical_graphlet_temporal",
    "knn_k": 16,
    "metric": "euclidean",
    "approximate_neighbors": True,
    "mutual_knn": True,
    "weight": "exp(-distance^2/sigma^2)",
    "resolution": 1.0,
    "louvain_seed": 42,
    "louvain_implementation": "networkx-3.6.1-compatible-undirected-with-node-move-tolerance",
    "louvain_node_move_gain_tolerance": 1e-20,
    "louvain_level_modularity_threshold": 1e-7,
    "louvain_max_local_move_sweeps": 100,
    "minimum_community_size": 10,
    "component_filter": None,
}
GRAPHLET_VORONOI_MATCHED_NEIGHBORS_SHA256 = (
    "7d12bb842637694bc7955074b2fa9a62a818ce208721228993262368ab73d597"
)
MATCHED_PCA_SCALER_SHA256 = (
    "caf5b1309259995f99adc4ea2196b505fcab292a722cf322a4e7810f8982b0ea"
)
GRAPHLET_METHOD_PROVENANCE_FIELDS = {
    "feature_preparation", "feature_preparation_sha256",
    "n_feature_successes_before_cohort_restriction", "restrict_ids",
    "graphlet_neighbor_method", "graphlet_neighbor_settings",
    "graphlet_feature_version", "source_graphlet_feature_version",
    "legacy_crystalnn_metadata_compatibility",
}
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
CUTOFF_TRAINED_SLURM_PATH = "cutoff_trained/slurm-3472690.out"
CUTOFF_TRAINED_SUMMARY_SHA256 = (
    "2cbb8cf5c69b09766e75037734723d71d4db4f64df2490c25265e1bfc84d67d9"
)
CUTOFF_TRAINED_REPORTING_MANIFEST_SHA256 = (
    "37f63dfeaf2d438f87043e31282c9352659be56d72213c57e26586690a5debb7"
)
CUTOFF_TRAINED_SLURM_SHA256 = (
    "0f953d8ec1cb385cd7dc7b41bc63a470ded25ed3f9983f28977946ab221b99c3"
)
CUTOFF_TRAINED_INDEPENDENT_VERIFICATION_SHA256 = (
    "09a5b32d7ac7f7bae2a0b49dc51df65e1c20567dcc5b3acc3b12acab83e47995"
)


def cutoff_trained_file_sets(package_dir):
    """Derive the required inventory and storage policy from a saved run."""
    package_dir = Path(package_dir)
    summary_path = package_dir / "cutoff_trained_retrospective_summary.json"
    if not summary_path.exists():
        return set(), set(), set()
    summary = json.loads(summary_path.read_text())
    relative = {
        "cutoff_trained_retrospective_summary.json",
        "verification/independent_verification.json",
        "reporting/cutoff_trained_composition_summary.json",
        "reporting/cutoff_trained_joint.csv",
        "reporting/cutoff_trained_rates.csv",
        "reporting/cutoff_trained_shared_strata.csv",
        "reporting/reporting_manifest.json",
    }
    audit_path = summary.get("production_filter_audit", {}).get(
        "assignments", {}
    ).get("path")
    if audit_path:
        relative.add(str(audit_path))
    for saved in summary.get("cutoffs", {}).values():
        relative.update(
            str(record["path"])
            for record in saved.get("outputs", {}).values()
            if isinstance(record, dict) and record.get("path")
        )
    job_id = summary.get("execution", {}).get("slurm_job_id")
    if job_id not in (None, ""):
        relative.add(f"slurm-{job_id}.out")
    else:
        # Legacy packages lack an execution block. Keeping their discovered log
        # in the inventory lets the schema check below issue the explicit legacy
        # failure instead of silently accepting it as a current result.
        relative.update(path.name for path in package_dir.glob("slurm-*.out"))

    required = {str(CUTOFF_TRAINED_ROOT / path) for path in relative}
    archive = set()
    for path_text in relative:
        path = package_dir / path_text
        if not path.exists():
            continue
        size = path.stat().st_size
        if (
            size > 5 * 1024 * 1024
            or path.suffix in {".npy", ".npz"}
            or (path.suffix in {".csv", ".json"} and size > 1024 * 1024)
        ):
            archive.add(str(CUTOFF_TRAINED_ROOT / path_text))
    return required, archive, required - archive


_DEFAULT_CUTOFF_TRAINED_DIR = (
    Path(__file__).resolve().parents[1]
    / "notes/feature_repair_2026_09/downstream/cutoff_trained"
)
(
    _CUTOFF_TRAINED_DISCOVERED_FILES,
    CUTOFF_TRAINED_ARCHIVE_FILES,
    _,
) = cutoff_trained_file_sets(_DEFAULT_CUTOFF_TRAINED_DIR)
CUTOFF_TRAINED_REQUIRED_FILES = {
    path
    for path in _CUTOFF_TRAINED_DISCOVERED_FILES
    if not Path(path).name.startswith("slurm-")
} | ({CUTOFF_TRAINED_SLURM_PATH} if CUTOFF_TRAINED_SLURM_PATH else set())
CUTOFF_TRAINED_SELECTED_FILES = (
    CUTOFF_TRAINED_REQUIRED_FILES - CUTOFF_TRAINED_ARCHIVE_FILES
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def boolean(series):
    values = series.astype(str).str.lower()
    if not values.isin(["true", "false", "0", "1"]).all():
        raise ValueError("Unrecognized boolean values")
    return values.isin(["true", "1"])


class Audit:
    def __init__(self, base):
        self.base = base
        self.inputs = {}
        self.checks = []

    def path(self, relative):
        path = self.base / relative
        self.inputs[str(relative)] = sha256(path)
        return path

    def js(self, relative):
        return json.loads(self.path(relative).read_text())

    def csv(self, relative):
        return pd.read_csv(self.path(relative), keep_default_na=False, low_memory=False)

    def check(self, name, condition, **details):
        self.checks.append({"check": name, "passed": bool(condition), **details})
        if not condition:
            raise AssertionError(name)

    def rate(self, name, flags, reported):
        k, n = int(np.sum(flags)), len(flags)
        self.check(name, k == reported["k"] and n == reported["n"]
                   and np.isclose(k / n, reported["rate"], rtol=0, atol=1e-12), k=k, n=n)


def exclusive(events):
    # Independently implement the published priority, retaining noise.
    c = pd.Series("same", index=events.index)
    existing = events.event_type.eq("existing_community")
    c[existing & events.n_active_other_communities.eq(1)] = "cross"
    c[existing & events.n_active_other_communities.ge(2)] = "bridge"
    c[events.event_type.eq("community_birth")] = "birth"
    c[events.community.lt(0)] = "outlier"
    return c


def is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def integer_ids(values):
    if not isinstance(values, list) or not values:
        raise ValueError("Expected a nonempty JSON list of IDs")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
               for value in values):
        raise ValueError("IDs must be positive integers")
    if len(values) != len(set(values)):
        raise ValueError("IDs must be unique")
    return values


def graphlet_provenance_matches(document, feature_metadata, feature_metadata_path,
                                method, analyzer_sha, label_hash, pca_ids_hash):
    legacy_crystalnn = (method == "crystalnn"
                        and feature_metadata.get("neighbor_method") is None
                        and feature_metadata.get("graphlet_feature_version") is None)
    packages = document.get("packages")
    return (
        document.get("production_feature_version") == VERSION
        and document.get("feature_preparation_sha256") == sha256(feature_metadata_path)
        and document.get("production_labels_sha256") == label_hash
        and document.get("production_labels_sha256") == feature_metadata.get("production_labels_sha256")
        and document.get("production_pca_ids_sha256") == pca_ids_hash
        and is_sha256(document.get("production_pca_sha256"))
        and document.get("source_sha256") == analyzer_sha
        and document.get("n_feature_successes_before_cohort_restriction") == feature_metadata.get("n_successful")
        and document.get("neighbor_search_n_jobs") == GRAPHLET_ANALYSIS_N_JOBS
        and document.get("neighbor_search_random_state") == 42
        and document.get("exact_audit_max_workers") == GRAPHLET_EXACT_AUDIT_MAX_WORKERS
        and document.get("graphlet_neighbor_method") == method
        and document.get("graphlet_neighbor_settings") == GRAPHLET_NEIGHBOR_SETTINGS[method]
        and document.get("graphlet_feature_version") == GRAPHLET_FEATURE_VERSIONS[method]
        and document.get("source_graphlet_feature_version") == (
            None if legacy_crystalnn else GRAPHLET_FEATURE_VERSIONS[method])
        and document.get("legacy_crystalnn_metadata_compatibility") is legacy_crystalnn
        and isinstance(packages, dict)
        and {"numpy", "scipy", "scikit-learn", "pynndescent", "networkx"} <= set(packages)
        and packages.get("networkx") == "3.6.1"
        and all(isinstance(name, str) and isinstance(version, str) and version
                for name, version in packages.items())
    )


def graphlet_louvain_diagnostics_valid(partition):
    """Validate the saved numerical-termination evidence for Louvain."""
    try:
        diagnostics = partition.get("louvain_diagnostics")
        if not isinstance(diagnostics, dict):
            return False
        records = diagnostics.get("level_diagnostics")
        levels = diagnostics.get("levels")
        termination = diagnostics.get("termination")
        if (not isinstance(records, list) or not isinstance(levels, int)
                or isinstance(levels, bool) or levels != len(records)):
            return False
        if partition.get("n_edges") == 0:
            return diagnostics == {
                "levels": 0,
                "level_diagnostics": [],
                "termination": "empty_graph",
            }
        if levels < 1 or termination not in {
                "no_local_improvement", "level_modularity_threshold"}:
            return False
        tolerance = float(partition["louvain_node_move_gain_tolerance"])
        level_threshold = float(partition["louvain_level_modularity_threshold"])
        sweep_cap = partition["louvain_max_local_move_sweeps"]
        if (not np.isfinite(tolerance) or tolerance < 0
                or not np.isfinite(level_threshold) or level_threshold < 0
                or not isinstance(sweep_cap, int) or isinstance(sweep_cap, bool)
                or sweep_cap < 1):
            return False
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                return False
            sweeps = record.get("sweeps")
            moves = record.get("accepted_moves")
            if (not isinstance(sweeps, int) or isinstance(sweeps, bool)
                    or not 1 <= sweeps <= sweep_cap
                    or not isinstance(moves, int) or isinstance(moves, bool)
                    or moves < 0):
                return False
            minimum_gain = record.get("minimum_accepted_gain")
            maximum_gain = record.get("maximum_accepted_gain")
            if moves == 0:
                if minimum_gain is not None or maximum_gain is not None:
                    return False
            else:
                minimum_gain = float(minimum_gain)
                maximum_gain = float(maximum_gain)
                if (not np.isfinite(minimum_gain) or not np.isfinite(maximum_gain)
                        or minimum_gain <= tolerance or maximum_gain < minimum_gain):
                    return False
            has_modularity = (
                "level_modularity" in record or "level_modularity_gain" in record
            )
            terminal_no_move = (
                termination == "no_local_improvement" and index == levels - 1
            )
            if terminal_no_move:
                if moves != 0 or has_modularity:
                    return False
            else:
                if not {"level_modularity", "level_modularity_gain"} <= set(record):
                    return False
                modularity = float(record["level_modularity"])
                gain = float(record["level_modularity_gain"])
                if not np.isfinite(modularity) or not np.isfinite(gain):
                    return False
                terminal_threshold = (
                    termination == "level_modularity_threshold" and index == levels - 1
                )
                if terminal_threshold:
                    if gain > level_threshold:
                        return False
                elif gain <= level_threshold:
                    return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def check_common_restriction(document, other_ids_path):
    restrictions = document.get("restrict_ids")
    return (isinstance(restrictions, list) and len(restrictions) == 1
            and isinstance(restrictions[0], dict)
            and restrictions[0].get("sha256") == sha256(other_ids_path))


def temporal_rows_match(saved_by_decade, expected_by_decade):
    if (not isinstance(saved_by_decade, dict)
            or set(saved_by_decade) != set(expected_by_decade)):
        return False
    integer_fields = ("n_total", "n_outlier", "n_birth", "n_same", "n_cross",
                      "n_bridge", "n_same_isolated")
    share_fields = ("share_outlier", "share_birth", "share_same", "share_cross",
                    "share_bridge", "share_any_attachment")
    for decade, expected in expected_by_decade.items():
        saved = saved_by_decade.get(decade)
        if not isinstance(saved, dict) or set(saved) != set(expected):
            return False
        if saved.get("decade") != decade:
            return False
        if any(saved.get(field) != expected[field] for field in integer_fields):
            return False
        if any(not np.isclose(saved.get(field), expected[field], rtol=0, atol=1e-12)
               for field in share_fields):
            return False
    return True


def partition_metrics(reference, alternate):
    """Independently compute every saved production-partition metric."""
    reference = np.asarray(reference, dtype=np.int64)
    alternate = np.asarray(alternate, dtype=np.int64)

    def one(left, right):
        table = contingency_matrix(left, right, sparse=True)

        def pairs(values):
            values = np.asarray(values, dtype=np.int64)
            return int(np.sum(values * (values - 1) // 2))

        intersection_pairs = pairs(table.data)
        reference_pairs = pairs(np.asarray(table.sum(axis=1)).ravel())
        alternate_pairs = pairs(np.asarray(table.sum(axis=0)).ravel())
        return {
            "n_entries": len(left),
            "ARI": float(adjusted_rand_score(left, right)),
            "NMI_arithmetic": float(normalized_mutual_info_score(
                left, right, average_method="arithmetic")),
            "same_community_pair_recall": (
                intersection_pairs / reference_pairs if reference_pairs else None),
            "same_community_pair_precision": (
                intersection_pairs / alternate_pairs if alternate_pairs else None),
            "same_community_pair_f1": (
                2 * intersection_pairs / (reference_pairs + alternate_pairs)
                if reference_pairs + alternate_pairs else None),
            "reference_same_pairs": reference_pairs,
            "alternate_same_pairs": alternate_pairs,
            "intersection_same_pairs": intersection_pairs,
        }

    keep = alternate >= 0
    return {
        "all_entries_noise_as_one_label": one(reference, alternate),
        "alternate_nonnoise_only": one(reference[keep], alternate[keep])
        if keep.any() else None,
        "n_alternate_noise": int(np.sum(~keep)),
    }


def nested_numbers_match(saved, expected, *, atol=1e-12):
    if isinstance(expected, dict):
        return (
            isinstance(saved, dict)
            and set(saved) == set(expected)
            and all(nested_numbers_match(saved[key], value, atol=atol)
                    for key, value in expected.items())
        )
    if isinstance(expected, float):
        return (
            isinstance(saved, (int, float)) and not isinstance(saved, bool)
            and np.isclose(saved, expected, rtol=0, atol=atol)
        )
    return saved == expected


def graphlet_science_documents_identical(left, right):
    """Compare outputs while allowing exactly the known method provenance fields."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if not GRAPHLET_METHOD_PROVENANCE_FIELDS <= set(left):
        return False
    if not GRAPHLET_METHOD_PROVENANCE_FIELDS <= set(right):
        return False
    return (
        {key: value for key, value in left.items()
         if key not in GRAPHLET_METHOD_PROVENANCE_FIELDS}
        == {key: value for key, value in right.items()
            if key not in GRAPHLET_METHOD_PROVENANCE_FIELDS}
    )


def selected_neighbor_rows_identical(left, right):
    """Compare the scientific 1-NN rows, excluding run-level provenance."""
    integer_fields = (
        "query_id", "query_community", "neighbor_id", "neighbor_community",
    )
    distance_fields = (
        "distance", "distance_l1_recomputed",
        "first_ten_chemical_channels_l1", "pure_geometry_four_channels_l1",
        "pair_triplet_chemical_50_channels_l1",
    )
    required = set(integer_fields + distance_fields + ("same_community",))
    if not required <= set(left.columns) or not required <= set(right.columns):
        return False
    return (
        len(left) == len(right)
        and all(np.array_equal(
            left[field].to_numpy(dtype=np.int64),
            right[field].to_numpy(dtype=np.int64),
        ) for field in integer_fields)
        and all(np.array_equal(
            left[field].to_numpy(dtype=float),
            right[field].to_numpy(dtype=float),
        ) for field in distance_fields)
        and np.array_equal(
            boolean(left.same_community).to_numpy(dtype=bool),
            boolean(right.same_community).to_numpy(dtype=bool),
        )
    )


def canonical_partition_labels(values):
    """Give equivalent partitions identical labels without an O(n^2) join."""
    values = np.asarray(values, dtype=np.int64)
    canonical = np.full(values.shape, -1, dtype=np.int64)
    mapping = {}
    next_label = 0
    for index, label in enumerate(values):
        label = int(label)
        if label < 0:
            continue
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        canonical[index] = mapping[label]
    return canonical


def partition_membership_and_noise_identical(left, right):
    """Compare partition membership while allowing a permutation of labels."""
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    return (
        left.shape == right.shape
        and np.array_equal(left < 0, right < 0)
        and np.array_equal(
            canonical_partition_labels(left), canonical_partition_labels(right),
        )
    )


def temporal_event_rows_identical(left, right):
    """Compare temporal science rows while allowing partition-label renaming."""
    if (set(left.columns) != set(right.columns)
            or "community" not in left.columns
            or len(left) != len(right)):
        return False
    if not partition_membership_and_noise_identical(
            left.community.to_numpy(dtype=np.int64),
            right.community.to_numpy(dtype=np.int64)):
        return False
    invariant_columns = sorted(set(left.columns) - {"community"})
    return left[invariant_columns].reset_index(drop=True).equals(
        right[invariant_columns].reset_index(drop=True)
    )


def archive_relative_path(value):
    """Return whether a private-artifact reference is safely archive-relative."""
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def worker_count_sensitivity_checks(document, canonical_recoveries,
                                    canonical_neighbor_paths,
                                    canonical_neighbor_hashes, cohort_size):
    """Validate the saved 112-to-48-worker comparison without old private CSVs."""
    checks = {
        "record and canonical-run provenance": False,
        "private archive references": False,
        "crystalnn transition arithmetic and canonical identity": False,
        "voronoinn transition arithmetic and canonical identity": False,
    }
    try:
        canonical_run = document.get("canonical_run", {})
        superseded_run = document.get("superseded_run", {})
        validation = document.get("validation", {})
        canonical_settings = {
            "neighbor_search_n_jobs": GRAPHLET_ANALYSIS_N_JOBS,
            "neighbor_search_random_state": 42,
            "exact_audit_max_workers": GRAPHLET_EXACT_AUDIT_MAX_WORKERS,
        }
        checks["record and canonical-run provenance"] = (
            document.get("schema_version") == 1
            and canonical_run.get("job_id") == 3471253
            and canonical_run.get("status") == "COMPLETED"
            and canonical_run.get("exit_code") == "0:0"
            and canonical_run.get("elapsed") == "00:18:29"
            and all(canonical_run.get(field) == value
                    for field, value in canonical_settings.items())
            and superseded_run.get("job_id") == 3471058
            and superseded_run.get("neighbor_search_n_jobs") == 112
            and isinstance(superseded_run.get("status"), str)
            and "cancelled" in superseded_run["status"].lower()
            and validation.get("status") == "passed"
            and validation.get("ordered_query_ids_identical_for_both_methods") is True
            and set(canonical_recoveries) == {"crystalnn", "voronoinn"}
            and all(len(run.get("cohort", [])) == cohort_size
                    and all(run.get("recovery", {}).get(field) == value
                            for field, value in canonical_settings.items())
                    for run in canonical_recoveries.values())
        )

        archive_root = superseded_run.get(
            "private_archive_root_relative_to_archive")
        recovery_hashes = superseded_run.get("recovery_json_sha256")
        archive_paths = [
            document.get("variants", {}).get(method, {}).get(
                "superseded_neighbor_csv_private_archive_path")
            for method in ("crystalnn", "voronoinn")
        ]
        checks["private archive references"] = (
            archive_relative_path(archive_root)
            and isinstance(recovery_hashes, dict)
            and set(recovery_hashes) == {"crystalnn", "voronoinn"}
            and all(is_sha256(value) for value in recovery_hashes.values())
            and all(archive_relative_path(path)
                    and path.startswith(archive_root.rstrip("/") + "/")
                    for path in archive_paths)
        )

        variants = document.get("variants", {})
        for method in ("crystalnn", "voronoinn"):
            check_name = (
                f"{method} transition arithmetic and canonical identity")
            variant = variants.get(method, {})
            transitions = variant.get(
                "hit_transitions_superseded_to_canonical", {})
            transition_fields = (
                "hit_to_hit", "hit_to_miss", "miss_to_hit", "miss_to_miss")
            transition_counts_valid = (
                isinstance(transitions, dict)
                and set(transitions) == set(transition_fields)
                and all(isinstance(transitions.get(field), int)
                        and not isinstance(transitions.get(field), bool)
                        and transitions[field] >= 0
                        for field in transition_fields)
            )
            if not transition_counts_valid:
                continue
            canonical_hits = variant.get("canonical_hits")
            superseded_hits = variant.get("superseded_hits")
            n_queries = variant.get("n_queries")
            hit_status_changed = variant.get("hit_status_changed")
            neighbor_id_changed = variant.get("selected_neighbor_id_changed")
            neighbor_community_changed = variant.get(
                "selected_neighbor_community_changed")
            integer_counts = (
                canonical_hits, superseded_hits, n_queries, hit_status_changed,
                neighbor_id_changed, neighbor_community_changed)
            if not all(isinstance(value, int) and not isinstance(value, bool)
                       for value in integer_counts):
                continue
            expected_path = canonical_neighbor_paths[method]
            saved_path = variant.get("canonical_neighbor_csv")
            old_hash = variant.get("superseded_neighbor_csv_sha256")
            expected_hash = canonical_neighbor_hashes[method]
            rate = canonical_hits / n_queries
            old_rate = superseded_hits / n_queries
            delta = canonical_hits - superseded_hits
            checks[check_name] = (
                n_queries == cohort_size > 0
                and sum(transitions.values()) == n_queries
                and canonical_hits
                == transitions["hit_to_hit"] + transitions["miss_to_hit"]
                and superseded_hits
                == transitions["hit_to_hit"] + transitions["hit_to_miss"]
                and hit_status_changed
                == transitions["hit_to_miss"] + transitions["miss_to_hit"]
                and variant.get("canonical_minus_superseded_hits") == delta
                and np.isclose(variant.get("canonical_rate"), rate,
                               rtol=0, atol=1e-15)
                and np.isclose(variant.get("superseded_rate"), old_rate,
                               rtol=0, atol=1e-15)
                and np.isclose(
                    variant.get(
                        "canonical_minus_superseded_percentage_points"),
                    100 * delta / n_queries, rtol=0, atol=1e-12)
                and 0 <= hit_status_changed <= neighbor_community_changed
                <= neighbor_id_changed <= n_queries
                and variant.get("query_ids_identical") is True
                and archive_relative_path(saved_path)
                and saved_path.replace("\\", "/").endswith(expected_path)
                and is_sha256(variant.get("canonical_neighbor_csv_sha256"))
                and variant.get("canonical_neighbor_csv_sha256") == expected_hash
                and is_sha256(old_hash) and old_hash != expected_hash
                and canonical_hits == canonical_recoveries[method].get("hits")
            )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        pass
    return checks


def ordered_ids_sha256(values):
    payload = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def family_partition_statistics(assignments, label_map, production_community):
    """Independently identify one production family's dominant alternate cluster."""
    production_labels = assignments.icsd_id.astype(int).map(label_map)
    family = assignments.loc[production_labels.eq(production_community)].copy()
    alternate = family.community.astype(int)
    counts = alternate.loc[alternate.ge(0)].value_counts()
    if counts.empty:
        raise ValueError(
            f"Production community {production_community} has no alternate cluster")
    overlap = int(counts.max())
    tied = [int(label) for label, count in counts.items() if int(count) == overlap]
    all_labels = assignments.community.astype(int)
    dominant = min(
        tied,
        key=lambda label: int(assignments.loc[
            all_labels.eq(label), "icsd_id"].astype(int).min()),
    )
    full_cluster = assignments.loc[all_labels.eq(dominant)].copy()
    captured = family.loc[alternate.eq(dominant)].copy()
    return {
        "family": family,
        "captured": captured,
        "full_cluster": full_cluster,
        "matched": len(family),
        "noise": int(alternate.lt(0).sum()),
        "dominant": dominant,
        "overlap": overlap,
        "cluster_size": len(full_cluster),
        "recall": overlap / len(family),
        "purity": overlap / len(full_cluster),
    }


def family_window_counts(frame, event_year):
    years = frame.year.astype(int)
    return {
        "pre_count": int(((years > event_year - 10) &
                          (years <= event_year)).sum()),
        "post_count": int(((years > event_year) &
                           (years <= min(event_year + 10, 2015))).sum()),
    }


def graphlet_family_method_checks(method_report, neighbors, assignments, label_map,
                                  descriptors, expected_critical,
                                  expected_summary):
    """Check manuscript-critical family metrics against the underlying rows."""
    checks = {"top20 structure and headline": False, "critical families": False}
    try:
        top20 = method_report.get("top20")
        summary = method_report.get("top20_summary")
        if (not isinstance(top20, list) or len(top20) != 20
                or not isinstance(summary, dict)):
            return checks
        by_community = {row.get("community"): row for row in top20
                        if isinstance(row, dict)}
        descriptor_rows = [
            (int(row.rank), int(row.community), int(row.event_year))
            for row in descriptors.itertuples()
        ]
        report_rows = [
            (row.get("rank"), row.get("community"), row.get("survey_event_year"))
            for row in top20
        ]
        structure_valid = (
            len(by_community) == 20
            and [rank for rank, _, _ in descriptor_rows] == list(range(1, 21))
            and report_rows == descriptor_rows
        )

        communities = [community for _, community, _ in descriptor_rows]
        query_communities = neighbors.query_community.astype(int)
        same = boolean(neighbors.same_community)
        selected = query_communities.isin(communities)
        local_queries = int(selected.sum())
        local_hits = int((selected & same).sum())
        post_gt_pre = 0
        twofold_or_zero = 0
        zero_baseline = 0
        survey_rows_valid = True
        statistics_by_community = {}
        for _, community, event_year in descriptor_rows:
            statistics = family_partition_statistics(
                assignments, label_map, community)
            statistics_by_community[community] = statistics
            counts = family_window_counts(statistics["full_cluster"], event_year)
            post_gt_pre += counts["post_count"] > counts["pre_count"]
            twofold_or_zero += (
                (counts["pre_count"] == 0 and counts["post_count"] > 0)
                or (counts["pre_count"] > 0
                    and counts["post_count"] >= 2 * counts["pre_count"])
            )
            zero_baseline += (
                counts["pre_count"] == 0 and counts["post_count"] > 0)
            saved = by_community[community]
            saved_partition = saved.get("independent_partition", {})
            saved_window = saved.get("survey_event", {}).get(
                "full_dominant_alternate_cluster", {})
            survey_rows_valid &= (
                saved_partition.get("dominant_alternate_cluster_label")
                == statistics["dominant"]
                and saved_partition.get("dominant_overlap_count")
                == statistics["overlap"]
                and saved_partition.get("dominant_cluster_size_in_matched_cohort")
                == statistics["cluster_size"]
                and saved_window.get("pre_count") == counts["pre_count"]
                and saved_window.get("post_count") == counts["post_count"]
            )
        checks["top20 structure and headline"] = (
            structure_valid and survey_rows_valid
            and summary.get("n_communities") == 20
            and summary.get("local_neighbor_queries") == local_queries
            == expected_summary["local_neighbor_queries"]
            and summary.get("local_neighbor_hits") == local_hits
            == expected_summary["local_neighbor_hits"]
            and summary.get("survey_year_dominant_clusters_with_post_gt_pre")
            == post_gt_pre == expected_summary["post_gt_pre"]
            and summary.get(
                "survey_year_dominant_clusters_with_at_least_twofold_or_zero_baseline_growth")
            == twofold_or_zero == expected_summary["twofold_or_zero"]
            and summary.get("zero_baseline_cases_in_previous_count")
            == zero_baseline == expected_summary["zero_baseline"]
        )

        critical_valid = True
        for community, expected in expected_critical.items():
            saved = by_community.get(community, {})
            statistics = statistics_by_community.get(community)
            local = saved.get("local_neighbor_recovery", {})
            partition = saved.get("independent_partition", {})
            targeted = saved.get("targeted_events")
            query_mask = query_communities.eq(community)
            local_expected = {
                "evaluated_members": int(query_mask.sum()),
                "exact_production_community_hits": int((query_mask & same).sum()),
            }
            if (statistics is None or not isinstance(targeted, list)
                    or len(targeted) != 1):
                critical_valid = False
                continue
            event = targeted[0]
            year = expected["target_year"]
            temporal_expected = {
                name: family_window_counts(statistics[source], year)
                for name, source in {
                    "family_members_in_matched_cohort": "family",
                    "family_members_captured_by_dominant_cluster": "captured",
                    "full_dominant_alternate_cluster": "full_cluster",
                }.items()
            }
            critical_valid &= (
                local_expected == expected["local"]
                and all(local.get(field) == value
                        for field, value in local_expected.items())
                and {
                    "matched_dated_members": statistics["matched"],
                    "alternate_noise_count": statistics["noise"],
                    "dominant_overlap_count": statistics["overlap"],
                    "dominant_cluster_size_in_matched_cohort":
                        statistics["cluster_size"],
                } == expected["partition"]
                and all(partition.get(field) == value
                        for field, value in expected["partition"].items())
                and np.isclose(partition.get("dominant_cluster_recall_all_members"),
                               statistics["recall"], rtol=0, atol=1e-15)
                and np.isclose(partition.get("dominant_cluster_purity"),
                               statistics["purity"], rtol=0, atol=1e-15)
                and event.get("year") == year
                and temporal_expected == expected["temporal"]
                and all(
                    event.get(name, {}).get(field) == value
                    for name, counts in temporal_expected.items()
                    for field, value in counts.items()
                )
            )
        checks["critical families"] = critical_valid
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        pass
    return checks


def numpy_int64_sha256(values):
    stream = io.BytesIO()
    np.save(stream, np.asarray(values, dtype=np.int64), allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def louvain_tolerance_audit_checks(document, canonical_partition,
                                   canonical_assignments, canonical_time_summary,
                                   expected_inputs):
    """Validate the saved 1e-20/1e-21 plateau on the canonical graph."""
    checks = {
        "fixed input identities and protocol": False,
        "both tolerances converged with expected metrics": False,
        "exact assignment plateau": False,
        "1e-20 result equals canonical Voronoi replay": False,
    }
    try:
        inputs = document.get("inputs")
        fixed = document.get("fixed_protocol")
        tolerances = document.get("tolerances")
        comparisons = document.get("comparisons")
        expected_fixed = {
            "n_entries": canonical_partition["n_entries"],
            "n_edges": canonical_partition["n_edges"],
            "knn_k": GRAPHLET_PARTITION_PROTOCOL["knn_k"],
            "metric": GRAPHLET_PARTITION_PROTOCOL["metric"],
            "mutual_knn": GRAPHLET_PARTITION_PROTOCOL["mutual_knn"],
            "sigma": canonical_partition["sigma"],
            "weight": GRAPHLET_PARTITION_PROTOCOL["weight"],
            "louvain_seed": GRAPHLET_PARTITION_PROTOCOL["louvain_seed"],
            "louvain_level_modularity_threshold":
                GRAPHLET_PARTITION_PROTOCOL[
                    "louvain_level_modularity_threshold"],
            "louvain_max_local_move_sweeps":
                GRAPHLET_PARTITION_PROTOCOL["louvain_max_local_move_sweeps"],
            "minimum_community_size":
                GRAPHLET_PARTITION_PROTOCOL["minimum_community_size"],
        }
        checks["fixed input identities and protocol"] = (
            document.get("purpose")
            == "isolate stable-Louvain node-move gain tolerance on one fixed saved NNDescent graph"
            and isinstance(inputs, dict) and inputs == expected_inputs
            and inputs.get("neighbors_sha256")
            == GRAPHLET_VORONOI_MATCHED_NEIGHBORS_SHA256
            and isinstance(fixed, dict) and set(fixed) == set(expected_fixed)
            and all(fixed.get(key) == value
                    for key, value in expected_fixed.items())
        )

        tolerance_values = {"1e-20": 1e-20, "1e-21": 1e-21}
        tolerance_records_valid = (
            isinstance(tolerances, dict)
            and set(tolerances) == set(tolerance_values)
        )
        if tolerance_records_valid:
            for key, tolerance in tolerance_values.items():
                record = tolerances.get(key, {})
                diagnostics = record.get("louvain_diagnostics", {})
                accepted_gains = [
                    level.get("minimum_accepted_gain")
                    for level in diagnostics.get("level_diagnostics", [])
                    if level.get("minimum_accepted_gain") is not None
                ]
                diagnostic_document = {
                    "n_edges": fixed["n_edges"],
                    "louvain_node_move_gain_tolerance": tolerance,
                    "louvain_level_modularity_threshold":
                        fixed["louvain_level_modularity_threshold"],
                    "louvain_max_local_move_sweeps":
                        fixed["louvain_max_local_move_sweeps"],
                    "louvain_diagnostics": diagnostics,
                }
                tolerance_records_valid &= (
                    record.get("node_move_gain_tolerance") == tolerance
                    and record.get("converged") is True
                    and graphlet_louvain_diagnostics_valid(diagnostic_document)
                    and accepted_gains
                    and np.isclose(record.get("minimum_accepted_gain"),
                                   min(accepted_gains), rtol=0, atol=1e-30)
                    and record.get("n_communities")
                    == canonical_partition.get("n_communities")
                    and record.get("n_noise")
                    == canonical_partition.get("n_noise")
                    and np.isclose(
                        record.get("modularity_before_small_community_filter"),
                        0.9949454015031324, rtol=0, atol=1e-15)
                    and is_sha256(record.get("assignments_sha256"))
                )
        checks["both tolerances converged with expected metrics"] = bool(
            tolerance_records_valid)

        comparison = (comparisons or {}).get("1e-20_vs_1e-21", {})
        noise = comparison.get("noise_transition_counts", {})
        left = tolerances.get("1e-20", {}) if isinstance(tolerances, dict) else {}
        right = tolerances.get("1e-21", {}) if isinstance(tolerances, dict) else {}
        left_without_tolerance = {
            key: value for key, value in left.items()
            if key != "node_move_gain_tolerance"
        }
        right_without_tolerance = {
            key: value for key, value in right.items()
            if key != "node_move_gain_tolerance"
        }
        labels = canonical_assignments.community.astype(int).to_numpy()
        expected_noise = {
            "nonnoise_to_nonnoise": int(np.sum(labels >= 0)),
            "nonnoise_to_noise": 0,
            "noise_to_nonnoise": 0,
            "noise_to_noise": int(np.sum(labels < 0)),
        }
        checks["exact assignment plateau"] = (
            isinstance(comparisons, dict)
            and set(comparisons) == {"1e-20_vs_1e-21"}
            and comparison.get("exact_label_array_equal") is True
            and comparison.get("n_numeric_labels_different") == 0
            and comparison.get("adjusted_rand_index") == 1.0
            and comparison.get("normalized_mutual_information_arithmetic") == 1.0
            and noise == expected_noise and sum(noise.values()) == len(labels)
            and left_without_tolerance == right_without_tolerance
            and left.get("assignments_sha256") == right.get("assignments_sha256")
        )

        canonical_assignment_hash = numpy_int64_sha256(labels)
        temporal_endpoints = {
            decade: canonical_time_summary.get("by_decade", {}).get(decade)
            for decade in ("1930s", "2010s")
        }
        checks["1e-20 result equals canonical Voronoi replay"] = (
            left.get("assignments_sha256") == canonical_assignment_hash
            and left.get("louvain_diagnostics")
            == canonical_partition.get("louvain_diagnostics")
            and left.get("n_communities")
            == canonical_partition.get("n_communities")
            and left.get("n_noise") == canonical_partition.get("n_noise")
            and left.get("comparison_to_production")
            == canonical_partition.get("comparison_to_corrected_production_labels")
            and left.get("temporal_endpoints") == temporal_endpoints
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        pass
    return checks


def read_graphlet_recovery(a, run_name, method, expected_ids, feature_metadata,
                           feature_metadata_path, other_ids_path, label_map,
                           label_hash, pca_ids_hash, analyzer_sha, shared):
    prefix = f"representations/{run_name}"
    run_summary = a.js(f"{prefix}/summary.json")
    run_metadata = a.js(f"{prefix}/run_metadata.json")
    cohort_path = a.path(f"{prefix}/cohort.ids.json")
    cohort = integer_ids(json.loads(cohort_path.read_text()))
    a.check(f"{run_name}: exact ordered graphlet recovery cohort",
            cohort == expected_ids
            and run_summary.get("n_candidates") == len(cohort)
            and run_summary.get("cohort_sha256") == sha256(cohort_path)
            and run_summary.get("complete") is True
            and run_summary.get("dated_only") is False
            and run_summary.get("target_year_range") is None
            and (check_common_restriction(run_summary, other_ids_path)
                 if shared else run_summary.get("restrict_ids") == []))
    a.check(f"{run_name}: method and source provenance",
            graphlet_provenance_matches(
                run_summary, feature_metadata, feature_metadata_path, method,
                analyzer_sha, label_hash, pca_ids_hash)
            and graphlet_provenance_matches(
                run_metadata, feature_metadata, feature_metadata_path, method,
                analyzer_sha, label_hash, pca_ids_hash)
            and all(run_summary.get(key) == run_metadata.get(key) for key in (
                "production_feature_version", "feature_preparation_sha256",
                "production_labels_sha256", "production_pca_sha256",
                "production_pca_ids_sha256", "source_sha256", "n_candidates",
                "n_feature_successes_before_cohort_restriction", "dated_only",
                "neighbor_search_n_jobs", "neighbor_search_random_state",
                "exact_audit_max_workers",
                "target_year_range", "restrict_ids", "cohort_sha256",
                "graphlet_neighbor_method", "graphlet_neighbor_settings",
                "graphlet_feature_version", "source_graphlet_feature_version",
                "legacy_crystalnn_metadata_compatibility")))

    recovery = a.js(f"{prefix}/graphlet_l1/recovery.json")
    neighbors = a.csv(f"{prefix}/graphlet_l1/nearest_neighbors.csv")
    a.check(f"{run_name}: recovery artifact provenance and protocol",
            graphlet_provenance_matches(
                recovery, feature_metadata, feature_metadata_path, method,
                analyzer_sha, label_hash, pca_ids_hash)
            and recovery.get("cohort_sha256") == sha256(cohort_path)
            and recovery.get("name") == "graphlet_l1"
            and recovery.get("metric") == "manhattan"
            and recovery.get("approximate") is True
            and recovery.get("estimator") == "pynndescent.NNDescent"
            and recovery.get("ann_requested_neighbors_including_self") == 15
            and (check_common_restriction(recovery, other_ids_path)
                 if shared else recovery.get("restrict_ids") == []))

    required_neighbor_columns = {
        "query_id", "query_community", "neighbor_id", "neighbor_community",
        "distance", "same_community", "distance_l1_recomputed",
        "first_ten_chemical_channels_l1", "pure_geometry_four_channels_l1",
        "pair_triplet_chemical_50_channels_l1",
    }
    if not required_neighbor_columns <= set(neighbors.columns):
        raise ValueError(f"{run_name} nearest-neighbor columns are incomplete")
    query_ids = neighbors.query_id.astype(int).tolist()
    neighbor_ids = neighbors.neighbor_id.astype(int)
    expected_set = set(cohort)
    same = neighbors.query_community.astype(int).eq(neighbors.neighbor_community.astype(int))
    stored_same = boolean(neighbors.same_community)
    distances = neighbors.distance.to_numpy(float)
    recomputed = neighbors.distance_l1_recomputed.to_numpy(float)
    parts = neighbors[["first_ten_chemical_channels_l1", "pure_geometry_four_channels_l1",
                       "pair_triplet_chemical_50_channels_l1"]].to_numpy(float)
    sizes = Counter(label_map[iid] for iid in cohort)
    chance = sum(size * (size - 1) for size in sizes.values()) / (len(cohort) * (len(cohort) - 1))
    hits = int(same.sum())
    a.check(f"{run_name}: complete nonself nearest-neighbor identities",
            len(neighbors) == len(cohort) and query_ids == cohort
            and neighbor_ids.isin(expected_set).all()
            and neighbors.query_id.astype(int).ne(neighbor_ids).all()
            and neighbors.query_id.astype(int).map(label_map).eq(neighbors.query_community.astype(int)).all()
            and neighbor_ids.map(label_map).eq(neighbors.neighbor_community.astype(int)).all())
    a.check(f"{run_name}: recovery and distance arithmetic",
            stored_same.eq(same).all()
            and recovery.get("n_entries") == recovery.get("n_candidates") == len(cohort)
            and recovery.get("n_communities") == len(sizes)
            and recovery.get("hits") == hits
            and recovery.get("misses") == len(cohort) - hits
            and np.isclose(recovery.get("same_community_1nn_agreement"), hits / len(cohort), rtol=0, atol=1e-12)
            and np.isclose(recovery.get("random_other_entry_chance"), chance, rtol=0, atol=1e-12)
            and np.isfinite(distances).all() and (distances >= 0).all()
            and np.isfinite(recomputed).all() and (recomputed >= 0).all()
            and np.isfinite(parts).all() and (parts >= 0).all()
            and np.allclose(parts.sum(axis=1), recomputed, rtol=0, atol=1e-10)
            and np.allclose(distances, recomputed, rtol=2e-6, atol=2e-5))
    confusion = a.csv(f"{prefix}/graphlet_l1/confusion_counts.csv")
    if set(confusion.columns) != {"from_community", "to_community", "count"}:
        raise ValueError(f"{run_name} confusion-count columns are invalid")
    observed_confusion = Counter(zip(
        neighbors.loc[~same, "query_community"].astype(int),
        neighbors.loc[~same, "neighbor_community"].astype(int),
    ))
    saved_confusion = {
        (int(row.from_community), int(row.to_community)): int(row.count)
        for row in confusion.itertuples()
    }
    a.check(f"{run_name}: exact miss-confusion arithmetic",
            len(confusion) == len(saved_confusion)
            and all(source != destination and count > 0
                    for (source, destination), count in saved_confusion.items())
            and saved_confusion == dict(observed_confusion)
            and sum(saved_confusion.values()) == len(cohort) - hits)
    reported = run_summary.get("results", {}).get("graphlet_l1", {})
    a.check(f"{run_name}: run summary matches recovery",
            reported.get("n_entries") == len(cohort)
            and reported.get("n_communities") == len(sizes)
            and np.isclose(reported.get("same_community_1nn_agreement"), hits / len(cohort), rtol=0, atol=1e-12)
            and np.isclose(reported.get("random_other_entry_chance"), chance, rtol=0, atol=1e-12))

    audit = a.js(f"{prefix}/graphlet_l1/exact_query_audit.json")
    queries = audit.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{run_name} exact query audit is empty")
    neighbor_by_query = neighbors.set_index(neighbors.query_id.astype(int)).to_dict("index")
    exact_hits = 0
    ann_exact = 0
    valid_queries = True
    for row in queries:
        query_id = row.get("query_id")
        tied = row.get("exact_nearest_tied_ids")
        if (not isinstance(query_id, int) or query_id not in expected_set
                or not isinstance(tied, list) or not tied
                or tied != sorted(set(tied))
                or any(not isinstance(iid, int) or iid not in expected_set or iid == query_id for iid in tied)):
            valid_queries = False
            break
        selected = row.get("exact_selected_neighbor_id")
        selected_community = label_map.get(selected)
        minimum = row.get("exact_minimum_l1")
        ann_distance = row.get("ann_selected_distance_l1")
        excess = row.get("ann_distance_excess_l1")
        current = neighbor_by_query.get(query_id, {})
        tied_communities = [label_map[iid] for iid in tied]
        is_exact = current.get("neighbor_id") in tied
        if not (
            selected == min(tied)
            and row.get("query_community") == label_map[query_id]
            and row.get("exact_selected_neighbor_community") == selected_community
            and row.get("ann_selected_neighbor_id") == current.get("neighbor_id")
            and np.isfinite([minimum, ann_distance, excess]).all()
            and minimum >= 0 and excess >= 0
            and np.isclose(ann_distance, current.get("distance_l1_recomputed"),
                           rtol=0, atol=1e-9)
            and np.isclose(ann_distance - minimum, excess, rtol=0, atol=1e-10)
            and row.get("ann_selected_is_exact_nearest") is is_exact
            and row.get("any_exact_tie_same_community") is any(
                community == label_map[query_id] for community in tied_communities)
            and row.get("all_exact_ties_same_community") is all(
                community == label_map[query_id] for community in tied_communities)
        ):
            valid_queries = False
            break
        ann_exact += int(is_exact)
        exact_hits += int(selected_community == label_map[query_id])
    embedded_audit = recovery.get("exact_query_audit", {})
    a.check(f"{run_name}: exact-query identities, ties and aggregate rates",
            valid_queries and len({row["query_id"] for row in queries}) == len(queries)
            and audit.get("n_queries") == len(queries)
            and 1 <= len(queries) <= len(cohort)
            and audit.get("query_seed") == 42
            and audit.get("parallel_workers") == recovery.get("exact_audit_max_workers")
            and audit.get("candidate_pool_size") == len(cohort)
            and audit.get("distance") == "exact float64 cityblock on stored float32 CDF values"
            and audit.get("tie_policy") == "all exact-equal ties retained; smallest ICSD ID selected"
            and np.isclose(audit.get("ann_exact_nearest_fraction"), ann_exact / len(queries), rtol=0, atol=1e-12)
            and np.isclose(audit.get("exact_selected_label_agreement"), exact_hits / len(queries), rtol=0, atol=1e-12)
            and all(embedded_audit.get(key) == audit.get(key) for key in (
                "n_queries", "query_seed", "parallel_workers", "candidate_pool_size", "distance",
                "tie_policy", "ann_exact_nearest_fraction", "exact_selected_label_agreement")))
    return {
        "run_name": run_name,
        "method": method,
        "cohort": cohort,
        "summary": run_summary,
        "recovery": recovery,
        "neighbors": neighbors,
        "same": same.to_numpy(bool),
        "hits": hits,
        "chance": chance,
        "cohort_path": cohort_path,
    }


def read_voronoi_graphlet_features(a, nonnoise, label_rows, label_hash):
    prefix = "representations/graphlet_voronoi_features"
    seed_prefix = "representations/graphlet_voronoi_seed"
    metadata_path = a.path(f"{prefix}/feature_preparation.json")
    metadata = json.loads(metadata_path.read_text())
    ids_path = a.path(f"{prefix}/features.ids.json")
    ids = integer_ids(json.loads(ids_path.read_text()))
    assignments = a.csv(f"{prefix}/production_assignments.csv")
    bin_path = a.path(f"{prefix}/bin_edges.json")
    bin_edges = json.loads(bin_path.read_text())
    crystal_bin_path = a.path("representations/graphlet_features/bin_edges.json")
    crystal_bin_edges = json.loads(crystal_bin_path.read_text())
    seed_report_path = a.path(f"{seed_prefix}/featurization_report.json")
    seed_report = json.loads(seed_report_path.read_text())
    seed_bin_path = a.path(f"{seed_prefix}/bin_edges.json")
    seed_ids_path = a.path(f"{seed_prefix}/graphlet_cdf.npy.ids.json")
    seed_ids = integer_ids(json.loads(seed_ids_path.read_text()))
    seed_plain_ids_path = a.path(f"{seed_prefix}/graphlet_cdf.ids.json")
    seed_plain_ids = integer_ids(json.loads(seed_plain_ids_path.read_text()))
    seed_metadata_path = a.path(f"{seed_prefix}/graphlet_cdf.npy.meta.json")
    seed_metadata = json.loads(seed_metadata_path.read_text())
    crystal_seed_report_path = a.path("../full_run_results/graphlets/featurization_report.json")
    crystal_seed_report = json.loads(crystal_seed_report_path.read_text())

    requested = integer_ids(metadata.get("requested_ids"))
    failures = metadata.get("failures")
    if not isinstance(failures, list):
        raise ValueError("VoronoiNN graphlet feature failures are not a list")
    try:
        failed_ids = [int(row["icsd_id"]) for row in failures]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("VoronoiNN graphlet failures lack integer ICSD IDs") from exc
    source_hashes = metadata.get("source_sha256", {})
    repo_root = a.base.parents[2]
    expected_source_hashes = {
        "notes/feature_repair_2026_09/downstream/prepare_representation_features.py":
            sha256(repo_root / "notes/feature_repair_2026_09/downstream/prepare_representation_features.py"),
        "experiments/graphlet_compare/graphlet_features.py":
            sha256(repo_root / "experiments/graphlet_compare/graphlet_features.py"),
        "scripts/crystal_neighbors.py": sha256(repo_root / "scripts/crystal_neighbors.py"),
    }
    source_metadata = metadata.get("existing_metadata", {})
    source_feature_path = Path(str(metadata.get("existing_features", "")))
    source_ids_path = Path(str(metadata.get("existing_ids", "")))
    source_bins_path = Path(str(source_metadata.get("bin_edges", "")))
    source_failure_path = Path(str(source_metadata.get("failure_report", "")))
    preparation_packages = metadata.get("packages", {})
    crystal_sampling = crystal_seed_report.get("bin_edge_sampling", {})
    seed_sampling_source = seed_report.get("bin_edge_sampling_source", {})
    seed_sampling = seed_report.get("bin_edge_sampling", {})
    seed_sampling_requested = seed_sampling.get("requested_icsd_ids")
    seed_sampling_successful = seed_sampling.get("successful_icsd_ids")
    seed_sampling_failures = seed_sampling.get("failures")
    if not all(isinstance(values, list) for values in (
            seed_sampling_requested, seed_sampling_successful, seed_sampling_failures)):
        raise ValueError("VoronoiNN bin-calibration identities are incomplete")
    try:
        seed_sampling_failed_ids = [int(row["icsd_id"]) for row in seed_sampling_failures]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("VoronoiNN bin-calibration failures lack integer ICSD IDs") from exc
    crystal_sampling_ids = crystal_sampling.get("requested_icsd_ids")
    a.check("VoronoiNN graphlets: exact independent bin calibration cohort",
            isinstance(crystal_sampling_ids, list)
            and len(crystal_sampling_ids) == len(set(crystal_sampling_ids)) == 5000
            and seed_sampling_requested == crystal_sampling_ids
            and seed_sampling.get("n_requested") == len(seed_sampling_requested)
            and len(seed_sampling_requested) == len(set(seed_sampling_requested))
            and all(isinstance(iid, int) and iid > 0 for iid in seed_sampling_requested)
            and seed_sampling_successful == sorted(seed_sampling_successful)
            and len(seed_sampling_successful) == len(set(seed_sampling_successful))
            and len(seed_sampling_failed_ids) == len(set(seed_sampling_failed_ids))
            and set(seed_sampling_successful).isdisjoint(seed_sampling_failed_ids)
            and set(seed_sampling_successful) | set(seed_sampling_failed_ids)
            == set(seed_sampling_requested)
            and seed_sampling.get("n_successful") == len(seed_sampling_successful)
            and seed_sampling.get("n_failed") == len(seed_sampling_failed_ids)
            and seed_sampling.get("n_successful") + seed_sampling.get("n_failed")
            == seed_sampling.get("n_requested")
            and seed_sampling_source.get("sha256") == sha256(crystal_seed_report_path)
            and Path(str(seed_sampling_source.get("path", ""))).name
            == "featurization_report.json")

    seed_features = seed_report.get("features", {})
    seed_feature_failures = seed_features.get("failures")
    if not isinstance(seed_feature_failures, list):
        raise ValueError("VoronoiNN seed feature failures are absent")
    try:
        seed_feature_failed_ids = [int(row["icsd_id"]) for row in seed_feature_failures]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("VoronoiNN seed feature failures lack integer ICSD IDs") from exc
    dated_targets = sorted(
        int(row.icsd_id) for row in label_rows.itertuples()
        if int(row.community) >= 0 and str(row.year) != ""
        and 1900 <= int(float(row.year)) <= 2025
    )
    a.check("VoronoiNN graphlets: seed feature success and failure identities",
            seed_report.get("complete") is True
            and seed_report.get("features_only") is True
            and seed_report.get("target_year_range") == [1900, 2025]
            and seed_features.get("n_requested") == len(dated_targets)
            and seed_features.get("n_successful") == len(seed_ids)
            and seed_features.get("n_failed") == len(seed_feature_failed_ids)
            and seed_features.get("n_successful") + seed_features.get("n_failed")
            == seed_features.get("n_requested")
            and len(seed_feature_failed_ids) == len(set(seed_feature_failed_ids))
            and set(seed_feature_failed_ids) <= set(dated_targets)
            and seed_ids == seed_plain_ids
            and seed_ids == [iid for iid in dated_targets if iid not in set(seed_feature_failed_ids)]
            and seed_metadata.get("n_requested") == seed_features.get("n_requested")
            and seed_metadata.get("n_successful") == seed_features.get("n_successful")
            and seed_metadata.get("n_failed") == seed_features.get("n_failed")
            and source_metadata == seed_metadata
            and metadata.get("existing_ids_sha256") == sha256(seed_ids_path))
    a.check("VoronoiNN graphlets: exact method, version and producer provenance",
            metadata.get("complete") is True
            and metadata.get("kind") == "graphlet"
            and metadata.get("production_feature_version") == VERSION
            and metadata.get("feature_version") == VERSION
            and metadata.get("production_labels_sha256") == label_hash
            and metadata.get("neighbor_method") == "voronoinn"
            and metadata.get("neighbor_settings") == GRAPHLET_NEIGHBOR_SETTINGS["voronoinn"]
            and metadata.get("graphlet_feature_version") == GRAPHLET_FEATURE_VERSIONS["voronoinn"]
            and metadata.get("representation") == GRAPHLET_REPRESENTATION
            and source_hashes == expected_source_hashes
            and is_sha256(metadata.get("features_sha256"))
            and source_metadata.get("feature_version") == VERSION
            and source_metadata.get("neighbor_method") == "voronoinn"
            and source_metadata.get("neighbor_settings") == GRAPHLET_NEIGHBOR_SETTINGS["voronoinn"]
            and source_metadata.get("graphlet_feature_version") == GRAPHLET_FEATURE_VERSIONS["voronoinn"]
            and source_metadata.get("representation") == GRAPHLET_REPRESENTATION
            and seed_report.get("feature_version") == VERSION
            and seed_report.get("neighbor_method") == "voronoinn"
            and seed_report.get("neighbor_settings") == GRAPHLET_NEIGHBOR_SETTINGS["voronoinn"]
            and seed_report.get("graphlet_feature_version") == GRAPHLET_FEATURE_VERSIONS["voronoinn"]
            and seed_report.get("representation") == GRAPHLET_REPRESENTATION
            and seed_report.get("cdf_row_order") == "ascending ICSD ID"
            and isinstance(preparation_packages, dict) and "pymatgen" in preparation_packages
            and all(isinstance(name, str) and isinstance(version, str) and version
                    for name, version in preparation_packages.items()))
    a.check("VoronoiNN graphlets: independently fitted bin provenance",
            isinstance(bin_edges, dict) and len(bin_edges) == 64
            and set(bin_edges) == set(crystal_bin_edges)
            and all(isinstance(bounds, list) and len(bounds) == 2
                    and np.isfinite(np.asarray(bounds, dtype=float)).all()
                    and float(bounds[1]) > float(bounds[0])
                    for bounds in bin_edges.values())
            and metadata.get("bin_edges_sha256") == sha256(bin_path)
            and sha256(bin_path) == sha256(seed_bin_path)
            and sha256(bin_path) != sha256(crystal_bin_path)
            and source_feature_path.parent.name == "graphlet_voronoi_seed"
            and source_ids_path.parent.name == "graphlet_voronoi_seed"
            and source_bins_path.parent.name == "graphlet_voronoi_seed"
            and source_failure_path.parent.name == "graphlet_voronoi_seed"
            and source_feature_path.name == "graphlet_cdf.npy"
            and source_ids_path.name == "graphlet_cdf.npy.ids.json"
            and source_bins_path.name == "bin_edges.json"
            and source_failure_path.name == "featurization_report.json"
            and source_bins_path == Path(str(seed_metadata.get("bin_edges")))
            and source_failure_path == Path(str(seed_metadata.get("failure_report")))
            and is_sha256(metadata.get("existing_features_sha256"))
            and is_sha256(metadata.get("existing_ids_sha256"))
            and source_metadata.get("cdf_row_order") == "ascending ICSD ID")

    expected_successes = [iid for iid in requested if iid not in set(failed_ids)]
    missing = metadata.get("missing_ids_to_attempt")
    if not isinstance(missing, list):
        raise ValueError("VoronoiNN graphlet missing-ID list is absent")
    assignment_ids = assignments.icsd_id.astype(int).tolist()
    a.check("VoronoiNN graphlets: target, success and failure identities",
            requested == sorted(nonnoise)
            and len(requested) == metadata.get("n_requested")
            and len(failed_ids) == len(set(failed_ids)) == metadata.get("n_failed")
            and set(failed_ids) <= set(requested)
            and ids == expected_successes
            and len(ids) == metadata.get("n_successful")
            and metadata.get("n_successful") + metadata.get("n_failed") == metadata.get("n_requested")
            and metadata.get("ids_sha256") == sha256(ids_path)
            and metadata.get("feature_shape") == [len(ids), 1280]
            and assignment_ids == ids
            and assignments.icsd_id.astype(int).is_unique)
    production_by_id = label_rows.set_index(label_rows.icsd_id.astype(int))
    joined = production_by_id.loc[assignment_ids]
    reuse_validation = metadata.get("reuse_validation")
    if not isinstance(reuse_validation, list):
        raise ValueError("VoronoiNN graphlet reuse validation is absent")
    a.check("VoronoiNN graphlets: saved labels, years and extension accounting",
            assignments.community.astype(int).tolist() == joined.community.astype(int).tolist()
            and assignments.year.astype(str).tolist() == joined.year.astype(str).tolist()
            and metadata.get("n_requested_undated") == sum(
                str(production_by_id.loc[iid, "year"]) == "" for iid in requested)
            and metadata.get("n_successful_undated") == sum(
                str(production_by_id.loc[iid, "year"]) == "" for iid in ids)
            and metadata.get("n_reused_candidates") == metadata.get("n_reused")
            and metadata.get("n_reused") + metadata.get("n_new_successful") == len(ids)
            and metadata.get("n_missing_to_attempt") == len(missing)
            and metadata.get("n_new_attempted") == len(missing)
            and len(missing) == len(set(missing)) and missing == sorted(missing)
            and set(missing) <= set(requested)
            and set(requested) - set(missing) <= set(ids)
            and set(failed_ids) <= set(missing)
            and metadata.get("n_new_successful") == len(missing) - len(failed_ids)
            and metadata.get("n_reused_candidates") == source_metadata.get("n_successful")
            and len(reuse_validation) == min(16, metadata.get("n_reused_candidates"))
            and len({row.get("icsd_id") for row in reuse_validation}) == len(reuse_validation)
            and all(row.get("icsd_id") in set(requested) - set(missing)
                    and row.get("matches") is True and row.get("failure") is None
                    and np.isfinite(row.get("max_abs_difference"))
                    and row.get("max_abs_difference") >= 0
                    for row in reuse_validation))
    return (metadata, ids, metadata_path, ids_path,
            len(seed_sampling_successful))


def read_graphlet_partition(a, run_name, method, expected_ids, feature_metadata,
                            feature_metadata_path, other_ids_path, label_rows,
                            label_hash, pca_ids_hash, analyzer_sha, shared,
                            analysis_name="graphlet"):
    prefix = f"representations/{run_name}"
    variant_prefix = f"{prefix}/{analysis_name}"
    label = f"{run_name}/{analysis_name}"
    run_summary = a.js(f"{prefix}/summary.json")
    run_metadata = a.js(f"{prefix}/run_metadata.json")
    cohort_path = a.path(f"{prefix}/cohort.ids.json")
    cohort = integer_ids(json.loads(cohort_path.read_text()))
    variant_ids_path = a.path(f"{variant_prefix}/ids.json")
    variant_ids = integer_ids(json.loads(variant_ids_path.read_text()))
    partition = a.js(f"{variant_prefix}/partition.json")
    assignments = a.csv(f"{variant_prefix}/community_assignments.csv")
    events = a.csv(f"{variant_prefix}/node_temporal_events.csv")
    exclusive_report = a.js(f"{variant_prefix}/exclusive_by_decade.json")
    time_summary = a.js(f"{variant_prefix}/graph_time_summary.json")
    growth = a.csv(f"{variant_prefix}/community_growth_by_decade.csv")
    top_communities = a.js(f"{variant_prefix}/top_communities.json")

    a.check(f"{label}: exact ordered dated cohort",
            cohort == expected_ids and variant_ids == cohort
            and run_summary.get("n_candidates") == len(cohort)
            and run_summary.get("cohort_sha256") == sha256(cohort_path)
            and run_summary.get("complete") is True
            and run_summary.get("dated_only") is True
            and run_summary.get("target_year_range") == [1900, 2025]
            and run_summary.get("n_undated") == 0
            and (check_common_restriction(run_summary, other_ids_path)
                 if shared else run_summary.get("restrict_ids") == []))
    provenance_documents = (run_summary, run_metadata, partition, exclusive_report,
                            time_summary)
    a.check(f"{label}: method and source provenance",
            all(graphlet_provenance_matches(
                document, feature_metadata, feature_metadata_path, method,
                analyzer_sha, label_hash, pca_ids_hash)
                for document in provenance_documents)
            and all(document.get("cohort_sha256") == sha256(cohort_path)
                    for document in provenance_documents)
            and all((check_common_restriction(document, other_ids_path)
                     if shared else document.get("restrict_ids") == [])
                    for document in provenance_documents))
    expected_protocol = dict(GRAPHLET_PARTITION_PROTOCOL)
    expected_protocol["knn_k"] = min(16, len(cohort) - 1)
    protocol = {key: partition.get(key) for key in expected_protocol}
    a.check(f"{label}: graphlet partition protocol",
            partition.get("name") == analysis_name
            and protocol == expected_protocol
            and all({key: document.get(key) for key in expected_protocol}
                    == expected_protocol
                    for document in (exclusive_report, time_summary))
            and np.isfinite(partition.get("sigma")) and partition.get("sigma") > 0
            and partition.get("sigma_population") == "positive retained neighbor distances")
    a.check(
        f"{label}: Louvain numerical convergence diagnostics",
        graphlet_louvain_diagnostics_valid(partition)
        and all(
            document.get("louvain_diagnostics") == partition.get("louvain_diagnostics")
            for document in (exclusive_report, time_summary)
        ),
    )

    assignment_ids = assignments.icsd_id.astype(int).tolist()
    assignment_labels = assignments.community.astype(int).to_numpy()
    nonnoise_labels = sorted(set(assignment_labels[assignment_labels >= 0].tolist()))
    expected_label_rows = label_rows.set_index(label_rows.icsd_id.astype(int)).loc[cohort]
    community_sizes = Counter(assignment_labels[assignment_labels >= 0])
    result = run_summary.get("results", {}).get(analysis_name, {})
    a.check(f"{label}: assignment identity and partition counts",
            assignment_ids == cohort and assignments.icsd_id.astype(int).is_unique
            and assignments.year.astype(int).tolist() == expected_label_rows.year.astype(int).tolist()
            and np.all(assignment_labels >= -1)
            and nonnoise_labels == list(range(len(nonnoise_labels)))
            and all(size >= GRAPHLET_PARTITION_PROTOCOL["minimum_community_size"]
                    for size in community_sizes.values())
            and partition.get("n_entries") == partition.get("n_candidates") == len(cohort)
            and partition.get("n_communities") == len(nonnoise_labels)
            and partition.get("n_noise") == int(np.sum(assignment_labels < 0))
            and isinstance(partition.get("n_edges"), int) and partition.get("n_edges") >= 0
            and result.get("n_entries") == len(cohort)
            and result.get("n_communities") == len(nonnoise_labels)
            and result.get("n_noise") == int(np.sum(assignment_labels < 0)))

    if not {"icsd_id", "year", "decade", "community", "event_type",
            "n_active_neighbors", "n_active_same_community_neighbors",
            "n_active_other_communities", "is_bridge_attachment"} <= set(events.columns):
        raise ValueError(f"{label} temporal-event columns are incomplete")
    event_ids = events.icsd_id.astype(int)
    event_by_id = events.set_index(event_ids)
    assignment_by_id = assignments.set_index(assignments.icsd_id.astype(int))
    event_categories = exclusive(events)
    birth_years = assignments.loc[assignments.community.astype(int).ge(0)].groupby(
        assignments.community.astype(int)).year.min().astype(int).to_dict()
    expected_event_type = [
        "outlier" if int(row.community) < 0 else
        "community_birth" if int(row.year) == birth_years[int(row.community)] else
        "existing_community"
        for row in events.itertuples()
    ]
    a.check(f"{label}: temporal event identity, year and partition labels",
            len(events) == len(cohort) and event_ids.is_unique and set(event_ids) == set(cohort)
            and event_ids.tolist() == sorted(
                cohort, key=lambda iid: (int(assignment_by_id.loc[iid, "year"]), iid))
            and event_by_id.loc[cohort].year.astype(int).tolist() == assignment_by_id.loc[cohort].year.astype(int).tolist()
            and event_by_id.loc[cohort].community.astype(int).tolist() == assignment_by_id.loc[cohort].community.astype(int).tolist()
            and events.decade.astype(str).tolist() == [f"{int(year) // 10 * 10}s" for year in events.year]
            and events.event_type.isin(["outlier", "community_birth", "existing_community"]).all()
            and events.event_type.tolist() == expected_event_type
            and boolean(events.is_bridge_attachment).eq(
                events.n_active_other_communities.astype(int).ge(2)).all()
            and events.n_active_neighbors.astype(int).ge(0).all()
            and events.n_active_same_community_neighbors.astype(int).ge(0).all()
            and events.n_active_other_communities.astype(int).ge(0).all()
            and events.n_active_same_community_neighbors.astype(int).le(
                events.n_active_neighbors.astype(int)).all()
            and events.n_active_other_communities.astype(int).le(
                events.n_active_neighbors.astype(int)).all()
            and int(events.n_active_neighbors.astype(int).sum())
            == partition.get("n_edges"))

    rows = exclusive_report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} exclusive temporal rows are absent")
    expected_row_fields = {
        "decade", "n_total", "n_outlier", "share_outlier", "n_birth",
        "share_birth", "n_same", "share_same", "n_cross", "share_cross",
        "n_bridge", "share_bridge", "share_any_attachment", "n_same_isolated",
    }
    row_by_decade = {row.get("decade"): row for row in rows}
    valid_rows = (len(row_by_decade) == len(rows)
                  and set(row_by_decade) == set(events.decade.astype(str))
                  and all(isinstance(row, dict) and set(row) == expected_row_fields
                          for row in rows))
    recomputed_rows = {}
    for decade, group in events.assign(exclusive_category=event_categories).groupby("decade", sort=False):
        counts = Counter(group.exclusive_category)
        n_total = len(group)
        calculated = {
            "decade": str(decade),
            "n_total": n_total,
            "n_outlier": counts["outlier"],
            "share_outlier": counts["outlier"] / n_total,
            "n_birth": counts["birth"],
            "share_birth": counts["birth"] / n_total,
            "n_same": counts["same"],
            "share_same": counts["same"] / n_total,
            "n_cross": counts["cross"],
            "share_cross": counts["cross"] / n_total,
            "n_bridge": counts["bridge"],
            "share_bridge": counts["bridge"] / n_total,
            "share_any_attachment": (counts["same"] + counts["cross"] + counts["bridge"]) / n_total,
            "n_same_isolated": int(((group.exclusive_category == "same")
                                    & (group.n_active_same_community_neighbors == 0)).sum()),
        }
        recomputed_rows[str(decade)] = calculated
        saved = row_by_decade.get(str(decade), {})
        integer_fields = ("n_total", "n_outlier", "n_birth", "n_same", "n_cross",
                          "n_bridge", "n_same_isolated")
        share_fields = ("share_outlier", "share_birth", "share_same", "share_cross",
                        "share_bridge", "share_any_attachment")
        valid_rows &= (all(saved.get(field) == calculated[field] for field in integer_fields)
                       and all(np.isclose(saved.get(field), calculated[field], rtol=0, atol=1e-12)
                               for field in share_fields)
                       and sum(calculated[field] for field in
                               ("n_outlier", "n_birth", "n_same", "n_cross", "n_bridge")) == n_total)
        by_decade = time_summary.get("by_decade", {}).get(str(decade), {})
        valid_rows &= (int(by_decade.get("n_total", -1)) == n_total
                       and int(by_decade.get("n_outlier", -1)) == counts["outlier"]
                       and int(by_decade.get("n_cluster_birth_point", -1)) == counts["birth"]
                       and int(by_decade.get("n_existing_cluster", -1))
                       == counts["same"] + counts["cross"] + counts["bridge"])
    a.check(f"{label}: exclusive temporal counts and shares",
            valid_rows and sum(row["n_total"] for row in recomputed_rows.values()) == len(cohort)
            and time_summary.get("n_points") == len(cohort)
            and time_summary.get("n_outliers") == int(np.sum(assignment_labels < 0)))

    reference_labels = expected_label_rows.community.astype(int).to_numpy()
    expected_comparison = partition_metrics(reference_labels, assignment_labels)
    a.check(f"{label}: independently recomputed production-partition metrics",
            nested_numbers_match(
                partition.get("comparison_to_corrected_production_labels"),
                expected_comparison)
            and nested_numbers_match(
                exclusive_report.get("comparison_to_corrected_production_labels"),
                expected_comparison)
            and nested_numbers_match(
                time_summary.get("comparison_to_corrected_production_labels"),
                expected_comparison)
            and nested_numbers_match(
                result.get("comparison_to_corrected_production_labels"),
                expected_comparison))
    a.check(f"{label}: partition fields propagated to temporal reports",
            all(exclusive_report.get(field) == value
                for field, value in partition.items())
            and all(time_summary.get(field) == value
                    for field, value in partition.items()))

    expected_birth_year = {}
    for row in assignments.itertuples():
        community = int(row.community)
        if community >= 0:
            key = str(community)
            year = int(row.year)
            expected_birth_year[key] = min(
                year, expected_birth_year.get(key, year))
    expected_time_decades = {}
    for decade, group in events.groupby(events.decade.astype(str), sort=True):
        n_total = len(group)
        nonnoise_group = group.community.astype(int).ge(0)
        counts = {
            "n_total": n_total,
            "n_outlier": int(group.event_type.eq("outlier").sum()),
            "n_cluster_birth_point": int(
                group.event_type.eq("community_birth").sum()),
            "n_existing_cluster": int(
                group.event_type.eq("existing_community").sum()),
            "n_same_community_attachment": int((
                nonnoise_group
                & group.n_active_same_community_neighbors.astype(int).gt(0)
            ).sum()),
            "n_cross_community_attachment": int((
                nonnoise_group
                & group.n_active_other_communities.astype(int).gt(0)
            ).sum()),
            "n_bridge_attachment": int((
                nonnoise_group & boolean(group.is_bridge_attachment)
            ).sum()),
            "n_core_attachment": int(group.core_periphery.astype(str).eq("core").sum()),
            "n_periphery_attachment": int(
                group.core_periphery.astype(str).eq("periphery").sum()),
        }
        for field in (
            "n_outlier", "n_cluster_birth_point", "n_existing_cluster",
            "n_same_community_attachment", "n_cross_community_attachment",
            "n_bridge_attachment", "n_core_attachment", "n_periphery_attachment",
        ):
            counts[field.removeprefix("n_") + "_ratio"] = counts[field] / n_total
        expected_time_decades[str(decade)] = counts
    a.check(f"{label}: complete graph-time summary from event rows",
            time_summary.get("n_points") == len(cohort)
            and time_summary.get("n_outliers") == int(np.sum(assignment_labels < 0))
            and np.isclose(
                time_summary.get("outlier_ratio"),
                np.mean(assignment_labels < 0), rtol=0, atol=1e-12)
            and time_summary.get("community_birth_year") == expected_birth_year
            and nested_numbers_match(
                time_summary.get("by_decade"), expected_time_decades))

    expected_top = [
        {"community": int(community), "size": int(size)}
        for community, size in Counter(
            int(value) for value in assignment_labels if int(value) >= 0
        ).most_common(25)
    ]
    decades = sorted(events.decade.astype(str).unique().tolist())
    growth_counts = Counter(
        (int(row.community), str(row.decade))
        for row in events.itertuples()
        if int(row.community) >= 0
    )
    expected_growth = []
    for item in expected_top:
        running = 0
        for decade in decades:
            running += growth_counts[(item["community"], decade)]
            expected_growth.append({
                "community": item["community"],
                "decade": decade,
                "cumulative_size": running,
            })
    growth_records = (
        growth.assign(
            community=growth.community.astype(int),
            decade=growth.decade.astype(str),
            cumulative_size=growth.cumulative_size.astype(int),
        ).to_dict("records")
        if list(growth.columns) == ["community", "decade", "cumulative_size"]
        else None
    )
    a.check(f"{label}: top-community and cumulative-growth outputs",
            top_communities == expected_top and growth_records == expected_growth)
    return {
        "run_name": run_name,
        "analysis_name": analysis_name,
        "method": method,
        "cohort": cohort,
        "summary": run_summary,
        "partition": partition,
        "assignments": assignments,
        "events": events,
        "exclusive_rows": recomputed_rows,
        "cohort_path": cohort_path,
        "variant_ids_path": variant_ids_path,
        "exclusive_report": exclusive_report,
        "time_summary": time_summary,
        "growth": growth,
        "top_communities": top_communities,
    }


def verify_graphlet_neighbor_sensitivity(a, nonnoise, label_rows, label_map, years,
                                         label_hash):
    crystal_feature_prefix = "representations/graphlet_features"
    crystal_metadata_path = a.path(f"{crystal_feature_prefix}/feature_preparation.json")
    crystal_metadata = json.loads(crystal_metadata_path.read_text())
    crystal_ids_path = a.path(f"{crystal_feature_prefix}/features.ids.json")
    crystal_ids = integer_ids(json.loads(crystal_ids_path.read_text()))
    (voronoi_metadata, voronoi_ids, voronoi_metadata_path, voronoi_ids_path,
     voronoi_calibration_successes) = \
        read_voronoi_graphlet_features(a, nonnoise, label_rows, label_hash)
    common_ids = sorted(set(crystal_ids) & set(voronoi_ids))
    if len(common_ids) < 2:
        raise ValueError("CrystalNN/VoronoiNN graphlet common cohort has fewer than two entries")
    dated_voronoi_ids = [iid for iid in voronoi_ids
                         if years[iid] != "" and 1900 <= int(float(years[iid])) <= 2025]
    dated_common_ids = [iid for iid in common_ids
                        if years[iid] != "" and 1900 <= int(float(years[iid])) <= 2025]
    analyzer_path = a.path("analyze_representations.py")
    analyzer_sha = sha256(analyzer_path)
    pca_ids_path = a.path("../full_run_results/production/sample_assignments.csv")
    pca_ids_hash = sha256(pca_ids_path)

    recoveries = {
        "voronoi_full": read_graphlet_recovery(
            a, "graphlet-voronoi-full-recovery", "voronoinn", voronoi_ids,
            voronoi_metadata, voronoi_metadata_path, crystal_ids_path,
            label_map, label_hash, pca_ids_hash, analyzer_sha, False),
        "voronoi_shared": read_graphlet_recovery(
            a, "graphlet-voronoi-shared-crystalnn-recovery", "voronoinn", common_ids,
            voronoi_metadata, voronoi_metadata_path, crystal_ids_path,
            label_map, label_hash, pca_ids_hash, analyzer_sha, True),
        "crystal_shared": read_graphlet_recovery(
            a, "graphlet-crystalnn-shared-voronoi-recovery", "crystalnn", common_ids,
            crystal_metadata, crystal_metadata_path, voronoi_ids_path,
            label_map, label_hash, pca_ids_hash, analyzer_sha, True),
    }
    canonical_neighbor_paths = {
        "crystalnn": (
            "representations/graphlet-crystalnn-shared-voronoi-recovery/"
            "graphlet_l1/nearest_neighbors.csv"),
        "voronoinn": (
            "representations/graphlet-voronoi-shared-crystalnn-recovery/"
            "graphlet_l1/nearest_neighbors.csv"),
    }
    worker_sensitivity = a.js(
        "representations/graphlet-neighbor-sensitivity/"
        "worker_count_sensitivity.json")
    worker_checks = worker_count_sensitivity_checks(
        worker_sensitivity,
        {
            "crystalnn": recoveries["crystal_shared"],
            "voronoinn": recoveries["voronoi_shared"],
        },
        canonical_neighbor_paths,
        {method: a.inputs[path]
         for method, path in canonical_neighbor_paths.items()},
        len(common_ids),
    )
    for detail, passed in worker_checks.items():
        a.check(f"graphlet worker-count sensitivity: {detail}", passed)
    partitions = {
        "voronoi_full": read_graphlet_partition(
            a, "graphlet-voronoi-dated-replay", "voronoinn", dated_voronoi_ids,
            voronoi_metadata, voronoi_metadata_path, crystal_ids_path,
            label_rows, label_hash, pca_ids_hash, analyzer_sha, False),
        "voronoi_shared": read_graphlet_partition(
            a, "graphlet-voronoi-shared-crystalnn-dated-replay", "voronoinn", dated_common_ids,
            voronoi_metadata, voronoi_metadata_path, crystal_ids_path,
            label_rows, label_hash, pca_ids_hash, analyzer_sha, True),
        "crystal_shared": read_graphlet_partition(
            a, "graphlet-crystalnn-shared-voronoi-dated-replay", "crystalnn", dated_common_ids,
            crystal_metadata, crystal_metadata_path, voronoi_ids_path,
            label_rows, label_hash, pca_ids_hash, analyzer_sha, True),
    }
    matched_control_names = {
        "raw": "production_pca_raw_same_protocol",
        "standardized": "production_pca_standardized_same_protocol",
    }
    matched_controls = {}
    for control, analysis_name in matched_control_names.items():
        matched_controls[control] = {
            "crystalnn": read_graphlet_partition(
                a, "graphlet-crystalnn-shared-voronoi-dated-replay",
                "crystalnn", dated_common_ids, crystal_metadata,
                crystal_metadata_path, voronoi_ids_path, label_rows,
                label_hash, pca_ids_hash, analyzer_sha, True,
                analysis_name=analysis_name),
            "voronoinn": read_graphlet_partition(
                a, "graphlet-voronoi-shared-crystalnn-dated-replay",
                "voronoinn", dated_common_ids, voronoi_metadata,
                voronoi_metadata_path, crystal_ids_path, label_rows,
                label_hash, pca_ids_hash, analyzer_sha, True,
                analysis_name=analysis_name),
        }
    analysis_documents = [
        run[artifact]
        for runs, artifact in ((recoveries, "recovery"), (partitions, "partition"))
        for run in runs.values()
    ] + [
        matched_controls[control][method]["partition"]
        for control in matched_control_names
        for method in ("crystalnn", "voronoinn")
    ]
    shared_analysis_provenance = (
        "production_feature_version", "production_labels_sha256",
        "production_pca_sha256", "production_pca_ids_sha256", "source_sha256",
        "neighbor_search_n_jobs", "neighbor_search_random_state",
        "exact_audit_max_workers", "packages",
    )
    a.check("graphlet neighbor sensitivity: common analysis environment and production provenance",
            all(all(document.get(field) == analysis_documents[0].get(field)
                    for field in shared_analysis_provenance)
                for document in analysis_documents[1:]))

    control_run_names = {
        "crystalnn": "graphlet-crystalnn-shared-voronoi-dated-replay",
        "voronoinn": "graphlet-voronoi-shared-crystalnn-dated-replay",
    }
    scalers = {
        method: a.js(f"representations/{run_name}/pca_subset_scaler.json")
        for method, run_name in control_run_names.items()
    }
    scaler_hashes = {
        method: a.inputs[f"representations/{run_name}/pca_subset_scaler.json"]
        for method, run_name in control_run_names.items()
    }
    scaler_valid = True
    for scaler in scalers.values():
        mean = scaler.get("mean") if isinstance(scaler, dict) else None
        scale = scaler.get("scale") if isinstance(scaler, dict) else None
        scaler_valid &= (
            isinstance(scaler, dict)
            and set(scaler) == {"mean", "scale", "n_samples"}
            and isinstance(mean, list) and isinstance(scale, list)
            and len(mean) == len(scale) == 32
            and np.isfinite(np.asarray(mean, dtype=float)).all()
            and np.isfinite(np.asarray(scale, dtype=float)).all()
            and np.all(np.asarray(scale, dtype=float) > 0)
            and scaler.get("n_samples") == len(dated_common_ids)
        )
    a.check("matched PCA controls: standardization scaler identity and validity",
            scaler_valid and scalers["crystalnn"] == scalers["voronoinn"]
            and set(scaler_hashes.values()) == {MATCHED_PCA_SCALER_SHA256})

    expected_control_results = {
        "raw": {
            "analysis_name": "production_pca_raw_same_protocol",
            "n_communities": 2094,
            "n_noise": 9339,
            "n_edges": 716156,
            "sigma": 1.4255766868591309,
            "comparison": {
                "ARI": 0.48890693803572416,
                "NMI_arithmetic": 0.9140181723972407,
                "same_community_pair_recall": 0.6498191597230659,
            },
            "headlines": {
                "1930s": {
                    "n_total": 1015, "n_birth": 354,
                    "n_same": 581, "n_cross": 1, "n_bridge": 0,
                    "share_birth": 0.34876847290640395,
                    "share_any_attachment": 0.5733990147783251,
                },
                "2010s": {
                    "n_total": 23881, "n_birth": 362,
                    "n_same": 19040, "n_cross": 2097, "n_bridge": 726,
                    "share_birth": 0.015158494200410369,
                    "share_any_attachment": 0.9154976759767179,
                },
            },
        },
        "standardized": {
            "analysis_name": "production_pca_standardized_same_protocol",
            "n_communities": 1994,
            "n_noise": 11591,
            "n_edges": 704985,
            "sigma": 0.7204698622226715,
            "comparison": {
                "ARI": 0.3306258141134392,
                "NMI_arithmetic": 0.865286531650013,
                "same_community_pair_recall": 0.48873044804527854,
            },
            "headlines": {
                "1930s": {
                    "n_total": 1015, "n_birth": 350,
                    "n_same": 566, "n_cross": 1, "n_bridge": 0,
                    "share_birth": 0.3448275862068966,
                    "share_any_attachment": 0.5586206896551724,
                },
                "2010s": {
                    "n_total": 23881, "n_birth": 330,
                    "n_same": 18776, "n_cross": 2074, "n_bridge": 649,
                    "share_birth": 0.013818516812528788,
                    "share_any_attachment": 0.9002554331895649,
                },
            },
        },
    }
    duplicate_hash_files = (
        "ids.json", "community_assignments.csv", "node_temporal_events.csv",
        "community_growth_by_decade.csv", "top_communities.json",
    )
    for control, expected in expected_control_results.items():
        left = matched_controls[control]["crystalnn"]
        right = matched_controls[control]["voronoinn"]
        analysis_name = expected["analysis_name"]
        hashes_identical = all(
            a.inputs[
                f"representations/{control_run_names['crystalnn']}/"
                f"{analysis_name}/{filename}"
            ]
            == a.inputs[
                f"representations/{control_run_names['voronoinn']}/"
                f"{analysis_name}/{filename}"
            ]
            for filename in duplicate_hash_files
        )
        a.check(f"matched PCA {control}: duplicate scientific outputs across graphlet runs",
                left["cohort"] == right["cohort"] == dated_common_ids
                and hashes_identical
                and np.array_equal(
                    left["assignments"].community.astype(int).to_numpy(),
                    right["assignments"].community.astype(int).to_numpy())
                and temporal_event_rows_identical(left["events"], right["events"])
                and left["top_communities"] == right["top_communities"]
                and left["growth"].equals(right["growth"])
                and temporal_rows_match(
                    left["exclusive_rows"], right["exclusive_rows"])
                and graphlet_science_documents_identical(
                    left["partition"], right["partition"])
                and graphlet_science_documents_identical(
                    left["exclusive_report"], right["exclusive_report"])
                and graphlet_science_documents_identical(
                    left["time_summary"], right["time_summary"]))
        partition = left["partition"]
        comparison = partition[
            "comparison_to_corrected_production_labels"
        ]["all_entries_noise_as_one_label"]
        counts_valid = (
            partition.get("name") == analysis_name
            and partition.get("n_entries") == len(dated_common_ids) == 146186
            and partition.get("n_communities") == expected["n_communities"]
            and partition.get("n_noise") == expected["n_noise"]
            and partition.get("n_edges") == expected["n_edges"]
            and np.isclose(partition.get("sigma"), expected["sigma"], rtol=0, atol=1e-15)
            and all(np.isclose(comparison.get(field), value, rtol=0, atol=1e-12)
                    for field, value in expected["comparison"].items())
        )
        headlines_valid = True
        for decade, values in expected["headlines"].items():
            row = left["exclusive_rows"].get(decade, {})
            headlines_valid &= (
                all(row.get(field) == value for field, value in values.items()
                    if field.startswith("n_"))
                and all(np.isclose(row.get(field), value, rtol=0, atol=1e-15)
                        for field, value in values.items()
                        if field.startswith("share_"))
                and row.get("n_same") + row.get("n_cross") + row.get("n_bridge")
                == round(values["share_any_attachment"] * values["n_total"])
            )
        a.check(f"matched PCA {control}: manuscript-critical partition and temporal values",
                counts_valid and headlines_valid)

    voronoi_full_recovery = recoveries["voronoi_full"]
    voronoi_shared_recovery = recoveries["voronoi_shared"]
    voronoi_full_partition = partitions["voronoi_full"]
    voronoi_shared_partition = partitions["voronoi_shared"]
    a.check("Voronoi full/shared: common restriction is a cohort no-op",
            voronoi_ids == common_ids
            and dated_voronoi_ids == dated_common_ids
            and voronoi_full_recovery["cohort"] == voronoi_shared_recovery["cohort"]
            and sha256(voronoi_full_recovery["cohort_path"])
            == sha256(voronoi_shared_recovery["cohort_path"])
            and voronoi_full_partition["cohort"] == voronoi_shared_partition["cohort"]
            and sha256(voronoi_full_partition["cohort_path"])
            == sha256(voronoi_shared_partition["cohort_path"]))
    a.check("Voronoi full/shared: identical selected-neighbor rows",
            selected_neighbor_rows_identical(
                voronoi_full_recovery["neighbors"],
                voronoi_shared_recovery["neighbors"],
            )
            and np.array_equal(
                voronoi_full_recovery["same"], voronoi_shared_recovery["same"])
            and voronoi_full_recovery["hits"] == voronoi_shared_recovery["hits"])
    voronoi_full_labels = (
        voronoi_full_partition["assignments"].community.astype(int).to_numpy()
    )
    voronoi_shared_labels = (
        voronoi_shared_partition["assignments"].community.astype(int).to_numpy()
    )
    a.check("Voronoi full/shared: identical partition membership and noise",
            partition_membership_and_noise_identical(
                voronoi_full_labels, voronoi_shared_labels))
    a.check("Voronoi full/shared: identical partition edge count",
            voronoi_full_partition["partition"].get("n_edges")
            == voronoi_shared_partition["partition"].get("n_edges"))
    a.check("Voronoi full/shared: identical temporal event rows",
            temporal_event_rows_identical(
                voronoi_full_partition["events"],
                voronoi_shared_partition["events"],
            ))
    a.check("Voronoi full/shared: identical temporal decade rows",
            temporal_rows_match(
                voronoi_full_partition["exclusive_rows"],
                voronoi_shared_partition["exclusive_rows"],
            ))

    tolerance_audit = a.js(
        "representations/louvain-tolerance-audit-final/tolerance_audit.json")
    canonical_voronoi_time_summary = a.js(
        "representations/graphlet-voronoi-shared-crystalnn-dated-replay/"
        "graphlet/graph_time_summary.json")
    tolerance_checks = louvain_tolerance_audit_checks(
        tolerance_audit,
        voronoi_shared_partition["partition"],
        voronoi_shared_partition["assignments"],
        canonical_voronoi_time_summary,
        {
            "features_sha256": voronoi_metadata.get("features_sha256"),
            "feature_ids_sha256": sha256(voronoi_ids_path),
            "neighbor_ids_sha256": sha256(
                voronoi_shared_partition["cohort_path"]),
            "neighbors_sha256": GRAPHLET_VORONOI_MATCHED_NEIGHBORS_SHA256,
            "production_labels_sha256": label_hash,
            "analyzer_sha256": analyzer_sha,
        },
    )
    for detail, passed in tolerance_checks.items():
        a.check(f"graphlet Louvain tolerance audit: {detail}", passed)

    family_report = a.js(
        "representations/graphlet-family-survival/summary.json")
    family_input_paths = {
        "analysis_source": (
            "analyze_graphlet_family_survival.py",
            "notes/feature_repair_2026_09/downstream/"
            "analyze_graphlet_family_survival.py"),
        "production_labels": (
            "../full_run_results/production/graph/community_assignments.csv",
            "notes/feature_repair_2026_09/full_run_results/production/graph/"
            "community_assignments.csv"),
        "current_family_manifest": (
            "community_evidence/current_family_manifest.json",
            "notes/feature_repair_2026_09/downstream/community_evidence/"
            "current_family_manifest.json"),
        "top20_checked_descriptors": (
            "community_evidence/renaissance_top20_checked_descriptors.csv",
            "notes/feature_repair_2026_09/downstream/community_evidence/"
            "renaissance_top20_checked_descriptors.csv"),
        "crystalnn_nearest_neighbors": (
            canonical_neighbor_paths["crystalnn"],
            "notes/feature_repair_2026_09/downstream/"
            + canonical_neighbor_paths["crystalnn"]),
        "voronoinn_nearest_neighbors": (
            canonical_neighbor_paths["voronoinn"],
            "notes/feature_repair_2026_09/downstream/"
            + canonical_neighbor_paths["voronoinn"]),
        "crystalnn_matched_dated_assignments": (
            "representations/graphlet-crystalnn-shared-voronoi-dated-replay/"
            "graphlet/community_assignments.csv",
            "notes/feature_repair_2026_09/downstream/representations/"
            "graphlet-crystalnn-shared-voronoi-dated-replay/graphlet/"
            "community_assignments.csv"),
        "voronoinn_matched_dated_assignments": (
            "representations/graphlet-voronoi-shared-crystalnn-dated-replay/"
            "graphlet/community_assignments.csv",
            "notes/feature_repair_2026_09/downstream/representations/"
            "graphlet-voronoi-shared-crystalnn-dated-replay/graphlet/"
            "community_assignments.csv"),
    }
    family_input_hashes = {}
    for name, (relative, _) in family_input_paths.items():
        family_input_hashes[name] = sha256(a.path(relative))
    saved_family_inputs = family_report.get("inputs")
    family_inputs_valid = (
        isinstance(saved_family_inputs, dict)
        and set(saved_family_inputs) == set(family_input_paths)
    )
    if family_inputs_valid:
        for name, (_, expected_suffix) in family_input_paths.items():
            entry = saved_family_inputs.get(name)
            family_inputs_valid &= (
                isinstance(entry, dict) and set(entry) == {"path", "sha256"}
                and isinstance(entry.get("path"), str)
                and entry["path"].replace("\\", "/").endswith(expected_suffix)
                and entry.get("sha256") == family_input_hashes[name]
            )
    a.check("graphlet family survival: exact input artifact manifest",
            family_report.get("analysis_version")
            == "graphlet-family-survival-v1" and family_inputs_valid)

    descriptors = a.csv(
        "community_evidence/renaissance_top20_checked_descriptors.csv")
    family_cohorts = family_report.get("cohorts", {})
    a.check("graphlet family survival: exact matched cohorts",
            family_cohorts.get("local_neighbor_recovery") == {
                "dated_only": False,
                "n_entries": len(common_ids),
                "ordered_ids_sha256": ordered_ids_sha256(common_ids),
            }
            and family_cohorts.get("independent_partitions") == {
                "dated_only": True,
                "exact_same_order_for_both_methods": True,
                "n_entries": len(dated_common_ids),
                "ordered_ids_sha256": ordered_ids_sha256(dated_common_ids),
            }
            and family_report.get("focus_communities")
            == [2662, 1802, 519, 74, 1294, 686, 1035, 2570])

    family_expectations = {
        "crystalnn_graphlet": {
            "critical": {
                2662: {
                    "local": {"evaluated_members": 807,
                              "exact_production_community_hits": 789},
                    "partition": {"matched_dated_members": 807,
                                  "alternate_noise_count": 44,
                                  "dominant_overlap_count": 305,
                                  "dominant_cluster_size_in_matched_cohort": 346},
                    "target_year": 1986,
                    "temporal": {
                        "family_members_in_matched_cohort": {
                            "pre_count": 0, "post_count": 504},
                        "family_members_captured_by_dominant_cluster": {
                            "pre_count": 0, "post_count": 224},
                        "full_dominant_alternate_cluster": {
                            "pre_count": 0, "post_count": 249},
                    },
                },
                74: {
                    "local": {"evaluated_members": 2839,
                              "exact_production_community_hits": 2598},
                    "partition": {"matched_dated_members": 2817,
                                  "alternate_noise_count": 204,
                                  "dominant_overlap_count": 192,
                                  "dominant_cluster_size_in_matched_cohort": 349},
                    "target_year": 1994,
                    "temporal": {
                        "family_members_in_matched_cohort": {
                            "pre_count": 140, "post_count": 995},
                        "family_members_captured_by_dominant_cluster": {
                            "pre_count": 0, "post_count": 89},
                        "full_dominant_alternate_cluster": {
                            "pre_count": 0, "post_count": 135},
                    },
                },
                1294: {
                    "local": {"evaluated_members": 903,
                              "exact_production_community_hits": 824},
                    "partition": {"matched_dated_members": 902,
                                  "alternate_noise_count": 67,
                                  "dominant_overlap_count": 259,
                                  "dominant_cluster_size_in_matched_cohort": 313},
                    "target_year": 1994,
                    "temporal": {
                        "family_members_in_matched_cohort": {
                            "pre_count": 17, "post_count": 368},
                        "family_members_captured_by_dominant_cluster": {
                            "pre_count": 0, "post_count": 96},
                        "full_dominant_alternate_cluster": {
                            "pre_count": 0, "post_count": 129},
                    },
                },
            },
            "summary": {"local_neighbor_queries": 14809,
                        "local_neighbor_hits": 13241,
                        "post_gt_pre": 19, "twofold_or_zero": 19,
                        "zero_baseline": 5},
        },
        "voronoinn_graphlet": {
            "critical": {
                2662: {
                    "local": {"evaluated_members": 807,
                              "exact_production_community_hits": 792},
                    "partition": {"matched_dated_members": 807,
                                  "alternate_noise_count": 61,
                                  "dominant_overlap_count": 291,
                                  "dominant_cluster_size_in_matched_cohort": 321},
                    "target_year": 1986,
                    "temporal": {
                        "family_members_in_matched_cohort": {
                            "pre_count": 0, "post_count": 504},
                        "family_members_captured_by_dominant_cluster": {
                            "pre_count": 0, "post_count": 222},
                        "full_dominant_alternate_cluster": {
                            "pre_count": 0, "post_count": 243},
                    },
                },
                74: {
                    "local": {"evaluated_members": 2839,
                              "exact_production_community_hits": 2592},
                    "partition": {"matched_dated_members": 2817,
                                  "alternate_noise_count": 162,
                                  "dominant_overlap_count": 309,
                                  "dominant_cluster_size_in_matched_cohort": 425},
                    "target_year": 1994,
                    "temporal": {
                        "family_members_in_matched_cohort": {
                            "pre_count": 140, "post_count": 995},
                        "family_members_captured_by_dominant_cluster": {
                            "pre_count": 7, "post_count": 146},
                        "full_dominant_alternate_cluster": {
                            "pre_count": 9, "post_count": 180},
                    },
                },
                1294: {
                    "local": {"evaluated_members": 903,
                              "exact_production_community_hits": 825},
                    "partition": {"matched_dated_members": 902,
                                  "alternate_noise_count": 70,
                                  "dominant_overlap_count": 239,
                                  "dominant_cluster_size_in_matched_cohort": 273},
                    "target_year": 1994,
                    "temporal": {
                        "family_members_in_matched_cohort": {
                            "pre_count": 17, "post_count": 368},
                        "family_members_captured_by_dominant_cluster": {
                            "pre_count": 0, "post_count": 81},
                        "full_dominant_alternate_cluster": {
                            "pre_count": 0, "post_count": 93},
                    },
                },
            },
            "summary": {"local_neighbor_queries": 14809,
                        "local_neighbor_hits": 13110,
                        "post_gt_pre": 20, "twofold_or_zero": 19,
                        "zero_baseline": 4},
        },
    }
    family_method_inputs = {
        "crystalnn_graphlet": (
            recoveries["crystal_shared"]["neighbors"],
            partitions["crystal_shared"]["assignments"]),
        "voronoinn_graphlet": (
            recoveries["voronoi_shared"]["neighbors"],
            partitions["voronoi_shared"]["assignments"]),
    }
    saved_family_methods = family_report.get("methods", {})
    a.check("graphlet family survival: exact method set",
            isinstance(saved_family_methods, dict)
            and set(saved_family_methods) == set(family_method_inputs))
    for method, (neighbors, assignments) in family_method_inputs.items():
        expected = family_expectations[method]
        method_checks = graphlet_family_method_checks(
            saved_family_methods.get(method, {}), neighbors, assignments,
            label_map, descriptors, expected["critical"], expected["summary"])
        for detail, passed in method_checks.items():
            a.check(f"graphlet family survival {method}: {detail}", passed)

    crystal_recovery = recoveries["crystal_shared"]
    voronoi_recovery = recoveries["voronoi_shared"]
    a.check("graphlet neighbor sensitivity: identical ordered recovery cohorts",
            crystal_recovery["cohort"] == voronoi_recovery["cohort"] == common_ids
            and sha256(crystal_recovery["cohort_path"]) == sha256(voronoi_recovery["cohort_path"]))
    recovery_transitions = Counter()
    crystal_neighbors = crystal_recovery["neighbors"]
    voronoi_neighbors = voronoi_recovery["neighbors"]
    for crystal_hit, voronoi_hit in zip(crystal_recovery["same"], voronoi_recovery["same"]):
        recovery_transitions[
            f"{'hit' if crystal_hit else 'miss'}_to_{'hit' if voronoi_hit else 'miss'}"] += 1
    recovery_transitions = {
        key: recovery_transitions[key] for key in
        ("hit_to_hit", "hit_to_miss", "miss_to_hit", "miss_to_miss")
    }
    a.check("graphlet neighbor sensitivity: direct recovery-transition arithmetic",
            sum(recovery_transitions.values()) == len(common_ids)
            and recovery_transitions["hit_to_hit"] + recovery_transitions["hit_to_miss"]
            == crystal_recovery["hits"]
            and recovery_transitions["hit_to_hit"] + recovery_transitions["miss_to_hit"]
            == voronoi_recovery["hits"])
    same_neighbor = int(np.sum(
        crystal_neighbors.neighbor_id.astype(int).to_numpy()
        == voronoi_neighbors.neighbor_id.astype(int).to_numpy()))
    same_neighbor_community = int(np.sum(
        crystal_neighbors.neighbor_community.astype(int).to_numpy()
        == voronoi_neighbors.neighbor_community.astype(int).to_numpy()))
    both_miss = ~crystal_recovery["same"] & ~voronoi_recovery["same"]
    both_miss_same_destination = int(np.sum(
        both_miss & (crystal_neighbors.neighbor_community.astype(int).to_numpy()
                     == voronoi_neighbors.neighbor_community.astype(int).to_numpy())))

    crystal_partition = partitions["crystal_shared"]
    voronoi_partition = partitions["voronoi_shared"]
    a.check("graphlet neighbor sensitivity: identical ordered partition cohorts",
            crystal_partition["cohort"] == voronoi_partition["cohort"] == dated_common_ids
            and sha256(crystal_partition["cohort_path"]) == sha256(voronoi_partition["cohort_path"]))
    crystal_labels = crystal_partition["assignments"].community.astype(int).to_numpy()
    voronoi_labels = voronoi_partition["assignments"].community.astype(int).to_numpy()
    both_assigned = (crystal_labels >= 0) & (voronoi_labels >= 0)
    noise_transitions = {
        "assigned_to_assigned": int(np.sum((crystal_labels >= 0) & (voronoi_labels >= 0))),
        "assigned_to_noise": int(np.sum((crystal_labels >= 0) & (voronoi_labels < 0))),
        "noise_to_assigned": int(np.sum((crystal_labels < 0) & (voronoi_labels >= 0))),
        "noise_to_noise": int(np.sum((crystal_labels < 0) & (voronoi_labels < 0))),
    }
    a.check("graphlet neighbor sensitivity: direct partition-transition arithmetic",
            sum(noise_transitions.values()) == len(dated_common_ids)
            and noise_transitions["assigned_to_assigned"]
            + noise_transitions["assigned_to_noise"] == int(np.sum(crystal_labels >= 0))
            and noise_transitions["assigned_to_assigned"]
            + noise_transitions["noise_to_assigned"] == int(np.sum(voronoi_labels >= 0)))
    all_scores = {
        "n_entries": len(crystal_labels),
        "ARI": adjusted_rand_score(crystal_labels, voronoi_labels),
        "NMI_arithmetic": normalized_mutual_info_score(
            crystal_labels, voronoi_labels, average_method="arithmetic"),
    }
    joint_scores = ({
        "n_entries": int(np.sum(both_assigned)),
        "ARI": adjusted_rand_score(crystal_labels[both_assigned], voronoi_labels[both_assigned]),
        "NMI_arithmetic": normalized_mutual_info_score(
            crystal_labels[both_assigned], voronoi_labels[both_assigned], average_method="arithmetic"),
    } if np.sum(both_assigned) >= 2 else None)

    sensitivity = a.js("representations/graphlet-neighbor-sensitivity/summary.json")
    paired_recovery = sensitivity.get("paired_recovery", {})
    saved_recovery = paired_recovery.get("recovery", {})
    saved_identity = paired_recovery.get("selected_neighbor_identity", {})
    saved_community = paired_recovery.get("selected_neighbor_reference_community", {})
    a.check("graphlet neighbor sensitivity: paired recovery summary",
            paired_recovery.get("cohort", {}).get("n_entries") == len(common_ids)
            and paired_recovery.get("cohort", {}).get("ordered_ids_sha256")
            == sha256(crystal_recovery["cohort_path"])
            and saved_recovery.get("crystalnn", {}).get("hits") == crystal_recovery["hits"]
            and saved_recovery.get("crystalnn", {}).get("misses") == len(common_ids) - crystal_recovery["hits"]
            and np.isclose(saved_recovery.get("crystalnn", {}).get("rate"),
                           crystal_recovery["hits"] / len(common_ids), rtol=0, atol=1e-12)
            and saved_recovery.get("voronoinn", {}).get("hits") == voronoi_recovery["hits"]
            and saved_recovery.get("voronoinn", {}).get("misses") == len(common_ids) - voronoi_recovery["hits"]
            and np.isclose(saved_recovery.get("voronoinn", {}).get("rate"),
                           voronoi_recovery["hits"] / len(common_ids), rtol=0, atol=1e-12)
            and np.isclose(saved_recovery.get("voronoinn_minus_crystalnn"),
                           (voronoi_recovery["hits"] - crystal_recovery["hits"]) / len(common_ids),
                           rtol=0, atol=1e-12)
            and np.isclose(saved_recovery.get("voronoinn_minus_crystalnn_percentage_points"),
                           100 * (voronoi_recovery["hits"] - crystal_recovery["hits"]) / len(common_ids),
                           rtol=0, atol=1e-12)
            and np.isclose(saved_recovery.get("crystalnn", {}).get("ann_exact_nearest_fraction"),
                           crystal_recovery["recovery"]["exact_query_audit"]["ann_exact_nearest_fraction"],
                           rtol=0, atol=1e-12)
            and np.isclose(saved_recovery.get("voronoinn", {}).get("ann_exact_nearest_fraction"),
                           voronoi_recovery["recovery"]["exact_query_audit"]["ann_exact_nearest_fraction"],
                           rtol=0, atol=1e-12)
            and paired_recovery.get("outcome_transitions_crystalnn_to_voronoinn") == recovery_transitions
            and saved_identity.get("same_id") == same_neighbor
            and saved_identity.get("different_id") == len(common_ids) - same_neighbor
            and np.isclose(saved_identity.get("same_id_fraction"), same_neighbor / len(common_ids), rtol=0, atol=1e-12)
            and saved_community.get("same_community_id") == same_neighbor_community
            and saved_community.get("different_community_id") == len(common_ids) - same_neighbor_community
            and np.isclose(saved_community.get("same_community_id_fraction"),
                           same_neighbor_community / len(common_ids), rtol=0, atol=1e-12)
            and saved_community.get("both_miss_same_destination_community") == both_miss_same_destination)

    saved_partition = sensitivity.get("common_cohort_partition", {})
    saved_partition_methods = saved_partition.get("methods", {})
    saved_agreement = saved_partition.get("agreement", {})
    a.check("graphlet neighbor sensitivity: paired partition summary",
            saved_partition.get("cohort", {}).get("n_entries") == len(dated_common_ids)
            and saved_partition.get("cohort", {}).get("ordered_ids_sha256")
            == sha256(crystal_partition["cohort_path"])
            and saved_partition.get("noise_transitions_crystalnn_to_voronoinn") == noise_transitions
            and all(saved_partition_methods.get(method, {}).get("n_communities")
                    == partitions[f"{'crystal' if method == 'crystalnn' else 'voronoi'}_shared"]["partition"]["n_communities"]
                    and saved_partition_methods.get(method, {}).get("n_noise")
                    == partitions[f"{'crystal' if method == 'crystalnn' else 'voronoi'}_shared"]["partition"]["n_noise"]
                    and saved_partition_methods.get(method, {}).get("n_edges")
                    == partitions[f"{'crystal' if method == 'crystalnn' else 'voronoi'}_shared"]["partition"]["n_edges"]
                    and saved_partition_methods.get(method, {}).get("partition_cohort_entries")
                    == len(dated_common_ids)
                    and np.isclose(saved_partition_methods.get(method, {}).get("noise_fraction"),
                                   partitions[f"{'crystal' if method == 'crystalnn' else 'voronoi'}_shared"]["partition"]["n_noise"]
                                   / len(dated_common_ids), rtol=0, atol=1e-12)
                    for method in ("crystalnn", "voronoinn"))
            and saved_agreement.get("all_entries_noise_as_one_label", {}).get("n_entries") == all_scores["n_entries"]
            and np.isclose(saved_agreement.get("all_entries_noise_as_one_label", {}).get("ARI"), all_scores["ARI"], rtol=0, atol=1e-12)
            and np.isclose(saved_agreement.get("all_entries_noise_as_one_label", {}).get("NMI_arithmetic"), all_scores["NMI_arithmetic"], rtol=0, atol=1e-12)
            and ((saved_agreement.get("jointly_assigned_only") is None and joint_scores is None)
                 or (joint_scores is not None
                     and saved_agreement.get("jointly_assigned_only", {}).get("n_entries") == joint_scores["n_entries"]
                     and np.isclose(saved_agreement.get("jointly_assigned_only", {}).get("ARI"), joint_scores["ARI"], rtol=0, atol=1e-12)
                     and np.isclose(saved_agreement.get("jointly_assigned_only", {}).get("NMI_arithmetic"), joint_scores["NMI_arithmetic"], rtol=0, atol=1e-12))))
    feature_metadata_by_method = {
        "crystalnn": crystal_metadata,
        "voronoinn": voronoi_metadata,
    }
    feature_coverage_valid = True
    for method, metadata in feature_metadata_by_method.items():
        coverage = saved_partition_methods.get(method, {}).get("feature_coverage", {})
        n_requested = metadata.get("n_requested")
        n_successful = metadata.get("n_successful")
        n_failed = metadata.get("n_failed")
        expected_legacy = (method == "crystalnn"
                           and metadata.get("neighbor_method") is None
                           and metadata.get("graphlet_feature_version") is None)
        feature_coverage_valid &= (
            coverage.get("n_requested") == n_requested
            and coverage.get("n_successful") == n_successful
            and coverage.get("n_failed") == n_failed
            and np.isclose(coverage.get("success_fraction"), n_successful / n_requested,
                           rtol=0, atol=1e-12)
            and coverage.get("legacy_crystalnn_metadata_compatibility") is expected_legacy
            and np.isclose(
                saved_partition_methods.get(method, {}).get(
                    "partition_cohort_fraction_of_feature_successes"),
                len(dated_common_ids) / n_successful, rtol=0, atol=1e-12)
        )
    a.check("graphlet neighbor sensitivity: paired feature coverage", feature_coverage_valid)

    expected_temporal = {
        "crystalnn": crystal_partition["exclusive_rows"],
        "voronoinn": voronoi_partition["exclusive_rows"],
    }
    saved_temporal = sensitivity.get("common_cohort_temporal", {})
    saved_temporal_methods = saved_temporal.get("methods", {})
    temporal_rows_valid = all(
        temporal_rows_match(saved_temporal_methods.get(method, {}).get("by_decade"), rows)
        for method, rows in expected_temporal.items()
    )
    expected_headlines = {}
    headlines_valid = True
    for method, rows in expected_temporal.items():
        expected_headlines[method] = {}
        for decade in ("1930s", "2010s"):
            row = rows.get(decade)
            if row is None:
                headlines_valid = False
                continue
            attachment = row["n_same"] + row["n_cross"] + row["n_bridge"]
            expected = {
                "n_entries": row["n_total"],
                "n_birth": row["n_birth"],
                "birth_share": row["share_birth"],
                "n_any_attachment": attachment,
                "any_attachment_share": row["share_any_attachment"],
            }
            expected_headlines[method][decade] = expected
            saved = saved_temporal_methods.get(method, {}).get("headline", {}).get(decade, {})
            headlines_valid &= (
                all(saved.get(field) == expected[field] for field in
                    ("n_entries", "n_birth", "n_any_attachment"))
                and all(np.isclose(saved.get(field), expected[field], rtol=0, atol=1e-12)
                        for field in ("birth_share", "any_attachment_share"))
            )
    expected_differences = {}
    differences_valid = headlines_valid
    saved_differences = saved_temporal.get(
        "headline_differences_voronoinn_minus_crystalnn", {})
    for decade in ("1930s", "2010s"):
        if not headlines_valid:
            break
        crystal_headline = expected_headlines["crystalnn"][decade]
        voronoi_headline = expected_headlines["voronoinn"][decade]
        birth_difference = (voronoi_headline["birth_share"]
                            - crystal_headline["birth_share"])
        attachment_difference = (voronoi_headline["any_attachment_share"]
                                 - crystal_headline["any_attachment_share"])
        expected_differences[decade] = {
            "birth_share": birth_difference,
            "birth_share_percentage_points": 100 * birth_difference,
            "any_attachment_share": attachment_difference,
            "any_attachment_share_percentage_points": 100 * attachment_difference,
        }
        saved = saved_differences.get(decade, {})
        differences_valid &= (set(saved) == set(expected_differences[decade])
                              and all(np.isclose(saved.get(field), value, rtol=0, atol=1e-12)
                                      for field, value in expected_differences[decade].items()))
    a.check("graphlet neighbor sensitivity: full temporal rows and headline differences",
            saved_temporal.get("cohort", {}).get("n_entries") == len(dated_common_ids)
            and saved_temporal.get("cohort", {}).get("ordered_ids_sha256")
            == sha256(crystal_partition["cohort_path"])
            and temporal_rows_valid and headlines_valid and differences_valid)

    saved_controls = sensitivity.get("matched_pca_controls", {})
    saved_control_records = saved_controls.get("controls", {})
    saved_controls_valid = (
        saved_controls.get("cohort", {}).get("n_entries") == len(dated_common_ids)
        and saved_controls.get("cohort", {}).get("ordered_ids_sha256")
        == sha256(matched_controls["raw"]["crystalnn"]["variant_ids_path"])
        and saved_controls.get("standardization_scaler") == {
            "n_samples": len(dated_common_ids),
            "n_features": 32,
            "byte_identical_across_graphlet_runs": True,
        }
        and set(saved_control_records) == set(expected_control_results)
    )
    expected_identity_fields = {
        "ids", "assignments", "temporal_events",
        "community_growth_by_decade", "top_communities",
    }
    saved_identity = saved_controls.get(
        "duplicate_science_identity_across_graphlet_runs", {})
    saved_controls_valid &= (
        set(saved_identity) == set(expected_control_results)
        and all(
            isinstance(saved_identity.get(control), dict)
            and set(saved_identity[control]) == expected_identity_fields
            and all(value is True for value in saved_identity[control].values())
            for control in expected_control_results
        )
    )
    for control, expected in expected_control_results.items():
        run = matched_controls[control]["crystalnn"]
        partition = run["partition"]
        record = saved_control_records.get(control, {})
        expected_headline = {}
        for decade in ("1930s", "2010s"):
            row = run["exclusive_rows"][decade]
            expected_headline[decade] = {
                "n_entries": row["n_total"],
                "n_birth": row["n_birth"],
                "birth_share": row["share_birth"],
                "n_any_attachment": (
                    row["n_same"] + row["n_cross"] + row["n_bridge"]),
                "any_attachment_share": row["share_any_attachment"],
            }
        saved_controls_valid &= (
            record.get("analysis_name") == expected["analysis_name"]
            and record.get("n_entries") == len(dated_common_ids)
            and record.get("n_communities") == expected["n_communities"]
            and record.get("n_noise") == expected["n_noise"]
            and np.isclose(
                record.get("noise_fraction"),
                expected["n_noise"] / len(dated_common_ids), rtol=0, atol=1e-15)
            and record.get("n_edges") == expected["n_edges"]
            and np.isclose(record.get("sigma"), expected["sigma"], rtol=0, atol=1e-15)
            and nested_numbers_match(
                record.get("comparison_to_corrected_production_labels"),
                partition.get("comparison_to_corrected_production_labels"))
            and temporal_rows_match(
                record.get("by_decade"), run["exclusive_rows"])
            and nested_numbers_match(record.get("headline"), expected_headline)
        )
    a.check("graphlet neighbor sensitivity: matched PCA control summary",
            saved_controls_valid)

    manifest_relatives = {
        "production_labels": "../full_run_results/production/graph/community_assignments.csv",
        "crystal_feature_metadata": "representations/graphlet_features/feature_preparation.json",
        "crystal_feature_ids": "representations/graphlet_features/features.ids.json",
        "voronoi_feature_metadata": "representations/graphlet_voronoi_features/feature_preparation.json",
        "voronoi_feature_ids": "representations/graphlet_voronoi_features/features.ids.json",
        "crystal_recovery": "representations/graphlet-crystalnn-shared-voronoi-recovery/graphlet_l1/recovery.json",
        "crystal_recovery_ids": "representations/graphlet-crystalnn-shared-voronoi-recovery/cohort.ids.json",
        "crystal_neighbors": "representations/graphlet-crystalnn-shared-voronoi-recovery/graphlet_l1/nearest_neighbors.csv",
        "crystal_exact_audit": "representations/graphlet-crystalnn-shared-voronoi-recovery/graphlet_l1/exact_query_audit.json",
        "voronoi_recovery": "representations/graphlet-voronoi-shared-crystalnn-recovery/graphlet_l1/recovery.json",
        "voronoi_recovery_ids": "representations/graphlet-voronoi-shared-crystalnn-recovery/cohort.ids.json",
        "voronoi_neighbors": "representations/graphlet-voronoi-shared-crystalnn-recovery/graphlet_l1/nearest_neighbors.csv",
        "voronoi_exact_audit": "representations/graphlet-voronoi-shared-crystalnn-recovery/graphlet_l1/exact_query_audit.json",
        "crystal_partition": "representations/graphlet-crystalnn-shared-voronoi-dated-replay/graphlet/partition.json",
        "crystal_partition_ids": "representations/graphlet-crystalnn-shared-voronoi-dated-replay/graphlet/ids.json",
        "crystal_assignments": "representations/graphlet-crystalnn-shared-voronoi-dated-replay/graphlet/community_assignments.csv",
        "crystal_temporal_events": "representations/graphlet-crystalnn-shared-voronoi-dated-replay/graphlet/node_temporal_events.csv",
        "crystal_exclusive_by_decade": "representations/graphlet-crystalnn-shared-voronoi-dated-replay/graphlet/exclusive_by_decade.json",
        "voronoi_partition": "representations/graphlet-voronoi-shared-crystalnn-dated-replay/graphlet/partition.json",
        "voronoi_partition_ids": "representations/graphlet-voronoi-shared-crystalnn-dated-replay/graphlet/ids.json",
        "voronoi_assignments": "representations/graphlet-voronoi-shared-crystalnn-dated-replay/graphlet/community_assignments.csv",
        "voronoi_temporal_events": "representations/graphlet-voronoi-shared-crystalnn-dated-replay/graphlet/node_temporal_events.csv",
        "voronoi_exclusive_by_decade": "representations/graphlet-voronoi-shared-crystalnn-dated-replay/graphlet/exclusive_by_decade.json",
        "crystalnn_pca_subset_scaler": "representations/graphlet-crystalnn-shared-voronoi-dated-replay/pca_subset_scaler.json",
        "voronoinn_pca_subset_scaler": "representations/graphlet-voronoi-shared-crystalnn-dated-replay/pca_subset_scaler.json",
    }
    for method, run_name in control_run_names.items():
        for control, analysis_name in matched_control_names.items():
            for artifact, filename in {
                "partition": "partition.json",
                "ids": "ids.json",
                "assignments": "community_assignments.csv",
                "temporal_events": "node_temporal_events.csv",
                "exclusive_by_decade": "exclusive_by_decade.json",
                "graph_time_summary": "graph_time_summary.json",
                "community_growth_by_decade": "community_growth_by_decade.csv",
                "top_communities": "top_communities.json",
            }.items():
                manifest_relatives[f"{method}_pca_{control}_{artifact}"] = (
                    f"representations/{run_name}/{analysis_name}/{filename}"
                )
    manifest = sensitivity.get("inputs")
    manifest_valid = isinstance(manifest, dict) and set(manifest) == set(manifest_relatives)
    if manifest_valid:
        for name, relative in manifest_relatives.items():
            entry = manifest.get(name)
            local_hash = a.inputs.get(relative)
            suffix = "/".join(Path(relative).parts[-3:])
            manifest_valid &= (
                isinstance(entry, dict) and set(entry) == {"path", "sha256"}
                and isinstance(entry.get("path"), str)
                and entry["path"].replace("\\", "/").endswith(suffix)
                and local_hash is not None and entry.get("sha256") == local_hash
            )
    a.check("graphlet neighbor sensitivity: exact input artifact manifest", manifest_valid)

    production_fields = (
        "production_feature_version", "production_labels_sha256",
        "production_pca_sha256", "production_pca_ids_sha256", "source_sha256",
    )
    expected_production_provenance = {
        field: crystal_recovery["recovery"].get(field) for field in production_fields
    }
    a.check("graphlet neighbor sensitivity: summary method provenance",
            sensitivity.get("schema_version") == 3
            and sensitivity.get("validation", {}).get("status") == "passed"
            and sensitivity.get("validation", {}).get("common_feature_intersection_entries")
            == len(common_ids)
            and sensitivity.get("validation", {}).get("common_dated_feature_intersection_entries")
            == len(dated_common_ids)
            and sensitivity.get("validation", {}).get("matched_pca_controls_validated")
            == ["raw", "standardized"]
            and sensitivity.get("validation", {}).get(
                "legacy_crystalnn_feature_metadata_compatibility") is True
            and sensitivity.get("validation", {}).get(
                "voronoinn_requires_explicit_method_and_version") is True
            and sensitivity.get("production_provenance") == expected_production_provenance
            and sensitivity.get("analysis_execution") == {
                "neighbor_search_n_jobs": GRAPHLET_ANALYSIS_N_JOBS,
                "neighbor_search_random_state": 42,
                "exact_audit_max_workers": GRAPHLET_EXACT_AUDIT_MAX_WORKERS,
            }
            and expected_production_provenance["production_feature_version"] == VERSION
            and expected_production_provenance["production_labels_sha256"] == label_hash
            and expected_production_provenance["production_pca_ids_sha256"] == pca_ids_hash
            and expected_production_provenance["source_sha256"] == analyzer_sha
            and sensitivity.get("graphlet_methods", {}).get("voronoinn", {}).get("effective_graphlet_feature_version")
            == GRAPHLET_FEATURE_VERSIONS["voronoinn"]
            and sensitivity.get("graphlet_methods", {}).get("voronoinn", {}).get("source_graphlet_feature_version")
            == GRAPHLET_FEATURE_VERSIONS["voronoinn"]
            and sensitivity.get("graphlet_methods", {}).get("voronoinn", {}).get("source_neighbor_method")
            == "voronoinn"
            and sensitivity.get("graphlet_methods", {}).get("voronoinn", {}).get(
                "legacy_crystalnn_metadata_compatibility") is False
            and sensitivity.get("graphlet_methods", {}).get("voronoinn", {}).get("neighbor_settings")
            == GRAPHLET_NEIGHBOR_SETTINGS["voronoinn"]
            and sensitivity.get("graphlet_methods", {}).get("crystalnn", {}).get("effective_graphlet_feature_version")
            == GRAPHLET_FEATURE_VERSIONS["crystalnn"]
            and sensitivity.get("graphlet_methods", {}).get("crystalnn", {}).get("source_graphlet_feature_version") is None
            and sensitivity.get("graphlet_methods", {}).get("crystalnn", {}).get("source_neighbor_method") is None
            and sensitivity.get("graphlet_methods", {}).get("crystalnn", {}).get(
                "legacy_crystalnn_metadata_compatibility") is True
            and sensitivity.get("graphlet_methods", {}).get("crystalnn", {}).get("neighbor_settings")
            == GRAPHLET_NEIGHBOR_SETTINGS["crystalnn"]
            and all(sensitivity.get("graphlet_methods", {}).get(method, {}).get("representation")
                    == GRAPHLET_REPRESENTATION for method in ("crystalnn", "voronoinn")))

    return {
        "voronoi_feature_successes": len(voronoi_ids),
        "voronoi_feature_failures": voronoi_metadata["n_failed"],
        "voronoi_calibration_successes": voronoi_calibration_successes,
        "recovery_common_pool": len(common_ids),
        "partition_common_pool": len(dated_common_ids),
        "recoveries": recoveries,
        "partitions": partitions,
        "matched_pca_controls": matched_controls,
        "summary": sensitivity,
        "crystal_metadata_path": crystal_metadata_path,
        "crystal_ids_path": crystal_ids_path,
        "voronoi_metadata_path": voronoi_metadata_path,
        "voronoi_ids_path": voronoi_ids_path,
    }


def verify_alab_mp_targets(a):
    """Verify the self-contained public A-Lab target-geometry validation."""
    package_dir = a.base / ALAB_MP_TARGET_ROOT
    package_symlinks = sorted(
        str(path.relative_to(a.base)) for path in package_dir.rglob("*")
        if path.is_symlink()
    )
    package_files = sorted(
        path.relative_to(a.base) for path in package_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    for relative in package_files:
        a.path(relative)
    recorded_package_files = {str(path) for path in package_files}
    a.check(
        "A-Lab MP targets: complete package file inventory hashed",
        not package_symlinks
        and ALAB_MP_TARGET_REQUIRED_FILES <= recorded_package_files
        and all(relative in a.inputs for relative in recorded_package_files),
        n_files=len(recorded_package_files),
    )

    checksum_entries = {}
    checksum_format_valid = True
    for line in (package_dir / "SHA256SUMS").read_text().splitlines():
        fields = line.split(None, 1)
        if len(fields) != 2 or not is_sha256(fields[0]):
            checksum_format_valid = False
            continue
        checksum_name = fields[1].strip()
        if checksum_name in checksum_entries or Path(checksum_name).is_absolute():
            checksum_format_valid = False
            continue
        checksum_entries[checksum_name] = fields[0]
    checksum_expected = {
        str(path.relative_to(ALAB_MP_TARGET_ROOT))
        for path in package_files
        if path not in {
            ALAB_MP_TARGET_ROOT / "SHA256SUMS",
            ALAB_MP_TARGET_ROOT / "verification/package_verification.json",
        }
    }
    a.check(
        "A-Lab MP targets: canonical SHA256SUMS covers every immutable payload",
        checksum_format_valid
        and a.inputs[str(ALAB_MP_TARGET_ROOT / "SHA256SUMS")]
        == ALAB_MP_TARGET_SHA256SUMS_SHA256
        and a.inputs[str(
            ALAB_MP_TARGET_ROOT / "verification/package_verification.json"
        )] == ALAB_MP_TARGET_VERIFICATION_SHA256
        and set(checksum_entries) == checksum_expected
        and all(
            checksum_entries[name]
            == a.inputs[str(ALAB_MP_TARGET_ROOT / name)]
            for name in checksum_expected
        ),
        n_payload_files=len(checksum_expected),
    )

    def rel(suffix):
        return str(ALAB_MP_TARGET_ROOT / suffix)

    targets = a.csv(rel("source/alab_targets.csv"))
    records = a.csv(rel("output/target_projection_records.csv"))
    failures = a.js(rel("output/failures.json"))
    projection_summary = a.js(rel("output/target_projection_summary.json"))
    sensitivity = a.js(rel("verification/outcome_sensitivity.json"))
    geometry = a.js(rel("verification/primary_materials_geometry_equivalence.json"))
    independent = a.js(rel("verification/independent_verification.json"))
    package_verification = a.js(rel("verification/package_verification.json"))
    summary_manifest = a.js(rel("source/mp_2022_10_28_summary_source_manifest.json"))
    materials_manifest = a.js(rel("source/mp_2022_10_28_materials_source_manifest.json"))
    summary_docs = a.js(rel("source/mp_2022_10_28_summary_target_docs.json"))
    materials_docs = a.js(rel("source/mp_2022_10_28_materials_target_docs.json"))

    target_ids = targets.mp_id.astype(str).tolist()
    record_ids = records.mp_id.astype(str).tolist()
    expected_outcomes = Counter({
        "made": 36,
        "not_obtained": 15,
        "offline_recovery": 2,
        "inconclusive": 4,
    })
    target_outcomes = Counter(targets.corrected_outcome.astype(str))
    record_outcomes = Counter(records.corrected_outcome.astype(str))
    a.check(
        "A-Lab MP targets: 57 unique target projections and no failures",
        len(target_ids) == len(set(target_ids)) == 57
        and record_ids == target_ids
        and records.mp_id.is_unique
        and failures == []
        and records.corrected_outcome.astype(str).tolist()
        == targets.corrected_outcome.astype(str).tolist(),
    )
    a.check(
        "A-Lab MP targets: corrected outcome counts",
        target_outcomes == record_outcomes == expected_outcomes
        and projection_summary.get("outcome_counts") == dict(expected_outcomes)
        and sensitivity.get("outcome_counts") == dict(expected_outcomes),
        counts=dict(record_outcomes),
    )

    feature_path = a.path(rel("output/features.npy"))
    projected_path = a.path(rel("output/features_pca.npy"))
    refined_path = a.path(rel("inputs/refined_features_pca.npy"))
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    projected = np.load(projected_path, mmap_mode="r", allow_pickle=False)
    refined = np.load(refined_path, mmap_mode="r", allow_pickle=False)
    a.check(
        "A-Lab MP targets: complete finite packaged arrays",
        features.shape == (57, 213)
        and projected.shape == (57, 32)
        and refined.shape == (42, 32)
        and np.isfinite(features).all()
        and np.isfinite(projected).all()
        and np.isfinite(refined).all(),
    )

    in_basin = boolean(records.in_basin).to_numpy()
    outcomes = records.corrected_outcome.astype(str).to_numpy()
    made = outcomes == "made"
    not_obtained = outcomes == "not_obtained"
    basin_table = [
        [int(np.sum(in_basin & made)), int(np.sum(~in_basin & made))],
        [int(np.sum(in_basin & not_obtained)),
         int(np.sum(~in_basin & not_obtained))],
    ]
    saved_year_tables = [
        sensitivity.get("comparisons_by_observation_year", {})
        .get(year, {}).get("made_vs_not_obtained", {})
        .get("table_rows_success_failure_cols_in_basin_frontier")
        for year in ("2022", "2023")
    ]
    a.check(
        "A-Lab MP targets: made/not-obtained basin table",
        basin_table == [[28, 8], [4, 11]]
        and projection_summary.get("primary_made_vs_not_obtained", {}).get(
            "in_basin_table_rows_success_failure_cols_in_out") == basin_table
        and saved_year_tables == [basin_table, basin_table]
        and package_verification.get("primary_basin_table") == basin_table,
        table=basin_table,
    )

    def source_manifest_valid(document, collection):
        rows = document.get("targets")
        objects = document.get("source_objects")
        if not isinstance(rows, list) or not isinstance(objects, list):
            return False
        object_keys = [obj.get("key") for obj in objects if isinstance(obj, dict)]
        return (
            document.get("source")
            == f"s3://materialsproject-build/collections/2022-10-28/{collection}/"
            and document.get("n_targets") == document.get("n_found")
            == document.get("n_structures") == 57
            and document.get("missing") == []
            and len(rows) == 57
            and [str(row.get("mp_id")) for row in rows] == target_ids
            and all(row.get("structure_present") is True for row in rows)
            and [str(row.get("outcome")) for row in rows]
            == targets.corrected_outcome.astype(str).tolist()
            and len(objects) == len(set(object_keys)) == 36
            and all(
                isinstance(obj, dict)
                and {"key", "bytes", "sha256", "etag", "last_modified",
                     "content_length"} <= set(obj)
                and isinstance(obj.get("bytes"), int)
                and obj["bytes"] > 0
                and obj["bytes"] == obj["content_length"]
                and is_sha256(obj.get("sha256"))
                for obj in objects
            )
            and all(row.get("source_object") in set(object_keys) for row in rows)
        )

    a.check(
        "A-Lab MP targets: public snapshot coverage and object provenance",
        source_manifest_valid(summary_manifest, "summary")
        and source_manifest_valid(materials_manifest, "materials")
        and set(summary_docs) == set(materials_docs) == set(target_ids),
    )

    def structures_have_same_geometry(summary_document, materials_document):
        left = summary_document.get("structure", {})
        right = materials_document.get("structure", {})
        left_sites = left.get("sites")
        right_sites = right.get("sites")
        return (
            left.get("lattice", {}).get("matrix")
            == right.get("lattice", {}).get("matrix")
            and isinstance(left_sites, list)
            and isinstance(right_sites, list)
            and len(left_sites) == len(right_sites)
            and all(
                left_site.get("species") == right_site.get("species")
                and left_site.get("abc") == right_site.get("abc")
                and left_site.get("xyz") == right_site.get("xyz")
                and left_site.get("properties") == right_site.get("properties")
                for left_site, right_site in zip(left_sites, right_sites)
            )
        )

    geometry_fields = (
        "exact_lattice_matrix",
        "exact_ordered_species",
        "exact_fractional_coordinates",
        "exact_cartesian_coordinates",
        "exact_site_properties",
        "structure_dict_equal_after_removing_summary_only_empty_properties_and_pbc",
    )
    geometry_rows = geometry.get("targets")
    geometry_by_id = {
        str(row.get("mp_id")): row for row in geometry_rows
    } if isinstance(geometry_rows, list) else {}
    direct_geometry_matches = sum(
        structures_have_same_geometry(summary_docs[mid], materials_docs[mid])
        for mid in target_ids
    )
    a.check(
        "A-Lab MP targets: all public primary/summary geometries agree",
        direct_geometry_matches == 57
        and len(geometry_by_id) == 57
        and set(geometry_by_id) == set(target_ids)
        and geometry.get("n_targets") == 57
        and geometry.get("n_exact_geometry_matches") == 57
        and geometry.get("clean_structure_dict_equal_n") == 57
        and geometry.get("raw_structure_dict_equal_n") == 0
        and all(
            all(geometry_by_id[mid].get(field) is True for field in geometry_fields)
            and geometry_by_id[mid].get("n_sites")
            == len(summary_docs[mid]["structure"]["sites"])
            for mid in target_ids
        ),
        exact_matches=direct_geometry_matches,
    )

    package_hash = lambda suffix: a.inputs[rel(suffix)]
    a.check(
        "A-Lab MP targets: geometry report is bound to packaged public records",
        geometry.get("summary_collection", {}).get("docs_sha256")
        == package_hash("source/mp_2022_10_28_summary_target_docs.json")
        and geometry.get("summary_collection", {}).get("manifest_sha256")
        == package_hash("source/mp_2022_10_28_summary_source_manifest.json")
        and geometry.get("primary_materials_collection", {}).get("docs_sha256")
        == package_hash("source/mp_2022_10_28_materials_target_docs.json")
        and geometry.get("primary_materials_collection", {}).get("manifest_sha256")
        == package_hash("source/mp_2022_10_28_materials_source_manifest.json")
        and geometry.get("script_sha256")
        == package_hash("code/check_primary_geometry_equivalence.py"),
    )

    basis_path = a.path("reference_basis/projection_basis.npz")
    score_path = a.path("accessibility/structural_accessibility_summary.json")
    refined_records_path = a.path("alab/alab_validation_records.csv")
    provenance = projection_summary.get("provenance", {})
    a.check(
        "A-Lab MP targets: projection provenance hashes",
        provenance.get("feature_version") == VERSION
        and provenance.get("docs_sha256")
        == package_hash("source/mp_2022_10_28_summary_target_docs.json")
        and provenance.get("source_summary_sha256")
        == package_hash("source/mp_2022_10_28_summary_source_manifest.json")
        and provenance.get("targets_sha256")
        == package_hash("source/alab_targets.csv")
        and provenance.get("basis_sha256") == sha256(basis_path)
        and provenance.get("score_summary_sha256") == sha256(score_path)
        and provenance.get("refined_records_sha256") == sha256(refined_records_path)
        and provenance.get("refined_pca_sha256")
        == package_hash("inputs/refined_features_pca.npy")
        and provenance.get("script_sha256")
        == package_hash("code/run_target_projection.py")
        and provenance.get("neighbor_settings")
        == provenance.get("basis_metadata", {}).get("neighbor_settings"),
    )

    sensitivity_provenance = sensitivity.get("provenance", {})
    a.check(
        "A-Lab MP targets: sensitivity report provenance hashes",
        sensitivity_provenance.get("target_records_sha256")
        == package_hash("output/target_projection_records.csv")
        and sensitivity_provenance.get("paired_records_sha256")
        == package_hash("output/target_vs_refinement_records.csv")
        and sensitivity_provenance.get("refined_records_sha256")
        == sha256(refined_records_path)
        and sensitivity_provenance.get("score_summary_sha256") == sha256(score_path)
        and sensitivity_provenance.get("script_sha256")
        == package_hash("code/outcome_sensitivity.py"),
    )

    package_checks = package_verification.get("checks")
    a.check(
        "A-Lab MP targets: packaged arithmetic verifier passed",
        package_verification.get("all_passed") is True
        and isinstance(package_checks, list)
        and package_verification.get("n_checks") == len(package_checks) == 332
        and package_verification.get("n_manifested_files_checked") == 28
        and all(check.get("passed") is True for check in package_checks)
        and package_verification.get("basis_sha256") == sha256(basis_path)
        and package_verification.get("score_summary_sha256") == sha256(score_path)
        and package_verification.get("verifier_sha256")
        == package_hash("code/verify_packaged_results.py"),
    )

    independent_checks = independent.get("checks")
    primary_recomputed = independent.get("primary_recomputed", {})
    run_to_package_files = {
        "source/alab_mp_snapshot_2022_10_28_docs.json":
            "source/mp_2022_10_28_summary_target_docs.json",
        "source/alab_mp_snapshot_2022_10_28_summary.json":
            "source/mp_2022_10_28_summary_source_manifest.json",
        "source/alab_targets.csv": "source/alab_targets.csv",
        "source/extracted_structures_manifest.json":
            "provenance/original_extracted_structures_manifest.json",
        "output/SHA256SUMS": "provenance/original_output_SHA256SUMS",
        "output/failures.json": "output/failures.json",
        "output/features.npy": "output/features.npy",
        "output/features_pca.npy": "output/features_pca.npy",
        "output/target_projection_records.csv": "output/target_projection_records.csv",
        "output/target_projection_summary.json": "output/target_projection_summary.json",
        "output/target_vs_refinement_records.csv": "output/target_vs_refinement_records.csv",
        "code/run_target_projection.py": "code/run_target_projection.py",
        "run.sbatch": "provenance/tacc_run_job3471430.sbatch",
    }
    independent_hashes = independent.get("files_sha256", {})
    archival_context = independent.get("archival_context", {})
    a.check(
        "A-Lab MP targets: independent verifier passed and matches package",
        independent.get("all_passed") is True
        and independent.get("n_checks") == 283
        and isinstance(independent_checks, list)
        and len(independent_checks) == 283
        and all(check.get("passed") is True for check in independent_checks)
        and independent.get("source_object_checks") == 36
        and independent.get("per_target_parse_checks") == 57
        and primary_recomputed.get("made_n") == 36
        and primary_recomputed.get("not_obtained_n") == 15
        and primary_recomputed.get("made_in_basin") == 28
        and primary_recomputed.get("not_obtained_in_basin") == 4
        and archival_context.get("all_original_hashed_files_preserved_in_package") is True
        and archival_context.get("all_mapped_hashes_reverified_when_packaged") is True
        and archival_context.get("original_to_package_path") == run_to_package_files
        and set(independent_hashes) == set(run_to_package_files)
        and all(
            independent_hashes.get(run_name) == package_hash(package_name)
            for run_name, package_name in run_to_package_files.items()
        ),
    )

    return {
        "n_targets": len(records),
        "outcome_counts": dict(record_outcomes),
        "basin_table": basin_table,
        "public_geometry_matches": direct_geometry_matches,
        "independent_checks": len(independent_checks),
        "package_files_hashed": len(recorded_package_files),
    }


def verify_cutoff_trained_retrospective(a, prod, source_rows, basis, label_hash):
    """Verify the occupancy-aware cutoff package through its standalone audit."""
    package_dir = a.base / CUTOFF_TRAINED_ROOT

    def rel(suffix):
        return str(CUTOFF_TRAINED_ROOT / suffix)

    summary_path = rel("cutoff_trained_retrospective_summary.json")
    summary = a.js(summary_path)
    protocol = summary.get("protocol", {})
    inputs = summary.get("inputs", {})
    cutoff_documents = summary.get("cutoffs", {})
    required_input_additions = {
        "occupancy_flags", "formula_conventions", "index_loader", "wrapper",
    }
    required_cutoff_additions = {
        "reference_composition_counts", "earliest_year_tie_audit",
    }
    current_schema = (
        required_input_additions <= set(inputs)
        and isinstance(cutoff_documents, dict)
        and bool(cutoff_documents)
        and all(required_cutoff_additions <= set(saved)
                for saved in cutoff_documents.values())
    )
    if not current_schema:
        raise ValueError(
            "FINAL-OUTPUT INTEGRATION REQUIRED: cutoff_trained is the legacy "
            "raw-formula package, not the occupancy-aware final run"
        )

    cutoffs = sorted(int(value) for value in cutoff_documents)
    execution = summary.get("execution", {})
    job_id = execution.get("slurm_job_id")
    if job_id in (None, ""):
        raise ValueError("Occupancy-aware cutoff package lacks its Slurm job ID")
    slurm_path = rel(f"slurm-{job_id}.out")
    reporting_manifest_path = rel("reporting/reporting_manifest.json")
    independent_path = rel("verification/independent_verification.json")

    required_files, archive_files, selected_files = cutoff_trained_file_sets(
        package_dir
    )
    package_symlinks = sorted(
        str(path.relative_to(a.base)) for path in package_dir.rglob("*")
        if path.is_symlink()
    )
    package_files = sorted(
        path.relative_to(a.base) for path in package_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    package_names = {str(path) for path in package_files}
    for relative in package_files:
        a.path(relative)
    a.check(
        "cutoff-trained: dynamic release inventory and complete hash coverage",
        not package_symlinks
        and package_names == required_files
        and all(name in a.inputs for name in package_names),
        n_files=len(package_names),
    )

    software = summary.get("software", {})
    integer_protocol_fields = {
        "max_year", "pca_components", "knn_k", "min_component_size",
        "min_community_size", "seed", "n_boot", "n_perm", "n_jobs",
    }
    textual_protocol_fields = {
        "scaler", "pca", "partition", "future_assignment", "formula_rule",
        "per_formula_identity", "per_formula_selection",
    }
    a.check(
        "cutoff-trained: current protocol, software and execution are explicit",
        protocol.get("cutoffs") == cutoffs
        and all(isinstance(protocol.get(key), int)
                for key in integer_protocol_fields)
        and all(protocol.get(key, 0) > 0 for key in integer_protocol_fields - {"seed"})
        and isinstance(protocol.get("seed"), int)
        and isinstance(protocol.get("louvain_resolution"), (int, float))
        and protocol.get("louvain_resolution", 0) > 0
        and isinstance(protocol.get("partial_occupancy_fraction_tolerance"),
                       (int, float))
        and 0 < protocol.get("partial_occupancy_fraction_tolerance", 0) < 1
        and all(isinstance(protocol.get(key), str) and protocol[key].strip()
                for key in textual_protocol_fields)
        and isinstance(protocol.get("formula_references"), list)
        and bool(protocol["formula_references"])
        and set(software) >= {
            "python", "numpy", "pandas", "scikit_learn", "networkx",
            "scipy", "pymatgen",
        }
        and all(isinstance(value, str) and value for value in software.values())
        and isinstance(execution.get("argv"), list) and bool(execution["argv"])
        and execution.get("hostname") not in (None, ""),
    )

    repo_root = Path(__file__).resolve().parents[1]
    expected_input_keys = {
        "features", "feature_metadata", "sample_assignments", "icsd_index",
        "occupancy_flags", "producer", "formula_conventions", "index_loader",
        "statistical_module", "wrapper", "production_pca", "production_labels",
        *(f"external_{source}_{kind}"
          for source in SOURCES
          for kind in ("features", "records", "feature_metadata")),
    }
    local_inputs = {
        "feature_metadata": a.base / "../full_run_results/production/summary.json",
        "sample_assignments": a.base / "../full_run_results/production/sample_assignments.csv",
        "icsd_index": a.base / "inputs/ICSD_index.csv",
        "occupancy_flags": a.base / "formula_layers/icsd_occupancy_flags.csv",
        "producer": repo_root / "scripts/analyze_cutoff_trained_retrospective.py",
        "formula_conventions": repo_root / "scripts/formula_conventions.py",
        "index_loader": repo_root / "scripts/analyze_synthesis_retrodiction.py",
        "statistical_module": repo_root / "notes/review_2026_08/retrospective_quadrant.py",
        "wrapper": repo_root / "notes/feature_repair_2026_09/downstream/run_cutoff_trained_retrospective.sh",
        "production_pca": a.base / "inputs/features_pca.npy",
        "production_labels": a.base / "../full_run_results/production/graph/community_assignments.csv",
    }
    for source, (directory, stem, _) in SOURCES.items():
        local_inputs[f"external_{source}_records"] = (
            a.base / f"external/{directory}/{stem}_frontier_records.csv"
        )
        local_inputs[f"external_{source}_feature_metadata"] = (
            a.base / f"external/{directory}/feature_metadata.json"
        )
    missing_raw_features = {
        "features", *(f"external_{source}_features" for source in SOURCES)
    }
    local_hashes_valid = all(
        path.exists()
        and inputs.get(name, {}).get("sha256") == sha256(path)
        for name, path in local_inputs.items()
    )
    a.check(
        "cutoff-trained: every local input and TACC-only feature hash is bound",
        set(inputs) == expected_input_keys
        and local_hashes_valid
        and all(is_sha256(inputs.get(name, {}).get("sha256"))
                for name in missing_raw_features)
        and inputs.get("features", {}).get("sha256")
        == basis["reference_files"]["icsd_features"]["sha256"]
        and inputs.get("sample_assignments", {}).get("sha256")
        == basis["reference_files"]["sample_assignments"]["sha256"]
        and inputs.get("production_pca", {}).get("sha256")
        == basis["reference_files"]["icsd_pca"]["sha256"]
        and inputs.get("production_labels", {}).get("sha256") == label_hash,
        n_inputs=len(inputs),
    )

    audit_summary = summary.get("production_filter_audit", {})
    audit_record = audit_summary.get("assignments", {})
    reported_audit = a.csv(rel(audit_record["path"]))
    reported_labels = prod.community.to_numpy(dtype=np.int64)
    reproduced_labels = reported_audit.reproduced_community.to_numpy(dtype=np.int64)
    a.check(
        "cutoff-trained: production partition reconstruction is exact",
        list(reported_audit.columns)
        == ["icsd_id", "reported_community", "reproduced_community"]
        and reported_audit.icsd_id.astype(int).tolist()
        == prod.icsd_id.astype(int).tolist()
        and np.array_equal(reported_audit.reported_community.to_numpy(np.int64),
                           reported_labels)
        and partition_membership_and_noise_identical(reported_labels,
                                                     reproduced_labels)
        and audit_record.get("sha256") == a.inputs[rel(audit_record["path"])]
        and audit_summary.get("n_train") == len(prod)
        and audit_summary.get("n_assigned") == int(np.sum(reproduced_labels >= 0))
        and audit_summary.get("n_outliers") == int(np.sum(reproduced_labels < 0)),
    )

    year_values = pd.to_numeric(prod.year, errors="coerce")
    cutoff_results = {}
    expected_external_ids = []
    expected_external_formulas = []
    expected_external_sources = []
    for source in SOURCES:
        rows = source_rows[source]
        expected_external_ids.extend(rows.material_id.astype(str))
        expected_external_formulas.extend(rows.reduced_formula.astype(str))
        expected_external_sources.extend([source] * len(rows))

    for cutoff in cutoffs:
        saved = cutoff_documents[str(cutoff)]
        outputs = saved.get("outputs", {})
        expected_output_names = {
            "training_partition", "cutoff_map", "heldout_entries",
            "external_classifications",
        }
        a.check(
            f"cutoff-trained T{cutoff}: complete output hashes",
            set(outputs) == expected_output_names
            and all(record.get("sha256") == a.inputs[rel(record.get("path"))]
                    for record in outputs.values()),
        )
        train = a.csv(rel(outputs["training_partition"]["path"]))
        held = a.csv(rel(outputs["heldout_entries"]["path"]))
        external = a.csv(rel(outputs["external_classifications"]["path"]))
        map_path = a.path(rel(outputs["cutoff_map"]["path"]))
        expected_train = prod.loc[
            year_values.notna() & year_values.gt(0) & year_values.le(cutoff),
            ["icsd_id", "year"],
        ]
        expected_held = prod.loc[
            year_values.gt(cutoff) & year_values.le(protocol["max_year"]),
            ["icsd_id", "year"],
        ]
        required_held_columns = {
            "icsd_id", "year", "formula", "has_partial_occupancy",
            "formula_identity", "stratum_coarse", "stratum_anon", "in_basin",
        }
        a.check(
            f"cutoff-trained T{cutoff}: exact dated cohorts and new formula schema",
            train.icsd_id.is_unique and held.icsd_id.is_unique
            and train.icsd_id.astype(int).tolist()
            == expected_train.icsd_id.astype(int).tolist()
            and train.year.astype(int).tolist() == expected_train.year.astype(int).tolist()
            and held.icsd_id.astype(int).tolist()
            == expected_held.icsd_id.astype(int).tolist()
            and held.year.astype(int).tolist() == expected_held.year.astype(int).tolist()
            and required_held_columns <= set(held)
            and set(pd.to_numeric(held.has_partial_occupancy,
                                  errors="coerce").dropna().astype(int)) <= {0, 1}
            and held.loc[held.formula.ne(""), "formula_identity"].ne("").all()
            and len(train) == saved.get("n_train_dated")
            and len(held) == saved.get("n_heldout_dated")
            and held.formula.ne("").sum() == saved.get("n_heldout_parseable_formula")
            and held.loc[held.formula.ne(""), "formula_identity"].nunique()
            == saved.get("n_first_postcutoff_entry_per_formula")
            and set(saved.get("reference_sizes", {}))
            == set(saved.get("reference_composition_counts", {}))
            == set(saved.get("analyses", {}))
            and set(saved.get("earliest_year_tie_audit", {})) == {
                "n_formula_identities",
                "n_with_multiple_entries_in_earliest_year",
                "n_with_mixed_basin_status_in_earliest_year",
                "deterministic_tie_rule",
                "by_formula_reference",
            },
        )
        partition = saved.get("partition", {})
        with np.load(map_path, allow_pickle=False) as cutoff_map:
            communities = np.asarray(cutoff_map["communities"], dtype=np.int64)
            radii = np.asarray(cutoff_map["radii_p95"], dtype=np.float64)
            sizes = np.asarray(cutoff_map["community_sizes"], dtype=np.int64)
            centroids = np.asarray(cutoff_map["centroids"], dtype=np.float64)
            scaler_mean = np.asarray(cutoff_map["scaler_mean"], dtype=np.float64)
            pca_components = np.asarray(cutoff_map["pca_components"], dtype=np.float64)
        a.check(
            f"cutoff-trained T{cutoff}: dynamic map and partition dimensions",
            len(communities) == partition.get("n_communities")
            and len(train) == partition.get("n_train")
            and partition.get("knn_k") == protocol["knn_k"]
            and partition.get("resolution") == protocol["louvain_resolution"]
            and partition.get("min_component_size") == protocol["min_component_size"]
            and partition.get("min_community_size") == protocol["min_community_size"]
            and partition.get("seed") == protocol["seed"]
            and communities.shape == radii.shape == sizes.shape
            and centroids.shape == (len(communities), protocol["pca_components"])
            and pca_components.shape
            == (protocol["pca_components"], len(scaler_mean))
            and np.isfinite(radii).all() and np.isfinite(centroids).all()
            and np.isfinite(scaler_mean).all() and np.isfinite(pca_components).all(),
        )
        a.check(
            f"cutoff-trained T{cutoff}: exact external cohort identity",
            external.source.astype(str).tolist() == expected_external_sources
            and external.material_id.astype(str).tolist() == expected_external_ids
            and external.formula.astype(str).tolist() == expected_external_formulas
            and set(saved.get("external_source_rates", {})) == set(SOURCES),
        )
        held_flags = boolean(held.in_basin)
        primary = saved["analyses"]["all_year_le_T_index"]["per_entry"]["quadrant"]
        cutoff_results[str(cutoff)] = {
            "n_train": len(train),
            "n_heldout": len(held),
            "n_communities": len(communities),
            "heldout_in_basin": int(held_flags.sum()),
            "primary_joint": primary["in_basin_and_match"],
            "primary_enrichment": primary[
                "enrichment_ratio_obs_over_independence"
            ],
        }

    rates = a.csv(rel("reporting/cutoff_trained_rates.csv"))
    shared = a.csv(rel("reporting/cutoff_trained_shared_strata.csv"))
    joint = a.csv(rel("reporting/cutoff_trained_joint.csv"))
    reporting_manifest = a.js(reporting_manifest_path)
    summary_hash = a.inputs[summary_path]
    reporting_output_names = {
        "cutoff_trained_composition_summary.json",
        "cutoff_trained_joint.csv",
        "cutoff_trained_rates.csv",
        "cutoff_trained_shared_strata.csv",
    }
    expected_reporting_hashes = {
        name: a.inputs[rel(f"reporting/{name}")] for name in reporting_output_names
    }
    n_references = len(next(iter(cutoff_documents.values()))["analyses"])
    a.check(
        "cutoff-trained: dynamic reporting inventory and hashes",
        len(rates) == (1 + len(SOURCES)) * len(cutoffs)
        and len(shared) == 2 * len(SOURCES) * len(cutoffs)
        and len(joint) == 2 * n_references * len(cutoffs)
        and not joint.duplicated(["cutoff", "reference", "unit"]).any()
        and reporting_manifest.get("source_summary_sha256") == summary_hash
        and reporting_manifest.get("producer_sha256")
        == sha256(repo_root / "scripts/summarize_cutoff_trained_retrospective.py")
        and reporting_manifest.get("outputs") == expected_reporting_hashes,
    )

    # Rerun the standalone verifier. This is the transitive scientific check:
    # it reconstructs occupancy-aware identities, hybrid precedent flags,
    # anonymous strata, every deterministic analysis, and the same-year tie
    # sensitivity directly from packaged rows and immutable local inputs.
    packaged_independent = a.js(independent_path)
    with tempfile.TemporaryDirectory(prefix="cutoff-global-verify-") as temporary:
        fresh_path = Path(temporary) / "independent_verification.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/verify_cutoff_trained_package.py"),
                "--package-dir", str(package_dir),
                "--report", str(fresh_path),
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        fresh_independent = (
            json.loads(fresh_path.read_text()) if fresh_path.exists() else {}
        )
        reports_identical = (
            fresh_path.exists()
            and fresh_path.read_bytes() == (a.base / independent_path).read_bytes()
        )
    a.check(
        "cutoff-trained: standalone verifier reruns byte-identically",
        completed.returncode == 0
        and fresh_independent.get("schema_version") == 2
        and fresh_independent.get("all_passed") is True
        and fresh_independent.get("failures") == []
        and fresh_independent == packaged_independent
        and reports_identical,
        stdout=completed.stdout[-2000:], stderr=completed.stderr[-2000:],
    )

    slurm_text = a.path(slurm_path).read_text()
    actual_canonical_hashes = {
        "summary": summary_hash,
        "reporting_manifest": a.inputs[reporting_manifest_path],
        "slurm": a.inputs[slurm_path],
        "independent_verification": a.inputs[independent_path],
    }
    configured_canonical_hashes = {
        "summary": CUTOFF_TRAINED_SUMMARY_SHA256,
        "reporting_manifest": CUTOFF_TRAINED_REPORTING_MANIFEST_SHA256,
        "slurm": CUTOFF_TRAINED_SLURM_SHA256,
        "independent_verification": CUTOFF_TRAINED_INDEPENDENT_VERIFICATION_SHA256,
    }
    if (
        CUTOFF_TRAINED_SLURM_PATH is None
        or not all(is_sha256(value) for value in configured_canonical_hashes.values())
    ):
        raise ValueError(
            "FINAL-OUTPUT INTEGRATION REQUIRED: set the final Slurm path and four "
            "canonical cutoff package hashes only after the final package passes "
            f"verification; path_candidate={slurm_path}; "
            f"hash_candidates={json.dumps(actual_canonical_hashes, sort_keys=True)}"
        )
    a.check(
        "cutoff-trained: canonical hashes and final Slurm log",
        slurm_path == CUTOFF_TRAINED_SLURM_PATH
        and actual_canonical_hashes == configured_canonical_hashes
        and "Traceback" not in slurm_text
        and "Production filter audit:" in slurm_text
        and all(f"T={cutoff}:" in slurm_text for cutoff in cutoffs),
    )
    return {
        "package_files_hashed": len(package_names),
        "archive_files": len(archive_files),
        "selected_files": len(selected_files),
        "cutoffs": cutoff_results,
        "reporting_rows": {
            "rates": len(rates), "shared_strata": len(shared), "joint": len(joint),
        },
        "independent_checks": packaged_independent["assertion_counts"]["total_checks"],
    }


def verify(a, array_root):
    prod = a.csv("../full_run_results/production/graph/community_assignments.csv")
    summary = a.js("../full_run_results/production/summary.json")
    basis = a.js("reference_basis/projection_basis_provenance.json")
    thresholds = a.js("reference_basis/community_thresholds.json")
    ids = set(prod.icsd_id.astype(int))
    labels = prod.set_index("icsd_id").community.to_dict()
    years = prod.set_index("icsd_id").year.to_dict()
    nonnoise = set(prod.loc[prod.community.ge(0), "icsd_id"].astype(int))
    label_hash = sha256(a.base / "../full_run_results/production/graph/community_assignments.csv")
    a.check("production accepted/requested/failure accounting",
            len(prod) == len(ids) == summary["sample_size_featurized"] == 167392
            and len(prod) + summary["n_failures"] == 181362,
            accepted=len(prod), failed=summary["n_failures"], archive_requested=181362)
    a.check("production current labels and version",
            summary["feature_version"] == basis["feature_version"] == VERSION
            and len(nonnoise) == 154025 and len(prod) - len(nonnoise) == 13367
            and len(set(labels.values()) - {-1}) == 2939
            and basis["reference_files"]["community_assignments"]["sha256"] == label_hash)
    a.check("saved PCA alignment provenance", basis["row_ids_and_years_verified"]
            and basis["fit_transform_reproduces_saved_pca"]
            and basis["transform_vs_saved_allclose"]
            and basis["n_reference_rows"] == len(prod))

    alab_mp_targets = verify_alab_mp_targets(a)

    events = a.csv("../full_run_results/production/time/node_temporal_events.csv")
    a.check("production temporal identity and label join",
            set(events.icsd_id) == ids and len(events) == len(ids)
            and events.icsd_id.map(labels).eq(events.community).all()
            and events.icsd_id.map(years).astype(str).eq(events.year.astype(str)).all())
    a.check("noise retained as temporal outlier",
            events.event_type.eq("outlier").eq(events.community.lt(0)).all())
    events["exclusive"] = exclusive(events)
    expected = a.csv("temporal/production/fig1_exclusive_by_decade.csv")
    for r in expected.to_dict("records"):
        sub = events[events.decade.eq(r["decade"])]
        counts = sub.exclusive.value_counts()
        a.check(f"production exclusive counts {r['decade']}", len(sub) == r["n_total"]
                and all(int(counts.get(k, 0)) == r[f"n_{k}"] for k in ["birth", "same", "cross", "bridge", "outlier"])
                and sum(int(counts.get(k, 0)) for k in ["birth", "same", "cross", "bridge", "outlier"]) == len(sub))

    dated = prod[prod.year.ne("") & prod.community.ge(0)].copy()
    dated["year"] = dated.year.astype(int)
    size_by_community = dated.groupby("community").size().to_dict()
    survey = a.csv("temporal/production/renaissance_survey_all.csv")
    top = a.csv("community_evidence/renaissance_top20_checked_descriptors.csv")
    a.check("renaissance eligibility and current top-twenty identities",
            set(survey.community) == {c for c,n in size_by_community.items() if n >= 50}
            and len(survey) == 516 and len(top) == 20
            and list(top.community) == list(survey.community.head(20)))
    for r in top.to_dict("records"):
        hist = dated[dated.community.eq(r["community"])].year
        pre = int(((hist > r["event_year"]-10) & (hist <= r["event_year"])).sum())
        post = int(((hist > r["event_year"]) & (hist <= r["event_year"]+10)).sum())
        score = post*post/pre if pre else post
        a.check(f"renaissance rank {r['rank']}: membership, windows and score",
                len(hist) == r["n_dated_survey_members"] and pre == r["n_pre"] and post == r["n_post"]
                and int(prod.community.eq(r["community"]).sum()) == r["n_all_members"]
                and np.isclose(score,r["score"],atol=1e-8,rtol=0))
    null = a.js("temporal/production/renaissance_null.json")
    a.check("renaissance null population", null["population"]["n_entries_permuted"] == len(dated)
            and null["population"]["n_communities_eligible"] == len(survey))
    for run in ["production_n200", "production_n1000"]:
        v = null["runs"][run]
        a.check(f"{run}: all twenty scores exceed per-rank null maxima",
                len(v["observed_top_k_vs_null"]) == 20
                and all(z["observed_score"] > z["null_max"] and z["exceeds_null_max"]
                        for z in v["observed_top_k_vs_null"]))
    probes = a.js("community_evidence/targeted_family_probe_candidates.json")
    for probe in probes["probes"]:
        for candidate in probe["candidate_communities"]:
            c, t = candidate["community"], probe["event_year"]
            hist = dated[dated.community.eq(c)].year
            a.check(f"historical probe {probe['probe']}, community {c}: whole-community windows",
                    len(hist) == candidate["dated_community_members"]
                    and int(((hist > t-10) & (hist <= t)).sum()) == candidate["n_pre"]
                    and int(((hist > t) & (hist <= t+10)).sum()) == candidate["n_post"])

    source_rows = {}
    for name, (directory, stem, target_n) in SOURCES.items():
        prefix = f"external/{directory}"
        frame = a.csv(f"{prefix}/{stem}_frontier_records.csv")
        attempts = a.csv(f"{prefix}/attempted_cohort.csv")
        meta = a.js(f"{prefix}/feature_metadata.json")
        report = a.js(f"{prefix}/{stem}_frontier_summary.json")
        failures = a.js(f"{prefix}/{stem}_frontier_failures.json")
        feature_ids = list(map(str, a.js(f"{prefix}/feature_ids.json")))
        key = "zip_member" if name == "MatterGen" else "material_id"
        keys, attempted = frame[key].astype(str).tolist(), attempts[key].astype(str).tolist()
        failed_keys = [str(r[key]) for r in failures]
        a.check(f"{name}: complete attempted/success/failure identity partition",
                len(frame) == target_n == report["n_featurized"] == meta["n_rows"]
                and len(keys) == len(set(keys)) and len(attempted) == len(set(attempted))
                and set(keys).isdisjoint(failed_keys)
                and set(keys) | set(failed_keys) == set(attempted)
                and len(failures) == report["n_failures"]
                and len(frame) + len(failures) == report["n_records_loaded"],
                attempted=len(attempts), successful=len(frame), failed=len(failures))
        a.check(f"{name}: feature order, metadata and reference hashes",
                keys == feature_ids and frame.feature_row.tolist() == list(range(len(frame)))
                and meta["feature_version"] == report["feature_version"] == VERSION
                and meta["neighbor_settings"] == basis["neighbor_settings"]
                and meta["n_features"] == 213 and meta["wl_iters"] == 3
                and report["threshold_file_sha256"] == sha256(a.base / "reference_basis/community_thresholds.json")
                and report["projection_basis"]["sha256"] == sha256(a.base / "reference_basis/projection_basis.npz"))
        distances = frame.nearest_centroid_distance.to_numpy(float)
        rad = frame.community_threshold_p95.to_numpy(float)
        joined_radius = frame.assigned_community.astype(str).map(thresholds["per_community_p95_threshold"]).to_numpy(float)
        a.check(f"{name}: finite populated projections and per-community full-map flags",
                np.isfinite(frame[["nearest_centroid_distance", "community_threshold_p95", "pca1", "pca2"]].to_numpy(float)).all()
                and (distances >= 0).all() and (rad > 0).all()
                and np.allclose(rad, joined_radius, rtol=0, atol=1e-12)
                and np.array_equal(boolean(frame.in_basin), distances <= rad)
                and np.array_equal(boolean(frame.outlier_like), distances > rad)
                and int(boolean(frame.in_basin).sum()) == report["n_in_basin"])
        a.check(f"{name}: original successful cohort preserved",
                not report["successful_id_changes"]["only_historical_successful"]
                and not report["successful_id_changes"]["newly_successful"])
        source_rows[name] = frame

    cutoff_trained = verify_cutoff_trained_retrospective(
        a, prod, source_rows, basis, label_hash,
    )

    full = a.js("calibration/fig3c_corrected_summary.json")
    flags = a.csv("calibration/external_cutoff_flags.csv")
    cutoff_maps = {}
    for cut in full["cutoffs"]:
        t = cut["cutoff"]
        held = a.csv(f"calibration/heldout_{t}.csv")
        wanted = {i for i,y in years.items() if y != "" and int(y) > t}
        a.check(f"T{t}: full dated held-out identities including original noise",
                set(held.icsd_id) == wanted and held.icsd_id.is_unique
                and held.icsd_id.map(labels).eq(held.fullmap_community).all())
        a.check(f"T{t}: held-out distance/threshold flags", np.isfinite(held[["distance", "threshold"]].to_numpy(float)).all()
                and np.array_equal(boolean(held.in_basin), held.distance.le(held.threshold)))
        block = cut["matchings"]["coarse"]
        a.rate(f"T{t}: held-out summary", boolean(held.in_basin), block["icsd_unmatched"])
        retro_rows = a.csv(f"synthesis/split_{t}/post_cutoff_accessibility_records.csv")
        joined = held.merge(retro_rows[["cif_id", "is_in_basin"]], left_on="icsd_id", right_on="cif_id", validate="one_to_one")
        a.check(f"T{t}: calibration agrees with independent retrodiction table", len(joined) == len(held)
                and boolean(joined.in_basin).eq(boolean(joined.is_in_basin)).all())
        cutoff_maps[t] = held.set_index("icsd_id").in_basin.astype(int).to_dict()
        for name, frame in source_rows.items():
            f = flags[flags.cutoff.eq(t) & flags.source.eq(name)]
            a.check(f"T{t} {name}: external row-order alignment", list(f.material_id.astype(str)) == list(frame.material_id.astype(str)))
            a.rate(f"T{t} {name}: full-cohort rate", boolean(f.in_basin), block["by_source"][name]["unmatched"])

    reference = {
        scale_invariant_formula_key(formula)
        for formula in a.csv("synthesis/split_1980/first_report_formulas.csv").reduced_formula.astype(str)
    }
    prior = a.js("prior/formula_synth_prior_summary.json")
    quadrant = a.csv("prior/quadrant_assignments.csv")
    a.check("formula reference count", len(reference) == prior["icsd_reference_size"] == 80944)
    for name, frame in source_rows.items():
        q = quadrant[quadrant.source.eq(name)]
        key = "zip_member" if name == "MatterGen" else "material_id"
        a.check(f"{name}: full quadrant row-order and formula identity",
                list(q[key].astype(str)) == list(frame[key].astype(str))
                and list(q.reduced_formula.astype(str)) == list(frame.reduced_formula.astype(str)))
        match = frame.reduced_formula.astype(str).map(scale_invariant_formula_key).isin(reference).to_numpy()
        inside = boolean(frame.in_basin).to_numpy()
        calc = {"in_basin_and_formula_match": int(np.sum(inside & match)),
                "frontier_and_formula_match": int(np.sum(~inside & match)),
                "in_basin_and_no_formula_match": int(np.sum(inside & ~match)),
                "frontier_and_no_formula_match": int(np.sum(~inside & ~match))}
        a.check(f"{name}: all four quadrants independently reconstructed", calc == prior["sources"][name]["quadrants"]
                and np.array_equal(boolean(q.formula_match), match)
                and np.array_equal(boolean(q.in_basin), inside), n=len(frame), counts=calc)

    candidates = a.csv("candidates/representative_candidates.csv")
    a.check("candidate table has ten current entries in each quadrant", len(candidates) == 40
            and candidates.groupby("quadrant").size().eq(10).all())
    for row in candidates.to_dict("records"):
        f = source_rows[row["source"]]
        key = "zip_member" if row["source"] == "MatterGen" else "material_id"
        key_value = row["zip_member"] if key == "zip_member" else row["material_id"]
        match = f[f[key].astype(str).eq(str(key_value))]
        if len(match) != 1:
            a.check("candidate unique source identity",False)
        r = match.iloc[0]
        inside = bool(r.nearest_centroid_distance <= r.community_threshold_p95)
        formula_match = scale_invariant_formula_key(str(r.reduced_formula)) in reference
        q = ("in_basin" if inside else "frontier") + "_and_" + ("formula_match" if formula_match else "no_formula_match")
        a.check(f"candidate {row['quadrant']} rank {row['rank']}: current source, label and axes",
                r.assigned_community == row["assigned_community"] and q == row["quadrant"]
                and np.isclose(float(row["d_over_tau"]),float(r.nearest_centroid_distance)/float(r.community_threshold_p95),atol=1e-12,rtol=0)
                and str(r.reduced_formula) == str(row["reduced_formula"]))

    retro = a.js("retrospective/retrospective_quadrant_summary.json")
    a.check("all retrospective cutoffs complete", set(retro["cutoffs"]) == {"1990", "2000", "2010"})
    for t, cut in retro["cutoffs"].items():
        for unit in ["per_entry", "per_formula"]:
            frame = a.csv(f"retrospective/T{t}_{unit}.csv")
            a.check(f"T{t} {unit}: identity and held-out flags", frame.icsd_id.is_unique
                    and frame.icsd_id.isin(cutoff_maps[int(t)]).all()
                    and frame.icsd_id.map(cutoff_maps[int(t)]).astype(int).eq(frame.in_basin.astype(int)).all())
            q = cut["analyses"]["post1980_le_T"][unit]["quadrant"]
            inside, match = boolean(frame.in_basin), boolean(frame.formula_match)
            a.check(f"T{t} {unit}: quadrant and enrichment arithmetic", len(frame) == q["n"]
                    and int((inside & match).sum()) == q["in_basin_and_match"]
                    and int((inside & ~match).sum()) == q["in_basin_and_no_match"]
                    and int((~inside & match).sum()) == q["frontier_and_match"]
                    and int((~inside & ~match).sum()) == q["frontier_and_no_match"]
                    and np.isclose(float((inside & match).mean()) / (float(inside.mean()) * float(match.mean())), q["enrichment_ratio_obs_over_independence"], atol=1e-12, rtol=0))

    # Representation coverage must use current non-noise IDs, including undated.
    success_sets = {}
    for kind in ["graphlet", "amd"]:
        p = f"representations/{kind}_features"
        meta = a.js(f"{p}/feature_preparation.json")
        success = list(map(int, a.js(f"{p}/features.ids.json")))
        failed = [int(x["icsd_id"]) for x in meta["failures"]]
        a.check(f"{kind}: full current target partition and retained IDs",
                meta["complete"] and set(meta["requested_ids"]) == nonnoise
                and set(success).isdisjoint(failed) and set(success) | set(failed) == nonnoise
                and len(success) == len(set(success)) == meta["n_successful"]
                and len(failed) == len(set(failed)) == meta["n_failed"]
                and meta["n_requested"] == len(nonnoise)
                and meta["production_labels_sha256"] == label_hash
                and meta["ids_sha256"] == sha256(a.base / f"{p}/features.ids.json"),
                successful=len(success), failed=len(failed), undated=sum(years[i] == "" for i in success))
        a.check(f"{kind}: reuse and undated coverage accounting", meta["n_reused"] + meta["n_new_successful"] == len(success)
                and meta["n_successful_undated"] == sum(years[i] == "" for i in success)
                and all(x["matches"] for x in meta["reuse_validation"]))
        success_sets[kind] = set(success)
    common = success_sets["graphlet"] & success_sets["amd"]
    for run in ["graphlet-full-recovery", "graphlet-common-recovery", "amd-full-recovery", "amd-common-recovery"]:
        p = f"representations/{run}"
        meta = a.js(f"{p}/summary.json")
        cohort = list(map(int, a.js(f"{p}/cohort.ids.json")))
        expected = common if "common" in run else success_sets[run.split("-")[0]]
        a.check(f"{run}: exact declared candidate pool", set(cohort) == expected
                and len(cohort) == len(expected) == meta["n_candidates"]
                and meta["cohort_sha256"] == sha256(a.base / f"{p}/cohort.ids.json")
                and meta["production_labels_sha256"] == label_hash and meta["complete"])
        for metric in meta["results"]:
            r = a.js(f"{p}/{metric}/recovery.json")
            nn = a.csv(f"{p}/{metric}/nearest_neighbors.csv")
            a.check(f"{run}/{metric}: complete nonself identities and current labels",
                    list(nn.query_id) == cohort and nn.neighbor_id.isin(expected).all()
                    and nn.query_id.ne(nn.neighbor_id).all()
                    and nn.query_id.map(labels).eq(nn.query_community).all()
                    and nn.neighbor_id.map(labels).eq(nn.neighbor_community).all()
                    and np.isfinite(nn.distance.to_numpy(float)).all() and nn.distance.ge(0).all())
            same = nn.query_community.eq(nn.neighbor_community)
            sizes = Counter(labels[i] for i in cohort)
            chance = sum(n*(n-1) for n in sizes.values())/(len(cohort)*(len(cohort)-1))
            a.check(f"{run}/{metric}: recovery and size-weighted null arithmetic",
                    boolean(nn.same_community).eq(same).all() and int(same.sum()) == r["hits"]
                    and len(nn)-int(same.sum()) == r["misses"]
                    and np.isclose(same.mean(),r["same_community_1nn_agreement"],atol=1e-12,rtol=0)
                    and np.isclose(chance,r["random_other_entry_chance"],atol=1e-12,rtol=0), hits=int(same.sum()), n=len(nn))
            confusion = a.csv(f"{p}/{metric}/confusion_counts.csv")
            observed = Counter(zip(nn.loc[~same,"query_community"],nn.loc[~same,"neighbor_community"]))
            saved = {(int(x.from_community),int(x.to_community)):int(x.count) for x in confusion.itertuples()}
            a.check(f"{run}/{metric}: confusion identities and counts", dict(observed) == saved)
            if metric == "graphlet_l1":
                columns = ["first_ten_chemical_channels_l1", "pure_geometry_four_channels_l1", "pair_triplet_chemical_50_channels_l1"]
                a.check(f"{run}: chemical/geometry contributions sum to exact selected distance",
                        np.isfinite(nn[columns].to_numpy(float)).all() and (nn[columns].to_numpy(float) >= 0).all()
                        and np.allclose(nn[columns].sum(axis=1),nn.distance_l1_recomputed,atol=1e-10,rtol=0))
                audit = a.js(f"{p}/{metric}/exact_query_audit.json")
                a.check(f"{run}: exact sample has valid retained ties", audit["n_queries"] == len(audit["queries"]) == 512
                        and audit["candidate_pool_size"] == len(cohort)
                        and all(z["query_id"] in expected and z["query_id"] not in z["exact_nearest_tied_ids"]
                                and set(z["exact_nearest_tied_ids"]) <= expected
                                and z["exact_selected_neighbor_id"] == min(z["exact_nearest_tied_ids"])
                                for z in audit["queries"])
                        and np.isclose(np.mean([z["ann_selected_is_exact_nearest"] for z in audit["queries"]]),audit["ann_exact_nearest_fraction"],atol=1e-12,rtol=0))

    for run in ["graphlet-dated-replay", "amd-full-partition"]:
        p = f"representations/{run}"
        meta = a.js(f"{p}/summary.json")
        cohort = set(map(int,a.js(f"{p}/cohort.ids.json")))
        for metric in meta["results"]:
            events = a.csv(f"{p}/{metric}/node_temporal_events.csv")
            partition = a.csv(f"{p}/{metric}/community_assignments.csv")
            table = a.js(f"{p}/{metric}/exclusive_by_decade.json")
            a.check(f"{run}/{metric}: chronological identity and labels", set(events.icsd_id) == cohort
                    and events.icsd_id.is_unique and set(partition.icsd_id) == cohort
                    and events.icsd_id.map(partition.set_index("icsd_id").community).eq(events.community).all())
            events["exclusive"] = exclusive(events)
            for r in table["rows"]:
                sub = events[events.decade.eq(r["decade"])]
                count = sub.exclusive.value_counts()
                a.check(f"{run}/{metric}: exclusive counts {r['decade']}", len(sub) == r["n_total"]
                        and all(int(count.get(k,0)) == r[f"n_{k}"] for k in ["birth","same","cross","bridge","outlier"]))

    neighbor_sensitivity = verify_graphlet_neighbor_sensitivity(
        a, nonnoise, prod, labels, years, label_hash)

    ablation_summary = a.js("no_message/summary.json")
    ablation = a.csv("no_message/graph/community_assignments.csv")
    ablation_events = a.csv("no_message/time/node_temporal_events.csv")
    comparison = a.js("no_message/comparison/partition_comparison.json")["no_message"]
    ablation_ids = set(ablation.icsd_id)
    a.check("no-message control changes only propagation setting",
            ablation_summary["feature_version"] == summary["feature_version"] == VERSION
            and ablation_summary["wl_iters"] == 0 and summary["wl_iters"] == 3
            and all(ablation_summary[k] == summary[k] for k in ["neighbor_settings", "local_mode", "max_sites"]))
    a.check("no-message accepted and rejected accounting",
            len(ablation) == len(ablation_ids) == ablation_summary["sample_size_featurized"]
            and len(ablation) + ablation_summary["n_failures"] == 181362
            and comparison["cohort"]["shared_count"] == len(ids & ablation_ids))
    a.check("no-message event identity, years and labels",
            ablation_events.icsd_id.is_unique and set(ablation_events.icsd_id) == ablation_ids
            and ablation_events.icsd_id.map(ablation.set_index("icsd_id").community).eq(ablation_events.community).all()
            and ablation_events.icsd_id.map(ablation.set_index("icsd_id").year).astype(str).eq(ablation_events.year.astype(str)).all())
    ablation_events["exclusive"] = exclusive(ablation_events)
    for decade, sub in ablation_events.groupby("decade"):
        temporal = comparison["temporal"]["by_decade"][decade]
        a.check(f"no-message birth share, {decade}",
                np.isclose(sub.exclusive.eq("birth").mean(), temporal["birth_share"], rtol=0, atol=1e-12))
    joined = prod[["icsd_id", "community"]].merge(ablation[["icsd_id", "community"]], on="icsd_id", suffixes=("_prod", "_abl"), validate="one_to_one")
    for scope, frame in [("including_noise", joined), ("nonoutlier_both", joined[joined.community_prod.ge(0) & joined.community_abl.ge(0)])]:
        metrics = comparison["agreement_vs_baseline"][scope]
        a.check(f"no-message partition agreement, {scope}",
                len(frame) == metrics["n"]
                and np.isclose(adjusted_rand_score(frame.community_prod, frame.community_abl), metrics["ARI"], atol=1e-12, rtol=0)
                and np.isclose(normalized_mutual_info_score(frame.community_prod, frame.community_abl), metrics["NMI"], atol=1e-12, rtol=0))

    if array_root is not None:
        for key,name in [("icsd_features","features.npy"),("icsd_pca","features_pca.npy")]:
            path = array_root/name
            a.check(f"production binary hash {name}", sha256(path) == basis["reference_files"][key]["sha256"])
            array = np.load(path,mmap_mode="r")
            a.check(f"production binary shape and finiteness {name}", len(array) == len(prod)
                    and all(np.isfinite(array[i:i+4096]).all() for i in range(0,len(array),4096)))
    return {"production_rows":len(prod),"production_nonnoise":len(nonnoise),
            "external_successes":{k:len(v) for k,v in source_rows.items()},
            "representation_common_pool":len(common),
            "voronoi_graphlet_successes":neighbor_sensitivity["voronoi_feature_successes"],
            "voronoi_graphlet_failures":neighbor_sensitivity["voronoi_feature_failures"],
            "voronoi_graphlet_calibration_successes":neighbor_sensitivity["voronoi_calibration_successes"],
            "graphlet_neighbor_recovery_common_pool":neighbor_sensitivity["recovery_common_pool"],
            "graphlet_neighbor_partition_common_pool":neighbor_sensitivity["partition_common_pool"],
            "alab_mp_targets":alab_mp_targets,
            "cutoff_trained":cutoff_trained,
            "no_message_rows":len(ablation),"raw_array_checks_run":array_root is not None}


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--downstream", type=Path, default=root/"notes/feature_repair_2026_09/downstream")
    p.add_argument("--out", type=Path)
    p.add_argument("--array-root", type=Path, help="Optional directory containing archived production features.npy and features_pca.npy")
    args = p.parse_args()
    a = Audit(args.downstream.resolve())
    result = {}
    error = None
    try:
        result = verify(a,args.array_root)
    except Exception as exc:
        error = {"type":type(exc).__name__}
        if isinstance(exc,AssertionError):
            error["check"] = str(exc)
        elif isinstance(exc, FileNotFoundError):
            error["path"] = exc.filename
    report = {"passed":error is None,"checks_completed":len(a.checks),"checks":a.checks,
              "summary":result,"error":error,"inputs_sha256":a.inputs,
              "scope":"Saved-artifact identity, count, classification and provenance consistency. No CIF extraction or new nearest-neighbor search; optional production binaries only when --array-root supplied.",
              "script_sha256":sha256(__file__)}
    output = args.out or args.downstream/"verification/downstream_consistency.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":report["passed"],"checks_completed":len(a.checks),"error":error,"report":str(output)}))
    return 0 if error is None else 1


if __name__ == "__main__":
    sys.exit(main())
