"""Focused checks for the saved graphlet Louvain tolerance audit."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_repaired_downstream as verifier


def fixture():
    diagnostics = {
        "levels": 2,
        "level_diagnostics": [
            {
                "sweeps": 2,
                "accepted_moves": 1,
                "minimum_accepted_gain": 1e-10,
                "maximum_accepted_gain": 1e-5,
                "level_modularity": 0.9,
                "level_modularity_gain": 0.1,
            },
            {
                "sweeps": 1,
                "accepted_moves": 0,
                "minimum_accepted_gain": None,
                "maximum_accepted_gain": None,
            },
        ],
        "termination": "no_local_improvement",
    }
    assignments = pd.DataFrame({"community": [0, 0, -1, 1]})
    assignment_hash = verifier.numpy_int64_sha256(assignments.community)
    comparison_to_production = {"n_alternate_noise": 1}
    canonical_partition = {
        "n_entries": 4,
        "n_edges": 3,
        "knn_k": 16,
        "metric": "euclidean",
        "mutual_knn": True,
        "sigma": 1.25,
        "weight": "exp(-distance^2/sigma^2)",
        "louvain_seed": 42,
        "louvain_level_modularity_threshold": 1e-7,
        "louvain_max_local_move_sweeps": 100,
        "minimum_community_size": 10,
        "n_communities": 2,
        "n_noise": 1,
        "louvain_diagnostics": diagnostics,
        "comparison_to_corrected_production_labels": comparison_to_production,
    }
    time_summary = {
        "by_decade": {
            "1930s": {"n_total": 1.0},
            "2010s": {"n_total": 3.0},
        }
    }
    inputs = {
        "features_sha256": "1" * 64,
        "feature_ids_sha256": "2" * 64,
        "neighbor_ids_sha256": "3" * 64,
        "neighbors_sha256": verifier.GRAPHLET_VORONOI_MATCHED_NEIGHBORS_SHA256,
        "production_labels_sha256": "4" * 64,
        "analyzer_sha256": "5" * 64,
    }
    fixed_protocol = {
        "n_entries": 4,
        "n_edges": 3,
        "knn_k": 16,
        "metric": "euclidean",
        "mutual_knn": True,
        "sigma": 1.25,
        "weight": "exp(-distance^2/sigma^2)",
        "louvain_seed": 42,
        "louvain_level_modularity_threshold": 1e-7,
        "louvain_max_local_move_sweeps": 100,
        "minimum_community_size": 10,
    }
    record = {
        "node_move_gain_tolerance": 1e-20,
        "converged": True,
        "louvain_diagnostics": diagnostics,
        "minimum_accepted_gain": 1e-10,
        "modularity_before_small_community_filter": 0.9949454015031324,
        "n_communities": 2,
        "n_noise": 1,
        "comparison_to_production": comparison_to_production,
        "temporal_endpoints": time_summary["by_decade"],
        "assignments_sha256": assignment_hash,
    }
    right_record = deepcopy(record)
    right_record["node_move_gain_tolerance"] = 1e-21
    document = {
        "purpose": (
            "isolate stable-Louvain node-move gain tolerance on one fixed "
            "saved NNDescent graph"),
        "inputs": inputs,
        "fixed_protocol": fixed_protocol,
        "tolerances": {"1e-20": record, "1e-21": right_record},
        "comparisons": {
            "1e-20_vs_1e-21": {
                "exact_label_array_equal": True,
                "n_numeric_labels_different": 0,
                "adjusted_rand_index": 1.0,
                "normalized_mutual_information_arithmetic": 1.0,
                "noise_transition_counts": {
                    "nonnoise_to_nonnoise": 3,
                    "nonnoise_to_noise": 0,
                    "noise_to_nonnoise": 0,
                    "noise_to_noise": 1,
                },
            }
        },
    }
    return document, canonical_partition, assignments, time_summary, inputs


def checks(values):
    return verifier.louvain_tolerance_audit_checks(*values)


class LouvainToleranceAuditTests(unittest.TestCase):
    def test_complete_plateau_record_passes(self):
        self.assertTrue(all(checks(fixture()).values()))

    def test_assignment_or_input_tampering_fails(self):
        values = list(fixture())
        values[0] = deepcopy(values[0])
        values[0]["tolerances"]["1e-20"]["assignments_sha256"] = "a" * 64
        result = checks(values)
        self.assertFalse(result["exact assignment plateau"])
        self.assertFalse(result["1e-20 result equals canonical Voronoi replay"])

        values = list(fixture())
        values[0] = deepcopy(values[0])
        values[0]["inputs"]["neighbor_ids_sha256"] = "b" * 64
        self.assertFalse(checks(values)["fixed input identities and protocol"])


if __name__ == "__main__":
    unittest.main()
