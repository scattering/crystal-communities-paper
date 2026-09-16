"""Focused checks for the duplicate Voronoi full/shared verifier."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_repaired_downstream as verifier


def neighbor_rows():
    return pd.DataFrame({
        "query_id": [11, 12, 13],
        "query_community": [1, 1, 2],
        "neighbor_id": [12, 11, 12],
        "neighbor_community": [1, 1, 1],
        "distance": [0.1, 0.1, 0.25],
        "same_community": [True, True, False],
        "distance_l1_recomputed": [0.1, 0.1, 0.25],
        "first_ten_chemical_channels_l1": [0.04, 0.04, 0.1],
        "pure_geometry_four_channels_l1": [0.01, 0.01, 0.05],
        "pair_triplet_chemical_50_channels_l1": [0.05, 0.05, 0.1],
    })


class DuplicateVoronoiIdentityTests(unittest.TestCase):
    def test_pca_control_documents_allow_only_method_provenance_differences(self):
        base = {field: f"left-{field}" for field in verifier.GRAPHLET_METHOD_PROVENANCE_FIELDS}
        base.update({"name": "production_pca_raw_same_protocol", "n_edges": 123, "sigma": 0.5})
        other = dict(base)
        for field in verifier.GRAPHLET_METHOD_PROVENANCE_FIELDS:
            other[field] = f"right-{field}"
        self.assertTrue(verifier.graphlet_science_documents_identical(base, other))

        changed = dict(other)
        changed["n_edges"] = 124
        self.assertFalse(verifier.graphlet_science_documents_identical(base, changed))

        incomplete = dict(other)
        incomplete.pop("restrict_ids")
        self.assertFalse(verifier.graphlet_science_documents_identical(base, incomplete))

    def test_selected_neighbor_identity_checks_ids_distances_and_hits(self):
        expected = neighbor_rows()
        duplicate = expected.copy()
        self.assertTrue(verifier.selected_neighbor_rows_identical(expected, duplicate))

        changes = {
            "neighbor_id": 99,
            "distance": 0.10000000000000002,
            "same_community": False,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                changed = duplicate.copy()
                changed.loc[0, field] = value
                self.assertFalse(
                    verifier.selected_neighbor_rows_identical(expected, changed)
                )

    def test_partition_identity_allows_label_permutation_only(self):
        original = np.asarray([0, 0, 3, 3, -1, 8])
        relabelled = np.asarray([7, 7, 2, 2, -1, 4])
        self.assertTrue(verifier.partition_membership_and_noise_identical(
            original, relabelled,
        ))

        changed_membership = relabelled.copy()
        changed_membership[1] = 2
        self.assertFalse(verifier.partition_membership_and_noise_identical(
            original, changed_membership,
        ))
        changed_noise = relabelled.copy()
        changed_noise[4] = 4
        self.assertFalse(verifier.partition_membership_and_noise_identical(
            original, changed_noise,
        ))

    def test_temporal_rows_allow_label_permutation_but_require_science_identity(self):
        original = pd.DataFrame({
            "icsd_id": [11, 12, 13, 14],
            "year": [1910, 1911, 1912, 1913],
            "decade": ["1910s"] * 4,
            "community": [0, 0, 3, -1],
            "event_type": [
                "community_birth", "existing_community",
                "community_birth", "outlier",
            ],
            "n_active_neighbors": [0, 1, 1, 0],
            "n_active_same_community_neighbors": [0, 1, 0, 0],
            "n_active_other_communities": [0, 0, 1, 0],
            "is_bridge_attachment": [False] * 4,
            "distance_to_centroid": [0.2, 0.2, 0.0, np.nan],
            "core_threshold": [0.2, 0.2, 0.0, np.nan],
            "core_periphery": ["core", "core", "core", ""],
        })
        duplicate = original.copy()
        duplicate["community"] = [9, 9, 4, -1]
        self.assertTrue(verifier.temporal_event_rows_identical(original, duplicate))

        changed = duplicate.copy()
        changed.loc[2, "n_active_other_communities"] = 0
        self.assertFalse(verifier.temporal_event_rows_identical(original, changed))


if __name__ == "__main__":
    unittest.main()
