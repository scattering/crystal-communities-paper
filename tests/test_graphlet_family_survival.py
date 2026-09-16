"""Focused tests for the graphlet family-survival metrics."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = ROOT / "notes" / "feature_repair_2026_09" / "downstream"
sys.path.insert(0, str(DOWNSTREAM))
import analyze_graphlet_family_survival as survival


class GraphletFamilySurvivalTests(unittest.TestCase):
    def setUp(self):
        # Production family 10 has two members in alternate cluster 5, one in
        # cluster 6, and one noise member. Cluster 5 also contains one member
        # from production community 20, so its family purity is 2/3.
        self.rows = [
            {"icsd_id": 1, "year": 1995, "production_community": 10, "alternate_community": 5},
            {"icsd_id": 2, "year": 2001, "production_community": 10, "alternate_community": 5},
            {"icsd_id": 3, "year": 2002, "production_community": 10, "alternate_community": 6},
            {"icsd_id": 4, "year": 2004, "production_community": 10, "alternate_community": -1},
            {"icsd_id": 5, "year": 1998, "production_community": 20, "alternate_community": 5},
            {"icsd_id": 6, "year": 2003, "production_community": 20, "alternate_community": 7},
        ]

    def test_fragmentation_purity_and_temporal_counts(self):
        metrics, dominant = survival.compute_partition_metrics(
            self.rows, {10}, n_all_members=5, n_dated_members=4
        )
        self.assertEqual(dominant, 5)
        self.assertEqual(metrics["matched_dated_members"], 4)
        self.assertEqual(metrics["alternate_noise_count"], 1)
        self.assertAlmostEqual(metrics["coverage_of_all_members"], 4 / 5)
        self.assertAlmostEqual(metrics["dominant_cluster_recall_all_members"], 1 / 2)
        self.assertAlmostEqual(metrics["dominant_cluster_recall_nonnoise_members"], 2 / 3)
        self.assertAlmostEqual(metrics["dominant_cluster_purity"], 2 / 3)
        self.assertEqual(metrics["clusters_for_90_percent_nonnoise"], 2)
        expected_neff = math.exp(-((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3)))
        self.assertAlmostEqual(
            metrics["fragmentation_effective_cluster_count_nonnoise"], expected_neff
        )
        self.assertAlmostEqual(metrics["pair_recall_all_family_pairs"], 1 / 6)

        event = survival.compute_temporal_event(
            self.rows,
            {10},
            dominant,
            {"kind": "test", "year": 2000, "context": "synthetic"},
        )
        self.assertEqual(
            event["family_members_captured_by_dominant_cluster"]["pre_count"], 1
        )
        self.assertEqual(
            event["family_members_captured_by_dominant_cluster"]["post_count"], 1
        )
        self.assertEqual(event["full_dominant_alternate_cluster"]["pre_count"], 2)
        self.assertEqual(event["full_dominant_alternate_cluster"]["post_count"], 1)

    def test_metrics_are_invariant_to_alternate_label_permutation(self):
        original, _ = survival.compute_partition_metrics(
            self.rows, {10}, n_all_members=5, n_dated_members=4
        )
        mapping = {-1: -1, 5: 101, 6: 3, 7: 88}
        permuted_rows = [
            {**row, "alternate_community": mapping[row["alternate_community"]]}
            for row in self.rows
        ]
        permuted, _ = survival.compute_partition_metrics(
            permuted_rows, {10}, n_all_members=5, n_dated_members=4
        )
        excluded = {"dominant_alternate_cluster_label"}
        self.assertEqual(
            {key: value for key, value in original.items() if key not in excluded},
            {key: value for key, value in permuted.items() if key not in excluded},
        )

        # Also exercise an exact overlap tie. The selected membership must stay
        # fixed even when its numeric label changes from larger to smaller.
        tied_rows = [
            {**row, "alternate_community": value}
            for row, value in zip(self.rows, (9, 9, 2, 2, 9, 7))
        ]
        tied_original, tied_label = survival.compute_partition_metrics(
            tied_rows, {10}, n_all_members=5, n_dated_members=4
        )
        tied_mapping = {-1: -1, 9: 50, 2: 1, 7: 3}
        tied_permuted_rows = [
            {**row, "alternate_community": tied_mapping[row["alternate_community"]]}
            for row in tied_rows
        ]
        tied_permuted, tied_permuted_label = survival.compute_partition_metrics(
            tied_permuted_rows, {10}, n_all_members=5, n_dated_members=4
        )
        self.assertEqual(tied_label, 9)
        self.assertEqual(tied_permuted_label, 50)
        self.assertEqual(
            {key: value for key, value in tied_original.items() if key not in excluded},
            {key: value for key, value in tied_permuted.items() if key not in excluded},
        )


if __name__ == "__main__":
    unittest.main()
