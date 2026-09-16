"""Checks for the frozen-transform sensitivity and paired decision accounting."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import magpie_coordinate_consistency as consistency


class MagpieCoordinateConsistencyTests(unittest.TestCase):
    def test_chunked_transform_matches_fitted_sklearn_transform(self):
        rng = np.random.default_rng(3471805)
        raw = rng.normal(size=(131, 39)) * np.arange(1, 40)
        scaler = StandardScaler().fit(raw)
        standardized = scaler.transform(raw)
        pca = PCA(n_components=8, svd_solver="randomized", iterated_power=0, random_state=42)
        saved_scores = pca.fit_transform(standardized)
        transform = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                     "pca_mean": pca.mean_, "pca_components": pca.components_}
        actual = consistency.transform_rows(raw, transform, chunk_size=7)
        expected = pca.transform(standardized)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        # Deliberately inaccurate randomized SVD exposes the exact distinction
        # under investigation; the utility follows external transform scores.
        self.assertGreater(np.max(np.abs(actual - saved_scores)), .1)

    def test_relative_displacement_excludes_noise_and_preserves_zero_radius_cases(self):
        before = np.array([[0., 0.], [2., 0.], [0., 5.]])
        after = before + np.array([[1., 0.], [0., 2.], [3., 4.]])
        basis = {"communities": np.array([5, 8]), "p95": np.array([2., 0.])}
        report = consistency.displacement_report(before, after, np.array([5, 8, -1]), basis)
        self.assertEqual(report["euclidean_coordinate_displacement_all_map_rows"]["maximum"], 5.)
        ratio = report["displacement_divided_by_original_own_community_p95_positive_radius"]
        self.assertEqual(ratio["n"], 1)
        self.assertEqual(ratio["maximum"], .5)
        self.assertEqual(report["n_nonnoise_zero_radius"], 1)
        self.assertEqual(report["n_zero_radius_members_with_nonzero_displacement"], 1)

    def test_paired_changes_keep_cancelling_switches_visible(self):
        before = [{"record_key": "a", "assigned_community": 1, "in_basin": "True"},
                  {"record_key": "b", "assigned_community": 2, "in_basin": "False"}]
        after = [{"record_key": "a", "assigned_community": 1, "in_basin": False},
                 {"record_key": "b", "assigned_community": 1, "in_basin": True}]
        result = consistency.paired_changes(before, after)
        self.assertEqual(result["rate_change_percentage_points"], 0.)
        self.assertEqual(result["n_membership_flags_changed"], 2)
        self.assertEqual(result["n_in_basin_to_frontier"], 1)
        self.assertEqual(result["n_frontier_to_in_basin"], 1)
        self.assertEqual(result["n_nearest_community_changed"], 1)
        with self.assertRaisesRegex(ValueError, "identities"):
            consistency.paired_changes(before, list(reversed(after)))


if __name__ == "__main__":
    unittest.main()
