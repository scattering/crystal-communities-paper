"""Projection semantics and frozen-identity regression tests for public ablations."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import external_representation_sensitivity as analysis


class ExternalRepresentationSensitivityTests(unittest.TestCase):
    def test_nearest_centroid_precedes_radius_and_equality_is_inside(self):
        basis = {"centroids": np.array([[0.], [10.]]), "p95": np.array([1., 100.]),
                 "communities": np.array([2, 9])}
        nearest, distances, inside = analysis.classify(np.array([[1.], [2.], [10.]]), basis)
        np.testing.assert_array_equal(nearest, [0, 0, 1])
        np.testing.assert_allclose(distances, [1, 2, 0])
        np.testing.assert_array_equal(inside, [True, False, True])

    def test_block_projection_agrees_with_independent_euclidean_oracle(self):
        rng = np.random.default_rng(391)
        x, centers = rng.normal(size=(71, 43)), rng.normal(size=(17, 43))
        radii = rng.uniform(2, 10, len(centers))
        matrix = cdist(x, centers)
        expected = matrix.argmin(axis=1)
        nearest, distances, inside = analysis.classify(x, {"centroids": centers, "p95": radii}, batch_size=7)
        np.testing.assert_array_equal(nearest, expected)
        np.testing.assert_allclose(distances, matrix[np.arange(len(x)), expected], atol=1e-12)
        np.testing.assert_array_equal(inside, distances <= radii[nearest])

    def test_later_members_cannot_widen_historical_radii_and_noise_cannot_define_basin(self):
        x = np.array([[0.], [2.], [100.], [1000.]])
        labels = np.array([4, 4, 4, -1])
        historic = analysis.community_basis(x, labels, np.array([True, True, False, True]))
        full = analysis.community_basis(x, labels, np.ones(4, dtype=bool))
        np.testing.assert_array_equal(historic["communities"], [4])
        np.testing.assert_allclose(historic["centroids"], [[1.]])
        np.testing.assert_allclose(historic["p95"], [1.])
        self.assertGreater(full["p95"][0], 60.)
        self.assertFalse(analysis.classify(x[2:3], historic)[2][0])

    def test_reference_ids_are_joined_in_feature_order_without_borrowing_labels(self):
        rows = [{"icsd_id": "11", "year": "1999", "community": "8"},
                {"icsd_id": "22", "year": "", "community": "-1"}]
        index, ids, years, labels = analysis.aligned_reference([22, 77, 11], rows)
        np.testing.assert_array_equal(index, [0, 2])
        np.testing.assert_array_equal(ids, [22, 11])
        np.testing.assert_array_equal(years, [-1, 1999])
        np.testing.assert_array_equal(labels, [-1, 8])
        with self.assertRaisesRegex(ValueError, "unrepresented"):
            analysis.aligned_reference([11], rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analysis.aligned_reference([11, 11, 22], rows)

    def test_singleton_radius_is_zero_and_exact_match_is_valid(self):
        basis = analysis.community_basis(np.array([[3.], [15.]]), np.array([0, -1]), np.ones(2, dtype=bool))
        np.testing.assert_allclose(basis["p95"], [0.])
        np.testing.assert_array_equal(analysis.classify(np.array([[3.], [3.00001]]), basis)[2], [True, False])

    def test_cache_requires_identity_signature_dimension_and_finite_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cached.npz"
            row = {"material_id": "mp-10"}
            analysis.save_cache(path, "frozen", row, np.ones(4))
            self.assertIsNotNone(analysis.load_cache(path, "frozen", "mp-10", 4))
            for signature, key, width in [("other", "mp-10", 4), ("frozen", "mp-20", 4), ("frozen", "mp-10", 5)]:
                self.assertIsNone(analysis.load_cache(path, signature, key, width))
            analysis.save_cache(path, "frozen", row, np.array([1., np.nan, 2., 3.]))
            self.assertIsNone(analysis.load_cache(path, "frozen", "mp-10", 4))

    def test_mattergen_cache_uses_zip_member_not_ambiguous_material_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cached.npz"
            row = {"material_id": "sample-1", "zip_member": "release/family/task/sample-1.cif"}
            analysis.save_cache(path, "frozen", row, np.ones(4))
            self.assertIsNotNone(analysis.load_cache(path, "frozen", row["zip_member"], 4))
            self.assertIsNone(analysis.load_cache(path, "frozen", row["material_id"], 4))

    def test_invalid_projection_cannot_be_silently_classified(self):
        basis = {"centroids": np.array([[0.]]), "p95": np.array([1.])}
        with self.assertRaisesRegex(ValueError, "Invalid projection"):
            analysis.classify(np.array([[np.nan]]), basis)
        with self.assertRaisesRegex(ValueError, "Invalid projection"):
            analysis.classify(np.array([[1., 2.]]), basis)

    def test_wilson_interval_stays_defined_at_zero_and_one_success_fraction(self):
        zero = analysis.rate([False] * 10)
        one = analysis.rate([True] * 10)
        self.assertAlmostEqual(zero["wilson_95_ci"][0], 0.)
        self.assertAlmostEqual(zero["wilson_95_ci"][1], 0.2775327998628892)
        self.assertAlmostEqual(one["wilson_95_ci"][0], 1 - zero["wilson_95_ci"][1])
        self.assertAlmostEqual(one["wilson_95_ci"][1], 1.)
        self.assertIsNone(analysis.rate([])["wilson_95_ci"])


if __name__ == "__main__":
    unittest.main()
