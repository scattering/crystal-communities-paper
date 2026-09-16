"""Independent artifact validation and shared-success denominator controls."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import external_representation_sensitivity as analysis
import verify_external_representation_sensitivity as verifier


class VerifyExternalRepresentationSensitivityTests(unittest.TestCase):
    def test_projection_audit_detects_wrong_membership_and_radius(self):
        basis = {"communities": np.array([8]), "centroids": np.array([[0.]]), "p95": np.array([2.])}
        row = {"record_key": "mp-1", "assigned_community": 8, "nearest_centroid_distance": 3.,
               "community_threshold_p95": 2., "in_basin": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projection.csv"
            analysis.write_csv(path, [row])
            self.assertEqual(verifier.audit_projection(path, ["mp-1"], basis)[2]["n_in_basin"], 0)
            analysis.write_csv(path, [{**row, "in_basin": True}])
            with self.assertRaisesRegex(ValueError, "In-basin flag"):
                verifier.audit_projection(path, ["mp-1"], basis)
            analysis.write_csv(path, [{**row, "community_threshold_p95": 4., "in_basin": True}])
            with self.assertRaisesRegex(ValueError, "Invalid radius"):
                verifier.audit_projection(path, ["mp-1"], basis)

    def test_common_support_uses_same_icsd_and_external_id_intersections(self):
        references = {"graphlet": {"full": {"1": True, "2": False, "3": True}},
                      "magpie": {"full": {"2": True, "3": True, "4": False}}}
        external = {name: {source: {"full": flags} for source in verifier.SOURCES}
                    for name, flags in [("graphlet", {"x": True, "y": False}),
                                        ("magpie", {"y": True, "z": False})]}
        rows = verifier.common_support(references, external)
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertEqual(row["common_icsd_n"], 2)
            self.assertEqual(row["common_external_n"], 1)
            if row["representation"] == "graphlet":
                self.assertEqual(row["icsd_in_basin_fraction"], .5)
                self.assertEqual(row["external_in_basin_fraction"], 0.)
                self.assertEqual(row["icsd_minus_external_percentage_points"], 50.)
            else:
                self.assertEqual(row["icsd_in_basin_fraction"], 1.)
                self.assertEqual(row["external_in_basin_fraction"], 1.)
                self.assertEqual(row["icsd_minus_external_percentage_points"], 0.)


if __name__ == "__main__":
    unittest.main()
