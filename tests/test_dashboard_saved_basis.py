"""Integration checks for the frozen dashboard; uses local regenerated artifacts."""
import copy
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from pymatgen.core import Structure

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "dashboard"))
from frozen_backend import load_bundle, score, sha256

DOWNSTREAM = REPO / "notes/feature_repair_2026_09/downstream"
MANIFEST = Path(os.environ.get("ICSD_DASHBOARD_MANIFEST", DOWNSTREAM / "dashboard/manifest.json"))


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


@unittest.skipUnless(MANIFEST.exists(), "Requires the local repaired dashboard manifest")
class SavedDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads(MANIFEST.read_text())
        missing = [role for role, entry in manifest["artifacts"].items()
                   if not (MANIFEST.parent / entry["path"]).is_file()]
        if missing:
            raise unittest.SkipTest("Requires archived dashboard artifacts: " + ", ".join(missing))
        cls.bundle = load_bundle(MANIFEST)
        cls.manifest = copy.deepcopy(cls.bundle["manifest"])
        for role, path in cls.bundle["paths"].items():
            cls.manifest["artifacts"][role]["path"] = str(path)

    def reject(self, manifest, message):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, message):
                load_bundle(path)

    def test_obsolete_version_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["feature_version"] = "legacy-unversioned"
        self.reject(manifest, "incompatible feature version")

    def test_changed_artifact_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["artifacts"]["representatives"]["sha256"] = "0" * 64
        self.reject(manifest, "hash mismatch: representatives")

    def test_rehashed_wrong_pca_cannot_be_mixed(self):
        manifest = copy.deepcopy(self.manifest)
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "wrong.npy"
            np.save(wrong, np.zeros((3, 32)))
            manifest["artifacts"]["pca"] = {"path": str(wrong), "sha256": sha256(wrong)}
            self.reject(manifest, "does not belong to this basis: pca")

    def test_public_cifs_match_regenerated_source_scores(self):
        manifest = read_csv(DOWNSTREAM / "candidates/cifs/extraction_manifest.csv")
        selected = {}
        for row in manifest:
            # One small-cell public example per source, spanning all five adapters.
            if row["source"] not in selected or int(row["n_sites"]) < int(selected[row["source"]]["n_sites"]):
                selected[row["source"]] = row
        self.assertEqual(set(selected), {"GNoME", "MP", "JARVIS", "Alexandria", "MatterGen"})
        comparisons = []
        for source, row in selected.items():
            with self.subTest(source=source, material=row["material_id"]):
                slug = source.lower()
                prefix = "mattergen-public" if slug == "mattergen" else slug
                records = read_csv(DOWNSTREAM / f"external/{slug}/{prefix}_frontier_records.csv")
                key = "zip_member" if slug == "mattergen" else "material_id"
                expected = [r for r in records if r[key] == row["record_key"]]
                self.assertEqual(len(expected), 1)
                expected = expected[0]
                path = DOWNSTREAM / "candidates/cifs" / row["cif_relative_path"]
                self.assertEqual(sha256(path), row["cif_sha256"])
                result = score(Structure.from_file(path), self.bundle)
                self.assertEqual(result["community"], int(expected["assigned_community"]))
                self.assertEqual(result["n_sites"], int(row["n_sites"]))
                self.assertEqual(result["frontier"], expected["in_basin"].lower() != "true")
                self.assertEqual(result["observation_year"], 2019)
                self.assertAlmostEqual(result["threshold"], float(expected["community_threshold_p95"]), places=12)
                np.testing.assert_allclose(result["distance"], float(expected["nearest_centroid_distance"]), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(result["xy"], [float(expected["pca1"]), float(expected["pca2"])], rtol=1e-6, atol=1e-6)
                if source == "GNoME":
                    costs = read_csv(DOWNSTREAM / "accessibility/gnome_accessibility_records.csv")
                    cost = next(r for r in costs if r["material_id"] == row["material_id"])
                    self.assertAlmostEqual(result["accessibility"], float(cost["accessibility_score"]), places=10)
                comparisons.append({"source": source, "material_id": row["material_id"],
                    "community": result["community"], "distance": result["distance"],
                    "distance_absolute_error": abs(result["distance"] - float(expected["nearest_centroid_distance"])),
                    "pca_xy_max_absolute_error": float(np.max(np.abs(result["xy"] - [float(expected["pca1"]), float(expected["pca2"])])))})
        print("Public CIF score comparisons: " + json.dumps(comparisons))

    def test_app_uses_saved_display_coordinates_and_builds_pages(self):
        import importlib
        from unittest.mock import patch
        with patch.dict(os.environ, {"ICSD_DASHBOARD_MANIFEST": str(MANIFEST.resolve())}):
            app = importlib.import_module("dash_app")
        self.assertEqual(app.APP_STATUS, "ready", app.APP_STATUS_ERROR)
        fmap = app.load_frozen_map()
        valid = np.flatnonzero(fmap.bundle["labels"] >= 0)
        indices = np.sort(np.random.default_rng(42).choice(valid, size=12000, replace=False))
        np.testing.assert_array_equal(fmap.background_xy, fmap.bundle["pca"][indices, :2])
        np.testing.assert_array_equal(fmap.centroids_xy, fmap.centroids[:, :2])
        self.assertEqual(app.year_text(-1), "unknown")
        for page in (app.overview_layout, app.scoring_layout, app.community_map_layout):
            self.assertIsNotNone(page())
        self.assertIsNotNone(app.placement_figure(None))
        self.assertIsNotNone(app.community_detail_panel(2662))
        client = app.server.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/_dash-layout").status_code, 200)


if __name__ == "__main__":
    unittest.main()
