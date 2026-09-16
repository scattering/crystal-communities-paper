"""Regression checks for repaired chemistry/geometry in the distinct Magpie ablation."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import secrets
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from pymatgen.core import Composition, Lattice, Species, Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import icsd_ablation_paper_text as driver
import icsd_ablation_paper_text_worker as worker
from crystal_neighbors import FEATURE_VERSION as PRODUCTION_VERSION, geometry_crystalnn


def bcc():
    mixture = {Species("Nb", 0): .75, Species("In", 0): .25}
    return Structure(Lattice.cubic(3.326), [mixture] * 2, [[0, 0, 0], [.5, .5, .5]])


def diagnostics():
    return {"n_sites": 2, "ordered": False,
            "sites_with_unrepresented_cn_mass": 0, "max_unrepresented_cn_mass": 0.0}


class MagpieAblationFeaturesTests(unittest.TestCase):
    def setUp(self):
        worker.init_worker(None, None, None, 3, 256)

    def test_oxidation_annotations_preserve_full_occupancy_weighted_chemistry(self):
        plain = worker.magpie_vector_from_species(Composition("Fe"))
        mixed = worker.magpie_vector_from_species(
            Composition({Species("Fe", 2): .4, Species("Fe", 3): .6})
        )
        np.testing.assert_allclose(mixed, plain, atol=1e-12)
        self.assertEqual(plain[0], 26)
        self.assertAlmostEqual(worker.magpie_vector_from_species(bcc()[0].species)[0], 43)
        self.assertGreater(np.count_nonzero(plain), 10)

    def test_disordered_geometry_and_distinct_concatenating_propagation(self):
        s = bcc()
        info = {}
        x0 = worker.build_structure_embedding_paper(s, 0)
        x1 = worker.build_structure_embedding_paper(s, 1)
        x3 = worker.build_structure_embedding_paper(s, 3, diagnostics=info)
        self.assertEqual(x3.shape, (4491,))
        self.assertTrue(np.isfinite(x3).all())
        self.assertEqual(x3[0], 43)
        self.assertGreater(np.count_nonzero(x3[22:83]), 0)
        # Equivalent BCC sites: retain self, append the equal neighbour mean,
        # and append zero neighbour spread, instead of adding the two vectors.
        np.testing.assert_allclose(x1[:83], x0[:83], atol=1e-10)
        np.testing.assert_allclose(x1[83:166], x0[:83], atol=1e-10)
        np.testing.assert_allclose(x1[166:249], 0, atol=1e-10)
        np.testing.assert_allclose(x3[-9:], x0[-9:], atol=1e-12)
        self.assertEqual(info, diagnostics())

    def test_weighted_neighbor_mean_and_spread_remain_the_ablation_rule(self):
        x = np.array([[1., 10.], [3., 4.], [7., 0.]])
        result = worker.paper_wl_round(x, [[(1, .25), (2, .75)], [], []])
        np.testing.assert_allclose(result[0], [1, 10, 6, 1, np.sqrt(3), np.sqrt(3)])
        np.testing.assert_allclose(result[1], [3, 4, 0, 0, 0, 0])

    def test_neighbour_failure_is_not_silently_successful(self):
        cnn = geometry_crystalnn()
        with patch.object(cnn, "get_nn_data", side_effect=RuntimeError("unavailable neighbours")):
            with self.assertRaisesRegex(RuntimeError, "unavailable neighbours"):
                worker.build_structure_embedding_paper(bcc(), 3, cnn=cnn)

    def test_missing_magpie_property_is_finite_without_hiding_lookup_errors(self):
        original = worker.WORKER_MAGPIE.get_elemental_property
        with patch.object(worker.WORKER_MAGPIE, "get_elemental_property", side_effect=(
            lambda element, name: np.nan if name == "Electronegativity" else original(element, name)
        )):
            vector = worker.magpie_vector_for_element("Fe")
            self.assertTrue(np.isfinite(vector).all())
            self.assertEqual(vector[worker.MAGPIE_PROPERTIES.index("Electronegativity")], 0)
        worker.WORKER_MAGPIE_CACHE.clear()
        with patch.object(worker.WORKER_MAGPIE, "get_elemental_property", side_effect=RuntimeError("bad property table")):
            with self.assertRaisesRegex(RuntimeError, "bad property table"):
                worker.magpie_vector_for_element("Fe")


class MagpieAblationCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.secret = secrets.token_urlsafe(24)
        self.args = argparse.Namespace(wl_iters=3, max_sites=256, zip_password=self.secret)
        self.rec = worker.Record(icsd_id=123, year=2001, row={})
        self.embedding = np.arange(4491, dtype=float)

    def test_distinct_version_rejects_legacy_and_wrong_metadata(self):
        self.assertNotEqual(worker.FEATURE_VERSION, PRODUCTION_VERSION)
        path = self.root / driver.cache_key(self.rec, self.args)
        # Even renaming an old ablation cache cannot make it compatible.
        np.savez_compressed(path, status="ok", embedding=self.embedding)
        self.assertIsNone(driver.load_cached_result(self.root, self.rec, self.args))
        metadata = driver.feature_metadata(self.args)
        metadata["feature_version"] = PRODUCTION_VERSION
        np.savez_compressed(path, status="ok", embedding=self.embedding,
                            feature_metadata=json.dumps(metadata),
                            feature_diagnostics=json.dumps(diagnostics()))
        self.assertIsNone(driver.load_cached_result(self.root, self.rec, self.args))

    def test_cache_roundtrip_retains_diagnostics_but_no_credential(self):
        info = {**diagnostics(), "private": self.secret}
        driver.store_cached_result(self.root, self.rec, self.args, True, self.embedding, info)
        ok, rec, actual, saved_info = driver.load_cached_result(self.root, self.rec, self.args)
        self.assertTrue(ok)
        self.assertEqual(rec, self.rec)
        np.testing.assert_array_equal(actual, self.embedding)
        self.assertEqual(saved_info, diagnostics())
        with np.load(self.root / driver.cache_key(self.rec, self.args), allow_pickle=False) as data:
            for name in data.files:
                self.assertNotIn(self.secret, str(data[name]))
        for invalid in (np.zeros(213), np.full(4491, np.nan)):
            driver.store_cached_result(self.root, self.rec, self.args, True, invalid, info)
            self.assertIsNone(driver.load_cached_result(self.root, self.rec, self.args))

    def test_protected_failure_details_are_omitted(self):
        failure = {"icsd_id": 123, "reason": "RuntimeError", "detail": self.secret[:12]}
        driver.store_cached_result(self.root, self.rec, self.args, False, None, failure)
        result = driver.load_cached_result(self.root, self.rec, self.args)
        self.assertFalse(result[0])
        self.assertNotIn(self.secret[:12], json.dumps(result[3]))

    def test_stdin_credential_and_summary_provenance(self):
        argv = ["--index-csv", "index.csv", "--output-dir", str(self.root), "--zip-password-stdin"]
        with patch("sys.stdin", io.StringIO(self.secret + "\n")):
            args = driver.parse_args(argv)
        self.assertEqual(args.zip_password, self.secret)
        self.assertNotIn(self.secret, json.dumps(driver.feature_metadata(args)))
        args.icsd_zip = "input.zip"
        args.n_jobs = 1
        args.sample_size = 3
        args.checkpoint_every = 1
        records = [worker.Record(i, 2000 + i, {}) for i in range(3)]
        result_by_id = {
            r.icsd_id: (True, r, self.embedding + r.icsd_id, diagnostics()) for r in records
        }
        clusterer = Mock()
        clusterer.fit_predict.return_value = np.array([0, 0, -1])
        with patch.object(driver, "parse_args", return_value=args), \
             patch.object(driver, "load_index_records", return_value=records), \
             patch.object(driver, "init_worker"), \
             patch.object(driver, "featurize_record", side_effect=lambda rec: result_by_id[rec.icsd_id]), \
             patch.object(driver.hdbscan, "HDBSCAN", return_value=clusterer), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(), 0)
        for name in ("summary.json", "progress.json"):
            saved = json.loads((self.root / name).read_text())
            self.assertEqual(saved["feature_version"], worker.FEATURE_VERSION)
            self.assertEqual(saved["feature_diagnostics"]["n_disordered"], 3)
            self.assertNotIn(self.secret, json.dumps(saved))
        self.assertEqual(np.load(self.root / "features.npy").shape, (3, 4491))


if __name__ == "__main__":
    unittest.main()
