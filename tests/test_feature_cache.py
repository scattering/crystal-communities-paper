"""Cache isolation, feature provenance, and credential-safe driver outputs."""
from __future__ import annotations

import argparse
import io
import json
import secrets
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import icsd_continuous_wl_densification as driver


def diagnostics(*, ordered=True, uncovered_sites=0, uncovered_mass=0.0):
    return {
        "n_sites": 4,
        "ordered": ordered,
        "sites_with_unrepresented_cn_mass": uncovered_sites,
        "max_unrepresented_cn_mass": uncovered_mass,
    }


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        # Generated only in memory: no real or fixed credential is test data.
        self.secret = secrets.token_urlsafe(32)
        self.args = argparse.Namespace(
            local_mode="matminer_ops", wl_iters=3, max_sites=256,
            zip_password=self.secret, zip_password_stdin=True,
        )
        self.rec = driver.Record(icsd_id=123, year=2001, row={})
        self.embedding = np.arange(213, dtype=float)

    def test_legacy_vector_is_not_reused_even_if_renamed(self):
        old = self.cache / "icsd_000123_v2_lm-matminer_ops_wl-3_ms-256.npz"
        np.savez_compressed(old, status="ok", embedding=self.embedding)
        self.assertIsNone(driver.load_cached_result(self.cache, self.rec, self.args))
        old.replace(self.cache / driver.cache_key(self.rec, self.args))
        self.assertIsNone(driver.load_cached_result(self.cache, self.rec, self.args))

    def test_new_cache_round_trips_vector_and_diagnostics_without_secrets(self):
        info = diagnostics(ordered=False, uncovered_sites=2, uncovered_mass=0.3)
        info["unexpected_private_field"] = self.secret
        driver.store_cached_result(self.cache, self.rec, self.args, True, self.embedding, info)
        ok, rec, embedding, actual = driver.load_cached_result(self.cache, self.rec, self.args)
        self.assertTrue(ok)
        self.assertEqual(rec, self.rec)
        np.testing.assert_array_equal(embedding, self.embedding)
        self.assertEqual(actual, diagnostics(ordered=False, uncovered_sites=2, uncovered_mass=0.3))
        with np.load(self.cache / driver.cache_key(self.rec, self.args), allow_pickle=False) as data:
            metadata = json.loads(str(data["feature_metadata"]))
            self.assertEqual(metadata["feature_version"], driver.FEATURE_VERSION)
            self.assertEqual(metadata["neighbor_settings"], driver.NEIGHBOR_SETTINGS)
            self.assertNotIn("zip_password", metadata)
            self.assertNotIn(self.secret, str(data["feature_metadata"]))
            self.assertNotIn(self.secret, str(data["feature_diagnostics"]))

    def test_version_and_neighbor_changes_invalidate_cache(self):
        driver.store_cached_result(self.cache, self.rec, self.args, True, self.embedding, diagnostics())
        original = driver.cache_key(self.rec, self.args)
        with patch.object(driver, "FEATURE_VERSION", "future-feature-version"):
            self.assertNotEqual(original, driver.cache_key(self.rec, self.args))
            self.assertIsNone(driver.load_cached_result(self.cache, self.rec, self.args))
        with patch.dict(driver.NEIGHBOR_SETTINGS, {"x_diff_weight": 1.0}):
            self.assertNotEqual(original, driver.cache_key(self.rec, self.args))
            self.assertIsNone(driver.load_cached_result(self.cache, self.rec, self.args))
        self.assertIsNotNone(driver.load_cached_result(self.cache, self.rec, self.args))

    def test_mismatched_metadata_and_nonfinite_vectors_are_rejected(self):
        path = self.cache / driver.cache_key(self.rec, self.args)
        metadata = driver.feature_metadata(self.args)
        metadata["feature_version"] = "old-feature-version"
        np.savez_compressed(path, status="ok", embedding=self.embedding,
                            feature_metadata=json.dumps(metadata),
                            feature_diagnostics=json.dumps(diagnostics()))
        self.assertIsNone(driver.load_cached_result(self.cache, self.rec, self.args))
        invalid = self.embedding.copy()
        invalid[0] = np.nan
        driver.store_cached_result(self.cache, self.rec, self.args, True, invalid, diagnostics())
        self.assertIsNone(driver.load_cached_result(self.cache, self.rec, self.args))

    def test_protected_failure_details_are_never_persisted(self):
        failure = {
            "icsd_id": self.rec.icsd_id,
            "reason": "RuntimeError",
            "detail": self.secret[:12],  # Even truncated exception text must not leak.
            "unexpected_private_field": self.secret,
        }
        driver.store_cached_result(self.cache, self.rec, self.args, False, None, failure)
        ok, _, _, actual = driver.load_cached_result(self.cache, self.rec, self.args)
        self.assertFalse(ok)
        self.assertEqual(actual["reason"], "RuntimeError")
        self.assertNotIn(self.secret[:12], json.dumps(actual))
        with np.load(self.cache / driver.cache_key(self.rec, self.args), allow_pickle=False) as data:
            for field in data.files:
                self.assertNotIn(self.secret[:12], str(data[field]))

    def test_stdin_password_and_legacy_flag_are_mutually_exclusive(self):
        required = ["--index-csv", "index.csv", "--output-dir", "output"]
        with patch.object(sys, "stdin", io.StringIO(self.secret + "\n")):
            parsed = driver.parse_args(required + ["--zip-password-stdin"])
        self.assertEqual(parsed.zip_password, self.secret)
        legacy = driver.parse_args(required + ["--zip-password", self.secret])
        self.assertEqual(legacy.zip_password, self.secret)
        with patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
            driver.parse_args(required + ["--zip-password", self.secret, "--zip-password-stdin"])
        with patch.object(sys, "stdin", io.StringIO("")), patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
            driver.parse_args(required + ["--zip-password-stdin"])

    def test_new_and_cached_diagnostics_reach_summary_and_progress(self):
        records = [driver.Record(i, 2000 + i, {}) for i in (1, 2, 3)]
        infos = {
            1: diagnostics(ordered=False, uncovered_sites=1, uncovered_mass=0.2),
            2: diagnostics(),
            3: diagnostics(ordered=False, uncovered_sites=2, uncovered_mass=0.4),
        }
        features = {i: np.array([i, i * i, i + 2.0]) for i in (1, 2, 3)}
        driver.store_cached_result(self.cache, records[0], self.args, True, features[1], infos[1])
        self.args.__dict__.update(
            icsd_zip="archive.zip", cif_root=None, index_csv="index.csv",
            output_dir=str(self.cache / "run"), cache_dir=str(self.cache),
            sample_size=3, seed=42, n_jobs=1, checkpoint_every=1,
            pca_dim=2, min_cluster_size=2, min_samples=1, run_umap=False,
        )
        clusterer = Mock()
        clusterer.fit_predict.return_value = np.zeros(3, dtype=int)
        fake_hdbscan = SimpleNamespace(HDBSCAN=Mock(return_value=clusterer))
        def featurize(rec):
            return True, rec, features[rec.icsd_id], infos[rec.icsd_id]
        with patch.object(driver, "parse_args", return_value=self.args), \
             patch.object(driver, "load_index_records", return_value=records), \
             patch.object(driver, "init_worker"), \
             patch.object(driver, "featurize_record", side_effect=featurize) as worker, \
             patch.object(driver, "hdbscan", fake_hdbscan), \
             patch.object(sys, "stdout", io.StringIO()) as stdout:
            self.assertEqual(driver.main(), 0)
        self.assertEqual(worker.call_count, 2)
        expected = {
            "n_structures": 3, "n_disordered": 2,
            "n_structures_with_unrepresented_cn_mass": 2,
            "n_sites_with_unrepresented_cn_mass": 3,
            "max_unrepresented_cn_mass": 0.4,
        }
        for filename in ("summary.json", "progress.json"):
            text = (self.cache / "run" / filename).read_text()
            result = json.loads(text)
            self.assertEqual(result["feature_version"], driver.FEATURE_VERSION)
            self.assertEqual(result["neighbor_settings"], driver.NEIGHBOR_SETTINGS)
            self.assertEqual(result["feature_diagnostics"], expected)
            self.assertNotIn(self.secret, text)
            self.assertNotIn("zip_password", result)
        self.assertNotIn(self.secret, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
