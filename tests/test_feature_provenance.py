"""Reject legacy or ambiguously versioned matrices before external projection."""
from __future__ import annotations

import json
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from crystal_neighbors import FEATURE_VERSION
from feature_provenance import require_feature_version


class FeatureProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.features = self.root / "features.npy"
        self.features.touch()

    def write_metadata(self, name, payload):
        (self.root / name).write_text(json.dumps(payload))

    def test_production_summary_accepts_current_version(self):
        metadata = {"feature_version": FEATURE_VERSION, "n_structures": 42}
        self.write_metadata("summary.json", metadata)
        self.assertEqual(require_feature_version(self.features), metadata)

    def test_packaged_metadata_accepts_current_version(self):
        self.write_metadata("summary.json", {"n_structures": 42})
        metadata = {"feature_version": FEATURE_VERSION}
        self.write_metadata("feature_metadata.json", metadata)
        self.assertEqual(require_feature_version(str(self.features)), metadata)

    def test_missing_and_legacy_metadata_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing feature_version.*Regenerate"):
            require_feature_version(self.features)
        self.write_metadata("summary.json", {"n_structures": 42})
        with self.assertRaisesRegex(ValueError, "Missing feature_version"):
            require_feature_version(self.features)

    def test_wrong_version_cannot_be_overridden_by_matching_sidecar(self):
        self.write_metadata("summary.json", {"feature_version": "legacy-v1"})
        self.write_metadata("feature_metadata.json", {"feature_version": FEATURE_VERSION})
        with self.assertRaisesRegex(ValueError, "Incompatible ICSD feature version 'legacy-v1'"):
            require_feature_version(self.features)

    def test_wrong_sidecar_cannot_be_hidden_by_matching_summary(self):
        self.write_metadata("summary.json", {"feature_version": FEATURE_VERSION})
        self.write_metadata("feature_metadata.json", {"feature_version": "legacy-v1"})
        with self.assertRaisesRegex(ValueError, "Incompatible ICSD feature version 'legacy-v1'"):
            require_feature_version(self.features)

    def test_invalid_metadata_is_rejected(self):
        (self.root / "summary.json").write_text("{truncated")
        with self.assertRaisesRegex(ValueError, "Cannot read feature provenance"):
            require_feature_version(self.features)
        self.write_metadata("summary.json", [])
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            require_feature_version(self.features)

    def test_symlink_uses_provenance_of_actual_feature_file(self):
        self.write_metadata("summary.json", {"feature_version": "legacy-v1"})
        link_dir = self.root / "repackaged"
        link_dir.mkdir()
        linked_features = link_dir / "features.npy"
        linked_features.symlink_to(self.features)
        (link_dir / "feature_metadata.json").write_text(
            json.dumps({"feature_version": FEATURE_VERSION})
        )
        with self.assertRaisesRegex(ValueError, "Incompatible ICSD feature version 'legacy-v1'"):
            require_feature_version(linked_features)

    @unittest.skipUnless(importlib.util.find_spec("dash"), "dashboard dependency is optional")
    def test_dashboard_displays_missing_version_without_loading_legacy_features(self):
        import numpy as np

        manifest = self.root / "dashboard_manifest.json"
        manifest.write_text(json.dumps({"artifacts": {}}))
        dashboard_path = (
            Path(__file__).resolve().parents[1] / "dashboard" / "dash_app.py"
        )
        spec = importlib.util.spec_from_file_location("feature_provenance_dashboard_test", dashboard_path)
        dashboard = importlib.util.module_from_spec(spec)
        self.addCleanup(sys.modules.pop, spec.name, None)
        sys.modules[spec.name] = dashboard
        env = {"ICSD_DASHBOARD_MANIFEST": str(manifest)}
        with patch.dict(os.environ, env), patch.object(
            np, "load", side_effect=AssertionError("Legacy feature matrix was loaded")
        ):
            spec.loader.exec_module(dashboard)
            self.assertEqual(dashboard.APP_STATUS, "incompatible_features")
            self.assertIn("incompatible feature version", dashboard.APP_STATUS_ERROR)
            self.assertIn("incompatible feature version", str(dashboard.config_hint()))
            dashboard.scoring_layout()
            with self.assertRaisesRegex(RuntimeError, "incompatible feature version"):
                dashboard.load_frozen_map()


if __name__ == "__main__":
    unittest.main()
