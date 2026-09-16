"""Focused integration check for the packaged A-Lab target validation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = ROOT / "notes/feature_repair_2026_09/downstream"
sys.path.insert(0, str(ROOT / "scripts"))
import manifest_repaired_downstream as manifest
import verify_repaired_downstream as verifier


@unittest.skipUnless((DOWNSTREAM / "alab_mp_targets/SHA256SUMS").is_file(),
                     "Requires the repaired A-Lab derived-analysis package")
class AlabMpTargetPackageTests(unittest.TestCase):
    def test_packaged_results_pass_global_invariants_and_hash_coverage(self):
        audit = verifier.Audit(DOWNSTREAM)
        result = verifier.verify_alab_mp_targets(audit)

        self.assertTrue(all(check["passed"] for check in audit.checks))
        self.assertEqual(result["n_targets"], 57)
        self.assertEqual(result["outcome_counts"], {
            "made": 36,
            "not_obtained": 15,
            "offline_recovery": 2,
            "inconclusive": 4,
        })
        self.assertEqual(result["basin_table"], [[28, 8], [4, 11]])
        self.assertEqual(result["public_geometry_matches"], 57)
        self.assertEqual(result["independent_checks"], 283)
        self.assertEqual(result["package_files_hashed"], 30)

        actual = {
            str(path.relative_to(DOWNSTREAM))
            for path in (DOWNSTREAM / "alab_mp_targets").rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        hashed = {
            path for path in audit.inputs if path.startswith("alab_mp_targets/")
        }
        self.assertEqual(hashed, actual)
        self.assertEqual(
            verifier.ALAB_MP_TARGET_REQUIRED_FILES,
            manifest.ALAB_MP_TARGET_REQUIRED_FILES,
        )
        self.assertEqual(
            verifier.ALAB_MP_TARGET_SHA256SUMS_SHA256,
            manifest.ALAB_MP_TARGET_SHA256SUMS_SHA256,
        )
        self.assertEqual(
            verifier.ALAB_MP_TARGET_VERIFICATION_SHA256,
            manifest.ALAB_MP_TARGET_VERIFICATION_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
