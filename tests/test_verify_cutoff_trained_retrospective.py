"""Focused integration checks for the cutoff-trained retrospective package."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = ROOT / "notes/feature_repair_2026_09/downstream"
sys.path.insert(0, str(ROOT / "scripts"))
import manifest_repaired_downstream as manifest
import verify_repaired_downstream as verifier


class CutoffTrainedPackageTests(unittest.TestCase):
    def require_current_formula_package(self):
        """Skip until the final layered-match/uniform-group TACC package is integrated."""
        summary_path = (
            DOWNSTREAM
            / "cutoff_trained/cutoff_trained_retrospective_summary.json"
        )
        if not summary_path.is_file():
            self.skipTest("Requires the repaired cutoff-trained derived-analysis package")
        summary = json.loads(summary_path.read_text())
        required_inputs = {
            "occupancy_flags",
            "formula_conventions",
            "index_loader",
            "wrapper",
        }
        required_cutoff_fields = {
            "reference_composition_counts",
            "earliest_year_tie_audit",
        }
        current_schema = required_inputs <= set(summary.get("inputs", {})) and all(
            required_cutoff_fields <= set(saved)
            for saved in summary.get("cutoffs", {}).values()
        )
        if not current_schema:
            self.skipTest(
                "FINAL-OUTPUT INTEGRATION POINT: replace the archived cutoff-trained "
                "package with the occupancy-aware final run, refresh package hashes/"
                "inventories, and then enable these invariant checks"
            )
        pin_names = (
            "CUTOFF_TRAINED_SLURM_PATH",
            "CUTOFF_TRAINED_SUMMARY_SHA256",
            "CUTOFF_TRAINED_REPORTING_MANIFEST_SHA256",
            "CUTOFF_TRAINED_SLURM_SHA256",
            "CUTOFF_TRAINED_INDEPENDENT_VERIFICATION_SHA256",
        )
        if any(getattr(verifier, name) is None for name in pin_names):
            self.skipTest(
                "FINAL-OUTPUT INTEGRATION POINT: run the standalone verifier on "
                "the completed package, then set its Slurm path and four canonical "
                "hash pins in the global verifier and manifest"
            )
        return summary

    def test_saved_results_pass_global_invariants_and_hash_coverage(self):
        summary = self.require_current_formula_package()
        audit = verifier.Audit(DOWNSTREAM)
        prod = audit.csv("../full_run_results/production/graph/community_assignments.csv")
        basis = audit.js("reference_basis/projection_basis_provenance.json")
        source_rows = {}
        for source, (directory, stem, _) in verifier.SOURCES.items():
            source_rows[source] = audit.csv(
                f"external/{directory}/{stem}_frontier_records.csv"
            )
        label_hash = verifier.sha256(
            DOWNSTREAM / "../full_run_results/production/graph/community_assignments.csv"
        )

        result = verifier.verify_cutoff_trained_retrospective(
            audit, prod, source_rows, basis, label_hash,
        )

        self.assertTrue(all(check["passed"] for check in audit.checks))
        self.assertGreater(len(audit.checks), 0)
        self.assertEqual(
            result["package_files_hashed"],
            len(verifier.CUTOFF_TRAINED_REQUIRED_FILES),
        )
        self.assertEqual(
            result["archive_files"], len(verifier.CUTOFF_TRAINED_ARCHIVE_FILES)
        )
        self.assertEqual(
            result["selected_files"], len(verifier.CUTOFF_TRAINED_SELECTED_FILES)
        )
        cutoffs = sorted(int(value) for value in summary["cutoffs"])
        n_sources = len(verifier.SOURCES)
        n_references = len(next(iter(summary["cutoffs"].values()))["analyses"])
        self.assertEqual(
            result["reporting_rows"],
            {
                "rates": (1 + n_sources) * len(cutoffs),
                "shared_strata": 2 * n_sources * len(cutoffs),
                "joint": 2 * n_references * len(cutoffs),
            },
        )
        self.assertGreater(result["independent_checks"], 0)
        self.assertEqual(set(result["cutoffs"]), {str(value) for value in cutoffs})
        for cutoff, saved in result["cutoffs"].items():
            self.assertGreater(saved["n_train"], 0, cutoff)
            self.assertGreater(saved["n_heldout"], 0, cutoff)
            self.assertGreater(saved["n_communities"], 0, cutoff)
            self.assertGreaterEqual(saved["heldout_in_basin"], 0, cutoff)
            self.assertLessEqual(
                saved["heldout_in_basin"], saved["n_heldout"], cutoff
            )
        hashed = {
            path for path in audit.inputs if path.startswith("cutoff_trained/")
        }
        self.assertEqual(hashed, verifier.CUTOFF_TRAINED_REQUIRED_FILES)
        self.assertEqual(
            verifier.CUTOFF_TRAINED_REQUIRED_FILES,
            manifest.CUTOFF_TRAINED_REQUIRED_FILES,
        )
        self.assertEqual(
            verifier.CUTOFF_TRAINED_ARCHIVE_FILES,
            manifest.CUTOFF_TRAINED_ARCHIVE_FILES,
        )
        self.assertEqual(
            verifier.CUTOFF_TRAINED_SELECTED_FILES,
            manifest.CUTOFF_TRAINED_SELECTED_FILES,
        )
        for name in (
            "CUTOFF_TRAINED_SLURM_PATH",
            "CUTOFF_TRAINED_SUMMARY_SHA256",
            "CUTOFF_TRAINED_REPORTING_MANIFEST_SHA256",
            "CUTOFF_TRAINED_SLURM_SHA256",
            "CUTOFF_TRAINED_INDEPENDENT_VERIFICATION_SHA256",
        ):
            self.assertEqual(getattr(verifier, name), getattr(manifest, name), name)

    def test_manifest_applies_release_storage_policy(self):
        self.require_current_formula_package()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "downstream"
            target.mkdir()

            def copy_tree(name):
                source = DOWNSTREAM / name
                destination = target / name
                try:
                    shutil.copytree(source, destination, copy_function=os.link)
                except OSError:
                    shutil.copytree(source, destination, copy_function=shutil.copy2)

            copy_tree("alab_mp_targets")
            copy_tree("cutoff_trained")
            (target / "reference_basis").mkdir()
            shutil.copy2(
                DOWNSTREAM / "reference_basis/projection_basis_provenance.json",
                target / "reference_basis/projection_basis_provenance.json",
            )
            with mock.patch.object(
                sys, "argv", ["manifest_repaired_downstream.py", "--root", str(target)]
            ):
                manifest.main()
            report = json.loads((target / "manifest.json").read_text())
            cutoff_records = {
                row["path"]: row["storage"]
                for row in report["artifacts"]
                if row["path"].startswith("cutoff_trained/")
            }
            self.assertEqual(set(cutoff_records), manifest.CUTOFF_TRAINED_REQUIRED_FILES)
            self.assertEqual(
                {path for path, storage in cutoff_records.items()
                 if storage == "derived_archive"},
                manifest.CUTOFF_TRAINED_ARCHIVE_FILES,
            )
            self.assertEqual(
                {path for path, storage in cutoff_records.items()
                 if storage == "git_selected"},
                manifest.CUTOFF_TRAINED_SELECTED_FILES,
            )


if __name__ == "__main__":
    unittest.main()
