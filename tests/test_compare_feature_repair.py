"""Shared-cohort birth scoring preserves fitted labels and checks its inputs."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_feature_repair as compare


class CommonCohortTests(unittest.TestCase):
    def setUp(self):
        # Historical community 10 loses its earliest member. Repaired community
        # 77 gains an earlier member. Shared year ties and noise must be retained.
        self.old = [(1, 1930, 10), (2, 2010, 10), (3, 2010, 10),
                    (4, 2010, -1), (5, None, 10), (6, 2010, 20)]
        self.new = [(2, 2010, 99), (3, 2010, 99), (4, 2010, -1),
                    (5, None, 99), (6, 2010, 77), (7, 1930, 77)]

    def supplement(self, old=None, new=None, **kwargs):
        old = self.old if old is None else old
        new = self.new if new is None else new
        ids = compare.cohort_comparison(old, new, "test")
        return compare.common_cohort_birth_shares(old, new, ids, **kwargs)

    def test_both_birth_definitions_count_ties_and_keep_noise(self):
        result = self.supplement()
        self.assertEqual(result["n_common"], 5)
        full = result["full_map_birth_year"]
        shared = result["common_cohort_birth_year"]
        self.assertEqual(full["old"]["by_decade"]["2010s"]["n_birth"], 1)
        self.assertEqual(full["new"]["by_decade"]["2010s"]["n_birth"], 2)
        self.assertEqual(full["birth_curve_comparison"]["2010s"]["change_percentage_points"], 25)
        for side in ("old", "new"):
            self.assertEqual(shared[side]["n_points"], 5)
            self.assertEqual(shared[side]["n_outliers"], 1)
            self.assertEqual(shared[side]["by_decade"]["2010s"]["n_total"], 4)
            self.assertEqual(shared[side]["by_decade"]["2010s"]["n_birth"], 3)
            self.assertEqual(shared[side]["by_decade"]["2010s"]["birth_share"], 0.75)
            self.assertEqual(shared[side]["by_decade"]["unknown"]["n_total"], 1)
            self.assertEqual(shared[side]["by_decade"]["unknown"]["n_birth"], 0)
            self.assertEqual(result["birth_year_changes"][side]["birth_years_shifted_later"], 1)
        self.assertEqual(shared["birth_curve_comparison"]["2010s"]["change_percentage_points"], 0)
        # Input populations and fitted labels are not modified.
        self.assertEqual(self.old[0], (1, 1930, 10))
        self.assertEqual(self.new[-1], (7, 1930, 77))

    def test_labels_need_not_match_between_partitions(self):
        renumbered = [(i, y, c + 1000 if c >= 0 else c) for i, y, c in self.new]
        self.assertEqual(self.supplement(), self.supplement(new=renumbered))

    def test_identical_ids_need_no_supplement_even_if_labels_differ(self):
        changed = [(i, y, 50) for i, y, _ in self.old]
        self.assertEqual(self.supplement(new=changed), {
            "status": "not_needed_identical_id_sets", "n_common": 6})

    def test_missing_labels_or_no_overlap_do_not_generate_metrics(self):
        result = self.supplement(historical_labels_available=False)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("--old-graphlet-assignments", result["reason"])
        self.assertNotIn("common_cohort_birth_year", result)
        result = self.supplement(new=[(100, 2010, 3)])
        self.assertEqual(result["n_common"], 0)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(compare.common_cohort_birth_shares(None, self.new, None)["status"], "unavailable")

    def test_changed_shared_year_is_rejected_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "Years changed"):
            self.supplement(new=[(2, 2011, 99), (7, 1930, 77)])
        with self.assertRaisesRegex(ValueError, "Years changed"):
            self.supplement(new=[(5, 2010, 99), (7, 1930, 77)])

    def test_duplicate_assignment_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignments.csv"
            path.write_text("icsd_id,year,community\n1,1930,1\n1,2010,2\n")
            with self.assertRaisesRegex(ValueError, "Duplicate ICSD ID"):
                compare.read_assignments(path)


class GraphletCliTests(unittest.TestCase):
    def test_only_actual_graphlet_labels_can_supply_common_cohort_births(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_prod = root / "old-production.csv"
            old_graph = root / "old-graphlets.csv"
            run = root / "new" / "graphlets"
            run.mkdir(parents=True)

            def assignments(path, rows):
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("icsd_id", "year", "community"))
                    writer.writerows(rows)

            assignments(old_prod, [(1, 1930, 7), (2, 2010, 7)])
            assignments(old_graph, [(1, 1930, 5), (2, 2010, 6)])
            assignments(run / "graphlet_community_assignments.csv",
                        [(1, 1930, 9), (2, 2010, 9), (3, 2000, 9)])

            def decade_row(born):
                return {"n_total": 1, "n_cluster_birth_point": born,
                        "cluster_birth_point_ratio": born, "n_outlier": 0,
                        "outlier_ratio": 0}

            old_summary = root / "old-graphlet-summary.json"
            old_summary.write_text(json.dumps({
                "n_points": 2, "n_outliers": 0, "outlier_ratio": 0,
                "community_birth_year": {"5": 1930, "6": 2010},
                "by_decade": {"1930s": decade_row(1), "2010s": decade_row(1)}}))
            (run / "graph_time_summary.json").write_text(json.dumps({
                "n_points": 3, "n_outliers": 0, "outlier_ratio": 0,
                "n_successful": 3, "n_failed": 0,
                "community_birth_year": {"9": 1930},
                "by_decade": {"1930s": decade_row(1), "2000s": decade_row(0),
                              "2010s": decade_row(0)}}))
            out = root / "comparison"
            argv = ["--new-run-root", str(root / "new"), "--out-dir", str(out),
                    "--representations", "graphlets", "--old-production-assignments", str(old_prod),
                    "--old-graphlet-summary", str(old_summary)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(argv), 0)
            result = json.loads((out / "comparison.json").read_text())["representations"]["graphlets"]
            self.assertEqual(result["successful_id_comparison"]["shared_count"], 2)
            self.assertEqual(result["common_cohort_birth_shares"]["status"], "unavailable")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(argv + ["--old-graphlet-assignments", str(old_graph)]), 0)
            result = json.loads((out / "comparison.json").read_text())["representations"]["graphlets"]
            shared = result["common_cohort_birth_shares"]
            self.assertEqual(shared["status"], "available")
            self.assertEqual(shared["full_map_birth_year"]["old"]["by_decade"]["2010s"]["birth_share"], 1)
            self.assertEqual(shared["common_cohort_birth_year"]["new"]["by_decade"]["2010s"]["birth_share"], 0)
            text = (out / "comparison.md").read_text()
            self.assertIn("no feature fitting, graph rebuilding, or reclustering", text)
            self.assertIn("Recompute each community's earliest known year among shared IDs", text)


if __name__ == "__main__":
    unittest.main()
