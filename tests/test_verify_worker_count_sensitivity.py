"""Focused checks for the graphlet worker-count sensitivity verifier."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_repaired_downstream as verifier


METHODS = ("crystalnn", "voronoinn")
CANONICAL_HASHES = {
    "crystalnn": "1" * 64,
    "voronoinn": "2" * 64,
}
CANONICAL_PATHS = {
    method: f"representations/{method}/nearest_neighbors.csv"
    for method in METHODS
}


def recovery(hits):
    return {
        "hits": hits,
        "cohort": list(range(1, 11)),
        "recovery": {
            "neighbor_search_n_jobs": 48,
            "neighbor_search_random_state": 42,
            "exact_audit_max_workers": 16,
        },
    }


def variant(method, canonical_hits, superseded_hits, transitions):
    n_queries = 10
    return {
        "canonical_hits": canonical_hits,
        "canonical_minus_superseded_hits": canonical_hits - superseded_hits,
        "canonical_minus_superseded_percentage_points": (
            100 * (canonical_hits - superseded_hits) / n_queries),
        "canonical_neighbor_csv": (
            "notes/feature_repair_2026_09/downstream/"
            + CANONICAL_PATHS[method]),
        "canonical_neighbor_csv_sha256": CANONICAL_HASHES[method],
        "canonical_rate": canonical_hits / n_queries,
        "hit_status_changed": (
            transitions["hit_to_miss"] + transitions["miss_to_hit"]),
        "hit_transitions_superseded_to_canonical": transitions,
        "n_queries": n_queries,
        "query_ids_identical": True,
        "selected_neighbor_community_changed": 2,
        "selected_neighbor_id_changed": 3,
        "superseded_hits": superseded_hits,
        "superseded_neighbor_csv_private_archive_path": (
            f"downstream_voronoi_20260905/superseded/job3471058-worker112/"
            f"{method}_nearest_neighbors.csv"),
        "superseded_neighbor_csv_sha256": (
            "3" * 64 if method == "crystalnn" else "4" * 64),
        "superseded_rate": superseded_hits / n_queries,
    }


def document():
    return {
        "schema_version": 1,
        "canonical_run": {
            "elapsed": "00:18:29",
            "exact_audit_max_workers": 16,
            "exit_code": "0:0",
            "job_id": 3471253,
            "neighbor_search_n_jobs": 48,
            "neighbor_search_random_state": 42,
            "status": "COMPLETED",
        },
        "superseded_run": {
            "job_id": 3471058,
            "neighbor_search_n_jobs": 112,
            "private_archive_root_relative_to_archive": (
                "downstream_voronoi_20260905/superseded/"
                "job3471058-worker112"),
            "recovery_json_sha256": {
                "crystalnn": "5" * 64,
                "voronoinn": "6" * 64,
            },
            "status": "scientific stages completed before the run was cancelled",
        },
        "validation": {
            "ordered_query_ids_identical_for_both_methods": True,
            "status": "passed",
        },
        "variants": {
            "crystalnn": variant("crystalnn", 8, 8, {
                "hit_to_hit": 7,
                "hit_to_miss": 1,
                "miss_to_hit": 1,
                "miss_to_miss": 1,
            }),
            "voronoinn": variant("voronoinn", 7, 8, {
                "hit_to_hit": 7,
                "hit_to_miss": 1,
                "miss_to_hit": 0,
                "miss_to_miss": 2,
            }),
        },
    }


def checks(value):
    return verifier.worker_count_sensitivity_checks(
        value,
        {"crystalnn": recovery(8), "voronoinn": recovery(7)},
        CANONICAL_PATHS,
        CANONICAL_HASHES,
        10,
    )


class WorkerCountSensitivityTests(unittest.TestCase):
    def test_complete_record_passes(self):
        self.assertTrue(all(checks(document()).values()))

    def test_transition_and_hash_tampering_fail_independently(self):
        cases = {
            "transition_sum": (
                "crystalnn transition arithmetic and canonical identity",
                ("variants", "crystalnn",
                 "hit_transitions_superseded_to_canonical", "miss_to_miss"),
                2),
            "changed_status": (
                "voronoinn transition arithmetic and canonical identity",
                ("variants", "voronoinn", "hit_status_changed"),
                2),
            "canonical_hash": (
                "crystalnn transition arithmetic and canonical identity",
                ("variants", "crystalnn", "canonical_neighbor_csv_sha256"),
                "a" * 64),
            "old_hash": (
                "voronoinn transition arithmetic and canonical identity",
                ("variants", "voronoinn", "superseded_neighbor_csv_sha256"),
                "not-a-hash"),
        }
        for name, (failed_check, path, replacement) in cases.items():
            with self.subTest(name=name):
                changed = deepcopy(document())
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                self.assertFalse(checks(changed)[failed_check])

    def test_archive_reference_and_canonical_provenance_are_required(self):
        escaped = deepcopy(document())
        escaped["variants"]["crystalnn"][
            "superseded_neighbor_csv_private_archive_path"] = "../old.csv"
        self.assertFalse(checks(escaped)["private archive references"])

        wrong_workers = deepcopy(document())
        wrong_workers["canonical_run"]["neighbor_search_n_jobs"] = 112
        self.assertFalse(
            checks(wrong_workers)["record and canonical-run provenance"])


if __name__ == "__main__":
    unittest.main()
