"""Exact search must preserve identities, ties, ranks and cohort exclusions."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_repaired_confused_pairs as comparison


class RepairedConfusedPairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = np.array([10, 20, 30, 40, 50])
        self.cdf = np.zeros((5, 64, 20), dtype=np.float32)
        self.cdf[:, :, -1] = 1
        self.cdf[2:4, 0, :-1] = .25
        self.cdf[4, 0, :-1] = .5
        self.cdf[4, 10, :-1] = .25
        self.cdf = self.cdf.reshape(5, -1)
        self.labels = {iid: {"year": 2000, "community": iid // 20} for iid in [10, 20, 30, 40, 50, 60]}
        self.labels[70] = {"year": None, "community": 3}
        self.labels[80] = {"year": 1880, "community": 4}
        self.labels[90] = {"year": 2000, "community": -1}
        self.pairs = [self.pair(q, n) for q, n in [(10, 30), (30, 10), (50, 40), (20, 10), (60, 10), (70, 10), (80, 10)]]
        self.metadata = {
            "feature_version": comparison.FEATURE_VERSION,
            "neighbor_settings": comparison.NEIGHBOR_SETTINGS,
            "representation": comparison.REPRESENTATION,
            "target_year_range": [1900, 2025],
            "cdf_row_order": "ascending ICSD ID",
            "n_requested": 6, "n_successful": 5, "n_failed": 1,
        }

    def pair(self, query, neighbor):
        return {"query_id": query, "neighbor_id": neighbor,
                "query_community": self.labels[query]["community"],
                "neighbor_community": self.labels[neighbor]["community"],
                "historical_distance_l1": 0.123}

    def compare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return comparison.compare(self.cdf, self.ids, self.labels, self.pairs, **kwargs)

    def write_inputs(self):
        np.save(self.root / "cdf.npy", self.cdf)
        (self.root / "ids.json").write_text(json.dumps(self.ids.tolist()))
        (self.root / "meta.json").write_text(json.dumps(self.metadata))
        lines = ["icsd_id,year,community"] + [f"{iid},{'' if r['year'] is None else r['year']},{r['community']}" for iid, r in self.labels.items()]
        (self.root / "labels.csv").write_text("\n".join(lines))
        old = [{**{k: v for k, v in p.items() if k != "historical_distance_l1"},
                "distance_l1": p["historical_distance_l1"]} for p in self.pairs]
        (self.root / "old.json").write_text(json.dumps(old))

    def load(self):
        return comparison.load_cdf(self.root / "cdf.npy", self.root / "ids.json", self.root / "meta.json", block_size=2)

    def test_ties_self_exclusion_ranks_and_distance_parts(self):
        result = self.compare(query_block_size=2, candidate_block_size=1)
        queries = {row["query_id"]: row for row in result["queries"]}
        self.assertEqual(queries[10]["nearest_id"], 20)
        self.assertEqual(queries[50]["nearest_tied_ids"], [30, 40])
        self.assertEqual(queries[50]["nearest_id"], 30)
        pairs = {(row["query_id"], row["neighbor_id"]): row for row in result["historical_pairs"]}
        row = pairs[10, 30]
        self.assertEqual((row["rank_first"], row["rank_last"], row["rank_with_smallest_id_tiebreak"]), (2, 3, 2))
        self.assertFalse(row["remains_nearest_including_ties"])
        row = pairs[50, 40]
        self.assertTrue(row["remains_nearest_including_ties"])
        self.assertFalse(row["is_selected_nearest"])
        self.assertEqual((row["rank_first"], row["rank_last"], row["rank_with_smallest_id_tiebreak"]), (1, 2, 2))
        self.assertEqual(row["repaired_distance"], {"distance_l1": 9.5,
            "first_10_elemental_channels_l1": 4.75, "remaining_54_pair_triplet_channels_l1": 4.75,
            "pure_geometry_four_channels_l1": 4.75, "pair_triplet_chemical_50_channels_l1": 0.0})

    def test_geometry_indices_match_registry_and_partitions_are_additive(self):
        graphlet_directory = Path(__file__).resolve().parents[1] / "experiments" / "graphlet_compare"
        sys.path.insert(0, str(graphlet_directory))
        self.addCleanup(sys.path.remove, str(graphlet_directory))
        import graphlet_features
        registry = graphlet_features.REGISTRY.all
        self.assertEqual(len(registry), 64)
        self.assertEqual({name: registry.index(name) for name in comparison.PURE_GEOMETRY_CHANNELS},
                         comparison.PURE_GEOMETRY_CHANNELS)
        self.assertEqual(len(comparison.PURE_GEOMETRY_COLUMNS), 4 * 20)
        self.assertEqual(len(comparison.PAIR_TRIPLET_CHEMICAL_COLUMNS), 50 * 20)
        # Every named geometric channel contributes; an adjacent chemical
        # channel must contribute only to the chemical partition.
        for channel in range(64):
            left, right = np.zeros(1280), np.zeros(1280)
            right[channel * 20:(channel + 1) * 20] = .25
            parts = comparison.distance_parts(left, right)
            self.assertEqual(parts["pure_geometry_four_channels_l1"],
                             5.0 if channel in comparison.PURE_GEOMETRY_CHANNELS.values() else 0.0)
            self.assertEqual(parts["pair_triplet_chemical_50_channels_l1"],
                             5.0 if channel >= 10 and channel not in comparison.PURE_GEOMETRY_CHANNELS.values() else 0.0)
        rng = np.random.default_rng(91)
        parts = comparison.distance_parts(rng.random(1280), rng.random(1280))
        self.assertAlmostEqual(parts["first_10_elemental_channels_l1"]
                               + parts["pure_geometry_four_channels_l1"]
                               + parts["pair_triplet_chemical_50_channels_l1"],
                               parts["distance_l1"], places=10)
        self.assertAlmostEqual(parts["pure_geometry_four_channels_l1"]
                               + parts["pair_triplet_chemical_50_channels_l1"],
                               parts["remaining_54_pair_triplet_channels_l1"], places=10)

    def test_chunking_matches_direct_brute_force_on_nontrivial_cdfs(self):
        rng = np.random.default_rng(19)
        histogram = rng.random((5, 64, 20))
        histogram /= histogram.sum(axis=2, keepdims=True)
        self.cdf = np.cumsum(histogram, axis=2).reshape(5, -1).astype(np.float32)
        self.cdf[3] = self.cdf[2]
        small = self.compare(query_block_size=1, candidate_block_size=2)
        large = self.compare(query_block_size=8, candidate_block_size=20)
        self.assertEqual(small, large)
        exact = cdist(self.cdf.astype(float), self.cdf.astype(float), metric="cityblock")
        np.fill_diagonal(exact, np.inf)
        index = {int(iid): i for i, iid in enumerate(self.ids)}
        for row in small["queries"]:
            q = index[row["query_id"]]
            ordered = sorted(range(5), key=lambda n: (exact[q, n], self.ids[n]))
            self.assertEqual(row["nearest_id"], self.ids[ordered[0]])
            self.assertEqual(row["nearest_tied_ids"], self.ids[exact[q] == np.min(exact[q])].tolist())
            self.assertEqual(row["nearest_distance"]["distance_l1"], float(np.min(exact[q])))
        for row in small["historical_pairs"]:
            if row["status"] != "compared":
                continue
            q, n = index[row["query_id"]], index[row["neighbor_id"]]
            ordered = sorted(range(5), key=lambda j: (exact[q, j], self.ids[j]))
            self.assertEqual(row["rank_with_smallest_id_tiebreak"], ordered.index(n) + 1)
            self.assertEqual(row["rank_first"], int(np.count_nonzero(exact[q] < exact[q, n])) + 1)
            self.assertEqual(row["rank_last"], int(np.count_nonzero(exact[q] <= exact[q, n])))

    def test_cohort_exclusions_are_explicit(self):
        result = self.compare()
        summary = result["summary"]
        self.assertEqual(summary["endpoint_queries_requested"], 8)
        self.assertEqual(summary["endpoint_queries_available"], 5)
        self.assertEqual({row["icsd_id"]: row["reason"] for row in summary["unavailable_queries"]}, {
            60: "eligible_but_absent_from_saved_cdf", 70: "no_year_in_historical_labels",
            80: "year_outside_repaired_target_range"})
        self.assertEqual(summary["candidate_pool"]["historical_nonnoise_label_count"], 8)
        self.assertEqual(summary["candidate_pool"]["historical_dated_1900_2025_nonnoise_count"], 6)
        self.assertEqual(summary["candidate_pool"]["dated_target_ids_absent_from_cdf"], [60])
        self.assertEqual(summary["historical_directed_pairs_unavailable"], 3)

    def test_rejects_bad_version_alignment_and_invalid_cdf(self):
        self.write_inputs()
        cdf, ids, _ = self.load()
        self.assertEqual(cdf.shape, (5, 1280))
        self.assertEqual(ids.tolist(), self.ids.tolist())
        for field, value in [("feature_version", "legacy"), ("neighbor_settings", {}),
                             ("representation", "histograms"), ("n_successful", 4),
                             ("n_failed", 0), ("target_year_range", [1800, 2025])]:
            with self.subTest(field=field):
                (self.root / "meta.json").write_text(json.dumps({**self.metadata, field: value}))
                with self.assertRaises(ValueError):
                    self.load()
        (self.root / "meta.json").write_text(json.dumps(self.metadata))
        (self.root / "ids.json").write_text(json.dumps([10, 20, 20, 40, 50]))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            self.load()
        (self.root / "ids.json").write_text(json.dumps(self.ids.tolist()))
        for column, value in [(1, float("nan")), (19, .9), (0, 2.0), (1, -.1)]:
            with self.subTest(column=column, value=value):
                invalid = self.cdf.copy()
                invalid[0, column] = value
                np.save(self.root / "cdf.npy", invalid)
                with self.assertRaises(ValueError):
                    self.load()

    def test_cli_writes_strict_json_and_validates_pair_communities(self):
        self.write_inputs()
        with contextlib.redirect_stdout(io.StringIO()):
            comparison.main(["--cdf", str(self.root / "cdf.npy"), "--ids", str(self.root / "ids.json"),
                "--metadata", str(self.root / "meta.json"), "--old-pairs", str(self.root / "old.json"),
                "--old-community-labels", str(self.root / "labels.csv"), "--outdir", str(self.root / "out"),
                "--candidate-block-size", "2"])
        result = json.loads((self.root / "out" / "repaired_confused_pairs.json").read_text())
        self.assertEqual(result["summary"]["historical_directed_pairs_compared"], 4)
        self.assertEqual(result["provenance"]["cdf_dtype"], "float32")
        self.assertTrue((self.root / "out" / "repaired_confused_pairs.md").is_file())
        rows = json.loads((self.root / "old.json").read_text())
        rows[0]["query_community"] = 99
        (self.root / "old.json").write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "community disagrees"):
            comparison.read_old_pairs(self.root / "old.json", self.labels)


if __name__ == "__main__":
    unittest.main()
