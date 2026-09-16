"""Focused checks for the strict graphlet-neighbor sensitivity join."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = ROOT / "notes" / "feature_repair_2026_09" / "downstream"
sys.path.insert(0, str(DOWNSTREAM))
import summarize_graphlet_neighbor_sensitivity as sensitivity


HEX = {
    "pca": "2" * 64,
    "pca_ids": "3" * 64,
    "features": "5" * 64,
    "source": "6" * 64,
}


class GraphletNeighborSensitivitySummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = list(range(1, 513))
        self.production_communities = {
            iid: 10 if iid <= 256 else 20 for iid in self.ids
        }
        self.years = {
            iid: 1930 + ((iid - 1) % 10) if iid <= 256 else 2010 + ((iid - 257) % 10)
            for iid in self.ids
        }
        self.paths = {("shared", "production_labels"): self.root / "production_labels.csv"}
        for method in sensitivity.METHODS:
            for name in (
                "feature_metadata", "feature_ids", "recovery", "recovery_ids", "neighbors",
                "exact_audit", "partition", "partition_ids", "assignments", "temporal_events",
                "exclusive_by_decade",
            ):
                suffix = ".csv" if name in ("neighbors", "assignments", "temporal_events") else ".json"
                self.paths[(method, name)] = self.root / f"{method}_{name}{suffix}"
            replay_root = self.root / f"{method}_replay"
            replay_root.mkdir()
            self.paths[(method, "pca_subset_scaler")] = replay_root / "pca_subset_scaler.json"
            for control, analysis_name in sensitivity.PCA_CONTROLS.items():
                directory = replay_root / analysis_name
                directory.mkdir()
                self.paths[(method, f"pca_{control}_dir")] = directory
        self._write_complete_fixture()

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")

    @staticmethod
    def write_csv(path, fields, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def feature_metadata(self, method):
        return {
            "kind": "graphlet",
            "complete": True,
            "feature_version": sensitivity.FEATURE_VERSION,
            "neighbor_method": method,
            "neighbor_settings": sensitivity.gf.neighbor_settings(method),
            "graphlet_feature_version": sensitivity.gf.GRAPHLET_FEATURE_VERSIONS[method],
            "representation": sensitivity.gf.GRAPHLET_REPRESENTATION,
            "n_requested": 513,
            "n_successful": len(self.ids),
            "n_failed": 1,
            "feature_shape": [len(self.ids), 1280],
            "production_labels_sha256": sensitivity.digest(
                self.paths[("shared", "production_labels")]
            ),
            "ids_sha256": sensitivity.digest(self.paths[(method, "feature_ids")]),
            "features_sha256": HEX["features"],
        }

    def run_provenance(self, method):
        metadata_path = self.paths[(method, "feature_metadata")]
        other = "voronoinn" if method == "crystalnn" else "crystalnn"
        return {
            "production_feature_version": sensitivity.FEATURE_VERSION,
            "feature_preparation": str(metadata_path),
            "feature_preparation_sha256": sensitivity.digest(metadata_path),
            "production_labels_sha256": sensitivity.digest(
                self.paths[("shared", "production_labels")]
            ),
            "production_pca_sha256": HEX["pca"],
            "production_pca_ids_sha256": HEX["pca_ids"],
            "source_sha256": HEX["source"],
            "neighbor_search_n_jobs": 4,
            "neighbor_search_random_state": 42,
            "exact_audit_max_workers": 4,
            "n_feature_successes_before_cohort_restriction": len(self.ids),
            "restrict_ids": [{
                "path": str(self.paths[(other, "feature_ids")]),
                "sha256": sensitivity.digest(self.paths[(other, "feature_ids")]),
            }],
            "packages": {
                "numpy": "test",
                "scikit-learn": "test",
                "networkx": "3.6.1",
            },
            "reference_labels_role": "corrected full-production Louvain labels joined by ID; no relabeling or inference from historical numeric IDs",
            "graphlet_neighbor_method": method,
            "graphlet_neighbor_settings": sensitivity.gf.neighbor_settings(method),
            "graphlet_feature_version": sensitivity.gf.GRAPHLET_FEATURE_VERSIONS[method],
            "source_graphlet_feature_version": sensitivity.gf.GRAPHLET_FEATURE_VERSIONS[method],
            "legacy_crystalnn_metadata_compatibility": False,
        }

    def _write_recovery(self, method, neighbors):
        ids_path = self.paths[(method, "recovery_ids")]
        self.write_json(ids_path, self.ids)
        rows = []
        for query, neighbor in zip(self.ids, neighbors):
            same = self.production_communities[query] == self.production_communities[neighbor]
            rows.append({
                "query_id": query,
                "query_community": self.production_communities[query],
                "neighbor_id": neighbor,
                "neighbor_community": self.production_communities[neighbor],
                "distance": 1.0,
                "distance_l1_recomputed": 1.0,
                "first_ten_chemical_channels_l1": 0.2,
                "pure_geometry_four_channels_l1": 0.3,
                "pair_triplet_chemical_50_channels_l1": 0.5,
                "same_community": same,
            })
        self.write_csv(
            self.paths[(method, "neighbors")],
            [
                "query_id", "query_community", "neighbor_id", "neighbor_community", "distance",
                "same_community", "distance_l1_recomputed", "first_ten_chemical_channels_l1",
                "pure_geometry_four_channels_l1", "pair_triplet_chemical_50_channels_l1",
            ],
            rows,
        )
        hits = sum(row["same_community"] for row in rows)
        queries = []
        for query, neighbor in zip(self.ids, neighbors):
            same = self.production_communities[query] == self.production_communities[neighbor]
            queries.append({
                "query_id": query,
                "query_community": self.production_communities[query],
                "exact_selected_neighbor_id": neighbor,
                "exact_selected_neighbor_community": self.production_communities[neighbor],
                "exact_minimum_l1": 1.0,
                "exact_nearest_tied_ids": [neighbor],
                "any_exact_tie_same_community": same,
                "all_exact_ties_same_community": same,
                "ann_selected_neighbor_id": neighbor,
                "ann_selected_distance_l1": 1.0,
                "ann_selected_is_exact_nearest": True,
                "ann_distance_excess_l1": 0.0,
            })
        audit = {
            "n_queries": len(queries),
            "query_seed": 42,
            "parallel_workers": 4,
            "candidate_pool_size": len(self.ids),
            "distance": "exact float64 cityblock on stored float32 CDF values",
            "tie_policy": "all exact-equal ties retained; smallest ICSD ID selected",
            "ann_exact_nearest_fraction": sum(q["ann_selected_is_exact_nearest"] for q in queries) / len(queries),
            "exact_selected_label_agreement": sum(
                self.production_communities[q["query_id"]]
                == self.production_communities[q["exact_selected_neighbor_id"]]
                for q in queries
            ) / len(queries),
            "queries": queries,
        }
        recovery = {
            **self.run_provenance(method),
            "cohort_sha256": sensitivity.digest(ids_path),
            "name": "graphlet_l1",
            "metric": "manhattan",
            "approximate": True,
            "estimator": "pynndescent.NNDescent",
            "ann_requested_neighbors_including_self": 15,
            "definition": "label of the selected single non-self nearest neighbor equals the query's corrected production label",
            "dated_only": False,
            "target_year_range": None,
            "n_undated": 0,
            "n_entries": len(self.ids),
            "n_candidates": len(self.ids),
            "n_communities": 2,
            "hits": hits,
            "misses": len(self.ids) - hits,
            "same_community_1nn_agreement": hits / len(self.ids),
            "exact_query_audit": {key: value for key, value in audit.items() if key != "queries"},
        }
        self.write_json(self.paths[(method, "recovery")], recovery)
        self.write_json(self.paths[(method, "exact_audit")], audit)

    def _write_temporal(self, method, assignment_records, partition, artifact_paths=None):
        birth_year = {}
        for row in assignment_records:
            if row["community"] >= 0:
                birth_year[row["community"]] = min(
                    row["year"], birth_year.get(row["community"], row["year"])
                )
        events = []
        for row in sorted(assignment_records, key=lambda item: (item["year"], item["icsd_id"])):
            if row["community"] < 0:
                event_type = "outlier"
            elif row["year"] == birth_year[row["community"]]:
                event_type = "community_birth"
            else:
                event_type = "existing_community"
            active = int(event_type == "existing_community")
            events.append({
                "icsd_id": row["icsd_id"],
                "year": row["year"],
                "decade": f"{(row['year'] // 10) * 10}s",
                "community": row["community"],
                "n_active_neighbors": active,
                "n_active_same_community_neighbors": active,
                "n_active_other_communities": 0,
                "is_bridge_attachment": False,
                "event_type": event_type,
                "distance_to_centroid": "" if row["community"] < 0 else 0.0,
                "core_threshold": "" if row["community"] < 0 else 0.0,
                "core_periphery": "" if row["community"] < 0 else "core",
            })
        event_path = (
            self.paths[(method, "temporal_events")]
            if artifact_paths is None
            else artifact_paths["temporal_events"]
        )
        self.write_csv(event_path, list(events[0]), events)
        validated_events = sensitivity.read_temporal_events(
            event_path,
            {
                "records": assignment_records,
                "document": partition,
                "ids": [row["icsd_id"] for row in assignment_records],
                "n_noise": partition["n_noise"],
            },
            f"{method} fixture",
        )
        exclusive = {
            **partition,
            "denominator": "all entries in the given decade, including representation-partition noise",
            "rows": sensitivity.recompute_exclusive_rows(validated_events),
        }
        exclusive_path = (
            self.paths[(method, "exclusive_by_decade")]
            if artifact_paths is None
            else artifact_paths["exclusive_by_decade"]
        )
        self.write_json(exclusive_path, exclusive)

        if artifact_paths is None:
            return
        by_decade = {}
        for decade in sorted({event["decade"] for event in validated_events}):
            rows = [event for event in validated_events if event["decade"] == decade]
            total = len(rows)
            values = {
                "n_total": total,
                "n_outlier": sum(event["event_type"] == "outlier" for event in rows),
                "n_cluster_birth_point": sum(
                    event["event_type"] == "community_birth" for event in rows
                ),
                "n_existing_cluster": sum(
                    event["event_type"] == "existing_community" for event in rows
                ),
                "n_same_community_attachment": sum(
                    event["community"] >= 0
                    and event["n_active_same_community_neighbors"] > 0
                    for event in rows
                ),
                "n_cross_community_attachment": sum(
                    event["community"] >= 0 and event["n_active_other_communities"] > 0
                    for event in rows
                ),
                "n_bridge_attachment": sum(
                    event["community"] >= 0 and event["is_bridge_attachment"]
                    for event in rows
                ),
                "n_core_attachment": sum(
                    event["core_periphery"] == "core" for event in rows
                ),
                "n_periphery_attachment": sum(
                    event["core_periphery"] == "periphery" for event in rows
                ),
            }
            for field in (
                "n_outlier", "n_cluster_birth_point", "n_existing_cluster",
                "n_same_community_attachment", "n_cross_community_attachment",
                "n_bridge_attachment", "n_core_attachment", "n_periphery_attachment",
            ):
                values[field.removeprefix("n_") + "_ratio"] = values[field] / total
            by_decade[decade] = values
        graph_time = {
            "n_points": len(assignment_records),
            "n_outliers": partition["n_noise"],
            "outlier_ratio": partition["n_noise"] / len(assignment_records),
            "community_birth_year": {
                str(key): value for key, value in sorted(birth_year.items())
            },
            "by_decade": by_decade,
            **partition,
        }
        self.write_json(artifact_paths["graph_time_summary"], graph_time)

        counts = sensitivity.Counter(
            row["community"] for row in assignment_records if row["community"] >= 0
        )
        top = [
            {"community": community, "size": size}
            for community, size in counts.most_common(25)
        ]
        self.write_json(artifact_paths["top_communities"], top)
        decades = sorted({event["decade"] for event in validated_events})
        event_counts = sensitivity.Counter(
            (event["community"], event["decade"])
            for event in validated_events
            if event["community"] >= 0
        )
        growth = []
        for item in top:
            running = 0
            for decade in decades:
                running += event_counts[(item["community"], decade)]
                growth.append({
                    "community": item["community"],
                    "decade": decade,
                    "cumulative_size": running,
                })
        self.write_csv(
            artifact_paths["community_growth_by_decade"],
            ["community", "decade", "cumulative_size"],
            growth,
        )

    def _write_partition(
        self, method, labels, ids=None, *, analysis_name="graphlet", artifact_paths=None
    ):
        ids = self.ids if ids is None else ids
        label_by_id = dict(zip(self.ids, labels))
        ids_path = (
            self.paths[(method, "partition_ids")]
            if artifact_paths is None
            else artifact_paths["ids"]
        )
        self.write_json(ids_path, ids)
        assignments = [
            {"icsd_id": iid, "year": self.years[iid], "community": label_by_id[iid]}
            for iid in ids
        ]
        assignments_path = (
            self.paths[(method, "assignments")]
            if artifact_paths is None
            else artifact_paths["assignments"]
        )
        self.write_csv(
            assignments_path,
            ["icsd_id", "year", "community"],
            assignments,
        )
        ordered_labels = [label_by_id[iid] for iid in ids]
        birth_year = {}
        for row in assignments:
            if row["community"] >= 0:
                birth_year[row["community"]] = min(
                    row["year"], birth_year.get(row["community"], row["year"])
                )
        n_edges = sum(
            row["community"] >= 0 and row["year"] > birth_year[row["community"]]
            for row in assignments
        )
        partition = {
            **self.run_provenance(method),
            "cohort_sha256": sensitivity.digest(ids_path),
            "name": analysis_name,
            "protocol": "historical_graphlet_temporal",
            "dated_only": True,
            "target_year_range": [1900, 2025],
            "knn_k": min(16, len(ids) - 1),
            "metric": "euclidean",
            "approximate_neighbors": True,
            "mutual_knn": True,
            "weight": "exp(-distance^2/sigma^2)",
            "sigma": 1.0,
            "sigma_population": "positive retained neighbor distances",
            "resolution": 1.0,
            "louvain_seed": 42,
            "louvain_implementation": "networkx-3.6.1-compatible-undirected-with-node-move-tolerance",
            "louvain_node_move_gain_tolerance": 1e-20,
            "louvain_level_modularity_threshold": 1e-7,
            "louvain_max_local_move_sweeps": 100,
            "louvain_diagnostics": {
                "levels": 2,
                "level_diagnostics": [
                    {
                        "sweeps": 2,
                        "accepted_moves": 2,
                        "minimum_accepted_gain": 1e-4,
                        "maximum_accepted_gain": 1e-2,
                        "level_modularity": 0.5,
                        "level_modularity_gain": 0.5,
                    },
                    {
                        "sweeps": 1,
                        "accepted_moves": 0,
                        "minimum_accepted_gain": None,
                        "maximum_accepted_gain": None,
                    },
                ],
                "termination": "no_local_improvement",
            },
            "minimum_community_size": 10,
            "component_filter": None,
            "n_undated": 0,
            "n_entries": len(ids),
            "n_candidates": len(ids),
            "n_communities": len(set(value for value in ordered_labels if value >= 0)),
            "n_noise": sum(value < 0 for value in ordered_labels),
            "n_edges": n_edges,
        }
        if artifact_paths is not None:
            reference = np.asarray(
                [self.production_communities[iid] for iid in ids], dtype=np.int64
            )
            partition["comparison_to_corrected_production_labels"] = (
                sensitivity.partition_metrics(reference, np.asarray(ordered_labels, dtype=np.int64))
            )
        partition_path = (
            self.paths[(method, "partition")]
            if artifact_paths is None
            else artifact_paths["partition"]
        )
        self.write_json(partition_path, partition)
        self._write_temporal(method, assignments, partition, artifact_paths)

    def _write_complete_fixture(self):
        self.write_csv(
            self.paths[("shared", "production_labels")],
            ["icsd_id", "year", "community"],
            [
                {"icsd_id": iid, "year": self.years[iid], "community": self.production_communities[iid]}
                for iid in self.ids
            ],
        )
        for method in sensitivity.METHODS:
            self.write_json(self.paths[(method, "feature_ids")], self.ids)
        for method in sensitivity.METHODS:
            self.write_json(self.paths[(method, "feature_metadata")], self.feature_metadata(method))
            self.write_json(
                self.paths[(method, "pca_subset_scaler")],
                {"mean": [0.0] * 32, "scale": [1.0] * 32, "n_samples": len(self.ids)},
            )

        crystal_neighbors = []
        for iid in self.ids:
            if iid <= 256:
                crystal_neighbors.append(iid + 1 if iid < 256 else 1)
            else:
                crystal_neighbors.append(iid + 1 if iid < 512 else 1)
        voronoi_neighbors = list(crystal_neighbors)
        voronoi_neighbors[1] = 257
        voronoi_neighbors[-1] = 257
        self._write_recovery("crystalnn", crystal_neighbors)
        self._write_recovery("voronoinn", voronoi_neighbors)

        crystal_labels = [0] * 255 + [1] * 255 + [-1, -1]
        voronoi_labels = [0] * 254 + [1, 0] + [1] * 254 + [-1, -1]
        self._write_partition("crystalnn", crystal_labels)
        self._write_partition("voronoinn", voronoi_labels)
        control_labels = {
            "raw": crystal_labels,
            "standardized": voronoi_labels,
        }
        for control, analysis_name in sensitivity.PCA_CONTROLS.items():
            for method in sensitivity.METHODS:
                directory = self.paths[(method, f"pca_{control}_dir")]
                self._write_partition(
                    method,
                    control_labels[control],
                    analysis_name=analysis_name,
                    artifact_paths=sensitivity.pca_control_paths(directory),
                )

    def args(self):
        return SimpleNamespace(
            production_labels=self.paths[("shared", "production_labels")],
            crystal_feature_metadata=self.paths[("crystalnn", "feature_metadata")],
            crystal_feature_ids=self.paths[("crystalnn", "feature_ids")],
            voronoi_feature_metadata=self.paths[("voronoinn", "feature_metadata")],
            voronoi_feature_ids=self.paths[("voronoinn", "feature_ids")],
            crystal_recovery=self.paths[("crystalnn", "recovery")],
            crystal_recovery_ids=self.paths[("crystalnn", "recovery_ids")],
            crystal_neighbors=self.paths[("crystalnn", "neighbors")],
            crystal_exact_audit=self.paths[("crystalnn", "exact_audit")],
            voronoi_recovery=self.paths[("voronoinn", "recovery")],
            voronoi_recovery_ids=self.paths[("voronoinn", "recovery_ids")],
            voronoi_neighbors=self.paths[("voronoinn", "neighbors")],
            voronoi_exact_audit=self.paths[("voronoinn", "exact_audit")],
            crystal_partition=self.paths[("crystalnn", "partition")],
            crystal_partition_ids=self.paths[("crystalnn", "partition_ids")],
            crystal_assignments=self.paths[("crystalnn", "assignments")],
            crystal_temporal_events=self.paths[("crystalnn", "temporal_events")],
            crystal_exclusive_by_decade=self.paths[("crystalnn", "exclusive_by_decade")],
            voronoi_partition=self.paths[("voronoinn", "partition")],
            voronoi_partition_ids=self.paths[("voronoinn", "partition_ids")],
            voronoi_assignments=self.paths[("voronoinn", "assignments")],
            voronoi_temporal_events=self.paths[("voronoinn", "temporal_events")],
            voronoi_exclusive_by_decade=self.paths[("voronoinn", "exclusive_by_decade")],
            crystal_pca_raw_dir=self.paths[("crystalnn", "pca_raw_dir")],
            crystal_pca_standardized_dir=self.paths[("crystalnn", "pca_standardized_dir")],
            voronoi_pca_raw_dir=self.paths[("voronoinn", "pca_raw_dir")],
            voronoi_pca_standardized_dir=self.paths[("voronoinn", "pca_standardized_dir")],
            output_json=self.root / "summary.json",
            output_markdown=self.root / "summary.md",
        )

    def test_valid_summary_reports_recovery_partition_and_temporal_results(self):
        summary = sensitivity.build_summary(self.args())
        self.assertEqual(
            summary["paired_recovery"]["outcome_transitions_crystalnn_to_voronoinn"],
            {"hit_to_hit": 510, "hit_to_miss": 1, "miss_to_hit": 1, "miss_to_miss": 0},
        )
        self.assertEqual(summary["paired_recovery"]["selected_neighbor_identity"]["same_id"], 510)
        self.assertEqual(
            summary["paired_recovery"]["recovery"]["crystalnn"]["ann_exact_nearest_fraction"],
            1.0,
        )
        self.assertEqual(summary["common_cohort_partition"]["cohort"]["n_entries"], 512)
        self.assertIn("ARI", summary["common_cohort_partition"]["agreement"]["all_entries_noise_as_one_label"])
        temporal = summary["common_cohort_temporal"]
        self.assertEqual(temporal["cohort"]["n_entries"], 512)
        self.assertEqual(set(temporal["methods"]["crystalnn"]["by_decade"]), {"1930s", "2010s"})
        self.assertIn("birth_share", temporal["methods"]["voronoinn"]["headline"]["1930s"])
        self.assertEqual(summary["schema_version"], 3)
        controls = summary["matched_pca_controls"]
        self.assertEqual(controls["cohort"]["n_entries"], 512)
        self.assertEqual(controls["controls"]["raw"]["n_communities"], 2)
        self.assertTrue(all(
            controls["duplicate_science_identity_across_graphlet_runs"]["raw"].values()
        ))
        markdown = sensitivity.render_markdown(summary)
        self.assertIn("Hit → miss | 1", markdown)
        self.assertIn("Arithmetic NMI", markdown)
        self.assertIn("Common-cohort temporal replay", markdown)
        self.assertIn("Matched PCA controls", markdown)

    def test_cli_writes_strict_json_and_markdown(self):
        args = self.args()
        argv = []
        for name, value in vars(args).items():
            argv.extend((f"--{name.replace('_', '-')}", str(value)))
        self.assertEqual(sensitivity.main(argv), 0)
        report = sensitivity.read_json(args.output_json)
        self.assertEqual(report["validation"]["status"], "passed")
        self.assertIn("Common-cohort partitions", args.output_markdown.read_text())
        self.assertIn("crystal_temporal_events", report["inputs"])
        self.assertIn("voronoi_exact_audit", report["inputs"])
        self.assertIn("crystalnn_pca_raw_graph_time_summary", report["inputs"])
        self.assertIn("voronoinn_pca_standardized_top_communities", report["inputs"])

    def test_wrong_graphlet_version_fails_before_comparison(self):
        path = self.paths[("voronoinn", "feature_metadata")]
        metadata = sensitivity.read_json(path)
        metadata["graphlet_feature_version"] = "wrong-version"
        self.write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, "graphlet feature version mismatch"):
            sensitivity.build_summary(self.args())

    def test_recovery_outcome_inconsistent_with_communities_fails(self):
        path = self.paths[("crystalnn", "neighbors")]
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["same_community"] = "False"
        self.write_csv(path, list(rows[0]), rows)
        with self.assertRaisesRegex(ValueError, "recovery outcome mismatch"):
            sensitivity.build_summary(self.args())

    def test_legacy_crystalnn_feature_metadata_is_explicitly_supported(self):
        method = "crystalnn"
        metadata_path = self.paths[(method, "feature_metadata")]
        metadata = sensitivity.read_json(metadata_path)
        metadata.pop("neighbor_method")
        metadata.pop("graphlet_feature_version")
        self.write_json(metadata_path, metadata)
        for name in ("recovery", "partition", "exclusive_by_decade"):
            path = self.paths[(method, name)]
            document = sensitivity.read_json(path)
            document["feature_preparation_sha256"] = sensitivity.digest(metadata_path)
            document["source_graphlet_feature_version"] = None
            document["legacy_crystalnn_metadata_compatibility"] = True
            self.write_json(path, document)
        for control in sensitivity.PCA_CONTROLS:
            artifact_paths = sensitivity.pca_control_paths(
                self.paths[(method, f"pca_{control}_dir")]
            )
            for name in ("partition", "exclusive_by_decade", "graph_time_summary"):
                path = artifact_paths[name]
                document = sensitivity.read_json(path)
                document["feature_preparation_sha256"] = sensitivity.digest(metadata_path)
                document["source_graphlet_feature_version"] = None
                document["legacy_crystalnn_metadata_compatibility"] = True
                self.write_json(path, document)
        summary = sensitivity.build_summary(self.args())
        self.assertTrue(
            summary["graphlet_methods"][method]["legacy_crystalnn_metadata_compatibility"]
        )
        self.assertIsNone(summary["graphlet_methods"][method]["source_graphlet_feature_version"])

    def test_partitions_must_be_the_exact_dated_feature_intersection(self):
        labels = [0] * 255 + [1] * 255 + [-1, -1]
        self._write_partition("voronoinn", labels, ids=list(reversed(self.ids)))
        with self.assertRaisesRegex(ValueError, "exact dated graphlet feature intersection"):
            sensitivity.build_summary(self.args())

    def test_analyzer_source_revision_must_match(self):
        path = self.paths[("voronoinn", "recovery")]
        document = sensitivity.read_json(path)
        document["source_sha256"] = "7" * 64
        self.write_json(path, document)
        with self.assertRaisesRegex(ValueError, "differs in source_sha256"):
            sensitivity.build_summary(self.args())

    def test_exact_audit_tie_policy_is_checked(self):
        path = self.paths[("crystalnn", "exact_audit")]
        document = sensitivity.read_json(path)
        document["queries"][0]["exact_selected_neighbor_id"] = 512
        self.write_json(path, document)
        with self.assertRaisesRegex(ValueError, "smallest-ID tie selection"):
            sensitivity.build_summary(self.args())

    def test_partition_declared_minimum_size_is_enforced(self):
        labels = [0] * 9 + [1] * 501 + [-1, -1]
        self._write_partition("voronoinn", labels)
        with self.assertRaisesRegex(ValueError, "below its declared minimum size"):
            sensitivity.build_summary(self.args())

    def test_partition_rejects_sub_tolerance_louvain_move(self):
        path = self.paths[("voronoinn", "partition")]
        document = sensitivity.read_json(path)
        document["louvain_diagnostics"]["level_diagnostics"][0][
            "minimum_accepted_gain"
        ] = 1e-21
        self.write_json(path, document)
        with self.assertRaisesRegex(ValueError, "sub-tolerance move"):
            sensitivity.build_summary(self.args())

    def test_temporal_saved_row_is_recomputed_from_events(self):
        path = self.paths[("voronoinn", "exclusive_by_decade")]
        document = sensitivity.read_json(path)
        document["rows"][0]["n_birth"] += 1
        self.write_json(path, document)
        with self.assertRaisesRegex(ValueError, "n_birth mismatch"):
            sensitivity.build_summary(self.args())

    def test_temporal_event_must_join_partition_identity(self):
        path = self.paths[("crystalnn", "temporal_events")]
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["community"] = "1"
        self.write_csv(path, list(rows[0]), rows)
        with self.assertRaisesRegex(ValueError, "temporal community mismatch"):
            sensitivity.build_summary(self.args())

    def test_pca_comparison_is_recomputed_from_assignments(self):
        path = sensitivity.pca_control_paths(
            self.paths[("voronoinn", "pca_raw_dir")]
        )["partition"]
        document = sensitivity.read_json(path)
        document["comparison_to_corrected_production_labels"][
            "all_entries_noise_as_one_label"
        ]["ARI"] += 0.01
        self.write_json(path, document)
        with self.assertRaisesRegex(ValueError, "production-label comparison.*ARI"):
            sensitivity.build_summary(self.args())

    def test_pca_science_outputs_must_match_across_graphlet_runs(self):
        paths = sensitivity.pca_control_paths(
            self.paths[("voronoinn", "pca_standardized_dir")]
        )
        for name in ("partition", "exclusive_by_decade", "graph_time_summary"):
            document = sensitivity.read_json(paths[name])
            document["sigma"] = 2.0
            self.write_json(paths[name], document)
        with self.assertRaisesRegex(ValueError, "partition science fields differ"):
            sensitivity.build_summary(self.args())


if __name__ == "__main__":
    unittest.main()
