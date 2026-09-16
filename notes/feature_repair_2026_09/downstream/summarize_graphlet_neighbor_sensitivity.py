#!/usr/bin/env python3
"""Summarize paired CrystalNN/VoronoiNN graphlet sensitivity outputs.

The comparison is deliberately strict. Recovery artifacts must describe the
same ordered query cohort, and partition artifacts must have been constructed
on the same ordered cohort. Provenance, feature-preparation hashes, neighbor
rules, saved counts, and per-query/per-entry rows are checked before any metric
is reported. A mismatch raises an error instead of falling back to an
intersection or comparing unlike runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments/graphlet_compare")]
from crystal_neighbors import FEATURE_VERSION  # noqa: E402
import graphlet_features as gf  # noqa: E402


METHODS = ("crystalnn", "voronoinn")
PCA_CONTROLS = {
    "raw": "production_pca_raw_same_protocol",
    "standardized": "production_pca_standardized_same_protocol",
}
SHARED_PROVENANCE_FIELDS = (
    "production_feature_version",
    "production_labels_sha256",
    "production_pca_sha256",
    "production_pca_ids_sha256",
    "source_sha256",
)
ANALYSIS_EXECUTION_FIELDS = (
    "neighbor_search_n_jobs",
    "neighbor_search_random_state",
    "exact_audit_max_workers",
)
COMMON_PROVENANCE_FIELDS = SHARED_PROVENANCE_FIELDS + ANALYSIS_EXECUTION_FIELDS
PARTITION_PROTOCOL_FIELDS = (
    "dated_only",
    "target_year_range",
    "protocol",
    "knn_k",
    "metric",
    "approximate_neighbors",
    "mutual_knn",
    "weight",
    "resolution",
    "louvain_seed",
    "louvain_implementation",
    "louvain_node_move_gain_tolerance",
    "louvain_level_modularity_threshold",
    "louvain_max_local_move_sweeps",
    "minimum_community_size",
    "component_filter",
)
RECOVERY_PROTOCOL_FIELDS = (
    "dated_only",
    "target_year_range",
    "name",
    "metric",
    "approximate",
    "estimator",
    "ann_requested_neighbors_including_self",
    "definition",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON value: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid JSON artifact {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value),
        f"{label} is not a lowercase SHA-256 digest",
    )
    return value


def require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} is not an integer")
    require(value >= minimum, f"{label} is below {minimum}")
    return value


def require_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    if minimum is not None:
        require(result >= minimum, f"{label} is below {minimum}")
    return result


def read_ids(path: Path) -> list[int]:
    values = read_json(path)
    require(isinstance(values, list), f"ID artifact {path} is not a JSON list")
    require(
        all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values),
        f"ID artifact {path} contains a non-positive or non-integer ID",
    )
    require(len(values) == len(set(values)), f"ID artifact {path} contains duplicate IDs")
    return values


def read_production_labels(path: Path) -> dict[int, dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames is not None and {"icsd_id", "year", "community"} <= set(reader.fieldnames),
            "Production-label CSV columns mismatch",
        )
        records = {}
        for number, row in enumerate(reader, 2):
            try:
                iid = int(row["icsd_id"])
                year_text = row["year"].strip()
                year = None
                if year_text:
                    year_number = float(year_text)
                    if not math.isfinite(year_number) or not year_number.is_integer():
                        raise ValueError("year is not an integer")
                    year = int(year_number)
                community = int(row["community"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid production-label row at line {number}: {exc}") from exc
            require(iid > 0, f"Invalid production-label ID at line {number}")
            require(iid not in records, f"Duplicate production-label ID {iid}")
            require(community >= -1, f"Production-label community below -1 for ID {iid}")
            records[iid] = {"year": year, "community": community}
    require(records, "Production-label CSV is empty")
    return records


def parse_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{label} must be true or false")


def method_fields(metadata: dict, method: str) -> dict:
    return {
        "neighbor_method": method,
        "neighbor_settings": gf.neighbor_settings(method),
        "graphlet_feature_version": gf.GRAPHLET_FEATURE_VERSIONS[method],
    }


def validate_feature_metadata(path: Path, method: str) -> tuple[dict, dict]:
    metadata = read_json(path)
    require(isinstance(metadata, dict), f"Feature metadata {path} is not an object")
    expected = method_fields(metadata, method)
    require(metadata.get("complete") is True, f"{method} feature preparation is incomplete")
    require(metadata.get("kind") == "graphlet", f"{method} feature metadata kind is not graphlet")
    require(metadata.get("feature_version") == FEATURE_VERSION, f"{method} production feature version mismatch")
    source_method = metadata.get("neighbor_method")
    source_graphlet_version = metadata.get("graphlet_feature_version")
    legacy_crystalnn = (
        method == "crystalnn"
        and source_method is None
        and source_graphlet_version is None
    )
    require(
        source_method == method or legacy_crystalnn,
        f"{method} neighbor-method metadata mismatch",
    )
    require(metadata.get("neighbor_settings") == expected["neighbor_settings"], f"{method} neighbor settings mismatch")
    require(
        source_graphlet_version == expected["graphlet_feature_version"] or legacy_crystalnn,
        f"{method} graphlet feature version mismatch",
    )
    require(
        metadata.get("representation") == gf.GRAPHLET_REPRESENTATION,
        f"{method} graphlet representation mismatch",
    )
    requested = require_int(metadata.get("n_requested"), f"{method} n_requested", minimum=1)
    successful = require_int(metadata.get("n_successful"), f"{method} n_successful", minimum=1)
    failed = require_int(metadata.get("n_failed"), f"{method} n_failed")
    require(successful + failed == requested, f"{method} feature coverage does not sum")
    shape = metadata.get("feature_shape")
    require(shape == [successful, gf.NUM_FEATURES * gf.NUM_BINS], f"{method} feature shape mismatch")
    require_sha(metadata.get("production_labels_sha256"), f"{method} production-label hash")
    require_sha(metadata.get("ids_sha256"), f"{method} feature-ID hash")
    require_sha(metadata.get("features_sha256"), f"{method} feature-matrix hash")
    coverage = {
        "n_requested": requested,
        "n_successful": successful,
        "n_failed": failed,
        "success_fraction": successful / requested,
        "legacy_crystalnn_metadata_compatibility": legacy_crystalnn,
    }
    return metadata, coverage


def validate_feature_ids(path: Path, metadata: dict, method: str, labels: dict[int, dict]) -> list[int]:
    ids = read_ids(path)
    require(digest(path) == metadata.get("ids_sha256"), f"{method} feature-ID hash mismatch")
    require(len(ids) == metadata.get("n_successful"), f"{method} feature-ID count mismatch")
    require(ids == sorted(ids), f"{method} feature IDs are not in canonical ascending order")
    require(
        all(iid in labels and labels[iid]["community"] >= 0 for iid in ids),
        f"{method} feature IDs extend outside the production non-noise population",
    )
    return ids


def validate_run_provenance(
    document: dict,
    feature_metadata: dict,
    feature_metadata_path: Path,
    method: str,
    label: str,
) -> None:
    require(document.get("production_feature_version") == FEATURE_VERSION, f"{label} production version mismatch")
    require(
        document.get("feature_preparation_sha256") == digest(feature_metadata_path),
        f"{label} feature-preparation hash mismatch",
    )
    require(
        document.get("production_labels_sha256") == feature_metadata.get("production_labels_sha256"),
        f"{label} production-label provenance mismatch",
    )
    expected = method_fields(feature_metadata, method)
    require(document.get("graphlet_neighbor_method") == method, f"{label} neighbor method mismatch")
    require(
        document.get("graphlet_neighbor_settings") == expected["neighbor_settings"],
        f"{label} neighbor settings mismatch",
    )
    require(
        document.get("graphlet_feature_version") == expected["graphlet_feature_version"],
        f"{label} graphlet feature version mismatch",
    )
    legacy_crystalnn = (
        method == "crystalnn"
        and feature_metadata.get("neighbor_method") is None
        and feature_metadata.get("graphlet_feature_version") is None
    )
    require(
        document.get("legacy_crystalnn_metadata_compatibility") is legacy_crystalnn,
        f"{label} legacy-compatibility flag mismatch",
    )
    expected_source_version = None if legacy_crystalnn else expected["graphlet_feature_version"]
    require(
        document.get("source_graphlet_feature_version") == expected_source_version,
        f"{label} source graphlet-version provenance mismatch",
    )
    require(
        document.get("n_feature_successes_before_cohort_restriction") == feature_metadata.get("n_successful"),
        f"{label} feature-success count mismatch",
    )
    require(
        document.get("reference_labels_role")
        == "corrected full-production Louvain labels joined by ID; no relabeling or inference from historical numeric IDs",
        f"{label} reference-label role mismatch",
    )
    for field in SHARED_PROVENANCE_FIELDS[1:]:
        require_sha(document.get(field), f"{label} {field}")
    packages = document.get("packages")
    require(
        isinstance(packages, dict)
        and packages
        and all(isinstance(key, str) and isinstance(value, str) for key, value in packages.items()),
        f"{label} package provenance is invalid",
    )
    require(packages.get("networkx") == "3.6.1", f"{label} NetworkX version mismatch")
    n_jobs = require_int(
        document.get("neighbor_search_n_jobs"), f"{label} neighbor-search n_jobs", minimum=1,
    )
    require(
        document.get("neighbor_search_random_state") == 42,
        f"{label} neighbor-search random state mismatch",
    )
    require(
        document.get("exact_audit_max_workers") == min(n_jobs, 16),
        f"{label} exact-audit worker cap mismatch",
    )


def validate_shared_provenance(left: dict, right: dict, label: str, fields=COMMON_PROVENANCE_FIELDS) -> None:
    for field in fields:
        require(left.get(field) == right.get(field), f"{label} differs in {field}")
    if fields == COMMON_PROVENANCE_FIELDS:
        require(left.get("packages") == right.get("packages"), f"{label} differs in packages")


def validate_common_restriction(document: dict, other_feature_ids: Path, label: str) -> None:
    restrictions = document.get("restrict_ids")
    require(isinstance(restrictions, list) and len(restrictions) == 1, f"{label} lacks one common-cohort restriction")
    restriction = restrictions[0]
    require(isinstance(restriction, dict), f"{label} restriction provenance is invalid")
    require(
        restriction.get("sha256") == digest(other_feature_ids),
        f"{label} restriction does not hash the other method's feature IDs",
    )


def read_nearest_neighbors(path: Path, ids: list[int], label: str) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "query_id", "query_community", "neighbor_id", "neighbor_community",
            "distance", "same_community", "distance_l1_recomputed",
            "first_ten_chemical_channels_l1", "pure_geometry_four_channels_l1",
            "pair_triplet_chemical_50_channels_l1",
        }
        require(reader.fieldnames is not None and required <= set(reader.fieldnames), f"{label} neighbor CSV columns mismatch")
        records = []
        for number, row in enumerate(reader, 2):
            try:
                record = {
                    "query_id": int(row["query_id"]),
                    "query_community": int(row["query_community"]),
                    "neighbor_id": int(row["neighbor_id"]),
                    "neighbor_community": int(row["neighbor_community"]),
                    "distance": require_float(float(row["distance"]), f"{label} line {number} distance", minimum=0),
                    "same_community": parse_bool(row["same_community"], f"{label} line {number} same_community"),
                }
                record["distance_l1_recomputed"] = require_float(
                    float(row["distance_l1_recomputed"]), f"{label} line {number} recomputed distance", minimum=0,
                )
                channel_fields = (
                    "first_ten_chemical_channels_l1",
                    "pure_geometry_four_channels_l1",
                    "pair_triplet_chemical_50_channels_l1",
                )
                record["channel_parts"] = [
                    require_float(float(row[field]), f"{label} line {number} {field}", minimum=0)
                    for field in channel_fields
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {label} neighbor row at line {number}: {exc}") from exc
            records.append(record)
    require([record["query_id"] for record in records] == ids, f"{label} neighbor query order differs from IDs")
    id_set = set(ids)
    require(
        all(record["neighbor_id"] in id_set and record["neighbor_id"] != record["query_id"] for record in records),
        f"{label} neighbor CSV contains a missing or self neighbor",
    )
    query_communities = {record["query_id"]: record["query_community"] for record in records}
    require(
        all(community >= 0 for community in query_communities.values()),
        f"{label} contains a noise query community",
    )
    for record in records:
        require(
            record["neighbor_community"] == query_communities[record["neighbor_id"]],
            f"{label} neighbor-community join mismatch for query {record['query_id']}",
        )
        require(
            record["same_community"]
            == (record["query_community"] == record["neighbor_community"]),
            f"{label} recovery outcome mismatch for query {record['query_id']}",
        )
        require(
            math.isclose(record["distance"], record["distance_l1_recomputed"], rel_tol=2e-6, abs_tol=2e-5),
            f"{label} stored/recomputed distance mismatch for query {record['query_id']}",
        )
        require(
            math.isclose(sum(record["channel_parts"]), record["distance_l1_recomputed"], rel_tol=1e-12, abs_tol=1e-10),
            f"{label} channel decomposition mismatch for query {record['query_id']}",
        )
    return records


def validate_exact_query_audit(
    path: Path,
    ids: list[int],
    records: list[dict],
    label: str,
    expected_workers: int,
) -> dict:
    audit = read_json(path)
    require(isinstance(audit, dict), f"{label} exact-query audit is not an object")
    queries = audit.get("queries")
    require(isinstance(queries, list), f"{label} exact-query audit queries are not a list")
    require(audit.get("n_queries") == 512 == len(queries), f"{label} exact-query audit does not contain 512 queries")
    require(audit.get("candidate_pool_size") == len(ids), f"{label} exact-query candidate-pool mismatch")
    require(audit.get("query_seed") == 42, f"{label} exact-query seed mismatch")
    require(
        audit.get("parallel_workers") == expected_workers,
        f"{label} exact-query worker count mismatch",
    )
    require(
        audit.get("distance") == "exact float64 cityblock on stored float32 CDF values",
        f"{label} exact-query distance definition mismatch",
    )
    require(
        audit.get("tie_policy") == "all exact-equal ties retained; smallest ICSD ID selected",
        f"{label} exact-query tie policy mismatch",
    )
    id_set = set(ids)
    recovery_by_id = {record["query_id"]: record for record in records}
    seen_queries = set()
    exact_flags = []
    exact_label_flags = []
    for number, query in enumerate(queries, 1):
        require(isinstance(query, dict), f"{label} exact-query row {number} is not an object")
        query_id = require_int(query.get("query_id"), f"{label} exact-query row {number} query ID", minimum=1)
        require(query_id in id_set, f"{label} exact-query row {number} query is outside the cohort")
        require(query_id not in seen_queries, f"{label} exact-query audit repeats query {query_id}")
        seen_queries.add(query_id)
        tied = query.get("exact_nearest_tied_ids")
        require(
            isinstance(tied, list)
            and tied
            and all(isinstance(iid, int) and not isinstance(iid, bool) for iid in tied),
            f"{label} exact-query row {number} has invalid tied IDs",
        )
        require(tied == sorted(set(tied)), f"{label} exact-query row {number} tied IDs are not unique and sorted")
        require(query_id not in tied and set(tied) <= id_set, f"{label} exact-query row {number} has invalid self/out-of-cohort tie")
        require(
            query.get("exact_selected_neighbor_id") == min(tied),
            f"{label} exact-query row {number} violates smallest-ID tie selection",
        )
        recovery = recovery_by_id[query_id]
        require(
            query.get("query_community") == recovery["query_community"],
            f"{label} exact-query row {number} query-community mismatch",
        )
        require(
            query.get("exact_selected_neighbor_community")
            == recovery_by_id[min(tied)]["query_community"],
            f"{label} exact-query row {number} exact-neighbor-community mismatch",
        )
        tied_same = [
            recovery_by_id[iid]["query_community"] == recovery["query_community"] for iid in tied
        ]
        require(
            query.get("any_exact_tie_same_community") is any(tied_same)
            and query.get("all_exact_ties_same_community") is all(tied_same),
            f"{label} exact-query row {number} tie-label flags mismatch",
        )
        require(
            query.get("ann_selected_neighbor_id") == recovery["neighbor_id"],
            f"{label} exact-query row {number} ANN neighbor mismatch",
        )
        exact_minimum = require_float(
            query.get("exact_minimum_l1"), f"{label} exact-query row {number} minimum", minimum=0,
        )
        ann_distance = require_float(
            query.get("ann_selected_distance_l1"), f"{label} exact-query row {number} ANN distance", minimum=0,
        )
        excess = require_float(
            query.get("ann_distance_excess_l1"), f"{label} exact-query row {number} excess", minimum=0,
        )
        require(
            math.isclose(excess, ann_distance - exact_minimum, rel_tol=2e-12, abs_tol=2e-12),
            f"{label} exact-query row {number} distance excess mismatch",
        )
        exact = query.get("ann_selected_is_exact_nearest")
        require(isinstance(exact, bool), f"{label} exact-query row {number} exact flag is not Boolean")
        require(exact == (recovery["neighbor_id"] in tied), f"{label} exact-query row {number} exact flag mismatch")
        exact_flags.append(exact)
        exact_label_flags.append(
            recovery["query_community"] == query["exact_selected_neighbor_community"]
        )
    fraction = require_float(audit.get("ann_exact_nearest_fraction"), f"{label} exact fraction", minimum=0)
    require(fraction <= 1, f"{label} exact fraction exceeds one")
    require(
        math.isclose(fraction, sum(exact_flags) / len(exact_flags), rel_tol=0, abs_tol=1e-15),
        f"{label} exact fraction does not match query rows",
    )
    agreement = require_float(
        audit.get("exact_selected_label_agreement"), f"{label} exact label agreement", minimum=0,
    )
    require(agreement <= 1, f"{label} exact label agreement exceeds one")
    require(
        math.isclose(agreement, sum(exact_label_flags) / len(exact_label_flags), rel_tol=0, abs_tol=1e-15),
        f"{label} exact label agreement does not match query rows",
    )
    return {
        "summary": {key: value for key, value in audit.items() if key != "queries"},
        "ann_exact_nearest_fraction": fraction,
        "exact_selected_label_agreement": agreement,
    }


def validate_recovery(
    recovery_path: Path,
    ids_path: Path,
    neighbors_path: Path,
    exact_audit_path: Path,
    feature_metadata: dict,
    feature_metadata_path: Path,
    method: str,
) -> dict:
    label = f"{method} recovery"
    recovery = read_json(recovery_path)
    require(isinstance(recovery, dict), f"{label} JSON is not an object")
    ids = read_ids(ids_path)
    require(recovery.get("cohort_sha256") == digest(ids_path), f"{label} cohort hash mismatch")
    validate_run_provenance(recovery, feature_metadata, feature_metadata_path, method, label)
    require(recovery.get("name") == "graphlet_l1", f"{label} is not graphlet_l1")
    require(
        recovery.get("dated_only") is False and recovery.get("target_year_range") is None,
        f"{label} is not the full common feature cohort",
    )
    require(recovery.get("metric") == "manhattan", f"{label} metric is not manhattan")
    require(recovery.get("approximate") is True, f"{label} is not the ANN graphlet protocol")
    require(
        recovery.get("estimator") == "pynndescent.NNDescent"
        and recovery.get("ann_requested_neighbors_including_self") == 15,
        f"{label} ANN protocol mismatch",
    )
    count = len(ids)
    require(count >= 2, f"{label} has fewer than two entries")
    require(count <= feature_metadata["n_successful"], f"{label} exceeds feature-success coverage")
    for field in ("n_entries", "n_candidates"):
        require(recovery.get(field) == count, f"{label} {field} differs from ID count")
    records = read_nearest_neighbors(neighbors_path, ids, label)
    exact_audit = validate_exact_query_audit(
        exact_audit_path,
        ids,
        records,
        label,
        recovery["exact_audit_max_workers"],
    )
    hits = sum(record["same_community"] for record in records)
    misses = count - hits
    require(recovery.get("hits") == hits and recovery.get("misses") == misses, f"{label} hit/miss counts mismatch")
    require(
        math.isclose(
            require_float(recovery.get("same_community_1nn_agreement"), f"{label} recovery rate"),
            hits / count,
            rel_tol=0,
            abs_tol=1e-15,
        ),
        f"{label} recovery rate mismatch",
    )
    require(
        recovery.get("n_communities") == len({record["query_community"] for record in records}),
        f"{label} community count mismatch",
    )
    require(
        recovery.get("exact_query_audit") == exact_audit["summary"],
        f"{label} embedded exact-query audit summary mismatch",
    )
    return {
        "document": recovery,
        "ids": ids,
        "records": records,
        "hits": hits,
        "misses": misses,
        "rate": hits / count,
        "ann_exact_nearest_fraction": exact_audit["ann_exact_nearest_fraction"],
        "paths": {
            "recovery_json": recovery_path,
            "ids_json": ids_path,
            "neighbors_csv": neighbors_path,
            "exact_query_audit_json": exact_audit_path,
        },
    }


def paired_recovery(crystal: dict, voronoi: dict) -> dict:
    require(crystal["ids"] == voronoi["ids"], "Recovery outputs do not use the same ordered common cohort")
    require(
        crystal["document"]["cohort_sha256"] == voronoi["document"]["cohort_sha256"],
        "Recovery ordered-cohort hashes differ",
    )
    validate_shared_provenance(crystal["document"], voronoi["document"], "Recovery provenance")
    validate_shared_provenance(
        crystal["document"], voronoi["document"], "Recovery protocol", RECOVERY_PROTOCOL_FIELDS,
    )
    transitions = {
        "hit_to_hit": 0,
        "hit_to_miss": 0,
        "miss_to_hit": 0,
        "miss_to_miss": 0,
    }
    same_neighbor = same_neighbor_community = both_miss_same_destination = 0
    for left, right in zip(crystal["records"], voronoi["records"]):
        require(left["query_id"] == right["query_id"], "Recovery query join changed order")
        require(left["query_community"] == right["query_community"], f"Recovery query-community mismatch for {left['query_id']}")
        source = "hit" if left["same_community"] else "miss"
        destination = "hit" if right["same_community"] else "miss"
        transitions[f"{source}_to_{destination}"] += 1
        same_neighbor += left["neighbor_id"] == right["neighbor_id"]
        same_neighbor_community += left["neighbor_community"] == right["neighbor_community"]
        if source == destination == "miss":
            both_miss_same_destination += left["neighbor_community"] == right["neighbor_community"]
    count = len(crystal["ids"])
    require(sum(transitions.values()) == count, "Recovery transition counts do not sum")
    require(transitions["hit_to_hit"] + transitions["hit_to_miss"] == crystal["hits"], "CrystalNN transition hits mismatch")
    require(transitions["hit_to_hit"] + transitions["miss_to_hit"] == voronoi["hits"], "VoronoiNN transition hits mismatch")
    return {
        "cohort": {
            "n_entries": count,
            "ordered_ids_sha256": crystal["document"]["cohort_sha256"],
            "definition": "exact ordered cohort shared by both recovery searches",
        },
        "recovery": {
            "crystalnn": {
                "hits": crystal["hits"], "misses": crystal["misses"], "rate": crystal["rate"],
                "ann_exact_nearest_fraction": crystal["ann_exact_nearest_fraction"],
            },
            "voronoinn": {
                "hits": voronoi["hits"], "misses": voronoi["misses"], "rate": voronoi["rate"],
                "ann_exact_nearest_fraction": voronoi["ann_exact_nearest_fraction"],
            },
            "voronoinn_minus_crystalnn": voronoi["rate"] - crystal["rate"],
            "voronoinn_minus_crystalnn_percentage_points": 100 * (voronoi["rate"] - crystal["rate"]),
        },
        "outcome_transitions_crystalnn_to_voronoinn": transitions,
        "selected_neighbor_identity": {
            "same_id": same_neighbor,
            "different_id": count - same_neighbor,
            "same_id_fraction": same_neighbor / count,
        },
        "selected_neighbor_reference_community": {
            "same_community_id": same_neighbor_community,
            "different_community_id": count - same_neighbor_community,
            "same_community_id_fraction": same_neighbor_community / count,
            "both_miss_same_destination_community": both_miss_same_destination,
        },
    }


def read_partition_assignments(path: Path, ids: list[int], label: str) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames is not None and {"icsd_id", "year", "community"} <= set(reader.fieldnames),
            f"{label} assignment CSV columns mismatch",
        )
        records = []
        for number, row in enumerate(reader, 2):
            try:
                year_text = row["year"].strip()
                year = None
                if year_text:
                    year_number = float(year_text)
                    if not math.isfinite(year_number) or not year_number.is_integer():
                        raise ValueError("year is not an integer")
                    year = int(year_number)
                records.append({
                    "icsd_id": int(row["icsd_id"]),
                    "year": year,
                    "community": int(row["community"]),
                })
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {label} assignment at line {number}: {exc}") from exc
    require([record["icsd_id"] for record in records] == ids, f"{label} assignment order differs from IDs")
    return records


def validate_louvain_diagnostics(partition: dict, label: str) -> None:
    diagnostics = partition.get("louvain_diagnostics")
    require(isinstance(diagnostics, dict), f"{label} Louvain diagnostics are absent")
    level_records = diagnostics.get("level_diagnostics")
    require(isinstance(level_records, list), f"{label} Louvain level diagnostics are not a list")
    level_count = require_int(diagnostics.get("levels"), f"{label} Louvain level count")
    require(level_count == len(level_records), f"{label} Louvain level count mismatch")
    termination = diagnostics.get("termination")
    if partition.get("n_edges") == 0:
        require(
            diagnostics == {
                "levels": 0,
                "level_diagnostics": [],
                "termination": "empty_graph",
            },
            f"{label} empty-graph Louvain diagnostics mismatch",
        )
        return
    require(level_count >= 1, f"{label} nonempty graph has no Louvain levels")
    require(
        termination in {"no_local_improvement", "level_modularity_threshold"},
        f"{label} Louvain termination is invalid",
    )
    tolerance = require_float(
        partition.get("louvain_node_move_gain_tolerance"),
        f"{label} Louvain move tolerance", minimum=0,
    )
    level_threshold = require_float(
        partition.get("louvain_level_modularity_threshold"),
        f"{label} Louvain level threshold", minimum=0,
    )
    sweep_cap = require_int(
        partition.get("louvain_max_local_move_sweeps"),
        f"{label} Louvain sweep cap", minimum=1,
    )
    for index, record in enumerate(level_records):
        prefix = f"{label} Louvain level {index + 1}"
        require(isinstance(record, dict), f"{prefix} diagnostics are not an object")
        sweeps = require_int(record.get("sweeps"), f"{prefix} sweep count", minimum=1)
        require(sweeps <= sweep_cap, f"{prefix} exceeds the declared sweep cap")
        moves = require_int(record.get("accepted_moves"), f"{prefix} accepted moves")
        minimum_gain = record.get("minimum_accepted_gain")
        maximum_gain = record.get("maximum_accepted_gain")
        if moves == 0:
            require(
                minimum_gain is None and maximum_gain is None,
                f"{prefix} reports gain extrema without accepted moves",
            )
        else:
            minimum_gain = require_float(minimum_gain, f"{prefix} minimum accepted gain")
            maximum_gain = require_float(maximum_gain, f"{prefix} maximum accepted gain")
            require(minimum_gain > tolerance, f"{prefix} accepted a sub-tolerance move")
            require(maximum_gain >= minimum_gain, f"{prefix} gain extrema are reversed")

        has_modularity = "level_modularity" in record or "level_modularity_gain" in record
        terminal_no_move = termination == "no_local_improvement" and index == level_count - 1
        if terminal_no_move:
            require(moves == 0, f"{prefix} no-improvement termination still has moves")
            require(not has_modularity, f"{prefix} terminal no-move level has modularity fields")
        else:
            require(
                set(("level_modularity", "level_modularity_gain")) <= set(record),
                f"{prefix} lacks level modularity diagnostics",
            )
            require_float(record["level_modularity"], f"{prefix} modularity")
            gain = require_float(record["level_modularity_gain"], f"{prefix} modularity gain")
            if termination == "level_modularity_threshold" and index == level_count - 1:
                require(gain <= level_threshold, f"{prefix} threshold termination exceeds threshold")
            else:
                require(gain > level_threshold, f"{prefix} advanced without exceeding threshold")


def validate_partition(
    partition_path: Path,
    ids_path: Path,
    assignments_path: Path,
    feature_metadata: dict,
    feature_metadata_path: Path,
    method: str,
    expected_name: str = "graphlet",
) -> dict:
    label = (
        f"{method} partition"
        if expected_name == "graphlet"
        else f"{method} {expected_name} partition"
    )
    partition = read_json(partition_path)
    require(isinstance(partition, dict), f"{label} JSON is not an object")
    ids = read_ids(ids_path)
    count = len(ids)
    require(count >= 2, f"{label} has fewer than two entries")
    require(partition.get("cohort_sha256") == digest(ids_path), f"{label} cohort hash mismatch")
    validate_run_provenance(partition, feature_metadata, feature_metadata_path, method, label)
    require(partition.get("name") == expected_name, f"{label} name mismatch")
    require(partition.get("protocol") == "historical_graphlet_temporal", f"{label} protocol mismatch")
    expected_protocol = {
        "dated_only": True,
        "target_year_range": [1900, 2025],
        "knn_k": min(16, count - 1),
        "metric": "euclidean",
        "approximate_neighbors": True,
        "mutual_knn": True,
        "weight": "exp(-distance^2/sigma^2)",
        "resolution": 1.0,
        "louvain_seed": 42,
        "louvain_implementation": "networkx-3.6.1-compatible-undirected-with-node-move-tolerance",
        "louvain_node_move_gain_tolerance": 1e-20,
        "louvain_level_modularity_threshold": 1e-7,
        "louvain_max_local_move_sweeps": 100,
        "minimum_community_size": 10,
        "component_filter": None,
    }
    for field, expected in expected_protocol.items():
        require(partition.get(field) == expected, f"{label} {field} protocol mismatch")
    require_float(partition.get("sigma"), f"{label} kernel sigma", minimum=0)
    require(partition.get("sigma") > 0, f"{label} kernel sigma is not positive")
    require(
        partition.get("sigma_population") == "positive retained neighbor distances",
        f"{label} sigma population mismatch",
    )
    validate_louvain_diagnostics(partition, label)
    records = read_partition_assignments(assignments_path, ids, label)
    labels = np.asarray([record["community"] for record in records], dtype=np.int64)
    require(np.all(labels >= -1), f"{label} contains a community label below -1")
    nonnoise = sorted(set(labels[labels >= 0].tolist()))
    noise = int(np.sum(labels < 0))
    require(nonnoise == list(range(len(nonnoise))), f"{label} community labels are not contiguous")
    if nonnoise:
        _, community_sizes = np.unique(labels[labels >= 0], return_counts=True)
        require(
            np.all(community_sizes >= partition["minimum_community_size"]),
            f"{label} contains a community below its declared minimum size",
        )
    require(count <= feature_metadata["n_successful"], f"{label} exceeds feature-success coverage")
    for field in ("n_entries", "n_candidates"):
        require(partition.get(field) == count, f"{label} {field} differs from ID count")
    require(partition.get("n_communities") == len(nonnoise), f"{label} community count mismatch")
    require(partition.get("n_noise") == noise, f"{label} noise count mismatch")
    if partition.get("dated_only"):
        year_range = partition.get("target_year_range")
        require(year_range == [1900, 2025], f"{label} dated range mismatch")
        require(
            all(record["year"] is not None and 1900 <= record["year"] <= 2025 for record in records),
            f"{label} contains an out-of-range or missing year",
        )
    return {
        "document": partition,
        "ids": ids,
        "records": records,
        "labels": labels,
        "n_communities": len(nonnoise),
        "n_noise": noise,
        "paths": {
            "partition_json": partition_path,
            "ids_json": ids_path,
            "assignments_csv": assignments_path,
        },
    }


def _csv_integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    require(math.isfinite(number) and number.is_integer(), f"{label} is not an integer")
    result = int(number)
    if minimum is not None:
        require(result >= minimum, f"{label} is below {minimum}")
    return result


def read_temporal_events(path: Path, partition: dict, label: str) -> list[dict]:
    required = {
        "icsd_id", "year", "decade", "community", "n_active_neighbors",
        "n_active_same_community_neighbors", "n_active_other_communities",
        "is_bridge_attachment", "event_type",
    }
    core_fields = {"distance_to_centroid", "core_threshold", "core_periphery"}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames is not None and required <= set(reader.fieldnames),
            f"{label} temporal-event CSV columns mismatch",
        )
        present_core_fields = core_fields & set(reader.fieldnames)
        require(
            not present_core_fields or present_core_fields == core_fields,
            f"{label} temporal-event CSV has an incomplete core-field group",
        )
        has_core_fields = present_core_fields == core_fields
        if not has_core_fields:
            require(
                partition["n_noise"] == len(partition["ids"]),
                f"{label} temporal-event CSV omits core fields for a non-noise partition",
            )
        events = []
        for number, row in enumerate(reader, 2):
            prefix = f"{label} temporal-event line {number}"
            try:
                event = {
                    "icsd_id": _csv_integer(row["icsd_id"], f"{prefix} ICSD ID", minimum=1),
                    "year": _csv_integer(row["year"], f"{prefix} year"),
                    "decade": row["decade"].strip(),
                    "community": _csv_integer(row["community"], f"{prefix} community", minimum=-1),
                    "n_active_neighbors": _csv_integer(
                        row["n_active_neighbors"], f"{prefix} active-neighbor count", minimum=0,
                    ),
                    "n_active_same_community_neighbors": _csv_integer(
                        row["n_active_same_community_neighbors"],
                        f"{prefix} active same-community count", minimum=0,
                    ),
                    "n_active_other_communities": _csv_integer(
                        row["n_active_other_communities"],
                        f"{prefix} active other-community count", minimum=0,
                    ),
                    "is_bridge_attachment": parse_bool(
                        row["is_bridge_attachment"], f"{prefix} bridge flag",
                    ),
                    "event_type": row["event_type"].strip(),
                }
                if not has_core_fields:
                    event["distance_to_centroid"] = None
                    event["core_threshold"] = None
                    event["core_periphery"] = None
                elif event["community"] < 0:
                    require(
                        not row["distance_to_centroid"].strip()
                        and not row["core_threshold"].strip()
                        and not row["core_periphery"].strip(),
                        f"{prefix} noise row has core/periphery values",
                    )
                    event["distance_to_centroid"] = None
                    event["core_threshold"] = None
                    event["core_periphery"] = None
                else:
                    event["distance_to_centroid"] = require_float(
                        float(row["distance_to_centroid"]), f"{prefix} centroid distance", minimum=0,
                    )
                    event["core_threshold"] = require_float(
                        float(row["core_threshold"]), f"{prefix} core threshold", minimum=0,
                    )
                    event["core_periphery"] = row["core_periphery"].strip()
                    require(
                        event["core_periphery"] in {"core", "periphery"},
                        f"{prefix} has an invalid core/periphery class",
                    )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {prefix}: {exc}") from exc
            require(
                event["n_active_same_community_neighbors"] <= event["n_active_neighbors"],
                f"{prefix} same-community neighbors exceed all active neighbors",
            )
            require(
                event["n_active_other_communities"] <= event["n_active_neighbors"],
                f"{prefix} other communities exceed all active neighbors",
            )
            require(
                event["n_active_same_community_neighbors"]
                + event["n_active_other_communities"]
                <= event["n_active_neighbors"],
                f"{prefix} active-neighbor decomposition exceeds all active neighbors",
            )
            require(
                event["n_active_neighbors"] <= partition["document"]["knn_k"],
                f"{prefix} active-neighbor count exceeds the graph k",
            )
            if event["community"] < 0:
                require(
                    event["n_active_same_community_neighbors"] == 0,
                    f"{prefix} noise row has same-community neighbors",
                )
            require(
                event["is_bridge_attachment"] == (event["n_active_other_communities"] >= 2),
                f"{prefix} bridge flag disagrees with active other communities",
            )
            events.append(event)

    assignment_by_id = {record["icsd_id"]: record for record in partition["records"]}
    expected_order = [
        record["icsd_id"]
        for record in sorted(
            partition["records"], key=lambda record: (record["year"], record["icsd_id"]),
        )
    ]
    require(
        [event["icsd_id"] for event in events] == expected_order,
        f"{label} temporal events are not the exact chronological partition cohort",
    )
    birth_year = {}
    for record in partition["records"]:
        if record["community"] >= 0:
            birth_year[record["community"]] = min(
                record["year"], birth_year.get(record["community"], record["year"]),
            )
    for event in events:
        assignment = assignment_by_id[event["icsd_id"]]
        require(event["year"] == assignment["year"], f"{label} temporal year mismatch for {event['icsd_id']}")
        require(
            event["community"] == assignment["community"],
            f"{label} temporal community mismatch for {event['icsd_id']}",
        )
        require(
            event["decade"] == f"{(event['year'] // 10) * 10}s",
            f"{label} temporal decade mismatch for {event['icsd_id']}",
        )
        expected_type = (
            "outlier"
            if event["community"] < 0
            else "community_birth"
            if event["year"] == birth_year[event["community"]]
            else "existing_community"
        )
        require(
            event["event_type"] == expected_type,
            f"{label} temporal event type mismatch for {event['icsd_id']}",
        )
        if event["community"] >= 0:
            expected_core = (
                "core"
                if event["distance_to_centroid"] <= event["core_threshold"]
                else "periphery"
            )
            require(
                event["core_periphery"] == expected_core,
                f"{label} core/periphery mismatch for {event['icsd_id']}",
            )
        if expected_type == "outlier":
            event["exclusive_category"] = "outlier"
        elif expected_type == "community_birth":
            event["exclusive_category"] = "birth"
        elif event["n_active_other_communities"] >= 2:
            event["exclusive_category"] = "bridge"
        elif event["n_active_other_communities"] == 1:
            event["exclusive_category"] = "cross"
        else:
            event["exclusive_category"] = "same"
    require(
        sum(event["n_active_neighbors"] for event in events) == partition["document"]["n_edges"],
        f"{label} active-neighbor counts do not sum to the partition edge count",
    )
    require(
        sum(event["community"] < 0 for event in events) == partition["n_noise"],
        f"{label} event noise count differs from the partition",
    )
    members_by_community = {community: [] for community in birth_year}
    for event in events:
        if event["community"] >= 0:
            members_by_community[event["community"]].append(event)
    for community in sorted(birth_year):
        members = members_by_community[community]
        thresholds = [event["core_threshold"] for event in members]
        require(
            all(math.isclose(value, thresholds[0], rel_tol=1e-12, abs_tol=1e-12) for value in thresholds),
            f"{label} core threshold varies within community {community}",
        )
        require(
            math.isclose(
                thresholds[0], float(np.median([event["distance_to_centroid"] for event in members])),
                rel_tol=1e-7, abs_tol=1e-6,
            ),
            f"{label} core threshold is not the median distance for community {community}",
        )
    return events


def recompute_exclusive_rows(events: list[dict]) -> list[dict]:
    categories = ("outlier", "birth", "same", "cross", "bridge")
    decades = sorted({event["decade"] for event in events})
    rows = []
    for decade in decades:
        group = [event for event in events if event["decade"] == decade]
        counts = {
            category: sum(event["exclusive_category"] == category for event in group)
            for category in categories
        }
        total = len(group)
        require(total > 0 and sum(counts.values()) == total, f"Temporal categories do not partition {decade}")
        row = {"decade": decade, "n_total": total}
        for category in categories:
            row[f"n_{category}"] = counts[category]
            row[f"share_{category}"] = counts[category] / total
        row["share_any_attachment"] = sum(
            counts[category] for category in ("same", "cross", "bridge")
        ) / total
        row["n_same_isolated"] = sum(
            event["exclusive_category"] == "same"
            and event["n_active_same_community_neighbors"] == 0
            for event in group
        )
        rows.append(row)
    return rows


def validate_temporal_replay(
    events_path: Path,
    exclusive_path: Path,
    partition: dict,
    method: str,
) -> dict:
    label = f"{method} temporal replay"
    document = read_json(exclusive_path)
    require(isinstance(document, dict), f"{label} exclusive-by-decade JSON is not an object")
    for field, expected in partition["document"].items():
        require(
            field in document and document[field] == expected,
            f"{label} differs from its partition in {field}",
        )
    require(
        document.get("denominator")
        == "all entries in the given decade, including representation-partition noise",
        f"{label} denominator mismatch",
    )
    events = read_temporal_events(events_path, partition, label)
    expected_rows = recompute_exclusive_rows(events)
    saved_rows = document.get("rows")
    require(isinstance(saved_rows, list), f"{label} rows are not a list")
    require(len(saved_rows) == len(expected_rows), f"{label} decade-row count mismatch")
    integer_fields = (
        "n_total", "n_outlier", "n_birth", "n_same", "n_cross", "n_bridge",
        "n_same_isolated",
    )
    share_fields = (
        "share_outlier", "share_birth", "share_same", "share_cross", "share_bridge",
        "share_any_attachment",
    )
    exact_fields = {"decade", *integer_fields, *share_fields}
    for saved, expected in zip(saved_rows, expected_rows):
        require(isinstance(saved, dict) and set(saved) == exact_fields, f"{label} row schema mismatch")
        require(saved.get("decade") == expected["decade"], f"{label} decade order mismatch")
        for field in integer_fields:
            require(
                require_int(saved.get(field), f"{label} {expected['decade']} {field}") == expected[field],
                f"{label} {expected['decade']} {field} mismatch",
            )
        for field in share_fields:
            actual = require_float(saved.get(field), f"{label} {expected['decade']} {field}", minimum=0)
            require(actual <= 1, f"{label} {expected['decade']} {field} exceeds one")
            require(
                math.isclose(actual, expected[field], rel_tol=0, abs_tol=1e-15),
                f"{label} {expected['decade']} {field} mismatch",
            )
    by_decade = {row["decade"]: row for row in saved_rows}
    for decade in ("1930s", "2010s"):
        require(decade in by_decade, f"{label} lacks required headline decade {decade}")
    headline = {}
    for decade in ("1930s", "2010s"):
        row = by_decade[decade]
        headline[decade] = {
            "n_entries": row["n_total"],
            "n_birth": row["n_birth"],
            "birth_share": row["share_birth"],
            "n_any_attachment": row["n_same"] + row["n_cross"] + row["n_bridge"],
            "any_attachment_share": row["share_any_attachment"],
        }
    return {
        "document": document,
        "events": events,
        "by_decade": by_decade,
        "headline": headline,
        "paths": {"node_temporal_events_csv": events_path, "exclusive_by_decade_json": exclusive_path},
    }


def paired_temporal(crystal: dict, voronoi: dict, partition: dict) -> dict:
    require(
        [(event["icsd_id"], event["year"], event["decade"]) for event in crystal["events"]]
        == [(event["icsd_id"], event["year"], event["decade"]) for event in voronoi["events"]],
        "Temporal replays do not contain the same chronological common cohort",
    )
    require(
        crystal["document"]["cohort_sha256"] == voronoi["document"]["cohort_sha256"],
        "Temporal ordered-cohort hashes differ",
    )
    validate_shared_provenance(crystal["document"], voronoi["document"], "Temporal provenance")
    validate_shared_provenance(
        crystal["document"], voronoi["document"], "Temporal partition protocol",
        PARTITION_PROTOCOL_FIELDS,
    )
    differences = {}
    for decade in ("1930s", "2010s"):
        left = crystal["headline"][decade]
        right = voronoi["headline"][decade]
        birth = right["birth_share"] - left["birth_share"]
        attachment = right["any_attachment_share"] - left["any_attachment_share"]
        differences[decade] = {
            "birth_share": birth,
            "birth_share_percentage_points": 100 * birth,
            "any_attachment_share": attachment,
            "any_attachment_share_percentage_points": 100 * attachment,
        }
    return {
        "cohort": {
            "n_entries": partition["cohort"]["n_entries"],
            "ordered_ids_sha256": partition["cohort"]["ordered_ids_sha256"],
            "definition": "exact dated feature intersection used for both partitions and temporal replays",
        },
        "methods": {
            method: {"by_decade": run["by_decade"], "headline": run["headline"]}
            for method, run in (("crystalnn", crystal), ("voronoinn", voronoi))
        },
        "headline_differences_voronoinn_minus_crystalnn": differences,
    }


def partition_metrics(reference: np.ndarray, alternate: np.ndarray) -> dict:
    """Recompute the saved production-partition comparison from assignments."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix

    def one(left: np.ndarray, right: np.ndarray) -> dict:
        table = contingency_matrix(left, right, sparse=True)

        def pairs(values) -> int:
            values = np.asarray(values, dtype=np.int64)
            return int(np.sum(values * (values - 1) // 2))

        intersection_pairs = pairs(table.data)
        reference_pairs = pairs(np.asarray(table.sum(axis=1)).ravel())
        alternate_pairs = pairs(np.asarray(table.sum(axis=0)).ravel())
        return {
            "n_entries": int(len(left)),
            "ARI": float(adjusted_rand_score(left, right)),
            "NMI_arithmetic": float(
                normalized_mutual_info_score(left, right, average_method="arithmetic")
            ),
            "same_community_pair_recall": (
                intersection_pairs / reference_pairs if reference_pairs else None
            ),
            "same_community_pair_precision": (
                intersection_pairs / alternate_pairs if alternate_pairs else None
            ),
            "same_community_pair_f1": (
                2 * intersection_pairs / (reference_pairs + alternate_pairs)
                if reference_pairs + alternate_pairs
                else None
            ),
            "reference_same_pairs": reference_pairs,
            "alternate_same_pairs": alternate_pairs,
            "intersection_same_pairs": intersection_pairs,
        }

    keep = alternate >= 0
    return {
        "all_entries_noise_as_one_label": one(reference, alternate),
        "alternate_nonnoise_only": one(reference[keep], alternate[keep]) if keep.any() else None,
        "n_alternate_noise": int(np.sum(~keep)),
    }


def require_nested_match(actual: Any, expected: Any, label: str) -> None:
    """Require the same JSON structure, allowing only tiny float roundoff."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{label} schema mismatch")
        for key, value in expected.items():
            require_nested_match(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, float):
        require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12),
            f"{label} value mismatch",
        )
    else:
        require(actual == expected, f"{label} value mismatch")


def validate_partition_comparison(
    partition: dict,
    production_labels: dict[int, dict],
    label: str,
) -> dict:
    reference = np.asarray(
        [production_labels[iid]["community"] for iid in partition["ids"]],
        dtype=np.int64,
    )
    expected = partition_metrics(reference, partition["labels"])
    require_nested_match(
        partition["document"].get("comparison_to_corrected_production_labels"),
        expected,
        f"{label} production-label comparison",
    )
    return expected


def pca_control_paths(directory: Path) -> dict[str, Path]:
    return {
        "partition": directory / "partition.json",
        "ids": directory / "ids.json",
        "assignments": directory / "community_assignments.csv",
        "temporal_events": directory / "node_temporal_events.csv",
        "exclusive_by_decade": directory / "exclusive_by_decade.json",
        "graph_time_summary": directory / "graph_time_summary.json",
        "community_growth_by_decade": directory / "community_growth_by_decade.csv",
        "top_communities": directory / "top_communities.json",
    }


def validate_pca_scaler(path: Path, expected_entries: int, label: str) -> dict:
    document = read_json(path)
    require(
        isinstance(document, dict) and set(document) == {"mean", "scale", "n_samples"},
        f"{label} PCA scaler schema mismatch",
    )
    mean = document.get("mean")
    scale = document.get("scale")
    require(
        isinstance(mean, list) and isinstance(scale, list)
        and len(mean) == len(scale) == 32,
        f"{label} PCA scaler dimension mismatch",
    )
    mean_values = np.asarray(mean, dtype=float)
    scale_values = np.asarray(scale, dtype=float)
    require(np.isfinite(mean_values).all(), f"{label} PCA scaler mean is non-finite")
    require(
        np.isfinite(scale_values).all() and np.all(scale_values > 0),
        f"{label} PCA scaler scale is invalid",
    )
    require(
        document.get("n_samples") == expected_entries,
        f"{label} PCA scaler sample count mismatch",
    )
    return document


def validate_temporal_supplements(
    graph_time_path: Path,
    growth_path: Path,
    top_path: Path,
    partition: dict,
    temporal: dict,
    label: str,
) -> dict:
    """Validate the remaining temporal outputs against assignments and event rows."""
    graph_time = read_json(graph_time_path)
    require(isinstance(graph_time, dict), f"{label} graph-time summary is not an object")
    for field, expected in partition["document"].items():
        require(
            graph_time.get(field) == expected,
            f"{label} graph-time summary differs from its partition in {field}",
        )

    labels = partition["labels"]
    events = temporal["events"]
    expected_birth_year = {}
    for record in partition["records"]:
        community = record["community"]
        if community >= 0:
            key = str(community)
            expected_birth_year[key] = min(
                record["year"], expected_birth_year.get(key, record["year"])
            )
    expected_by_decade = {}
    for decade in sorted({event["decade"] for event in events}):
        rows = [event for event in events if event["decade"] == decade]
        count = len(rows)
        values = {
            "n_total": count,
            "n_outlier": sum(event["event_type"] == "outlier" for event in rows),
            "n_cluster_birth_point": sum(
                event["event_type"] == "community_birth" for event in rows
            ),
            "n_existing_cluster": sum(
                event["event_type"] == "existing_community" for event in rows
            ),
            "n_same_community_attachment": sum(
                event["community"] >= 0 and event["n_active_same_community_neighbors"] > 0
                for event in rows
            ),
            "n_cross_community_attachment": sum(
                event["community"] >= 0 and event["n_active_other_communities"] > 0
                for event in rows
            ),
            "n_bridge_attachment": sum(
                event["community"] >= 0 and event["is_bridge_attachment"] for event in rows
            ),
            "n_core_attachment": sum(event["core_periphery"] == "core" for event in rows),
            "n_periphery_attachment": sum(
                event["core_periphery"] == "periphery" for event in rows
            ),
        }
        for field in (
            "n_outlier", "n_cluster_birth_point", "n_existing_cluster",
            "n_same_community_attachment", "n_cross_community_attachment",
            "n_bridge_attachment", "n_core_attachment", "n_periphery_attachment",
        ):
            values[field.removeprefix("n_") + "_ratio"] = values[field] / count
        expected_by_decade[decade] = values
    expected_graph_time = {
        "n_points": len(labels),
        "n_outliers": int(np.sum(labels < 0)),
        "outlier_ratio": float(np.mean(labels < 0)),
        "community_birth_year": expected_birth_year,
        "by_decade": expected_by_decade,
    }
    for field, expected in expected_graph_time.items():
        require_nested_match(graph_time.get(field), expected, f"{label} graph-time {field}")

    counts = Counter(int(value) for value in labels if int(value) >= 0)
    expected_top = [
        {"community": community, "size": size}
        for community, size in counts.most_common(25)
    ]
    top = read_json(top_path)
    require(top == expected_top, f"{label} top-community output mismatch")

    with growth_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames == ["community", "decade", "cumulative_size"],
            f"{label} growth CSV columns mismatch",
        )
        growth = [
            {
                "community": _csv_integer(row["community"], f"{label} growth community", minimum=0),
                "decade": row["decade"].strip(),
                "cumulative_size": _csv_integer(
                    row["cumulative_size"], f"{label} cumulative size", minimum=0
                ),
            }
            for row in reader
        ]
    decades = sorted({event["decade"] for event in events})
    event_counts = Counter(
        (event["community"], event["decade"])
        for event in events
        if event["community"] >= 0
    )
    expected_growth = []
    for item in expected_top:
        running = 0
        for decade in decades:
            running += event_counts[(item["community"], decade)]
            expected_growth.append({
                "community": item["community"],
                "decade": decade,
                "cumulative_size": running,
            })
    require(growth == expected_growth, f"{label} cumulative-growth output mismatch")
    return {
        "graph_time": graph_time,
        "growth": growth,
        "top_communities": top,
    }


def matched_pca_control_summary(
    controls: dict,
    cohort_sha256: str,
    scalers: dict[str, dict],
    scaler_paths: dict[str, Path],
) -> dict:
    summaries = {}
    identical = {}
    method_specific_fields = {
        "feature_preparation", "feature_preparation_sha256",
        "n_feature_successes_before_cohort_restriction", "restrict_ids",
        "graphlet_neighbor_method", "graphlet_neighbor_settings",
        "graphlet_feature_version", "source_graphlet_feature_version",
        "legacy_crystalnn_metadata_compatibility",
    }

    def science_document(document: dict) -> dict:
        return {
            key: value for key, value in document.items()
            if key not in method_specific_fields
        }

    for control, analysis_name in PCA_CONTROLS.items():
        crystal = controls[control]["crystalnn"]
        voronoi = controls[control]["voronoinn"]
        left = crystal["partition"]
        right = voronoi["partition"]
        require(left["ids"] == right["ids"], f"{analysis_name} ordered cohorts differ")
        require(
            left["document"]["cohort_sha256"] == cohort_sha256
            == right["document"]["cohort_sha256"],
            f"{analysis_name} cohort hashes differ",
        )
        validate_shared_provenance(
            left["document"], right["document"], f"{analysis_name} production provenance"
        )
        validate_shared_provenance(
            left["document"], right["document"], f"{analysis_name} partition protocol",
            PARTITION_PROTOCOL_FIELDS,
        )
        require(
            method_specific_fields <= set(left["document"])
            and method_specific_fields <= set(right["document"]),
            f"{analysis_name} method-provenance fields are incomplete",
        )
        require(
            science_document(left["document"]) == science_document(right["document"]),
            f"{analysis_name} partition science fields differ",
        )
        require(
            np.array_equal(left["labels"], right["labels"]),
            f"{analysis_name} assignments differ between graphlet runs",
        )
        require(
            crystal["temporal"]["events"] == voronoi["temporal"]["events"],
            f"{analysis_name} temporal event rows differ between graphlet runs",
        )
        require(
            crystal["temporal"]["by_decade"] == voronoi["temporal"]["by_decade"],
            f"{analysis_name} temporal decade rows differ between graphlet runs",
        )
        require(
            science_document(crystal["temporal"]["document"])
            == science_document(voronoi["temporal"]["document"]),
            f"{analysis_name} exclusive temporal science fields differ",
        )
        require(
            science_document(crystal["supplements"]["graph_time"])
            == science_document(voronoi["supplements"]["graph_time"]),
            f"{analysis_name} graph-time science fields differ",
        )
        artifact_identity = {
            key: digest(crystal["paths"][key]) == digest(voronoi["paths"][key])
            for key in (
                "ids", "assignments", "temporal_events",
                "community_growth_by_decade", "top_communities",
            )
        }
        require(all(artifact_identity.values()), f"{analysis_name} duplicate science files differ")
        count = len(left["ids"])
        summaries[control] = {
            "analysis_name": analysis_name,
            "n_entries": count,
            "n_communities": left["n_communities"],
            "n_noise": left["n_noise"],
            "noise_fraction": left["n_noise"] / count,
            "n_edges": require_int(left["document"].get("n_edges"), f"{analysis_name} edge count"),
            "sigma": require_float(left["document"].get("sigma"), f"{analysis_name} sigma", minimum=0),
            "comparison_to_corrected_production_labels": crystal["comparison"],
            "by_decade": crystal["temporal"]["by_decade"],
            "headline": crystal["temporal"]["headline"],
        }
        identical[control] = artifact_identity
    require(
        scalers["crystalnn"] == scalers["voronoinn"]
        and digest(scaler_paths["crystalnn"]) == digest(scaler_paths["voronoinn"]),
        "Matched PCA standardization scalers differ between graphlet runs",
    )
    return {
        "cohort": {
            "n_entries": next(iter(summaries.values()))["n_entries"],
            "ordered_ids_sha256": cohort_sha256,
            "definition": "exact dated graphlet feature intersection used for both matched PCA controls",
        },
        "controls": summaries,
        "duplicate_science_identity_across_graphlet_runs": identical,
        "standardization_scaler": {
            "n_samples": scalers["crystalnn"]["n_samples"],
            "n_features": len(scalers["crystalnn"]["mean"]),
            "byte_identical_across_graphlet_runs": (
                digest(scaler_paths["crystalnn"]) == digest(scaler_paths["voronoinn"])
            ),
        },
    }


def clustering_scores(left: np.ndarray, right: np.ndarray) -> dict | None:
    if len(left) < 2:
        return None
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    return {
        "n_entries": int(len(left)),
        "ARI": float(adjusted_rand_score(left, right)),
        "NMI_arithmetic": float(normalized_mutual_info_score(left, right, average_method="arithmetic")),
    }


def paired_partition(crystal: dict, voronoi: dict, crystal_coverage: dict, voronoi_coverage: dict) -> dict:
    require(crystal["ids"] == voronoi["ids"], "Partition outputs were not constructed on the same ordered common cohort")
    require(
        crystal["document"]["cohort_sha256"] == voronoi["document"]["cohort_sha256"],
        "Partition ordered-cohort hashes differ",
    )
    validate_shared_provenance(crystal["document"], voronoi["document"], "Partition provenance")
    validate_shared_provenance(
        crystal["document"], voronoi["document"], "Partition protocol", PARTITION_PROTOCOL_FIELDS,
    )
    for left, right in zip(crystal["records"], voronoi["records"]):
        require(left["icsd_id"] == right["icsd_id"], "Partition ID join changed order")
        require(left["year"] == right["year"], f"Partition year mismatch for {left['icsd_id']}")
    left_labels, right_labels = crystal["labels"], voronoi["labels"]
    both_assigned = (left_labels >= 0) & (right_labels >= 0)
    noise_transitions = {
        "assigned_to_assigned": int(np.sum((left_labels >= 0) & (right_labels >= 0))),
        "assigned_to_noise": int(np.sum((left_labels >= 0) & (right_labels < 0))),
        "noise_to_assigned": int(np.sum((left_labels < 0) & (right_labels >= 0))),
        "noise_to_noise": int(np.sum((left_labels < 0) & (right_labels < 0))),
    }
    count = len(left_labels)
    require(sum(noise_transitions.values()) == count, "Partition noise-transition counts do not sum")
    method_summary = {}
    for method, run, coverage in (
        ("crystalnn", crystal, crystal_coverage),
        ("voronoinn", voronoi, voronoi_coverage),
    ):
        method_summary[method] = {
            "feature_coverage": coverage,
            "partition_cohort_entries": count,
            "partition_cohort_fraction_of_feature_successes": count / coverage["n_successful"],
            "n_communities": run["n_communities"],
            "n_noise": run["n_noise"],
            "noise_fraction": run["n_noise"] / count,
            "n_edges": require_int(run["document"].get("n_edges"), f"{method} partition n_edges"),
        }
    return {
        "cohort": {
            "n_entries": count,
            "ordered_ids_sha256": crystal["document"]["cohort_sha256"],
            "definition": "exact ordered cohort used to construct both graphlet partitions",
        },
        "methods": method_summary,
        "agreement": {
            "all_entries_noise_as_one_label": clustering_scores(left_labels, right_labels),
            "jointly_assigned_only": clustering_scores(left_labels[both_assigned], right_labels[both_assigned]),
        },
        "noise_transitions_crystalnn_to_voronoinn": noise_transitions,
    }


def artifact_manifest(paths: dict[str, Path]) -> dict:
    return {name: {"path": str(path), "sha256": digest(path)} for name, path in sorted(paths.items())}


def build_summary(args) -> dict:
    production_labels = read_production_labels(args.production_labels)
    production_labels_sha256 = digest(args.production_labels)
    metadata = {}
    coverage = {}
    for method, path in (
        ("crystalnn", args.crystal_feature_metadata),
        ("voronoinn", args.voronoi_feature_metadata),
    ):
        metadata[method], coverage[method] = validate_feature_metadata(path, method)
    require(
        metadata["crystalnn"]["production_labels_sha256"]
        == metadata["voronoinn"]["production_labels_sha256"],
        "Feature preparations use different production labels",
    )
    require(
        metadata["crystalnn"]["production_labels_sha256"] == production_labels_sha256,
        "Feature preparations do not hash the supplied production labels",
    )
    feature_ids = {
        "crystalnn": validate_feature_ids(
            args.crystal_feature_ids, metadata["crystalnn"], "crystalnn", production_labels,
        ),
        "voronoinn": validate_feature_ids(
            args.voronoi_feature_ids, metadata["voronoinn"], "voronoinn", production_labels,
        ),
    }
    common_feature_ids = sorted(set(feature_ids["crystalnn"]) & set(feature_ids["voronoinn"]))
    require(len(common_feature_ids) >= 2, "Graphlet feature intersection has fewer than two entries")
    common_dated_ids = [
        iid for iid in common_feature_ids
        if production_labels[iid]["year"] is not None
        and 1900 <= production_labels[iid]["year"] <= 2025
    ]
    require(len(common_dated_ids) >= 2, "Dated graphlet feature intersection has fewer than two entries")
    recovery = {
        "crystalnn": validate_recovery(
            args.crystal_recovery, args.crystal_recovery_ids, args.crystal_neighbors,
            args.crystal_exact_audit,
            metadata["crystalnn"], args.crystal_feature_metadata, "crystalnn",
        ),
        "voronoinn": validate_recovery(
            args.voronoi_recovery, args.voronoi_recovery_ids, args.voronoi_neighbors,
            args.voronoi_exact_audit,
            metadata["voronoinn"], args.voronoi_feature_metadata, "voronoinn",
        ),
    }
    partitions = {
        "crystalnn": validate_partition(
            args.crystal_partition, args.crystal_partition_ids, args.crystal_assignments,
            metadata["crystalnn"], args.crystal_feature_metadata, "crystalnn",
        ),
        "voronoinn": validate_partition(
            args.voronoi_partition, args.voronoi_partition_ids, args.voronoi_assignments,
            metadata["voronoinn"], args.voronoi_feature_metadata, "voronoinn",
        ),
    }
    for method, feature_ids_path in (
        ("crystalnn", args.voronoi_feature_ids),
        ("voronoinn", args.crystal_feature_ids),
    ):
        require(
            recovery[method]["ids"] == common_feature_ids,
            f"{method} recovery cohort is not the exact graphlet feature intersection",
        )
        require(
            partitions[method]["ids"] == common_dated_ids,
            f"{method} partition cohort is not the exact dated graphlet feature intersection",
        )
        validate_common_restriction(recovery[method]["document"], feature_ids_path, f"{method} recovery")
        validate_common_restriction(partitions[method]["document"], feature_ids_path, f"{method} partition")
        for record in recovery[method]["records"]:
            require(
                record["query_community"] == production_labels[record["query_id"]]["community"],
                f"{method} recovery production-community mismatch for {record['query_id']}",
            )
        for record in partitions[method]["records"]:
            require(
                record["year"] == production_labels[record["icsd_id"]]["year"],
                f"{method} partition production-year mismatch for {record['icsd_id']}",
            )
        require(
            recovery[method]["document"].get("n_undated")
            == sum(production_labels[iid]["year"] is None for iid in common_feature_ids),
            f"{method} recovery undated count mismatch",
        )
        require(
            partitions[method]["document"].get("n_undated") == 0,
            f"{method} partition reports undated entries",
        )
    temporal = {
        "crystalnn": validate_temporal_replay(
            args.crystal_temporal_events, args.crystal_exclusive_by_decade,
            partitions["crystalnn"], "crystalnn",
        ),
        "voronoinn": validate_temporal_replay(
            args.voronoi_temporal_events, args.voronoi_exclusive_by_decade,
            partitions["voronoinn"], "voronoinn",
        ),
    }
    control_directories = {
        "raw": {
            "crystalnn": args.crystal_pca_raw_dir,
            "voronoinn": args.voronoi_pca_raw_dir,
        },
        "standardized": {
            "crystalnn": args.crystal_pca_standardized_dir,
            "voronoinn": args.voronoi_pca_standardized_dir,
        },
    }
    for method in METHODS:
        require(
            control_directories["raw"][method].parent
            == control_directories["standardized"][method].parent,
            f"{method} PCA controls do not share one replay root",
        )
    scaler_paths = {
        method: control_directories["raw"][method].parent / "pca_subset_scaler.json"
        for method in METHODS
    }
    scalers = {
        method: validate_pca_scaler(path, len(common_dated_ids), method)
        for method, path in scaler_paths.items()
    }
    controls = {control: {} for control in PCA_CONTROLS}
    control_input_paths = {
        f"{method}_pca_subset_scaler": path for method, path in scaler_paths.items()
    }
    for control, analysis_name in PCA_CONTROLS.items():
        for method in METHODS:
            artifact_paths = pca_control_paths(control_directories[control][method])
            partition = validate_partition(
                artifact_paths["partition"],
                artifact_paths["ids"],
                artifact_paths["assignments"],
                metadata[method],
                (
                    args.crystal_feature_metadata
                    if method == "crystalnn"
                    else args.voronoi_feature_metadata
                ),
                method,
                expected_name=analysis_name,
            )
            require(
                partition["ids"] == common_dated_ids,
                f"{method} {analysis_name} cohort is not the exact dated graphlet feature intersection",
            )
            other_feature_ids = (
                args.voronoi_feature_ids if method == "crystalnn" else args.crystal_feature_ids
            )
            validate_common_restriction(
                partition["document"], other_feature_ids, f"{method} {analysis_name}"
            )
            for record in partition["records"]:
                require(
                    record["year"] == production_labels[record["icsd_id"]]["year"],
                    f"{method} {analysis_name} production-year mismatch for {record['icsd_id']}",
                )
            require(
                partition["document"].get("n_undated") == 0,
                f"{method} {analysis_name} reports undated entries",
            )
            comparison = validate_partition_comparison(
                partition, production_labels, f"{method} {analysis_name}"
            )
            replay = validate_temporal_replay(
                artifact_paths["temporal_events"],
                artifact_paths["exclusive_by_decade"],
                partition,
                f"{method} {analysis_name}",
            )
            supplements = validate_temporal_supplements(
                artifact_paths["graph_time_summary"],
                artifact_paths["community_growth_by_decade"],
                artifact_paths["top_communities"],
                partition,
                replay,
                f"{method} {analysis_name}",
            )
            controls[control][method] = {
                "partition": partition,
                "temporal": replay,
                "supplements": supplements,
                "comparison": comparison,
                "paths": artifact_paths,
            }
            for artifact, path in artifact_paths.items():
                control_input_paths[f"{method}_pca_{control}_{artifact}"] = path

    all_documents = (
        [run["document"] for run in recovery.values()]
        + [run["document"] for run in partitions.values()]
        + [
            controls[control][method]["partition"]["document"]
            for control in PCA_CONTROLS
            for method in METHODS
        ]
    )
    for document in all_documents[1:]:
        validate_shared_provenance(all_documents[0], document, "Cross-analysis production provenance")
    paths = {
        "production_labels": args.production_labels,
        "crystal_feature_metadata": args.crystal_feature_metadata,
        "crystal_feature_ids": args.crystal_feature_ids,
        "voronoi_feature_metadata": args.voronoi_feature_metadata,
        "voronoi_feature_ids": args.voronoi_feature_ids,
        "crystal_recovery": args.crystal_recovery,
        "crystal_recovery_ids": args.crystal_recovery_ids,
        "crystal_neighbors": args.crystal_neighbors,
        "crystal_exact_audit": args.crystal_exact_audit,
        "voronoi_recovery": args.voronoi_recovery,
        "voronoi_recovery_ids": args.voronoi_recovery_ids,
        "voronoi_neighbors": args.voronoi_neighbors,
        "voronoi_exact_audit": args.voronoi_exact_audit,
        "crystal_partition": args.crystal_partition,
        "crystal_partition_ids": args.crystal_partition_ids,
        "crystal_assignments": args.crystal_assignments,
        "crystal_temporal_events": args.crystal_temporal_events,
        "crystal_exclusive_by_decade": args.crystal_exclusive_by_decade,
        "voronoi_partition": args.voronoi_partition,
        "voronoi_partition_ids": args.voronoi_partition_ids,
        "voronoi_assignments": args.voronoi_assignments,
        "voronoi_temporal_events": args.voronoi_temporal_events,
        "voronoi_exclusive_by_decade": args.voronoi_exclusive_by_decade,
        **control_input_paths,
    }
    partition_summary = paired_partition(
        partitions["crystalnn"], partitions["voronoinn"],
        coverage["crystalnn"], coverage["voronoinn"],
    )
    return {
        "schema_version": 3,
        "comparison": "graphlet neighbor-rule sensitivity: CrystalNN versus radius-screened VoronoiNN",
        "production_provenance": {field: all_documents[0][field] for field in SHARED_PROVENANCE_FIELDS},
        "analysis_execution": {
            field: all_documents[0][field] for field in ANALYSIS_EXECUTION_FIELDS
        },
        "graphlet_methods": {
            method: {
                "source_graphlet_feature_version": metadata[method].get("graphlet_feature_version"),
                "effective_graphlet_feature_version": gf.GRAPHLET_FEATURE_VERSIONS[method],
                "source_neighbor_method": metadata[method].get("neighbor_method"),
                "neighbor_settings": metadata[method]["neighbor_settings"],
                "representation": metadata[method]["representation"],
                "legacy_crystalnn_metadata_compatibility": coverage[method]["legacy_crystalnn_metadata_compatibility"],
            }
            for method in METHODS
        },
        "paired_recovery": paired_recovery(recovery["crystalnn"], recovery["voronoinn"]),
        "common_cohort_partition": partition_summary,
        "common_cohort_temporal": paired_temporal(
            temporal["crystalnn"], temporal["voronoinn"], partition_summary,
        ),
        "matched_pca_controls": matched_pca_control_summary(
            controls,
            partition_summary["cohort"]["ordered_ids_sha256"],
            scalers,
            scaler_paths,
        ),
        "inputs": artifact_manifest(paths),
        "validation": {
            "status": "passed",
            "policy": "exact feature-intersection cohorts and matching production, method, feature-preparation, protocol, partition, temporal, and matched-PCA-control provenance required",
            "common_feature_intersection_entries": len(common_feature_ids),
            "common_dated_feature_intersection_entries": len(common_dated_ids),
            "matched_pca_controls_validated": list(PCA_CONTROLS),
            "legacy_crystalnn_feature_metadata_compatibility": coverage["crystalnn"]["legacy_crystalnn_metadata_compatibility"],
            "voronoinn_requires_explicit_method_and_version": True,
        },
    }


def percent(value: float) -> str:
    return f"{100 * value:.3f}%"


def render_markdown(summary: dict) -> str:
    recovery = summary["paired_recovery"]
    partition = summary["common_cohort_partition"]
    crystal = recovery["recovery"]["crystalnn"]
    voronoi = recovery["recovery"]["voronoinn"]
    transitions = recovery["outcome_transitions_crystalnn_to_voronoinn"]
    all_scores = partition["agreement"]["all_entries_noise_as_one_label"]
    joint_scores = partition["agreement"]["jointly_assigned_only"]
    temporal = summary["common_cohort_temporal"]
    lines = [
        "# CrystalNN versus VoronoiNN graphlet sensitivity",
        "",
        "All provenance and exact ordered-cohort checks passed.",
        "",
        "## Paired 1-nearest-neighbor recovery",
        "",
        f"Common cohort: {recovery['cohort']['n_entries']:,} entries.",
        "",
        "| Neighbor rule | Hits | Misses | Recovery | ANN exact nearest (512-query audit) |",
        "|---|---:|---:|---:|---:|",
        (
            f"| CrystalNN | {crystal['hits']:,} | {crystal['misses']:,} | {percent(crystal['rate'])} | "
            f"{percent(crystal['ann_exact_nearest_fraction'])} |"
        ),
        (
            f"| VoronoiNN | {voronoi['hits']:,} | {voronoi['misses']:,} | {percent(voronoi['rate'])} | "
            f"{percent(voronoi['ann_exact_nearest_fraction'])} |"
        ),
        "",
        f"VoronoiNN minus CrystalNN: {recovery['recovery']['voronoinn_minus_crystalnn_percentage_points']:+.3f} percentage points.",
        "",
        "| CrystalNN → VoronoiNN outcome | Count |",
        "|---|---:|",
        f"| Hit → hit | {transitions['hit_to_hit']:,} |",
        f"| Hit → miss | {transitions['hit_to_miss']:,} |",
        f"| Miss → hit | {transitions['miss_to_hit']:,} |",
        f"| Miss → miss | {transitions['miss_to_miss']:,} |",
        "",
        f"The selected neighbor ID is unchanged for {recovery['selected_neighbor_identity']['same_id']:,} entries ({percent(recovery['selected_neighbor_identity']['same_id_fraction'])}).",
        "",
        "## Common-cohort partitions",
        "",
        f"Both partitions were constructed on the same {partition['cohort']['n_entries']:,}-entry ordered cohort.",
        "",
        "| Neighbor rule | Feature successes / requested | Communities | Noise | Noise share |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, display in (("crystalnn", "CrystalNN"), ("voronoinn", "VoronoiNN")):
        record = partition["methods"][method]
        cover = record["feature_coverage"]
        lines.append(
            f"| {display} | {cover['n_successful']:,} / {cover['n_requested']:,} "
            f"({percent(cover['success_fraction'])}) | {record['n_communities']:,} | "
            f"{record['n_noise']:,} | {percent(record['noise_fraction'])} |"
        )
    lines.extend([
        "",
        "| Partition comparison | Entries | ARI | Arithmetic NMI |",
        "|---|---:|---:|---:|",
        f"| All entries, noise as one label | {all_scores['n_entries']:,} | {all_scores['ARI']:.6f} | {all_scores['NMI_arithmetic']:.6f} |",
        (
            f"| Jointly assigned only | {joint_scores['n_entries']:,} | {joint_scores['ARI']:.6f} | {joint_scores['NMI_arithmetic']:.6f} |"
            if joint_scores is not None
            else "| Jointly assigned only | <2 | — | — |"
        ),
        "",
        "## Common-cohort temporal replay",
        "",
        (
            f"Both temporal replays use the exact {temporal['cohort']['n_entries']:,}-entry "
            "dated partition cohort. Shares include partition noise in the denominator."
        ),
        "",
        "| Neighbor rule | Decade | Entries | Birth share | Any attachment share |",
        "|---|---:|---:|---:|---:|",
    ])
    for method, display in (("crystalnn", "CrystalNN"), ("voronoinn", "VoronoiNN")):
        for decade in ("1930s", "2010s"):
            row = temporal["methods"][method]["headline"][decade]
            lines.append(
                f"| {display} | {decade} | {row['n_entries']:,} | "
                f"{percent(row['birth_share'])} | {percent(row['any_attachment_share'])} |"
            )
    lines.extend([
        "",
        "| VoronoiNN minus CrystalNN | Birth share | Any attachment share |",
        "|---|---:|---:|",
    ])
    differences = temporal["headline_differences_voronoinn_minus_crystalnn"]
    for decade in ("1930s", "2010s"):
        row = differences[decade]
        lines.append(
            f"| {decade} | {row['birth_share_percentage_points']:+.3f} pp | "
            f"{row['any_attachment_share_percentage_points']:+.3f} pp |"
        )
    pca_controls = summary["matched_pca_controls"]
    lines.extend([
        "",
        "## Matched PCA controls",
        "",
        (
            f"Raw and standardized production PCA were replayed on the same "
            f"{pca_controls['cohort']['n_entries']:,}-entry cohort and independently emitted "
            "under both graphlet runs. Their assignments and temporal-event rows are byte-identical."
        ),
        "",
        "| PCA control | Communities | Noise | ARI | Arithmetic NMI | Pair recall |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for control, display in (("raw", "Raw"), ("standardized", "Standardized")):
        record = pca_controls["controls"][control]
        comparison = record["comparison_to_corrected_production_labels"][
            "all_entries_noise_as_one_label"
        ]
        lines.append(
            f"| {display} | {record['n_communities']:,} | {record['n_noise']:,} | "
            f"{comparison['ARI']:.6f} | {comparison['NMI_arithmetic']:.6f} | "
            f"{percent(comparison['same_community_pair_recall'])} |"
        )
    lines.extend([
        "",
        "| PCA control | Decade | Entries | Birth share | Any attachment share |",
        "|---|---:|---:|---:|---:|",
    ])
    for control, display in (("raw", "Raw"), ("standardized", "Standardized")):
        for decade in ("1930s", "2010s"):
            row = pca_controls["controls"][control]["headline"][decade]
            lines.append(
                f"| {display} | {decade} | {row['n_entries']:,} | "
                f"{percent(row['birth_share'])} | {percent(row['any_attachment_share'])} |"
            )
    lines.extend([
        "",
        "The JSON report retains exact input paths and SHA-256 digests for every compared artifact.",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-labels", type=Path, required=True)
    parser.add_argument("--crystal-feature-metadata", type=Path, required=True)
    parser.add_argument("--crystal-feature-ids", type=Path, required=True)
    parser.add_argument("--voronoi-feature-metadata", type=Path, required=True)
    parser.add_argument("--voronoi-feature-ids", type=Path, required=True)
    parser.add_argument("--crystal-recovery", type=Path, required=True)
    parser.add_argument("--crystal-recovery-ids", type=Path, required=True)
    parser.add_argument("--crystal-neighbors", type=Path, required=True)
    parser.add_argument("--crystal-exact-audit", type=Path, required=True)
    parser.add_argument("--voronoi-recovery", type=Path, required=True)
    parser.add_argument("--voronoi-recovery-ids", type=Path, required=True)
    parser.add_argument("--voronoi-neighbors", type=Path, required=True)
    parser.add_argument("--voronoi-exact-audit", type=Path, required=True)
    parser.add_argument("--crystal-partition", type=Path, required=True)
    parser.add_argument("--crystal-partition-ids", type=Path, required=True)
    parser.add_argument("--crystal-assignments", type=Path, required=True)
    parser.add_argument("--crystal-temporal-events", type=Path, required=True)
    parser.add_argument("--crystal-exclusive-by-decade", type=Path, required=True)
    parser.add_argument("--voronoi-partition", type=Path, required=True)
    parser.add_argument("--voronoi-partition-ids", type=Path, required=True)
    parser.add_argument("--voronoi-assignments", type=Path, required=True)
    parser.add_argument("--voronoi-temporal-events", type=Path, required=True)
    parser.add_argument("--voronoi-exclusive-by-decade", type=Path, required=True)
    parser.add_argument("--crystal-pca-raw-dir", type=Path, required=True)
    parser.add_argument("--crystal-pca-standardized-dir", type=Path, required=True)
    parser.add_argument("--voronoi-pca-raw-dir", type=Path, required=True)
    parser.add_argument("--voronoi-pca-standardized-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    inputs = [value for name, value in vars(args).items() if not name.startswith("output_")]
    outputs = [args.output_json, args.output_markdown]
    require(len(set(outputs)) == 2, "JSON and Markdown outputs must be different paths")
    require(not set(inputs) & set(outputs), "An output path cannot overwrite an input artifact")
    summary = build_summary(args)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    args.output_markdown.write_text(render_markdown(summary))
    print(json.dumps({
        "status": "passed",
        "recovery_entries": summary["paired_recovery"]["cohort"]["n_entries"],
        "partition_entries": summary["common_cohort_partition"]["cohort"]["n_entries"],
        "temporal_entries": summary["common_cohort_temporal"]["cohort"]["n_entries"],
        "output_json": str(args.output_json),
        "output_markdown": str(args.output_markdown),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
