#!/usr/bin/env python3
"""Exact L1 neighbours of historical confused endpoints in a repaired CDF pool.

This is a focused comparison against external historical community labels,
not a new community assignment or a full-population recovery estimate. The
original recovery used 152122 non-noise structures; the repaired densification
CDF targets the 148425 of those with years in 1900--2025. Actual pool sizes and
missing endpoints are computed from the supplied files, not assumed.

Distances sum all 64 x 20 CDF coordinates without rescaling. The first ten
channels describe elemental properties; the other 54 are pair/triplet channels
and include both geometry and chemistry. Their difference is not a pure
geometric contribution. These 54 are also split into four pure geometric
channels (bond distance, angle cosine, short arm, long arm) and 50 chemical
channels. Nearest ties mean exactly equal float64 cityblock
results on the saved CDF values; the smallest ICSD ID breaks a tie. Both
directions of every historical pair enter the query set, but historical-partner
ranks are reported only for the supplied directed pairs.

Only query-block x candidate-block distances are materialized. No feature
generation, approximate search, or structure access is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS


REPRESENTATION = "graphlet-64x20-periodic-images-short-long-arms"
N_CHANNELS = 64
N_BINS = 20
FIRST_ORDER_WIDTH = 10 * N_BINS
PURE_GEOMETRY_CHANNELS = {
    "f2_distance": 10,
    "f3_cos_angle": 31,
    "f3_d_ij": 32,
    "f3_d_jk": 33,
}
PURE_GEOMETRY_COLUMNS = np.asarray([
    channel * N_BINS + bin_index
    for channel in PURE_GEOMETRY_CHANNELS.values() for bin_index in range(N_BINS)
])
PAIR_TRIPLET_CHEMICAL_COLUMNS = np.asarray([
    channel * N_BINS + bin_index
    for channel in range(10, N_CHANNELS) if channel not in PURE_GEOMETRY_CHANNELS.values()
    for bin_index in range(N_BINS)
])
YEAR_RANGE = (1900, 2025)


def _integer(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = float(value)
        result = int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(number) or number != result or (positive and result <= 0):
        raise ValueError(f"Invalid {name}")
    return result


def read_labels(path):
    labels = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"icsd_id", "year", "community"}.issubset(reader.fieldnames or []):
            raise ValueError("Historical labels need icsd_id, year, community columns")
        for row in reader:
            iid = _integer(row["icsd_id"], "label ICSD ID", positive=True)
            if iid in labels:
                raise ValueError(f"Duplicate historical-label ICSD ID {iid}")
            value = row["year"].strip()
            year = None if value.lower() in ("", "nan", "none", "null") else _integer(value, "year")
            labels[iid] = {"year": year, "community": _integer(row["community"], "community")}
    return labels


def read_old_pairs(path, labels):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("Old pairs must be a non-empty JSON list")
    pairs, seen = [], set()
    fields = {"query_id", "neighbor_id", "query_community", "neighbor_community", "distance_l1"}
    for row in payload:
        if not isinstance(row, dict) or not fields.issubset(row):
            raise ValueError("An old pair is missing required fields")
        result = {name: _integer(row[name], name, positive=name.endswith("_id"))
                  for name in fields - {"distance_l1"}}
        try:
            distance = float(row["distance_l1"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid historical distance_l1") from exc
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Historical distance_l1 must be finite and nonnegative")
        result["historical_distance_l1"] = distance
        key = result["query_id"], result["neighbor_id"]
        if key[0] == key[1] or key in seen:
            raise ValueError(f"Self-pair or duplicate historical pair {key}")
        seen.add(key)
        for prefix in ("query", "neighbor"):
            iid = result[f"{prefix}_id"]
            if iid not in labels:
                raise ValueError(f"Historical endpoint {iid} is absent from old labels")
            if labels[iid]["community"] != result[f"{prefix}_community"]:
                raise ValueError(f"Historical community disagrees for endpoint {iid}")
            if labels[iid]["community"] < 0:
                raise ValueError(f"Historical endpoint {iid} has a noise label")
        pairs.append(result)
    return pairs


def load_cdf(cdf_path, ids_path, metadata_path, *, block_size=4096):
    metadata = json.loads(Path(metadata_path).read_text())
    if not isinstance(metadata, dict):
        raise ValueError("CDF metadata must be a JSON object")
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("CDF feature_version does not match the repaired implementation")
    if metadata.get("neighbor_settings") != NEIGHBOR_SETTINGS:
        raise ValueError("CDF neighbor_settings do not match the repaired implementation")
    if metadata.get("representation") != REPRESENTATION:
        raise ValueError("CDF representation is not the repaired 64 x 20 graphlet representation")
    if metadata.get("target_year_range") != list(YEAR_RANGE):
        raise ValueError("CDF metadata must declare the dated 1900--2025 target cohort")
    payload = json.loads(Path(ids_path).read_text())
    if not isinstance(payload, list):
        raise ValueError("CDF IDs must be a JSON list in matrix-row order")
    ids = np.asarray([_integer(value, "CDF ICSD ID", positive=True) for value in payload], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("CDF IDs contain duplicates")
    cdf = np.load(cdf_path, mmap_mode="r", allow_pickle=False)
    if cdf.ndim != 2 or cdf.shape != (len(ids), N_CHANNELS * N_BINS):
        raise ValueError("CDF shape must be (number of IDs, 1280)")
    if not np.issubdtype(cdf.dtype, np.floating):
        raise ValueError("CDF matrix must have floating-point dtype")
    if len(ids) < 2:
        raise ValueError("At least two candidate structures are needed")
    counts = [metadata[name] for name in ("n_successful", "n_structures") if name in metadata]
    if not counts or any(_integer(value, "metadata row count") != len(ids) for value in counts):
        raise ValueError("CDF metadata row count is missing or disagrees with IDs/matrix")
    if "n_requested" in metadata and "n_failed" in metadata:
        requested = _integer(metadata["n_requested"], "n_requested")
        failed = _integer(metadata["n_failed"], "n_failed")
        if failed < 0 or requested != len(ids) + failed:
            raise ValueError("CDF metadata requested/successful/failed counts disagree")
    if metadata.get("cdf_row_order") != "ascending ICSD ID" or np.any(np.diff(ids) <= 0):
        raise ValueError("CDF IDs must match the declared ascending ICSD ID row order")
    for start in range(0, len(ids), block_size):
        values = cdf[start:start + block_size].reshape(-1, N_CHANNELS, N_BINS)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite CDF values in row block starting {start}")
        # Float32 cumulative sums can overshoot one by several ulps.
        tolerance = 2e-6
        if (np.any(values < -tolerance) or np.any(values > 1 + tolerance)
                or np.any(np.diff(values, axis=2) < -tolerance)
                or np.any(np.abs(values[:, :, -1] - 1) > tolerance)):
            raise ValueError(f"Invalid normalized cumulative histograms in row block starting {start}")
    return cdf, ids, metadata


def _eligible(record):
    year = record["year"]
    return record["community"] >= 0 and year is not None and YEAR_RANGE[0] <= year <= YEAR_RANGE[1]


def _missing_reason(iid, labels):
    record = labels[iid]
    if record["year"] is None:
        return "no_year_in_historical_labels"
    if not YEAR_RANGE[0] <= record["year"] <= YEAR_RANGE[1]:
        return "year_outside_repaired_target_range"
    return "eligible_but_absent_from_saved_cdf"


def distance_parts(left, right):
    """Split raw L1 into first-order chemistry, pure geometry and other chemistry.

    The 50 pair/triplet chemical channels describe elemental properties sampled
    on the neighbour graph, so their weights and membership still depend on
    structure. This is a partition of descriptor distance, not causal attribution.
    """
    left = np.asarray(left, dtype=np.float64).reshape(1, -1)
    right = np.asarray(right, dtype=np.float64).reshape(1, -1)
    return {
        "distance_l1": float(cdist(left, right, metric="cityblock")[0, 0]),
        "first_10_elemental_channels_l1": float(cdist(left[:, :FIRST_ORDER_WIDTH], right[:, :FIRST_ORDER_WIDTH], metric="cityblock")[0, 0]),
        "remaining_54_pair_triplet_channels_l1": float(cdist(left[:, FIRST_ORDER_WIDTH:], right[:, FIRST_ORDER_WIDTH:], metric="cityblock")[0, 0]),
        "pure_geometry_four_channels_l1": float(cdist(left[:, PURE_GEOMETRY_COLUMNS], right[:, PURE_GEOMETRY_COLUMNS], metric="cityblock")[0, 0]),
        "pair_triplet_chemical_50_channels_l1": float(cdist(left[:, PAIR_TRIPLET_CHEMICAL_COLUMNS], right[:, PAIR_TRIPLET_CHEMICAL_COLUMNS], metric="cityblock")[0, 0]),
    }


def compare(cdf, ids, labels, pairs, *, query_block_size=8, candidate_block_size=4096):
    """Stream exact search and historical-neighbour ranks over the saved pool."""
    if query_block_size < 1 or candidate_block_size < 1:
        raise ValueError("Block sizes must be positive")
    index = {int(iid): row for row, iid in enumerate(ids)}
    for iid in index:
        if iid not in labels or not _eligible(labels[iid]):
            raise ValueError(f"CDF candidate {iid} is not in the historical dated non-noise cohort")
    requested_queries = sorted({pair[key] for pair in pairs for key in ("query_id", "neighbor_id")})
    queries = [iid for iid in requested_queries if iid in index]
    unavailable_queries = [{"icsd_id": iid, "reason": _missing_reason(iid, labels), **labels[iid]}
                           for iid in requested_queries if iid not in index]
    pair_results, outgoing = [], {}
    for pair in pairs:
        result = dict(pair)
        missing = [{"role": prefix, "icsd_id": pair[f"{prefix}_id"],
                    "reason": _missing_reason(pair[f"{prefix}_id"], labels)}
                   for prefix in ("query", "neighbor") if pair[f"{prefix}_id"] not in index]
        result["status"] = "unavailable_endpoint" if missing else "compared"
        result["unavailable_endpoints"] = missing
        if not missing:
            result["repaired_distance"] = distance_parts(cdf[index[pair["query_id"]]], cdf[index[pair["neighbor_id"]]])
            result.update(n_strictly_closer=0, n_at_same_distance=0, n_tied_smaller_id=0)
            outgoing.setdefault(pair["query_id"], []).append(result)
        pair_results.append(result)

    query_results = []
    for q_start in range(0, len(queries), query_block_size):
        query_ids = queries[q_start:q_start + query_block_size]
        vectors = np.asarray(cdf[[index[iid] for iid in query_ids]], dtype=np.float64)
        best_distances = np.full(len(query_ids), np.inf)
        best_ids = [[] for _ in query_ids]
        for start in range(0, len(ids), candidate_block_size):
            candidate_ids = ids[start:start + candidate_block_size]
            candidates = np.asarray(cdf[start:start + candidate_block_size], dtype=np.float64)
            distances = cdist(vectors, candidates, metric="cityblock")
            for q, iid in enumerate(query_ids):
                values = distances[q]
                values[candidate_ids == iid] = np.inf  # Identity, never assumed ANN position.
                minimum = float(np.min(values))
                if minimum < best_distances[q]:
                    best_distances[q] = minimum
                    best_ids[q] = candidate_ids[values == minimum].tolist()
                elif minimum == best_distances[q] and math.isfinite(minimum):
                    best_ids[q].extend(candidate_ids[values == minimum].tolist())
                for result in outgoing.get(iid, []):
                    threshold = result["repaired_distance"]["distance_l1"]
                    equal = values == threshold
                    result["n_strictly_closer"] += int(np.count_nonzero(values < threshold))
                    result["n_at_same_distance"] += int(np.count_nonzero(equal))
                    result["n_tied_smaller_id"] += int(np.count_nonzero(equal & (candidate_ids < result["neighbor_id"])))
        for q, iid in enumerate(query_ids):
            tied_ids = sorted(best_ids[q])
            if not tied_ids:
                raise ValueError(f"No non-self nearest neighbour found for {iid}")
            neighbor_id = tied_ids[0]
            result = {
                "query_id": iid,
                "query_old_community": labels[iid]["community"],
                "nearest_id": neighbor_id,
                "nearest_old_community": labels[neighbor_id]["community"],
                "nearest_distance": distance_parts(cdf[index[iid]], cdf[index[neighbor_id]]),
                "nearest_tie_count": len(tied_ids),
                "nearest_tied_ids": tied_ids,
                "nearest_tie_old_community_counts": dict(sorted(Counter(labels[j]["community"] for j in tied_ids).items())),
                "selected_neighbor_same_old_community": labels[iid]["community"] == labels[neighbor_id]["community"],
                "any_tied_neighbor_same_old_community": any(labels[iid]["community"] == labels[j]["community"] for j in tied_ids),
            }
            query_results.append(result)
            for pair_result in outgoing.get(iid, []):
                closer, equal = pair_result["n_strictly_closer"], pair_result["n_at_same_distance"]
                if equal < 1:
                    raise ValueError("Historical partner was not found at its own exact distance")
                pair_result.update(
                    rank_first=closer + 1,
                    rank_last=closer + equal,
                    rank_with_smallest_id_tiebreak=closer + pair_result["n_tied_smaller_id"] + 1,
                    remains_nearest_including_ties=closer == 0,
                    is_selected_nearest=pair_result["neighbor_id"] == neighbor_id,
                    repaired_nearest_id=neighbor_id,
                    repaired_nearest_old_community=labels[neighbor_id]["community"],
                    repaired_nearest_distance_l1=float(best_distances[q]),
                )
        print(f"Exact L1 queries completed: {min(q_start + query_block_size, len(queries))}/{len(queries)}", flush=True)

    historical_pool = {iid for iid, row in labels.items() if row["community"] >= 0}
    dated_pool = {iid for iid, row in labels.items() if _eligible(row)}
    compared = [row for row in pair_results if row["status"] == "compared"]
    summary = {
        "candidate_pool": {
            "historical_nonnoise_label_count": len(historical_pool),
            "historical_dated_1900_2025_nonnoise_count": len(dated_pool),
            "saved_repaired_cdf_count": len(ids),
            "matches_full_historical_recovery_pool": set(index) == historical_pool,
            "matches_dated_target_pool": set(index) == dated_pool,
            "historical_ids_outside_dated_target_count": len(historical_pool - dated_pool),
            "dated_target_ids_absent_from_cdf": sorted(dated_pool - index.keys()),
        },
        "endpoint_queries_requested": len(requested_queries),
        "endpoint_queries_available": len(query_results),
        "unavailable_queries": unavailable_queries,
        "historical_directed_pairs_requested": len(pairs),
        "historical_directed_pairs_compared": len(compared),
        "historical_directed_pairs_unavailable": len(pair_results) - len(compared),
        "old_partner_remains_nearest_including_ties": sum(row["remains_nearest_including_ties"] for row in compared),
        "old_partner_is_selected_nearest": sum(row["is_selected_nearest"] for row in compared),
        "endpoint_queries_selected_neighbor_same_old_community": sum(row["selected_neighbor_same_old_community"] for row in query_results),
        "endpoint_queries_any_tied_neighbor_same_old_community": sum(row["any_tied_neighbor_same_old_community"] for row in query_results),
    }
    return {"summary": summary, "queries": query_results, "historical_pairs": pair_results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("cdf", "ids", "metadata", "old-pairs", "old-community-labels", "outdir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--query-block-size", type=int, default=8)
    parser.add_argument("--candidate-block-size", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.query_block_size < 1 or args.candidate_block_size < 1:
        parser.error("Block sizes must be positive")
    labels = read_labels(args.old_community_labels)
    pairs = read_old_pairs(args.old_pairs, labels)
    cdf, ids, metadata = load_cdf(args.cdf, args.ids, args.metadata, block_size=args.candidate_block_size)
    print(f"Validated repaired CDF: {cdf.shape[0]} candidates, {cdf.shape[1]} coordinates", flush=True)
    results = compare(cdf, ids, labels, pairs, query_block_size=args.query_block_size,
                      candidate_block_size=args.candidate_block_size)
    results["provenance"] = {
        "feature_version": FEATURE_VERSION,
        "neighbor_settings": dict(NEIGHBOR_SETTINGS),
        "representation": REPRESENTATION,
        "input_files": {name: str(getattr(args, name).resolve())
                        for name in ("cdf", "ids", "metadata", "old_pairs", "old_community_labels")},
        "cdf_metadata": metadata,
        "cdf_shape": list(cdf.shape),
        "cdf_dtype": str(cdf.dtype),
        "distance": "scipy.spatial.distance.cdist cityblock, float64 arithmetic on stored CDF values; raw sum over 1280 coordinates",
        "tie_rule": "exact float64 distance equality; select smallest ICSD ID; exclude query ICSD ID",
        "historical_partner_rank": "one-based among non-self candidates; interval spans exact ties; deterministic rank breaks ties by ICSD ID",
        "distance_partition": "channels 0:10 elemental properties; channels 10:64 pair/triplet chemistry and geometry, additionally split into pure geometric channels 10,31,32,33 and the remaining 50 chemical channels",
        "pure_geometry_channel_indices": dict(PURE_GEOMETRY_CHANNELS),
        "community_labels_role": "external historical reference labels, not regenerated or validated as repaired communities",
        "interpretation_limits": [
            "Focused endpoint queries are selected historical confusions, not an unbiased full-population recovery estimate.",
            "The historical recovery used 152122 non-noise entries; the dated target contains 148425. Actual supplied candidate coverage is reported explicitly.",
            "Historical and repaired descriptors/bin edges differ, so absolute old and repaired distances are not calibrated to one another.",
            "ID uniqueness/order, metadata counts and CDF validity are checked; row identity relies on the IDs sidecar produced with the matrix.",
            "Eligible IDs absent from the saved CDF are reported as absent; feature failure is not inferred without its separate failure report.",
            "The four geometric channels contain distances/angles. The 50 pair/triplet chemical channels contain elemental-property statistics on neighbour-selected graphlets, so this partition is not causal attribution.",
        ],
        "query_block_size": args.query_block_size,
        "candidate_block_size": args.candidate_block_size,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / "repaired_confused_pairs.json"
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    summary = results["summary"]
    pool = summary["candidate_pool"]
    lines = [
        "Exact comparison of historical confused endpoints",
        "",
        f"Candidate pool: {pool['saved_repaired_cdf_count']} corrected CDF rows; {pool['historical_dated_1900_2025_nonnoise_count']} eligible dated historical entries; {pool['historical_nonnoise_label_count']} historical non-noise entries overall.",
        f"Available endpoint queries: {summary['endpoint_queries_available']}/{summary['endpoint_queries_requested']}.",
        f"Comparable directed historical pairs: {summary['historical_directed_pairs_compared']}/{summary['historical_directed_pairs_requested']}.",
        f"Historical partner remains an exact nearest neighbour (ties included): {summary['old_partner_remains_nearest_including_ties']}/{summary['historical_directed_pairs_compared']}.",
        f"Historical partner selected after smallest-ID tie break: {summary['old_partner_is_selected_nearest']}/{summary['historical_directed_pairs_compared']}.",
        f"Endpoint queries selecting the same historical community: {summary['endpoint_queries_selected_neighbor_same_old_community']}/{summary['endpoint_queries_available']}.",
        "",
        "These are selected endpoint queries against historical reference labels. They do not estimate full-population recovery, and the dated candidate pool differs from the original full recovery pool. Remaining-54-channel distances contain both pair/triplet chemistry and geometry; separate contributions for four pure geometric channels and 50 chemical channels are also reported.",
        "See repaired_confused_pairs.json for every nearest-neighbour identity, all exact ties, historical-partner ranks, distance partitions and unavailable endpoints.",
    ]
    (args.outdir / "repaired_confused_pairs.md").write_text("\n\n".join(lines) + "\n")
    print(f"Wrote {output}", flush=True)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
