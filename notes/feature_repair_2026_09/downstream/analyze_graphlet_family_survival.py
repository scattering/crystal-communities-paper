#!/usr/bin/env python3
"""Audit named production communities in matched graphlet comparisons.

This analysis deliberately separates two questions:

1. local 1-nearest-neighbor recovery of corrected production labels; and
2. preservation or fragmentation after an independent graphlet partition.

Alternative-partition community labels are method-local arbitrary integers.
They are never matched numerically between CrystalNN and VoronoiNN runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_VERSION = "graphlet-family-survival-v1"
RECORD_END_YEAR = 2015

# These values freeze the checked systematic-survey rows used in the paper.
# The descriptor CSV remains the source of descriptions, member counts, and
# evidence status; disagreement with this map is treated as a stale input.
SURVEY_EVENT_YEARS = {
    701: 1984,
    1035: 2005,
    2378: 1989,
    1999: 1981,
    519: 1986,
    1294: 1995,
    74: 1993,
    686: 1997,
    961: 1996,
    2570: 1993,
    2722: 1996,
    1911: 2004,
    479: 2006,
    1802: 1991,
    2815: 2009,
    786: 1997,
    1533: 1996,
    855: 2004,
    2146: 1993,
    2662: 1987,
}

# These are the independently specified literature-year probes in S8, rather
# than the years that maximize the systematic survey score.
TARGETED_EVENTS = {
    2662: {
        "name": "YBa2Cu3O7-related 123 neighborhood",
        "year": 1986,
        "context": "cuprate superconductivity reference year",
    },
    74: {
        "name": "dominant manganite-seed mixed-oxide neighborhood",
        "year": 1994,
        "context": "colossal-magnetoresistance reference year",
    },
    1294: {
        "name": "supplementary rhombohedral manganite-rich neighborhood",
        "year": 1994,
        "context": "colossal-magnetoresistance reference year",
    },
    1035: {
        "name": "LaFeAsO-containing neighborhood",
        "year": 2008,
        "context": "iron-pnictide superconductivity reference year",
    },
}

FOCUS_COMMUNITIES = (2662, 1802, 519, 74, 1294, 686, 1035, 2570)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def sha256_ids(ids: Iterable[int]) -> str:
    payload = json.dumps(list(ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_intlike(value: Any, label: str, *, allow_blank: bool = False) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text:
        if allow_blank:
            return None
        raise ValueError(f"{label} is blank")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{label} is not numeric: {text!r}") from exc
    require(math.isfinite(number) and number.is_integer(), f"{label} is not an integer: {text!r}")
    return int(number)


def parse_bool(value: Any, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{label} must be true or false")


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        require(required <= set(reader.fieldnames), f"CSV columns mismatch for {path}")
        rows = list(reader)
    require(bool(rows), f"CSV is empty: {path}")
    return rows


def read_production(path: Path) -> tuple[list[int], dict[int, dict[str, int | None]]]:
    rows = read_csv(path, {"icsd_id", "year", "community"})
    order: list[int] = []
    records: dict[int, dict[str, int | None]] = {}
    for line, row in enumerate(rows, 2):
        iid = parse_intlike(row["icsd_id"], f"production line {line} ID")
        year = parse_intlike(row["year"], f"production line {line} year", allow_blank=True)
        community = parse_intlike(row["community"], f"production line {line} community")
        assert iid is not None and community is not None
        require(iid > 0, f"production line {line} has non-positive ID")
        require(community >= -1, f"production line {line} has community below -1")
        require(iid not in records, f"duplicate production ID {iid}")
        order.append(iid)
        records[iid] = {"year": year, "community": community}
    return order, records


def read_alternate_assignments(
    path: Path,
    production: dict[int, dict[str, int | None]],
) -> list[dict[str, int]]:
    rows = read_csv(path, {"icsd_id", "year", "community"})
    result: list[dict[str, int]] = []
    seen: set[int] = set()
    for line, row in enumerate(rows, 2):
        iid = parse_intlike(row["icsd_id"], f"assignment line {line} ID")
        year = parse_intlike(row["year"], f"assignment line {line} year")
        alternate = parse_intlike(row["community"], f"assignment line {line} community")
        assert iid is not None and year is not None and alternate is not None
        require(iid not in seen, f"duplicate alternate-assignment ID {iid}")
        require(iid in production, f"alternate assignment contains unknown ID {iid}")
        require(production[iid]["year"] == year, f"publication year mismatch for ID {iid}")
        production_community = production[iid]["community"]
        require(
            isinstance(production_community, int) and production_community >= 0,
            f"alternate assignment includes production-noise ID {iid}",
        )
        require(alternate >= -1, f"alternate community below -1 for ID {iid}")
        seen.add(iid)
        result.append(
            {
                "icsd_id": iid,
                "year": year,
                "production_community": production_community,
                "alternate_community": alternate,
            }
        )
    return result


def read_neighbors(
    path: Path,
    production: dict[int, dict[str, int | None]],
) -> list[dict[str, int | bool]]:
    rows = read_csv(
        path,
        {
            "query_id",
            "query_community",
            "neighbor_id",
            "neighbor_community",
            "same_community",
        },
    )
    result: list[dict[str, int | bool]] = []
    seen: set[int] = set()
    for line, row in enumerate(rows, 2):
        query_id = parse_intlike(row["query_id"], f"neighbor line {line} query ID")
        query_community = parse_intlike(
            row["query_community"], f"neighbor line {line} query community"
        )
        neighbor_id = parse_intlike(row["neighbor_id"], f"neighbor line {line} neighbor ID")
        neighbor_community = parse_intlike(
            row["neighbor_community"], f"neighbor line {line} neighbor community"
        )
        assert None not in (query_id, query_community, neighbor_id, neighbor_community)
        query_id = int(query_id)
        neighbor_id = int(neighbor_id)
        query_community = int(query_community)
        neighbor_community = int(neighbor_community)
        same = parse_bool(row["same_community"], f"neighbor line {line} same_community")
        require(query_id not in seen, f"duplicate neighbor query ID {query_id}")
        require(query_id != neighbor_id, f"self neighbor retained for ID {query_id}")
        require(query_id in production and neighbor_id in production, "neighbor CSV contains unknown ID")
        require(
            production[query_id]["community"] == query_community,
            f"query production-label mismatch for ID {query_id}",
        )
        require(
            production[neighbor_id]["community"] == neighbor_community,
            f"neighbor production-label mismatch for ID {neighbor_id}",
        )
        require(same == (query_community == neighbor_community), f"same_community mismatch for ID {query_id}")
        seen.add(query_id)
        result.append(
            {
                "query_id": query_id,
                "query_community": query_community,
                "neighbor_id": neighbor_id,
                "neighbor_community": neighbor_community,
                "same_community": same,
            }
        )
    query_ids = set(seen)
    require(
        all(int(row["neighbor_id"]) in query_ids for row in result),
        "selected neighbor lies outside the query candidate cohort",
    )
    return result


def read_top20(path: Path) -> list[dict[str, Any]]:
    required = {
        "rank",
        "community",
        "n_all_members",
        "n_dated_survey_members",
        "event_year",
        "description",
        "space_group_evidence",
        "top_reduced_formulas",
        "post_window_complete_through_2015",
        "status",
    }
    rows = read_csv(path, required)
    result: list[dict[str, Any]] = []
    for line, row in enumerate(rows, 2):
        record = {
            "rank": parse_intlike(row["rank"], f"top20 line {line} rank"),
            "community": parse_intlike(row["community"], f"top20 line {line} community"),
            "n_all_members": parse_intlike(row["n_all_members"], f"top20 line {line} n_all_members"),
            "n_dated_members": parse_intlike(
                row["n_dated_survey_members"], f"top20 line {line} n_dated_members"
            ),
            "survey_event_year": parse_intlike(row["event_year"], f"top20 line {line} event year"),
            "description": row["description"].strip(),
            "space_group_evidence": row["space_group_evidence"].strip(),
            "top_reduced_formulas": row["top_reduced_formulas"].strip(),
            "post_window_complete_through_2015": parse_bool(
                row["post_window_complete_through_2015"],
                f"top20 line {line} post-window completeness",
            ),
            "label_status": row["status"].strip(),
        }
        require(all(record[key] for key in ("description", "space_group_evidence", "label_status")), f"blank evidence field at top20 line {line}")
        result.append(record)
    require(len(result) == 20, "checked top-20 descriptor file does not contain 20 rows")
    require([row["rank"] for row in result] == list(range(1, 21)), "top-20 ranks are not 1..20")
    observed = {int(row["community"]): int(row["survey_event_year"]) for row in result}
    require(observed == SURVEY_EVENT_YEARS, "top-20 event-year mapping differs from the checked map")
    return result


def read_family_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict) and isinstance(value.get("groups"), list), "family manifest schema mismatch")
    seen: set[int] = set()
    for index, group in enumerate(value["groups"]):
        require(isinstance(group, dict), f"family group {index} is not an object")
        require(isinstance(group.get("name"), str) and group["name"].strip(), f"family group {index} has no name")
        require(isinstance(group.get("communities"), list) and group["communities"], f"family group {index} has no communities")
        communities = group["communities"]
        require(all(isinstance(item, int) and item >= 0 for item in communities), f"family group {index} has invalid community")
        require(len(communities) == len(set(communities)), f"family group {index} repeats a community")
        require(not (seen & set(communities)), "family manifest groups overlap")
        seen.update(communities)
    return value


def choose2(value: int) -> int:
    return value * (value - 1) // 2 if value >= 2 else 0


def clusters_for_fraction(counts: list[int], fraction: float) -> int | None:
    require(0 < fraction <= 1, "cluster coverage fraction must be in (0, 1]")
    total = sum(counts)
    if total == 0:
        return None
    running = 0
    for index, count in enumerate(sorted(counts, reverse=True), 1):
        running += count
        if running / total >= fraction:
            return index
    raise AssertionError("cluster coverage accumulation did not reach its total")


def compute_partition_metrics(
    rows: list[dict[str, int]],
    communities: set[int],
    *,
    n_all_members: int,
    n_dated_members: int,
) -> tuple[dict[str, Any], int | None]:
    members = [row for row in rows if row["production_community"] in communities]
    n_members = len(members)
    noise = sum(row["alternate_community"] == -1 for row in members)
    counts = Counter(
        row["alternate_community"]
        for row in members
        if row["alternate_community"] >= 0
    )
    nonnoise = n_members - noise
    dominant: int | None = None
    overlap = 0
    cluster_size = 0
    purity: float | None = None
    if counts:
        overlap = max(counts.values())
        tied_labels = [label for label, count in counts.items() if count == overlap]
        # Numeric community labels are arbitrary. Resolve an exact overlap tie
        # with cluster membership, which is invariant to a label permutation.
        dominant = min(
            tied_labels,
            key=lambda label: min(
                row["icsd_id"] for row in rows if row["alternate_community"] == label
            ),
        )
        cluster_size = sum(row["alternate_community"] == dominant for row in rows)
        purity = overlap / cluster_size
    probabilities = [count / nonnoise for count in counts.values()] if nonnoise else []
    entropy = -sum(value * math.log(value) for value in probabilities)
    same_nonnoise_pairs = sum(choose2(count) for count in counts.values())
    result = {
        "production_communities": sorted(communities),
        "production_all_members": n_all_members,
        "production_dated_members": n_dated_members,
        "matched_dated_members": n_members,
        "coverage_of_all_members": n_members / n_all_members,
        "coverage_of_dated_members": n_members / n_dated_members,
        "alternate_noise_count": noise,
        "alternate_noise_fraction": noise / n_members if n_members else None,
        "alternate_nonnoise_count": nonnoise,
        "alternate_nonnoise_fraction": nonnoise / n_members if n_members else None,
        "nonnoise_cluster_count": len(counts),
        "dominant_alternate_cluster_label": dominant,
        "dominant_label_scope": "method-local arbitrary label; not matched numerically across representations",
        "dominant_overlap_tie_break": "smallest ICSD ID in the complete alternate cluster",
        "dominant_overlap_count": overlap,
        "dominant_cluster_recall_all_members": overlap / n_members if n_members else None,
        "dominant_cluster_recall_nonnoise_members": overlap / nonnoise if nonnoise else None,
        "dominant_cluster_size_in_matched_cohort": cluster_size,
        "dominant_cluster_purity": purity,
        "fragmentation_entropy_nats_nonnoise": entropy if nonnoise else None,
        "fragmentation_effective_cluster_count_nonnoise": math.exp(entropy) if nonnoise else None,
        "clusters_for_50_percent_nonnoise": clusters_for_fraction(list(counts.values()), 0.5),
        "clusters_for_80_percent_nonnoise": clusters_for_fraction(list(counts.values()), 0.8),
        "clusters_for_90_percent_nonnoise": clusters_for_fraction(list(counts.values()), 0.9),
        "same_nonnoise_cluster_pair_count": same_nonnoise_pairs,
        "pair_recall_all_family_pairs": (
            same_nonnoise_pairs / choose2(n_members) if choose2(n_members) else None
        ),
        "pair_recall_nonnoise_family_pairs": (
            same_nonnoise_pairs / choose2(nonnoise) if choose2(nonnoise) else None
        ),
    }
    return result, dominant


def compute_local_recovery(
    rows: list[dict[str, int | bool]],
    communities: set[int],
    *,
    n_all_members: int,
) -> dict[str, Any]:
    members = [row for row in rows if int(row["query_community"]) in communities]
    exact_hits = sum(bool(row["same_community"]) for row in members)
    group_hits = sum(int(row["neighbor_community"]) in communities for row in members)
    return {
        "production_communities": sorted(communities),
        "evaluated_members": len(members),
        "coverage_of_all_members": len(members) / n_all_members,
        "exact_production_community_hits": exact_hits,
        "exact_production_community_recovery": exact_hits / len(members) if members else None,
        "family_union_hits": group_hits,
        "family_union_recovery": group_hits / len(members) if members else None,
        "cross_community_hits_within_family_union": group_hits - exact_hits,
    }


def window_counts(rows: list[dict[str, int]], event_year: int) -> dict[str, Any]:
    post_end = min(event_year + 10, RECORD_END_YEAR)
    pre = sum(event_year - 10 < row["year"] <= event_year for row in rows)
    post = sum(event_year < row["year"] <= post_end for row in rows)
    return {
        "n_total": len(rows),
        "pre_count": pre,
        "post_count": post,
        "pre_rate_per_nominal_ten_years": pre / 10,
        "post_rate_per_nominal_ten_years": post / 10,
        "fold": post / pre if pre else None,
        "fold_status": "defined" if pre else "undefined_zero_pre",
        "survey_score": post * post / pre if pre else post,
    }


def compute_temporal_event(
    rows: list[dict[str, int]],
    communities: set[int],
    dominant: int | None,
    event: dict[str, Any],
) -> dict[str, Any]:
    year = int(event["year"])
    family = [row for row in rows if row["production_community"] in communities]
    captured = [row for row in family if row["alternate_community"] == dominant]
    full_cluster = [row for row in rows if row["alternate_community"] == dominant]
    return {
        **event,
        "pre_window": f"({year - 10}, {year}]",
        "post_window": f"({year}, {min(year + 10, RECORD_END_YEAR)}]",
        "nominal_post_window_complete_through_2015": year <= RECORD_END_YEAR - 10,
        "available_post_years": max(0, min(10, RECORD_END_YEAR - year)),
        "family_members_in_matched_cohort": window_counts(family, year),
        "family_members_captured_by_dominant_cluster": window_counts(captured, year),
        "full_dominant_alternate_cluster": window_counts(full_cluster, year),
    }


def describe_top20(
    rows: list[dict[str, int]],
    neighbors: list[dict[str, int | bool]],
    top20: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for descriptor in top20:
        community = int(descriptor["community"])
        communities = {community}
        partition, dominant = compute_partition_metrics(
            rows,
            communities,
            n_all_members=int(descriptor["n_all_members"]),
            n_dated_members=int(descriptor["n_dated_members"]),
        )
        local = compute_local_recovery(
            neighbors,
            communities,
            n_all_members=int(descriptor["n_all_members"]),
        )
        survey_event = compute_temporal_event(
            rows,
            communities,
            dominant,
            {
                "kind": "systematic_survey_maximizing_year",
                "year": int(descriptor["survey_event_year"]),
                "context": "checked production-partition top-20 survey year",
            },
        )
        targeted: list[dict[str, Any]] = []
        if community in TARGETED_EVENTS:
            targeted.append(
                compute_temporal_event(
                    rows,
                    communities,
                    dominant,
                    {"kind": "targeted_literature_year", **TARGETED_EVENTS[community]},
                )
            )
        results.append(
            {
                **descriptor,
                "local_neighbor_recovery": local,
                "independent_partition": partition,
                "survey_event": survey_event,
                "targeted_events": targeted,
            }
        )

    partition_values = [row["independent_partition"] for row in results]
    local_values = [row["local_neighbor_recovery"] for row in results]
    temporal_values = [
        row["survey_event"]["full_dominant_alternate_cluster"] for row in results
    ]
    local_queries = sum(int(value["evaluated_members"]) for value in local_values)
    local_hits = sum(int(value["exact_production_community_hits"]) for value in local_values)
    twofold_or_zero = sum(
        (value["pre_count"] == 0 and value["post_count"] > 0)
        or (value["pre_count"] > 0 and value["post_count"] >= 2 * value["pre_count"])
        for value in temporal_values
    )
    summary = {
        "n_communities": len(results),
        "local_neighbor_queries": local_queries,
        "local_neighbor_hits": local_hits,
        "local_entry_weighted_recovery": local_hits / local_queries,
        "local_median_community_recovery": median(
            float(value["exact_production_community_recovery"]) for value in local_values
        ),
        "partition_median_dominant_cluster_recall": median(
            float(value["dominant_cluster_recall_all_members"])
            for value in partition_values
        ),
        "partition_median_dominant_cluster_purity": median(
            float(value["dominant_cluster_purity"]) for value in partition_values
        ),
        "partition_median_effective_cluster_count_nonnoise": median(
            float(value["fragmentation_effective_cluster_count_nonnoise"])
            for value in partition_values
        ),
        "partition_median_pair_recall_all_family_pairs": median(
            float(value["pair_recall_all_family_pairs"]) for value in partition_values
        ),
        "partition_median_alternate_noise_fraction": median(
            float(value["alternate_noise_fraction"]) for value in partition_values
        ),
        "survey_year_dominant_clusters_with_post_gt_pre": sum(
            value["post_count"] > value["pre_count"] for value in temporal_values
        ),
        "survey_year_dominant_clusters_with_at_least_twofold_or_zero_baseline_growth": twofold_or_zero,
        "zero_baseline_cases_in_previous_count": sum(
            value["pre_count"] == 0 and value["post_count"] > 0
            for value in temporal_values
        ),
        "temporal_summary_scope": (
            "Dominant alternate clusters were selected by production-membership overlap, "
            "not by their years; this is not an independent event scan or rank validation."
        ),
    }
    return results, summary


def describe_named_families(
    rows: list[dict[str, int]],
    neighbors: list[dict[str, int | bool]],
    manifest: dict[str, Any],
    top_by_community: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for group in manifest["groups"]:
        communities = set(int(value) for value in group["communities"])
        n_all = int(group["n_all_members"])
        n_dated = int(group["n_dated_members"])
        partition, dominant = compute_partition_metrics(
            rows,
            communities,
            n_all_members=n_all,
            n_dated_members=n_dated,
        )
        events: list[dict[str, Any]] = []
        if len(communities) == 1:
            community = next(iter(communities))
            descriptor = top_by_community.get(community)
            if descriptor is not None:
                events.append(
                    compute_temporal_event(
                        rows,
                        communities,
                        dominant,
                        {
                            "kind": "systematic_survey_maximizing_year",
                            "year": int(descriptor["survey_event_year"]),
                            "context": "checked production-partition top-20 survey year",
                        },
                    )
                )
            if community in TARGETED_EVENTS:
                events.append(
                    compute_temporal_event(
                        rows,
                        communities,
                        dominant,
                        {"kind": "targeted_literature_year", **TARGETED_EVENTS[community]},
                    )
                )
        result.append(
            {
                "name": group["name"],
                "label_status": group.get("label_status"),
                "source": group.get("source"),
                "local_neighbor_recovery": compute_local_recovery(
                    neighbors, communities, n_all_members=n_all
                ),
                "independent_partition": partition,
                "events": events,
            }
        )
    return result


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def input_record(path: Path) -> dict[str, str]:
    return {"path": relative_or_absolute(path), "sha256": sha256_file(path)}


def fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def fmt_count_window(value: dict[str, Any]) -> str:
    fold = value["fold"]
    suffix = "undefined" if fold is None else f"{fold:.2f}×"
    return f"{value['pre_count']}→{value['post_count']} ({suffix})"


def render_markdown(document: dict[str, Any]) -> str:
    lines = [
        "# Graphlet family-survival audit",
        "",
        "This audit separates local 1-nearest-neighbor recovery from preservation after an independently fitted graphlet partition. Alternative community numbers are arbitrary within each method and are not matched numerically between methods.",
        "",
        f"Analysis version: `{document['analysis_version']}`. The local recovery cohort contains {document['cohorts']['local_neighbor_recovery']['n_entries']:,} entries; the matched dated partition cohort contains {document['cohorts']['independent_partitions']['n_entries']:,} entries.",
        "",
        "## Input provenance",
        "",
        "| Input | SHA-256 |",
        "|---|---|",
    ]
    for name, value in document["inputs"].items():
        lines.append(f"| `{value['path']}` | `{value['sha256']}` |")
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "For a production family $F_c$ and an alternate partition $A_k$, $n_{ck}=|F_c\\cap A_k|$ and the dominant alternate cluster is $k^*=\\arg\\max_{k\\geq0}n_{ck}$. Dominant-cluster recall is $n_{ck^*}/|F_c|$ and purity is $n_{ck^*}/|A_{k^*}|$. Noise (`-1`) remains in the recall denominator and is never treated as one coherent cluster.",
            "",
            "Nonnoise fragmentation uses $q_k=n_{ck}/\\sum_{j\\geq0}n_{cj}$ and $N_{\\mathrm{eff}}=\\exp[-\\sum_k q_k\\ln q_k]$. K90 is the smallest number of largest nonnoise clusters covering at least 90% of nonnoise family members. Pair recall is $\\sum_{k\\geq0}\\binom{n_{ck}}2/\\binom{|F_c|}2$.",
            "",
            "Temporal windows are $(t-10,t]$ and $(t,\\min(t+10,2015)]$. The selected alternate cluster is chosen from membership overlap only; its dates do not enter selection.",
            "",
            "## Focus communities",
            "",
            "| Community | Method | Local 1-NN | Dominant recall | Purity | Noise | Neff | K90 | Selected-cluster temporal window |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for method_name, method in document["methods"].items():
        by_community = {int(row["community"]): row for row in method["top20"]}
        for community in FOCUS_COMMUNITIES:
            row = by_community[community]
            local = row["local_neighbor_recovery"]
            partition = row["independent_partition"]
            event = row["targeted_events"][0] if row["targeted_events"] else row["survey_event"]
            temporal = event["full_dominant_alternate_cluster"]
            lines.append(
                "| {community} | {method} | {local} | {recall} | {purity} | {noise} | {neff:.2f} | {k90} | {kind} {year}: {window} |".format(
                    community=community,
                    method=method_name,
                    local=fmt_pct(local["exact_production_community_recovery"]),
                    recall=fmt_pct(partition["dominant_cluster_recall_all_members"]),
                    purity=fmt_pct(partition["dominant_cluster_purity"]),
                    noise=fmt_pct(partition["alternate_noise_fraction"]),
                    neff=partition["fragmentation_effective_cluster_count_nonnoise"],
                    k90=partition["clusters_for_90_percent_nonnoise"],
                    kind="target" if event["kind"] == "targeted_literature_year" else "survey",
                    year=event["year"],
                    window=fmt_count_window(temporal),
                )
            )
    lines.extend(
        [
            "",
            "## Top-20 summary",
            "",
            "| Method | Local entry-weighted recovery | Median dominant recall | Median purity | Median Neff | Median pair recall | Post > pre | ≥2× or zero-baseline growth |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method_name, method in document["methods"].items():
        summary = method["top20_summary"]
        lines.append(
            "| {method} | {local} | {recall} | {purity} | {neff:.2f} | {pair} | {post}/20 | {two}/20 |".format(
                method=method_name,
                local=fmt_pct(summary["local_entry_weighted_recovery"]),
                recall=fmt_pct(summary["partition_median_dominant_cluster_recall"]),
                purity=fmt_pct(summary["partition_median_dominant_cluster_purity"]),
                neff=summary["partition_median_effective_cluster_count_nonnoise"],
                pair=fmt_pct(summary["partition_median_pair_recall_all_family_pairs"]),
                post=summary["survey_year_dominant_clusters_with_post_gt_pre"],
                two=summary["survey_year_dominant_clusters_with_at_least_twofold_or_zero_baseline_growth"],
            )
        )
    lines.extend(
        [
            "",
            "The strong local-recovery rates show that named production-community members usually retain a same-label nearest neighbor. The substantially smaller dominant-cluster recalls and pair recalls show that independently fitted graphlet partitions often divide the broader production neighborhoods. The temporal counts are an overlap-anchored preservation check, not an independent rediscovery or causal attribution of the events.",
            "",
        ]
    )
    return "\n".join(lines)


def build_document(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "analysis_source": Path(__file__),
        "production_labels": args.production_labels,
        "current_family_manifest": args.family_manifest,
        "top20_checked_descriptors": args.top20_descriptors,
        "crystalnn_nearest_neighbors": args.crystal_neighbors,
        "voronoinn_nearest_neighbors": args.voronoi_neighbors,
        "crystalnn_matched_dated_assignments": args.crystal_assignments,
        "voronoinn_matched_dated_assignments": args.voronoi_assignments,
    }
    _, production = read_production(args.production_labels)
    top20 = read_top20(args.top20_descriptors)
    manifest = read_family_manifest(args.family_manifest)
    methods = {
        "crystalnn_graphlet": {
            "assignments": read_alternate_assignments(args.crystal_assignments, production),
            "neighbors": read_neighbors(args.crystal_neighbors, production),
        },
        "voronoinn_graphlet": {
            "assignments": read_alternate_assignments(args.voronoi_assignments, production),
            "neighbors": read_neighbors(args.voronoi_neighbors, production),
        },
    }
    crystal_assignment_ids = [row["icsd_id"] for row in methods["crystalnn_graphlet"]["assignments"]]
    voronoi_assignment_ids = [row["icsd_id"] for row in methods["voronoinn_graphlet"]["assignments"]]
    require(
        crystal_assignment_ids == voronoi_assignment_ids,
        "CrystalNN and VoronoiNN partition cohorts are not the same ordered IDs",
    )
    crystal_neighbor_ids = [int(row["query_id"]) for row in methods["crystalnn_graphlet"]["neighbors"]]
    voronoi_neighbor_ids = [int(row["query_id"]) for row in methods["voronoinn_graphlet"]["neighbors"]]
    require(
        crystal_neighbor_ids == voronoi_neighbor_ids,
        "CrystalNN and VoronoiNN recovery cohorts are not the same ordered IDs",
    )
    require(
        set(crystal_assignment_ids) <= set(crystal_neighbor_ids),
        "dated partition cohort is not a subset of the local-recovery cohort",
    )
    top_by_community = {int(row["community"]): row for row in top20}
    require(set(FOCUS_COMMUNITIES) <= set(top_by_community), "focus community is absent from top-20 descriptors")
    require(set(TARGETED_EVENTS) <= set(top_by_community), "targeted community is absent from top-20 descriptors")

    output_methods: dict[str, Any] = {}
    for name, inputs in methods.items():
        top_results, top_summary = describe_top20(
            inputs["assignments"], inputs["neighbors"], top20
        )
        output_methods[name] = {
            "alternate_label_scope": "arbitrary within this method; no numeric correspondence is asserted across methods",
            "top20_summary": top_summary,
            "top20": top_results,
            "named_families": describe_named_families(
                inputs["assignments"],
                inputs["neighbors"],
                manifest,
                top_by_community,
            ),
        }

    return {
        "analysis_version": ANALYSIS_VERSION,
        "scope": (
            "Checked production families joined by ICSD ID to exact matched CrystalNN/VoronoiNN "
            "graphlet recovery and dated partition cohorts."
        ),
        "interpretation_limits": [
            "Local 1-NN recovery and independent-partition preservation answer different questions.",
            "Alternative community labels are arbitrary method-local integers and are never matched numerically.",
            "Dominant alternate clusters are selected by membership overlap without using publication years.",
            "Temporal windows test preservation of an anchored signal; they are not an independent scan, rank validation, or causal event attribution.",
            "Family descriptions inherit the chemistry/space-group evidence limits in the checked descriptor and family-manifest inputs.",
        ],
        "inputs": {name: input_record(path) for name, path in paths.items()},
        "cohorts": {
            "local_neighbor_recovery": {
                "n_entries": len(crystal_neighbor_ids),
                "ordered_ids_sha256": sha256_ids(crystal_neighbor_ids),
                "dated_only": False,
            },
            "independent_partitions": {
                "n_entries": len(crystal_assignment_ids),
                "ordered_ids_sha256": sha256_ids(crystal_assignment_ids),
                "dated_only": True,
                "exact_same_order_for_both_methods": True,
            },
        },
        "definitions": {
            "dominant_cluster": "k* = argmax over nonnoise alternate labels k of n_ck = |F_c intersection A_k|; the smallest ICSD ID in the complete alternate cluster breaks exact overlap ties without using the arbitrary numeric label",
            "dominant_cluster_recall_all_members": "n_ck* / |F_c|; alternate noise remains in the denominator",
            "dominant_cluster_purity": "n_ck* / |A_k*| on the complete matched dated cohort",
            "fragmentation_effective_cluster_count_nonnoise": "exp(-sum_k q_k ln(q_k)), q_k = n_ck / sum_{j>=0} n_cj",
            "clusters_for_90_percent_nonnoise": "minimum number of largest nonnoise alternate clusters covering at least 90% of nonnoise family members",
            "pair_recall_all_family_pairs": "sum_{k>=0} choose(n_ck,2) / choose(|F_c|,2); noise pairs are not counted together",
            "pre_window": "event_year - 10 < publication_year <= event_year",
            "post_window": "event_year < publication_year <= min(event_year + 10, 2015)",
            "survey_score": "post_count^2 / pre_count when pre_count > 0, otherwise post_count",
        },
        "event_years": {
            "systematic_survey_by_production_community": {
                str(key): value for key, value in SURVEY_EVENT_YEARS.items()
            },
            "targeted_literature_events": {
                str(key): value for key, value in TARGETED_EVENTS.items()
            },
            "record_end_year": RECORD_END_YEAR,
        },
        "family_manifest_scope": manifest.get("scope"),
        "focus_communities": list(FOCUS_COMMUNITIES),
        "methods": output_methods,
    }


def parse_args() -> argparse.Namespace:
    base = ROOT / "notes" / "feature_repair_2026_09" / "downstream"
    representations = base / "representations"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-labels",
        type=Path,
        default=ROOT / "notes" / "feature_repair_2026_09" / "full_run_results" / "production" / "graph" / "community_assignments.csv",
    )
    parser.add_argument(
        "--family-manifest",
        type=Path,
        default=base / "community_evidence" / "current_family_manifest.json",
    )
    parser.add_argument(
        "--top20-descriptors",
        type=Path,
        default=base / "community_evidence" / "renaissance_top20_checked_descriptors.csv",
    )
    parser.add_argument(
        "--crystal-neighbors",
        type=Path,
        default=representations / "graphlet-crystalnn-shared-voronoi-recovery" / "graphlet_l1" / "nearest_neighbors.csv",
    )
    parser.add_argument(
        "--voronoi-neighbors",
        type=Path,
        default=representations / "graphlet-voronoi-shared-crystalnn-recovery" / "graphlet_l1" / "nearest_neighbors.csv",
    )
    parser.add_argument(
        "--crystal-assignments",
        type=Path,
        default=representations / "graphlet-crystalnn-shared-voronoi-dated-replay" / "graphlet" / "community_assignments.csv",
    )
    parser.add_argument(
        "--voronoi-assignments",
        type=Path,
        default=representations / "graphlet-voronoi-shared-crystalnn-dated-replay" / "graphlet" / "community_assignments.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=representations / "graphlet-family-survival" / "summary.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=representations / "graphlet-family-survival" / "summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.output_json.resolve() != args.output_markdown.resolve(), "output paths must differ")
    document = build_document(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(render_markdown(document), encoding="utf-8")
    print(args.output_json)
    print(args.output_markdown)


if __name__ == "__main__":
    main()
