#!/usr/bin/env python3
"""Freeze chemistry/size-matched comparators for the GNoME bonding pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
DIAGNOSTICS = HERE.parent
RESULTS = DIAGNOSTICS.parent / "representations" / "amd-external-full-map" / "results"
GNOME_SAMPLE = DIAGNOSTICS / "local_property_pilot" / "sample.json"
COMMON_SUPPORT = DIAGNOSTICS / "factor_ablations" / "inputs" / "common_support_ids.json"

SOURCES = ("mattergen", "mp", "jarvis", "alexandria")
SUPPORT_NAMES = {
    "gnome": "GNoME",
    "mattergen": "MatterGen",
    "mp": "MP",
    "jarvis": "JARVIS",
    "alexandria": "Alexandria",
}
SUPPORT_ID_FIELDS = {
    "gnome": "material_id",
    "mattergen": "zip_member",
    "mp": "material_id",
    "jarvis": "material_id",
    "alexandria": "material_id",
}
SEED = 20260909
ELEMENT_RE = re.compile(r"[A-Z][a-z]?")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def elements(formula: str) -> set[str]:
    found = set(ELEMENT_RE.findall(formula))
    if not found:
        raise ValueError(f"Could not parse any elements from formula {formula!r}")
    return found


def chemistry_class(formula: str) -> str:
    """Assign the first matching class in the prespecified priority order."""
    els = elements(formula)
    for label, members in (
        ("O", {"O"}),
        ("FClBrI", {"F", "Cl", "Br", "I"}),
        ("SSeTe", {"S", "Se", "Te"}),
        ("NPAs", {"N", "P", "As"}),
        ("HBC", {"H", "B", "C"}),
    ):
        if els & members:
            return label
    return "other"


def seeded_jitter(source: str, gnome_id: str, candidate_key: str) -> float:
    payload = f"{SEED}|{source}|{gnome_id}|{candidate_key}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


def source_location(source: str, csv_row_number: int, row: dict[str, str]) -> dict[str, object]:
    location: dict[str, object] = {
        "successful_records_csv": str(
            Path("notes/feature_repair_2026_09/downstream/representations")
            / "amd-external-full-map"
            / "results"
            / source
            / "successful_records.csv"
        ),
        # Header is line 1, so the first data record is line 2.
        "csv_line": csv_row_number + 2,
        "common_support_key": row[SUPPORT_ID_FIELDS[source]],
    }
    if source == "mattergen":
        location["zip_member"] = row["zip_member"]
    return location


def output_record(
    source: str,
    row: dict[str, str],
    csv_row_number: int,
    target: dict[str, object],
) -> dict[str, object]:
    formula = row["reduced_formula"]
    n_sites = int(row["n_sites"])
    n_elements = len(elements(formula))
    target_sites = int(target["n_sites"])
    target_elements = int(target["n_elements"])
    ratio = n_sites / target_sites
    return {
        "source": source,
        # Use the exact five-map identifier. For MatterGen this is zip_member;
        # source_record retains the shorter native material_id as well.
        "material_id": row[SUPPORT_ID_FIELDS[source]],
        "matching_id": target["material_id"],
        "matched_gnome_id": target["material_id"],
        "cell": target["cell"],
        "formula": formula,
        "n_sites": n_sites,
        "n_elements": n_elements,
        "chemistry_class": chemistry_class(formula),
        "match_differences": {
            "n_elements_signed": n_elements - target_elements,
            "n_elements_absolute": abs(n_elements - target_elements),
            "n_sites_ratio": ratio,
            "abs_log2_n_sites_ratio": abs(math.log2(ratio)),
        },
        "source_location": source_location(source, csv_row_number, row),
        "source_record": dict(row),
    }


def match_source(
    source: str,
    targets: list[dict[str, object]],
    candidates: list[tuple[int, dict[str, str]]],
) -> tuple[dict[int, int], dict[str, object]]:
    """Maximize feasible pair count, then minimize size differences globally."""
    n_targets = len(targets)
    n_candidates = len(candidates)
    infeasible_cost = 1.0e12
    unmatched_cost = 1.0e6
    # Since every eligible abs(log2 ratio) <= 1, this makes one unit of total
    # element-count difference dominate all possible log-ratio differences.
    element_weight = n_targets + 1.0
    cost = np.full((n_targets, n_candidates + n_targets), infeasible_cost, dtype=float)
    eligible_counts: list[int] = []

    for i, target in enumerate(targets):
        target_class = str(target["chemistry_class"])
        target_elements = int(target["n_elements"])
        target_sites = int(target["n_sites"])
        eligible = 0
        for j, (_, row) in enumerate(candidates):
            formula = row["reduced_formula"]
            candidate_elements = len(elements(formula))
            candidate_sites = int(row["n_sites"])
            element_difference = abs(candidate_elements - target_elements)
            log_ratio_difference = abs(math.log2(candidate_sites / target_sites))
            if (
                chemistry_class(formula) == target_class
                and element_difference <= 1
                and log_ratio_difference <= 1.0 + 1e-12
            ):
                key = row[SUPPORT_ID_FIELDS[source]]
                cost[i, j] = (
                    element_difference * element_weight
                    + log_ratio_difference
                    + seeded_jitter(source, str(target["material_id"]), key) * 1e-9
                )
                eligible += 1
        eligible_counts.append(eligible)
        # Any target can use any dummy. Tiny deterministic offsets settle dummy ties.
        for dummy in range(n_targets):
            cost[i, n_candidates + dummy] = unmatched_cost + (i * n_targets + dummy) * 1e-9

    row_indices, column_indices = linear_sum_assignment(cost)
    assignments = {
        int(i): int(j)
        for i, j in zip(row_indices, column_indices)
        if j < n_candidates and cost[i, j] < unmatched_cost
    }
    unmatched_indices = [i for i in range(n_targets) if i not in assignments]
    report = {
        "common_five_map_candidates": n_candidates,
        "targets_with_at_least_one_individually_eligible_candidate": sum(n > 0 for n in eligible_counts),
        "matched": len(assignments),
        "unmatched": len(unmatched_indices),
        "unmatched_gnome_ids": [targets[i]["material_id"] for i in unmatched_indices],
        "unmatched_due_to_no_individually_eligible_candidate": [
            targets[i]["material_id"] for i in unmatched_indices if eligible_counts[i] == 0
        ],
        "unmatched_due_to_without_replacement_contention": [
            targets[i]["material_id"] for i in unmatched_indices if eligible_counts[i] > 0
        ],
        "eligible_candidate_count_by_gnome_id": {
            str(targets[i]["material_id"]): eligible_counts[i] for i in range(n_targets)
        },
    }
    return assignments, report


def main() -> None:
    original = json.loads(GNOME_SAMPLE.read_text(encoding="utf-8"))
    common = json.loads(COMMON_SUPPORT.read_text(encoding="utf-8"))["populations"]
    success_rows = {source: read_csv(RESULTS / source / "successful_records.csv") for source in SUPPORT_NAMES}
    attempted_rows = {source: read_csv(RESULTS / source / "attempted_cohort.csv") for source in SUPPORT_NAMES}

    gnome_by_id = {row["material_id"]: (i, row) for i, row in enumerate(success_rows["gnome"])}
    targets: list[dict[str, object]] = []
    expected_old_classes = {
        "oxide": "O",
        "halide": "FClBrI",
        "chalcogenide": "SSeTe",
        "pnictide": "NPAs",
        "HBC": "HBC",
        "other": "other",
    }
    for old in original:
        material_id = old["material_id"]
        if material_id not in gnome_by_id:
            raise ValueError(f"Pilot GNoME id absent from successful records: {material_id}")
        _, row = gnome_by_id[material_id]
        derived_class = chemistry_class(row["reduced_formula"])
        if derived_class != expected_old_classes[old["chemistry_stratum"]]:
            raise ValueError(f"Chemistry class disagrees for {material_id}")
        if row["reduced_formula"] != old["reduced_formula"] or int(row["n_sites"]) != int(old["n_sites"]):
            raise ValueError(f"Frozen pilot metadata disagrees with successful records for {material_id}")
        targets.append(
            {
                "material_id": material_id,
                "cell": old["cell"],
                "formula": row["reduced_formula"],
                "n_sites": int(row["n_sites"]),
                "n_elements": len(elements(row["reduced_formula"])),
                "chemistry_class": derived_class,
            }
        )

    records: list[dict[str, object]] = []
    for target in targets:
        csv_index, row = gnome_by_id[str(target["material_id"])]
        records.append(output_record("gnome", row, csv_index, target))

    cohort_reports: dict[str, object] = {}
    support_schema: dict[str, object] = {}
    for source in SOURCES:
        support_ids = set(common[SUPPORT_NAMES[source]]["ids"])
        field = SUPPORT_ID_FIELDS[source]
        candidates = [
            (i, row)
            for i, row in enumerate(success_rows[source])
            if row[field] in support_ids
        ]
        candidates.sort(key=lambda item: item[1][field])
        assignments, cohort_report = match_source(source, targets, candidates)
        cohort_report["successful_records"] = len(success_rows[source])
        cohort_report["attempted_cohort_records"] = len(attempted_rows[source])
        cohort_report["common_support_declared"] = int(common[SUPPORT_NAMES[source]]["n"])
        cohort_reports[source] = cohort_report
        support_schema[source] = {
            "native_material_id_field": "material_id",
            "output_material_id_field": field,
            "five_map_support_id_field": field,
        }
        for target_index, candidate_index in sorted(assignments.items()):
            csv_index, row = candidates[candidate_index]
            records.append(output_record(source, row, csv_index, targets[target_index]))

    if len(records) > 500:
        raise AssertionError(f"Bounded sample exceeds 500 records: {len(records)}")
    if len({(r["source"], r["source_location"]["common_support_key"]) for r in records}) != len(records):
        raise AssertionError("A source record was selected more than once")

    sample_path = HERE / "sample.json"
    sample_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    sample_sha256 = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    source_counts = Counter(str(record["source"]) for record in records)
    report = {
        "status": "frozen",
        "seed": SEED,
        "target_definition": {
            "source": "local_property_pilot/sample.json",
            "gnome_targets": len(targets),
            "all_gnome_targets_preserved": True,
            "original_cell_label_propagated_to_comparators": True,
        },
        "selection_guardrails": {
            "required_five_representation_common_support": True,
            "chemistry_class_priority": ["O", "FClBrI", "SSeTe", "NPAs", "HBC", "other"],
            "exact_chemistry_class": True,
            "maximum_absolute_n_elements_difference": 1,
            "n_sites_ratio_range_inclusive": [0.5, 2.0],
            "without_replacement_within_cohort": True,
            "assignment": "scipy.optimize.linear_sum_assignment with dummy unmatched columns",
            "objective_order": [
                "maximize feasible matched pairs",
                "minimize total absolute n_elements difference",
                "minimize total absolute log2 n_sites ratio",
                "seeded SHA-256 tie break",
            ],
            "selected_on_bonding_outcomes": False,
            "selected_on_map_basin_membership": False,
            "criteria_relaxed": False,
        },
        "identifier_schema": support_schema,
        "cohorts": cohort_reports,
        "output": {
            "records": len(records),
            "maximum_records": 500,
            "source_counts": dict(sorted(source_counts.items())),
            "sample_sha256": sample_sha256,
        },
        "caveats": [
            "Matching controls only broad chemistry class, element count, and crystallographic site count; it does not control stoichiometry, space group, volume, oxidation state, stability, or provenance-specific generation filters.",
            "MatterGen five-map identity is its zip_member path rather than its material_id; both are retained for traceability.",
            "Any unmatched target remains unmatched because the prespecified feasibility constraints and within-cohort uniqueness were not relaxed.",
        ],
    }
    (HERE / "sampling_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
