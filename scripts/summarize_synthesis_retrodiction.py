#!/usr/bin/env python3
"""Regroup saved synthesis-retrodiction scores by chemical composition.

The structural score in ``post_cutoff_accessibility_records.csv`` does not
depend on formula identity.  This postprocessor therefore repairs the
formula-level summaries without recomputing the fixed-map structural scores.
It joins every ICSD identifier to the frozen index and occupancy audit, groups
all records with one normalized nominal-composition identity, and
deterministically selects the lowest ICSD identifier when several records
share the earliest year.  It also evaluates the earlier conditional identity
used by ``analyze_cutoff_trained_retrospective`` so its population can be
cross-checked without making occupancy class part of the grouping unit.

The grouping identity is the element-sorted normalized atomic-fraction vector
rounded to 12 decimal places.  The separate precedent-matching analysis still
uses exact rational keys for fully occupied references and fraction tolerances
for partial-occupancy references; those are matching rules rather than a
reason to give one nominal formula two grouping identities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_synthesis_retrodiction import (  # noqa: E402
    load_icsd_index,
    mean,
    rankdata,
    scatter_plot,
    spearman,
    stdev,
)
from formula_conventions import (  # noqa: E402
    normalized_fraction_key,
    scale_invariant_formula_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-csv", type=Path, required=True)
    parser.add_argument("--icsd-index", type=Path, required=True)
    parser.add_argument("--occupancy-flags", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-year", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null-repeats", type=int, default=1000)
    parser.add_argument(
        "--base-summary",
        type=Path,
        help="Existing structural summary to preserve; defaults to OUTPUT_DIR/synthesis_retrodiction_summary.json",
    )
    parser.add_argument(
        "--expected-first-formulas",
        type=int,
        help="Fail if the normalized nominal-composition population differs",
    )
    parser.add_argument(
        "--expected-cutoff-hybrid-first-formulas",
        type=int,
        help="Fail if the audit population under the cutoff producer's prior hybrid token differs",
    )
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def normalized_formula_identity_token(formula: str) -> str:
    """Serialize element-sorted normalized atomic fractions."""
    if not str(formula).strip():
        return ""
    elements, fractions = normalized_fraction_key(formula)
    return "|".join(elements) + "::" + ",".join(
        f"{value:.12g}" for value in fractions
    )


def legacy_hybrid_formula_identity_token(formula: str, is_partial: bool) -> str:
    """Return the superseded conditional token used by job 3472607."""
    if not str(formula).strip():
        return ""
    if is_partial:
        return normalized_formula_identity_token(formula)
    key = scale_invariant_formula_key(formula)
    total = sum(amount for _, amount in key)
    elements = tuple(element for element, _ in key)
    fractions = tuple(
        float(value)
        for value in np.round([amount / total for _, amount in key], 12)
    )
    return "|".join(elements) + "::" + ",".join(
        f"{value:.12g}" for value in fractions
    )


def nominal_formula_identity_token(formula: str) -> str:
    """Return the occupancy-independent nominal-composition identity."""
    return normalized_formula_identity_token(formula)


def read_scored(path: Path) -> list[dict[str, object]]:
    required = {
        "cif_id",
        "reduced_formula",
        "year",
        "assigned_community",
        "nearest_centroid_distance",
        "is_in_basin",
        "A_i",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} misses columns {sorted(missing)}")
        rows: list[dict[str, object]] = []
        for raw in reader:
            rows.append(
                {
                    "cif_id": int(raw["cif_id"]),
                    "reduced_formula": raw["reduced_formula"],
                    "year": int(raw["year"]),
                    "assigned_community": int(raw["assigned_community"]),
                    "nearest_centroid_distance": float(raw["nearest_centroid_distance"]),
                    "is_in_basin": int(raw["is_in_basin"]),
                    "A_i": float(raw["A_i"]),
                }
            )
    identifiers = [int(row["cif_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path} contains duplicate ICSD identifiers")
    return rows


def read_occupancy_flags(path: Path) -> dict[int, bool]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"icsd_id", "has_partial_occupancy"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} misses columns {sorted(missing)}")
        result: dict[int, bool] = {}
        for row in reader:
            identifier = int(row["icsd_id"])
            value = row["has_partial_occupancy"].strip()
            if value not in {"0", "1"}:
                raise ValueError(f"Invalid occupancy flag for ICSD {identifier}: {value!r}")
            if identifier in result:
                raise ValueError(f"Duplicate occupancy flag for ICSD {identifier}")
            result[identifier] = value == "1"
    return result


def attach_formula_identities(
    scored_rows: Iterable[dict[str, object]],
    index: dict[int, dict[str, object]],
    occupancy: dict[int, bool],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Join authoritative formula/occupancy metadata and compute identities."""
    joined: list[dict[str, object]] = []
    n_display_disagreements = 0
    n_saved_formula_blank = 0
    for source in scored_rows:
        identifier = int(source["cif_id"])
        if identifier not in index:
            raise ValueError(f"ICSD {identifier} is absent from the index")
        if identifier not in occupancy:
            raise ValueError(f"ICSD {identifier} is absent from the occupancy audit")
        formula = str(index[identifier].get("reduced_formula") or "")
        if not formula:
            raise ValueError(f"ICSD {identifier} has no parseable indexed formula")
        saved_formula = str(source.get("reduced_formula") or "").strip()
        if not saved_formula:
            n_saved_formula_blank += 1
        elif saved_formula != formula:
            n_display_disagreements += 1
        is_partial = bool(occupancy[identifier])
        identity = nominal_formula_identity_token(formula)
        cutoff_hybrid_identity = legacy_hybrid_formula_identity_token(
            formula, is_partial
        )
        if not identity:
            raise ValueError(f"ICSD {identifier} produced an empty formula identity")
        row = dict(source)
        row["reduced_formula"] = formula
        row["formula_identity"] = identity
        row["cutoff_hybrid_formula_identity"] = cutoff_hybrid_identity
        row["has_partial_occupancy"] = int(is_partial)
        joined.append(row)
    return joined, {
        "n_saved_formula_blank_recovered_from_index": n_saved_formula_blank,
        "n_saved_vs_index_display_formula_disagreements": n_display_disagreements,
    }


def identity_comparison_audit(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    """Quantify differences from the conditional full/partial token."""
    nominal_to_hybrid: dict[str, set[str]] = {}
    nominal_occupancy: dict[str, set[int]] = {}
    nominal_counts: dict[str, int] = {}
    hybrid_to_nominal: dict[str, set[str]] = {}
    hybrid_counts: dict[str, int] = {}
    for row in rows:
        nominal = str(row["formula_identity"])
        hybrid = str(row["cutoff_hybrid_formula_identity"])
        nominal_to_hybrid.setdefault(nominal, set()).add(hybrid)
        nominal_occupancy.setdefault(nominal, set()).add(
            int(row["has_partial_occupancy"])
        )
        nominal_counts[nominal] = nominal_counts.get(nominal, 0) + 1
        hybrid_to_nominal.setdefault(hybrid, set()).add(nominal)
        hybrid_counts[hybrid] = hybrid_counts.get(hybrid, 0) + 1
    cross_occupancy_splits = [
        nominal
        for nominal, hybrids in nominal_to_hybrid.items()
        if len(hybrids) > 1 and len(nominal_occupancy[nominal]) > 1
    ]
    hybrid_merges = [
        hybrid for hybrid, nominals in hybrid_to_nominal.items() if len(nominals) > 1
    ]
    occupancy_strata = {"full_only": 0, "partial_only": 0, "mixed": 0}
    for values in nominal_occupancy.values():
        if values == {0}:
            occupancy_strata["full_only"] += 1
        elif values == {1}:
            occupancy_strata["partial_only"] += 1
        else:
            occupancy_strata["mixed"] += 1
    return {
        "n_normalized_nominal_identities": len(nominal_to_hybrid),
        "n_cutoff_hybrid_identities": len(hybrid_to_nominal),
        "normalized_minus_cutoff_hybrid": len(nominal_to_hybrid)
        - len(hybrid_to_nominal),
        "n_nominal_identities_split_by_hybrid_across_occupancy": len(
            cross_occupancy_splits
        ),
        "n_records_in_cross_occupancy_split_identities": sum(
            nominal_counts[nominal] for nominal in cross_occupancy_splits
        ),
        "n_hybrid_identities_merging_distinct_nominal_compositions": len(
            hybrid_merges
        ),
        "n_records_in_hybrid_merge_identities": sum(
            hybrid_counts[hybrid] for hybrid in hybrid_merges
        ),
        "n_full_only_nominal_identities": occupancy_strata["full_only"],
        "n_partial_only_nominal_identities": occupancy_strata["partial_only"],
        "n_mixed_occupancy_nominal_identities": occupancy_strata["mixed"],
    }


def earliest_year_mean_sensitivity(
    rows: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Average tied earliest-year records rather than selecting one identifier."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["formula_identity"]), []).append(row)
    years: list[float] = []
    scores: list[float] = []
    basin_fractions: list[float] = []
    for members in grouped.values():
        first_year = min(int(row["year"]) for row in members)
        earliest = [row for row in members if int(row["year"]) == first_year]
        years.append(float(first_year))
        scores.append(mean([float(row["A_i"]) for row in earliest]))
        basin_fractions.append(
            mean([float(row["is_in_basin"]) for row in earliest])
        )
    return {
        "n_formula_identities": len(grouped),
        "spearman_mean_Ai_vs_first_report_year": spearman(scores, years),
        "spearman_mean_in_basin_vs_first_report_year": spearman(
            basin_fractions, years
        ),
        "rule": "mean across every record tied in the earliest represented year",
    }


def group_first_reports(
    rows: Iterable[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select earliest year and then lowest ICSD identifier per identity."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["formula_identity"]), []).append(row)
    selected: list[dict[str, object]] = []
    n_multiple_earliest = 0
    n_mixed_basin = 0
    for identity, members in grouped.items():
        first_year = min(int(row["year"]) for row in members)
        earliest = [row for row in members if int(row["year"]) == first_year]
        if len(earliest) > 1:
            n_multiple_earliest += 1
        if len({int(row["is_in_basin"]) for row in earliest}) > 1:
            n_mixed_basin += 1
        chosen = min(earliest, key=lambda row: int(row["cif_id"]))
        selected.append(chosen)
    selected.sort(key=lambda row: (int(row["year"]), int(row["cif_id"])))
    return selected, {
        "n_formula_identities": len(grouped),
        "n_with_multiple_entries_in_earliest_year": n_multiple_earliest,
        "n_with_mixed_basin_status_in_earliest_year": n_mixed_basin,
        "deterministic_tie_rule": "lowest ICSD identifier",
    }


def correlation_nulls(
    first_reports: list[dict[str, object]], seed: int, repeats: int
) -> dict[str, object]:
    """Compute observed Spearman correlations and a shared year-shuffle null."""
    a_values = [float(row["A_i"]) for row in first_reports]
    basin_values = [float(row["is_in_basin"]) for row in first_reports]
    years = [float(row["year"]) for row in first_reports]
    rho_a = spearman(a_values, years)
    rho_basin = spearman(basin_values, years)

    # Spearman correlation is Pearson correlation of ranks.  Shuffling raw
    # years and reranking them is exactly equivalent to applying the same
    # permutation to their precomputed ranks, including tied years.
    a_rank = np.asarray(rankdata(a_values), dtype=float)
    basin_rank = np.asarray(rankdata(basin_values), dtype=float)
    year_rank = np.asarray(rankdata(years), dtype=float)

    def centered(values: np.ndarray) -> tuple[np.ndarray, float]:
        shifted = values - values.mean()
        return shifted, float(np.linalg.norm(shifted))

    a_centered, a_norm = centered(a_rank)
    basin_centered, basin_norm = centered(basin_rank)
    year_centered, year_norm = centered(year_rank)
    rng = random.Random(seed)
    indices = list(range(len(first_reports)))
    null_a: list[float] = []
    null_basin: list[float] = []
    for _ in range(repeats):
        rng.shuffle(indices)
        permuted = year_centered[np.asarray(indices, dtype=np.int64)]
        if a_norm and year_norm:
            null_a.append(float(np.dot(a_centered, permuted) / (a_norm * year_norm)))
        if basin_norm and year_norm:
            null_basin.append(
                float(np.dot(basin_centered, permuted) / (basin_norm * year_norm))
            )
        # The legacy producer starts each shuffle from the unpermuted year
        # sequence, rather than repeatedly shuffling the preceding result.
        indices[:] = range(len(first_reports))

    mean_a = mean(null_a) if null_a else None
    std_a = stdev(null_a) if len(null_a) > 1 else None
    mean_basin = mean(null_basin) if null_basin else None
    std_basin = stdev(null_basin) if len(null_basin) > 1 else None
    return {
        "spearman_Ai_vs_first_report_year": rho_a,
        "spearman_in_basin_vs_first_report_year": rho_basin,
        "null_mean_rho": mean_a,
        "null_std_rho": std_a,
        "null_abs_rho_gt_3sigma": None
        if rho_a is None or std_a is None
        else abs(float(rho_a) - float(mean_a)) > 3.0 * float(std_a),
        "null_mean_rho_in_basin": mean_basin,
        "null_std_rho_in_basin": std_basin,
        "null_abs_rho_in_basin_gt_3sigma": None
        if rho_basin is None or std_basin is None
        else abs(float(rho_basin) - float(mean_basin)) > 3.0 * float(std_basin),
        "null_repeats": repeats,
        "null_seed": seed,
        "null_method": "publication-year permutation; shared permutation for score and basin correlations",
    }


def repeated_formula_comparisons(
    rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Compare earliest-year and later records without asserting polymorphism."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["formula_identity"]), []).append(row)
    result: list[dict[str, object]] = []
    for identity, members in grouped.items():
        years = {int(row["year"]) for row in members}
        if len(years) < 2:
            continue
        first_year = min(years)
        first = [float(row["A_i"]) for row in members if int(row["year"]) == first_year]
        later = [float(row["A_i"]) for row in members if int(row["year"]) > first_year]
        if not later:
            continue
        representative = min(
            (row for row in members if int(row["year"]) == first_year),
            key=lambda row: int(row["cif_id"]),
        )
        first_mean = mean(first)
        later_mean = mean(later)
        result.append(
            {
                "reduced_formula": representative["reduced_formula"],
                "formula_identity": identity,
                "first_year": first_year,
                "first_A_i_mean": first_mean,
                "later_A_i_mean": later_mean,
                "earliest_mean_Ai_lower_than_later": first_mean < later_mean,
                "n_records": len(members),
                "n_publication_years": len(years),
            }
        )
    result.sort(key=lambda row: (int(row["first_year"]), str(row["formula_identity"])))
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_summary_path = args.base_summary or (
        args.output_dir / "synthesis_retrodiction_summary.json"
    )
    base_summary = json.loads(base_summary_path.read_text()) if base_summary_path.exists() else {}
    prior_base_record = (
        base_summary.get("regrouping_provenance", {})
        .get("inputs", {})
        .get("base_summary")
    )
    base_summary_record = prior_base_record or {
        "path": str(base_summary_path),
        "sha256_before_regrouping": digest(base_summary_path)
        if base_summary_path.exists()
        else None,
    }

    scored = read_scored(args.scored_csv)
    index = load_icsd_index(args.icsd_index)
    occupancy = read_occupancy_flags(args.occupancy_flags)
    joined, join_audit = attach_formula_identities(scored, index, occupancy)
    identity_audit = identity_comparison_audit(joined)
    unexpected_years = [row for row in joined if int(row["year"]) <= args.holdout_year]
    if unexpected_years:
        raise ValueError(
            f"{len(unexpected_years)} scored rows are not after cutoff {args.holdout_year}"
        )

    first_reports, tie_audit = group_first_reports(joined)
    if (
        args.expected_first_formulas is not None
        and len(first_reports) != args.expected_first_formulas
    ):
        raise ValueError(
            f"Expected {args.expected_first_formulas:,} first-formula rows, "
            f"found {len(first_reports):,}"
        )
    if (
        args.expected_cutoff_hybrid_first_formulas is not None
        and identity_audit["n_cutoff_hybrid_identities"]
        != args.expected_cutoff_hybrid_first_formulas
    ):
        raise ValueError(
            f"Expected {args.expected_cutoff_hybrid_first_formulas:,} identities "
            "under the cutoff producer's hybrid token, found "
            f"{identity_audit['n_cutoff_hybrid_identities']:,}"
        )
    correlations = correlation_nulls(first_reports, args.seed, args.null_repeats)
    tie_mean_sensitivity = earliest_year_mean_sensitivity(joined)
    repeated_rows = repeated_formula_comparisons(joined)

    producer = Path(__file__).resolve()
    conventions = ROOT / "scripts" / "formula_conventions.py"
    index_loader = ROOT / "scripts" / "analyze_synthesis_retrodiction.py"
    cutoff_producer = ROOT / "scripts" / "analyze_cutoff_trained_retrospective.py"
    summary = dict(base_summary)
    # Remove the legacy labels: multiple reports of one nominal formula are
    # not, by themselves, evidence for distinct polymorphs.
    summary.pop("n_temporal_polymorph_formulas", None)
    summary.pop("polymorph_first_easier_fraction", None)
    summary.update(
        {
            "holdout_year": int(args.holdout_year),
            "n_post_cutoff_icsd": len(joined),
            "n_post_cutoff_parseable_formula": len(joined),
            "n_first_report_formulas": len(first_reports),
            **correlations,
            "post_cutoff_in_basin_fraction": mean(
                [float(row["is_in_basin"]) for row in joined]
            ),
            "first_report_in_basin_fraction": mean(
                [float(row["is_in_basin"]) for row in first_reports]
            ),
            "n_repeated_formula_identities_with_multiple_years": len(
                repeated_rows
            ),
            "repeated_formula_earliest_mean_Ai_lower_fraction": mean(
                [
                    float(bool(row["earliest_mean_Ai_lower_than_later"]))
                    for row in repeated_rows
                ]
            )
            if repeated_rows
            else None,
            "formula_identity": {
                "rule": (
                    "all heldout records use element-sorted normalized atomic fractions "
                    "rounded to 12 decimal places; occupancy is retained as metadata and "
                    "does not change the nominal-composition grouping unit"
                ),
                "rule_name": "normalized_fraction_all_records_decimals_12",
                "formula_source": "ICSD index name parsed by load_icsd_index; saved display formula is not used as an identifier",
                "occupancy_source": str(args.occupancy_flags),
                "tie_audit": tie_audit,
                "comparison_to_cutoff_hybrid_token": identity_audit,
                **join_audit,
            },
            "earliest_year_tie_mean_sensitivity": tie_mean_sensitivity,
            "regrouping_provenance": {
                "schema_version": 1,
                "purpose": "occupancy-aware scale-invariant regrouping of saved structural scores",
                "argv": [str(value) for value in sys.argv],
                "hostname": platform.node(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "inputs": {
                    "scored_csv": {"path": str(args.scored_csv), "sha256": digest(args.scored_csv)},
                    "icsd_index": {"path": str(args.icsd_index), "sha256": digest(args.icsd_index)},
                    "occupancy_flags": {"path": str(args.occupancy_flags), "sha256": digest(args.occupancy_flags)},
                    "base_summary": base_summary_record,
                },
                "code": {
                    "producer": {"path": str(producer), "sha256": digest(producer)},
                    "formula_conventions": {"path": str(conventions), "sha256": digest(conventions)},
                    "index_loader": {"path": str(index_loader), "sha256": digest(index_loader)},
                    "current_cutoff_producer": {
                        "path": str(cutoff_producer),
                        "sha256": digest(cutoff_producer),
                    },
                },
            },
        }
    )

    first_path = args.output_dir / "first_report_formulas.csv"
    repeated_path = args.output_dir / "repeated_formula_results.csv"
    legacy_repeated_path = args.output_dir / "polymorph_sibling_results.csv"
    summary_path = args.output_dir / "synthesis_retrodiction_summary.json"
    scatter_path = args.output_dir / "retrodiction_first_report_scatter.png"
    first_fields = [
        "cif_id",
        "reduced_formula",
        "year",
        "assigned_community",
        "nearest_centroid_distance",
        "is_in_basin",
        "A_i",
        "formula_identity",
        "cutoff_hybrid_formula_identity",
        "has_partial_occupancy",
    ]
    repeated_fields = [
        "reduced_formula",
        "formula_identity",
        "first_year",
        "first_A_i_mean",
        "later_A_i_mean",
        "earliest_mean_Ai_lower_than_later",
        "n_records",
        "n_publication_years",
    ]
    atomic_csv(first_path, first_reports, first_fields)
    atomic_csv(repeated_path, repeated_rows, repeated_fields)
    # Preserve the old path for archive compatibility, but give it the same
    # corrected schema and identify the canonical name in the manifest.
    atomic_csv(legacy_repeated_path, repeated_rows, repeated_fields)
    atomic_text(summary_path, json.dumps(summary, indent=2, allow_nan=False) + "\n")
    scatter_plot(
        first_reports,
        correlations["spearman_Ai_vs_first_report_year"],
        scatter_path,
        args.holdout_year,
    )
    manifest = {
        "schema_version": 1,
        "holdout_year": args.holdout_year,
        "n_first_report_formulas": len(first_reports),
        "outputs": {
            path.name: {"path": str(path), "sha256": digest(path)}
            for path in (
                summary_path,
                first_path,
                repeated_path,
                legacy_repeated_path,
                scatter_path,
            )
        },
        "canonical_repeated_formula_output": repeated_path.name,
        "legacy_compatibility_alias": legacy_repeated_path.name,
        "crosscheck": {
            "expected_first_formulas": args.expected_first_formulas,
            "normalized_identity_matched": args.expected_first_formulas is None
            or len(first_reports) == args.expected_first_formulas,
            "expected_cutoff_hybrid_first_formulas": args.expected_cutoff_hybrid_first_formulas,
            "observed_cutoff_hybrid_first_formulas": identity_audit[
                "n_cutoff_hybrid_identities"
            ],
            "cutoff_hybrid_identity_matched": args.expected_cutoff_hybrid_first_formulas
            is None
            or identity_audit["n_cutoff_hybrid_identities"]
            == args.expected_cutoff_hybrid_first_formulas,
        },
    }
    atomic_text(
        args.output_dir / "formula_regrouping_manifest.json",
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
    )
    print(
        f"T={args.holdout_year}: entries={len(joined):,}; "
        f"normalized formula identities={len(first_reports):,}; "
        f"cutoff hybrid identities={identity_audit['n_cutoff_hybrid_identities']:,}; "
        f"rho(A, year)={correlations['spearman_Ai_vs_first_report_year']:.6f}; "
        f"rho(basin, year)={correlations['spearman_in_basin_vs_first_report_year']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
