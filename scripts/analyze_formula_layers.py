#!/usr/bin/env python3
"""Regenerate the layered composition-precedent analysis.

The analysis keeps four composition questions separate: exact normalized
element ratios, element-set overlap, anonymous stoichiometry, and continuous
same-element ratio proximity.  Fully occupied ICSD records use the tested
normalized-first integer key in :mod:`formula_conventions`; partial-occupancy
records use normalized atomic fractions with a declared tolerance.

The producer accepts every data path on the command line.  It does not read
licensed CIFs and does not contain machine-specific archive paths.  A single
run writes the scored-cohort layers, occupancy-stratified and tolerance
sensitivities, Figure-4 quadrant cells, full-release GNoME layers, and the
small exact-formula candidate gate used by the separate disorder-aware
structure matcher.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy
from scipy.spatial import cKDTree

from formula_conventions import (
    normalized_fraction_vector,
    scale_invariant_formula_key,
)


Key = tuple[tuple[str, int], ...]
Elements = tuple[str, ...]


@dataclass(frozen=True)
class FormulaRow:
    identifier: str
    formula: str
    key: Key
    elements: Elements
    fractions: tuple[float, ...]
    anonymous: tuple[int, ...]
    year: int | None = None
    partial: bool | None = None
    in_basin: bool | None = None


@dataclass
class Reference:
    name: str
    rows: list[FormulaRow]
    full_keys: set[Key]
    all_keys: set[Key]
    element_sets: set[Elements]
    anonymous_keys: set[tuple[int, ...]]
    partial_trees: dict[Elements, cKDTree]
    all_trees: dict[Elements, cKDTree]
    full_rows_by_key: dict[Key, list[FormulaRow]]
    partial_rows_by_elements: dict[Elements, list[FormulaRow]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icsd-index", type=Path, required=True)
    parser.add_argument("--occupancy-flags", type=Path, required=True)
    parser.add_argument(
        "--first-reports",
        type=Path,
        required=True,
        help="Analyzed post-cutoff first-report CSV (cif_id, reduced_formula, year)",
    )
    parser.add_argument(
        "--external",
        nargs=3,
        action="append",
        metavar=("NAME", "ATTEMPTED_CSV", "SCORED_CSV"),
        required=True,
        help="Repeat once per external cohort. ATTEMPTED and SCORED may be the same file.",
    )
    parser.add_argument(
        "--full-gnome",
        type=Path,
        required=True,
        help="Complete public GNoME stable-materials summary CSV",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--post-year-exclusive", type=int, default=1980)
    parser.add_argument("--max-year-inclusive", type=int, default=2015)
    parser.add_argument("--primary-partial-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--partial-tolerances",
        type=float,
        nargs="+",
        default=[1e-8, 1e-6, 1e-4, 1e-3, 5e-3, 1e-2],
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical_identifier(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(int(text))
    except ValueError:
        return text


def truth(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return True
    if text in {"0", "false", "f", "no"}:
        return False
    raise ValueError(f"Invalid Boolean value {value!r}")


def optional_year(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def first_value(row: dict[str, str], names: Iterable[str]) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def make_formula_row(
    identifier: object,
    formula: object,
    *,
    year: int | None = None,
    partial: bool | None = None,
    in_basin: bool | None = None,
) -> FormulaRow:
    formula_text = str(formula or "").strip()
    if not formula_text:
        raise ValueError("missing formula")
    key = scale_invariant_formula_key(formula_text)
    elements, fractions = normalized_fraction_vector(formula_text)
    anonymous = tuple(sorted((amount for _, amount in key), reverse=True))
    return FormulaRow(
        identifier=canonical_identifier(identifier),
        formula=formula_text,
        key=key,
        elements=elements,
        fractions=tuple(float(value) for value in fractions),
        anonymous=anonymous,
        year=year,
        partial=partial,
        in_basin=in_basin,
    )


def load_occupancy(path: Path) -> dict[str, bool]:
    result: dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identifier = canonical_identifier(first_value(row, ("icsd_id", "cif_id", "cif_names")))
            if not identifier:
                continue
            value = truth(first_value(row, ("has_partial_occupancy", "partial")))
            if identifier in result and result[identifier] != value:
                raise ValueError(f"Conflicting occupancy flags for ICSD {identifier}")
            result[identifier] = value
    return result


def load_icsd(path: Path, occupancy: dict[str, bool]) -> tuple[list[FormulaRow], list[dict[str, str]]]:
    rows: list[FormulaRow] = []
    failures: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for line, raw in enumerate(csv.DictReader(handle), start=2):
            identifier = canonical_identifier(first_value(raw, ("icsd_id", "cif_names", "cif_id")))
            formula = first_value(raw, ("reduced_formula", "name", "formula"))
            year_text = first_value(raw, ("publication_year", "year"))
            try:
                year = optional_year(year_text)
                partial = occupancy[identifier]
                rows.append(make_formula_row(identifier, formula, year=year, partial=partial))
            except (KeyError, TypeError, ValueError) as error:
                failures.append(
                    {"line": str(line), "identifier": identifier, "formula": formula, "error": repr(error)}
                )
    return rows, failures


def load_first_reports(path: Path, occupancy: dict[str, bool]) -> tuple[list[FormulaRow], list[dict[str, str]]]:
    rows: list[FormulaRow] = []
    failures: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for line, raw in enumerate(csv.DictReader(handle), start=2):
            identifier = canonical_identifier(first_value(raw, ("icsd_id", "cif_id", "cif_names")))
            formula = first_value(raw, ("reduced_formula", "formula", "name"))
            year_text = first_value(raw, ("year", "publication_year"))
            try:
                year = optional_year(year_text)
                partial = occupancy[identifier]
                rows.append(make_formula_row(identifier, formula, year=year, partial=partial))
            except (KeyError, TypeError, ValueError) as error:
                failures.append(
                    {"line": str(line), "identifier": identifier, "formula": formula, "error": repr(error)}
                )
    return rows, failures


def load_external(path: Path, *, scored: bool) -> tuple[list[FormulaRow], list[dict[str, str]]]:
    rows: list[FormulaRow] = []
    failures: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for line, raw in enumerate(csv.DictReader(handle), start=2):
            identifier = first_value(raw, ("material_id", "id", "identifier"))
            formula = first_value(raw, ("reduced_formula", "formula", "name"))
            try:
                inside = truth(first_value(raw, ("in_basin",))) if scored else None
                rows.append(make_formula_row(identifier, formula, in_basin=inside))
            except (TypeError, ValueError) as error:
                failures.append(
                    {"line": str(line), "identifier": identifier, "formula": formula, "error": repr(error)}
                )
    return rows, failures


def load_full_gnome(path: Path) -> tuple[list[FormulaRow], list[dict[str, str]]]:
    rows: list[FormulaRow] = []
    failures: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for line, raw in enumerate(csv.DictReader(handle), start=2):
            identifier = first_value(raw, ("material_id", "MaterialId", "id"))
            formula = first_value(raw, ("reduced_formula", "Reduced Formula", "formula"))
            try:
                rows.append(make_formula_row(identifier, formula))
            except (TypeError, ValueError) as error:
                failures.append(
                    {"line": str(line), "identifier": identifier, "formula": formula, "error": repr(error)}
                )
            if line % 100_000 == 0:
                print(f"  parsed {line - 1:,} full-release GNoME rows", flush=True)
    return rows, failures


def unique_fraction_rows(rows: Iterable[FormulaRow]) -> tuple[np.ndarray, list[FormulaRow]]:
    by_fraction: dict[tuple[float, ...], FormulaRow] = {}
    for row in rows:
        token = tuple(round(value, 14) for value in row.fractions)
        by_fraction.setdefault(token, row)
    ordered = [by_fraction[token] for token in sorted(by_fraction)]
    array = np.asarray([row.fractions for row in ordered], dtype=float)
    return array, ordered


def build_reference(name: str, rows: list[FormulaRow]) -> Reference:
    full = [row for row in rows if row.partial is False]
    partial = [row for row in rows if row.partial is True]
    all_by_elements: dict[Elements, list[FormulaRow]] = defaultdict(list)
    partial_by_elements: dict[Elements, list[FormulaRow]] = defaultdict(list)
    full_rows_by_key: dict[Key, list[FormulaRow]] = defaultdict(list)
    for row in rows:
        all_by_elements[row.elements].append(row)
        if row.partial:
            partial_by_elements[row.elements].append(row)
        else:
            full_rows_by_key[row.key].append(row)
    partial_trees = {
        elements: cKDTree(unique_fraction_rows(group)[0])
        for elements, group in partial_by_elements.items()
    }
    all_trees = {
        elements: cKDTree(unique_fraction_rows(group)[0])
        for elements, group in all_by_elements.items()
    }
    return Reference(
        name=name,
        rows=rows,
        full_keys={row.key for row in full},
        all_keys={row.key for row in rows},
        element_sets=set(all_by_elements),
        anonymous_keys={row.anonymous for row in rows},
        partial_trees=partial_trees,
        all_trees=all_trees,
        full_rows_by_key=dict(full_rows_by_key),
        partial_rows_by_elements=dict(partial_by_elements),
    )


def nearest_distances(rows: list[FormulaRow], trees: dict[Elements, cKDTree]) -> np.ndarray:
    result = np.full(len(rows), np.inf, dtype=float)
    by_elements: dict[Elements, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.elements in trees:
            by_elements[row.elements].append(index)
    for elements, indices in by_elements.items():
        queries = np.asarray([rows[index].fractions for index in indices], dtype=float)
        distance, _ = trees[elements].query(queries, k=1, p=np.inf, workers=-1)
        result[np.asarray(indices, dtype=int)] = np.asarray(distance, dtype=float)
    return result


def analyze_rows(
    rows: list[FormulaRow],
    reference: Reference,
    tolerances: list[float],
    primary_tolerance: float,
    *,
    return_flags: bool = False,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    n = len(rows)
    integer = np.fromiter((row.key in reference.all_keys for row in rows), dtype=bool, count=n)
    full = np.fromiter((row.key in reference.full_keys for row in rows), dtype=bool, count=n)
    element = np.fromiter((row.elements in reference.element_sets for row in rows), dtype=bool, count=n)
    anonymous = np.fromiter(
        (row.anonymous in reference.anonymous_keys for row in rows), dtype=bool, count=n
    )
    partial_distance = nearest_distances(rows, reference.partial_trees)
    all_distance = nearest_distances(rows, reference.all_trees)
    partial = partial_distance <= primary_tolerance + 1e-15
    hybrid = full | partial

    def metric(mask: np.ndarray) -> tuple[int, float]:
        count = int(mask.sum())
        return count, count / n if n else 0.0

    integer_count, integer_rate = metric(integer)
    hybrid_count, hybrid_rate = metric(hybrid)
    element_count, element_rate = metric(element)
    anonymous_count, anonymous_rate = metric(anonymous)
    origin = Counter(
        "both" if full_match and partial_match else
        "full_only" if full_match else
        "partial_only" if partial_match else
        "neither"
        for full_match, partial_match in zip(full, partial)
    )
    output: dict[str, object] = {
        "n": n,
        "integer_formula_exact_count": integer_count,
        "integer_formula_exact_rate": integer_rate,
        "hybrid_formula_precedent_count": hybrid_count,
        "hybrid_formula_precedent_rate": hybrid_rate,
        "element_set_count": element_count,
        "element_set_rate": element_rate,
        "anonymous_integer_stoichiometry_count": anonymous_count,
        "anonymous_integer_stoichiometry_rate": anonymous_rate,
        "primary_match_origin": dict(origin),
        "same_element_normalized_ratio_tolerance": {},
        "hybrid_formula_precedent_tolerance": {},
    }
    for tolerance in tolerances:
        ratio_mask = all_distance <= tolerance + 1e-15
        tolerance_hybrid = full | (partial_distance <= tolerance + 1e-15)
        ratio_count, ratio_rate = metric(ratio_mask)
        tolerance_count, tolerance_rate = metric(tolerance_hybrid)
        output["same_element_normalized_ratio_tolerance"][f"{tolerance:g}"] = {
            "count": ratio_count,
            "rate": ratio_rate,
        }
        output["hybrid_formula_precedent_tolerance"][f"{tolerance:g}"] = {
            "count": tolerance_count,
            "rate": tolerance_rate,
        }
    flags = None
    if return_flags:
        flags = {
            "integer": integer,
            "full": full,
            "partial": partial,
            "hybrid": hybrid,
            "element": element,
            "anonymous": anonymous,
            "partial_distance": partial_distance,
            "all_distance": all_distance,
        }
    return output, flags


def reference_summary(reference: Reference) -> dict[str, object]:
    full_rows = [row for row in reference.rows if row.partial is False]
    partial_rows = [row for row in reference.rows if row.partial is True]
    full_keys = {row.key for row in full_rows}
    partial_keys = {row.key for row in partial_rows}
    return {
        "records": len(reference.rows),
        "fully_occupied_records": len(full_rows),
        "partial_occupancy_records": len(partial_rows),
        "unique_normalized_integer_formulas": len(reference.all_keys),
        "unique_fully_occupied_integer_formulas": len(reference.full_keys),
        "unique_partial_occupancy_integer_formulas": len(partial_keys),
        "integer_formula_keys_present_in_both_occupancy_classes": len(
            full_keys & partial_keys
        ),
        "unique_element_sets": len(reference.element_sets),
        "unique_anonymous_stoichiometries": len(reference.anonymous_keys),
        "partial_occupancy_element_sets": len(reference.partial_trees),
        "partial_occupancy_fraction_vectors": int(sum(tree.n for tree in reference.partial_trees.values())),
    }


def quadrant_counts(rows: list[FormulaRow], flags: dict[str, np.ndarray]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for row, match in zip(rows, flags["hybrid"]):
        if row.in_basin is None:
            raise ValueError("Quadrant input lacks in_basin")
        location = "in_basin" if row.in_basin else "frontier"
        precedent = "formula_match" if bool(match) else "no_formula_match"
        result[f"{location}__{precedent}"] += 1
    return dict(result)


def candidate_pairs(
    queries: list[FormulaRow],
    flags: dict[str, np.ndarray],
    reference: Reference,
    tolerance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query, matched in zip(queries, flags["hybrid"]):
        if not matched:
            continue
        seen: set[str] = set()
        for target in reference.full_rows_by_key.get(query.key, []):
            seen.add(target.identifier)
            rows.append(
                {
                    "reference_scope": reference.name,
                    "material_id": query.identifier,
                    "query_formula": query.formula,
                    "icsd_id": target.identifier,
                    "icsd_formula": target.formula,
                    "icsd_year": target.year,
                    "icsd_partial_occupancy": False,
                    "match_kind": "fully_occupied_integer_key",
                    "max_atomic_fraction_difference": 0.0,
                }
            )
        partial_rows = reference.partial_rows_by_elements.get(query.elements, [])
        for target in partial_rows:
            difference = float(
                np.max(np.abs(np.asarray(query.fractions) - np.asarray(target.fractions)))
            )
            if difference <= tolerance + 1e-15 and target.identifier not in seen:
                rows.append(
                    {
                        "reference_scope": reference.name,
                        "material_id": query.identifier,
                        "query_formula": query.formula,
                        "icsd_id": target.identifier,
                        "icsd_formula": target.formula,
                        "icsd_year": target.year,
                        "icsd_partial_occupancy": True,
                        "match_kind": "partial_occupancy_fraction_tolerance",
                        "max_atomic_fraction_difference": difference,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_full_gnome_records(
    path: Path,
    rows: list[FormulaRow],
    flags_by_scope: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = ["material_id", "reduced_formula"]
    for scope in flags_by_scope:
        fields.extend(
            [
                f"{scope}_formula_precedent",
                f"{scope}_element_set",
                f"{scope}_anonymous_stoichiometry",
            ]
        )
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            record: dict[str, object] = {
                "material_id": row.identifier,
                "reduced_formula": row.formula,
            }
            for scope, flags in flags_by_scope.items():
                record[f"{scope}_formula_precedent"] = int(flags["hybrid"][index])
                record[f"{scope}_element_set"] = int(flags["element"][index])
                record[f"{scope}_anonymous_stoichiometry"] = int(flags["anonymous"][index])
            writer.writerow(record)


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)
    tolerances = sorted(set(float(value) for value in args.partial_tolerances))
    primary = float(args.primary_partial_tolerance)
    if primary not in tolerances:
        tolerances.append(primary)
        tolerances.sort()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading occupancy flags and ICSD references", flush=True)
    occupancy = load_occupancy(args.occupancy_flags)
    icsd, icsd_failures = load_icsd(args.icsd_index, occupancy)
    first_reports, first_failures = load_first_reports(args.first_reports, occupancy)
    if icsd_failures or first_failures:
        raise ValueError(
            f"Formula/reference loading failed: ICSD={len(icsd_failures)}, first reports={len(first_failures)}"
        )
    if len(icsd) != len(occupancy):
        raise ValueError(
            f"ICSD/occupancy population mismatch: {len(icsd):,} index rows vs {len(occupancy):,} flags"
        )

    post = args.post_year_exclusive
    maximum = args.max_year_inclusive
    references = {
        "scored_post1980_first_reports": build_reference(
            "scored_post1980_first_reports", first_reports
        ),
        "full_index_post1980": build_reference(
            "full_index_post1980",
            [row for row in icsd if row.year is not None and post < row.year <= maximum],
        ),
        "full_index_dated_all_years": build_reference(
            "full_index_dated_all_years", [row for row in icsd if row.year is not None]
        ),
        "full_index_all_records": build_reference("full_index_all_records", icsd),
    }

    external_inputs: dict[str, dict[str, object]] = {}
    input_paths = {
        "icsd_index": args.icsd_index,
        "occupancy_flags": args.occupancy_flags,
        "first_reports": args.first_reports,
        "full_gnome": args.full_gnome,
    }
    for name, attempted_text, scored_text in args.external:
        attempted_path = Path(attempted_text)
        scored_path = Path(scored_text)
        attempted, attempted_failures = load_external(attempted_path, scored=False)
        scored, scored_failures = load_external(scored_path, scored=True)
        if attempted_failures or scored_failures:
            raise ValueError(
                f"{name} formula parsing failed: attempted={len(attempted_failures)}, scored={len(scored_failures)}"
            )
        external_inputs[name] = {"attempted": attempted, "scored": scored}
        input_paths[f"external_{name}_attempted"] = attempted_path
        input_paths[f"external_{name}_scored"] = scored_path

    scored_support: dict[str, object] = {
        "method": {
            "element_set": "alphabetically sorted element symbols, following OPTIMADE elements semantics",
            "normalized_ratios": "atomic amounts divided by their sum and aligned to the element set, following OPTIMADE elements_ratios semantics",
            "integer_key": "collapse species and isotope labels to elements; normalize total amount to one; apply pymatgen get_integer_formula_and_factor; gcd-reduce and sort by element",
            "anonymous_stoichiometry": "gcd-reduced integer coefficients sorted largest to smallest, following OPTIMADE chemical_formula_anonymous ordering",
            "formula_precedent": "fully occupied reference: exact integer key; partial-occupancy reference: same elements and max normalized-fraction difference within tolerance",
            "primary_partial_tolerance": primary,
        },
        "denominators": {
            name: len(populations["scored"])
            for name, populations in external_inputs.items()
        },
        "references": {},
        "sources": {},
    }
    occupancy_output: dict[str, object] = {
        "method": scored_support["method"],
        "references": {},
        "sources": {},
    }
    hybrid_output: dict[str, object] = {
        "method": scored_support["method"],
        "tolerances": tolerances,
        "sources": {},
    }

    scored_reference_names = (
        "scored_post1980_first_reports",
        "full_index_post1980",
        "full_index_dated_all_years",
    )
    # Retain the established short keys consumed by the figure producer while
    # using explicit names internally and in the full-release artifact.
    scored_output_names = {
        "scored_post1980_first_reports": "scored_post1980",
        "full_index_post1980": "full_index_post1980",
        "full_index_dated_all_years": "full_index_all_year",
    }
    scored_flags: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    for reference_name in scored_reference_names:
        reference = references[reference_name]
        output_name = scored_output_names[reference_name]
        scored_support["references"][output_name] = reference_summary(reference)
        occupancy_output["references"][output_name] = reference_summary(reference)
        scored_support["sources"][output_name] = {}
        occupancy_output["sources"][output_name] = {}
        for source, populations in external_inputs.items():
            result, flags = analyze_rows(
                populations["scored"], reference, tolerances, primary, return_flags=True
            )
            assert flags is not None
            scored_support["sources"][output_name][source] = result
            occupancy_output["sources"][output_name][source] = {
                "n": result["n"],
                "primary_match_origin": result["primary_match_origin"],
            }
            hybrid_output["sources"].setdefault(source, {})[output_name] = {
                "n": result["n"],
                "hybrid_formula_precedent_tolerance": result[
                    "hybrid_formula_precedent_tolerance"
                ],
            }
            scored_flags[reference_name][source] = flags

    primary_name = "scored_post1980_first_reports"
    quadrants: dict[str, object] = {}
    for source, populations in external_inputs.items():
        rows = populations["scored"]
        flags = scored_flags[primary_name][source]
        quadrants[source] = {
            "n": len(rows),
            "scale_invariant_quadrant": quadrant_counts(rows, flags),
            "formula_precedent_rule": scored_support["method"]["formula_precedent"],
            "partial_tolerance": primary,
        }

    dump(args.output_dir / "scored_support_layers.json", scored_support)
    dump(args.output_dir / "occupancy_stratified_formula_layers.json", occupancy_output)
    dump(args.output_dir / "hybrid_ordered_partial_formula_layers.json", hybrid_output)
    dump(args.output_dir / "scale_invariant_quadrants.json", quadrants)

    print("Loading and analyzing the complete GNoME release", flush=True)
    full_gnome, full_gnome_failures = load_full_gnome(args.full_gnome)
    if full_gnome_failures:
        raise ValueError(f"Full GNoME formula parsing failed for {len(full_gnome_failures)} rows")
    full_summary: dict[str, object] = {
        "method": scored_support["method"],
        "source": {"input_records": len(full_gnome), "parse_failures": 0},
        "references": {},
        "results": {},
    }
    full_flags: dict[str, dict[str, np.ndarray]] = {}
    for reference_name, reference in references.items():
        full_summary["references"][reference_name] = reference_summary(reference)
        result, flags = analyze_rows(
            full_gnome, reference, tolerances, primary, return_flags=True
        )
        assert flags is not None
        full_summary["results"][reference_name] = result
        full_flags[reference_name] = flags
        print(
            f"  {reference_name}: formula={result['hybrid_formula_precedent_count']:,}, "
            f"elements={result['element_set_count']:,}, anonymous={result['anonymous_integer_stoichiometry_count']:,}",
            flush=True,
        )
    dump(args.output_dir / "full_gnome_formula_layers_summary.json", full_summary)
    write_full_gnome_records(
        args.output_dir / "full_gnome_formula_layer_records.csv.gz", full_gnome, full_flags
    )

    gate_rows: list[dict[str, object]] = []
    for scope in ("full_index_post1980", "full_index_all_records"):
        gate_rows.extend(
            candidate_pairs(full_gnome, full_flags[scope], references[scope], primary)
        )
    gate_rows.sort(
        key=lambda row: (str(row["reference_scope"]), str(row["material_id"]), str(row["icsd_id"]))
    )
    write_csv(args.output_dir / "full_gnome_exact_formula_candidate_gate.csv", gate_rows)
    gate_summary = {}
    for scope in ("full_index_post1980", "full_index_all_records"):
        selected = [row for row in gate_rows if row["reference_scope"] == scope]
        gate_summary[scope] = {
            "candidate_count": len({row["material_id"] for row in selected}),
            "candidate_reference_pairs": len(selected),
            "partial_reference_pairs": sum(row["icsd_partial_occupancy"] for row in selected),
            "fully_occupied_reference_pairs": sum(
                not row["icsd_partial_occupancy"] for row in selected
            ),
            "formula_precedent_rule": scored_support["method"]["formula_precedent"],
            "partial_tolerance": primary,
        }
    dump(args.output_dir / "full_gnome_exact_formula_candidate_gate_summary.json", gate_summary)

    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": str(Path(__file__).resolve()),
        "producer_sha256": sha256(Path(__file__)),
        "formula_conventions_sha256": sha256(Path(__file__).with_name("formula_conventions.py")),
        "command": sys.argv,
        "parameters": {
            "post_year_exclusive": post,
            "max_year_inclusive": maximum,
            "primary_partial_tolerance": primary,
            "partial_tolerances": tolerances,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pymatgen": importlib.metadata.version("pymatgen"),
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
        "populations": {
            "icsd": len(icsd),
            "occupancy_flags": len(occupancy),
            "first_reports": len(first_reports),
            "full_gnome": len(full_gnome),
            "external": {
                name: {
                    "attempted": len(populations["attempted"]),
                    "scored": len(populations["scored"]),
                }
                for name, populations in external_inputs.items()
            },
        },
        "runtime_seconds": time.perf_counter() - started,
        "outputs": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
    }
    dump(args.output_dir / "formula_layers_provenance.json", provenance)
    print(f"Wrote formula-layer outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
