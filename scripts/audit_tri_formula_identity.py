#!/usr/bin/env python3
"""Audit TRI/ICSD overlap and timing under the declared formula identity.

The analysis scripts retain readable reduced formulas for output, but formulas
are joined and grouped by element-sorted normalized atomic fractions rounded
to 12 decimal places.  This audit independently reconstructs the overlap and
the stepping-stone timing categories from the ICSD index and CrystalWeave
community assignments.  An optional legacy role table quantifies the effect
of replacing raw reduced-formula-string identity; an optional corrected role
table is checked row by row against the reconstruction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from pymatgen.core import Composition

from formula_conventions import normalized_fraction_key


FormulaIdentity = tuple[tuple[str, ...], tuple[float, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tri-json", required=True, type=Path)
    parser.add_argument("--icsd-index", required=True, type=Path)
    parser.add_argument("--community-assignments", required=True, type=Path)
    parser.add_argument("--legacy-role-records", type=Path)
    parser.add_argument("--corrected-role-records", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(formula: str) -> FormulaIdentity | None:
    try:
        return normalized_fraction_key(str(formula).strip(), decimals=12)
    except Exception:
        return None


def reduced_formula(formula: str) -> str | None:
    try:
        return Composition(str(formula).strip()).reduced_formula
    except Exception:
        return None


def load_tri(path: Path) -> tuple[dict[FormulaIdentity, str], dict[str, object]]:
    raw = json.loads(path.read_text())
    by_identity: dict[FormulaIdentity, list[str]] = defaultdict(list)
    failures = []
    for formula in raw:
        key = identity(formula)
        if key is None:
            failures.append(formula)
        else:
            by_identity[key].append(formula)
    collisions = [values for values in by_identity.values() if len(values) > 1]
    if collisions:
        raise ValueError(f"TRI contains {len(collisions)} normalized-identity collisions")
    return (
        {key: values[0] for key, values in by_identity.items()},
        {
            "raw_records": len(raw),
            "parse_failures": len(failures),
            "normalized_identities": len(by_identity),
            "identity_collisions": len(collisions),
        },
    )


def load_tri_raw_reduced_formula(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text())
    out = {}
    for formula in raw:
        key = reduced_formula(formula)
        if key is not None:
            out[key] = formula
    return out


def load_index(path: Path) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    failures = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                identifier = int(row["cif_names"])
            except Exception:
                continue
            formula = str(row.get("name") or "").strip()
            key = identity(formula)
            if key is None:
                failures.append(identifier)
                continue
            try:
                year = int(row["publication_year"])
            except Exception:
                year = None
            records[identifier] = {
                "identity": key,
                "formula": formula,
                "reduced_formula": reduced_formula(formula),
                "year": year,
            }
    return records, {
        "parsed_records": len(records),
        "parse_failures": len(failures),
        "normalized_identities_all_index": len({r["identity"] for r in records.values()}),
    }


def load_assignments(path: Path) -> list[dict[str, int | None]]:
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                identifier = int(row["icsd_id"])
                community = int(row["community"])
            except Exception:
                continue
            try:
                year = int(row["year"])
            except Exception:
                year = None
            records.append({"icsd_id": identifier, "community": community, "year": year})
    return records


def reconstruct(
    tri: dict[object, str],
    index: dict[int, dict[str, object]],
    assignments: list[dict[str, int | None]],
    *,
    key_field: str = "identity",
) -> tuple[dict[object, dict[str, object]], dict[str, object]]:
    community_birth: dict[int, int] = {}
    entries: dict[FormulaIdentity, list[int]] = defaultdict(list)
    years: dict[FormulaIdentity, list[int]] = defaultdict(list)
    communities: dict[FormulaIdentity, list[int]] = defaultdict(list)
    raw_formulas: dict[FormulaIdentity, set[str]] = defaultdict(set)
    for row in assignments:
        identifier = int(row["icsd_id"])
        community = int(row["community"])
        assigned_year = row["year"]
        if community >= 0 and assigned_year is not None:
            year = int(assigned_year)
            community_birth[community] = min(year, community_birth.get(community, year))
        meta = index.get(identifier)
        if meta is None:
            continue
        key = meta[key_field]
        if key is None:
            continue
        entries[key].append(identifier)
        raw_formulas[key].add(str(meta["formula"]))
        if meta["year"] is not None:
            years[key].append(int(meta["year"]))
        communities[key].append(community)

    shared = sorted(set(tri) & set(entries))
    states: dict[FormulaIdentity, dict[str, object]] = {}
    for key in shared:
        nonnoise = {community for community in communities[key] if community >= 0}
        first_community_birth = min(
            (community_birth[c] for c in nonnoise if c in community_birth), default=None
        )
        first_year = min(years[key]) if years[key] else None
        if first_year is None or first_community_birth is None:
            category = None
        elif first_community_birth < first_year:
            category = "joins_existing"
        elif first_community_birth == first_year:
            category = "co_birth"
        else:
            category = "community_after_formula"
        states[key] = {
            "tri_formula_raw": tri[key],
            "icsd_n_entries": len(entries[key]),
            "icsd_first_year": first_year,
            "icsd_first_associated_community_birth": first_community_birth,
            "stepping_stone_class": category,
            "icsd_raw_formulas": sorted(raw_formulas[key]),
        }
    counts = Counter(state["stepping_stone_class"] for state in states.values())
    classifiable = len(states) - counts[None]
    summary = {
        "n_shared": len(states),
        "n_classifiable": classifiable,
        "stepping_stone_counts": {
            "joins_existing": counts["joins_existing"],
            "co_birth": counts["co_birth"],
            "community_after_formula": counts["community_after_formula"],
            "unclassifiable": counts[None],
        },
        "joins_existing_rate_classifiable": (
            counts["joins_existing"] / classifiable if classifiable else None
        ),
    }
    return states, summary


def load_role_records(path: Path) -> dict[FormulaIdentity, dict[str, object]]:
    out = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = identity(row.get("formula") or row.get("tri_formula_raw") or "")
            if key is None:
                continue
            if key in out:
                raise ValueError(f"Duplicate identity in role table: {row.get('formula')!r}")
            try:
                first_year = int(float(row["icsd_first_year"])) if row.get("icsd_first_year") else None
            except Exception:
                first_year = None
            try:
                n_entries = int(float(row["icsd_n_entries"]))
            except Exception:
                n_entries = None
            out[key] = {
                "formula": row.get("formula"),
                "icsd_n_entries": n_entries,
                "icsd_first_year": first_year,
                "stepping_stone_class": row.get("stepping_stone_class") or None,
            }
    return out


def serialized_identity(key: FormulaIdentity) -> dict[str, object]:
    return {"elements": list(key[0]), "fractions": list(key[1])}


def main() -> int:
    args = parse_args()
    tri, tri_summary = load_tri(args.tri_json)
    index, index_summary = load_index(args.icsd_index)
    assignments = load_assignments(args.community_assignments)
    states, headline = reconstruct(tri, index, assignments)
    accepted_identities = {
        index[int(row["icsd_id"])]["identity"]
        for row in assignments
        if int(row["icsd_id"]) in index
    }

    inputs = {
        "tri_json": {"path": str(args.tri_json), "sha256": sha256_file(args.tri_json)},
        "icsd_index": {"path": str(args.icsd_index), "sha256": sha256_file(args.icsd_index)},
        "community_assignments": {
            "path": str(args.community_assignments),
            "sha256": sha256_file(args.community_assignments),
        },
        "producer": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "purpose": "TRI/ICSD overlap and stepping-stone timing audit under normalized nominal formula identity",
        "formula_identity": "element-sorted normalized atomic fractions rounded to 12 decimal places",
        "inputs": inputs,
        "tri": tri_summary,
        "icsd": {
            **index_summary,
            "accepted_records": len(assignments),
            "normalized_identities_accepted": len(accepted_identities),
        },
        "overlap": {
            "all_index": len(set(tri) & {r["identity"] for r in index.values()}),
            "accepted": len(states),
        },
        "headline": headline,
    }

    # Reconstruct the previously used raw reduced-formula-string analysis
    # directly from the same source inputs.  This keeps the change audit
    # reproducible without archiving a second multi-megabyte role table.
    legacy_raw, legacy_headline = reconstruct(
        load_tri_raw_reduced_formula(args.tri_json),
        index,
        assignments,
        key_field="reduced_formula",
    )
    legacy: dict[FormulaIdentity, dict[str, object]] = {}
    for raw_key, state in legacy_raw.items():
        normalized = identity(str(raw_key))
        if normalized is None:
            continue
        if normalized in legacy:
            raise ValueError("Legacy reduced-formula states collide after normalization")
        legacy[normalized] = {"formula": raw_key, **state}
    changed = []
    common_class_changes = 0
    for key in sorted(set(states) | set(legacy)):
        state = states.get(key)
        old = legacy.get(key)
        changed_fields = []
        for field in ("icsd_n_entries", "icsd_first_year", "stepping_stone_class"):
            if (state or {}).get(field) != (old or {}).get(field):
                changed_fields.append(field)
        if changed_fields:
            if state is not None and old is not None and "stepping_stone_class" in changed_fields:
                common_class_changes += 1
            changed.append(
                {
                    "identity": serialized_identity(key),
                    "tri_formula_raw": (state or old or {}).get("tri_formula_raw"),
                    "changed_fields": changed_fields,
                    "legacy": old,
                    "corrected": state,
                }
            )
    result["comparison_to_raw_reduced_formula_identity"] = {
        "legacy_headline": legacy_headline,
        "new_shared_identities": len(set(states) - set(legacy)),
        "removed_shared_identities": len(set(legacy) - set(states)),
        "changed_identities": len(changed),
        "class_changes_among_common_identities": common_class_changes,
        "changed_records": changed,
    }

    if args.corrected_role_records:
        corrected = load_role_records(args.corrected_role_records)
        mismatches = []
        for key in sorted(set(states) | set(corrected)):
            state = states.get(key)
            saved = corrected.get(key)
            if state is None or saved is None or any(
                state[field] != saved[field]
                for field in ("icsd_n_entries", "icsd_first_year", "stepping_stone_class")
            ):
                mismatches.append(serialized_identity(key))
        inputs["corrected_role_records"] = {
            "path": str(args.corrected_role_records),
            "sha256": sha256_file(args.corrected_role_records),
        }
        result["corrected_role_validation"] = {
            "saved_rows": len(corrected),
            "identity_set_matches": set(corrected) == set(states),
            "n_row_mismatches": len(mismatches),
            "mismatch_identities": mismatches[:25],
        }

    if args.legacy_role_records:
        legacy = load_role_records(args.legacy_role_records)
        changed = []
        common_class_changes = 0
        for key in sorted(set(states) | set(legacy)):
            state = states.get(key)
            old = legacy.get(key)
            changed_fields = []
            for field in ("icsd_n_entries", "icsd_first_year", "stepping_stone_class"):
                if (state or {}).get(field) != (old or {}).get(field):
                    changed_fields.append(field)
            if changed_fields:
                if state is not None and old is not None and "stepping_stone_class" in changed_fields:
                    common_class_changes += 1
                changed.append(
                    {
                        "identity": serialized_identity(key),
                        "tri_formula_raw": (state or {}).get("tri_formula_raw"),
                        "changed_fields": changed_fields,
                        "legacy": old,
                        "corrected": state,
                    }
                )
        legacy_counts = Counter(row["stepping_stone_class"] for row in legacy.values())
        inputs["legacy_role_records"] = {
            "path": str(args.legacy_role_records),
            "sha256": sha256_file(args.legacy_role_records),
        }
        result["legacy_role_table_validation"] = {
            "legacy_shared": len(legacy),
            "legacy_classifiable": len(legacy) - legacy_counts[None],
            "legacy_stepping_stone_counts": {
                "joins_existing": legacy_counts["joins_existing"],
                "co_birth": legacy_counts["co_birth"],
                "community_after_formula": legacy_counts["community_after_formula"],
                "unclassifiable": legacy_counts[None],
            },
            "new_shared_identities": len(set(states) - set(legacy)),
            "removed_shared_identities": len(set(legacy) - set(states)),
            "changed_identities": len(changed),
            "class_changes_among_common_identities": common_class_changes,
            "changed_records": changed,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "headline": headline}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
