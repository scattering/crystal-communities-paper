#!/usr/bin/env python3
"""Choose new SI S7.6 examples from the repaired full map, without fetching CIFs.

The previous examples were hand-picked. This deterministic selection preserves
their source balance, central in-basin examples, and moderate frontier distances;
it does not preserve obsolete community IDs or claim prototype identities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from crystal_neighbors import FEATURE_VERSION
from formula_conventions import scale_invariant_formula_key

QUADRANTS = (
    ("in_basin_and_formula_match", "A", "In-basin; post-1980 formula match"),
    ("frontier_and_formula_match", "B", "Frontier; post-1980 formula match"),
    ("in_basin_and_no_formula_match", "C", "In-basin; no post-1980 formula match"),
    ("frontier_and_no_formula_match", "D", "Frontier; no post-1980 formula match"),
)
SOURCES = ("GNoME", "MatterGen", "MP", "JARVIS", "Alexandria")


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flag(value):
    if str(value).lower() not in ("true", "false", "1", "0"):
        raise ValueError(f"Invalid Boolean: {value!r}")
    return str(value).lower() in ("true", "1")


def quota_for(pool):
    available = Counter(r["source"] for r in pool)
    quota = {s: min(2, available[s]) for s in SOURCES if available[s]}
    # Preserve the historical 2 MatterGen / 3 MP / 3 JARVIS / 2 Alexandria
    # allocation when GNoME has no eligible formula matches.
    while sum(quota.values()) < 10:
        choices = [s for s in ("MP", "JARVIS", "Alexandria", "GNoME", "MatterGen")
                   if quota.get(s, 0) < min(3, available[s])]
        if not choices:
            raise ValueError(f"Cannot choose ten examples with <=3/source: {dict(available)}")
        quota[choices[0]] = quota.get(choices[0], 0) + 1
    return quota


def choose(pool, in_basin):
    quota = quota_for(pool)
    selected, used_communities, used_formula_identities, diagnostics = [], set(), set(), {}
    for source in SOURCES:
        if source not in quota:
            continue
        candidates = [r for r in pool if r["source"] == source]
        moderate = [r for r in candidates if 1.2 <= r["d_over_tau"] <= 3]
        eligible = candidates if in_basin else moderate
        if len(eligible) < quota[source]:
            # A sparse cell can require a documented distance exception to keep
            # the original source balance. Add only the nearest missing cases.
            outside = [r for r in candidates if r not in eligible]
            outside.sort(key=lambda r: (max(1.2 - r["d_over_tau"], r["d_over_tau"] - 3, 0), r["record_key"]))
            eligible = eligible + outside[:quota[source] - len(eligible)]
        diagnostics[source] = {"available": len(candidates), "distance_eligible": len(eligible),
                               "selected": quota[source],
                               "distance_exceptions": [r["record_key"] for r in eligible if not in_basin and r not in moderate]}
        for i in range(quota[source]):
            options = [r for r in eligible if r not in selected]
            if source == "MatterGen" and i == 0:
                generated = [r for r in options if r["mattergen_family"] == "mattergen"]
                if generated:
                    options = generated
            # Distinct communities and n>=20 are preferences, not exclusions.
            # Qualified names break distance ties only; an unlabelled community
            # is described by its actual full-map central exemplars instead.
            def score(r):
                distance = r["d_over_tau"] if in_basin else abs(r["d_over_tau"] - 1.8)
                identity = scale_invariant_formula_key(r["reduced_formula"])
                return (r["assigned_community"] in used_communities, identity in used_formula_identities, r["community_size"] < 20,
                        distance, r["family_name_kind"] != "checked_description", r["record_key"])
            picked = min(options, key=score)
            selected.append(picked)
            used_communities.add(picked["assigned_community"])
            used_formula_identities.add(scale_invariant_formula_key(picked["reduced_formula"]))
    return selected, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream", required=True, type=Path)
    parser.add_argument("--full-map-assignments", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.downstream
    paths = {"quadrants": root / "prior/quadrant_assignments.csv",
             "thresholds": root / "reference_basis/community_thresholds.json",
             "profiles": root / "community_evidence/production_full_membership_profiles.json",
             "families": root / "community_evidence/current_family_manifest.json",
             "representatives": root / "community_evidence/production_central_representatives_top20.csv",
             "first_report": root / "synthesis/split_1980/first_report_formulas.csv",
             "full_map": args.full_map_assignments}
    thresholds = json.loads(paths["thresholds"].read_text())
    if thresholds["feature_version"] != FEATURE_VERSION:
        raise ValueError("Candidate selection requires the current repaired encoder")
    profiles = json.loads(paths["profiles"].read_text())
    names, descriptions = {}, defaultdict(list)
    for group in json.loads(paths["families"].read_text())["groups"]:
        for community in group["communities"]:
            names[str(community)] = (group["name"], group["label_status"])
    for row in read_csv(paths["representatives"]):
        c = row["community"]
        if row["reduced_formula"] not in descriptions[c]:
            descriptions[c].append(row["reduced_formula"])
    first = {}
    for record in read_csv(paths["first_report"]):
        key = scale_invariant_formula_key(record["reduced_formula"])
        previous = first.get(key)
        if previous is None or int(record["year"]) < int(previous["year"]):
            first[key] = record
    assignments = read_csv(paths["full_map"])
    full_map = {r["icsd_id"]: r["community"] for r in assignments}
    if len(full_map) != len(assignments):
        raise ValueError("Duplicate full-map ICSD IDs")
    sizes = Counter(r["community"] for r in assignments if r["community"] != "-1")
    if dict(sizes) != {k: int(v) for k, v in thresholds["per_community_size"].items()}:
        raise ValueError("Full-map assignments and repaired thresholds disagree")
    rows, seen = [], set()
    for row in read_csv(paths["quadrants"]):
        c = row["assigned_community"]
        source = row["source"]
        member = row.get("zip_member", "") if source == "MatterGen" else ""
        if source == "MatterGen" and not member:
            raise ValueError("MatterGen identity requires zip_member")
        key = member or row["material_id"]
        if (source, key) in seen:
            raise ValueError(f"Duplicate candidate identity: {source}/{key}")
        seen.add((source, key))
        tau = float(thresholds["per_community_p95_threshold"][c])
        distance = float(row["nearest_centroid_distance"])
        if not math.isfinite(distance) or not math.isfinite(tau) or tau <= 0:
            raise ValueError(f"Nonfinite distance or invalid radius: {source}/{key}")
        formula_key = scale_invariant_formula_key(row["reduced_formula"])
        inside, match = distance <= tau, formula_key in first
        short_q = ("in_basin" if inside else "frontier") + ("_match" if match else "_no_match")
        q = ("in_basin" if inside else "frontier") + ("_and_formula_match" if match else "_and_no_formula_match")
        if (flag(row["in_basin"]) != inside or flag(row["formula_match"]) != match
                or row["quadrant"] not in (q, short_q)):
            raise ValueError(f"Quadrant/threshold/reference disagreement: {source}/{key}")
        if not math.isclose(float(row["community_threshold_p95"]), tau, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("External and full-map p95 thresholds disagree")
        if profiles[c]["size"] != sizes[c]:
            raise ValueError("Community evidence belongs to a different partition")
        if c in names:
            name, status = names[c]
            kind = "checked_description"
        else:
            # The central-representative producer omits some small communities.
            # Use explicitly identified membership frequencies there, never a
            # split-map community with the same numerical ID.
            central = [f for f in descriptions[c] if len(f) <= 32]
            formulas = central or [f for f, _ in profiles[c]["top_reduced_formulas"] if len(f) <= 32]
            prefix = "central examples" if central else "frequent formulas"
            sg, count = profiles[c]["top_space_groups"][0]
            name = f"SG {sg}: {count}/{sizes[c]} members"
            if formulas:
                name += f"; {prefix}: " + ", ".join(formulas[:2])
            status = "Full-map space-group/member evidence; no prototype identity inferred."
            kind = "full_map_exemplars" if central else "full_membership_evidence"
        found = first.get(formula_key, {})
        cid = found.get("cif_id", "")
        fc = full_map.get(cid, "") if cid else ""
        rows.append({"quadrant": q, "source": source, "material_id": row["material_id"],
                     "record_key": key, "zip_member": member, "reduced_formula": row["reduced_formula"],
                     "mattergen_family": row.get("family", row.get("mattergen_family", "")) if source == "MatterGen" else "",
                     "mattergen_task": row.get("task", "") if source == "MatterGen" else "",
                     "assigned_community": c, "community_size": sizes[c], "canonical_family_name": name,
                     "family_name_kind": kind, "label_status": status, "nearest_centroid_distance": distance,
                     "tau_c": tau, "d_over_tau": distance / tau, "in_basin": inside, "formula_match": match,
                     "matching_icsd_id": cid, "matching_icsd_year": found.get("year", ""),
                     "matching_icsd_full_map_community": fc,
                     "matching_icsd_same_community": fc == c if fc else "",
                     "cif_source_for_rendering": "Frozen public release; exact member recorded in extraction_manifest.csv"})
    output, diagnostics = [], {}
    for q, _, _ in QUADRANTS:
        picks, diag = choose([r for r in rows if r["quadrant"] == q], q.startswith("in_basin"))
        diagnostics[q] = diag
        for rank, row in enumerate(picks, 1):
            row["rank"] = rank
            output.append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "representative_candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    manifest = {"feature_version": FEATURE_VERSION, "n_candidates": len(output),
                "n_validated_source_records": len(rows), "n_formula_reference": len(first),
                "formula_key": "collapse species/isotopes to elements; normalize total amount to one; pymatgen get_integer_formula_and_factor; gcd reduce; sort elements alphabetically",
                "selection": "10/quadrant; 2-3/source where available; distinct communities and size>=20 preferred; in-basin smallest d/tau; frontier 1.2<=d/tau<=3 ranked nearest 1.8, with the closest distance exceptions only if needed for source balance; first MatterGen example generated where eligible",
                "scope": "Illustrative examples, not a random representative sample or individual prototype validation.",
                "diagnostics": diagnostics,
                "sha256": {name: sha256(path) for name, path in paths.items()},
                "producer_sha256": sha256(__file__)}
    (args.out_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    md = ["# Repaired-map representative candidates", "", manifest["selection"] + ".", "", manifest["scope"], "",
          "Family text uses checked descriptions of the current full-map communities or their central ICSD formulas. It does not certify a candidate prototype. Formula matches use the repaired post-1980 reference; matching ICSD entries are joined to full-map assignments by collection code.", ""]
    exceptions = [r for r in output if not r["in_basin"] and not 1.2 <= r["d_over_tau"] <= 3]
    if exceptions:
        md += ["Distance-window exceptions needed to retain source balance: " + "; ".join(
            f"{r['source']} {r['material_id']} ({r['quadrant']}, d/τc={r['d_over_tau']:.3f})" for r in exceptions) + ".", ""]
    for q, letter, title in QUADRANTS:
        md += [f"## {letter}. {title}", "", "| # | Source / family | ID | Formula | Community (size) | d/τc | Community description | Matching ICSD (year; full-map community) |", "|---:|---|---|---|---|---:|---|---|"]
        for row in output:
            if row["quadrant"] != q:
                continue
            src = row["source"] + (" / " + row["mattergen_family"] if row["mattergen_family"] else "")
            match = (f"{row['matching_icsd_id']} ({row['matching_icsd_year']}; {row['matching_icsd_full_map_community'] or 'unassigned'})" if row["formula_match"] else "—")
            md.append(f"| {row['rank']} | {src} | {row['material_id']} | {row['reduced_formula']} | {row['assigned_community']} ({row['community_size']}) | {row['d_over_tau']:.3f} | {row['canonical_family_name']} | {match} |")
        md.append("")
    (args.out_dir / "representative_candidates.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"selected": len(output), "out_dir": str(args.out_dir), "by_quadrant_source": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
