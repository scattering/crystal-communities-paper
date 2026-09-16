#!/usr/bin/env python3
"""Prepare repaired community evidence and Magpie comparisons without name transfer.

Uses the complete ICSD index and saved production PCA. Formula/space-group
descriptors are evidence for curation, not verified prototype or event labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from pymatgen.core import Composition
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import icsd_ablation_comparison as ablation
from analyze_renaissance_communities import SEED_PATTERNS
from analyze_renaissance_extra import PROBES
from compare_feature_repair import read_assignments
from formula_conventions import normalized_fraction_key
from regenerate_repaired_temporal import dump, file_hash, write_csv

SCRIPTS = Path(__file__).resolve().parent


def metadata(path):
    out, cache = {}, {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            iid = int(row["cif_names"])
            formula = row["name"].strip()
            if formula not in cache:
                try:
                    composition = Composition(formula)
                    cache[formula] = (
                        composition.reduced_formula,
                        sorted(composition.get_el_amt_dict()),
                        normalized_fraction_key(formula),
                    )
                except (ValueError, TypeError):
                    cache[formula] = (None, [], None)
            reduced, elements, nominal_identity = cache[formula]
            out[iid] = {"formula": formula, "reduced_formula": reduced, "elements": elements,
                        "nominal_formula_identity": nominal_identity,
                        "space_group": int(row["sym_group"]) if row["sym_group"].strip() else None}
    return out


def characterize(records, meta):
    groups = defaultdict(list)
    for iid, year, c in records:
        if c >= 0:
            groups[c].append((iid, year))
    profiles = {}
    for c, members in sorted(groups.items()):
        formulas, sgs, elements = Counter(), Counter(), Counter()
        indexed, parsed = 0, 0
        years = [y for _, y in members if y is not None]
        for iid, _ in members:
            row = meta.get(iid)
            if row is None:
                continue
            indexed += 1
            if row["space_group"] is not None:
                sgs[row["space_group"]] += 1
            if row["reduced_formula"]:
                parsed += 1
                formulas[row["reduced_formula"]] += 1
                elements.update(row["elements"])
        profiles[c] = {"community": c, "size": len(members), "n_dated": len(years),
                       "birth_year": min(years) if years else None,
                       "n_indexed": indexed, "n_parsed_formulas": parsed,
                       "n_unique_space_groups": len(sgs), "top_space_groups": sgs.most_common(10),
                       "top_reduced_formulas": formulas.most_common(10),
                       "element_member_counts": dict(elements.most_common()),
                       "dominant_space_group_share_all_members": sgs.most_common(1)[0][1] / len(members) if sgs else None,
                       "label_status": "Uncurated: formula/SG evidence alone does not verify a prototype or research-event identity."}
    return profiles


def first_reports(records, meta, *, identity_field="reduced_formula"):
    """Select the first post-cutoff record for each requested formula identity.

    Ties retain assignment row order, matching the producer's strict year
    comparison. ``reduced_formula`` retains the frozen first-report tables used
    by the original community-evidence run. ``nominal_formula_identity``
    supplies the single 12-decimal normalized-fraction identity used for the
    corrected S1.5 temporal attachment analysis.
    """
    union = {}
    tables = {}
    for cutoff in (1980, 1990, 2000, 2010):
        first = {}
        for iid, year, _ in records:
            identity = meta.get(iid, {}).get(identity_field)
            if year is None or year <= cutoff or not identity:
                continue
            if identity not in first or year < first[identity][1]:
                first[identity] = (iid, year)
        table = [
            {
                "cif_id": iid,
                "reduced_formula": meta[iid]["reduced_formula"],
                "year": year,
            }
            for _, (iid, year) in first.items()
        ]
        tables[cutoff] = table
        for identity, (iid, year) in first.items():
            union.setdefault(iid, (identity, year))
    return union, tables


def magpie_comparison(prod_records, mag_records, meta, out, *, input_provenance=None):
    prod = {i: (y, c) for i, y, c in prod_records}
    mag = {i: (y, c) for i, y, c in mag_records}
    for iid in prod.keys() & mag.keys():
        if prod[iid][0] != mag[iid][0]:
            raise ValueError("Publication years differ across repaired representations")
    p, a, common = ablation.aligned_label_arrays(prod, mag, drop_outliers=True)
    all_p, all_a, common_all = ablation.aligned_label_arrays(prod, mag, drop_outliers=False)
    sg = {i: v["space_group"] for i, v in meta.items() if v["space_group"] is not None}
    legacy_first, tables = first_reports(prod_records, meta)
    first, normalized_tables = first_reports(
        prod_records, meta, identity_field="nominal_formula_identity"
    )
    own_mag_first, own_normalized_tables = first_reports(
        mag_records, meta, identity_field="nominal_formula_identity"
    )
    for cutoff, rows in tables.items():
        folder = out / f"formula_selection/split_{cutoff}"
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / "first_report_formulas.csv", rows)
    counts = Counter(c for _, _, c in prod_records if c >= 0)
    purity = [ablation.per_community_purity(prod, mag, c) for c in sorted(counts)]
    sg_prod, sg_prod_rows = ablation.mean_space_groups_top10(prod, sg)
    sg_mag, sg_mag_rows = ablation.mean_space_groups_top10(mag, sg)
    corrected_attachment = {
        "production": ablation.formula_attachment_rate(prod, first),
        "magpie": ablation.formula_attachment_rate(mag, first),
    }
    legacy_attachment = {
        "production": ablation.formula_attachment_rate(prod, legacy_first),
        "magpie": ablation.formula_attachment_rate(mag, legacy_first),
    }
    identity_audit = {
        "purpose": "S1.5 formula-attachment identity correction",
        "formula_identity": (
            "Element-sorted normalized atomic fractions rounded to 12 decimal "
            "places for every ICSD record"
        ),
        "selection_scope": (
            "First post-cutoff record for each nominal composition at 1980, "
            "1990, 2000 and 2010; union by ICSD ID; earliest record per identity "
            "over that union"
        ),
        "frozen_table_note": (
            "formula_selection/split_*/first_report_formulas.csv remains the "
            "previous reduced-formula selection because it is an input to the "
            "separately versioned Fig. 4 formula-precedent analysis"
        ),
        "formula_counts_by_cutoff": {
            str(cutoff): {
                "legacy_reduced_formula": len(tables[cutoff]),
                "normalized_nominal_identity": len(normalized_tables[cutoff]),
                "collapsed_by_correction": (
                    len(tables[cutoff]) - len(normalized_tables[cutoff])
                ),
            }
            for cutoff in sorted(tables)
        },
        "production_reference_union": {
            "legacy_rows": len(legacy_first),
            "normalized_identity_rows": len(first),
        },
        "same_production_reference": {
            "legacy_reduced_formula": legacy_attachment,
            "normalized_nominal_identity": corrected_attachment,
        },
        "magpie_own_reference": {
            "normalized_identity_rows": len(own_mag_first),
            "formula_counts_by_cutoff": {
                str(cutoff): len(own_normalized_tables[cutoff])
                for cutoff in sorted(own_normalized_tables)
            },
        },
        "inputs": input_provenance or {},
        "producer_sha256": file_hash(Path(__file__)),
        "formula_conventions_sha256": file_hash(SCRIPTS / "formula_conventions.py"),
    }
    report = {"n_common_total": len(common_all), "n_common_nonoutlier_both": len(common),
              "agreement_nonoutlier_both": {"ARI": adjusted_rand_score(p, a), "NMI": normalized_mutual_info_score(p, a)},
              "agreement_including_noise": {"ARI": adjusted_rand_score(all_p, all_a), "NMI": normalized_mutual_info_score(all_p, all_a)},
              "decade_birth": {"production": ablation.decade_birth_ratios(prod), "magpie": ablation.decade_birth_ratios(mag)},
              "sg_top10": {"production": {"mean": sg_prod, "per_community": sg_prod_rows},
                           "magpie": {"mean": sg_mag, "per_community": sg_mag_rows}},
              "formula_reference": "First post-cutoff production records at 1980/1990/2000/2010 under one element-sorted normalized atomic-fraction identity (12 decimals), union by ICSD ID; earliest nominal-composition entry over that union. Selection rebuilt from the full index and repaired rows, with the same strict-year tie rule as retrodiction.",
              "formula_attachment_same_production_reference": corrected_attachment,
              "magpie_formula_attachment_own_reference_supplement": ablation.formula_attachment_rate(mag, own_mag_first),
              "per_community_purity": purity,
              "family_labels": "Not copied from historical IDs; see full-membership profiles and reconciliation."}
    dump(out / "magpie_comparison.json", report)
    dump(out / "s1p5_formula_identity_audit.json", identity_audit)
    write_csv(out / "production_to_magpie_purity.csv", purity)
    lines = ["# Repaired production versus Magpie ablation", "", f"Shared entries: {len(common_all):,}; assigned in both: {len(common):,}.",
             f"ARI/NMI among assigned-in-both entries: {report['agreement_nonoutlier_both']['ARI']:.6f} / {report['agreement_nonoutlier_both']['NMI']:.6f}.",
             f"ARI/NMI including noise: {report['agreement_including_noise']['ARI']:.6f} / {report['agreement_including_noise']['NMI']:.6f}.", "",
             "| Decade | Production birth share | Magpie birth share |", "|---|---:|---:|"]
    for d in report["decade_birth"]["production"]:
        lines.append(f"| {d} | {100*report['decade_birth']['production'][d]:.3f}% | {100*report['decade_birth']['magpie'][d]:.3f}% |")
    lines += ["", f"Top-ten communities: mean distinct SG count {sg_prod:.1f} (production), {sg_mag:.1f} (Magpie).", "",
              report["formula_reference"], "", "| Formula-entry classification | Production | Magpie |", "|---|---:|---:|"]
    rates = report["formula_attachment_same_production_reference"]
    for key in ("n_classifiable", "n_unclassifiable", "pre_existing_rate", "community_birth_rate", "precedes_rate"):
        lines.append(f"| {key} | {rates['production'][key]} | {rates['magpie'][key]} |")
    lines += ["", "Per-community mapping purity is in production_to_magpie_purity.csv; its denominator excludes Magpie outliers and missing IDs, as in the historical comparison."]
    (out / "magpie_comparison.md").write_text("\n".join(lines) + "\n")


def representatives(records, sample_path, pca_path, meta, out):
    sample = read_assignments(sample_path)
    if [(i, y) for i, y, _ in sample] != [(i, y) for i, y, _ in records]:
        raise ValueError("Saved PCA/sample order differs from production assignment order")
    X = np.load(pca_path, allow_pickle=False, mmap_mode="r")
    if X.shape != (len(records), 32) or not np.isfinite(X).all():
        raise ValueError("Expected finite saved production PCA with32 columns")
    grouped = defaultdict(list)
    for index, (_, _, c) in enumerate(records):
        if c >= 0:
            grouped[c].append(index)
    rows = []
    for c, indexes in sorted(grouped.items()):
        if len(indexes) < 25:
            continue
        values = X[indexes]
        distances = np.linalg.norm(values - values.mean(axis=0), axis=1)
        order = np.argsort(distances, kind="stable")[:20]
        for rank, offset in enumerate(order, 1):
            iid, year, _ = records[indexes[offset]]
            row = meta.get(iid, {})
            rows.append({"community": c, "community_size": len(indexes), "rank_by_centroid_distance": rank,
                         "icsd_id": iid, "year": year, "centroid_distance": float(distances[offset]),
                         "formula": row.get("formula"), "reduced_formula": row.get("reduced_formula"),
                         "space_group": row.get("space_group"), "community_label": "", "label_status": "requires curation"})
    write_csv(out / "production_central_representatives_top20.csv", rows)


def family_probes(records, meta, out):
    assignment = {i: (y, c) for i, y, c in records}
    definitions = []
    for key, pattern in SEED_PATTERNS.items():
        definitions.append((key, pattern["event_year"], pattern["seed_formulas"],
                            pattern["element_seed_required"], pattern["element_seed_any_of"]))
    tmd = next(p for p in PROBES if p["key"] == "tmd_2d")
    definitions += [("tmd_exact_compositions", 2010, tmd["exact_seeds"], set(), set()),
                    ("LaFeAsO_exact_parent", 2008, ["LaFeAsO"], set(), set()),
                    ("BaFe2As2_exact_parent", 2008, ["BaFe2As2"], set(), set()),
                    ("LiFeAs_exact_parent", 2008, ["LiFeAs"], set(), set()),
                    ("RE_In_TM_212_exact_seeds", 1989, ["La2InCu2", "La2InPd2"], set(), set())]
    results = []
    year_end = max(y for _, y, _ in records if y is not None)
    for key, event, exact, required, any_of in definitions:
        exact = {Composition(f).reduced_formula for f in exact}
        matched = [i for i, row in meta.items() if i in assignment and
                   (row["reduced_formula"] in exact or (required and required <= set(row["elements"]) and (not any_of or any_of & set(row["elements"]))))]
        counts = Counter(assignment[i][1] for i in matched if assignment[i][1] is not None and assignment[i][1] >= 0)
        communities = []
        for c, count in counts.most_common(5):
            hist = Counter(y for _, y, label in records if label == c and y is not None)
            pre = sum(n for y, n in hist.items() if event - 10 < y <= event)
            post = sum(n for y, n in hist.items() if event < y <= event + 10)
            available = min(10, year_end - event)
            communities.append({"community": c, "matched_seed_members": count, "dated_community_members": sum(hist.values()),
                                "year_histogram": dict(sorted(hist.items())), "n_pre": pre, "n_post": post,
                                "pre_rate_10yr": pre/10, "post_rate_10yr": post/10,
                                "post_available_years": available, "post_rate_available": post/available if available else None,
                                "fold_10yr": post/pre if pre else None,
                                "fold_available_years": (post/available)/(pre/10) if available and pre else None})
        results.append({"probe": key, "event_year": event, "exact_reduced_seeds": sorted(exact),
                        "required_elements": sorted(required), "any_of_elements": sorted(any_of),
                        "n_matched_successful_entries": len(matched), "n_matched_outlier_entries": sum(assignment[i][1] < 0 for i in matched),
                        "candidate_communities": communities})
    dump(out / "targeted_family_probe_candidates.json", {"scope": "Full-index composition-seed selection; whole-community histories. Candidates require actual structure verification; no historical IDs forced or prototype names assigned.", "probes": results})


def historical_label_candidates(old_records, new_records, old_labels_path, out):
    old = {i: c for i, _, c in old_records}
    new = {i: c for i, _, c in new_records}
    old_size, new_size = Counter(old.values()), Counter(new.values())
    overlap = defaultdict(Counter)
    for iid, c in old.items():
        if iid in new:
            overlap[c][new[iid]] += 1
    rows = []
    with old_labels_path.open(newline="") as handle:
        for label in csv.DictReader(handle):
            if label["kind"] != "graph_community":
                continue
            c = int(label["id"])
            for new_c, count in overlap[c].most_common(5):
                rows.append({"historical_community": c, "historical_family_name_NOT_TRANSFERRED": label["canonical_family_name"],
                             "new_community": new_c, "shared_members": count,
                             "fraction_of_historical_members": count/old_size[c], "fraction_of_new_members": count/new_size[new_c],
                             "status": "Membership link only; verify whole-community chemistry and actual structures before naming."})
    write_csv(out / "historical_family_reconciliation_candidates.csv", rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--icsd-index", type=Path, required=True)
    p.add_argument("--production-pca", type=Path, required=True)
    p.add_argument("--old-production-assignments", type=Path, required=True)
    p.add_argument("--historical-family-labels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = metadata(args.icsd_index)
    print(f"Loaded complete index: {len(meta):,} entries", flush=True)
    records = {}
    for name in ("production", "magpie", "graphlets"):
        path = args.run_root / name / ("graphlet_community_assignments.csv" if name == "graphlets" else "graph/community_assignments.csv")
        records[name] = read_assignments(path)
        profiles = characterize(records[name], meta)
        dump(args.out_dir / f"{name}_full_membership_profiles.json", profiles)
        subprocess.run([sys.executable, str(SCRIPTS / "analyze_prototype_collapse.py"), "--icsd-index", str(args.icsd_index),
                        "--community-assignments", str(path), "--output-dir", str(args.out_dir / name / "prototype_collapse")], check=True)
    representatives(records["production"], args.run_root / "production/sample_assignments.csv", args.production_pca, meta, args.out_dir)
    magpie_comparison(
        records["production"],
        records["magpie"],
        meta,
        args.out_dir,
        input_provenance={
            "icsd_index": {
                "path": str(args.icsd_index),
                "sha256": file_hash(args.icsd_index),
            },
            "production_assignments": {
                "path": str(args.run_root / "production/graph/community_assignments.csv"),
                "sha256": file_hash(args.run_root / "production/graph/community_assignments.csv"),
            },
            "magpie_assignments": {
                "path": str(args.run_root / "magpie/graph/community_assignments.csv"),
                "sha256": file_hash(args.run_root / "magpie/graph/community_assignments.csv"),
            },
        },
    )
    family_probes(records["production"], meta, args.out_dir)
    historical_label_candidates(read_assignments(args.old_production_assignments), records["production"], args.historical_family_labels, args.out_dir)
    dump(args.out_dir / "community_evidence_manifest.json", {"producer_sha256": file_hash(Path(__file__)),
         "index_sha256": file_hash(args.icsd_index), "production_pca_sha256": file_hash(args.production_pca),
         "basis": "Saved repaired PCA; no refitting", "n_index_rows": len(meta),
         "family_labels": "No canonical labels produced; verified chemistry/SG summaries and membership candidates only."})
    print(f"Wrote community evidence and Magpie comparison to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
