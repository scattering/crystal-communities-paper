#!/usr/bin/env python3
"""Join completed current-map recovery reports to newly inspected community evidence.

No old community IDs or names are embedded. Optional family JSON must be an
explicit current-map list: {"groups": [{"name": ..., "label_status": ...,
"communities": [integer IDs]}]}. Candidate names are not upgraded to verified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from prepare_representation_features import digest, read_labels
from analyze_representations import dump, write_csv


def descriptor_field(row, current, earlier):
    # The evidence agent's two explicitly known export schemas carry the same
    # quantities; do not infer a semantic label from any other column.
    return row[current] if current in row else row[earlier]


def aggregate_group(per, community_ids, production_sizes, confusion):
    communities = set(community_ids)
    present = [per[c] for c in communities if c in per]
    evaluated = sum(r["size"] for r in present)
    hits = sum(r["hits"] for r in present)
    internal_misses = sum(n for (a,b),n in confusion.items() if a in communities and b in communities)
    rates = [r["recovery"] for r in present]
    return {"communities": sorted(communities), "n_communities_requested": len(communities),
        "n_communities_available": len(present), "missing_communities": sorted(communities-set(per)),
        "n_production_entries": sum(production_sizes[c] for c in communities),
        "n_evaluated_entries": evaluated, "hits": hits,
        "coverage_fraction": evaluated/sum(production_sizes[c] for c in communities),
        "entry_weighted_recovery": hits/evaluated if evaluated else None,
        "community_mean_recovery": float(np.mean(rates)) if rates else None,
        "community_median_recovery": float(np.median(rates)) if rates else None,
        "community_min_recovery": min(rates) if rates else None,
        "community_max_recovery": max(rates) if rates else None,
        "misses_between_listed_communities": internal_misses,
        "merged_group_recovery": (hits+internal_misses)/evaluated if evaluated else None,
        "merged_group_definition": "Count nearest neighbors in any listed community as group matches; candidate pool and neighbor identities remain unchanged"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recovery", action="append", required=True, help="report_name=/path/to/completed/recovery.json")
    ap.add_argument("--production-labels", type=Path, required=True)
    ap.add_argument("--renaissance", type=Path, required=True)
    ap.add_argument("--renaissance-manifest", type=Path, required=True)
    ap.add_argument("--descriptors", type=Path, required=True, help="new renaissance_top20_checked_descriptors.csv")
    ap.add_argument("--families", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    labels = read_labels(args.production_labels)
    sizes = Counter(r["community"] for r in labels.values() if r["community"] >= 0)
    label_hash = digest(args.production_labels)
    manifest = json.loads(args.renaissance_manifest.read_text())
    if manifest.get("assignments_sha256") != label_hash:
        raise ValueError("Renaissance manifest does not use these corrected production assignments")
    renaissance = json.loads(args.renaissance.read_text())
    top = renaissance["top"][:20]
    top_ids = [int(r["community"]) for r in top]
    if len(top_ids) != 20 or len(set(top_ids)) != 20 or any(c not in sizes for c in top_ids):
        raise ValueError("Expected twenty unique current-map renaissance communities")
    with args.descriptors.open(newline="") as handle:
        descriptors = {int(r["community"]): r for r in csv.DictReader(handle)}
    for r in top:
        c = int(r["community"])
        if c not in descriptors or int(descriptor_field(descriptors[c],"community_size_all","n_all_members")) != sizes[c] or int(descriptor_field(descriptors[c],"community_size_dated","n_dated_survey_members")) != r["size"]:
            raise ValueError("Checked descriptor counts do not align to this current-map survey")
    families = json.loads(args.families.read_text())["groups"] if args.families else []
    for group in families:
        if not group.get("name") or not group.get("label_status") or not group.get("communities"):
            raise ValueError("Family group requires name, explicit label_status and community list")
        if any(not isinstance(c,int) or c not in sizes for c in group["communities"]):
            raise ValueError("Family groups must reference current-map communities")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"production_labels_sha256": label_hash,
        "sources": {str(p): digest(p) for p in [args.renaissance,args.renaissance_manifest,args.descriptors]+([args.families] if args.families else [])},
        "renaissance_definition": {k:v for k,v in renaissance.items() if k not in ("top","all_eligible")},
        "interpretation": "Checked descriptions retain their stated evidence limits; top-twenty ranking is a survey statistic, not verification of twenty scientific events.",
        "results": {}}
    markdown = ["# Current-map representation comparison joins", "",
        "All recovery rates use the corrected production labels. Compare numerical methods directly only when their candidate-cohort hashes match. Renaissance survey membership is dated; recovery coverage below counts all current community members, including undated entries.", "",
        "| Report | Candidates | 1-NN agreement | Random-other-entry chance | Top-20 available | Top-20 coverage | Top-20 entry-weighted recovery |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for item in args.recovery:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or name in report["results"]:
            ap.error("Each recovery must have a unique name=/path/to/recovery.json")
        path = Path(raw_path)
        d = json.loads(path.read_text())
        if d.get("production_labels_sha256") != label_hash:
            raise ValueError("Recovery does not use these corrected production labels")
        if d.get("approximate") and "exact_query_audit" not in d:
            raise ValueError("Approximate recovery exact-query audit has not completed")
        per = {int(r["community"]): r for r in d["per_community"]}
        if sum(r["size"] for r in per.values()) != d["n_entries"] or sum(r["hits"] for r in per.values()) != d["hits"]:
            raise ValueError("Recovery integer counts do not sum")
        if any(c not in sizes or r["size"] > sizes[c] for c,r in per.items()):
            raise ValueError("Recovery community counts exceed the corrected map")
        confusion_path = path.parent / "confusion_counts.csv"
        with confusion_path.open(newline="") as handle:
            confusion = Counter({(int(r["from_community"]),int(r["to_community"])):int(r["count"]) for r in csv.DictReader(handle)})
        if sum(confusion.values()) != d["misses"]:
            raise ValueError("Complete confusion counts do not sum to recovery misses")
        rows = []
        for rank,r in enumerate(top,1):
            c = int(r["community"])
            rec = per.get(c)
            rows.append({"rank": rank, "community": c, "description": descriptor_field(descriptors[c],"checked_descriptor","description"),
                "description_status": descriptor_field(descriptors[c],"validation_status","status"), "survey_event_year": r["event_year"],
                "survey_dated_size": r["size"], "production_all_members": sizes[c],
                "evaluated_members": rec["size"] if rec else 0,
                "coverage_fraction": rec["size"]/sizes[c] if rec else 0,
                "recovery": rec["recovery"] if rec else None,
                "hits": rec["hits"] if rec else 0, "misses": rec["misses"] if rec else 0,
                "top_miss_targets": rec["top_three_miss_targets"] if rec else [],
                "post_window_complete_through_2015": descriptor_field(descriptors[c],"full_post_window_available","post_window_complete_through_2015")})
        top_summary = aggregate_group(per,top_ids,sizes,confusion)
        # The ranking is not one material family: do not give its union a
        # merged-family interpretation merely because it occupies twenty slots.
        for key in ("misses_between_listed_communities","merged_group_recovery","merged_group_definition"):
            top_summary.pop(key)
        group_results = [{"name": g["name"], "label_status": g["label_status"],
            **aggregate_group(per,g["communities"],sizes,confusion)} for g in families]
        result = {"recovery_path": str(path), "recovery_sha256": digest(path),
            "confusion_sha256": digest(confusion_path), "n_candidates": d["n_entries"],
            "cohort_sha256": d["cohort_sha256"], "metric": d["metric"], "approximate": d["approximate"],
            "same_community_1nn_agreement": d["same_community_1nn_agreement"],
            "random_other_entry_chance": d["random_other_entry_chance"],
            "renaissance_top20_summary": top_summary, "renaissance_top20": rows, "family_groups": group_results}
        report["results"][name] = result
        write_csv(args.out_dir / f"{name}_renaissance_top20.csv", [{**r,"top_miss_targets":json.dumps(r["top_miss_targets"])} for r in rows])
        pct = lambda value: f"{100*value:.2f}%" if value is not None else "unavailable"
        markdown.append(f"| {name} | {d['n_entries']:,} | {pct(d['same_community_1nn_agreement'])} | {pct(d['random_other_entry_chance'])} | {top_summary['n_communities_available']}/20 | {pct(top_summary['coverage_fraction'])} | {pct(top_summary['entry_weighted_recovery'])} |")
    markdown.extend(["", "Per-community values, missing communities, coverage denominators, current descriptors and evidence status are retained in the accompanying JSON and CSV files. Family unions are included only when an explicit current-map family manifest was supplied.", ""])
    dump(args.out_dir / "s9_representation_joins.json",report)
    (args.out_dir / "s9_representation_joins.md").write_text("\n".join(markdown))


if __name__ == "__main__":
    main()
