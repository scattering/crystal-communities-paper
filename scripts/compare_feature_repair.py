#!/usr/bin/env python3
"""Compare historical and repaired temporal results without interpreting them.

Writes only comparison.json and comparison.md under the required --out-dir.
New-run defaults follow run_{production,magpie,graphlets}.sh. Feature matrices
are optional: their headers are checked when present; no arrays are loaded.
Historical Magpie assignments can be supplied separately because the saved
ablation report contains birth shares, not its full successful-ID population.
When successful-ID populations differ, a supplementary shared-cohort comparison
retains fitted labels and scores both original and recomputed community births.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import struct
from collections import Counter
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text())


def read_assignments(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    seen = set()
    for row in rows:
        iid = int(row["icsd_id"])
        if iid in seen:
            raise ValueError(f"Duplicate ICSD ID {iid} in {path}")
        seen.add(iid)
        year = int(row["year"]) if row.get("year", "").strip() else None
        records.append((iid, year, int(row.get("community", row.get("cluster", -1)))))
    return records


def decade(year):
    return "unknown" if year is None else f"{year // 10 * 10}s"


def temporal_stats(summary):
    return {
        "n_points": summary.get("n_points"),
        "n_outliers": summary.get("n_outliers"),
        "outlier_ratio": summary.get("outlier_ratio"),
        "n_communities_with_birth_year": len(summary.get("community_birth_year", {})),
        "by_decade": {
            d: {"n_total": int(v["n_total"]),
                "n_birth": int(v.get("n_cluster_birth_point", 0)),
                "birth_share": v.get("cluster_birth_point_ratio"),
                "n_outliers": int(v.get("n_outlier", 0)),
                "outlier_ratio": v.get("outlier_ratio"),
                "existing_community_share": v.get("existing_cluster_ratio"),
                "same_community_nonexclusive_share": v.get("same_community_attachment_ratio")}
            for d, v in sorted(summary.get("by_decade", {}).items())
        },
    }


def validate_rows(records, stats, path):
    """Validate temporal summary against its actual ID/year/label records."""
    if stats["n_points"] is not None and len(records) != stats["n_points"]:
        raise ValueError(f"Row count disagrees with n_points: {path}")
    noise = sum(c < 0 for _, _, c in records)
    if stats["n_outliers"] is not None and noise != stats["n_outliers"]:
        raise ValueError(f"Noise count disagrees with summary: {path}")
    births = {}
    for _, year, c in records:
        if c >= 0 and year is not None:
            births[c] = min(year, births.get(c, year))
    totals = Counter(decade(y) for _, y, _ in records)
    born = Counter(decade(y) for _, y, c in records if c >= 0 and y is not None and births[c] == y)
    if stats["by_decade"] and all("n_total" in v for v in stats["by_decade"].values()):
        if totals.keys() != stats["by_decade"].keys():
            raise ValueError(f"Summary decade coverage disagrees with assignments: {path}")
    for d, values in stats["by_decade"].items():
        if "n_total" in values and totals[d] != values["n_total"]:
            raise ValueError(f"Decade row count disagrees for {d}: {path}")
        if values["birth_share"] is not None and totals[d]:
            if not math.isclose(born[d] / totals[d], values["birth_share"], abs_tol=1e-9):
                raise ValueError(f"Birth share disagrees for {d}: {path}")
    return {"unique_ids": True, "row_count": len(records),
            "summary_counts_and_birth_shares_match": True}


def cohort_comparison(old, new, reference):
    old_years = {iid: y for iid, y, _ in old}
    new_years = {iid: y for iid, y, _ in new}
    common = old_years.keys() & new_years.keys()
    mismatches = [iid for iid in sorted(common) if old_years[iid] != new_years[iid]]
    if mismatches:
        raise ValueError(f"Years changed for {len(mismatches)} shared IDs ({reference}); first IDs {mismatches[:10]}")
    removed = sorted(old_years.keys() - new_years.keys())
    added = sorted(new_years.keys() - old_years.keys())
    old_counts = Counter(decade(y) for y in old_years.values())
    new_counts = Counter(decade(y) for y in new_years.values())
    removed_counts = Counter(decade(old_years[i]) for i in removed)
    added_counts = Counter(decade(new_years[i]) for i in added)
    return {
        "reference": reference, "old_successful_count": len(old),
        "new_successful_count": len(new), "shared_count": len(common),
        "same_years_on_overlap": True, "only_old_ids": removed, "only_new_ids": added,
        "by_decade": {d: {"old": old_counts[d], "new": new_counts[d],
                           "only_old": removed_counts[d], "only_new": added_counts[d]}
                      for d in sorted(old_counts.keys() | new_counts.keys())},
    }


def birth_years(records):
    births = {}
    for _, year, community in records:
        if community >= 0 and year is not None:
            births[community] = min(year, births.get(community, year))
    return births


def score_birth_rows(records, births):
    """Score supplied rows against given birth years; keep noise in denominators."""
    totals = Counter(decade(y) for _, y, _ in records)
    noise = Counter(decade(y) for _, y, c in records if c < 0)
    born = Counter(decade(y) for _, y, c in records
                   if c >= 0 and y is not None and births.get(c) == y)
    n_noise = sum(noise.values())
    return {
        "n_points": len(records), "n_outliers": n_noise,
        "outlier_ratio": n_noise / len(records) if records else None,
        "by_decade": {d: {"n_total": totals[d], "n_birth": born[d],
                           "birth_share": born[d] / totals[d],
                           "n_outliers": noise[d], "outlier_ratio": noise[d] / totals[d]}
                      for d in sorted(totals)},
    }


def compare_birth_curves(old, new):
    curve = {}
    for d in sorted(old["by_decade"].keys() | new["by_decade"].keys()):
        a = old["by_decade"].get(d, {}).get("birth_share")
        b = new["by_decade"].get(d, {}).get("birth_share")
        curve[d] = {"old": a, "new": b,
                    "change_percentage_points": 100 * (b - a) if a is not None and b is not None else None}
    return curve


def common_cohort_birth_shares(old, new, comparison, *, historical_labels_available=True):
    """Use an ID/year comparison already validated by cohort_comparison().

    Each partition retains its own fitted labels; label numbers are never matched
    across partitions. Neither scoring method refits features or reclusters.
    """
    if comparison is None:
        return {"status": "unavailable", "reason": "Historical assignments are required."}
    if not comparison["only_old_ids"] and not comparison["only_new_ids"]:
        return {"status": "not_needed_identical_id_sets", "n_common": comparison["shared_count"]}
    if not historical_labels_available:
        return {"status": "unavailable", "reason": "Historical graphlet labels are required; the production-derived ID cohort cannot supply them. Use --old-graphlet-assignments."}
    common = {r[0] for r in old} & {r[0] for r in new}
    if not common:
        return {"status": "unavailable", "reason": "The successful-ID populations have no overlap.", "n_common": 0}
    result = {
        "status": "available", "n_common": len(common),
        "scope": "Supplementary restriction of two fitted partitions to common IDs, with identical years and denominators. Labels and noise membership stay fixed; no feature fitting, graph rebuilding, or reclustering occurs. The partitions still depend on their different full fitting populations.",
        "full_map_birth_year": {
            "definition": "Keep each community's earliest known year from its full partition; score only shared IDs."},
        "common_cohort_birth_year": {
            "definition": "Recompute each community's earliest known year among shared IDs, retaining its fitted labels."},
        "birth_year_changes": {},
    }
    for name, records in (("old", old), ("new", new)):
        shared = [r for r in records if r[0] in common]
        full_births, shared_births = birth_years(records), birth_years(shared)
        result["full_map_birth_year"][name] = score_birth_rows(shared, full_births)
        result["common_cohort_birth_year"][name] = score_birth_rows(shared, shared_births)
        result["birth_year_changes"][name] = {
            "retained_communities_with_dated_members": len(shared_births),
            "birth_years_shifted_later": sum(y > full_births[c] for c, y in shared_births.items()),
        }
    for method in ("full_map_birth_year", "common_cohort_birth_year"):
        values = result[method]
        values["birth_curve_comparison"] = compare_birth_curves(values["old"], values["new"])
    return result


def check_feature_rows(run_dir, records, graphlets=False):
    """Read .npy headers and row-ID sidecars; never load feature arrays."""
    result = {}
    names = ["graphlet_cdf.npy"] if graphlets else ["features.npy", "features_pca.npy"]
    for name in names:
        path = run_dir / name
        if not path.exists():
            result[name] = {"checked": False, "reason": "matrix not present in this results copy"}
            continue
        with path.open("rb") as handle:
            if handle.read(6) != b"\x93NUMPY":
                raise ValueError(f"Invalid numpy header: {path}")
            major, minor = handle.read(2)
            if major not in (1, 2, 3):
                raise ValueError(f"Unsupported numpy format {major}.{minor}: {path}")
            width = 2 if major == 1 else 4
            size = struct.unpack("<H" if width == 2 else "<I", handle.read(width))[0]
            header = ast.literal_eval(handle.read(size).decode("utf-8" if major == 3 else "latin1").strip())
        shape = header["shape"]
        if len(shape) != 2 or shape[0] != len(records):
            raise ValueError(f"Feature/assignment row count mismatch: {path}")
        result[name] = {"checked": True, "shape": list(shape), "dtype": header["descr"]}
    ids = [r[0] for r in records]
    if graphlets:
        sidecar = run_dir / "graphlet_cdf.ids.json"
        if not sidecar.exists():
            sidecar = run_dir / "graphlet_cdf.npy.ids.json"
        if sidecar.exists():
            if read_json(sidecar) != ids:
                raise ValueError(f"CDF/assignment ID order mismatch: {sidecar}")
            result["id_order"] = {"checked": True, "source": str(sidecar)}
        else:
            result["id_order"] = {"checked": False, "reason": "CDF ID sidecar not present"}
    else:
        sample = run_dir / "sample_assignments.csv"
        if sample.exists():
            source = read_assignments(sample)
            if [(i, y) for i, y, _ in source] != [(i, y) for i, y, _ in records]:
                raise ValueError(f"Feature-source/community ID or year order mismatch: {sample}")
            result["id_order"] = {"checked": True, "source": str(sample)}
        else:
            result["id_order"] = {"checked": False, "reason": "sample assignments not present"}
    return result


def percent(value):
    return "unavailable" if value is None else f"{100 * value:.3f}%"


def markdown(report):
    lines = ["# Feature repair comparison", "", "Comparative facts only; no conclusion about validity or robustness is inferred.", "",
             "Birth shares include community outliers in each decade's denominator. Unknown-year entries are counted separately.", ""]
    for name, result in report["representations"].items():
        lines += [f"## {name}", ""]
        if "error" in result:
            lines += [f"Unavailable / validation failed: {result['error']}", ""]
            continue
        old, new = result["old"], result["new"]
        lines += [f"Cohort: {result['cohort_note']}", "",
                  "| Metric | Historical | Repaired |", "|---|---:|---:|",
                  f"| Successful feature rows | {old['n_points'] if old['n_points'] is not None else 'unavailable'} | {new['n_points']} |",
                  f"| Community outliers | {old['n_outliers'] if old['n_outliers'] is not None else 'unavailable'} | {new['n_outliers']} |",
                  f"| Outlier fraction | {percent(old['outlier_ratio'])} | {percent(new['outlier_ratio'])} |"]
        for d in ("1930s", "2010s"):
            a = old["by_decade"].get(d, {}).get("birth_share")
            b = new["by_decade"].get(d, {}).get("birth_share")
            lines.append(f"| {d} birth share | {percent(a)} | {percent(b)} |")
        cov = result["coverage"]
        lines += ["", f"Repaired featurization coverage: {cov.get('n_successful')} / {cov.get('n_attempted', 'unavailable')} attempted; failures: {cov.get('n_failed', 'unavailable')}."]
        changes = result.get("successful_id_comparison")
        if changes is None:
            lines += ["Historical successful-ID list is unavailable; no same-representation old/new ID comparison is asserted."]
            changes = result.get("historical_production_cohort_comparison")
        if changes:
            lines += [f"ID comparison reference: {changes['reference']}. Shared: {changes['shared_count']}; only historical: {len(changes['only_old_ids'])}; only repaired: {len(changes['only_new_ids'])}. Years match on the overlap."]
        unchecked = [k for k, v in result["feature_row_checks"].items() if not v["checked"]]
        if unchecked:
            lines += ["Unavailable row checks in this results copy: " + ", ".join(unchecked) + "."]
        lines += ["", "| Decade | Historical birth share | Repaired birth share | Change (pp) |", "|---|---:|---:|---:|"]
        for d, v in result["birth_curve_comparison"].items():
            delta = "unavailable" if v["change_percentage_points"] is None else f"{v['change_percentage_points']:+.3f}"
            lines.append(f"| {d} | {percent(v['old'])} | {percent(v['new'])} | {delta} |")
        if changes:
            lines += ["", "| Decade | Historical IDs | Repaired IDs | Only historical | Only repaired |", "|---|---:|---:|---:|---:|"]
            for d, v in changes["by_decade"].items():
                lines.append(f"| {d} | {v['old']} | {v['new']} | {v['only_old']} | {v['only_new']} |")
        shared = result["common_cohort_birth_shares"]
        lines += ["", "Supplementary common-ID birth shares:", ""]
        if shared["status"] == "not_needed_identical_id_sets":
            lines += ["Successful-ID sets are identical; no population restriction is needed."]
        elif shared["status"] != "available":
            lines += ["Unavailable: " + shared["reason"]]
        else:
            lines += [shared["scope"], "", f"Shared entries: {shared['n_common']:,}."]
            for method in ("full_map_birth_year", "common_cohort_birth_year"):
                values = shared[method]
                lines += ["", values["definition"], "",
                          "| Decade | Shared IDs | Historical birth share | Repaired birth share | Change (pp) |", "|---|---:|---:|---:|---:|"]
                for d, v in values["birth_curve_comparison"].items():
                    n = values["old"]["by_decade"][d]["n_total"]
                    lines.append(f"| {d} | {n} | {percent(v['old'])} | {percent(v['new'])} | {v['change_percentage_points']:+.3f} |")
            shifted = shared["birth_year_changes"]
            lines += ["", f"Community birth years shifted later after restriction: historical {shifted['old']['birth_years_shifted_later']}; repaired {shifted['new']['birth_years_shifted_later']}."]
        lines += [""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--new-run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--representations", nargs="+", choices=["production", "magpie", "graphlets"], default=["production", "magpie", "graphlets"])
    for name in ("production", "magpie", "graphlets"):
        parser.add_argument(f"--{name}-dir", type=Path)
    for name in ("production-summary", "production-assignments", "magpie-report", "magpie-assignments", "graphlet-summary", "graphlet-assignments"):
        parser.add_argument(f"--old-{name}", type=Path)
    args = parser.parse_args(argv)
    root = args.old_repo
    prod_path = args.old_production_assignments or root / "notes/icsd_community_assignments/community_assignments_labels3.csv"
    prod = read_assignments(prod_path)
    old_paths = {
        "production": args.old_production_summary or root / "notes/graph_time_summary.json",
        "magpie": args.old_magpie_report or root / "notes/ablation_paper_text_report.json",
        "graphlets": args.old_graphlet_summary or root / "experiments/graphlet_compare/results/graphlet_graph_time_summary.json",
    }
    report = {"old_repo": str(root), "new_run_root": str(args.new_run_root),
              "historical_production_assignments": str(prod_path), "representations": {}}
    for name in args.representations:
        try:
            run = getattr(args, name + "_dir") or args.new_run_root / name
            time_path = run / ("graph_time_summary.json" if name == "graphlets" else "time/graph_time_summary.json")
            assignment_path = run / ("graphlet_community_assignments.csv" if name == "graphlets" else "graph/community_assignments.csv")
            raw_new = read_json(time_path)
            new = temporal_stats(raw_new)
            records = read_assignments(assignment_path)
            validation = validate_rows(records, new, assignment_path)
            raw_old = read_json(old_paths[name])
            old_records = None
            if name == "magpie":
                old = {"n_points": None, "n_outliers": None, "outlier_ratio": None,
                       "n_common_with_historical_production": raw_old.get("n_common_total"),
                       "by_decade": {d: {"birth_share": v} for d, v in raw_old["decade_birth"]["ablation"].items()}}
                if args.old_magpie_assignments:
                    old_records = read_assignments(args.old_magpie_assignments)
                    old.update(n_points=len(old_records), n_outliers=sum(c < 0 for _, _, c in old_records))
                    old["outlier_ratio"] = old["n_outliers"] / old["n_points"] if old_records else None
                note = "Magpie representation; historical birth shares from the ablation report."
                if old_records is None:
                    note += " Historical totals and ID coverage require --old-magpie-assignments."
            else:
                old = temporal_stats(raw_old)
                if name == "production":
                    old_records = prod
                    note = f"All successfully featurized ICSD entries, including community noise and unknown years (historical: {len(prod):,})."
                elif args.old_graphlet_assignments:
                    old_records = read_assignments(args.old_graphlet_assignments)
                    note = "Graphlet replay, with supplied historical graphlet assignments."
                else:
                    old_records = [r for r in prod if r[2] >= 0 and r[1] is not None and 1900 <= r[1] <= 2025]
                    if len(old_records) != old["n_points"]:
                        raise ValueError("Historical graphlet count differs from the derived dated/non-noise cohort; supply --old-graphlet-assignments")
                    note = f"Historical graphlet cohort: {len(old_records):,} dated, non-noise production IDs. Derived from historical production membership; these are not historical graphlet labels."
            if old_records is not None and (name != "graphlets" or args.old_graphlet_assignments):
                validate_rows(old_records, old, old_paths[name])
            feature_summary = raw_new if name == "graphlets" else read_json(run / "summary.json")
            failed = feature_summary.get("n_failed", feature_summary.get("n_failures"))
            successful = feature_summary.get("n_successful", feature_summary.get("sample_size_featurized", new["n_points"]))
            if successful != new["n_points"]:
                raise ValueError(f"Feature success count differs from temporal rows: {run}")
            attempted = feature_summary.get("n_requested")
            if attempted is None and failed is not None:
                attempted = successful + failed
            coverage = {"n_successful": successful, "n_attempted": attempted, "n_failed": failed,
                        "success_fraction": successful / attempted if attempted else None,
                        "sample_size_requested": feature_summary.get("sample_size_requested")}
            curve = compare_birth_curves(old, new)
            old_assignment_source = (str(args.old_magpie_assignments) if name == "magpie" and args.old_magpie_assignments
                                     else str(args.old_graphlet_assignments) if name == "graphlets" and args.old_graphlet_assignments
                                     else str(prod_path) if name != "magpie" else None)
            result = {"sources": {"old_summary": str(old_paths[name]), "old_assignments": old_assignment_source,
                                  "new_summary": str(time_path), "new_assignments": str(assignment_path)},
                      "cohort_note": note, "old": old, "new": new, "coverage": coverage,
                      "feature_version": feature_summary.get("feature_version"),
                      "neighbor_settings": feature_summary.get("neighbor_settings"),
                      "assignment_validation": validation, "feature_row_checks": check_feature_rows(run, records, name == "graphlets"),
                      "birth_curve_comparison": curve,
                      "successful_id_comparison": cohort_comparison(old_records, records, f"historical {name} successful IDs") if old_records is not None else None}
            result["common_cohort_birth_shares"] = common_cohort_birth_shares(
                old_records, records, result["successful_id_comparison"],
                historical_labels_available=name != "graphlets" or bool(args.old_graphlet_assignments))
            if old_records is None:
                result["historical_production_cohort_comparison"] = cohort_comparison(prod, records, "historical production successful IDs; not asserted to be the full historical Magpie ID set")
            report["representations"][name] = result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["representations"][name] = {"error": str(exc)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "comparison.md").write_text(markdown(report) + "\n")
    print(f"Wrote {args.out_dir / 'comparison.json'} and {args.out_dir / 'comparison.md'}")
    return int(any("error" in v for v in report["representations"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
