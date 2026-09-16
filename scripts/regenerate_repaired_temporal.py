#!/usr/bin/env python3
"""Regenerate temporal diagnostics in an explicit output directory.

Consumes completed repaired assignments/events. Reuses the released survey,
shuffle-null, and exclusive-event scoring functions; never copies historical
community names to new numeric IDs. Historical assignments are used only for
membership-overlap reconciliation. No CIFs, remote access, or feature fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import analyze_renaissance_null as renaissance_null
import analyze_renaissance_survey as survey
import analyze_temporal_null as temporal_null
from compare_feature_repair import read_assignments, temporal_stats, validate_rows, cohort_comparison
from compute_fig1_exclusive_events import classify

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def birth_tables(summary, events_path, records, out):
    rows = []
    for decade, v in sorted(summary["by_decade"].items()):
        rows.append({"decade": decade, "n_total": int(v["n_total"]),
                     "n_birth": int(v["n_cluster_birth_point"]), "n_outlier": int(v["n_outlier"]),
                     "birth_share": v["cluster_birth_point_ratio"], "outlier_share": v["outlier_ratio"],
                     "birth_plus_outlier_upper_bound": v["cluster_birth_point_ratio"] + v["outlier_ratio"]})
    write_csv(out / "outliers_and_birth_bound.csv", rows)
    if not events_path.exists():
        dump(out / "exclusive_events_status.json", {"status": "unavailable", "reason": "Per-node events were not saved for this representation."})
        return
    events = pd.read_csv(events_path, keep_default_na=False, low_memory=False)
    ids = events.icsd_id.astype(int).tolist()
    if len(ids) != len(set(ids)) or set(ids) != {r[0] for r in records}:
        raise ValueError("Temporal event IDs disagree with validated assignments")
    by_id = {i: (y, c) for i, y, c in records}
    for row in events.itertuples():
        year = int(row.year) if str(row.year).strip() else None
        if (year, int(row.community)) != by_id[int(row.icsd_id)]:
            raise ValueError("Temporal event year/label disagrees with assignments")
    if not set(events.event_type).issubset({"outlier", "community_birth", "existing_community"}):
        raise ValueError("Unrecognized temporal event class")
    events["class"] = classify(events)
    rows = []
    for decade, sub in events.groupby("decade", sort=True):
        counts = sub["class"].value_counts()
        n = len(sub)
        row = {"decade": decade, "n_total": n}
        for name in ("outlier", "birth", "same", "cross", "bridge"):
            row[f"n_{name}"] = int(counts.get(name, 0))
            row[f"share_{name}"] = row[f"n_{name}"] / n
        row["share_attach_incl_outlier"] = (row["n_same"] + row["n_cross"] + row["n_bridge"]) / n
        row["n_same_isolated"] = int(((sub["class"] == "same") & (sub.n_active_same_community_neighbors == 0)).sum())
        v = summary["by_decade"][decade]
        if (sum(row[f"n_{name}"] for name in ("outlier", "birth", "same", "cross", "bridge")) != n
                or n != v["n_total"] or row["n_birth"] != v["n_cluster_birth_point"]
                or row["n_outlier"] != v["n_outlier"]
                or row["n_same"] + row["n_cross"] + row["n_bridge"] != v["n_existing_cluster"]):
            raise ValueError(f"Exclusive-event counters disagree with summary in {decade}")
        rows.append(row)
    csv_path = out / "fig1_exclusive_by_decade.csv"
    write_csv(csv_path, rows)
    for extension in ("png", "svg"):
        subprocess.run([sys.executable, str(SCRIPTS / "make_fig_temporal_cliff_revised.py"),
                        "--exclusive-csv", str(csv_path), "--output", str(out / f"fig1_temporal_cliff.{extension}")], check=True)


def renaissance_tables(assignments, out):
    records = survey.load_community_assignments(assignments)
    histories = defaultdict(Counter)
    for _, year, community in records:
        histories[community][year] += 1
    eligible = sorted(c for c, hist in histories.items() if sum(hist.values()) >= survey.N_MIN)
    rows = []
    for c in eligible:
        _, scores = survey.best_event_year(histories[c], survey.EVENT_YEARS, survey.WINDOW)
        rows.append({"community": c, "size": sum(histories[c].values()), **scores})
    rows.sort(key=lambda row: row["score"], reverse=True)
    # JSON null represents an infinite fold when n_pre=0; counts and score remain explicit.
    for row in rows:
        row["fold"] = None if math.isinf(row["fold"]) else row["fold"]
    result = {"assignments": str(assignments), "label_status": "New community IDs only; family/event identities require membership and physical verification.",
              "n_dated_assigned": len(records), "n_eligible": len(eligible), "n_min": survey.N_MIN,
              "window": survey.WINDOW, "event_years": [1970, 2010],
              "pre_window": "event_year-W < year <= event_year", "post_window": "event_year < year <= event_year+W",
              "year_end": max(y for _, y, _ in records), "zero_pre_fold": "null denotes infinite fold when n_post > 0",
              "top": rows[:survey.TOP_K], "all_eligible": rows}
    dump(out / "renaissance_survey.json", result)
    write_csv(out / "renaissance_survey_all.csv", rows)
    md = ["# Repaired renaissance survey", "", result["label_status"], "",
          "Pre: (event−10, event]; post: (event, event+10]. Rates retain the survey's ten-year normalization.", "",
          "| Rank | New community | Dated members | Event year | Pre | Post | Fold | Score |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(rows[:20], 1):
        fold = "∞" if row["fold"] is None else f"{row['fold']:.2f}"
        md.append(f"| {rank} | {row['community']} | {row['size']} | {row['event_year']} | {row['n_pre']} | {row['n_post']} | {fold} | {row['score']:.2f} |")
    (out / "renaissance_survey.md").write_text("\n".join(md) + "\n")
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    for ax, row in zip(axes.flat, rows[:8]):
        hist = histories[row["community"]]
        ax.bar(list(hist), list(hist.values()), width=0.85, color="#555555")
        ax.axvline(row["event_year"], color="#0072B2", linestyle="--")
        ax.set_title(f"Community {row['community']} (n={row['size']}), step year {row['event_year']}", fontsize=10)
        ax.set_xlim(1925, 2020)
        ax.set_ylabel("Entries / year")
    fig.suptitle("Repaired top-eight step-change communities; family identities not yet curated")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(out / f"renaissance_survey_histograms.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    # Same released scorer under alternative half-widths and the audited gap convention.
    all_years = np.asarray([y for _, y, _ in records])
    all_comms = np.asarray([c for _, _, c in records])
    mask = np.isin(all_comms, eligible)
    lookup = {c: i for i, c in enumerate(eligible)}
    comm_idx = np.asarray([lookup[c] for c in all_comms[mask]])
    probes = []
    for window in (5, 10, 15):
        scorer = renaissance_null.VectorScorer(len(eligible), int(all_years.min()), int(all_years.max()), survey.EVENT_YEARS, window)
        for convention in ("production", "gap_year"):
            scores, years, pre, post = scorer.score(comm_idx, all_years[mask], convention)
            order = np.argsort(-scores, kind="stable")
            for rank, index in enumerate(order, 1):
                year = int(years[index])
                post_years = max(0, min(window, int(all_years.max()) - year))
                rate_post = int(post[index]) / post_years if post_years else None
                rate_pre = int(pre[index]) / window
                probes.append({"window": window, "convention": convention, "rank": rank,
                               "community": eligible[index], "event_year": year,
                               "n_pre": int(pre[index]), "n_post": int(post[index]), "score": float(scores[index]),
                               "pre_rate_survey": rate_pre, "post_rate_survey": int(post[index]) / window,
                               "available_post_years": post_years, "post_rate_available_years": rate_post,
                               "fold_available_years": rate_post / rate_pre if rate_pre and rate_post is not None else None})
    write_csv(out / "renaissance_window_gap_probes.csv", probes)
    return result


def run_renaissance_null(assignments, out, n_shuffles, seed):
    renaissance_null.COMMUNITY_ASSIGN = assignments
    renaissance_null.OUT_TABLE_JSON = out / "renaissance_survey.json"
    saved = sys.argv
    try:
        sys.argv = ["analyze_renaissance_null.py", "--n-shuffles", *map(str, n_shuffles),
                    "--seed", str(seed), "--out", str(out / "renaissance_null.json")]
        renaissance_null.main()
    finally:
        sys.argv = saved


def run_temporal_null(assignments, summary, out, n_shuffles, seed):
    null_dir = out / "year_shuffle"
    subprocess.run([sys.executable, str(SCRIPTS / "analyze_temporal_null.py"),
                    "--community-assignments", str(assignments), "--output-dir", str(null_dir),
                    "--n-shuffles", str(n_shuffles), "--seed", str(seed)], check=True)
    result_path = null_dir / "temporal_null_summary.json"
    result = json.loads(result_path.read_text())
    result["seed"] = seed
    result["population"] = "All dated successful entries, including the community outlier class; labels fixed."
    result["interval_definition"] = "5th to95th shuffle percentiles (central90% interval)"
    for decade, row in result["by_decade"].items():
        if not math.isclose(row["observed_birth_ratio"], summary["by_decade"][decade]["cluster_birth_point_ratio"], abs_tol=1e-12):
            raise ValueError("Observed temporal null curve disagrees with completed temporal summary")
        value = row["observed_birth_ratio"]
        row["position_relative_to_5th_95th_percentile_band"] = ("below" if value < row["shuffle_p05_birth_ratio"]
                                                   else "above" if value > row["shuffle_p95_birth_ratio"] else "inside")
    dump(result_path, result)
    # Render directly: old figure helper embeds historical regime claims and a fixed count of200.
    decades = list(result["by_decade"])
    rows = list(result["by_decade"].values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(decades, [v["shuffle_p05_birth_ratio"] for v in rows],
                    [v["shuffle_p95_birth_ratio"] for v in rows], color="#cccccc", label=f"Year-shuffle 5–95% ({n_shuffles} shuffles)")
    ax.plot(decades, [v["shuffle_mean_birth_ratio"] for v in rows], color="#666666", linestyle="--", label="Shuffle mean")
    ax.plot(decades, [v["observed_birth_ratio"] for v in rows], color="#0072B2", marker="o", label="Observed")
    ax.set(xlabel="Publication decade", ylabel="Community-birth share", ylim=(0, 1), title="Community-birth share and year-shuffle null")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(null_dir / f"temporal_null_birth_ratio.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def reconcile(old_path, new, out):
    old = read_assignments(old_path)
    comparison = cohort_comparison(old, new, str(old_path))
    old_map = {i: c for i, _, c in old}
    new_map = {i: c for i, _, c in new}
    old_sizes, new_sizes = Counter(old_map.values()), Counter(new_map.values())
    pairs = Counter((c, new_map.get(i)) for i, c in old_map.items())
    rows = [{"old_community": a, "new_community": b, "shared_members": n,
             "fraction_of_old_community": n / old_sizes[a],
             "fraction_of_new_community": n / new_sizes[b] if b is not None else None}
            for (a, b), n in sorted(pairs.items(), key=lambda p: (p[0][0], -p[1], -2 if p[0][1] is None else p[0][1]))]
    write_csv(out / "old_to_new_membership.csv", rows)
    common = sorted(old_map.keys() & new_map.keys())
    comparison["ARI_shared_including_noise"] = adjusted_rand_score([old_map[i] for i in common], [new_map[i] for i in common])
    comparison["NMI_shared_including_noise"] = normalized_mutual_info_score([old_map[i] for i in common], [new_map[i] for i in common])
    comparison["label_transfer"] = "None. Overlap is a curation aid, not physical verification of any new family label."
    dump(out / "membership_reconciliation.json", comparison)


def hdbscan_summary(sample_path, records, out):
    with sample_path.open(newline="") as handle:
        samples = list(csv.DictReader(handle))
    graph = {i: (y, c) for i, y, c in records}
    if {int(row["icsd_id"]) for row in samples} != graph.keys():
        raise ValueError("HDBSCAN sample IDs differ from Louvain IDs")
    rows = [{"icsd_id": int(r["icsd_id"]), "year": int(r["year"]) if r["year"] else None,
             "community": int(r["cluster"])} for r in samples]
    labels = [r["community"] for r in rows]
    result = {"source": str(sample_path), "n_rows": len(rows), "n_clusters": len({c for c in labels if c >= 0}),
              "n_outliers": sum(c < 0 for c in labels),
              "ARI_vs_Louvain": adjusted_rand_score(labels, [graph[r["icsd_id"]][1] for r in rows]),
              "NMI_vs_Louvain": normalized_mutual_info_score(labels, [graph[r["icsd_id"]][1] for r in rows]),
              "dated_birth_curve": temporal_null.summarize([r for r in rows if r["year"] is not None])}
    assigned = [r for r in rows if r["community"] >= 0 and graph[r["icsd_id"]][1] >= 0]
    result["agreement_scope"] = "Top-level ARI/NMI include the noise label; separate assigned-in-both metrics below."
    result["nonoutlier_both"] = {
        "n": len(assigned),
        "ARI": adjusted_rand_score([r["community"] for r in assigned], [graph[r["icsd_id"]][1] for r in assigned]),
        "NMI": normalized_mutual_info_score([r["community"] for r in assigned], [graph[r["icsd_id"]][1] for r in assigned]),
    }
    dump(out / "hdbscan_louvain_comparison.json", result)


def keyword_audit(flags_path, records, out):
    flags = {int(line.strip()) for line in flags_path.read_text().splitlines() if line.strip()}
    totals, flagged = Counter(), Counter()
    for iid, year, _ in records:
        decade = f"{year // 10 * 10}s" if year is not None else "unknown"
        totals[decade] += 1
        flagged[decade] += iid in flags
    rows = [{"decade": d, "n_total": n, "n_keyword_flagged": flagged[d], "fraction_keyword_flagged": flagged[d] / n}
            for d, n in sorted(totals.items())]
    dump(out / "theoretical_keyword_audit_intersection.json", {
        "source_flags": str(flags_path), "source_flags_sha256": file_hash(flags_path),
        "scope": "Intersection with saved CIF-header keyword flags; not a new CIF scan or a classification of structures as theoretical.",
        "n_entries": len(records), "n_flagged": sum(flagged.values()),
        "fraction_flagged": sum(flagged.values()) / len(records),
        "max_publication_year": max(y for _, y, _ in records if y is not None), "by_decade": rows})
    write_csv(out / "theoretical_keyword_audit_by_decade.csv", rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-run-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--old-production-assignments", required=True, type=Path)
    p.add_argument("--old-magpie-assignments", required=True, type=Path)
    p.add_argument("--old-graphlet-assignments", required=True, type=Path)
    p.add_argument("--representations", nargs="+", choices=["production", "magpie", "graphlets"], default=["production", "magpie", "graphlets"])
    p.add_argument("--steps", nargs="+", choices=["tables", "renaissance", "nulls", "reconciliation", "keyword-audit"], default=["tables", "renaissance", "nulls", "reconciliation"])
    p.add_argument("--theoretical-flags", type=Path, help="Saved header-keyword flag IDs for the keyword-audit step")
    p.add_argument("--temporal-shuffles", type=int, default=200)
    p.add_argument("--renaissance-shuffles", type=int, nargs="+", default=[200, 1000])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if "keyword-audit" in args.steps and args.theoretical_flags is None:
        p.error("--theoretical-flags is required for keyword-audit")
    for name in args.representations:
        run = args.new_run_root / name
        assignments = run / ("graphlet_community_assignments.csv" if name == "graphlets" else "graph/community_assignments.csv")
        summary_path = run / ("graph_time_summary.json" if name == "graphlets" else "time/graph_time_summary.json")
        summary = json.loads(summary_path.read_text())
        records = read_assignments(assignments)
        validation = validate_rows(records, temporal_stats(summary), assignments)
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        provenance = {"producer": str(Path(__file__).resolve()), "steps": args.steps,
                      "assignments": str(assignments), "assignments_sha256": file_hash(assignments),
                      "summary": str(summary_path), "summary_sha256": file_hash(summary_path),
                      "validation": validation, "seed": args.seed,
                      "source_sha256": {path.name: file_hash(path) for path in (Path(__file__), SCRIPTS / "analyze_renaissance_survey.py", SCRIPTS / "analyze_renaissance_null.py", SCRIPTS / "analyze_temporal_null.py", SCRIPTS / "compute_fig1_exclusive_events.py")},
                      "completed_steps": []}
        manifest_path = out / "temporal_regeneration_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if (previous["assignments_sha256"] != provenance["assignments_sha256"]
                    or previous["summary_sha256"] != provenance["summary_sha256"]):
                raise ValueError("Use a fresh output directory for changed temporal inputs")
            provenance["completed_steps"] = list(previous["completed_steps"])
            provenance["prior_executions"] = previous.get("prior_executions", []) + [
                {key: value for key, value in previous.items() if key != "prior_executions"}]
        dump(manifest_path, provenance)
        print(f"Starting {name}: {len(records):,} validated rows", flush=True)
        for step in args.steps:
            if step == "tables":
                birth_tables(summary, run / "time/node_temporal_events.csv", records, out)
                if name != "graphlets":
                    hdbscan_summary(run / "sample_assignments.csv", records, out)
            elif step == "renaissance":
                renaissance_tables(assignments, out)
            elif step == "nulls":
                run_renaissance_null(assignments, out, args.renaissance_shuffles, args.seed)
                run_temporal_null(assignments, summary, out, args.temporal_shuffles, args.seed)
            elif step == "reconciliation":
                old_path = getattr(args, f"old_{'graphlet' if name == 'graphlets' else name}_assignments")
                reconcile(old_path, records, out)
            elif step == "keyword-audit":
                keyword_audit(args.theoretical_flags, records, out)
            if step not in provenance["completed_steps"]:
                provenance["completed_steps"].append(step)
            dump(out / "temporal_regeneration_manifest.json", provenance)
            print(f"Completed {name}/{step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
