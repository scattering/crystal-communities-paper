#!/usr/bin/env python3
"""Summarize actual repaired source cohorts, failures, and MatterGen subgroups.

Writes explicit downstream outputs only; does not edit the manuscript or infer
that a full-map ordering establishes the held-out or composition-matched result.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from external_frozen_cohorts import EXPECTED_POOLS, record_key
from regenerate_external_projection import write_csv


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    coverage, families, changes = [], [], {}
    for source, pool in EXPECTED_POOLS.items():
        directory = args.external_root / source
        slug = "mattergen-public" if source == "mattergen" else source
        summary = json.loads((directory / f"{slug}_frontier_summary.json").read_text())
        rows = read_csv(directory / f"{slug}_frontier_records.csv")
        attempts = read_csv(directory / "attempted_cohort.csv")
        failures = json.loads((directory / f"{slug}_frontier_failures.json").read_text())
        ids = [record_key(r) for r in rows]
        attempted_ids = [record_key(r) for r in attempts]
        if (len(set(ids)) != len(ids) or len(set(attempted_ids)) != len(attempted_ids)
                or set(ids) | {record_key(r) for r in failures} != set(attempted_ids)
                or set(ids) & {record_key(r) for r in failures}
                or len(rows) + len(failures) != len(attempts)
                or summary["n_featurized"] != len(rows) or summary["n_failures"] != len(failures)
                or summary["cohort"]["candidate_pool_size"] != pool):
            raise ValueError(f"Incomplete or inconsistent attempted/scored/failure accounting: {source}")
        if ids != json.loads((directory / "feature_ids.json").read_text()):
            raise ValueError(f"Projection record/feature ID order mismatch: {source}")
        in_basin = [float(r["nearest_centroid_distance"]) <= float(r["community_threshold_p95"]) for r in rows]
        if any(flag != (r["in_basin"].lower() == "true") for flag, r in zip(in_basin, rows)):
            raise ValueError(f"Per-community classification mismatch: {source}")
        coverage.append({"source": source, "seed": 42, "candidate_pool": pool, "attempted": len(attempts),
                         "historical_scored": summary["cohort"]["n_historical_successful"],
                         "repaired_scored": len(rows), "failed": len(failures),
                         "projection_fraction": len(rows) / len(attempts), "n_in_basin": sum(in_basin),
                         "full_map_in_basin_fraction": sum(in_basin) / len(rows),
                         "failure_reasons": json.dumps(dict(Counter(r["reason"] for r in failures)), sort_keys=True)})
        changes[source] = summary["successful_id_changes"]
        if source == "mattergen":
            for level, key in (("family", lambda r: r["family"]),
                               ("family_task", lambda r: r["family"] + "/" + r["task"]),
                               ("all", lambda r: "all")):
                for group in sorted({key(r) for r in attempts}):
                    selected = [r for r in rows if key(r) == group]
                    n = len(selected)
                    k = sum(r["in_basin"].lower() == "true" for r in selected)
                    families.append({"level": level, "group": group,
                                     "attempted": sum(key(r) == group for r in attempts), "scored": n,
                                     "n_in_basin": k, "full_map_in_basin_fraction": k / n if n else None})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "external_projection_counts.csv", coverage)
    write_csv(args.out_dir / "mattergen_projection_by_family_task.csv", families)
    (args.out_dir / "external_successful_id_changes.json").write_text(json.dumps(changes, indent=2) + "\n")
    lines = ["# Repaired external-source accounting", "", "Full-map per-community p95 classification; these are not held-out rates.", "",
             "| Source | Candidate pool | Attempted | Historical scored | Repaired scored | Failed | In-basin |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in coverage:
        lines.append(f"| {row['source']} | {row['candidate_pool']} | {row['attempted']} | {row['historical_scored']} | {row['repaired_scored']} | {row['failed']} | {row['full_map_in_basin_fraction']:.3%} |")
    lines += ["", "The original releases, filters and attempted IDs remain fixed. Successful-set changes and complete failure reasons are in the accompanying JSON/CSV files. MatterGen generated and baseline categories remain separate."]
    (args.out_dir / "external_projection_counts.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(coverage, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
