#!/usr/bin/env python3
"""Year-shuffle null model for community birth-vs-densification dynamics.

Backs Extended Data Fig 1 (year-shuffle null). For each decade we report
the observed fraction of new ICSD entries that *open a new structural
community* vs *attach to an existing one*. To rule out a partition
artifact, this script holds the community partition fixed and randomly
permutes the publication-year column ``args.n_shuffles`` times,
recomputing the same per-decade ratios under each shuffle to obtain
5–95% null bands.

Inputs:
  --community-assignments    CSV with columns (icsd_id, year, community).
                             The same file consumed by every other
                             production analyzer.
  --n-shuffles               Number of year permutations (default 200).
  --seed                     Permutation seed (default 42); used by
                             ``random.Random``.

Outputs (under --output-dir):
  temporal_null_summary.json  Per-decade observed birth ratios with
                              5/50/95-percentile null bands.
  temporal_null_birth_ratio.png  In-script preview plot. The
                              publication PNG is rendered separately
                              by ``make_fig_temporal_null.py`` reading
                              the JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Timestamp-shuffle null for community birth/existing ratios.")
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, int]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                year = int(row["year"])
                community = int(row["community"])
                icsd_id = int(row["icsd_id"])
            except Exception:
                continue
            rows.append({"icsd_id": icsd_id, "year": year, "community": community})
    return rows


def decade_from_year(year: int) -> str:
    return f"{(year // 10) * 10}s"


def summarize(rows: list[dict[str, int]], years: list[int] | None = None) -> dict[str, dict[str, float]]:
    if years is None:
        years = [row["year"] for row in rows]
    by_comm: dict[int, int] = {}
    for row, year in zip(rows, years):
        comm = row["community"]
        if comm < 0:
            continue
        by_comm[comm] = min(year, by_comm.get(comm, year))

    by_decade: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row, year in zip(rows, years):
        decade = decade_from_year(year)
        by_decade[decade]["n_total"] += 1
        comm = row["community"]
        if comm < 0:
            by_decade[decade]["n_outlier"] += 1
            continue
        birth = by_comm.get(comm)
        if birth is None:
            continue
        if year == birth:
            by_decade[decade]["n_birth"] += 1
        elif year > birth:
            by_decade[decade]["n_existing"] += 1
    for decade, stats in by_decade.items():
        total = stats["n_total"] or 1.0
        stats["birth_ratio"] = stats["n_birth"] / total
        stats["existing_ratio"] = stats["n_existing"] / total
        stats["outlier_ratio"] = stats["n_outlier"] / total
    return {k: dict(v) for k, v in sorted(by_decade.items())}


def plot(decades: list[str], observed: list[float], low: list[float], high: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(decades))
    ax.plot(x, observed, color="#d62728", lw=2, label="Observed")
    ax.fill_between(x, low, high, color="#c7c7c7", alpha=0.5, label="Timestamp-shuffle 5-95%")
    ax.set_xticks(x)
    ax.set_xticklabels(decades, rotation=45, ha="right")
    ax.set_ylabel("Community-birth ratio")
    ax.set_title("Observed graph-community birth ratios exceed timestamp-shuffle null")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.community_assignments))
    observed = summarize(rows)
    decades = sorted(observed)
    rng = random.Random(args.seed)
    base_years = [row["year"] for row in rows]

    shuffle_births: dict[str, list[float]] = {d: [] for d in decades}
    for _ in range(args.n_shuffles):
        years = base_years[:]
        rng.shuffle(years)
        shuffled = summarize(rows, years)
        for decade in decades:
            shuffle_births[decade].append(float(shuffled.get(decade, {}).get("birth_ratio", 0.0)))

    envelope = {
        decade: {
            "observed_birth_ratio": float(observed[decade]["birth_ratio"]),
            "shuffle_mean_birth_ratio": float(np.mean(vals)),
            "shuffle_p05_birth_ratio": float(np.percentile(vals, 5)),
            "shuffle_p95_birth_ratio": float(np.percentile(vals, 95)),
        }
        for decade, vals in shuffle_births.items()
    }
    summary = {
        "n_rows": int(len(rows)),
        "n_shuffles": int(args.n_shuffles),
        "by_decade": envelope,
    }
    (out_dir / "temporal_null_summary.json").write_text(json.dumps(summary, indent=2))

    plot(
        decades,
        [envelope[d]["observed_birth_ratio"] for d in decades],
        [envelope[d]["shuffle_p05_birth_ratio"] for d in decades],
        [envelope[d]["shuffle_p95_birth_ratio"] for d in decades],
        out_dir / "temporal_null_birth_ratio.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
