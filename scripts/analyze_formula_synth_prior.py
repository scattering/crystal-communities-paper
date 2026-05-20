#!/usr/bin/env python3
"""Synthesizability prior via reduced-formula overlap with ICSD.

For each external source (GNoME, MatterGen, MP, Alexandria, JARVIS), compute
the fraction of proposals whose reduced_formula appears in any ICSD entry.
A formula match is a strong "this composition has been made" signal — much
stronger than the geometric in-basin classification, which captures
structural similarity but says nothing about whether the composition itself
has ever been synthesized.

Cross-tabulated against in-basin status, this yields four quadrants per
source:

  ┌──────────────────────┬──────────────────────────┐
  │ in-basin, formula ✔  │ in-basin, formula ✗      │
  │ (highest synth prior)│ (novel composition,      │
  │                      │  structurally familiar)  │
  ├──────────────────────┼──────────────────────────┤
  │ frontier, formula ✔  │ frontier, formula ✗      │
  │ (familiar comp,      │ (lowest synth prior;     │
  │  structurally exotic)│  most exploratory)       │
  └──────────────────────┴──────────────────────────┘

ICSD reference is the union of post_cutoff first-report formulas, which
captures every composition first reported in ICSD after 1980 (~81K unique).
Pre-1980 ICSD compositions are not covered (caveat documented in writeup).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--icsd-formulas-dir", required=True,
                   help="Dir containing split_<year>/first_report_formulas.csv "
                   "for the per-cutoff first-report files. Union builds the "
                   "ICSD reference set (post-1980 ICSD, ~81K formulas).")
    p.add_argument("--icsd-formulas-splits", nargs="+", default=["1980", "1990", "2000", "2010"])
    p.add_argument("--source", action="append", nargs=2, metavar=("NAME", "PATH"),
                   help="External source records CSV. Repeatable.", required=True)
    p.add_argument("--per-community-thresholds",
                   help="Optional JSON with {'per_community_p95_threshold': "
                        "{community_id: threshold}}. When provided, the "
                        "outlier_like classification is recomputed per-record "
                        "from (assigned_community, nearest_centroid_distance) "
                        "against the community-specific threshold, overriding "
                        "the legacy outlier_like column in the records CSVs "
                        "(which was emitted under the pooled-threshold "
                        "convention). This is the canonical mode for the "
                        "Nature manuscript.")
    p.add_argument("--output-summary", required=True)
    p.add_argument("--output-table", required=True,
                   help="Markdown table for SI §S7 inclusion.")
    return p.parse_args()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def load_icsd_formulas(formulas_dir: Path, splits: Iterable[str]) -> dict[str, int]:
    """Return {reduced_formula: first_year}, taking the minimum first_year
    across the per-split files (since the same composition may appear in
    multiple splits as a later second/third entry)."""
    out: dict[str, int] = {}
    for s in splits:
        path = formulas_dir / f"split_{s}" / "first_report_formulas.csv"
        if not path.exists():
            print(f"  WARN: missing {path}", flush=True)
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                formula = (row.get("reduced_formula") or "").strip()
                if not formula:
                    continue
                try:
                    year = int(row["year"])
                except (KeyError, ValueError):
                    continue
                if formula not in out or year < out[formula]:
                    out[formula] = year
    return out


def load_external_records(
    path: Path,
    per_community_thr: dict[int, float] | None = None,
) -> list[dict]:
    """Load external records and optionally recompute outlier_like under
    per-community thresholds.

    If ``per_community_thr`` is provided, the legacy ``outlier_like`` column
    in the records CSV (emitted by the per-source frontier producers under
    the pooled-threshold convention) is ignored, and ``outlier_like`` is
    recomputed as ``nearest_centroid_distance > threshold[assigned_community]``.
    Records whose assigned community is missing from the threshold dict
    (e.g. HDBSCAN noise label = -1) are classified as outlier_like.
    """
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            formula = (row.get("reduced_formula") or "").strip()
            if not formula:
                continue
            if per_community_thr is None:
                outlier = (row.get("outlier_like") or "").strip().lower() == "true"
            else:
                try:
                    c = int(row["assigned_community"])
                    d = float(row["nearest_centroid_distance"])
                except (KeyError, ValueError):
                    outlier = True
                else:
                    thr = per_community_thr.get(c)
                    outlier = (thr is None) or (d > thr)
            rows.append({"formula": formula, "outlier_like": outlier})
    return rows


def main() -> int:
    args = parse_args()

    print("loading ICSD reference formulas...", flush=True)
    icsd_formulas = load_icsd_formulas(Path(args.icsd_formulas_dir), args.icsd_formulas_splits)
    icsd_formula_set = set(icsd_formulas.keys())
    print(f"  {len(icsd_formula_set)} unique ICSD reduced_formulas (post-1980 union)", flush=True)

    per_community_thr = None
    if args.per_community_thresholds:
        thr_data = json.loads(Path(args.per_community_thresholds).read_text())
        per_community_thr = {
            int(k): float(v)
            for k, v in thr_data["per_community_p95_threshold"].items()
        }
        print(f"  loaded {len(per_community_thr)} per-community thresholds; "
              f"pooled-equivalent (for reference) = "
              f"{thr_data.get('pooled_p95_threshold', 'unknown')}", flush=True)

    summary: dict[str, dict] = {
        "icsd_reference_size": len(icsd_formula_set),
        "icsd_reference_caveat": "Union of post-1980 ICSD first-report formulas; "
                                  "pre-1980 ICSD compositions are NOT included.",
        "threshold_mode": "per_community_p95" if per_community_thr else "pooled_p95_legacy",
        "sources": {},
    }

    table_lines = []
    table_lines.append(
        "| Source | N | formula∈ICSD | rate [95% CI] | "
        "in-basin & match | frontier & match | in-basin & no-match | frontier & no-match |"
    )
    table_lines.append("|--|---:|---:|--:|---:|---:|---:|---:|")

    for name, path in args.source:
        rows = load_external_records(Path(path), per_community_thr)
        n = len(rows)
        match = [(r["formula"] in icsd_formula_set, r["outlier_like"]) for r in rows]
        n_match = sum(1 for m, _ in match if m)
        # Quadrants
        ib_match = sum(1 for m, o in match if m and not o)
        fr_match = sum(1 for m, o in match if m and o)
        ib_nomatch = sum(1 for m, o in match if not m and not o)
        fr_nomatch = sum(1 for m, o in match if not m and o)
        rate = n_match / n if n else float("nan")
        lo, hi = wilson(n_match, n)
        summary["sources"][name] = {
            "n": n,
            "n_formula_in_icsd": n_match,
            "rate": rate,
            "ci95": [lo, hi],
            "quadrants": {
                "in_basin_and_formula_match": ib_match,
                "frontier_and_formula_match": fr_match,
                "in_basin_and_no_formula_match": ib_nomatch,
                "frontier_and_no_formula_match": fr_nomatch,
            },
        }
        table_lines.append(
            f"| {name} | {n} | {n_match} | {rate:.3f} [{lo:.3f}, {hi:.3f}] | "
            f"{ib_match} ({ib_match/n:.1%}) | {fr_match} ({fr_match/n:.1%}) | "
            f"{ib_nomatch} ({ib_nomatch/n:.1%}) | {fr_nomatch} ({fr_nomatch/n:.1%}) |"
        )
        print(f"  {name}: {n_match}/{n}={rate:.3f} formula-in-ICSD; "
              f"in-basin&match={ib_match} frontier&match={fr_match} "
              f"in-basin&no-match={ib_nomatch} frontier&no-match={fr_nomatch}",
              flush=True)

    Path(args.output_summary).write_text(json.dumps(summary, indent=2))
    Path(args.output_table).write_text("\n".join(table_lines) + "\n")
    print(f"\nwrote {args.output_summary} and {args.output_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
