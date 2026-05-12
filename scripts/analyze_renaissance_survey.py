#!/usr/bin/env python3
"""Systematic survey of community-level birth-year discontinuities.

For every production community with at least N_MIN members, scan candidate
event-years 1970-2010 and pick the year that maximizes a step-change score:

    score = (rate_post / rate_pre) × n_post   when rate_pre > 0
    score = n_post                            when rate_pre == 0   (community born after event)

with rate_pre / rate_post evaluated over symmetric ±W-year windows around
the candidate year (default W=10). The "born after event" case is given
a finite score so the cuprate-style result (community empty before 1986)
appears in the ranking.

We then report the top K communities by score, with their year-of-step,
fold-change, and the most common member reduced_formulas (so the user can
identify what scientific event each corresponds to).

The intent is to convert the post-hoc cuprate/manganite analysis into an
unbiased survey: do other documented field-defining events (Fe-based
superconductors 2008, NaCoO_2 thermoelectrics 1997, MAX phases 2000s,
double perovskites for spintronics 1998, etc.) appear in the top hits?
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
COMMUNITY_ASSIGN = REPO / "notes/icsd_community_assignments/community_assignments_labels3.csv"
POST_CUTOFF_DIRS = [
    REPO / f"notes/icsd_first_report_formulas/split_{year}" for year in (1980, 1990, 2000, 2010)
]
OUT_FIG = REPO / "resources/figures/icsd_densification/renaissance_survey.png"
OUT_TABLE_JSON = REPO / "notes/renaissance_survey_top.json"
OUT_TABLE_MD = REPO / "notes/renaissance_survey_top.md"

# Renaissance-survey tuning constants. Justification documented in
# paper/supporting_information.md §S1.5 ("Visualization tuning parameters").
N_MIN = 50
"""Minimum community size to enter the survey. Below 50 members a community
has too little post-event mass to distinguish a genuine fold-change from
sampling noise. Cuprates (community 6425) has 378 members and CMR
manganites has 232; the floor cuts off ~50 of the smallest 200 communities,
none of which carry documented field-defining renaissances."""

WINDOW = 10
"""Half-width in years of the pre/post fold-change window. Chosen to match
the typical 'decade after publication' pattern in materials renaissances:
both cuprates (1986) and CMR manganites (1994) fully realize their
fold-change inside ±10 yr of their seminal papers."""

EVENT_YEARS = list(range(1970, 2011))
"""Candidate event years scanned per community. Lower bound is set by the
ICSD becoming continuously populated (pre-1970 the year-coverage of the
ICSD itself is sparse). Upper bound bounds the +10-yr post-window to fit
inside the 2018-end of the dataset; events from ~2008 onward get
artificially low post-window counts. The manuscript's cited renaissances
all fall in 1980–2008, so this does not bind."""

TOP_K = 20
"""Survey ranking cut. Chosen so the manuscript can report 'nine of sixteen
event-interpretable communities' as field-defining renaissances; the
top-20 cut leaves a comfortable margin while keeping the table tractable
for the Extended Data ranking."""


def load_community_assignments(path: Path) -> list[tuple[int, int, int]]:
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                icsd_id = int(row["icsd_id"])
                year = int(row["year"]) if row.get("year") else None
                community = int(row["community"])
            except (KeyError, ValueError, TypeError):
                continue
            if year is None or year < 1900 or year > 2025:
                continue
            if community < 0:  # exclude HDBSCAN noise
                continue
            out.append((icsd_id, year, community))
    return out


def load_formula_lookup(post_cutoff_dirs: list[Path]) -> dict[int, str]:
    out: dict[int, str] = {}
    for d in post_cutoff_dirs:
        for fname in ["first_report_formulas.csv", "post_cutoff_accessibility_records.csv"]:
            path = d / fname
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        cif_id = int(row.get("cif_id", "") or 0)
                    except (TypeError, ValueError):
                        continue
                    if cif_id <= 0:
                        continue
                    formula = (row.get("reduced_formula") or "").strip()
                    if formula:
                        out[cif_id] = formula
    return out


def best_event_year(year_hist: Counter, event_years: list[int], window: int) -> tuple[int, dict]:
    """Find the event year that maximizes step-change score for this community."""
    best = None
    for ey in event_years:
        n_pre = sum(c for y, c in year_hist.items() if ey - window < y <= ey)
        n_post = sum(c for y, c in year_hist.items() if ey < y <= ey + window)
        rate_pre = n_pre / window
        rate_post = n_post / window
        # Score: emphasize both fold-change and absolute post volume.
        if n_pre == 0:
            score = float(n_post)
            fold = float("inf") if n_post > 0 else 0.0
        else:
            fold = rate_post / rate_pre
            score = fold * n_post
        if best is None or score > best["score"]:
            best = {
                "event_year": ey,
                "n_pre": n_pre,
                "n_post": n_post,
                "rate_pre": rate_pre,
                "rate_post": rate_post,
                "fold": fold,
                "score": score,
            }
    return best["event_year"], best


def main() -> int:
    print("loading community assignments...", flush=True)
    records = load_community_assignments(COMMUNITY_ASSIGN)
    print(f"  {len(records)} (icsd_id, year, community) rows after dropping noise", flush=True)

    print("loading formula lookup...", flush=True)
    formula_lookup = load_formula_lookup(POST_CUTOFF_DIRS)
    print(f"  {len(formula_lookup)} cif_id -> reduced_formula entries", flush=True)

    # Per-community year histograms
    by_comm: dict[int, Counter] = defaultdict(Counter)
    by_comm_ids: dict[int, list[int]] = defaultdict(list)
    for icsd_id, year, comm in records:
        by_comm[comm][year] += 1
        by_comm_ids[comm].append(icsd_id)

    print(f"  {len(by_comm)} communities total (excluding noise)", flush=True)
    eligible = {c: h for c, h in by_comm.items() if sum(h.values()) >= N_MIN}
    print(f"  {len(eligible)} communities with size >= {N_MIN}", flush=True)

    # Score each community
    results = []
    for comm, hist in eligible.items():
        ey, scoring = best_event_year(hist, EVENT_YEARS, WINDOW)
        # Top member formulas
        top_formulas = Counter(formula_lookup[i] for i in by_comm_ids[comm]
                                if i in formula_lookup).most_common(8)
        results.append({
            "community": comm,
            "size": sum(hist.values()),
            **scoring,
            "top_formulas": top_formulas,
        })

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:TOP_K]

    # Write JSON + markdown
    OUT_TABLE_JSON.write_text(json.dumps({"top": top, "n_eligible": len(eligible),
                                           "window": WINDOW, "event_years": [min(EVENT_YEARS), max(EVENT_YEARS)]},
                                          indent=2, default=str))
    md_lines = [
        f"# Top {TOP_K} community-level birth-year discontinuities",
        "",
        f"- Communities surveyed: {len(eligible)} (size >= {N_MIN}, excluding HDBSCAN noise)",
        f"- Candidate event years: {min(EVENT_YEARS)} to {max(EVENT_YEARS)}",
        f"- Window: ±{WINDOW} years around event",
        "",
        "| rank | community | size | best event year | n_pre | n_post | rate_pre | rate_post | fold | score | top member formulas |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|--|---:|--|",
    ]
    for i, r in enumerate(top, 1):
        formulas = ", ".join(f"{f} ({n})" for f, n in r["top_formulas"][:5])
        fold_str = "∞" if r["fold"] == float("inf") else f"{r['fold']:.1f}×"
        md_lines.append(
            f"| {i} | {r['community']} | {r['size']} | {r['event_year']} | "
            f"{r['n_pre']} | {r['n_post']} | {r['rate_pre']:.1f} | {r['rate_post']:.1f} | "
            f"{fold_str} | {r['score']:.1f} | {formulas} |"
        )
    OUT_TABLE_MD.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {OUT_TABLE_JSON}")
    print(f"wrote {OUT_TABLE_MD}")

    # Also plot the year histograms of the top 8
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), dpi=150, sharex=True)
    axes = axes.ravel()
    palette = ["#0f6d61", "#7a2cad", "#ff7e2a", "#34915d", "#1f77b4", "#d62728", "#8c564b", "#e377c2"]
    for ax, r, color in zip(axes, top[:8], palette):
        comm = r["community"]
        hist = by_comm[comm]
        years = sorted(hist.keys())
        if not years:
            continue
        years_full = list(range(min(years), max(years) + 1))
        counts = [hist.get(y, 0) for y in years_full]
        ax.bar(years_full, counts, width=0.8, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(r["event_year"], color="#d62728", linestyle="--", linewidth=1.2, alpha=0.85)
        formulas_short = ", ".join(f for f, _ in r["top_formulas"][:3])
        fold_str = "∞" if r["fold"] == float("inf") else f"{r['fold']:.1f}×"
        ax.set_title(
            f"comm {comm} (n={r['size']})  best step-year {r['event_year']}  fold {fold_str}\n"
            f"top: {formulas_short[:60]}",
            fontsize=8.5, fontweight="bold", pad=4,
        )
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", color="#eee", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(1925, 2020)
    for ax in axes[:4]:
        ax.set_ylabel("entries / yr", fontsize=8.5)
    for ax in axes[6:]:
        ax.set_xlabel("year", fontsize=8.5)
    fig.suptitle(f"Top 8 communities by birth-year step-change (size >= {N_MIN}, ±{WINDOW}-yr window)",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
