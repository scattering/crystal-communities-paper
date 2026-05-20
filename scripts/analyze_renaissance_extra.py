#!/usr/bin/env python3
"""Targeted renaissance probes for events NOT covered by the systematic top-20.

Three new test cases asked for explicitly:

  1. Fe-based superconductors: Kamihara/Hosono 2008 LaFeAsO
     (the Fe-pnictide / Fe-chalcogenide superconductor family)

  2. 2D layered materials renaissance: post-graphene 2004, transition-metal
     dichalcogenides MoS2 / WSe2 / etc.; hexagonal boron nitride etc.

  3. Hybrid organic-inorganic halide perovskites for photovoltaics:
     Miyasaka 2009 (CH3NH3PbI3 first PV cell), Snaith/Park 2012-13 (efficiency
     leap). Family: methylammonium / formamidinium lead iodides/bromides.

Plus a deep-dive on the cuprate-precursor "2:1:2 intermetallic" community
2349 from the systematic survey (top members La2InCu2, La2InPd2, etc.)
that scored at rank 6 with a 1989 step.

Each probe identifies the dominant production community for its seed family
(excluding HDBSCAN noise), then plots the year-histogram of that community.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frontier_common import require_zenodo_file


REPO = Path(__file__).resolve().parent.parent
COMMUNITY_ASSIGN = REPO / "notes/icsd_community_assignments/community_assignments_labels3.csv"
POST_CUTOFF_DIRS = [
    REPO / f"notes/icsd_first_report_formulas/split_{year}" for year in (1980, 1990, 2000, 2010)
]
OUT_FIG = REPO / "resources/figures/icsd_densification/renaissance_extra.png"
OUT_JSON = REPO / "notes/renaissance_extra_summary.json"


def parse_elements(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", formula))


def load_records(community_assign: Path = None):
    out = []
    path = community_assign if community_assign is not None else COMMUNITY_ASSIGN
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.append((int(row["icsd_id"]), int(row["year"]), int(row["community"])))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def load_formula_lookup(post_cutoff_dirs: list[Path] = None) -> dict[int, str]:
    out = {}
    dirs = post_cutoff_dirs if post_cutoff_dirs is not None else POST_CUTOFF_DIRS
    for d in dirs:
        for fname in ["first_report_formulas.csv", "post_cutoff_accessibility_records.csv"]:
            path = d / fname
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        cid = int(row.get("cif_id", "") or 0)
                    except (TypeError, ValueError):
                        continue
                    if cid <= 0:
                        continue
                    f2 = (row.get("reduced_formula") or "").strip()
                    if f2:
                        out[cid] = f2
    return out


def find_dominant_community(records, formula_lookup, *, exact_seeds=None,
                              required_elems=None, any_of_elems=None,
                              substring_filter=None,
                              max_n_elements=None):
    """Return (top_community, member_icsd_ids, family_size).

    max_n_elements (optional): tighter constraint that the formula has at
    most this many distinct elements. Useful to exclude e.g. complicated
    5-element intermetallics from "Fe + As" Fe-pnictide search.
    """
    exact = set(exact_seeds or [])
    required = set(required_elems or [])
    any_of = set(any_of_elems or [])

    family_ids = set()
    has_loose_filter = bool(required) or bool(any_of) or bool(substring_filter)
    for cid, f in formula_lookup.items():
        if f in exact:
            family_ids.add(cid)
            continue
        if not has_loose_filter:
            # Exact-seed-only mode: skip the loose filter entirely.
            continue
        if substring_filter and substring_filter not in f:
            continue
        elems = parse_elements(f)
        if max_n_elements is not None and len(elems) > max_n_elements:
            continue
        if required.issubset(elems) and (not any_of or elems & any_of):
            family_ids.add(cid)

    id_to_comm = {cid: c for cid, _, c in records if c >= 0}
    family_comms = [id_to_comm[i] for i in family_ids if i in id_to_comm]
    if not family_comms:
        return None, [], len(family_ids)
    counter = Counter(family_comms)
    top_comm, top_count = counter.most_common(1)[0]
    members = [i for i, _, c in records if c == top_comm]
    return top_comm, members, len(family_ids)


def per_year(records, comm_id):
    return Counter(y for _, y, c in records if c == comm_id)


def annual_rate_summary(year_hist, event_year, window=10):
    n_pre = sum(c for y, c in year_hist.items() if event_year - window < y <= event_year)
    n_post = sum(c for y, c in year_hist.items() if event_year < y <= event_year + window)
    rate_pre = n_pre / window
    rate_post = n_post / window
    if n_pre == 0 and n_post == 0:
        fold = "n/a"
    elif n_pre == 0:
        fold = f"∞ (community born after {event_year})"
    else:
        fold = f"{rate_post / rate_pre:.1f}×"
    return {
        "n_pre": n_pre, "n_post": n_post,
        "rate_pre": rate_pre, "rate_post": rate_post,
        "fold_change_str": fold,
    }


PROBES = [
    {
        "key": "fe_pnictides",
        "label": "Fe-based superconductors — 1111-type LaFeAsO family (community 5450)",
        "event_year": 2008,
        "event_caption": "Kamihara/Hosono 2008 (LaFeAsO)",
        # The Fe-pnictide superconductor family splits across THREE sister
        # communities by structural prototype: 1111 (ZrCuSiAs, LaFeAsO),
        # 122 (ThCr2Si2, BaFe2As2), 111 (Cu2Sb, LiFeAs). The 122 community
        # is dominated by pre-existing ternary intermetallics that share
        # the prototype, so the post-2008 superconductor surge is diluted.
        # The 1111 community is the cleanest single-community signal.
        "force_community": 5450,
        "color": "#0f6d61",
    },
    {
        "key": "tmd_2d",
        "label": "Transition-metal dichalcogenides (post-graphene 2D-materials renaissance)",
        "event_year": 2010,
        "event_caption": "post-graphene 2D-materials boom (~2010–)",
        # Tight seed: exactly 2-element compounds with TM + chalcogen, AND
        # exact-seed list covers the famous canonical TMDs.
        "exact_seeds": ["MoS2", "MoSe2", "WS2", "WSe2", "MoTe2", "WTe2",
                          "ReS2", "ReSe2", "NbSe2", "NbS2", "TaS2", "TaSe2",
                          "TiS2", "TiSe2", "ZrS2", "ZrSe2", "HfS2", "HfSe2",
                          "VSe2", "PdSe2", "PtSe2"],
        "required_elems": set(),  # exact-seed match only
        "any_of_elems": set(),
        "max_n_elements": 2,  # exclude doped variants
        "color": "#7a2cad",
    },
    # NOTE: a hybrid organic-inorganic halide perovskite probe was previously
    # included here; it has been dropped because our ICSD snapshot ends with
    # publication-year 2015, before the photovoltaic renaissance triggered by
    # Snaith/Park 2012 matured (2016-2020). The renaissance is therefore a
    # snapshot-bounded non-test rather than something this figure can probe;
    # see Methods / SI §S8.5 for the disclosure. Restore the probe block with
    # a post-2020 ICSD snapshot if available.
    {
        "key": "comm_2349_deep_dive",
        "label": "Deep-dive: community 2349 (rank #6, La2InCu2 family, 1989 step)",
        "event_year": 1989,
        "event_caption": "rank-#6 community step year",
        # Force it to community 2349 directly
        "force_community": 2349,
        "color": "#34915d",
    },
]


def parse_args() -> argparse.Namespace:
    """CLI overrides for input/output paths (defaults match production layout)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--community-assignments", default=str(COMMUNITY_ASSIGN),
                        help="Path to community_assignments_labels3.csv.")
    parser.add_argument("--post-cutoff-dir", action="append", default=None,
                        help="Per-cutoff first-report directory. Repeatable. "
                             "Defaults to the four production splits 1980/1990/2000/2010.")
    parser.add_argument("--output-fig", default=str(OUT_FIG), help="Output PNG path.")
    parser.add_argument("--output-json", default=str(OUT_JSON), help="Output summary JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    community_assign = require_zenodo_file(
        args.community_assignments,
        what="canonical 167.5K-row ICSD community assignments table",
    )
    # post_cutoff_dirs is best-effort (load_formula_lookup skips missing
    # dirs). Don't gate on it.
    post_cutoff_dirs = [Path(p) for p in (args.post_cutoff_dir or POST_CUTOFF_DIRS)]
    out_fig = Path(args.output_fig)
    out_json = Path(args.output_json)

    print("loading data...", flush=True)
    records = load_records(community_assign)
    formula_lookup = load_formula_lookup(post_cutoff_dirs)
    print(f"  {len(records)} records, {len(formula_lookup)} formulas", flush=True)

    fig, axes = plt.subplots(len(PROBES), 1, figsize=(11, 3 * len(PROBES)), dpi=160, sharex=True)

    summary = {}
    for ax, probe in zip(axes, PROBES):
        print(f"\n=== {probe['label']} ===", flush=True)

        if "force_community" in probe:
            comm = probe["force_community"]
            members = [i for i, _, c in records if c == comm]
            family_size = "(forced)"
            print(f"  forcing community {comm} ({len(members)} members)", flush=True)
        else:
            comm, members, family_size = find_dominant_community(
                records, formula_lookup,
                exact_seeds=probe.get("exact_seeds"),
                required_elems=probe.get("required_elems"),
                any_of_elems=probe.get("any_of_elems"),
            )
            if comm is None:
                print("  NO seed matches found", flush=True)
                continue
            print(f"  family icsd_ids: {family_size}, dominant community {comm} "
                  f"({len(members)} total members)", flush=True)

        hist = per_year(records, comm)
        rs = annual_rate_summary(hist, probe["event_year"])
        member_formulas = Counter(formula_lookup[i] for i in members if i in formula_lookup).most_common(8)
        print(f"  pre-event ({probe['event_year']-10}-{probe['event_year']}): {rs['n_pre']} entries ({rs['rate_pre']:.1f}/yr)", flush=True)
        print(f"  post-event ({probe['event_year']+1}-{probe['event_year']+10}): {rs['n_post']} entries ({rs['rate_post']:.1f}/yr)", flush=True)
        print(f"  fold: {rs['fold_change_str']}", flush=True)
        print(f"  top member formulas: {member_formulas[:5]}", flush=True)

        years = sorted(hist.keys())
        if not years:
            continue
        years_full = list(range(min(years), max(years) + 1))
        counts = [hist.get(y, 0) for y in years_full]
        ax.bar(years_full, counts, width=0.8, color=probe["color"], edgecolor="white", linewidth=0.3)
        ax.axvline(probe["event_year"], color="#d62728", linestyle="--", linewidth=1.2, alpha=0.85)
        ymax = max(counts) if counts else 1
        ax.text(probe["event_year"] + 0.5, ymax * 0.95, probe["event_caption"],
                color="#d62728", fontsize=8.5, va="top", fontweight="bold")
        title = (f"{probe['label']} — community {comm} (n={len(members)})  "
                 f"→ {rs['rate_pre']:.1f}/yr pre, {rs['rate_post']:.1f}/yr post, fold {rs['fold_change_str']}")
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)
        ax.set_ylabel("entries / yr", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", color="#eee", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(1925, 2020)

        summary[probe["key"]] = {
            "label": probe["label"],
            "event_year": probe["event_year"],
            "community": comm,
            "size": len(members),
            "family_seed_size": family_size,
            "rate_summary": rs,
            "top_member_formulas": member_formulas,
        }

    axes[-1].set_xlabel("Publication year", fontsize=9)
    fig.suptitle("Renaissance probes: targeted families and a survey deep-dive", fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160, bbox_inches="tight")
    plt.close(fig)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out_fig}\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
