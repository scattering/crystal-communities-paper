#!/usr/bin/env python3
"""Look for explosive growth in named communities at known scientific events.

Two test cases:

  - Cuprate high-Tc renaissance: Bednorz & Müller 1986 (Nobel Prize 1987).
    Look for step-up in the community containing La2CuO4 / YBa2Cu3O7-type
    cuprates after 1986.

  - CMR manganite renaissance: Jin et al. 1994 reported colossal
    magnetoresistance in La0.67Ca0.33MnO3 thin films, triggering a wave
    of follow-up work despite Goodenough's foundational 1950s-60s
    investigations. Look for step-up after ~1993.

Pipeline:
  1. Load production community_assignments.csv (icsd_id, year, community).
  2. Load post-cutoff CSVs (which carry reduced_formula) and build a
     formula -> {(icsd_id, year, community)} lookup for the post-1980 subset.
  3. Find ICSD entries whose reduced_formula matches a "seed" pattern for
     each renaissance — e.g. canonical cuprate formulas. The dominant
     production community among those seeds is the "renaissance community."
  4. From community_assignments.csv, retrieve all members of that community
     (all years, including pre-1980), and plot histogram of first-year by
     year, annotating the relevant scientific event.
  5. Quantify by comparing pre-event vs post-event annual rates.
"""
from __future__ import annotations

import csv
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
OUT_FIG = REPO / "resources/figures/icsd_densification/renaissance_communities.png"
OUT_JSON = REPO / "notes/renaissance_communities_summary.json"

# Seed formulas for each test case. We pick canonical exemplars whose
# production community we want to identify.
SEED_PATTERNS = {
    "cuprates_high_tc": {
        "label": "High-T_c cuprates",
        "event_year": 1986,
        "event_caption": "Bednorz–Müller 1986",
        # Bednorz-Müller 1986 reported high-Tc in La-Ba-Cu-O; YBa2Cu3O7
        # (YBCO, canonical Ba2YCu3O7 in pymatgen) followed in 1987. The
        # superconducting-cuprate family is dominated by Ba/Sr/La-cuprates
        # with layered perovskite-related stoichiometries.
        "seed_formulas": ["La2CuO4", "Ba2YCu3O7", "Bi2Sr2CaCu2O8", "Bi2Sr2Ca2Cu3O10"],
        # Looser element-set seed for the family: any reduced_formula
        # containing Cu AND (Ba or Sr or La or Y) AND O.
        "element_seed_required": {"Cu", "O"},
        "element_seed_any_of": {"Ba", "Sr", "La", "Y"},
    },
    "cmr_manganites": {
        "label": "CMR manganites",
        "event_year": 1994,
        "event_caption": "Jin et al. 1994 (CMR in La-Ca-MnO3 films)",
        # Jin 1994 reported CMR in La0.67Ca0.33MnO3 thin films, triggering
        # a wave of follow-up after foundational Goodenough et al. work in
        # the 1950s-60s. The manganite family is rare-earth + alkaline-earth
        # MnO3 perovskites.
        "seed_formulas": ["LaMnO3", "PrMnO3", "NdMnO3", "SmMnO3"],
        # Looser element-set seed: any formula containing Mn AND O AND a
        # rare-earth (La/Pr/Nd/Sm/Eu/Gd/Tb/Dy/Ho) or alkaline-earth (Ca/Sr/Ba).
        "element_seed_required": {"Mn", "O"},
        "element_seed_any_of": {"La", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
                                  "Ca", "Sr", "Ba"},
    },
}


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
            out.append((icsd_id, year, community))
    return out


def load_formula_lookup(post_cutoff_dirs: list[Path]) -> dict[int, str]:
    """Build cif_id -> reduced_formula for all entries in the post-cutoff
    CSVs (covers post-1980 ICSD)."""
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


def parse_elements(formula: str) -> set[str]:
    """Best-effort element extraction from a reduced_formula string."""
    import re
    return set(re.findall(r"[A-Z][a-z]?", formula))


def find_renaissance_community(
    seed_pattern: dict,
    community_records: list[tuple[int, int, int]],
    formula_lookup: dict[int, str],
) -> tuple[int, list[int]]:
    """Find the production community most populated by the seed family.

    Strategy: build a 'family' set of icsd_ids using both exact seed
    formulas AND a loose element-set filter. Then count community
    membership across that family, excluding HDBSCAN noise (community -1),
    and return the dominant community + ALL its members across all years.
    """
    exact_seeds = set(seed_pattern["seed_formulas"])
    required = set(seed_pattern.get("element_seed_required", set()))
    any_of = set(seed_pattern.get("element_seed_any_of", set()))

    # Step 1: build the family icsd_id set
    family_ids: set[int] = set()
    for cid, f in formula_lookup.items():
        if f in exact_seeds:
            family_ids.add(cid)
            continue
        elems = parse_elements(f)
        if required.issubset(elems) and (not any_of or elems & any_of):
            family_ids.add(cid)
    print(f"  family icsd_ids (exact + element-set): {len(family_ids)}", flush=True)

    # Step 2: production communities of those family members,
    # excluding noise (community -1).
    id_to_comm = {icsd_id: comm for icsd_id, _, comm in community_records}
    family_comms = [id_to_comm[i] for i in family_ids if i in id_to_comm and id_to_comm[i] >= 0]
    if not family_comms:
        return -1, []
    counter = Counter(family_comms)
    top3 = counter.most_common(3)
    top_comm, top_count = top3[0]
    print(f"  top 3 communities for {seed_pattern['label']}: {top3}", flush=True)

    # All members of the top community across all years (for the histogram).
    members_in_top = [i for i, _, c in community_records if c == top_comm]

    # Sanity check: how many community members are in our family vs not?
    members_in_family = [i for i in members_in_top if i in family_ids]
    print(f"  community {top_comm} has {len(members_in_top)} total members; "
          f"{len(members_in_family)} are family members "
          f"(by element-set seed; the rest are chemically related but "
          f"don't match the seed elements)", flush=True)

    return top_comm, members_in_top


def per_year_first_reports(community_records: list[tuple[int, int, int]],
                           comm_id: int) -> dict[int, int]:
    """For a given community, count NEW ICSD entries per year (i.e., the
    histogram of publication years of its members)."""
    bins = Counter()
    for icsd_id, year, community in community_records:
        if community == comm_id:
            bins[year] += 1
    return dict(bins)


def annual_rate_summary(year_hist: dict[int, int],
                        event_year: int,
                        pre_window: int = 10,
                        post_window: int = 10) -> dict[str, float]:
    """Compare mean annual report rate in (event-pre_window, event] to
    (event, event+post_window]."""
    pre = sum(c for y, c in year_hist.items() if event_year - pre_window < y <= event_year)
    post = sum(c for y, c in year_hist.items() if event_year < y <= event_year + post_window)
    rate_pre = pre / pre_window
    rate_post = post / post_window
    if pre == 0 and post == 0:
        fold_str = "n/a"
    elif pre == 0:
        # Community had no members in the pre-window: report as "post/0" symbolically
        fold_str = f"∞ (community born after {event_year})"
    else:
        fold_str = f"{rate_post / rate_pre:.1f}×"
    return {
        "pre_window": pre_window,
        "post_window": post_window,
        "n_pre": pre,
        "n_post": post,
        "rate_pre_per_year": rate_pre,
        "rate_post_per_year": rate_post,
        "fold_change_str": fold_str,
    }


def main() -> int:
    print("loading community assignments...", flush=True)
    records = load_community_assignments(COMMUNITY_ASSIGN)
    print(f"  {len(records)} (icsd_id, year, community) rows", flush=True)

    print("loading formula lookup from post-cutoff CSVs...", flush=True)
    formula_lookup = load_formula_lookup(POST_CUTOFF_DIRS)
    print(f"  {len(formula_lookup)} cif_id -> reduced_formula entries", flush=True)

    summary = {}

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=180, sharex=True)

    for ax, (key, pat) in zip(axes, SEED_PATTERNS.items()):
        print(f"\n=== {pat['label']} ===", flush=True)
        comm_id, members = find_renaissance_community(pat, records, formula_lookup)
        if comm_id < 0:
            print(f"  NO seed matches found; skipping", flush=True)
            continue

        year_hist = per_year_first_reports(records, comm_id)
        rate_summary = annual_rate_summary(year_hist, pat["event_year"], 10, 10)

        # Sample formulas of community members for sanity check
        member_formulas = [formula_lookup.get(i) for i in members if i in formula_lookup]
        member_formulas = [f for f in member_formulas if f]
        formula_counter = Counter(member_formulas).most_common(8)
        print(f"  top member formulas: {formula_counter}", flush=True)
        print(f"  pre-event ({pat['event_year']-10}-{pat['event_year']}): "
              f"{rate_summary['n_pre']} entries ({rate_summary['rate_pre_per_year']:.1f}/yr)", flush=True)
        print(f"  post-event ({pat['event_year']+1}-{pat['event_year']+10}): "
              f"{rate_summary['n_post']} entries ({rate_summary['rate_post_per_year']:.1f}/yr)", flush=True)
        print(f"  fold change: {rate_summary['fold_change_str']}", flush=True)

        years = sorted(year_hist.keys())
        if not years:
            continue
        years_full = list(range(min(years), max(years) + 1))
        counts = [year_hist.get(y, 0) for y in years_full]

        bar_color = "#0f6d61" if "cuprate" in key else "#7a2cad"
        ax.bar(years_full, counts, width=0.8, color=bar_color, edgecolor="white", linewidth=0.3)
        ax.axvline(pat["event_year"], color="#d62728", linestyle="--", linewidth=1.4, alpha=0.85)
        ax.text(pat["event_year"] + 0.5, max(counts) * 0.95,
                pat["event_caption"], color="#d62728", fontsize=9, va="top",
                fontweight="bold")
        ax.set_title(
            f"{pat['label']} — community {comm_id} (n = {len(members)})  "
            f"→ {rate_summary['rate_pre_per_year']:.1f}/yr × 10 yr pre-event,  "
            f"{rate_summary['rate_post_per_year']:.1f}/yr × 10 yr post-event,  "
            f"fold-change {rate_summary['fold_change_str']}",
            fontsize=10.5, fontweight="bold", pad=8,
        )
        ax.set_ylabel("New ICSD entries / year", fontsize=9.5)
        ax.set_xlim(1925, 2025)
        ax.grid(True, axis="y", color="#eee", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        summary[key] = {
            "label": pat["label"],
            "event_year": pat["event_year"],
            "event_caption": pat["event_caption"],
            "community_id": comm_id,
            "community_size_total": len(members),
            "year_histogram": {str(y): year_hist.get(y, 0) for y in years_full},
            "rate_summary": rate_summary,
            "top_member_formulas": formula_counter,
        }

    axes[-1].set_xlabel("Publication year", fontsize=10)
    fig.suptitle("Community-size growth around scientific events", fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    import json
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote figure to {OUT_FIG}")
    print(f"wrote summary to {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
