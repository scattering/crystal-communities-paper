#!/usr/bin/env python3
"""Combined Figure 2 for the Nature submission: renaissance-community
validation in three panels (cuprate, CMR manganite, top-16 ranked).

Panels:
  (a) Cuprate community (community 6425, 378 members) year-histogram
      with Bednorz-Müller 1986 annotation.
  (b) CMR-manganite community (community 160, 571 members) year-histogram
      with Jin et al. 1994 annotation.
  (c) Top-16 step-change communities, with documented-event labels
      where identified.

Inputs are in notes/ and require no TACC access.
"""
from __future__ import annotations

import argparse
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
SURVEY_JSON = REPO / "notes/renaissance_survey_top.json"
OUT_FIG = REPO / "resources/figures/icsd_densification/fig2_renaissance_validation.png"

# Documented-event identifications for the top-16 communities (from §S8.4)
EVENT_IDS = {
    1: ("Sm-Fe-N magnets (Coey ~1990)", "magnets"),
    2: ("CMR manganites (Jin 1994)", "renaissance"),
    3: ("RE-light-metal binaries", None),
    4: ("Heavy fermion 1:1:2 (Steglich 1979 era)", "renaissance"),
    5: ("SOFC perovskite cathodes (mid-1990s)", "renaissance"),
    6: ("2:1:2 RE-In-Cu/Pd (Pöttgen-era survey)", "renaissance"),
    7: ("Doped MgAl2O4 spinels", None),
    8: ("Dilute magnetic semiconductors (Dietl 2000)", "renaissance"),
    9: ("MAX phases (Barsoum 1996+)", "renaissance"),
    10: ("RE-aluminate substrates (post high-Tc)", None),
    11: ("Double perovskites for spintronics (Kobayashi 1998)", "renaissance"),
    12: ("NaCoO2 thermoelectrics (Terasaki 1997)", "renaissance"),
    13: ("Corundum/ilmenite solid solutions", None),
    14: ("Double perovskites (companion to #11)", "renaissance"),
    15: ("Li-ion battery cathodes (Sony LiCoO2 1991)", "renaissance"),
    16: ("Layered Ruddlesden-Popper cuprates/manganites", "renaissance"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--community-assignments",
        default=str(COMMUNITY_ASSIGN),
        help="Path to community_assignments_labels3.csv.",
    )
    parser.add_argument(
        "--survey-json",
        default=str(SURVEY_JSON),
        help="Path to renaissance_survey_top.json.",
    )
    parser.add_argument(
        "--output",
        default=str(OUT_FIG),
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_records(path: Path):
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.append((int(row["icsd_id"]), int(row["year"]), int(row["community"])))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def per_year(records, comm_id):
    return Counter(y for _, y, c in records if c == comm_id)


def main() -> int:
    args = parse_args()
    community_assign = Path(args.community_assignments)
    survey_json = Path(args.survey_json)
    out_fig = Path(args.output)

    print("loading data...", flush=True)
    records = load_records(community_assign)
    survey = json.loads(survey_json.read_text())
    top = survey["top"]
    print(f"  {len(records)} records, {len(top)} top survey communities", flush=True)

    fig = plt.figure(figsize=(13.5, 9.0), dpi=170)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.4, wspace=0.18,
                          left=0.07, right=0.985, top=0.94, bottom=0.07)

    # Panel (a): cuprate community 6425
    ax_a = fig.add_subplot(gs[0, 0])
    hist = per_year(records, 6425)
    years = sorted(hist.keys())
    yrs_full = list(range(1925, 2020))
    counts = [hist.get(y, 0) for y in yrs_full]
    ax_a.bar(yrs_full, counts, width=0.85, color="#0f6d61", edgecolor="white", linewidth=0.3)
    ax_a.axvline(1986, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.85)
    ax_a.text(1986.4, max(counts) * 0.96, "Bednorz–Müller\n1986",
               color="#d62728", fontsize=9, fontweight="bold", va="top")
    n_pre = sum(c for y, c in hist.items() if 1976 < y <= 1986)
    n_post = sum(c for y, c in hist.items() if 1986 < y <= 1996)
    ax_a.set_title(f"(a) High-T$_c$ cuprate community (n=378) — pre = {n_pre}, post = {n_post} entries; community born after 1986",
                    fontsize=10, fontweight="bold", pad=4)
    ax_a.set_ylabel("New ICSD entries / yr")
    ax_a.set_xlim(1925, 2020)
    ax_a.grid(True, axis="y", color="#eee", linewidth=0.6)
    ax_a.set_axisbelow(True)
    for s in ("top", "right"): ax_a.spines[s].set_visible(False)

    # Panel (b): CMR manganite community 160
    ax_b = fig.add_subplot(gs[0, 1])
    hist = per_year(records, 160)
    counts = [hist.get(y, 0) for y in yrs_full]
    ax_b.bar(yrs_full, counts, width=0.85, color="#7a2cad", edgecolor="white", linewidth=0.3)
    ax_b.axvline(1994, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.85)
    ax_b.text(1994.4, max(counts) * 0.96, "Jin et al. 1994\n(CMR thin films)",
               color="#d62728", fontsize=9, fontweight="bold", va="top")
    n_pre = sum(c for y, c in hist.items() if 1984 < y <= 1994)
    n_post = sum(c for y, c in hist.items() if 1994 < y <= 2004)
    fold = (n_post / 10) / (n_pre / 10) if n_pre > 0 else float("inf")
    ax_b.set_title(f"(b) CMR-manganite community (n=571) — pre = {n_pre}, post = {n_post}; "
                    f"{fold:.0f}× fold-change",
                    fontsize=10, fontweight="bold", pad=4)
    ax_b.set_ylabel("New ICSD entries / yr")
    ax_b.set_xlim(1925, 2020)
    ax_b.grid(True, axis="y", color="#eee", linewidth=0.6)
    ax_b.set_axisbelow(True)
    for s in ("top", "right"): ax_b.spines[s].set_visible(False)

    # Panel (c): top-16 step-change ranking, horizontal bars
    ax_c = fig.add_subplot(gs[1, :])
    top16 = top[:16]
    # Reverse so #1 is on top
    top16_rev = list(reversed(top16))
    y_pos = np.arange(len(top16_rev))
    fold_values = []
    fold_labels = []
    bar_colors = []
    for r in top16_rev:
        if isinstance(r["fold"], str) or (isinstance(r["fold"], float) and r["fold"] == float("inf")):
            # Treat as max-on-chart for visualization, but label "∞"
            fold_values.append(150.0)
            fold_labels.append("∞")
        else:
            fold_values.append(min(r["fold"], 150.0))  # cap visualization at 150
            fold_labels.append(f"{r['fold']:.0f}×")
        # Color: renaissance vs. other
        rank = top.index(r) + 1
        kind = EVENT_IDS.get(rank, ("", None))[1]
        if kind == "renaissance":
            bar_colors.append("#0f6d61")
        elif kind == "magnets":
            bar_colors.append("#ff7e2a")
        else:
            bar_colors.append("#9aa5b1")

    ax_c.barh(y_pos, fold_values, color=bar_colors, edgecolor="white", linewidth=0.4)
    # Annotate each bar with rank, event name (if known), and fold value
    labels = []
    for r in top16_rev:
        rank = top.index(r) + 1
        event_name = EVENT_IDS.get(rank, ("(no clear event)", None))[0]
        labels.append(f"#{rank} ({r['event_year']}): {event_name}")
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels, fontsize=8.5)
    for i, (yp, fv, fl) in enumerate(zip(y_pos, fold_values, fold_labels)):
        ax_c.text(fv + 1, yp, fl, va="center", fontsize=7.5, color="#222")
    ax_c.set_xlabel("Pre/post fold-change (capped at 150 for visualization; ∞ = community born after event)")
    ax_c.set_title("(c) Top-16 communities by birth-year step-change — 9 of 16 identifiable communities map to documented field-defining events (teal). "
                    "Permanent magnets at rank #1 (orange).",
                    fontsize=10, fontweight="bold", pad=4)
    ax_c.set_xlim(0, 165)
    ax_c.grid(True, axis="x", color="#eee", linewidth=0.6)
    ax_c.set_axisbelow(True)
    for s in ("top", "right"): ax_c.spines[s].set_visible(False)

    fig.suptitle("Communities track field-defining scientific events",
                 fontsize=13, fontweight="bold", y=0.97)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
