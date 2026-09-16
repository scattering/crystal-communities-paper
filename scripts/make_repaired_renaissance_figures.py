#!/usr/bin/env python3
"""Render repaired Fig2/ED3 using checked descriptions and explicit new inputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def history(ax, row, event, title, letter, *, compact=False):
    hist = {int(y): n for y, n in row["year_histogram"].items()}
    first = min(1930, min(hist))
    years = np.arange(first, 2016)
    ax.bar(years, [hist.get(int(y), 0) for y in years], width=.9, color="#0072B2", linewidth=0)
    ax.axvline(event, color="#D55E00", linestyle="--", linewidth=1.2)
    ax.set(xlim=(first-1, 2016))
    if compact:
        ax.set_xlabel("Publication year", fontsize=8.5, labelpad=2)
        ax.set_ylabel("Entries / year", fontsize=8.5, labelpad=2)
        ax.set_xticks([1940, 1960, 1980, 2000])
        ax.tick_params(labelsize=8, pad=2)
        ax.set_title(
            f"({letter}) {title}\n"
            f"c{row['community']}; n = {row['dated_community_members']:,}; reference {event}",
            fontsize=8.5, loc="left", pad=5,
        )
        ax.text(.03, .94, f"Pre/post: {row['n_pre']:,} / {row['n_post']:,}",
                transform=ax.transAxes, va="top", fontsize=8)
    else:
        ax.set(ylabel="New ICSD entries / year", xlabel="First-publication year")
        ax.set_title(f"({letter}) {title}\nCommunity {row['community']}; n = {row['dated_community_members']:,}; reference year {event}", fontsize=9.5, loc="left")
        ax.text(.03, .95, f"Pre/post entries: {row['n_pre']:,} / {row['n_post']:,}", transform=ax.transAxes, va="top", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)


def save(fig, out, stem):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--community-evidence", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probes = {v["probe"]: v for v in json.loads((args.community_evidence / "targeted_family_probe_candidates.json").read_text())["probes"]}
    with (args.community_evidence / "renaissance_top20_checked_descriptors.csv").open(newline="") as f:
        descriptors = list(csv.DictReader(f))
    # Design at the manuscript's physical display scale. The previous 14-inch
    # canvas shrank labels below 4 pt when embedded in a 5.83-inch Word column.
    fig = plt.figure(figsize=(6.5, 7.25))
    cu = probes["cuprates_high_tc"]
    mn = probes["cmr_manganites"]
    history(fig.add_axes([.105, .80, .365, .145]),
            cu["candidate_communities"][0], cu["event_year"],
            "YBa2Cu3O7-related cuprates", "a", compact=True)
    history(fig.add_axes([.62, .80, .365, .145]),
            mn["candidate_communities"][0], mn["event_year"],
            "Mn-seed mixed-oxide community", "b", compact=True)
    ax = fig.add_axes([.525, .072, .445, .635])
    top = descriptors[:20]
    scores = [float(r["score"]) for r in top]
    labels = [textwrap.fill(f"#{r['rank']} c{r['community']}: {r['description']}", 44)
              for r in top]
    y = np.arange(len(top))
    ax.barh(y, scores, color="#0072B2", height=.58)
    ax.set_yticks(y, labels, fontsize=8)
    for label in ax.get_yticklabels():
        label.set_linespacing(1.0)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", labelsize=8, pad=2)
    ax.set_xticks([0, 10000, 20000, 30000], ["0", "10,000", "20,000", "30,000"])
    for yi, row, score in zip(y, top, scores):
        mark = "*" if row["post_window_complete_through_2015"] == "False" else ""
        ax.text(score + max(scores)*.025, yi, f"{int(row['event_year'])}{mark}",
                fontsize=8, va="center")
    ax.set(xlim=(0, max(scores)*1.20), ylim=(len(top)-.35, -.65))
    ax.set_xlabel("Step-change score (post-count² / pre-count)", fontsize=8.5, labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.035, .735, "(c) All 20 highest-scoring communities", fontsize=10,
             weight="bold", va="bottom")
    fig.text(.035, .008, "* Post-window ends at the 2015 snapshot limit.", fontsize=8.5)
    save(fig, args.output_dir, "fig2_renaissance_validation_repaired")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), layout="constrained")
    selected = [("LaFeAsO_exact_parent", "LaFeAsO-anchored iron arsenide oxides"),
                ("tmd_exact_compositions", "Dominant exact-TMD-seed community"),
                ("RE_In_TM_212_exact_seeds", "La2InCu2/La2InPd2-anchored intermetallic community")]
    for ax, (key, title), letter in zip(axes, selected, "abc"):
        probe = probes[key]
        history(ax, probe["candidate_communities"][0], probe["event_year"], title, letter)
    fig.suptitle("Targeted composition-anchored temporal probes", fontsize=13)
    fig.supxlabel("Whole-community counts. Post-windows for 2008 and 2010 contain 7 and 5 available years; physical prototype identity is not inferred from the seed alone.", fontsize=9)
    save(fig, args.output_dir, "ed3_targeted_renaissance_probes_repaired")
    print(args.output_dir)


if __name__ == "__main__":
    main()
