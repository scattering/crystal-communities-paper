#!/usr/bin/env python3
"""Replace the prototype-collapse bar chart with one that uses chemical and
structural family labels instead of bare community IDs.

Reads:
  prototype_collapse_summary.json   →  per-community space-group counts
  canonical_family_names_labels3.csv →  curated family labels per community

For each top-N community, look up:
  - canonical_family_name  (when present, for the curated rows)
  - raw_label              (always present; e.g. "JBW zeolite family",
                            "Vanadium Gallide (1/1)")
  - community id           (last-resort identifier)

Bar label = canonical name → raw label → "community N", with a one-line
"N space groups" subtitle that the eye can read at a glance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prototype-summary", required=True)
    p.add_argument("--canonical-csv", required=True)
    p.add_argument("--inferred-csv", default=None,
                   help="optional output of scripts/infer_community_families.py; "
                        "used as a fallback when no curated canonical name exists")
    p.add_argument("--output", required=True)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--label-max-chars", type=int, default=42)
    return p.parse_args()


def load_canonical(path: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("kind") or "").strip() != "graph_community":
                continue
            try:
                comm = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            out[comm] = {
                "canonical": (row.get("canonical_family_name") or "").strip(),
                "raw_label": (row.get("raw_label") or "").strip(),
                "confidence": (row.get("confidence") or "").strip().lower(),
            }
    return out


def load_inferred(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    out: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                comm = int(row["community"])
            except (KeyError, TypeError, ValueError):
                continue
            out[comm] = {
                "family": (row.get("inferred_family") or "").strip(),
                "confidence": (row.get("confidence") or "").strip().lower(),
            }
    return out


def shorten(label: str, n: int) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def main() -> int:
    args = parse_args()
    summary = json.loads(Path(args.prototype_summary).read_text())
    rows = summary["top_communities"][: args.top_n]
    canonical = load_canonical(Path(args.canonical_csv))
    inferred = load_inferred(Path(args.inferred_csv) if args.inferred_csv else None)

    bar_labels: list[str] = []
    bar_values: list[int] = []
    bar_colors: list[str] = []
    bar_subtitle: list[str] = []

    # Color palette per source/confidence — curated names dominate, the
    # heuristic-inferred fallbacks ride a desaturated palette so the
    # reader can tell at a glance which labels are chemist-curated.
    curated_color = {"high": "#0f6d61", "medium": "#34915d", "low": "#b56200"}
    inferred_color = {"high": "#3a6b8a", "medium": "#6e83a4", "low": "#9b9bbd"}

    for r in rows:
        comm = int(r["community"])
        n_sg = int(r["n_unique_space_groups"])
        size = int(r["size"])
        rec = canonical.get(comm, {})
        canon = rec.get("canonical")
        raw = rec.get("raw_label")
        inf = inferred.get(comm, {})
        if canon and canon.lower() not in {"unknown", "n/a"}:
            label = canon
            color = curated_color.get(rec.get("confidence", ""), "#0f6d61")
        elif inf.get("family"):
            label = inf["family"] + " *"  # mark heuristic
            color = inferred_color.get(inf.get("confidence", ""), "#5b7a92")
        elif raw:
            label = raw
            color = "#5b6672"
        else:
            label = f"community {comm}"
            color = "#5b6672"
        bar_labels.append(shorten(label, args.label_max_chars))
        bar_values.append(n_sg)
        bar_colors.append(color)
        bar_subtitle.append(f"{n_sg} space groups · n = {size}")

    # Sort by space-group count so the eye sees the heaviest collapses first.
    order = np.argsort(bar_values)  # ascending → top of barh = largest
    bar_labels = [bar_labels[i] for i in order]
    bar_values = [bar_values[i] for i in order]
    bar_colors = [bar_colors[i] for i in order]
    bar_subtitle = [bar_subtitle[i] for i in order]

    fig, ax = plt.subplots(figsize=(11.0, 6.0), dpi=180)
    y = np.arange(len(bar_labels))
    ax.barh(y, bar_values, color=bar_colors, edgecolor="white", height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(bar_labels, fontsize=10)
    for i, (v, st) in enumerate(zip(bar_values, bar_subtitle)):
        ax.text(v + 1.2, i, st, va="center", fontsize=8.5, color="#5b6672")
    ax.set_xlabel("Unique crystallographic space groups within community")
    ax.set_xlim(0, max(bar_values) * 1.32)
    ax.set_title(
        "Continuous structural basins absorb dozens of discrete space groups",
        fontsize=12, fontweight="bold", pad=10,
    )
    # Reference line at the mean for visual anchor
    mean_sg = float(np.mean(bar_values))
    ax.axvline(mean_sg, color="#888", linestyle=":", linewidth=1.0)
    ax.text(mean_sg, len(bar_labels) - 0.1, f"  mean {mean_sg:.1f}", color="#888", fontsize=8.5, va="top")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#eee", linewidth=0.8)
    ax.set_axisbelow(True)
    if any("*" in lbl for lbl in bar_labels):
        ax.text(
            0.99, 0.02,
            "* heuristic family inferred from member-set (SG, stoichiometry); "
            "non-starred labels are chemist-curated",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#777", style="italic",
        )
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size} bytes; n={len(bar_labels)} communities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
