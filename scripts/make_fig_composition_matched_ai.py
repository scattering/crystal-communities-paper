#!/usr/bin/env python3
"""Render the composition-matched AI vs held-out ICSD figure (Extended Data Fig 2).

Reads the JSON written by ``analyze_composition_matched_ai.py`` and
produces the two-panel bar chart that the manuscript's Extended Data
Fig 2 caption describes. The plot block was originally inline at the
end of ``analyze_composition_matched_ai.py``; splitting it out brings
this figure into the same ``analyze → make_fig`` pattern used by the
other manuscript figures and lets a reviewer tweak styling without
rerunning the Wilson-CI computation.

Inputs:
  --summary  Path to ``composition_matched_ai_summary.json`` (the
             ``"cutoffs"`` key contains a list of per-cutoff blocks
             with ``matchings.{coarse,anonymized}.by_source.<NAME>.{
             unmatched,matched,icsd_matched_to_source,n_strata_common
             }``).
  --output   Path to the PNG to write.

Source order in the figure is taken from the order of keys under
``by_source`` of the first cutoff's ``coarse`` matching, preserving
the order the analyzer wrote them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Distinct color per source; the ICSD-matched companion bar uses a
# desaturated grey to keep the figure readable as the source count grows.
BASE_PALETTE = [
    "#0f6d61",  # MP / teal
    "#ff7e2a",  # GNoME / orange
    "#34915d",  # green
    "#7a2cad",  # MatterGen / purple
    "#1f77b4",  # JARVIS / blue
    "#d62728",  # Alexandria / red
    "#8c564b",  # spare
    "#e377c2",  # spare
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True,
                   help="Path to composition_matched_ai_summary.json")
    p.add_argument("--output", required=True,
                   help="Path to write the figure PNG")
    return p.parse_args()


def render(summary: dict, out_fig: Path) -> None:
    """Render the two-panel composition-matched figure.

    Two side-by-side panels, one per matching strategy (``coarse`` and
    ``anonymized``). Within each panel, one bar per (source, role)
    where role is "ICSD ↔ source-matched" (grey) or "source (matched
    strata)" (per-source color). The ICSD-matched bar is per-source
    because each source has its own stratum intersection with ICSD.
    """
    cutoff_results = summary["cutoffs"]
    cutoff_labels = [str(r["cutoff"]) for r in cutoff_results]
    matchers = [("coarse", None), ("anonymized", None)]

    # Recover source order from the analyzer's JSON. We use the first
    # cutoff's coarse-matching by_source dict, which preserves the order
    # the analyzer wrote them (Python dict insertion order is stable
    # across json.load).
    first_block = cutoff_results[0]["matchings"]["coarse"]["by_source"]
    source_names = list(first_block.keys())

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(11.0, 3.8 + 1.6 * len(source_names)), 5.0),
        dpi=180,
        sharey=True,
    )
    x = np.arange(len(cutoff_labels))
    n_sources = len(source_names)
    n_bars = 2 * n_sources
    w = 0.85 / max(n_bars, 1)

    for ax, (matcher_name, _) in zip(axes, matchers):
        for i, src_name in enumerate(source_names):
            ai_color = BASE_PALETTE[i % len(BASE_PALETTE)]
            # ICSD-matched bar at slot 2i, AI bar at slot 2i+1.
            for j, (role, color, key) in enumerate([
                (f"ICSD ↔ {src_name}-matched", "#5b6672", "icsd_matched_to_source"),
                (f"{src_name} (matched strata)", ai_color, "matched"),
            ]):
                slot = 2 * i + j
                off = (slot - (n_bars - 1) / 2.0) * w
                rates = np.array(
                    [(r["matchings"][matcher_name]["by_source"][src_name][key]["rate"] or 0)
                     for r in cutoff_results], dtype=float)
                cis = [r["matchings"][matcher_name]["by_source"][src_name][key]["ci95"]
                       for r in cutoff_results]
                # Wilson CI returns NaN for n=0 strata; clamp to zero for
                # the error bars so matplotlib doesn't drop the whisker.
                lo = np.array(
                    [c[0] if c[0] is not None and not np.isnan(c[0]) else 0 for c in cis])
                hi = np.array(
                    [c[1] if c[1] is not None and not np.isnan(c[1]) else 0 for c in cis])
                yerr_lo = np.maximum(rates - lo, 0)
                yerr_hi = np.maximum(hi - rates, 0)
                # Only legend the source itself once (the AI bar); the
                # ICSD-matched bar is consistent across sources visually
                # so we omit per-source legend entries to keep the
                # legend compact.
                label = role if (j == 1 or i == 0) else None
                ax.bar(x + off, rates, w, color=color, label=label,
                       edgecolor="white", linewidth=0.3)
                ax.errorbar(x + off, rates, yerr=[yerr_lo, yerr_hi], fmt="none",
                            ecolor="#222", elinewidth=0.5, capsize=1.5)

        # Annotate per-source common-strata counts above each cutoff.
        for i_cut, r in enumerate(cutoff_results):
            counts = " / ".join(
                f"{src_name[:3]}={r['matchings'][matcher_name]['by_source'][src_name]['n_strata_common']}"
                for src_name in source_names
            )
            ax.text(x[i_cut], 0.96, counts, ha="center", va="top",
                    fontsize=7, color="#5b6672")
        ax.set_xticks(x)
        ax.set_xticklabels(cutoff_labels)
        ax.set_xlabel("Training cutoff year")
        ax.set_title(f"{matcher_name} matching", fontsize=11,
                     fontweight="bold", pad=8)
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", color="#eee", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("In-basin rate (95th-percentile threshold)")
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False, ncol=1)
    fig.suptitle(
        "Composition-matched in-basin rates: AI vs held-out ICSD",
        fontsize=12, fontweight="bold", y=0.99,
    )

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figure to {out_fig}")


def main() -> int:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text())
    render(summary, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
