#!/usr/bin/env python3
"""Chemistry of bridge attachments in the ICSD graph chronology.

Compares the elemental composition of structural-graph *bridge*
attachments (entries whose nearest-neighbor edges span ≥ 2 communities
when first inserted) against core attachments in the same chronology.
For each year ≥ ``--year-min`` the script tabulates per-element
enrichment in bridge vs. core attachments, and the per-event element
count distribution to test whether bridges are systematically more
chemically complex than core entries.

Inputs:
  --icsd-index    ICSD index CSV with formulas.
  --node-events   Per-entry node-event table (event_type,
                   is_bridge_attachment, year).
  --year-min      Lower year bound (default 1990).

Outputs (under ``--output-dir``):
  bridge_chemistry_summary.json       Per-element enrichment + complexity
                                       moments.
  bridge_element_enrichment.csv       Per-element bridge-vs-core odds.
  bridge_complexity_boxplot.png       Element count distributions.
  bridge_element_enrichment.png       Top enriched/depleted elements.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Composition


ID_COLUMN_CANDIDATES = ["cif_names", "ICSDid", "ICSD ID", "Collection Code"]
NAME_COLUMN_CANDIDATES = ["name", "formula", "Name", "Formula"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze chemistry of bridge attachments in ICSD graph chronology.")
    parser.add_argument("--icsd-index", required=True)
    parser.add_argument("--node-events", required=True)
    parser.add_argument("--year-min", type=int, default=1990)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def parse_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def load_metadata(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header in {path}")
        id_col = find_column(reader.fieldnames, ID_COLUMN_CANDIDATES)
        name_col = find_column(reader.fieldnames, NAME_COLUMN_CANDIDATES)
        if id_col is None or name_col is None:
            raise ValueError(f"Missing id/name columns in {path}")
        out = {}
        for row in reader:
            icsd_id = parse_int(row.get(id_col, ""))
            if icsd_id is None:
                continue
            out[icsd_id] = row.get(name_col, "").strip()
    return out


def parse_formula_stats(formula: str) -> tuple[int | None, list[str]]:
    try:
        comp = Composition(formula)
        els = sorted(el.symbol for el in comp.elements)
        return len(els), els
    except Exception:
        return None, []


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def top_enrichment(
    bridge_elements: Counter[str],
    core_elements: Counter[str],
    n_bridge: int,
    n_core: int,
    top_n: int = 15,
) -> list[dict[str, float | str]]:
    universe = sorted(set(bridge_elements) | set(core_elements))
    rows = []
    for el in universe:
        pb = (bridge_elements[el] + 1.0) / (n_bridge + 2.0)
        pc = (core_elements[el] + 1.0) / (n_core + 2.0)
        rows.append(
            {
                "element": el,
                "bridge_fraction": pb,
                "core_fraction": pc,
                "log2_enrichment": float(__import__("math").log2(pb / pc)),
            }
        )
    rows.sort(key=lambda x: (-x["log2_enrichment"], -x["bridge_fraction"]))
    return rows[:top_n]


def plot_complexity(bridge_values: list[float], core_values: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.boxplot([core_values, bridge_values], labels=["Core", "Bridge"], showfliers=False)
    ax.set_ylabel("Number of unique elements in formula")
    ax.set_title("Bridge attachments are chemically more complex")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_enrichment(rows: list[dict[str, float | str]], path: Path) -> None:
    labels = [str(r["element"]) for r in rows][::-1]
    values = [float(r["log2_enrichment"]) for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = ["#b2182b" if v > 0 else "#2166ac" for v in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("log2 enrichment in bridge nodes versus core nodes")
    ax.set_title("Elements enriched in post-1990 bridge attachments")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(Path(args.icsd_index))

    bridge_complexity: list[float] = []
    core_complexity: list[float] = []
    bridge_elements: Counter[str] = Counter()
    core_elements: Counter[str] = Counter()
    bridge_examples: list[dict[str, object]] = []
    core_examples: list[dict[str, object]] = []
    skipped = 0

    with Path(args.node_events).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            year = parse_int(row.get("year", ""))
            if year is None or year < args.year_min:
                continue
            if row.get("event_type") == "outlier":
                continue
            icsd_id = parse_int(row.get("icsd_id", ""))
            if icsd_id is None:
                continue
            formula = metadata.get(icsd_id, "")
            n_unique, elements = parse_formula_stats(formula)
            if n_unique is None:
                skipped += 1
                continue

            is_bridge = str(row.get("is_bridge_attachment", "")).lower() == "true"
            core_periphery = (row.get("core_periphery") or "").strip().lower()
            record = {
                "icsd_id": icsd_id,
                "year": year,
                "formula": formula,
                "n_unique_elements": n_unique,
                "community": parse_int(row.get("community", "")),
            }
            if is_bridge:
                bridge_complexity.append(float(n_unique))
                bridge_elements.update(set(elements))
                if len(bridge_examples) < 15:
                    bridge_examples.append(record)
            elif core_periphery == "core":
                core_complexity.append(float(n_unique))
                core_elements.update(set(elements))
                if len(core_examples) < 15:
                    core_examples.append(record)

    enriched = top_enrichment(
        bridge_elements,
        core_elements,
        len(bridge_complexity),
        len(core_complexity),
        top_n=15,
    )

    summary = {
        "year_min": args.year_min,
        "n_bridge": len(bridge_complexity),
        "n_core": len(core_complexity),
        "n_skipped_formula_parse": skipped,
        "bridge_complexity_mean": mean(bridge_complexity),
        "bridge_complexity_median": median(bridge_complexity),
        "core_complexity_mean": mean(core_complexity),
        "core_complexity_median": median(core_complexity),
        "top_enriched_elements": enriched,
        "bridge_examples": bridge_examples,
        "core_examples": core_examples,
    }
    (out_dir / "bridge_chemistry_summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "bridge_element_enrichment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0].keys()))
        writer.writeheader()
        writer.writerows(enriched)

    plot_complexity(bridge_complexity, core_complexity, out_dir / "bridge_complexity_boxplot.png")
    plot_enrichment(enriched, out_dir / "bridge_element_enrichment.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
