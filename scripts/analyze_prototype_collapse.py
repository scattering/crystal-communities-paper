#!/usr/bin/env python3
"""Quantify discrete-prototype collapse inside continuous structural communities.

For each community of size ≥ a small minimum, counts the number of
distinct ICSD-reported space groups represented and the dominant-space-
group fraction; small unique-space-group counts inside large
communities are evidence that the continuous Voronoi-graph embedding
is collapsing onto a small set of classical prototypes (i.e. the
communities are not just a re-coloring of the formula-anonymous
space-group label).

Inputs:
  --icsd-index           ICSD index CSV (formula + space group).
  --community-assignments  Production community labels.
  --top-n                Number of largest communities to highlight
                          in the plot (default 10).

Outputs (under ``--output-dir``):
  prototype_collapse_summary.json         Per-community unique-SG
                                           count + dominant-SG fraction.
  prototype_collapse_space_groups.png     Bar plot for the top-N.
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


ID_COLUMN_CANDIDATES = ["cif_names", "ICSDid", "ICSD ID", "Collection Code"]
SG_COLUMN_CANDIDATES = ["sym_group", "Space Group", "space_group", "Space_Group"]
NAME_COLUMN_CANDIDATES = ["name", "formula", "Name", "Formula"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify discrete prototype collapse inside continuous communities.")
    parser.add_argument("--icsd-index", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def load_index(path: Path) -> dict[int, dict[str, object]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header in {path}")
        id_col = find_column(reader.fieldnames, ID_COLUMN_CANDIDATES)
        sg_col = find_column(reader.fieldnames, SG_COLUMN_CANDIDATES)
        name_col = find_column(reader.fieldnames, NAME_COLUMN_CANDIDATES)
        if id_col is None or sg_col is None or name_col is None:
            raise ValueError(f"Missing required columns in {path}")
        out = {}
        for row in reader:
            icsd_id = parse_int(row.get(id_col, ""))
            if icsd_id is None:
                continue
            out[icsd_id] = {
                "space_group": parse_int(row.get(sg_col, "")),
                "formula": row.get(name_col, "").strip(),
            }
    return out


def load_assignments(path: Path) -> list[tuple[int, int]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            icsd_id = parse_int(row.get("icsd_id", ""))
            comm = parse_int(row.get("community", ""))
            if icsd_id is None or comm is None or comm < 0:
                continue
            rows.append((icsd_id, comm))
    return rows


def plot_unique_sg(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(r["community"]) for r in rows][::-1]
    values = [int(r["n_unique_space_groups"]) for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(labels, values, color="#4c78a8")
    ax.set_xlabel("Unique space groups within community")
    ax.set_ylabel("Community")
    ax.set_title("Prototype collapse: continuous communities span many space groups")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_index(Path(args.icsd_index))
    assignments = load_assignments(Path(args.community_assignments))

    community_sizes = Counter()
    community_sg: dict[int, Counter[int]] = defaultdict(Counter)
    community_formula: dict[int, Counter[str]] = defaultdict(Counter)

    for icsd_id, comm in assignments:
        community_sizes[comm] += 1
        entry = meta.get(icsd_id)
        if entry is None:
            continue
        sg = entry.get("space_group")
        formula = str(entry.get("formula", ""))
        if sg is not None:
            community_sg[comm][int(sg)] += 1
        if formula:
            community_formula[comm][formula] += 1

    top_rows = []
    for comm, size in community_sizes.most_common(args.top_n):
        sg_counts = community_sg.get(comm, Counter())
        formula_counts = community_formula.get(comm, Counter())
        top_rows.append(
            {
                "community": int(comm),
                "size": int(size),
                "n_unique_space_groups": int(len(sg_counts)),
                "top_space_groups": [
                    {"space_group": int(sg), "count": int(count)}
                    for sg, count in sg_counts.most_common(10)
                ],
                "n_unique_formulas": int(len(formula_counts)),
                "top_formulas": [
                    {"formula": formula, "count": int(count)}
                    for formula, count in formula_counts.most_common(10)
                ],
            }
        )

    summary = {
        "top_communities": top_rows,
    }
    (out_dir / "prototype_collapse_summary.json").write_text(json.dumps(summary, indent=2))
    plot_unique_sg(top_rows, out_dir / "prototype_collapse_space_groups.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
