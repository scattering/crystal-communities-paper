#!/usr/bin/env python3
"""Validate communities on hand-curated prototype-family slices.

Sanity check that learned Louvain communities recover canonical
chemistry families. From the ICSD index the script extracts entries
whose stoichiometry + space group match curated rules for
classical families (perovskite-like, spinel-like, Heusler-like, etc.),
filters families to those with at least ``--min-family-size`` entries,
and reports per-family ARI / NMI plus the dominant community for each
family.

Inputs:
  --index-csv, --community-assignments, --min-family-size,
  --output-dir.

Outputs (under ``--output-dir``):
  benchmark_family_validation_summary.json  Per-family dominant-
                                             community fraction +
                                             global ARI / NMI.
  benchmark_family_validation_records.csv   Per-entry family tag and
                                             community.
  benchmark_family_validation.png            Bar plot.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate learned communities on hand-curated prototype-family slices.")
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-family-size", type=int, default=25)
    return parser.parse_args()


def parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def approx_ratio(values: list[float], target: list[float], tol: float = 0.05) -> bool:
    if len(values) != len(target):
        return False
    norm_values = sorted(v / sum(values) for v in values)
    norm_target = sorted(target)
    return all(abs(a - b) <= tol for a, b in zip(norm_values, norm_target))


def is_metal(symbol: str) -> bool:
    try:
        return bool(Element(symbol).is_metal)
    except Exception:
        return False


def family_label(formula: str, sg: int | None) -> str | None:
    if sg is None:
        return None
    try:
        comp = Composition(formula).reduced_composition
    except Exception:
        return None

    parts = comp.get_el_amt_dict()
    if not parts:
        return None
    elems = list(parts)
    amounts = list(parts.values())
    has_oxygen = "O" in parts

    if len(parts) == 2 and sg == 225 and approx_ratio(amounts, [0.5, 0.5]):
        return "rocksalt-like"
    if len(parts) == 2 and sg == 225 and approx_ratio(amounts, [1 / 3, 2 / 3]):
        return "fluorite-like"
    if len(parts) == 2 and sg == 136 and approx_ratio(amounts, [1 / 3, 2 / 3]):
        return "rutile-like"

    if has_oxygen and len(parts) == 3:
        o_amt = parts["O"]
        non_o = [amt for el, amt in parts.items() if el != "O"]
        if sg == 221 and approx_ratio(non_o + [o_amt], [0.2, 0.2, 0.6]):
            return "perovskite-like"
        if sg == 227 and approx_ratio(non_o + [o_amt], [1 / 7, 2 / 7, 4 / 7]):
            return "spinel-like"
        if sg == 227 and approx_ratio(non_o + [o_amt], [2 / 11, 2 / 11, 7 / 11]):
            return "pyrochlore-like"
        if sg == 166 and approx_ratio(non_o + [o_amt], [0.25, 0.25, 0.5]):
            return "delafossite-like"

    if not has_oxygen and len(parts) == 3 and all(is_metal(el) for el in elems):
        if sg == 225 and approx_ratio(amounts, [0.5, 0.25, 0.25]):
            return "full-Heusler-like"
        if sg == 216 and approx_ratio(amounts, [1 / 3, 1 / 3, 1 / 3]):
            return "half-Heusler-like"

    return None


def load_index(path: Path) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            icsd_id = parse_int(row.get("cif_names", ""))
            if icsd_id is None:
                continue
            out[icsd_id] = {
                "formula": row.get("name", ""),
                "space_group": parse_int(row.get("sym_group", "")),
            }
    return out


def load_assignments(path: Path) -> list[dict[str, int]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            icsd_id = parse_int(row.get("icsd_id", ""))
            community = parse_int(row.get("community", ""))
            if icsd_id is None or community is None or community < 0:
                continue
            rows.append({"icsd_id": icsd_id, "community": community})
    return rows


def summarize(rows: list[dict[str, object]], min_family_size: int) -> dict[str, object]:
    family_counts = Counter(str(r["family"]) for r in rows)
    family_to_comm: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        family_to_comm[str(row["family"])][int(row["community"])] += 1

    family_rows = []
    for family, count in family_counts.items():
        if count < min_family_size:
            continue
        comm_counts = family_to_comm[family]
        dominant_comm, dominant_count = comm_counts.most_common(1)[0]
        family_rows.append(
            {
                "family": family,
                "count": int(count),
                "n_communities": int(len(comm_counts)),
                "dominant_community": int(dominant_comm),
                "dominant_community_fraction": float(dominant_count / count),
            }
        )
    family_rows.sort(key=lambda row: (-row["count"], row["family"]))

    kept_families = {row["family"] for row in family_rows}
    kept_rows = [row for row in rows if row["family"] in kept_families]
    return {
        "n_labeled_rows": int(len(rows)),
        "n_families": int(len(kept_families)),
        "adjusted_rand_index": float(
            adjusted_rand_score([r["community"] for r in kept_rows], [r["family"] for r in kept_rows])
        )
        if kept_rows
        else 0.0,
        "normalized_mutual_info": float(
            normalized_mutual_info_score([r["community"] for r in kept_rows], [r["family"] for r in kept_rows])
        )
        if kept_rows
        else 0.0,
        "family_purity_mean": float(np.mean([r["dominant_community_fraction"] for r in family_rows])) if family_rows else 0.0,
        "family_purity_weighted_mean": float(
            np.average([r["dominant_community_fraction"] for r in family_rows], weights=[r["count"] for r in family_rows])
        )
        if family_rows
        else 0.0,
        "families": family_rows,
    }


def plot(summary: dict[str, object], out_path: Path) -> None:
    rows = list(summary["families"])
    if not rows:
        return
    labels = [str(r["family"]) for r in rows][::-1]
    purity = [float(r["dominant_community_fraction"]) for r in rows][::-1]
    counts = [int(r["count"]) for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, purity, color="#4c78a8")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Dominant-community fraction")
    ax.set_title("Coherence of hand-curated benchmark families")
    for bar, count in zip(bars, counts):
        ax.text(min(bar.get_width() + 0.01, 0.98), bar.get_y() + bar.get_height() / 2, f"n={count}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(Path(args.index_csv))
    assignments = load_assignments(Path(args.community_assignments))

    labeled_rows = []
    for row in assignments:
        meta = index.get(int(row["icsd_id"]))
        if not meta:
            continue
        family = family_label(str(meta["formula"]), meta["space_group"])
        if family is None:
            continue
        labeled_rows.append(
            {
                "icsd_id": int(row["icsd_id"]),
                "community": int(row["community"]),
                "family": family,
                "formula": str(meta["formula"]),
                "space_group": int(meta["space_group"]),
            }
        )

    summary = summarize(labeled_rows, args.min_family_size)
    (out_dir / "benchmark_family_validation_summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "benchmark_family_validation_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["icsd_id", "community", "family", "formula", "space_group"])
        writer.writeheader()
        writer.writerows(labeled_rows)
    plot(summary, out_dir / "benchmark_family_validation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
