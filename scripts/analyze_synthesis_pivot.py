#!/usr/bin/env python3
"""Composition-matched and strict-polymorph robustness checks for the 𝒜ᵢ-vs-time
relation.

Reads the post-cutoff accessibility records emitted by
``analyze_synthesis_retrodiction.py`` and runs two pivots designed to
rule out trivial explanations of the 𝒜ᵢ-vs-first-report-year
correlation:

  1. *Composition-matched first reports*: group post-cutoff entries by
     anonymized composition class (Composition.anonymized_formula); for
     classes with at least ``--min-class-size`` entries, the
     class-centered first-report year is correlated against the
     class-centered 𝒜ᵢ. This removes the across-chemistry main effect.
  2. *Strict polymorph controls*: among reduced formulas reported in
     multiple distinct years (year gap ≥ ``--min-year-gap``) and whose
     entries fall in ≥ ``--min-community-changes`` distinct
     communities, asks whether the *first*-reported polymorph has lower
     𝒜ᵢ than later polymorphs of the same formula. This removes
     formula identity entirely.

Inputs:
  --scored-csv  post_cutoff_accessibility_records.csv from
                ``analyze_synthesis_retrodiction.py``.

Outputs (under ``--output-dir``):
  synthesis_pivot_summary.json             Class-centered Spearman ρ
                                            and strict-polymorph win
                                            rate.
  composition_matched_first_reports.csv    Per-class first-report rows.
  strict_polymorph_controls.csv            Per-formula sibling rows.
  synthesis_pivot_summary.png              QA scatter / bar plot.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Composition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composition-matched and strict-polymorph synthesis validation.")
    parser.add_argument("--scored-csv", required=True, help="post_cutoff_accessibility_records.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-class-size", type=int, default=25)
    parser.add_argument("--min-year-gap", type=int, default=2)
    parser.add_argument("--min-community-changes", type=int, default=2)
    return parser.parse_args()


def parse_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def parse_float(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    num = sum(a * b for a, b in zip(dx, dy))
    denx = sum(a * a for a in dx) ** 0.5
    deny = sum(b * b for b in dy) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return num / denx / deny


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def anonymized_formula(formula: str) -> str | None:
    try:
        return Composition(formula).anonymized_formula
    except Exception:
        return None


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            formula = (row.get("reduced_formula") or "").strip()
            year = parse_int(row.get("year", ""))
            score = parse_float(row.get("A_i", ""))
            comm = parse_int(row.get("assigned_community", ""))
            in_basin = parse_int(row.get("is_in_basin", ""))
            if not formula or year is None or score is None or comm is None or in_basin is None:
                continue
            anon = anonymized_formula(formula)
            if anon is None:
                continue
            try:
                comp = Composition(formula)
                n_elements = len(comp.elements)
            except Exception:
                continue
            rows.append(
                {
                    "cif_id": parse_int(row.get("cif_id", "")),
                    "reduced_formula": formula,
                    "year": year,
                    "A_i": score,
                    "assigned_community": comm,
                    "is_in_basin": in_basin,
                    "anonymized_formula": anon,
                    "n_elements": n_elements,
                    "class_key": f"{anon}|{n_elements}",
                }
            )
    return rows


def class_centered_first_reports(rows: list[dict[str, object]], min_class_size: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    first_by_formula: dict[str, dict[str, object]] = {}
    for row in rows:
        formula = str(row["reduced_formula"])
        prev = first_by_formula.get(formula)
        if prev is None or int(row["year"]) < int(prev["year"]):
            first_by_formula[formula] = row

    first_reports = list(first_by_formula.values())
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in first_reports:
        by_class[str(row["class_key"])].append(row)

    class_sizes = {k: len(v) for k, v in by_class.items()}
    usable_rows: list[dict[str, object]] = []
    for key, group in by_class.items():
        if len(group) < min_class_size:
            continue
        center = mean([float(r["A_i"]) for r in group])
        years = [int(r["year"]) for r in group]
        for row in group:
            out = dict(row)
            out["class_centered_A_i"] = float(row["A_i"]) - center
            out["class_year_rank"] = sorted(years).index(int(row["year"])) + 1
            usable_rows.append(out)
    return usable_rows, class_sizes


def strict_polymorph_rows(
    rows: list[dict[str, object]], min_year_gap: int, min_community_changes: int
) -> list[dict[str, object]]:
    by_formula: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_formula[str(row["reduced_formula"])].append(row)

    out: list[dict[str, object]] = []
    for formula, group in by_formula.items():
        years = sorted({int(r["year"]) for r in group})
        comms = sorted({int(r["assigned_community"]) for r in group})
        if len(years) < 2 or len(comms) < min_community_changes:
            continue
        first_year = min(years)
        later_years = [y for y in years if y >= first_year + min_year_gap]
        if not later_years:
            continue
        first_rows = [r for r in group if int(r["year"]) == first_year]
        later_rows = [r for r in group if int(r["year"]) in later_years]
        later_distinct = [r for r in later_rows if int(r["assigned_community"]) not in {int(fr["assigned_community"]) for fr in first_rows}]
        if not later_distinct:
            continue
        first_a = mean([float(r["A_i"]) for r in first_rows])
        later_a = mean([float(r["A_i"]) for r in later_distinct])
        first_basin = mean([float(r["is_in_basin"]) for r in first_rows])
        later_basin = mean([float(r["is_in_basin"]) for r in later_distinct])
        out.append(
            {
                "reduced_formula": formula,
                "first_year": first_year,
                "later_min_year": min(int(r["year"]) for r in later_distinct),
                "first_A_i_mean": first_a,
                "later_distinct_A_i_mean": later_a,
                "first_lower_A_i": first_a < later_a,
                "first_in_basin_mean": first_basin,
                "later_distinct_in_basin_mean": later_basin,
                "first_more_in_basin": first_basin > later_basin,
            }
        )
    return out


def make_plot(class_rows: list[dict[str, object]], strict_rows: list[dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    if class_rows:
        x = [float(r["class_centered_A_i"]) for r in class_rows]
        y = [float(r["year"]) for r in class_rows]
        axes[0].scatter(x, y, s=9, alpha=0.3, c="#1b9e77", linewidths=0)
        rho = spearman(x, y)
        axes[0].set_title("Composition-matched first reports")
        axes[0].set_xlabel("Class-centered accessibility")
        axes[0].set_ylabel("First report year")
        if rho is not None:
            axes[0].text(0.03, 0.97, f"Spearman rho = {rho:.3f}", transform=axes[0].transAxes, va="top")

    labels = ["first lower A_i", "first more in-basin"]
    values = [0.0, 0.0]
    if strict_rows:
        values[0] = mean([1.0 if bool(r["first_lower_A_i"]) else 0.0 for r in strict_rows])
        values[1] = mean([1.0 if bool(r["first_more_in_basin"]) else 0.0 for r in strict_rows])
    axes[1].bar(labels, values, color=["#7570b3", "#d95f02"])
    axes[1].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("Strict polymorph controls")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.scored_csv))
    class_rows, class_sizes = class_centered_first_reports(rows, min_class_size=args.min_class_size)
    strict_rows = strict_polymorph_rows(
        rows,
        min_year_gap=args.min_year_gap,
        min_community_changes=args.min_community_changes,
    )

    summary = {
        "n_rows": len(rows),
        "n_class_centered_first_reports": len(class_rows),
        "n_classes_total": len(class_sizes),
        "n_classes_ge_min_size": sum(1 for v in class_sizes.values() if v >= args.min_class_size),
        "spearman_class_centered_A_i_vs_year": spearman(
            [float(r["class_centered_A_i"]) for r in class_rows],
            [float(r["year"]) for r in class_rows],
        )
        if class_rows
        else None,
        "n_strict_polymorph_formulas": len(strict_rows),
        "strict_polymorph_first_lower_A_i_fraction": mean(
            [1.0 if bool(r["first_lower_A_i"]) else 0.0 for r in strict_rows]
        )
        if strict_rows
        else None,
        "strict_polymorph_first_more_in_basin_fraction": mean(
            [1.0 if bool(r["first_more_in_basin"]) else 0.0 for r in strict_rows]
        )
        if strict_rows
        else None,
        "min_class_size": int(args.min_class_size),
        "min_year_gap": int(args.min_year_gap),
        "min_community_changes": int(args.min_community_changes),
    }

    (out_dir / "synthesis_pivot_summary.json").write_text(json.dumps(summary, indent=2))

    if class_rows:
        with (out_dir / "composition_matched_first_reports.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(class_rows[0].keys()))
            writer.writeheader()
            writer.writerows(class_rows)

    if strict_rows:
        with (out_dir / "strict_polymorph_controls.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(strict_rows[0].keys()))
            writer.writeheader()
            writer.writerows(strict_rows)

    make_plot(class_rows, strict_rows, out_dir / "synthesis_pivot_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
