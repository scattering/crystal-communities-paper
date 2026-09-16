#!/usr/bin/env python3
"""External validation of 𝒜ᵢ against the Kononova text-mined synthesis corpus.

Supports SI §S3.3. The Kononova et al. corpus provides reduced formulas
that have been *reported as synthesis targets in the literature*; this
script asks whether post-cutoff ICSD entries with a Kononova match
(``--kononova-json-xz``) have systematically *lower* 𝒜ᵢ scores than
post-cutoff entries with no Kononova match. A lower 𝒜ᵢ for the
Kononova-positive set would confirm that the score tracks
human-relevant synthesizability rather than just embedding distance.

Pipeline:
  1. Load Kononova targets from the .json.xz dump and convert parseable
     targets to normalized-first, scale-invariant composition keys.
  2. Read the post-cutoff scored records emitted by
     ``analyze_synthesis_retrodiction.py``.
  3. Retain the earliest post-cutoff report per scale-invariant formula and
     class-center 𝒜ᵢ within OPTIMADE-style anonymous stoichiometry classes
     (``--min-class-size`` per class) to avoid pure-chemistry
     confounding (mirrors ``analyze_synthesis_pivot.py``).
  4. Compare Kononova-positive vs. Kononova-negative class-centered
     𝒜ᵢ with a Kolmogorov-Smirnov statistic.

Inputs:
  --kononova-json-xz    Kononova synthesis corpus (.json.xz).
  --scored-csv          post_cutoff_accessibility_records.csv.

Outputs (under ``--output-dir``):
  kononova_validation_summary.json              KS statistic + group
                                                 means and counts.
  kononova_class_centered_first_reports.csv     Per-formula class-
                                                 centered rows.
  kononova_validation_summary.png               QA distribution plot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import lzma
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from formula_conventions import (
    anonymous_stoichiometry_key,
    scale_invariant_formula_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate accessibility score against Kononova text-mined synthesis targets.")
    parser.add_argument("--kononova-json-xz", required=True)
    parser.add_argument("--scored-csv", required=True, help="post_cutoff_accessibility_records.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-class-size", type=int, default=25)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def formula_key(text: str) -> tuple[tuple[str, int], ...] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return scale_invariant_formula_key(text)
    except Exception:
        return None


def anonymized_formula(text: str) -> tuple[int, ...] | None:
    try:
        return anonymous_stoichiometry_key(text)
    except Exception:
        return None


def ks_statistic(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    xa = np.sort(np.asarray(a, dtype=float))
    xb = np.sort(np.asarray(b, dtype=float))
    grid = np.sort(np.unique(np.concatenate([xa, xb])))
    cdfa = np.searchsorted(xa, grid, side="right") / len(xa)
    cdfb = np.searchsorted(xb, grid, side="right") / len(xb)
    return float(np.max(np.abs(cdfa - cdfb)))


def load_kononova_formulas(
    path: Path,
) -> tuple[set[tuple[tuple[str, int], ...]], dict[str, int]]:
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    formulas: set[tuple[tuple[str, int], ...]] = set()
    target_count = 0
    parse_failures = 0
    for rxn in data.get("reactions", []):
        target = rxn.get("target") or {}
        formula = target.get("material_formula") or target.get("material_string")
        if not formula:
            candidates = rxn.get("targets_string")
            if isinstance(candidates, list) and candidates:
                formula = candidates[0]
            elif isinstance(candidates, str):
                formula = candidates
        if formula:
            target_count += 1
        key = formula_key(formula or "")
        if key:
            formulas.add(key)
        elif formula:
            parse_failures += 1
    return formulas, {
        "reactions": len(data.get("reactions", [])),
        "target_records": target_count,
        "target_parse_failures": parse_failures,
    }


def load_first_reports(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            display_formula = (row.get("reduced_formula", "") or "").strip()
            key = formula_key(display_formula)
            score = parse_float(row.get("A_i", ""))
            year = parse_int(row.get("year", ""))
            if key is None or score is None or year is None:
                continue
            anon = anonymized_formula(display_formula)
            if anon is None:
                continue
            rows.append(
                {
                    "reduced_formula": display_formula,
                    "formula_key": key,
                    "A_i": score,
                    "year": year,
                    "class_key": f"{anon}|{len(anon)}",
                }
            )

    first_by_formula: dict[tuple[tuple[str, int], ...], dict[str, object]] = {}
    for row in rows:
        key = row["formula_key"]
        prev = first_by_formula.get(key)
        if prev is None or int(row["year"]) < int(prev["year"]):
            first_by_formula[key] = row
    return list(first_by_formula.values())


def class_center_rows(rows: list[dict[str, object]], min_class_size: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["class_key"])].append(row)
    class_sizes = {k: len(v) for k, v in by_class.items()}
    out: list[dict[str, object]] = []
    for key, group in by_class.items():
        if len(group) < min_class_size:
            continue
        center = mean([float(r["A_i"]) for r in group])
        for row in group:
            new = dict(row)
            new["class_centered_A_i"] = float(row["A_i"]) - center
            out.append(new)
    return out, class_sizes


def add_kononova_labels(
    rows: list[dict[str, object]],
    positives: set[tuple[tuple[str, int], ...]],
) -> list[dict[str, object]]:
    out = []
    for row in rows:
        new = dict(row)
        new["in_kononova"] = int(row["formula_key"] in positives)
        out.append(new)
    return out


def make_plot(pos_vals: list[float], neg_vals: list[float], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    if pos_vals or neg_vals:
        bins = np.linspace(
            min(pos_vals + neg_vals) if (pos_vals or neg_vals) else -1,
            max(pos_vals + neg_vals) if (pos_vals or neg_vals) else 1,
            40,
        )
        if neg_vals:
            axes[0].hist(neg_vals, bins=bins, alpha=0.6, density=True, label="not in Kononova", color="#7570b3")
        if pos_vals:
            axes[0].hist(pos_vals, bins=bins, alpha=0.6, density=True, label="in Kononova", color="#1b9e77")
        axes[0].set_xlabel("Class-centered accessibility")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Kononova positives vs controls")
        axes[0].legend(frameon=False)

    groups = ["bottom decile", "top decile"]
    rates = [0.0, 0.0]
    pooled = sorted(pos_vals + neg_vals)
    if pooled:
        lo = np.percentile(pooled, 10)
        hi = np.percentile(pooled, 90)
        values = np.asarray(pos_vals + neg_vals, dtype=float)
        labels = np.asarray([1.0] * len(pos_vals) + [0.0] * len(neg_vals))
        bottom = labels[values <= lo]
        top = labels[values >= hi]
        rates[0] = float(bottom.mean()) if len(bottom) else 0.0
        rates[1] = float(top.mean()) if len(top) else 0.0
    axes[1].bar(groups, rates, color=["#1b9e77", "#d95f02"])
    axes[1].set_ylim(0, max(0.05, max(rates) * 1.2))
    axes[1].set_ylabel("Kononova positive fraction")
    axes[1].set_title("Positive rate by accessibility decile")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kononova_path = Path(args.kononova_json_xz)
    scored_path = Path(args.scored_csv)
    positives, corpus_stats = load_kononova_formulas(kononova_path)
    rows = load_first_reports(scored_path)
    rows, class_sizes = class_center_rows(rows, min_class_size=args.min_class_size)
    rows = add_kononova_labels(rows, positives)

    pos = [float(r["class_centered_A_i"]) for r in rows if int(r["in_kononova"]) == 1]
    neg = [float(r["class_centered_A_i"]) for r in rows if int(r["in_kononova"]) == 0]
    pooled = [float(r["class_centered_A_i"]) for r in rows]
    dec10 = float(np.percentile(pooled, 10)) if pooled else None
    dec90 = float(np.percentile(pooled, 90)) if pooled else None
    bottom = [r for r in rows if dec10 is not None and float(r["class_centered_A_i"]) <= dec10]
    top = [r for r in rows if dec90 is not None and float(r["class_centered_A_i"]) >= dec90]

    summary = {
        "method": {
            "formula_identity": "element-collapsed normalized-first scale_invariant_formula_key",
            "composition_class": "OPTIMADE-style anonymous_stoichiometry_key",
            "first_report": "earliest year per scale-invariant composition key",
            "min_class_size": int(args.min_class_size),
            "pymatgen_version": importlib.metadata.version("pymatgen"),
        },
        "inputs": {
            "kononova_json_xz": str(kononova_path.resolve()),
            "kononova_sha256": sha256_file(kononova_path),
            "scored_csv": str(scored_path.resolve()),
            "scored_csv_sha256": sha256_file(scored_path),
        },
        "corpus_parse": corpus_stats,
        "n_kononova_unique_formulas": len(positives),
        "n_first_reports_considered": len(rows),
        "n_classes_ge_min_size": sum(1 for v in class_sizes.values() if v >= args.min_class_size),
        "n_kononova_positive_first_reports": sum(int(r["in_kononova"]) for r in rows),
        "class_centered_A_i_mean_kononova": mean(pos) if pos else None,
        "class_centered_A_i_mean_controls": mean(neg) if neg else None,
        "class_centered_A_i_ks": ks_statistic(pos, neg),
        "kononova_positive_fraction_bottom_decile": mean([float(r["in_kononova"]) for r in bottom]) if bottom else None,
        "kononova_positive_fraction_top_decile": mean([float(r["in_kononova"]) for r in top]) if top else None,
        "bottom_to_top_positive_rate_ratio": (
            (mean([float(r["in_kononova"]) for r in bottom]) / max(mean([float(r["in_kononova"]) for r in top]), 1e-9))
            if bottom and top
            else None
        ),
        "min_class_size": int(args.min_class_size),
    }

    (out_dir / "kononova_validation_summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "kononova_class_centered_first_reports.csv").open("w", newline="", encoding="utf-8") as handle:
        serializable_rows = []
        for row in rows:
            record = dict(row)
            if "formula_key" in record:
                record["formula_key"] = json.dumps(record["formula_key"], separators=(",", ":"))
            serializable_rows.append(record)
        writer = csv.DictWriter(
            handle,
            fieldnames=list(serializable_rows[0].keys()) if serializable_rows else ["reduced_formula"],
        )
        writer.writeheader()
        writer.writerows(serializable_rows)

    make_plot(pos, neg, out_dir / "kononova_validation_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
