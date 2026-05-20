#!/usr/bin/env python3
"""Frontier rates stratified by community-level functional class.

Supports SI §S4. Aggregates frontier (= ``outlier_like``) rates by
community functional class for three populations: held-out ICSD
(>2010 map), the GNoME public sample, and the MatterGen public bundle.
Communities are mapped to one of five hand-curated functional
signatures (battery_electrode_candidate, framework, magnet_candidate,
superconductor_candidate, thermoelectric_candidate) via the
``--community-labels`` CSV, and frontier rates within each class are
computed with bootstrap 95% CIs.

Inputs:
  --community-labels       CSV mapping community -> functional_signature.
  --heldout-records        post_cutoff_accessibility_records.csv (held-out
                            ICSD >2010 map).
  --gnome-records          gnome_frontier_records.csv.
  --mattergen-records      mattergen_frontier_records.csv.
  --bootstrap-iters        Number of bootstrap resamples (default 2000).
  --seed                   RNG seed (default 42).

Outputs (under ``--output-dir``):
  functional_frontier_stratification_summary.json   Per-class rates +
                                                     CIs per population.
  functional_frontier_stratification_rows.csv       Same, tabular.
  functional_frontier_stratification_records.csv    Per-record frontier
                                                     flag with population
                                                     and class.
  functional_frontier_stratification.png            Per-class bar plot.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute frontier rates by community-level functional class for held-out ICSD, GNoME, and MatterGen."
    )
    parser.add_argument("--community-labels", required=True, help="CSV with community -> functional_signature labels.")
    parser.add_argument("--heldout-records", required=True, help="Held-out ICSD accessibility records CSV.")
    parser.add_argument("--gnome-records", required=True, help="GNoME frontier records CSV.")
    parser.add_argument("--mattergen-records", required=True, help="MatterGen frontier records CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


CLASS_ORDER = [
    "battery_electrode_candidate",
    "framework",
    "magnet_candidate",
    "superconductor_candidate",
    "thermoelectric_candidate",
]

POP_STYLE = {
    "heldout_icsd_2010": {"label": "Held-out ICSD (>2010 map)", "color": "#4c78a8"},
    "mattergen_public": {"label": "MatterGen public bundle", "color": "#f58518"},
    "gnome_public": {"label": "GNoME public sample", "color": "#54a24b"},
}


def parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def parse_bool(value: str) -> bool | None:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return True
    if text in {"0", "false", "f", "no"}:
        return False
    return None


def load_label_map(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            community = parse_int(row.get("community", ""))
            label = (row.get("functional_signature") or "").strip()
            if community is None or not label:
                continue
            out[community] = label
    return out


def load_records(
    path: Path,
    community_field: str,
    frontier_fn: Callable[[dict[str, str]], bool | None],
    label_map: dict[int, str],
    population_key: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            community = parse_int(row.get(community_field, ""))
            if community is None:
                continue
            functional_class = label_map.get(community)
            if not functional_class:
                continue
            frontier = frontier_fn(row)
            if frontier is None:
                continue
            rows.append(
                {
                    "population": population_key,
                    "community": community,
                    "functional_class": functional_class,
                    "frontier": bool(frontier),
                }
            )
    return rows


def bootstrap_ci(values: list[int], rng: np.random.Generator, iters: int) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    means = []
    for _ in range(iters):
        idx = rng.integers(0, len(arr), size=len(arr))
        means.append(float(arr[idx].mean()))
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def summarize(records: list[dict[str, object]], bootstrap_iters: int, seed: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    communities: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in records:
        key = (str(row["population"]), str(row["functional_class"]))
        grouped[key].append(1 if bool(row["frontier"]) else 0)
        communities[key].add(int(row["community"]))

    rng = np.random.default_rng(seed)
    out = []
    for population, functional_class in sorted(grouped):
        values = grouped[(population, functional_class)]
        ci_low, ci_high = bootstrap_ci(values, rng, bootstrap_iters)
        out.append(
            {
                "population": population,
                "functional_class": functional_class,
                "n_structures": int(len(values)),
                "n_frontier": int(sum(values)),
                "frontier_rate": float(np.mean(values)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_unique_communities": int(len(communities[(population, functional_class)])),
            }
        )
    return out


def plot(summary_rows: list[dict[str, object]], out_path: Path) -> None:
    rows_by_pop = defaultdict(dict)
    for row in summary_rows:
        rows_by_pop[str(row["population"])][str(row["functional_class"])] = row

    classes = [c for c in CLASS_ORDER if any(c in rows_by_pop[p] for p in rows_by_pop)]
    if not classes:
        return

    x = np.arange(len(classes), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.4))

    for i, population in enumerate(["heldout_icsd_2010", "mattergen_public", "gnome_public"]):
        style = POP_STYLE[population]
        vals = []
        err_low = []
        err_high = []
        for cls in classes:
            row = rows_by_pop.get(population, {}).get(cls)
            if row is None:
                vals.append(np.nan)
                err_low.append(0.0)
                err_high.append(0.0)
            else:
                vals.append(float(row["frontier_rate"]))
                err_low.append(float(row["frontier_rate"]) - float(row["ci_low"]))
                err_high.append(float(row["ci_high"]) - float(row["frontier_rate"]))
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=style["label"],
            color=style["color"],
            yerr=np.vstack([err_low, err_high]),
            capsize=3,
            linewidth=0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_candidate", "").replace("_", " ") for c in classes], rotation=20, ha="right")
    ax.set_ylabel("Frontier rate")
    ax.set_title("Frontier rate by community-level functional class")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(Path(args.community_labels))
    heldout_records = load_records(
        Path(args.heldout_records),
        community_field="assigned_community",
        frontier_fn=lambda row: (not parse_bool(row.get("is_in_basin", ""))) if parse_bool(row.get("is_in_basin", "")) is not None else None,
        label_map=label_map,
        population_key="heldout_icsd_2010",
    )
    gnome_records = load_records(
        Path(args.gnome_records),
        community_field="assigned_community",
        frontier_fn=lambda row: parse_bool(row.get("outlier_like", "")),
        label_map=label_map,
        population_key="gnome_public",
    )
    mattergen_records = load_records(
        Path(args.mattergen_records),
        community_field="assigned_community",
        frontier_fn=lambda row: parse_bool(row.get("outlier_like", "")),
        label_map=label_map,
        population_key="mattergen_public",
    )

    all_records = heldout_records + mattergen_records + gnome_records
    summary_rows = summarize(all_records, bootstrap_iters=args.bootstrap_iters, seed=args.seed)

    overall = {}
    for pop in ["heldout_icsd_2010", "mattergen_public", "gnome_public"]:
        pop_rows = [r for r in all_records if r["population"] == pop]
        overall[pop] = {
            "n_labeled_structures": int(len(pop_rows)),
            "n_labeled_communities_hit": int(len({int(r["community"]) for r in pop_rows})),
            "class_counts": dict(Counter(str(r["functional_class"]) for r in pop_rows)),
        }

    summary = {
        "community_labels_path": str(Path(args.community_labels)),
        "n_labeled_communities": int(len(label_map)),
        "classes_present": sorted({row["functional_class"] for row in summary_rows}),
        "overall": overall,
        "rows": summary_rows,
    }
    (out_dir / "functional_frontier_stratification_summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "functional_frontier_stratification_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "population",
                "functional_class",
                "n_structures",
                "n_frontier",
                "frontier_rate",
                "ci_low",
                "ci_high",
                "n_unique_communities",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "functional_frontier_stratification_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["population", "community", "functional_class", "frontier"])
        writer.writeheader()
        writer.writerows(all_records)

    plot(summary_rows, out_dir / "functional_frontier_stratification.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
