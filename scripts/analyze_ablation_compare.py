#!/usr/bin/env python3
"""Cross-ablation comparison of densification and partition agreement.

Compares the densification-by-decade trends and the cluster partitions
across the four featurization ablations referenced in Methods (full
matminer + 3-round message passing, no-message-passing, fast-local
geometry, chemistry-only). For each pair of runs, computes adjusted
Rand index and normalized mutual information between the cluster
labels; for each run, summarizes per-decade ratios and a
ratio-vs-decade slope.

Inputs:
  --base-dir   Directory containing per-ablation subdirectories.
  --runs       Subdirectory names under base-dir to compare.

Outputs (under ``--output-dir``):
  ablation_compare_summary.json         All-pairs ARI/NMI plus per-run
                                         densification stats.
  ablation_partition_agreement.csv      Pairwise ARI/NMI table.
  ablation_trend_slopes.csv             Per-run ratio-vs-decade slopes.
  ablation_densification_compare.png    Side-by-side decade trends.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare densification and partition agreement across ablations.")
    parser.add_argument("--base-dir", required=True, help="Directory containing ablation run subdirectories")
    parser.add_argument("--runs", nargs="+", required=True, help="Run subdirectory names under base-dir")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_assignments(path: Path) -> list[dict[str, int]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "icsd_id": int(row["icsd_id"]),
                        "year": int(row["year"]),
                        "cluster": int(row["cluster"]),
                    }
                )
            except Exception:
                continue
    return rows


def relabel(name: str) -> str:
    mapping = {
        "ref_matminer_wl3": "full",
        "no_message_matminer_wl0": "no message",
        "fastlocal_wl3": "fast local",
        "chemonly_wl0": "chemistry only",
    }
    return mapping.get(name, name)


def plot_densification(run_data: dict[str, dict], out_path: Path) -> None:
    decades = sorted({d for data in run_data.values() for d in data["summary"]["densification"]["by_decade"].keys()})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    colors = {
        "ref_matminer_wl3": "#1f77b4",
        "no_message_matminer_wl0": "#ff7f0e",
        "fastlocal_wl3": "#2ca02c",
        "chemonly_wl0": "#d62728",
    }
    for name, data in run_data.items():
        by_decade = data["summary"]["densification"]["by_decade"]
        birth = [by_decade.get(decade, {}).get("cluster_birth_point_ratio", 0.0) for decade in decades]
        existing = [by_decade.get(decade, {}).get("existing_cluster_ratio", 0.0) for decade in decades]
        axes[0].plot(decades, birth, marker="o", ms=3, lw=1.8, color=colors.get(name), label=relabel(name))
        axes[1].plot(decades, existing, marker="o", ms=3, lw=1.8, color=colors.get(name), label=relabel(name))
    axes[0].set_title("Cluster-birth ratio by decade")
    axes[1].set_title("Existing-cluster ratio by decade")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("ratio")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compare_partitions(run_data: dict[str, dict]) -> list[dict[str, float]]:
    out = []
    for a, b in combinations(run_data.keys(), 2):
        rows_a = {row["icsd_id"]: row["cluster"] for row in run_data[a]["assignments"]}
        rows_b = {row["icsd_id"]: row["cluster"] for row in run_data[b]["assignments"]}
        common = sorted(set(rows_a) & set(rows_b))
        labels_a = [rows_a[i] for i in common]
        labels_b = [rows_b[i] for i in common]
        out.append(
            {
                "run_a": a,
                "run_b": b,
                "n_common": int(len(common)),
                "ari": float(adjusted_rand_score(labels_a, labels_b)),
                "nmi": float(normalized_mutual_info_score(labels_a, labels_b)),
            }
        )
    return out


def summarize_slopes(run_data: dict[str, dict]) -> list[dict[str, float]]:
    out = []
    for name, data in run_data.items():
        by_decade = data["summary"]["densification"]["by_decade"]
        decades = sorted(by_decade)
        xs = np.arange(len(decades), dtype=float)
        birth = np.array([by_decade[d].get("cluster_birth_point_ratio", 0.0) for d in decades], dtype=float)
        existing = np.array([by_decade[d].get("existing_cluster_ratio", 0.0) for d in decades], dtype=float)
        birth_slope = float(np.polyfit(xs, birth, 1)[0]) if len(xs) >= 2 else 0.0
        existing_slope = float(np.polyfit(xs, existing, 1)[0]) if len(xs) >= 2 else 0.0
        out.append(
            {
                "run": name,
                "label": relabel(name),
                "birth_slope_per_decade_index": birth_slope,
                "existing_slope_per_decade_index": existing_slope,
            }
        )
    return out


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_data = {}
    for name in args.runs:
        run_dir = base_dir / name
        run_data[name] = {
            "summary": load_summary(run_dir / "summary.json"),
            "assignments": load_assignments(run_dir / "sample_assignments.csv"),
        }

    plot_densification(run_data, out_dir / "ablation_densification_compare.png")
    partition_rows = compare_partitions(run_data)
    slope_rows = summarize_slopes(run_data)
    summary = {"partition_agreement": partition_rows, "trend_slopes": slope_rows}
    (out_dir / "ablation_compare_summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "ablation_partition_agreement.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_a", "run_b", "n_common", "ari", "nmi"])
        writer.writeheader()
        writer.writerows(partition_rows)
    with (out_dir / "ablation_trend_slopes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "label", "birth_slope_per_decade_index", "existing_slope_per_decade_index"])
        writer.writeheader()
        writer.writerows(slope_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
