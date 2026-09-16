#!/usr/bin/env python3
"""Compare validated repaired graph partitions on common ICSD identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from compare_feature_repair import cohort_comparison, read_assignments, temporal_stats, validate_rows
from regenerate_repaired_temporal import dump, file_hash, write_csv


def load(directory):
    path = directory / "graph/community_assignments.csv"
    rows = read_assignments(path)
    temporal = json.loads((directory / "time/graph_time_summary.json").read_text())
    stats = temporal_stats(temporal)
    validate_rows(rows, stats, path)
    graph = json.loads((directory / "graph/community_summary.json").read_text())
    return rows, temporal, stats, graph, path


def agreement(left, right):
    a = {i: c for i, _, c in left}
    b = {i: c for i, _, c in right}
    shared = sorted(a.keys() & b.keys())
    out = {}
    for name, ids in (("including_noise", shared), ("nonoutlier_both", [i for i in shared if a[i] >= 0 and b[i] >= 0])):
        out[name] = {"n": len(ids), "ARI": adjusted_rand_score([a[i] for i in ids], [b[i] for i in ids]),
                     "NMI": normalized_mutual_info_score([a[i] for i in ids], [b[i] for i in ids])}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--variant", action="append", required=True, help="NAME=run_directory, containing graph/ and time/")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline, *_ = load(args.baseline)
    runs = [("baseline", args.baseline)] + [(v.split("=", 1)[0], Path(v.split("=", 1)[1])) for v in args.variant]
    report, rows = {}, []
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    decades = [f"{y}s" for y in range(1930, 2020, 10)]
    for name, directory in runs:
        entries, temporal, stats, graph, path = load(directory)
        comparison = cohort_comparison(baseline, entries, "repaired baseline")
        metrics = agreement(baseline, entries)
        shares = {d: stats["by_decade"][d]["birth_share"] for d in decades}
        report[name] = {"run_directory": str(directory), "graph_settings": graph["graph"],
                        "assignment_sha256": file_hash(path), "cohort": comparison,
                        "n_communities": len({c for _, _, c in entries if c >= 0}), "temporal": stats,
                        "agreement_vs_baseline": metrics,
                        "monotonic_nonincreasing_1930s_to_2010s": all(shares[b] <= shares[a] for a, b in zip(decades[:-1], decades[1:]))}
        r = {"variant": name, "n_successful": len(entries), "n_communities": report[name]["n_communities"],
             "outlier_ratio": stats["outlier_ratio"], "k": graph["graph"]["k"], "resolution": graph["graph"]["resolution"],
             "birth_share_1930s": shares["1930s"], "birth_share_2010s": shares["2010s"],
             "monotonic_1930s_to_2010s": report[name]["monotonic_nonincreasing_1930s_to_2010s"]}
        r.update({f"{key}_{scope}": value for scope, vals in metrics.items() for key, value in vals.items()})
        rows.append(r)
        ax.plot(decades, list(shares.values()), marker="o", markersize=3, label=name, linewidth=2 if name == "baseline" else 1)
    ax.set(xlabel="First-publication decade", ylabel="Community-birth share", title="Repaired partition sensitivity")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "svg"):
        fig.savefig(args.output_dir / f"partition_birth_share_comparison.{ext}", dpi=200)
    plt.close(fig)
    dump(args.output_dir / "partition_comparison.json", report)
    write_csv(args.output_dir / "partition_comparison.csv", rows)
    lines = ["# Repaired partition comparison", "", "Labels are independently fitted in each graph; agreement is on shared ICSD IDs. Temporal curves use each full successful population. No external projection rates are inferred.", "", "| Variant | Communities | Outlier share |1930s birth|2010s birth|ARI/NMI, non-outlier in both|", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['variant']}|{r['n_communities']:,}|{100*r['outlier_ratio']:.2f}%|{100*r['birth_share_1930s']:.3f}%|{100*r['birth_share_2010s']:.3f}%|{r['ARI_nonoutlier_both']:.4f}/{r['NMI_nonoutlier_both']:.4f}|")
    (args.output_dir / "partition_comparison.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
