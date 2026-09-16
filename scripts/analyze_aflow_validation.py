#!/usr/bin/env python3
"""External validation: do learned communities recover AFLOW prototype labels?

Sanity check that the unsupervised structural communities recover the
classical AFLOW prototype taxonomy. Samples ``--sample-size`` ICSD
entries (default 5000), reads their CIFs from ``--icsd-zip`` (with
``--zip-password`` if needed), runs ``AflowPrototypeMatcher`` on each
sampled structure to obtain a prototype label, joins to the production
community label, and reports per-prototype dominant-community counts
along with global ARI / NMI of the prototype labeling vs. the Louvain
partition. Prototype labels appearing fewer than ``--min-prototype-count``
times are dropped to suppress small-class noise.

Inputs:
  --community-assignments, --icsd-zip [+ --zip-password],
  --sample-size, --seed, --min-prototype-count, --top-n.

Outputs (under ``--output-dir``):
  aflow_validation_summary.json            Top prototypes + dominant
                                            community fraction; global
                                            ARI / NMI.
  aflow_validation_records.csv             Per-sampled-entry prototype
                                            tag and community.
  aflow_validation_failures.json           Per-entry AflowMatcher
                                            failures (capped at 500).
  aflow_validation_top_prototypes.png      Bar plot of dominant
                                            community fractions.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.analysis.prototypes import AflowPrototypeMatcher
from pymatgen.core import Structure
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate graph communities against sampled AFLOW prototype labels.")
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--icsd-zip", required=True)
    parser.add_argument("--zip-password")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-prototype-count", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_tag(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "unknown", "?", ".", "nan"}:
        return None
    return text


def preferred_label(tags: dict[str, object]) -> str | None:
    for key in ("aflow", "strukturbericht", "mineral"):
        label = normalize_tag(tags.get(key))
        if label:
            return label
    for value in tags.values():
        label = normalize_tag(value)
        if label:
            return label
    return None


def read_structure(zf: zipfile.ZipFile, icsd_id: int, pwd: bytes | None) -> Structure:
    member = f"FindIt_CIFs/icsd_{icsd_id:06d}.cif"
    with zf.open(member, pwd=pwd) as handle:
        text = handle.read().decode("utf-8", errors="replace")
    return Structure.from_str(text, fmt="cif")


def summarize(records: list[dict[str, object]], min_count: int, top_n: int) -> dict[str, object]:
    proto_counts = Counter(str(r["prototype"]) for r in records)
    proto_to_comm: dict[str, Counter[int]] = defaultdict(Counter)
    comm_to_proto: dict[int, Counter[str]] = defaultdict(Counter)
    for row in records:
        proto = str(row["prototype"])
        comm = int(row["community"])
        proto_to_comm[proto][comm] += 1
        comm_to_proto[comm][proto] += 1

    prototype_rows = []
    for proto, count in proto_counts.items():
        if count < min_count:
            continue
        comm_counts = proto_to_comm[proto]
        dominant_comm, dominant_count = comm_counts.most_common(1)[0]
        prototype_rows.append(
            {
                "prototype": proto,
                "count": int(count),
                "n_communities": int(len(comm_counts)),
                "dominant_community": int(dominant_comm),
                "dominant_community_fraction": float(dominant_count / count),
            }
        )
    prototype_rows.sort(key=lambda row: (-row["count"], row["prototype"]))

    community_rows = []
    for comm, counts in comm_to_proto.items():
        total = sum(counts.values())
        dominant_proto, dominant_count = counts.most_common(1)[0]
        community_rows.append(
            {
                "community": int(comm),
                "count": int(total),
                "n_prototypes": int(len(counts)),
                "dominant_prototype": dominant_proto,
                "dominant_prototype_fraction": float(dominant_count / total),
            }
        )
    community_rows.sort(key=lambda row: (-row["count"], row["community"]))

    return {
        "n_records": int(len(records)),
        "n_unique_prototypes": int(len(proto_counts)),
        "adjusted_rand_index": float(
            adjusted_rand_score([r["community"] for r in records], [r["prototype"] for r in records])
        ),
        "normalized_mutual_info": float(
            normalized_mutual_info_score([r["community"] for r in records], [r["prototype"] for r in records])
        ),
        "prototype_purity_mean": float(
            np.mean([row["dominant_community_fraction"] for row in prototype_rows]) if prototype_rows else 0.0
        ),
        "prototype_purity_weighted_mean": float(
            np.average(
                [row["dominant_community_fraction"] for row in prototype_rows],
                weights=[row["count"] for row in prototype_rows],
            )
            if prototype_rows
            else 0.0
        ),
        "top_prototypes": prototype_rows[:top_n],
        "top_communities": community_rows[:top_n],
    }


def plot_top_prototypes(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    labels = [str(r["prototype"])[:30] for r in rows][::-1]
    purity = [float(r["dominant_community_fraction"]) for r in rows][::-1]
    counts = [int(r["count"]) for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, purity, color="#4c78a8")
    ax.set_xlabel("Dominant-community fraction")
    ax.set_title("AFLOW prototype coherence within learned communities")
    ax.set_xlim(0.0, 1.0)
    for bar, count in zip(bars, counts):
        ax.text(min(bar.get_width() + 0.01, 0.98), bar.get_y() + bar.get_height() / 2, f"n={count}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.community_assignments))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.sample_size]

    matcher = AflowPrototypeMatcher(initial_ltol=0.25, initial_stol=0.35, initial_angle_tol=8.0)
    pwd = args.zip_password.encode("utf-8") if args.zip_password else None

    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with zipfile.ZipFile(args.icsd_zip) as zf:
        for row in rows:
            icsd_id = int(row["icsd_id"])
            community = int(row["community"])
            if community < 0:
                continue
            try:
                structure = read_structure(zf, icsd_id, pwd)
                tags = matcher.get_prototypes(structure)
                if not tags:
                    continue
                label = preferred_label(tags[0].tags)
                if not label:
                    continue
                records.append(
                    {
                        "icsd_id": icsd_id,
                        "community": community,
                        "prototype": label,
                    }
                )
            except Exception as exc:
                failures.append({"icsd_id": icsd_id, "detail": str(exc)})

    summary = summarize(records, args.min_prototype_count, args.top_n)
    summary["n_sampled"] = int(len(rows))
    summary["n_matched"] = int(len(records))
    summary["n_failures"] = int(len(failures))

    (out_dir / "aflow_validation_summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "aflow_validation_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["icsd_id", "community", "prototype"])
        writer.writeheader()
        writer.writerows(records)
    (out_dir / "aflow_validation_failures.json").write_text(json.dumps(failures[:500], indent=2))
    plot_top_prototypes(summary["top_prototypes"], out_dir / "aflow_validation_top_prototypes.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
