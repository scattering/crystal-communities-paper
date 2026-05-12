#!/usr/bin/env python3
"""Keyword-based mineral / prototype family enrichment for communities.

Tags each ICSD entry with one or more curated family labels by
keyword-matching the formula / mineral-name field in the ICSD index
against canonical patterns (perovskite, spinel, garnet, MOF, zeolite,
pyroxene, amphibole, Heusler), then reports per-community counts of
each family and the top-N communities for each family. Optionally
also runs the same pass on HDBSCAN clusters via
``--cluster-assignments``. Used as a cross-check that the learned
communities cluster known mineral families together.

Inputs:
  --icsd-index, --community-assignments, --cluster-assignments
  (optional), --top-n.

Outputs (under ``--output-dir``):
  community_family_enrichment.json   Per-family ranking of communities.
  community_family_enrichment.csv    Same, tabular.
  cluster_family_enrichment.{json,csv}   When --cluster-assignments
                                          is provided.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ID_COLUMN_CANDIDATES = ["cif_names", "ICSDid", "ICSD ID", "Collection Code"]
NAME_COLUMN_CANDIDATES = ["name", "formula", "Name", "Formula"]

FAMILY_PATTERNS = {
    "perovskite": [
        r"\bperovskite\b",
        r"\belpasolite\b",
        r"\bbrownmillerite\b",
    ],
    "spinel": [
        r"\bspinel\b",
    ],
    "garnet": [
        r"\bgarnet\b",
        r"\byag\b",
    ],
    "mof": [
        r"\bmof\b",
        r"\bmetal-organic\b",
        r"\bmetal organic\b",
    ],
    "zeolite": [
        r"\bzeolite\b",
        r"\bchabazite\b",
        r"\bfaujasite\b",
        r"\bnatrolite\b",
        r"\bclinoptilolite\b",
        r"\bmordenite\b",
        r"\banalcime\b",
        r"\bsodalite\b",
        r"\bheulandite\b",
        r"\bstilbite\b",
    ],
    "pyroxene": [
        r"\bpyroxene\b",
        r"\bdiopside\b",
        r"\benstatite\b",
        r"\bjadeite\b",
        r"\baugite\b",
        r"\bhedenbergite\b",
        r"\baegirine\b",
    ],
    "amphibole": [
        r"\bamphibole\b",
        r"\brichterite\b",
        r"\bhornblende\b",
        r"\btremolite\b",
        r"\bactinolite\b",
        r"\bglaucophane\b",
        r"\barfvedsonite\b",
        r"\briebeckite\b",
    ],
    "heusler": [
        r"\bheusler\b",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyword-based family enrichment for ICSD communities/clusters.")
    parser.add_argument("--icsd-index", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--cluster-assignments")
    parser.add_argument("--top-n", type=int, default=10)
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
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def load_metadata(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header in {path}")
        id_col = find_column(reader.fieldnames, ID_COLUMN_CANDIDATES)
        name_col = find_column(reader.fieldnames, NAME_COLUMN_CANDIDATES)
        if id_col is None or name_col is None:
            raise ValueError(f"Missing id/name columns in {path}")
        out: dict[int, dict[str, str]] = {}
        for row in reader:
            icsd_id = parse_int(row.get(id_col, ""))
            if icsd_id is None:
                continue
            out[icsd_id] = {"name": row.get(name_col, "").strip()}
        return out


def load_assignments(path: Path, label_col: str) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            icsd_id = parse_int(row.get("icsd_id", ""))
            label = parse_int(row.get(label_col, ""))
            if icsd_id is None or label is None:
                continue
            rows.append((icsd_id, label))
    return rows


def tag_families(name: str) -> list[str]:
    text = (name or "").lower()
    found = []
    for family, patterns in FAMILY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text):
                found.append(family)
                break
    return found


def summarize_partition(
    assignments: list[tuple[int, int]],
    metadata: dict[int, dict[str, str]],
    top_n: int,
) -> dict[str, object]:
    family_totals = Counter()
    label_sizes = Counter()
    label_family_counts: dict[int, Counter[str]] = defaultdict(Counter)
    label_examples: dict[int, list[dict[str, object]]] = defaultdict(list)

    for icsd_id, label in assignments:
        if label < 0:
            continue
        label_sizes[label] += 1
        name = metadata.get(icsd_id, {}).get("name", "")
        families = tag_families(name)
        for family in families:
            family_totals[family] += 1
            label_family_counts[label][family] += 1
            if len(label_examples[label]) < 5:
                label_examples[label].append({"icsd_id": icsd_id, "name": name, "family": family})

    n_total = sum(label_sizes.values()) or 1
    family_baseline = {
        family: family_totals[family] / n_total for family in sorted(family_totals)
    }

    top_by_family: dict[str, list[dict[str, object]]] = {}
    for family in sorted(family_totals):
        rows = []
        baseline = family_baseline[family]
        for label, counts in label_family_counts.items():
            hits = counts.get(family, 0)
            if hits <= 0:
                continue
            size = label_sizes[label]
            frac = hits / size
            enrich = frac / baseline if baseline > 0 else math.inf
            rows.append(
                {
                    "label": int(label),
                    "size": int(size),
                    "hits": int(hits),
                    "fraction": frac,
                    "enrichment": enrich,
                    "examples": label_examples.get(label, []),
                }
            )
        rows.sort(key=lambda x: (-x["enrichment"], -x["hits"], -x["size"]))
        top_by_family[family] = rows[:top_n]

    top_labels = []
    for label, size in label_sizes.most_common(top_n):
        fam_counts = label_family_counts.get(label, Counter())
        ranked = [
            {"family": family, "hits": int(hits), "fraction": hits / size}
            for family, hits in fam_counts.most_common()
        ]
        top_labels.append(
            {
                "label": int(label),
                "size": int(size),
                "top_families": ranked[:5],
                "examples": label_examples.get(label, []),
            }
        )

    return {
        "n_labeled_entries": int(n_total),
        "family_totals": dict(family_totals),
        "family_baseline_fraction": family_baseline,
        "top_by_family": top_by_family,
        "top_labels": top_labels,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_top_by_family(top_by_family: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    out = []
    for family, rows in top_by_family.items():
        for row in rows:
            out.append(
                {
                    "family": family,
                    "label": row["label"],
                    "size": row["size"],
                    "hits": row["hits"],
                    "fraction": row["fraction"],
                    "enrichment": row["enrichment"],
                }
            )
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(Path(args.icsd_index))

    community = summarize_partition(
        load_assignments(Path(args.community_assignments), "community"),
        metadata,
        args.top_n,
    )
    (out_dir / "community_family_enrichment.json").write_text(json.dumps(community, indent=2))
    write_csv(out_dir / "community_family_enrichment.csv", flatten_top_by_family(community["top_by_family"]))

    if args.cluster_assignments:
        cluster = summarize_partition(
            load_assignments(Path(args.cluster_assignments), "cluster"),
            metadata,
            args.top_n,
        )
        (out_dir / "cluster_family_enrichment.json").write_text(json.dumps(cluster, indent=2))
        write_csv(out_dir / "cluster_family_enrichment.csv", flatten_top_by_family(cluster["top_by_family"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
