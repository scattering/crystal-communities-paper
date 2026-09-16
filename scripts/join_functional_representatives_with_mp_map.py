#!/usr/bin/env python3
"""Join community representatives against a prebuilt MP-ICSD mapping.

Helper: offline alternative to
``enrich_functional_representatives_with_mp.py``. Takes a
representatives CSV and a prebuilt MP-ICSD map (from
``build_mp_icsd_map.py``) and writes the joined output without
hitting the MP API. Useful for reproducing the functional-labeling
inputs without an MP_API_KEY.

Inputs:
  --representatives, --mp-map, --max-communities.

Outputs:
  --output-csv     Joined representatives with MP fields.
  --summary-json   Coverage statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Join community representative rows with a prebuilt MP ICSD map.")
    parser.add_argument("--representatives", required=True)
    parser.add_argument("--mp-map", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-communities", type=int, default=50)
    args = parser.parse_args()

    reps = read_rows(Path(args.representatives))
    mp_rows = read_rows(Path(args.mp_map))

    seen_communities: set[str] = set()
    kept_reps: list[dict[str, str]] = []
    for row in reps:
        community = row["community"]
        if community not in seen_communities and len(seen_communities) >= args.max_communities:
            continue
        seen_communities.add(community)
        kept_reps.append(row)

    mp_by_icsd: dict[str, list[dict[str, str]]] = {}
    for row in mp_rows:
        mp_by_icsd.setdefault((row.get("icsd_id") or "").strip(), []).append(row)

    out_rows: list[dict[str, str]] = []
    matched_reps = 0
    matched_communities: set[str] = set()
    for row in kept_reps:
        icsd_id = (row.get("icsd_id") or "").strip()
        hits = mp_by_icsd.get(icsd_id, [])
        if hits:
            matched_reps += 1
            matched_communities.add(row["community"])
            for hit in hits:
                out_rows.append(
                    {
                        **row,
                        "mp_material_id": hit.get("mp_material_id", ""),
                        "mp_formula_pretty": hit.get("mp_formula_pretty", ""),
                    }
                )
        else:
            out_rows.append({**row, "mp_material_id": "", "mp_formula_pretty": ""})

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {
        "n_representative_rows": len(kept_reps),
        "n_unique_communities": len(seen_communities),
        "n_mp_map_rows": len(mp_rows),
        "n_matched_representative_rows": matched_reps,
        "match_rate_rows": matched_reps / len(kept_reps) if kept_reps else 0.0,
        "n_communities_with_any_match": len(matched_communities),
        "community_match_rate": len(matched_communities) / len(seen_communities) if seen_communities else 0.0,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
