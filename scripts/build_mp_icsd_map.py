#!/usr/bin/env python3
"""Build an ICSD-id to Materials Project material_id mapping via the MP API.

Helper: streams MP summary documents in chunks (``--chunk-size``,
optional ``--num-chunks`` for probe runs) and emits one row per
ICSD-id linked to an MP material from the ``database_IDs.icsd``
field. Requires ``MP_API_KEY`` in the environment.

Outputs:
  --output-csv     icsd_id, mp_material_id, mp_formula_pretty.
  --summary-json   Coverage statistics for the mapping pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def normalize_icsd_id(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("icsd-"):
        return text.split("-", 1)[1]
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Materials Project ICSD-to-MP mapping from summary docs.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--num-chunks", type=int, default=None, help="Limit chunks for a probe run")
    args = parser.parse_args()

    mp_key = os.environ.get("MP_API_KEY", "").strip()
    if not mp_key:
        raise RuntimeError("MP_API_KEY is not set in the environment.")

    from mp_api.client import MPRester

    records: list[dict[str, object]] = []
    n_docs = 0
    with MPRester(mp_key, use_document_model=False, monty_decode=False) as mpr:
        docs = mpr.materials.summary.search(
            fields=["material_id", "formula_pretty", "database_IDs"],
            chunk_size=args.chunk_size,
            num_chunks=args.num_chunks,
        )
        for doc in docs:
            n_docs += 1
            db_ids = doc.get("database_IDs") or {}
            icsd_ids = db_ids.get("icsd") or []
            for icsd_id in icsd_ids:
                norm_icsd = normalize_icsd_id(icsd_id)
                if not norm_icsd:
                    continue
                records.append(
                    {
                        "icsd_id": norm_icsd,
                        "mp_material_id": doc.get("material_id", ""),
                        "mp_formula_pretty": doc.get("formula_pretty", ""),
                    }
                )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["icsd_id", "mp_material_id", "mp_formula_pretty"])
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "n_summary_docs_seen": n_docs,
        "n_mapping_rows": len(records),
        "n_unique_icsd_ids": len({row["icsd_id"] for row in records}),
        "n_unique_mp_ids": len({row["mp_material_id"] for row in records}),
        "chunk_size": args.chunk_size,
        "num_chunks": args.num_chunks,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
