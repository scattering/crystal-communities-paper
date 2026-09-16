#!/usr/bin/env python3
"""Enrich community-representative ICSD ids with Materials Project metadata.

Helper: takes the representatives CSV from
``extract_functional_community_representatives.py``, queries MP for
matching ICSD ids in ``--chunk-size`` batches, and writes an enriched
CSV with MP material id, formula, and selected MP fields.
Requires ``MP_API_KEY`` in the environment.

Inputs:
  --representatives, --max-communities, --chunk-size.

Outputs:
  --output-csv      Enriched representatives.
  --summary-json    Coverage statistics.
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unique_icsd_ids(rows: list[dict[str, str]], max_communities: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    kept_rows: list[dict[str, str]] = []
    seen_communities: set[str] = set()
    ids: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        community = row["community"]
        if max_communities is not None and community not in seen_communities and len(seen_communities) >= max_communities:
            continue
        seen_communities.add(community)
        kept_rows.append(row)
        icsd_id = (row.get("icsd_id") or "").strip()
        if icsd_id and icsd_id not in seen_ids:
            seen_ids.add(icsd_id)
            ids.append(icsd_id)
    return ids, kept_rows


def chunked(seq: list[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enrich community representative ICSD ids with Materials Project metadata."
    )
    parser.add_argument("--representatives", required=True, help="Representative CSV from extract_functional_community_representatives.py")
    parser.add_argument("--output-csv", required=True, help="Enriched representative CSV")
    parser.add_argument("--summary-json", required=True, help="Summary JSON output")
    parser.add_argument("--max-communities", type=int, default=50, help="Only enrich the first N communities by file order")
    parser.add_argument("--chunk-size", type=int, default=100, help="Batch size for MP queries")
    args = parser.parse_args()

    mp_key = os.environ.get("MP_API_KEY", "").strip()
    if not mp_key:
        raise RuntimeError("MP_API_KEY is not set in the environment.")

    try:
        from mp_api.client import MPRester
    except Exception as exc:
        raise RuntimeError("mp_api client is not importable in this environment.") from exc

    rows = read_rows(Path(args.representatives))
    icsd_ids, kept_rows = unique_icsd_ids(rows, max_communities=args.max_communities)

    method_used = None
    mp_by_icsd: dict[str, list[dict[str, object]]] = {}
    query_errors: list[dict[str, str]] = []

    with MPRester(mp_key, use_document_model=False, monty_decode=False) as mpr:
        for batch in chunked(icsd_ids, args.chunk_size):
            docs = None
            batch_errors: list[str] = []
            # Try the most likely supported paths first. Keep the first one that works.
            attempts = [
                ("search_database_IDs_str", lambda: mpr.materials.summary.search(database_IDs=batch, fields=["material_id", "formula_pretty", "database_IDs", "band_gap", "is_metal", "theoretical", "symmetry"])),
                ("search_database_IDs_dict", lambda: mpr.materials.summary.search(database_IDs={"icsd": batch}, fields=["material_id", "formula_pretty", "database_IDs", "band_gap", "is_metal", "theoretical", "symmetry"])),
                ("_search_database_ids_str", lambda: mpr.materials.summary._search(database_ids=batch, fields=["material_id", "formula_pretty", "database_IDs", "band_gap", "is_metal", "theoretical", "symmetry"])),
                ("_search_database_ids_prefixed", lambda: mpr.materials.summary._search(database_ids=[f"icsd-{x}" for x in batch], fields=["material_id", "formula_pretty", "database_IDs", "band_gap", "is_metal", "theoretical", "symmetry"])),
            ]
            for attempt_name, attempt_fn in attempts:
                try:
                    docs = attempt_fn()
                    method_used = attempt_name
                    break
                except Exception as exc:
                    batch_errors.append(f"{attempt_name}: {type(exc).__name__}: {str(exc)[:200]}")
            if docs is None:
                query_errors.append({"batch_first_icsd": batch[0], "detail": " | ".join(batch_errors)})
                continue
            for doc in docs:
                db_ids = doc.get("database_IDs") or {}
                ids = db_ids.get("icsd") or []
                record = {
                    "material_id": doc.get("material_id", ""),
                    "formula_pretty": doc.get("formula_pretty", ""),
                    "band_gap": doc.get("band_gap", ""),
                    "is_metal": doc.get("is_metal", ""),
                    "theoretical": doc.get("theoretical", ""),
                    "symmetry": json.dumps(doc.get("symmetry", {}), ensure_ascii=True),
                }
                for icsd_id in ids:
                    key = normalize_icsd_id(icsd_id)
                    if not key:
                        continue
                    mp_by_icsd.setdefault(key, []).append(record)

        # Optional robocrys enrichment for the matched material ids.
        robocrys_by_mpid: dict[str, str] = {}
        material_ids = sorted({rec["material_id"] for records in mp_by_icsd.values() for rec in records if rec.get("material_id")})
        if material_ids:
            try:
                for batch in chunked(material_ids, args.chunk_size):
                    docs = mpr.materials.robocrys.search(material_ids=batch, fields=["material_id", "description"])
                    for doc in docs:
                        mpid = doc.get("material_id", "")
                        if mpid:
                            robocrys_by_mpid[mpid] = doc.get("description", "")
            except Exception as exc:
                query_errors.append({"batch_first_icsd": "robocrys", "detail": f"robocrys: {type(exc).__name__}: {str(exc)[:200]}"})

    out_rows: list[dict[str, object]] = []
    for row in kept_rows:
        recs = mp_by_icsd.get(normalize_icsd_id(row.get("icsd_id") or ""), [])
        if not recs:
            out_rows.append({**row, "mp_material_id": "", "mp_formula_pretty": "", "mp_band_gap": "", "mp_is_metal": "", "mp_theoretical": "", "mp_symmetry": "", "mp_robocrys": ""})
            continue
        for rec in recs:
            out_rows.append(
                {
                    **row,
                    "mp_material_id": rec.get("material_id", ""),
                    "mp_formula_pretty": rec.get("formula_pretty", ""),
                    "mp_band_gap": rec.get("band_gap", ""),
                    "mp_is_metal": rec.get("is_metal", ""),
                    "mp_theoretical": rec.get("theoretical", ""),
                    "mp_symmetry": rec.get("symmetry", ""),
                    "mp_robocrys": robocrys_by_mpid.get(rec.get("material_id", ""), ""),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    matched_ids = sum(1 for icsd_id in icsd_ids if icsd_id in mp_by_icsd)
    summary = {
        "n_input_rows": len(rows),
        "n_kept_rows": len(kept_rows),
        "n_unique_icsd_ids": len(icsd_ids),
        "n_matched_icsd_ids": matched_ids,
        "match_rate": matched_ids / len(icsd_ids) if icsd_ids else 0.0,
        "query_method_used": method_used,
        "n_material_ids_with_robocrys": len(robocrys_by_mpid),
        "query_errors": query_errors,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
