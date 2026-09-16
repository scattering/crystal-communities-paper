#!/usr/bin/env python3
"""Append existing five-representation map diagnostics to the frozen sample."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parents[1]
SAMPLE = HERE / "sample.json"
REPORT = HERE / "sampling_report.json"
SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")


def projection_paths(source: str) -> dict[str, tuple[Path, str]]:
    crystalweave_name = "mattergen-public" if source == "mattergen" else source
    return {
        "CrystalWeave": (
            DOWNSTREAM / "external" / source / f"{crystalweave_name}_frontier_records.csv",
            "zip_member" if source == "mattergen" else "material_id",
        ),
        "Magpie": (
            DOWNSTREAM / "external_representation" / "consistent_transform" / "external" / source / "projection_full.csv",
            "record_key",
        ),
        "CrystalNN_graphlets": (
            DOWNSTREAM / "external_representation" / "graphlet" / "external" / source / "projection_full.csv",
            "record_key",
        ),
        "VoronoiNN_graphlets": (
            DOWNSTREAM
            / "representation_mechanism_diagnostics"
            / "voronoi_external"
            / "results"
            / "external"
            / source
            / "projection_full.csv",
            "record_key",
        ),
        "AMD": (
            DOWNSTREAM
            / "representations"
            / "amd-external-full-map"
            / "results"
            / source
            / "projection_full.csv",
            "record_key",
        ),
    }


def retained_path(path: Path) -> str:
    repo = next(parent for parent in HERE.parents if (parent / ".git").exists())
    return path.relative_to(repo).as_posix()


def load_projection(path: Path, key_field: str) -> dict[str, tuple[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[str, tuple[int, dict[str, str]]] = {}
    for i, row in enumerate(rows):
        key = row[key_field]
        if key in indexed:
            raise ValueError(f"Duplicate {key_field}={key!r} in {path}")
        indexed[key] = (i, row)
    return indexed


def bool_value(value: str) -> bool:
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"Unexpected boolean value {value!r}")
    return lowered == "true"


def main() -> None:
    records = json.loads(SAMPLE.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    indexes: dict[tuple[str, str], dict[str, tuple[int, dict[str, str]]]] = {}
    inputs: dict[str, dict[str, object]] = {}

    for source in SOURCES:
        for representation, (path, key_field) in projection_paths(source).items():
            if not path.exists():
                raise FileNotFoundError(path)
            indexes[(source, representation)] = load_projection(path, key_field)
            inputs[f"{source}:{representation}"] = {
                "path": retained_path(path),
                "key_field": key_field,
            }

    missing: list[dict[str, str]] = []
    for record in records:
        source = record["source"]
        material_id = record["material_id"]
        joined: dict[str, dict[str, object]] = {}
        for representation, (path, _) in projection_paths(source).items():
            located = indexes[(source, representation)].get(material_id)
            if located is None:
                missing.append(
                    {"source": source, "material_id": material_id, "representation": representation}
                )
                continue
            row_index, row = located
            distance = float(row["nearest_centroid_distance"])
            radius = float(row["community_threshold_p95"])
            flag = bool_value(row["in_basin"])
            reproduced_flag = distance <= radius
            if flag != reproduced_flag:
                raise ValueError(
                    f"Saved in_basin does not reproduce for {source} {material_id} {representation}"
                )
            joined[representation] = {
                "assigned_community": int(row["assigned_community"]),
                "nearest_centroid_distance": distance,
                "community_threshold_p95": radius,
                "in_basin": flag,
                "source_location": {
                    "path": retained_path(path),
                    "csv_line": row_index + 2,
                },
            }
        record["representations"] = joined

    if missing:
        raise ValueError(f"Missing {len(missing)} representation joins; first entries: {missing[:10]}")
    if any(len(record["representations"]) != 5 for record in records):
        raise AssertionError("Every frozen record must join to all five representation maps")

    SAMPLE.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    report["map_join"] = {
        "status": "complete",
        "representations": [
            "CrystalWeave",
            "Magpie",
            "CrystalNN_graphlets",
            "VoronoiNN_graphlets",
            "AMD",
        ],
        "records_with_all_five_maps": len(records),
        "missing_joins": 0,
        "saved_flags_reproduced_from_distance_le_radius": True,
        "inputs": inputs,
        "selection_used_map_fields": False,
    }
    report["output"]["sample_sha256"] = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
