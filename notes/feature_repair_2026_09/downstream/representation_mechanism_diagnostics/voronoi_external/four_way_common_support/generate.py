#!/usr/bin/env python3
"""Build the exact full-map common-support table for four representations."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parents[2]
SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")
SOURCE_LABELS = {
    "gnome": "GNoME",
    "mattergen": "MatterGen",
    "mp": "MP",
    "jarvis": "JARVIS",
    "alexandria": "Alexandria",
}
REPRESENTATIONS = {
    "CrystalWeave": {
        "icsd": DOWNSTREAM / "external_representation/production_reference/icsd_full.csv",
        "external": {
            "gnome": DOWNSTREAM / "external/gnome/gnome_frontier_records.csv",
            "mattergen": DOWNSTREAM / "external/mattergen/mattergen-public_frontier_records.csv",
            "mp": DOWNSTREAM / "external/mp/mp_frontier_records.csv",
            "jarvis": DOWNSTREAM / "external/jarvis/jarvis_frontier_records.csv",
            "alexandria": DOWNSTREAM / "external/alexandria/alexandria_frontier_records.csv",
        },
    },
    "Magpie": {
        "icsd": DOWNSTREAM / "external_representation/consistent_transform/basis/icsd_full.csv",
        "external_root": DOWNSTREAM / "external_representation/consistent_transform/external",
    },
    "CrystalNN graphlets": {
        "icsd": DOWNSTREAM / "external_representation/graphlet/basis/icsd_full.csv",
        "external_root": DOWNSTREAM / "external_representation/graphlet/external",
    },
    "Voronoi graphlets": {
        "icsd": HERE.parent / "results/basis/icsd_full.csv",
        "external_root": HERE.parent / "results/external",
    },
}
EXPECTED_COMMON = {
    "ICSD": (146186, "b70d79e1aa1072d879fcd90f4413a86772c2f17936653e86192c3d623be8d004"),
    "GNoME": (5000, "d7fde4416366366da3bc748a0f7b0603b15a6214790a6c4c36469ea0b4e7f3d5"),
    "MatterGen": (385, "4592ecd721086567c670afd1c80cbb74a63df9b18a8d29582f456598dca5b343"),
    "MP": (4969, "14fe3cdfdd27ffa7bf4ccd14b296da6b17ae2daffa4744cd3a836abb42c64481"),
    "JARVIS": (4930, "f7f933af7cfa7779606d464a01e0680721d331e7d66c84f9eb9803905c9dbdaa"),
    "Alexandria": (4982, "111b16ba50077022fddeb174fe6c42f09e319b225d376bcdf0baa89847ee7f4b"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def parse_bool(series: pd.Series, path: Path) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    if not values.isin(["true", "false"]).all():
        raise ValueError(f"Unexpected in_basin value in {path}")
    return values.eq("true")


def load_projection(path: Path, key: str) -> pd.Series:
    frame = pd.read_csv(path, dtype={key: str}, float_precision="round_trip")
    if key not in frame or frame[key].isna().any() or not frame[key].is_unique:
        raise ValueError(f"Invalid key column {key} in {path}")
    required = {"nearest_centroid_distance", "community_threshold_p95", "in_basin"}
    if not required.issubset(frame):
        raise ValueError(f"Missing projection columns in {path}")
    inside = parse_bool(frame.in_basin, path)
    reproduced = frame.nearest_centroid_distance.le(frame.community_threshold_p95)
    if not inside.equals(reproduced):
        raise ValueError(f"Saved in_basin flags do not match distance <= p95 in {path}")
    inside.index = frame[key].astype(str)
    inside.index.name = "record_key"
    return inside


def external_path(specification: dict, source: str) -> Path:
    if "external" in specification:
        return specification["external"][source]
    return specification["external_root"] / source / "projection_full.csv"


def main() -> None:
    inputs: dict[str, dict[str, int | str]] = {}
    projections: dict[str, dict[str, pd.Series]] = {}
    for representation, specification in REPRESENTATIONS.items():
        paths = {"ICSD": specification["icsd"]}
        paths.update({SOURCE_LABELS[source]: external_path(specification, source) for source in SOURCES})
        projections[representation] = {}
        for population, path in paths.items():
            path = path.resolve()
            key = "record_key"
            if representation == "CrystalWeave" and population != "ICSD":
                key = "zip_member" if population == "MatterGen" else "material_id"
            projections[representation][population] = load_projection(path, key)
            inputs[str(path.relative_to(DOWNSTREAM.parents[2]))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }

    populations = ("ICSD",) + tuple(SOURCE_LABELS[source] for source in SOURCES)
    common_ids: dict[str, set[str]] = {}
    for population in populations:
        common_ids[population] = set.intersection(
            *(set(projections[representation][population].index) for representation in REPRESENTATIONS)
        )
        if not common_ids[population]:
            raise ValueError(f"Empty four-way intersection for {population}")
        observed = (len(common_ids[population]), ids_sha256(common_ids[population]))
        if observed != EXPECTED_COMMON[population]:
            raise ValueError(f"Four-way support changed for {population}: {observed}")

    long_rows = []
    for representation in REPRESENTATIONS:
        icsd_flags = projections[representation]["ICSD"].loc[sorted(common_ids["ICSD"])]
        icsd_n = len(icsd_flags)
        icsd_inside = int(icsd_flags.sum())
        icsd_rate = icsd_inside / icsd_n
        long_rows.append({
            "representation": representation,
            "population": "ICSD",
            "n_common": icsd_n,
            "n_in_basin": icsd_inside,
            "in_basin_fraction": icsd_rate,
            "icsd_minus_source_percentage_points": "",
        })
        for source in SOURCES:
            population = SOURCE_LABELS[source]
            flags = projections[representation][population].loc[sorted(common_ids[population])]
            n = len(flags)
            n_inside = int(flags.sum())
            rate = n_inside / n
            long_rows.append({
                "representation": representation,
                "population": population,
                "n_common": n,
                "n_in_basin": n_inside,
                "in_basin_fraction": rate,
                "icsd_minus_source_percentage_points": 100 * (icsd_rate - rate),
            })

    csv_path = HERE / "four_way_common_support.csv"
    fields = list(long_rows[0])
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(long_rows)

    rows_by_representation = {
        representation: {
            row["population"]: row
            for row in long_rows
            if row["representation"] == representation
        }
        for representation in REPRESENTATIONS
    }
    lines = [
        "# Four-representation exact-common-support full-map comparison",
        "",
        "All four fitted maps retain their own centroids, member-based p95 radii, and nearest-centroid classifications. Only the evaluation identifiers are intersected. Cells show in-basin percentages; parenthetical values are exact common denominators.",
        "",
        "| Representation | ICSD | GNoME | MatterGen | MP | JARVIS | Alexandria |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for representation in REPRESENTATIONS:
        values = rows_by_representation[representation]
        cells = []
        for population in populations:
            row = values[population]
            cells.append(f"{100 * row['in_basin_fraction']:.2f}% ({row['n_common']:,})")
        lines.append(f"| {representation} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "ICSD-minus-source gaps are retained at full precision in `four_way_common_support.csv`. The comparison controls feature-success coverage, but each representation still defines a different map and basin geometry.",
    ]
    (HERE / "four_way_common_support.md").write_text("\n".join(lines) + "\n")

    report = {
        "status": "complete",
        "scope": "Full-record member-radius maps; exact evaluation-ID intersection across CrystalWeave, Magpie, CrystalNN graphlets, and Voronoi graphlets. Maps, centroids, radii, and saved classifications remain representation-specific.",
        "common_populations": {
            population: {
                "n": len(values),
                "ids_sha256": ids_sha256(values),
            }
            for population, values in common_ids.items()
        },
        "rows": long_rows,
        "inputs": inputs,
        "script_sha256": sha256(Path(__file__)),
        "checks": {
            "unique_identifiers": True,
            "saved_flags_reproduced_from_distance_and_threshold": True,
            "same_evaluation_identifiers_across_all_four_representations": True,
        },
    }
    (HERE / "four_way_common_support.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
