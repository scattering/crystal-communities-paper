#!/usr/bin/env python3
"""Independent stdlib check of nearest-cell thresholds, IDs and row-level flags."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DOWN = Path("notes/feature_repair_2026_09/downstream")
EXT = DOWN / "external_representation"
METHODS = ("production", "magpie", "graphlet")
SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")


def table(path, key):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows)
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def linear_p95(values):
    ordered = sorted(values)
    at = .95 * (len(ordered) - 1)
    lo = math.floor(at)
    hi = math.ceil(at)
    return ordered[lo] + (at - lo) * (ordered[hi] - ordered[lo])


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    for path, meta in summary["inputs"].items():
        assert digest(ROOT / path) == meta["sha256"], path
    assert digest(HERE / "reproduce.py") == summary["script_sha256"]
    refs = {
        "production": table(ROOT / EXT / "production_reference/icsd_full.csv", "record_key"),
        "magpie": table(ROOT / EXT / "consistent_transform/basis/icsd_full.csv", "record_key"),
        "graphlet": table(ROOT / EXT / "graphlet/basis/icsd_full.csv", "record_key"),
    }
    common = set.intersection(*(set(v) for v in refs.values()))
    assert len(common) == 150247
    assert hashlib.sha256("\n".join(sorted(common)).encode()).hexdigest() == summary["common_icsd_ids_sha256"]
    years = {key: int(float(refs["production"][key]["year"])) for key in common}
    ext = {method: {} for method in METHODS}
    for method in METHODS:
        for source in SOURCES:
            if method == "production":
                matches = list((ROOT / DOWN / "external" / source).glob("*frontier_records.csv"))
                assert len(matches) == 1
                path = matches[0]
                key = "zip_member" if source == "mattergen" else "material_id"
            else:
                directory = "consistent_transform" if method == "magpie" else "graphlet"
                path = ROOT / EXT / directory / "external" / source / "projection_full.csv"
                key = "record_key"
            ext[method][source] = table(path, key)
    ext_common = {source: set.intersection(*(set(ext[m][source]) for m in METHODS)) for source in SOURCES}
    evaluated_rows = 0
    verified_cells = 0
    for method in METHODS:
        for cutoff in (1990, 2000, 2010):
            directory = HERE / method / f"T{cutoff}"
            stats = summary["methods"][method][str(cutoff)]
            cells = table(directory / "cell_thresholds.csv", "assigned_community")
            values = defaultdict(list)
            for key in common:
                if years[key] <= cutoff:
                    row = refs[method][key]
                    values[row["assigned_community"]].append(float(row["nearest_centroid_distance"]))
            assert set(values) == set(cells)
            for cell, data in values.items():
                assert len(data) == int(cells[cell]["n_calibration_members"])
                assert math.isclose(linear_p95(data), float(cells[cell]["nearest_cell_p95"]), rel_tol=2e-15, abs_tol=1e-14)
                verified_cells += 1
            assert sum(map(len, values.values())) == stats["n_calibration_icsd"]
            for source in ("icsd",) + SOURCES:
                result = table(directory / f"{source}.csv", "record_key")
                inputs = refs[method] if source == "icsd" else ext[method][source]
                expected = {key for key in common if years[key] > cutoff} if source == "icsd" else set(inputs)
                assert set(result) == expected
                for key, row in result.items():
                    original = inputs[key]
                    cell = row["assigned_community"]
                    distance = float(row["nearest_centroid_distance"])
                    assert cell == original["assigned_community"]
                    assert distance == float(original["nearest_centroid_distance"])
                    assert row["original_full_map_in_basin"] == original["in_basin"]
                    absent = cell not in cells
                    assert (row["cell_absent_at_cutoff"] == "True") == absent
                    assert int(row["n_calibration_members"]) == (0 if absent else len(values[cell]))
                    expected_in = not absent and distance <= float(cells[cell]["nearest_cell_p95"])
                    assert (row["in_basin"] == "True") == expected_in
                    if absent:
                        assert row["nearest_cell_p95"] == ""
                    else:
                        assert float(row["nearest_cell_p95"]) == float(cells[cell]["nearest_cell_p95"])
                    if source != "icsd":
                        assert (row["common_three_representation_success"] == "True") == (key in ext_common[source])
                    evaluated_rows += 1
                groups = {"icsd": expected} if source == "icsd" else {"own_successes": expected, "common_successes": ext_common[source]}
                for population, keys in groups.items():
                    expected_stats = stats["icsd"] if source == "icsd" else stats["external"][source][population]
                    assert len(keys) == expected_stats["n"]
                    assert sum(result[key]["in_basin"] == "True" for key in keys) == expected_stats["n_in_basin"]
                    assert sum(result[key]["cell_absent_at_cutoff"] == "True" for key in keys) == expected_stats["n_assigned_to_absent_cell"]
    report = {"status": "pass", "hashed_inputs_verified": len(summary["inputs"]),
              "common_icsd_ids_verified": len(common), "threshold_cells_verified": verified_cells,
              "row_classifications_verified": evaluated_rows,
              "threshold_verification": "Independent sorted-array linear interpolation, stdlib only; numeric tolerance 2e-15 relative or 1e-14 absolute.",
              "row_verification": "Exact IDs, assignment, saved distance, original flag, historical count, absent-cell flag, nearest-cell radius, new binary flag and summary counts.",
              "summary_sha256": digest(HERE / "summary.json"),
              "reproduce_script_sha256": digest(HERE / "reproduce.py"),
              "verify_script_sha256": digest(Path(__file__))}
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
