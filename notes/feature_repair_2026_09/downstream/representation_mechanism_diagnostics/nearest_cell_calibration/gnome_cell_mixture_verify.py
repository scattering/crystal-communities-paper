#!/usr/bin/env python3
"""Independently verify saved cell-mixture estimates and bootstrap intervals."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (index - lo) * (values[hi] - values[lo])


def main():
    data = json.loads((HERE / "gnome_cell_mixture_summary.json").read_text())
    for path, meta in data["inputs"].items():
        assert sha(HERE / path) == meta["sha256"]
    assert sha(HERE / "gnome_cell_mixture.py") == data["script_sha256"]
    boot = defaultdict(list)
    for row in rows(HERE / "gnome_cell_mixture_bootstrap.csv"):
        boot[row["method"], int(row["cutoff"])].append(row)
    for result in data["results"]:
        directory = HERE / result["method"] / f"T{result['cutoff']}"
        grouped = {}
        for source in ("icsd", "gnome"):
            grouped[source] = defaultdict(lambda: [0, 0])
            source_rows = rows(directory / f"{source}.csv")
            assert len({r["record_key"] for r in source_rows}) == len(source_rows)
            for row in source_rows:
                val = grouped[source][row["assigned_community"]]
                val[0] += 1
                val[1] += row["in_basin"] == "True"
        shared = set(grouped["icsd"]) & set(grouped["gnome"])
        assert len(shared) == result["n_shared_cells"]
        n_ic = sum(grouped["icsd"][c][0] for c in shared)
        n_gn = sum(grouped["gnome"][c][0] for c in shared)
        weighted = sum(grouped["gnome"][c][0] * grouped["icsd"][c][1] / grouped["icsd"][c][0] for c in shared) / n_gn
        gn_rate = sum(grouped["gnome"][c][1] for c in shared) / n_gn
        assert n_ic == result["icsd_support_n"] and n_gn == result["gnome_support_n"]
        assert math.isclose(100 * weighted, result["icsd_standardized_percent"], abs_tol=1e-12)
        assert math.isclose(100 * gn_rate, result["gnome_support_percent_in_basin"], abs_tol=1e-12)
        assert math.isclose(100 * (weighted - gn_rate), result["standardized_gap_pp"], abs_tol=1e-12)
        draws = boot[result["method"], result["cutoff"]]
        assert len(draws) == result["bootstrap_n"] == 5000
        assert sorted(int(r["replicate"]) for r in draws) == list(range(5000))
        for column, field in (("whole_cell_gap_pp", "whole_cell_bootstrap_95_percentile_gap_pp"),
                              ("conditional_within_cell_gap_pp", "conditional_within_cell_bootstrap_95_percentile_gap_pp")):
            values = [float(r[column]) for r in draws]
            assert all(math.isfinite(v) for v in values)
            for actual, expected in zip((percentile(values, .025), percentile(values, .975)), result[field]):
                assert math.isclose(actual, expected, abs_tol=1e-12)
    output = {"status": "pass", "input_hashes_verified": len(data["inputs"]),
              "method_cutoff_point_estimates_independently_verified": len(data["results"]),
              "bootstrap_values_and_percentile_intervals_verified": 90000,
              "scope": "Stdlib aggregation from original row-level diagnostic flags independently reproduces supported cell counts, cohort denominators and standardized rates. Saved bootstrap replicate IDs, finiteness and both percentile intervals are independently verified; random draws are not independently regenerated.",
              "summary_sha256": sha(HERE / "gnome_cell_mixture_summary.json"),
              "script_sha256": sha(Path(__file__))}
    (HERE / "gnome_cell_mixture_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
