#!/usr/bin/env python3
"""Read-only independent reconstruction of the A-Lab binary outcome statistics.

Rebuilds tables from target IDs/outcomes and projection CSVs, enumerates central
hypergeometric Fisher probabilities, and root-inverts the noncentral Fisher
distribution for conditional odds ratios and exact intervals. It does not call
the producer's scipy.stats.contingency.odds_ratio implementation or write files.
Run with an environment containing NumPy and SciPy; results print as JSON.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent
TARGETS = DOWNSTREAM / "alab_mp_targets/source/alab_targets.csv"
COMPARISON = HERE / "outcome_comparison.json"
MODELS = {
    "CrystalWeave": DOWNSTREAM / "alab_mp_targets/output/target_projection_records.csv",
    "Graphlet (CrystalNN)": HERE / "graphlet_crystalnn/projection_records.csv",
    "Graphlet (VoronoiNN)": HERE / "graphlet_voronoinn/projection_records.csv",
    "AMD-100": HERE / "amd/projection_records.csv",
}


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_statistics(table):
    a, b, c, d = table.ravel()
    population = int(table.sum())
    in_basin = int(a + c)
    made = int(a + b)

    def distribution(log_odds):
        return stats.nchypergeom_fisher(population, in_basin, made, math.exp(log_odds))

    def root(function):
        return math.exp(optimize.brentq(function, -50, 50, xtol=1e-13))

    estimate = root(lambda value: distribution(value).mean() - a)
    lower = root(lambda value: distribution(value).sf(a - 1) - 0.025)
    upper = root(lambda value: distribution(value).cdf(a) - 0.025)
    support = np.arange(max(0, made - (population - in_basin)), min(made, in_basin) + 1)
    probabilities = stats.hypergeom.pmf(support, population, in_basin, made)
    observed = stats.hypergeom.pmf(a, population, in_basin, made)
    fisher = min(1.0, float(probabilities[probabilities <= observed * (1 + 1e-12)].sum()))
    return {
        "table": table.tolist(),
        "n": population,
        "made_fraction_counts": [[int(a), int(a + c)], [int(b), int(b + d)]],
        "conditional_odds_ratio": estimate,
        "conditional_exact_95ci": [lower, upper],
        "fisher_two_sided": fisher,
    }


def main():
    target_rows = rows(TARGETS)
    truth = {row["mp_id"]: row["corrected_outcome"] for row in target_rows}
    assert len(target_rows) == len(truth) == 57
    definitive = {mid for mid, outcome in truth.items() if outcome in ("made", "not_obtained")}
    assert len(definitive) == 51
    assert sum(outcome == "made" for outcome in truth.values()) == 36
    assert sum(outcome == "not_obtained" for outcome in truth.values()) == 15
    reported = json.loads(COMPARISON.read_text())
    results = {}
    for name, path in MODELS.items():
        records = rows(path)
        ids = [row["mp_id"] for row in records]
        assert len(ids) == len(set(ids)) == 57 and set(ids) == set(truth), name
        assert all(row["corrected_outcome"] == truth[row["mp_id"]] for row in records), name
        table = np.zeros((2, 2), dtype=int)
        for row in records:
            outcome = truth[row["mp_id"]]
            if outcome not in ("made", "not_obtained"):
                continue
            assert row["in_basin"] in ("True", "False")
            table[0 if outcome == "made" else 1, 0 if row["in_basin"] == "True" else 1] += 1
        result = independent_statistics(table)
        saved = reported["representations"][name]["common_successful_support"]
        assert table.tolist() == saved["table_rows_made_not_obtained_cols_in_frontier"], name
        assert abs(result["conditional_odds_ratio"] - saved["conditional_exact_odds_ratio"]) < 1e-9, name
        assert np.allclose(result["conditional_exact_95ci"], saved["conditional_exact_95ci"],
                           rtol=1e-8, atol=1e-10), name
        assert abs(result["fisher_two_sided"] - saved["fisher_exact_p_two_sided"]) < 1e-12, name
        results[name] = result
    alternatives = [name for name in MODELS if name != "CrystalWeave"]
    order = sorted(alternatives, key=lambda name: results[name]["fisher_two_sided"])
    previous = 0.0
    for rank, name in enumerate(order):
        adjusted = max(previous, min(1.0, (len(order) - rank) * results[name]["fisher_two_sided"]))
        previous = adjusted
        saved = reported["representations"][name]["common_successful_support"]
        assert abs(adjusted - saved["holm_adjusted_p_three_alternative_maps"]) < 1e-12, name
        results[name]["holm_adjusted_p_three_alternatives"] = adjusted
    assert set(reported["common_definitive_ids"]) == definitive
    files = [TARGETS, *MODELS.values(), COMPARISON, Path(__file__)]
    print(json.dumps({
        "status": "PASS",
        "checks": "Identical 57 unique IDs and corrected labels; same 51 definitive IDs; independently reconstructed tables, enumerated Fisher, root-inverted conditional OR/CI and Holm-3 match producer.",
        "results": results,
        "input_sha256": {str(path.relative_to(DOWNSTREAM)): digest(path) for path in files},
        "software": {"numpy": np.__version__, "scipy": __import__("scipy").__version__},
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
