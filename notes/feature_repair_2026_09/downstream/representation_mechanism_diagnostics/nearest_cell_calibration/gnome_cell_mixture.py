#!/usr/bin/env python3
"""Standardize later-ICSD acceptance to GNoME's fixed-center cell mixture.

Uses only the saved nearest-cell-calibration diagnostic, with two explicitly
scoped bootstrap intervals. Writes only beside this script and within its
existing method/cutoff subdirectories. No features/maps are fitted or changed.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METHODS = ("production", "magpie", "graphlet")
LABELS = {"production": "Crystal-WL", "magpie": "Magpie", "graphlet": "CrystalNN graphlets"}
N_BOOTSTRAP = 5000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, inputs):
    inputs[str(path.relative_to(HERE))] = {"sha256": sha(path), "bytes": path.stat().st_size}
    frame = pd.read_csv(path, dtype={"record_key": str}, float_precision="round_trip")
    assert frame.record_key.is_unique and frame.record_key.notna().all()
    assert frame.in_basin.isin([True, False]).all()
    return frame


def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975], method="linear")]


def main():
    input_summary = json.loads((HERE / "summary.json").read_text())
    inputs = {"summary.json": {"sha256": sha(HERE / "summary.json"), "bytes": (HERE / "summary.json").stat().st_size}}
    results = []
    bootstrap_frames = []
    for method_index, method in enumerate(METHODS):
        for cutoff in (1990, 2000, 2010):
            directory = HERE / method / f"T{cutoff}"
            ic = load(directory / "icsd.csv", inputs)
            gn = load(directory / "gnome.csv", inputs)
            assert len(gn) == 5000
            assert len(ic) == input_summary["methods"][method][str(cutoff)]["icsd"]["n"]
            ic_cells = ic.groupby("assigned_community").in_basin.agg(icsd_n="size", icsd_in="sum")
            gn_cells = gn.groupby("assigned_community").in_basin.agg(gnome_n="size", gnome_in="sum")
            cells = ic_cells.join(gn_cells, how="inner").sort_index()
            assert len(cells) > 1
            cells["icsd_rate"] = cells.icsd_in / cells.icsd_n
            cells["gnome_rate"] = cells.gnome_in / cells.gnome_n
            cells["gnome_weight"] = cells.gnome_n / cells.gnome_n.sum()
            cells["gap_pp"] = 100 * (cells.icsd_rate - cells.gnome_rate)
            cells["weighted_gap_pp"] = cells.gnome_weight * cells.gap_pp
            cells.to_csv(directory / "gnome_cell_mixture_cells.csv", float_format="%.17g")
            ni, ng = cells.icsd_n.to_numpy(), cells.gnome_n.to_numpy()
            pi, pg = cells.icsd_rate.to_numpy(), cells.gnome_rate.to_numpy()
            weighted_ic = float(np.dot(ng, pi) / ng.sum())
            supported_gn = float(cells.gnome_in.sum() / ng.sum())
            gap = 100 * (weighted_ic - supported_gn)
            assert np.isclose(gap, cells.weighted_gap_pp.sum(), rtol=0, atol=1e-12)
            # Direct row-weight reconstruction independent of the grouped dot product.
            row_weights = ic.assigned_community.map(cells.gnome_n / cells.icsd_n).fillna(0)
            assert np.isclose((row_weights * ic.in_basin).sum() / row_weights.sum(), weighted_ic, rtol=0, atol=2e-15)

            rng = np.random.default_rng(np.random.SeedSequence([42, method_index, cutoff]))
            k = len(cells)
            cluster_values, conditional_values = [], []
            for start in range(0, N_BOOTSTRAP, 250):
                batch = min(250, N_BOOTSTRAP - start)
                # Resample K complete paired cells. Preserve both cohorts' counts
                # and flags within a chosen cell; normalize by resampled GNoME mass.
                draws = rng.multinomial(k, np.full(k, 1 / k), size=batch)
                cluster_values.extend(100 * (draws @ (ng * (pi - pg))) / (draws @ ng))
                # Independent nonparametric binary resampling within each fixed
                # cell is exactly binomial(n, observed cell acceptance proportion).
                sampled_ic = rng.binomial(ni, pi, size=(batch, k)) / ni
                sampled_gn = rng.binomial(ng, pg, size=(batch, k)) / ng
                conditional_values.extend(100 * ((sampled_ic - sampled_gn) @ ng) / ng.sum())
            cluster_values = np.asarray(cluster_values)
            conditional_values = np.asarray(conditional_values)
            bootstrap_frames.append(pd.DataFrame({"method": method, "cutoff": cutoff,
                "replicate": np.arange(N_BOOTSTRAP), "whole_cell_gap_pp": cluster_values,
                "conditional_within_cell_gap_pp": conditional_values}))

            no_ic = gn.loc[~gn.assigned_community.isin(cells.index)]
            no_gn = ic.loc[~ic.assigned_community.isin(cells.index)]
            result = {
                "method": method, "label": LABELS[method], "cutoff": cutoff,
                "n_shared_cells": k,
                "gnome_support_n": int(ng.sum()), "gnome_total_n": len(gn),
                "gnome_support_percent": 100 * float(ng.sum()) / len(gn),
                "gnome_excluded_n": len(no_ic), "gnome_excluded_cells": int(no_ic.assigned_community.nunique()),
                "gnome_excluded_in_basin_n": int(no_ic.in_basin.sum()),
                "icsd_support_n": int(ni.sum()), "icsd_total_n": len(ic),
                "icsd_support_percent": 100 * float(ni.sum()) / len(ic),
                "icsd_excluded_n": len(no_gn), "icsd_excluded_cells": int(no_gn.assigned_community.nunique()),
                "icsd_raw_full_percent": 100 * float(ic.in_basin.mean()),
                "gnome_raw_full_percent": 100 * float(gn.in_basin.mean()),
                "raw_full_gap_pp": 100 * float(ic.in_basin.mean() - gn.in_basin.mean()),
                "icsd_unweighted_support_percent": 100 * float(cells.icsd_in.sum() / ni.sum()),
                "gnome_support_percent_in_basin": 100 * supported_gn,
                "icsd_standardized_percent": 100 * weighted_ic,
                "standardized_gap_pp": gap,
                "whole_cell_bootstrap_95_percentile_gap_pp": interval(cluster_values),
                "conditional_within_cell_bootstrap_95_percentile_gap_pp": interval(conditional_values),
                "effective_weighted_cell_count": float(1 / np.sum((ng / ng.sum()) ** 2)),
                "largest_cell_gnome_weight": float(cells.gnome_weight.max()),
                "gnome_mass_in_cells_with_later_icsd_n_le_5": float(cells.loc[cells.icsd_n <= 5, "gnome_weight"].sum()),
                "gnome_support_absent_historical_cell_n": int(gn.loc[gn.assigned_community.isin(cells.index), "cell_absent_at_cutoff"].sum()),
                "bootstrap_n": N_BOOTSTRAP, "seed_sequence": [42, method_index, cutoff],
            }
            # Bounds if unsupported GNoME cells are retained and their missing
            # ICSD acceptance is allowed to range over [0,1]. These are not CIs.
            covered_mass = ng.sum() / len(gn)
            raw_gn = float(gn.in_basin.mean())
            result["all_gnome_missing_cell_identification_gap_bounds_pp"] = [
                100 * (covered_mass * weighted_ic - raw_gn),
                100 * (covered_mass * weighted_ic + 1 - covered_mass - raw_gn),
            ]
            results.append(result)
    output = {
        "diagnostic": "Later-ICSD acceptance standardized to GNoME nearest-cell mixture after cell-consistent historical radius calibration",
        "definition": "On cells occupied by both later ICSD and GNoME, adjusted ICSD acceptance is sum_c [n_GNoME,c / sum_c n_GNoME,c] * [in_ICSD,c / n_ICSD,c]. Compare with GNoME acceptance on those exact supported cells. The existing in_basin flags are unchanged.",
        "support": "A cell must contain at least one later-ICSD entry and one GNoME entry. Cells absent in the historical calibration remain included when they meet this support rule; all their existing flags are frontier. Excluded IDs and mass are reported explicitly.",
        "whole_cell_bootstrap": "5,000 replicates sample K shared cells with replacement uniformly, where K is observed shared-cell count. A selected cell carries both cohorts' entire observed counts and in-basin flags. Each replicate re-normalizes by its total GNoME mass. The 2.5th and 97.5th percentiles treat cells as independent resampling units and preserve within-cell dependence; the represented cell mixture can change between replicates.",
        "conditional_within_cell_bootstrap": "Separate 5,000-replicate interval holds observed cells and GNoME mixture weights fixed. Independently resample binary outcomes within each cohort/cell via Binomial(n, observed proportion). It assumes conditional independence of records within cells, so it is a finite-count sensitivity rather than the cell-aware primary interval.",
        "scope": "Conditional diagnostic of observed full-map cells and historical flags; it does not refit centers/representations, remove future information from the maps, estimate synthesis success, or establish representation-independent validation.",
        "results": results, "inputs": inputs,
        "script_sha256": sha(Path(__file__)),
        "checks": "Input unique IDs, binary flags, prescribed denominators, grouped weighted rate and independent row-weight reconstruction all pass.",
    }
    (HERE / "gnome_cell_mixture_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    pd.DataFrame(results).to_csv(HERE / "gnome_cell_mixture_rates.csv", index=False, float_format="%.17g")
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(HERE / "gnome_cell_mixture_bootstrap.csv", index=False, float_format="%.17g")
    lines = ["# GNoME cell-mixture standardization after nearest-cell calibration", "", output["definition"], "",
             "Positive gaps mean later ICSD has higher acceptance. Rates use shared later-ICSD/GNoME cell support; unsupported GNoME structures are excluded from both the target mixture and comparison rate.", "",
             "| Method | Cutoff | GNoME support | Later-ICSD support | Adjusted ICSD | GNoME | Adjusted gap (pp) | Whole-cell bootstrap 95% interval |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results:
        lo, hi = r["whole_cell_bootstrap_95_percentile_gap_pp"]
        lines.append(f"| {r['label']} | {r['cutoff']} | {r['gnome_support_n']:,}/5,000 ({r['gnome_support_percent']:.2f}%) | {r['icsd_support_n']:,}/{r['icsd_total_n']:,} | {r['icsd_standardized_percent']:.2f}% | {r['gnome_support_percent_in_basin']:.2f}% | {r['standardized_gap_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "## Support and finite-count checks", "",
              "| Method | Cutoff | Shared cells | GNoME excluded / cells | Conditional within-cell 95% interval (pp) | Missing-cell identification bounds, all GNoME (pp) |",
              "| --- | ---: | ---: | ---: | --- | --- |"]
    for r in results:
        lo, hi = r["conditional_within_cell_bootstrap_95_percentile_gap_pp"]
        blo, bhi = r["all_gnome_missing_cell_identification_gap_bounds_pp"]
        lines.append(f"| {r['label']} | {r['cutoff']} | {r['n_shared_cells']} | {r['gnome_excluded_n']} / {r['gnome_excluded_cells']} | [{lo:+.2f}, {hi:+.2f}] | [{blo:+.2f}, {bhi:+.2f}] |")
    lines += ["", "The identification bounds retain all 5,000 GNoME entries and allow the unobserved later-ICSD acceptance in unsupported cells to range from zero to one. They describe missing-support uncertainty, not sampling confidence intervals.", "",
              "## Bootstrap interpretation", "", output["whole_cell_bootstrap"], "", output["conditional_within_cell_bootstrap"], "", output["scope"], "",
              "## Reproduction", "",
              "Run `python gnome_cell_mixture.py` with NumPy and pandas. The JSON records input hashes, unrounded counts/rates, support diagnostics, both intervals and seeds. Each method/cutoff directory contains `gnome_cell_mixture_cells.csv`; all 45,000 paired bootstrap draws are retained in `gnome_cell_mixture_bootstrap.csv`."]
    (HERE / "gnome_cell_mixture_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[6:17]))
    print(f"Wrote mixture extension; {len(inputs)} hashed inputs; {len(results)} method/cutoff comparisons.")


if __name__ == "__main__":
    main()
