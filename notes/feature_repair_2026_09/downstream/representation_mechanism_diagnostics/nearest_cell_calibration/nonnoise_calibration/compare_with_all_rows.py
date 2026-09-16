#!/usr/bin/env python3
"""Compare original-nonnoise calibration with the existing all-row diagnostic."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
METHODS = ("production", "magpie", "graphlet")
LABELS = {"production": "Crystal-WL", "magpie": "Magpie", "graphlet": "CrystalNN graphlets"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = {}
    def read(path):
        inputs[str(path.relative_to(BASE))] = sha(path)
        return json.loads(path.read_text())
    before, after = read(BASE / "summary.json"), read(HERE / "summary.json")
    mix_before = {(r["method"], r["cutoff"]): r for r in read(BASE / "gnome_cell_mixture_summary.json")["results"]}
    mix_after = {(r["method"], r["cutoff"]): r for r in read(HERE / "gnome_cell_mixture_summary.json")["results"]}
    assert before["common_icsd_ids_sha256"] == after["common_icsd_ids_sha256"]
    assert before["external_available_n"] == after["external_available_n"]
    assert before["external_common_n"] == after["external_common_n"]
    for path in set(before["inputs"]) & set(after["inputs"]):
        assert before["inputs"][path] == after["inputs"][path]
    comparisons = []
    production_identical = []
    for method in METHODS:
        for cutoff in (1990, 2000, 2010):
            a, b = before["methods"][method][str(cutoff)], after["methods"][method][str(cutoff)]
            am, bm = mix_before[method, cutoff], mix_after[method, cutoff]
            assert a["icsd"]["ids_sha256"] == b["icsd"]["ids_sha256"]
            for source in ("gnome", "mattergen", "mp", "jarvis", "alexandria"):
                assert a["external"][source]["common_successes"]["ids_sha256"] == b["external"][source]["common_successes"]["ids_sha256"]
            assert am["gnome_support_n"] == bm["gnome_support_n"]
            assert am["icsd_support_n"] == bm["icsd_support_n"]
            assert am["n_shared_cells"] == bm["n_shared_cells"]
            if method == "production":
                for name in ("cell_thresholds.csv", "icsd.csv", "gnome.csv", "mattergen.csv", "mp.csv", "jarvis.csv", "alexandria.csv"):
                    relative = Path(method) / f"T{cutoff}" / name
                    digest = sha(BASE / relative)
                    assert digest == sha(HERE / relative), relative
                    production_identical.append({"relative_path": str(relative), "sha256": digest})
            comparisons.append({
                "method": method, "cutoff": cutoff,
                "all_row_calibration_n": a["n_calibration_icsd"],
                "nonnoise_calibration_n": b["n_calibration_icsd"],
                "original_noise_excluded": b["n_original_noise_excluded_from_calibration"],
                "all_row_calibrated_cells": a["n_calibrated_cells"],
                "nonnoise_calibrated_cells": b["n_calibrated_cells"],
                "all_row_icsd_rate": a["icsd"]["in_basin_percent"],
                "nonnoise_icsd_rate": b["icsd"]["in_basin_percent"],
                "all_row_gnome_rate": a["external"]["gnome"]["common_successes"]["in_basin_percent"],
                "nonnoise_gnome_rate": b["external"]["gnome"]["common_successes"]["in_basin_percent"],
                "all_row_unadjusted_gap_pp": a["external"]["gnome"]["common_successes"]["icsd_minus_external_percentage_points"],
                "nonnoise_unadjusted_gap_pp": b["external"]["gnome"]["common_successes"]["icsd_minus_external_percentage_points"],
                "all_row_mixture_adjusted_gap_pp": am["standardized_gap_pp"],
                "nonnoise_mixture_adjusted_gap_pp": bm["standardized_gap_pp"],
                "nonnoise_whole_cell_bootstrap_95_gap_pp": bm["whole_cell_bootstrap_95_percentile_gap_pp"],
                "gnome_mixture_support_n": bm["gnome_support_n"],
                "icsd_mixture_support_n": bm["icsd_support_n"],
                "n_shared_mixture_cells": bm["n_shared_cells"],
            })
    report = {"scope": "Same full maps, exact common ICSD IDs, saved assignments and distances, external evaluation IDs, and later-ICSD/GNoME mixture support. The new variant removes original-partition noise from historical radius calibration before grouping remaining entries into fixed nearest-centroid cells.",
              "interpretation": "On common nonnoise historical calibration members this isolates the grouping change relative to original graph-community radius membership; centers are still fixed full-map centers. It is a diagnostic, not cutoff-trained validation.",
              "comparisons": comparisons, "checks": {"all_evaluation_ID_hashes_unchanged": True,
              "common_input_hashes_unchanged": True, "all_mixture_support_counts_unchanged": True,
              "production_row_outputs_and_thresholds_identical": production_identical},
              "inputs": inputs, "script_sha256": sha(Path(__file__))}
    (HERE / "comparison_with_all_rows.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Nonnoise historical calibration compared with all-row calibration", "", report["scope"], "", report["interpretation"], "",
             "Crystal-WL is unchanged because every entry in the common ICSD population has a nonnoise Crystal-WL label. All 21 Crystal-WL threshold and row-evaluation files are byte-identical between variants. Magpie and graphlet calibration exclude their own original noise; later evaluation populations remain unchanged.", "",
             "## Calibration membership and unadjusted GNoME contrast", "",
             "Gaps are ICSD minus GNoME acceptance in percentage points.", "",
             "| Method | Cutoff | All-row calibration n | Nonnoise calibration n | Noise excluded | All-row gap | Nonnoise gap |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in comparisons:
        lines.append(f"| {LABELS[r['method']]} | {r['cutoff']} | {r['all_row_calibration_n']:,} | {r['nonnoise_calibration_n']:,} | {r['original_noise_excluded']:,} | {r['all_row_unadjusted_gap_pp']:+.2f} | {r['nonnoise_unadjusted_gap_pp']:+.2f} |")
    lines += ["", "## GNoME cell-mixture standardization", "",
              "Support is identical between variants. The whole-cell interval uses 5,000 bootstrap replicates, retaining both cohorts' entire observed counts within each resampled cell.", "",
              "| Method | Cutoff | GNoME support / 5,000 | All-row adjusted gap | Nonnoise adjusted gap | Nonnoise whole-cell bootstrap 95% interval |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for r in comparisons:
        lo, hi = r["nonnoise_whole_cell_bootstrap_95_gap_pp"]
        lines.append(f"| {LABELS[r['method']]} | {r['cutoff']} | {r['gnome_mixture_support_n']:,} | {r['all_row_mixture_adjusted_gap_pp']:+.2f} | {r['nonnoise_mixture_adjusted_gap_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "Excluding noise from calibration leaves the qualitative diagnostic unchanged: Crystal-WL retains higher ICSD acceptance, the Magpie adjusted gaps are unresolved by whole-cell bootstrap intervals, and graphlets retain higher GNoME acceptance after mixture standardization. The graphlet Alexandria reversal remains absent in the corresponding unadjusted five-cohort tables.", "",
              "The 2010 mixture comparison remains locally sparse: 25.3% of supported GNoME weight in graphlets lies in cells containing at most five later ICSD entries. Complete support, missing-cell bounds, per-cell counts and both bootstrap definitions remain in the variant's separate mixture report and JSON.", "",
              "## Reproduction", "",
              "Run `reproduce.py`, `verify.py`, `gnome_cell_mixture.py`, `gnome_cell_mixture_verify.py`, then `compare_with_all_rows.py` in this directory. The first and third scripts require NumPy/pandas; the checks use the Python standard library. `verification.json` and `gnome_cell_mixture_verification.json` contain the independent checks. All-row results in the parent directory are preserved."]
    (HERE / "comparison_with_all_rows.md").write_text("\n".join(lines) + "\n")
    print("Comparison verified: nine method/cutoff pairs; unchanged IDs/support; 21 byte-identical Crystal-WL files.")


if __name__ == "__main__":
    main()
