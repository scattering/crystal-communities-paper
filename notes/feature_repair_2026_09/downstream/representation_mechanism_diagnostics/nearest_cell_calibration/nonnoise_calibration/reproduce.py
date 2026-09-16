#!/usr/bin/env python3
"""Calibrate fixed full-map nearest-centroid cells using historical ICSD distances.

This diagnostic changes p95 calibration membership, not centers, representations,
partitions, assignments or distances. It is not cutoff-trained validation.
Writes only beside this script. Dependencies: numpy, pandas.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
DOWN = Path("notes/feature_repair_2026_09/downstream")
EXT = DOWN / "external_representation"
METHODS = ("production", "magpie", "graphlet")
SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")
LABELS = {"production": "Crystal-WL", "magpie": "Magpie", "graphlet": "CrystalNN graphlets",
          "icsd": "ICSD", "gnome": "GNoME", "mattergen": "MatterGen", "mp": "MP",
          "jarvis": "JARVIS", "alexandria": "Alexandria"}
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for blob in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(blob)
    return h.hexdigest()


def read_csv(path, key):
    path = Path(path)
    absolute = ROOT / path
    INPUTS[str(path)] = {"sha256": sha(absolute), "bytes": absolute.stat().st_size}
    frame = pd.read_csv(absolute, dtype={key: str}, float_precision="round_trip")
    assert frame[key].notna().all() and frame[key].is_unique, (path, "invalid IDs")
    return frame.set_index(key)


def projections(path, key="record_key"):
    frame = read_csv(path, key)
    assert frame.assigned_community.ge(0).all(), (path, "missing nearest centroid")
    assert np.isfinite(frame.nearest_centroid_distance).all(), (path, "nonfinite distance")
    assert frame.nearest_centroid_distance.ge(0).all(), path
    assert frame.in_basin.isin([True, False]).all(), (path, "invalid saved flag")
    assert np.array_equal(frame.in_basin.to_numpy(),
                          (frame.nearest_centroid_distance <= frame.community_threshold_p95).to_numpy()), (path, "saved flag mismatch")
    return frame


def one(directory, pattern):
    paths = sorted((ROOT / directory).glob(pattern))
    assert len(paths) == 1, (directory, pattern)
    return paths[0].relative_to(ROOT)


def ids_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def score(frame, cells):
    out = frame[["assigned_community", "nearest_centroid_distance", "in_basin"]].copy()
    out.rename(columns={"in_basin": "original_full_map_in_basin"}, inplace=True)
    out["n_calibration_members"] = out.assigned_community.map(cells.n_calibration_members).fillna(0).astype(int)
    out["cell_absent_at_cutoff"] = out.n_calibration_members.eq(0)
    out["nearest_cell_p95"] = out.assigned_community.map(cells.nearest_cell_p95)
    out["in_basin"] = ~out.cell_absent_at_cutoff & out.nearest_centroid_distance.le(out.nearest_cell_p95)
    out.index.name = "record_key"
    return out


def count(out):
    n = len(out)
    return {"n": n, "n_in_basin": int(out.in_basin.sum()),
            "in_basin_percent": 100 * float(out.in_basin.mean()),
            "n_frontier": int((~out.in_basin).sum()),
            "n_assigned_to_absent_cell": int(out.cell_absent_at_cutoff.sum()),
            "n_distinct_absent_cells_selected": int(out.loc[out.cell_absent_at_cutoff, "assigned_community"].nunique()),
            "n_assigned_to_singleton_calibration_cell": int(out.n_calibration_members.eq(1).sum()),
            "n_assigned_to_zero_radius_cell": int(out.nearest_cell_p95.eq(0).sum()),
            "original_full_map_in_basin_n": int(out.original_full_map_in_basin.sum()),
            "original_full_map_in_basin_percent": 100 * float(out.original_full_map_in_basin.mean()),
            "ids_sha256": ids_hash(out.index)}


def main():
    references = {
        "production": projections(EXT / "production_reference/icsd_full.csv"),
        "magpie": projections(EXT / "consistent_transform/basis/icsd_full.csv"),
        "graphlet": projections(EXT / "graphlet/basis/icsd_full.csv"),
    }
    common_ids = sorted(set.intersection(*(set(f.index) for f in references.values())))
    assert len(common_ids) == 150247
    years = pd.to_numeric(references["production"].loc[common_ids, "year"], errors="raise")
    assert years.notna().all() and np.equal(years, years.astype(int)).all()
    years = years.astype(int)
    for path in (Path("notes/feature_repair_2026_09/full_run_results/magpie/graph/community_assignments.csv"),
                 DOWN / "representations/graphlet-dated-replay/graphlet/community_assignments.csv"):
        alt = read_csv(path, "icsd_id")
        check = pd.to_numeric(alt.loc[common_ids, "year"], errors="raise").astype(int)
        assert np.array_equal(years.to_numpy(), check.to_numpy()), (path, "year discrepancy")

    partition_paths = {
        "production": Path("notes/feature_repair_2026_09/full_run_results/production/graph/community_assignments.csv"),
        "magpie": Path("notes/feature_repair_2026_09/full_run_results/magpie/graph/community_assignments.csv"),
        "graphlet": DOWN / "representations/graphlet-dated-replay/graphlet/community_assignments.csv",
    }
    original_labels = {}
    for method, path in partition_paths.items():
        part = read_csv(path, "icsd_id")
        original_labels[method] = part.loc[common_ids, "community"].astype(int)
    assert original_labels["production"].ge(0).all()

    external = {m: {} for m in METHODS}
    external_shared = {}
    for source in SOURCES:
        external["production"][source] = projections(one(DOWN / "external" / source, "*frontier_records.csv"),
                                                     "zip_member" if source == "mattergen" else "material_id")
        for method, directory in (("magpie", "consistent_transform"), ("graphlet", "graphlet")):
            external[method][source] = projections(EXT / directory / "external" / source / "projection_full.csv")
        external_shared[source] = sorted(set.intersection(*(set(external[m][source].index) for m in METHODS)))
    assert [len(external_shared[s]) for s in SOURCES] == [5000, 386, 4999, 4964, 5000]

    manifest = {
        "diagnostic": "Historical nonnoise p95 calibration within fixed full-map nearest-centroid cells",
        "scope": "All methods use exactly 150,247 shared dated ICSD IDs. At each T, calibration uses only entries dated <=T with a nonnoise original graph-community label for that method. Those entries are grouped by their nearest full-map centroid and their saved distances define the cell p95. Score later common-ID ICSD and the external cohorts against those cell-distance p95 values. Fixed full-map centers, fitted representations and partitions retain future information; this is not cutoff-trained validation.",
        "rule": "in_basin = cell existed by cutoff AND saved_distance <= numpy.quantile(calibration_distances, 0.95, method='linear')",
        "absent_cells": "Always frontier, retained in denominator; missing radius serialized as an empty CSV field.",
        "singleton_cells": "Radius is that member's distance to the fixed full-map centroid; it need not be zero because the centroid is not refitted.",
        "external_denominators": "Both each method's own successful population and the source-wise common successful IDs are reported. Primary method-gap comparison uses the latter.",
        "missing_data": "Existing feature failures stay excluded; no new feature extraction is attempted.",
        "calibration_population": "Historically dated, original-graph-nonnoise members within the common ICSD IDs; all later common-ID ICSD entries remain eligible evaluation rows, including original noise.",
        "year_check": "Production projection years agree exactly with both Magpie and graphlet original partition years on all common IDs.",
        "common_icsd_n": len(common_ids), "common_icsd_ids_sha256": ids_hash(common_ids),
        "ids_hash_convention": "UTF-8 sorted keys joined by LF, no trailing LF",
        "external_common_n": {s: len(ids) for s, ids in external_shared.items()},
        "external_available_n": {m: {s: len(external[m][s]) for s in SOURCES} for m in METHODS},
        "methods": {},
        "packages": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    flat_rows = []
    for method in METHODS:
        ref = references[method].loc[common_ids].copy()
        ref["year"] = years
        ref["original_graph_community"] = original_labels[method]
        manifest["methods"][method] = {}
        all_full_map_cells = set(ref.assigned_community)
        for cutoff in (1990, 2000, 2010):
            historical = ref.loc[ref.year <= cutoff]
            training = historical.loc[historical.original_graph_community >= 0]
            later = ref.loc[ref.year > cutoff]
            cells = training.groupby("assigned_community").nearest_centroid_distance.agg(
                n_calibration_members="size",
                nearest_cell_p95=lambda vals: float(np.quantile(vals.to_numpy(), .95, method="linear")),
            )
            assert int(cells.n_calibration_members.sum()) == len(training)
            directory = HERE / method / f"T{cutoff}"
            directory.mkdir(parents=True, exist_ok=True)
            cells.to_csv(directory / "cell_thresholds.csv", float_format="%.17g")
            scored = score(later, cells)
            scored["year"] = years.loc[scored.index]
            scored.to_csv(directory / "icsd.csv", float_format="%.17g")
            ic = count(scored)
            summary = {"n_calibration_icsd": len(training), "n_historical_common_icsd_total": len(historical),
                       "n_original_noise_excluded_from_calibration": len(historical) - len(training), "n_later_icsd": len(later),
                       "n_calibrated_cells": len(cells),
                       "n_cells_occupied_in_common_full_record": len(all_full_map_cells),
                       "n_common_full_record_cells_absent_at_cutoff": len(all_full_map_cells - set(cells.index)),
                       "n_singleton_cells": int(cells.n_calibration_members.eq(1).sum()),
                       "n_zero_radius_cells": int(cells.nearest_cell_p95.eq(0).sum()),
                       "icsd": ic, "external": {}}
            flat_rows.append({"method": method, "cutoff": cutoff, "source": "icsd", "denominator": "common", **ic})
            for source in SOURCES:
                out = score(external[method][source], cells)
                out["common_three_representation_success"] = out.index.isin(external_shared[source])
                out.to_csv(directory / f"{source}.csv", float_format="%.17g")
                stats = {"own_successes": count(out), "common_successes": count(out.loc[external_shared[source]])}
                for population, values in stats.items():
                    values["icsd_minus_external_percentage_points"] = ic["in_basin_percent"] - values["in_basin_percent"]
                    flat_rows.append({"method": method, "cutoff": cutoff, "source": source, "denominator": population, **values})
                summary["external"][source] = stats
            manifest["methods"][method][str(cutoff)] = summary

    manifest["inputs"] = INPUTS
    manifest["script_sha256"] = sha(Path(__file__))
    pd.DataFrame(flat_rows).to_csv(HERE / "rates.csv", index=False, float_format="%.17g")
    (HERE / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Historical calibration of nearest-centroid cells using original nonnoise members", "", manifest["scope"], "",
             "No features, centroids, graph partitions or nearest-centroid assignments were recomputed. The p95 radius is calibrated from distances of historically dated original-nonnoise members reassigned into nearest-centroid cells. The historically dated original-noise entries are excluded from calibration; all later ICSD and external evaluation rows are retained. Cells without historical members count as frontier. A singleton's radius equals its saved distance to the fixed center, which can be positive.", "",
             "## Rates on identical evaluation populations", "",
             "External shared-success counts: GNoME 5,000; MatterGen 386; MP 4,999; JARVIS 4,964; Alexandria 5,000.", "",
             "| Method | Cutoff | Historical calibration n | Later ICSD rate (n) | GNoME | MatterGen | MP | JARVIS | Alexandria | ICSD higher than cohorts |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        for cutoff in (1990, 2000, 2010):
            s = manifest["methods"][method][str(cutoff)]
            rates = [s["external"][source]["common_successes"] for source in SOURCES]
            high = sum(v["icsd_minus_external_percentage_points"] > 0 for v in rates)
            lines.append(f"| {LABELS[method]} | {cutoff} | {s['n_calibration_icsd']:,} | {s['icsd']['in_basin_percent']:.2f}% ({s['icsd']['n']:,}) | "
                         + " | ".join(f"{v['in_basin_percent']:.2f}%" for v in rates) + f" | {high}/5 |")
    lines += ["", "## ICSD minus computed-cohort gaps (percentage points)", "",
              "| Method | Cutoff | GNoME | MatterGen | MP | JARVIS | Alexandria |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        for cutoff in (1990, 2000, 2010):
            s = manifest["methods"][method][str(cutoff)]
            lines.append(f"| {LABELS[method]} | {cutoff} | " + " | ".join(
                f"{s['external'][source]['common_successes']['icsd_minus_external_percentage_points']:+.2f}" for source in SOURCES) + " |")
    lines += ["", "## Coverage: entries assigned to cells absent at cutoff", "",
              "All such entries are frontier and remain in the denominators above. Counts below use common external successes; JSON also reports each method's own-success counts.", "",
              "| Method | Cutoff | Calibrated cells | Full-record occupied cells absent | Later ICSD | GNoME | MatterGen | MP | JARVIS | Alexandria |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        for cutoff in (1990, 2000, 2010):
            s = manifest["methods"][method][str(cutoff)]
            lines.append(f"| {LABELS[method]} | {cutoff} | {s['n_calibrated_cells']} | {s['n_common_full_record_cells_absent_at_cutoff']} | {s['icsd']['n_assigned_to_absent_cell']} | "
                         + " | ".join(str(s["external"][source]["common_successes"]["n_assigned_to_absent_cell"]) for source in SOURCES) + " |")
    lines += ["", "## Reproduction and limits", "",
              "Run `python reproduce.py` with NumPy and pandas. `summary.json` records exact integer counts, all denominator choices, cell coverage, singleton and zero-radius cases, verified IDs/years and every input SHA-256. `rates.csv` provides all unrounded rates. Each method/cutoff directory contains its cell thresholds and all row-level evaluations.", "",
              "The original full-map flag is retained beside each new classification for traceability. It uses full-record graph-community radii and is not a historical comparator. This diagnostic fixes the grouping mismatch in radius calibration while holding the original-nonnoise calibration population and fitted centers fixed; it does not remove all dependence on representation, clustering or future data. It provides no synthesis-success calibration."]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote diagnostic to {HERE}; {len(INPUTS)} hashed inputs; {len(common_ids)} common ICSD IDs.")
    print("\n".join(lines[8:20]))


if __name__ == "__main__":
    main()
