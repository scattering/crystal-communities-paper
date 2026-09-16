#!/usr/bin/env python3
"""Regenerate calibration and joint-prior analyses from explicit repaired inputs.

The reference coordinates and memberships are the saved full-record map.
Cutoff centroids/radii use dated members through T; external records retain
their full-map assignments for the historical manuscript calibration. Every
projected record counts in calibration; composition matching is a separate
restricted-population control. Retrospective statistics reuse the reviewed
bootstrap/permutation implementations with newly computed inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notes/review_2026_08"))
import fig3c_corrected_denominators as calibration
import retrospective_quadrant as retrospective
import analyze_composition_matched_ai as composition
import analyze_synthesis_retrodiction as synthesis
import analyze_structural_accessibility_revised as accessibility
from formula_conventions import scale_invariant_formula_key
from stampede_formula_reference import composition_class, anonymized_formula


SOURCE_SLUGS = {"GNoME": "gnome", "MatterGen": "mattergen", "MP": "mp",
                "JARVIS": "jarvis", "Alexandria": "alexandria"}


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=lambda x: x.item()) + "\n")


def frame(path):
    return pd.read_csv(path, keep_default_na=False)


def formula_keys(values):
    """Return scale-invariant keys while preserving row alignment."""
    return values.map(lambda value: scale_invariant_formula_key(str(value)))


def command(name, *args):
    argv = [sys.executable, str(ROOT / "scripts" / name), *map(str, args)]
    print("Running", name, flush=True)
    subprocess.run(argv, check=True)


def load_inputs(args):
    p = args.production
    x = np.load(p / "features_pca.npy")
    labels = frame(p / "graph/community_assignments.csv")
    samples = frame(p / "sample_assignments.csv")
    assert len(labels) == len(x) == len(samples)
    assert labels.icsd_id.is_unique and np.array_equal(labels.icsd_id, samples.icsd_id)
    assert np.array_equal(labels.year, samples.year) and np.isfinite(x).all()
    ids = labels.icsd_id.to_numpy(dtype=int)
    years = pd.to_numeric(labels.year, errors="coerce").fillna(-1).to_numpy(dtype=int)
    communities = labels.community.to_numpy(dtype=int)
    index = synthesis.load_icsd_index(args.icsd_index)
    sources, paths = {}, {}
    for name, slug in SOURCE_SLUGS.items():
        prefix = "mattergen-public" if slug == "mattergen" else slug
        path = args.downstream / "external" / slug / f"{prefix}_frontier_records.csv"
        metadata = json.loads(path.with_name(f"{prefix}_frontier_summary.json").read_text())
        assert metadata["feature_version"].startswith("crystal-features-v2")
        assert metadata["threshold_mode"] == "per_community_p95"
        sources[name] = frame(path)
        assert len(sources[name]) == metadata["n_featurized"]
        paths[name] = path
    return x, ids, years, communities, index, sources, paths


def source_flags(source, thresholds):
    radius = source.assigned_community.map(thresholds)
    return (radius.notna() & (source.nearest_centroid_distance <= radius)).to_numpy(dtype=bool)


def rate(values):
    return calibration.rate_block(int(np.asarray(values).sum()), len(values))


def build_calibration(args, x, ids, years, communities, sources, paths):
    out = args.downstream / "calibration"
    out.mkdir(exist_ok=True)
    results, sweeps, classification = [], [], []
    for cutoff in [1990, 2000, 2010]:
        cids, centers, radii, thresholds, ntrain = calibration.cutoff_map(x, years, communities, cutoff)
        held = calibration.heldout_classify(x, ids, years, communities, cutoff, cids, centers, radii)
        # A second implementation from the retrodiction run is a row-level check.
        scored = frame(args.downstream / f"synthesis/split_{cutoff}/post_cutoff_accessibility_records.csv")
        check = held.merge(scored, left_on="icsd_id", right_on="cif_id", validate="one_to_one")
        mismatch = ((check.community != check.assigned_community) |
                    (check.in_basin != check.is_in_basin))
        if mismatch.any():
            # Resolve any cancellation/centroid tie using direct norms, exactly
            # matching the retrodiction implementation and sorted community IDs.
            row_lookup = {int(i): j for j, i in enumerate(ids)}
            for iid in check.loc[mismatch, "icsd_id"]:
                distances = np.linalg.norm(centers - x[row_lookup[int(iid)]], axis=1)
                j = int(np.argmin(distances))
                mask = held.icsd_id == iid
                held.loc[mask, ["community", "distance", "threshold", "in_basin"]] = [cids[j], distances[j], radii[j], int(distances[j] <= radii[j])]
            check = held.merge(scored, left_on="icsd_id", right_on="cif_id", validate="one_to_one")
        assert (check.community == check.assigned_community).all()
        assert (check.in_basin == check.is_in_basin).all()
        assert np.allclose(check.distance, check.nearest_centroid_distance, rtol=1e-10, atol=1e-10)
        held.to_csv(out / f"heldout_{cutoff}.csv", index=False)
        full = {"icsd_unmatched": rate(held.in_basin), "by_source": {}}
        for name, source in sources.items():
            flags = source_flags(source, thresholds)
            full["by_source"][name] = {"unmatched": rate(flags)}
            classification.extend({"cutoff": cutoff, "source": name,
                                   "material_id": str(r.material_id),
                                   "in_basin": bool(flag)}
                                  for r, flag in zip(source.itertuples(), flags))
        results.append({"cutoff": cutoff, "n_train": ntrain,
                        "verification": {"n_retrodiction_overlap": len(check),
                                         "exact_identity_and_flag_agreement": True},
                        "matchings": {"coarse": full}})
        train = (years > 0) & (years <= cutoff)
        for percentile in [90, 95, 99]:
            thr = composition.compute_thresholds(x[train], communities[train], percentile)
            sweeps.append({"cutoff": cutoff, "percentile": percentile,
                           "ICSD_p95": rate(held.in_basin),
                           "sources": {name: rate(source_flags(source, thr)) for name, source in sources.items()}})
        print("Full-denominator cutoff", cutoff, full, flush=True)
    dump(out / "fig3c_corrected_summary.json", {"cutoffs": results,
         "population_rule": "all dated post-cutoff ICSD and all successfully projected source records; absent-at-T external community is frontier"})
    dump(out / "threshold_sweep.json", {"heldout_threshold_percentile": 95,
         "external_percentiles": [90, 95, 99], "all_records_counted": True, "results": sweeps})
    pd.DataFrame(classification).to_csv(out / "external_cutoff_flags.csv", index=False)
    source_args = [v for name, path in paths.items() for v in ("--ai-source", name, path)]
    command("analyze_composition_matched_ai.py", "--features", args.production / "features.npy",
            "--features-pca", args.production / "features_pca.npy",
            "--community-assignments", args.production / "graph/community_assignments.csv",
            "--sample-assignments", args.production / "sample_assignments.csv",
            "--post-cutoff-dir", args.downstream / "synthesis",
            "--post-cutoff-pattern", "split_{cutoff}/post_cutoff_accessibility_records.csv",
            *source_args, "--output-summary", out / "composition_matched_ai_summary.json",
            "--output-records", out / "composition_matched_ai_records.csv")


def build_prior(args, index, sources, paths):
    out = args.downstream / "prior"
    out.mkdir(exist_ok=True)
    # Formula precedent keeps the separately versioned occupancy-aware
    # first-report reference used by Fig. 4.  The synthesis split uses one
    # normalized nominal identity for retrospective per-formula sampling and
    # therefore must not silently replace this reference.
    formula_precedent_dir = args.downstream / "community_evidence/formula_selection"
    source_args = [v for name, path in paths.items() for v in ("--source", name, path)]
    command("analyze_formula_synth_prior.py", "--icsd-formulas-dir", formula_precedent_dir,
            *source_args, "--per-community-thresholds", args.downstream / "reference_basis/community_thresholds.json",
            "--output-summary", out / "formula_synth_prior_summary.json",
            "--output-table", out / "formula_synth_prior_table.md")
    first = frame(formula_precedent_dir / "split_1980/first_report_formulas.csv")
    nominal_first = frame(args.downstream / "synthesis/split_1980/first_report_formulas.csv")
    reference = set(formula_keys(first.reduced_formula))
    all_year = {scale_invariant_formula_key(r["reduced_formula"]) for r in index.values() if r["reduced_formula"] and r["year"] is not None}
    after1980 = {scale_invariant_formula_key(r["reduced_formula"]) for r in index.values() if r["reduced_formula"] and r["year"] is not None and r["year"] > 1980}
    before1980 = {scale_invariant_formula_key(r["reduced_formula"]) for r in index.values() if r["reduced_formula"] and r["year"] is not None and r["year"] <= 1980}
    rows, undercount = [], {}
    for name, source in sources.items():
        source_keys = formula_keys(source.reduced_formula)
        matches = source_keys.isin(reference)
        in_basin = source.in_basin.astype(str).str.lower() == "true"
        for (_, r), match, inside in zip(source.iterrows(), matches, in_basin):
            rows.append({**r.to_dict(), "source": name, "formula_match": bool(match),
                         "in_basin": bool(inside), "quadrant": ("in_basin" if inside else "frontier") + ("_match" if match else "_no_match")})
        undercount[name] = {"scored_post1980": rate(matches),
                           "index_post1980": rate(source_keys.isin(after1980)),
                           "index_all_year": rate(source_keys.isin(all_year))}
    pd.DataFrame(rows).to_csv(out / "quadrant_assignments.csv", index=False)
    dump(out / "pre1980_formula_undercount.json", {"n_index_parseable_dated": sum(bool(r["reduced_formula"]) and r["year"] is not None for r in index.values()),
         "n_all_year_formulas": len(all_year), "n_index_post1980_formulas": len(after1980),
         "n_index_through1980_formulas": len(before1980), "n_only_through1980": len(before1980 - after1980),
         "n_scored_post1980_reference_records": len(first),
         "n_scored_post1980_formulas": len(reference),
         "n_uniform_nominal_first_report_records": len(nominal_first),
         "same_integer_key_set_in_both_scored_reference_views":
             reference == set(formula_keys(nominal_first.reduced_formula)),
         "sources": undercount,
         "formula_key": "collapse species/isotopes to elements; normalize to atomic fractions; pymatgen get_integer_formula_and_factor; gcd reduce; sort elements alphabetically"})


def build_retrospective(args, index, sources):
    out = args.downstream / "retrospective"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    first1980 = frame(args.downstream / "synthesis/split_1980/first_report_formulas.csv")
    full_reference = set(formula_keys(first1980.reduced_formula))
    results = {"seed": 42, "max_year": 2015, "n_boot": 2000, "n_perm": 2000, "cutoffs": {}}
    formula_by_id = {i: r["reduced_formula"] for i, r in index.items()}
    for cutoff in [2010, 2000, 1990]:
        held = frame(args.downstream / f"calibration/heldout_{cutoff}.csv")
        held = held[held.year <= 2015].copy()
        held["formula"] = held.icsd_id.map(formula_by_id)
        known = held.formula.notna() & (held.formula != "")
        all_n = len(held)
        held = held[known].copy()
        held["formula_key"] = formula_keys(held.formula)
        held["stratum_coarse"] = held.formula.map(composition_class)
        held["stratum_anon"] = held.formula.map(anonymized_formula)
        fr = frame(args.downstream / f"synthesis/split_{cutoff}/first_report_formulas.csv")
        first = held[held.icsd_id.isin(fr.cif_id)].copy()
        references = {
            "post1980_le_T": set(formula_keys(first1980.loc[first1980.year <= cutoff, "reduced_formula"])),
            "allyear_le_T_index": {scale_invariant_formula_key(r["reduced_formula"]) for r in index.values() if r["reduced_formula"] and r["year"] is not None and r["year"] <= cutoff},
            "post1980_le_T_index": {scale_invariant_formula_key(r["reduced_formula"]) for r in index.values() if r["reduced_formula"] and r["year"] is not None and 1980 < r["year"] <= cutoff},
        }
        cut = {"coverage": {"n_all_entries": all_n, "n_parseable_formula": len(held),
                            "n_unparseable_formula": all_n - len(held)}, "reference_sizes": {k: len(v) for k, v in references.items()}, "analyses": {}}
        for ref_name, reference in references.items():
            units = {}
            for unit, source in [("per_entry", held), ("per_formula", first)]:
                df = source.copy()
                df["formula_match"] = df.formula_key.isin(reference).astype(int)
                units[unit] = {"quadrant": retrospective.quadrant(df), "by_year": retrospective.by_year(df),
                               "bootstrap": retrospective.bootstrap(df, rng),
                               "permutation_global": retrospective.permutation(df, rng),
                               "permutation_within_coarse_strata": retrospective.permutation(df, rng, "stratum_coarse"),
                               "permutation_within_anonymized_strata": retrospective.permutation(df, rng, "stratum_anon")}
                if ref_name == "post1980_le_T":
                    df.to_csv(out / f"T{cutoff}_{unit}.csv", index=False)
            cut["analyses"][ref_name] = units
        external_flags = frame(args.downstream / "calibration/external_cutoff_flags.csv")
        external_flags = external_flags[external_flags.cutoff == cutoff]
        cut["computed_sources_same_T_conditions"] = {}
        for name, source in sources.items():
            # Saved feature row order is retained in the classification file.
            flags = external_flags[external_flags.source == name]
            assert list(flags.material_id.astype(str)) == list(source.material_id.astype(str))
            df = pd.DataFrame({"in_basin": (flags.in_basin.astype(str).str.lower() == "true").to_numpy(dtype=int),
                               "formula_match": formula_keys(source.reduced_formula).isin(references["post1980_le_T"]).to_numpy(dtype=int)})
            cut["computed_sources_same_T_conditions"][name] = {"quadrant_T_map_T_post1980_reference": retrospective.quadrant(df)}
        results["cutoffs"][str(cutoff)] = cut
        dump(out / "retrospective_quadrant_summary.json", results)
        print("Retrospective cutoff complete", cutoff, cut["analyses"]["post1980_le_T"]["per_entry"]["quadrant"], flush=True)


def build_accessibility(args, sources, paths):
    out = args.downstream / "accessibility"
    out.mkdir(exist_ok=True)
    command("analyze_structural_accessibility_revised.py",
            "--community-assignments", args.production / "graph/community_assignments.csv",
            "--node-events", args.production / "time/node_temporal_events.csv",
            "--gnome-records", paths["GNoME"], "--quadrant-assignments", args.downstream / "prior/quadrant_assignments.csv",
            "--output-dir", out, "--dump-stats")
    # Re-evaluate the reviewed 6x6 grid under the same role convention as ED5.
    meta = accessibility.load_community_metadata(args.production / "graph/community_assignments.csv", args.production / "time/node_temporal_events.csv")
    events = frame(args.production / "time/node_temporal_events.csv")
    terms, roles = [], []
    for r in events.itertuples():
        if int(r.community) < 0 or r.year == "" or r.distance_to_centroid == "":
            continue
        m = meta[int(r.community)]
        terms.append([np.log1p(float(r.distance_to_centroid) / max(m["core_threshold"], 1e-6)), np.log1p(m["size"]), np.log1p(max(float(r.year) - m["birth_year"], 0))])
        roles.append((r.event_type == "community_birth", "bridge" if str(r.is_bridge_attachment).lower() == "true" else r.core_periphery))
    terms = np.asarray(terms)
    masks = {"ICSD birth": np.array([r[0] for r in roles]), **{f"ICSD {role}": np.array([r[1] == role for r in roles]) for role in ["core", "periphery", "bridge"]}}
    gterms = []
    g = sources["GNoME"]
    inside = (g.in_basin.astype(str).str.lower() == "true").to_numpy()
    for r in g.itertuples():
        m = meta[int(r.assigned_community)]
        gterms.append([np.log1p(r.nearest_centroid_distance / max(m["core_threshold"], 1e-6)), np.log1p(m["size"]), np.log1p(max(2019 - m["birth_year"], 0))])
    gterms = np.asarray(gterms)
    results = []
    for alpha, beta in [(a, b) for a in np.linspace(0, 1, 6) for b in np.linspace(0, 1, 6)] + [(.5, .5)]:
        raw = terms @ np.array([1, -alpha, -beta])
        z = (raw - raw.mean()) / raw.std(ddof=1)
        gz = (gterms @ np.array([1, -alpha, -beta]) - raw.mean()) / raw.std(ddof=1)
        means = {k: float(z[m].mean()) for k, m in masks.items()}
        means.update({"GNoME in-basin": float(gz[inside].mean()), "GNoME frontier": float(gz[~inside].mean())})
        results.append({"alpha": float(alpha), "beta": float(beta), "is_production_baseline": bool(alpha == .5 and beta == .5), "means": means,
                        "methods_order": means["ICSD core"] < means["ICSD periphery"] < means["ICSD bridge"] < means["ICSD birth"] and means["GNoME in-basin"] < means["GNoME frontier"],
                        "order": sorted(means, key=means.get)})
    dump(out / "alpha_beta_grid.json", {"role_convention": "ED5: birth may overlap; bridge precedes core/periphery", "cells": results})
    # A-Lab uses the same full-map score moments, with age evaluated in 2023.
    moments = json.loads((out / "structural_accessibility_summary.json").read_text())
    alab = frame(args.downstream / "external/alab/alab_projection_records.csv")
    scores = []
    for r in alab.itertuples():
        m = meta[int(r.assigned_community)]
        raw = accessibility.raw_accessibility(r.nearest_centroid_distance, m["core_threshold"], m["size"], 2023-m["birth_year"])
        scores.append(accessibility.zscore(raw, moments["icsd_raw_mu"], moments["icsd_raw_sigma"]))
    alab["accessibility_score"] = scores
    alab_out = args.downstream / "alab"
    alab_out.mkdir(exist_ok=True)
    alab.to_csv(alab_out / "alab_validation_records.csv", index=False)
    outcomes = {}
    for name, group in alab.groupby("corrected_outcome"):
        outcomes[name] = {"n": len(group), "mean_accessibility": float(group.accessibility_score.mean()),
                          "median_accessibility": float(group.accessibility_score.median()),
                          "frontier_rate": float((group.in_basin.astype(str).str.lower() != "true").mean())}
    dump(alab_out / "alab_validation_summary.json", {"n_scored": len(alab), "observation_year": 2023,
         "score_moments": {k: moments[k] for k in ["icsd_raw_mu", "icsd_raw_sigma"]}, "outcome_means": outcomes,
         "limitation": "The 15 not-obtained targets have no released CIF and cannot be scored; this is not a success/failure validation."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production", type=Path, required=True)
    p.add_argument("--downstream", type=Path, required=True)
    p.add_argument("--icsd-index", type=Path, required=True)
    p.add_argument("--stages", nargs="+", choices=["calibration", "prior", "retrospective", "accessibility"], default=["calibration", "prior", "retrospective", "accessibility"])
    args = p.parse_args()
    x, ids, years, communities, index, sources, paths = load_inputs(args)
    if "calibration" in args.stages:
        build_calibration(args, x, ids, years, communities, sources, paths)
    if "prior" in args.stages:
        build_prior(args, index, sources, paths)
    if "retrospective" in args.stages:
        build_retrospective(args, index, sources)
    if "accessibility" in args.stages:
        build_accessibility(args, sources, paths)


if __name__ == "__main__":
    main()
