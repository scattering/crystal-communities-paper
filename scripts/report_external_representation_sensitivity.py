#!/usr/bin/env python3
"""Write an SI-ready representation sensitivity subsection from verified results.

All five cohorts, both representations and both independent verification reports
must be complete. Magpie source rates use the consistent-transform sensitivity;
original saved-score comparisons remain in the numerical provenance report.
This script writes a standalone replacement and never edits a manuscript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from external_representation_sensitivity import SOURCES, csv_rows, dump, sha

MAPS = ("full", "T1990", "T2000", "T2010")
NAMES = {"gnome": "GNoME", "mattergen": "MatterGen", "mp": "MP", "jarvis": "JARVIS", "alexandria": "Alexandria"}
MAP_NAMES = {"full": "Full record", "T1990": "$T=1990$", "T2000": "$T=2000$", "T2010": "$T=2010$"}


def pct(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{value:+.2f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def load_json(path, inputs):
    value = json.loads(Path(path).read_text())
    inputs[str(path)] = sha(path)
    return value


def complete_run(directory, kind, inputs):
    manifest_path = directory / "basis/basis_manifest.json"
    manifest = load_json(manifest_path, inputs)
    if manifest["kind"] != kind or set(manifest["reference_rates"]) != set(MAPS):
        raise ValueError(f"Incomplete or wrong reference representation: {directory}")
    sources = {}
    for source in SOURCES:
        summary = load_json(directory / "external" / source / "summary.json", inputs)
        expected_attempts = 386 if source == "mattergen" else 5000
        if (summary["kind"] != kind or summary["source"] != source or set(summary["rates"]) != set(MAPS)
                or summary["n_attempted"] != expected_attempts
                or summary["n_successful"] + summary["n_failures"] != expected_attempts
                or summary["basis_manifest_sha256"] != inputs[str(manifest_path)]):
            raise ValueError(f"Incomplete or inconsistent source: {directory}/{source}")
        for key in MAPS:
            report = summary["rates"][key]
            if report["n"] != summary["n_successful"] or report["icsd_reference"]["n"] != manifest["reference_rates"][key]["n"]:
                raise ValueError("Reference or successful external denominator changed")
        sources[source] = summary
    return {"manifest": manifest, "sources": sources, "manifest_sha256": inputs[str(manifest_path)]}


def verified_report(path, expected_manifests, inputs):
    report = load_json(path, inputs)
    if report.get("n_failed") != 0 or report.get("n_passed") != report.get("n_checks") or not report.get("n_checks"):
        raise ValueError(f"Independent verification is incomplete or failed: {path}")
    if not all(check.get("passed") for check in report["checks"]):
        raise ValueError(f"An independent check failed: {path}")
    if set(report["input_manifest_sha256"].values()) != set(expected_manifests):
        raise ValueError(f"Verification refers to different results: {path}")
    return report


def shared_data(report):
    rows = report["common_support"]
    keyed = {(row["representation"], row["map"], row["source"]): row for row in rows}
    if len(keyed) != 40 or len(rows) != 40:
        raise ValueError("Expected all 40 representation/map/source common-support comparisons")
    for model in ("magpie", "graphlet"):
        for key in MAPS:
            for source in SOURCES:
                row = keyed[(model, key, source)]
                if row["common_icsd_n"] < 1 or row["common_external_n"] < 1:
                    raise ValueError("Common represented support is empty")
    return keyed


def gap_summary(rows):
    gaps = [row["icsd_minus_external_percentage_points"] for row in rows]
    return f"{pp(min(gaps))} to {pp(max(gaps))}", sum(gap > 0 for gap in gaps)


def production_control(path, magpie_dir, graphlet_dir, matched, inputs):
    control = load_json(path, inputs)
    if (not all(control["checks"].values()) or not control["GNoME"]["recorded_flags_and_radii_reverified"]
            or not control["GNoME"]["same_production_basis_hash_verified"]):
        raise ValueError("Production protocol control is not verified")
    id_sets = []
    for directory in (magpie_dir, graphlet_dir):
        projection = directory / "basis/icsd_full.csv"
        ids = [int(row["record_key"]) for row in csv_rows(projection)]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate common-control reference IDs")
        id_sets.append(set(ids))
        inputs[str(projection)] = sha(projection)
    common = id_sets[0] & id_sets[1]
    digest = hashlib.sha256("".join(f"{iid}\n" for iid in sorted(common)).encode()).hexdigest()
    reference = control["experimental_reference"]
    if (reference["common_ids_sha256"] != digest or reference["common_three_representations"]["n"] != len(common)
            or any(matched[(model, "full", "gnome")]["common_icsd_n"] != len(common) for model in ("magpie", "graphlet"))
            or any(matched[(model, "full", "gnome")]["common_external_n"] != control["GNoME"]["n"] for model in ("magpie", "graphlet"))):
        raise ValueError("Production control and alternative maps have different common evaluation IDs")
    return control


def build_section(magpie, graphlet, original, matched, coordinate, production=None):
    lines = ["## S1.12. External-cohort sensitivity to the crystal representation", "",
        "We featurized the same five public cohorts under the Magpie-22 ablation (§S1.5) and CrystalNN graphlets (§S9). "
        "The Magpie ablation changes the site chemistry, concatenating neighbor aggregation and structure pooling together; "
        "the graphlet comparison uses the frozen 64-channel, 20-bin CDF representation and Euclidean distance of its independent temporal map. "
        "Each external structure is assigned to the nearest community centroid and classified in basin when its distance is at most that community's 95th-percentile member distance.", "",
        "The frozen draws contain 5,000 attempted structures for each sampled source and all 386 public MatterGen structures. "
        "The table gives successful feature counts and the intersection of successful identifiers across the two representations. "
        "Failed attempts remain recorded individually and are excluded from the reported rate denominators.", ""]
    coverage = []
    for source in SOURCES:
        m, g = magpie["sources"][source], graphlet["sources"][source]
        common = matched[("magpie", "full", source)]["common_external_n"]
        coverage.append([NAMES[source], f'{m["n_attempted"]:,}', f'{m["n_successful"]:,}', f'{g["n_successful"]:,}', f'{common:,}'])
    lines += [table(["Source", "Attempted", "Magpie successes", "Graphlet successes", "Shared successes"], coverage), "",
              "Table: Frozen external cohorts and representation-specific successful denominators. {#tbl:external-representation-coverage}", ""]
    mref, gref = magpie["manifest"], graphlet["manifest"]
    lines += [f'The Magpie map contains {mref["n_map_rows"]:,} accepted ICSD entries, including {mref["n_map_noise"]:,} map outliers. '
              f'The graphlet map contains {gref["n_map_rows"]:,} dated represented entries from the production-assigned population, '
              f'including {gref["n_map_noise"]:,} outliers under its own partition. '
              "These outliers remain in the experimental evaluation denominators; only assigned members define centroids and radii.", "",
              "The full-record rows compare external cohorts with the ICSD entries used to construct their basins. "
              "For each cutoff row, centroids and radii use only dated members published by $T$, and later dated ICSD entries form the experimental comparison. "
              "The representation and partition retain full-record information, so these are full-map sensitivities; the independently cutoff-trained analysis is reported in §S7. "
              "The frozen Magpie scaler/PCA transform is applied identically to ICSD and external features before estimating these centroids and radii.", ""]
    for kind, name, run in [("magpie", "Magpie", magpie), ("graphlet", "CrystalNN graphlet", graphlet)]:
        rates = []
        for key in MAPS:
            ref = run["manifest"]["reference_rates"][key]
            rates.append([MAP_NAMES[key], f'{pct(ref["in_basin_fraction"])} ({ref["n"]:,})']
                         + [pct(run["sources"][source]["rates"][key]["in_basin_fraction"]) for source in SOURCES])
        lines += [table(["Reference", "ICSD rate (n)"] + [NAMES[s] for s in SOURCES], rates), "",
                  f"Table: {name} in-basin rates. External denominators are the {name} successes in the preceding coverage table; ICSD denominators are shown with each rate. "
                  f"{{#tbl:external-representation-{kind}}}", ""]
    # Summarize matched-support effects without replacing the source-specific
    # rates above with an average over heterogeneous public releases.
    support_rows = []
    for key in MAPS:
        common_n = matched[("magpie", key, SOURCES[0])]["common_icsd_n"]
        pieces = []
        for model in ("magpie", "graphlet"):
            selected = [matched[(model, key, source)] for source in SOURCES]
            if any(row["common_icsd_n"] != common_n for row in selected):
                raise ValueError("Matched ICSD denominator differs across a comparison")
            gaps, positive = gap_summary(selected)
            pieces += [gaps, f"{positive}/5"]
        support_rows.append([MAP_NAMES[key], f"{common_n:,}"] + pieces)
    lines += ["Restricting each comparison to the same successful ICSD identifiers and the same successful external identifiers controls the different feature-coverage denominators. "
              "The next table reports the range of ICSD-minus-source gaps over the five cohorts and the number of positive gaps. "
              "A negative gap means the computed cohort has a higher in-basin fraction.", "",
              table(["Reference", "Shared ICSD n", "Magpie gap range (pp)", "ICSD higher", "Graphlet gap range (pp)", "ICSD higher"], support_rows), "",
              "Table: Source comparisons on common represented identifiers. External intersection counts are listed in the coverage table. {#tbl:external-representation-common-support}", ""]
    if production is not None:
        pref, pgnome = production["experimental_reference"]["common_three_representations"], production["GNoME"]
        control_rows = [["Production", pct(pref["in_basin_fraction"]), pct(pgnome["in_basin_fraction"]),
                         pp(100 * (pref["in_basin_fraction"] - pgnome["in_basin_fraction"]))]]
        for model, name in [("magpie", "Magpie"), ("graphlet", "CrystalNN graphlet")]:
            row = matched[(model, "full", "gnome")]
            control_rows.append([name, pct(row["icsd_in_basin_fraction"]), pct(row["external_in_basin_fraction"]),
                                 pp(row["icsd_minus_external_percentage_points"])])
        lines += [f'The full-map protocol control applies the same nearest-centroid/per-community-p95 rule to all three representations on exactly {pref["n"]:,} shared ICSD entries and {pgnome["n"]:,} GNoME entries. '
                  "Each fitted map and its basin definitions are retained; only the experimental evaluation identifiers are intersected.", "",
                  table(["Representation", "Shared ICSD", "GNoME", "ICSD minus GNoME (pp)"], control_rows), "",
                  "Table: Full-map comparison under a common classification rule and identical evaluation identifiers. {#tbl:external-representation-production-control}", ""]
    reversals = []
    for model, name in [("magpie", "Magpie"), ("graphlet", "CrystalNN graphlet")]:
        negative_maps = [key for key in MAPS if matched[(model, key, "gnome")]["icsd_minus_external_percentage_points"] < 0]
        if negative_maps:
            location = "all four comparisons" if len(negative_maps) == 4 else ", ".join(MAP_NAMES[key] for key in negative_maps)
            reversals.append(f"Under {name}, GNoME has a higher in-basin fraction than ICSD in {location} on the common represented identifiers.")
    if reversals:
        if all(matched[(model, key, "gnome")]["icsd_minus_external_percentage_points"] < 0
               for model in ("magpie", "graphlet") for key in MAPS):
            observation = "GNoME has a higher in-basin fraction than ICSD under both alternative representations in all four comparisons on the common represented identifiers."
        else:
            observation = " ".join(reversals)
        if all(matched[("graphlet", key, "alexandria")]["icsd_minus_external_percentage_points"] < 0 for key in MAPS if key != "full"):
            observation += " Alexandria also has a higher in-basin fraction than ICSD under CrystalNN graphlets at all three historical cutoffs on the common represented identifiers."
        lines += [observation + " The experimental-versus-computed source ordering depends on the representation. "
                  "The temporal decline in community-birth share remains supported by the separate cross-representation tests (§S9).", ""]
    else:
        lines += ["The tables report source-specific point estimates and their dependence on represented population and partition. "
                  "The temporal decline in community-birth share is assessed separately in §S9.", ""]
    diagnostics = coordinate["coordinate_displacement"]
    relative = diagnostics["displacement_divided_by_original_own_community_p95_positive_radius"]
    changes = coordinate["paired_membership_changes"]
    icsd_changes = [row["rate_change_percentage_points"] for row in changes if row["population"] == "ICSD"]
    external_changes = [row["rate_change_percentage_points"] for row in changes if row["population"] != "ICSD"]
    lines += ["As a numerical coordinate check, we compared the common-transform Magpie results above with centroids and radii calculated from the randomized-PCA fit scores. "
              f'The maximum coordinate-component difference is {diagnostics["maximum_component_absolute_displacement"]:.6g}; '
              f'the median and 95th-percentile Euclidean displacements are {100 * relative["p50"]:.3f}% and {100 * relative["p95"]:.3f}% of the original member-community p95 radius among members with positive radii. '
              f'Applying the common transform changes the four ICSD rates by {pp(min(icsd_changes))} to {pp(max(icsd_changes))} percentage points '
              f'and the 20 external rates by {pp(min(external_changes))} to {pp(max(external_changes))} percentage points. '
              "The partition, training-member identities and external feature-success sets are identical in this check. "
              "Per-community displacements and paired membership switches are retained in the numerical audit.", ""]
    if diagnostics["n_nonnoise_zero_radius"]:
        lines += [f'{diagnostics["n_nonnoise_zero_radius"]:,} original nonnoise members have zero community radius; '
                  f'{diagnostics["n_zero_radius_members_with_nonzero_displacement"]:,} of these have nonzero coordinate displacement and are excluded from the radius-normalized summaries.', ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magpie", required=True, type=Path, help="Original saved-score Magpie run")
    parser.add_argument("--consistent-magpie", required=True, type=Path)
    parser.add_argument("--graphlet", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path, help="Original Magpie/graphlet verification.json")
    parser.add_argument("--consistent-verification", required=True, type=Path)
    parser.add_argument("--production-control", type=Path, help="Verified full-map production control on the exact shared ICSD/GNoME identifiers")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = {}
    original = complete_run(args.magpie, "magpie", inputs)
    magpie = complete_run(args.consistent_magpie, "magpie", inputs)
    graphlet = complete_run(args.graphlet, "graphlet", inputs)
    verified_report(args.verification, [original["manifest_sha256"], graphlet["manifest_sha256"]], inputs)
    verified = verified_report(args.consistent_verification, [magpie["manifest_sha256"], graphlet["manifest_sha256"]], inputs)
    if magpie["manifest"].get("coordinate_variant") != "consistent_transform":
        raise ValueError("Reporting requires consistent-transform Magpie coordinates")
    matched = shared_data(verified)
    coordinate = load_json(args.consistent_magpie / "coordinate_consistency.json", inputs)
    if not coordinate.get("same_external_feature_successes_and_coordinates") or not coordinate.get("same_partition_and_training_member_counts"):
        raise ValueError("Coordinate-consistency identity verification is incomplete")
    if coordinate["original_basis_manifest_sha256"] != original["manifest_sha256"]:
        raise ValueError("Coordinate diagnostic refers to a different original map")
    for source in SOURCES:
        if (original["sources"][source]["feature_ids_sha256"] != magpie["sources"][source]["feature_ids_sha256"]
                or original["sources"][source]["n_successful"] != magpie["sources"][source]["n_successful"]):
            raise ValueError("Coordinate comparison changed a source's successful identities")
    if {(row["population"], row["map"]) for row in coordinate["paired_membership_changes"]} != {
            (population, key) for population in ("ICSD", *SOURCES) for key in MAPS}:
        raise ValueError("Coordinate comparison is incomplete")
    production = production_control(args.production_control, args.consistent_magpie, args.graphlet, matched, inputs) if args.production_control else None
    text = build_section(magpie, graphlet, original, matched, coordinate, production)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "replacement_s1p12.md"
    output.write_text(text)
    dump(args.out_dir / "report_provenance.json", {"inputs_sha256": inputs, "script_sha256": sha(__file__),
         "output": str(output), "output_sha256": sha(output), "word_count": len(text.split()),
         "reported_magpie_coordinate_variant": "consistent_transform", "all_five_sources_complete_and_verified": True})
    print(json.dumps({"output": str(output), "words": len(text.split()), "input_files": len(inputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
