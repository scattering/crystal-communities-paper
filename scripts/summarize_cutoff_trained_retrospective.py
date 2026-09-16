#!/usr/bin/env python3
"""Create compact reporting tables from cutoff-trained retrospective output."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate_legacy(block: dict) -> dict:
    return {
        "k": block["k"],
        "n": block["n"],
        "rate": block["rate"],
        "ci95": block["wilson95"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    source = json.loads(args.summary.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rates: list[dict] = []
    shared: list[dict] = []
    joint: list[dict] = []
    compatibility = {
        "schema_version": 1,
        "purpose": "composition-support rendering from independently cutoff-trained maps",
        "source_summary": {
            "path": str(args.summary),
            "sha256": digest(args.summary),
        },
        "cutoffs": [],
    }

    for cutoff_text, result in sorted(
        source["cutoffs"].items(), key=lambda item: int(item[0])
    ):
        cutoff = int(cutoff_text)
        held = result["heldout_rate"]
        rates.append(
            {
                "cutoff": cutoff,
                "source": "ICSD",
                "k": held["k"],
                "n": held["n"],
                "rate": held["rate"],
                "ci95_low": held["wilson95"][0],
                "ci95_high": held["wilson95"][1],
                "heldout_minus_external": "",
            }
        )
        matchings = {}
        for matcher, field in (
            ("coarse", "shared_coarse_strata"),
            ("anonymized", "shared_anonymized_strata"),
        ):
            by_source = {}
            for name, comparison in result["external_source_rates"].items():
                common = comparison[field]
                by_source[name] = {
                    "unmatched": rate_legacy(comparison["all_records"]),
                    "matched": rate_legacy(common["external"]),
                    "icsd_matched_to_source": rate_legacy(common["heldout"]),
                    "n_strata_common": common["n_shared_strata"],
                }
                shared.append(
                    {
                        "cutoff": cutoff,
                        "matching": matcher,
                        "source": name,
                        "n_shared_strata": common["n_shared_strata"],
                        "icsd_k": common["heldout"]["k"],
                        "icsd_n": common["heldout"]["n"],
                        "icsd_rate": common["heldout"]["rate"],
                        "external_k": common["external"]["k"],
                        "external_n": common["external"]["n"],
                        "external_rate": common["external"]["rate"],
                        "icsd_minus_external": common["heldout_minus_external"],
                    }
                )
            matchings[matcher] = {
                "icsd_unmatched": rate_legacy(held),
                "by_source": by_source,
            }
        compatibility["cutoffs"].append(
            {"cutoff": cutoff, "matchings": matchings}
        )

        for name, comparison in result["external_source_rates"].items():
            rate = comparison["all_records"]
            rates.append(
                {
                    "cutoff": cutoff,
                    "source": name,
                    "k": rate["k"],
                    "n": rate["n"],
                    "rate": rate["rate"],
                    "ci95_low": rate["wilson95"][0],
                    "ci95_high": rate["wilson95"][1],
                    "heldout_minus_external": comparison["heldout_minus_external"],
                }
            )

        for reference, reference_result in result["analyses"].items():
            for unit, analysis in reference_result.items():
                q = analysis["quadrant"]
                boot = analysis["bootstrap"]
                row = {
                    "cutoff": cutoff,
                    "reference": reference,
                    "unit": unit,
                    "n": q["n"],
                    "in_basin_and_match": q["in_basin_and_match"],
                    "share_in_basin_and_match": q["share_in_basin_and_match"],
                    "p_in_basin": q["p_in_basin"],
                    "p_formula_match": q["p_match"],
                    "enrichment_ratio": q[
                        "enrichment_ratio_obs_over_independence"
                    ],
                    "enrichment_ci95_low": boot["enrichment_ratio_ci95"][0],
                    "enrichment_ci95_high": boot["enrichment_ratio_ci95"][1],
                    "relative_risk": q["relative_risk_in_basin_match_vs_no_match"],
                    "permutation_global_p_one_sided": analysis[
                        "permutation_global"
                    ]["p_one_sided"],
                    "permutation_within_year_p_one_sided": analysis.get(
                        "permutation_within_year", {}
                    ).get("p_one_sided", ""),
                    "permutation_within_coarse_p_one_sided": analysis[
                        "permutation_within_coarse_strata"
                    ]["p_one_sided"],
                    "permutation_within_anonymized_p_one_sided": analysis[
                        "permutation_within_anonymized_strata"
                    ]["p_one_sided"],
                }
                joint.append(row)

    write_csv(args.output_dir / "cutoff_trained_rates.csv", rates)
    write_csv(args.output_dir / "cutoff_trained_shared_strata.csv", shared)
    write_csv(args.output_dir / "cutoff_trained_joint.csv", joint)
    compatibility_path = args.output_dir / "cutoff_trained_composition_summary.json"
    compatibility_path.write_text(json.dumps(compatibility, indent=2) + "\n")

    manifest = {
        "schema_version": 1,
        "producer": str(Path(__file__).resolve()),
        "producer_sha256": digest(Path(__file__)),
        "source_summary_sha256": digest(args.summary),
        "outputs": {
            path.name: digest(path)
            for path in (
                args.output_dir / "cutoff_trained_rates.csv",
                args.output_dir / "cutoff_trained_shared_strata.csv",
                args.output_dir / "cutoff_trained_joint.csv",
                compatibility_path,
            )
        },
    }
    (args.output_dir / "reporting_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"wrote {len(rates)} rates, {len(shared)} shared-strata rows and "
        f"{len(joint)} joint rows to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
