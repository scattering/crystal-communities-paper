#!/usr/bin/env python3
"""Independently verify the packaged cutoff-trained retrospective analysis.

The verifier starts from the packaged row tables and immutable local source
files. It checks recorded hashes, temporal partitions, cutoff-map metadata,
row identities, saved classifications, source rates, shared-stratum rates, and
the deterministic quadrant arithmetic for all 24 analyses. It also reruns the
reporting renderer in a temporary directory and requires byte-identical output.

The six large raw feature matrices are intentionally not copied into the local
package. Their recorded TACC paths and hashes are retained as an explicit
limitation; consequently this verifier does not rerun feature projection or
graph construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_package = root / "notes/feature_repair_2026_09/downstream/cutoff_trained"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=default_package,
        help="Cutoff-trained package to verify.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Output report. Defaults to PACKAGE/verification/"
            "independent_verification.json."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
ROOT = Path(__file__).resolve().parents[1]
BASE = ARGS.package_dir.resolve()
REPORT_PATH = (
    ARGS.report.resolve()
    if ARGS.report
    else BASE / "verification/independent_verification.json"
)
SUMMARY_PATH = BASE / "cutoff_trained_retrospective_summary.json"
SUMMARY = json.loads(SUMMARY_PATH.read_text())
PROTOCOL = SUMMARY["protocol"]
CUTOFFS = sorted(int(value) for value in SUMMARY["cutoffs"])
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "notes/review_2026_08"))
from analyze_synthesis_retrodiction import load_icsd_index  # noqa: E402
from formula_conventions import (  # noqa: E402
    anonymous_stoichiometry_key,
    build_formula_precedent_index,
    formula_has_precedent,
    normalized_fraction_key,
    scale_invariant_formula_key,
)
from retrospective_quadrant import composition_class  # noqa: E402
N = 0
FAIL: list[dict[str, str]] = []
HASH_AUDIT: dict[str, object] = {
    "local_recorded_inputs": {},
    "analysis_outputs": {},
    "reporting_outputs": {},
}

# Formula normalization emits harmless warnings for noble gases and Rf. They
# do not affect the exact reference-set comparisons made below.
warnings.filterwarnings("ignore", message="No Pauling electronegativity")


def ck(cond, label, detail=""):
    global N
    N += 1
    if not bool(cond):
        FAIL.append({"label": label, "detail": detail})


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        try:
            return math.isclose(float(a), float(b), rel_tol=2e-14, abs_tol=2e-14)
        except (TypeError, ValueError):
            return False
    if isinstance(a, (int, np.integer, bool, np.bool_)) and isinstance(
        b, (int, np.integer, bool, np.bool_)
    ):
        return int(a) == int(b)
    return a == b


def eq(a, b, label):
    ck(same(a, b), label, f"actual={a!r}; expected={b!r}")


def comp(a, b, label):
    if isinstance(b, dict):
        ck(isinstance(a, dict), label + ".type")
        if not isinstance(a, dict):
            return
        ck(
            set(a) == set(b),
            label + ".keys",
            f"actual-only={set(a)-set(b)}, expected-only={set(b)-set(a)}",
        )
        for k in b:
            if k in a:
                comp(a[k], b[k], f"{label}.{k}")
    elif isinstance(b, (list, tuple)):
        ck(isinstance(a, (list, tuple)), label + ".type")
        if not isinstance(a, (list, tuple)):
            return
        eq(len(a), len(b), label + ".len")
        for i, (x, y) in enumerate(zip(a, b)):
            comp(x, y, f"{label}[{i}]")
    else:
        eq(a, b, label)


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [cen - half, cen + half]


def quadrant(frame):
    ib = frame["in_basin"].to_numpy(dtype=np.int64)
    fm = frame["formula_match"].to_numpy(dtype=np.int64)
    n = len(frame)
    ul = int(((ib == 1) & (fm == 1)).sum())
    ur = int(((ib == 1) & (fm == 0)).sum())
    ll = int(((ib == 0) & (fm == 1)).sum())
    lr = int(((ib == 0) & (fm == 0)).sum())
    out = {
        "n": n,
        "in_basin_and_match": ul,
        "in_basin_and_no_match": ur,
        "frontier_and_match": ll,
        "frontier_and_no_match": lr,
    }
    if n:
        for k, v in list(out.items())[1:]:
            out["share_" + k] = v / n
        pib = (ul + ur) / n
        pm = (ul + ll) / n
        out["p_in_basin"] = pib
        out["p_match"] = pm
        out["expected_share_in_basin_and_match_independence"] = pib * pm
        out["enrichment_ratio_obs_over_independence"] = (
            (ul / n) / (pib * pm) if pib * pm > 0 else None
        )
        out["p_in_basin_given_match"] = ul / (ul + ll) if ul + ll else None
        out["p_in_basin_given_no_match"] = ur / (ur + lr) if ur + lr else None
        out["wilson_in_basin_given_match"] = wilson(ul, ul + ll)
        out["wilson_in_basin_given_no_match"] = wilson(ur, ur + lr)
        if out["p_in_basin_given_match"] and out["p_in_basin_given_no_match"]:
            out["relative_risk_in_basin_match_vs_no_match"] = (
                out["p_in_basin_given_match"] / out["p_in_basin_given_no_match"]
            )
        out["odds_ratio"] = (ul * lr) / (ur * ll) if ur * ll else None
        out["wilson_share_in_basin_and_match"] = wilson(ul, n)
    return out


def byyear(frame):
    out = {}
    fields = (
        "n",
        "in_basin_and_match",
        "in_basin_and_no_match",
        "frontier_and_match",
        "frontier_and_no_match",
        "share_in_basin_and_match",
        "p_in_basin",
        "p_match",
        "enrichment_ratio_obs_over_independence",
        "p_in_basin_given_match",
        "p_in_basin_given_no_match",
    )
    for y, g in frame.groupby("year"):
        q = quadrant(g)
        out[str(int(y))] = {k: q[k] for k in fields}
    return out


def rate(vals):
    x = np.asarray(vals, dtype=bool)
    n = len(x)
    k = int(x.sum())
    if not n:
        return {"k": 0, "n": 0, "rate": None, "wilson95": [None, None]}
    return {"k": k, "n": n, "rate": k / n, "wilson95": wilson(k, n)}


def shared(held, ext, field):
    h = held[(held.formula != "") & held[field].notna()].copy()
    e = ext[(ext.formula != "") & ext[field].notna()].copy()
    s = set(h[field].astype(str)) & set(e[field].astype(str))
    h = h[h[field].astype(str).isin(s)]
    e = e[e[field].astype(str).isin(s)]
    hr = rate(h.in_basin)
    er = rate(e.in_basin)
    return {
        "n_shared_strata": len(s),
        "heldout": hr,
        "external": er,
        "heldout_minus_external": hr["rate"] - er["rate"],
        "definition": "restrict each population to strata represented in both; retain original stratum frequencies",
    }


def normalized_formula_identity_token(formula: str) -> str:
    """Reconstruct the producer's scale-invariant nominal formula token."""
    if not str(formula).strip():
        return ""
    elements, fractions = normalized_fraction_key(formula)
    return "|".join(elements) + "::" + ",".join(
        f"{value:.12g}" for value in fractions
    )


def formula_identity_token(formula: str, is_partial: bool) -> str:
    """Reconstruct the occupancy-independent nominal identity."""
    del is_partial
    return normalized_formula_identity_token(formula)


def anonymous_stoichiometry_token(formula: str) -> str:
    """Reconstruct the scale-invariant anonymous-composition stratum."""
    if not str(formula).strip():
        return ""
    return "|".join(map(str, anonymous_stoichiometry_key(formula)))


def expected_quadrant_for_uniform_tie_choice(frame: pd.DataFrame) -> dict[str, object]:
    """Expected quadrant terms after uniform choice among earliest-year ties."""
    in_basin_probability = frame.in_basin.to_numpy(dtype=float)
    formula_match = frame.formula_match.to_numpy(dtype=bool)
    p_in_basin = float(in_basin_probability.mean())
    p_match = float(formula_match.mean())
    joint = float((in_basin_probability * formula_match).mean())
    expected = p_in_basin * p_match
    return {
        "n": len(frame),
        "expected_in_basin_and_match_count": float(
            np.sum(in_basin_probability * formula_match)
        ),
        "share_in_basin_and_match": joint,
        "p_in_basin": p_in_basin,
        "p_match": p_match,
        "expected_share_in_basin_and_match_independence": expected,
        "enrichment_ratio_obs_over_independence": (
            joint / expected if expected > 0 else None
        ),
    }


def earliest_year_tie_sensitivity(
    earliest_year_rows: pd.DataFrame, match_column: str
) -> dict[str, object]:
    """Independently reconstruct the saved same-year tie sensitivity."""
    grouped_match_counts = earliest_year_rows.groupby("formula_identity")[
        match_column
    ].nunique()
    ck(
        bool((grouped_match_counts == 1).all()),
        f"tie.{match_column}.formula_match_constant_within_identity",
    )
    outside = (
        earliest_year_rows.sort_values(
            ["formula_identity", "in_basin", "icsd_id"],
            ascending=[True, True, True],
            kind="stable",
        )
        .drop_duplicates("formula_identity", keep="first")
        .copy()
    )
    inside = (
        earliest_year_rows.sort_values(
            ["formula_identity", "in_basin", "icsd_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("formula_identity", keep="first")
        .copy()
    )
    for frame in (outside, inside):
        frame["formula_match"] = frame[match_column]
    uniform = (
        earliest_year_rows.groupby("formula_identity", sort=False)
        .agg(in_basin=("in_basin", "mean"), formula_match=(match_column, "first"))
        .reset_index()
    )
    return {
        "outside_when_available": quadrant(outside),
        "inside_when_available": quadrant(inside),
        "uniform_random_choice_expectation": expected_quadrant_for_uniform_tie_choice(
            uniform
        ),
    }


# Recorded input and output hashes.
inputs_local = {
    "feature_metadata": ROOT
    / "notes/feature_repair_2026_09/full_run_results/production/summary.json",
    "sample_assignments": ROOT
    / "notes/feature_repair_2026_09/full_run_results/production/sample_assignments.csv",
    "icsd_index": ROOT
    / "notes/feature_repair_2026_09/downstream/inputs/ICSD_index.csv",
    "occupancy_flags": ROOT
    / "notes/feature_repair_2026_09/downstream/formula_layers/icsd_occupancy_flags.csv",
    "producer": ROOT / "scripts/analyze_cutoff_trained_retrospective.py",
    "formula_conventions": ROOT / "scripts/formula_conventions.py",
    "index_loader": ROOT / "scripts/analyze_synthesis_retrodiction.py",
    "statistical_module": ROOT / "notes/review_2026_08/retrospective_quadrant.py",
    "wrapper": ROOT
    / "notes/feature_repair_2026_09/downstream/run_cutoff_trained_retrospective.sh",
    "production_pca": ROOT
    / "notes/feature_repair_2026_09/downstream/inputs/features_pca.npy",
    "production_labels": ROOT
    / "notes/feature_repair_2026_09/full_run_results/production/graph/community_assignments.csv",
}
source_slug = {
    "GNoME": "gnome",
    "MatterGen": "mattergen",
    "MP": "mp",
    "JARVIS": "jarvis",
    "Alexandria": "alexandria",
}
source_file = {
    "GNoME": "gnome_frontier_records.csv",
    "MatterGen": "mattergen-public_frontier_records.csv",
    "MP": "mp_frontier_records.csv",
    "JARVIS": "jarvis_frontier_records.csv",
    "Alexandria": "alexandria_frontier_records.csv",
}
for s in source_slug:
    leaf = ROOT / "notes/feature_repair_2026_09/downstream/external" / source_slug[s]
    inputs_local[f"external_{s}_records"] = leaf / source_file[s]
    inputs_local[f"external_{s}_feature_metadata"] = leaf / "feature_metadata.json"
missing_recorded_local_inputs = sorted(set(inputs_local) - set(SUMMARY.get("inputs", {})))
if missing_recorded_local_inputs:
    raise SystemExit(
        "FINAL-OUTPUT INTEGRATION REQUIRED: the packaged summary predates the "
        "occupancy-aware verifier schema and does not record local inputs: "
        + ", ".join(missing_recorded_local_inputs)
    )
for k, p in inputs_local.items():
    ck(p.exists(), f"input.{k}.exists")
    if p.exists():
        actual_hash = sha(p)
        expected_hash = SUMMARY["inputs"][k]["sha256"]
        HASH_AUDIT["local_recorded_inputs"][k] = {
            "local_path": str(p.relative_to(ROOT)),
            "recorded_tacc_path": SUMMARY["inputs"][k]["path"],
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "matches": actual_hash == expected_hash,
        }
        eq(actual_hash, expected_hash, f"input.{k}.sha256")
missing_inputs = sorted(set(SUMMARY["inputs"]) - set(inputs_local))
eq(
    missing_inputs,
    sorted(["features"] + [f"external_{s}_features" for s in source_slug]),
    "input.missing_expected_feature_blobs",
)

# Protocol and provenance schema checks deliberately avoid pinning run-specific
# values such as final row counts, hashes, package paths, or software versions.
eq(PROTOCOL["cutoffs"], CUTOFFS, "protocol.cutoffs_match_summary")
ck(bool(CUTOFFS), "protocol.cutoffs_nonempty")
ck(
    all(a < b for a, b in zip(CUTOFFS, CUTOFFS[1:])),
    "protocol.cutoffs_strictly_increasing",
)
for key in (
    "max_year",
    "pca_components",
    "knn_k",
    "min_component_size",
    "min_community_size",
    "seed",
    "n_boot",
    "n_perm",
    "n_jobs",
):
    ck(
        isinstance(PROTOCOL.get(key), int),
        f"protocol.{key}.integer",
        repr(PROTOCOL.get(key)),
    )
for key in (
    "pca_components",
    "knn_k",
    "min_component_size",
    "min_community_size",
    "n_boot",
    "n_perm",
    "n_jobs",
):
    ck(PROTOCOL.get(key, 0) > 0, f"protocol.{key}.positive")
ck(
    isinstance(PROTOCOL.get("louvain_resolution"), (int, float))
    and PROTOCOL["louvain_resolution"] > 0,
    "protocol.louvain_resolution.positive_numeric",
)
ck(
    isinstance(PROTOCOL.get("partial_occupancy_fraction_tolerance"), (int, float))
    and 0 < PROTOCOL["partial_occupancy_fraction_tolerance"] < 1,
    "protocol.partial_occupancy_fraction_tolerance.valid",
)
for key in (
    "scaler",
    "pca",
    "partition",
    "formula_rule",
    "per_formula_identity",
    "per_formula_selection",
    "future_assignment",
):
    ck(
        isinstance(PROTOCOL.get(key), str) and bool(PROTOCOL[key].strip()),
        f"protocol.{key}.documented",
    )
ck(
    isinstance(PROTOCOL.get("formula_references"), list)
    and bool(PROTOCOL["formula_references"])
    and all(
        isinstance(value, str) and bool(value.strip())
        for value in PROTOCOL["formula_references"]
    ),
    "protocol.formula_references.documented",
)
ck(
    isinstance(SUMMARY.get("execution", {}).get("argv"), list)
    and bool(SUMMARY["execution"]["argv"]),
    "execution.argv.recorded",
)
for key in ("slurm_job_id", "hostname"):
    ck(
        SUMMARY.get("execution", {}).get(key) not in (None, ""),
        f"execution.{key}.recorded",
    )
for key in (
    "python",
    "numpy",
    "pandas",
    "scikit_learn",
    "networkx",
    "scipy",
    "pymatgen",
):
    ck(
        isinstance(SUMMARY.get("software", {}).get(key), str)
        and bool(SUMMARY["software"][key].strip()),
        f"software.{key}.recorded",
    )
for T, c in SUMMARY["cutoffs"].items():
    expected_outputs = {
        "training_partition": f"T{T}/training_partition.csv",
        "cutoff_map": f"T{T}/cutoff_map.npz",
        "heldout_entries": f"T{T}/heldout_entries.csv",
        "external_classifications": f"T{T}/external_classifications.csv",
    }
    eq(set(c["outputs"]), set(expected_outputs), f"T{T}.outputs.names")
    for name, expected_path in expected_outputs.items():
        eq(c["outputs"][name]["path"], expected_path, f"T{T}.output.{name}.path")
    for name, rec in c["outputs"].items():
        p = BASE / rec["path"]
        ck(p.exists(), f"T{T}.output.{name}.exists")
        if p.exists():
            actual_hash = sha(p)
            HASH_AUDIT["analysis_outputs"][f"T{T}/{name}"] = {
                "path": rec["path"],
                "expected_sha256": rec["sha256"],
                "actual_sha256": actual_hash,
                "matches": actual_hash == rec["sha256"],
            }
            eq(actual_hash, rec["sha256"], f"T{T}.output.{name}.sha256")
pa = SUMMARY["production_filter_audit"]["assignments"]
p = BASE / pa["path"]
ck(p.exists(), "production_audit.output.exists")
production_audit_hash = sha(p)
HASH_AUDIT["analysis_outputs"]["production_filter_audit_assignments"] = {
    "path": pa["path"],
    "expected_sha256": pa["sha256"],
    "actual_sha256": production_audit_hash,
    "matches": production_audit_hash == pa["sha256"],
}
eq(production_audit_hash, pa["sha256"], "production_audit.output.sha256")
manifest = json.loads((BASE / "reporting/reporting_manifest.json").read_text())
eq(
    sha(SUMMARY_PATH),
    manifest["source_summary_sha256"],
    "reporting.source_summary.sha256",
)
eq(
    sha(ROOT / "scripts/summarize_cutoff_trained_retrospective.py"),
    manifest["producer_sha256"],
    "reporting.producer.sha256",
)
for name, h in manifest["outputs"].items():
    actual_hash = sha(BASE / "reporting" / name)
    HASH_AUDIT["reporting_outputs"][name] = {
        "expected_sha256": h,
        "actual_sha256": actual_hash,
        "matches": actual_hash == h,
    }
    eq(actual_hash, h, f"reporting.{name}.sha256")

# Exact source rows and formula reference reconstruction.
sample = pd.read_csv(inputs_local["sample_assignments"], keep_default_na=False)
ids = sample.icsd_id.to_numpy(np.int64)
years = pd.to_numeric(sample.year, errors="coerce").fillna(-1).to_numpy(np.int64)
ck(sample.icsd_id.is_unique, "source.sample_ids_unique")
feature_metadata = json.loads(inputs_local["feature_metadata"].read_text())
eq(
    len(sample),
    feature_metadata["feature_diagnostics"]["n_structures"],
    "source.sample_rows_match_feature_metadata",
)
index = load_icsd_index(inputs_local["icsd_index"])
ck(set(map(int, ids)).issubset(index), "source.sample_ids_covered_by_index")
formulas = np.asarray(
    [str(index.get(int(i), {}).get("reduced_formula") or "") for i in ids], dtype=object
)
index_years = np.asarray(
    [
        int(index[int(identifier)]["year"])
        if index[int(identifier)].get("year") is not None
        else -1
        for identifier in ids
    ],
    dtype=np.int64,
)
ck(np.array_equal(years, index_years), "source.sample_years_exact_index")
occupancy_rows = pd.read_csv(inputs_local["occupancy_flags"], keep_default_na=False)
ck(
    {"icsd_id", "has_partial_occupancy"}.issubset(occupancy_rows.columns),
    "source.occupancy_required_columns",
)
ck(occupancy_rows.icsd_id.is_unique, "source.occupancy_ids_unique")
occupancy_values = pd.to_numeric(
    occupancy_rows.has_partial_occupancy, errors="coerce"
)
ck(occupancy_values.notna().all(), "source.occupancy_values_numeric")
ck(
    set(occupancy_values.dropna().astype(np.int64).unique()).issubset({0, 1}),
    "source.occupancy_values_binary",
)
occupancy_by_id = dict(
    zip(
        occupancy_rows.icsd_id.astype(np.int64),
        occupancy_values.fillna(-1).astype(np.int64).astype(bool),
    )
)
missing_occupancy = sorted(set(map(int, index)) - set(occupancy_by_id))
eq(missing_occupancy, [], "source.occupancy_covers_index")
partial_flags = np.asarray(
    [occupancy_by_id.get(int(identifier), False) for identifier in ids], dtype=bool
)

# Production reconstruction label equivalence.
prod = pd.read_csv(inputs_local["production_labels"], keep_default_na=False)
aud = pd.read_csv(
    BASE / "production_filter_audit_assignments.csv", keep_default_na=False
)
eq(len(aud), len(sample), "production_audit.rows")
ck(aud.icsd_id.is_unique, "production_audit.ids_unique")
ck(
    np.array_equal(aud.icsd_id.to_numpy(np.int64), ids),
    "production_audit.ids_exact_sample_order",
)
ck(
    np.array_equal(prod.icsd_id.to_numpy(np.int64), ids),
    "production_labels.ids_exact_sample_order",
)
ck(
    np.array_equal(
        aud.reported_community.to_numpy(np.int64), prod.community.to_numpy(np.int64)
    ),
    "production_audit.reported_labels_exact",
)
reported = aud.reported_community.to_numpy(np.int64)
reproduced = aud.reproduced_community.to_numpy(np.int64)
paudit = SUMMARY["production_filter_audit"]
pcmp = paudit["comparison_to_reported_partition"]
eq(
    len(set(reported[reported >= 0])),
    pcmp["reported_n_communities"],
    "production_audit.reported_n_communities",
)
eq(
    int((reported < 0).sum()),
    pcmp["reported_n_outliers"],
    "production_audit.reported_n_outliers",
)
eq(
    np.array_equal(reported < 0, reproduced < 0),
    pcmp["noise_mask_identical"],
    "production_audit.noise_identical",
)
eq(
    adjusted_rand_score(reported, reproduced),
    pcmp["adjusted_rand_index_all_entries"],
    "production_audit.ari",
)
eq(
    normalized_mutual_info_score(reported, reproduced, average_method="arithmetic"),
    pcmp["normalized_mutual_information_all_entries"],
    "production_audit.nmi",
)
eq(int((reproduced >= 0).sum()), paudit["n_assigned"], "production_audit.n_assigned")
eq(int((reproduced < 0).sum()), paudit["n_outliers"], "production_audit.n_outliers")
eq(
    len(set(reproduced[reproduced >= 0])),
    paudit["n_communities"],
    "production_audit.n_communities",
)
comp(
    paudit["largest_communities"],
    [[int(k), int(v)] for k, v in Counter(reproduced[reproduced >= 0]).most_common(20)],
    "production_audit.largest_communities",
)
eq(
    paudit["n_entries_in_components_below_minimum"]
    + paudit["n_nodes_after_component_filter"],
    paudit["n_train"],
    "production_audit.component_arithmetic",
)
eq(
    paudit["n_entries_in_components_below_minimum"]
    + paudit["n_entries_in_louvain_groups_below_minimum"],
    paudit["n_outliers"],
    "production_audit.outlier_arithmetic",
)
eq(
    paudit["n_louvain_groups_before_size_filter"]
    - paudit["n_louvain_groups_below_minimum"],
    paudit["n_communities"],
    "production_audit.community_arithmetic",
)

all_unions = []
tables = {}
primary = {}
formula_identity_match_constancy = {}
for Ttxt, c in sorted(SUMMARY["cutoffs"].items(), key=lambda x: int(x[0])):
    T = int(Ttxt)
    leaf = BASE / f"T{T}"
    train = pd.read_csv(leaf / "training_partition.csv", keep_default_na=False)
    held = pd.read_csv(leaf / "heldout_entries.csv", keep_default_na=False)
    ext = pd.read_csv(
        leaf / "external_classifications.csv",
        keep_default_na=False,
        dtype={"material_id": str},
    )
    formula_columns = {
        f"formula_match_{reference}" for reference in c["analyses"]
    }
    eq(
        set(train.columns),
        {"icsd_id", "year", "cutoff_community"},
        f"T{T}.train.columns",
    )
    eq(
        set(held.columns),
        {
            "icsd_id",
            "year",
            "formula",
            "assigned_community",
            "nearest_centroid_distance",
            "community_threshold_p95",
            "in_basin",
            "has_partial_occupancy",
            "stratum_coarse",
            "stratum_anon",
            "formula_identity",
            *formula_columns,
        },
        f"T{T}.held.columns",
    )
    eq(
        set(ext.columns),
        {
            "source",
            "material_id",
            "formula",
            "assigned_community",
            "nearest_centroid_distance",
            "community_threshold_p95",
            "in_basin",
            "stratum_coarse",
            "stratum_anon",
            "formula_identity",
            *formula_columns,
        },
        f"T{T}.external.columns",
    )
    tables[T] = (train, held, ext)
    trainmask = (years > 0) & (years <= T)
    heldmask = (years > T) & (years <= PROTOCOL["max_year"])
    # Row counts, ID order, temporal partitions.
    eq(len(train), int(trainmask.sum()), f"T{T}.train.row_count_source_mask")
    eq(len(train), c["n_train_dated"], f"T{T}.train.row_count_summary")
    eq(len(held), int(heldmask.sum()), f"T{T}.held.row_count_source_mask")
    eq(len(held), c["n_heldout_dated"], f"T{T}.held.row_count_summary")
    ck(train.icsd_id.is_unique, f"T{T}.train.ids_unique")
    ck(held.icsd_id.is_unique, f"T{T}.held.ids_unique")
    ck(
        np.array_equal(train.icsd_id.to_numpy(np.int64), ids[trainmask]),
        f"T{T}.train.ids_exact_source_order",
    )
    ck(
        np.array_equal(train.year.to_numpy(np.int64), years[trainmask]),
        f"T{T}.train.years_exact_source_order",
    )
    ck(
        np.array_equal(held.icsd_id.to_numpy(np.int64), ids[heldmask]),
        f"T{T}.held.ids_exact_source_order",
    )
    ck(
        np.array_equal(held.year.to_numpy(np.int64), years[heldmask]),
        f"T{T}.held.years_exact_source_order",
    )
    ck(
        set(train.icsd_id).isdisjoint(set(held.icsd_id)),
        f"T{T}.train_held_ids_disjoint",
    )
    ck((train.year.gt(0) & train.year.le(T)).all(), f"T{T}.train.cutoff_boundary")
    ck(
        (held.year.gt(T) & held.year.le(PROTOCOL["max_year"])).all(),
        f"T{T}.held.cutoff_boundary",
    )
    all_unions.append(set(train.icsd_id) | set(held.icsd_id))
    ck(
        np.array_equal(
            held.formula.astype(str).to_numpy(), formulas[heldmask].astype(str)
        ),
        f"T{T}.held.formulas_exact_index",
    )
    eq(
        int((held.formula != "").sum()),
        c["n_heldout_parseable_formula"],
        f"T{T}.held.parseable_count",
    )
    ck(
        "has_partial_occupancy" in held.columns,
        f"T{T}.held.has_partial_occupancy_column",
    )
    ck(
        np.array_equal(
            held.has_partial_occupancy.to_numpy(np.int64),
            partial_flags[heldmask].astype(np.int64),
        ),
        f"T{T}.held.has_partial_occupancy_exact_source",
    )
    expected_held_identities = np.asarray(
        [
            formula_identity_token(formula, bool(is_partial))
            for formula, is_partial in zip(
                formulas[heldmask], partial_flags[heldmask], strict=True
            )
        ],
        dtype=object,
    )
    ck(
        np.array_equal(
            held.formula_identity.astype(str).to_numpy(), expected_held_identities
        ),
        f"T{T}.held.formula_identity_exact_nominal_rule",
    )
    ck(
        np.array_equal(
            held.stratum_anon.astype(str).to_numpy(),
            held.formula.map(anonymous_stoichiometry_token).to_numpy(dtype=object),
        ),
        f"T{T}.held.anonymous_stoichiometry_exact_rule",
    )
    ck(
        np.array_equal(
            held.stratum_coarse.astype(str).to_numpy(),
            held.formula.map(composition_class).fillna("").astype(str).to_numpy(),
        ),
        f"T{T}.held.coarse_composition_stratum_exact_rule",
    )
    # Map and partition invariants.
    z = np.load(leaf / "cutoff_map.npz")
    expected_keys = {
        "scaler_mean",
        "scaler_scale",
        "pca_mean",
        "pca_components",
        "pca_explained_variance_ratio",
        "communities",
        "centroids",
        "radii_p95",
        "community_sizes",
    }
    eq(set(z.files), expected_keys, f"T{T}.map.keys")
    comm = z["communities"]
    sizes = z["community_sizes"]
    radii = z["radii_p95"]
    labels = train.cutoff_community.to_numpy(np.int64)
    part = c["partition"]
    for partition_key, protocol_key in (
        ("knn_k", "knn_k"),
        ("resolution", "louvain_resolution"),
        ("min_component_size", "min_component_size"),
        ("min_community_size", "min_community_size"),
        ("seed", "seed"),
    ):
        eq(
            part[partition_key],
            PROTOCOL[protocol_key],
            f"T{T}.partition.{partition_key}_matches_protocol",
        )
    feature_dimension = int(z["scaler_mean"].shape[0])
    pca_components = int(PROTOCOL["pca_components"])
    eq(z["scaler_mean"].shape, (feature_dimension,), f"T{T}.map.scaler_mean_shape")
    eq(z["scaler_scale"].shape, (feature_dimension,), f"T{T}.map.scaler_scale_shape")
    eq(z["pca_mean"].shape, (feature_dimension,), f"T{T}.map.pca_mean_shape")
    eq(
        z["pca_components"].shape,
        (pca_components, feature_dimension),
        f"T{T}.map.pca_shape",
    )
    eq(
        z["centroids"].shape,
        (len(comm), pca_components),
        f"T{T}.map.centroids_shape",
    )
    eq(
        z["pca_explained_variance_ratio"].shape,
        (pca_components,),
        f"T{T}.map.evr_shape",
    )
    eq(radii.shape, (len(comm),), f"T{T}.map.radii_shape")
    eq(sizes.shape, (len(comm),), f"T{T}.map.sizes_shape")
    for k in (
        "scaler_mean",
        "scaler_scale",
        "pca_mean",
        "pca_components",
        "pca_explained_variance_ratio",
        "centroids",
        "radii_p95",
    ):
        ck(np.isfinite(z[k]).all(), f"T{T}.map.{k}_finite")
    ck((z["scaler_scale"] > 0).all(), f"T{T}.map.scaler_scale_positive")
    ck((radii >= 0).all(), f"T{T}.map.radii_nonnegative")
    ck((sizes >= part["min_community_size"]).all(), f"T{T}.map.community_min_size")
    ck(
        np.array_equal(comm, np.arange(len(comm), dtype=np.int64)),
        f"T{T}.map.communities_contiguous",
    )
    eq(len(comm), part["n_communities"], f"T{T}.partition.n_communities_map")
    eq(
        int((labels >= 0).sum()),
        part["n_assigned"],
        f"T{T}.partition.n_assigned_labels",
    )
    eq(int((labels < 0).sum()), part["n_outliers"], f"T{T}.partition.n_outliers_labels")
    eq(
        len(set(labels[labels >= 0])),
        part["n_communities"],
        f"T{T}.partition.n_communities_labels",
    )
    label_counts = np.asarray([(labels == x).sum() for x in comm], dtype=np.int64)
    ck(np.array_equal(label_counts, sizes), f"T{T}.partition.community_sizes_exact")
    comp(
        part["largest_communities"],
        [[int(k), int(v)] for k, v in Counter(labels[labels >= 0]).most_common(20)],
        f"T{T}.partition.largest_communities",
    )
    eq(
        part["n_entries_in_components_below_minimum"]
        + part["n_nodes_after_component_filter"],
        part["n_train"],
        f"T{T}.partition.component_arithmetic",
    )
    eq(
        part["n_entries_in_components_below_minimum"]
        + part["n_entries_in_louvain_groups_below_minimum"],
        part["n_outliers"],
        f"T{T}.partition.outlier_arithmetic",
    )
    eq(
        part["n_louvain_groups_before_size_filter"]
        - part["n_louvain_groups_below_minimum"],
        part["n_communities"],
        f"T{T}.partition.community_arithmetic",
    )
    eq(
        float(z["pca_explained_variance_ratio"].sum()),
        c["pca_explained_variance_ratio_sum"],
        f"T{T}.map.evr_sum",
    )
    # Classification saved-map invariants for heldout and external rows.
    for name, frame in [("held", held), ("external", ext)]:
        vals = frame.assigned_community.to_numpy(np.int64)
        pos = np.searchsorted(comm, vals)
        ck(
            (pos < len(comm)).all() and np.array_equal(comm[pos], vals),
            f"T{T}.{name}.assigned_communities_exist",
        )
        expthr = radii[pos]
        ck(
            np.allclose(
                frame.community_threshold_p95.to_numpy(float),
                expthr,
                rtol=2e-14,
                atol=2e-14,
            ),
            f"T{T}.{name}.thresholds_exact_map",
        )
        dist = frame.nearest_centroid_distance.to_numpy(float)
        ib = frame.in_basin.to_numpy(np.int64)
        ck(
            np.isfinite(dist).all() and (dist >= 0).all(),
            f"T{T}.{name}.distances_finite_nonnegative",
        )
        ck(
            np.array_equal(ib, (dist <= expthr).astype(np.int64)),
            f"T{T}.{name}.in_basin_exact_inequality",
        )
        ck(set(np.unique(ib)).issubset({0, 1}), f"T{T}.{name}.in_basin_binary")
    # Reconstruct the declared hybrid formula reference: exact integer ratios
    # for fully occupied ICSD records and same-element-set atomic fractions for
    # partial-occupancy records.
    reference_records = {
        "all_year_le_T": [
            (str(formula), bool(is_partial))
            for formula, is_partial, year in zip(
                formulas, partial_flags, years, strict=True
            )
            if formula != "" and 0 < int(year) <= T
        ],
        "post1980_le_T": [
            (str(formula), bool(is_partial))
            for formula, is_partial, year in zip(
                formulas, partial_flags, years, strict=True
            )
            if formula != "" and 1980 < int(year) <= T
        ],
        "all_year_le_T_index": [
            (str(meta["reduced_formula"]), occupancy_by_id[int(identifier)])
            for identifier, meta in index.items()
            if meta.get("reduced_formula")
            and meta.get("year") is not None
            and int(meta["year"]) <= T
        ],
        "post1980_le_T_index": [
            (str(meta["reduced_formula"]), occupancy_by_id[int(identifier)])
            for identifier, meta in index.items()
            if meta.get("reduced_formula")
            and meta.get("year") is not None
            and 1980 < int(meta["year"]) <= T
        ],
    }
    tolerance = float(PROTOCOL["partial_occupancy_fraction_tolerance"])
    refs = {
        name: build_formula_precedent_index(records, tolerance=tolerance)
        for name, records in reference_records.items()
    }
    formula_identity_match_constancy[str(T)] = {}
    eq(set(refs), set(c["reference_sizes"]), f"T{T}.references.names")
    eq(
        set(refs),
        set(c["reference_composition_counts"]),
        f"T{T}.reference_composition_counts.names",
    )
    for ref, reference in refs.items():
        records = reference_records[ref]
        expected_size = len(
            {normalized_formula_identity_token(formula) for formula, _ in records}
        )
        eq(expected_size, c["reference_sizes"][ref], f"T{T}.reference.{ref}.size")
        expected_counts = {
            "records": len(records),
            "fully_occupied_integer_keys": len(reference.full_integer_keys),
            "partial_occupancy_element_sets": len(reference.partial_fraction_trees),
            "partial_occupancy_fraction_vectors": int(
                sum(tree.n for tree in reference.partial_fraction_trees.values())
            ),
        }
        comp(
            expected_counts,
            c["reference_composition_counts"][ref],
            f"T{T}.reference.{ref}.composition_counts",
        )
        held_flags = np.asarray(
            [formula_has_precedent(formula, reference) for formula in held.formula],
            dtype=np.int64,
        )
        external_flags = np.asarray(
            [formula_has_precedent(formula, reference) for formula in ext.formula],
            dtype=np.int64,
        )
        ck(
            np.array_equal(
                held[f"formula_match_{ref}"].to_numpy(np.int64), held_flags
            ),
            f"T{T}.held.reference.{ref}.flags",
        )
        nonblank = held.formula_identity.astype(str).ne("").to_numpy()
        constancy_frame = pd.DataFrame(
            {
                "formula_identity": held.loc[nonblank, "formula_identity"].astype(str),
                "formula_match": held_flags[nonblank],
            }
        )
        mixed = constancy_frame.groupby("formula_identity")[
            "formula_match"
        ].nunique()
        n_mixed = int((mixed > 1).sum())
        ck(
            n_mixed == 0,
            f"T{T}.held.reference.{ref}.formula_match_constant_within_nominal_identity",
            f"mixed_formula_identities={n_mixed}",
        )
        formula_identity_match_constancy[str(T)][ref] = {
            "n_heldout_records": int(nonblank.sum()),
            "n_nominal_formula_identities": int(len(mixed)),
            "reconstructed_formula_match_count": int(held_flags[nonblank].sum()),
            "n_identities_with_mixed_formula_precedent": n_mixed,
        }
        ck(
            np.array_equal(
                ext[f"formula_match_{ref}"].to_numpy(np.int64), external_flags
            ),
            f"T{T}.external.reference.{ref}.flags",
        )
    # External source row identities and raw metadata.
    expected_external_rows = sum(
        int(
            json.loads(
                inputs_local[f"external_{source}_feature_metadata"].read_text()
            )["n_rows"]
        )
        for source in source_slug
    )
    eq(len(ext), expected_external_rows, f"T{T}.external.total_rows")
    eq(
        list(ext.source.drop_duplicates()),
        list(source_slug),
        f"T{T}.external.source_order",
    )
    expected_external_identities = ext.formula.map(
        normalized_formula_identity_token
    ).to_numpy(dtype=object)
    ck(
        np.array_equal(
            ext.formula_identity.astype(str).to_numpy(),
            expected_external_identities,
        ),
        f"T{T}.external.formula_identity_exact_nominal_rule",
    )
    ck(
        np.array_equal(
            ext.stratum_anon.astype(str).to_numpy(),
            ext.formula.map(anonymous_stoichiometry_token).to_numpy(dtype=object),
        ),
        f"T{T}.external.anonymous_stoichiometry_exact_rule",
    )
    ck(
        np.array_equal(
            ext.stratum_coarse.astype(str).to_numpy(),
            ext.formula.map(composition_class).fillna("").astype(str).to_numpy(),
        ),
        f"T{T}.external.coarse_composition_stratum_exact_rule",
    )
    for s in source_slug:
        g = ext[ext.source == s].reset_index(drop=True)
        raw = pd.read_csv(
            inputs_local[f"external_{s}_records"],
            keep_default_na=False,
            dtype={"material_id": str},
        )
        meta = json.loads(inputs_local[f"external_{s}_feature_metadata"].read_text())
        eq(len(g), len(raw), f"T{T}.external.{s}.rows")
        eq(len(g), meta["n_rows"], f"T{T}.external.{s}.metadata_rows")
        eq(
            meta["n_features"],
            feature_dimension,
            f"T{T}.external.{s}.metadata_features",
        )
        ck(
            np.array_equal(
                g.material_id.astype(str).to_numpy(),
                raw.material_id.astype(str).to_numpy(),
            ),
            f"T{T}.external.{s}.ids_exact_raw_order",
        )
        ck(
            np.array_equal(
                g.formula.astype(str).to_numpy(),
                raw.reduced_formula.astype(str).to_numpy(),
            ),
            f"T{T}.external.{s}.formulas_exact_raw_order",
        )
        if "feature_row" in raw:
            ck(
                np.array_equal(raw.feature_row.to_numpy(np.int64), np.arange(len(raw))),
                f"T{T}.external.{s}.feature_row_sequence",
            )
    # Rate and shared-strata arithmetic from row-level CSVs against summary.
    comp(rate(held.in_basin), c["heldout_rate"], f"T{T}.heldout_rate")
    for s, v in c["external_source_rates"].items():
        g = ext[ext.source == s]
        comp(rate(g.in_basin), v["all_records"], f"T{T}.external_rate.{s}.all")
        eq(
            float(held.in_basin.mean() - g.in_basin.mean()),
            v["heldout_minus_external"],
            f"T{T}.external_rate.{s}.gap",
        )
        comp(
            shared(held, g, "stratum_coarse"),
            v["shared_coarse_strata"],
            f"T{T}.external_rate.{s}.shared_coarse",
        )
        comp(
            shared(held, g, "stratum_anon"),
            v["shared_anonymized_strata"],
            f"T{T}.external_rate.{s}.shared_anon",
        )
    # All 8 deterministic analyses per cutoff: quadrants and every by-year cell.
    parseable = held[held.formula != ""].copy()
    ck(
        (parseable.formula_identity != "").all(),
        f"T{T}.parseable.formula_identity_nonempty",
    )
    first_year = parseable.groupby("formula_identity").year.transform("min")
    earliest_year_rows = parseable[parseable.year == first_year].copy()
    first = (
        parseable.sort_values(["year", "icsd_id"], kind="stable")
        .drop_duplicates("formula_identity", keep="first")
        .copy()
    )
    eq(
        len(first),
        c["n_first_postcutoff_entry_per_formula"],
        f"T{T}.first_formula.count",
    )
    ck(first.formula_identity.is_unique, f"T{T}.first_formula.identity_unique")
    earliest_counts = earliest_year_rows.groupby("formula_identity").size()
    earliest_basin_counts = earliest_year_rows.groupby("formula_identity")[
        "in_basin"
    ].nunique()
    expected_tie_audit = {
        "n_formula_identities": int(len(earliest_counts)),
        "n_with_multiple_entries_in_earliest_year": int(
            np.sum(earliest_counts.to_numpy() > 1)
        ),
        "n_with_mixed_basin_status_in_earliest_year": int(
            np.sum(earliest_basin_counts.to_numpy() > 1)
        ),
        "deterministic_tie_rule": "lowest ICSD identifier",
        "by_formula_reference": {
            ref: earliest_year_tie_sensitivity(
                earliest_year_rows, f"formula_match_{ref}"
            )
            for ref in refs
        },
    }
    comp(
        expected_tie_audit,
        c["earliest_year_tie_audit"],
        f"T{T}.earliest_year_tie_audit",
    )
    for ref in refs:
        for unit, frame0 in [("per_entry", parseable), ("per_formula", first)]:
            frame = frame0.copy()
            frame["formula_match"] = frame[f"formula_match_{ref}"]
            a = c["analyses"][ref][unit]
            q = quadrant(frame)
            by = byyear(frame)
            comp(q, a["quadrant"], f"T{T}.analysis.{ref}.{unit}.quadrant")
            comp(by, a["by_year"], f"T{T}.analysis.{ref}.{unit}.by_year")
            eq(
                a["bootstrap"]["n_boot"],
                PROTOCOL["n_boot"],
                f"T{T}.analysis.{ref}.{unit}.bootstrap_n",
            )
            ck(
                a["bootstrap"]["enrichment_ratio_ci95"][0]
                <= q["enrichment_ratio_obs_over_independence"]
                <= a["bootstrap"]["enrichment_ratio_ci95"][1],
                f"T{T}.analysis.{ref}.{unit}.enrichment_ci_contains_point",
            )
            ck(
                a["bootstrap"]["share_in_basin_and_match_ci95"][0]
                <= q["share_in_basin_and_match"]
                <= a["bootstrap"]["share_in_basin_and_match_ci95"][1],
                f"T{T}.analysis.{ref}.{unit}.share_ci_contains_point",
            )
            ck(
                a["bootstrap"]["relative_risk_ci95"][0]
                <= q["relative_risk_in_basin_match_vs_no_match"]
                <= a["bootstrap"]["relative_risk_ci95"][1],
                f"T{T}.analysis.{ref}.{unit}.rr_ci_contains_point",
            )
            for pn, pv in (
                (k, v) for k, v in a.items() if k.startswith("permutation_")
            ):
                eq(
                    pv["observed_share"],
                    q["share_in_basin_and_match"],
                    f"T{T}.analysis.{ref}.{unit}.{pn}.observed_share",
                )
                eq(
                    pv["n_perm"],
                    PROTOCOL["n_perm"],
                    f"T{T}.analysis.{ref}.{unit}.{pn}.n_perm",
                )
                ck(
                    0 < pv["p_one_sided"] <= 1,
                    f"T{T}.analysis.{ref}.{unit}.{pn}.p_valid",
                )
                permutation_numerator = pv["p_one_sided"] * (
                    PROTOCOL["n_perm"] + 1
                )
                ck(
                    math.isclose(
                        permutation_numerator,
                        round(permutation_numerator),
                        rel_tol=0,
                        abs_tol=1e-12,
                    ),
                    f"T{T}.analysis.{ref}.{unit}.{pn}.p_on_exact_grid",
                )
            eq(
                a["permutation_global"]["n_strata"],
                1,
                f"T{T}.analysis.{ref}.{unit}.global_strata",
            )
            eq(
                a["permutation_within_year"]["n_strata"],
                frame.year.astype(str).nunique(),
                f"T{T}.analysis.{ref}.{unit}.year_strata",
            )
            eq(
                a["permutation_within_coarse_strata"]["n_strata"],
                frame.stratum_coarse.fillna("__none__").astype(str).nunique(),
                f"T{T}.analysis.{ref}.{unit}.coarse_strata",
            )
            eq(
                a["permutation_within_anonymized_strata"]["n_strata"],
                frame.stratum_anon.fillna("__none__").astype(str).nunique(),
                f"T{T}.analysis.{ref}.{unit}.anon_strata",
            )
            if ref == "all_year_le_T_index":
                primary[(T, unit)] = {
                    "quadrant": q,
                    "bootstrap": a["bootstrap"],
                    "permutation_p": {
                        k: v["p_one_sided"]
                        for k, v in a.items()
                        if k.startswith("permutation_")
                    },
                }

# Same dated ID universe and exact chronological nesting across cutoffs.
ck(all(x == all_unions[0] for x in all_unions[1:]), "cross_cutoff.same_dated_universe")
eq(
    len(all_unions[0]),
    int(((years > 0) & (years <= PROTOCOL["max_year"])).sum()),
    "cross_cutoff.dated_universe_count",
)
for A, B in zip(CUTOFFS, CUTOFFS[1:]):
    ta, ha, _ = tables[A]
    tb, hb, _ = tables[B]
    ck(set(ta.icsd_id).issubset(set(tb.icsd_id)), f"cross_cutoff.train_{A}_subset_{B}")
    ck(set(hb.icsd_id).issubset(set(ha.icsd_id)), f"cross_cutoff.held_{B}_subset_{A}")
    ck(
        set(tb.icsd_id) == set(ta.icsd_id) | set(ha.loc[ha.year <= B, "icsd_id"]),
        f"cross_cutoff.train_{B}_exact_transition",
    )
    ck(
        set(hb.icsd_id) == set(ha.loc[ha.year > B, "icsd_id"]),
        f"cross_cutoff.held_{B}_exact_transition",
    )

# Reporting row counts and direct equality for every saved joint row.
rates = pd.read_csv(BASE / "reporting/cutoff_trained_rates.csv", keep_default_na=False)
sharedrep = pd.read_csv(
    BASE / "reporting/cutoff_trained_shared_strata.csv", keep_default_na=False
)
joint = pd.read_csv(BASE / "reporting/cutoff_trained_joint.csv", keep_default_na=False)
n_cutoffs = len(CUTOFFS)
n_sources = len(source_slug)
n_references = len(next(iter(SUMMARY["cutoffs"].values()))["analyses"])
eq(len(rates), (1 + n_sources) * n_cutoffs, "reporting.rates.rows")
eq(len(sharedrep), 2 * n_sources * n_cutoffs, "reporting.shared.rows")
eq(len(joint), 2 * n_references * n_cutoffs, "reporting.joint.rows")
ck(
    not joint.duplicated(["cutoff", "reference", "unit"]).any(),
    "reporting.joint.keys_unique",
)
for _, row in joint.iterrows():
    T = int(row.cutoff)
    ref = row.reference
    unit = row.unit
    a = SUMMARY["cutoffs"][str(T)]["analyses"][ref][unit]
    q = a["quadrant"]
    b = a["bootstrap"]
    expected = {
        "n": q["n"],
        "in_basin_and_match": q["in_basin_and_match"],
        "share_in_basin_and_match": q["share_in_basin_and_match"],
        "p_in_basin": q["p_in_basin"],
        "p_formula_match": q["p_match"],
        "enrichment_ratio": q["enrichment_ratio_obs_over_independence"],
        "enrichment_ci95_low": b["enrichment_ratio_ci95"][0],
        "enrichment_ci95_high": b["enrichment_ratio_ci95"][1],
        "relative_risk": q["relative_risk_in_basin_match_vs_no_match"],
        "permutation_global_p_one_sided": a["permutation_global"]["p_one_sided"],
        "permutation_within_year_p_one_sided": a["permutation_within_year"][
            "p_one_sided"
        ],
        "permutation_within_coarse_p_one_sided": a["permutation_within_coarse_strata"][
            "p_one_sided"
        ],
        "permutation_within_anonymized_p_one_sided": a[
            "permutation_within_anonymized_strata"
        ]["p_one_sided"],
    }
    for k, v in expected.items():
        eq(row[k], v, f"reporting.joint.T{T}.{ref}.{unit}.{k}")

# The stochastic arrays are intentionally not regenerated here because doing so
# repeats the expensive resampling analysis. Bind their declared stream order
# to the hashed producer and verify exact p-value grid and sample-count
# invariants above.
producer_text = inputs_local["producer"].read_text()
rng_contract_snippets = [
    "rng = np.random.default_rng(args.seed + cutoff)",
    "for reference_name in reference_sets:",
    'for unit_name, unit in (("per_entry", parseable), ("per_formula", first_formula)):',
    "units[unit_name] = statistical_block(frame, rng, args.n_boot, args.n_perm)",
]
statistical_call_order = [
    '"quadrant": quadrant(frame)',
    '"by_year": by_year(frame)',
    '"bootstrap": bootstrap(frame, rng, n_boot=n_boot)',
    '"permutation_global": permutation(frame, rng, n_perm=n_perm)',
    '"permutation_within_year": permutation(frame, rng, "year", n_perm=n_perm)',
    '"permutation_within_coarse_strata": permutation(',
    '"permutation_within_anonymized_strata": permutation(',
]
statistical_call_positions = [
    producer_text.find(snippet) for snippet in statistical_call_order
]
rng_contract = {
    "seed_rule": "default_rng(protocol seed + cutoff)",
    "reference_order": list(next(iter(SUMMARY["cutoffs"].values()))["analyses"]),
    "unit_order": ["per_entry", "per_formula"],
    "statistical_call_order": [
        "quadrant",
        "by_year",
        "bootstrap",
        "permutation_global",
        "permutation_within_year",
        "permutation_within_coarse_strata",
        "permutation_within_anonymized_strata",
    ],
    "numpy_version_recorded": SUMMARY["software"]["numpy"],
    "producer_contract_found": all(
        snippet in producer_text for snippet in rng_contract_snippets
    )
    and all(position >= 0 for position in statistical_call_positions)
    and all(
        earlier < later
        for earlier, later in zip(
            statistical_call_positions,
            statistical_call_positions[1:],
        )
    ),
}
ck(rng_contract["producer_contract_found"], "stochastic.producer_stream_order_contract")

# Rebuild the compact reports in a temporary directory. The renderer does not
# depend on any raw feature matrix, so all five outputs should reproduce byte
# for byte, including its manifest.
logical_assertions = N
report_names = [
    "cutoff_trained_composition_summary.json",
    "cutoff_trained_joint.csv",
    "cutoff_trained_rates.csv",
    "cutoff_trained_shared_strata.csv",
    "reporting_manifest.json",
]
reporting_rebuild = {}
try:
    summary_argument = str(SUMMARY_PATH.relative_to(ROOT))
except ValueError:
    summary_argument = str(SUMMARY_PATH)
with tempfile.TemporaryDirectory(prefix="cutoff-trained-reporting-") as temporary:
    command = [
        sys.executable,
        str(ROOT / "scripts/summarize_cutoff_trained_retrospective.py"),
        "--summary",
        summary_argument,
        "--output-dir",
        temporary,
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    for name in report_names:
        original = BASE / "reporting" / name
        rebuilt = Path(temporary) / name
        byte_identical = (
            completed.returncode == 0
            and original.exists()
            and rebuilt.exists()
            and original.read_bytes() == rebuilt.read_bytes()
        )
        reporting_rebuild[name] = {
            "byte_identical": byte_identical,
            "original_sha256": sha(original) if original.exists() else None,
            "rebuilt_sha256": sha(rebuilt) if rebuilt.exists() else None,
        }
        ck(byte_identical, f"reporting.rebuild.{name}.byte_identical", completed.stderr)

missing_tacc_inputs = [
    {
        "key": key,
        "recorded_tacc_path": SUMMARY["inputs"][key]["path"],
        "recorded_sha256": SUMMARY["inputs"][key]["sha256"],
        "reason": "large raw feature matrix is not copied into the local checkout",
    }
    for key in missing_inputs
]

producer_lines = (
    (ROOT / "scripts/analyze_cutoff_trained_retrospective.py").read_text().splitlines()
)
console_selector_line = next(
    (
        line_number
        for line_number, line in enumerate(producer_lines, start=1)
        if 'primary = analyses["all_year_le_T"]' in line
    ),
    None,
)
cosmetic_note = {
    "kind": "console-only formula-reference selection",
    "producer_path": "scripts/analyze_cutoff_trained_retrospective.py",
    "producer_line": console_selector_line,
    "packaged_slurm_log_paths": sorted(path.name for path in BASE.glob("slurm-*.out")),
    "observed_console_reference": "all_year_le_T",
    "chosen_reporting_reference": "all_year_le_T_index",
    "console_enrichment_values": [
        SUMMARY["cutoffs"][str(cutoff)]["analyses"]["all_year_le_T"]["per_entry"][
            "quadrant"
        ]["enrichment_ratio_obs_over_independence"]
        for cutoff in CUTOFFS
    ],
    "chosen_primary_enrichment_values": [
        SUMMARY["cutoffs"][str(cutoff)]["analyses"]["all_year_le_T_index"]["per_entry"][
            "quadrant"
        ]["enrichment_ratio_obs_over_independence"]
        for cutoff in CUTOFFS
    ],
    "impact": (
        "None on the summary JSON, reporting tables, or manuscript values; only the "
        "terminal progress line uses the successfully-featurized reference."
    ),
}

report = {
    "schema_version": 2,
    "purpose": "independent verification of the local cutoff-trained package",
    "all_passed": not FAIL,
    "assertion_counts": {
        "deterministic_package_assertions": logical_assertions,
        "reporting_byte_comparisons": len(report_names),
        "total_checks": N,
        "failed_checks": len(FAIL),
    },
    "verifier": {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha(Path(__file__).resolve()),
    },
    "source_package_hashes": {
        "cutoff_summary_sha256": sha(SUMMARY_PATH),
        "reporting_manifest_sha256": sha(BASE / "reporting/reporting_manifest.json"),
        "analysis_producer_sha256": sha(
            ROOT / "scripts/analyze_cutoff_trained_retrospective.py"
        ),
        "formula_conventions_sha256": sha(ROOT / "scripts/formula_conventions.py"),
        "index_loader_sha256": sha(
            ROOT / "scripts/analyze_synthesis_retrodiction.py"
        ),
        "occupancy_flags_sha256": sha(inputs_local["occupancy_flags"]),
        "statistical_module_sha256": sha(
            ROOT / "notes/review_2026_08/retrospective_quadrant.py"
        ),
        "execution_wrapper_sha256": sha(inputs_local["wrapper"]),
        "reporting_producer_sha256": sha(
            ROOT / "scripts/summarize_cutoff_trained_retrospective.py"
        ),
    },
    "recorded_hash_audit": HASH_AUDIT,
    "stochastic_stream_contract": rng_contract,
    "reporting_rebuild": {
        "producer_exit_code": completed.returncode,
        "files": reporting_rebuild,
    },
    "missing_tacc_only_raw_inputs": missing_tacc_inputs,
    "scope_limitation": (
        "The saved maps, partitions, classifications, and all downstream arithmetic "
        "are independently checked locally. Feature projection and graph fitting "
        "cannot be rerun without the six TACC-only raw feature matrices."
    ),
    "cosmetic_slurm_console_note": cosmetic_note,
    "dated_icsd_universe": len(all_unions[0]),
    "cutoff_partitions": {
        str(cutoff): {
            "n_train": SUMMARY["cutoffs"][str(cutoff)]["n_train_dated"],
            "n_heldout": SUMMARY["cutoffs"][str(cutoff)]["n_heldout_dated"],
            "n_communities": SUMMARY["cutoffs"][str(cutoff)]["partition"][
                "n_communities"
            ],
            "n_assigned": SUMMARY["cutoffs"][str(cutoff)]["partition"]["n_assigned"],
            "n_outliers": SUMMARY["cutoffs"][str(cutoff)]["partition"]["n_outliers"],
        }
        for cutoff in CUTOFFS
    },
    "report_row_counts": {
        "rates": len(rates),
        "shared_strata": len(sharedrep),
        "joint_analyses": len(joint),
    },
    "primary_all_year_le_T_index": {
        f"T{cutoff}_{unit}": value for (cutoff, unit), value in primary.items()
    },
    "formula_identity_match_constancy": {
        "rule": (
            "all held-out records sharing a 12-decimal normalized nominal-"
            "composition identity must have the same reconstructed formula-"
            "precedent flag within each cutoff and reference scope"
        ),
        "n_cutoff_reference_checks": int(
            sum(len(value) for value in formula_identity_match_constancy.values())
        ),
        "n_checks_with_mixed_flags": int(
            sum(
                saved["n_identities_with_mixed_formula_precedent"] > 0
                for by_reference in formula_identity_match_constancy.values()
                for saved in by_reference.values()
            )
        ),
        "cutoffs": formula_identity_match_constancy,
    },
    "failures": FAIL,
}
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(
    json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
)
print(
    json.dumps(
        {
            "all_passed": report["all_passed"],
            "total_checks": N,
            "failed_checks": len(FAIL),
            "report": str(REPORT_PATH),
            "verifier_sha256": report["verifier"]["sha256"],
        },
        indent=2,
    )
)
if FAIL:
    for failure in FAIL[:50]:
        print("FAIL", failure)
    raise SystemExit(1)
