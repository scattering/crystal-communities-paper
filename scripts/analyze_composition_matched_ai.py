#!/usr/bin/env python3
"""Composition-matched AI control for the held-out frontier-rate comparison.

Reviewer concern: the embedding is chemistry-aware, so a reviewer can argue
that GNoME / MatterGen look frontier-like simply because their compositions
differ from held-out ICSD. This script tests that explanation directly.

For each cutoff (1990, 2000, 2010):

  1. Load per-cutoff post_cutoff_accessibility_records.csv (held-out
     ICSD entries with reduced_formula and is_in_basin).
  2. Recompute per-community 95th-percentile centroid-distance thresholds
     from the cutoff's training-only data using features_pca.npy +
     community_assignments.csv + sample_assignments.csv.
  3. Re-classify the public GNoME and MatterGen samples against those
     cutoff thresholds, using their existing nearest_centroid_distance
     and assigned_community in the records CSVs (same approximation as
     analyze_gnome_temporal_sweep.py).
  4. Compute a composition class for every structure: (anonymized
     stoichiometry pattern, dominant anion class, n_unique_elements).
  5. Within each composition stratum that has both AI proposals AND
     held-out ICSD, compute matched in-basin rates with Wilson 95% CIs.

Outputs:
  - notes/composition_matched_ai_summary.json
  - notes/composition_matched_ai_records.csv

The Extended Data Fig 2 PNG is rendered separately by
``scripts/make_fig_composition_matched_ai.py`` reading the summary JSON;
the legacy ``--output-figure`` argument here calls into the same code
path for backward compatibility but is no longer required.

If the AI vs human gap survives composition matching, the original
claim is robust. If it shrinks substantially, the gap is mostly a
composition artifact and the central claim should be softened.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

ANIONS = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="features_pca.npy")
    p.add_argument("--community-assignments", required=True)
    p.add_argument("--sample-assignments", required=True,
                   help="if not separate, pass community-assignments here too (same schema)")
    p.add_argument("--post-cutoff-dir", required=True,
                   help="directory containing post_cutoff_<year>.csv per cutoff")
    p.add_argument("--post-cutoff-pattern", default="post_cutoff_{cutoff}.csv",
                   help="filename pattern relative to --post-cutoff-dir, with "
                   "literal '{cutoff}' placeholder. Use e.g. "
                   "'split_{cutoff}/post_cutoff_accessibility_records.csv' for "
                   "the synthesis_retrodiction layout.")
    # AI sources: pass --ai-source NAME PATH once per source. Replaces the
    # older --gnome-records / --mattergen-records pair. NAME shows up as the
    # series key in the summary JSON and the legend label in the figure.
    p.add_argument("--ai-source", action="append", nargs=2, metavar=("NAME", "PATH"),
                   help="AI source as 'NAME PATH' pair. Repeatable.")
    # Deprecated aliases kept for backward compat with the old slurm script.
    p.add_argument("--gnome-records", help="(deprecated) alias for --ai-source GNoME PATH")
    p.add_argument("--mattergen-records", help="(deprecated) alias for --ai-source MatterGen PATH")
    p.add_argument("--cutoffs", type=int, nargs="+", default=[1990, 2000, 2010])
    p.add_argument("--threshold-percentile", type=float, default=95.0)
    p.add_argument("--output-summary", required=True)
    p.add_argument("--output-records", required=True)
    p.add_argument(
        "--output-figure",
        default=None,
        help="Optional. If provided, also render the Extended Data Fig 2 "
             "two-panel bar chart inline. The preferred path is to skip "
             "this argument and render the figure separately via "
             "scripts/make_fig_composition_matched_ai.py from the JSON "
             "summary, so styling tweaks do not require re-running the "
             "Wilson-CI computation.",
    )
    return p.parse_args()


# --- formula parsing & composition class -----------------------------------

def parse_formula(formula: str) -> Optional[dict[str, float]]:
    if not formula:
        return None
    out: dict[str, float] = {}
    pattern = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)?")
    pos = 0
    while pos < len(formula):
        m = pattern.match(formula, pos)
        if not m:
            return None
        el = m.group(1)
        amt = float(m.group(2)) if m.group(2) else 1.0
        out[el] = out.get(el, 0.0) + amt
        if m.end() == m.start():
            return None
        pos = m.end()
    return out if out else None


def composition_class(formula: str) -> Optional[str]:
    """Coarse composition descriptor: (anion class, n_elements bucket, ratio
    bucket). ~40-50 strata. Defensive against shrinking match counts to
    near-zero on the small MatterGen sample."""
    elems = parse_formula(formula)
    if not elems:
        return None
    cations = {e: n for e, n in elems.items() if e not in ANIONS}
    anions = {e: n for e, n in elems.items() if e in ANIONS}
    n_elem = len(elems)
    cat_total = sum(cations.values())
    ani_total = sum(anions.values())

    # anion-class bucket
    if not anions:
        anion_cls = "intermetallic"
    else:
        anion_dominant = max(anions, key=anions.get)
        anion_cls = {
            "O": "oxide", "S": "sulfide", "Se": "selenide", "Te": "telluride",
            "F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide",
            "N": "nitride",
        }.get(anion_dominant, "mixed-anion")
        if len(anions) > 1:
            anion_cls = "mixed-anion"

    # anion/cation ratio bucket
    if cat_total > 0 and anions:
        r = ani_total / cat_total
        if r < 0.6: ratio_bucket = "lo"
        elif r < 1.1: ratio_bucket = "11"
        elif r < 1.7: ratio_bucket = "15"
        elif r < 2.3: ratio_bucket = "20"
        elif r < 3.5: ratio_bucket = "30"
        else: ratio_bucket = "hi"
    elif anions:
        ratio_bucket = "all-anion"
    else:
        ratio_bucket = "no-anion"

    n_bucket = str(n_elem) if n_elem <= 4 else "5+"
    return f"{anion_cls}|n{n_bucket}|r{ratio_bucket}"


def anonymized_formula(formula: str) -> Optional[str]:
    """Pymatgen-style anonymized stoichiometry. MgAl2O4 → 'A2B1C4',
    ZnFe2O4 → 'A2B1C4', LaMnO3 → 'A1B1C3'. Sites carry no element identity;
    integer ratios after reduction to smallest common form.

    This is the textbook clean composition-matching descriptor: two
    structures share an anonymized formula iff they are the same
    stoichiometry, regardless of which elements are involved. ~hundreds
    of strata in the full ICSD; AI populations may not populate all of
    them (so matched-stratum counts shrink relative to the coarse
    descriptor above).
    """
    elems = parse_formula(formula)
    if not elems:
        return None
    # Round non-integer occupancies to the nearest reasonable rational; for
    # solid-solution formulas like Cu1.5Se2Y0.84 we round to 0.05 then take
    # gcd-like reduction. We use a simple multiply-by-100 + integer gcd
    # reduction to keep this stdlib-only and deterministic.
    amts = sorted(elems.values(), reverse=True)
    int_amts = [int(round(a * 100)) for a in amts]
    from math import gcd
    g = int_amts[0]
    for v in int_amts[1:]:
        g = gcd(g, v)
    if g == 0:
        return None
    reduced = [v // g for v in int_amts]
    # Cap to reasonable display: drop trailing 0s, sort
    parts = "".join(f"{chr(ord('A') + i)}{n}" for i, n in enumerate(reduced))
    return parts


# --- loaders --------------------------------------------------------------

def load_id_year_community(path: Path) -> tuple[list[int], list[Optional[int]], list[int]]:
    icsd_ids: list[int] = []
    years: list[Optional[int]] = []
    comms: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                icsd_ids.append(int(row["icsd_id"]))
            except (KeyError, ValueError):
                continue
            year = row.get("year", "").strip()
            years.append(int(year) if year else None)
            try:
                comms.append(int(row["community"]))
            except (KeyError, ValueError):
                comms.append(-1)
    return icsd_ids, years, comms


def load_post_cutoff(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "cif_id": int(row["cif_id"]),
                    "formula": (row.get("reduced_formula") or "").strip(),
                    "year": int(row["year"]) if row.get("year") else None,
                    "is_in_basin": int(row["is_in_basin"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def load_ai_records(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "id": row.get("material_id") or row.get("zip_member") or "",
                    "formula": (row.get("reduced_formula") or "").strip(),
                    "assigned_community": int(row["assigned_community"]),
                    "nearest_centroid_distance": float(row["nearest_centroid_distance"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


# --- threshold computation ------------------------------------------------

def compute_thresholds(X: np.ndarray, labels: np.ndarray, percentile: float) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in sorted({int(v) for v in labels if int(v) >= 0}):
        members = np.flatnonzero(labels == c)
        if len(members) == 0:
            continue
        center = X[members].mean(axis=0)
        dists = np.linalg.norm(X[members] - center, axis=1)
        out[c] = float(np.percentile(dists, percentile))
    return out


# --- statistics -----------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def matched_rate(records: list[tuple[str, int]], strata_keep: set[str]) -> tuple[int, int, float, float, float]:
    """records = [(stratum, in_basin_flag), ...]; restricts to strata_keep
    and returns (k, n, rate, lo, hi)."""
    kept = [(s, x) for s, x in records if s in strata_keep]
    n = len(kept)
    k = sum(x for _, x in kept)
    rate = k / n if n else float("nan")
    lo, hi = wilson(k, n)
    return k, n, rate, lo, hi


# --- main -----------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print("loading features and community assignments...", flush=True)
    X = np.load(args.features)
    # match analyze_gnome_temporal_sweep.py: standardize, then PCA(32)
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    from sklearn.decomposition import PCA
    Xp = PCA(n_components=min(32, Xs.shape[0], Xs.shape[1]), random_state=42).fit_transform(Xs)

    icsd_ids, years_list, comms_list = load_id_year_community(Path(args.community_assignments))
    if len(icsd_ids) != Xp.shape[0]:
        raise SystemExit(f"row mismatch: features {Xp.shape[0]} vs assignments {len(icsd_ids)}")
    years_arr = np.array([y if y is not None else -1 for y in years_list], dtype=int)
    comms_arr = np.array(comms_list, dtype=int)

    # Build the (name, records) source list from --ai-source plus the
    # deprecated --gnome-records / --mattergen-records aliases.
    source_specs: list[tuple[str, str]] = list(args.ai_source or [])
    if args.gnome_records:
        source_specs.append(("GNoME", args.gnome_records))
    if args.mattergen_records:
        source_specs.append(("MatterGen", args.mattergen_records))
    if not source_specs:
        raise SystemExit("no AI sources provided; use --ai-source NAME PATH (repeatable)")

    print("loading AI records...", flush=True)
    sources: list[tuple[str, list[dict]]] = []
    for name, path in source_specs:
        rows = load_ai_records(Path(path))
        sources.append((name, rows))
        print(f"  {len(rows)} {name}", flush=True)

    cutoff_results: list[dict[str, object]] = []
    all_records: list[dict[str, object]] = []  # for the per-record CSV

    # Loop over both matching strategies: coarse strata + anonymized formula.
    # Anonymized matching is the textbook clean version; coarse matching is
    # a defensive fallback that keeps more AI proposals matched.
    matchers = [("coarse", composition_class), ("anonymized", anonymized_formula)]

    for cutoff in args.cutoffs:
        print(f"\n=== cutoff {cutoff} ===", flush=True)
        train_mask = (years_arr > 0) & (years_arr <= cutoff)
        train_X = Xp[train_mask]
        train_labels = comms_arr[train_mask]
        thresholds = compute_thresholds(train_X, train_labels, args.threshold_percentile)
        print(f"  trained {len(thresholds)} community thresholds at p{args.threshold_percentile:.0f}", flush=True)

        # held-out ICSD: from post_cutoff CSV
        pc_path = Path(args.post_cutoff_dir) / args.post_cutoff_pattern.format(cutoff=cutoff)
        pc_rows = load_post_cutoff(pc_path)
        print(f"  {len(pc_rows)} held-out ICSD records loaded from {pc_path}", flush=True)

        # Pre-classify each AI source against the cutoff thresholds. The
        # classification is reused across both matching strategies (coarse
        # and anonymized).
        classified_by_source: dict[str, list[dict]] = {}
        for src_name, rows in sources:
            cl = []
            for r in rows:
                thr = thresholds.get(int(r["assigned_community"]))
                if thr is None:
                    continue
                in_basin = 1 if float(r["nearest_centroid_distance"]) <= thr else 0
                cl.append({"id": r["id"], "formula": r["formula"], "in_basin": in_basin})
            classified_by_source[src_name] = cl

        cutoff_block: dict[str, object] = {"cutoff": cutoff, "matchings": {}}

        # Run the analysis under each matching strategy (coarse + anonymized).
        for matcher_name, matcher in matchers:
            icsd_strata = []
            for r in pc_rows:
                cls = matcher(r["formula"])
                if cls is None:
                    continue
                icsd_strata.append((cls, int(r["is_in_basin"])))
                if matcher_name == "coarse":  # avoid duplicating per-record rows
                    all_records.append({
                        "cutoff": cutoff, "series": "ICSD", "id": r["cif_id"],
                        "formula": r["formula"],
                        "stratum_coarse": cls,
                        "stratum_anon": anonymized_formula(r["formula"]) or "",
                        "in_basin": int(r["is_in_basin"]),
                    })
            icsd_strata_set = {s for s, _ in icsd_strata}
            ki_all = sum(x for _, x in icsd_strata); ni_all = len(icsd_strata)

            sources_block: dict[str, object] = {
                "icsd_unmatched": {
                    "k": ki_all, "n": ni_all,
                    "rate": ki_all/ni_all if ni_all else None,
                    "ci95": list(wilson(ki_all, ni_all)),
                },
                "by_source": {},
            }

            for src_name, _ in sources:
                src_rows = classified_by_source[src_name]
                src_strata = []
                for r in src_rows:
                    cls = matcher(r["formula"])
                    if cls is None:
                        continue
                    src_strata.append((cls, int(r["in_basin"])))
                    if matcher_name == "coarse":
                        all_records.append({
                            "cutoff": cutoff, "series": src_name, "id": r["id"],
                            "formula": r["formula"],
                            "stratum_coarse": cls,
                            "stratum_anon": anonymized_formula(r["formula"]) or "",
                            "in_basin": int(r["in_basin"]),
                        })

                src_strata_set = {s for s, _ in src_strata}
                common = icsd_strata_set & src_strata_set
                k_src_all = sum(x for _, x in src_strata); n_src_all = len(src_strata)
                k_src, n_src, r_src, lo_src, hi_src = matched_rate(src_strata, common)
                k_icsd, n_icsd, r_icsd, lo_icsd, hi_icsd = matched_rate(icsd_strata, common)

                sources_block["by_source"][src_name] = {
                    "unmatched": {
                        "k": k_src_all, "n": n_src_all,
                        "rate": k_src_all/n_src_all if n_src_all else None,
                        "ci95": list(wilson(k_src_all, n_src_all)),
                    },
                    "matched": {"k": k_src, "n": n_src, "rate": r_src, "ci95": [lo_src, hi_src]},
                    "icsd_matched_to_source": {"k": k_icsd, "n": n_icsd, "rate": r_icsd, "ci95": [lo_icsd, hi_icsd]},
                    "n_strata_common": len(common),
                }
                gap = (r_icsd - r_src) if (r_icsd is not None and not (isinstance(r_icsd, float) and math.isnan(r_icsd)) and r_src is not None and not (isinstance(r_src, float) and math.isnan(r_src))) else float("nan")
                print(f"  [{matcher_name}] {src_name}: common {len(common)}, "
                      f"ICSD {k_icsd}/{n_icsd}={r_icsd:.3f}  AI {k_src}/{n_src}={r_src:.3f}  gap {gap:+.3f}", flush=True)

            cutoff_block["matchings"][matcher_name] = sources_block

        cutoff_results.append(cutoff_block)

    # Write summary
    Path(args.output_summary).write_text(json.dumps({"cutoffs": cutoff_results}, indent=2))
    print(f"\nwrote summary to {args.output_summary}")

    # Write records CSV with both stratum descriptors so the user can audit.
    out_records = Path(args.output_records)
    out_records.parent.mkdir(parents=True, exist_ok=True)
    with out_records.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cutoff", "series", "id", "formula",
            "stratum_coarse", "stratum_anon", "in_basin",
        ])
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)
    print(f"wrote per-record CSV ({len(all_records)} rows) to {out_records}")

    # Optional inline rendering. The canonical path is to render the
    # figure separately via scripts/make_fig_composition_matched_ai.py
    # reading the JSON we just wrote, so styling tweaks do not require
    # re-running the Wilson-CI computation. Kept here only for backward
    # compatibility with older invocations that pass --output-figure.
    if args.output_figure:
        from make_fig_composition_matched_ai import render
        out_fig = Path(args.output_figure)
        # Re-load the JSON we just wrote so the inline render path uses
        # exactly the same code path as the standalone figure script.
        summary = json.loads(Path(args.output_summary).read_text())
        render(summary, out_fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
