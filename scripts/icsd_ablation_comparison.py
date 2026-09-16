#!/usr/bin/env python3
"""Compare the production Louvain partition against the paper-text-faithful
ablation partition on the same ICSD cohort.

Inputs:
  --production-csv  CSV with columns icsd_id,year,community (the canonical
                    partition; the default points at the superseded June 2026 labels3 file)
  --ablation-csv    CSV with the same columns from the ablation pipeline
  --icsd-index      ICSD_index.csv (for sym_group lookup; needed for the
                    "mean space groups per top-10 community" metric)
  --report-out      Output Markdown report path

Computed comparisons (intersection of icsd_ids only, outliers excluded
where the metric requires it):

  Partition agreement:
    - Adjusted Rand Index (sklearn)
    - Normalized Mutual Information (sklearn)

  Per-community purity (fraction of a production community that lands
  in a single alternative community):
    - Production community 6425 (cuprate, expected n=378)
    - Production community 160  (CMR manganite, expected n=571)
    - Top-10 production communities by size

  Headline-number recompute under the ablation partition:
    - Decade-level community-birth ratio (1930s -> 2010s)
      Compared to production's 40.2% -> 2.6% collapse.
    - Mean number of distinct space groups in the top-10 communities by
      size. Compared to production's 44.2.
    - Reduced-formula stepping-stone rate: of unique formulas with at
      least one assignment, what fraction had their first ICSD entry
      land in a community with strictly earlier members? Compared to
      production's 82.9%.
    - Renaissance survey: top-16 communities by step-change score (same
      scoring as scripts/analyze_renaissance_survey.py:best_event_year),
      with their event-year, fold-change, n_post, and the top-3 most
      common reduced formulas among members. Manual cross-reference to
      the 16 documented field-defining events listed in the manuscript
      is left to the reader; the report surfaces the data.

Outputs a Markdown report and a JSON sidecar with the same numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REPO = Path(__file__).resolve().parent.parent

DEFAULT_PRODUCTION = REPO / "notes/icsd_community_assignments/community_assignments_labels3.csv"
DEFAULT_INDEX = REPO / "notes/ICSD_index.csv"  # may not exist locally
DEFAULT_FIRST_REPORT_DIRS = [
    REPO / f"notes/icsd_first_report_formulas/split_{year}" for year in (1980, 1990, 2000, 2010)
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare production vs paper-text-ablation community partitions.")
    p.add_argument("--production-csv", default=str(DEFAULT_PRODUCTION))
    p.add_argument("--ablation-csv", required=True)
    p.add_argument(
        "--icsd-index",
        default=str(DEFAULT_INDEX),
        help="ICSD_index.csv for sym_group lookup; if absent, the SG-per-community metric is skipped.",
    )
    p.add_argument(
        "--first-report-dirs",
        nargs="*",
        default=[str(d) for d in DEFAULT_FIRST_REPORT_DIRS],
        help="Directories with first_report_formulas.csv (one per cutoff year).",
    )
    p.add_argument("--report-out", required=True)
    p.add_argument("--json-out", default=None, help="Defaults to report-out with .json suffix")
    return p.parse_args()


def load_assignments(path: Path) -> dict[int, tuple[int | None, int]]:
    """Returns icsd_id -> (year, community)."""
    out: dict[int, tuple[int | None, int]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                icsd_id = int(row["icsd_id"])
            except (KeyError, ValueError):
                continue
            year_raw = row.get("year") or ""
            try:
                year = int(year_raw) if year_raw else None
            except ValueError:
                year = None
            comm_raw = row.get("community") or row.get("cluster") or ""
            try:
                comm = int(comm_raw)
            except ValueError:
                continue
            out[icsd_id] = (year, comm)
    return out


def load_sym_group_lookup(index_csv: Path) -> dict[int, int]:
    """Returns icsd_id -> space group number from the ICSD index CSV."""
    if not index_csv.exists():
        return {}
    out: dict[int, int] = {}
    with index_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = (row.get("cif_names") or row.get("ICSDid") or "").strip()
            if not raw_id:
                continue
            try:
                icsd_id = int(float(raw_id))
            except ValueError:
                continue
            sg_raw = (row.get("sym_group") or "").strip()
            try:
                sg = int(float(sg_raw))
            except ValueError:
                continue
            out[icsd_id] = sg
    return out


def load_first_report_formulas(dirs: list[Path]) -> dict[int, tuple[str, int]]:
    """cif_id -> (reduced_formula, year). Pre-computed in production runs."""
    out: dict[int, tuple[str, int]] = {}
    for d in dirs:
        path = d / "first_report_formulas.csv"
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    cif_id = int(row.get("cif_id") or 0)
                except ValueError:
                    continue
                if cif_id <= 0:
                    continue
                formula = (row.get("reduced_formula") or "").strip()
                try:
                    year = int(row.get("year") or 0)
                except ValueError:
                    continue
                if formula and year > 0 and cif_id not in out:
                    out[cif_id] = (formula, year)
    return out


def aligned_label_arrays(
    prod: dict[int, tuple[int | None, int]],
    abl: dict[int, tuple[int | None, int]],
    drop_outliers: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    common = sorted(set(prod) & set(abl))
    if drop_outliers:
        common = [k for k in common if prod[k][1] >= 0 and abl[k][1] >= 0]
    p = np.array([prod[k][1] for k in common], dtype=int)
    a = np.array([abl[k][1] for k in common], dtype=int)
    return p, a, common


def per_community_purity(
    prod: dict[int, tuple[int | None, int]],
    abl: dict[int, tuple[int | None, int]],
    target_prod_community: int,
) -> dict:
    """Of all production-community-X members that survive in the ablation,
    what is the largest count in any single ablation community?"""
    members = [k for k, (_, c) in prod.items() if c == target_prod_community]
    survived = [k for k in members if k in abl and abl[k][1] >= 0]
    abl_labels = [abl[k][1] for k in survived]
    counts = Counter(abl_labels)
    if not counts:
        return {
            "production_community": target_prod_community,
            "production_size": len(members),
            "survived_in_ablation": 0,
            "purity": None,
            "modal_ablation_community": None,
            "modal_count": 0,
        }
    modal_comm, modal_count = counts.most_common(1)[0]
    return {
        "production_community": int(target_prod_community),
        "production_size": int(len(members)),
        "survived_in_ablation": int(len(survived)),
        "purity": float(modal_count) / float(len(survived)) if survived else None,
        "modal_ablation_community": int(modal_comm),
        "modal_count": int(modal_count),
    }


def decade_birth_ratios(assignments: dict[int, tuple[int | None, int]]) -> dict[str, float]:
    cluster_birth: dict[int, int] = {}
    for k, (y, c) in assignments.items():
        if c < 0 or y is None:
            continue
        if c not in cluster_birth or y < cluster_birth[c]:
            cluster_birth[c] = y
    by_decade_total: Counter = Counter()
    by_decade_birth: Counter = Counter()
    for k, (y, c) in assignments.items():
        if y is None or y < 1900 or y > 2025:
            continue
        decade = f"{(y // 10) * 10}s"
        by_decade_total[decade] += 1
        if c >= 0 and cluster_birth.get(c) == y:
            by_decade_birth[decade] += 1
    out = {}
    for decade in sorted(by_decade_total):
        n = by_decade_total[decade]
        out[decade] = (by_decade_birth[decade] / n) if n else 0.0
    return out


def mean_space_groups_top10(
    assignments: dict[int, tuple[int | None, int]],
    sg_lookup: dict[int, int],
) -> tuple[float | None, list[tuple[int, int, int]]]:
    if not sg_lookup:
        return None, []
    sizes: Counter = Counter()
    for _, (_, c) in assignments.items():
        if c >= 0:
            sizes[c] += 1
    top10 = [c for c, _ in sizes.most_common(10)]
    per_comm: list[tuple[int, int, int]] = []  # (community_id, size, n_distinct_sg)
    distinct_counts: list[int] = []
    for c in top10:
        members = [k for k, (_, cc) in assignments.items() if cc == c]
        sgs = {sg_lookup[k] for k in members if k in sg_lookup}
        per_comm.append((int(c), int(len(members)), int(len(sgs))))
        distinct_counts.append(len(sgs))
    if not distinct_counts:
        return None, per_comm
    return float(np.mean(distinct_counts)), per_comm


def formula_attachment_rate(
    assignments: dict[int, tuple[int | None, int]],
    first_report: dict[int, tuple[object, int]],
) -> dict | None:
    """Nominal-composition stepping-stone rate.

    For each unique composition identity supplied by the caller, take the
    earliest-year ICSD entry with that identity. Classify its community
    membership:
      - 'pre-existing' if the community has a strictly earlier birth year
      - 'community-birth' if the community's earliest member is this entry
      - 'precedes-community' if this entry's year < community birth year
        (cannot happen by definition unless the partition was imported
        independently of year ordering; surfaced as a sanity-check).
    """
    if not first_report:
        return None
    cluster_birth: dict[int, int] = {}
    for k, (y, c) in assignments.items():
        if c < 0 or y is None:
            continue
        if c not in cluster_birth or y < cluster_birth[c]:
            cluster_birth[c] = y

    # First (year-min) ICSD entry per caller-defined composition identity.
    by_formula: dict[object, tuple[int, int]] = {}
    for cif_id, (identity, year) in first_report.items():
        prev = by_formula.get(identity)
        if prev is None or year < prev[0]:
            by_formula[identity] = (year, cif_id)

    n_total = 0
    n_pre_existing = 0
    n_birth = 0
    n_precedes = 0
    n_unclassifiable = 0
    for _, (year, cif_id) in by_formula.items():
        if cif_id not in assignments:
            n_unclassifiable += 1
            continue
        y_assign, c = assignments[cif_id]
        if c < 0:
            n_unclassifiable += 1
            continue
        n_total += 1
        birth = cluster_birth.get(c)
        if birth is None:
            n_unclassifiable += 1
            n_total -= 1
            continue
        if year > birth:
            n_pre_existing += 1
        elif year == birth:
            n_birth += 1
        else:
            n_precedes += 1
    return {
        "n_classifiable": int(n_total),
        "n_pre_existing": int(n_pre_existing),
        "n_community_birth": int(n_birth),
        "n_precedes_community": int(n_precedes),
        "n_unclassifiable": int(n_unclassifiable),
        "pre_existing_rate": (n_pre_existing / n_total) if n_total else None,
        "community_birth_rate": (n_birth / n_total) if n_total else None,
        "precedes_rate": (n_precedes / n_total) if n_total else None,
    }


def best_event_year(year_hist: Counter, event_years: list[int], window: int) -> dict:
    best = None
    for ey in event_years:
        n_pre = sum(c for y, c in year_hist.items() if ey - window < y <= ey)
        n_post = sum(c for y, c in year_hist.items() if ey < y <= ey + window)
        if n_pre == 0:
            score = float(n_post)
            fold = float("inf") if n_post > 0 else 0.0
        else:
            fold = (n_post / window) / (n_pre / window)
            score = fold * n_post
        cand = {"event_year": ey, "n_pre": n_pre, "n_post": n_post, "fold": fold, "score": score}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best or {"event_year": None, "n_pre": 0, "n_post": 0, "fold": 0.0, "score": 0.0}


def renaissance_survey(
    assignments: dict[int, tuple[int | None, int]],
    first_report: dict[int, tuple[str, int]],
    n_min: int = 50,
    window: int = 10,
    top_k: int = 16,
) -> list[dict]:
    members_by_comm: dict[int, list[tuple[int, int]]] = defaultdict(list)  # comm -> [(year, cif_id)]
    for k, (y, c) in assignments.items():
        if c < 0 or y is None or y < 1900 or y > 2025:
            continue
        members_by_comm[c].append((y, k))
    out = []
    event_years = list(range(1970, 2011))
    for comm, members in members_by_comm.items():
        if len(members) < n_min:
            continue
        year_hist = Counter(y for y, _ in members)
        best = best_event_year(year_hist, event_years, window)
        formula_counter: Counter = Counter()
        for _, cif_id in members:
            fr = first_report.get(cif_id)
            if fr:
                formula_counter[fr[0]] += 1
        top_formulas = [f for f, _ in formula_counter.most_common(3)]
        out.append({
            "community": int(comm),
            "size": int(len(members)),
            **best,
            "top_formulas": top_formulas,
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:top_k]


def write_report(path: Path, report: dict) -> None:
    L: list[str] = []
    L.append("# Methods-robustness ablation: production vs paper-text-faithful partition\n")
    L.append("Comparing the canonical Louvain partition (Magpie-7-dim hand-rolled chem,\n"
            "additive WL, mean+max+var pool) against a re-implementation of the metric\n"
            "as the manuscript Methods text actually describes (Magpie-22 elemental,\n"
            "concat-mean-std WL, mean+std pool).\n")
    L.append(f"- Production CSV: `{report['inputs']['production_csv']}`")
    L.append(f"- Ablation  CSV: `{report['inputs']['ablation_csv']}`")
    L.append(f"- ICSD ids in both: **{report['n_common_total']}**")
    L.append(f"- After excluding outliers in either: **{report['n_common_no_outliers']}**\n")

    L.append("## Partition agreement (sklearn)\n")
    L.append(f"- Adjusted Rand Index: **{report['agreement']['ARI']:.4f}**")
    L.append(f"- Normalized Mutual Information: **{report['agreement']['NMI']:.4f}**\n")

    L.append("## Per-community purity\n")
    L.append("Of each production community's members that survive (non-outlier) in the\n"
            "ablation, what fraction land in the single most common alternative\n"
            "community? Purity = 1.0 means the production community maps cleanly to\n"
            "one alternative community; 0.5 means the production community is split.\n")
    L.append("| Production community | Prod size | Survived | Purity | Modal alt-comm | Modal count |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for row in report["per_community_purity"]:
        purity = f"{row['purity']:.3f}" if row["purity"] is not None else "—"
        L.append(f"| {row['production_community']} | {row['production_size']} | "
                 f"{row['survived_in_ablation']} | {purity} | "
                 f"{row['modal_ablation_community']} | {row['modal_count']} |")
    L.append("")

    L.append("## Decade-level community-birth ratio\n")
    L.append("Production reports collapse from 40.2% in the 1930s to 2.6% in the 2010s.\n"
            "The ablation values below are computed under exactly the same definition\n"
            "(community-birth = entry whose year equals its community's earliest year).\n")
    L.append("| Decade | Production | Ablation |")
    L.append("|---|---:|---:|")
    decades = sorted(set(report["decade_birth"]["production"]) | set(report["decade_birth"]["ablation"]))
    for d in decades:
        p = report["decade_birth"]["production"].get(d)
        a = report["decade_birth"]["ablation"].get(d)
        L.append(f"| {d} | {p:.3f} | {a:.3f} |" if p is not None and a is not None
                 else f"| {d} | {('%.3f' % p) if p else '—'} | {('%.3f' % a) if a else '—'} |")
    L.append("")

    L.append("## Mean distinct space groups per top-10 community\n")
    L.append("Production reports 44.2 (Methods, main text). Larger value = the metric\n"
            "absorbs more crystallographic relabellings into a single basin.\n")
    if report["sg_top10"]["production"]["mean"] is not None:
        L.append(f"- Production: **{report['sg_top10']['production']['mean']:.2f}** distinct SGs / top-10 community")
    else:
        L.append("- Production: SG lookup unavailable (ICSD index CSV not on disk).")
    if report["sg_top10"]["ablation"]["mean"] is not None:
        L.append(f"- Ablation:  **{report['sg_top10']['ablation']['mean']:.2f}**")
    else:
        L.append("- Ablation:  SG lookup unavailable (ICSD index CSV not on disk).")
    L.append("")

    L.append("## Reduced-formula stepping-stone rate\n")
    L.append("Production reports 82.9% of newly reported reduced formulas enter\n"
            "pre-existing structural communities, vs 16.2% community-birth and 1.0%\n"
            "preceding-community.\n")
    if report["formula_attachment"]["production"] is not None:
        for label, key in [("pre-existing rate", "pre_existing_rate"),
                           ("community-birth rate", "community_birth_rate"),
                           ("precedes-community rate", "precedes_rate")]:
            p = report["formula_attachment"]["production"][key]
            a = report["formula_attachment"]["ablation"][key]
            L.append(f"- {label}: production **{p:.3f}**, ablation **{a:.3f}**")
    else:
        L.append("- First-report-formula CSVs missing on disk; skipped.\n")
    L.append("")

    L.append("## Renaissance survey: top-16 communities by step-change score\n")
    L.append("Same scoring as `scripts/analyze_renaissance_survey.py:best_event_year`.\n"
            "Manuscript reports 9 of the top-16 production communities map to documented\n"
            "field-defining renaissances (Sm-Fe-N, CMR, SOFC, MAX phases, NaCoO2, double\n"
            "perovskites, Li-ion cathodes, dilute magnetic semiconductors, RP cuprates).\n"
            "The table below is the same survey run on the ablation partition.\n")
    L.append("\n### Production (re-derived for sanity check)\n")
    L.append("| Rank | Community | Size | Event yr | n_pre | n_post | Fold | Top-3 formulas |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(report["renaissance"]["production"], 1):
        fold = "inf" if r["fold"] == float("inf") else f"{r['fold']:.2f}"
        L.append(f"| {i} | {r['community']} | {r['size']} | {r['event_year']} | "
                 f"{r['n_pre']} | {r['n_post']} | {fold} | {', '.join(r['top_formulas']) or '—'} |")
    L.append("\n### Ablation (paper-text faithful)\n")
    L.append("| Rank | Community | Size | Event yr | n_pre | n_post | Fold | Top-3 formulas |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(report["renaissance"]["ablation"], 1):
        fold = "inf" if r["fold"] == float("inf") else f"{r['fold']:.2f}"
        L.append(f"| {i} | {r['community']} | {r['size']} | {r['event_year']} | "
                 f"{r['n_pre']} | {r['n_post']} | {fold} | {', '.join(r['top_formulas']) or '—'} |")

    L.append("\n## How to read this report\n")
    L.append("- ARI > 0.7 and NMI > 0.85 would mean the two partitions agree closely;\n"
            "  the headline numbers should match within ~1-2 pp.\n"
            "- Per-community purity > 0.7 for the cuprate / CMR / top-10 communities\n"
            "  means those basins are stable to the featurization swap.\n"
            "- A decade-birth-ratio collapse from ~40% (1930s) to ~3% (2010s) on the\n"
            "  ablation, similar to production, supports the headline claim.\n"
            "- The renaissance-top-16 list under the ablation should overlap substantially\n"
            "  with the production list (same well-known renaissance communities).\n")
    path.write_text("\n".join(L))


def main() -> int:
    args = parse_args()
    report_out = Path(args.report_out)
    json_out = Path(args.json_out) if args.json_out else report_out.with_suffix(".json")

    prod = load_assignments(Path(args.production_csv))
    abl = load_assignments(Path(args.ablation_csv))
    sg_lookup = load_sym_group_lookup(Path(args.icsd_index))
    first_report = load_first_report_formulas([Path(d) for d in args.first_report_dirs])

    p_arr, a_arr, common = aligned_label_arrays(prod, abl, drop_outliers=True)
    common_total = len(set(prod) & set(abl))

    ari = float(adjusted_rand_score(p_arr, a_arr)) if len(p_arr) else None
    nmi = float(normalized_mutual_info_score(p_arr, a_arr)) if len(p_arr) else None

    # Per-community purity: 6425, 160, plus top-10 production communities by size.
    prod_sizes = Counter(c for _, c in prod.values() if c >= 0)
    top10_prod = [c for c, _ in prod_sizes.most_common(10)]
    target_communities = sorted(set([6425, 160, *top10_prod]))
    purity = [per_community_purity(prod, abl, c) for c in target_communities]

    decade_prod = decade_birth_ratios(prod)
    decade_abl = decade_birth_ratios(abl)

    sg_prod_mean, sg_prod_per_comm = mean_space_groups_top10(prod, sg_lookup)
    sg_abl_mean, sg_abl_per_comm = mean_space_groups_top10(abl, sg_lookup)

    fa_prod = formula_attachment_rate(prod, first_report)
    fa_abl = formula_attachment_rate(abl, first_report)

    ren_prod = renaissance_survey(prod, first_report)
    ren_abl = renaissance_survey(abl, first_report)

    report = {
        "inputs": {
            "production_csv": str(args.production_csv),
            "ablation_csv": str(args.ablation_csv),
            "icsd_index": str(args.icsd_index),
        },
        "n_common_total": common_total,
        "n_common_no_outliers": len(common),
        "agreement": {"ARI": ari, "NMI": nmi},
        "per_community_purity": purity,
        "decade_birth": {"production": decade_prod, "ablation": decade_abl},
        "sg_top10": {
            "production": {"mean": sg_prod_mean, "per_community": sg_prod_per_comm},
            "ablation":   {"mean": sg_abl_mean,  "per_community": sg_abl_per_comm},
        },
        "formula_attachment": {"production": fa_prod, "ablation": fa_abl},
        "renaissance": {"production": ren_prod, "ablation": ren_abl},
    }

    json_out.write_text(json.dumps(report, indent=2, default=lambda o: None if o == float("inf") else o))
    write_report(report_out, report)
    print(f"Wrote report -> {report_out}")
    print(f"Wrote JSON   -> {json_out}")
    if ari is not None:
        print(f"ARI = {ari:.4f}, NMI = {nmi:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
