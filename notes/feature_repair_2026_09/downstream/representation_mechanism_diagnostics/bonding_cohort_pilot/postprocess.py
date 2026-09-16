#!/usr/bin/env python3
"""Summarize the frozen matched bonding-cohort pilot.

This is a matched-sample analysis, not an estimate of source-wide prevalence.
Uncertainty is obtained by resampling connected components of matched pairs so
that reuse of either ICSD structure cannot create pseudoreplication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")
COMPARATORS = SOURCES[1:]
FAMILIES = ("joint", "chemistry", "geometry")
KINDS = ("actual", "delta")
N_BOOTSTRAPS = 2000
SEED = 20260909


def finite_float(value: object) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"Expected finite value, got {value!r}")
    return x


def percentile_ci(values: Iterable[float]) -> list[float] | None:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return [float(x) for x in np.percentile(a, [2.5, 97.5])]


def median(rows: list[dict[str, str]], field: str) -> float | None:
    values = [finite_float(r[field]) for r in rows if r.get(field, "") != ""]
    return float(np.median(values)) if values else None


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def pair_clusters(pairs: list[dict[str, object]]) -> list[list[int]]:
    """Connect pairs sharing a GNoME ID or any query/reference comparator ICSD ID."""
    uf = UnionFind(len(pairs))
    owners: dict[tuple[str, str], int] = {}
    for i, pair in enumerate(pairs):
        g = pair["gnome"]
        o = pair["other"]
        tokens = {("gnome", str(pair["matched_gnome_id"]))}
        for row in (g, o):
            for field in ("icsd_reference", "icsd_comparator"):
                value = row.get(field, "")
                if value != "":
                    tokens.add(("icsd", str(value)))
        for token in tokens:
            if token in owners:
                uf.union(i, owners[token])
            else:
                owners[token] = i
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(pairs)):
        groups[uf.find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def resample_indices(
    clusters: list[list[int]], rng: np.random.Generator
) -> np.ndarray:
    chosen = rng.integers(0, len(clusters), size=len(clusters))
    return np.fromiter(
        (i for cluster_index in chosen for i in clusters[int(cluster_index)]),
        dtype=int,
    )


def deterministic_rng(label: str) -> np.random.Generator:
    offset = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    return np.random.default_rng((SEED + offset) % (2**64))


def make_pairs(rows: list[dict[str, str]], source: str) -> list[dict[str, object]]:
    gnome = {r["matched_gnome_id"]: r for r in rows if r["source"] == "gnome"}
    other: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        if row["source"] != source:
            continue
        gid = row["matched_gnome_id"]
        if gid in other:
            duplicates.append(gid)
        other[gid] = row
    if duplicates:
        raise ValueError(f"Duplicate {source} matches for GNoME IDs: {duplicates[:5]}")
    return [
        {"matched_gnome_id": gid, "gnome": gnome[gid], "other": row}
        for gid, row in other.items()
        if gid in gnome
    ]


def metric_comparison(
    pairs: list[dict[str, object]],
    clusters: list[list[int]],
    field: str,
    source: str,
) -> dict[str, object]:
    g = np.array([finite_float(p["gnome"][field]) for p in pairs])
    o = np.array([finite_float(p["other"][field]) for p in pairs])
    paired = g - o
    observed_dom = float(np.median(g) - np.median(o))
    observed_paired = float(np.median(paired))
    rng = deterministic_rng(f"metric|{source}|{field}")
    boot_dom: list[float] = []
    boot_paired: list[float] = []
    for _ in range(N_BOOTSTRAPS):
        ix = resample_indices(clusters, rng)
        boot_dom.append(float(np.median(g[ix]) - np.median(o[ix])))
        boot_paired.append(float(np.median(paired[ix])))
    return {
        "gnome_median": float(np.median(g)),
        "comparator_median": float(np.median(o)),
        "gnome_minus_comparator_difference_of_medians": observed_dom,
        "difference_of_medians_ci95": percentile_ci(boot_dom),
        "median_paired_difference": observed_paired,
        "median_paired_difference_ci95": percentile_ci(boot_paired),
    }


def regression_fit(y: np.ndarray, x: np.ndarray) -> dict[str, object]:
    covariate_names = [
        "difference_log1p_amd_distance",
        "difference_n_elements",
        "difference_log_n_sites",
    ]
    retained = [i for i in range(x.shape[1]) if not np.all(x[:, i] == 0)]
    dropped = [i for i in range(x.shape[1]) if i not in retained]
    design = np.column_stack([np.ones(len(y)), x[:, retained]])
    rank = int(np.linalg.matrix_rank(design))
    coef, _, _, singular = np.linalg.lstsq(design, y, rcond=None)
    raw_condition = float(np.inf if singular[-1] == 0 else singular[0] / singular[-1])
    coefficients: dict[str, float | None] = {
        "intercept_geometry_adjusted_source_contrast": float(coef[0]),
        **{name: None for name in covariate_names},
    }
    for coefficient, column in zip(coef[1:], retained):
        coefficients[covariate_names[column]] = float(coefficient)
    return {
        "coefficients": coefficients,
        "retained_covariates": [covariate_names[i] for i in retained],
        "dropped_identically_zero_covariates": [covariate_names[i] for i in dropped],
        "n": int(len(y)),
        "design_columns": int(design.shape[1]),
        "rank": rank,
        "full_rank": rank == design.shape[1],
        "condition_number": raw_condition if math.isfinite(raw_condition) else None,
    }


def adjusted_regression(
    pairs: list[dict[str, object]], clusters: list[list[int]], source: str
) -> dict[str, object]:
    def values(p: dict[str, object]) -> tuple[float, list[float]]:
        g, o = p["gnome"], p["other"]
        y = finite_float(g["joint_actual"]) - finite_float(o["joint_actual"])
        x = [
            math.log1p(finite_float(g["amd_distance"]))
            - math.log1p(finite_float(o["amd_distance"])),
            finite_float(g["n_elements"]) - finite_float(o["n_elements"]),
            math.log(finite_float(g["n_sites"]))
            - math.log(finite_float(o["n_sites"])),
        ]
        return y, x

    unpacked = [values(p) for p in pairs]
    y = np.asarray([v[0] for v in unpacked])
    x = np.asarray([v[1] for v in unpacked])
    result = regression_fit(y, x)
    rng = deterministic_rng(f"regression|{source}")
    intercepts: list[float] = []
    full_rank = 0
    for _ in range(N_BOOTSTRAPS):
        ix = resample_indices(clusters, rng)
        fit = regression_fit(y[ix], x[ix])
        if fit["full_rank"]:
            full_rank += 1
            intercepts.append(
                fit["coefficients"]["intercept_geometry_adjusted_source_contrast"]
            )
    result.update(
        {
            "intercept_ci95_full_rank_bootstraps": percentile_ci(intercepts),
            "bootstrap_full_rank_replicates": full_rank,
            "bootstrap_replicates": N_BOOTSTRAPS,
            "bootstrap_design_note": (
                "Each bootstrap fit also drops covariates that are identically zero in "
                "that draw before checking design rank; other rank deficiency is refused."
            ),
            "interpretation": (
                "The intercept is the fitted GNoME-minus-comparator joint-score "
                "contrast when the three observed pair differences are zero. It is "
                "a limited imbalance sensitivity analysis, not causal mediation."
            ),
        }
    )
    return result


def stratum_summaries(pairs: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    strata = sorted({str(p["gnome"].get("cell", "unlabeled")) for p in pairs})
    for stratum in strata:
        pp = [p for p in pairs if str(p["gnome"].get("cell", "unlabeled")) == stratum]
        item: dict[str, object] = {"n_pairs": len(pp)}
        for family in FAMILIES:
            for kind in KINDS:
                field = f"{family}_{kind}"
                diffs = [
                    finite_float(p["gnome"][field]) - finite_float(p["other"][field])
                    for p in pp
                ]
                item[field] = {
                    "gnome_median": float(
                        np.median([finite_float(p["gnome"][field]) for p in pp])
                    ),
                    "comparator_median": float(
                        np.median([finite_float(p["other"][field]) for p in pp])
                    ),
                    "median_paired_difference": float(np.median(diffs)),
                }
        result[stratum] = item
    return result


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    start = 0
    while start < len(a):
        end = start + 1
        while end < len(a) and sorted_a[end] == sorted_a[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = rankdata(np.asarray(x)), rankdata(np.asarray(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def map_associations(
    rows: list[dict[str, str]], sample_path: Path
) -> dict[str, object]:
    sample = json.loads(sample_path.read_text())
    frozen = {(r["source"], r["material_id"]): r for r in sample}
    result: dict[str, object] = {}
    for source in SOURCES:
        source_rows = [r for r in rows if r["source"] == source]
        names = sorted(
            {
                name
                for row in source_rows
                for name in frozen.get((source, row["material_id"]), {}).get(
                    "representations", {}
                )
            }
        )
        maps: dict[str, object] = {}
        for name in names:
            ratios: list[float] = []
            scores: list[float] = []
            excluded_zero_radius = 0
            for row in source_rows:
                rep = frozen.get((source, row["material_id"]), {}).get(
                    "representations", {}
                ).get(name)
                if not rep:
                    continue
                radius = finite_float(rep["community_threshold_p95"])
                if radius <= 0:
                    excluded_zero_radius += 1
                    continue
                ratios.append(finite_float(rep["nearest_centroid_distance"]) / radius)
                scores.append(finite_float(row["joint_actual"]))
            maps[name] = {
                "n_positive_radius": len(ratios),
                "n_excluded_nonpositive_radius": excluded_zero_radius,
                "median_distance_over_radius": (
                    float(np.median(ratios)) if ratios else None
                ),
                "spearman_rho_joint_actual_vs_distance_over_radius": spearman(
                    ratios, scores
                ),
            }
        result[source] = maps
    return result


def cohort_summaries(rows: list[dict[str, str]]) -> dict[str, object]:
    result: dict[str, object] = {}
    fields = [f"{family}_{kind}" for family in FAMILIES for kind in KINDS]
    fields += [f"baseline_{field}" for field in fields]
    for source in SOURCES:
        rr = [r for r in rows if r["source"] == source]
        result[source] = {
            "n_analyzed": len(rr),
            "matched_gnome_stratum_counts": dict(
                sorted(Counter(r.get("cell", "unlabeled") for r in rr).items())
            ),
            "medians": {
                field: median(rr, field)
                for field in fields
                if rr and field in rr[0]
            },
            "median_amd_distance": median(rr, "amd_distance"),
            "median_icsd_baseline_amd_distance": median(
                rr, "icsd_baseline_amd_distance"
            ),
        }
    return result


def comparison_summary(
    pairs: list[dict[str, object]], source: str
) -> dict[str, object]:
    clusters = pair_clusters(pairs)
    metric_fields = [f"{family}_{kind}" for family in FAMILIES for kind in KINDS]
    metrics = {
        field: metric_comparison(pairs, clusters, field, source)
        for field in metric_fields
    }
    amd = metric_comparison(pairs, clusters, "amd_distance", source)
    return {
        "n_pairs": len(pairs),
        "n_connected_components": len(clusters),
        "connected_component_sizes": sorted((len(x) for x in clusters), reverse=True),
        "bootstrap_inference_limited": len(clusters) < 2,
        "cluster_tokens": (
            "matched GNoME ID plus both rows' icsd_reference and "
            "icsd_comparator IDs"
        ),
        "metrics": metrics,
        "amd_reference_match_imbalance": amd,
        "matched_gnome_strata": stratum_summaries(pairs),
        "joint_actual_geometry_imbalance_sensitivity": adjusted_regression(
            pairs, clusters, source
        ),
    }


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}g}"


def ci(value: object) -> str:
    if value is None:
        return "[NA, NA]"
    return f"[{fmt(value[0])}, {fmt(value[1])}]"


def render_readme(summary: dict[str, object]) -> str:
    lines = [
        "# Bonding-cohort pilot results",
        "",
        "This analysis compares each available chemistry/size-matched computed-library "
        "record with its frozen GNoME partner. Results describe this matched pilot sample; "
        "they do not estimate prevalence in any full source library.",
        "",
        "## Pair-specific comparisons",
        "",
        "The primary paired quantity below is the median within-pair GNoME-minus-comparator "
        "difference. The separate difference of medians is retained in "
        "`bootstrap_summary.json`. Positive values mean the GNoME score is larger.",
        "",
        "| Comparator | n pairs | components | Outcome | GNoME median | Comparator median | Median paired difference (95% cluster-bootstrap CI) |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for source, comp in summary["pair_comparisons"].items():
        for field, stat in comp["metrics"].items():
            lines.append(
                f"| {source} | {comp['n_pairs']} | {comp['n_connected_components']} | "
                f"{field} | {fmt(stat['gnome_median'])} | "
                f"{fmt(stat['comparator_median'])} | "
                f"{fmt(stat['median_paired_difference'])} "
                f"{ci(stat['median_paired_difference_ci95'])} |"
            )
    lines += [
        "",
        "Cluster bootstraps resample connected components of comparison pairs. Pairs are "
        "connected when they share a matched GNoME ID or either row's ICSD reference or "
        "ICSD comparator ID. This conservative rule accounts for reused ICSD structures. "
        "If a comparison has only one component, its bootstrap interval is degenerate "
        "and provides no independent-cluster uncertainty information.",
        "",
        "## AMD reference-match imbalance",
        "",
        "| Comparator | GNoME median AMD | Comparator median AMD | Median paired difference (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for source, comp in summary["pair_comparisons"].items():
        stat = comp["amd_reference_match_imbalance"]
        lines.append(
            f"| {source} | {fmt(stat['gnome_median'])} | "
            f"{fmt(stat['comparator_median'])} | "
            f"{fmt(stat['median_paired_difference'])} "
            f"{ci(stat['median_paired_difference_ci95'])} |"
        )
    lines += [
        "",
        "## Limited geometry-imbalance sensitivity",
        "",
        "For each comparator, ordinary least squares regresses the paired `joint_actual` "
        "difference on paired differences in `log1p(amd_distance)`, element count, and "
        "log site count. The intercept is the fitted source contrast when those observed "
        "differences are zero. This is a sensitivity analysis for unequal reference "
        "geometry, not a causal mediation analysis.",
        "",
        "| Comparator | Intercept (95% cluster-bootstrap CI) | rank / columns | condition number | full-rank bootstraps |",
        "|---|---:|---:|---:|---:|",
    ]
    for source, comp in summary["pair_comparisons"].items():
        reg = comp["joint_actual_geometry_imbalance_sensitivity"]
        intercept = reg["coefficients"]["intercept_geometry_adjusted_source_contrast"]
        lines.append(
            f"| {source} | {fmt(intercept)} "
            f"{ci(reg['intercept_ci95_full_rank_bootstraps'])} | "
            f"{reg['rank']} / {reg['design_columns']} | "
            f"{fmt(reg['condition_number'])} | "
            f"{reg['bootstrap_full_rank_replicates']} / {reg['bootstrap_replicates']} |"
        )
    lines += [
        "",
        "Rank deficiency or a large condition number limits the adjusted estimate. "
        "Bootstrap CIs use only full-rank replicates, with their count shown explicitly.",
        "",
        "## GNoME-stratum and map context",
        "",
        "`matched_gnome_strata` in the JSON gives descriptive results within each frozen "
        "GNoME map-disagreement cell. A comparator row's `cell` is its partner's GNoME "
        "stratum, not the comparator's own basin status.",
        "",
        "`map_distance_ratio_associations` reports descriptive Spearman correlations "
        "between `joint_actual` and source-own centroid-distance/radius for each frozen "
        "representation. Records with a nonpositive community radius are excluded. These "
        "associations do not identify a causal mechanism.",
        "",
        "## Design limits",
        "",
        "Matching controls broad chemistry class, element count within one, and site count "
        "within a factor of two, subject to without-replacement availability. It does not "
        "control stoichiometry, symmetry, volume, oxidation state, stability, or "
        "source-specific generation and filtering. Cohort medians and all contrasts are "
        "conditional on successful matching and bonding-feature analysis.",
        "",
    ]
    return "\n".join(lines)


def validate_rows(rows: list[dict[str, str]]) -> None:
    required = {
        "source",
        "material_id",
        "matched_gnome_id",
        "cell",
        "icsd_reference",
        "icsd_comparator",
        "amd_distance",
        "n_elements",
        "n_sites",
        *[f"{family}_{kind}" for family in FAMILIES for kind in KINDS],
    }
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"per_target.csv lacks required columns: {sorted(missing)}")
    unknown = sorted({r["source"] for r in rows} - set(SOURCES))
    if unknown:
        raise ValueError(f"Unexpected sources: {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results" / "per_target.csv")
    parser.add_argument("--sample", type=Path, default=HERE / "sample.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_rows(rows)
    comparisons: dict[str, object] = {}
    unmatched_after_analysis: dict[str, object] = {}
    for source in COMPARATORS:
        source_analyzed = sum(r["source"] == source for r in rows)
        pairs = make_pairs(rows, source)
        missing_gnome = source_analyzed - len(pairs)
        unmatched_after_analysis[source] = {
            "analyzed_comparator_rows": source_analyzed,
            "complete_pairs": len(pairs),
            "comparator_rows_without_analyzed_gnome_partner": missing_gnome,
        }
        if pairs:
            comparisons[source] = comparison_summary(pairs, source)
    summary = {
        "analysis_scope": "frozen chemistry/size-matched pilot sample",
        "not_full_cohort_prevalence": True,
        "bootstrap": {
            "replicates": N_BOOTSTRAPS,
            "seed": SEED,
            "interval": "percentile 95%",
            "unit": "connected component of comparison pairs",
        },
        "cohort_summaries": cohort_summaries(rows),
        "pair_availability": unmatched_after_analysis,
        "pair_comparisons": comparisons,
        "map_distance_ratio_associations": map_associations(rows, args.sample),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "bootstrap_summary.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "README_results.md").write_text(render_readme(summary))
    print(f"Wrote {output}")
    print(f"Wrote {args.output_dir / 'README_results.md'}")


if __name__ == "__main__":
    main()
