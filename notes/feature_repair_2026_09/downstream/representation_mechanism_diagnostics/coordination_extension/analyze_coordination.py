#!/usr/bin/env python3
"""Analyze the frozen full-cohort coordination extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))
from formula_conventions import scale_invariant_formula_key  # noqa: E402

COMPARATORS = ("mattergen", "mp", "jarvis", "alexandria")
METHODS = ("crystalnn", "voronoinn")
ANALYSES = ("all", "unused")


class UnionFind:
    def __init__(self, n: int): self.parent = list(range(n))
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[b] = a


def normalize_source(value: str) -> str:
    x = value.strip().lower().replace("-theoretical", "").replace("_theoretical", "")
    if x not in ("gnome", *COMPARATORS):
        raise ValueError(f"Unknown source {value!r}")
    return x


def finite_float(value: object, label: str = "value") -> float:
    x = float(value)
    if not math.isfinite(x): raise ValueError(f"Non-finite {label}: {value!r}")
    return x


def pair_clusters(pairs: list[dict[str, object]]) -> list[list[int]]:
    """Connect pairs sharing source-qualified formula at either endpoint."""
    uf, owners = UnionFind(len(pairs)), {}
    for i, pair in enumerate(pairs):
        tokens = (("gnome", str(pair["gnome_formula"])),
                  (str(pair["source"]), str(pair["comparator_formula"])))
        for token in tokens:
            if token in owners: uf.union(i, owners[token])
            else: owners[token] = i
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(pairs)): groups[uf.find(i)].append(i)
    return sorted(groups.values(), key=lambda x: x[0])


def resample_clusters(clusters: list[list[int]], rng: np.random.Generator) -> np.ndarray:
    selected = rng.integers(0, len(clusters), len(clusters))
    return np.fromiter((i for c in selected for i in clusters[int(c)]), dtype=int)


def seed_for(seed: int, label: str) -> int:
    offset = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    return (seed + offset) % (2**64)


def summarize_differences(diffs: np.ndarray, clusters: list[list[int]], nboot: int,
                          seed: int, label: str) -> dict[str, object]:
    if not len(diffs): raise ValueError(f"No complete pairs for {label}")
    rng = np.random.default_rng(seed_for(seed, label + "|cluster"))
    rng_pair = np.random.default_rng(seed_for(seed, label + "|ordinary"))
    boot_cluster = np.empty(nboot); boot_pair = np.empty(nboot)
    for b in range(nboot):
        boot_cluster[b] = np.median(diffs[resample_clusters(clusters, rng)])
        boot_pair[b] = np.median(diffs[rng_pair.integers(0, len(diffs), len(diffs))])
    q = np.quantile(diffs, [.1, .25, .5, .75, .9])
    return {
        "n_pairs": int(len(diffs)), "n_formula_components": len(clusters),
        "formula_component_sizes_descending": sorted(map(len, clusters), reverse=True),
        "largest_component_fraction": max(map(len, clusters)) / len(diffs),
        "median_paired_difference": float(np.median(diffs)),
        "mean_paired_difference": float(np.mean(diffs)),
        "fraction_gnome_higher": float(np.mean(diffs > 1e-10)),
        "fraction_tied": float(np.mean(np.abs(diffs) <= 1e-10)),
        "difference_quantiles_p10_p25_p50_p75_p90": q.tolist(),
        "bonferroni_ci9875_cluster_percentile": np.quantile(boot_cluster, [.00625, .99375]).tolist(),
        "pointwise_ci95_cluster_percentile": np.quantile(boot_cluster, [.025, .975]).tolist(),
        "bonferroni_ci9875_ordinary_pairs_sensitivity": np.quantile(boot_pair, [.00625, .99375]).tolist(),
        "cluster_inference_limited": len(clusters) < 20 or max(map(len, clusters)) / len(diffs) > .25,
    }


def read_manifest(path: Path) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    raw = json.loads(path.read_text())
    rows = raw if isinstance(raw, list) else raw.get("rows", raw.get("records"))
    if rows is None and isinstance(raw, dict) and raw.get("cohort_manifest"):
        csv_path = path.parent / str(raw["cohort_manifest"])
        if raw.get("cohort_manifest_sha256") and hashlib.sha256(csv_path.read_bytes()).hexdigest() != raw["cohort_manifest_sha256"]:
            raise ValueError("Referenced cohort manifest hash does not match frozen descriptor")
        with csv_path.open(newline="") as handle: rows = list(csv.DictReader(handle))
    if not isinstance(rows, list): raise ValueError("Manifest must contain flat rows or reference a cohort-manifest CSV")
    index = {}
    for row0 in rows:
        row = dict(row0); row["source"] = normalize_source(str(row["source"])); row["material_id"] = str(row["material_id"])
        row["n_sites"] = int(row["n_sites"]); row["n_elements"] = int(row["n_elements"])
        row["reduced_formula"] = scale_invariant_formula_key(str(row["formula"])); row["was_pilot"] = str(row["was_pilot"]).lower() in ("1", "true", "yes")
        key = (row["source"], row["material_id"])
        if key in index: raise ValueError(f"Duplicate manifest identity {key}")
        index[key] = row
    return list(index.values()), index


def read_coordination(path: Path, manifest: dict[tuple[str, str], dict[str, object]]) -> dict[tuple[str, str, str], float]:
    index = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            source, mid, method = normalize_source(row["source"]), str(row["material_id"]), row["neighbor_method"].strip().lower()
            if method not in METHODS: raise ValueError(f"Unknown neighbor method {method!r}")
            if (source, mid) not in manifest: raise ValueError(f"Coordination row absent from manifest: {(source, mid)}")
            key = (source, mid, method)
            if key in index: raise ValueError(f"Duplicate coordination identity {key}")
            index[key] = finite_float(row["mean_cn"], "mean_cn")
    return index


def read_pairs(path: Path, manifest: dict[tuple[str, str], dict[str, object]]) -> list[dict[str, object]]:
    out, seen = [], set()
    used_endpoints = set()
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            analysis, source = raw["analysis"].strip().lower(), normalize_source(raw["source"])
            if analysis not in ANALYSES or source == "gnome": raise ValueError(f"Invalid pair analysis/source: {analysis}/{source}")
            gid, oid = str(raw["gnome_id"]), str(raw["comparator_id"])
            identity = (analysis, source, gid, oid)
            if identity in seen: raise ValueError(f"Duplicate pair {identity}")
            seen.add(identity)
            if ("gnome", gid) not in manifest or (source, oid) not in manifest:
                raise ValueError(f"Pair endpoint absent from manifest: {identity}")
            if source == "gnome" or (source == "gnome" and gid == oid): raise ValueError(f"Self pair: {identity}")
            g, o = manifest[("gnome", gid)], manifest[(source, oid)]
            for role, mid in (("gnome", gid), ("comparator", oid)):
                endpoint = (analysis, source, role, mid)
                if endpoint in used_endpoints:
                    raise ValueError(f"Reused endpoint within matched comparison: {endpoint}")
                used_endpoints.add(endpoint)
            if g["chemistry_class"] != o["chemistry_class"] or abs(g["n_elements"] - o["n_elements"]) > 1:
                raise ValueError(f"Pair violates frozen chemistry restrictions: {identity}")
            if not .5 <= g["n_sites"] / o["n_sites"] <= 2:
                raise ValueError(f"Pair violates frozen site-count restriction: {identity}")
            if analysis == "unused" and (g["was_pilot"] or o["was_pilot"]):
                raise ValueError(f"Pilot record in unused sensitivity: {identity}")
            out.append({"analysis": analysis, "source": source, "gnome_id": gid, "comparator_id": oid,
                        "gnome_formula": g["reduced_formula"], "comparator_formula": o["reduced_formula"],
                        "chemistry_class": str(g["chemistry_class"]),
                        "gnome_n_elements": g["n_elements"], "comparator_n_elements": o["n_elements"]})
    return out


def descriptive(values: list[float]) -> dict[str, object]:
    a = np.asarray(values)
    return {"n": len(values), "mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1)) if len(a) > 1 else None,
            "median": float(np.median(a)), "p10_p25_p75_p90": np.quantile(a, [.1, .25, .75, .9]).tolist()}


def analyze(args: argparse.Namespace) -> dict[str, object]:
    manifest_rows, manifest = read_manifest(args.manifest)
    cn = read_coordination(args.coordination, manifest)
    pairs = read_pairs(args.pairs, manifest)
    out: dict[str, object] = {
        "status": "post-hoc full-cohort coordination extension",
        "estimand": "GNoME minus comparator difference in per-structure mean coordination",
        "multiplicity": "Within each neighbor method and analysis set, four comparator intervals use Bonferroni 98.75% percentile bounds for approximate familywise 95% coverage.",
        "bootstrap": {"replicates": args.n_bootstrap, "seed": args.seed,
                      "primary_resampling": "connected components of pairs sharing a source-qualified reduced formula at either endpoint",
                      "sensitivity": "ordinary pairs bootstrap"},
        "input_sha256": {p: hashlib.sha256(getattr(args, p).read_bytes()).hexdigest() for p in ("manifest", "coordination", "pairs")},
        "full_cohort_descriptives": {}, "matched": {},
    }
    for method in METHODS:
        out["full_cohort_descriptives"][method] = {}
        for source in ("gnome", *COMPARATORS):
            vals = [cn[(source, str(r["material_id"]), method)] for r in manifest_rows
                    if r["source"] == source and (source, str(r["material_id"]), method) in cn]
            planned = sum(r["source"] == source for r in manifest_rows)
            out["full_cohort_descriptives"][method][source] = {"n_planned": planned, "n_available": len(vals),
                "n_missing": planned-len(vals), **(descriptive(vals) if vals else {})}
        out["matched"][method] = {}
        for analysis in ANALYSES:
            out["matched"][method][analysis] = {}
            for source in COMPARATORS:
                planned_pairs = [p for p in pairs if p["analysis"] == analysis and p["source"] == source]
                complete, diffs, missing_gnome, missing_comparator = [], [], 0, 0
                for p in planned_pairs:
                    gk = ("gnome", str(p["gnome_id"]), method); ok = (source, str(p["comparator_id"]), method)
                    missing_gnome += gk not in cn; missing_comparator += ok not in cn
                    if gk in cn and ok in cn: complete.append(p); diffs.append(cn[gk] - cn[ok])
                record: dict[str, object] = {"n_planned": len(planned_pairs), "n_complete": len(complete),
                                             "n_attrited": len(planned_pairs)-len(complete),
                                             "n_missing_gnome_endpoint": missing_gnome,
                                             "n_missing_comparator_endpoint": missing_comparator}
                g_available = sum(r['source'] == 'gnome' and (analysis == 'all' or not r['was_pilot']) for r in manifest_rows)
                c_available = sum(r['source'] == source and (analysis == 'all' or not r['was_pilot']) for r in manifest_rows)
                record['matching_coverage'] = {'gnome_available': g_available, 'comparator_available': c_available,
                                              'gnome_fraction_matched': len(planned_pairs) / g_available if g_available else None,
                                              'comparator_fraction_matched': len(planned_pairs) / c_available if c_available else None}
                if complete:
                    d = np.asarray(diffs); clusters = pair_clusters(complete)
                    record["primary"] = summarize_differences(d, clusters, args.n_bootstrap, args.seed, f"{method}|{analysis}|{source}")
                    print(f"{method}/{analysis}/{source}: {len(d)} pairs, median {np.median(d):+.4f}, interval {record['primary']['bonferroni_ci9875_cluster_percentile']}", flush=True)
                    exact_ix = [i for i,p in enumerate(complete) if p["gnome_n_elements"] == p["comparator_n_elements"]]
                    if exact_ix:
                        exact = d[exact_ix]
                        record["exact_n_elements_sensitivity"] = {
                            **descriptive(exact.tolist()),
                            "mean_paired_difference": float(np.mean(exact)),
                            "median_paired_difference": float(np.median(exact)),
                            "fraction_gnome_higher": float(np.mean(exact > 1e-10)),
                            "note": "Low-cost effect-distribution sensitivity; no additional bootstrap inference.",
                        }
                    classes = defaultdict(list)
                    for p, value in zip(complete, d): classes[str(p["chemistry_class"])].append(float(value))
                    record["by_gnome_chemistry_class"] = {k: {**descriptive(v), "fraction_gnome_higher": float(np.mean(np.asarray(v)>1e-10))}
                                                           for k,v in sorted(classes.items())}
                out["matched"][method][analysis][source] = record
    return out


def fmt(x: float) -> str: return f"{x:+.3f}"


def report(summary: dict[str, object]) -> str:
    lines = ["# Full-cohort coordination extension", "",
             "This extension tests the higher-coordination finding across the complete analyzed public cohorts. CrystalNN supplies a species-independent effective coordination; the existing VoronoiNN rule counts neighbors retained by an atomic-radius distance screen. These quantities have distinct numerical scales. The comparisons describe coordination differences and their consistency across neighbor definitions.", ""]
    for analysis in ANALYSES:
        lines += [f"## Matched comparisons: {analysis}", "",
                  "| Method | Comparator | complete/planned | GNoME / comparator coverage | Median paired difference (98.75% cluster CI) | Mean difference | GNoME higher |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            for source in COMPARATORS:
                r = summary["matched"][method][analysis][source]
                if "primary" not in r:
                    lines.append(f"| {method} | {source} | 0/{r['n_planned']} | unavailable | — | — |")
                    continue
                s=r["primary"]; lo,hi=s["bonferroni_ci9875_cluster_percentile"]
                coverage=r['matching_coverage']
                lines.append(f"| {method} | {source} | {r['n_complete']}/{r['n_planned']} | {coverage['gnome_fraction_matched']:.1%} / {coverage['comparator_fraction_matched']:.1%} | {fmt(s['median_paired_difference'])} [{fmt(lo)}, {fmt(hi)}] | {fmt(s['mean_paired_difference'])} | {s['fraction_gnome_higher']:.1%} |")
        lines += ["", "Intervals are percentile cluster-bootstrap intervals. They are approximate; sparse or dominant formula components are flagged in `summary.json`. Ordinary-pair and exact-element-count sensitivities are also recorded.", ""]
    lines += ["## Full-cohort coverage", "", "| Method | Source | available/planned | Median mean CN |", "|---|---|---:|---:|"]
    for method in METHODS:
        for source in ("gnome", *COMPARATORS):
            d=summary["full_cohort_descriptives"][method][source]
            lines.append(f"| {method} | {source} | {d['n_available']}/{d['n_planned']} | {d.get('median', float('nan')):.3f} |")
    lines += ["", "Broad chemistry-class effect distributions are provided in `summary.json`. A credible replication should be judged from direction, effect magnitude and distributional consistency across neighbor definitions and sensitivities; no rule requires every adjusted interval to exclude zero.", ""]
    return "\n".join(lines)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--coordination", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=20000); p.add_argument("--seed", type=int, default=20260910)
    args=p.parse_args()
    if args.n_bootstrap < 100: raise ValueError("At least 100 bootstrap replicates required")
    result=analyze(args); args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (args.output_dir/"report.md").write_text(report(result))


if __name__ == "__main__": main()
