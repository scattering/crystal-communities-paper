#!/usr/bin/env python3
"""Historical structural and formula-precedent contingency statistics. Current regeneration scripts import these helpers with repaired inputs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTES = ROOT / "notes"
STAMPEDE_DIR = HERE / "stampede"
sys.path.insert(0, str(HERE))
from stampede_formula_reference import anonymized_formula, composition_class  # noqa: E402

SEED = 42
CUTOFFS = [2010, 2000, 1990]
MAX_YEAR = 2015
N_BOOT = 2000
N_PERM = 2000
SOURCES = {
    "GNoME": NOTES / "external_frontier_runs/gnome_frontier_20260419/gnome_frontier_records.csv",
    "MatterGen": NOTES / "external_frontier_runs/mattergen_frontier_20260419/mattergen-public_frontier_records.csv",
    "MP": NOTES / "external_frontier_runs/mp_frontier_20260427/mp_frontier_records.csv",
    "JARVIS": NOTES / "external_frontier_runs/jarvis_frontier_20260427/jarvis_frontier_records.csv",
    "Alexandria": NOTES / "external_frontier_runs/alexandria_frontier_20260427/alexandria_frontier_records.csv",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


# ---------------------------------------------------------------- map ----
def build_map():
    X = np.load(NOTES / "features.npy")
    sd = X.std(axis=0)
    Xs = (X - X.mean(axis=0)) / np.where(sd == 0, 1.0, sd)
    Xp = PCA(n_components=32, random_state=SEED).fit_transform(Xs)
    ca = pd.read_csv(NOTES / "icsd_community_assignments/community_assignments_labels3.csv")
    assert len(ca) == Xp.shape[0] == 167_500
    years = ca["year"].fillna(-1).astype(int).to_numpy()
    comms = ca["community"].fillna(-1).astype(int).to_numpy()
    ids = ca["icsd_id"].astype(int).to_numpy()
    return Xp, ids, years, comms


def heldout_classification(Xp, ids, years, comms, T):
    """Exact logic of scripts/analyze_synthesis_retrodiction.py steps 2-3."""
    train = (years > 0) & (years <= T)
    test = years > T
    tX, tl = Xp[train], comms[train]
    cids = sorted({int(c) for c in tl if c >= 0})
    cent = np.zeros((len(cids), Xp.shape[1]))
    thr = np.zeros(len(cids))
    for i, c in enumerate(cids):
        m = tX[tl == c]
        cent[i] = m.mean(axis=0)
        thr[i] = np.percentile(np.linalg.norm(m - cent[i], axis=1), 95)
    hX = Xp[test]
    # nearest centroid: squared distances via the matrix-product expansion
    # (chunked, ~100 MB per chunk), then the exact Euclidean distance is
    # recomputed for the 3 closest candidates so the argmin / threshold
    # comparison is not affected by floating-point cancellation.
    idx = np.empty(len(hX), dtype=int)
    dist = np.empty(len(hX))
    c2 = (cent ** 2).sum(axis=1)
    kk = min(3, len(cent))
    for s in range(0, len(hX), 2000):
        h = hX[s:s + 2000]
        d2 = (h ** 2).sum(axis=1)[:, None] + c2[None, :] - 2.0 * (h @ cent.T)
        cand = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        exact = np.linalg.norm(h[:, None, :] - cent[cand], axis=2)
        j = exact.argmin(axis=1)
        idx[s:s + 2000] = cand[np.arange(len(h)), j]
        dist[s:s + 2000] = exact[np.arange(len(h)), j]
    cid_arr = np.asarray(cids)
    df = pd.DataFrame({
        "icsd_id": ids[test],
        "year": years[test],
        "assigned_community": cid_arr[idx],
        "nearest_centroid_distance": dist,
        "in_basin": (dist <= thr[idx]).astype(int),
    })
    thr_map = {int(c): float(t) for c, t in zip(cids, thr)}
    return df, thr_map, int(train.sum())


# ------------------------------------------------------------ formulas ----
def local_formula_table(T, cm_all):
    """Per-entry formula + strata for post-T entries from local derived files."""
    # keep_default_na=False: the reduced formula "NaN" (sodium nitride) must stay a string
    fr = pd.read_csv(NOTES / f"icsd_first_report_formulas/split_{T}/first_report_formulas.csv", keep_default_na=False)
    fr = fr.rename(columns={"cif_id": "icsd_id", "reduced_formula": "formula"})[
        ["icsd_id", "formula", "is_in_basin", "nearest_centroid_distance", "assigned_community"]]
    cm = cm_all[(cm_all.series == "ICSD") & (cm_all.cutoff == T)].copy()
    cm["icsd_id"] = cm["id"].astype(int)
    cm = cm[["icsd_id", "formula", "stratum_coarse", "stratum_anon", "in_basin"]]
    # strata via pymatgen parser for every formula (verify vs. stored strata)
    allf = pd.concat([fr[["icsd_id", "formula"]], cm[["icsd_id", "formula"]]]).drop_duplicates("icsd_id")
    allf["stratum_coarse_pm"] = allf["formula"].map(composition_class)
    allf["stratum_anon_pm"] = allf["formula"].map(anonymized_formula)
    chk = allf.merge(cm[["icsd_id", "stratum_coarse", "stratum_anon"]], on="icsd_id")
    agree_c = float((chk.stratum_coarse_pm == chk.stratum_coarse).mean())
    agree_a = float((chk.stratum_anon_pm == chk.stratum_anon).mean())
    return fr, cm, allf, {"n_cm_rows": int(len(cm)), "n_first_report_rows": int(len(fr)),
                          "n_union_entries_with_formula": int(len(allf)),
                          "strata_coarse_agreement_pm_vs_stored": agree_c,
                          "strata_anon_agreement_pm_vs_stored": agree_a}


def post1980_reference(T, fr1980):
    return set(fr1980.loc[fr1980.year <= T, "reduced_formula"])


# ---------------------------------------------------------------- stats ----
def quadrant(df, ib="in_basin", fm="formula_match"):
    n = len(df)
    ul = int(((df[ib] == 1) & (df[fm] == 1)).sum())
    ur = int(((df[ib] == 1) & (df[fm] == 0)).sum())
    ll = int(((df[ib] == 0) & (df[fm] == 1)).sum())
    lr = int(((df[ib] == 0) & (df[fm] == 0)).sum())
    out = {"n": n,
           "in_basin_and_match": ul, "in_basin_and_no_match": ur,
           "frontier_and_match": ll, "frontier_and_no_match": lr}
    if n:
        out.update({f"share_{k}": v / n for k, v in list(out.items())[1:]})
        p_ib = (ul + ur) / n
        p_m = (ul + ll) / n
        out["p_in_basin"] = p_ib
        out["p_match"] = p_m
        out["expected_share_in_basin_and_match_independence"] = p_ib * p_m
        out["enrichment_ratio_obs_over_independence"] = (ul / n) / (p_ib * p_m) if p_ib * p_m > 0 else None
        out["p_in_basin_given_match"] = ul / (ul + ll) if ul + ll else None
        out["p_in_basin_given_no_match"] = ur / (ur + lr) if ur + lr else None
        out["wilson_in_basin_given_match"] = wilson(ul, ul + ll)
        out["wilson_in_basin_given_no_match"] = wilson(ur, ur + lr)
        if out["p_in_basin_given_match"] and out["p_in_basin_given_no_match"]:
            out["relative_risk_in_basin_match_vs_no_match"] = out["p_in_basin_given_match"] / out["p_in_basin_given_no_match"]
        out["odds_ratio"] = (ul * lr) / (ur * ll) if ur * ll else None
        out["wilson_share_in_basin_and_match"] = wilson(ul, n)
    return out


def bootstrap(df, rng, n_boot=N_BOOT):
    ib = df["in_basin"].to_numpy()
    fm = df["formula_match"].to_numpy()
    n = len(df)
    enr, rr, ul = [], [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        a, b = ib[s], fm[s]
        p_ul = float((a & b).mean())
        p_ib, p_m = a.mean(), b.mean()
        enr.append(p_ul / (p_ib * p_m))
        ul.append(p_ul)
        pm_ib = (a & b).sum() / max(b.sum(), 1)
        pnm_ib = (a & (1 - b)).sum() / max((1 - b).sum(), 1)
        rr.append(pm_ib / pnm_ib if pnm_ib else np.nan)
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"enrichment_ratio_ci95": q(enr), "share_in_basin_and_match_ci95": q(ul),
            "relative_risk_ci95": q(rr), "n_boot": n_boot}


def permutation(df, rng, stratum=None, n_perm=N_PERM):
    """Shuffle formula_match labels (globally or within stratum groups);
    return null distribution of the in-basin&match share and a p-value."""
    ib = df["in_basin"].to_numpy().astype(bool)
    fm = df["formula_match"].to_numpy().astype(bool)
    n = len(df)
    obs = float((ib & fm).mean())
    if stratum is None:
        inv = np.zeros(n, dtype=int)
        n_strata = 1
    else:
        codes = df[stratum].fillna("__none__").astype(str).to_numpy()
        _, inv = np.unique(codes, return_inverse=True)
        n_strata = int(inv.max()) + 1
    # Vectorised within-stratum shuffle: `pos` lists entries grouped by
    # stratum; `order` lists entries grouped by stratum in a random
    # within-stratum order, so fm[order] -> positions `pos` permutes labels
    # only among entries sharing a stratum.
    pos = np.argsort(inv, kind="stable")
    null = np.empty(n_perm)
    fm_perm = np.empty_like(fm)
    for r in range(n_perm):
        order = np.lexsort((rng.random(n), inv))
        fm_perm[pos] = fm[order]
        null[r] = (ib & fm_perm).mean()
    p = (1 + int((null >= obs).sum())) / (n_perm + 1)
    return {"observed_share": obs, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_ci95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
            "z": float((obs - null.mean()) / null.std(ddof=1)) if null.std(ddof=1) > 0 else None,
            "p_one_sided": p, "n_perm": n_perm,
            "n_strata": n_strata}


def by_year(df):
    rows = {}
    for y, g in df.groupby("year"):
        q = quadrant(g)
        rows[int(y)] = {k: q[k] for k in ("n", "in_basin_and_match", "in_basin_and_no_match",
                                            "frontier_and_match", "frontier_and_no_match",
                                            "share_in_basin_and_match", "p_in_basin", "p_match",
                                            "enrichment_ratio_obs_over_independence",
                                            "p_in_basin_given_match", "p_in_basin_given_no_match")}
    return rows


def computed_sources_at_T(thr_map, ref_post1980, ref_full, ref_all=None):
    """Apples-to-apples: classify each computed-source sample against the
    T-cutoff per-community thresholds (same approximation as SI S7: the
    record's assigned_community and nearest_centroid_distance are relative to
    the full-map centroids) and against the T-truncated formula reference."""
    out = {}
    for name, path in SOURCES.items():
        r = pd.read_csv(path, keep_default_na=False)
        r = r[r["reduced_formula"].notna() & (r["reduced_formula"].astype(str).str.strip() != "")].copy()
        thr = r["assigned_community"].map(lambda c: thr_map.get(int(c), np.nan))
        r["in_basin"] = ((r["nearest_centroid_distance"] <= thr) & thr.notna()).astype(int)
        r["formula_match"] = r["reduced_formula"].isin(ref_post1980).astype(int)
        q_T = quadrant(r)
        r["formula_match"] = r["reduced_formula"].isin(ref_full).astype(int)
        q_full = quadrant(r)
        entry = {"n": int(len(r)),
                 "quadrant_T_map_T_post1980_reference": q_T,
                 "quadrant_T_map_full_post1980_reference": q_full}
        if ref_all is not None:
            r["formula_match"] = r["reduced_formula"].isin(ref_all).astype(int)
            entry["quadrant_T_map_T_allyear_reference"] = quadrant(r)
        out[name] = entry
    return out


# ----------------------------------------------------------------- main ----
def main() -> int:
    rng = np.random.default_rng(SEED)
    print("building map (PCA 32) ...", flush=True)
    Xp, ids, years, comms = build_map()
    cm_all = pd.read_csv(NOTES / "composition_matched_ai_records.csv", dtype={"id": str}, keep_default_na=False)
    fr1980 = pd.read_csv(NOTES / "icsd_first_report_formulas/split_1980/first_report_formulas.csv", keep_default_na=False)
    ref_full = set(fr1980["reduced_formula"])  # manuscript Fig. 4 reference (81,531 formulas)

    synth = json.load(open(NOTES / "formula_synth_prior_summary.json"))
    baseline_fig4 = {}
    for name, s in synth["sources"].items():
        q = s["quadrants"]
        n = s["n"]
        baseline_fig4[name] = {"n": n, **{k: v for k, v in q.items()},
                               **{f"share_{k}": v / n for k, v in q.items()}}

    results = {"seed": SEED, "max_year": MAX_YEAR, "n_boot": N_BOOT, "n_perm": N_PERM,
               "manuscript_fig4_reference_size": len(ref_full),
               "baseline_fig4_computed_sources": baseline_fig4, "cutoffs": {}}

    for T in CUTOFFS:
        print(f"\n=== T = {T} ===", flush=True)
        ho, thr_map, n_train = heldout_classification(Xp, ids, years, comms, T)
        fr, cm, allf, cov = local_formula_table(T, cm_all)

        # --- verification of the recomputed in-basin flags vs production ---
        v1 = ho.merge(fr, on="icsd_id", suffixes=("", "_prod"))
        v2 = ho.merge(cm[["icsd_id", "in_basin"]], on="icsd_id", suffixes=("", "_cm"))
        verify = {
            "n_train_year_le_T": n_train,
            "n_heldout_year_gt_T_all": int(len(ho)),
            "n_heldout_T_lt_year_le_max": int((ho.year <= MAX_YEAR).sum()),
            "in_basin_rate_all_heldout": float(ho.in_basin.mean()),
            "vs_first_report_split": {
                "n_overlap": int(len(v1)),
                "in_basin_agreement": float((v1.in_basin == v1.is_in_basin).mean()),
                "community_agreement": float((v1.assigned_community == v1.assigned_community_prod).mean()),
                "max_abs_distance_diff": float((v1.nearest_centroid_distance - v1.nearest_centroid_distance_prod).abs().max()),
            },
            "vs_composition_matched_records": {
                "n_overlap": int(len(v2)),
                "in_basin_agreement": float((v2.in_basin == v2.in_basin_cm).mean()),
                "in_basin_rate_on_this_population": float(v2.in_basin.mean()),
            },
        }
        print("  verify:", json.dumps(verify, indent=None), flush=True)
        # diagnostic for entries whose recomputed flag differs from the production flag
        dis = v2[v2.in_basin != v2.in_basin_cm]
        for _, row in dis.iterrows():
            thr_c = thr_map[int(row.assigned_community)]
            print(f"  DISAGREE icsd_id={int(row.icsd_id)} community={int(row.assigned_community)} dist={row.nearest_centroid_distance:.15g} thr={thr_c:.15g} dist-thr={row.nearest_centroid_distance - thr_c:.3g}", flush=True)
        verify["disagreements_vs_composition_matched"] = [{"icsd_id": int(r.icsd_id), "dist_minus_threshold": float(r.nearest_centroid_distance - thr_map[int(r.assigned_community)])} for _, r in dis.iterrows()]

        # --- references ---
        ref_post = post1980_reference(T, fr1980)
        refs = {"post1980_le_T": ref_post}
        ref_sizes = {"n_reference_formulas_1980lt_year_le_T_featureset": len(ref_post)}
        ref_all = None
        stam_entries = None
        if STAMPEDE_DIR.exists():
            f_all = STAMPEDE_DIR / f"icsd_reduced_formulas_year_le_{T}.txt"
            f_post = STAMPEDE_DIR / f"icsd_reduced_formulas_1980lt_year_le_{T}.txt"
            f_ent = STAMPEDE_DIR / f"post_cutoff_formula_match_T{T}.csv"
            if f_all.exists():
                ref_all = {l.strip() for l in f_all.read_text().splitlines() if l.strip()}
                refs["allyear_le_T_index"] = ref_all
                ref_sizes["n_reference_formulas_year_le_T_index"] = len(ref_all)
            if f_post.exists():
                ref_post_idx = {l.strip() for l in f_post.read_text().splitlines() if l.strip()}
                refs["post1980_le_T_index"] = ref_post_idx
                ref_sizes["n_reference_formulas_1980lt_year_le_T_index"] = len(ref_post_idx)
            if f_ent.exists():
                stam_entries = pd.read_csv(f_ent)

        # --- per-entry table ---
        ent = ho[ho.year <= MAX_YEAR].merge(allf, on="icsd_id", how="left")
        ent = ent.merge(cm[["icsd_id", "stratum_coarse", "stratum_anon"]], on="icsd_id", how="left")
        ent["stratum_coarse"] = ent["stratum_coarse"].fillna(ent["stratum_coarse_pm"])
        ent["stratum_anon"] = ent["stratum_anon"].fillna(ent["stratum_anon_pm"])
        n_all = len(ent)
        known = ent["formula"].notna() & (ent["formula"].astype(str) != "")
        cov["n_post_T_entries_T_lt_year_le_max"] = int(n_all)
        cov["n_with_local_formula"] = int(known.sum())
        cov["coverage_fraction"] = float(known.mean())
        cov["in_basin_rate_entries_without_local_formula"] = float(ent.loc[~known, "in_basin"].mean()) if (~known).any() else None

        cut = {"verification": verify, "coverage": cov, "reference_sizes": ref_sizes, "analyses": {}}

        for ref_name, ref in refs.items():
            e = ent[known].copy()
            e["formula_match"] = e["formula"].isin(ref).astype(int)
            unit_entries = {
                "unit": "post-T ICSD entries with a locally available reduced formula",
                "quadrant": quadrant(e),
                "by_year": by_year(e),
                "bootstrap": bootstrap(e, rng),
                "permutation_global": permutation(e, rng),
                "permutation_within_coarse_strata": permutation(e, rng, "stratum_coarse"),
                "permutation_within_anonymized_strata": permutation(e, rng, "stratum_anon"),
            }
            # per-formula unit: earliest post-T entry per formula (Fig. 3c/SI population)
            f = ho.merge(fr[["icsd_id", "formula"]], on="icsd_id")
            f = f[f.year <= MAX_YEAR].merge(allf[["icsd_id", "stratum_coarse_pm", "stratum_anon_pm"]], on="icsd_id", how="left")
            f = f.rename(columns={"stratum_coarse_pm": "stratum_coarse", "stratum_anon_pm": "stratum_anon"})
            f["formula_match"] = f["formula"].isin(ref).astype(int)
            unit_formulas = {
                "unit": "post-T first-report formulas (earliest post-T entry per reduced formula)",
                "quadrant": quadrant(f),
                "by_year": by_year(f),
                "bootstrap": bootstrap(f, rng),
                "permutation_global": permutation(f, rng),
                "permutation_within_coarse_strata": permutation(f, rng, "stratum_coarse"),
                "permutation_within_anonymized_strata": permutation(f, rng, "stratum_anon"),
            }
            cut["analyses"][ref_name] = {"per_entry": unit_entries, "per_formula": unit_formulas}
            q = unit_entries["quadrant"]
            print(f"  [{ref_name}] per-entry n={q['n']} UL={q['share_in_basin_and_match']:.4f} "
                  f"P(ib)={q['p_in_basin']:.4f} P(m)={q['p_match']:.4f} "
                  f"enrich={q['enrichment_ratio_obs_over_independence']:.3f} "
                  f"P(ib|m)={q['p_in_basin_given_match']:.4f} P(ib|!m)={q['p_in_basin_given_no_match']:.4f}", flush=True)
            q = unit_formulas["quadrant"]
            print(f"  [{ref_name}] per-formula n={q['n']} UL={q['share_in_basin_and_match']:.4f} "
                  f"enrich={q['enrichment_ratio_obs_over_independence']:.3f}", flush=True)

            if ref_name == "post1980_le_T":
                out_csv = HERE / f"retrospective_quadrant_T{T}.csv"
                e[["icsd_id", "year", "in_basin", "formula_match"]].sort_values("icsd_id").to_csv(out_csv, index=False)

        # Stampede complete per-entry table (if present): join in_basin on all entries
        if stam_entries is not None:
            s = ho[ho.year <= MAX_YEAR][["icsd_id", "year", "in_basin"]].merge(
                stam_entries.drop(columns=["year"]), on="icsd_id", how="inner")
            comp = {}
            for col, label in (("formula_match_post1980", "post1980_le_T_index"), ("formula_match_allyear", "allyear_le_T_index")):
                s2 = s.rename(columns={col: "formula_match"})
                comp[label] = {"quadrant": quadrant(s2), "by_year": by_year(s2), "bootstrap": bootstrap(s2, rng),
                               "permutation_global": permutation(s2, rng),
                               "permutation_within_coarse_strata": permutation(s2, rng, "stratum_coarse"),
                               "permutation_within_anonymized_strata": permutation(s2, rng, "stratum_anon")}
            cut["analyses_stampede_complete_per_entry"] = {"n": int(len(s)), **comp}
            s.rename(columns={"formula_match_allyear": "formula_match"})[["icsd_id", "year", "in_basin", "formula_match"]] \
                .sort_values("icsd_id").to_csv(HERE / f"retrospective_quadrant_T{T}_allyear_reference.csv", index=False)

        cut["computed_sources_same_T_conditions"] = computed_sources_at_T(thr_map, ref_post, ref_full, ref_all)
        for name, v in cut["computed_sources_same_T_conditions"].items():
            q = v["quadrant_T_map_T_post1980_reference"]
            print(f"  source {name}: n={q['n']} UL={q['share_in_basin_and_match']:.4f} LR={q['share_frontier_and_no_match']:.4f} "
                  f"P(ib)={q['p_in_basin']:.4f} P(m)={q['p_match']:.4f}", flush=True)
        results["cutoffs"][str(T)] = cut

    (HERE / "retrospective_quadrant_summary.json").write_text(json.dumps(results, indent=2, default=float))
    print("\nwrote", HERE / "retrospective_quadrant_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
