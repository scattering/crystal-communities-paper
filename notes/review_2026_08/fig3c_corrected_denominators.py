#!/usr/bin/env python3
"""Historical basin calibration helpers. Current callers pass repaired arrays and assignments; legacy standalone defaults are retained for provenance."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTES = ROOT / "notes"
SEED = 42
CUTOFFS = [1990, 2000, 2010]
SOURCES = OrderedDict([
    ("GNoME", NOTES / "external_frontier_runs/gnome_frontier_20260419/gnome_frontier_records.csv"),
    ("MatterGen", NOTES / "external_frontier_runs/mattergen_frontier_20260419/mattergen-public_frontier_records.csv"),
    ("MP", NOTES / "external_frontier_runs/mp_frontier_20260427/mp_frontier_records.csv"),
    ("JARVIS", NOTES / "external_frontier_runs/jarvis_frontier_20260427/jarvis_frontier_records.csv"),
    ("Alexandria", NOTES / "external_frontier_runs/alexandria_frontier_20260427/alexandria_frontier_records.csv"),
])
# Panel-(c) display order in scripts/make_fig_5source_calibration_revised.py
ORDER = ["ICSD", "MatterGen", "GNoME", "MP", "JARVIS", "Alexandria"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Identical to scripts/analyze_composition_matched_ai.py::wilson."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def load_coarse_parser():
    """Import composition_class from the production script without running it."""
    path = ROOT / "scripts" / "analyze_composition_matched_ai.py"
    spec = importlib.util.spec_from_file_location("acm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.composition_class


# ------------------------------------------------------------------ map ---
def build_map():
    X = np.load(NOTES / "features.npy")
    sd = X.std(axis=0)
    Xs = (X - X.mean(axis=0)) / np.where(sd == 0, 1.0, sd)
    Xp = PCA(n_components=32, random_state=SEED).fit_transform(Xs)
    ca = pd.read_csv(NOTES / "icsd_community_assignments/community_assignments_labels3.csv")
    assert len(ca) == Xp.shape[0] == 167_500, (len(ca), Xp.shape)
    years = ca["year"].fillna(-1).astype(int).to_numpy()
    comms = ca["community"].fillna(-1).astype(int).to_numpy()
    ids = ca["icsd_id"].astype(int).to_numpy()
    return Xp, ids, years, comms


def alignment_checks(Xp, years, comms) -> dict:
    """(1) recomputed basis == archived Zenodo basis; (2) full-map per-community
    p95 thresholds reproduce notes/per_community_thresholds_fullmap_p95.json,
    which proves the features.npy rows are aligned with the assignments CSV."""
    out = {}
    Xz = np.load(HERE / "features_pca_zenodo.npy")
    sign = np.sign((Xz * Xp).sum(axis=0))
    out["max_abs_diff_vs_zenodo_basis_sign_aligned"] = float(np.abs(Xz - Xp * sign).max())
    ref = json.load(open(NOTES / "per_community_thresholds_fullmap_p95.json"))
    thr_ref = ref["per_community_p95_threshold"]
    size_ref = ref["per_community_size"]
    # try both "all rows with label c" and "dated rows with label c"
    rows = []
    rng = np.random.default_rng(0)
    test_comms = [0, 1, 2, 43, 1485, 5450, 4549, 2958] + [int(c) for c in rng.choice(6756, 12, replace=False)]
    max_all = 0.0
    max_dated = 0.0
    for c in test_comms:
        m_all = np.flatnonzero(comms == c)
        m_dated = np.flatnonzero((comms == c) & (years > 0))
        for tag, m in (("all", m_all), ("dated", m_dated)):
            cen = Xp[m].mean(axis=0)
            p95 = float(np.percentile(np.linalg.norm(Xp[m] - cen, axis=1), 95))
            d = abs(p95 - float(thr_ref[str(c)]))
            if tag == "all":
                max_all = max(max_all, d)
            else:
                max_dated = max(max_dated, d)
        rows.append({"community": c, "n_all": int(len(m_all)), "n_dated": int(len(m_dated)),
                     "size_in_json": int(size_ref[str(c)]), "p95_json": float(thr_ref[str(c)])})
    out["p95_reproduction_max_abs_diff_all_rows"] = max_all
    out["p95_reproduction_max_abs_diff_dated_rows_only"] = max_dated
    out["p95_reproduction_sample"] = rows
    out["n_communities_in_json"] = int(ref["n_communities_with_threshold"])
    out["n_communities_in_assignments"] = int(len(set(comms[comms >= 0].tolist())))
    return out


def cutoff_map(Xp, years, comms, T):
    """Per-community centroid + p95 threshold from dated members with year <= T.
    Exact logic of scripts/analyze_synthesis_retrodiction.py::centroid_thresholds
    and scripts/analyze_composition_matched_ai.py::compute_thresholds."""
    train = (years > 0) & (years <= T)
    tX, tl = Xp[train], comms[train]
    cids = sorted({int(c) for c in tl if c >= 0})
    cent = np.zeros((len(cids), Xp.shape[1]))
    thr = np.zeros(len(cids))
    for i, c in enumerate(cids):
        m = tX[tl == c]
        cent[i] = m.mean(axis=0)
        thr[i] = float(np.percentile(np.linalg.norm(m - cent[i], axis=1), 95))
    thr_map = {int(c): float(t) for c, t in zip(cids, thr)}
    return np.asarray(cids), cent, thr, thr_map, int(train.sum())


def heldout_classify(Xp, ids, years, comms, T, cids, cent, thr):
    """Every entry with year > T (no formula requirement; outliers included),
    assigned to the nearest <= T centroid; in-basin iff d <= tau_c(T)."""
    test = years > T
    hX = Xp[test]
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
    return pd.DataFrame({
        "icsd_id": ids[test],
        "year": years[test],
        "fullmap_community": comms[test],
        "community": cids[idx],
        "distance": dist,
        "threshold": thr[idx],
        "in_basin": (dist <= thr[idx]).astype(int),
    })


def rate_block(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "ci95": [lo, hi]}


def main() -> int:
    log = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    composition_class = load_coarse_parser()

    say("building map: StandardScaler + PCA(32, seed 42) on all 167,500 rows ...")
    Xp, ids, years, comms = build_map()
    align = alignment_checks(Xp, years, comms)
    say(f"  basis vs archived Zenodo basis: max|diff| = {align['max_abs_diff_vs_zenodo_basis_sign_aligned']:.3e}")
    say(f"  full-map p95 thresholds reproduced (all rows w/ label): max|diff| = {align['p95_reproduction_max_abs_diff_all_rows']:.3e}; "
        f"(dated rows only): {align['p95_reproduction_max_abs_diff_dated_rows_only']:.3e}")
    say(f"  communities: {align['n_communities_in_assignments']} in assignments, {align['n_communities_in_json']} in JSON")
    say(f"  dated entries: {(years > 0).sum()}, undated: {(years <= 0).sum()}, full-map outliers (label -1): {(comms < 0).sum()}")

    # published numbers + populations
    pub = json.load(open(NOTES / "composition_matched_ai_summary.json"))
    pub_by_T = {c["cutoff"]: c["matchings"]["coarse"] for c in pub["cutoffs"]}
    cm = pd.read_csv(NOTES / "composition_matched_ai_records.csv", dtype={"id": str}, keep_default_na=False)

    # external records (all rows, as the frontier scripts wrote them)
    ext = {}
    for name, path in SOURCES.items():
        r = pd.read_csv(path, keep_default_na=False, dtype={"material_id": str})
        r["id"] = r["material_id"].astype(str)
        r["formula"] = r["reduced_formula"].astype(str).str.strip()
        r["assigned_community"] = r["assigned_community"].astype(int)
        r["nearest_centroid_distance"] = r["nearest_centroid_distance"].astype(float)
        r["coarse_parseable"] = r["formula"].map(lambda f: composition_class(f) is not None)
        ext[name] = r
        say(f"  {name}: {len(r)} records; coarse-unparseable formula: {(~r['coarse_parseable']).sum()}")

    results = {"seed": SEED, "alignment_checks": align, "cutoffs": {}}
    summary_cutoffs = []
    record_rows = []

    for T in CUTOFFS:
        say(f"\n=== T = {T} ===")
        cids, cent, thr, thr_map, n_train = cutoff_map(Xp, years, comms, T)
        say(f"  train (dated, year <= {T}): {n_train} entries; communities present at T: {len(cids)}")

        # ---------------- held-out ICSD ----------------
        ho = heldout_classify(Xp, ids, years, comms, T, cids, cent, thr)
        n_all = len(ho)
        k_all = int(ho.in_basin.sum())
        pub_icsd = pub_by_T[T]["icsd_unmatched"]
        cm_icsd = cm[(cm.series == "ICSD") & (cm.cutoff == T)].copy()
        cm_icsd["icsd_id"] = cm_icsd["id"].astype(int)
        rep = ho.merge(cm_icsd[["icsd_id", "in_basin"]], on="icsd_id", suffixes=("", "_pub"))
        rep_k, rep_n = int(rep.in_basin.sum()), len(rep)
        agree = float((rep.in_basin == rep.in_basin_pub).mean())
        n_dis = int((rep.in_basin != rep.in_basin_pub).sum())
        # which held-out rows were omitted from the published population?
        omitted = ho[~ho.icsd_id.isin(cm_icsd.icsd_id)]
        icsd_block = {
            "population_all": rate_block(k_all, n_all),
            "population_published": {"k_recomputed": rep_k, "n": rep_n, "rate_recomputed": rep_k / rep_n,
                                      "k_published": pub_icsd["k"], "n_published": pub_icsd["n"],
                                      "rate_published": pub_icsd["rate"],
                                      "per_record_agreement": agree, "n_disagreements": n_dis,
                                      "all_published_ids_found": bool(rep_n == len(cm_icsd))},
            "n_omitted_from_published": int(len(omitted)),
            "omitted_in_basin_rate": float(omitted.in_basin.mean()) if len(omitted) else None,
            "n_heldout_fullmap_outliers": int((ho.fullmap_community < 0).sum()),
            "n_heldout_fullmap_outliers_in_published_population": int((rep.icsd_id.isin(ho[ho.fullmap_community < 0].icsd_id)).sum()),
            "n_heldout_assigned_to_community_absent_at_T": 0,  # by construction: nearest <= T centroid
        }
        say(f"  held-out ICSD: published pop n={rep_n} (JSON n={pub_icsd['n']}), recomputed rate={rep_k / rep_n:.4f} "
            f"(published {pub_icsd['rate']:.4f}), per-record agreement={agree:.6f} ({n_dis} disagreements)")
        say(f"  held-out ICSD: ALL post-T n={n_all}, k={k_all}, rate={k_all / n_all:.4f}; "
            f"omitted-from-published n={len(omitted)} with in-basin rate {icsd_block['omitted_in_basin_rate']:.4f}; "
            f"full-map outliers among held-out: {icsd_block['n_heldout_fullmap_outliers']} "
            f"({icsd_block['n_heldout_fullmap_outliers_in_published_population']} of them in the published population)")
        # per-record rows (vectorised append)
        record_rows.append(pd.DataFrame({
            "source": "ICSD", "id": ho.icsd_id.astype(str), "cutoff": T,
            "community": ho.community, "community_present_at_T": 1,
            "distance": ho.distance, "threshold": ho.threshold,
            "in_basin": ho.in_basin, "in_published_population": ho.icsd_id.isin(cm_icsd.icsd_id).astype(int),
        }))

        # ---------------- externals ----------------
        src_blocks = OrderedDict()
        for name, r in ext.items():
            thr_s = r["assigned_community"].map(lambda c: thr_map.get(int(c), np.nan))
            present = thr_s.notna()
            in_basin = (present & (r["nearest_centroid_distance"] <= thr_s)).astype(int)
            # reproduction: present-at-T AND coarse-parseable (production semantics)
            pub_mask = present & r["coarse_parseable"]
            k_rep, n_rep = int(in_basin[pub_mask].sum()), int(pub_mask.sum())
            pub_src = pub_by_T[T]["by_source"][name]["unmatched"]
            # per-record check against composition_matched_ai_records.csv
            cm_src = cm[(cm.series == name) & (cm.cutoff == T)][["id", "in_basin"]]
            chk = r.loc[pub_mask, ["id"]].assign(in_basin=in_basin[pub_mask].values).merge(cm_src, on="id", suffixes=("", "_pub"))
            agree_s = float((chk.in_basin == chk.in_basin_pub).mean()) if len(chk) else float("nan")
            # ids that occur more than once in the records CSV (MatterGen: 3
            # formulas shipped under two task folders) cross-match in the
            # merge; restrict to unique ids for the exact per-record check
            dup_ids = set(r.loc[r["id"].duplicated(keep=False), "id"])
            chk_u = chk[~chk.id.isin(dup_ids)]
            agree_u = float((chk_u.in_basin == chk_u.in_basin_pub).mean()) if len(chk_u) else float("nan")
            n_absent = int((~present).sum())
            n_unpars = int((~r["coarse_parseable"]).sum())
            n_absent_and_unpars = int(((~present) & (~r["coarse_parseable"])).sum())
            n_all_s = len(r)
            k_all_s = int(in_basin.sum())
            # what the omitted records look like
            om = ~pub_mask
            k_om_present = int(in_basin[om & present].sum())
            n_om_present = int((om & present).sum())
            src_blocks[name] = {
                "population_all": rate_block(k_all_s, n_all_s),
                "population_published": {"k_recomputed": k_rep, "n": n_rep, "rate_recomputed": k_rep / n_rep,
                                          "k_published": pub_src["k"], "n_published": pub_src["n"],
                                          "rate_published": pub_src["rate"],
                                          "per_record_agreement": agree_s, "n_matched_ids": int(len(chk)),
                                          "n_duplicated_ids_in_records_csv": int(len(dup_ids)),
                                          "per_record_agreement_unique_ids": agree_u, "n_unique_ids_checked": int(len(chk_u)),
                                          "n_published_rows_in_records_csv": int(len(cm_src))},
                "n_records": n_all_s,
                "n_community_absent_at_T": n_absent,
                "n_coarse_unparseable_formula": n_unpars,
                "n_absent_and_unparseable": n_absent_and_unpars,
                "n_omitted_total": int(om.sum()),
                "omitted_present_at_T_in_basin_rate": (k_om_present / n_om_present) if n_om_present else None,
                "n_omitted_present_at_T": n_om_present,
            }
            say(f"  {name:<10s} published pop n={n_rep} (JSON n={pub_src['n']}), recomputed k={k_rep} (JSON k={pub_src['k']}), "
                f"rate={k_rep / n_rep:.4f} (published {pub_src['rate']:.4f}), per-record agreement={agree_s:.6f} on {len(chk)} ids "
                f"({agree_u:.6f} on {len(chk_u)} unique ids; {len(dup_ids)} duplicated ids)")
            say(f"  {name:<10s} ALL n={n_all_s}, k={k_all_s}, rate={k_all_s / n_all_s:.4f}; absent-at-T={n_absent}, "
                f"unparseable={n_unpars}, both={n_absent_and_unpars}, omitted total={om.sum()}")
            record_rows.append(pd.DataFrame({
                "source": name, "id": r["id"], "cutoff": T,
                "community": r["assigned_community"], "community_present_at_T": present.astype(int),
                "distance": r["nearest_centroid_distance"], "threshold": thr_s,
                "in_basin": in_basin, "in_published_population": pub_mask.astype(int),
            }))

        # ---------------- sensitivity: held-out ICSD under the EXTERNAL rule ----
        # (full-map community label, distance to the full-map centroid of that
        # community, in-basin iff the community is present at T and
        # d <= tau_c(T); full-map outliers and absent-at-T communities ->
        # frontier).  Not used in the figure; quantifies the asymmetry between
        # the held-out (nearest <= T centroid) and external (full-map
        # assignment) treatments on the same population.
        # External records carry `assigned_community` = argmin over ALL 6,756
        # full-map centroids and `nearest_centroid_distance` = distance to that
        # centroid (scripts/analyze_gnome_frontier.py L229, L262-266; centroids
        # from every row carrying the label).  Apply exactly that to the
        # held-out ICSD rows.
        test = years > T
        if "fullmap_cids" not in results:
            fc_ids = sorted({int(c) for c in comms if c >= 0})
            fc_cent = np.vstack([Xp[comms == c].mean(axis=0) for c in fc_ids])
            results["fullmap_cids"] = fc_ids
            results["_fullmap_cent"] = fc_cent
        fc_ids, fc_cent = results["fullmap_cids"], results["_fullmap_cent"]
        hX = Xp[test]
        fm_idx = np.empty(len(hX), dtype=int)
        fm_dist = np.empty(len(hX))
        c2 = (fc_cent ** 2).sum(axis=1)
        kk = 3
        for s in range(0, len(hX), 2000):
            h = hX[s:s + 2000]
            d2 = (h ** 2).sum(axis=1)[:, None] + c2[None, :] - 2.0 * (h @ fc_cent.T)
            cand = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
            exact = np.linalg.norm(h[:, None, :] - fc_cent[cand], axis=2)
            j = exact.argmin(axis=1)
            fm_idx[s:s + 2000] = cand[np.arange(len(h)), j]
            fm_dist[s:s + 2000] = exact[np.arange(len(h)), j]
        fm_lab = np.asarray(fc_ids)[fm_idx]
        fm_thr = np.array([thr_map.get(int(c), np.nan) for c in fm_lab])
        fm_present = ~np.isnan(fm_thr)
        fm_in = fm_present & (fm_dist <= fm_thr)
        in_pub = pd.Series(ids[test]).isin(cm_icsd.icsd_id).to_numpy()
        icsd_block["sensitivity_external_style_rule"] = {
            "description": "held-out ICSD classified exactly like the external records: nearest FULL-map centroid "
                           "(argmin over all 6,756), distance to that centroid, in-basin iff that community is present "
                           "at T and d <= tau_c(T); absent-at-T -> frontier",
            "n": int(test.sum()),
            "n_community_absent_at_T": int((~fm_present).sum()),
            "rate_all_records": rate_block(int(fm_in.sum()), int(test.sum())),
            "rate_present_only": rate_block(int(fm_in.sum()), int(fm_present.sum())),
            "audit_1_4_check_published_population_present_only": rate_block(int(fm_in[in_pub].sum()), int((fm_present & in_pub).sum())),
        }
        say(f"  sensitivity (external-style rule on held-out ICSD): all records rate={fm_in.sum() / test.sum():.4f} "
            f"(n={test.sum()}, absent-at-T={int((~fm_present).sum())}); present-only rate={fm_in.sum() / fm_present.sum():.4f} "
            f"(n={int(fm_present.sum())}); audit-1.4 check on published pop, present-only: "
            f"{fm_in[in_pub].sum() / (fm_present & in_pub).sum():.4f} (n={int((fm_present & in_pub).sum())})")

        results["cutoffs"][str(T)] = {"n_train": n_train, "n_communities_present": int(len(cids)),
                                      "ICSD": icsd_block, "sources": src_blocks}

        # producer-compatible block (same shape as composition_matched_ai_summary.json, coarse matcher)
        by_source = OrderedDict()
        for name in SOURCES:
            by_source[name] = {"unmatched": src_blocks[name]["population_all"]}
        summary_cutoffs.append({"cutoff": T, "matchings": {"coarse": {
            "icsd_unmatched": icsd_block["population_all"], "by_source": by_source}}})

    # ---------------- comparison table + ordering ----------------
    table = []
    ordering = {}
    for T in CUTOFFS:
        blk = results["cutoffs"][str(T)]
        rates = {}
        for name in ORDER:
            if name == "ICSD":
                pb, cb = pub_by_T[T]["icsd_unmatched"], blk["ICSD"]["population_all"]
                n_total = blk["ICSD"]["population_all"]["n"]
            else:
                pb, cb = pub_by_T[T]["by_source"][name]["unmatched"], blk["sources"][name]["population_all"]
                n_total = blk["sources"][name]["n_records"]
            row = {"source": name, "cutoff": T,
                   "published_rate": pb["rate"], "published_k": pb["k"], "published_n": pb["n"],
                   "published_ci95": pb["ci95"],
                   "corrected_rate": cb["rate"], "corrected_k": cb["k"], "corrected_n": cb["n"],
                   "corrected_ci95": cb["ci95"],
                   "delta_pp": 100.0 * (cb["rate"] - pb["rate"])}
            table.append(row)
            rates[name] = cb
        # ordering verdict: point estimates + CI overlap of adjacent pairs
        expected = ["ICSD", "MatterGen", "GNoME", "MP", "JARVIS", "Alexandria"]
        pts = [rates[s]["rate"] for s in expected]
        ok_strict = all(pts[i] > pts[i + 1] for i in range(len(pts) - 1))
        # allow GNoME ~ MP: check ordering with the two treated as a block
        block = ["ICSD", "MatterGen", ("GNoME", "MP"), "JARVIS", "Alexandria"]
        vals = [rates[s]["rate"] if isinstance(s, str) else (rates[s[0]]["rate"], rates[s[1]]["rate"]) for s in block]
        ok_block = (vals[0] > vals[1] > max(vals[2]) and min(vals[2]) > vals[3] > vals[4])
        gm_lo = max(rates["GNoME"]["ci95"][0], rates["MP"]["ci95"][0])
        gm_hi = min(rates["GNoME"]["ci95"][1], rates["MP"]["ci95"][1])
        pairs = {}
        for a, b in [("ICSD", "MatterGen"), ("MatterGen", "GNoME"), ("MatterGen", "MP"), ("GNoME", "MP"),
                     ("GNoME", "JARVIS"), ("MP", "JARVIS"), ("JARVIS", "Alexandria")]:
            lo = max(rates[a]["ci95"][0], rates[b]["ci95"][0])
            hi = min(rates[a]["ci95"][1], rates[b]["ci95"][1])
            pairs[f"{a} vs {b}"] = {"ci_overlap": bool(lo <= hi), "rate_a": rates[a]["rate"], "rate_b": rates[b]["rate"]}
        ordering[str(T)] = {"point_estimate_order_desc": sorted(expected, key=lambda s: -rates[s]["rate"]),
                            "strict_order_ICSD>MatterGen>GNoME>MP>JARVIS>Alexandria": ok_strict,
                            "block_order_ICSD>MatterGen>{GNoME,MP}>JARVIS>Alexandria": ok_block,
                            "GNoME_vs_MP_ci_overlap": bool(gm_lo <= gm_hi),
                            "GNoME_vs_MP_gap_pp": 100.0 * (rates["GNoME"]["rate"] - rates["MP"]["rate"]),
                            "adjacent_pairs": pairs}
        say(f"\n  ordering at T={T}: {' > '.join(ordering[str(T)]['point_estimate_order_desc'])}; "
            f"block order holds={ok_block}; GNoME vs MP CI overlap={ordering[str(T)]['GNoME_vs_MP_ci_overlap']} "
            f"(gap {ordering[str(T)]['GNoME_vs_MP_gap_pp']:+.2f} pp)")

    results["comparison_table"] = table
    results["ordering"] = ordering
    results.pop("_fullmap_cent", None)

    say("\n  comparison table (rate, n): published -> corrected")
    for row in table:
        say(f"    {row['source']:<10s} {row['cutoff']}  {row['published_rate']:.4f} (n={row['published_n']:>6d})  ->  "
            f"{row['corrected_rate']:.4f} [{row['corrected_ci95'][0]:.4f},{row['corrected_ci95'][1]:.4f}] (n={row['corrected_n']:>6d})  "
            f"delta={row['delta_pp']:+.2f} pp")

    # ---------------- outputs ----------------
    summary = {
        "provenance": {
            "script": "notes/review_2026_08/fig3c_corrected_denominators.py",
            "description": "Fig. 3c in-basin rates with every record counted: held-out ICSD = all entries with "
                           "year > T (no formula requirement; nearest <= T centroid; in-basin iff d <= tau_c(T)); "
                           "externals = all records, full-map assigned_community and nearest_centroid_distance, "
                           "in-basin iff community present at T and d <= tau_c(T), else frontier. Same key layout "
                           "as notes/composition_matched_ai_summary.json (coarse matcher) for the fields read by "
                           "scripts/make_fig_5source_calibration_revised.py::render_bar_panel.",
            "threshold_percentile": 95,
            "pca_components": 32, "seed": SEED,
        },
        "cutoffs": summary_cutoffs,
    }
    (HERE / "fig3c_corrected_summary.json").write_text(json.dumps(summary, indent=2))
    (HERE / "fig3c_corrected_table.json").write_text(json.dumps(results, indent=2, default=float))
    rec = pd.concat(record_rows, ignore_index=True)
    rec = rec[["source", "id", "cutoff", "community", "community_present_at_T", "distance", "threshold",
               "in_basin", "in_published_population"]]
    rec.to_csv(HERE / "fig3c_corrected_records.csv", index=False, float_format="%.12g")
    say(f"\nwrote {HERE / 'fig3c_corrected_summary.json'}")
    say(f"wrote {HERE / 'fig3c_corrected_table.json'}")
    say(f"wrote {HERE / 'fig3c_corrected_records.csv'} ({len(rec)} rows)")
    (HERE / "fig3c_corrected_denominators.log").write_text("\n".join(log) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
