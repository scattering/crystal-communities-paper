#!/usr/bin/env python3
"""Recompute the review's source-level checks using repaired projections.

Adapted from notes/review_2026_08/source_complexity_and_sampling.py. Input
directories and thresholds are explicit; original outputs remain untouched.
The C tables include both attempted and successfully projected cohorts.

(A) MatterGen-public in-basin rate by family / task.
(B) Site counts per source for the sampled records, and in-basin rate
    binned by site count (is MatterGen's rate a size artefact?).
(C) Sample-vs-full representativeness for the 5,000-record uniform random
    samples.

Two in-basin conventions are reported everywhere (SI S1.6, last paragraph):
  pooled        - the legacy `outlier_like` column in the records CSVs
                  (single pooled p95 within-community distance = 4.756).
  per_community - the manuscript convention: outlier iff
                  nearest_centroid_distance > thr[assigned_community], with
                  thr from notes/per_community_thresholds_fullmap_p95.json;
                  a record whose community has no threshold is frontier.
                  This is byte-for-byte the rule in
                  scripts/analyze_formula_synth_prior.py::load_external_records.

Inputs
  notes/external_frontier_runs/*/*_frontier_records.csv   (frozen paper records)
  <derived>/<source>_full.csv           one row per candidate in the paper's
                                        candidate pool (filters reproduced from
                                        scripts/analyze_<source>_frontier.py and
                                        the original TACC batch wrappers, not distributed)
  <derived>/<source>_sampled_sites.csv  cell + primitive site counts computed
                                        from the actual structures of the sampled
                                        records (pymatgen get_primitive_structure)
The derived tables were produced from public, non-ICSD source files (GNoME
stable_materials_summary.csv + by_id.zip, Alexandria PBE 2025.07.02 shards
00000/00019/00038, JARVIS jdft_3d-12-12-2022.json, the MP theoretical-only
candidate cache mp_theoretical_candidates_20260427.jsonl, and
microsoft/mattergen data-release/cifs.zip) in a session scratch directory;
none of the source files are stored in this repo.

Usage: python analyze_repaired_source_complexity.py --derived <covariates>
       --external-root <new external runs> --thresholds <new thresholds.json>
       --out <new output directory>
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from crystal_neighbors import FEATURE_VERSION

RECORDS = {}
SLUG = {"GNoME": "gnome", "MatterGen": "mattergen", "MP": "mp", "JARVIS": "jarvis", "Alexandria": "alexandria"}
ORDER = ["GNoME", "MatterGen", "JARVIS", "MP", "Alexandria"]
BINS = [(1, 10, "<=10"), (11, 20, "11-20"), (21, 40, "21-40"), (41, 10**9, ">40")]
Z = 1.959964

PC_THR = {}
POOLED_THR = None
CONV = {"pooled": "in_basin_pooled", "per_community": "in_basin_pc"}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def load_records(name: str) -> pd.DataFrame:
    df = pd.read_csv(RECORDS[name], dtype={"material_id": str}, keep_default_na=False, na_values=[""])
    # The new producer uses per-community outlier_like. This adapted report
    # retains its historical INTERNAL column names so both conventions remain
    # directly comparable in the A/B output tables.
    stated_pc = df["outlier_like"].astype(str).str.lower().eq("true")
    df["outlier_like"] = df["outlier_like_pooled"].astype(str).str.lower().eq("true")
    df["in_basin_pooled"] = ~df["outlier_like"]
    # sanity: the legacy column must equal the pooled rule at the stored threshold
    assert (df["outlier_like"] == (df["nearest_centroid_distance"] > POOLED_THR)).all(), name
    thr = df["assigned_community"].astype(int).map(PC_THR)
    df["outlier_like_pc"] = thr.isna() | (df["nearest_centroid_distance"] > thr)
    assert (df["outlier_like_pc"] == stated_pc).all(), name
    df["in_basin_pc"] = ~df["outlier_like_pc"]
    df = df.rename(columns={"n_sites": "feature_n_sites", "n_elements": "feature_n_elements"})
    return df


def rate_table(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        n, k = len(sub), int(sub[col].sum())
        lo, hi = wilson(k, n)
        rows.append(dict(zip(by, key)) | {"n": n, "n_in_basin": k, "in_basin_rate": k / n, "wilson95_lo": lo, "wilson95_hi": hi})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def crystal_system(spg: float) -> str | None:
    if pd.isna(spg):
        return None
    s = int(spg)
    if not 1 <= s <= 230:
        return None
    if s <= 2:
        return "triclinic"
    if s <= 15:
        return "monoclinic"
    if s <= 74:
        return "orthorhombic"
    if s <= 142:
        return "tetragonal"
    if s <= 167:
        return "trigonal"
    if s <= 194:
        return "hexagonal"
    return "cubic"


def main() -> int:
    global RECORDS, PC_THR, POOLED_THR
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", required=True)
    ap.add_argument("--external-root", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D = Path(args.derived)
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads(args.thresholds.read_text())
    if thresholds.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Use the repaired community threshold file")
    PC_THR = {int(k): float(v) for k, v in thresholds["per_community_p95_threshold"].items()}
    POOLED_THR = float(thresholds["pooled_p95_threshold"])
    RECORDS = {name: args.external_root / slug / f"{'mattergen-public' if slug == 'mattergen' else slug}_frontier_records.csv"
               for name, slug in SLUG.items()}
    frag: list[str] = [f"pooled p95 threshold = {POOLED_THR:.4f}; per-community thresholds for {len(PC_THR)} communities", ""]

    # ------------------------------------------------------------------ (A)
    mg = load_records("MatterGen")
    parts = mg["zip_member"].str.split("/", expand=True)
    assert (parts[0] == "cifs").all() and parts.shape[1] == 4
    assert (parts[1] == mg["family"]).all()
    mg["task"] = parts[2]
    for conv, col in CONV.items():
        a_family = rate_table(mg, ["family"], col).sort_values("n", ascending=False)
        a_task = rate_table(mg, ["task"], col).sort_values("n", ascending=False)
        a_ft = rate_table(mg, ["family", "task"], col).sort_values(["family", "n"], ascending=[True, False])
        n, k = len(mg), int(mg[col].sum())
        lo, hi = wilson(k, n)
        a_all = pd.DataFrame([{"family": "ALL", "n": n, "n_in_basin": k, "in_basin_rate": k / n, "wilson95_lo": lo, "wilson95_hi": hi}])
        a_out = pd.concat([a_family, a_all], ignore_index=True)
        a_out.to_csv(OUT / f"A_mattergen_in_basin_by_family_{conv}.csv", index=False)
        a_task.to_csv(OUT / f"A_mattergen_in_basin_by_task_{conv}.csv", index=False)
        a_ft.to_csv(OUT / f"A_mattergen_in_basin_by_family_task_{conv}.csv", index=False)
        frag += [f"## A. MatterGen-public by family [{conv}]", md_table(a_out), "",
                 f"## A. MatterGen-public by task (zip path cifs/<family>/<task>/) [{conv}]", md_table(a_task), "",
                 f"## A. MatterGen-public by family x task [{conv}]", md_table(a_ft), ""]

    # ------------------------------------------------------------------ (B)
    detail = []
    for name in ORDER:
        rec = load_records(name)
        samp = pd.read_csv(D / f"{SLUG[name]}_sampled_sites.csv", dtype={"material_id": str})
        if name == "MatterGen":
            m = rec.merge(samp, on=["material_id", "zip_member"], how="left", validate="one_to_one")
            m["task"] = m["zip_member"].str.split("/", expand=True)[2]
        else:
            m = rec.merge(samp, on="material_id", how="left", validate="one_to_one")
        if (m["n_sites_cell"].isna().any() or m["n_elements"].isna().any()
                or (m["n_sites_cell"] != m["feature_n_sites"]).any()
                or (m["n_elements"] != m["feature_n_elements"]).any()):
            raise ValueError(f"{name}: raw-structure covariates disagree with repaired feature records")
        if name == "GNoME":  # cross-check NSites in summary CSV vs. parsed CIF
            full = pd.read_csv(D / "gnome_full.csv", dtype={"material_id": str})
            chk = m.merge(full[["material_id", "n_sites"]], on="material_id", how="left")
            mism = int((chk["n_sites"] != chk["n_sites_cell"]).sum())
            frag.append(f"GNoME NSites (summary CSV) vs parsed by_id CIF site count mismatches: {mism} of {len(chk)}")
        m["source"] = name
        detail.append(m[["source", "material_id", "n_sites_cell", "n_sites_primitive", "n_elements", "outlier_like", "outlier_like_pc",
                         "in_basin_pooled", "in_basin_pc"] + (["zip_member", "family", "task"] if name == "MatterGen" else [])])
    det = pd.concat(detail, ignore_index=True)
    # Cross-validate the per-community flag against the coordinator's
    # notes/review_2026_08/quadrant_assignments.csv (in_basin = per-community rule, reproduces Fig. 4).
    qa_path = OUT / "quadrant_assignments.csv"
    if qa_path.exists():
        qa = pd.read_csv(qa_path, dtype={"material_id": str})
        qa["in_basin"] = qa["in_basin"].astype(str).str.lower().eq("true")
        mg_q = qa[qa["source"] == "MatterGen"][["zip_member", "in_basin"]]
        ot_q = qa[qa["source"] != "MatterGen"][["source", "material_id", "in_basin"]]
        chk_mg = det[det["source"] == "MatterGen"].merge(mg_q, on="zip_member", how="left", validate="one_to_one")
        chk_ot = det[det["source"] != "MatterGen"].merge(ot_q, on=["source", "material_id"], how="left", validate="one_to_one")
        n_dis = int((chk_mg["in_basin"] != chk_mg["in_basin_pc"]).sum() + (chk_ot["in_basin"] != chk_ot["in_basin_pc"]).sum())
        n_miss = int(chk_mg["in_basin"].isna().sum() + chk_ot["in_basin"].isna().sum())
        frag.append(f"per-community flag vs quadrant_assignments.csv in_basin: {n_dis} disagreements, {n_miss} unmatched, of {len(det)}")
        assert n_dis == 0 and n_miss == 0
    det["n_sites"] = det["n_sites_cell"]
    det["in_basin"] = det["in_basin_pc"]
    det["outlier_like_pooled"] = det["outlier_like"]
    det[["source", "material_id", "n_sites", "in_basin", "outlier_like_pooled"]].to_csv(OUT / "site_counts_by_source.csv", index=False)
    det.drop(columns=["n_sites", "in_basin", "outlier_like_pooled"]).to_csv(OUT / "site_counts_by_source_detail.csv", index=False)
    frag.append(f"site_counts_by_source.csv rows: {len(det)}; unmatched n_sites: {int(det['n_sites_cell'].isna().sum())}; "
                f"unmatched primitive: {int(det['n_sites_primitive'].isna().sum())}")

    def site_summary(col: str) -> pd.DataFrame:
        rows = []
        for name in ORDER:
            s = det.loc[det["source"] == name, col].dropna()
            rows.append({"source": name, "n_records": int((det["source"] == name).sum()), "n_matched": len(s),
                         "min": int(s.min()), "p10": float(s.quantile(0.10)), "median": float(s.median()),
                         "p90": float(s.quantile(0.90)), "max": int(s.max()),
                         "frac_le_20": float((s <= 20).mean()), "frac_le_10": float((s <= 10).mean())})
        return pd.DataFrame(rows)

    b_cell = site_summary("n_sites_cell")
    b_prim = site_summary("n_sites_primitive")
    b_cell.to_csv(OUT / "B_site_count_summary_cell.csv", index=False)
    b_prim.to_csv(OUT / "B_site_count_summary_primitive.csv", index=False)
    frag += ["## B. Site counts per source (cell as distributed)", md_table(b_cell), "",
             "## B. Site counts per source (pymatgen primitive cell)", md_table(b_prim), ""]

    overall = []
    for conv, ind in CONV.items():
        for name in ORDER:
            s = det[det["source"] == name]
            n, k = len(s), int(s[ind].sum())
            wl, wh = wilson(k, n)
            overall.append({"convention": conv, "source": name, "n": n, "n_in_basin": k, "in_basin_rate": k / n, "wilson95_lo": wl, "wilson95_hi": wh})
    overall = pd.DataFrame(overall)
    overall.to_csv(OUT / "B_overall_in_basin_by_source.csv", index=False)
    frag += ["## B. Overall full-map in-basin rate by source, both conventions", md_table(overall), ""]

    def binned(col: str, ind: str) -> pd.DataFrame:
        rows = []
        for lo_, hi_, lab in BINS:
            for name in ORDER:
                s = det[(det["source"] == name) & (det[col] >= lo_) & (det[col] <= hi_)]
                n, k = len(s), int(s[ind].sum())
                wl, wh = wilson(k, n)
                rows.append({"site_bin": lab, "source": name, "n": n, "n_in_basin": k,
                             "in_basin_rate": (k / n if n else np.nan), "wilson95_lo": wl, "wilson95_hi": wh})
        return pd.DataFrame(rows)

    for conv, ind in CONV.items():
        b_bin_cell = binned("n_sites_cell", ind)
        b_bin_prim = binned("n_sites_primitive", ind)
        b_bin_cell.to_csv(OUT / f"B_in_basin_by_site_bin_cell_{conv}.csv", index=False)
        b_bin_prim.to_csv(OUT / f"B_in_basin_by_site_bin_primitive_{conv}.csv", index=False)
        frag += [f"## B. In-basin rate by site-count bin (cell as distributed) [{conv}]", md_table(b_bin_cell), "",
                 f"## B. In-basin rate by site-count bin (primitive cell) [{conv}]", md_table(b_bin_prim), ""]
        for label, tab in (("cell", b_bin_cell), ("primitive", b_bin_prim)):
            lines = [f"## B. Source ordering by in-basin rate within each bin ({label}) [{conv}]"]
            for _, _, lab in BINS:
                t = tab[tab["site_bin"] == lab].sort_values("in_basin_rate", ascending=False)
                lines.append(f"- {lab}: " + " > ".join(f"{r.source} {r.in_basin_rate:.3f} (n={r.n})" for r in t.itertuples()))
            frag += lines + [""]
        # direct standardisation to MatterGen's cell-size distribution, and a <=20-site-only comparison
        mg_bins = det[det["source"] == "MatterGen"]["n_sites_cell"]
        w = {lab: float(((mg_bins >= lo_) & (mg_bins <= hi_)).mean()) for lo_, hi_, lab in BINS}
        rows = []
        for name in ORDER:
            t = b_bin_cell[b_bin_cell["source"] == name].set_index("site_bin")
            std = (sum(w[lab] * t.loc[lab, "in_basin_rate"] for lab in w if w[lab] > 0)
                   if all(w[lab] == 0 or not pd.isna(t.loc[lab, "in_basin_rate"]) for lab in w) else np.nan)
            crude = det[det["source"] == name][ind].mean()
            le20 = det[(det["source"] == name) & (det["n_sites_cell"] <= 20)]
            k20, n20 = int(le20[ind].sum()), len(le20)
            wl, wh = wilson(k20, n20)
            rows.append({"source": name, "crude_in_basin": crude, "standardised_to_mattergen_bins": std,
                         "n_le20": n20, "in_basin_le20": k20 / n20 if n20 else np.nan, "le20_wilson95_lo": wl, "le20_wilson95_hi": wh})
        b_std = pd.DataFrame(rows)
        b_std.to_csv(OUT / f"B_size_standardised_rates_{conv}.csv", index=False)
        frag += [f"## B. Size-standardised in-basin rates [{conv}] (weights = MatterGen cell-size bin shares {json.dumps({k: round(v, 3) for k, v in w.items()})})",
                 md_table(b_std), ""]

    # ------------------------------------------------------------------ (C)
    c_rows, c_ks = [], []
    for name, cohort in ((name, cohort) for name in ("Alexandria", "JARVIS", "MP", "GNoME")
                         for cohort in ("attempted", "scored")):
        full = pd.read_csv(D / f"{SLUG[name]}_full.csv", dtype={"material_id": str})
        rec = (load_records(name) if cohort == "scored" else
               pd.read_csv(args.external_root / SLUG[name] / "attempted_cohort.csv", dtype={"material_id": str}))
        samp = full[full["material_id"].isin(set(rec["material_id"]))]
        assert len(samp) == len(rec), (name, len(samp), len(rec))
        for var in ("n_elements", "n_sites", "e_above_hull", "spg_number"):
            f, s = full[var].dropna().astype(float), samp[var].dropna().astype(float)
            ks = stats.ks_2samp(s, f, alternative="two-sided", method="asymp")
            qf, qs = f.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values, s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
            c_rows.append({"source": name, "cohort": cohort, "variable": var, "n_full": len(f), "n_sample": len(s),
                           "full_q10": qf[0], "full_q25": qf[1], "full_q50": qf[2], "full_q75": qf[3], "full_q90": qf[4],
                           "samp_q10": qs[0], "samp_q25": qs[1], "samp_q50": qs[2], "samp_q75": qs[3], "samp_q90": qs[4],
                           "full_mean": f.mean(), "samp_mean": s.mean(), "KS_D": ks.statistic, "KS_p": ks.pvalue})
        fc = full["spg_number"].map(crystal_system).value_counts()
        sc = samp["spg_number"].map(crystal_system).value_counts()
        cats = [c for c in ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"] if c in fc.index]
        obs = np.array([sc.get(c, 0) for c in cats], dtype=float)
        exp = np.array([fc[c] for c in cats], dtype=float) / fc[cats].sum() * obs.sum()
        chi = stats.chisquare(obs, exp)
        c_ks.append({"source": name, "cohort": cohort, "test": "crystal_system_chi2", "n_full": int(fc[cats].sum()), "n_sample": int(obs.sum()),
                     "statistic": chi.statistic, "dof": len(cats) - 1, "p": chi.pvalue,
                     "full_shares": json.dumps({c: round(float(fc[c] / fc[cats].sum()), 3) for c in cats}),
                     "sample_shares": json.dumps({c: round(float(sc.get(c, 0) / obs.sum()), 3) for c in cats})})
    c_df = pd.DataFrame(c_rows)
    c_df.to_csv(OUT / "C_sample_vs_full_quantiles_ks.csv", index=False)
    c_chi = pd.DataFrame(c_ks)
    c_chi.to_csv(OUT / "C_crystal_system_chi2.csv", index=False)
    show = c_df[["source", "cohort", "variable", "n_full", "n_sample", "full_q10", "samp_q10", "full_q25", "samp_q25", "full_q50", "samp_q50",
                 "full_q75", "samp_q75", "full_q90", "samp_q90", "full_mean", "samp_mean", "KS_D", "KS_p"]]
    frag += ["## C. Sample vs full: quantiles + two-sample KS", md_table(show, "{:.4g}"), "",
             "## C. Crystal-system chi-square (sample vs full proportions)",
             md_table(c_chi[["source", "cohort", "n_full", "n_sample", "statistic", "dof", "p", "full_shares", "sample_shares"]], "{:.3g}"), ""]

    (OUT / "tables_autogen.md").write_text("\n".join(frag))
    print("\n".join(frag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
