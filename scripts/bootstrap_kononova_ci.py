#!/usr/bin/env python3
"""Recompute the Kononova bottom-vs-top decile positive-rate ratio and its
5,000-resample nonparametric bootstrap 95% CI for the 1990 / 2000 / 2010
cutoffs (main text Discussion; SI §S3.3–S3.4).

Inputs are the per-formula class-centred CSVs written by
``scripts/analyze_kononova_validation.py`` on Stampede3
(``$WORK/icsd_graph_runs/kononova_validation_20260419/split_<T>/
kononova_class_centered_first_reports.csv``), pulled to
``notes/review_2026_08/kononova/stampede_pull/split_<T>/``.

Method (identical to the archived producer ``bootstrap_kononova.py``,
Stampede3 job 3060583, 2026-04-26):

* unit of resampling = one first-report reduced formula (one CSV row);
* each of the ``N_BOOT`` draws resamples all N rows with replacement;
* the 10th / 90th percentile thresholds of ``class_centered_A_i`` are
  RECOMPUTED inside every draw (``np.quantile``, linear interpolation),
  bottom = score <= q10, top = score >= q90;
* ratio = mean(in_kononova | bottom) / mean(in_kononova | top);
* CI = 2.5 / 97.5 percentiles of the finite bootstrap ratios.

Two seed conventions are reported:

* ``seed42``        : ``np.random.default_rng(42)`` for every cutoff (the
                      convention requested by the audit);
* ``seed42_plus_T`` : ``np.random.default_rng(42 + T)`` — what the archived
                      producer actually used (``RNG_SEED + int(cutoff)``),
                      so this run should reproduce the archived
                      ``bootstrap_decile_ratio_ci.json`` to the last digit.

A fixed-threshold sensitivity variant (deciles taken once from the full
sample and held fixed across draws) is also reported so readers can see
how much of the CI width comes from re-estimating the thresholds.

Usage:
    python scripts/bootstrap_kononova_ci.py [--input-dir DIR] [--n-boot 5000] [--out results.json]

    DIR holds split_{1990,2000,2010}/kononova_class_centered_first_reports.csv (the per-formula
    class-centred tables written by scripts/analyze_kononova_validation.py; released with the
    Zenodo deposit) and, optionally, bootstrap_decile_ratio_ci.json for the exact-reproduction check.
    Default DIR: notes/review_2026_08/kononova/stampede_pull relative to the repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PULL = REPO / "notes/review_2026_08/kononova/stampede_pull"  # overridden by --input-dir
CUTOFFS = (1990, 2000, 2010)
QUOTED = {  # main text + SI §S3.3 / §S3.4 as submitted
    1990: {"ratio": 1.08, "ci": [0.88, 1.31], "N": 23764},
    2000: {"ratio": 1.15, "ci": [0.90, 1.44], "N": 14198},
    2010: {"ratio": 1.40, "ci": [1.02, 1.98], "N": 4227},
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    score, label = [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                a = float(row["class_centered_A_i"])
                y = int(row["in_kononova"])
            except (KeyError, ValueError, TypeError):
                continue
            score.append(a)
            label.append(y)
    return np.asarray(score, dtype=float), np.asarray(label, dtype=int)


def decile_rates(score: np.ndarray, label: np.ndarray,
                 q10: float | None = None, q90: float | None = None) -> tuple[float, float, float]:
    """(bottom_rate, top_rate, ratio). Thresholds recomputed unless given."""
    if q10 is None:
        q10 = float(np.quantile(score, 0.10))
    if q90 is None:
        q90 = float(np.quantile(score, 0.90))
    lo = score <= q10
    hi = score >= q90
    lo_rate = float(label[lo].mean()) if lo.any() else float("nan")
    hi_rate = float(label[hi].mean()) if hi.any() else float("nan")
    ratio = lo_rate / hi_rate if hi_rate > 0 else float("inf")
    return lo_rate, hi_rate, ratio


def bootstrap(score: np.ndarray, label: np.ndarray, n_boot: int, seed: int,
              fixed_thresholds: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    n = len(score)
    q10 = q90 = None
    if fixed_thresholds:
        q10 = float(np.quantile(score, 0.10))
        q90 = float(np.quantile(score, 0.90))
    ratios = np.empty(n_boot)
    bots = np.empty(n_boot)
    tops = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bots[b], tops[b], ratios[b] = decile_rates(score[idx], label[idx], q10, q90)
    finite = np.isfinite(ratios)
    r = ratios[finite]
    return {
        "seed": int(seed),
        "n_boot": int(n_boot),
        "n_finite": int(finite.sum()),
        "thresholds_recomputed_per_draw": not fixed_thresholds,
        "ratio_mean": float(r.mean()),
        "ratio_median": float(np.median(r)),
        "ratio_ci95": [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))],
        "bottom_rate_mean": float(np.nanmean(bots)),
        "top_rate_mean": float(np.nanmean(tops)),
        "p_ratio_le_1": float(np.mean(r <= 1.0)),
    }


def main() -> int:
    global PULL
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(PULL))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default=str(PULL.parent / "kononova_bootstrap_results.json"))
    args = ap.parse_args()
    PULL = Path(args.input_dir)

    archived_path = PULL / "bootstrap_decile_ratio_ci.json"
    archived = json.loads(archived_path.read_text()) if archived_path.exists() else None

    out = {
        "method": {
            "unit": "first-report reduced formula (one row of kononova_class_centered_first_reports.csv)",
            "n_boot": args.n_boot,
            "deciles": "10th/90th percentiles of class_centered_A_i recomputed within each resample "
                       "(np.quantile linear); bottom = score <= q10, top = score >= q90",
            "ci": "percentile bootstrap, 2.5/97.5 of finite ratios",
            "seed_conventions": {"seed42": "default_rng(42) every cutoff",
                                 "seed42_plus_T": "default_rng(42 + T), as in archived producer"},
        },
        "inputs": {},
        "cutoffs": {},
    }

    for T in CUTOFFS:
        csv_path = PULL / f"split_{T}" / "kononova_class_centered_first_reports.csv"
        score, label = load_split(csv_path)
        out["inputs"][str(T)] = {"csv": str(csv_path.relative_to(REPO)), "md5": md5(csv_path),
                                 "n_rows": int(len(score)), "n_positive": int(label.sum())}
        lo, hi, ratio = decile_rates(score, label)
        q10 = float(np.quantile(score, 0.10))
        q90 = float(np.quantile(score, 0.90))
        rec = {
            "N": int(len(score)),
            "n_positive": int(label.sum()),
            "positive_fraction_overall": float(label.mean()),
            "q10_threshold": q10,
            "q90_threshold": q90,
            "n_bottom_decile": int((score <= q10).sum()),
            "n_top_decile": int((score >= q90).sum()),
            "bottom_decile_positive_fraction": lo,
            "top_decile_positive_fraction": hi,
            "ratio": ratio,
            "bootstrap_seed42": bootstrap(score, label, args.n_boot, 42),
            "bootstrap_seed42_plus_T": bootstrap(score, label, args.n_boot, 42 + T),
            "bootstrap_seed42_fixed_thresholds": bootstrap(score, label, args.n_boot, 42, fixed_thresholds=True),
            "quoted_in_manuscript": QUOTED[T],
        }
        if archived is not None:
            a = archived["splits"][str(T)]
            rec["archived_stampede_job_3060583"] = {
                "ratio": a["point_estimate"]["ratio"],
                "ratio_ci95": a["bootstrap"]["ratio_ci95"],
                "ratio_mean": a["bootstrap"]["ratio_mean"],
                "N": a["n_total_formulas"],
            }
            b = rec["bootstrap_seed42_plus_T"]
            rec["reproduces_archived_exactly"] = bool(
                np.allclose(b["ratio_ci95"], a["bootstrap"]["ratio_ci95"], rtol=0, atol=1e-12)
                and abs(b["ratio_mean"] - a["bootstrap"]["ratio_mean"]) < 1e-12
                and abs(ratio - a["point_estimate"]["ratio"]) < 1e-12
            )
        out["cutoffs"][str(T)] = rec

        q = QUOTED[T]
        b42 = rec["bootstrap_seed42"]
        bT = rec["bootstrap_seed42_plus_T"]
        bF = rec["bootstrap_seed42_fixed_thresholds"]
        print(f"T={T}: N={len(score)} pos={int(label.sum())} "
              f"bottom={lo:.4f} top={hi:.4f} ratio={ratio:.3f} "
              f"| seed42 CI=[{b42['ratio_ci95'][0]:.3f}, {b42['ratio_ci95'][1]:.3f}] "
              f"| seed42+T CI=[{bT['ratio_ci95'][0]:.3f}, {bT['ratio_ci95'][1]:.3f}] "
              f"| fixed-thr CI=[{bF['ratio_ci95'][0]:.3f}, {bF['ratio_ci95'][1]:.3f}] "
              f"| quoted {q['ratio']} {q['ci']}"
              + (f" | reproduces archived: {rec['reproduces_archived_exactly']}" if archived else ""))

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
