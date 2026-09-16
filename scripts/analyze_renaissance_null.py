#!/usr/bin/env python3
"""Recompute the renaissance-survey year-shuffle null with the PRODUCTION scoring.

Background (2026-09 review audit)
---------------------------------
`notes/renaissance_null_summary.json` (committed in e95d502, no producer script
committed) reports per-rank null bands for the top-20 step-change scores of
the renaissance survey (SI §S8.4).  Its "observed" scores do NOT match the
production survey (`notes/renaissance_survey_top.json`, produced by
`scripts/analyze_renaissance_survey.py`): rank 1 is community 160 at 24,864.5
in the null file versus community 2904 at 19,600.0 in production.

Reverse-engineering (see renaissance_null_recompute.md, section A) shows the
archived file scored every community with a *gap-year* window convention

    pre  = [ey-10, ey-1]   (10 years, event year excluded)
    post = [ey+1,  ey+10]  (10 years)

whereas the production `best_event_year` uses

    pre  = (ey-10, ey]  = [ey-9, ey]   (10 years, event year INCLUDED in pre)
    post = (ey, ey+10]  = [ey+1, ey+10]

Everything else (population of 148,425 dated community-assigned entries,
N_MIN = 50 -> 378 eligible communities, event years 1970-2010, score =
n_post^2 / n_pre or n_post when n_pre == 0, `random.Random(42).shuffle` of
the year column, 200 shuffles, `np.percentile` linear interpolation) is
identical, and this script reproduces the archived bands to machine
precision when the gap-year convention is switched on (`--legacy-check`).

This script:
  1. imports the production scorer and constants from
     scripts/analyze_renaissance_survey.py (no re-implementation of the
     scoring rule for the observed table);
  2. verifies that the observed top-20 reproduces
     notes/renaissance_survey_top.json exactly (community, event year, n_pre,
     n_post, score);
  3. verifies a vectorised scorer (used inside the shuffle loop for speed)
     agrees with `best_event_year` on every eligible community;
  4. runs the year-shuffle null with the production convention for 200 and
     1,000 shuffles (seed 42) and reports per-rank p50 / p95 / p99;
  5. optionally re-runs the legacy gap-year convention to confirm the
     diagnosis of the archived file.

Population permuted
-------------------
The year column is permuted over ALL 148,425 dated community-assigned ICSD
entries (year in 1900-2025, community >= 0 -- exactly the rows
`load_community_assignments` returns), not only the members of the 378
eligible communities.  Reasons: (i) this is what the SI text describes
("permutations of the same 148,425 community-assigned ICSD entries");
(ii) it preserves the global ICSD year distribution that the null is meant
to represent, so an eligible community's shuffled years are draws from the
whole record rather than from the size>=50 subset, which is slightly
younger; (iii) community sizes are invariant under the permutation, so the
eligible set (378 communities) is identical in every shuffle.  A sensitivity
run permuting only within the eligible subset is also reported.

Usage
-----
    python scripts/analyze_renaissance_null.py \
        [--n-shuffles 200 1000] [--seed 42] [--legacy-check]

Output: notes/review_2026_08/renaissance_null_recomputed.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analyze_renaissance_survey import (  # noqa: E402
    COMMUNITY_ASSIGN,
    EVENT_YEARS,
    N_MIN,
    OUT_TABLE_JSON,
    TOP_K,
    WINDOW,
    best_event_year,
    load_community_assignments,
)

LEGACY_JSON = REPO / "notes/renaissance_null_summary.json"
OUT_JSON = REPO / "notes/review_2026_08/renaissance_null_recomputed.json"


# --------------------------------------------------------------------------
# Vectorised scorer (same rule as best_event_year; checked against it below)
# --------------------------------------------------------------------------
class VectorScorer:
    """Score every eligible community for every candidate event year at once.

    Convention "production":  pre = [ey-W+1, ey], post = [ey+1, ey+W]
    Convention "gap_year":    pre = [ey-W,   ey-1], post = [ey+1, ey+W]
    """

    def __init__(self, n_comm: int, ymin: int, ymax: int, event_years: list[int], window: int):
        self.n_comm = n_comm
        self.ymin = ymin
        self.ny = ymax - ymin + 1
        self.ey = np.asarray(event_years)
        self.W = window

    def _range_sum(self, C: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        lo_i = np.clip(lo - self.ymin, 0, self.ny)
        hi_i = np.clip(hi - self.ymin + 1, 0, self.ny)
        return C[:, hi_i] - C[:, lo_i]

    def score(self, comm_idx: np.ndarray, years: np.ndarray, convention: str = "production"):
        H = np.zeros((self.n_comm, self.ny))
        np.add.at(H, (comm_idx, years - self.ymin), 1.0)
        C = np.concatenate([np.zeros((self.n_comm, 1)), np.cumsum(H, axis=1)], axis=1)
        ey, W = self.ey, self.W
        if convention == "production":
            n_pre = self._range_sum(C, ey - W + 1, ey)
        elif convention == "gap_year":
            n_pre = self._range_sum(C, ey - W, ey - 1)
        else:
            raise ValueError(convention)
        n_post = self._range_sum(C, ey + 1, ey + W)
        # Same float arithmetic as best_event_year: fold = rate_post / rate_pre,
        # score = fold * n_post (NOT the algebraically equal n_post**2 / n_pre,
        # whose rounding differs and can flip near-ties between event years,
        # e.g. community 5979: 143.99999999999997 at 1992 vs 144.0 at 2001).
        rate_pre = n_pre / W
        rate_post = n_post / W
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(n_pre > 0, (rate_post / rate_pre) * n_post, n_post)
        j = np.argmax(score, axis=1)  # first maximum, matching `score > best` in production
        rows = np.arange(self.n_comm)
        return score[rows, j], ey[j], n_pre[rows, j].astype(int), n_post[rows, j].astype(int)


def percentile_table(tops: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "p50": np.percentile(tops, 50, axis=0),
        "p95": np.percentile(tops, 95, axis=0),
        "p99": np.percentile(tops, 99, axis=0),
    }


def run_null(scorer: VectorScorer, comm_idx_elig: np.ndarray, elig_mask: np.ndarray,
             years_all: np.ndarray, n_shuffles: int, seed: int, convention: str,
             permute_population: str = "all") -> np.ndarray:
    """Return (n_shuffles, TOP_K) array of sorted top-K scores per shuffle.

    permute_population = "all": shuffle the year column over every dated
    community-assigned entry, then score the eligible communities.
    permute_population = "eligible": shuffle only within the eligible subset.
    The shuffle uses random.Random(seed).shuffle on a Python list, the same
    idiom as scripts/analyze_temporal_null.py and the archived null.
    """
    rng = random.Random(seed)
    base = years_all.tolist() if permute_population == "all" else years_all[elig_mask].tolist()
    tops = np.zeros((n_shuffles, TOP_K))
    for t in range(n_shuffles):
        # Fresh copy of the ORIGINAL year list each iteration, then shuffle --
        # the exact idiom of scripts/analyze_temporal_null.py (and of the
        # archived producer: this reproduces its bands to machine precision).
        years_t = base[:]
        rng.shuffle(years_t)
        perm = np.asarray(years_t)
        yv = perm[elig_mask] if permute_population == "all" else perm
        s, _, _, _ = scorer.score(comm_idx_elig, yv, convention)
        tops[t] = np.sort(s)[::-1][:TOP_K]
    return tops


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-shuffles", type=int, nargs="+", default=[200, 1000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--legacy-check", action="store_true",
                    help="also re-run the gap-year convention and compare to the archived JSON")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    t0 = time.time()
    records = load_community_assignments(COMMUNITY_ASSIGN)
    years_all = np.asarray([y for _, y, _ in records])
    comms_all = np.asarray([c for _, _, c in records])
    print(f"{len(records)} dated community-assigned entries (production population)")

    # ---- observed table via the PRODUCTION function ----------------------
    by_comm: dict[int, Counter] = defaultdict(Counter)
    for _, y, c in records:
        by_comm[c][y] += 1
    eligible = sorted(c for c, h in by_comm.items() if sum(h.values()) >= N_MIN)
    print(f"{len(by_comm)} communities; {len(eligible)} eligible (size >= {N_MIN})")

    observed = []
    for c in eligible:
        ey, sc = best_event_year(by_comm[c], EVENT_YEARS, WINDOW)
        observed.append({"community": c, "size": sum(by_comm[c].values()), **sc})
    observed.sort(key=lambda r: r["score"], reverse=True)  # stable, as in production
    obs_top = observed[:TOP_K]

    # ---- check 1: reproduce notes/renaissance_survey_top.json exactly ----
    prod = json.loads(OUT_TABLE_JSON.read_text())["top"]
    assert len(prod) == TOP_K
    mismatches = []
    for r, (a, b) in enumerate(zip(obs_top, prod), 1):
        same = (a["community"] == b["community"] and a["event_year"] == b["event_year"]
                and a["n_pre"] == b["n_pre"] and a["n_post"] == b["n_post"]
                and abs(a["score"] - b["score"]) < 1e-9)
        if not same:
            mismatches.append((r, a, b))
    if mismatches:
        for m in mismatches:
            print("MISMATCH", m)
        raise SystemExit("observed top-20 does not reproduce notes/renaissance_survey_top.json")
    print(f"check 1 OK: observed top-{TOP_K} reproduces {OUT_TABLE_JSON.name} exactly "
          f"(community, event_year, n_pre, n_post, score)")

    # ---- check 2: vectorised scorer == best_event_year on every community --
    cidx = {c: i for i, c in enumerate(eligible)}
    elig_mask = np.isin(comms_all, eligible)
    comm_idx_elig = np.asarray([cidx[c] for c in comms_all[elig_mask]])
    scorer = VectorScorer(len(eligible), int(years_all.min()), int(years_all.max()), EVENT_YEARS, WINDOW)
    vs, vey, vpre, vpost = scorer.score(comm_idx_elig, years_all[elig_mask], "production")
    for c in eligible:
        ey, sc = best_event_year(by_comm[c], EVENT_YEARS, WINDOW)
        i = cidx[c]
        assert abs(vs[i] - sc["score"]) < 1e-9 and vey[i] == ey and vpre[i] == sc["n_pre"] and vpost[i] == sc["n_post"], c
    print(f"check 2 OK: vectorised scorer agrees with best_event_year on all {len(eligible)} eligible communities")

    out = {
        "produced_by": "scripts/analyze_renaissance_null.py",
        "scoring": {
            "source": "scripts/analyze_renaissance_survey.py::best_event_year",
            "window": WINDOW,
            "event_years_range": [min(EVENT_YEARS), max(EVENT_YEARS)],
            "pre_window": "ey-W < y <= ey (event year included in pre)",
            "post_window": "ey < y <= ey+W",
            "score": "(rate_post/rate_pre)*n_post = n_post^2/n_pre; n_post when n_pre == 0",
            "n_min": N_MIN,
            "top_k": TOP_K,
        },
        "population": {
            "n_entries_permuted": int(len(records)),
            "description": "all dated community-assigned ICSD entries (year 1900-2025, community >= 0); "
                           "year column permuted over this whole set; scoring restricted to eligible communities",
            "n_communities_total": len(by_comm),
            "n_communities_eligible": len(eligible),
        },
        "seed": args.seed,
        "shuffle_rng": "random.Random(seed).shuffle on the year list (same idiom as scripts/analyze_temporal_null.py)",
        "percentile_method": "numpy.percentile, linear interpolation",
        "observed_top_k": [
            {"rank": r, "community": o["community"], "size": o["size"], "event_year": o["event_year"],
             "n_pre": o["n_pre"], "n_post": o["n_post"], "fold": (None if o["fold"] == float("inf") else o["fold"]),
             "score": o["score"]}
            for r, o in enumerate(obs_top, 1)
        ],
        "runs": {},
    }

    # ---- production-convention null runs ----------------------------------
    for n_sh in args.n_shuffles:
        t1 = time.time()
        tops = run_null(scorer, comm_idx_elig, elig_mask, years_all, n_sh, args.seed, "production", "all")
        pt = percentile_table(tops)
        rows = []
        for r, o in enumerate(obs_top):
            rows.append({
                "rank": r + 1, "community": o["community"], "observed_score": o["score"],
                "null_p50": float(pt["p50"][r]), "null_p95": float(pt["p95"][r]), "null_p99": float(pt["p99"][r]),
                "null_max": float(tops[:, r].max()),
                "ratio_obs_over_p95": float(o["score"] / pt["p95"][r]),
                "ratio_obs_over_p99": float(o["score"] / pt["p99"][r]),
                "exceeds_p95": bool(o["score"] > pt["p95"][r]),
                "exceeds_p99": bool(o["score"] > pt["p99"][r]),
                "exceeds_null_max": bool(o["score"] > tops[:, r].max()),
            })
        key = f"production_n{n_sh}"
        out["runs"][key] = {
            "convention": "production", "permute_population": "all", "n_shuffles": n_sh,
            "observed_top_k_vs_null": rows,
            "summary": {
                "n_observed_exceed_null_p95": sum(x["exceeds_p95"] for x in rows),
                "n_observed_exceed_null_p99": sum(x["exceeds_p99"] for x in rows),
                "n_observed_exceed_null_max": sum(x["exceeds_null_max"] for x in rows),
                "rank1_ratio_p95": rows[0]["ratio_obs_over_p95"], "rank1_ratio_p99": rows[0]["ratio_obs_over_p99"],
                "rank20_ratio_p95": rows[-1]["ratio_obs_over_p95"], "rank20_ratio_p99": rows[-1]["ratio_obs_over_p99"],
                "global_null_max_top1": float(tops[:, 0].max()),
            },
            "elapsed_s": round(time.time() - t1, 1),
        }
        s = out["runs"][key]["summary"]
        print(f"{key}: rank1 obs {rows[0]['observed_score']:.1f} p50/p95/p99 = {rows[0]['null_p50']:.1f}/"
              f"{rows[0]['null_p95']:.1f}/{rows[0]['null_p99']:.1f}  ratios {s['rank1_ratio_p95']:.2f}x / {s['rank1_ratio_p99']:.2f}x | "
              f"rank20 obs {rows[-1]['observed_score']:.1f} p95/p99 = {rows[-1]['null_p95']:.1f}/{rows[-1]['null_p99']:.1f} "
              f"ratios {s['rank20_ratio_p95']:.2f}x / {s['rank20_ratio_p99']:.2f}x | exceed p95 {s['n_observed_exceed_null_p95']}/20, "
              f"p99 {s['n_observed_exceed_null_p99']}/20, null max {s['n_observed_exceed_null_max']}/20 "
              f"({out['runs'][key]['elapsed_s']} s)")

    # ---- sensitivity: permute within the eligible subset only --------------
    n_sh = args.n_shuffles[0]
    tops = run_null(scorer, comm_idx_elig, elig_mask, years_all, n_sh, args.seed, "production", "eligible")
    pt = percentile_table(tops)
    out["runs"][f"sensitivity_eligible_only_n{n_sh}"] = {
        "convention": "production", "permute_population": "eligible", "n_shuffles": n_sh,
        "n_entries_permuted": int(elig_mask.sum()),
        "per_rank": [{"rank": r + 1, "observed_score": o["score"], "null_p50": float(pt["p50"][r]),
                      "null_p95": float(pt["p95"][r]), "null_p99": float(pt["p99"][r]),
                      "exceeds_p99": bool(o["score"] > pt["p99"][r])} for r, o in enumerate(obs_top)],
    }
    print(f"sensitivity (permute within {int(elig_mask.sum())} eligible-community entries, n={n_sh}): "
          f"rank1 p95/p99 = {pt['p95'][0]:.1f}/{pt['p99'][0]:.1f}; rank20 p95/p99 = {pt['p95'][19]:.1f}/{pt['p99'][19]:.1f}; "
          f"exceed p99 {sum(o['score'] > pt['p99'][r] for r, o in enumerate(obs_top))}/20")

    # ---- legacy check: reproduce the archived gap-year file ---------------
    if args.legacy_check and LEGACY_JSON.exists():
        legacy = json.loads(LEGACY_JSON.read_text())
        ls, ley, lpre, lpost = scorer.score(comm_idx_elig, years_all[elig_mask], "gap_year")
        order = np.argsort(-ls, kind="stable")[:TOP_K]
        obs_ok = all(eligible[k] == row["community"] and abs(ls[k] - row["observed_score"]) < 1e-6
                     for k, row in zip(order, legacy["observed_top_k_vs_null"]))
        tops_l = run_null(scorer, comm_idx_elig, elig_mask, years_all, legacy["n_shuffles"], args.seed, "gap_year", "all")
        ptl = percentile_table(tops_l)
        maxdiff = max(abs(ptl[p][r] - legacy["observed_top_k_vs_null"][r][f"null_{p}"])
                      for p in ("p50", "p95", "p99") for r in range(TOP_K))
        out["legacy_gap_year_check"] = {
            "archived_file": str(LEGACY_JSON.relative_to(REPO)),
            "convention": "gap_year: pre [ey-W, ey-1], post [ey+1, ey+W]",
            "observed_top_k_reproduced": bool(obs_ok),
            "legacy_observed_top_k": [{"rank": r + 1, "community": int(eligible[k]), "score": float(ls[k]),
                                       "event_year": int(ley[k]), "n_pre": int(lpre[k]), "n_post": int(lpost[k])}
                                      for r, k in enumerate(order)],
            "max_abs_diff_percentiles": float(maxdiff),
        }
        print(f"legacy check: gap-year observed top-{TOP_K} reproduced = {obs_ok}; "
              f"max |Δ| over archived p50/p95/p99 = {maxdiff:.3e}")

    out["elapsed_s_total"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
