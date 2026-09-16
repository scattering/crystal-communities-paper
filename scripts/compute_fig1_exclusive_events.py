#!/usr/bin/env python3
"""Build a MUTUALLY EXCLUSIVE per-decade discovery-event partition for Fig. 1.

Input : notes/node_temporal_events.csv   (per-entry events written by
        scripts/icsd_graph_time_evolution.py::compute_temporal_metrics)
Output: notes/review_2026_08/fig1_exclusive_by_decade.csv

Exclusive class per entry (precedence top -> bottom):
  outlier          community < 0            (event_type == 'outlier')
  community_birth  year == birth year of its community (event_type == 'community_birth')
  bridge           existing_community and n_active_other_communities >= 2
                   (identical to is_bridge_attachment for non-outlier, non-birth rows)
  cross            existing_community and n_active_other_communities == 1
  same             existing_community and n_active_other_communities == 0
                   (joins a previously occupied community with no contact to any
                    other community; includes entries whose active neighbours are
                    all same-community, and entries with no active non-outlier
                    neighbour at the moment of attachment -- counted separately
                    in n_same_isolated)

The JSON counters n_same_community_attachment / n_cross_community_attachment /
n_bridge_attachment in graph_time_summary.json are NOT exclusive: each is
incremented independently for every non-outlier entry (births included), so
they overlap and cannot be stacked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "notes" / "node_temporal_events.csv"
SUMMARY = ROOT / "notes" / "graph_time_summary.json"
OUT = ROOT / "notes" / "review_2026_08" / "fig1_exclusive_by_decade.csv"

DECADES = [f"{y}s" for y in range(1910, 2020, 10)]


def classify(df: pd.DataFrame) -> pd.Series:
    cls = pd.Series("same", index=df.index, dtype=object)
    existing = df["event_type"] == "existing_community"
    cls[existing & (df["n_active_other_communities"] == 1)] = "cross"
    cls[existing & (df["n_active_other_communities"] >= 2)] = "bridge"
    cls[df["event_type"] == "community_birth"] = "birth"
    cls[df["event_type"] == "outlier"] = "outlier"
    return cls


def main() -> int:
    df = pd.read_csv(EVENTS)
    assert len(df) == 167_500, len(df)
    # Sanity: code-level identities
    assert ((df["event_type"] == "outlier") == (df["community"] < 0)).all()
    assert (df["is_bridge_attachment"] == (df["n_active_other_communities"] >= 2)).all()

    df["cls"] = classify(df)
    # Every existing_community row must land in exactly one of same/cross/bridge
    ex = df[df["event_type"] == "existing_community"]
    assert set(ex["cls"].unique()) <= {"same", "cross", "bridge"}
    assert (df.loc[df.event_type == "community_birth", "cls"] == "birth").all()
    assert (df.loc[df.event_type == "outlier", "cls"] == "outlier").all()

    rows = []
    for d in DECADES + ["unknown"]:
        sub = df[df["decade"] == d]
        n_total = len(sub)
        c = sub["cls"].value_counts()
        n_outlier = int(c.get("outlier", 0))
        n_birth = int(c.get("birth", 0))
        n_same = int(c.get("same", 0))
        n_cross = int(c.get("cross", 0))
        n_bridge = int(c.get("bridge", 0))
        n_same_isolated = int(((sub["cls"] == "same") & (sub["n_active_same_community_neighbors"] == 0)).sum())
        n_attach = n_same + n_cross + n_bridge
        n_nonout = n_total - n_outlier
        rows.append({
            "decade": d,
            "n_total": n_total,
            "n_outlier": n_outlier,
            "n_birth": n_birth,
            "n_same": n_same,
            "n_cross": n_cross,
            "n_bridge": n_bridge,
            "share_outlier": n_outlier / n_total,
            "share_birth": n_birth / n_total,
            "share_same": n_same / n_total,
            "share_cross": n_cross / n_total,
            "share_bridge": n_bridge / n_total,
            # two-class summary, outliers INCLUDED in denominator
            "share_attach_incl_outlier": n_attach / n_total,
            # two-class summary, outliers EXCLUDED from denominator
            "share_birth_excl_outlier": n_birth / n_nonout if n_nonout else np.nan,
            "share_attach_excl_outlier": n_attach / n_nonout if n_nonout else np.nan,
            "n_same_isolated": n_same_isolated,
        })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, float_format="%.6f")

    # ---- verification against graph_time_summary.json ----
    bd = json.loads(SUMMARY.read_text())["by_decade"]
    ok = True
    for _, r in out.iterrows():
        d = r["decade"]
        j = bd[d]
        s5 = r[["share_outlier", "share_birth", "share_same", "share_cross", "share_bridge"]].sum()
        checks = {
            "sum5==1": abs(s5 - 1.0) < 1e-9,
            "n_total": int(j["n_total"]) == int(r["n_total"]),
            "n_birth==n_cluster_birth_point": int(j["n_cluster_birth_point"]) == int(r["n_birth"]),
            "n_outlier": int(j["n_outlier"]) == int(r["n_outlier"]),
            "same+cross+bridge==n_existing_cluster": int(j["n_existing_cluster"]) == int(r["n_same"] + r["n_cross"] + r["n_bridge"]),
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            ok = False
        print(f"{d:>8}  sum5={s5:.9f}  " + ("OK" if not bad else "MISMATCH: " + ", ".join(bad)))
    tot = int(out["n_total"].sum())
    print(f"sum over all decades (incl. unknown) = {tot}  (expect 167392) -> {'OK' if tot == 167392 else 'MISMATCH'}")
    tot_known = int(out[out.decade != "unknown"]["n_total"].sum())
    print(f"sum over 1910s..2010s = {tot_known}")
    print(f"class totals: {df['cls'].value_counts().to_dict()}")
    # cross-check with global JSON counters
    print(f"bridge True in CSV = {int(df.is_bridge_attachment.sum())} = existing-bridge {int((df.cls=='bridge').sum())} + birth-with-bridge {int((df.event_type=='community_birth') & df.is_bridge_attachment).sum() if False else int(((df.event_type=='community_birth') & df.is_bridge_attachment).sum())} + outlier-with-bridge {int(((df.event_type=='outlier') & df.is_bridge_attachment).sum())}")
    print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
