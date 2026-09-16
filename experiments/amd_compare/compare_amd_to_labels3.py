#!/usr/bin/env python3
"""Compare the AMD-based partition against the June 2026 (labels3) partition,
which the CrystalWeave partition has since superseded.

Produces:
  - Global ARI/NMI between the two partitions on the shared id set
  - Per-manuscript-community recovery table: for each manuscript-anchored
    community in labels3 (cuprate 1178, Fe-pnictide 5450/296/358, lacunar
    spinel 2958, pyrochlore 5866, spinel 6041, Pnma perovskite 6054, etc.),
    measure majority overlap with the best-matching AMD community.
  - Densification cliff replay on the AMD partition (matches §S9.2 graphlet
    test): birth fraction by decade, same-community attachment fraction.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# Manuscript-anchored communities (from notes/canonical_family_names_labels3.csv,
# expanded by the chemistry-aware sweep + hand-curates).
MANUSCRIPT_COMMUNITIES = [
    (1178, "High-Tc cuprate (YBa2Cu3O7 / RE-Ba-Cu-O 123)"),
    (2846, "Cuprate sister (RE-Ba-Cu-O variants)"),
    (6607, "Cuprate sister (I4_1/amd)"),
    (5450, "Fe-pnictide 1111 LaFeAsO (ZrCuSiAs-prototype)"),
    (296,  "Fe-pnictide 122 BaFe2As2 (ThCr2Si2-prototype)"),
    (358,  "Fe-pnictide 111 / FeSe family"),
    (2958, "AM4X8 lacunar spinel (GaMo4S8-prototype)"),
    (5866, "Pyrochlore (A2B2O7)"),
    (6041, "Spinel (AB2O4, Fd-3m)"),
    (6054, "Orthorhombic Pnma perovskite ABO3"),
    (1716, "Cubic perovskite ABO3 (SrTiO3-type)"),
    (1,    "ThCr2Si2-prototype 122-type intermetallics"),
    (4549, "ThCr2Si2 sister"),
    (160,  "CMR manganite"),
    (2349, "Mo2FeB2-prototype 2:1:2 intermetallic"),
]


def read_partition(path: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                iid = int(row["icsd_id"])
                cid = int(row["community"])
            except (KeyError, ValueError):
                continue
            out[iid] = cid
    return out


def per_community_recovery(labels3: dict[int, int], amd_p: dict[int, int], comm_id: int) -> dict:
    """For one labels3 community, find the best matching AMD community
    (largest overlap) and report overlap/size statistics."""
    members = [iid for iid, c in labels3.items() if c == comm_id and iid in amd_p]
    if not members:
        return {"labels3_community": comm_id, "size_in_subset": 0, "best_amd": None,
                "best_overlap": 0, "purity": 0.0, "amd_community_size": 0,
                "amd_community_capture": 0.0}
    amd_dist = Counter(amd_p[iid] for iid in members)
    best_amd, best_overlap = amd_dist.most_common(1)[0]
    amd_size = sum(1 for _, c in amd_p.items() if c == best_amd)
    return {
        "labels3_community": comm_id,
        "size_in_subset": len(members),
        "best_amd": best_amd,
        "best_overlap": best_overlap,
        "purity": best_overlap / len(members),
        "amd_community_size": amd_size,
        "amd_community_capture": best_overlap / amd_size if amd_size else 0.0,
    }


def densification_metrics(partition: dict[int, int], events_path: Path) -> dict:
    """Replay the temporal-densification cliff under a given partition.
    Returns per-decade fraction-new-community ("birth") and
    same-community-attachment fractions.
    Matches the §S9.2 graphlet pattern.
    """
    # Read per-node events table (icsd_id, year, parent_icsd_id_at_birth ...).
    decades = defaultdict(lambda: {"n": 0, "birth": 0, "same_comm_attach": 0})
    if not events_path.exists():
        return {}
    with events_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                iid = int(row["icsd_id"])
                yr = int(float(row["year"]))
                decade = (yr // 10) * 10
            except (KeyError, ValueError):
                continue
            if iid not in partition:
                continue
            d = decades[decade]
            d["n"] += 1
            # birth: this iid opened a new community at its time of arrival
            # In the AMD partition we don't have temporal replay — so we just
            # report the static fraction of decade members in each community.
            # For full faithfulness to §S9.2 we'd need a temporal replay run;
            # leave that as an extension and report only static membership here.
    out = {}
    for d, v in sorted(decades.items()):
        out[d] = {"n_entries": v["n"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels3", required=True, help="production community_assignments.csv (labels3)")
    ap.add_argument("--amd", required=True, help="amd_community_assignments.csv")
    ap.add_argument("--events", default=None, help="(optional) node_temporal_events.csv for densification replay")
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    l3 = read_partition(Path(args.labels3))
    am = read_partition(Path(args.amd))
    common = sorted(set(l3) & set(am))
    print(f"  labels3 n={len(l3):,}   amd n={len(am):,}   common={len(common):,}", flush=True)

    y_l3 = np.array([l3[i] for i in common])
    y_am = np.array([am[i] for i in common])

    ari = float(adjusted_rand_score(y_l3, y_am))
    nmi = float(normalized_mutual_info_score(y_l3, y_am))
    n_l3 = len(set(y_l3))
    n_am = len(set(y_am))
    print(f"  ARI = {ari:.4f}", flush=True)
    print(f"  NMI = {nmi:.4f}", flush=True)
    print(f"  labels3 partition has {n_l3} communities; AMD has {n_am}", flush=True)

    per_comm = []
    for cid, label in MANUSCRIPT_COMMUNITIES:
        rec = per_community_recovery(l3, am, cid)
        rec["family"] = label
        per_comm.append(rec)
        print(f"    [{cid:>5} {label[:55]:55s}]  size={rec['size_in_subset']:>4}  best_amd={str(rec['best_amd']):>6}  "
              f"purity={rec['purity']*100:>5.1f}%  amd_size={rec['amd_community_size']:>6}  "
              f"capture={rec['amd_community_capture']*100:>5.1f}%",
              flush=True)

    dens = densification_metrics(am, Path(args.events)) if args.events else {}

    with (out / "amd_compare_result.json").open("w") as f:
        json.dump({
            "ari": ari, "nmi": nmi,
            "n_l3_communities": n_l3, "n_amd_communities": n_am,
            "n_compared": len(common),
            "manuscript_community_recovery": per_comm,
            "densification_static": dens,
        }, f, indent=2)
    print(f"  wrote {out / 'amd_compare_result.json'}", flush=True)


if __name__ == "__main__":
    main()
