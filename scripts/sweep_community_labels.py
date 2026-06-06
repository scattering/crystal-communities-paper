#!/usr/bin/env python3
"""Sweep v2 — chemistry-aware, with corrected A1B1/A1B2 stoichiometry mappings.

Two changes vs v1:
  1. Fix stoich/SG for rocksalt, CsCl-type, zincblende, wurtzite to A1B1 (not A1B2)
  2. Chemistry filter: oxide-specific labels (perovskite, spinel, pyrochlore,
     K2NiF4, ordered double perovskite, olivine) only fire when the community's
     dominant anion is actually O.
"""
from __future__ import annotations
import csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/work2/09870/williamratcliff/stampede3")
LABELS3 = REPO / "icsd_graph_runs/full_matminer_ops_spr_20260419_labels3/community_assignments.csv"
ICSD_INDEX = REPO / "reference_data/ICSD_index.csv"
CANONICAL = REPO / "icsd_graph_runs/full_matminer_ops_spr_20260419_labels3/canonical_family_names_labels3.csv"
INFERRED = REPO / "icsd_graph_runs/full_matminer_ops_spr_20260419_labels3/community_families_inferred.csv"
PROTOTYPE = REPO / "icsd_graph_runs/full_matminer_ops_spr_20260419_labels3/community_prototype_labels.json"
OUT = REPO / "proposed_community_labels_v2.csv"
MIN_COMMUNITY_SIZE = 10

# Value can be:
#   - str  → chemistry-agnostic label, fires unconditionally
#   - (str, "O") → requires dominant anion to be O ('O' for oxide-specific)
PROTOTYPES = {
    # ---- Chemistry-agnostic binary 1:1 prototypes ----
    ("A1B1", 225):  "Rocksalt (NaCl-type)",
    ("A1B1", 221):  "Cesium chloride (CsCl-type)",
    ("A1B1", 216):  "Zincblende (ZnS-type)",
    ("A1B1", 186):  "Wurtzite (ZnS hex)",
    # ---- Chemistry-agnostic AB2 prototypes ----
    ("A1B2", 225):  "Fluorite (CaF2-type) / antifluorite",
    ("A1B2", 194):  "Laves C14 (MgZn2-prototype)",
    ("A1B2", 227):  "Laves C15 (Cu2Mg-prototype)",
    ("A1B2", 166):  "Laves C36 (MgNi2-prototype hexagonal)",
    # ---- Chemistry-agnostic intermetallic prototypes ----
    ("A1B5", 191):  "CaCu5-prototype (LaNi5 hydrogen-storage family)",
    ("A1B1C1", 216):"Half-Heusler ABX (F-43m)",
    ("A1B1C1", 225):"Full-Heusler X2YZ (Fm-3m)",
    ("A1B1C1", 189):"Half-Heusler ScAuSn-type (P-62m)",
    ("A1B2C2", 139):"ThCr2Si2-prototype 122-type (Fe-pnictide 122, heavy-fermion CeCu2Si2, BaFe2As2-family)",
    ("A2B2C1", 139):"Mo2FeB2-prototype 2:1:2 intermetallic (Pottgen-era RE-In-TM)",
    # ---- Lacunar spinel — chemistry-permissive (chalcogenide family with cubic F-43m AM4X8) ----
    ("A1B4C8", 216):"AM4X8 lacunar spinel (GaMo4S8-prototype, cluster magnets)",
    ("A1B4C8", 215):"AM4X8 lacunar spinel (low-T rhombohedral variant)",
    ("A1B4C8", 113):"AM4X8 lacunar spinel (tetragonal variant)",
    # ---- Oxide-specific labels (require dominant anion = O) ----
    ("A1B1C3", 221):("Cubic perovskite ABO3", "O"),
    ("A1B1C3", 62): ("Orthorhombic Pnma perovskite ABO3", "O"),
    ("A1B1C3", 167):("Rhombohedral perovskite ABO3 (R-3c)", "O"),
    ("A1B2C4", 227):("Spinel (AB2O4, Fd-3m)", "O"),
    ("A2B2C7", 227):("Pyrochlore (A2B2O7)", "O"),
    ("A1B1C2D6", 225):("Ordered double perovskite (A2BB'O6, Fm-3m)", "O"),
    ("A1B2C1D4", 139):("K2NiF4-prototype (T214 cuprate / RP n=1)", "O"),
    ("A2B1C4", 62):("Olivine (A2BO4, Pnma)", "O"),
    # ---- Layered chalcogenide ----
    ("A1B1C2", 166):("CdI2-prototype (1T-MX2 dichalcogenide)", "chalc"),
}

ANION_PRIORITY = ["O","F","Cl","Br","I","S","Se","Te","N","P","As","H","C"]
CHALCOGEN = {"S","Se","Te"}

def anonymize_formula(formula):
    toks = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula)
    counts = []
    for el, n in toks:
        if not el: continue
        try: count = float(n) if n else 1.0
        except ValueError: count = 1.0
        counts.append((el, count))
    if not counts: return ""
    counts.sort(key=lambda t: (t[1], t[0]))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = []
    for i, (el, n) in enumerate(counts):
        if i >= len(letters): break
        out.append(f"{letters[i]}{int(n) if n == int(n) else n}")
    return "".join(out)

def dominant_anion(metas):
    """Across all member formulas, return the most common dominant-anion element."""
    counts = Counter()
    for m in metas:
        toks = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", m.get("formula",""))
        elems = []
        for el, n in toks:
            if not el: continue
            try: c = float(n) if n else 1.0
            except ValueError: c = 1.0
            elems.append((el, c))
        candidates = [(el, c) for el, c in elems if el in ANION_PRIORITY]
        if candidates:
            counts[max(candidates, key=lambda x: x[1])[0]] += 1
    return counts.most_common(1)[0][0] if counts else None

def main():
    labeled = set()
    for src in [CANONICAL, INFERRED]:
        if src.exists():
            with src.open() as f:
                for r in csv.DictReader(f):
                    try: labeled.add(int(r["community"]))
                    except (KeyError, ValueError): pass
    if PROTOTYPE.exists():
        for r in json.load(PROTOTYPE.open()):
            try: labeled.add(int(r["community"]))
            except (KeyError, ValueError): pass
    print(f"  already labeled: {len(labeled)} communities", flush=True)

    members = defaultdict(list)
    with LABELS3.open() as f:
        for r in csv.DictReader(f):
            try:
                c = int(r["community"])
                if c < 0: continue
                members[c].append(int(r["icsd_id"]))
            except (KeyError, ValueError): continue
    unlabeled = {c: set(m) for c, m in members.items() if len(m) >= MIN_COMMUNITY_SIZE and c not in labeled}
    print(f"  unlabeled, size ≥ {MIN_COMMUNITY_SIZE}: {len(unlabeled)}", flush=True)

    icsd_records = {}
    needed = set().union(*unlabeled.values())
    with ICSD_INDEX.open() as f:
        for r in csv.DictReader(f):
            try: iid = int(r["cif_names"])
            except (KeyError, ValueError): continue
            if iid in needed:
                icsd_records[iid] = {
                    "formula": r.get("name","").strip(),
                    "sg": int(r["sym_group"]) if r.get("sym_group","").strip().isdigit() else None,
                }
    print(f"  metadata coverage: {len(icsd_records)}/{len(needed)}", flush=True)

    rows = []
    chem_filtered = 0
    for c, iids in sorted(unlabeled.items(), key=lambda kv: -len(kv[1])):
        metas = [icsd_records[i] for i in iids if i in icsd_records]
        if not metas: continue
        stoich_sg = Counter()
        for m in metas:
            f_, sg_ = m["formula"], m["sg"]
            if f_ and sg_:
                stoich_sg[(anonymize_formula(f_), sg_)] += 1
        if not stoich_sg: continue
        top_stoich_sg, top_count = stoich_sg.most_common(1)[0]
        dominant_frac = top_count / len(metas)
        entry = PROTOTYPES.get(top_stoich_sg)
        anion = dominant_anion(metas) if entry else None
        if entry is None:
            proposal = f"{top_stoich_sg[0]} stoichiometry, SG {top_stoich_sg[1]} (auto)"
        elif isinstance(entry, tuple):
            label, required = entry
            if required == "O":
                if anion == "O":
                    proposal = label
                else:
                    proposal = f"{top_stoich_sg[0]} stoichiometry, SG {top_stoich_sg[1]} (auto; non-oxide chemistry, dominant anion {anion})"
                    chem_filtered += 1
            elif required == "chalc":
                if anion in CHALCOGEN:
                    proposal = label
                else:
                    proposal = f"{top_stoich_sg[0]} stoichiometry, SG {top_stoich_sg[1]} (auto; non-chalcogenide chemistry, dominant anion {anion})"
                    chem_filtered += 1
            else:
                proposal = label
        else:
            proposal = entry

        conf = "HIGH" if dominant_frac > 0.7 else ("MED" if dominant_frac > 0.5 else "LOW")
        formulas = [m["formula"] for m in metas if m["formula"]]
        rows.append({
            "community": c, "size": len(iids),
            "metadata_coverage": f"{len(metas)}/{len(iids)}",
            "top_anonymized_stoich": top_stoich_sg[0],
            "top_sg": top_stoich_sg[1],
            "dominant_frac": f"{dominant_frac:.2f}",
            "dominant_anion": anion or "",
            "confidence": conf,
            "proposed_label": proposal,
            "sample_formulas": "; ".join(formulas[:5]),
        })
    print(f"  chemistry filter rejected {chem_filtered} oxide-specific labels", flush=True)
    rows.sort(key=lambda r: (r["confidence"] != "HIGH", -r["size"]))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["community","size","metadata_coverage","top_anonymized_stoich","top_sg","dominant_frac","dominant_anion","confidence","proposed_label","sample_formulas"])
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {OUT} ({len(rows)} rows)", flush=True)

    # quick stats
    mapped = [r for r in rows if "stoichiometry, SG" not in r["proposed_label"]]
    print(f"  mapped to a prototype name: {len(mapped)} ({100*len(mapped)/len(rows):.1f}%)", flush=True)
    for r in rows:
        if r["community"] == 2958:
            print(f"\n  community 2958: {r['proposed_label']}  (conf={r['confidence']}, dominant_frac={r['dominant_frac']}, anion={r['dominant_anion']})")
            break

if __name__ == "__main__":
    main()
