#!/usr/bin/env python3
"""Provisional chemistry-based functional labels for a repaired partition.

This is a new, explicit conservative rubric, not a reproduction of the historical
manual seed labels. It never transfers historical community IDs or names. These
labels indicate candidate chemical families, not measured functionality. Framework
labels are deliberately left unassigned: formula/SG data do not verify topology.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import json
from pathlib import Path

from pymatgen.core import Composition

from compare_feature_repair import read_assignments
from regenerate_repaired_temporal import dump, file_hash, write_csv

RE = set("Y La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())
AE = {"Mg", "Ca", "Sr", "Ba"}
TM = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Nb", "Mo", "W"}
CLASS = {"battery": "battery_electrode_candidate", "magnet": "magnet_candidate",
         "superconductor": "superconductor_candidate", "thermoelectric": "thermoelectric_candidate"}
RULE_DESCRIPTIONS = {
    "alkali_TM_oxide_or_phosphate": "Li/Na + Ti/V/Cr/Mn/Fe/Co/Ni + O; alkali/TM>=0.25, Si<=TM; no H/C/N/S/Se/Te/halogens/rare earths; remaining elements restricted to oxide/phosphate dopants.",
    "cuprate_related": "Cu+O and rare-earth/alkaline-earth cations, optionally Bi/Tl/Hg/Pb; at least two O per Cu; SG47/65/69/123/139/166/191/221; excludes other transition metals and non-oxide anions (chain/green-phase families excluded).",
    "iron_pnictide_chalcogenide": "Fe with As or Se/Te in known Fe-superconductor SG families; permits Co/Ni/Ru/Ir dopants only while Fe dominates the transition-metal sites.",
    "A15_Nb_V_compound": "SG223 with Nb/V + Al/Ga/Ge/Sn/Si in approximately3:1 stoichiometry.",
    "R_Ni_borocarbide": "Rare earth + Ni + B + C with SG139.",
    "MgB2": "Mg/B-only2:1 boron-to-magnesium stoichiometry in SG191.",
    "RE_Fe_Co_intermetallic": "Rare-earth Fe/Co-rich intermetallic/interstitial family; Fe+Co per rare-earth cation>=2; excludes O/S/Se/Te/halogens.",
    "manganite_oxide": "RE/Ca/Sr/Ba + Mn + O, Mn>=50% of transition-metal content, O/Mn between2and5; oxide dopants only.",
    "ferrite_oxide": "Fe-rich oxide, Fe>=50% of transition-metal content, supported spinel/hexaferrite SG194/227.",
    "heavy_chalcogenide": "Chalcogenide of Bi/Sb/Pb/Sn/Ge/Cu/Ag/In/Tl/Zn/Cd/Hg with Te/Se/S; requires Bi/Sb/Pb or Sn+Se/Te or Cu/Ag+Se/Te; no oxygen/halogens/transition-metal magnetic cations.",
    "half_Heusler_nickel_stannide": "Ti/Zr/Hf + Ni/Pd + Sn/Sb in approximately1:1:1 stoichiometry, SG216.",
}


@lru_cache(maxsize=None)
def match_rules(formula: str, sg: int):
    try:
        a = Composition(formula).get_el_amt_dict()
    except (ValueError, TypeError):
        return ()
    e = set(a)
    amount = lambda group: sum(a.get(x, 0) for x in group)
    rules = []
    if e & {"Li", "Na"} and e & {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"} and "O" in e:
        if e <= {"Li", "Na", "Mg", "Ca", "Al", "Si", "P", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Zn", "O"} and amount({"Li", "Na"}) >= .25*amount(TM) and a.get("Si", 0) <= amount(TM):
            rules.append((CLASS["battery"], "alkali_TM_oxide_or_phosphate"))
    if {"Cu", "O"} <= e and e & (RE | AE) and a["O"] >= 2*a["Cu"] and sg in {47, 65, 69, 123, 139, 166, 191, 221}:
        if e <= RE | AE | {"Cu", "O", "Bi", "Tl", "Hg", "Pb"}:
            rules.append((CLASS["superconductor"], "cuprate_related"))
    if "Fe" in e and e & {"As", "Se", "Te"} and sg in {63, 67, 129, 139}:
        if e <= RE | AE | {"Li", "Na", "K", "Rb", "Cs", "Fe", "Co", "Ni", "Ru", "Ir", "As", "P", "Se", "Te", "O", "F"} and a["Fe"] >= amount({"Co", "Ni", "Ru", "Ir"}):
            rules.append((CLASS["superconductor"], "iron_pnictide_chalcogenide"))
    if sg == 223 and e & {"Nb", "V"} and e & {"Al", "Ga", "Ge", "Sn", "Si"}:
        if e <= {"Nb", "V", "Al", "Ga", "Ge", "Sn", "Si"} and 2.7 <= amount({"Nb", "V"})/amount({"Al", "Ga", "Ge", "Sn", "Si"}) <= 3.3:
            rules.append((CLASS["superconductor"], "A15_Nb_V_compound"))
    if sg == 139 and {"Ni", "B", "C"} <= e and e & RE and e <= RE | {"Ni", "B", "C", "Co", "Pt", "Pd"}:
        rules.append((CLASS["superconductor"], "R_Ni_borocarbide"))
    if e == {"Mg", "B"} and sg == 191 and abs(a["B"]/a["Mg"]-2) < .05:
        rules.append((CLASS["superconductor"], "MgB2"))
    if e & RE and e & {"Fe", "Co"} and amount({"Fe", "Co"}) >= 2*amount(RE):
        if e <= RE | {"Fe", "Co", "Ni", "Mn", "Cu", "Al", "Ga", "Si", "Ge", "Ti", "Zr", "Hf", "V", "Nb", "Mo", "B", "C", "N", "H"}:
            rules.append((CLASS["magnet"], "RE_Fe_Co_intermetallic"))
    if {"Mn", "O"} <= e and e & (RE | {"Ca", "Sr", "Ba"}):
        if e <= RE | AE | TM | {"Li", "Na", "K", "Al", "Ga", "O"} and a["Mn"] >= .5*amount(TM) and 2 <= a["O"]/a["Mn"] <= 5:
            rules.append((CLASS["magnet"], "manganite_oxide"))
    if {"Fe", "O"} <= e and sg in {194, 227} and a["Fe"] >= .5*amount(TM):
        if e <= RE | AE | TM | {"Al", "Ga", "O"}:
            rules.append((CLASS["magnet"], "ferrite_oxide"))
    if e & {"S", "Se", "Te"} and e <= {"S", "Se", "Te", "Bi", "Sb", "Pb", "Sn", "Ge", "Cu", "Ag", "In", "Tl", "Zn", "Cd", "Hg"}:
        if e & {"Bi", "Sb", "Pb"} or (e & {"Sn", "Cu", "Ag"} and e & {"Se", "Te"}):
            rules.append((CLASS["thermoelectric"], "heavy_chalcogenide"))
    if sg == 216 and e & {"Ti", "Zr", "Hf"} and e & {"Ni", "Pd"} and e & {"Sn", "Sb"}:
        if e <= {"Ti", "Zr", "Hf", "Ni", "Pd", "Sn", "Sb"}:
            amounts = [amount({"Ti", "Zr", "Hf"}), amount({"Ni", "Pd"}), amount({"Sn", "Sb"})]
            if max(amounts)/min(amounts) <= 1.2:
                rules.append((CLASS["thermoelectric"], "half_Heusler_nickel_stannide"))
    return tuple(rules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    with args.index.open(newline="") as f:
        for row in csv.DictReader(f):
            meta[int(row["cif_names"])] = (row["name"], int(row["sym_group"] or 0))
    members = defaultdict(list)
    for iid, _, c in read_assignments(args.assignments):
        if c >= 0:
            members[c].append(iid)
    reps = defaultdict(list)
    with args.representatives.open(newline="") as f:
        for row in csv.DictReader(f):
            reps[int(row["community"])].append(int(row["icsd_id"]))
    labels, detailed = [], []
    for c, ids in sorted(members.items()):
        mc, rc, mr, rr = Counter(), Counter(), Counter(), Counter()
        for iid in ids:
            matches = match_rules(*meta.get(iid, ("", 0)))
            mc.update(set(cls for cls, _ in matches))
            mr.update(rule for _, rule in matches)
        for iid in reps.get(c, []):
            if iid not in ids:
                raise ValueError(f"Representative {iid} not in community {c}")
            matches = match_rules(*meta.get(iid, ("", 0)))
            rc.update(set(cls for cls, _ in matches))
            rr.update(rule for _, rule in matches)
        supported = [cls for cls in CLASS.values() if len(reps.get(c, [])) == 20
                     and rc[cls] >= 12 and mc[cls] >= .5*len(ids)]
        label = supported[0] if len(supported) == 1 else ""
        labels.append({"community": c, "community_size": len(ids), "functional_signature": label,
                       "label_status": "provisional_chemistry_heuristic" if label else "unassigned",
                       "n_representatives": len(reps.get(c, [])),
                       "n_representatives_supporting_label": rc[label] if label else "",
                       "n_members_supporting_label": mc[label] if label else "",
                       "member_fraction_supporting_label": mc[label]/len(ids) if label else "",
                       "supporting_rules": ";".join(k for k in mr if k in rr) if label else "",
                       "notes": "Candidate family, not measured functionality; no historical community-label transfer."})
        detailed.append({"community": c, "member_class_counts": dict(mc), "representative_class_counts": dict(rc),
                         "member_rule_counts": dict(mr), "representative_rule_counts": dict(rr),
                         "representative_icsd_ids": reps.get(c, [])})
    write_csv(args.output_dir / "community_functional_labels.csv", labels)
    dump(args.output_dir / "functional_label_evidence.json", detailed)
    manifest = {"method": "new conservative chemistry heuristic; historical labels were manual and are not reproduced",
                "thresholds": "size>=25,20 centroid-nearest representatives;>=12/20 representatives AND>=50% all members match one class; conflicting classes blank",
                "framework_status": "unassigned; composition/SG and an isolated legacy zeolite exemplar do not establish whole-community framework topology",
                "rule_descriptions": RULE_DESCRIPTIONS,
                "counts": dict(Counter(r["functional_signature"] or "unassigned" for r in labels)),
                "inputs": {str(p): file_hash(p) for p in (args.assignments, args.index, args.representatives)},
                "producer_sha256": file_hash(Path(__file__))}
    dump(args.output_dir / "functional_label_manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
