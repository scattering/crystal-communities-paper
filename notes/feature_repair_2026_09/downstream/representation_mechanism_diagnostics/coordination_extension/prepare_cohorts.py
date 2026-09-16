#!/usr/bin/env python3
"""Freeze full common-support chemistry/size matches before coordination outcomes."""
from __future__ import annotations

import csv, hashlib, json, math, re
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Composition
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
RESULTS = DIAG.parent / "representations" / "amd-external-full-map" / "results"
SUPPORT = DIAG / "factor_ablations" / "inputs" / "common_support_ids.json"
PILOT = DIAG / "bonding_cohort_pilot" / "sample.json"
SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")
COMPARATORS = SOURCES[1:]
DISPLAY = {"gnome":"GNoME", "mattergen":"MatterGen", "mp":"MP", "jarvis":"JARVIS", "alexandria":"Alexandria"}
ID_FIELD = {s:("zip_member" if s == "mattergen" else "material_id") for s in SOURCES}
ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
SEED = 20260910

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def elements(formula: str) -> set[str]:
    out = {element.symbol for element in Composition(formula).elements}
    if not out: raise ValueError(f"No elements parsed: {formula!r}")
    lexical = set(ELEMENT_RE.findall(formula))
    if out != lexical: raise ValueError(f"Formula parser disagreement for {formula!r}: {out} != {lexical}")
    return out

def chemistry_class(formula: str) -> str:
    els = elements(formula)
    for label, members in (("O",{"O"}),("FClBrI",{"F","Cl","Br","I"}),
                           ("SSeTe",{"S","Se","Te"}),("NPAs",{"N","P","As"}),
                           ("HBC",{"H","B","C"})):
        if els & members: return label
    return "other"

def make_rows():
    support_doc = json.loads(SUPPORT.read_text())
    populations = support_doc["populations"]
    pilot = json.loads(PILOT.read_text())
    pilot_keys = {(r["source"], str(r["material_id"])) for r in pilot}
    rows, inputs = [], {str(SUPPORT.relative_to(HERE.parents[5])): sha(SUPPORT), str(PILOT.relative_to(HERE.parents[5])): sha(PILOT)}
    for source in SOURCES:
        path = RESULTS/source/"successful_records.csv"
        inputs[str(path.relative_to(HERE.parents[5]))] = sha(path)
        field = ID_FIELD[source]
        wanted = set(map(str, populations[DISPLAY[source]]["ids"]))
        seen = set()
        for raw in read_csv(path):
            ident = str(raw[field])
            if ident not in wanted: continue
            if ident in seen: raise ValueError(f"duplicate {source} identity {ident}")
            seen.add(ident)
            formula = raw["reduced_formula"]
            rows.append({"source":source,"material_id":ident,"formula":formula,
                         "n_sites":int(raw["n_sites"]),"n_elements":len(elements(formula)),
                         "chemistry_class":chemistry_class(formula),
                         "was_pilot":(source,ident) in pilot_keys,"record_key":ident})
        if seen != wanted: raise ValueError(f"{source}: support/success mismatch missing={len(wanted-seen)} extra={len(seen-wanted)}")
    return rows, inputs, support_doc, pilot_keys

def match_class(targets, candidates, source, analysis, label):
    nt, nc = len(targets), len(candidates)
    if not nt or not nc: return {}, [0]*nt
    te=np.array([r["n_elements"] for r in targets]); ts=np.array([r["n_sites"] for r in targets],float)
    ce=np.array([r["n_elements"] for r in candidates]); cs=np.array([r["n_sites"] for r in candidates],float)
    ed=np.abs(te[:,None]-ce[None,:]); ld=np.abs(np.log2(cs[None,:]/ts[:,None]))
    eligible=(ed<=1)&(ld<=1+1e-12)
    eligible_counts=eligible.sum(axis=1).astype(int).tolist()
    # One unmatched assignment must dominate the entire possible secondary
    # objective, so cardinality is genuinely optimized first at this scale.
    infeasible, unmatched = 1e15, 1e9
    cost=np.full((nt,nc+nt),infeasible,float)
    seed=int.from_bytes(hashlib.sha256(f"{SEED}|{analysis}|{source}|{label}".encode()).digest()[:8],"big")
    jitter=np.random.default_rng(seed).random((nt,nc))*1e-9
    real=ed*(nt+1.0)+ld+jitter
    cost[:,:nc][eligible]=real[eligible]
    # Distinct tiny offsets give deterministic dummy assignments without changing cardinality.
    cost[:,nc:]=unmatched + (np.arange(nt)[:,None]*nt+np.arange(nt)[None,:])*1e-12
    ri,ci=linear_sum_assignment(cost)
    return {int(i):int(j) for i,j in zip(ri,ci) if j<nc and cost[i,j]<unmatched}, eligible_counts

def match_analysis(all_rows, analysis):
    eligible_rows=[r for r in all_rows if analysis=="all" or not r["was_pilot"]]
    gnome=[r for r in eligible_rows if r["source"]=="gnome"]
    output=[]; reports={}
    for source in COMPARATORS:
        candidates=[r for r in eligible_rows if r["source"]==source]
        matched_g=set(); matched_c=set(); eligibility={}
        for label in ("O","FClBrI","SSeTe","NPAs","HBC","other"):
            tg=[r for r in gnome if r["chemistry_class"]==label]
            cc=[r for r in candidates if r["chemistry_class"]==label]
            assignments, counts=match_class(tg,cc,source,analysis,label)
            eligibility.update({r["material_id"]:n for r,n in zip(tg,counts)})
            for i,j in assignments.items():
                gr,cr=tg[i],cc[j]; matched_g.add(gr["material_id"]); matched_c.add(cr["material_id"])
                output.append({"analysis":analysis,"source":source,"gnome_id":gr["material_id"],"comparator_id":cr["material_id"]})
        no_candidate=sum(eligibility.get(r["material_id"],0)==0 for r in gnome if r["material_id"] not in matched_g)
        reports[source]={"gnome_available":len(gnome),"comparator_available":len(candidates),"matched":len(matched_g),
                         "gnome_unmatched":len(gnome)-len(matched_g),"gnome_unmatched_no_individually_eligible_candidate":no_candidate,
                         "gnome_unmatched_without_replacement_contention":len(gnome)-len(matched_g)-no_candidate,
                         "comparator_unused":len(candidates)-len(matched_c),"gnome_coverage":len(matched_g)/len(gnome) if gnome else 0.0,
                         "comparator_coverage":len(matched_c)/len(candidates) if candidates else 0.0}
    return output,reports

def write_csv(path, rows, fields):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    rows,inputs,support_doc,pilot_keys=make_rows()
    rows.sort(key=lambda r:(SOURCES.index(r["source"]),r["material_id"]))
    (HERE/"cohort_manifest.json").write_text(json.dumps(rows,indent=2)+"\n")
    pairs=[]; analyses={}
    for analysis in ("all","unused"):
        out,report=match_analysis(rows,analysis); pairs.extend(out); analyses[analysis]=report
    pairs.sort(key=lambda r:(("all","unused").index(r["analysis"]),COMPARATORS.index(r["source"]),r["gnome_id"]))
    write_csv(HERE/"matched_pairs.csv",pairs,["analysis","source","gnome_id","comparator_id"])
    manifest_hash=sha(HERE/"cohort_manifest.json"); pairs_hash=sha(HERE/"matched_pairs.csv")
    counts=Counter(r["source"] for r in rows); pilot_counts=Counter(r["source"] for r in rows if r["was_pilot"])
    report={"status":"frozen_before_coordination_outcomes","seed":SEED,
            "scope":"All five-representation common-support public records; no basin selection.",
            "selection":{"chemistry_class_priority":["O","FClBrI","SSeTe","NPAs","HBC","other"],"exact_chemistry_class":True,
              "maximum_absolute_n_elements_difference":1,"n_sites_ratio_range_inclusive":[0.5,2.0],"without_replacement_within_each_comparison":True,
              "objective_order":["maximize cardinality","minimize total absolute element-count difference","minimize total absolute log2 site-count ratio","seeded tie break"],
              "implementation":"vectorized eligibility and cost matrices; scipy linear_sum_assignment separately by chemistry class","selected_on_map_flags":False,"selected_on_coordination_outcomes":False},
            "manifest":{"rows":len(rows),"source_counts":dict(counts),"pilot_counts":dict(pilot_counts),"sha256":manifest_hash},
            "analyses":analyses,"matched_pairs":{"rows":len(pairs),"sha256":pairs_hash},"inputs_sha256":inputs,
            "support_ids_hash_convention":support_doc.get("ids_hash_convention"),"identity":{"mattergen":"zip_member","other_sources":"material_id","record_key":"exact support identity"}}
    (HERE/"sampling_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"records":len(rows),"pairs":len(pairs),"analyses":analyses},indent=2))

if __name__ == "__main__": main()
