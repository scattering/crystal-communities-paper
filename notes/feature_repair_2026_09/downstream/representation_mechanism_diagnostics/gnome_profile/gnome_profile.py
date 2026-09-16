#!/usr/bin/env python3
"""Per-structure profile of the 5,000 GNoME records across the five representation maps.

Question: which GNoME structures sit inside experimental basins under the graphlet maps but outside under
CrystalWeave, and what distinguishes them (composition novelty, size, symmetry, distance geometry)?
All inputs are saved per-record projections and covariates in this branch; nothing is re-featurised.
"""
from __future__ import annotations
import csv, hashlib, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[5]
D = ROOT / "notes/feature_repair_2026_09/downstream"
OUT = Path(__file__).resolve().parent
MAPS = {
    "CrystalWeave": None,  # from quadrant_assignments
    "Magpie": D / "external_representation/magpie/external/gnome/projection_full.csv",
    "Graphlet_CrystalNN": D / "external_representation/graphlet/external/gnome/projection_full.csv",
    "Graphlet_VoronoiNN": D / "representation_mechanism_diagnostics/voronoi_external/results/external/gnome/projection_full.csv",
    "AMD": D / "representations/amd-external-full-map/results/gnome/projection_full.csv",
}
GRAPHLET_ASSIGNMENTS = D / "representations/graphlet-dated-replay/graphlet/community_assignments.csv"
ICSD_INDEX = D / "inputs/ICSD_index.csv"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def ids_sha256(values):
    return hashlib.sha256("\n".join(map(str, sorted(map(int, values)))).encode()).hexdigest()

def load():
    q = pd.read_csv(D / "prior/quadrant_assignments.csv", dtype={"material_id": str})
    g = q[q.source == "GNoME"].copy().set_index("material_id")
    g = g.rename(columns={"in_basin": "in_CrystalWeave", "assigned_community": "comm_CrystalWeave",
                          "nearest_centroid_distance": "d_CrystalWeave", "community_threshold_p95": "p95_CrystalWeave"})
    for name, path in MAPS.items():
        if path is None: continue
        p = pd.read_csv(path, dtype={"record_key": str}).set_index("record_key")
        g[f"in_{name}"] = p["in_basin"].reindex(g.index).astype("boolean")
        g[f"comm_{name}"] = p["assigned_community"].reindex(g.index)
        g[f"d_{name}"] = p["nearest_centroid_distance"].reindex(g.index)
        g[f"p95_{name}"] = p["community_threshold_p95"].reindex(g.index)
    cov = pd.read_csv(D / "external_covariates/gnome_full.csv", dtype={"material_id": str}).set_index("material_id")
    g["spg_number"] = cov["spg_number"].reindex(g.index)
    g["e_above_hull_cov"] = cov["e_above_hull"].reindex(g.index)
    el = pd.read_csv(D / "formula_layers/nearest_icsd_elmd_records.csv", dtype={"material_id": str})
    el = el[el.source == "gnome"].set_index("material_id")
    g["elmd_all_icsd"] = el["all_icsd_nearest_elmd"].reindex(g.index)
    g["elmd_nearest_formula"] = el["all_icsd_nearest_icsd_formula"].reindex(g.index)
    return g

def composition_keys(formula):
    try:
        c = Composition(formula)
    except Exception:
        return None
    els = tuple(sorted(e.symbol for e in c.elements))
    ik, _ = c.get_integer_formula_and_factor()
    ic = Composition(ik)
    amounts = sorted((int(round(v)) for v in ic.get_el_amt_dict().values()), reverse=True)
    gcd = math.gcd(*amounts) if len(amounts) > 1 else amounts[0]
    anon = tuple(a // gcd for a in amounts)
    key = tuple(sorted((e.symbol, int(round(v)) // gcd) for e, v in ic.items()))
    return els, anon, key

def icsd_reference():
    """Element sets, anonymous stoichiometries and integer keys over the full local ICSD index names."""
    cache = OUT / "icsd_reference_keys.json"
    if cache.exists():
        d = json.load(open(cache))
        if d.get("icsd_index_sha256") == sha256(ICSD_INDEX):
            return set(map(tuple, d["els"])), set(map(tuple, d["anon"])), set(tuple(tuple(x) for x in k) for k in d["keys"])
    els, anon, keys = set(), set(), set()
    with open(ICSD_INDEX) as f:
        for row in csv.DictReader(f):
            r = composition_keys(row["name"])
            if r: els.add(r[0]); anon.add(r[1]); keys.add(r[2])
    json.dump({"icsd_index_sha256": sha256(ICSD_INDEX), "els": [list(x) for x in sorted(els)], "anon": [list(x) for x in sorted(anon)], "keys": [[list(t) for t in k] for k in sorted(keys)]}, open(cache, "w"))
    return els, anon, keys

def current_graphlet_profiles():
    """Profile exactly the historical partition used by the saved GNoME projection."""
    assignments = pd.read_csv(GRAPHLET_ASSIGNMENTS)
    index = pd.read_csv(ICSD_INDEX, dtype=str)
    if assignments["icsd_id"].isna().any() or assignments["icsd_id"].duplicated().any():
        raise ValueError("Graphlet assignments must contain one non-null row per ICSD id")
    assignments["icsd_id"] = assignments["icsd_id"].astype(int)
    index["icsd_id"] = index["cif_names"].astype(int)  # canonicalizes licensed zero-padded ids
    if index["icsd_id"].duplicated().any():
        raise ValueError("ICSD_index.csv contains duplicate canonical identifiers")
    missing = set(assignments.icsd_id) - set(index.icsd_id)
    if missing:
        raise ValueError(f"{len(missing)} graphlet assignment ids are absent from ICSD_index.csv")
    joined = assignments.merge(
        index[["icsd_id", "name", "sym_group"]], on="icsd_id", how="left", validate="one_to_one"
    )
    if len(joined) != len(assignments) or joined[["name", "sym_group"]].isna().any().any():
        raise ValueError("Graphlet-to-ICSD metadata alignment is incomplete")

    formula_cache = {}
    def reduced(formula):
        if formula not in formula_cache:
            try:
                formula_cache[formula] = Composition(formula).reduced_formula
            except Exception:
                formula_cache[formula] = None
        return formula_cache[formula]
    joined["reduced_formula"] = joined["name"].map(reduced)
    profiles = {}
    members = joined[joined.community >= 0]
    for community, frame in members.groupby("community", sort=True):
        formulas = Counter(frame.reduced_formula.dropna())
        sgs = Counter(frame.sym_group.astype(int))
        profiles[str(int(community))] = {
            "community": int(community), "size": len(frame),
            "n_indexed": len(frame), "n_parsed_formulas": int(frame.reduced_formula.notna().sum()),
            "n_unique_space_groups": len(sgs), "top_space_groups": sgs.most_common(10),
            "top_reduced_formulas": formulas.most_common(10),
            "dominant_space_group_share_all_members": sgs.most_common(1)[0][1] / len(frame),
            "label_status": "Uncurated: formula/SG evidence alone does not verify a prototype identity.",
        }
    expected_sizes = members.groupby("community").size().to_dict()
    observed_sizes = {int(c): p["size"] for c, p in profiles.items()}
    if observed_sizes != expected_sizes:
        raise AssertionError("Generated profiles do not reproduce the current graphlet partition")
    return profiles, {
        "assignment_rows": len(assignments), "indexed_assignment_rows": len(joined),
        "non_noise_rows": len(members), "non_noise_communities": len(profiles),
        "assignment_ids_sha256": ids_sha256(assignments.icsd_id),
        "joined_ids_sha256": ids_sha256(joined.icsd_id),
    }

def main():
    g = load(); n = len(g); print("GNoME records:", n)
    els_ref, anon_ref, key_ref = icsd_reference()
    flags = g.reduced_formula.map(composition_keys)
    g["element_set_in_icsd"] = flags.map(lambda r: r is not None and r[0] in els_ref)
    g["anon_stoich_in_icsd"] = flags.map(lambda r: r is not None and r[1] in anon_ref)
    g["exact_formula_in_icsd"] = flags.map(lambda r: r is not None and r[2] in key_ref)
    # community profiles: dominant space group of the assigned community
    prof_cw = json.load(open(D / "community_evidence/production_full_membership_profiles.json"))
    prof_gr, alignment = current_graphlet_profiles()
    graphlet_profiles_path = OUT / "graphlet_replay_membership_profiles.json"
    with open(graphlet_profiles_path, "w") as f:
        json.dump(prof_gr, f, indent=2, sort_keys=True)
    def top_spg(prof, c):
        p = prof.get(str(int(c))) if pd.notna(c) else None
        return p["top_space_groups"][0][0] if p and p.get("top_space_groups") else None
    g["cw_comm_top_spg"] = g.comm_CrystalWeave.map(lambda c: top_spg(prof_cw, c))
    g["gr_comm_top_spg"] = g.comm_Graphlet_CrystalNN.map(lambda c: top_spg(prof_gr, c))
    g["spg_matches_cw_comm"] = g.spg_number == g.cw_comm_top_spg
    g["spg_matches_gr_comm"] = g.spg_number == g.gr_comm_top_spg
    g["r_CrystalWeave"] = g.d_CrystalWeave / g.p95_CrystalWeave
    g["r_Graphlet_CrystalNN"] = g.d_Graphlet_CrystalNN / g.p95_Graphlet_CrystalNN
    g["high_symmetry"] = g.spg_number >= 143  # trigonal, hexagonal, cubic
    # 1. pattern table across the five maps
    cols = [f"in_{m}" for m in MAPS]
    pat = g[cols].astype(int).astype(str).agg("".join, axis=1)
    pattern_counts = pat.value_counts()
    # 2. CW x graphlet 2x2 and per-cell profile
    g["cell"] = np.where(g.in_CrystalWeave & g.in_Graphlet_CrystalNN.astype(bool), "in both",
                np.where(~g.in_CrystalWeave & g.in_Graphlet_CrystalNN.astype(bool), "graphlet-in / CrystalWeave-out",
                np.where(g.in_CrystalWeave & ~g.in_Graphlet_CrystalNN.astype(bool), "CrystalWeave-in / graphlet-out", "out of both")))
    metrics = {
        "n": ("cell", "size"), "median n_sites": ("n_sites", "median"), "median n_elements": ("n_elements", "median"),
        "share n_elements>=4": ("n_elements", lambda s: (s >= 4).mean()), "median decomposition energy (eV/atom)": ("decomposition_energy_per_atom", "median"),
        "share exact ICSD formula": ("exact_formula_in_icsd", "mean"), "share element set in ICSD": ("element_set_in_icsd", "mean"),
        "share anonymous stoichiometry in ICSD": ("anon_stoich_in_icsd", "mean"), "median ElMD to nearest ICSD formula": ("elmd_all_icsd", "median"),
        "share high-symmetry (trigonal/hexagonal/cubic)": ("high_symmetry", "mean"), "share spg = CrystalWeave community top spg": ("spg_matches_cw_comm", "mean"),
        "share spg = graphlet community top spg": ("spg_matches_gr_comm", "mean"), "median d/p95 CrystalWeave": ("r_CrystalWeave", "median"),
        "median d/p95 graphlet": ("r_Graphlet_CrystalNN", "median"), "share in Magpie": ("in_Magpie", lambda s: s.astype(float).mean()),
        "share in VoronoiNN graphlets": ("in_Graphlet_VoronoiNN", lambda s: s.astype(float).mean()), "share in AMD": ("in_AMD", lambda s: s.astype(float).mean()),
    }
    order = ["in both", "graphlet-in / CrystalWeave-out", "CrystalWeave-in / graphlet-out", "out of both"]
    prof = pd.DataFrame({k: g.groupby("cell")[c].agg(f) for k, (c, f) in metrics.items()}).reindex(order).T
    # 3. contrasts: graphlet-in/CW-out vs in-both
    a, b = g[g.cell == "graphlet-in / CrystalWeave-out"], g[g.cell == "in both"]
    tests = []
    for col, label in [("n_elements", "n_elements"), ("n_sites", "n_sites"), ("elmd_all_icsd", "ElMD to nearest ICSD"), ("decomposition_energy_per_atom", "decomposition energy"), ("r_Graphlet_CrystalNN", "graphlet d/p95")]:
        x, y = a[col].dropna(), b[col].dropna()
        if len(x) and len(y):
            u, p = mannwhitneyu(x, y, alternative="two-sided"); tests.append((label, float(x.median()), float(y.median()), float(p)))
    fish = []
    for col, label in [("exact_formula_in_icsd", "exact formula in ICSD"), ("element_set_in_icsd", "element set in ICSD"), ("anon_stoich_in_icsd", "anonymous stoichiometry in ICSD"), ("spg_matches_cw_comm", "spg = CrystalWeave community top spg"), ("spg_matches_gr_comm", "spg = graphlet community top spg"), ("high_symmetry", "high symmetry")]:
        t = [[int(a[col].sum()), int((~a[col]).sum())], [int(b[col].sum()), int((~b[col]).sum())]]
        orr, p = fisher_exact(t); fish.append((label, a[col].mean(), b[col].mean(), orr, p))
    # 4. receiving graphlet communities for graphlet-in/CW-out; where CrystalWeave would put them
    def describe(prof, c):
        p = prof.get(str(int(c)));
        if not p: return ("?", "?", 0)
        return (", ".join(f"{f}×{k}" for f, k in p["top_reduced_formulas"][:4]), ", ".join(f"{s}×{k}" for s, k in p["top_space_groups"][:3]), p["size"])
    recv = a.groupby("comm_Graphlet_CrystalNN").size().sort_values(ascending=False).head(12)
    recv_rows = [(int(c), int(k), *describe(prof_gr, c)) for c, k in recv.items()]
    cw_rows = [(int(c), int(k), *describe(prof_cw, c)) for c, k in a.groupby("comm_CrystalWeave").size().sort_values(ascending=False).head(12).items()]
    # 5. write
    g.to_csv(OUT / "gnome_profile_records.csv")
    prof.to_csv(OUT / "gnome_profile_by_cell.csv")
    with open(OUT / "gnome_profile.md", "w") as f:
        f.write("# GNoME per-structure profile across the five representation maps\n\n")
        f.write(f"{n} GNoME records; full-record maps, member-p95 basins. In-basin flags: " + ", ".join(f"{m} {g[f'in_{m}'].astype(float).mean()*100:.1f}%" for m in MAPS) + ".\n\n")
        f.write("## In-basin pattern across maps (order: CrystalWeave, Magpie, Graphlet CrystalNN, Graphlet VoronoiNN, AMD; 1 = in basin)\n\n| pattern | n |\n|---|---:|\n")
        for k, v in pattern_counts.head(12).items(): f.write(f"| {k} | {v} |\n")
        f.write(f"\nCrystalWeave × Graphlet (CrystalNN) 2×2: " + ", ".join(f"{k}: {v}" for k, v in g.cell.value_counts().reindex(order).items()) + ".\n\n")
        f.write("## Profile by cell\n\n" + prof.to_markdown(floatfmt=".3f") + "\n\n")
        f.write("## Graphlet-in / CrystalWeave-out versus in-both\n\n| quantity | median (graphlet-in/CW-out) | median (in both) | Mann–Whitney p |\n|---|---:|---:|---:|\n")
        for l, x, y, p in tests: f.write(f"| {l} | {x:.3f} | {y:.3f} | {p:.2e} |\n")
        f.write("\n| flag | share (graphlet-in/CW-out) | share (in both) | odds ratio | Fisher p |\n|---|---:|---:|---:|---:|\n")
        for l, x, y, orr, p in fish: f.write(f"| {l} | {x:.3f} | {y:.3f} | {orr:.2f} | {p:.2e} |\n")
        f.write("\n## Where the graphlet-in / CrystalWeave-out structures land\n\n### Receiving graphlet communities (top 12)\n\n| graphlet community | GNoME n | top ICSD formulas | top space groups | community size |\n|---:|---:|---|---|---:|\n")
        for r in recv_rows: f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n")
        f.write("\n### Nearest CrystalWeave communities of the same structures (outside their radius)\n\n| CrystalWeave community | GNoME n | top ICSD formulas | top space groups | community size |\n|---:|---:|---|---|---:|\n")
        for r in cw_rows: f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("The graphlet-in / CrystalWeave-out structures are assigned to broad replay-partition communities rather than uniformly to small, single-prototype families. Their space group matches the dominant space group of the receiving graphlet community in "
                f"{a.spg_matches_gr_comm.mean()*100:.1f}% of cases, compared with {b.spg_matches_gr_comm.mean()*100:.1f}% among structures in both basins. This agreement test therefore does not support exact prototype reuse as the explanation for graphlet acceptance. The saved local-motif representation establishes neighborhood similarity, while the present profile does not identify a unique physical mechanism for that similarity.\n")
    outputs = [OUT / "icsd_reference_keys.json", graphlet_profiles_path, OUT / "gnome_profile_records.csv", OUT / "gnome_profile_by_cell.csv", OUT / "gnome_profile.md"]
    provenance = {
        "purpose": "GNoME receiver profiles for the exact CrystalNN graphlet replay partition used by projection_full.csv",
        "inputs": {
            str(GRAPHLET_ASSIGNMENTS.relative_to(ROOT)): sha256(GRAPHLET_ASSIGNMENTS),
            str(ICSD_INDEX.relative_to(ROOT)): sha256(ICSD_INDEX),
        },
        "alignment_checks": alignment,
        "outputs": {p.name: sha256(p) for p in outputs},
        "script_sha256": sha256(Path(__file__)),
    }
    with open(OUT / "gnome_profile_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
    print(open(OUT / "gnome_profile.md").read())

if __name__ == "__main__":
    main()
