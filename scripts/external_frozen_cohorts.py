"""Local-only readers for the exact five historical external sampling rules.

No download or authenticated API fallback is provided. The 5,000 attempted IDs
are replayed before feature failures, including the old MP/JARVIS NaN drops.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

EXPECTED_POOLS = {"gnome": 554054, "mp": 20479, "jarvis": 22022,
                  "alexandria": 154942, "mattergen": 386}


def record_key(record):
    return record.get("zip_member") or record["material_id"]


def public_fields(record):
    return {k: v for k, v in record.items() if not k.startswith("_")}


def load_cohort(source, source_path, historical_records):
    path = Path(source_path)
    files, candidates = [], []
    if source == "gnome":
        from analyze_gnome_frontier import load_gnome_summary
        files = [path / "stable_materials_summary.csv", path / "by_id.zip"]
        for file in files:
            if not file.is_file():
                raise FileNotFoundError(file)
        # This reader already performs Random(42).shuffle; retain its order.
        rows = load_gnome_summary(files[0], sample_size=10**9, seed=42, max_sites=256)
        candidates = [{"material_id": r.material_id, "reduced_formula": r.reduced_formula,
                       "decomposition_energy_per_atom": r.decomp_e, "_kind": "gnome"} for r in rows]
    elif source == "mp":
        files = [path]
        with path.open() as handle:
            for line in handle:
                r = json.loads(line)
                structure = r["structure"]
                if not structure.get("sites") or len(structure["sites"]) > 256:
                    continue
                # The frozen API cache was already filtered to theoretical,
                # no ICSD provenance, and 0 <= E_hull <= .2. Do not re-query.
                candidates.append({"material_id": r["material_id"], "reduced_formula": r["reduced_formula"],
                                   "energy_above_hull": r.get("energy_above_hull"), "spg": r.get("spg"),
                                   "_kind": "pymatgen", "_structure": structure})
    elif source == "jarvis":
        from analyze_jarvis_frontier import reduced_formula_from_atoms
        files = [path]
        with path.open() as handle:
            raw = json.load(handle)
        for r in raw:
            atoms = r.get("atoms")
            if not atoms or not atoms.get("elements") or (r.get("icsd") or "").strip():
                continue
            try:
                energy = float(r["ehull"])
            except (KeyError, TypeError, ValueError):
                continue
            if not .05 <= energy <= .5 or len(atoms["elements"]) > 256:
                continue
            formula = reduced_formula_from_atoms(atoms)
            if formula:
                candidates.append({"material_id": r.get("jid", ""), "reduced_formula": formula,
                                   "ehull": energy, "formation_energy": r.get("formation_energy_peratom"),
                                   "_kind": "jarvis", "_structure": atoms})
    elif source == "alexandria":
        from analyze_alexandria_frontier import iter_alexandria_records
        files = [path / f"alexandria_{i:05d}.json.bz2" for i in (0, 19, 38)]
        for file in files:
            # Read existing shards directly; never call the download helper.
            for r in iter_alexandria_records(file):
                if r.energy_above_hull is None or not .05 <= r.energy_above_hull <= .5:
                    continue
                if r.structure_dict.get("sites") and len(r.structure_dict["sites"]) > 256:
                    continue
                candidates.append({"material_id": r.material_id, "reduced_formula": r.reduced_formula,
                                   "energy_above_hull": r.energy_above_hull, "spg": r.spg,
                                   "_kind": "pymatgen", "_structure": r.structure_dict})
    elif source == "mattergen":
        from analyze_external_cif_zip_frontier import load_records
        files = [path]
        for r in load_records(path, ".cif", "symmetrized", 0, 42):
            candidates.append({"material_id": r.material_id, "zip_member": r.zip_member,
                               "family": r.family, "task": r.zip_member.split("/")[2],
                               "reduced_formula": None, "_kind": "mattergen"})
    else:
        raise ValueError(f"Unknown source: {source}")
    n_pool = len(candidates)
    if n_pool != EXPECTED_POOLS[source]:
        raise ValueError(f"{source} candidate pool changed: expected {EXPECTED_POOLS[source]}, found {n_pool}")
    if source not in ("gnome", "mattergen"):
        random.Random(42).shuffle(candidates)
    selected = candidates if source == "mattergen" else candidates[:5000]
    keys = [record_key(r) for r in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("Attempted cohort has duplicate keys")
    with Path(historical_records).open(newline="") as handle:
        old_keys = [record_key(r) for r in csv.DictReader(handle)]
    old_set = set(old_keys)
    if len(old_set) != len(old_keys) or not old_set.issubset(keys):
        raise ValueError("Historical successful IDs are not a unique subset of the replayed attempted cohort")
    if [key for key in keys if key in old_set] != old_keys:
        raise ValueError("Historical successful ID order differs from the replayed sample")
    return selected, old_keys, files, {
        "candidate_pool_size": n_pool, "n_attempted": len(selected), "seed": 42,
        "n_historical_successful": len(old_keys), "historical_successful_order_reproduced": True,
        "historically_omitted_attempted_keys": [key for key in keys if key not in old_set],
    }
