#!/usr/bin/env python3
"""Rebuild frozen full-pool covariates and attempted-cohort primitive site counts.

Reads existing public files only. Uses the historical source filters and the
repaired projector's exact attempted manifest. No embeddings are regenerated.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.groups import SpaceGroup

from external_frozen_cohorts import EXPECTED_POOLS, record_key
from regenerate_external_projection import write_csv
from prepare_repaired_projection_basis import sha256_file

SITE_ZIP = None


@lru_cache(maxsize=512)
def sg_number(value):
    if value is None or str(value).strip() in ("", "na", "None"):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        try:
            number = SpaceGroup(str(value)).int_number
        except ValueError:
            return None
    return number if 1 <= number <= 230 else None


def covariate(mid, formula, n_sites, energy, space_group):
    return {"material_id": mid, "n_elements": len(Composition(formula).elements),
            "n_sites": n_sites, "e_above_hull": energy, "spg_number": sg_number(space_group)}


def load_full(source, path, attempted):
    wanted = {record_key(r) for r in attempted}
    full, selected, files = [], {}, []
    if source == "gnome":
        files = [path / "stable_materials_summary.csv", path / "by_id.zip"]
        with files[0].open(newline="") as handle:
            for row in csv.DictReader(handle):
                mid, formula = row["MaterialId"].strip(), row["Reduced Formula"].strip()
                n = int(float(row["NSites"])) if row["NSites"].strip() else None
                if not mid or not formula or n is not None and n > 256:
                    continue
                energy = row.get("Decomposition Energy Per Atom All") or row.get("Decomposition Energy Per Atom")
                full.append(covariate(mid, formula, n, float(energy) if energy else None, row["Space Group Number"]))
                if mid in wanted:
                    selected[mid] = {"material_id": mid, "_kind": "gnome"}
    elif source == "mp":
        files = [path]
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                structure = row["structure"]
                if not structure.get("sites") or len(structure["sites"]) > 256:
                    continue
                mid = row["material_id"]
                full.append(covariate(mid, row["reduced_formula"], len(structure["sites"]), row.get("energy_above_hull"), row.get("spg")))
                if mid in wanted:
                    selected[mid] = {"material_id": mid, "_kind": "pymatgen", "_structure": structure}
    elif source == "jarvis":
        from analyze_jarvis_frontier import reduced_formula_from_atoms
        files = [path]
        with path.open() as handle:
            raw = json.load(handle)
        for row in raw:
            atoms = row.get("atoms")
            if not atoms or not atoms.get("elements") or (row.get("icsd") or "").strip():
                continue
            try:
                energy = float(row["ehull"])
            except (KeyError, ValueError, TypeError):
                continue
            if not .05 <= energy <= .5 or len(atoms["elements"]) > 256:
                continue
            formula = reduced_formula_from_atoms(atoms)
            if not formula:
                continue
            mid = row.get("jid", "")
            full.append(covariate(mid, formula, len(atoms["elements"]), energy, row.get("spg_number")))
            if mid in wanted:
                selected[mid] = {"material_id": mid, "_kind": "jarvis", "_structure": atoms}
    elif source == "alexandria":
        from analyze_alexandria_frontier import iter_alexandria_records
        files = [path / f"alexandria_{i:05d}.json.bz2" for i in (0, 19, 38)]
        for file in files:
            for row in iter_alexandria_records(file):
                sites = row.structure_dict.get("sites")
                if row.energy_above_hull is None or not .05 <= row.energy_above_hull <= .5 or sites and len(sites) > 256:
                    continue
                mid = row.material_id
                full.append(covariate(mid, row.reduced_formula, len(sites) if sites else None, row.energy_above_hull, row.spg))
                if mid in wanted:
                    selected[mid] = {"material_id": mid, "_kind": "pymatgen", "_structure": row.structure_dict}
    elif source == "mattergen":
        from analyze_external_cif_zip_frontier import load_records
        files = [path]
        for row in load_records(path, ".cif", "symmetrized", 0, 42):
            full.append({"material_id": row.material_id, "zip_member": row.zip_member})
            selected[row.zip_member] = {"material_id": row.material_id, "zip_member": row.zip_member, "_kind": "mattergen"}
    if len(full) != EXPECTED_POOLS[source] or len({record_key(r) for r in full}) != len(full):
        raise ValueError(f"{source}: full pool size/unique keys differ from historical cohort")
    if set(selected) != wanted:
        raise ValueError(f"{source}: attempted IDs are missing from the frozen source pool")
    return full, [selected[record_key(r)] for r in attempted], files


def init_sites(source, path):
    global SITE_ZIP
    if source in ("gnome", "mattergen"):
        SITE_ZIP = zipfile.ZipFile(Path(path) / "by_id.zip" if source == "gnome" else path)


def site_counts(record):
    result = {"material_id": record["material_id"]}
    if record.get("zip_member"):
        result["zip_member"] = record["zip_member"]
    try:
        if record["_kind"] == "pymatgen":
            structure = Structure.from_dict(record["_structure"])
        elif record["_kind"] == "jarvis":
            from analyze_jarvis_frontier import jarvis_atoms_to_structure
            structure = jarvis_atoms_to_structure(record["_structure"])
        else:
            mid = record["material_id"]
            names = [record["zip_member"]] if record["_kind"] == "mattergen" else [
                f"{mid}.cif", f"{mid}.CIF", f"by_id/{mid}.cif", f"by_id/{mid}.CIF", f"{mid}.vasp.cif"]
            for name in names:
                try:
                    data = SITE_ZIP.read(name)
                    break
                except KeyError:
                    continue
            else:
                raise KeyError(mid)
            structure = Structure.from_str(data.decode("utf-8", errors="replace"), fmt="cif")
        result.update(n_sites_cell=len(structure), n_elements=len(structure.composition.elements))
        result["n_sites_primitive"] = len(structure.get_primitive_structure())
        result["site_count_status"] = "ok"
    except Exception as exc:
        result["site_count_status"] = type(exc).__name__
        result["site_count_error"] = str(exc)[:200]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=tuple(EXPECTED_POOLS), required=True)
    for name in ("source-path", "external-root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args(argv)
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    attempt_path = args.external_root / args.source / "attempted_cohort.csv"
    with attempt_path.open(newline="") as handle:
        attempted = list(csv.DictReader(handle))
    if len(attempted) != (386 if args.source == "mattergen" else 5000):
        raise ValueError("Wrong attempted sample size")
    full, selected, files = load_full(args.source, args.source_path, attempted)
    print(f"{args.source}: verified {len(full)} candidate covariates, {len(selected)} attempted structures", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context("spawn"),
            initializer=init_sites, initargs=(args.source, str(args.source_path))) as pool:
        sites = list(pool.map(site_counts, selected, chunksize=16))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.source}_full.csv", full)
    write_csv(args.out_dir / f"{args.source}_sampled_sites.csv", sites)
    metadata = {"source": args.source, "n_full": len(full), "n_attempted": len(attempted),
                "n_cell_counts": sum("n_sites_cell" in r for r in sites),
                "n_primitive_counts": sum("n_sites_primitive" in r for r in sites),
                "n_site_count_failures": sum(r["site_count_status"] != "ok" for r in sites),
                "source_files": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in files],
                "attempted_manifest_sha256": sha256_file(attempt_path),
                "primitive_rule": "pymatgen Structure.get_primitive_structure(), default tolerance; no encoder involved"}
    (args.out_dir / f"{args.source}_covariate_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
