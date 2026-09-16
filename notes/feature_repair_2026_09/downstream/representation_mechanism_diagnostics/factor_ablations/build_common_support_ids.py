#!/usr/bin/env python3
"""Materialize the exact five-representation common-support identifiers.

Reproduces the intersection logic of
representations/amd-external-full-map/build_five_representation_common_support.py
(same retained projection tables, same flag reconstruction check) and writes the
sorted identifier lists so a remote job can evaluate every ablation on exactly the
same evaluation populations. The per-population SHA-256 must equal the values in
five_representation_common_support.json; the script refuses to write otherwise.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / ".git").exists())
DOWN = REPO / "notes/feature_repair_2026_09/downstream"
AMD = DOWN / "representations/amd-external-full-map"
SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")
LABELS = {"gnome": "GNoME", "mattergen": "MatterGen", "mp": "MP", "jarvis": "JARVIS", "alexandria": "Alexandria"}
REPS = {
    "CrystalWeave": {"icsd": DOWN / "external_representation/production_reference/icsd_full.csv",
                     "external": {s: DOWN / "external" / s / (("mattergen-public" if s == "mattergen" else s) + "_frontier_records.csv") for s in SOURCES}},
    "Magpie": {"icsd": DOWN / "external_representation/consistent_transform/basis/icsd_full.csv", "root": DOWN / "external_representation/consistent_transform/external"},
    "CrystalNN graphlets": {"icsd": DOWN / "external_representation/graphlet/basis/icsd_full.csv", "root": DOWN / "external_representation/graphlet/external"},
    "VoronoiNN graphlets": {"icsd": DOWN / "representation_mechanism_diagnostics/voronoi_external/results/basis/icsd_full.csv", "root": DOWN / "representation_mechanism_diagnostics/voronoi_external/results/external"},
    "AMD": {"icsd": AMD / "results/icsd_projection_full.csv", "root": AMD / "results"},
}


def digest_ids(values):
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path, key):
    frame = pd.read_csv(path, dtype={key: str}, float_precision="round_trip")
    if frame[key].isna().any() or not frame[key].is_unique:
        raise ValueError(path)
    values = frame.in_basin.astype(str).str.lower()
    if not values.isin(["true", "false"]).all():
        raise ValueError(path)
    flags = values.eq("true")
    if not flags.equals(frame.nearest_centroid_distance.le(frame.community_threshold_p95)):
        raise ValueError(f"flags do not reproduce {path}")
    return set(frame[key].astype(str))


def main():
    reference = json.loads((AMD / "five_representation_common_support.json").read_text())["common_populations"]
    ids, inputs = {}, {}
    pops = ("ICSD",) + tuple(LABELS[s] for s in SOURCES)
    for rep, spec in REPS.items():
        paths = {"ICSD": spec["icsd"], **{LABELS[s]: (spec["external"][s] if "external" in spec else spec["root"] / s / "projection_full.csv") for s in SOURCES}}
        for pop, path in paths.items():
            key = "record_key"
            if rep == "CrystalWeave" and pop != "ICSD":
                key = "zip_member" if pop == "MatterGen" else "material_id"
            ids.setdefault(pop, []).append(load(path, key))
            inputs[path.relative_to(REPO).as_posix()] = digest_file(path)
    common = {pop: sorted(set.intersection(*ids[pop])) for pop in pops}
    for pop in pops:
        if len(common[pop]) != reference[pop]["n"] or digest_ids(common[pop]) != reference[pop]["ids_sha256"]:
            raise SystemExit(f"Common support for {pop} does not reproduce the five-way table")
    out = HERE / "inputs" / "common_support_ids.json"
    out.write_text(json.dumps({
        "scope": "Exact five-representation common evaluation identifiers (CrystalWeave, Magpie, CrystalNN graphlets, VoronoiNN graphlets, AMD); reproduced from the retained projection tables.",
        "reference": "representations/amd-external-full-map/five_representation_common_support.json",
        "ids_hash_convention": "sha256 of sorted identifiers joined by LF, no trailing LF",
        "populations": {pop: {"n": len(common[pop]), "ids_sha256": digest_ids(common[pop]), "ids": common[pop]} for pop in pops},
        "inputs_sha256": inputs,
        "script_sha256": digest_file(__file__),
    }, indent=1) + "\n")
    print(json.dumps({pop: {"n": len(common[pop]), "ids_sha256": digest_ids(common[pop])} for pop in pops}, indent=1))
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
