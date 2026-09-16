#!/usr/bin/env python3
"""Validate v2 on audited neighbour endpoints and a seeded ICSD control sample.

Read licensed CIFs only in memory. The archive credential is read from stdin,
and is never included in metadata. All output paths must name a new directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import random
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "graphlet_compare"))
import graphlet_features as gf
from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS
from icsd_densify_worker import build_structure_embedding, read_structure_from_zip

_ZIP = _PASSWORD = _OLD = _ROWS = None


def initialize(zip_path, password, reference, rows):
    global _ZIP, _PASSWORD, _OLD, _ROWS
    warnings.filterwarnings("ignore")
    _ZIP = zipfile.ZipFile(zip_path)
    _PASSWORD = password
    _OLD = np.load(reference, mmap_mode="r")
    _ROWS = rows


def inspect_one(i):
    started = time.monotonic()
    result = {"icsd_id": i}
    try:
        structure = read_structure_from_zip(_ZIP, i, _PASSWORD)
        result.update(n_sites=len(structure), ordered=bool(structure.is_ordered))
        old = _OLD[_ROWS[i]]
        result["old_first204_nonzero"] = int(np.count_nonzero(old[:204]))
        diagnostics = {}
        x0 = build_structure_embedding(structure, 0)
        x3 = build_structure_embedding(structure, 3, diagnostics=diagnostics)
        chemical = np.r_[0:7, 68:75, 136:143]
        geometry = np.r_[7:68, 75:136, 143:204]
        result.update(
            production_ok=True,
            chemical_nonzero=int(np.count_nonzero(x3[chemical])),
            geometry_nonzero=int(np.count_nonzero(x3[geometry])),
            wl_max_change=float(np.max(np.abs(x3 - x0))),
            old_new_max_change=float(np.max(np.abs(x3 - old))),
            diagnostics=diagnostics,
        )
        result["new_vector"] = x3.tolist()
    except Exception as exc:
        result["production_ok"] = False
        result["production_error"] = (type(exc).__name__ + ": " + str(exc)).replace(_PASSWORD, "[redacted]")
    try:
        if "n_sites" not in result:
            raise ValueError("Structure was not parsed")
        raw = gf.all_raw_features(structure)
        # Per-structure ranges suffice to check that every histogram is valid;
        # these pilot histograms are not used for cross-structure NN distances.
        edges = gf.global_bin_edges({str(i): raw})
        hist = gf.histogram_tensor(raw, edges)
        result.update(graphlet_ok=True, graphlet_features=int(hist.shape[0]),
                      graphlet_geometry_observations=sum(len(raw[k]) for k in gf.REGISTRY.second_order + gf.REGISTRY.third_order))
    except Exception as exc:
        result["graphlet_ok"] = False
        result["graphlet_error"] = (type(exc).__name__ + ": " + str(exc)).replace(_PASSWORD, "[redacted]")
    result["runtime_sec"] = time.monotonic() - started
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--icsd-zip", required=True)
    p.add_argument("--reference-features", required=True)
    p.add_argument("--assignments", required=True)
    p.add_argument("--pairs-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--controls-per-stratum", type=int, default=32)
    p.add_argument("--processes", type=int, default=8)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    pairs = json.loads(Path(args.pairs_json).read_text())
    endpoints = {int(row[key]) for row in pairs for key in ("query_id", "neighbor_id")}
    with open(args.assignments) as f:
        ids = [int(row["icsd_id"]) for row in csv.DictReader(f)]
    rows = {i: j for j, i in enumerate(ids)}
    old = np.load(args.reference_features, mmap_mode="r")
    zero = np.all(old[:, :204] == 0, axis=1)
    rng = random.Random(42)
    controls = set()
    for flag in (False, True):
        pool = [ids[j] for j in np.flatnonzero(zero == flag) if ids[j] not in endpoints]
        controls.update(rng.sample(pool, min(args.controls_per_stratum, len(pool))))
    wanted = sorted(endpoints | controls)
    print("READY_FOR_ARCHIVE_CREDENTIAL_ON_STDIN", flush=True)
    password = sys.stdin.readline().rstrip("\r\n")
    if not password:
        raise SystemExit("Archive credential was not supplied")
    started = time.monotonic()
    records = []
    with mp.get_context("spawn").Pool(
        args.processes, initializer=initialize,
        initargs=(args.icsd_zip, password, args.reference_features, rows),
    ) as pool:
        for record in pool.imap_unordered(inspect_one, wanted, chunksize=1):
            record["stratum"] = "audited_endpoint" if record["icsd_id"] in endpoints else "control"
            records.append(record)
            if len(records) % 20 == 0:
                print("INSPECTED", len(records), "OF", len(wanted), flush=True)
    password = None
    records.sort(key=lambda r: r["icsd_id"])
    summary = {"feature_version": FEATURE_VERSION, "neighbor_settings": NEIGHBOR_SETTINGS,
               "n": len(records), "n_audited_endpoints": len(endpoints),
               "runtime_sec": time.monotonic() - started,
               "packages": {k: importlib.metadata.version(k) for k in ("pymatgen", "pymatgen-core", "matminer", "numpy")},
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                  (ROOT/"scripts/icsd_densify_worker.py", ROOT/"scripts/crystal_neighbors.py", ROOT/"experiments/graphlet_compare/graphlet_features.py")}}
    for group in ("all", "audited_endpoint", "control"):
        selected = [r for r in records if group == "all" or r["stratum"] == group]
        summary[group] = {
            "n": len(selected),
            "production_ok": sum(r["production_ok"] for r in selected),
            "graphlet_ok": sum(r["graphlet_ok"] for r in selected),
            "nonzero_chemistry": sum(r.get("chemical_nonzero", 0) > 0 for r in selected),
            "nonzero_geometry": sum(r.get("geometry_nonzero", 0) > 0 for r in selected),
            "message_passing_changes": sum(r.get("wl_max_change", 0) > 0 for r in selected),
            "structures_with_unrepresented_cn_mass": sum(r.get("diagnostics", {}).get("sites_with_unrepresented_cn_mass", 0) > 0 for r in selected),
        }
    vectors = [r.pop("new_vector") for r in records if "new_vector" in r]
    np.save(out/"pilot_features.npy", np.asarray(vectors))
    (out/"pilot_feature_ids.json").write_text(json.dumps([r["icsd_id"] for r in records if r["production_ok"]]))
    (out/"records.json").write_text(json.dumps(records, indent=2))
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
