#!/usr/bin/env python3
"""Compute Average Minimum Distance (AMD) fingerprints for the ICSD subset.

Streams CIFs from the password-protected ICSD_CIFs.zip (no on-disk extraction),
converts each to pymatgen Structure -> amd.PeriodicSet, and computes
amd.AMD(structure, k=100) per crystal. Saves a (N, 100) numpy array of AMD
fingerprints + a parallel list of icsd_ids.

Parallel via joblib across the available cores on the SLURM node.

Reads ICSD_ZIP_PASSWORD from environment (one-shot, never written to disk).
"""
from __future__ import annotations
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import amd
from amd.io import periodicset_from_pymatgen_structure
from pymatgen.core import Structure

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(os.environ.get("CRYSTAL_COMMUNITIES_TACC_ROOT", os.environ.get("WORK", "/path/to/tacc/work")))  # TACC work root; set explicitly off-site
ICSD_ZIP = REPO_ROOT / "reference_data/ICSD_CIFs.zip"
ICSD_INDEX = REPO_ROOT / "reference_data/ICSD_index.csv"
ICSD_ID_SUBSET = REPO_ROOT / "icsd_graph_runs/full_matminer_ops_spr_20260419_labels3/community_assignments.csv"
OUT_DIR = REPO_ROOT / "amd_compare_runs" / f"amd_features_{os.environ.get('SLURM_JOB_ID', 'local')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 100               # AMD neighbour count, matching Widdowson-Kurlin published default
N_JOBS = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
BATCH = 500           # progress reporting cadence

# --------------------------------------------------------------------------- #
# CIF name resolution (zip member naming conventions vary)
# --------------------------------------------------------------------------- #
def build_zip_name_index(zf: zipfile.ZipFile) -> dict[int, str]:
    """Map icsd_id -> zip member name. Handles both 'icsd_NNN.cif' and 'NNN.cif'."""
    idx: dict[int, str] = {}
    for name in zf.namelist():
        base = name.rsplit("/", 1)[-1]
        if not base.lower().endswith(".cif"):
            continue
        stem = base[:-4]
        if stem.lower().startswith("icsd_"):
            stem = stem[5:]
        try:
            iid = int(stem)
            idx[iid] = name
        except ValueError:
            continue
    return idx


# --------------------------------------------------------------------------- #
# Worker (zipfile + password handle per process to avoid pickle problems)
# --------------------------------------------------------------------------- #
def _amd_worker(payloads: list[tuple[int, bytes]], k: int) -> list[tuple[int, np.ndarray | None, str]]:
    """Process a batch of (icsd_id, raw_cif_bytes). Returns (id, amd_vec, error)."""
    results: list[tuple[int, np.ndarray | None, str]] = []
    for iid, raw in payloads:
        try:
            text = raw.decode("utf-8", errors="replace")
            struct = Structure.from_str(text, fmt="cif")
            pset = periodicset_from_pymatgen_structure(struct)
            vec = amd.AMD(pset, k)
            results.append((iid, np.asarray(vec, dtype=np.float32), ""))
        except Exception as e:
            results.append((iid, None, f"{type(e).__name__}: {e}"))
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pw = os.environ.get("ICSD_ZIP_PASSWORD", "").encode() or None
    if not pw:
        sys.exit("ICSD_ZIP_PASSWORD not set in env; aborting.")

    # 1. Resolve the ICSD subset (must match labels3 partition exactly)
    target_ids: list[int] = []
    import csv as csvlib
    with ICSD_ID_SUBSET.open() as f:
        r = csvlib.DictReader(f)
        for row in r:
            try:
                if int(row["community"]) < 0:
                    continue
                target_ids.append(int(row["icsd_id"]))
            except (KeyError, ValueError):
                continue
    target_ids = sorted(set(target_ids))
    print(f"  target ICSD subset (community-assigned): {len(target_ids):,}", flush=True)

    # 2. Open zip + resolve member-name mapping
    zf = zipfile.ZipFile(ICSD_ZIP)
    zf.setpassword(pw)
    name_idx = build_zip_name_index(zf)
    resolved = [(iid, name_idx[iid]) for iid in target_ids if iid in name_idx]
    print(f"  zip coverage: {len(resolved):,}/{len(target_ids):,}", flush=True)

    # 3. Streaming feed: read CIFs in batches, dispatch to workers
    pw = None  # drop password from this scope; sub-processes don't need it

    def feed_batches():
        batch: list[tuple[int, bytes]] = []
        for iid, member in resolved:
            try:
                batch.append((iid, zf.read(member)))
            except Exception as e:
                print(f"    zip-read fail for {iid}: {e}", flush=True)
                continue
            if len(batch) >= BATCH:
                yield batch
                batch = []
        if batch:
            yield batch

    t0 = time.time()
    ids_done: list[int] = []
    amds: list[np.ndarray] = []
    failures: list[tuple[int, str]] = []

    # Read all CIF bytes upfront (fits comfortably — ~300 MB), then dispatch
    # in batches to the worker pool. Avoids the joblib generator-streaming API
    # that requires a newer loky backend than this env has.
    all_batches = list(feed_batches())
    print(f"  prepared {len(all_batches)} batches ({sum(len(b) for b in all_batches):,} CIFs in memory)", flush=True)

    # Periodic-progress callback wrapper for joblib
    batch_results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(_amd_worker)(b, K) for b in all_batches
    )
    for results in batch_results:
        for iid, vec, err in results:
            if vec is None:
                failures.append((iid, err))
                continue
            ids_done.append(iid)
            amds.append(vec)
        elapsed = time.time() - t0
        if len(ids_done) % 5000 == 0 or len(ids_done) == len(resolved):
            rate = len(ids_done) / max(elapsed, 1e-6)
            print(f"    [{elapsed:6.0f}s]  done={len(ids_done):>6,}  failed={len(failures):>4}  rate={rate:5.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  done in {elapsed/60:.1f} min: {len(ids_done):,} computed, {len(failures)} failed", flush=True)

    # 4. Save
    arr = np.stack(amds).astype(np.float32) if amds else np.empty((0, K), dtype=np.float32)
    np.save(OUT_DIR / "amd_features.npy", arr)
    with (OUT_DIR / "amd_features.npy.ids.json").open("w") as f:
        json.dump(ids_done, f)
    if failures:
        with (OUT_DIR / "amd_failures.json").open("w") as f:
            json.dump(failures, f, indent=2)
    print(f"  saved: {OUT_DIR / 'amd_features.npy'}  shape={arr.shape}", flush=True)


if __name__ == "__main__":
    main()
