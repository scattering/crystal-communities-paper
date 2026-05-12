#!/usr/bin/env python3
"""Project an Alexandria-PBE sample into the frozen ICSD structural map.

Companion to analyze_gnome_frontier.py / analyze_mp_frontier.py. Alexandria
is a DFT calculation database (Schmidt+ 2023, Wang+ 2021) that publishes
~5.8M PBE-relaxed structures, including a large predicted/off-hull subset.

Caveats:
  - Alexandria entries do NOT carry direct ICSD provenance flags. We filter
    by energy_above_hull > 0 (off the convex hull → likely-predicted) as a
    proxy for "not experimentally known," and additionally deduplicate
    against the ICSD reduced_formula+space_group set if --icsd-dedup-csv
    is provided.
  - We sample from a small number of bz2 files (default: 1 file, 100K
    entries) rather than streaming the full 58-file release. For 5K-scale
    comparisons with GNoME this is sufficient; for tighter CIs, increase
    --n-source-files.

Output schema matches gnome_frontier_records.csv exactly.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from icsd_densify_worker import build_structure_embedding


ALEXANDRIA_BASE = "https://alexandria.icams.rub.de/data/pbe/2025.07.02"


@dataclass
class AlexRecord:
    material_id: str        # mat_id like "agm005737469"
    reduced_formula: str
    energy_above_hull: float | None
    spg: int | None
    structure_dict: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icsd-features", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--alexandria-dir", required=True,
                        help="Local directory to cache the alexandria_NNNNN.json.bz2 files. "
                        "Files are downloaded on first run.")
    parser.add_argument("--n-source-files", type=int, default=1,
                        help="Number of alexandria_NNNNN.json.bz2 files to download "
                        "(each ~60MB compressed, ~100K entries).")
    parser.add_argument("--source-file-stride", type=int, default=10,
                        help="Stride between source-file indices to span the dataset "
                        "(file index = i * stride). Default 10 spans the 58-file release.")
    parser.add_argument("--min-e-above-hull", type=float, default=0.0,
                        help="Filter candidates with energy_above_hull >= this. Default 0 keeps "
                        "everything; set to e.g. 0.05 to keep only off-hull predictions.")
    parser.add_argument("--max-e-above-hull", type=float, default=0.5,
                        help="Filter candidates with energy_above_hull <= this. Default 0.5 "
                        "drops obviously-unphysical entries.")
    parser.add_argument("--icsd-dedup-csv", default=None,
                        help="Optional CSV with columns reduced_formula, spg from the ICSD set; "
                        "Alexandria entries matching any (reduced_formula, spg) here are dropped.")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sites", type=int, default=256)
    parser.add_argument("--wl-iters", type=int, default=3)
    parser.add_argument("--local-mode", choices=["matminer_ops", "fast_local"], default="matminer_ops")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


from frontier_common import (
    parse_int,
    load_community_rows,
    centroid_thresholds,
)


def load_icsd_dedup_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            f = (row.get("reduced_formula") or "").strip()
            s = parse_int(row.get("spg") or row.get("space_group") or "")
            if f and s is not None:
                keys.add((f, s))
    return keys


def fetch_alexandria_file(alex_dir: Path, idx: int, timeout: int = 120, retries: int = 3) -> Path:
    """Download one Alexandria-PBE shard with timeout + retry + atomic rename.

    Each shard is ~50–80 MB. The Alexandria mirror at icams.rub.de can
    be flaky on the reviewer's first run, so we (a) bound the read with
    a timeout, (b) retry up to ``retries`` times with exponential
    backoff, and (c) write to a ``.tmp`` sibling before renaming so a
    partial-download crash does not leave a corrupt file that the
    ``dest.exists() and st_size > 0`` short-circuit will silently
    accept on the next run.
    """
    import socket
    import time
    from urllib.error import URLError

    name = f"alexandria_{idx:05d}.json.bz2"
    dest = alex_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{ALEXANDRIA_BASE}/{name}"
    alex_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        print(f"downloading {url} -> {dest} (attempt {attempt}/{retries}) ...", flush=True)
        try:
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            try:
                urlretrieve(url, tmp)
            finally:
                socket.setdefaulttimeout(old_timeout)
            if tmp.stat().st_size == 0:
                raise IOError(f"downloaded {tmp} is empty")
            tmp.replace(dest)
            return dest
        except (URLError, IOError, socket.timeout) as exc:
            last_err = exc
            print(f"  attempt {attempt} failed: {exc}", flush=True)
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                backoff = 2 ** attempt
                print(f"  retrying in {backoff}s ...", flush=True)
                time.sleep(backoff)
    raise RuntimeError(f"failed to download {url} after {retries} attempts: {last_err}")


def iter_alexandria_records(path: Path) -> list[AlexRecord]:
    out = []
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data["entries"] if isinstance(data, dict) and "entries" in data else data
    for e in entries:
        d = e.get("data") or {}
        mat_id = d.get("mat_id") or e.get("entry_id")
        formula = d.get("formula")
        spg = d.get("spg")
        eah = d.get("e_above_hull")
        struct = e.get("structure")
        if not mat_id or not formula or not struct:
            continue
        out.append(AlexRecord(
            material_id=str(mat_id),
            reduced_formula=str(formula),
            energy_above_hull=float(eah) if eah is not None else None,
            spg=int(spg) if spg is not None else None,
            structure_dict=struct,
        ))
    return out


def featurize_alex_record(args_tuple) -> tuple[bool, AlexRecord, np.ndarray | None, str | None]:
    rec, wl_iters, local_mode = args_tuple
    try:
        from pymatgen.core import Structure
        structure = Structure.from_dict(rec.structure_dict)
        emb = build_structure_embedding(structure, wl_iters=wl_iters, local_mode=local_mode)
        return True, rec, emb, None
    except Exception as exc:
        return False, rec, None, str(exc)[:200]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.icsd_features)
    community_rows = load_community_rows(Path(args.community_assignments))
    if len(X) != len(community_rows):
        raise ValueError(f"ICSD features rows ({len(X)}) != community rows ({len(community_rows)})")
    community_labels = [int(r["community"]) if r["community"] is not None else -1 for r in community_rows]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca32 = PCA(n_components=32, random_state=args.seed)
    Xp = pca32.fit_transform(Xs)
    pca2 = PCA(n_components=2, random_state=args.seed)
    pca2.fit(Xs)

    communities, centroids, threshold = centroid_thresholds(Xp, community_labels)
    print(f"trained {len(communities)} community thresholds; p95 = {threshold:.4f}", flush=True)

    icsd_dedup: set[tuple[str, int]] = set()
    if args.icsd_dedup_csv:
        icsd_dedup = load_icsd_dedup_keys(Path(args.icsd_dedup_csv))
        print(f"loaded {len(icsd_dedup)} (formula, spg) ICSD dedup keys", flush=True)

    alex_dir = Path(args.alexandria_dir)
    candidates: list[AlexRecord] = []
    for i in range(args.n_source_files):
        idx = i * args.source_file_stride
        path = fetch_alexandria_file(alex_dir, idx)
        recs = iter_alexandria_records(path)
        print(f"file {idx:05d}: {len(recs)} entries before filter", flush=True)
        for r in recs:
            if r.energy_above_hull is None:
                continue
            if not (args.min_e_above_hull <= r.energy_above_hull <= args.max_e_above_hull):
                continue
            if r.structure_dict.get("sites") and len(r.structure_dict["sites"]) > args.max_sites:
                continue
            if icsd_dedup and (r.reduced_formula, r.spg) in icsd_dedup:
                continue
            candidates.append(r)
    print(f"after filters: {len(candidates)} candidates", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = candidates[:args.sample_size]
    print(f"sampling {len(sample)} for featurization", flush=True)

    import concurrent.futures
    results = []
    work = [(rec, args.wl_iters, args.local_mode) for rec in sample]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        for item in ex.map(featurize_alex_record, work, chunksize=16):
            results.append(item)

    kept = [(rec, emb) for ok, rec, emb, _ in results if ok and emb is not None]
    failures = [{"material_id": rec.material_id, "formula": rec.reduced_formula, "detail": err}
                for ok, rec, _, err in results if not ok]
    print(f"featurized {len(kept)}; failures {len(failures)}", flush=True)
    if not kept:
        raise RuntimeError("No Alexandria structures featurized successfully")

    recs = [rec for rec, _ in kept]
    Y = np.vstack([emb for _, emb in kept])
    # Drop rows with NaN features. Some structures (e.g. noble-gas-containing)
    # produce NaN matminer features that crash PCA.transform downstream.
    nan_mask = np.isnan(Y).any(axis=1)
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        print(f"dropping {n_dropped} structures with NaN features", flush=True)
        Y = Y[~nan_mask]
        recs = [r for r, m in zip(recs, nan_mask) if not m]
    Ys = scaler.transform(Y)
    Yp = pca32.transform(Ys)
    Y2 = pca2.transform(Ys)

    dmat = np.linalg.norm(Yp[:, None, :] - centroids[None, :, :], axis=2)
    nearest_idx = np.argmin(dmat, axis=1)
    nearest_comm = communities[nearest_idx]
    nearest_dist = dmat[np.arange(len(Yp)), nearest_idx]
    outlier_like = nearest_dist > threshold

    top_communities = Counter(int(c) for c in nearest_comm[~outlier_like]).most_common(15)
    summary = {
        "source": "alexandria_pbe_2025_07_02",
        "n_source_files_used": args.n_source_files,
        "candidate_pool_size": len(candidates),
        "sample_size_requested": args.sample_size,
        "n_featurized": len(recs),
        "n_failures": len(failures),
        "icsd_centroid_distance_threshold_p95": threshold,
        "n_outlier_like": int(np.count_nonzero(outlier_like)),
        "outlier_like_ratio": float(np.mean(outlier_like)),
        "top_existing_communities": top_communities,
        "mean_nearest_centroid_distance": float(np.mean(nearest_dist)),
        "min_e_above_hull_filter": args.min_e_above_hull,
        "max_e_above_hull_filter": args.max_e_above_hull,
        "icsd_dedup_keys_used": len(icsd_dedup),
    }
    (out_dir / "alexandria_frontier_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "alexandria_frontier_failures.json").write_text(json.dumps(failures[:500], indent=2))

    with (out_dir / "alexandria_frontier_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "material_id", "reduced_formula", "energy_above_hull",
                "assigned_community", "nearest_centroid_distance", "outlier_like",
                "pca1", "pca2",
            ],
        )
        writer.writeheader()
        for rec, comm, dist, is_out, xy in zip(recs, nearest_comm, nearest_dist, outlier_like, Y2):
            writer.writerow({
                "material_id": rec.material_id,
                "reduced_formula": rec.reduced_formula,
                "energy_above_hull": rec.energy_above_hull,
                "assigned_community": int(comm),
                "nearest_centroid_distance": float(dist),
                "outlier_like": bool(is_out),
                "pca1": float(xy[0]),
                "pca2": float(xy[1]),
            })

    print(f"wrote {len(recs)} records to {out_dir}/alexandria_frontier_records.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
