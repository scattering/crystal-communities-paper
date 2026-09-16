#!/usr/bin/env python3
"""Full-ICSD graphlet community-recovery (the cheap, scalable variant).

Question: across ALL ICSD entries, does an INDEPENDENT structural
representation (graphlet histograms) recover the production Louvain
communities? If a structure's graphlet-nearest-neighbour sits in the
same community at a high rate, the communities are representation-robust
structural objects, not artifacts of our matminer / message-passing /
moment-pooling embedding.

Key feasibility trick: graphlet-EMD between two structures
  = mean_f  sum_b |CDF_a[f,b] - CDF_b[f,b]|
  = (1/F) * || cdf_a - cdf_b ||_1   on the flattened (F*B)=1280-d CDF.
So nearest-neighbour under graphlet-EMD == L1 nearest-neighbour in CDF
space → approximate-NN (pynndescent, metric='manhattan'), O(N log N),
NOT an infeasible O(N^2) dense matrix.

NO GP. Retains feature CDFs, row IDs and neighbour identities for auditing;
no CIFs/structures are written.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
import gc
import multiprocessing as mp
# 'spawn' instead of fork(): workers do NOT inherit the parent's heap,
# so the large pass-A subsample (~5 GB) can't cause os.fork() ENOMEM
# when the pass-B Pool launches 112 children.
_MP_CTX = mp.get_context("spawn")
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))

from icsd_densify_worker import read_structure_from_zip  # noqa: E402
from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS  # noqa: E402
import graphlet_features as gf  # noqa: E402

_ZIP = None
_PWD = None
_EDGES = None


def _winit(zip_path, pwd, edges=None):
    global _ZIP, _PWD, _EDGES
    _ZIP = zipfile.ZipFile(zip_path)
    _PWD = pwd
    _EDGES = edges


def _feat_raw(icsd_id):
    """Pass A sends numeric values only; histogram weights do not set bin edges."""
    stage = "read_structure"
    try:
        s = read_structure_from_zip(_ZIP, icsd_id, _PWD)
        stage = "raw_features"
        raw = gf.all_raw_features(s)
        stage = "bin_values"
        values = {name: np.fromiter((v for v, _ in raw[name]), dtype=np.float64,
                                    count=len(raw[name])) for name in gf.REGISTRY.all}
        return int(icsd_id), values, None
    except Exception as exc:
        return int(icsd_id), None, _failure(icsd_id, stage, exc)


def _feat_cdf(icsd_id):
    """Pass B: return ONLY the flattened (F*B) CDF — never the raw lists,
    so parent peak memory is the CDF matrix, not all raw features."""
    stage = "read_structure"
    try:
        s = read_structure_from_zip(_ZIP, icsd_id, _PWD)
        stage = "raw_features"
        raw = gf.all_raw_features(s)
        stage = "histogram"
        H = gf.histogram_tensor(raw, _EDGES)            # (F,B)
        return int(icsd_id), np.cumsum(H, axis=1).reshape(-1).astype(np.float32), None
    except Exception as exc:
        return int(icsd_id), None, _failure(icsd_id, stage, exc)


def _failure(icsd_id, stage, exc):
    # Exception messages may include archive contents or credentials.
    return {"icsd_id": int(icsd_id), "stage": stage,
            "exception_type": type(exc).__name__}


def _coverage(requested, successful, failures):
    return {"n_requested": int(requested), "n_successful": int(successful),
            "n_failed": len(failures)}


def _bin_edges_from_values(samples):
    """Consume numeric pass-A arrays, preserving global_bin_edges exactly.

    The original rule uses unweighted sample values and the 0.5/99.5
    percentiles. Release each feature's sample arrays once its edges are set.
    """
    edges = {}
    for name in gf.REGISTRY.all:
        arrays = [sample.pop(name) for sample in samples.values()]
        if not any(a.size for a in arrays):
            raise ValueError(f"No observations for bin-edge feature: {name}")
        arr = np.concatenate(arrays)
        del arrays
        lo, hi = float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            lo, hi = float(arr.min()), float(arr.max())
            if hi - lo < 1e-9:
                hi = lo + 1.0
        edges[name] = [lo, hi]
        del arr
    return edges


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assignments", required=True,
                   help="row-aligned icsd_id source (sample_assignments.csv)")
    p.add_argument("--community-labels", required=True,
                   help="community_assignments_labels3.csv (icsd_id,year,community)")
    p.add_argument("--icsd-zip", required=True)
    passwords = p.add_mutually_exclusive_group()
    passwords.add_argument("--zip-password", help="Legacy; prefer --zip-password-stdin")
    passwords.add_argument("--zip-password-stdin", action="store_true",
                           help="Read one password line from stdin; otherwise use ICSD_ZIP_PASSWORD")
    p.add_argument("--limit", type=int, default=0,
                   help="cap #structures (0 = all valid); for smoke tests")
    p.add_argument("--procs", type=int, default=0)
    p.add_argument("--nn", type=int, default=15)
    p.add_argument("--embedding-baseline", type=float, default=None,
                   help="Optional external reference rate; not calculated or version-validated here")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.zip_password_stdin:
        args.zip_password = sys.stdin.readline().rstrip("\r\n")
        if not args.zip_password:
            p.error("--zip-password-stdin requires a non-empty input line")
    elif args.zip_password is None:
        args.zip_password = os.environ.get("ICSD_ZIP_PASSWORD") or None
    return args


def load_icsd_order(path):
    ids = []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        cols = {c.lower(): c for c in rdr.fieldnames}
        idc = next((cols[k] for k in ("icsd_id", "cif_id", "id", "coll_code")
                    if k in cols), rdr.fieldnames[0])
        for r in rdr:
            try:
                ids.append(int(float(r[idc])))
            except Exception:
                ids.append(-1)
    return ids


def load_comm(path):
    m = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                m[int(float(r["icsd_id"]))] = int(float(r["community"]))
            except Exception:
                continue
    return m


def main():
    a = parse_args()
    t0 = time.time()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    comm = load_comm(a.community_labels)
    order = load_icsd_order(a.assignments)
    # all entries with a valid (non-noise) production community + an id
    targets = []
    seen = set()
    for iid in order:
        if iid > 0 and iid not in seen and comm.get(iid, -1) >= 0:
            seen.add(iid)
            targets.append(iid)
    if a.limit:
        targets = targets[:a.limit]
    print(f"targets: {len(targets)} ICSD entries with valid Louvain community",
          flush=True)

    procs = a.procs or os.cpu_count() or 1
    pwd = a.zip_password
    F, B = gf.NUM_FEATURES, gf.NUM_BINS
    min_keep = max(50, len(targets) // 5)  # scales with --limit; smoke-safe
    provenance = {
        "feature_version": FEATURE_VERSION,
        "neighbor_settings": dict(NEIGHBOR_SETTINGS),
        "representation": "graphlet-64x20-periodic-images-short-long-arms",
        "community_labels_source": {"path": str(a.community_labels),
            "role": "external comparison labels; not regenerated or feature-version-validated here"},
        "assignments_source": str(a.assignments),
        "cdf_row_order": "ascending ICSD ID",
    }
    report = {**provenance, "bin_edge_sampling": None, "features": None}
    report_path = out / "featurization_report.json"

    # ---- Pass A: bin edges from a seeded representative subsample only
    # (percentile edges are stable; avoids holding all raw features) ----
    rng = np.random.default_rng(42)
    edge_n = min(len(targets), 5000)
    edge_ids = list(rng.choice(targets, size=edge_n, replace=False))
    raw_sub = {}
    edge_failures = []
    with _MP_CTX.Pool(procs, initializer=_winit, initargs=(a.icsd_zip, pwd)) as pool:
        for iid, raw, failure in pool.imap_unordered(_feat_raw, edge_ids, chunksize=1):
            if raw is not None:
                raw_sub[iid] = raw
            else:
                edge_failures.append(failure)
    report["bin_edge_sampling"] = {
        **_coverage(edge_n, len(raw_sub), edge_failures), "failures": edge_failures,
        "requested_icsd_ids": [int(i) for i in edge_ids],
        "successful_icsd_ids": sorted(raw_sub),
    }
    report_path.write_text(json.dumps(report, indent=2))
    try:
        edges = _bin_edges_from_values(raw_sub)
    except Exception as exc:
        report["run_failure"] = {"stage": "bin_edges", "exception_type": type(exc).__name__}
        report_path.write_text(json.dumps(report, indent=2))
        raise SystemExit(f"Bin-edge generation failed ({type(exc).__name__}); see {report_path}")
    (out / "bin_edges.json").write_text(json.dumps(edges, indent=2))
    print(f"bin edges from {len(raw_sub)}/{edge_n} subsample "
          f"({time.time()-t0:.0f}s)", flush=True)
    del raw_sub
    gc.collect()
    try:  # release freed heap back to the OS (extra belt-and-braces vs fork ENOMEM)
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    # ---- Pass B: featurize ALL targets, workers emit ONLY the 1280-d CDF;
    # parent peak = CDF matrix (~0.86 GB at full ICSD), not raw features ----
    cdf = np.empty((len(targets), F * B), dtype=np.float32)
    ids = []
    failures = []
    with _MP_CTX.Pool(procs, initializer=_winit,
              initargs=(a.icsd_zip, pwd, edges)) as pool:
        for k, (iid, v, failure) in enumerate(
                pool.imap_unordered(_feat_cdf, targets, chunksize=16)):
            if v is not None:
                cdf[len(ids)] = v
                ids.append(iid)
            else:
                failures.append(failure)
            if (k + 1) % 20000 == 0:
                print(f"  featurized {k+1}/{len(targets)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    cdf = cdf[:len(ids)]
    row_order = np.argsort(ids)
    cdf = cdf[row_order]
    ids = [int(ids[i]) for i in row_order]
    report["features"] = {**_coverage(len(targets), len(ids), failures),
                          "failures": failures}
    report_path.write_text(json.dumps(report, indent=2))
    np.save(out / "graphlet_cdf.npy", cdf)
    (out / "graphlet_cdf.ids.json").write_text(json.dumps(ids))
    print(f"featurized {len(ids)}/{len(targets)} ({procs} procs); "
          f"CDF {cdf.shape} ({time.time()-t0:.0f}s)", flush=True)
    if len(ids) < min_keep:
        raise SystemExit(f"too few featurized ({len(ids)} < {min_keep})")

    try:
        from pynndescent import NNDescent
    except Exception as e:
        raise SystemExit(
            f"pynndescent required for scalable L1 ANN ({type(e).__name__}); "
            f"env lacks it — install or run the n=5000 variant instead.")

    index = NNDescent(cdf, metric="manhattan", n_neighbors=a.nn,
                      random_state=42, low_memory=True)
    nbr_idx, nbr_dist = index.neighbor_graph
    np.save(out / "ann_neighbor_indices.npy", nbr_idx)
    np.save(out / "ann_neighbor_distances.npy", nbr_dist)
    print(f"ANN graph built ({time.time()-t0:.0f}s)", flush=True)

    comm_arr = np.array([comm[i] for i in ids])
    hits, percomm = [], defaultdict(list)
    miss_targets = defaultdict(Counter)        # comm_r -> Counter of comm_j (misses)
    confusion_pairs = Counter()                # (comm_r, comm_j) raw counts, misses only
    for r in range(len(ids)):
        j = next((int(x) for x in nbr_idx[r] if 0 <= x < len(ids) and x != r), None)
        if j is None:
            continue
        cr, cj = int(comm_arr[r]), int(comm_arr[j])
        same = (cr == cj)
        hits.append(1.0 if same else 0.0)
        percomm[cr].append(1.0 if same else 0.0)
        if not same:
            miss_targets[cr][cj] += 1
            confusion_pairs[(cr, cj)] += 1

    overall = float(np.mean(hits)) if hits else float("nan")
    per = {}
    for c, v in sorted(percomm.items()):
        tops = miss_targets[c].most_common(3)
        per[str(c)] = {"size": len(v), "recovery": float(np.mean(v)),
                       "top_miss_targets": [[int(t), int(n)] for t, n in tops]}
    top_pairs = [{"from": int(a_), "to": int(b_), "count": int(n)}
                 for (a_, b_), n in confusion_pairs.most_common(50)]
    res = {
        **provenance,
        "variant": "full_ICSD_community_recovery",
        **_coverage(len(targets), len(ids), failures),
        "bin_edge_coverage": _coverage(edge_n, edge_n - len(edge_failures), edge_failures),
        "n_structures": len(ids),
        "n_scored": len(hits),
        "n_without_nonself_neighbor": len(ids) - len(hits),
        "n_communities": len(percomm),
        "graphlet_fullICSD_same_community_recovery": overall,
        "embedding_fullICSD_same_community_baseline": a.embedding_baseline,
        "embedding_baseline_source": "optional external reference; not computed here",
        "ann_metric": "manhattan(==graphlet-EMD on CDF)",
        "ann_n_neighbors": a.nn,
        "procs": procs,
        "runtime_sec": round(time.time() - t0, 1),
        "top_confusion_pairs": top_pairs,
        "per_community": per,
        "artifacts": {"cdf": "graphlet_cdf.npy", "cdf_ids": "graphlet_cdf.ids.json",
            "bin_edges": "bin_edges.json", "ann_indices": "ann_neighbor_indices.npy",
            "ann_distances": "ann_neighbor_distances.npy", "failures": report_path.name},
    }
    (out / "full_recovery_result.json").write_text(json.dumps(res, indent=2))
    print("=== RESULT ===")
    for k, v in res.items():
        if k != "per_community":
            print(f"  {k}: {v}")
    print(f"wrote {out/'full_recovery_result.json'}")


if __name__ == "__main__":
    main()
