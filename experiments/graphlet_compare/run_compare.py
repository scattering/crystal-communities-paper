#!/usr/bin/env python3
"""Structural-geometry comparison: graphlet-EMD vs. our PCA embedding (v2).

NO GP. Purely structural. v2 fixes the two confounds from the n=400 pilot:
  (1) uses the PRODUCTION Louvain `community` labels
      (community_assignments_labels3.csv), not raw HDBSCAN.
  (2) samples WHOLE communities (not uniform-random), so within-sample
      nearest neighbours are real — removes the sparsity deflation.
Plus: featurization parallelised across all node cores; an extra
full-ICSD same-community baseline calibrates the embedding's true
neighbour-recovery rate.

Primary (label-free): Mantel/Spearman of pairwise distances + mean
k-NN overlap. Secondary: same-Louvain-community nearest-neighbour
recovery, embedding vs. graphlet, with the full-ICSD baseline.
Aggregate JSON only — no CIFs/structures written.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool
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


def _winit(zip_path, pwd):
    global _ZIP, _PWD
    _ZIP = zipfile.ZipFile(zip_path)
    _PWD = pwd


def _featurize_one(icsd_id):
    """Return raw features or a sanitized, stage-specific failure record."""
    stage = "read_structure"
    try:
        s = read_structure_from_zip(_ZIP, icsd_id, _PWD)
        stage = "raw_features"
        return int(icsd_id), gf.all_raw_features(s), None
    except Exception as exc:
        return int(icsd_id), None, _failure(icsd_id, stage, exc)


def _failure(icsd_id, stage, exc):
    return {"icsd_id": int(icsd_id), "stage": stage,
            "exception_type": type(exc).__name__}


def _coverage(requested, successful, failures):
    return {"n_requested": int(requested), "n_successful": int(successful),
            "n_failed": len(failures)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-pca", required=True)
    p.add_argument("--assignments", required=True,
                   help="row-aligned with features_pca: row i -> icsd_id")
    p.add_argument("--community-labels", required=True,
                   help="community_assignments_labels3.csv (icsd_id,year,community)")
    p.add_argument("--icsd-zip", required=True)
    passwords = p.add_mutually_exclusive_group()
    passwords.add_argument("--zip-password", help="Legacy; prefer --zip-password-stdin")
    passwords.add_argument("--zip-password-stdin", action="store_true",
                           help="Read one password line from stdin; otherwise use ICSD_ZIP_PASSWORD")
    p.add_argument("--n-sample", type=int, default=5000)
    p.add_argument("--per-community-cap", type=int, default=250)
    p.add_argument("--min-community-size", type=int, default=30)
    p.add_argument("--knn", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--procs", type=int, default=0, help="0 = all node cores")
    p.add_argument("--full-baseline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.zip_password_stdin:
        args.zip_password = sys.stdin.readline().rstrip("\r\n")
        if not args.zip_password:
            p.error("--zip-password-stdin requires a non-empty input line")
    elif args.zip_password is None:
        args.zip_password = os.environ.get("ICSD_ZIP_PASSWORD") or None
    return args


def load_rowmap(path):
    """row index -> icsd_id (order == features_pca rows)."""
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


def load_communities(path):
    """icsd_id -> production Louvain community (int)."""
    m = {}
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                m[int(float(r["icsd_id"]))] = int(float(r["community"]))
            except Exception:
                continue
    return m


def emd_cdf_distance_matrix(H):
    C = np.cumsum(H, axis=2)
    n = C.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        D[a] = np.abs(C[a][None] - C).sum(axis=2).mean(axis=1)
    return D


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def knn_sets(D, k):
    out = []
    for i in range(D.shape[0]):
        order = np.argsort(D[i])
        nn = [int(j) for j in order if j != i][:k]
        out.append(set(nn))
    return out


def nn_same_comm(D, comm):
    hit = []
    for i in range(D.shape[0]):
        j = next((int(j) for j in np.argsort(D[i], kind="stable") if j != i), None)
        if j is None:
            continue
        if comm[i] >= 0 and comm[j] >= 0:
            hit.append(1.0 if comm[i] == comm[j] else 0.0)
    return float(np.mean(hit)) if hit else float("nan")


def full_embedding_baseline(Xall, comm_all, sample_rows, comm_sample, chunk=256):
    """For each sampled point, nearest neighbour over ALL ICSD embedding
    rows (true same-community recovery — calibrates deflation).

    Memory-safe: squared-distance identity ||a-b||^2 = |a|^2+|b|^2-2a.b,
    so the only large temporary is (chunk, N) — never (chunk, N, dim)."""
    hit = []
    sample_rows = np.asarray(sample_rows)
    Xs = Xall[sample_rows]
    n_all = Xall.shape[0]
    all_sq = (Xall ** 2).sum(1)                      # (N,)
    for s0 in range(0, len(sample_rows), chunk):
        blk = Xs[s0:s0 + chunk]                       # (b, dim)
        b_sq = (blk ** 2).sum(1)                      # (b,)
        d2 = b_sq[:, None] + all_sq[None, :] - 2.0 * (blk @ Xall.T)  # (b, N)
        for bi in range(blk.shape[0]):
            srow = s0 + bi
            d2[bi, sample_rows[srow]] = np.inf        # exclude self
            j = int(np.argmin(d2[bi]))
            ci, cj = comm_sample[srow], comm_all[j]
            if ci >= 0 and cj >= 0:
                hit.append(1.0 if ci == cj else 0.0)
    return float(np.mean(hit)) if hit else float("nan")


def main():
    a = parse_args()
    t0 = time.time()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    X = np.load(a.features_pca, mmap_mode="r")
    row_icsd = load_rowmap(a.assignments)
    if len(row_icsd) != X.shape[0]:
        raise ValueError("PCA rows and assignment rows differ; row alignment is required")
    n_rows = X.shape[0]
    comm_map = load_communities(a.community_labels)
    print(f"features_pca {X.shape} | assignments {len(row_icsd)} | "
          f"community map {len(comm_map)} ids", flush=True)

    # row -> community (production Louvain); keep rows with valid id+comm+finite emb
    rows_by_comm = {}
    for i in range(n_rows):
        iid = row_icsd[i]
        c = comm_map.get(iid, -2)
        if iid > 0 and c >= 0:
            rows_by_comm.setdefault(c, []).append(i)

    comms = [c for c, rs in rows_by_comm.items() if len(rs) >= a.min_community_size]
    rng.shuffle(comms)
    sample_rows = []
    for c in comms:
        rs = rows_by_comm[c][:]
        rng.shuffle(rs)
        sample_rows.extend(rs[:a.per_community_cap])
        if len(sample_rows) >= a.n_sample:
            break
    sample_rows = sample_rows[:a.n_sample]
    samp_icsd = [row_icsd[i] for i in sample_rows]
    print(f"sampled {len(sample_rows)} rows from {len(set(comm_map[i] for i in samp_icsd))} "
          f"communities (cap {a.per_community_cap})", flush=True)

    procs = a.procs or os.cpu_count() or 1
    pwd = a.zip_password
    provenance = {
        "feature_version": FEATURE_VERSION,
        "neighbor_settings": dict(NEIGHBOR_SETTINGS),
        "representation": "graphlet-64x20-periodic-images-short-long-arms",
        "community_labels_source": {"path": str(a.community_labels),
            "role": "external comparison labels; not regenerated or feature-version-validated here"},
        "embedding_source": {"path": str(a.features_pca),
            "role": "external comparison embedding; not regenerated or feature-version-validated here"},
        "assignments_source": str(a.assignments),
    }
    report_path = out / "featurization_report.json"
    report = {**provenance, "features": None}
    failures = []
    raw_by_icsd = {}
    with Pool(procs, initializer=_winit, initargs=(a.icsd_zip, pwd)) as pool:
        for iid, raw, failure in pool.imap_unordered(_featurize_one, samp_icsd, chunksize=8):
            if raw is not None:
                raw_by_icsd[iid] = raw
            else:
                failures.append(failure)
    print(f"featurized {len(raw_by_icsd)}/{len(samp_icsd)} "
          f"({procs} procs, {time.time()-t0:.0f}s)", flush=True)

    keep = [(r, iid) for r, iid in zip(sample_rows, samp_icsd) if iid in raw_by_icsd]
    report["raw_features"] = {**_coverage(len(samp_icsd), len(keep), failures),
                              "failures": list(failures)}
    report_path.write_text(json.dumps(report, indent=2))
    try:
        edges = gf.global_bin_edges(raw_by_icsd)
    except Exception as exc:
        report["run_failure"] = {"stage": "bin_edges", "exception_type": type(exc).__name__}
        report_path.write_text(json.dumps(report, indent=2))
        raise SystemExit(f"Bin-edge generation failed ({type(exc).__name__}); see {report_path}")
    (out / "bin_edges.json").write_text(json.dumps(edges, indent=2))
    successful = []
    histograms = []
    for r, iid in keep:
        try:
            histogram = gf.histogram_tensor(raw_by_icsd[iid], edges)
        except Exception as exc:
            failures.append(_failure(iid, "histogram", exc))
            continue
        successful.append((r, iid))
        histograms.append(histogram)
    keep = successful
    krows = [r for r, _ in keep]
    kic = [iid for _, iid in keep]
    report["features"] = {**_coverage(len(samp_icsd), len(keep), failures),
                          "failures": failures, "successful_icsd_ids": kic}
    report_path.write_text(json.dumps(report, indent=2))
    if len(keep) < 50:
        raise SystemExit(f"Too few featurized entries ({len(keep)}); see {report_path}")
    comm_s = np.array([comm_map[i] for i in kic])
    H = np.stack(histograms)
    Emb = np.asarray(X[krows], dtype=np.float64)
    Emb = (Emb - Emb.mean(0)) / (Emb.std(0) + 1e-9)

    De = np.sqrt(((Emb[:, None, :] - Emb[None, :, :]) ** 2).sum(-1))
    Dg = emd_cdf_distance_matrix(H)
    iu = np.triu_indices(len(keep), 1)
    rho = spearman(De[iu], Dg[iu])
    ke, kg = knn_sets(De, a.knn), knn_sets(Dg, a.knn)
    overlap = float(np.mean([len(ke[i] & kg[i]) / a.knn for i in range(len(keep))]))

    res = {
        **provenance,
        "version": 3,
        **_coverage(len(samp_icsd), len(keep), failures),
        "n_structures": len(keep),
        "n_sample_requested": a.n_sample,
        "n_communities_sampled": int(len(set(comm_s.tolist()))),
        "per_community_cap": a.per_community_cap,
        "knn_k": a.knn, "seed": a.seed,
        "embedding_dim": int(Emb.shape[1]),
        "graphlet_features": int(H.shape[1]), "graphlet_bins": int(H.shape[2]),
        "mantel_spearman_pairwise": rho,
        "mean_knn_overlap_frac": overlap,
        "nn_same_community_embedding_insample": nn_same_comm(De, comm_s),
        "nn_same_community_graphlet_insample": nn_same_comm(Dg, comm_s),
        "procs": procs,
        "failure_report": report_path.name,
        "bin_edges": "bin_edges.json",
    }
    # Write PRIMARY metrics first — a later baseline failure must not lose
    # the science.
    res["runtime_sec_primary"] = round(time.time() - t0, 1)
    rp = out / "graphlet_compare_result.json"
    rp.write_text(json.dumps(res, indent=2))
    print("=== PRIMARY RESULT (written) ===")
    for k, v in res.items():
        print(f"  {k}: {v}")

    if a.full_baseline:
        try:
            Xall = np.asarray(X[:n_rows], dtype=np.float64)
            Xall = (Xall - Xall.mean(0)) / (Xall.std(0) + 1e-9)
            comm_all = np.array([comm_map.get(row_icsd[i], -1) for i in range(n_rows)])
            res["nn_same_community_embedding_fullICSD_baseline"] = (
                full_embedding_baseline(Xall, comm_all, np.array(krows), comm_s))
            res["runtime_sec_total"] = round(time.time() - t0, 1)
            rp.write_text(json.dumps(res, indent=2))
            print("full-ICSD baseline:",
                  res["nn_same_community_embedding_fullICSD_baseline"])
        except Exception as e:
            res["baseline_failure"] = {"stage": "embedding_baseline",
                                       "exception_type": type(e).__name__}
            rp.write_text(json.dumps(res, indent=2))
            print(f"WARNING: full-ICSD baseline failed ({type(e).__name__}); "
                  f"primary metrics already saved.", flush=True)


if __name__ == "__main__":
    main()
