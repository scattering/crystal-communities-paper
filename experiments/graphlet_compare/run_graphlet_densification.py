#!/usr/bin/env python3
"""Densification stability test under an independent structural representation.

The manuscript's central claim is *temporal*: experimental discovery densifies
the structural map (per-decade community-birth share collapses ~40% (1930s) ->
~3% (2010s)). Does that curve survive when the structural representation is
swapped from our matminer+message-passing+PCA-32 embedding to graphlet
histograms (Lesser et al. 2025)? Same Louvain partitioning, same temporal
replay -- only the feature matrix changes.

Pipeline:
  1. Featurize each ICSD entry (valid year + community) -> graphlet CDF
     (two-pass: bin edges from 5k subsample, then 1280-d CDF per structure).
  2. Mutual-kNN graph on the CDF (k=16, Euclidean via pynndescent; same as the
     production parameters in scripts/icsd_graph_community_postprocess.py).
  3. Networkx Louvain at resolution=1.0, seed=42 (identical to production).
  4. Per-decade temporal replay: for each community c, birth_year(c) =
     min(year of members); per-decade community-birth ratio = entries that
     open a new community / valid-year entries in that decade.
Output: graphlet_densification_result.json, by-decade comparable to
graph_time_summary.json's cluster_birth_point_ratio.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))

from icsd_densify_worker import read_structure_from_zip  # noqa: E402
from crystal_neighbors import FEATURE_VERSION  # noqa: E402
from icsd_graph_time_evolution import compute_temporal_metrics  # noqa: E402
import graphlet_features as gf  # noqa: E402

_MP = get_context("spawn")
_ZIP = None
_PWD = None
_EDGES = None
_NEIGHBOR_METHOD = None


def _winit(zip_path, pwd, edges=None, neighbor_method="crystalnn"):
    global _ZIP, _PWD, _EDGES, _NEIGHBOR_METHOD
    _ZIP = zipfile.ZipFile(zip_path)
    _PWD = pwd
    _EDGES = edges
    _NEIGHBOR_METHOD = neighbor_method


def _feat_raw(icsd_id):
    """Pass A sends numeric values only; histogram weights do not set bin edges."""
    stage = "read_structure"
    try:
        s = read_structure_from_zip(_ZIP, icsd_id, _PWD)
        stage = "raw_features"
        raw = gf.all_raw_features(s, neighbor_method=_NEIGHBOR_METHOD)
        stage = "bin_values"
        values = {name: np.fromiter((v for v, _ in raw[name]), dtype=np.float64,
                                    count=len(raw[name])) for name in gf.REGISTRY.all}
        return int(icsd_id), values, None
    except Exception as exc:
        return int(icsd_id), None, _failure(icsd_id, stage, exc)


def _feat_cdf(icsd_id):
    stage = "read_structure"
    try:
        s = read_structure_from_zip(_ZIP, icsd_id, _PWD)
        stage = "raw_features"
        raw = gf.all_raw_features(s, neighbor_method=_NEIGHBOR_METHOD)
        stage = "histogram"
        H = gf.histogram_tensor(raw, _EDGES)
        return int(icsd_id), np.cumsum(H, axis=1).reshape(-1).astype(np.float32), None
    except Exception as exc:
        return int(icsd_id), None, _failure(icsd_id, stage, exc)


def _failure(icsd_id, stage, exc):
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
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--procs", type=int, default=0)
    p.add_argument("--k", type=int, default=16, help="kNN k (same as production)")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Louvain resolution (same as production)")
    p.add_argument("--min-community-size", type=int, default=10)
    p.add_argument("--neighbor-method", choices=gf.NEIGHBOR_METHODS,
                   default="crystalnn")
    p.add_argument("--edge-ids-report", type=Path,
                   help="Reuse requested_icsd_ids from another featurization report")
    p.add_argument("--features-only", action="store_true",
                   help="Stop after writing the validated CDF artifacts")
    p.add_argument("--save-cdf", default="",
                   help="if set, write the CDF matrix here (.npy) for reuse")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.zip_password_stdin:
        args.zip_password = sys.stdin.readline().rstrip("\r\n")
        if not args.zip_password:
            p.error("--zip-password-stdin requires a non-empty input line")
    elif args.zip_password is None:
        args.zip_password = os.environ.get("ICSD_ZIP_PASSWORD") or None
    return args


def load_assignments(comm_path):
    """Return icsd_id -> (year, community). Drops noise (community < 0)."""
    out = {}
    with open(comm_path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                iid = int(float(r["icsd_id"]))
                y = int(float(r["year"]))
                c = int(float(r["community"]))
            except Exception:
                continue
            if 1900 <= y <= 2025 and c >= 0:
                out[iid] = (y, c)
    return out


def build_mutual_knn_graph(cdf, k):
    """Mirrors scripts/icsd_graph_community_postprocess.build_weighted_graph
    (mutual-kNN + Gaussian-weighted edges, sigma=median positive distance),
    but uses pynndescent for kNN so 150k x 1280 finishes in minutes, not hours."""
    import networkx as nx
    from pynndescent import NNDescent

    n = cdf.shape[0]
    idx_struct = NNDescent(cdf, metric="euclidean", n_neighbors=min(k + 1, n),
                           random_state=42, low_memory=True)
    indices, distances = idx_struct.neighbor_graph  # (n, k+1)

    neighbors = [
        [(int(j), float(d)) for j, d in zip(indices[i], distances[i])
         if 0 <= int(j) < n and int(j) != i][:k]
        for i in range(n)
    ]
    neighbor_sets = [{j for j, _ in row} for row in neighbors]
    pos = np.asarray([d for row in neighbors for _, d in row])
    sigma = float(np.median(pos[pos > 0])) if np.any(pos > 0) else 1.0
    sigma = max(sigma, 1e-8)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    edges_added = 0
    for i in range(n):
        for j, dist in neighbors[i]:
            if i not in neighbor_sets[j]:        # mutual-kNN intersection
                continue
            w = math.exp(-((float(dist) / sigma) ** 2))
            if graph.has_edge(i, j):
                if w > graph[i][j]["weight"]:
                    graph[i][j]["weight"] = w
            else:
                graph.add_edge(i, j, weight=w)
                edges_added += 1
    return graph, edges_added


def run_louvain(graph, resolution, min_size):
    """networkx Louvain at the production parameters; communities below
    min_size are folded into noise label -1."""
    import networkx as nx
    comms = nx.community.louvain_communities(
        graph, weight="weight", resolution=resolution, seed=42)
    labels = np.full(graph.number_of_nodes(), -1, dtype=int)
    next_id = 0
    for members in comms:
        if len(members) < min_size:
            continue
        for v in members:
            labels[v] = next_id
        next_id += 1
    return labels, next_id


def main():
    a = parse_args()
    t0 = time.time()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    yc = load_assignments(a.community_labels)            # iid -> (year, prodcomm)
    print(f"valid-year, non-noise prodcomm entries: {len(yc)}", flush=True)
    targets = sorted(yc.keys())
    if a.limit:
        targets = targets[:a.limit]
    print(f"targets after --limit={a.limit}: {len(targets)}", flush=True)

    procs = a.procs or os.cpu_count() or 1
    pwd = a.zip_password
    F, B = gf.NUM_FEATURES, gf.NUM_BINS
    provenance = {
        "feature_version": FEATURE_VERSION,
        "neighbor_method": a.neighbor_method,
        "neighbor_settings": gf.neighbor_settings(a.neighbor_method),
        "graphlet_feature_version": gf.GRAPHLET_FEATURE_VERSIONS[a.neighbor_method],
        "representation": gf.GRAPHLET_REPRESENTATION,
        "community_labels_source": {"path": str(a.community_labels),
            "role": "external population filter; not regenerated or feature-version-validated here"},
        "assignments_source": {"path": str(a.assignments),
            "role": "legacy CLI input; target IDs and years come from community_labels"},
        "target_year_range": [1900, 2025],
        "cdf_row_order": "ascending ICSD ID",
    }
    report = {**provenance, "bin_edge_sampling": None, "features": None}
    report_path = out / "featurization_report.json"

    # ---- Pass A: bin edges from seeded ~5k subsample
    if a.edge_ids_report:
        source_report = json.loads(a.edge_ids_report.read_text())
        edge_ids = source_report["bin_edge_sampling"]["requested_icsd_ids"]
        if (not edge_ids or len(edge_ids) != len(set(edge_ids))
                or not all(isinstance(i, int) and i > 0 for i in edge_ids)):
            raise ValueError("Invalid requested edge IDs in source report")
        report["bin_edge_sampling_source"] = {
            "path": str(a.edge_ids_report),
            "sha256": hashlib.sha256(a.edge_ids_report.read_bytes()).hexdigest(),
            "role": "same structures as the CrystalNN calibration; numeric edges are refitted from VoronoiNN observations",
        }
    else:
        rng = np.random.default_rng(42)
        edge_n = min(len(targets), 5000)
        edge_ids = list(rng.choice(targets, size=edge_n, replace=False))
    edge_n = len(edge_ids)
    raw_sub = {}
    edge_failures = []
    with _MP.Pool(
        procs, initializer=_winit,
        initargs=(a.icsd_zip, pwd, None, a.neighbor_method),
    ) as pool:
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
    del raw_sub; gc.collect()
    try:
        import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    # ---- Pass B: featurize all -> CDF; parent peak = CDF matrix only
    cdf = np.empty((len(targets), F * B), dtype=np.float32)
    ids = []
    failures = []
    with _MP.Pool(
        procs, initializer=_winit,
        initargs=(a.icsd_zip, pwd, edges, a.neighbor_method),
    ) as pool:
        for k, (iid, v, failure) in enumerate(
                pool.imap_unordered(_feat_cdf, targets, chunksize=16)):
            if v is not None:
                cdf[len(ids)] = v; ids.append(iid)
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
    (out / "graphlet_cdf.ids.json").write_text(json.dumps(ids))
    print(f"featurized {len(ids)}/{len(targets)} "
          f"({time.time()-t0:.0f}s); CDF {cdf.shape}", flush=True)
    if a.save_cdf:
        np.save(a.save_cdf, cdf)
        with open(a.save_cdf + ".ids.json", "w") as fh:
            json.dump(ids, fh)
        print(f"saved CDF to {a.save_cdf} ({time.time()-t0:.0f}s)", flush=True)
        (Path(str(a.save_cdf) + ".meta.json")).write_text(json.dumps({
            **provenance, **_coverage(len(targets), len(ids), failures),
            "bin_edges": str(out / "bin_edges.json"),
            "failure_report": str(report_path),
        }, indent=2))
    if len(ids) < 2:
        raise SystemExit(f"Too few featurized entries ({len(ids)}); see {report_path}")
    if a.features_only:
        report["complete"] = True
        report["features_only"] = True
        report["runtime_sec"] = round(time.time() - t0, 1)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"feature-only run complete ({time.time()-t0:.0f}s)", flush=True)
        return

    # ---- Mutual-kNN graph + Louvain (production parameters)
    graph, n_edges = build_mutual_knn_graph(cdf, a.k)
    print(f"mutual-kNN graph: {graph.number_of_nodes()} nodes, "
          f"{n_edges} edges ({time.time()-t0:.0f}s)", flush=True)
    labels, n_comms = run_louvain(graph, a.resolution, a.min_community_size)
    print(f"Louvain: {n_comms} communities (after min_size={a.min_community_size}); "
          f"{(labels<0).sum()} entries in noise ({time.time()-t0:.0f}s)", flush=True)

    # ---- Persist graphlet community assignments for provenance / reuse
    years = np.array([yc[iid][0] for iid in ids])
    with open(out / "graphlet_community_assignments.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["icsd_id", "year", "community"])
        for i, iid in enumerate(ids):
            w.writerow([iid, int(years[i]), int(labels[i])])

    # ---- Temporal replay: use the PRODUCTION compute_temporal_metrics
    # (imported from scripts/icsd_graph_time_evolution.py) on OUR graph,
    # labels, and CDF features. Output is byte-identical in structure to
    # the manuscript's graph_time_summary.json -- the only thing that
    # changes is the feature representation. That is the apples-to-apples
    # comparison.
    rows = [{"icsd_id": int(iid), "year": int(years[i]), "community": int(labels[i])}
            for i, iid in enumerate(ids)]
    summary, node_events, cumulative_rows, top_counts = compute_temporal_metrics(
        cdf, rows, labels, graph, top_n_communities=25)
    summary.update({**provenance, **_coverage(len(targets), len(ids), failures),
                    "failure_report": report_path.name})
    (out / "graph_time_summary.json").write_text(json.dumps(summary, indent=2))
    top_communities = [{"community": int(c), "size": int(s)} for c, s in top_counts]
    (out / "top_communities.json").write_text(json.dumps(top_communities, indent=2))
    with open(out / "community_growth_by_decade.csv", "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=["community", "decade", "cumulative_size"])
        wcsv.writeheader()
        for r in cumulative_rows:
            wcsv.writerow(r)

    # ---- Run-metadata sidecar (graphlet-specific run config, doesn't
    # contaminate the production-format graph_time_summary.json) ----
    meta = {
        **provenance,
        "variant": "graphlet_densification_stability",
        **_coverage(len(targets), len(ids), failures),
        "bin_edge_coverage": _coverage(edge_n, edge_n - len(edge_failures), edge_failures),
        "n_structures": len(ids),
        "n_communities": n_comms,
        "n_noise": int((labels < 0).sum()),
        "knn_k": a.k, "louvain_resolution": a.resolution,
        "min_community_size": a.min_community_size,
        "procs": procs,
        "runtime_sec": round(time.time() - t0, 1),
        "artifacts": {"cdf": str(a.save_cdf) if a.save_cdf else None,
            "cdf_ids": "graphlet_cdf.ids.json", "bin_edges": "bin_edges.json",
            "failures": report_path.name},
    }
    (out / "graphlet_densification_meta.json").write_text(json.dumps(meta, indent=2))

    print("=== RESULT (per-decade ratios from production replay) ===")
    for d, s in summary["by_decade"].items():
        n_t = int(s.get("n_total", 0))
        print(f"  {d}: n={n_t:>6d} "
              f"birth={s.get('cluster_birth_point_ratio', float('nan')):.3f} "
              f"existing={s.get('existing_cluster_ratio', float('nan')):.3f} "
              f"same={s.get('same_community_attachment_ratio', float('nan')):.3f} "
              f"cross={s.get('cross_community_attachment_ratio', float('nan')):.3f} "
              f"bridge={s.get('bridge_attachment_ratio', float('nan')):.3f} "
              f"outlier={s.get('outlier_ratio', float('nan')):.3f}")
    print(f"wrote {out/'graph_time_summary.json'} (production format)")
    print(f"wrote {out/'graphlet_densification_meta.json'} (graphlet meta)")


if __name__ == "__main__":
    main()
