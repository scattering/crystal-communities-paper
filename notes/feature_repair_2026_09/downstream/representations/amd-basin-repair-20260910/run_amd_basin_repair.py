#!/usr/bin/env python3
"""Repair AMD-100 basin calibration and fit genuinely cutoff-trained AMD maps.

The fixed arm preserves the saved full-map coordinates, graph, and Louvain
labels, then applies the production component/community filters and excludes
zero-radius communities before reassignment.  The cutoff arm independently
fits every map object on dated ICSD rows available by T.  Query rows are never
dropped; filtered/noise outcomes are retained in the output accounting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn import __version__ as sklearn_version, config_context
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--icsd-features", type=Path, required=True)
    p.add_argument("--icsd-ids", type=Path, required=True)
    p.add_argument("--icsd-records", type=Path, required=True,
                   help="CSV containing unique icsd_id and year columns")
    p.add_argument("--full-scaler", type=Path, required=True)
    p.add_argument("--full-assignments", type=Path, required=True)
    p.add_argument("--full-graph-edges", type=Path, required=True,
                   help="Saved graph_edges.npy; endpoints are AMD row indices")
    p.add_argument("--external", nargs=3, action="append", default=[],
                   metavar=("SOURCE", "FEATURES", "IDS"))
    p.add_argument("--five-way-support", type=Path, required=True,
                   help="JSON object mapping icsd and each external source to record IDs")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arms", choices=("all", "fixed", "cutoff"), default="all")
    p.add_argument("--cutoffs", type=int, nargs="+", default=[1990, 2000, 2010])
    p.add_argument("--max-year", type=int, default=2015)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--min-component-size", type=int, default=8)
    p.add_argument("--min-community-size", type=int, default=4)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tiny-radius", type=float, default=1e-12,
                   help="Reporting threshold only; eligibility is strictly radius > 0")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def load_ids(path):
    values = json.loads(Path(path).read_text())
    if isinstance(values, dict):
        values = values.get("ids", values.get("record_ids"))
    values = [str(x) for x in values]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs: {path}")
    return values


def load_icsd_ids(path):
    values = load_ids(path)
    if not all(v.isdigit() and int(v) > 0 for v in values):
        raise ValueError("ICSD IDs must be positive integers")
    return [int(v) for v in values]


def load_years(path, ids):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    table = {}
    for r in rows:
        key = int(r["icsd_id"])
        if key in table:
            raise ValueError(f"Duplicate ICSD record {key}")
        raw = str(r.get("year", "")).strip()
        table[key] = int(float(raw)) if raw else None
    if any(i not in table for i in ids):
        raise ValueError("ICSD records do not cover feature IDs")
    return np.asarray([table[i] if table[i] is not None else -1 for i in ids], dtype=np.int64)


def validate_features(path, n):
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != (n, 100) or not np.isfinite(x).all():
        raise ValueError(f"Invalid AMD-100 array {path}: {x.shape}")
    if np.any(x < 0) or np.any(np.diff(x, axis=1) < -1e-6):
        raise ValueError(f"Invalid AMD monotonicity/nonnegativity: {path}")
    return x


def load_stable_louvain():
    source = Path(__file__).resolve().parents[2] / "analyze_representations.py"
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("amd_repair_existing_analysis", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stable_louvain_communities, source


def actual_neighbors(x, k, n_jobs):
    if len(x) <= k:
        raise ValueError(f"Need more than k={k} training rows")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs).fit(x)
    indices = np.empty((len(x), k), dtype=np.int64)
    distances = np.empty((len(x), k), dtype=np.float64)
    with config_context(working_memory=512):
        for start in range(0, len(x), 2048):
            d, j = nn.kneighbors(x[start:start + 2048])
            for local, row in enumerate(range(start, min(start + 2048, len(x)))):
                keep = np.flatnonzero(j[local] != row)[:k]
                if len(keep) != k:
                    raise ValueError(f"Too few non-self neighbors for row {row}")
                indices[row], distances[row] = j[local, keep], d[local, keep]
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite neighbor distance")
    return indices, distances


def mutual_graph(indices, distances):
    # Historical AMD definition: median of all retained distances (zeros included)
    # and exp(-d^2/(2 sigma^2)). A nonpositive median requires an explicit
    # method decision rather than an unreported fallback.
    sigma = float(np.median(distances))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Historical AMD kernel has nonpositive median distance")
    neighbor_sets = [set(map(int, row)) for row in indices]
    graph = nx.Graph(); graph.add_nodes_from(range(len(indices)))
    for i, (js, ds) in enumerate(zip(indices, distances)):
        for j, d in zip(js, ds):
            j = int(j)
            if i not in neighbor_sets[j]:
                continue
            weight = math.exp(-(float(d) ** 2) / (2 * sigma ** 2))
            if not graph.has_edge(i, j) or weight > graph[i][j]["weight"]:
                graph.add_edge(i, j, weight=weight)
    return graph, sigma


def component_mask(graph, n, minimum):
    keep = np.zeros(n, dtype=bool)
    sizes = []
    for members in nx.connected_components(graph):
        sizes.append(len(members))
        if len(members) >= minimum:
            keep[list(members)] = True
    return keep, sizes


def filtered_basis(x, labels, component_keep, min_community, tiny):
    eligible, rejected = [], []
    candidate_zero = candidate_tiny = candidate_exact_identical = 0
    for label in sorted(set(map(int, labels)) - {-1}):
        members = np.flatnonzero((labels == label) & component_keep)
        radius = None
        center = None
        if len(members):
            values = np.asarray(x[members], dtype=np.float64)
            center = values.mean(axis=0)
            exact_identical = bool(np.all(values == values[0]))
            candidate_exact_identical += exact_identical
            radius = 0.0 if exact_identical else float(np.quantile(np.linalg.norm(values - center, axis=1), .95))
            candidate_zero += radius == 0
            candidate_tiny += 0 < radius <= tiny
        reason = None
        if len(members) < min_community:
            reason = "community_below_minimum"
        elif np.any(~component_keep[np.flatnonzero(labels == label)]):
            reason = "community_intersects_filtered_component"
        if reason:
            rejected.append({"community": label, "n_members": int(len(members)), "radius_p95": radius, "reason": reason})
            continue
        if radius <= 0:
            rejected.append({"community": label, "n_members": int(len(members)), "radius_p95": radius, "reason": "zero_radius"})
            continue
        eligible.append((label, members, center, radius))
    if not eligible:
        raise ValueError("No positive-radius eligible communities")
    return {
        "communities": np.asarray([e[0] for e in eligible], dtype=np.int64),
        "centroids": np.stack([e[2] for e in eligible]),
        "radii": np.asarray([e[3] for e in eligible]),
        "counts": np.asarray([len(e[1]) for e in eligible], dtype=np.int64),
        "n_tiny_positive_radii": int(sum(0 < e[3] <= tiny for e in eligible)),
        "n_candidate_zero_radii_before_size_filter": int(candidate_zero),
        "n_candidate_tiny_positive_radii_before_size_filter": int(candidate_tiny),
        "n_candidate_exact_identical_member_communities": int(candidate_exact_identical),
        "rejected": rejected,
    }


def assign(x, basis, batch_size):
    centers, radii = basis["centroids"], basis["radii"]
    nearest = np.empty(len(x), dtype=np.int64); distance = np.empty(len(x))
    c2 = np.einsum("ij,ij->i", centers, centers)
    for start in range(0, len(x), batch_size):
        block = np.asarray(x[start:start + batch_size], dtype=np.float64)
        d2 = np.maximum(np.einsum("ij,ij->i", block, block)[:, None] + c2 - 2 * block @ centers.T, 0)
        # np.argmin gives deterministic lowest sorted community on an exact tie.
        chosen = np.argmin(d2, axis=1)
        nearest[start:start + len(block)] = chosen
        # Select cheaply by the dot-product identity, then calculate the saved
        # distance directly to avoid cancellation at zero/tiny-radius boundaries.
        distance[start:start + len(block)] = np.linalg.norm(block - centers[chosen], axis=1)
    return nearest, distance, distance <= radii[nearest]


def projection_rows(source, ids, nearest, distance, inside, basis, years=None, training=None):
    rows = []
    for n, key in enumerate(ids):
        j = nearest[n]
        row = {"source": source, "record_key": str(key),
               "assigned_community": int(basis["communities"][j]),
               "nearest_centroid_distance": float(distance[n]),
               "community_radius_p95": float(basis["radii"][j]),
               "in_basin": bool(inside[n])}
        if years is not None:
            row["year"] = "" if years[n] < 0 else int(years[n])
        if training is not None:
            row["is_training"] = bool(training[n])
        rows.append(row)
    return rows


FIELDS = ["source", "record_key", "year", "is_training", "assigned_community",
          "nearest_centroid_distance", "community_radius_p95", "in_basin"]


def counts(rows, support=None):
    selected = rows if support is None else [r for r in rows if r["record_key"] in support]
    return {"n": len(selected), "n_in_basin": sum(bool(r["in_basin"]) for r in selected),
            "in_basin_fraction": (sum(bool(r["in_basin"]) for r in selected) / len(selected)) if selected else None}


def load_support(path):
    raw = json.loads(Path(path).read_text())
    raw = raw.get("populations", raw)
    aliases = {"icsd": "icsd", "gnome": "gnome", "mattergen": "mattergen",
               "mp": "mp", "jarvis": "jarvis", "alexandria": "alexandria"}
    normalized = {}
    for supplied, value in raw.items():
        key = aliases.get(str(supplied).lower())
        if key is None:
            continue
        if isinstance(value, dict):
            value = value["ids"]
        normalized[key] = set(map(str, value))
    expected = set(("icsd",) + SOURCES)
    if set(normalized) != expected:
        raise ValueError(f"Five-way support requires {sorted(expected)}; got {sorted(normalized)}")
    return normalized


def save_arm(out, basis, projections, support, metadata):
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "basis.npz", communities=basis["communities"],
                       centroids=basis["centroids"], radii=basis["radii"], counts=basis["counts"])
    report = {**metadata, "basis": {"n_eligible_communities": len(basis["communities"]),
              "n_members": int(basis["counts"].sum()),
              "n_zero_radius_excluded": sum(r["reason"] == "zero_radius" for r in basis["rejected"]),
              "n_tiny_positive_radii": basis["n_tiny_positive_radii"],
              "n_candidate_zero_radii_before_size_filter": basis["n_candidate_zero_radii_before_size_filter"],
              "n_candidate_tiny_positive_radii_before_size_filter": basis["n_candidate_tiny_positive_radii_before_size_filter"],
              "n_candidate_exact_identical_member_communities": basis["n_candidate_exact_identical_member_communities"],
              "rejected_communities": basis["rejected"]}, "populations": {}}
    for source, rows in projections.items():
        path = out / f"{source}_projection.csv"
        write_csv(path, rows, FIELDS)
        report["populations"][source] = {"full": counts(rows), "five_way_common_support": counts(rows, support[source]),
                                         "projection_sha256": sha256(path)}
    if metadata.get("arm") == "independent_cutoff_refit":
        cutoff, max_year = metadata["cutoff"], metadata["max_year"]
        evaluation = [r for r in projections["icsd"]
                      if r["year"] != "" and cutoff < int(r["year"]) <= max_year]
        report["populations"]["icsd"]["later_through_max_year"] = counts(evaluation)
        report["populations"]["icsd"]["later_through_max_year_five_way_common_support"] = counts(evaluation, support["icsd"])
        report["populations"]["icsd"]["n_outside_later_evaluation_window"] = len(projections["icsd"]) - len(evaluation)
    dump(out / "summary.json", report)


def main():
    args = parse_args(); started = time.time()
    if args.output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {args.output_dir}")
    if set(s for s, _, _ in args.external) != set(SOURCES) or len(args.external) != len(SOURCES):
        raise ValueError(f"Provide exactly one --external for each of {SOURCES}")
    ids = load_icsd_ids(args.icsd_ids); raw = validate_features(args.icsd_features, len(ids))
    years = load_years(args.icsd_records, ids)
    support = load_support(args.five_way_support)
    external = {}
    for source, feature_path, id_path in args.external:
        ext_ids = load_ids(id_path); ext = validate_features(feature_path, len(ext_ids))
        external[source] = (ext_ids, ext, Path(feature_path), Path(id_path))
        if not support[source] <= set(ext_ids):
            raise ValueError(f"Five-way support contains unavailable {source} IDs")
    if not support["icsd"] <= set(map(str, ids)):
        raise ValueError("Five-way support contains unavailable ICSD IDs")
    args.output_dir.mkdir(parents=True)
    helper_path = Path(__file__).resolve().parents[2] / "analyze_representations.py"
    stable_louvain = None
    if args.arms in ("all", "cutoff"):
        stable_louvain, helper_path = load_stable_louvain()
    input_paths = [args.icsd_features, args.icsd_ids, args.icsd_records, args.full_scaler,
                   args.full_assignments, args.full_graph_edges, args.five_way_support]
    input_paths += [p for _, _, fp, ip in external.values() for p in (fp, ip)]
    def json_value(value):
        if isinstance(value, Path): return str(value)
        if isinstance(value, (list, tuple)): return [json_value(v) for v in value]
        return value
    configuration = {key: json_value(value) for key, value in vars(args).items()}
    provenance = {"configuration": configuration, "producer_sha256": sha256(__file__),
                  "inputs": {str(p): sha256(p) for p in input_paths},
                  "software": {"python": platform.python_version(), "numpy": np.__version__,
                               "sklearn": sklearn_version, "networkx": nx.__version__},
                  "stable_louvain_helper": {"path": str(helper_path), "sha256": sha256(helper_path)}}
    # A: preserve full-map coordinates, graph and partition; repair eligibility/calibration only.
    scaler = json.loads(args.full_scaler.read_text()); mean = np.asarray(scaler["mean"]); scale = np.asarray(scaler["scale"])
    if mean.shape != (100,) or scale.shape != (100,) or np.any(scale <= 0):
        raise ValueError("Invalid full-map scaler")
    x_full = (np.asarray(raw, dtype=np.float64) - mean) / scale
    with args.full_assignments.open(newline="") as f:
        label_map = {int(r["icsd_id"]): int(r["community"]) for r in csv.DictReader(f)}
    if set(label_map) != set(ids): raise ValueError("Full assignments and AMD IDs differ")
    labels = np.asarray([label_map[i] for i in ids])
    edges = np.load(args.full_graph_edges, allow_pickle=False)
    graph = nx.Graph(); graph.add_nodes_from(range(len(ids)))
    if edges.ndim != 2 or edges.shape[1] < 2: raise ValueError("Invalid graph edges")
    graph.add_weighted_edges_from((int(a), int(b), float(w) if edges.shape[1] > 2 else 1.0) for a, b, *rest in edges for w in [rest[0] if rest else 1.0])
    if any(n < 0 or n >= len(ids) for n in graph): raise ValueError("Graph endpoint out of range")
    keep, component_sizes = component_mask(graph, len(ids), args.min_component_size)
    if args.arms in ("all", "fixed"):
        basis = filtered_basis(x_full, labels, keep, args.min_community_size, args.tiny_radius)
        projections = {}
        q = [("icsd", list(map(str, ids)), x_full, years, None)]
        q += [(s, ext_ids, (np.asarray(ext) - mean) / scale, None, None) for s, (ext_ids, ext, _, _) in external.items()]
        for source, keys, x, ys, training in q:
            nearest, distance, inside = assign(x, basis, args.batch_size)
            projections[source] = projection_rows(source, keys, nearest, distance, inside, basis, ys, training)
        save_arm(args.output_dir / "full_map_fixed_partition_repair", basis, projections, support,
                 {"arm": "fixed_partition_filtering", "semantics": "saved full-map scaler, coordinates, graph and Louvain partition retained; component/community/positive-radius eligibility repaired before reassignment",
                  "n_graph_components": len(component_sizes), "component_sizes": sorted(component_sizes, reverse=True),
                  "n_nodes_component_filtered": int((~keep).sum()), "n_saved_partition_noise": int((labels < 0).sum()),
                  "runtime_seconds": time.time() - started})
    # B: every object is independently trained on pre-T dated ICSD only.
    cutoff_summaries = {}
    for cutoff in args.cutoffs if args.arms in ("all", "cutoff") else []:
        arm_started = time.time(); training = (years >= 0) & (years <= cutoff)
        print(f"AMD cutoff {cutoff}: fitting {int(training.sum())} training rows", flush=True)
        train_raw = np.asarray(raw[training], dtype=np.float64)
        fitted = StandardScaler().fit(train_raw); train_x = fitted.transform(train_raw)
        neighbor_i, neighbor_d = actual_neighbors(train_x, args.k, args.n_jobs)
        g, sigma = mutual_graph(neighbor_i, neighbor_d)
        component_keep, sizes = component_mask(g, len(train_x), args.min_component_size)
        retained = g.subgraph(np.flatnonzero(component_keep)).copy()
        communities, louvain_diag = stable_louvain(retained, weight="weight", resolution=args.resolution, seed=args.seed) if retained.number_of_edges() else ([{n} for n in retained], {"termination": "empty_graph", "levels": 0, "level_diagnostics": []})
        train_labels = np.full(len(train_x), -1, dtype=np.int64)
        for label, members in enumerate(sorted(communities, key=lambda m: min(m))): train_labels[list(members)] = label
        cutoff_basis = filtered_basis(train_x, train_labels, component_keep, args.min_community_size, args.tiny_radius)
        cutoff_out = args.output_dir / f"cutoff_{cutoff}"
        all_x = fitted.transform(np.asarray(raw, dtype=np.float64))
        cutoff_projections = {}
        nearest, distance, inside = assign(all_x, cutoff_basis, args.batch_size)
        cutoff_projections["icsd"] = projection_rows("icsd", list(map(str, ids)), nearest, distance, inside, cutoff_basis, years, training)
        for source, (ext_ids, ext, _, _) in external.items():
            x = fitted.transform(np.asarray(ext, dtype=np.float64)); nearest, distance, inside = assign(x, cutoff_basis, args.batch_size)
            cutoff_projections[source] = projection_rows(source, ext_ids, nearest, distance, inside, cutoff_basis)
        metadata = {"arm": "independent_cutoff_refit", "cutoff": cutoff, "max_year": args.max_year,
                    "evaluation_icsd_year_rule": f"{cutoff} < year <= {args.max_year}",
                    "n_training": int(training.sum()), "n_undated_icsd_queries": int((years < 0).sum()),
                    "n_later_through_max_year": int(((years > cutoff) & (years <= args.max_year)).sum()),
                    "training_partition": {"n_component_filtered": int((~component_keep).sum()),
                                           "n_not_in_positive_radius_eligible_communities": int(len(train_x) - cutoff_basis["counts"].sum())},
                    "graph": {"k": args.k, "exact_neighbors": True, "mutual": True, "metric": "euclidean", "sigma": sigma,
                              "sigma_definition": "median of all retained k-neighbor distances; nonpositive median is an error",
                              "kernel": "exp(-distance^2/(2*sigma^2))", "n_edges": g.number_of_edges(),
                              "component_sizes": sorted(sizes, reverse=True), "louvain": louvain_diag},
                    "scaler": {"mean": fitted.mean_.tolist(), "scale": fitted.scale_.tolist()}, "runtime_seconds": time.time() - arm_started}
        save_arm(cutoff_out, cutoff_basis, cutoff_projections, support, metadata)
        neighbor_path = cutoff_out / "training_neighbors.npz"
        np.savez_compressed(neighbor_path, indices=neighbor_i, distances=neighbor_d)
        eligible_labels = set(map(int, cutoff_basis["communities"]))
        assignment_path = cutoff_out / "training_assignments.csv"
        write_csv(assignment_path, [
            {"icsd_id": int(record_id), "year": int(year), "louvain_label": int(label),
             "eligible_positive_radius_community": bool(int(label) in eligible_labels)}
            for record_id, year, label in zip(np.asarray(ids)[training], years[training], train_labels)
        ], ["icsd_id", "year", "louvain_label", "eligible_positive_radius_community"])
        summary_path = cutoff_out / "summary.json"
        saved_summary = json.loads(summary_path.read_text())
        saved_summary["reconstruction_artifacts"] = {
            "training_neighbors_npz": {"path": str(neighbor_path), "sha256": sha256(neighbor_path),
                                        "arrays": ["indices", "distances"]},
            "training_assignments_csv": {"path": str(assignment_path), "sha256": sha256(assignment_path)},
        }
        dump(summary_path, saved_summary)
        cutoff_summaries[str(cutoff)] = {"summary": str(cutoff_out / "summary.json"), "sha256": sha256(cutoff_out / "summary.json")}
        print(f"AMD cutoff {cutoff}: completed in {time.time() - arm_started:.1f}s with {len(cutoff_basis['communities'])} eligible communities", flush=True)
    dump(args.output_dir / "run_metadata.json", {**provenance, "status": "complete", "cutoffs": cutoff_summaries,
         "runtime_seconds": time.time() - started, "query_policy": "all ICSD and external feature rows retained; no epsilon radius floor; zero-radius centers excluded before assignment"})


if __name__ == "__main__":
    main()
