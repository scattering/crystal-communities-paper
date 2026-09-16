#!/usr/bin/env python3
"""Recompute representation recovery, partitions, and temporal replay.

Inputs are the completed, coverage-audited feature extensions. All labels are
joined by ICSD ID to the corrected production partition. No historical numeric
community ID is treated as a persistent family identity. Run on a compute node;
this script never reads CIFs or credentials.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.metadata
import json
from numbers import Real
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments/graphlet_compare")]
from crystal_neighbors import FEATURE_VERSION
from prepare_representation_features import digest, read_labels


# NetworkX 3.6.1 accepts any strictly positive local modularity gain. On the
# VoronoiNN graph, three nodes cycle indefinitely at gains near 1.83e-23. A
# fixed-graph audit found identical assignments at 1e-20 and 1e-21 (and
# convergence at both), while 1e-20 remains more than 500 times the diagnosed
# cycling gain. This floor therefore terminates numerical non-improvement while
# retaining the lowest empirically stable optimization path.
LOUVAIN_NODE_MOVE_GAIN_TOLERANCE = 1e-20
LOUVAIN_LEVEL_MODULARITY_THRESHOLD = 1e-7
LOUVAIN_MAX_LOCAL_MOVE_SWEEPS = 100
NEIGHBOR_SEARCH_RANDOM_STATE = 42
EXACT_AUDIT_MAX_WORKERS = 16


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_ids(path):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            ids = [int(row["icsd_id"]) for row in csv.DictReader(handle)]
    else:
        ids = json.loads(path.read_text())
    if not all(isinstance(i, int) and i > 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("IDs must be unique positive integers")
    return np.asarray(ids, dtype=np.int64)


def actual_neighbors(X, k, metric, approximate, procs):
    """Retain actual non-self rows, preserving the estimator's distance/tie order."""
    k = min(k, len(X) - 1)
    if k < 1:
        raise ValueError("At least two candidates are required")
    if approximate:
        from pynndescent import NNDescent
        estimator = NNDescent(
            X,
            metric=metric,
            n_neighbors=k + 1,
            random_state=NEIGHBOR_SEARCH_RANDOM_STATE,
            low_memory=True,
            n_jobs=procs,
        )
        raw_indices, raw_distances = estimator.neighbor_graph
    else:
        from sklearn import config_context
        from sklearn.neighbors import NearestNeighbors
        estimator = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=procs).fit(X)
        raw_indices = np.empty((len(X), k + 1), dtype=np.int64)
        raw_distances = np.empty((len(X), k + 1), dtype=np.float64)
        with config_context(working_memory=256):
            for start in range(0, len(X), 1024):
                stop = min(start + 1024, len(X))
                raw_distances[start:stop], raw_indices[start:stop] = estimator.kneighbors(X[start:stop])
    indices = np.empty((len(X), k), dtype=np.int64)
    distances = np.empty((len(X), k), dtype=np.float64)
    for row in range(len(X)):
        keep = [p for p, j in enumerate(raw_indices[row]) if 0 <= j < len(X) and j != row][:k]
        if len(keep) != k:
            raise ValueError("Neighbor search returned too few valid non-self candidates")
        indices[row] = raw_indices[row, keep]
        distances[row] = raw_distances[row, keep]
        if len(set(indices[row])) != k or not np.isfinite(distances[row]).all():
            raise ValueError("Invalid retained neighbor rows")
    return indices, distances


def recovery_stats(labels, selected, null_reps=200):
    """Entrywise 1-NN agreement; chance and confusion null use this exact cohort."""
    n = len(labels)
    sizes = Counter(int(c) for c in labels)
    hits = Counter(int(labels[i]) for i in range(n) if labels[i] == labels[selected[i]])
    confusion = Counter((int(labels[i]), int(labels[selected[i]])) for i in range(n)
                        if labels[i] != labels[selected[i]])
    by_source = defaultdict(Counter)
    for (source, destination), count in confusion.items():
        by_source[source][destination] = count
    communities = sorted(sizes)
    weights = np.asarray([sizes[c] for c in communities], dtype=float)
    misses = np.asarray([sizes[c] - hits[c] for c in communities], dtype=int)
    total_misses = int(misses.sum())
    rng = np.random.default_rng(42)
    null_global = np.zeros(10)
    per = []
    for index, c in enumerate(communities):
        m = int(misses[index])
        captured = sum(count for _, count in by_source[c].most_common(3))
        record = {"community": c, "size": sizes[c], "hits": hits[c], "misses": m,
                  "recovery": hits[c] / sizes[c], "random_other_entry_chance": (sizes[c]-1)/(n-1),
                  "top_three_miss_targets": by_source[c].most_common(3),
                  "top_three_captured": captured, "top_three_share": captured/m if m else None}
        if m:
            p = weights.copy()
            p[index] = 0
            p /= p.sum()
            reps = null_reps if m >= 100 else 10
            draws = rng.multinomial(m, p, size=reps)
            top_k = min(3, draws.shape[1])
            values = np.partition(draws, draws.shape[1]-top_k, axis=1)[:, -top_k:].sum(axis=1)
            null_global += values[:10]
            if m >= 100:
                record["top_three_sizeweighted_null"] = {
                    "repetitions": reps, "mean_share": float(values.mean()/m),
                    "p97_5_share": float(np.percentile(values, 97.5)/m),
                    "definition": "condition on this community's observed miss count; draw other communities by cohort size and reselect top three in every draw"}
        per.append(record)
    strata = []
    for lo, hi in [(1,4), (5,19), (20,49), (50,99), (100,999), (1000,10**9)]:
        subset = [r for r in per if lo <= r["size"] <= hi]
        count = sum(r["size"] for r in subset)
        if count:
            strata.append({"size_range": [lo, None if hi == 10**9 else hi],
                "n_communities": len(subset), "n_entries": count,
                "entry_weighted_recovery": sum(r["hits"] for r in subset)/count,
                "community_mean_recovery": float(np.mean([r["recovery"] for r in subset])),
                "random_other_entry_chance": sum(r["size"]*r["random_other_entry_chance"] for r in subset)/count})
    return {"n_entries": n, "n_communities": len(sizes), "hits": n-total_misses,
        "misses": total_misses, "same_community_1nn_agreement": 1-total_misses/n,
        "random_other_entry_chance": sum(s*(s-1) for s in sizes.values())/(n*(n-1)),
        "size_strata": strata, "per_community": per,
        "top_three_concentration": {"definition": "sum of each source community's three largest miss-destination counts / all misses",
            "observed": sum(r["top_three_captured"] for r in per)/total_misses if total_misses else None,
            "sizeweighted_reselected_null_repetitions": 10,
            "sizeweighted_reselected_null_mean": float(null_global.mean()/total_misses) if total_misses else None,
            "sizeweighted_reselected_null_sd": float(null_global.std()/total_misses) if total_misses else None}}, confusion


def graphlet_selected_channels(X, selected):
    import graphlet_features as gf
    expected = {10: "f2_distance", 31: "f3_cos_angle", 32: "f3_d_ij", 33: "f3_d_jk"}
    if any(gf.REGISTRY.all[index] != name for index, name in expected.items()):
        raise ValueError("Pure geometry channel registry changed")
    result = np.empty((len(X), 4))
    for start in range(0, len(X), 1024):
        stop = min(start+1024, len(X))
        d = np.abs(np.asarray(X[start:stop], dtype=np.float64)-X[selected[start:stop]]).reshape(-1,64,20).sum(axis=2)
        first = d[:,:10].sum(axis=1)
        geometry = d[:,list(expected)].sum(axis=1)
        other = d[:,[i for i in range(10,64) if i not in expected]].sum(axis=1)
        result[start:stop] = np.column_stack((d.sum(axis=1), first, geometry, other))
    return result


def exact_graphlet_audit(X, ids, labels, selected, count, procs):
    """Exact L1 over the entire candidate set for seeded queries; bounded memory."""
    from concurrent.futures import ThreadPoolExecutor
    from scipy.spatial.distance import cdist
    queries = sorted(
        np.random.default_rng(NEIGHBOR_SEARCH_RANDOM_STATE)
        .choice(len(X), min(count, len(X)), replace=False)
        .tolist()
    )
    def one(q):
        distances = np.empty(len(X), dtype=np.float64)
        for start in range(0, len(X), 4096):
            stop = min(start+4096,len(X))
            distances[start:stop] = cdist(X[q:q+1], X[start:stop], metric="cityblock")[0]
        distances[q] = np.inf
        minimum = distances.min()
        tied = np.flatnonzero(distances == minimum)
        chosen = int(tied[np.argmin(ids[tied])])
        ann_distance = float(distances[selected[q]])
        return {"query_id": int(ids[q]), "query_community": int(labels[q]),
            "exact_selected_neighbor_id": int(ids[chosen]), "exact_selected_neighbor_community": int(labels[chosen]),
            "exact_minimum_l1": float(minimum), "exact_nearest_tied_ids": sorted(ids[tied].tolist()),
            "any_exact_tie_same_community": bool(np.any(labels[tied] == labels[q])),
            "all_exact_ties_same_community": bool(np.all(labels[tied] == labels[q])),
            "ann_selected_neighbor_id": int(ids[selected[q]]), "ann_selected_distance_l1": ann_distance,
            "ann_selected_is_exact_nearest": bool(ann_distance == minimum),
            "ann_distance_excess_l1": ann_distance-float(minimum)}
    audit_workers = min(procs, EXACT_AUDIT_MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=audit_workers) as executor:
        rows = list(executor.map(one, queries))
    return {"n_queries": len(rows), "query_seed": NEIGHBOR_SEARCH_RANDOM_STATE,
        "parallel_workers": audit_workers, "candidate_pool_size": len(X),
        "distance": "exact float64 cityblock on stored float32 CDF values",
        "tie_policy": "all exact-equal ties retained; smallest ICSD ID selected",
        "ann_exact_nearest_fraction": float(np.mean([r["ann_selected_is_exact_nearest"] for r in rows])),
        "exact_selected_label_agreement": float(np.mean([r["query_community"] == r["exact_selected_neighbor_community"] for r in rows])),
        "queries": rows}


def do_recovery(X, ids, labels, out, name, metric, approximate, args, provenance):
    out.mkdir(parents=True, exist_ok=True)
    print(f"Recovery {name}: {len(X)} candidates, metric={metric}, approximate={approximate}", flush=True)
    # Historical graphlet recovery requested 15 including self. Exact metrics
    # only need two including self; membership is checked, never assumed first.
    indices, distances = actual_neighbors(X, 14 if approximate else 1, metric, approximate, args.procs)
    np.savez(out / "neighbors.npz", indices=indices, distances=distances)
    dump(out / "ids.json", ids.tolist())
    selected = indices[:,0]
    summary, confusion = recovery_stats(labels, selected)
    channels = graphlet_selected_channels(X, selected) if name == "graphlet_l1" else None
    nearest = []
    for row, iid in enumerate(ids):
        record = {"query_id": int(iid), "query_community": int(labels[row]),
            "neighbor_id": int(ids[selected[row]]), "neighbor_community": int(labels[selected[row]]),
            "distance": float(distances[row,0]), "same_community": bool(labels[row] == labels[selected[row]])}
        if channels is not None:
            record.update(zip(("distance_l1_recomputed", "first_ten_chemical_channels_l1", "pure_geometry_four_channels_l1", "pair_triplet_chemical_50_channels_l1"), channels[row].tolist()))
        nearest.append(record)
    write_csv(out / "nearest_neighbors.csv", nearest)
    write_csv(out / "confusion_counts.csv", [{"from_community": c, "to_community": d, "count": n}
        for (c,d),n in sorted(confusion.items(), key=lambda x:(-x[1], x[0]))], ["from_community","to_community","count"])
    summary.update({**provenance, "name": name, "metric": metric, "approximate": approximate,
        "definition": "label of the selected single non-self nearest neighbor equals the query's corrected production label",
        "tie_policy": "estimator distance order; equal-distance choice is estimator-dependent; exact graphlet sample retains all ties",
        "estimator": "pynndescent.NNDescent" if approximate else "sklearn.neighbors.NearestNeighbors exact search",
        "ann_requested_neighbors_including_self": 15 if approximate else None,
        "emd_note": "L1 sums 64 channels; divide by 64 for the mean per-channel discrete CDF distance" if approximate else None})
    if approximate:
        dump(out / "recovery.json", summary)
        audit = exact_graphlet_audit(X, ids, labels, selected, args.exact_audit_queries, args.procs)
        dump(out / "exact_query_audit.json", audit)
        summary["exact_query_audit"] = {k:v for k,v in audit.items() if k != "queries"}
    dump(out / "recovery.json", summary)
    return summary


def mutual_graph(indices, distances, protocol):
    import networkx as nx
    if protocol == "historical_amd":
        sigma = float(np.median(distances))
        if sigma <= 0:
            raise ValueError("Historical AMD kernel has zero median distance; explicit method decision required")
        denominator = 2
    else:
        positive = distances[distances > 0]
        sigma = max(float(np.median(positive)) if len(positive) else 1.0, 1e-8)
        denominator = 1
    neighbor_sets = [set(row) for row in indices]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(indices)))
    for i in range(len(indices)):
        for j, distance in zip(indices[i], distances[i]):
            j = int(j)
            if i not in neighbor_sets[j]:
                continue
            weight = float(np.exp(-((distance/sigma)**2)/denominator))
            if not graph.has_edge(i,j) or weight > graph[i][j]["weight"]:
                graph.add_edge(i,j,weight=weight)
    return graph, sigma


def _stable_neighbor_weights(neighbors, node_to_community):
    """Sum one node's edge weights by neighboring community.

    This is the small helper used by NetworkX 3.6.1's undirected Louvain
    implementation, kept locally so the numerical safeguard does not depend on
    an unversioned private NetworkX API.
    """
    weights = defaultdict(float)
    for neighbor, edge_weight in neighbors.items():
        weights[node_to_community[neighbor]] += edge_weight
    return weights


def _stable_aggregate_graph(graph, partition):
    """Aggregate a weighted graph exactly as NetworkX 3.6.1 Louvain does."""
    aggregate = graph.__class__()
    node_to_community = {}
    for community, members in enumerate(partition):
        original_nodes = set()
        for node in members:
            node_to_community[node] = community
            original_nodes.update(graph.nodes[node].get("nodes", {node}))
        aggregate.add_node(community, nodes=original_nodes)
    for left, right, data in graph.edges(data=True):
        community_left = node_to_community[left]
        community_right = node_to_community[right]
        existing = aggregate.get_edge_data(
            community_left, community_right, {"weight": 0},
        )["weight"]
        aggregate.add_edge(
            community_left, community_right,
            weight=data["weight"] + existing,
        )
    return aggregate


def _stable_louvain_one_level(
    graph, total_weight, partition, resolution, random_state,
    *, move_gain_tolerance=LOUVAIN_NODE_MOVE_GAIN_TOLERANCE,
    max_sweeps=LOUVAIN_MAX_LOCAL_MOVE_SWEEPS,
):
    """NetworkX's undirected Louvain local-move step with a numeric guard."""
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Stable Louvain helper requires a simple undirected graph")
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Stable Louvain helper requires positive finite total edge weight")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Stable Louvain helper requires positive finite resolution")
    if not np.isfinite(move_gain_tolerance) or move_gain_tolerance < 0:
        raise ValueError("Stable Louvain helper requires a finite non-negative move tolerance")
    if not isinstance(max_sweeps, int) or isinstance(max_sweeps, bool) or max_sweeps < 1:
        raise ValueError("Stable Louvain helper requires at least one local-move sweep")
    node_to_community = {node: index for index, node in enumerate(graph.nodes())}
    inner_partition = [{node} for node in graph.nodes()]
    degrees = dict(graph.degree(weight="weight"))
    community_degree = list(degrees.values())
    neighbors = {
        node: {
            neighbor: data["weight"]
            for neighbor, data in graph[node].items()
            if neighbor != node
        }
        for node in graph
    }
    shuffled_nodes = list(graph.nodes)
    random_state.shuffle(shuffled_nodes)
    improvement = False
    accepted_moves = 0
    minimum_accepted_gain = None
    maximum_accepted_gain = None

    for sweep in range(1, max_sweeps + 1):
        moves_this_sweep = 0
        for node in shuffled_nodes:
            original_community = node_to_community[node]
            best_community = original_community
            best_gain = move_gain_tolerance
            weights_to_community = _stable_neighbor_weights(
                neighbors[node], node_to_community,
            )
            degree = degrees[node]
            community_degree[original_community] -= degree
            remove_cost = (
                -weights_to_community[original_community] / total_weight
                + resolution
                * community_degree[original_community]
                * degree
                / (2 * total_weight**2)
            )
            for candidate_community, weight in weights_to_community.items():
                gain = (
                    remove_cost
                    + weight / total_weight
                    - resolution
                    * community_degree[candidate_community]
                    * degree
                    / (2 * total_weight**2)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_community = candidate_community
            community_degree[best_community] += degree
            if best_community != original_community:
                original_nodes = graph.nodes[node].get("nodes", {node})
                partition[original_community].difference_update(original_nodes)
                inner_partition[original_community].remove(node)
                partition[best_community].update(original_nodes)
                inner_partition[best_community].add(node)
                node_to_community[node] = best_community
                improvement = True
                moves_this_sweep += 1
                accepted_moves += 1
                minimum_accepted_gain = (
                    best_gain if minimum_accepted_gain is None
                    else min(minimum_accepted_gain, best_gain)
                )
                maximum_accepted_gain = (
                    best_gain if maximum_accepted_gain is None
                    else max(maximum_accepted_gain, best_gain)
                )
        if moves_this_sweep == 0:
            break
    else:
        raise RuntimeError(
            "Louvain local moves did not converge within "
            f"{max_sweeps} sweeps at gain tolerance {move_gain_tolerance}"
        )

    return (
        list(filter(len, partition)),
        list(filter(len, inner_partition)),
        improvement,
        {
            "sweeps": sweep,
            "accepted_moves": accepted_moves,
            "minimum_accepted_gain": minimum_accepted_gain,
            "maximum_accepted_gain": maximum_accepted_gain,
        },
    )


def stable_louvain_communities(
    graph, *, weight="weight", resolution=1.0, seed=42,
    move_gain_tolerance=LOUVAIN_NODE_MOVE_GAIN_TOLERANCE,
    level_threshold=LOUVAIN_LEVEL_MODULARITY_THRESHOLD,
):
    """Return Louvain communities and convergence diagnostics.

    This retains NetworkX 3.6.1's undirected update and aggregation semantics
    while adding a per-node gain tolerance, an explicit sweep cap, validation,
    and convergence diagnostics. NetworkX's public ``threshold`` applies
    between aggregation levels and cannot stop a sub-resolution cycle within
    one level.
    """
    import networkx as nx
    from networkx.algorithms.community.quality import modularity
    from networkx.utils import create_py_random_state

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Stable Louvain helper requires a simple undirected graph")
    if weight is not None and not isinstance(weight, str):
        raise ValueError("Louvain weight must be an edge-attribute name or None")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Louvain resolution must be positive and finite")
    if not np.isfinite(move_gain_tolerance) or move_gain_tolerance < 0:
        raise ValueError("Louvain move tolerance must be finite and non-negative")
    if not np.isfinite(level_threshold) or level_threshold < 0:
        raise ValueError("Louvain level threshold must be finite and non-negative")
    partition = [{node} for node in graph.nodes()]
    if nx.is_empty(graph):
        return partition, {
            "levels": 0,
            "level_diagnostics": [],
            "termination": "empty_graph",
        }
    random_state = create_py_random_state(seed)
    working_graph = graph.__class__()
    working_graph.add_nodes_from(graph)
    working_graph.add_weighted_edges_from(
        graph.edges(data=weight, default=1),
    )
    edge_weights = [
        data["weight"] for _, _, data in working_graph.edges(data=True)
    ]
    if any(
        not isinstance(edge_weight, Real)
        or not np.isfinite(edge_weight)
        or edge_weight < 0
        for edge_weight in edge_weights
    ):
        raise ValueError("Louvain edge weights must be finite and non-negative")
    total_weight = working_graph.size(weight="weight")
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Louvain requires positive finite total edge weight")
    previous_modularity = modularity(
        graph, partition, resolution=resolution, weight=weight,
    )
    partition, inner_partition, improvement, diagnostics = _stable_louvain_one_level(
        working_graph, total_weight, partition, resolution, random_state,
        move_gain_tolerance=move_gain_tolerance,
    )
    levels = [diagnostics]
    final_partition = [members.copy() for members in partition]
    termination = "no_local_improvement"
    while improvement:
        final_partition = [members.copy() for members in partition]
        current_modularity = modularity(
            working_graph, inner_partition, resolution=resolution, weight="weight",
        )
        gain = current_modularity - previous_modularity
        levels[-1]["level_modularity"] = float(current_modularity)
        levels[-1]["level_modularity_gain"] = float(gain)
        if gain <= level_threshold:
            termination = "level_modularity_threshold"
            break
        previous_modularity = current_modularity
        working_graph = _stable_aggregate_graph(working_graph, inner_partition)
        partition, inner_partition, improvement, diagnostics = _stable_louvain_one_level(
            working_graph, total_weight, partition, resolution, random_state,
            move_gain_tolerance=move_gain_tolerance,
        )
        levels.append(diagnostics)
        if not improvement:
            termination = "no_local_improvement"
    return final_partition, {
        "levels": len(levels),
        "level_diagnostics": levels,
        "termination": termination,
    }


def partition_metrics(reference, alternate):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix
    def one(a,b):
        table = contingency_matrix(a,b,sparse=True)
        pair = lambda v: int(np.sum(np.asarray(v, dtype=np.int64)*(np.asarray(v, dtype=np.int64)-1)//2))
        true_positive = pair(table.data)
        reference_pairs = pair(np.asarray(table.sum(axis=1)).ravel())
        alternate_pairs = pair(np.asarray(table.sum(axis=0)).ravel())
        recall = true_positive/reference_pairs if reference_pairs else None
        precision = true_positive/alternate_pairs if alternate_pairs else None
        return {"n_entries": len(a), "ARI": float(adjusted_rand_score(a,b)),
            "NMI_arithmetic": float(normalized_mutual_info_score(a,b)),
            "same_community_pair_recall": recall, "same_community_pair_precision": precision,
            "same_community_pair_f1": 2*true_positive/(reference_pairs+alternate_pairs) if reference_pairs+alternate_pairs else None,
            "reference_same_pairs": reference_pairs, "alternate_same_pairs": alternate_pairs,
            "intersection_same_pairs": true_positive}
    keep = alternate >= 0
    return {"all_entries_noise_as_one_label": one(reference,alternate),
        "alternate_nonnoise_only": one(reference[keep], alternate[keep]) if keep.any() else None,
        "n_alternate_noise": int((~keep).sum())}


def temporal_replay(X, ids, years, labels, graph, out, provenance):
    import pandas as pd
    from icsd_graph_time_evolution import compute_temporal_metrics
    from compute_fig1_exclusive_events import classify
    rows = [{"icsd_id": int(i), "year": y, "community": int(c)} for i,y,c in zip(ids,years,labels)]
    summary, events, growth, top = compute_temporal_metrics(X, rows, labels, graph, 25)
    summary.update(provenance)
    dump(out / "graph_time_summary.json", summary)
    event_frame = pd.DataFrame(events)
    event_frame.to_csv(out / "node_temporal_events.csv", index=False)
    write_csv(out / "community_growth_by_decade.csv", growth, ["community","decade","cumulative_size"])
    dump(out / "top_communities.json", [{"community": int(c), "size": int(s)} for c,s in top])
    event_frame["exclusive_category"] = classify(event_frame)
    exclusive = []
    for decade, group in event_frame.groupby("decade", sort=True):
        counts = Counter(group.exclusive_category)
        if sum(counts.values()) != len(group) or set(counts)-{"outlier","birth","same","cross","bridge"}:
            raise ValueError("Invalid exclusive temporal partition")
        if counts["birth"] != summary["by_decade"][decade]["n_cluster_birth_point"] or counts["outlier"] != summary["by_decade"][decade]["n_outlier"]:
            raise ValueError("Exclusive temporal counts do not match production replay")
        entry = {"decade": decade, "n_total": len(group)}
        for category in ("outlier","birth","same","cross","bridge"):
            entry[f"n_{category}"] = counts[category]
            entry[f"share_{category}"] = counts[category]/len(group)
        entry["share_any_attachment"] = sum(counts[c] for c in ("same","cross","bridge"))/len(group)
        entry["n_same_isolated"] = int(((group.exclusive_category == "same") & (group.n_active_same_community_neighbors == 0)).sum())
        exclusive.append(entry)
    dump(out / "exclusive_by_decade.json", {**provenance, "denominator": "all entries in the given decade, including representation-partition noise", "rows": exclusive})


def do_partition(X, ids, years, reference_labels, out, name, protocol, approximate, args, provenance):
    import networkx as nx
    out.mkdir(parents=True, exist_ok=True)
    print(f"Partition {name}: {len(X)} rows, protocol={protocol}", flush=True)
    indices, distances = actual_neighbors(X, 16, "euclidean", approximate, args.procs)
    np.savez(out / "neighbors.npz", indices=indices, distances=distances)
    dump(out / "ids.json", ids.tolist())
    graph, sigma = mutual_graph(indices,distances,protocol)
    if graph.number_of_edges():
        communities, louvain_diagnostics = stable_louvain_communities(
            graph, weight="weight", resolution=1.0, seed=42,
        )
    else:
        communities = [{i} for i in range(len(X))]
        louvain_diagnostics = {
            "levels": 0,
            "level_diagnostics": [],
            "termination": "empty_graph",
        }
    min_size = 1 if protocol == "historical_amd" else 10
    alternate = np.full(len(X), -1, dtype=np.int64)
    next_label = 0
    for members in communities:
        if len(members) >= min_size:
            alternate[list(members)] = next_label
            next_label += 1
    write_csv(out / "community_assignments.csv", [{"icsd_id": int(i), "year": y, "community": int(c)} for i,y,c in zip(ids,years,alternate)])
    edges = np.asarray([(a,b,data["weight"]) for a,b,data in graph.edges(data=True)], dtype=float).reshape(-1,3)
    np.save(out / "graph_edges.npy", edges)
    result = {**provenance, "name": name, "protocol": protocol, "knn_k": min(16,len(X)-1),
        "metric": "euclidean", "approximate_neighbors": approximate, "mutual_knn": True,
        "sigma": sigma, "sigma_population": "all retained neighbor distances" if protocol == "historical_amd" else "positive retained neighbor distances",
        "weight": "exp(-distance^2/(2*sigma^2))" if protocol == "historical_amd" else "exp(-distance^2/sigma^2)",
        "resolution": 1.0, "louvain_seed": 42,
        "louvain_implementation": "networkx-3.6.1-compatible-undirected-with-node-move-tolerance",
        "louvain_node_move_gain_tolerance": LOUVAIN_NODE_MOVE_GAIN_TOLERANCE,
        "louvain_level_modularity_threshold": LOUVAIN_LEVEL_MODULARITY_THRESHOLD,
        "louvain_max_local_move_sweeps": LOUVAIN_MAX_LOCAL_MOVE_SWEEPS,
        "louvain_diagnostics": louvain_diagnostics,
        "minimum_community_size": min_size,
        "component_filter": None, "n_entries": len(X), "n_edges": len(edges),
        "n_communities": next_label, "n_noise": int(np.sum(alternate<0)),
        "comparison_to_corrected_production_labels": partition_metrics(reference_labels,alternate)}
    dump(out / "partition.json", result)
    temporal_replay(X,ids,years,alternate,graph,out,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("recovery","partition"), required=True)
    parser.add_argument("--kind", choices=("graphlet","amd"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--production-labels", type=Path, required=True)
    parser.add_argument("--production-metadata", type=Path, required=True)
    parser.add_argument("--production-features", type=Path, required=True, help="corrected PCA-32 array")
    parser.add_argument("--production-ids", type=Path, required=True, help="ordered sample_assignments.csv or ordered JSON IDs for PCA rows")
    parser.add_argument("--restrict-ids", type=Path, action="append", default=[])
    parser.add_argument("--dated-only", action="store_true", help="retain 1900 <= year <= 2025; distinct temporal cohort")
    parser.add_argument("--with-subset-baselines", action="store_true", help="recluster raw and standardized PCA on identical cohort and graph protocol")
    parser.add_argument("--procs", type=int, default=32)
    parser.add_argument("--exact-audit-queries", type=int, default=512)
    parser.add_argument("--expected-graphlet-neighbor-method", choices=("crystalnn", "voronoinn"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.procs < 1 or args.exact_audit_queries < 1:
        parser.error("Positive process/audit counts are required")
    started = time.time()
    ids = read_ids(args.ids)
    n_feature_successes = len(ids)
    meta = json.loads(args.metadata.read_text())
    production_meta = json.loads(args.production_metadata.read_text())
    if production_meta.get("feature_version") != FEATURE_VERSION or not meta.get("complete"):
        raise ValueError("Completed feature extension and current production version are required")
    if args.kind == "graphlet":
        import graphlet_features as gf
        # The completed CrystalNN cache predates the method tag and stores it
        # as JSON null.  That one legacy case is identified by its exact
        # settings/representation below; every new cache must name its method.
        neighbor_method = meta.get("neighbor_method") or "crystalnn"
        if (args.expected_graphlet_neighbor_method is not None
                and neighbor_method != args.expected_graphlet_neighbor_method):
            raise ValueError("Graphlet input uses the wrong neighbor method")
        expected_graphlet_version = gf.GRAPHLET_FEATURE_VERSIONS.get(neighbor_method)
        saved_graphlet_version = meta.get("graphlet_feature_version")
        legacy_crystalnn_metadata = (
            neighbor_method == "crystalnn" and saved_graphlet_version is None
        )
        graphlet_version_matches = (
            saved_graphlet_version == expected_graphlet_version
            or legacy_crystalnn_metadata
        )
        if (neighbor_method not in gf.NEIGHBOR_METHODS
                or meta.get("feature_version") != FEATURE_VERSION
                or not graphlet_version_matches
                or meta.get("neighbor_settings") != gf.neighbor_settings(neighbor_method)
                or meta.get("representation") != gf.GRAPHLET_REPRESENTATION):
            raise ValueError("Graphlet feature version/settings mismatch")
    for field, path in (("ids_sha256", args.ids), ("features_sha256", args.features), ("production_labels_sha256", args.production_labels)):
        if meta.get(field) != digest(path):
            raise ValueError(f"Feature extension metadata does not match {field}")
    labels = read_labels(args.production_labels)
    X = np.load(args.features, mmap_mode="r", allow_pickle=False)
    if X.shape != (len(ids), 1280 if args.kind == "graphlet" else 100) or not np.isfinite(X).all():
        raise ValueError("Invalid feature dimensions/values")
    keep_ids = set(ids.tolist())
    if any(i not in labels or labels[i]["community"] < 0 for i in keep_ids):
        raise ValueError("Feature extension contains IDs outside the corrected non-noise population")
    for restriction in args.restrict_ids:
        keep_ids &= set(read_ids(restriction).tolist())
    if args.dated_only:
        keep_ids = {i for i in keep_ids if labels[i]["year"] is not None and 1900 <= labels[i]["year"] <= 2025}
    selected_rows = [r for r,i in enumerate(ids) if i in keep_ids]
    ids, X = ids[selected_rows], np.asarray(X[selected_rows])
    if len(ids) < 2:
        raise ValueError("Too few available candidates")
    reference = np.asarray([labels[i]["community"] for i in ids], dtype=np.int64)
    years = [labels[i]["year"] for i in ids]
    production_ids = read_ids(args.production_ids)
    production_rows = {i:r for r,i in enumerate(production_ids)}
    pca = np.load(args.production_features, mmap_mode="r", allow_pickle=False)
    if pca.shape != (len(production_ids),32) or not np.isfinite(pca).all() or any(i not in production_rows for i in ids):
        raise ValueError("PCA-32 array/ordered ID alignment check failed")
    pca = np.asarray(pca[[production_rows[i] for i in ids]])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump(args.out_dir / "cohort.ids.json", ids.tolist())
    provenance = {"production_feature_version": FEATURE_VERSION, "feature_preparation": str(args.metadata),
        "feature_preparation_sha256": digest(args.metadata), "production_labels_sha256": digest(args.production_labels),
        "production_pca_sha256": digest(args.production_features), "production_pca_ids_sha256": digest(args.production_ids),
        "source_sha256": digest(Path(__file__)), "n_candidates": len(ids), "n_undated": sum(y is None for y in years),
        "n_feature_successes_before_cohort_restriction": n_feature_successes,
        "neighbor_search_n_jobs": args.procs,
        "neighbor_search_random_state": NEIGHBOR_SEARCH_RANDOM_STATE,
        "exact_audit_max_workers": min(args.procs, EXACT_AUDIT_MAX_WORKERS),
        "dated_only": args.dated_only, "target_year_range": [1900,2025] if args.dated_only else None,
        "restrict_ids": [{"path": str(p), "sha256": digest(p)} for p in args.restrict_ids],
        "reference_labels_role": "corrected full-production Louvain labels joined by ID; no relabeling or inference from historical numeric IDs",
        "cohort_sha256": digest(args.out_dir / "cohort.ids.json")}
    if args.kind == "graphlet":
        provenance.update(
            graphlet_neighbor_method=neighbor_method,
            graphlet_neighbor_settings=meta["neighbor_settings"],
            graphlet_feature_version=expected_graphlet_version,
            source_graphlet_feature_version=saved_graphlet_version,
            legacy_crystalnn_metadata_compatibility=legacy_crystalnn_metadata,
        )
    provenance["packages"] = {}
    for package in ("numpy","scipy","scikit-learn","pynndescent","networkx"):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    dump(args.out_dir / "run_metadata.json", provenance)
    from sklearn.preprocessing import StandardScaler
    scaled = None
    if args.kind == "amd":
        scaler = StandardScaler().fit(X.astype(np.float64))
        scaled = scaler.transform(X.astype(np.float64))
        dump(args.out_dir / "amd_scaler.json", {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "n_samples": len(X)})
    summaries = {}
    if args.operation == "recovery":
        metrics = [("graphlet_l1", X, "manhattan", True)] if args.kind == "graphlet" else [
            ("amd_raw_linf", X, "chebyshev", False), ("amd_raw_euclidean", X, "euclidean", False),
            ("amd_standardized_euclidean", scaled, "euclidean", False)]
        metrics.append(("production_pca32_ceiling", pca, "euclidean", False))
        for name, matrix, metric, approximate in metrics:
            result = do_recovery(matrix,ids,reference,args.out_dir/name,name,metric,approximate,args,provenance)
            summaries[name] = {k:result[k] for k in ("same_community_1nn_agreement","random_other_entry_chance","n_entries","n_communities")}
        provenance["ceiling_note"] = "Production labels were constructed in PCA-32; agreement there is a same-embedding ceiling, not an independent representation benchmark"
    else:
        protocol = "historical_amd" if args.kind == "amd" else "historical_graphlet_temporal"
        variants = [(args.kind, scaled if args.kind == "amd" else X)]
        if args.with_subset_baselines:
            pca_scaler = StandardScaler().fit(pca.astype(np.float64))
            dump(args.out_dir / "pca_subset_scaler.json", {"mean": pca_scaler.mean_.tolist(), "scale": pca_scaler.scale_.tolist(), "n_samples": len(pca)})
            variants += [("production_pca_raw_same_protocol",pca), ("production_pca_standardized_same_protocol",pca_scaler.transform(pca.astype(np.float64)))]
        for name,matrix in variants:
            result = do_partition(matrix,ids,years,reference,args.out_dir/name,name,protocol,args.kind == "graphlet",args,provenance)
            summaries[name] = {k:result[k] for k in ("n_entries","n_communities","n_noise","comparison_to_corrected_production_labels")}
    dump(args.out_dir / "summary.json", {**provenance, "results": summaries, "runtime_seconds": time.time()-started, "complete": True})


if __name__ == "__main__":
    main()
