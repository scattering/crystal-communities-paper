#!/usr/bin/env python3
"""Why the two parent partitions are not bit-identical to the saved maps.

Check 1: run the CrystalWeave protocol (build_weighted_graph + run_louvain) on the
saved production features_pca.npy itself (bit-identical input) and compare with the
saved production partition. Exact agreement isolates the driver's 1.6e-9 PCA
reconstruction round-off + Louvain sensitivity as the cause of the parent deviation.

Check 2: on the driver's parent_graphlet neighbor graph (which equals the saved
neighbors.npz), run plain networkx louvain_communities(seed=42) - the code path that
produced the saved CrystalNN graphlet partition - and compare with (a) the saved
partition and (b) the driver's stable-Louvain partition.

Check 3: for ablation B (randomized-SVD PCA on 1,233 components), quantify the
fit_transform-versus-transform coordinate difference and the ICSD in-basin rate if
ICSD rows were projected with transform() like the externals.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

STAGE = Path(os.environ["FA_STAGE"]).resolve()
RESULTS = Path(os.environ["FA_OUT"])
WORK = Path(os.environ.get("FA_WORK", os.environ.get("WORK", "/path/to/tacc/work")))
for p in (STAGE / "scripts", STAGE / "experiments/graphlet_compare", STAGE / "notes/feature_repair_2026_09/downstream"):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_factor_ablations as fa  # noqa: E402


def main():
    import networkx as nx
    from icsd_graph_community_postprocess import build_weighted_graph, run_louvain
    import analyze_representations as ar
    out = {"networkx": nx.__version__}
    production = fa.load_production_population()
    # Check 1
    t0 = time.time()
    saved_pca = np.asarray(production["saved_pca"], dtype=np.float64)
    graph, base = build_weighted_graph(saved_pca, 16, True, 8)
    labels = run_louvain(graph, len(saved_pca), 1.0, 4, base)
    out["check1_saved_pca_crystalweave_protocol"] = {
        "n_nodes_after_component_filter": graph.number_of_nodes(), "n_edges": graph.number_of_edges(),
        "vs_saved_production_partition": fa.compare_partitions(production["saved_labels"], labels),
        "exact_label_array_equal": bool(np.array_equal(production["saved_labels"], labels)), "seconds": time.time() - t0}
    import gzip, csv
    with gzip.open(RESULTS / "maps/parent_crystalweave/community_assignments.csv.gz", "rt") as h:
        driver_labels = np.asarray([int(r["community"]) for r in csv.DictReader(h)])
    out["check1_saved_pca_crystalweave_protocol"]["vs_driver_parent_crystalweave"] = fa.compare_partitions(driver_labels, labels)
    print(json.dumps(out["check1_saved_pca_crystalweave_protocol"], indent=1), flush=True)
    # Check 2
    t0 = time.time()
    graphlet = fa.load_graphlet_population(production)
    with np.load(RESULTS / "maps/parent_graphlet/neighbors.npz", allow_pickle=False) as saved:
        indices, distances = saved["indices"], saved["distances"]
    with np.load(WORK / "feature_repair_runs/downstream_v2_20260905/representations/graphlet-dated-replay/graphlet/neighbors.npz", allow_pickle=False) as saved:
        same_order = bool(np.array_equal(indices, saved["indices"])) and bool(np.array_equal(distances, saved["distances"]))
    graph, sigma = ar.mutual_graph(indices, distances, "historical_graphlet_temporal")
    communities = nx.community.louvain_communities(graph, weight="weight", resolution=1.0, seed=42)
    plain = np.full(len(indices), -1, dtype=np.int64)
    nxt = 0
    for members in communities:
        if len(members) >= 10:
            plain[list(members)] = nxt
            nxt += 1
    with gzip.open(RESULTS / "maps/parent_graphlet/community_assignments.csv.gz", "rt") as h:
        stable = np.asarray([int(r["community"]) for r in csv.DictReader(h)])
    out["check2_plain_networkx_louvain_on_driver_graphlet_graph"] = {
        "driver_neighbors_identical_to_saved_neighbors_npz_including_order": same_order, "sigma": float(sigma), "n_edges": graph.number_of_edges(),
        "plain_vs_saved_graphlet_partition": fa.compare_partitions(graphlet["saved_labels"], plain),
        "plain_exact_label_array_equal_to_saved": bool(np.array_equal(graphlet["saved_labels"], plain)),
        "plain_vs_driver_stable_louvain": fa.compare_partitions(stable, plain), "seconds": time.time() - t0}
    print(json.dumps(out["check2_plain_networkx_louvain_on_driver_graphlet_graph"], indent=1), flush=True)
    # Check 3
    t0 = time.time()
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from external_representation_sensitivity import classify, rate
    _, variance = fa.column_variance(graphlet["cdf"])
    columns = [c for c in range(1280) if variance[c] > 0]
    standardized = StandardScaler().fit_transform(np.asarray(graphlet["cdf"][:, columns], dtype=np.float64))
    pca = PCA(n_components=32, random_state=42)
    fit_coords = pca.fit_transform(standardized)
    transform_coords = pca.transform(standardized)
    with np.load(RESULTS / "maps/B_cdf_zscore_pca32_crystalweave_protocol/basis.npz", allow_pickle=False) as saved:
        basis = {k: saved[k] for k in ("communities", "centroids", "p95", "counts")}
        saved_components = saved["transform_pca_components"]
    delta = np.linalg.norm(fit_coords - transform_coords, axis=1)
    _, _, inside_fit = classify(fit_coords, basis)
    _, _, inside_transform = classify(transform_coords, basis)
    with gzip.open(RESULTS / "maps/B_cdf_zscore_pca32_crystalweave_protocol/icsd_full.csv.gz", "rt") as h:
        driver_inside = np.asarray([r["in_basin"] == "True" for r in csv.DictReader(h)])
    out["check3_B_randomized_pca_fit_vs_transform"] = {
        "pca_solver": pca._fit_svd_solver, "components_reproduced": bool(np.allclose(saved_components, pca.components_)),
        "row_displacement_max": float(delta.max()), "row_displacement_median": float(np.median(delta)), "row_displacement_p95": float(np.quantile(delta, .95)),
        "median_p95_radius": float(np.median(basis["p95"])), "median_nearest_distance_fit": None,
        "icsd_rate_fit_transform_coordinates": rate(inside_fit)["in_basin_fraction"], "icsd_rate_transform_coordinates": rate(inside_transform)["in_basin_fraction"],
        "driver_icsd_rate_reproduced": bool(np.array_equal(driver_inside, inside_fit)), "seconds": time.time() - t0}
    print(json.dumps(out["check3_B_randomized_pca_fit_vs_transform"], indent=1), flush=True)
    fa.dump(RESULTS / "parent_reproduction_checks.json", out)


if __name__ == "__main__":
    main()
