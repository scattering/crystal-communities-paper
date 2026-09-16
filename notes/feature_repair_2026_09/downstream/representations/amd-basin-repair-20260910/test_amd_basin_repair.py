import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import json
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("repair", HERE / "run_amd_basin_repair.py")
repair = importlib.util.module_from_spec(spec); spec.loader.exec_module(repair)


def test_singletons_and_zero_radius_are_excluded_before_reassignment():
    x = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.], [10., 0.], [10., 1.], [10., 2.], [10., 3.]])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    basis = repair.filtered_basis(x, labels, np.ones(8, bool), 4, 1e-12)
    assert basis["communities"].tolist() == [1]
    near, dist, inside = repair.assign(np.array([[0., 0.], [10., 1.5]]), basis, 2)
    assert basis["communities"][near].tolist() == [1, 1]
    assert inside.tolist() == [False, True]
    assert any(r["reason"] == "zero_radius" for r in basis["rejected"])
    assert basis["n_candidate_exact_identical_member_communities"] == 1


def test_component_and_community_filters_capture_small_groups():
    g = nx.Graph(); g.add_nodes_from(range(9)); g.add_edges_from((i, i + 1) for i in range(6)); g.add_edge(7, 8)
    keep, sizes = repair.component_mask(g, 9, 4)
    assert sizes == [7, 2]
    assert keep.tolist() == [True] * 7 + [False, False]
    x = np.arange(18, dtype=float).reshape(9, 2)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
    basis = repair.filtered_basis(x, labels, keep, 4, 1e-12)
    assert basis["communities"].tolist() == [0]
    assert {r["community"] for r in basis["rejected"]} == {1, 2}


def test_holdout_mask_ties_and_finite_zero_distance_behavior():
    years = np.array([1989, 1990, 1991, -1])
    training = (years >= 0) & (years <= 1990)
    assert training.tolist() == [True, True, False, False]
    basis = {"communities": np.array([3, 8]), "centroids": np.array([[0., 0.], [2., 0.]]),
             "radii": np.array([1., 1.]), "counts": np.array([4, 4])}
    near, dist, inside = repair.assign(np.array([[1., 0.], [0., 0.], [4., 0.]]), basis, 2)
    assert near.tolist() == [0, 0, 1]  # exact tie chooses sorted lower community
    assert np.isfinite(dist).all() and dist[1] == 0
    assert inside.tolist() == [True, True, False]


def test_canonical_nested_five_way_support(tmp_path):
    populations = {}
    for name in ("ICSD", "GNoME", "MatterGen", "MP", "JARVIS", "Alexandria"):
        populations[name] = {"n": 1, "ids": [name + "-1"]}
    path = tmp_path / "support.json"
    path.write_text(json.dumps({"populations": populations}))
    support = repair.load_support(path)
    assert set(support) == {"icsd", "gnome", "mattergen", "mp", "jarvis", "alexandria"}
    assert support["mattergen"] == {"MatterGen-1"}


def test_historical_kernel_rejects_nonpositive_median():
    indices = np.array([[1], [0]])
    with pytest.raises(ValueError, match="nonpositive median"):
        repair.mutual_graph(indices, np.zeros((2, 1)))
