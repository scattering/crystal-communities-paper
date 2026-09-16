import numpy as np
import pytest
from pymatgen.core import Lattice, PeriodicSite, Structure
from scipy.spatial.distance import cdist

import bond_metrics as bm


def test_extract_preserves_reciprocal_and_periodic_self_image_distances(monkeypatch):
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0, 0]])
    cl = PeriodicSite("Cl", [0.5, 0, 0], lattice)
    na = PeriodicSite("Na", [0, 0, 0], lattice)
    na_image = PeriodicSite("Na", [1, 0, 0], lattice)
    neighbors = [
        [{"site_index": 1, "site": cl, "weight": 0.5},
         {"site_index": 0, "site": na_image, "weight": 0.25}],
        [{"site_index": 0, "site": na, "weight": 0.5}],
    ]
    monkeypatch.setattr(bm.gf, "_structure_neighbor_info", lambda _structure: neighbors)

    cache = bm.extract(structure)

    np.testing.assert_array_equal(cache["i"], [0, 0, 1])
    np.testing.assert_array_equal(cache["j"], [1, 0, 0])
    np.testing.assert_allclose(cache["d"], [2.0, 4.0, 2.0])
    np.testing.assert_allclose(cache["cn"], [0.75, 0.5])


def test_extract_rejects_missing_properties_and_disorder():
    lattice = Lattice.cubic(4.0)
    with pytest.raises(ValueError, match="electronegativity"):
        bm.extract(Structure(lattice, ["Ar"], [[0, 0, 0]]))
    with pytest.raises(ValueError, match="ordered elemental site"):
        bm.extract(Structure(lattice, [{"Na": 0.5, "K": 0.5}], [[0, 0, 0]]))


def _cache():
    return {
        "z": np.array([3.0, 8.0, 17.0]),
        "X": np.array([1.0, 2.0, 4.0]),
        "rcov": np.array([1.2, 0.7, 1.0]),
        "i": np.array([0, 1, 1, 2]),
        "j": np.array([1, 0, 2, 1]),
        "w": np.array([1.0, 1.0, 0.25, 0.25]),
        "d": np.array([1.8, 1.8, 2.7, 2.7]),
        "cn": np.array([1.0, 1.25, 0.25]),
    }


def test_shuffle_preserves_geometry_and_composition_but_changes_joint():
    cache = _cache()
    actual = bm.vectors(cache)
    shuffled = bm.vectors(cache, permutation=[2, 0, 1])

    np.testing.assert_array_equal(actual["geometry"][0], shuffled["geometry"][0])
    np.testing.assert_array_equal(actual["geometry"][1], shuffled["geometry"][1])
    assert not np.array_equal(actual["joint"][0], shuffled["joint"][0])
    np.testing.assert_array_equal(np.sort(cache["z"]), np.sort(cache["z"][[2, 0, 1]]))
    np.testing.assert_allclose(actual["joint"][1].sum(), 1.0)


def test_weighted_energy_matches_direct_unweighted_v_statistic_and_scaling():
    A = np.array([[0.0], [1.0]])
    B = np.array([[2.0], [3.0]])
    weights = np.ones(2)
    expected = (
        2 * cdist(A, B).mean()
        - cdist(A, A).mean()
        - cdist(B, B).mean()
    )
    assert bm.weighted_energy(A, weights, B, weights, [1.0]) == pytest.approx(expected)
    assert expected == pytest.approx(3.0)
    assert bm.weighted_energy(A, weights, B, weights, [2.0]) == pytest.approx(expected / 2.0)
    assert bm.weighted_energy(A, weights, A, weights, [1.0]) == pytest.approx(0.0)


def test_marginal_w1_is_columnwise_weighted_and_scaled():
    A = np.array([[0.0, 0.0], [2.0, 4.0]])
    B = np.array([[1.0, 2.0], [3.0, 6.0]])
    result = bm.marginal_w1(A, [1, 1], B, [1, 1], [1.0, 2.0])
    np.testing.assert_allclose(result, [1.0, 1.0])


def test_vector_validation_rejects_nonfinite_data_without_imputation():
    cache = _cache()
    cache["X"] = cache["X"].copy()
    cache["X"][1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        bm.vectors(cache)
