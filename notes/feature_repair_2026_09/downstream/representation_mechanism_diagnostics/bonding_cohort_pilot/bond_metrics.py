"""Weighted local-contact metrics for the bonding cohort pilot.

The module deliberately treats CrystalNN weights as contact weights, not bond
orders.  Site labels may be permuted while the neighbor graph, image distances,
and weighted coordination numbers remain fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


REPO_ROOT = next(
    parent
    for parent in [Path(__file__).resolve().parent / "stage", *Path(__file__).resolve().parents]
    if (parent / "experiments" / "graphlet_compare" / "graphlet_features.py").is_file()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments.graphlet_compare import graphlet_features as gf  # noqa: E402

try:  # Current pymatgen location.
    from pymatgen.core.molecule_structure_comparator import CovalentRadius
except ImportError:  # Compatibility with the older environment used remotely.
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius


ArrayPair = Tuple[np.ndarray, np.ndarray]


def _as_finite_float(value, description: str) -> float:
    """Convert a scalar-like pymatgen value, rejecting absent data."""
    if hasattr(value, "magnitude"):
        value = value.magnitude
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported {description}: {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"Nonfinite {description}: {value!r}")
    return result


def extract(structure) -> dict[str, np.ndarray]:
    """Extract site properties and directed periodic CrystalNN contacts.

    Ordered, unit-occupancy elemental sites are required.  Missing Pauling
    electronegativities or Cordero covalent radii are errors: joint vectors are
    never formed by imputing or dropping individual contacts.
    """
    z_values, x_values, radius_values = [], [], []
    for site_index, site in enumerate(structure):
        if not site.is_ordered or len(site.species) != 1:
            raise ValueError(f"Site {site_index} is not an ordered elemental site")
        species, occupancy = next(iter(site.species.items()))
        if not np.isclose(float(occupancy), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"Site {site_index} does not have unit occupancy")
        element = getattr(species, "element", species)
        symbol = getattr(element, "symbol", None)
        if symbol is None or symbol not in CovalentRadius.radius:
            raise ValueError(f"No Cordero covalent radius for site {site_index}: {symbol!r}")
        z_values.append(_as_finite_float(getattr(element, "Z", None), f"Z at site {site_index}"))
        x_values.append(
            _as_finite_float(getattr(element, "X", None), f"electronegativity at site {site_index}")
        )
        radius_values.append(
            _as_finite_float(CovalentRadius.radius[symbol], f"covalent radius at site {site_index}")
        )

    neighbor_info = gf._structure_neighbor_info(structure)
    if len(neighbor_info) != len(structure):
        raise ValueError("CrystalNN returned the wrong number of site neighbor lists")

    centers, neighbors, weights, distances = [], [], [], []
    coordination = np.zeros(len(structure), dtype=float)
    for i, entries in enumerate(neighbor_info):
        center_cart = np.asarray(structure[i].coords, dtype=float)
        for entry in entries:
            j = int(entry["site_index"])
            if j < 0 or j >= len(structure):
                raise ValueError(f"CrystalNN neighbor index {j} is out of range")
            weight = _as_finite_float(entry.get("weight", 1.0), "CrystalNN weight")
            if weight < 0.0:
                raise ValueError("CrystalNN weights must be nonnegative")
            neighbor_site = entry["site"]
            distance = float(np.linalg.norm(np.asarray(neighbor_site.coords, dtype=float) - center_cart))
            if not np.isfinite(distance) or distance <= 0.0:
                raise ValueError(f"Invalid periodic contact distance {distance!r}")
            centers.append(i)
            neighbors.append(j)
            weights.append(weight)
            distances.append(distance)
            coordination[i] += weight

    if not weights or sum(weights) <= 0.0:
        raise ValueError("No positive CrystalNN contact mass")
    return {
        "z": np.asarray(z_values, dtype=float),
        "X": np.asarray(x_values, dtype=float),
        "rcov": np.asarray(radius_values, dtype=float),
        "i": np.asarray(centers, dtype=np.intp),
        "j": np.asarray(neighbors, dtype=np.intp),
        "w": np.asarray(weights, dtype=float),
        "d": np.asarray(distances, dtype=float),
        "cn": coordination,
    }


def _normalized_weights(weights) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("Weights must be a finite, nonnegative one-dimensional array")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Weights must have positive total mass")
    return weights / total


def vectors(cache: Mapping[str, np.ndarray], permutation=None) -> dict[str, ArrayPair]:
    """Build joint, chemistry-only, and geometry-only contact clouds.

    ``permutation`` maps each site to a reassigned complete label row; Z,
    electronegativity, and covalent radius therefore always move together.
    Coordination, graph indices, weights, and image distances stay fixed.
    """
    z = np.asarray(cache["z"], dtype=float)
    electronegativity = np.asarray(cache["X"], dtype=float)
    radius = np.asarray(cache["rcov"], dtype=float)
    coordination = np.asarray(cache["cn"], dtype=float)
    n_sites = z.size
    if any(a.shape != (n_sites,) for a in (electronegativity, radius, coordination)):
        raise ValueError("Site arrays must be one-dimensional and equally sized")
    if permutation is None:
        permutation = np.arange(n_sites, dtype=np.intp)
    else:
        permutation = np.asarray(permutation, dtype=np.intp)
        if permutation.shape != (n_sites,) or not np.array_equal(np.sort(permutation), np.arange(n_sites)):
            raise ValueError("permutation must contain every site index exactly once")
    # Index Z as part of the label row even though it is not itself a vector coordinate.
    z, electronegativity, radius = (
        a[permutation] for a in (z, electronegativity, radius)
    )
    del z

    i = np.asarray(cache["i"], dtype=np.intp)
    j = np.asarray(cache["j"], dtype=np.intp)
    weights = np.asarray(cache["w"], dtype=float)
    distance = np.asarray(cache["d"], dtype=float)
    if not (i.ndim == j.ndim == weights.ndim == distance.ndim == 1):
        raise ValueError("Contact arrays must be one-dimensional")
    if not (i.size == j.size == weights.size == distance.size):
        raise ValueError("Contact arrays must have equal length")
    if ((i < 0) | (i >= n_sites) | (j < 0) | (j >= n_sites)).any():
        raise ValueError("Contact index is out of range")
    if not all(np.isfinite(a).all() for a in (electronegativity, radius, coordination, distance)):
        raise ValueError("Vector inputs must be finite")
    if (radius <= 0.0).any() or (distance <= 0.0).any():
        raise ValueError("Radii and contact distances must be positive")
    normalized_weights = _normalized_weights(weights)

    mean_x = (electronegativity[i] + electronegativity[j]) / 2.0
    diff_x = np.abs(electronegativity[i] - electronegativity[j])
    mean_radius = (radius[i] + radius[j]) / 2.0
    diff_radius = np.abs(radius[i] - radius[j])
    mean_cn = (coordination[i] + coordination[j]) / 2.0
    diff_cn = np.abs(coordination[i] - coordination[j])

    joint = np.column_stack((distance / (radius[i] + radius[j]), mean_x, diff_x, mean_cn, diff_cn))
    chemistry = np.column_stack((mean_x, diff_x, mean_radius, diff_radius))
    geometry = np.column_stack((distance, mean_cn, diff_cn))
    return {
        "joint": (joint, normalized_weights.copy()),
        "chemistry": (chemistry, normalized_weights.copy()),
        "geometry": (geometry, normalized_weights.copy()),
    }


def _cloud(points, weights, scale) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or not np.isfinite(points).all():
        raise ValueError("Points must be a nonempty finite two-dimensional array")
    weights = _normalized_weights(weights)
    if weights.size != points.shape[0]:
        raise ValueError("One weight is required per point")
    scale = np.asarray(scale, dtype=float)
    if scale.ndim == 0:
        scale = np.full(points.shape[1], float(scale))
    if scale.shape != (points.shape[1],) or not np.isfinite(scale).all() or (scale <= 0.0).any():
        raise ValueError("Scale must contain one finite positive value per column")
    points = points / scale

    # Compression is exact and can greatly reduce repeated reciprocal contacts.
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    compressed_weights = np.zeros(unique.shape[0], dtype=float)
    np.add.at(compressed_weights, inverse, weights)
    return unique, compressed_weights


def weighted_energy(A, wa, B, wb, scale) -> float:
    """Return the weighted multivariate energy V-statistic (without a square root)."""
    A, wa = _cloud(A, wa, scale)
    B, wb = _cloud(B, wb, scale)
    if A.shape[1] != B.shape[1]:
        raise ValueError("Point clouds must have the same dimensionality")
    cross = float(wa @ cdist(A, B, metric="euclidean") @ wb)
    within_a = float(wa @ cdist(A, A, metric="euclidean") @ wa)
    within_b = float(wb @ cdist(B, B, metric="euclidean") @ wb)
    value = 2.0 * cross - within_a - within_b
    if value < -1e-9 * (1.0 + 2.0 * abs(cross) + abs(within_a) + abs(within_b)):
        raise ValueError("Substantially negative energy statistic")
    # Exact energy distance is nonnegative; suppress floating-point cancellation.
    return max(0.0, float(value))


def marginal_w1(A, wa, B, wb, scale) -> np.ndarray:
    """Return independently scaled weighted Wasserstein-1 distance per column."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("Point clouds must be two-dimensional with matching columns")
    if A.shape[0] == 0 or B.shape[0] == 0 or not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError("Point clouds must be nonempty and finite")
    wa, wb = _normalized_weights(wa), _normalized_weights(wb)
    if wa.size != A.shape[0] or wb.size != B.shape[0]:
        raise ValueError("One weight is required per point")
    scale = np.asarray(scale, dtype=float)
    if scale.ndim == 0:
        scale = np.full(A.shape[1], float(scale))
    if scale.shape != (A.shape[1],) or not np.isfinite(scale).all() or (scale <= 0.0).any():
        raise ValueError("Scale must contain one finite positive value per column")
    return np.asarray([
        wasserstein_distance(A[:, column], B[:, column], u_weights=wa, v_weights=wb) / scale[column]
        for column in range(A.shape[1])
    ])
