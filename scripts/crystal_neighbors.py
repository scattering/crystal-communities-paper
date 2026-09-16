"""Shared, species-independent neighbour geometry for feature version 2.

CrystalNN's documented geometric mode operates directly on crystallographic
mixed/partially occupied sites. It describes the average site skeleton; it does
not infer a locally ordered alloy or sample vacancy configurations. Chemistry
is calculated separately from the original site occupancies.
"""
from __future__ import annotations

import numpy as np
from pymatgen.analysis.local_env import CrystalNN


FEATURE_VERSION = "crystal-features-v2-geometric-crystalnn"
NEIGHBOR_SETTINGS = {
    "weighted_cn": True,
    "cation_anion": False,
    "distance_cutoffs": None,
    "x_diff_weight": 0.0,
    "porous_adjustment": False,
}


def geometry_crystalnn() -> CrystalNN:
    """Use the same documented geometric rule for ordered and disordered sites."""
    return CrystalNN(**NEIGHBOR_SETTINGS)


def validate_neighbor_data(nndata, site_idx: int) -> None:
    probabilities = np.asarray(list(nndata.cn_weights.values()), dtype=float)
    if (
        not probabilities.size
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < -1e-10)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-8)
    ):
        raise ValueError(f"Invalid coordination probabilities at site {site_idx}")


def validate_neighbor_info(structure, site_idx: int, neighbors: list[dict]) -> None:
    """Reject unavailable geometry rather than returning a successful zero vector.

    Periodic images of the central site are valid neighbours. Their Cartesian
    displacement must be nonzero even when ``site_index == site_idx``.
    """
    if not neighbors:
        raise ValueError(f"No geometric neighbours at site {site_idx}")
    for entry in neighbors:
        index = int(entry["site_index"])
        weight = float(entry["weight"])
        displacement = np.asarray(entry["site"].coords) - structure[site_idx].coords
        if not 0 <= index < len(structure):
            raise ValueError(f"Invalid neighbour index {index} at site {site_idx}")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid neighbour weight at site {site_idx}: {weight}")
        if not np.isfinite(displacement).all() or np.linalg.norm(displacement) <= 1e-10:
            raise ValueError(f"Invalid periodic neighbour displacement at site {site_idx}")
