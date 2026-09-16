"""Continuous first- and second-order metrics on a frozen neighbour graph.

This small module deliberately reuses the elemental properties and neighbour
weights of ``experiments.graphlet_compare.graphlet_features``.  It is limited
to ordered structures: one elemental label, with unit occupancy, per site.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments.graphlet_compare import graphlet_features as gf  # noqa: E402


PROPERTY_NAMES = tuple(gf.ELEMENTAL_PROPERTIES)
CHANNEL_NAMES = tuple(
    f"{family}/{prop}"
    for family in ("f1", "f2_mean", "f2_absdiff")
    for prop in PROPERTY_NAMES
)


class FixedEdges(NamedTuple):
    """Directed base-site indices and graphlet weights for neighbour entries."""

    i: np.ndarray
    j: np.ndarray
    w: np.ndarray


Distribution = Tuple[np.ndarray, np.ndarray]
DistributionMap = Dict[str, Distribution]


def ordered_site_properties(structure) -> np.ndarray:
    """Return an ``(n_sites, 10)`` property matrix for an ordered structure."""
    rows = []
    for site_index, site in enumerate(structure):
        if not site.is_ordered or len(site.species) != 1:
            raise ValueError(f"Site {site_index} is not ordered")
        species, occupancy = next(iter(site.species.items()))
        if not np.isclose(float(occupancy), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"Site {site_index} does not have unit occupancy")
        element = getattr(species, "element", species)
        values = gf._element_property_dict(element)
        rows.append([values.get(prop, np.nan) for prop in PROPERTY_NAMES])
    return np.asarray(rows, dtype=float).reshape(len(structure), len(PROPERTY_NAMES))


def fixed_edges(structure, neighbor_info) -> FixedEdges:
    """Freeze graphlet directed neighbour entries as base indices and weights.

    Every supplied entry produces one edge.  Thus periodic-image multiedges and
    edges with ``i == j`` are retained.  Geometry is intentionally absent: the
    pilot changes labels on this fixed graph, while using the same neighbour
    weights as graphlet second-order accumulation.
    """
    if len(neighbor_info) != len(structure):
        raise ValueError("neighbor_info must contain one list per structure site")
    centers, neighbors, weights = [], [], []
    for i, entries in enumerate(neighbor_info):
        for entry in entries:
            j = int(entry["site_index"])
            if j < 0 or j >= len(structure):
                raise ValueError(f"Neighbor site index {j} is out of range")
            weight = float(entry.get("weight", 1.0))
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError("Neighbor weights must be finite and nonnegative")
            centers.append(i)
            neighbors.append(j)
            weights.append(weight)
    return FixedEdges(
        np.asarray(centers, dtype=np.intp),
        np.asarray(neighbors, dtype=np.intp),
        np.asarray(weights, dtype=float),
    )


def _coerce_edges(edges, n_sites: int) -> FixedEdges:
    try:
        i, j, w = (np.asarray(part) for part in edges)
    except Exception as exc:
        raise ValueError("edges must be an (i, j, weight) triple") from exc
    i = np.asarray(i, dtype=np.intp)
    j = np.asarray(j, dtype=np.intp)
    w = np.asarray(w, dtype=float)
    if i.ndim != 1 or j.ndim != 1 or w.ndim != 1 or not (i.size == j.size == w.size):
        raise ValueError("edge arrays must be one-dimensional and equally sized")
    if ((i < 0) | (i >= n_sites) | (j < 0) | (j >= n_sites)).any():
        raise ValueError("edge index is out of range")
    if not np.isfinite(w).all() or (w < 0.0).any():
        raise ValueError("edge weights must be finite and nonnegative")
    return FixedEdges(i, j, w)


def _normalized(values, weights) -> Tuple[Distribution, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    missing_mass = float(weights[np.isfinite(weights) & ~np.isfinite(values)].sum())
    values, weights = values[valid], weights[valid]
    valid_mass = float(weights.sum())
    if valid_mass <= 0.0:
        return (np.empty(0, dtype=float), np.empty(0, dtype=float)), missing_mass
    return (values, weights / valid_mass), missing_mass


def distributions(
    properties,
    edges,
    *,
    return_missing_mass: bool = False,
) -> Union[DistributionMap, Tuple[DistributionMap, Dict[str, float]]]:
    """Build normalized continuous distributions for the 30 pilot channels.

    Nonfinite property observations are omitted rather than replaced by zero.
    With ``return_missing_mass=True``, the second result maps each channel to
    the omitted unnormalized site or edge weight.  A wholly missing or
    zero-mass channel is represented by two empty arrays.
    """
    props = np.asarray(properties, dtype=float)
    if props.ndim != 2 or props.shape[1] != len(PROPERTY_NAMES):
        raise ValueError(f"properties must have shape (n_sites, {len(PROPERTY_NAMES)})")
    edge_data = _coerce_edges(edges, props.shape[0])
    result: DistributionMap = {}
    missing: Dict[str, float] = {}
    site_weights = np.ones(props.shape[0], dtype=float)
    for column, prop in enumerate(PROPERTY_NAMES):
        f1_name = f"f1/{prop}"
        result[f1_name], missing[f1_name] = _normalized(props[:, column], site_weights)

        left = props[edge_data.i, column]
        right = props[edge_data.j, column]
        pair_valid = np.isfinite(left) & np.isfinite(right)
        for family, values in (
            ("f2_mean", (left + right) / 2.0),
            ("f2_absdiff", np.abs(left - right)),
        ):
            name = f"{family}/{prop}"
            # Make either invalid endpoint explicitly invalid before filtering.
            values = np.where(pair_valid, values, np.nan)
            result[name], missing[name] = _normalized(values, edge_data.w)
    return (result, missing) if return_missing_mass else result


def shuffled_properties(
    properties,
    *,
    rng: Optional[np.random.Generator] = None,
    permutation=None,
) -> np.ndarray:
    """Permute complete site-property rows, preserving overall composition."""
    props = np.asarray(properties, dtype=float)
    if props.ndim != 2 or props.shape[1] != len(PROPERTY_NAMES):
        raise ValueError(f"properties must have shape (n_sites, {len(PROPERTY_NAMES)})")
    if permutation is None:
        generator = np.random.default_rng() if rng is None else rng
        permutation = generator.permutation(props.shape[0])
    permutation = np.asarray(permutation, dtype=np.intp)
    if permutation.shape != (props.shape[0],) or not np.array_equal(
        np.sort(permutation), np.arange(props.shape[0])
    ):
        raise ValueError("permutation must contain every site index exactly once")
    return props[permutation].copy()


def distance(
    raw_a: Mapping[str, Distribution],
    raw_b: Mapping[str, Distribution],
    scales: Mapping[str, float],
) -> Dict[str, float]:
    """Return scaled per-channel weighted Wasserstein distances.

    ``scales`` should normally contain a separate value for every channel,
    because first-order, pair means, and pair differences have different
    empirical spreads.  A property-name entry is accepted as a fallback.  A
    channel is omitted when its selected scale is absent, nonfinite, or zero.
    """
    output = {}
    for channel in CHANNEL_NAMES:
        prop = channel.split("/", 1)[1]
        try:
            values_a, weights_a = raw_a[channel]
            values_b, weights_b = raw_b[channel]
        except KeyError as exc:
            raise ValueError(f"Missing metric input for {channel}") from exc
        scale_value = scales.get(channel, scales.get(prop))
        if scale_value is None:
            continue
        scale = float(scale_value)
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        values_a, weights_a = np.asarray(values_a), np.asarray(weights_a)
        values_b, weights_b = np.asarray(values_b), np.asarray(weights_b)
        for values, weights in ((values_a, weights_a), (values_b, weights_b)):
            if values.ndim != 1 or weights.ndim != 1 or values.size != weights.size:
                raise ValueError(f"Malformed distribution for {channel}")
            if values.size == 0:
                raise ValueError(f"No valid mass for {channel}")
            if not np.isfinite(values).all() or not np.isfinite(weights).all():
                raise ValueError(f"Nonfinite distribution for {channel}")
            if (weights < 0.0).any() or not np.isclose(weights.sum(), 1.0):
                raise ValueError(f"Weights for {channel} must be normalized")
        output[channel] = float(wasserstein_distance(
            values_a, values_b, u_weights=weights_a, v_weights=weights_b,
        ) / scale)
    return output
