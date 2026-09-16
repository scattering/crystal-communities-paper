"""Graphlet histogram featurization — VENDORED for the structural-comparison
experiment (Graphlet branch).

Adapted with attribution from
  our group's earlier graphlet-histogram port (ansatz-bcs-toy, src/graphlets.py; not distributed)
which is our local adaptation of Lesser et al. 2025 (arXiv:2510.07373v2),
not a copy of the authors' implementation. It differs in neighbour selection,
elemental properties, disorder averaging, and triplet statistics; see
Supporting Information §S9.1 of the accompanying paper.
No GP here — we only use the histogram representation + EMD distance to ask
a purely structural question: does the graphlet-EMD geometry partition ICSD
the same way our moment-pooled PCA embedding does?

First-order: 10 elemental-property histograms. Second-order: distance + 10
elemental means + 10 abs-diffs over directed neighbour occurrences selected by
the declared CrystalNN or VoronoiNN rule.
Both directions of a reciprocal bond contribute; histogram normalization
removes the common factor without discarding directional weights or periodic
images. Third-order: unordered pairs of distinct neighbour images about each
centre, with cos(angle), shorter/longer arm lengths, and 30 elemental stats.
The historical names f3_d_ij and f3_d_jk now mean the shorter and longer arm,
respectively, so their values do not depend on neighbour-list ordering.
All distances use the actual Cartesian neighbour images. Both neighbour rules
operate on the same crystallographic site geometry and retain native disorder.
64 features, 20 bins each; global bin edges are shared across the subsample.
Missing or zero-mass histograms are errors, never fabricated uniform data.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core.local_env import VoronoiNN

from crystal_neighbors import (
    NEIGHBOR_SETTINGS,
    geometry_crystalnn,
    validate_neighbor_info,
)

log = logging.getLogger(__name__)

NUM_BINS = 20
DISTANCE_CUTOFF_TRIPLET = 6.0  # Angstrom
NEIGHBOR_METHODS = ("crystalnn", "voronoinn")
GRAPHLET_REPRESENTATION = "graphlet-64x20-periodic-images-short-long-arms"
GRAPHLET_FEATURE_VERSIONS = {
    "crystalnn": "graphlet-64x20-geometric-crystalnn-v1",
    "voronoinn": "graphlet-64x20-voronoinn-slater15-uniform-v1",
}
VORONOINN_SETTINGS = {
    "constructor": {
        "tol": 0.0,
        "targets": None,
        "cutoff": 13.0,
        "allow_pathological": False,
        "weight": "solid_angle",
        "extra_nn_info": False,
        "compute_adj_neighbors": False,
    },
    "distance_screen": "d <= 1.5 * (occupancy-normalized Slater radius_i + radius_j)",
    "distance_screen_multiplier": 1.5,
    "retained_neighbor_weight": "uniform",
}

ELEMENTAL_PROPERTIES = [
    "Z", "electronegativity", "electron_affinity", "atomic_radius",
    "average_ionic_radius", "num_valence_electrons", "group", "row",
    "atomic_mass", "ionization_energy",
]


def _safe_attr(elem, name):
    try:
        v = getattr(elem, name, None)
        if v is None:
            return np.nan
        if hasattr(v, "magnitude"):
            v = v.magnitude
        return float(v)
    except Exception:
        return np.nan


def _element_property_dict(element):
    out = {}
    for prop in ELEMENTAL_PROPERTIES:
        if prop == "Z":
            out[prop] = float(element.Z)
        elif prop == "electronegativity":
            out[prop] = _safe_attr(element, "X")
        elif prop == "electron_affinity":
            out[prop] = _safe_attr(element, "electron_affinity")
        elif prop == "atomic_radius":
            out[prop] = _safe_attr(element, "atomic_radius")
        elif prop == "average_ionic_radius":
            out[prop] = _safe_attr(element, "average_ionic_radius")
        elif prop == "num_valence_electrons":
            try:
                out[prop] = float(
                    sum(occ for n, l, occ in element.full_electronic_structure
                        if n == element.row)
                )
            except Exception:
                out[prop] = np.nan
        elif prop == "group":
            out[prop] = float(element.group) if element.group is not None else np.nan
        elif prop == "row":
            out[prop] = float(element.row) if element.row is not None else np.nan
        elif prop == "atomic_mass":
            out[prop] = _safe_attr(element, "atomic_mass")
        elif prop == "ionization_energy":
            out[prop] = _safe_attr(element, "ionization_energy")
    return out


def neighbor_settings(method: str) -> dict:
    if method == "crystalnn":
        return dict(NEIGHBOR_SETTINGS)
    if method == "voronoinn":
        return dict(VORONOINN_SETTINGS)
    raise ValueError(f"Unknown graphlet neighbour method: {method}")


def make_neighbor_finder(method: str):
    if method == "crystalnn":
        return geometry_crystalnn()
    if method == "voronoinn":
        return VoronoiNN(**VORONOINN_SETTINGS["constructor"])
    raise ValueError(f"Unknown graphlet neighbour method: {method}")


def _effective_atomic_radius(site, site_index: int) -> float:
    """Occupancy-normalized Slater radius used by the published NN screen."""
    numerator = denominator = 0.0
    for species, occupancy in site.species.items():
        element = getattr(species, "element", species)
        radius = getattr(element, "atomic_radius", None)
        if radius is None:
            raise ValueError(f"Missing atomic radius at site {site_index}")
        value = float(getattr(radius, "magnitude", radius))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid atomic radius at site {site_index}")
        numerator += float(occupancy) * value
        denominator += float(occupancy)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"Zero occupied mass at site {site_index}")
    return numerator / denominator


def _screen_voronoi_neighbors(
    structure, site_index: int, neighbors: list[dict], radii: list[float],
):
    """Apply the paper's 1.5 x summed-radius screen and count each graphlet once."""
    retained = []
    for entry in neighbors:
        neighbor_index = int(entry["site_index"])
        displacement = np.asarray(entry["site"].coords) - structure[site_index].coords
        distance = float(np.linalg.norm(displacement))
        cutoff = VORONOINN_SETTINGS["distance_screen_multiplier"] * (
            radii[site_index] + radii[neighbor_index]
        )
        if distance <= cutoff + 1e-12:
            record = dict(entry)
            record["weight"] = 1.0
            retained.append(record)
    return retained


def _structure_neighbor_info(structure, neighbor_finder=None, *, neighbor_method="crystalnn"):
    if neighbor_finder is None:
        neighbor_finder = make_neighbor_finder(neighbor_method)
    radii = (
        [_effective_atomic_radius(site, i) for i, site in enumerate(structure)]
        if neighbor_method == "voronoinn" else None
    )
    nn = []
    for i, _ in enumerate(structure):
        info = neighbor_finder.get_nn_info(structure, i)
        if neighbor_method == "voronoinn":
            info = _screen_voronoi_neighbors(structure, i, info, radii)
        validate_neighbor_info(structure, i, info)
        nn.append(info)
    return nn


@dataclass
class FeatureRegistry:
    first_order: List[str]
    second_order: List[str]
    third_order: List[str]

    @property
    def all(self) -> List[str]:
        return self.first_order + self.second_order + self.third_order


def build_feature_registry() -> FeatureRegistry:
    first = [f"f1_{p}" for p in ELEMENTAL_PROPERTIES]
    second = (["f2_distance"]
              + [f"f2_mean_{p}" for p in ELEMENTAL_PROPERTIES]
              + [f"f2_absdiff_{p}" for p in ELEMENTAL_PROPERTIES])
    third = (["f3_cos_angle", "f3_d_ij", "f3_d_jk"]
             + [f"f3_mean_{p}" for p in ELEMENTAL_PROPERTIES]
             + [f"f3_std_{p}" for p in ELEMENTAL_PROPERTIES]
             + [f"f3_range_{p}" for p in ELEMENTAL_PROPERTIES])
    return FeatureRegistry(first, second, third)


REGISTRY = build_feature_registry()
NUM_FEATURES = len(REGISTRY.all)


def _site_props(structure):
    from pymatgen.core import Element, Species
    out = []
    for i, site in enumerate(structure):
        sp_props = {}
        for sp, occ in site.species.items():
            try:
                el = sp.element if isinstance(sp, Species) else sp
                if not isinstance(el, Element):
                    el = Element(el.symbol)
                old_occ = sp_props.get(el.symbol, (0.0, None))[0]
                sp_props[el.symbol] = (
                    float(old_occ) + float(occ), _element_property_dict(el)
                )
            except Exception as exc:
                raise ValueError(f"Unsupported species {sp!r} at site {i}") from exc
        out.append(sp_props)
    return out


def _accumulate_first_order(structure):
    site_props = _site_props(structure)
    by_prop = {p: [] for p in ELEMENTAL_PROPERTIES}
    for sp_dict in site_props:
        for _el, (occ, props) in sp_dict.items():
            for p in ELEMENTAL_PROPERTIES:
                v = props.get(p, np.nan)
                if np.isfinite(v):
                    by_prop[p].append((v, float(occ)))
    return by_prop


def _accumulate_second_order(structure, nn_info_list):
    site_props = _site_props(structure)
    by_feat: Dict[str, List[Tuple[float, float]]] = {"f2_distance": []}
    for p in ELEMENTAL_PROPERTIES:
        by_feat[f"f2_mean_{p}"] = []
        by_feat[f"f2_absdiff_{p}"] = []
    for i, neighbors in enumerate(nn_info_list):
        for nbr in neighbors:
            j = nbr["site_index"]
            d = float(np.linalg.norm(nbr["site"].coords - structure[i].coords))
            wij = float(nbr.get("weight", 1.0))
            for _ei, (oi, pi) in site_props[i].items():
                for _ej, (oj, pj) in site_props[j].items():
                    w = wij * float(oi) * float(oj)
                    by_feat["f2_distance"].append((d, w))
                    for p in ELEMENTAL_PROPERTIES:
                        vi, vj = pi.get(p, np.nan), pj.get(p, np.nan)
                        if np.isfinite(vi) and np.isfinite(vj):
                            by_feat[f"f2_mean_{p}"].append(((vi + vj) / 2.0, w))
                            by_feat[f"f2_absdiff_{p}"].append((abs(vi - vj), w))
    return by_feat


def _accumulate_third_order(structure, nn_info_list):
    site_props = _site_props(structure)
    by_feat: Dict[str, List[Tuple[float, float]]] = {
        "f3_cos_angle": [], "f3_d_ij": [], "f3_d_jk": []}
    for p in ELEMENTAL_PROPERTIES:
        by_feat[f"f3_mean_{p}"] = []
        by_feat[f"f3_std_{p}"] = []
        by_feat[f"f3_range_{p}"] = []
    coords = structure.cart_coords
    for j in range(len(structure)):
        nbrs = nn_info_list[j]
        if len(nbrs) < 2:
            continue
        for a, b in combinations(range(len(nbrs)), 2):
            na, nb = nbrs[a], nbrs[b]
            i, k = na["site_index"], nb["site_index"]
            # Equal site indices may identify distinct periodic images.
            ri = np.asarray(na["site"].coords) - coords[j]
            rk = np.asarray(nb["site"].coords) - coords[j]
            d_ij = float(np.linalg.norm(ri))
            d_jk = float(np.linalg.norm(rk))
            if d_ij > DISTANCE_CUTOFF_TRIPLET or d_jk > DISTANCE_CUTOFF_TRIPLET:
                continue
            denom = d_ij * d_jk
            if denom <= 0 or not np.isfinite(denom):
                raise ValueError(f"Invalid triplet arm at site {j}")
            cos_t = float(np.clip(np.dot(ri, rk) / denom, -1.0, 1.0))
            d_short, d_long = sorted((d_ij, d_jk))
            wjk = float(na.get("weight", 1.0)) * float(nb.get("weight", 1.0))
            for _ei, (oi, pi) in site_props[i].items():
                for _ej, (oj, pj) in site_props[j].items():
                    for _ek, (ok, pk) in site_props[k].items():
                        w = wjk * float(oi) * float(oj) * float(ok)
                        by_feat["f3_cos_angle"].append((cos_t, w))
                        by_feat["f3_d_ij"].append((d_short, w))
                        by_feat["f3_d_jk"].append((d_long, w))
                        for p in ELEMENTAL_PROPERTIES:
                            vs = [pi.get(p, np.nan), pj.get(p, np.nan), pk.get(p, np.nan)]
                            if not all(np.isfinite(v) for v in vs):
                                continue
                            arr = np.asarray(vs, dtype=float)
                            by_feat[f"f3_mean_{p}"].append((float(arr.mean()), w))
                            by_feat[f"f3_std_{p}"].append(
                                (float(arr.std()) if arr.size > 1 else 0.0, w))
                            by_feat[f"f3_range_{p}"].append(
                                (float(arr.max() - arr.min()) if arr.size > 1 else 0.0, w))
    return by_feat


def all_raw_features(structure, *, neighbor_method="crystalnn", neighbor_finder=None):
    nn = _structure_neighbor_info(
        structure, neighbor_finder, neighbor_method=neighbor_method,
    )
    raw: Dict[str, List[Tuple[float, float]]] = {}
    for p, vals in _accumulate_first_order(structure).items():
        raw[f"f1_{p}"] = vals
    raw.update(_accumulate_second_order(structure, nn))
    raw.update(_accumulate_third_order(structure, nn))
    for k in REGISTRY.all:
        raw.setdefault(k, [])
    for k in ("f2_distance", "f3_cos_angle", "f3_d_ij", "f3_d_jk"):
        if not raw[k] or sum(w for _, w in raw[k]) <= 0:
            raise ValueError(f"Missing or zero-mass graphlet feature: {k}")
    return raw


def global_bin_edges(all_raw: Dict[str, Dict]) -> Dict[str, list]:
    edges = {}
    for fname in REGISTRY.all:
        vals = []
        for raw in all_raw.values():
            vals.extend(v for v, _ in raw.get(fname, []))
        if not vals:
            raise ValueError(f"No observations for bin-edge feature: {fname}")
        arr = np.asarray(vals, dtype=float)
        lo, hi = float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            lo, hi = float(arr.min()), float(arr.max())
            if hi - lo < 1e-9:
                hi = lo + 1.0
        edges[fname] = [lo, hi]
    return edges


def _histogram(values_weights, lo, hi, num_bins=NUM_BINS, *, feature_name="unknown"):
    if not values_weights:
        raise ValueError(f"Missing graphlet feature: {feature_name}")
    vals = np.asarray([v for v, _ in values_weights], dtype=float)
    wts = np.asarray([w for _, w in values_weights], dtype=float)
    if not np.isfinite(vals).all() or not np.isfinite(wts).all() or (wts < 0).any():
        raise ValueError(f"Invalid values or weights for graphlet feature: {feature_name}")
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid bin edges for graphlet feature: {feature_name}")
    # Equivalent cells can put exact symmetry angles infinitesimally either
    # side of a bin edge. Snap only roundoff-scale offsets, in bin units, so
    # an orthogonal angle does not move bins when a cell is re-expressed.
    positions = (np.clip(vals, lo, hi) - lo) * (num_bins / (hi - lo))
    nearest = np.rint(positions)
    positions = np.where(np.abs(positions - nearest) <= 1e-10, nearest, positions)
    indices = np.clip(np.floor(positions), 0, num_bins - 1).astype(int)
    h = np.bincount(indices, weights=wts, minlength=num_bins)
    s = h.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Zero-mass graphlet feature: {feature_name}")
    return (h / s).astype(np.float32)


def histogram_tensor(raw, bin_edges) -> np.ndarray:
    """(NUM_FEATURES, NUM_BINS) normalized-histogram tensor for one structure."""
    out = np.zeros((NUM_FEATURES, NUM_BINS), dtype=np.float32)
    for k, fname in enumerate(REGISTRY.all):
        lo, hi = bin_edges[fname]
        out[k] = _histogram(raw.get(fname, []), lo, hi, feature_name=fname)
    return out
