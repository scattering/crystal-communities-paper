#!/usr/bin/env python3
"""Production featurization worker for the ICSD densification pipeline.

This module is the single source of truth for the structural embedding
used by every downstream analysis in the manuscript: the frozen ICSD
PCA basis, the per-community centroids, the held-out frontier rates,
and the projection of every external source (GNoME, MatterGen, MP,
JARVIS, Alexandria, AFLOW, A-Lab). Importing
``build_structure_embedding`` from here is what guarantees that the
external sources are scored against the *same* descriptor that defines
the ICSD map; the function is reused unchanged by
``analyze_alab_validation.py`` and by every external-frontier producer
referenced from the Methods section.

Featurization (per CIF):
  * per-site chemistry vector (one-hot over Element, weighted by
    occupancy) concatenated with a local-geometry block computed
    either from matminer's ``CrystalNNFingerprint("ops")`` (the
    production setting, ``--local-mode matminer_ops``) or a fast
    fallback that uses CN/distance/species moments (``fast_local``);
  * Voronoi-graph adjacency from ``pymatgen.analysis.local_env.CrystalNN``
    weighted by neighbor weight, used as the message-passing graph;
  * ``--wl-iters`` rounds (default 3) of weighted Weisfeiler-Lehman
    update ``x_i <- x_i + sum_j w_ij x_j / sum_j w_ij``;
  * pooled to a structure-level vector by concatenating per-site
    mean / max / variance across all sites, plus a 9-d global block
    (a, b, c, alpha, beta, gamma, volume, density, n_sites).

The module is invoked by the multi-process driver
(``icsd_continuous_wl_densification.py``) which calls ``init_worker``
once per process to bind the ICSD zip handle and the per-worker
CrystalNN / fingerprint instances, then dispatches CIFs to
``featurize_record`` / ``featurize_record_batch``. Structures with
more than ``WORKER_MAX_SITES`` (default 256) are rejected as
``too_many_sites`` to keep the per-CIF wall-time bounded.

This file has no ``__main__`` entry point: it is purely a worker
library imported by the driver and by external-source projection
scripts.
"""

from __future__ import annotations

import re
import warnings
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Element, Structure

try:
    from matminer.featurizers.site import CrystalNNFingerprint
except ImportError:  # pragma: no cover - handled at runtime on TACC
    CrystalNNFingerprint = None


ID_COLUMN_CANDIDATES = [
    "ICSDid",
    "ICSD ID",
    "ICSD_ID",
    "Collection Code",
    "Coll Code",
    "coll_code",
    "cif_names",
]

YEAR_COLUMN_CANDIDATES = [
    "publication_year",
    "Publication Year",
    "Publication year",
    "PubYear",
    "Year",
    "year",
]


@dataclass
class Record:
    icsd_id: int
    year: int | None
    row: dict[str, str]


WORKER_ZIP: zipfile.ZipFile | None = None
WORKER_ZIP_PASSWORD: str | None = None
WORKER_CIF_ROOT: Path | None = None
WORKER_WL_ITERS: int = 3
WORKER_MAX_SITES: int = 256
WORKER_LOCAL_MODE: str = "matminer_ops"
WORKER_CNN: CrystalNN | None = None
WORKER_SITE_FP: CrystalNNFingerprint | None = None
WORKER_GEOM_DIM: int = 4


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def parse_icsd_id(raw_id: str) -> int | None:
    value = raw_id.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        match = re.search(r"(\d+)", value)
        if match:
            return int(match.group(1))
    return None


def cif_member_name(icsd_id: int) -> str:
    return f"FindIt_CIFs/icsd_{icsd_id:06d}.cif"


def cif_file_path(cif_root: Path, icsd_id: int) -> Path:
    return cif_root / f"icsd_{icsd_id:06d}.cif"


def read_structure_from_zip(zf: zipfile.ZipFile, icsd_id: int, zip_password: str | None = None) -> Structure:
    member = cif_member_name(icsd_id)
    pwd = zip_password.encode("utf-8") if zip_password else None
    with zf.open(member, pwd=pwd) as handle:
        text = handle.read().decode("utf-8", errors="replace")
    return Structure.from_str(text, fmt="cif")


def read_structure_from_dir(cif_root: Path, icsd_id: int) -> Structure:
    return Structure.from_file(cif_file_path(cif_root, icsd_id))


def element_vector_from_species(species) -> np.ndarray:
    vec = np.zeros(7, dtype=float)
    total_occ = 0.0
    for sp, occ in species.items():
        try:
            el = Element(str(sp))
        except Exception:
            continue
        total_occ += float(occ)
        atomic_radius = float(el.atomic_radius or 0.0)
        avg_ionic = float(el.average_ionic_radius or 0.0)
        x = float(el.X or 0.0)
        row = float(el.row or 0.0)
        group = float(el.group or 0.0)
        block = {"s": 0.0, "p": 1.0, "d": 2.0, "f": 3.0}.get(getattr(el, "block", ""), -1.0)
        contrib = np.array(
            [float(el.Z), row, group, atomic_radius, avg_ionic, x, block],
            dtype=float,
        )
        vec += float(occ) * contrib
    if total_occ > 0:
        vec /= total_occ
    return vec


def geometry_dim(site_fp: CrystalNNFingerprint | None) -> int:
    if site_fp is None:
        return 4
    try:
        return len(site_fp.feature_labels())
    except Exception:
        return 4


def fast_local_geometry(
    structure: Structure,
    site_idx: int,
    nn: list[dict],
) -> np.ndarray:
    if not nn:
        return np.zeros(12, dtype=float)

    weights = np.array([float(item.get("weight", 1.0)) for item in nn], dtype=float)
    weights = np.clip(weights, 0.0, None)
    weight_sum = float(weights.sum()) or 1.0

    distances = []
    neighbor_vectors = []
    for item, w in zip(nn, weights):
        nbr_site = item["site"]
        try:
            d = float(structure[site_idx].distance(nbr_site))
        except Exception:
            d = 0.0
        distances.append(d)
        neighbor_vectors.append(element_vector_from_species(nbr_site.species))

    distances_arr = np.array(distances, dtype=float)
    nbr_mean = sum(w * v for w, v in zip(weights, neighbor_vectors)) / weight_sum
    return np.concatenate(
        [
            np.array(
                [
                    weight_sum,
                    float(distances_arr.mean()) if len(distances_arr) else 0.0,
                    float(distances_arr.std()) if len(distances_arr) else 0.0,
                    float(np.count_nonzero(weights > 0)),
                    float(np.max(weights)) if len(weights) else 0.0,
                    float(np.min(distances_arr)) if len(distances_arr) else 0.0,
                    float(np.max(distances_arr)) if len(distances_arr) else 0.0,
                    float(len({str(sp) for item in nn for sp in item["site"].species.keys()})),
                ],
                dtype=float,
            ),
            nbr_mean[:4],
        ],
        axis=0,
    )


def weighted_nn_from_nndata(site_fp: CrystalNNFingerprint, nndata) -> list[dict]:
    if not site_fp.cnn.weighted_cn:
        max_key = max(nndata.cn_weights, key=lambda k: nndata.cn_weights[k])
        nn = [dict(entry) for entry in nndata.cn_nninfo[max_key]]
        for entry in nn:
            entry["weight"] = 1.0
        return nn

    all_nninfo = [dict(entry) for entry in nndata.all_nninfo]
    for entry in all_nninfo:
        weight = 0.0
        for cn in nndata.cn_nninfo:
            for cn_entry in nndata.cn_nninfo[cn]:
                if entry["site"] == cn_entry["site"]:
                    weight += nndata.cn_weights[cn]
        entry["weight"] = weight
    return all_nninfo


def fingerprint_from_nndata(site_fp: CrystalNNFingerprint, struct: Structure, idx: int, nndata) -> np.ndarray:
    max_cn = sorted(site_fp.op_types)[-1]
    cn_fingerprint: list[float] = []
    for k in range(max_cn):
        cn = k + 1
        wt = nndata.cn_weights.get(cn, 0)
        if cn not in site_fp.ops:
            continue
        for op in site_fp.ops[cn]:
            if op == "wt":
                cn_fingerprint.append(wt)
            elif wt == 0:
                cn_fingerprint.append(0.0)
            else:
                neigh_sites = [d["site"] for d in nndata.cn_nninfo[cn]]
                opval = op.get_order_parameters(
                    [struct[idx]] + neigh_sites,
                    0,
                    indices_neighs=[i for i in range(1, len(neigh_sites) + 1)],
                )[0]
                opval = opval or 0.0
                cn_fingerprint.append(wt * opval)
    return np.asarray(cn_fingerprint, dtype=float)


def local_geometry_vector(
    structure: Structure,
    site_idx: int,
    cnn: CrystalNN,
    site_fp: CrystalNNFingerprint | None,
    geom_dim: int,
    local_mode: str,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    try:
        nndata = cnn.get_nn_data(structure, site_idx)
        if local_mode == "matminer_ops" and site_fp is not None:
            nn = weighted_nn_from_nndata(site_fp, nndata)
        else:
            nn = cnn.get_nn_info(structure, site_idx)
    except Exception:
        nndata = None
        nn = []

    geom_fp = None
    if local_mode == "matminer_ops" and site_fp is not None:
        try:
            if nndata is None:
                raise ValueError("Missing CrystalNN data")
            geom_fp = fingerprint_from_nndata(site_fp, structure, site_idx, nndata)
            if geom_fp.shape[0] != geom_dim:
                raise ValueError(f"Unexpected geometry feature length {geom_fp.shape[0]} != {geom_dim}")
        except Exception:
            geom_fp = None
    elif local_mode == "fast_local":
        geom_fp = fast_local_geometry(structure, site_idx, nn)
    elif local_mode == "chem_only":
        geom_fp = np.zeros(geom_dim, dtype=float)
        nn = []

    if not nn:
        if geom_fp is not None:
            return geom_fp, []
        return np.zeros(geom_dim, dtype=float), []

    weights = np.array([float(item.get("weight", 1.0)) for item in nn], dtype=float)
    weights = np.clip(weights, 0.0, None)

    distances = []
    neighbor_species = Counter()
    neighbors: list[tuple[int, float]] = []
    for item, w in zip(nn, weights):
        nbr_site = item["site"]
        nbr_idx = int(item["site_index"])
        neighbors.append((nbr_idx, float(w)))
        try:
            d = float(structure[site_idx].distance(nbr_site))
        except Exception:
            d = 0.0
        distances.append(d)
        for sp, occ in nbr_site.species.items():
            neighbor_species[str(sp)] += float(occ) * float(w)

    distances_arr = np.array(distances, dtype=float)
    if geom_fp is not None:
        geom = geom_fp
    else:
        geom = np.array(
            [
                float(weights.sum()),
                float(distances_arr.mean()) if len(distances_arr) else 0.0,
                float(distances_arr.std()) if len(distances_arr) else 0.0,
                float(len(neighbor_species)),
            ],
            dtype=float,
        )
    return geom, neighbors


def build_structure_embedding(
    structure: Structure,
    wl_iters: int,
    cnn: CrystalNN | None = None,
    site_fp: CrystalNNFingerprint | None = None,
    geom_dim: int | None = None,
    local_mode: str = "matminer_ops",
) -> np.ndarray:
    n_sites = len(structure)
    cnn = cnn or CrystalNN(weighted_cn=True, x_diff_weight=0.0, porous_adjustment=False)
    if local_mode == "matminer_ops":
        site_fp = site_fp or (CrystalNNFingerprint.from_preset("ops") if CrystalNNFingerprint is not None else None)
        geom_dim = geom_dim or geometry_dim(site_fp)
    elif local_mode == "fast_local":
        site_fp = None
        geom_dim = geom_dim or 12
    else:
        site_fp = None
        geom_dim = geom_dim or 12

    x0 = []
    adjacency: list[list[tuple[int, float]]] = []
    for i, site in enumerate(structure):
        chem = element_vector_from_species(site.species)
        geom, nbrs = local_geometry_vector(structure, i, cnn, site_fp, geom_dim, local_mode)
        x0.append(np.concatenate([chem, geom], axis=0))
        adjacency.append(nbrs)

    x = np.vstack(x0)
    for _ in range(wl_iters):
        x_next = np.zeros_like(x)
        for i in range(n_sites):
            nbrs = adjacency[i]
            if not nbrs:
                x_next[i] = x[i]
                continue
            wsum = sum(w for _, w in nbrs) or 1.0
            nbr_mean = sum(w * x[j] for j, w in nbrs) / wsum
            x_next[i] = x[i] + nbr_mean
        x = x_next

    pooled = np.concatenate([x.mean(axis=0), x.max(axis=0), x.var(axis=0)], axis=0)
    lattice = structure.lattice
    global_vec = np.array(
        [
            float(lattice.a),
            float(lattice.b),
            float(lattice.c),
            float(lattice.alpha),
            float(lattice.beta),
            float(lattice.gamma),
            float(structure.volume),
            float(structure.density),
            float(n_sites),
        ],
        dtype=float,
    )
    return np.concatenate([pooled, global_vec], axis=0)


def init_worker(
    icsd_zip: str | None,
    zip_password: str | None,
    cif_root: str | None,
    wl_iters: int,
    max_sites: int,
    local_mode: str,
) -> None:
    global WORKER_ZIP
    global WORKER_ZIP_PASSWORD
    global WORKER_CIF_ROOT
    global WORKER_WL_ITERS
    global WORKER_MAX_SITES
    global WORKER_LOCAL_MODE
    global WORKER_CNN
    global WORKER_SITE_FP
    global WORKER_GEOM_DIM

    warnings.filterwarnings(
        "ignore",
        message=r"CrystalNN: cannot locate an appropriate radius.*",
        category=UserWarning,
    )

    WORKER_ZIP = zipfile.ZipFile(icsd_zip) if icsd_zip else None
    WORKER_ZIP_PASSWORD = zip_password
    WORKER_CIF_ROOT = Path(cif_root) if cif_root else None
    WORKER_WL_ITERS = wl_iters
    WORKER_MAX_SITES = max_sites
    WORKER_LOCAL_MODE = local_mode
    WORKER_CNN = CrystalNN(weighted_cn=True, x_diff_weight=0.0, porous_adjustment=False)
    if local_mode == "matminer_ops":
        WORKER_SITE_FP = CrystalNNFingerprint.from_preset("ops") if CrystalNNFingerprint is not None else None
        WORKER_GEOM_DIM = geometry_dim(WORKER_SITE_FP)
    else:
        WORKER_SITE_FP = None
        WORKER_GEOM_DIM = 12


def featurize_record(rec: Record) -> tuple[bool, Record, np.ndarray | None, dict | None]:
    try:
        if WORKER_ZIP is not None:
            structure = read_structure_from_zip(WORKER_ZIP, rec.icsd_id, WORKER_ZIP_PASSWORD)
        elif WORKER_CIF_ROOT is not None:
            structure = read_structure_from_dir(WORKER_CIF_ROOT, rec.icsd_id)
        else:
            raise RuntimeError("Worker not initialized with data source")

        if len(structure) > WORKER_MAX_SITES:
            return False, rec, None, {"icsd_id": rec.icsd_id, "reason": f"too_many_sites:{len(structure)}"}

        emb = build_structure_embedding(
            structure,
            WORKER_WL_ITERS,
            cnn=WORKER_CNN,
            site_fp=WORKER_SITE_FP,
            geom_dim=WORKER_GEOM_DIM,
            local_mode=WORKER_LOCAL_MODE,
        )
        return True, rec, emb, None
    except Exception as exc:  # pragma: no cover - exploratory pipeline
        return False, rec, None, {"icsd_id": rec.icsd_id, "reason": type(exc).__name__, "detail": str(exc)[:200]}


def featurize_record_batch(records: list[Record]) -> list[tuple[bool, Record, np.ndarray | None, dict | None]]:
    return [featurize_record(rec) for rec in records]
