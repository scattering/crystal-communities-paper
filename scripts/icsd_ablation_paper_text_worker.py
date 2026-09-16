#!/usr/bin/env python3
"""Paper-text-faithful featurization worker for the methods-robustness ablation.

This is a *deliberate* re-implementation of the structural embedding
described in the Nature manuscript Methods paragraph, which differs in
three ways from the production worker (`icsd_densify_worker.py`):

1. Per-site chemistry vector: 22-dim occupancy-weighted Magpie elemental
   properties (matminer.utils.data.MagpieData), not the production
   7-dim hand-rolled (Z, row, group, atomic_radius, average_ionic_radius,
   X, block) vector.

2. WL update rule: at each round, for every site i, compute the weighted
   mean and weighted standard deviation of the current per-site features
   over its neighbors j, then *concatenate* onto the central atom:
       x^{t+1}_i = [ x^t_i || mean_j(x^t_j) || std_j(x^t_j) ]
   (production rule is additive: x^{t+1}_i = x^t_i + mean_j(x^t_j),
   no std, dimension fixed.)

3. Structure-level pool: mean + std over sites (production: mean + max + var).

Geometry side is identical to production: matminer
CrystalNNFingerprint("ops") fingerprint over a CrystalNN bond graph,
weighted CN. The same MAX_SITES=256 cap, same 9-d global lattice block,
and the same downstream StandardScaler->PCA(32) are reused unchanged so
that any partition difference reflects the chem+WL+pool changes only.

Exposed interface mirrors `icsd_densify_worker.py` (Record dataclass,
init_worker, featurize_record, featurize_record_batch, find_column,
parse_icsd_id, ID_COLUMN_CANDIDATES, YEAR_COLUMN_CANDIDATES) so the
existing parallel driver scaffolding and external-source projection
scripts could in principle re-use it; the ablation driver
`icsd_ablation_paper_text.py` imports from here.

Pre-PCA dimension accounting (verified against installed matminer in
the production conda env on Stampede3, see ablation README):
    chem (Magpie elemental):           22
    geom (CrystalNNFingerprint ops):   61
    per-site initial:                  22 + 61 = 83
    per-site after 3 paper-WL rounds:  83 * 3**3 = 2,241
    structure pool [mean || std]:      2,241 * 2 = 4,482
    + globals (a,b,c,al,be,ga,V,d,N):  9
    final pre-PCA dim:                 4,491

This is materially smaller than the back-of-envelope ~10,431 estimate
because Magpie is a 22-element-property table; the 132-dim
ElementProperty composition featurizer reaches 132 via 6 stats applied
across multiple atoms in a Composition, which doesn't apply in the
per-site occupancy-weighted setting.
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
from pymatgen.core import Element, Species, Structure

from crystal_neighbors import FEATURE_VERSION as GEOMETRY_FEATURE_VERSION, NEIGHBOR_SETTINGS, geometry_crystalnn
from icsd_densify_worker import geometry_dim, local_geometry_vector as production_local_geometry_vector

FEATURE_VERSION = f"{GEOMETRY_FEATURE_VERSION}-magpie22-concat-mean-std"

try:
    from matminer.featurizers.site import CrystalNNFingerprint
except ImportError:  # pragma: no cover - handled at runtime on TACC
    CrystalNNFingerprint = None

try:
    from matminer.utils.data import MagpieData
except ImportError:  # pragma: no cover - handled at runtime on TACC
    MagpieData = None


# Magpie's 22 element-level properties used by the composition-level
# ElementProperty.from_preset("magpie") featurizer. The composition
# featurizer applies 6 stats over these 22 to reach 132 dims; in the
# per-site occupancy-weighted setting we use the 22 raw values directly.
MAGPIE_PROPERTIES = [
    "Number",
    "MendeleevNumber",
    "AtomicWeight",
    "MeltingT",
    "Column",
    "Row",
    "CovalentRadius",
    "Electronegativity",
    "NsValence",
    "NpValence",
    "NdValence",
    "NfValence",
    "NValence",
    "NsUnfilled",
    "NpUnfilled",
    "NdUnfilled",
    "NfUnfilled",
    "NUnfilled",
    "GSvolume_pa",
    "GSbandgap",
    "GSmagmom",
    "SpaceGroupNumber",
]

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
WORKER_CNN: CrystalNN | None = None
WORKER_SITE_FP: CrystalNNFingerprint | None = None
WORKER_GEOM_DIM: int = 61
WORKER_MAGPIE: "MagpieData | None" = None
# Cached per-element Magpie vectors; element symbol -> np.ndarray(22).
WORKER_MAGPIE_CACHE: dict[str, np.ndarray] = {}


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


def magpie_vector_for_element(symbol: str) -> np.ndarray:
    """22-d Magpie elemental property vector for a single element symbol.

    Cached after first lookup; missing table values are filled with 0.0.
    """
    if symbol in WORKER_MAGPIE_CACHE:
        return WORKER_MAGPIE_CACHE[symbol]
    if WORKER_MAGPIE is None:
        raise RuntimeError("Worker not initialized: MagpieData unavailable")
    out = np.zeros(len(MAGPIE_PROPERTIES), dtype=float)
    for i, prop in enumerate(MAGPIE_PROPERTIES):
        value = WORKER_MAGPIE.get_elemental_property(Element(symbol), prop)
        out[i] = float(value) if value is not None and np.isfinite(value) else 0.0
    WORKER_MAGPIE_CACHE[symbol] = out
    return out


def magpie_vector_from_species(species) -> np.ndarray:
    """22-d occupancy-weighted Magpie vector for a (possibly disordered) site.

    Mirrors the structure of `element_vector_from_species` in the production
    worker: iterates over (specie, occupancy), accumulates occ*vec, then
    divides by total_occ to renormalize. Vacancy fractions are therefore
    invisible (same caveat as production); aliovalent vs isovalent
    substitution is treated identically (Magpie is neutral-atom only).
    """
    vec = np.zeros(len(MAGPIE_PROPERTIES), dtype=float)
    total_occ = 0.0
    for sp, occ in species.items():
        el = sp.element if isinstance(sp, Species) else sp
        if not isinstance(el, Element):
            raise ValueError(f"Unsupported chemical species: {sp!r}")
        if not np.isfinite(occ) or occ <= 0:
            raise ValueError(f"Invalid occupancy for {sp!r}: {occ}")
        total_occ += float(occ)
        vec += float(occ) * magpie_vector_for_element(el.symbol)
    if total_occ <= 0 or not np.isfinite(vec).all():
        raise ValueError("Unavailable or non-finite Magpie elemental properties")
    vec /= total_occ
    return vec


def local_geometry_vector(
    structure: Structure,
    site_idx: int,
    cnn: CrystalNN,
    site_fp: CrystalNNFingerprint,
    geom_dim: int,
    diagnostics: dict | None = None,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Use the repaired production geometry while retaining distinct chemistry/WL."""
    return production_local_geometry_vector(
        structure, site_idx, cnn, site_fp, geom_dim, "matminer_ops", diagnostics
    )


def paper_wl_round(x: np.ndarray, adjacency: list[list[tuple[int, float]]]) -> np.ndarray:
    """One round of paper-text WL: x^{t+1}_i = [x^t_i || w-mean_j(x^t_j) || w-std_j(x^t_j)].

    For sites with no neighbors, mean and std are zero vectors of shape (D,).
    Output shape: (n_sites, 3 * D).
    """
    n_sites, D = x.shape
    out = np.zeros((n_sites, 3 * D), dtype=float)
    for i in range(n_sites):
        nbrs = adjacency[i]
        if not nbrs:
            out[i, :D] = x[i]
            # mean and std remain zeros
            continue
        idxs = np.array([j for j, _ in nbrs], dtype=int)
        ws = np.array([w for _, w in nbrs], dtype=float)
        wsum = float(ws.sum())
        if wsum <= 0:
            out[i, :D] = x[i]
            continue
        nbr_x = x[idxs]                                  # shape (k, D)
        ws_norm = ws / wsum                              # weights sum to 1
        mean = (ws_norm[:, None] * nbr_x).sum(axis=0)    # shape (D,)
        var = (ws_norm[:, None] * (nbr_x - mean) ** 2).sum(axis=0)
        std = np.sqrt(np.clip(var, 0.0, None))           # shape (D,)
        out[i, :D] = x[i]
        out[i, D : 2 * D] = mean
        out[i, 2 * D : 3 * D] = std
    return out


def build_structure_embedding_paper(
    structure: Structure,
    wl_iters: int,
    cnn: CrystalNN | None = None,
    site_fp: CrystalNNFingerprint | None = None,
    geom_dim: int | None = None,
    diagnostics: dict | None = None,
) -> np.ndarray:
    """Paper-text-faithful structure embedding.

    1. Per-site initial vector = [Magpie22 || CrystalNN-ops61], dim 83.
    2. wl_iters rounds of paper-text WL (concat self || mean || std), each
       round multiplies per-site dim by 3. After 3 rounds: 83 * 27 = 2,241.
    3. Structure pool = [mean_i x_i || std_i x_i], dim 2,241 * 2 = 4,482.
    4. Concatenate 9-d global lattice block. Final dim 4,491.
    """
    n_sites = len(structure)
    if not n_sites or wl_iters < 0:
        raise ValueError("A nonempty structure and nonnegative wl_iters are required")
    cnn = cnn or geometry_crystalnn()
    if diagnostics is not None:
        diagnostics.update({"n_sites": n_sites, "ordered": bool(structure.is_ordered),
                            "sites_with_unrepresented_cn_mass": 0,
                            "max_unrepresented_cn_mass": 0.0})
    if site_fp is None:
        if CrystalNNFingerprint is None:
            raise RuntimeError("CrystalNNFingerprint not available; install matminer")
        site_fp = CrystalNNFingerprint.from_preset("ops")
    geom_dim = geom_dim or geometry_dim(site_fp)

    x0 = []
    adjacency: list[list[tuple[int, float]]] = []
    for i, site in enumerate(structure):
        chem = magpie_vector_from_species(site.species)
        geom, nbrs = local_geometry_vector(structure, i, cnn, site_fp, geom_dim, diagnostics)
        x0.append(np.concatenate([chem, geom], axis=0))
        adjacency.append(nbrs)

    x = np.vstack(x0)  # shape (n_sites, 83)
    for _ in range(wl_iters):
        x = paper_wl_round(x, adjacency)

    pooled = np.concatenate([x.mean(axis=0), x.std(axis=0)], axis=0)
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
    embedding = np.concatenate([pooled, global_vec], axis=0)
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite structure embedding")
    return embedding


def init_worker(
    icsd_zip: str | None,
    zip_password: str | None,
    cif_root: str | None,
    wl_iters: int,
    max_sites: int,
    local_mode: str = "paper_text",
) -> None:
    """Per-worker initialization. The local_mode kw is accepted for parity with
    the production worker's signature so the existing process-pool initargs
    pattern (`(icsd_zip, zip_password, cif_root, wl_iters, max_sites, local_mode)`)
    works unchanged; this worker only implements the paper_text mode.
    """
    global WORKER_ZIP, WORKER_ZIP_PASSWORD, WORKER_CIF_ROOT
    global WORKER_WL_ITERS, WORKER_MAX_SITES
    global WORKER_CNN, WORKER_SITE_FP, WORKER_GEOM_DIM
    global WORKER_MAGPIE, WORKER_MAGPIE_CACHE

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
    WORKER_CNN = geometry_crystalnn()
    if CrystalNNFingerprint is None:
        raise RuntimeError("matminer.featurizers.site.CrystalNNFingerprint missing")
    if MagpieData is None:
        raise RuntimeError("matminer.utils.data.MagpieData missing")
    WORKER_SITE_FP = CrystalNNFingerprint.from_preset("ops")
    WORKER_GEOM_DIM = geometry_dim(WORKER_SITE_FP)
    WORKER_MAGPIE = MagpieData()
    WORKER_MAGPIE_CACHE = {}


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

        diagnostics: dict = {}
        emb = build_structure_embedding_paper(
            structure,
            WORKER_WL_ITERS,
            cnn=WORKER_CNN,
            site_fp=WORKER_SITE_FP,
            geom_dim=WORKER_GEOM_DIM,
            diagnostics=diagnostics,
        )
        return True, rec, emb, diagnostics
    except Exception as exc:  # pragma: no cover - exploratory pipeline
        return False, rec, None, {"icsd_id": rec.icsd_id, "reason": type(exc).__name__, "detail": str(exc)[:200]}


def featurize_record_batch(records: list[Record]) -> list[tuple[bool, Record, np.ndarray | None, dict | None]]:
    return [featurize_record(rec) for rec in records]
