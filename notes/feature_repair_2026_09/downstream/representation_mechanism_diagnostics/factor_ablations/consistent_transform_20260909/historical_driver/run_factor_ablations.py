#!/usr/bin/env python3
"""One-factor ablations for the graphlet-versus-CrystalWeave cohort-occupancy reversal.

Every map is a full-record ICSD map fitted with one of the two protocols the branch
already uses, imported (not re-implemented) from the staged checkout:

* CrystalWeave family: ``icsd_graph_community_postprocess.build_weighted_graph`` and
  ``run_louvain`` (exact Euclidean mutual-16 graph, exp[-(d/sigma)^2] with sigma the
  median positive retained distance, components < 8 dropped, NetworkX Louvain at
  resolution 1 with seed 42, communities < 4 dropped). Coordinates are the production
  transform: StandardScaler then PCA-32 with random_state 42.
* Graphlet family: ``analyze_representations.actual_neighbors`` (pynndescent Euclidean
  ANN, random_state 42), ``mutual_graph`` (protocol ``historical_graphlet_temporal``) and
  ``stable_louvain_communities`` (resolution 1, seed 42), minimum community size 10 and
  no component filter, exactly as ``do_partition`` does. Coordinates are unscaled CDFs.

Basins use ``external_representation_sensitivity.community_basis`` (arithmetic centroid
and member-distance p95) and ``classify`` (nearest Euclidean centroid, distance <= its
own p95). These are full-map calibration sensitivities, not cutoff-trained validation.
No CIF, network or credential access is used; everything comes from saved matrices.
"""
from __future__ import annotations

import concurrent.futures
import csv
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

STAGE = Path(os.environ["FA_STAGE"]).resolve()
WORK = Path(os.environ.get("FA_WORK", os.environ.get("WORK", "/path/to/tacc/work")))
OUT = Path(os.environ["FA_OUT"])
COMMON_PATH = Path(os.environ["FA_COMMON_IDS"])
SMOKE = os.environ.get("FA_SMOKE", "") == "1"
for _path in (STAGE / "scripts", STAGE / "experiments/graphlet_compare",
              STAGE / "notes/feature_repair_2026_09/downstream"):
    sys.path.insert(0, str(_path))

PROD = WORK / "feature_repair_runs/full_v2_20260904/production"
DOWN = WORK / "feature_repair_runs/downstream_v2_20260905"
GRAPHLET_FEATURES = DOWN / "representations/graphlet_features"
GRAPHLET_PARTITION = DOWN / "representations/graphlet-dated-replay"
EXTERNAL_CRYSTALWEAVE = DOWN / "external"
EXTERNAL_GRAPHLET = WORK / "feature_repair_runs/external_representation_20260905T224322Z/graphlet/external"
SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")
LABELS = {"gnome": "GNoME", "mattergen": "MatterGen", "mp": "MP", "jarvis": "JARVIS", "alexandria": "Alexandria"}
CUTOFFS = (1990, 2000, 2010)

# CrystalWeave 213 = [mean(68) | max(68) | var(68)] + 9 cell scalars; each 68 = [7 chemistry | 61 OPS].
CHEMISTRY_COLUMNS = [c for base in (0, 68, 136) for c in range(base, base + 7)]
CELL_COLUMNS = list(range(204, 213))
# Graphlet 1280 = 64 channels x 20 CDF bins in REGISTRY.all order; channel k at 20k..20k+19.
STRUCTURAL_CHANNELS = [10, 31, 32, 33]
CHEMISTRY_CHANNELS = [k for k in range(64) if k not in STRUCTURAL_CHANNELS]
N_BINS = 20

ABLATIONS = {
    "parent_crystalweave": {"family": "crystalweave", "population": "production", "coordinates": "crystalweave_pca", "columns": "all",
        "description": "Parent control: raw 213-d CrystalWeave, z-score + PCA-32 (seed 42), CrystalWeave protocol."},
    "parent_graphlet": {"family": "graphlet", "population": "graphlet", "coordinates": "cdf", "channels": "all",
        "description": "Parent control: unscaled 1,280-d CrystalNN graphlet CDF, graphlet protocol."},
    "A1_cdf_plus_cell_block_balanced": {"family": "graphlet", "population": "graphlet", "coordinates": "cdf_plus_cell", "cell_scaling": "block_balanced",
        "description": "Graphlet CDF + CrystalWeave's 9 cell scalars (ICSD z-score, scaled so the 9-scalar block variance equals the 1,280-CDF block variance), graphlet protocol."},
    "A2_cdf_plus_cell_equal_component": {"family": "graphlet", "population": "graphlet", "coordinates": "cdf_plus_cell", "cell_scaling": "equal_per_component",
        "description": "Graphlet CDF + 9 cell scalars (ICSD z-score, each scaled to the mean per-component CDF variance), graphlet protocol."},
    "B_cdf_zscore_pca32_crystalweave_protocol": {"family": "crystalweave", "population": "graphlet", "coordinates": "cdf_pca",
        "description": "Graphlet CDF, ICSD z-score (zero-variance components dropped), PCA-32 (seed 42), CrystalWeave protocol."},
    "C1_crystalweave_minus_chemistry": {"family": "crystalweave", "population": "production", "coordinates": "crystalweave_pca", "columns": "drop_chemistry",
        "description": "CrystalWeave raw minus the 21 chemistry channels (192-d), z-score + PCA-32, CrystalWeave protocol."},
    "C2_crystalweave_minus_cell": {"family": "crystalweave", "population": "production", "coordinates": "crystalweave_pca", "columns": "drop_cell",
        "description": "CrystalWeave raw minus the 9 cell scalars (204-d), z-score + PCA-32, CrystalWeave protocol."},
    "C3_crystalweave_geometry_pooling_only": {"family": "crystalweave", "population": "production", "coordinates": "crystalweave_pca", "columns": "drop_chemistry_and_cell",
        "description": "CrystalWeave raw minus chemistry and cell blocks (183-d geometry pooling only), z-score + PCA-32, CrystalWeave protocol."},
    "D1_graphlet_structural_channels": {"family": "graphlet", "population": "graphlet", "coordinates": "cdf", "channels": "structural",
        "description": "Graphlet structural channels only (f2_distance, f3_cos_angle, f3_d_ij, f3_d_jk; 80-d), graphlet protocol."},
    "D2_graphlet_chemistry_channels": {"family": "graphlet", "population": "graphlet", "coordinates": "cdf", "channels": "chemistry",
        "description": "Graphlet chemistry channels only (60 channels, 1,200-d), graphlet protocol."},
}
GRAPHLET_FAMILY = [name for name, spec in ABLATIONS.items() if spec["family"] == "graphlet"]


# ----------------------------------------------------------------------------- utilities
def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def ids_sha256(values):
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def column_variance(X, chunk=8192):
    """Population (ddof=0) per-column mean and variance in float64, chunked."""
    n, d = X.shape
    total = np.zeros(d)
    total_sq = np.zeros(d)
    for start in range(0, n, chunk):
        block = np.asarray(X[start:start + chunk], dtype=np.float64)
        total += block.sum(axis=0)
        total_sq += np.einsum("ij,ij->j", block, block)
    mean = total / n
    variance = np.maximum(total_sq / n - mean * mean, 0.0)
    return mean, variance


# ----------------------------------------------------------------------------- inputs
def input_paths():
    paths = {
        "production_features": PROD / "features.npy",
        "production_pca": PROD / "features_pca.npy",
        "production_sample_assignments": PROD / "sample_assignments.csv",
        "production_partition": PROD / "graph/community_assignments.csv",
        "production_summary": PROD / "summary.json",
        "graphlet_features": GRAPHLET_FEATURES / "features.npy",
        "graphlet_ids": GRAPHLET_FEATURES / "features.ids.json",
        "graphlet_bin_edges": GRAPHLET_FEATURES / "bin_edges.json",
        "graphlet_feature_preparation": GRAPHLET_FEATURES / "feature_preparation.json",
        "graphlet_production_assignments": GRAPHLET_FEATURES / "production_assignments.csv",
        "graphlet_cohort_ids": GRAPHLET_PARTITION / "cohort.ids.json",
        "graphlet_partition": GRAPHLET_PARTITION / "graphlet/community_assignments.csv",
        "graphlet_partition_json": GRAPHLET_PARTITION / "graphlet/partition.json",
        "graphlet_neighbors": GRAPHLET_PARTITION / "graphlet/neighbors.npz",
        "common_support_ids": COMMON_PATH,
    }
    for source in SOURCES:
        paths[f"external_crystalweave_{source}_features"] = EXTERNAL_CRYSTALWEAVE / source / "features.npy"
        paths[f"external_crystalweave_{source}_ids"] = EXTERNAL_CRYSTALWEAVE / source / "feature_ids.json"
        paths[f"external_crystalweave_{source}_metadata"] = EXTERNAL_CRYSTALWEAVE / source / "feature_metadata.json"
        paths[f"external_graphlet_{source}_features"] = EXTERNAL_GRAPHLET / source / "features.npy"
        paths[f"external_graphlet_{source}_ids"] = EXTERNAL_GRAPHLET / source / "feature_ids.json"
    return paths


def assignment_rows(path):
    rows = csv_rows(path)
    ids = np.asarray([int(r["icsd_id"]) for r in rows], dtype=np.int64)
    years = np.asarray([int(float(r["year"])) if r["year"].strip() else -1 for r in rows], dtype=np.int64)
    label_key = "community" if "community" in rows[0] else "cluster"
    labels = np.asarray([int(r[label_key]) for r in rows], dtype=np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"Duplicate ICSD IDs in {path}")
    return ids, years, labels


class SyntheticStore:
    """Small deterministic stand-in for the saved matrices (FA_SMOKE=1 only)."""

    def __init__(self):
        rng = np.random.default_rng(0)
        n_prod, n_graphlet = 2600, 2200
        self.prod_ids = np.arange(1, n_prod + 1) * 3
        self.prod_years = rng.integers(1920, 2024, n_prod)
        self.prod_years[:40] = -1
        self.prod_raw = rng.normal(size=(n_prod, 213)) * np.linspace(0.2, 3, 213)
        self.prod_raw[:, 204:213] = np.abs(self.prod_raw[:, 204:213]) * 10 + 3
        self.prod_raw[:, 5] = 0.0  # a constant raw column
        self.prod_labels = rng.integers(-1, 30, n_prod)
        centers = rng.random((25, 1280))
        self.graphlet_ids = np.sort(rng.choice(self.prod_ids[40:], n_graphlet, replace=False))
        choice = rng.integers(0, 25, n_graphlet)
        raw = np.sort(np.abs(centers[choice] + rng.normal(scale=0.05, size=(n_graphlet, 1280))).reshape(n_graphlet, 64, 20), axis=2)
        raw = raw / raw[:, :, -1:]
        self.graphlet_cdf = raw.reshape(n_graphlet, 1280).astype(np.float32)
        self.graphlet_cdf[:, 19::20] = 1.0
        self.graphlet_labels = rng.integers(-1, 20, n_graphlet)
        self.ext = {}
        for source in SOURCES:
            n = {"gnome": 300, "mattergen": 40, "mp": 290, "jarvis": 280, "alexandria": 300}[source]
            keys = [f"{source}-{i}" for i in range(n)]
            raw213 = rng.normal(size=(n, 213)) * np.linspace(0.2, 3, 213)
            raw213[:, 204:213] = np.abs(raw213[:, 204:213]) * 10 + 3
            raw213[:, 5] = 0.0
            choice = rng.integers(0, 25, n)
            cdf = np.sort(np.abs(centers[choice] + rng.normal(scale=0.05, size=(n, 1280))).reshape(n, 64, 20), axis=2)
            cdf = (cdf / cdf[:, :, -1:]).reshape(n, 1280).astype(np.float32)
            cdf[:, 19::20] = 1.0
            drop_cw = set(rng.choice(n, 3, replace=False).tolist())
            drop_gr = set(rng.choice(n, 4, replace=False).tolist())
            self.ext[source] = {
                "crystalweave": ([k for i, k in enumerate(keys) if i not in drop_cw], raw213[[i for i in range(n) if i not in drop_cw]]),
                "graphlet": ([k for i, k in enumerate(keys) if i not in drop_gr], cdf[[i for i in range(n) if i not in drop_gr]]),
            }


_SYNTHETIC = None


def synthetic():
    global _SYNTHETIC
    if _SYNTHETIC is None:
        _SYNTHETIC = SyntheticStore()
    return _SYNTHETIC


def load_common_support():
    if SMOKE:
        store = synthetic()
        rng = np.random.default_rng(1)
        common = {"ICSD": sorted(map(str, rng.choice(store.graphlet_ids, 1500, replace=False).tolist()))}
        for source in SOURCES:
            keys = sorted(set(store.ext[source]["crystalweave"][0]) & set(store.ext[source]["graphlet"][0]))
            common[LABELS[source]] = sorted(rng.choice(keys, len(keys) - 5, replace=False).tolist())
        return common
    payload = json.loads(COMMON_PATH.read_text())
    common = {}
    for population, entry in payload["populations"].items():
        ids = [str(v) for v in entry["ids"]]
        if len(ids) != entry["n"] or len(set(ids)) != len(ids) or ids_sha256(ids) != entry["ids_sha256"]:
            raise ValueError(f"Common-support identifiers for {population} fail their recorded hash")
        common[population] = ids
    expected = {"ICSD": 83661, "GNoME": 5000, "MatterGen": 384, "MP": 4969, "JARVIS": 4930, "Alexandria": 4982}
    if {k: len(v) for k, v in common.items()} != expected:
        raise ValueError("Common-support population sizes differ from the five-representation table")
    return common


def load_production_population():
    """All 167,392 production successes in features.npy row order."""
    if SMOKE:
        store = synthetic()
        return {"ids": store.prod_ids, "years": store.prod_years, "saved_labels": store.prod_labels,
                "raw": store.prod_raw, "saved_pca": None}
    ids, years, hdbscan = assignment_rows(PROD / "sample_assignments.csv")
    ids2, years2, labels = assignment_rows(PROD / "graph/community_assignments.csv")
    if not np.array_equal(ids, ids2) or not np.array_equal(years, years2):
        raise ValueError("Production sample/community assignment order differs")
    summary = json.loads((PROD / "summary.json").read_text())
    if summary.get("feature_version") != "crystal-features-v2-geometric-crystalnn" or summary.get("local_mode") != "matminer_ops" or summary.get("wl_iters") != 3:
        raise ValueError("Unexpected production feature metadata")
    raw = np.load(PROD / "features.npy", mmap_mode="r", allow_pickle=False)
    saved_pca = np.load(PROD / "features_pca.npy", mmap_mode="r", allow_pickle=False)
    if raw.shape != (len(ids), 213) or saved_pca.shape != (len(ids), 32) or summary.get("sample_size_featurized") != len(ids):
        raise ValueError("Production feature/PCA dimensions disagree with assignments")
    return {"ids": ids, "years": years, "saved_labels": labels, "raw": raw, "saved_pca": saved_pca}


def load_graphlet_population(production=None):
    """The 150,247 dated production-nonnoise entries of the CrystalNN graphlet map, in the parent's row order."""
    if SMOKE:
        store = synthetic()
        production = production or load_production_population()
        prod_row = {int(i): r for r, i in enumerate(production["ids"])}
        rows = [prod_row[int(i)] for i in store.graphlet_ids]
        return {"ids": store.graphlet_ids, "years": production["years"][rows], "saved_labels": store.graphlet_labels,
                "cdf": store.graphlet_cdf, "cell": np.asarray(production["raw"][rows][:, CELL_COLUMNS], dtype=np.float64),
                "production_rows": np.asarray(rows)}
    production = production or load_production_population()
    all_ids = np.asarray(json.loads((GRAPHLET_FEATURES / "features.ids.json").read_text()), dtype=np.int64)
    matrix = np.load(GRAPHLET_FEATURES / "features.npy", mmap_mode="r", allow_pickle=False)
    preparation = json.loads((GRAPHLET_FEATURES / "feature_preparation.json").read_text())
    if matrix.shape != (len(all_ids), 1280) or not preparation.get("complete") or preparation.get("n_successful") != len(all_ids):
        raise ValueError("Graphlet feature matrix disagrees with its preparation report")
    if (preparation.get("neighbor_method") or "crystalnn") != "crystalnn":
        raise ValueError("Expected the CrystalNN graphlet matrix")
    cohort = np.asarray(json.loads((GRAPHLET_PARTITION / "cohort.ids.json").read_text()), dtype=np.int64)
    cohort_set = set(cohort.tolist())
    rows = np.asarray([r for r, i in enumerate(all_ids) if int(i) in cohort_set], dtype=np.int64)
    if not np.array_equal(all_ids[rows], cohort):
        raise ValueError("Graphlet cohort order does not follow the feature-row order")
    part_ids, part_years, part_labels = assignment_rows(GRAPHLET_PARTITION / "graphlet/community_assignments.csv")
    if not np.array_equal(part_ids, cohort):
        raise ValueError("Saved graphlet partition order differs from the cohort")
    prod_row = {int(i): r for r, i in enumerate(production["ids"])}
    production_rows = np.asarray([prod_row[int(i)] for i in cohort], dtype=np.int64)
    years = production["years"][production_rows]
    if not np.array_equal(years, part_years) or np.any(years < 1900) or np.any(years > 2025):
        raise ValueError("Graphlet cohort years disagree with production or leave the dated range")
    if np.any(production["saved_labels"][production_rows] < 0):
        raise ValueError("Graphlet cohort contains production noise")
    cdf = np.asarray(matrix[rows], dtype=np.float32)
    reshaped = cdf.reshape(-1, 64, N_BINS)
    if (np.any(reshaped < -2e-6) or np.any(reshaped > 1 + 2e-6) or np.any(np.diff(reshaped, axis=2) < -2e-6)
            or np.any(np.abs(reshaped[:, :, -1] - 1) > 2e-6)):
        raise ValueError("Graphlet rows are not normalized CDFs")
    cell = np.asarray(production["raw"][production_rows][:, CELL_COLUMNS], dtype=np.float64)
    return {"ids": cohort, "years": years, "saved_labels": part_labels, "cdf": cdf, "cell": cell, "production_rows": production_rows}


def load_external(source, family):
    if SMOKE:
        keys, matrix = synthetic().ext[source][family]
        return list(keys), np.asarray(matrix)
    root = EXTERNAL_CRYSTALWEAVE if family == "crystalweave" else EXTERNAL_GRAPHLET
    keys = [str(k) for k in json.loads((root / source / "feature_ids.json").read_text())]
    matrix = np.load(root / source / "features.npy", allow_pickle=False)
    width = 213 if family == "crystalweave" else 1280
    if matrix.shape != (len(keys), width) or len(set(keys)) != len(keys) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid {family} external matrix for {source}")
    if family == "crystalweave":
        meta = json.loads((root / source / "feature_metadata.json").read_text())
        if meta.get("feature_version") != "crystal-features-v2-geometric-crystalnn" or meta.get("n_features") != 213 or meta.get("n_rows") != len(keys):
            raise ValueError(f"Unexpected CrystalWeave external metadata for {source}")
    else:
        reshaped = matrix.reshape(-1, 64, N_BINS)
        if (np.any(reshaped < -2e-6) or np.any(reshaped > 1 + 2e-6) or np.any(np.diff(reshaped, axis=2) < -2e-6)
                or np.any(np.abs(reshaped[:, :, -1] - 1) > 2e-6)):
            raise ValueError(f"External graphlet rows for {source} are not normalized CDFs")
    return keys, matrix


def check_graphlet_registry():
    if SMOKE:
        return {"checked": False}
    import graphlet_features as gf
    names = gf.REGISTRY.all
    expected = {10: "f2_distance", 31: "f3_cos_angle", 32: "f3_d_ij", 33: "f3_d_jk"}
    if len(names) != 64 or gf.NUM_BINS != N_BINS or any(names[i] != n for i, n in expected.items()):
        raise ValueError("Graphlet channel registry differs from the assumed layout")
    if not all(n.startswith("f1_") for n in names[:10]) or sum(n.startswith("f2_") for n in names) != 21 or sum(n.startswith("f3_") for n in names) != 33:
        raise ValueError("Graphlet registry order counts differ from 10/21/33")
    return {"checked": True, "structural_channels": {i: names[i] for i in STRUCTURAL_CHANNELS},
            "n_chemistry_channels": len(CHEMISTRY_CHANNELS), "registry": names}


# ----------------------------------------------------------------------------- coordinates
def graphlet_columns(channels):
    keys = {"all": list(range(64)), "structural": STRUCTURAL_CHANNELS, "chemistry": CHEMISTRY_CHANNELS}[channels]
    return [k * N_BINS + b for k in keys for b in range(N_BINS)]


def crystalweave_columns(columns):
    drop = set()
    if columns in ("drop_chemistry", "drop_chemistry_and_cell"):
        drop |= set(CHEMISTRY_COLUMNS)
    if columns in ("drop_cell", "drop_chemistry_and_cell"):
        drop |= set(CELL_COLUMNS)
    return [c for c in range(213) if c not in drop]


def build_coordinates(name, production, graphlet, externals):
    """ICSD coordinates plus identically transformed externals for one ablation.

    Returns (icsd_coordinates, {source: (keys, coordinates)}, transform_record).
    """
    spec = ABLATIONS[name]
    record = {"ablation": name, **{k: v for k, v in spec.items() if k != "description"}}
    if spec["coordinates"] in ("crystalweave_pca", "cdf_pca"):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        if spec["coordinates"] == "crystalweave_pca":
            columns = crystalweave_columns(spec["columns"])
            icsd_raw = np.asarray(production["raw"][:, columns], dtype=np.float64)
            ext_family = "crystalweave"
        else:
            _, variance = column_variance(graphlet["cdf"])
            columns = [c for c in range(1280) if variance[c] > 0]
            record["n_zero_variance_components_dropped"] = 1280 - len(columns)
            icsd_raw = np.asarray(graphlet["cdf"][:, columns], dtype=np.float64)
            ext_family = "graphlet"
        record["n_input_components"] = len(columns)
        record["input_columns_sha256"] = ids_sha256(columns)
        scaler = StandardScaler()
        standardized = scaler.fit_transform(icsd_raw)
        pca = PCA(n_components=32, random_state=42)
        icsd = pca.fit_transform(standardized)
        record.update(n_zero_scale_columns=int(np.sum(scaler.var_ == 0)), pca_solver=pca._fit_svd_solver,
                      explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
                      pca_fit_transform_vs_transform_max_abs_error=float(np.max(np.abs(pca.transform(standardized) - icsd))))
        if name == "parent_crystalweave" and production["saved_pca"] is not None:
            saved = np.asarray(production["saved_pca"], dtype=np.float64)
            record["saved_pca_reconstruction_max_abs_error"] = float(np.max(np.abs(saved - icsd)))
            record["saved_pca_reconstruction_allclose"] = bool(np.allclose(saved, icsd, rtol=1e-7, atol=1e-8))
            if not record["saved_pca_reconstruction_allclose"]:
                raise ValueError("Reconstructed production PCA differs from the saved map")
        transform = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "pca_mean": pca.mean_, "pca_components": pca.components_, "columns": np.asarray(columns)}
        ext = {}
        for source, (keys, matrix) in externals[ext_family].items():
            ext[source] = (keys, pca.transform(scaler.transform(np.asarray(matrix[:, columns], dtype=np.float64))))
        del standardized
        return icsd, ext, record, transform
    if spec["coordinates"] == "cdf":
        columns = graphlet_columns(spec["channels"])
        record["n_input_components"] = len(columns)
        icsd = np.ascontiguousarray(graphlet["cdf"][:, columns], dtype=np.float32)
        ext = {source: (keys, np.ascontiguousarray(matrix[:, columns], dtype=np.float32)) for source, (keys, matrix) in externals["graphlet"].items()}
        return icsd, ext, record, {"columns": np.asarray(columns)}
    if spec["coordinates"] == "cdf_plus_cell":
        _, cdf_variance = column_variance(graphlet["cdf"])
        block_variance = float(cdf_variance.sum())
        cell_mean, cell_variance = column_variance(graphlet["cell"])
        cell_scale = np.sqrt(cell_variance)
        if np.any(cell_scale <= 0):
            raise ValueError("A cell scalar has zero variance on the graphlet population")
        factor = np.sqrt(block_variance / 9) if spec["cell_scaling"] == "block_balanced" else np.sqrt(block_variance / 1280)
        record.update(cdf_block_total_variance=block_variance, cdf_mean_component_variance=block_variance / 1280,
                      cell_scale_factor=float(factor), cell_block_total_variance_after_scaling=float(9 * factor * factor),
                      n_input_components=1289)
        def combine(cdf, cell):
            scaled = ((np.asarray(cell, dtype=np.float64) - cell_mean) / cell_scale) * factor
            return np.ascontiguousarray(np.hstack([np.asarray(cdf, dtype=np.float32), scaled.astype(np.float32)]), dtype=np.float32)
        icsd = combine(graphlet["cdf"], graphlet["cell"])
        ext = {}
        for source in SOURCES:
            g_keys, g_matrix = externals["graphlet"][source]
            c_keys, c_matrix = externals["crystalweave"][source]
            c_index = {k: i for i, k in enumerate(c_keys)}
            keys = [k for k in g_keys if k in c_index]
            g_rows = [i for i, k in enumerate(g_keys) if k in c_index]
            c_rows = [c_index[k] for k in keys]
            ext[source] = (keys, combine(g_matrix[g_rows], c_matrix[c_rows][:, CELL_COLUMNS]))
        return icsd, ext, record, {"cell_mean": cell_mean, "cell_scale": cell_scale, "cell_factor": np.asarray([factor])}
    raise ValueError(name)


# ----------------------------------------------------------------------------- map fitting
def fit_crystalweave_map(X):
    from icsd_graph_community_postprocess import build_weighted_graph, run_louvain
    import networkx as nx
    started = time.time()
    graph, base = build_weighted_graph(np.asarray(X, dtype=np.float64), 16, True, 8)
    graph_seconds = time.time() - started
    labels = run_louvain(graph, len(X), 1.0, 4, base)
    stats = {"protocol": "crystalweave", "knn_k": 16, "metric": "euclidean", "neighbor_search": "sklearn NearestNeighbors exact (algorithm=auto)",
             "mutual_knn": True, "weight": "exp(-(distance/sigma)^2), sigma = median positive retained distance", "min_component_size": 8,
             "louvain": f"networkx {nx.__version__} louvain_communities, resolution 1.0, seed 42", "min_community_size": 4,
             "n_nodes_after_component_filter": int(graph.number_of_nodes()), "n_edges_after_component_filter": int(graph.number_of_edges()),
             "graph_seconds": graph_seconds, "louvain_seconds": time.time() - started - graph_seconds}
    return labels, stats


def fit_graphlet_map(indices, distances, n):
    import analyze_representations as ar
    started = time.time()
    graph, sigma = ar.mutual_graph(indices, distances, "historical_graphlet_temporal")
    graph_seconds = time.time() - started
    if graph.number_of_edges():
        communities, diagnostics = ar.stable_louvain_communities(graph, weight="weight", resolution=1.0, seed=42)
    else:
        communities, diagnostics = [{i} for i in range(n)], {"levels": 0, "termination": "empty_graph"}
    labels = np.full(n, -1, dtype=np.int64)
    next_label = 0
    for members in communities:
        if len(members) >= 10:
            labels[list(members)] = next_label
            next_label += 1
    stats = {"protocol": "graphlet", "knn_k": 16, "metric": "euclidean", "neighbor_search": "pynndescent NNDescent (random_state 42, low_memory) via analyze_representations.actual_neighbors",
             "mutual_knn": True, "weight": "exp(-(distance/sigma)^2), sigma = median positive retained distance", "sigma": float(sigma),
             "min_component_size": None, "louvain": "analyze_representations.stable_louvain_communities, resolution 1.0, seed 42",
             "louvain_node_move_gain_tolerance": ar.LOUVAIN_NODE_MOVE_GAIN_TOLERANCE, "min_community_size": 10,
             "n_edges": int(graph.number_of_edges()), "louvain_levels": diagnostics.get("levels"), "louvain_termination": diagnostics.get("termination"),
             "graph_seconds": graph_seconds, "louvain_seconds": time.time() - started - graph_seconds}
    return labels, stats


def compare_partitions(reference, alternate):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    reference = np.asarray(reference)
    alternate = np.asarray(alternate)
    both = (reference >= 0) & (alternate >= 0)
    pairs = set(zip(reference[both].tolist(), alternate[both].tolist()))
    one_to_one = len(pairs) == len(set(reference[both].tolist())) == len(set(alternate[both].tolist()))
    return {"n": int(len(reference)), "ARI_all_entries_noise_as_label": float(adjusted_rand_score(reference, alternate)),
            "NMI_all_entries": float(normalized_mutual_info_score(reference, alternate)),
            "noise_agreement": bool(np.array_equal(reference < 0, alternate < 0)), "n_reference_noise": int((reference < 0).sum()),
            "n_alternate_noise": int((alternate < 0).sum()), "identical_up_to_relabeling": bool(one_to_one and np.array_equal(reference < 0, alternate < 0)),
            "n_reference_communities": int(len(set(reference[reference >= 0].tolist()))), "n_alternate_communities": int(len(set(alternate[alternate >= 0].tolist())))}


# ----------------------------------------------------------------------------- evaluation
def evaluate(name, icsd_coords, ext_coords, labels, population, common):
    from external_representation_sensitivity import community_basis, classify, rate
    from scipy.spatial.distance import cdist
    basis = community_basis(icsd_coords, labels, np.ones(len(labels), dtype=bool))
    nearest, distance, inside = classify(icsd_coords, basis)
    icsd_keys = [str(int(i)) for i in population["ids"]]
    results = {"icsd": {"keys": icsd_keys, "nearest": nearest, "distance": distance, "inside": inside}}
    for source, (keys, coords) in ext_coords.items():
        n, d, i = classify(coords, basis)
        results[source] = {"keys": list(keys), "nearest": n, "distance": d, "inside": i}
    # Rates on own support and on the exact five-representation common support.
    rates = {}
    common_masks = {}
    for population_name, entry in results.items():
        label = "ICSD" if population_name == "icsd" else LABELS[population_name]
        common_set = set(common[label])
        mask = np.asarray([k in common_set for k in entry["keys"]], dtype=bool)
        if int(mask.sum()) != len(common_set):
            raise ValueError(f"{name}: common support for {label} is not contained in the evaluated population ({int(mask.sum())} of {len(common_set)})")
        common_masks[population_name] = mask
        rates[population_name] = {"own": rate(entry["inside"]), "common": rate(entry["inside"][mask])}
    for population_name in rates:
        for support in ("own", "common"):
            rates[population_name][support]["icsd_minus_source_pp"] = None if population_name == "icsd" else 100 * (
                rates["icsd"][support]["in_basin_fraction"] - rates[population_name][support]["in_basin_fraction"])
    # Independent SciPy checks on fixed samples, plus exhaustive flag reconstruction.
    rng = np.random.default_rng(42)
    verification = {"flags_equal_distance_le_p95_all_rows": True, "samples": {}}
    for population_name, entry in results.items():
        if not np.array_equal(entry["inside"], entry["distance"] <= basis["p95"][entry["nearest"]]):
            verification["flags_equal_distance_le_p95_all_rows"] = False
        count = min(256 if population_name == "icsd" else 128, len(entry["keys"]))
        sample = np.sort(rng.choice(len(entry["keys"]), count, replace=False))
        coords = icsd_coords if population_name == "icsd" else ext_coords[population_name][1]
        oracle = cdist(np.asarray(coords[sample], dtype=np.float64), basis["centroids"], metric="euclidean")
        oracle_min = oracle.min(axis=1)
        chosen = oracle[np.arange(count), entry["nearest"][sample]]
        verification["samples"][population_name] = {
            "n_checked": int(count),
            "nearest_distance_matches_scipy_minimum": bool(np.allclose(chosen, oracle_min, rtol=1e-9, atol=1e-9)),
            "argmin_agreement_fraction": float(np.mean(oracle.argmin(axis=1) == entry["nearest"][sample])),
            "distance_max_abs_error_vs_scipy": float(np.max(np.abs(entry["distance"][sample] - chosen)))}
        if not verification["samples"][population_name]["nearest_distance_matches_scipy_minimum"]:
            raise ValueError(f"{name}: SciPy nearest-centroid check failed for {population_name}")
        if not np.allclose(entry["distance"][sample], chosen, rtol=1e-9, atol=1e-9):
            raise ValueError(f"{name}: SciPy distance check failed for {population_name}")
    if not verification["flags_equal_distance_le_p95_all_rows"]:
        raise ValueError(f"{name}: saved flags do not reproduce distance <= p95")
    return basis, results, rates, common_masks, verification


def write_map_outputs(name, out, population, labels, basis, results, transform):
    write_csv(out / "community_assignments.csv.gz",
              [{"icsd_id": int(i), "year": (int(y) if y >= 0 else ""), "community": int(c)} for i, y, c in zip(population["ids"], population["years"], labels)],
              ["icsd_id", "year", "community"])
    icsd = results["icsd"]
    write_csv(out / "icsd_full.csv.gz", [{"record_key": k, "year": (int(y) if y >= 0 else ""), "map_community": int(c),
                                          "assigned_community": int(basis["communities"][j]), "nearest_centroid_distance": float(d),
                                          "community_threshold_p95": float(basis["p95"][j]), "in_basin": bool(f)}
                                         for k, y, c, j, d, f in zip(icsd["keys"], population["years"], labels, icsd["nearest"], icsd["distance"], icsd["inside"])],
              ["record_key", "year", "map_community", "assigned_community", "nearest_centroid_distance", "community_threshold_p95", "in_basin"])
    for source in SOURCES:
        entry = results[source]
        (out / "external" / source).mkdir(parents=True, exist_ok=True)
        write_csv(out / "external" / source / "projection_full.csv",
                  [{"record_key": k, "assigned_community": int(basis["communities"][j]), "nearest_centroid_distance": float(d),
                    "community_threshold_p95": float(basis["p95"][j]), "in_basin": bool(f)}
                   for k, j, d, f in zip(entry["keys"], entry["nearest"], entry["distance"], entry["inside"])],
                  ["record_key", "assigned_community", "nearest_centroid_distance", "community_threshold_p95", "in_basin"])
    np.savez_compressed(out / "basis.npz", **basis, **{f"transform_{k}": v for k, v in transform.items()})
    projections = {"icsd_nearest": icsd["nearest"], "icsd_distance": icsd["distance"], "icsd_inside": icsd["inside"],
                   "icsd_year": population["years"], "icsd_label": labels, "icsd_ids": population["ids"], "p95": basis["p95"], "counts": basis["counts"]}
    for source in SOURCES:
        projections[f"{source}_nearest"] = results[source]["nearest"]
        projections[f"{source}_distance"] = results[source]["distance"]
        projections[f"{source}_inside"] = results[source]["inside"]
        projections[f"{source}_keys"] = np.asarray(results[source]["keys"])
    np.savez_compressed(out / "projections.npz", **projections)


def run_map(name):
    """Worker: build coordinates, fit the map, classify, verify and write one ablation."""
    started = time.time()
    out = OUT / "maps" / name
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    try:
        with log_path.open("a") as handle:
            handle.write(f"start {time.strftime('%Y-%m-%dT%H:%M:%S')} pid {os.getpid()}\n")
        spec = ABLATIONS[name]
        production = load_production_population()
        graphlet = load_graphlet_population(production) if spec["population"] == "graphlet" or spec["coordinates"] != "crystalweave_pca" else None
        population = production if spec["population"] == "production" else graphlet
        common = load_common_support()
        externals = {"crystalweave": {s: load_external(s, "crystalweave") for s in SOURCES},
                     "graphlet": {s: load_external(s, "graphlet") for s in SOURCES}}
        icsd_coords, ext_coords, transform_record, transform = build_coordinates(name, production, graphlet, externals)
        coordinate_seconds = time.time() - started
        if spec["family"] == "crystalweave":
            labels, fit_stats = fit_crystalweave_map(icsd_coords)
        else:
            with np.load(out / "neighbors.npz", allow_pickle=False) as saved:
                if str(saved["coordinates_sha256"]) != sha256_array(icsd_coords):
                    raise ValueError("Worker coordinates differ from the neighbor-search coordinates")
                indices, distances = saved["indices"], saved["distances"]
            labels, fit_stats = fit_graphlet_map(indices, distances, len(icsd_coords))
        fit_seconds = time.time() - started - coordinate_seconds
        comparison = {"saved_parent_partition": compare_partitions(population["saved_labels"], labels)} if name.startswith("parent_") else {}
        basis, results, rates, common_masks, verification = evaluate(name, icsd_coords, ext_coords, labels, population, common)
        write_map_outputs(name, out, population, labels, basis, results, transform)
        n_noise = int((labels < 0).sum())
        summary = {
            "ablation": name, "description": spec["description"], "family": spec["family"], "population": spec["population"],
            "transform": transform_record, "fit": fit_stats,
            "map": {"n_fit_rows": int(len(labels)), "n_dimensions": int(icsd_coords.shape[1]), "n_communities": int(len(basis["communities"])),
                    "n_noise": n_noise, "noise_share": n_noise / len(labels), "coordinate_dtype": str(icsd_coords.dtype),
                    "largest_community": int(basis["counts"].max()), "median_p95_radius": float(np.median(basis["p95"]))},
            "external_support": {s: {"n_evaluated": len(results[s]["keys"]), "n_common": int(common_masks[s].sum())} for s in SOURCES},
            "rates": rates, "verification": verification, **comparison,
            "seconds": {"coordinates": coordinate_seconds, "fit": fit_seconds, "total": time.time() - started},
        }
        dump(out / "map_summary.json", summary)
        with log_path.open("a") as handle:
            handle.write(f"done {time.strftime('%Y-%m-%dT%H:%M:%S')} {summary['seconds']}\n")
        return name, summary, None
    except Exception:
        text = traceback.format_exc()
        with log_path.open("a") as handle:
            handle.write(text)
        return name, None, text


# ----------------------------------------------------------------------------- nearest-cell recalibration
def nearest_cell_rates(summaries, common):
    """Nonnoise nearest-centroid-cell p95 recalibration, mirroring nearest_cell_calibration/nonnoise_calibration.

    Calibration members are ICSD fit rows with a nonnoise map label (full map) or additionally
    dated <= T (T variants); they are grouped by the nearest full-map centroid and their saved
    distances define the cell p95 (numpy linear quantile). Cells without calibration members
    are frontier. Centers, partitions, coordinates and distances are unchanged.
    """
    from external_representation_sensitivity import rate
    rows = []
    for name in ABLATIONS:
        with np.load(OUT / "maps" / name / "projections.npz", allow_pickle=False) as saved:
            p = {k: saved[k] for k in saved.files}
        n_cells = len(p["p95"])
        icsd_keys = np.asarray([str(int(i)) for i in p["icsd_ids"]])
        icsd_common = np.isin(icsd_keys, np.asarray(common["ICSD"]))
        for variant, cutoff in [("full_map_nonnoise", None)] + [(f"T{c}_nonnoise", c) for c in CUTOFFS]:
            calibration = p["icsd_label"] >= 0
            evaluation = np.ones(len(icsd_keys), dtype=bool)
            if cutoff is not None:
                calibration &= (p["icsd_year"] >= 0) & (p["icsd_year"] <= cutoff)
                evaluation = p["icsd_year"] > cutoff
            cell_p95 = np.full(n_cells, np.nan)
            members = np.bincount(p["icsd_nearest"][calibration], minlength=n_cells)
            for cell in np.flatnonzero(members > 0):
                cell_p95[cell] = np.quantile(p["icsd_distance"][calibration & (p["icsd_nearest"] == cell)], 0.95, method="linear")
            def flags(nearest, distance):
                present = members[nearest] > 0
                return present & (distance <= np.where(present, cell_p95[nearest], np.inf))
            icsd_flags = flags(p["icsd_nearest"], p["icsd_distance"])
            base = {"ablation": name, "variant": variant, "n_calibration": int(calibration.sum()), "n_calibrated_cells": int((members > 0).sum()),
                    "n_absent_cells": int((members == 0).sum())}
            icsd_rates = {"own": rate(icsd_flags[evaluation]), "common": rate(icsd_flags[evaluation & icsd_common])}
            for support, r in icsd_rates.items():
                rows.append({**base, "population": "ICSD", "support": support, "n": r["n"], "n_in_basin": r["n_in_basin"], "rate": r["in_basin_fraction"], "icsd_minus_source_pp": ""})
            for source in SOURCES:
                ext_flags = flags(p[f"{source}_nearest"], p[f"{source}_distance"])
                ext_common = np.isin(p[f"{source}_keys"].astype(str), np.asarray(common[LABELS[source]]))
                for support, mask in (("own", np.ones(len(ext_flags), dtype=bool)), ("common", ext_common)):
                    r = rate(ext_flags[mask])
                    rows.append({**base, "population": LABELS[source], "support": support, "n": r["n"], "n_in_basin": r["n_in_basin"], "rate": r["in_basin_fraction"],
                                 "icsd_minus_source_pp": 100 * (icsd_rates[support]["in_basin_fraction"] - r["in_basin_fraction"])})
    return rows


# ----------------------------------------------------------------------------- reporting
def percent(value):
    return f"{100 * value:.2f}%"


def signed(value):
    return f"{value:+.2f}"


def write_report(summaries, nearest_rows, provenance):
    populations = ["ICSD"] + [LABELS[s] for s in SOURCES]
    rows = []
    for name, summary in summaries.items():
        for population_name, entry in summary["rates"].items():
            label = "ICSD" if population_name == "icsd" else LABELS[population_name]
            for support in ("own", "common"):
                r = entry[support]
                rows.append({"ablation": name, "population": label, "n": r["n"], "n_in_basin": r["n_in_basin"], "rate": r["in_basin_fraction"],
                             "icsd_minus_source_pp": "" if r["icsd_minus_source_pp"] is None else r["icsd_minus_source_pp"], "support": support})
    write_csv(OUT / "factor_ablations.csv", rows, ["ablation", "population", "n", "n_in_basin", "rate", "icsd_minus_source_pp", "support"])
    stats_rows = []
    for name, summary in summaries.items():
        m, f = summary["map"], summary["fit"]
        stats_rows.append({"ablation": name, "family": summary["family"], "population": summary["population"], "n_fit_rows": m["n_fit_rows"], "n_dimensions": m["n_dimensions"],
                           "n_communities": m["n_communities"], "n_noise": m["n_noise"], "noise_share": m["noise_share"], "largest_community": m["largest_community"],
                           "median_p95_radius": m["median_p95_radius"], "sigma": f.get("sigma", ""), "n_edges": f.get("n_edges", f.get("n_edges_after_component_filter", "")),
                           "n_nodes_after_component_filter": f.get("n_nodes_after_component_filter", ""), "fit_seconds": summary["seconds"]["fit"]})
    write_csv(OUT / "map_statistics.csv", stats_rows)
    write_csv(OUT / "nearest_cell_recalibration.csv", nearest_rows, ["ablation", "variant", "population", "support", "n", "n_in_basin", "rate", "icsd_minus_source_pp", "n_calibration", "n_calibrated_cells", "n_absent_cells"])
    by = {(r["ablation"], r["population"], r["support"]): r for r in rows}
    nearest_by = {(r["ablation"], r["variant"], r["population"], r["support"]): r for r in nearest_rows}
    lines = ["# One-factor ablations: graphlet versus CrystalWeave design choices", "",
             "Full-map member-p95 in-basin percentages on the exact five-representation common support "
             "(83,661 ICSD / 5,000 GNoME / 384 MatterGen / 4,969 MP / 4,930 JARVIS / 4,982 Alexandria). "
             "Each map is fitted on its parent family's ICSD population and keeps its own partition, centroids and member-p95 radii. "
             "These are full-map calibration sensitivities, not cutoff-trained validation.", "",
             "## Table 1. In-basin rates on the common support (member p95)", "",
             "| Ablation | " + " | ".join(populations) + " | ICSD - GNoME (pp) |", "|---|" + "---:|" * (len(populations) + 1)]
    for name in ABLATIONS:
        cells = [percent(by[(name, p, "common")]["rate"]) for p in populations]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {signed(by[(name, 'GNoME', 'common')]['icsd_minus_source_pp'])} |")
    lines += ["", "## Table 2. Nonnoise nearest-cell recalibration, full map (common support)", "",
              "Radii recalibrated as the p95 of nonnoise fit-row distances within each nearest-centroid cell; centers, partitions and distances unchanged.", "",
              "| Ablation | " + " | ".join(populations) + " | ICSD - GNoME (pp) |", "|---|" + "---:|" * (len(populations) + 1)]
    for name in ABLATIONS:
        cells = [percent(nearest_by[(name, "full_map_nonnoise", p, "common")]["rate"]) for p in populations]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {signed(nearest_by[(name, 'full_map_nonnoise', 'GNoME', 'common')]['icsd_minus_source_pp'])} |")
    lines += ["", "## Table 3. Map statistics", "",
              "| Ablation | Family protocol | Fit rows | Dimensions | Communities | Noise share | Edges | ICSD own-support rate | GNoME own-support rate |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        m = summary["map"]
        edges = summary["fit"].get("n_edges", summary["fit"].get("n_edges_after_component_filter"))
        lines.append(f"| {name} | {summary['family']} | {m['n_fit_rows']:,} | {m['n_dimensions']} | {m['n_communities']:,} | {percent(m['noise_share'])} | {edges:,} | "
                     f"{percent(by[(name, 'ICSD', 'own')]['rate'])} | {percent(by[(name, 'GNoME', 'own')]['rate'])} |")
    lines += ["", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by run_factor_ablations.py (job {provenance.get('slurm_job_id')}).", ""]
    (OUT / "factor_ablations.md").write_text("\n".join(lines))
    return rows


# ----------------------------------------------------------------------------- main
def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(exist_ok=True)
    ann_jobs = int(os.environ.get("FA_ANN_JOBS", "16"))
    map_workers = int(os.environ.get("FA_MAP_WORKERS", "10"))
    only = [n for n in os.environ.get("FA_ONLY", "").split(",") if n]
    names = [n for n in ABLATIONS if not only or n in only]
    provenance = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": os.uname().nodename, "smoke": SMOKE, "ann_jobs": ann_jobs, "map_workers": map_workers,
                  "stage": str(STAGE), "packages": {}}
    for package in ("numpy", "scipy", "scikit-learn", "pynndescent", "networkx", "numba"):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    provenance["staged_code_sha256"] = {str(p.relative_to(STAGE)): sha256_file(p) for p in sorted(STAGE.rglob("*.py"))}
    provenance["driver_sha256"] = sha256_file(__file__)
    if not SMOKE:
        log("Hashing inputs")
        provenance["input_sha256"] = {k: {"path": str(v), "sha256": sha256_file(v), "bytes": v.stat().st_size} for k, v in input_paths().items()}
    provenance["graphlet_registry"] = check_graphlet_registry()
    provenance["column_layout"] = {"crystalweave_chemistry_columns": CHEMISTRY_COLUMNS, "crystalweave_cell_columns": CELL_COLUMNS,
                                   "graphlet_structural_channels": STRUCTURAL_CHANNELS, "graphlet_chemistry_channel_count": len(CHEMISTRY_CHANNELS)}
    dump(OUT / "provenance.json", provenance)

    log("Loading populations")
    production = load_production_population()
    graphlet = load_graphlet_population(production)
    externals = {"crystalweave": {s: load_external(s, "crystalweave") for s in SOURCES}, "graphlet": {s: load_external(s, "graphlet") for s in SOURCES}}
    common = load_common_support()
    provenance["populations"] = {"production": {"n": int(len(production["ids"])), "ids_sha256": ids_sha256(production["ids"].tolist()), "n_undated": int((production["years"] < 0).sum())},
                                 "graphlet": {"n": int(len(graphlet["ids"])), "ids_sha256": ids_sha256(graphlet["ids"].tolist())},
                                 "external": {s: {f: len(externals[f][s][0]) for f in ("crystalweave", "graphlet")} for s in SOURCES},
                                 "common": {k: len(v) for k, v in common.items()}}
    dump(OUT / "provenance.json", provenance)

    # Stage 1: graphlet-family neighbor searches, sequential with the full core count (the parent used
    # SLURM_CPUS_PER_TASK workers), so each map sees the same estimator configuration.
    import analyze_representations as ar
    for name in names:
        if ABLATIONS[name]["family"] != "graphlet":
            continue
        out = OUT / "maps" / name
        out.mkdir(parents=True, exist_ok=True)
        if (out / "neighbors.npz").exists():
            log(f"neighbors exist for {name}; skipping search")
            continue
        log(f"Neighbor search {name}")
        t0 = time.time()
        icsd_coords, _, record, _ = build_coordinates(name, production, graphlet, externals)
        indices, distances = ar.actual_neighbors(icsd_coords, 16, "euclidean", True, ann_jobs)
        extra = {}
        if name == "parent_graphlet" and not SMOKE:
            with np.load(GRAPHLET_PARTITION / "graphlet/neighbors.npz", allow_pickle=False) as saved:
                same_rows = np.mean([set(a.tolist()) == set(b.tolist()) for a, b in zip(indices, saved["indices"])])
                extra = {"saved_parent_neighbor_set_agreement_fraction": float(same_rows),
                         "saved_parent_neighbor_distance_max_abs_error_on_agreeing_rows": float(np.max(np.abs(np.sort(distances, axis=1) - np.sort(saved["distances"], axis=1))[[set(a.tolist()) == set(b.tolist()) for a, b in zip(indices, saved["indices"])]])) if same_rows > 0 else None}
        np.savez_compressed(out / "neighbors.npz", indices=indices, distances=distances, coordinates_sha256=np.array(sha256_array(icsd_coords)),
                            n_jobs=np.array(ann_jobs), seconds=np.array(time.time() - t0))
        dump(out / "neighbor_search.json", {"ablation": name, "n_rows": int(len(icsd_coords)), "n_dimensions": int(icsd_coords.shape[1]), "k": 16, "n_jobs": ann_jobs,
                                            "seconds": time.time() - t0, "coordinates_sha256": sha256_array(icsd_coords), **extra, "transform": record})
        log(f"  {name}: {time.time() - t0:.1f} s")
        del icsd_coords, indices, distances
    del production, graphlet, externals

    # Stage 2: fit, classify, verify and write every map in parallel worker processes.
    log(f"Fitting {len(names)} maps with {map_workers} workers")
    summaries, failures = {}, {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=map_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for name, summary, error in pool.map(run_map, names):
            if error:
                failures[name] = error
                log(f"FAILED {name}\n{error}")
            else:
                summaries[name] = summary
                r = summary["rates"]
                log(f"done {name}: communities {summary['map']['n_communities']}, noise {summary['map']['noise_share']:.4f}, "
                    f"ICSD common {100 * r['icsd']['common']['in_basin_fraction']:.2f}%, GNoME common {100 * r['gnome']['common']['in_basin_fraction']:.2f}% "
                    f"({summary['seconds']['total']:.0f} s)")
    if failures:
        dump(OUT / "failures.json", failures)
        raise SystemExit(f"{len(failures)} map(s) failed: {sorted(failures)}")
    summaries = {name: summaries[name] for name in ABLATIONS if name in summaries}
    log("Nearest-cell recalibration")
    nearest_rows = nearest_cell_rates({n: summaries[n] for n in summaries}, common) if not only else []
    rows = write_report(summaries, nearest_rows, provenance)
    checks = {
        "parent_crystalweave_common_rates": {"ICSD": summaries["parent_crystalweave"]["rates"]["icsd"]["common"]["in_basin_fraction"], "GNoME": summaries["parent_crystalweave"]["rates"]["gnome"]["common"]["in_basin_fraction"]},
        "parent_graphlet_common_rates": {"ICSD": summaries["parent_graphlet"]["rates"]["icsd"]["common"]["in_basin_fraction"], "GNoME": summaries["parent_graphlet"]["rates"]["gnome"]["common"]["in_basin_fraction"]},
        "expected_saved_five_way": {"parent_crystalweave": {"ICSD": 0.7997872365857448, "GNoME": 0.6204}, "parent_graphlet": {"ICSD": 0.7350856432507381, "GNoME": 0.841}},
    } if not only else {}
    dump(OUT / "summary.json", {"status": "complete", "runtime_seconds": time.time() - started, "provenance": provenance, "parent_checks": checks,
                                "maps": summaries, "n_rate_rows": len(rows), "n_nearest_cell_rows": len(nearest_rows)})
    log(f"Complete in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
