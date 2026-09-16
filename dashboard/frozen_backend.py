"""Read-only, version-bound dashboard projection from the saved production basis."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS
from icsd_densify_worker import build_structure_embedding

REQUIRED = ("basis", "pca", "sample_assignments", "assignments", "node_events",
            "profiles", "families", "representatives", "layout", "accessibility",
            "figure_graph_time_ratios", "figure_prototype_collapse", "figure_stepping_stone",
            "figure_gnome_frontier", "figure_graph_growth_gif")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_bundle(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Dashboard manifest has an incompatible feature version")
    paths = {}
    if not set(REQUIRED).issubset(manifest["artifacts"]):
        raise ValueError("Dashboard manifest is missing a required artifact")
    for role in manifest["artifacts"]:
        record = manifest["artifacts"][role]
        path = (manifest_path.parent / record["path"]).resolve()
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Dashboard artifact hash mismatch: {role}")
        paths[role] = path
    with np.load(paths["basis"], allow_pickle=False) as archive:
        basis = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(basis.pop("metadata")))
    if (metadata.get("feature_version") != FEATURE_VERSION
            or metadata.get("neighbor_settings") != NEIGHBOR_SETTINGS
            or metadata.get("local_mode") != "matminer_ops" or metadata.get("wl_iters") != 3):
        raise ValueError("Saved basis does not match the production encoder/settings")
    import icsd_densify_worker
    scripts = Path(icsd_densify_worker.__file__).parent
    for name, expected in metadata["worker_sha256"].items():
        if sha256(scripts / name) != expected:
            raise ValueError(f"Encoder source differs from saved basis: {name}")
    for role, source in (("pca", "icsd_pca"), ("sample_assignments", "sample_assignments"),
                         ("assignments", "community_assignments")):
        if manifest["artifacts"][role]["sha256"] != metadata["reference_files"][source]["sha256"]:
            raise ValueError(f"Artifact does not belong to this basis: {role}")
    rows = csv_rows(paths["assignments"])
    samples = csv_rows(paths["sample_assignments"])
    identities = lambda records: [(r["icsd_id"], r["year"]) for r in records]
    if identities(rows) != identities(samples) or len({r["icsd_id"] for r in rows}) != len(rows):
        raise ValueError("PCA/assignment ID order is not the production order")
    pca = np.load(paths["pca"], mmap_mode="r", allow_pickle=False)
    if pca.shape != (len(rows), 32) or len(rows) != metadata["n_reference_rows"]:
        raise ValueError("Saved PCA row count/dimension differs from the basis")
    if not np.isfinite(pca).all() or any(not np.isfinite(v).all() for v in basis.values()):
        raise ValueError("Saved projection contains nonfinite values")
    if (basis["pca_components"].shape != (32, 213)
            or basis["scaler_scale"].shape != (213,) or (basis["scaler_scale"] <= 0).any()):
        raise ValueError("Saved projection dimensions/scales are invalid")
    labels = np.array([int(r["community"]) for r in rows])
    communities, counts = np.unique(labels[labels >= 0], return_counts=True)
    if not np.array_equal(communities, basis["communities"]) or not np.array_equal(counts, basis["counts"]):
        raise ValueError("Partition membership differs from the saved community basis")
    profiles = json.loads(paths["profiles"].read_text())
    if set(profiles) != {str(c) for c in communities} or any(profiles[str(c)]["size"] != n for c, n in zip(communities, counts)):
        raise ValueError("Community descriptions belong to a different partition")
    by_id = {r["icsd_id"]: r for r in rows}
    events = csv_rows(paths["node_events"])
    if len(events) != len(rows) or len({r["icsd_id"] for r in events}) != len(events):
        raise ValueError("Node-event identities differ from production")
    for row in events:
        expected = by_id.get(row["icsd_id"])
        if expected is None or (row["year"], row["community"]) != (expected["year"], expected["community"]):
            raise ValueError("Node-event years/communities differ from production")
    representatives = csv_rows(paths["representatives"])
    for row in representatives:
        expected = by_id.get(row["icsd_id"])
        if expected is None or expected["community"] != row["community"] or expected["year"] != row["year"]:
            raise ValueError("Representative metadata belongs to another partition")
    accessibility = json.loads(paths["accessibility"].read_text())
    n_dated = sum(r["year"] != "" and int(r["community"]) >= 0 for r in rows)
    if accessibility["n_icsd_scored"] != n_dated or accessibility["alpha"] != .5 or accessibility["beta"] != .5:
        raise ValueError("Accessibility summary uses a different population or weights")
    if not np.isfinite([accessibility["icsd_raw_mu"], accessibility["icsd_raw_sigma"]]).all() or accessibility["icsd_raw_sigma"] <= 0:
        raise ValueError("Invalid saved accessibility moments")
    families = json.loads(paths["families"].read_text())
    for group in families["groups"]:
        if not set(group["communities"]).issubset(set(communities)):
            raise ValueError("Family description has an unknown community")
        if sum(profiles[str(c)]["size"] for c in group["communities"]) != group["n_all_members"]:
            raise ValueError("Family membership count differs from production")
    layout = csv_rows(paths["layout"])
    if len({r["community"] for r in layout}) != len(layout):
        raise ValueError("Duplicate community layout rows")
    for row in layout:
        profile = profiles.get(row["community"])
        if profile is None or int(row["size"]) != profile["size"]:
            raise ValueError("Layout belongs to another partition")
        if not np.isfinite([float(row["x"]), float(row["y"])]).all():
            raise ValueError("Nonfinite community layout")
    return {"basis": basis, "metadata": metadata, "manifest": manifest, "paths": paths,
            "pca": pca, "rows": rows, "labels": labels, "profiles": profiles,
            "families": families, "representatives": representatives, "accessibility": accessibility}


def project(embedding, basis):
    vector = np.asarray(embedding, dtype=float)
    if vector.shape != (213,) or not np.isfinite(vector).all():
        raise ValueError("Upload did not produce a finite production embedding")
    pooled = vector[:-9].reshape(3, 68)
    if not np.any(pooled[:, :7]) or not np.any(pooled[:, 7:]):
        raise ValueError("Upload has an empty chemistry or local-geometry feature block")
    return (((vector - basis["scaler_mean"]) / basis["scaler_scale"] - basis["pca_mean"])
            @ basis["pca_components"].T)


def score(structure, bundle, observation_year=2019):
    if not 0 < len(structure) <= 256:
        raise ValueError("The production representation supports 1–256 sites")
    diagnostics = {}
    embedding = build_structure_embedding(structure, wl_iters=3, local_mode="matminer_ops", diagnostics=diagnostics)
    basis = bundle["basis"]
    point = project(embedding, basis)
    distances = np.linalg.norm(basis["centroids"] - point, axis=1)
    i = int(np.argmin(distances))
    distance, radius = float(distances[i]), float(basis["p95"][i])
    birth = int(basis["birth_years"][i])
    # Match the saved accessibility producer's fallback without presenting an
    # imputed year as a known first report.
    cost_birth = birth if birth >= 0 else 2010
    age = max(observation_year - cost_birth, 0)
    raw = (np.log1p(distance / max(float(basis["p50"][i]), 1e-6))
           - .5 * np.log1p(float(basis["counts"][i])) - .5 * np.log1p(age))
    moments = bundle["accessibility"]
    return {"formula": structure.composition.reduced_formula, "n_sites": len(structure),
            "community": int(basis["communities"][i]), "distance": distance, "threshold": radius,
            "d_over_tau": distance / radius if radius > 0 else None, "frontier": distance > radius,
            "accessibility": float((raw - moments["icsd_raw_mu"]) / moments["icsd_raw_sigma"]),
            "community_size": int(basis["counts"][i]), "community_birth_year": birth,
            "cost_birth_year": cost_birth, "cost_birth_year_imputed": birth < 0,
            "observation_year": observation_year, "xy": point[:2], "centroid_xy": basis["centroids"][i, :2],
            "diagnostics": diagnostics}
