#!/usr/bin/env python3
"""Run the reviewer-requested retrospective test with a cutoff-trained map.

For each cutoff T, this producer fits the scaler and PCA only on dated ICSD
entries available through T, constructs the mutual-16-nearest-neighbour graph
and Louvain partition only on those entries, and estimates community centroids
and 95th-percentile radii in that map.  Every dated post-T ICSD entry is then
transformed and assigned without refitting.  Reduced-formula references are
also restricted to entries available through T.

The output retains per-entry classifications, cutoff partitions, fitted numeric
parameters and the same bootstrap/permutation summaries used by the narrower
fixed-partition retrospective analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn import config_context
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "notes" / "review_2026_08"
EXPECTED_FEATURE_VERSION = "crystal-features-v2-geometric-crystalnn"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(REVIEW))
from analyze_synthesis_retrodiction import load_icsd_index  # noqa: E402
from formula_conventions import (  # noqa: E402
    anonymous_stoichiometry_key,
    build_formula_precedent_index,
    formula_has_precedent,
    normalized_fraction_key,
    scale_invariant_formula_key,
)
from retrospective_quadrant import (  # noqa: E402
    bootstrap,
    by_year,
    composition_class,
    permutation,
    quadrant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-metadata", type=Path, required=True)
    parser.add_argument("--sample-assignments", type=Path, required=True)
    parser.add_argument("--icsd-index", type=Path, required=True)
    parser.add_argument(
        "--occupancy-flags",
        type=Path,
        required=True,
        help="Per-ICSD-record full/partial site-occupancy classification",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--production-pca", type=Path)
    parser.add_argument("--production-labels", type=Path)
    parser.add_argument(
        "--external",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "FEATURES", "RECORDS"),
        help="Repeat for each computed cohort; raw feature rows must match RECORDS",
    )
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1990, 2000, 2010])
    parser.add_argument("--max-year", type=int, default=2015)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-component-size", type=int, default=8)
    parser.add_argument("--min-community-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n")


def normalized_formula_identity_token(formula: str) -> str:
    """Stable scale-invariant token for a nominal composition."""
    if not str(formula).strip():
        return ""
    elements, fractions = normalized_fraction_key(formula)
    return "|".join(elements) + "::" + ",".join(f"{value:.12g}" for value in fractions)


def formula_identity_token(formula: str, is_partial: bool) -> str:
    """Nominal-composition identity for the earliest-per-formula analysis.

    Formula grouping uses one transform for every held-out record. Occupancy
    remains available as metadata and determines how an ICSD record enters the
    *precedent-matching* index, but it does not change the grouping identity.
    The unused ``is_partial`` argument is retained to keep the row-building
    call explicit and backwards compatible.
    """
    del is_partial
    return normalized_formula_identity_token(formula)


def anonymous_stoichiometry_token(formula: str) -> str:
    """Scale-invariant anonymous-composition key in OPTIMADE coefficient order."""
    if not str(formula).strip():
        return ""
    return "|".join(map(str, anonymous_stoichiometry_key(formula)))


def actual_neighbors(X: np.ndarray, k: int, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return k actual non-self neighbours in the estimator's tie order."""
    if len(X) <= k:
        raise ValueError(f"Need more than k={k} training rows")
    estimator = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        algorithm="auto",
        n_jobs=n_jobs,
    ).fit(X)
    indices = np.empty((len(X), k), dtype=np.int64)
    distances = np.empty((len(X), k), dtype=np.float64)
    with config_context(working_memory=512):
        for start in range(0, len(X), 4096):
            stop = min(start + 4096, len(X))
            raw_distances, raw_indices = estimator.kneighbors(X[start:stop])
            for local, row_index in enumerate(range(start, stop)):
                keep = np.flatnonzero(raw_indices[local] != row_index)[:k]
                if len(keep) != k:
                    raise ValueError(
                        f"Neighbour search returned fewer than {k} non-self rows for {row_index}"
                    )
                indices[row_index] = raw_indices[local, keep]
                distances[row_index] = raw_distances[local, keep]
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite nearest-neighbour distance")
    if any(len(set(map(int, row))) != k for row in indices):
        raise ValueError("Duplicate retained nearest-neighbour row")
    return indices, distances


def build_partition(
    X: np.ndarray,
    *,
    k: int,
    resolution: float,
    min_component_size: int,
    min_community_size: int,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, object]]:
    started = time.time()
    indices, distances = actual_neighbors(X, k, n_jobs)
    positive = distances[distances > 0]
    sigma = max(float(np.median(positive)) if len(positive) else 1.0, 1e-8)
    neighbor_sets = [set(map(int, row)) for row in indices]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(X)))
    for i, (row_indices, row_distances) in enumerate(zip(indices, distances)):
        for j, distance in zip(row_indices, row_distances):
            j = int(j)
            if i not in neighbor_sets[j]:
                continue
            weight = float(math.exp(-((float(distance) / sigma) ** 2)))
            if not graph.has_edge(i, j) or weight > graph[i][j]["weight"]:
                graph.add_edge(i, j, weight=weight)

    component_sizes = sorted((len(group) for group in nx.connected_components(graph)), reverse=True)
    kept_nodes = {
        node
        for group in nx.connected_components(graph)
        if len(group) >= min_component_size
        for node in group
    }
    retained = graph.subgraph(kept_nodes).copy()
    communities = nx.community.louvain_communities(
        retained,
        weight="weight",
        resolution=resolution,
        seed=seed,
    ) if retained.number_of_nodes() else []
    labels = np.full(len(X), -1, dtype=np.int64)
    kept_communities = []
    small_communities = []
    for members in communities:
        target = kept_communities if len(members) >= min_community_size else small_communities
        target.append(members)
    # Stable numeric labels make reruns directly comparable.
    kept_communities.sort(key=lambda members: min(members))
    for label, members in enumerate(kept_communities):
        labels[np.fromiter(sorted(members), dtype=np.int64)] = label

    summary = {
        "n_train": int(len(X)),
        "knn_k": int(k),
        "metric": "euclidean",
        "mutual_knn": True,
        "edge_weight": "exp[-(distance/sigma)^2]",
        "sigma": sigma,
        "n_edges_before_component_filter": int(graph.number_of_edges()),
        "n_connected_components_before_filter": int(len(component_sizes)),
        "n_components_below_minimum": int(sum(size < min_component_size for size in component_sizes)),
        "n_entries_in_components_below_minimum": int(sum(size for size in component_sizes if size < min_component_size)),
        "min_component_size": int(min_component_size),
        "n_nodes_after_component_filter": int(retained.number_of_nodes()),
        "louvain_implementation": "networkx.community.louvain_communities",
        "networkx_version": nx.__version__,
        "resolution": float(resolution),
        "seed": int(seed),
        "n_louvain_groups_before_size_filter": int(len(communities)),
        "n_louvain_groups_below_minimum": int(len(small_communities)),
        "n_entries_in_louvain_groups_below_minimum": int(sum(map(len, small_communities))),
        "min_community_size": int(min_community_size),
        "n_communities": int(len(kept_communities)),
        "n_assigned": int(np.sum(labels >= 0)),
        "n_outliers": int(np.sum(labels < 0)),
        "largest_communities": Counter(labels[labels >= 0].tolist()).most_common(20),
        "elapsed_seconds": float(time.time() - started),
    }
    return labels, summary


def centroid_map(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    communities = np.asarray(sorted(set(labels[labels >= 0].tolist())), dtype=np.int64)
    if not len(communities):
        raise ValueError("No retained cutoff communities")
    centroids = np.empty((len(communities), X.shape[1]), dtype=np.float64)
    radii = np.empty(len(communities), dtype=np.float64)
    sizes = np.empty(len(communities), dtype=np.int64)
    for position, community in enumerate(communities):
        members = X[labels == community]
        center = members.mean(axis=0)
        distances = np.linalg.norm(members - center, axis=1)
        centroids[position] = center
        radii[position] = np.percentile(distances, 95)
        sizes[position] = len(members)
    return communities, centroids, radii, sizes


def classify(
    X: np.ndarray,
    communities: np.ndarray,
    centroids: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.empty(len(X), dtype=np.int64)
    distance = np.empty(len(X), dtype=np.float64)
    centre_norm = np.sum(centroids * centroids, axis=1)
    n_candidates = min(3, len(centroids))
    for start in range(0, len(X), 2048):
        stop = min(start + 2048, len(X))
        block = X[start:stop]
        distance_squared = (
            np.sum(block * block, axis=1)[:, None]
            + centre_norm[None, :]
            - 2.0 * block @ centroids.T
        )
        candidates = np.argpartition(distance_squared, n_candidates - 1, axis=1)[:, :n_candidates]
        exact = np.linalg.norm(block[:, None, :] - centroids[candidates], axis=2)
        local = np.argmin(exact, axis=1)
        chosen = candidates[np.arange(len(block)), local]
        selected[start:stop] = chosen
        distance[start:stop] = exact[np.arange(len(block)), local]
    return communities[selected], distance, distance <= radii[selected]


def statistical_block(frame: pd.DataFrame, rng: np.random.Generator, n_boot: int, n_perm: int) -> dict[str, object]:
    return {
        "quadrant": quadrant(frame),
        "by_year": by_year(frame),
        "bootstrap": bootstrap(frame, rng, n_boot=n_boot),
        "permutation_global": permutation(frame, rng, n_perm=n_perm),
        "permutation_within_year": permutation(frame, rng, "year", n_perm=n_perm),
        "permutation_within_coarse_strata": permutation(
            frame, rng, "stratum_coarse", n_perm=n_perm
        ),
        "permutation_within_anonymized_strata": permutation(
            frame, rng, "stratum_anon", n_perm=n_perm
        ),
    }


def expected_quadrant_for_uniform_tie_choice(frame: pd.DataFrame) -> dict[str, object]:
    """Expected quadrant values when one earliest-year tied entry is sampled."""
    n = len(frame)
    in_basin_probability = frame.in_basin.to_numpy(dtype=float)
    formula_match = frame.formula_match.to_numpy(dtype=bool)
    p_in_basin = float(in_basin_probability.mean())
    p_match = float(formula_match.mean())
    joint = float((in_basin_probability * formula_match).mean())
    expected = p_in_basin * p_match
    return {
        "n": n,
        "expected_in_basin_and_match_count": float(
            np.sum(in_basin_probability * formula_match)
        ),
        "share_in_basin_and_match": joint,
        "p_in_basin": p_in_basin,
        "p_match": p_match,
        "expected_share_in_basin_and_match_independence": expected,
        "enrichment_ratio_obs_over_independence": (
            joint / expected if expected > 0 else None
        ),
    }


def earliest_year_tie_sensitivity(
    earliest_year_rows: pd.DataFrame,
    match_column: str,
) -> dict[str, object]:
    """Summarize sensitivity to structures tied in a formula's first year."""
    grouped_match_counts = earliest_year_rows.groupby("formula_identity")[
        match_column
    ].nunique()
    if bool((grouped_match_counts != 1).any()):
        raise ValueError(
            f"Formula-match status varies within a formula identity for {match_column}"
        )
    outside_when_available = (
        earliest_year_rows.sort_values(
            ["formula_identity", "in_basin", "icsd_id"],
            ascending=[True, True, True],
            kind="stable",
        )
        .drop_duplicates("formula_identity", keep="first")
        .copy()
    )
    inside_when_available = (
        earliest_year_rows.sort_values(
            ["formula_identity", "in_basin", "icsd_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("formula_identity", keep="first")
        .copy()
    )
    for frame in (outside_when_available, inside_when_available):
        frame["formula_match"] = frame[match_column]
    uniform = (
        earliest_year_rows.groupby("formula_identity", sort=False)
        .agg(in_basin=("in_basin", "mean"), formula_match=(match_column, "first"))
        .reset_index()
    )
    return {
        "outside_when_available": quadrant(outside_when_available),
        "inside_when_available": quadrant(inside_when_available),
        "uniform_random_choice_expectation": expected_quadrant_for_uniform_tie_choice(
            uniform
        ),
    }


def rate_block(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=bool)
    n = len(values)
    k = int(values.sum())
    if not n:
        return {"k": 0, "n": 0, "rate": None, "wilson95": [None, None]}
    z = 1.96
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"k": k, "n": n, "rate": p, "wilson95": [center - half, center + half]}


def shared_strata_comparison(
    heldout: pd.DataFrame,
    external: pd.DataFrame,
    field: str,
) -> dict[str, object]:
    held = heldout[(heldout.formula != "") & heldout[field].notna()].copy()
    source = external[(external.formula != "") & external[field].notna()].copy()
    shared = set(held[field].astype(str)) & set(source[field].astype(str))
    held = held[held[field].astype(str).isin(shared)]
    source = source[source[field].astype(str).isin(shared)]
    held_rate = rate_block(held.in_basin.to_numpy(dtype=bool))
    source_rate = rate_block(source.in_basin.to_numpy(dtype=bool))
    gap = None
    if held_rate["rate"] is not None and source_rate["rate"] is not None:
        gap = held_rate["rate"] - source_rate["rate"]
    return {
        "n_shared_strata": len(shared),
        "heldout": held_rate,
        "external": source_rate,
        "heldout_minus_external": gap,
        "definition": "restrict each population to strata represented in both; retain original stratum frequencies",
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = np.load(args.features, mmap_mode="r")
    feature_metadata = json.loads(args.feature_metadata.read_text())
    if feature_metadata.get("feature_version") != EXPECTED_FEATURE_VERSION:
        raise ValueError(
            "Unexpected production feature version: "
            f"{feature_metadata.get('feature_version')!r}"
        )
    if int(feature_metadata.get("feature_diagnostics", {}).get("n_structures", -1)) != len(
        features
    ):
        raise ValueError("Production feature metadata row count does not match matrix")
    rows = pd.read_csv(args.sample_assignments, keep_default_na=False)
    if len(rows) != len(features) or not rows.icsd_id.is_unique:
        raise ValueError("Feature/assignment rows are not one-to-one")
    ids = rows.icsd_id.to_numpy(dtype=np.int64)
    years = pd.to_numeric(rows.year, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    index = load_icsd_index(args.icsd_index)
    formulas = np.asarray(
        [str(index.get(int(identifier), {}).get("reduced_formula") or "") for identifier in ids],
        dtype=object,
    )
    occupancy_rows = pd.read_csv(args.occupancy_flags, keep_default_na=False)
    required_occupancy_columns = {"icsd_id", "has_partial_occupancy"}
    if not required_occupancy_columns.issubset(occupancy_rows.columns):
        raise ValueError(
            f"Occupancy flags require columns {sorted(required_occupancy_columns)}"
        )
    if not occupancy_rows.icsd_id.is_unique:
        raise ValueError("Occupancy flags contain duplicate ICSD identifiers")
    occupancy_by_id = dict(
        zip(
            occupancy_rows.icsd_id.astype(np.int64),
            occupancy_rows.has_partial_occupancy.astype(bool),
        )
    )
    missing_occupancy = sorted(set(map(int, index)) - set(occupancy_by_id))
    if missing_occupancy:
        raise ValueError(
            f"Occupancy flags miss {len(missing_occupancy)} ICSD identifiers; "
            f"first={missing_occupancy[:5]}"
        )
    partial_flags = np.asarray([occupancy_by_id[int(identifier)] for identifier in ids])
    index_loader_path = ROOT / "scripts" / "analyze_synthesis_retrodiction.py"
    wrapper_path = (
        ROOT
        / "notes"
        / "feature_repair_2026_09"
        / "downstream"
        / "run_cutoff_trained_retrospective.sh"
    )
    source_paths = {
        "features": args.features,
        "feature_metadata": args.feature_metadata,
        "sample_assignments": args.sample_assignments,
        "icsd_index": args.icsd_index,
        "occupancy_flags": args.occupancy_flags,
        "producer": Path(__file__),
        "formula_conventions": ROOT / "scripts" / "formula_conventions.py",
        "index_loader": index_loader_path,
        "statistical_module": REVIEW / "retrospective_quadrant.py",
        "wrapper": wrapper_path,
    }
    if bool(args.production_pca) != bool(args.production_labels):
        raise ValueError("Provide --production-pca and --production-labels together")
    if args.production_pca:
        source_paths.update({
            "production_pca": args.production_pca,
            "production_labels": args.production_labels,
        })
    external_inputs: dict[str, tuple[np.ndarray, pd.DataFrame, Path, Path]] = {}
    for name, feature_text, record_text in args.external:
        if name in external_inputs:
            raise ValueError(f"Duplicate external source {name}")
        feature_path = Path(feature_text)
        record_path = Path(record_text)
        matrix = np.load(feature_path, mmap_mode="r")
        records = pd.read_csv(record_path, keep_default_na=False)
        if len(matrix) != len(records):
            raise ValueError(f"{name} feature/record length mismatch")
        if "feature_row" in records and not np.array_equal(
            records.feature_row.to_numpy(dtype=np.int64), np.arange(len(records))
        ):
            raise ValueError(f"{name} feature_row does not match stored matrix order")
        if "material_id" not in records or "reduced_formula" not in records:
            raise ValueError(f"{name} records need material_id and reduced_formula")
        metadata_path = feature_path.parent / "feature_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"{name} is missing {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("feature_version") != EXPECTED_FEATURE_VERSION:
            raise ValueError(f"{name} uses unexpected feature version")
        if int(metadata.get("n_rows", -1)) != len(matrix):
            raise ValueError(f"{name} feature metadata row count does not match matrix")
        if int(metadata.get("n_features", -1)) != features.shape[1]:
            raise ValueError(f"{name} feature dimension does not match ICSD")
        external_inputs[name] = (matrix, records, feature_path, record_path)
        source_paths[f"external_{name}_features"] = feature_path
        source_paths[f"external_{name}_records"] = record_path
        source_paths[f"external_{name}_feature_metadata"] = metadata_path
    result: dict[str, object] = {
        "schema_version": 1,
        "purpose": "reviewer-requested retrospective validation with independently trained cutoff maps",
        "protocol": {
            "cutoffs": sorted(set(args.cutoffs)),
            "max_year": args.max_year,
            "scaler": "sklearn.preprocessing.StandardScaler fitted only on dated entries with year <= T",
            "pca": "sklearn.decomposition.PCA fitted only on dated entries with year <= T",
            "pca_components": 32,
            "knn_k": args.k,
            "louvain_resolution": args.resolution,
            "min_component_size": args.min_component_size,
            "min_community_size": args.min_community_size,
            "seed": args.seed,
            "partition": "Euclidean mutual-k-NN, exp[-(d/sigma)^2], seeded Louvain",
            "future_assignment": "nearest cutoff-trained community centroid; in-basin at distance <= cutoff-community p95",
            "partial_occupancy_fraction_tolerance": 1e-6,
            "formula_rule": (
                "exact scale-invariant integer ratios for fully occupied ICSD "
                "records; same element set and maximum absolute atomic-fraction "
                "difference <= 1e-6 for partial-occupancy ICSD records"
            ),
            "per_formula_identity": (
                "every heldout record uses the same element-sorted normalized "
                "atomic-fraction vector rounded to 12 decimal places; occupancy "
                "is retained as metadata and does not change the grouping unit"
            ),
            "per_formula_selection": (
                "select the earliest post-cutoff year for each nominal formula "
                "identity; break same-year ties by the lowest ICSD identifier"
            ),
            "formula_references": [
                "successfully featurized dated ICSD compositions with year <= T",
                "successfully featurized dated post-1980 ICSD compositions with year <= T",
                "complete indexed dated ICSD compositions with year <= T",
                "complete indexed dated post-1980 ICSD compositions with year <= T",
            ],
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "n_jobs": args.n_jobs,
        },
        "execution": {
            "argv": [str(value) for value in sys.argv],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": platform.node(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "networkx": nx.__version__,
            "scipy": scipy.__version__,
            "pymatgen": package_version("pymatgen"),
        },
        "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in source_paths.items()},
        "cutoffs": {},
    }
    if args.production_pca:
        production_pca = np.load(args.production_pca, mmap_mode="r")
        production_rows = pd.read_csv(args.production_labels, keep_default_na=False)
        if (
            len(production_pca) != len(rows)
            or len(production_rows) != len(rows)
            or not np.array_equal(production_rows.icsd_id.to_numpy(dtype=np.int64), ids)
        ):
            raise ValueError("Production audit inputs do not share feature row order")
        reproduced, audit = build_partition(
            np.asarray(production_pca, dtype=np.float64),
            k=args.k,
            resolution=args.resolution,
            min_component_size=args.min_component_size,
            min_community_size=args.min_community_size,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        reported = production_rows.community.to_numpy(dtype=np.int64)
        comparison = {
            "reported_n_communities": int(len(set(reported[reported >= 0].tolist()))),
            "reported_n_outliers": int(np.sum(reported < 0)),
            "noise_mask_identical": bool(np.array_equal(reproduced < 0, reported < 0)),
            "adjusted_rand_index_all_entries": float(adjusted_rand_score(reported, reproduced)),
            "normalized_mutual_information_all_entries": float(
                normalized_mutual_info_score(reported, reproduced, average_method="arithmetic")
            ),
        }
        if not comparison["noise_mask_identical"] or not np.isclose(
            comparison["adjusted_rand_index_all_entries"], 1.0, rtol=0, atol=1e-14
        ):
            raise ValueError(f"Production graph reconstruction differs: {comparison}")
        audit_rows = pd.DataFrame({
            "icsd_id": ids,
            "reported_community": reported,
            "reproduced_community": reproduced,
        })
        audit_path = args.output_dir / "production_filter_audit_assignments.csv"
        audit_rows.to_csv(audit_path, index=False)
        result["production_filter_audit"] = {
            **audit,
            "comparison_to_reported_partition": comparison,
            "assignments": {
                "path": audit_path.name,
                "sha256": digest(audit_path),
            },
        }
        dump(args.output_dir / "cutoff_trained_retrospective_summary.json", result)
        print(
            "Production filter audit: "
            f"{audit['n_components_below_minimum']:,} components/"
            f"{audit['n_entries_in_components_below_minimum']:,} entries; "
            f"{audit['n_louvain_groups_below_minimum']:,} Louvain groups/"
            f"{audit['n_entries_in_louvain_groups_below_minimum']:,} entries",
            flush=True,
        )
    for cutoff in sorted(set(args.cutoffs)):
        started = time.time()
        train = (years > 0) & (years <= cutoff)
        heldout = (years > cutoff) & (years <= args.max_year)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(np.asarray(features[train], dtype=np.float64))
        pca = PCA(n_components=32, random_state=args.seed)
        train_pca = pca.fit_transform(train_scaled)
        heldout_pca = pca.transform(scaler.transform(np.asarray(features[heldout], dtype=np.float64)))
        labels, partition = build_partition(
            train_pca,
            k=args.k,
            resolution=args.resolution,
            min_component_size=args.min_component_size,
            min_community_size=args.min_community_size,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        communities, centroids, radii, sizes = centroid_map(train_pca, labels)
        assigned, distances, in_basin = classify(heldout_pca, communities, centroids, radii)

        leaf = args.output_dir / f"T{cutoff}"
        leaf.mkdir(parents=True, exist_ok=True)
        train_rows = pd.DataFrame({
            "icsd_id": ids[train],
            "year": years[train],
            "cutoff_community": labels,
        })
        train_rows.to_csv(leaf / "training_partition.csv", index=False)
        np.savez(
            leaf / "cutoff_map.npz",
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            pca_mean=pca.mean_,
            pca_components=pca.components_,
            pca_explained_variance_ratio=pca.explained_variance_ratio_,
            communities=communities,
            centroids=centroids,
            radii_p95=radii,
            community_sizes=sizes,
        )

        held = pd.DataFrame({
            "icsd_id": ids[heldout],
            "year": years[heldout],
            "formula": formulas[heldout],
            "assigned_community": assigned,
            "nearest_centroid_distance": distances,
            "community_threshold_p95": radii[np.searchsorted(communities, assigned)],
            "in_basin": in_basin.astype(np.int64),
            "has_partial_occupancy": partial_flags[heldout].astype(np.int64),
        })
        held["stratum_coarse"] = held.formula.map(composition_class)
        held["stratum_anon"] = held.formula.map(anonymous_stoichiometry_token)
        held["formula_identity"] = [
            formula_identity_token(formula, bool(is_partial))
            for formula, is_partial in zip(
                held.formula, held.has_partial_occupancy, strict=True
            )
        ]
        reference_records = {
            "all_year_le_T": [
                (str(formula), bool(is_partial))
                for formula, is_partial, year in zip(formulas, partial_flags, years)
                if formula != "" and 0 < int(year) <= cutoff
            ],
            "post1980_le_T": [
                (str(formula), bool(is_partial))
                for formula, is_partial, year in zip(formulas, partial_flags, years)
                if formula != "" and 1980 < int(year) <= cutoff
            ],
            "all_year_le_T_index": [
                (str(meta["reduced_formula"]), occupancy_by_id[int(identifier)])
                for identifier, meta in index.items()
                if meta.get("reduced_formula")
                and meta.get("year") is not None
                and int(meta["year"]) <= cutoff
            ],
            "post1980_le_T_index": [
                (str(meta["reduced_formula"]), occupancy_by_id[int(identifier)])
                for identifier, meta in index.items()
                if meta.get("reduced_formula")
                and meta.get("year") is not None
                and 1980 < int(meta["year"]) <= cutoff
            ],
        }
        reference_sets = {
            name: build_formula_precedent_index(records, tolerance=1e-6)
            for name, records in reference_records.items()
        }
        for reference_name, reference in reference_sets.items():
            held[f"formula_match_{reference_name}"] = np.asarray(
                [formula_has_precedent(formula, reference) for formula in held.formula],
                dtype=np.int64,
            )
        held.to_csv(leaf / "heldout_entries.csv", index=False)

        external_rows = []
        source_rates = {}
        for name, (matrix, records, _, _) in external_inputs.items():
            projected = pca.transform(scaler.transform(np.asarray(matrix, dtype=np.float64)))
            source_community, source_distance, source_in_basin = classify(
                projected, communities, centroids, radii
            )
            source = pd.DataFrame({
                "source": name,
                "material_id": records.material_id.astype(str),
                "formula": records.reduced_formula.astype(str),
                "assigned_community": source_community,
                "nearest_centroid_distance": source_distance,
                "community_threshold_p95": radii[np.searchsorted(communities, source_community)],
                "in_basin": source_in_basin.astype(np.int64),
            })
            source["stratum_coarse"] = source.formula.map(composition_class)
            source["stratum_anon"] = source.formula.map(anonymous_stoichiometry_token)
            source["formula_identity"] = source.formula.map(
                normalized_formula_identity_token
            )
            for reference_name, reference in reference_sets.items():
                source[f"formula_match_{reference_name}"] = np.asarray(
                    [formula_has_precedent(formula, reference) for formula in source.formula],
                    dtype=np.int64,
                )
            external_rows.append(source)
            source_rates[name] = {
                "all_records": rate_block(source_in_basin),
                "heldout_minus_external": float(in_basin.mean() - source_in_basin.mean()),
                "shared_coarse_strata": shared_strata_comparison(held, source, "stratum_coarse"),
                "shared_anonymized_strata": shared_strata_comparison(held, source, "stratum_anon"),
            }
        external_path = leaf / "external_classifications.csv"
        if external_rows:
            pd.concat(external_rows, ignore_index=True).to_csv(external_path, index=False)

        parseable = held[held.formula != ""].copy()
        first_year = parseable.groupby("formula_identity").year.transform("min")
        earliest_year_rows = parseable[parseable.year == first_year].copy()
        first_formula = (
            parseable.sort_values(["year", "icsd_id"], kind="stable")
            .drop_duplicates("formula_identity", keep="first")
            .copy()
        )
        analyses = {}
        tie_sensitivity = {}
        rng = np.random.default_rng(args.seed + cutoff)
        for reference_name in reference_sets:
            column = f"formula_match_{reference_name}"
            units = {}
            for unit_name, unit in (("per_entry", parseable), ("per_formula", first_formula)):
                frame = unit.copy()
                frame["formula_match"] = frame[column]
                units[unit_name] = statistical_block(frame, rng, args.n_boot, args.n_perm)
            analyses[reference_name] = units
            tie_sensitivity[reference_name] = earliest_year_tie_sensitivity(
                earliest_year_rows, column
            )

        earliest_counts = earliest_year_rows.groupby("formula_identity").size()
        earliest_basin_counts = earliest_year_rows.groupby("formula_identity")[
            "in_basin"
        ].nunique()

        cutoff_result = {
            "cutoff": cutoff,
            "n_train_dated": int(train.sum()),
            "n_heldout_dated": int(heldout.sum()),
            "n_heldout_parseable_formula": int(len(parseable)),
            "n_first_postcutoff_entry_per_formula": int(len(first_formula)),
            "earliest_year_tie_audit": {
                "n_formula_identities": int(len(earliest_counts)),
                "n_with_multiple_entries_in_earliest_year": int(
                    np.sum(earliest_counts.to_numpy() > 1)
                ),
                "n_with_mixed_basin_status_in_earliest_year": int(
                    np.sum(earliest_basin_counts.to_numpy() > 1)
                ),
                "deterministic_tie_rule": "lowest ICSD identifier",
                "by_formula_reference": tie_sensitivity,
            },
            "reference_sizes": {
                name: len(
                    {
                        normalized_formula_identity_token(formula)
                        for formula, _ in records
                    }
                )
                for name, records in reference_records.items()
            },
            "reference_composition_counts": {
                name: {
                    "records": len(records),
                    "fully_occupied_integer_keys": len(reference_sets[name].full_integer_keys),
                    "partial_occupancy_element_sets": len(reference_sets[name].partial_fraction_trees),
                    "partial_occupancy_fraction_vectors": int(
                        sum(tree.n for tree in reference_sets[name].partial_fraction_trees.values())
                    ),
                }
                for name, records in reference_records.items()
            },
            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "pca_solver_used": str(pca._fit_svd_solver),
            "partition": partition,
            "heldout_rate": rate_block(in_basin),
            "external_source_rates": source_rates,
            "analyses": analyses,
            "outputs": {
                name: {"path": str(path.relative_to(args.output_dir)), "sha256": digest(path)}
                for name, path in {
                    "training_partition": leaf / "training_partition.csv",
                    "cutoff_map": leaf / "cutoff_map.npz",
                    "heldout_entries": leaf / "heldout_entries.csv",
                    **({"external_classifications": external_path} if external_rows else {}),
                }.items()
            },
            "elapsed_seconds": float(time.time() - started),
        }
        result["cutoffs"][str(cutoff)] = cutoff_result
        dump(args.output_dir / "cutoff_trained_retrospective_summary.json", result)
        primary = analyses["all_year_le_T"]["per_entry"]["quadrant"]
        print(
            f"T={cutoff}: train={train.sum():,}, heldout={heldout.sum():,}, "
            f"communities={partition['n_communities']:,}, in-basin={primary['p_in_basin']:.4f}, "
            f"joint enrichment={primary['enrichment_ratio_obs_over_independence']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
