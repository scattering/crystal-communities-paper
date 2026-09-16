#!/usr/bin/env python3
"""Post-hoc nested electronegativity diagnostic from frozen bonding caches.

This control uses the mean/difference electronegativity coordinates already in
the five-dimensional primary joint vector, with their frozen joint scales.  It
does not alter or replace the frozen primary analysis.
"""
from __future__ import annotations

import os

# Keep each worker single-threaded before NumPy/SciPy initialize BLAS.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import bond_metrics as metrics


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE = RESULTS / "cache"
SEED = 20260909
N_SHUFFLES = 32
N_BOOTSTRAPS = 2000
N_WORKERS = 8
SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")
X2_COLUMNS = (1, 2)


def load_cache(tag: str) -> dict[str, np.ndarray]:
    with np.load(CACHE / f"{tag}.npz") as archive:
        return {name: archive[name] for name in archive.files}


def query_seed(query_tag: str) -> int:
    return SEED + int(hashlib.sha256(query_tag.encode()).hexdigest()[:8], 16)


def x2(cache: dict[str, np.ndarray], permutation=None) -> tuple[np.ndarray, np.ndarray]:
    points, weights = metrics.vectors(cache, permutation=permutation)["joint"]
    return points[:, X2_COLUMNS], weights


def score(row: dict[str, str], scale: np.ndarray) -> dict[str, object]:
    query = load_cache(row["query_tag"])
    reference = load_cache(f"icsd_{row['icsd_reference']}")
    query_actual, query_weights = x2(query)
    reference_points, reference_weights = x2(reference)
    actual = metrics.weighted_energy(
        query_actual, query_weights, reference_points, reference_weights, scale
    )
    rng = np.random.default_rng(query_seed(row["query_tag"]))
    shuffled = []
    for _ in range(N_SHUFFLES):
        points, weights = x2(query, rng.permutation(len(query["z"])))
        shuffled.append(
            metrics.weighted_energy(points, weights, reference_points, reference_weights, scale)
        )
    shuffled_array = np.asarray(shuffled, dtype=float)
    shuffled_median = float(np.median(shuffled_array))
    return {
        "source": row["source"],
        "material_id": row["material_id"],
        "matched_gnome_id": row["matched_gnome_id"],
        "cell": row["cell"],
        "query_tag": row["query_tag"],
        "icsd_reference": row["icsd_reference"],
        "nested_x2_actual": float(actual),
        "nested_x2_shuffle_median": shuffled_median,
        "nested_x2_delta": float(actual - shuffled_median),
        "nested_x2_shuffle_sd": float(shuffled_array.std()),
        "joint5_actual": float(row["joint_actual"]),
        "joint5_delta": float(row["joint_delta"]),
        "nested_minus_joint5_actual": float(actual - float(row["joint_actual"])),
        "nested_minus_joint5_delta": float(actual - shuffled_median - float(row["joint_delta"])),
    }


def verify_primary_seed(row: dict[str, str], joint_scale: np.ndarray) -> dict[str, object]:
    query = load_cache(row["query_tag"])
    reference = load_cache(f"icsd_{row['icsd_reference']}")
    query_actual = metrics.vectors(query)["joint"]
    reference_actual = metrics.vectors(reference)["joint"]
    actual = metrics.weighted_energy(*query_actual, *reference_actual, joint_scale)
    rng = np.random.default_rng(query_seed(row["query_tag"]))
    permutation0 = metrics.vectors(query, rng.permutation(len(query["z"])))["joint"]
    shuffled0 = metrics.weighted_energy(*permutation0, *reference_actual, joint_scale)
    saved_null = json.loads((RESULTS / "null_scores.json").read_text())[row["query_tag"]]
    saved_actual = float(row["joint_actual"])
    saved_shuffled0 = float(saved_null[0]["joint"])
    actual_error = abs(actual - saved_actual)
    shuffled0_error = abs(shuffled0 - saved_shuffled0)
    if actual_error > 1e-12 or shuffled0_error > 1e-12:
        raise ValueError(
            f"Primary seed/procedure mismatch: actual={actual_error}, permutation0={shuffled0_error}"
        )
    return {
        "query_tag": row["query_tag"],
        "saved_joint5_actual": saved_actual,
        "recomputed_joint5_actual": float(actual),
        "absolute_actual_error": float(actual_error),
        "saved_joint5_permutation0": saved_shuffled0,
        "recomputed_joint5_permutation0": float(shuffled0),
        "absolute_permutation0_error": float(shuffled0_error),
        "tolerance": 1e-12,
        "passed": True,
    }


def bootstrap_summary(rows: list[dict[str, object]], label: str, inferential: bool) -> dict[str, object]:
    references: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        references.setdefault(str(row["icsd_reference"]), []).append(index)
    clusters = list(references.values())
    actual = np.asarray([row["nested_x2_actual"] for row in rows], dtype=float)
    shuffle = np.asarray([row["nested_x2_shuffle_median"] for row in rows], dtype=float)
    delta = np.asarray([row["nested_x2_delta"] for row in rows], dtype=float)
    joint_delta = np.asarray([row["joint5_delta"] for row in rows], dtype=float)
    nested_minus_joint = delta - joint_delta
    rng_offset = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    rng = np.random.default_rng((SEED + rng_offset) % (2**64))
    bootstrap_delta, bootstrap_nested_minus_joint = [], []
    for _ in range(N_BOOTSTRAPS):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.fromiter(
            (i for cluster_index in chosen for i in clusters[int(cluster_index)]), dtype=int
        )
        bootstrap_delta.append(float(np.median(delta[indices])))
        bootstrap_nested_minus_joint.append(float(np.median(nested_minus_joint[indices])))
    return {
        "n": len(rows),
        "n_icsd_reference_clusters": len(clusters),
        "largest_icsd_reference_cluster": max(map(len, clusters)),
        "inferential_role": "post-hoc diagnostic" if inferential else "descriptive post-hoc diagnostic",
        "median_actual": float(np.median(actual)),
        "median_shuffle_median": float(np.median(shuffle)),
        "median_paired_actual_minus_shuffle": float(np.median(delta)),
        "paired_delta_cluster_bootstrap_ci95": [
            float(value) for value in np.percentile(bootstrap_delta, [2.5, 97.5])
        ],
        "n_actual_below_shuffle_median": int(np.sum(delta < 0.0)),
        "median_joint5_delta": float(np.median(joint_delta)),
        "median_within_target_nested_minus_joint5_delta": float(np.median(nested_minus_joint)),
        "nested_minus_joint5_delta_cluster_bootstrap_ci95": [
            float(value)
            for value in np.percentile(bootstrap_nested_minus_joint, [2.5, 97.5])
        ],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with (RESULTS / "per_target.csv").open(newline="", encoding="utf-8") as handle:
        primary_rows = list(csv.DictReader(handle))
    if len(primary_rows) != 467:
        raise ValueError(f"Expected 467 completed targets, found {len(primary_rows)}")
    scale_record = json.loads((RESULTS / "scales.json").read_text())
    joint_scale = np.asarray(scale_record["joint"]["sd"], dtype=float)
    nested_scale = joint_scale[list(X2_COLUMNS)]
    verification = verify_primary_seed(primary_rows[0], joint_scale)

    tasks = [(row, nested_scale) for row in primary_rows]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        output_rows = list(pool.map(_score_task, tasks))

    cohort_results = {}
    for source in SOURCES:
        source_rows = [row for row in output_rows if row["source"] == source]
        cohort_results[source] = bootstrap_summary(
            source_rows, f"cohort|{source}", inferential=source == "gnome"
        )
    gnome_rows = [row for row in output_rows if row["source"] == "gnome"]
    strata_results = {}
    for cell in sorted({str(row["cell"]) for row in gnome_rows}):
        cell_rows = [row for row in gnome_rows if row["cell"] == cell]
        strata_results[cell] = bootstrap_summary(cell_rows, f"gnome-stratum|{cell}", True)

    csv_path = RESULTS / "nested_chemistry_check.csv"
    json_path = RESULTS / "nested_chemistry_check.json"
    write_csv(csv_path, output_rows)
    result = {
        "analysis_status": "post-hoc diagnostic; frozen primary results preserved",
        "purpose": (
            "Compare the primary joint result with a nested energy distance using only "
            "the same mean and absolute-difference electronegativity coordinates."
        ),
        "metric": "weighted energy V-statistic without square root",
        "coordinates": ["mean_electronegativity", "absolute_electronegativity_difference"],
        "primary_joint_column_indices": list(X2_COLUMNS),
        "scale": nested_scale.tolist(),
        "scale_source": "results/scales.json joint SD columns 1 and 2",
        "n_targets": len(output_rows),
        "n_shuffles_per_target": N_SHUFFLES,
        "shuffle_seed": "20260909 + int(sha256(query_tag)[:8], 16)",
        "bootstrap": {
            "replicates": N_BOOTSTRAPS,
            "interval": "percentile 95%",
            "cluster": "shared icsd_reference within each reported group",
            "seed": SEED,
        },
        "verification": verification,
        "gnome_overall": cohort_results["gnome"],
        "gnome_strata": strata_results,
        "other_cohorts_descriptive": {
            source: cohort_results[source] for source in SOURCES if source != "gnome"
        },
        "interpretive_limit": (
            "The nested and five-dimensional energy statistics share the same metric and "
            "electronegativity basis, but their dimensional distances are not an additive "
            "decomposition. Comparisons are diagnostic rather than causal attribution."
        ),
        "input_sha256": {
            "per_target.csv": sha256(RESULTS / "per_target.csv"),
            "scales.json": sha256(RESULTS / "scales.json"),
            "null_scores.json": sha256(RESULTS / "null_scores.json"),
            "bond_metrics.py": sha256(HERE / "bond_metrics.py"),
        },
    }
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(json.dumps({"gnome_overall": result["gnome_overall"], "verification": verification}))


def _score_task(task: tuple[dict[str, str], np.ndarray]) -> dict[str, object]:
    return score(*task)


if __name__ == "__main__":
    main()
