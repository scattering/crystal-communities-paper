#!/usr/bin/env python3
"""Cluster-bootstrap summaries for the frozen local-property pilot."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SEED = 20260909
N_BOOTSTRAP = 2_000
FAMILIES = ("f1", "f2_mean", "f2_absdiff", "pair")
REQUIRED_BASE = ("material_id", "cell", "icsd_reference", "amd_distance", "icsd_baseline_amd_distance")
INFORMATIVE_EPS = 1e-10


def _number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or missing {field!r} for {row.get('material_id', '<unknown>')}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {field!r} for {row.get('material_id', '<unknown>')}")
    return value


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def _bootstrap_indices(rows: list[dict[str, str]], rng: np.random.Generator) -> list[np.ndarray]:
    """Resample reference IDs, retaining every target attached to a sampled ID."""
    members: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        members.setdefault(row["icsd_reference"], []).append(index)
    clusters = sorted(members)
    if not clusters:
        raise ValueError("Cannot bootstrap an empty group")
    member_arrays = {key: np.asarray(indices, dtype=np.intp) for key, indices in members.items()}
    draws = rng.integers(0, len(clusters), size=(N_BOOTSTRAP, len(clusters)))
    return [np.concatenate([member_arrays[clusters[i]] for i in draw]) for draw in draws]


def _metric_summary(
    rows: list[dict[str, str]], family: str, bootstrap_indices: list[np.ndarray]
) -> dict[str, object]:
    actual = np.asarray([_number(row, f"{family}_actual") for row in rows])
    delta = np.asarray([_number(row, f"{family}_delta") for row in rows])
    shuffle_sd = np.asarray([_number(row, f"{family}_shuffle_sd") for row in rows])
    baseline_actual = np.asarray([_number(row, f"baseline_{family}_actual") for row in rows])
    baseline_delta = np.asarray([_number(row, f"baseline_{family}_delta") for row in rows])
    baseline_shuffle_sd = np.asarray([_number(row, f"baseline_{family}_shuffle_sd") for row in rows])
    paired_difference = actual - baseline_actual

    boot_delta = np.asarray([np.median(delta[index]) for index in bootstrap_indices])
    boot_paired = np.asarray([np.median(paired_difference[index]) for index in bootstrap_indices])
    return {
        "target_actual_median": float(np.median(actual)),
        "target_delta_median": float(np.median(delta)),
        "target_delta_cluster_bootstrap_ci95": _ci(boot_delta),
        "baseline_actual_median": float(np.median(baseline_actual)),
        "baseline_delta_median": float(np.median(baseline_delta)),
        "paired_target_minus_baseline_actual_median": float(np.median(paired_difference)),
        "paired_target_minus_baseline_actual_cluster_bootstrap_ci95": _ci(boot_paired),
        "target_informative_null_fraction": float(np.mean(shuffle_sd > INFORMATIVE_EPS)),
        "baseline_informative_null_fraction": float(np.mean(baseline_shuffle_sd > INFORMATIVE_EPS)),
        "shuffle_inference": family != "f1",
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _estimate(summary: dict[str, object], key: str, ci_key: str) -> str:
    ci = summary[ci_key]
    assert isinstance(ci, list)
    return f"{_fmt(float(summary[key]))} [{_fmt(float(ci[0]))}, {_fmt(float(ci[1]))}]"


def _markdown(result: dict[str, object]) -> str:
    groups = result["groups"]
    assert isinstance(groups, dict)
    lines = [
        "# Local-property pilot results",
        "",
        (
            f"Input: `{result['input_path']}`  \n"
            f"Postprocessing runtime: {float(result['runtime_seconds']):.3f} seconds. "
            "Scores are dimensionless W1 distances divided by the ICSD channel SD."
        ),
        "",
        str(result["pilot_execution_report"]),
        "",
        "The frozen strata are reported separately; they are not weighted or combined as a population prevalence estimate.",
        "",
        "| Group | n (references) | Target AMD | Baseline AMD | f2 mean delta (95% CI) | f2 mean target-baseline (95% CI) | f2 absdiff delta (95% CI) | f2 absdiff target-baseline (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in groups.items():
        metrics = group["metrics"]
        distance = group["amd_distance"]
        baseline_distance = group["icsd_baseline_amd_distance"]
        mean = metrics["f2_mean"]
        diff = metrics["f2_absdiff"]
        lines.append(
            f"| {name} | {group['n_targets']} ({group['n_reference_clusters']}) "
            f"| {_fmt(distance['median'])} [{_fmt(distance['q25'])}, {_fmt(distance['q75'])}] "
            f"| {_fmt(baseline_distance['median'])} [{_fmt(baseline_distance['q25'])}, {_fmt(baseline_distance['q75'])}] "
            f"| {_estimate(mean, 'target_delta_median', 'target_delta_cluster_bootstrap_ci95')} "
            f"| {_estimate(mean, 'paired_target_minus_baseline_actual_median', 'paired_target_minus_baseline_actual_cluster_bootstrap_ci95')} "
            f"| {_estimate(diff, 'target_delta_median', 'target_delta_cluster_bootstrap_ci95')} "
            f"| {_estimate(diff, 'paired_target_minus_baseline_actual_median', 'paired_target_minus_baseline_actual_cluster_bootstrap_ci95')} |"
        )
    lines.extend([
        "",
        "| Group | f2 mean baseline delta | f2 mean informative T/B | f2 absdiff baseline delta | f2 absdiff informative T/B |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, group in groups.items():
        metrics = group["metrics"]
        mean = metrics["f2_mean"]
        diff = metrics["f2_absdiff"]
        lines.append(
            f"| {name} | {_fmt(mean['baseline_delta_median'])} "
            f"| {_fmt(mean['target_informative_null_fraction'])}/{_fmt(mean['baseline_informative_null_fraction'])} "
            f"| {_fmt(diff['baseline_delta_median'])} "
            f"| {_fmt(diff['target_informative_null_fraction'])}/{_fmt(diff['baseline_informative_null_fraction'])} |"
        )
    lines.extend([
        "",
        "CIs use 2,000 fixed-seed cluster bootstrap replicates over shared `icsd_reference` IDs within each stratum. Delta is target actual minus its shuffle median; negative values indicate a smaller distance than the shuffled labels. Target-baseline is paired within target. The ICSD baseline uses geometry-matched ICSD queries, and its AMD matching distance can differ from the target-to-reference AMD distance; these descriptive comparisons do not establish equivalence.",
        "",
        "Baseline deltas and target/baseline informative-null fractions for every metric are in `bootstrap_summary.json`. First-order (`f1`) distributions are invariant under label permutation, so no inference is drawn from their shuffle deltas.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results" / "per_target.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--elapsed-seconds", type=float, default=None)
    parser.add_argument("--allocated-cpus", type=int, default=None)
    parser.add_argument("--total-cpu-seconds", type=float, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    input_path = args.input.expanduser().resolve()
    output_dir = (args.output_dir or input_path.parent).expanduser().resolve()

    with input_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No result rows in {input_path}")
    missing = [field for field in REQUIRED_BASE if field not in rows[0]]
    for family in FAMILIES:
        missing.extend(
            field for field in (
                f"{family}_actual", f"{family}_delta", f"{family}_shuffle_sd",
                f"baseline_{family}_actual", f"baseline_{family}_delta",
                f"baseline_{family}_shuffle_sd",
            ) if field not in rows[0]
        )
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(set(missing))))
    material_ids = [row["material_id"] for row in rows]
    if len(material_ids) != len(set(material_ids)):
        raise ValueError("per_target.csv contains duplicate material_id values")

    group_names = list(dict.fromkeys(row["cell"] for row in rows))
    if len(group_names) != 4 or any(not name for name in group_names):
        raise ValueError(f"Expected four nonempty cell groups, found {group_names}")
    rng = np.random.default_rng(SEED)
    group_results: dict[str, object] = {}
    for name in group_names:
        group_rows = [row for row in rows if row["cell"] == name]
        indices = _bootstrap_indices(group_rows, rng)
        group_results[name] = {
            "n_targets": len(group_rows),
            "n_reference_clusters": len({row["icsd_reference"] for row in group_rows}),
            "amd_distance": _distribution_summary(np.asarray([_number(row, "amd_distance") for row in group_rows])),
            "icsd_baseline_amd_distance": _distribution_summary(np.asarray([_number(row, "icsd_baseline_amd_distance") for row in group_rows])),
            "metrics": {family: _metric_summary(group_rows, family, indices) for family in FAMILIES},
        }

    result_dir = input_path.parent
    pilot_summary_path = result_dir / "summary.json"
    provenance_path = result_dir / "provenance.json"
    pilot_summary = json.loads(pilot_summary_path.read_text()) if pilot_summary_path.exists() else {}
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    execution_parts = []
    job_id = provenance.get("slurm_job_id")
    if job_id:
        execution_parts.append(f"Pilot job `{job_id}`")
    if "runtime_seconds" in pilot_summary:
        execution_parts.append(f"algorithm runtime {float(pilot_summary['runtime_seconds']):.3f} s")
    if args.elapsed_seconds is not None:
        execution_parts.append(f"elapsed {args.elapsed_seconds:g} s")
    if args.allocated_cpus is not None:
        execution_parts.append(f"allocation {args.allocated_cpus} CPUs")
    if args.elapsed_seconds is not None and args.allocated_cpus is not None:
        core_seconds = args.elapsed_seconds * args.allocated_cpus
        execution_parts.append(f"{core_seconds:g} allocated core-seconds ({core_seconds / 3600:.2f} core-hours)")
    if args.total_cpu_seconds is not None:
        execution_parts.append(f"TotalCPU {args.total_cpu_seconds:g} s")
    pilot_execution_report = "; ".join(execution_parts) + "." if execution_parts else "Pilot execution metadata unavailable."

    result: dict[str, object] = {
        "input_path": str(input_path),
        "output_directory": str(output_dir),
        "runtime_seconds": time.perf_counter() - started,
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "confidence_interval": "percentile 95%",
        "resampling_unit": "icsd_reference within each frozen cell group",
        "units": "dimensionless W1 / ICSD channel SD",
        "n_rows": len(rows),
        "strata_are_population_prevalence": False,
        "f1_shuffle_inference": False,
        "pilot_execution": {
            "slurm_job_id": job_id,
            "algorithm_runtime_seconds": pilot_summary.get("runtime_seconds"),
            "elapsed_seconds": args.elapsed_seconds,
            "allocated_cpus": args.allocated_cpus,
            "allocated_core_seconds": (
                args.elapsed_seconds * args.allocated_cpus
                if args.elapsed_seconds is not None and args.allocated_cpus is not None else None
            ),
            "allocated_core_hours": (
                args.elapsed_seconds * args.allocated_cpus / 3600
                if args.elapsed_seconds is not None and args.allocated_cpus is not None else None
            ),
            "total_cpu_seconds": args.total_cpu_seconds,
        },
        "pilot_execution_report": pilot_execution_report,
        "groups": group_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bootstrap_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "README_results.md").write_text(_markdown(result))


if __name__ == "__main__":
    main()
