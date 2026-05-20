#!/usr/bin/env python3
"""Project an arbitrary CIF-zip bundle into the frozen ICSD structural map.

Producer used for the MatterGen-public release in the Nature manuscript;
the script is generic and can ingest any zip of CIFs (``--dataset-label``
controls the output filename slug).

Pipeline (one CIF):
  1. read CIF text from the zip member,
  2. parse with pymatgen (skip CIFs above ``--max-sites``),
  3. embed via ``icsd_densify_worker.build_structure_embedding`` with
     ``--wl-iters`` rounds of Voronoi-graph message passing,
  4. project into the frozen ICSD PCA basis,
  5. assign to the nearest Louvain community,
  6. classify as ``outlier_like`` (frontier) if the nearest-centroid
     distance exceeds the 95th-percentile within-community threshold.

Outputs (under ``--output-dir``):
  ``{label}_frontier_records.csv``   per-CIF table with columns
                                     ``material_id, zip_member, family,
                                     reduced_formula, assigned_community,
                                     nearest_centroid_distance,
                                     outlier_like, pca1, pca2``.
                                     The shared subset matches the four
                                     other ``analyze_*_frontier.py``
                                     producers; ``zip_member`` and
                                     ``family`` are extras.
  ``{label}_frontier_summary.json``  aggregate counts and rates.
  ``{label}_frontier_failures.json`` per-CIF parse / embedding errors.
  ``{label}_frontier_pca.png``       overlay scatter for QA.

Manuscript invocation (MatterGen) is recorded at
``scripts/tacc/run_mattergen_frontier_skxdev.sh``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import random
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from icsd_densify_worker import build_structure_embedding


ZIP_HANDLE: zipfile.ZipFile | None = None
WL_ITERS = 3
LOCAL_MODE = "matminer_ops"


@dataclass
class ExternalRecord:
    material_id: str
    zip_member: str
    family: str
    reduced_formula: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project an arbitrary CIF zip bundle into the frozen ICSD structural map.")
    parser.add_argument("--icsd-features", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--cif-zip", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--sample-size", type=int, default=0, help="0 means use all matching CIFs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sites", type=int, default=256)
    parser.add_argument("--wl-iters", type=int, default=3)
    parser.add_argument("--local-mode", choices=["matminer_ops", "fast_local"], default="matminer_ops")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--include-pattern", default=".cif")
    parser.add_argument("--exclude-pattern", default="symmetrized")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def load_community_rows(path: Path) -> list[dict[str, int | None]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "icsd_id": parse_int(row.get("icsd_id", "")),
                    "year": parse_int(row.get("year", "")),
                    "community": parse_int(row.get("community", "")),
                }
            )
    return rows


def centroid_thresholds(Xp: np.ndarray, labels: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
    communities = sorted({c for c in labels if c is not None and c >= 0})
    index = {c: i for i, c in enumerate(communities)}
    centroids = np.zeros((len(communities), Xp.shape[1]), dtype=float)
    counts = np.zeros(len(communities), dtype=float)
    for x, c in zip(Xp, labels):
        if c is None or c < 0:
            continue
        idx = index[c]
        centroids[idx] += x
        counts[idx] += 1
    for i in range(len(communities)):
        centroids[i] /= max(counts[i], 1.0)

    within = []
    for x, c in zip(Xp, labels):
        if c is None or c < 0:
            continue
        d = float(np.linalg.norm(x - centroids[index[c]]))
        within.append(d)
    threshold = float(np.quantile(within, 0.95)) if within else 0.0
    return np.array(communities, dtype=int), centroids, threshold


def member_family(name: str) -> str:
    parts = [p for p in name.split("/") if p]
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def member_id(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def load_records(zip_path: Path, include_pattern: str, exclude_pattern: str, sample_size: int, seed: int) -> list[ExternalRecord]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            n
            for n in zf.namelist()
            if n.endswith(".cif")
            and include_pattern in n
            and (not exclude_pattern or exclude_pattern not in n)
        ]
    records = [
        ExternalRecord(
            material_id=member_id(name),
            zip_member=name,
            family=member_family(name),
            reduced_formula=None,
        )
        for name in names
    ]
    if sample_size and sample_size < len(records):
        rng = random.Random(seed)
        rng.shuffle(records)
        records = records[:sample_size]
    return records


def init_worker(zip_path: str, wl_iters: int, local_mode: str) -> None:
    global ZIP_HANDLE
    global WL_ITERS
    global LOCAL_MODE
    ZIP_HANDLE = zipfile.ZipFile(zip_path)
    WL_ITERS = wl_iters
    LOCAL_MODE = local_mode


def featurize_record(rec: ExternalRecord) -> tuple[bool, ExternalRecord, np.ndarray | None, str | None]:
    if ZIP_HANDLE is None:
        raise RuntimeError("ZIP handle not initialized")
    try:
        with ZIP_HANDLE.open(rec.zip_member) as handle:
            text = handle.read().decode("utf-8", errors="replace")
        structure = Structure.from_str(text, fmt="cif")
        emb = build_structure_embedding(structure, wl_iters=WL_ITERS, local_mode=LOCAL_MODE)
        rec.reduced_formula = structure.composition.reduced_formula
        return True, rec, emb, None
    except Exception as exc:
        return False, rec, None, str(exc)[:200]


def plot_frontier(ic_x2: np.ndarray, ai_x2: np.ndarray, outlier_like: np.ndarray, out_path: Path, label: str) -> None:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ic_x2), size=min(25000, len(ic_x2)), replace=False)
    bg = ic_x2[idx]

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.scatter(bg[:, 0], bg[:, 1], s=3, c="#c7c7c7", alpha=0.25, linewidths=0)
    mask_in = ~outlier_like
    ax.scatter(ai_x2[mask_in, 0], ai_x2[mask_in, 1], s=8, c="#1b9e77", alpha=0.6, linewidths=0, label=f"{label} in existing basin")
    ax.scatter(ai_x2[ai_out := outlier_like, 0], ai_x2[ai_out, 1], s=10, c="#d95f02", alpha=0.8, linewidths=0, label=f"{label} outlier-like")
    ax.set_xlabel("Frozen ICSD PCA-1")
    ax.set_ylabel("Frozen ICSD PCA-2")
    ax.set_title(f"{label} structures on the frozen ICSD structural map")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.icsd_features)
    community_rows = load_community_rows(Path(args.community_assignments))
    if len(X) != len(community_rows):
        raise ValueError(f"ICSD features rows ({len(X)}) != community rows ({len(community_rows)})")
    community_labels = [int(r["community"]) if r["community"] is not None else -1 for r in community_rows]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca32 = PCA(n_components=32, random_state=args.seed)
    Xp = pca32.fit_transform(Xs)
    pca2 = PCA(n_components=2, random_state=args.seed)
    X2 = pca2.fit_transform(Xs)
    communities, centroids, threshold = centroid_thresholds(Xp, community_labels)

    records = load_records(
        Path(args.cif_zip),
        include_pattern=args.include_pattern,
        exclude_pattern=args.exclude_pattern,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.n_jobs,
        initializer=init_worker,
        initargs=(args.cif_zip, args.wl_iters, args.local_mode),
    ) as ex:
        for item in ex.map(featurize_record, records, chunksize=16):
            results.append(item)

    kept = [(rec, emb) for ok, rec, emb, _ in results if ok and emb is not None]
    failures = [
        {"material_id": rec.material_id, "zip_member": rec.zip_member, "family": rec.family, "detail": err}
        for ok, rec, _, err in results
        if not ok
    ]
    if not kept:
        raise RuntimeError("No external structures featurized successfully")

    recs = [rec for rec, _ in kept]
    Y = np.vstack([emb for _, emb in kept])
    Ys = scaler.transform(Y)
    Yp = pca32.transform(Ys)
    Y2 = pca2.transform(Ys)

    dmat = np.linalg.norm(Yp[:, None, :] - centroids[None, :, :], axis=2)
    nearest_idx = np.argmin(dmat, axis=1)
    nearest_comm = communities[nearest_idx]
    nearest_dist = dmat[np.arange(len(Yp)), nearest_idx]
    outlier_like = nearest_dist > threshold

    top_communities = Counter(int(c) for c in nearest_comm[~outlier_like]).most_common(15)
    family_counts = Counter(rec.family for rec in recs)
    summary = {
        "dataset_label": args.dataset_label,
        "sample_size_requested": args.sample_size,
        "n_records_loaded": len(records),
        "n_featurized": len(recs),
        "n_failures": len(failures),
        "icsd_centroid_distance_threshold_p95": threshold,
        "n_outlier_like": int(np.count_nonzero(outlier_like)),
        "outlier_like_ratio": float(np.mean(outlier_like)),
        "top_existing_communities": top_communities,
        "mean_nearest_centroid_distance": float(np.mean(nearest_dist)),
        "family_counts": dict(family_counts),
    }

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.dataset_label.lower())
    (out_dir / f"{slug}_frontier_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{slug}_frontier_failures.json").write_text(json.dumps(failures[:500], indent=2))

    with (out_dir / f"{slug}_frontier_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "material_id",
                "zip_member",
                "family",
                "reduced_formula",
                "assigned_community",
                "nearest_centroid_distance",
                "outlier_like",
                "pca1",
                "pca2",
            ],
        )
        writer.writeheader()
        for rec, comm, dist, is_out, xy in zip(recs, nearest_comm, nearest_dist, outlier_like, Y2):
            writer.writerow(
                {
                    "material_id": rec.material_id,
                    "zip_member": rec.zip_member,
                    "family": rec.family,
                    "reduced_formula": rec.reduced_formula,
                    "assigned_community": int(comm),
                    "nearest_centroid_distance": float(dist),
                    "outlier_like": bool(is_out),
                    "pca1": float(xy[0]),
                    "pca2": float(xy[1]),
                }
            )

    plot_frontier(X2, Y2, outlier_like, out_dir / f"{slug}_frontier_pca.png", args.dataset_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
