#!/usr/bin/env python3
"""Project the GNoME public structure release onto the frozen ICSD map.

Producer for ``gnome_frontier_records.csv`` consumed by Figure 3 (the
``make_fig_5source_calibration.py`` 5-source calibration figure) and
by Figure 4 (the ``make_fig_synth_prior_quadrant.py`` 2x2 synthesizability
quadrant). The schema of the output CSV is the canonical reference for
the four sister producers (``analyze_{mp,jarvis,alexandria}_frontier.py``)
which all emit a strict superset of these columns.

Pipeline (one CIF, sub-process pool):
  1. read CIF text from the GNoME zip member,
  2. parse with pymatgen,
  3. embed via ``icsd_densify_worker.build_structure_embedding`` with
     ``--wl-iters`` rounds of Voronoi-graph message passing
     (matching the production ICSD featurization),
  4. project into the frozen ICSD PCA basis (loaded from
     ``--icsd-features``, the same .npy used by every figure script),
  5. assign to the nearest Louvain community (centroids loaded from
     ``--community-assignments``),
  6. classify as ``outlier_like`` (frontier) if the nearest-centroid
     distance exceeds the 95th-percentile within-community threshold.

The ``ZIP_HANDLE`` module global is intentional: it carries the open
``zipfile.ZipFile`` into worker subprocesses so each worker doesn't
re-open the zip per-CIF. This pattern is replicated in the four sister
producers; the comment block at the top of
``analyze_external_cif_zip_frontier.py`` documents the same idiom.

Outputs (under ``--output-dir``):
  ``gnome_frontier_records.csv``    per-CIF table.
  ``gnome_frontier_summary.json``   aggregate counts, p95 thresholds.
  ``gnome_frontier_failures.json``  per-CIF parse / embedding errors.
  ``gnome_frontier_pca.png``        overlay scatter for QA.

Manuscript invocation: ``scripts/tacc/run_gnome_frontier_skxdev.sh``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import random
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


GNOME_ZIP: zipfile.ZipFile | None = None
GNOME_WL_ITERS = 3
GNOME_LOCAL_MODE = "matminer_ops"


@dataclass
class GNoMERecord:
    material_id: str
    reduced_formula: str
    decomp_e: float | None
    nsites: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project GNoME structures into the frozen ICSD structural map.")
    parser.add_argument("--icsd-features", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--gnome-summary-csv", required=True)
    parser.add_argument("--gnome-by-id-zip", required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sites", type=int, default=256)
    parser.add_argument("--wl-iters", type=int, default=3)
    parser.add_argument("--local-mode", choices=["matminer_ops", "fast_local"], default="matminer_ops")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


from frontier_common import (
    parse_int,
    parse_float,
    load_community_rows,
    centroid_thresholds,
)


def load_gnome_summary(path: Path, sample_size: int, seed: int, max_sites: int) -> list[GNoMERecord]:
    records = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            material_id = (row.get("MaterialId") or row.get("material_id") or "").strip()
            reduced_formula = (row.get("Reduced Formula") or row.get("reduced_formula") or "").strip()
            nsites = parse_int(row.get("NSites", "") or row.get("n_sites", ""))
            if not material_id or not reduced_formula:
                continue
            if nsites is not None and nsites > max_sites:
                continue
            records.append(
                GNoMERecord(
                    material_id=material_id,
                    reduced_formula=reduced_formula,
                    decomp_e=parse_float(
                        row.get("Decomposition Energy Per Atom All", "")
                        or row.get("Decomposition Energy Per Atom", "")
                        or row.get("decomposition_energy_per_atom", "")
                    ),
                    nsites=nsites,
                )
            )
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:sample_size]


def init_gnome_worker(zip_path: str, wl_iters: int, local_mode: str) -> None:
    global GNOME_ZIP
    global GNOME_WL_ITERS
    global GNOME_LOCAL_MODE
    GNOME_ZIP = zipfile.ZipFile(zip_path)
    GNOME_WL_ITERS = wl_iters
    GNOME_LOCAL_MODE = local_mode


def open_material_from_zip(material_id: str) -> Structure:
    if GNOME_ZIP is None:
        raise RuntimeError("GNoME zip not initialized")
    candidates = [
        f"{material_id}.cif",
        f"{material_id}.CIF",
        f"by_id/{material_id}.cif",
        f"by_id/{material_id}.CIF",
        f"{material_id}.vasp.cif",
    ]
    for name in candidates:
        try:
            with GNOME_ZIP.open(name) as handle:
                text = handle.read().decode("utf-8", errors="replace")
            return Structure.from_str(text, fmt="cif")
        except KeyError:
            continue
    raise KeyError(material_id)


def featurize_gnome_record(rec: GNoMERecord) -> tuple[bool, GNoMERecord, np.ndarray | None, str | None]:
    try:
        structure = open_material_from_zip(rec.material_id)
        emb = build_structure_embedding(
            structure,
            wl_iters=GNOME_WL_ITERS,
            local_mode=GNOME_LOCAL_MODE,
        )
        return True, rec, emb, None
    except Exception as exc:
        return False, rec, None, str(exc)[:200]


def plot_frontier(
    ic_x2: np.ndarray,
    ic_labels: np.ndarray,
    ai_x2: np.ndarray,
    ai_outlier_like: np.ndarray,
    out_path: Path,
) -> None:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ic_x2), size=min(25000, len(ic_x2)), replace=False)
    bg = ic_x2[idx]

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    # The dashboard surface frames this figure as AI vs Human, so the legend
    # must explicitly name the human-ICSD background instead of leaving it as
    # an unlabeled scatter underneath.
    ax.scatter(
        bg[:, 0], bg[:, 1],
        s=3, c="#c7c7c7", alpha=0.25, linewidths=0,
        label="Human (ICSD background)",
    )
    mask_in = ~ai_outlier_like
    ax.scatter(
        ai_x2[mask_in, 0], ai_x2[mask_in, 1],
        s=8, c="#1b9e77", alpha=0.6, linewidths=0,
        label="AI: GNoME → existing basin",
    )
    ax.scatter(
        ai_x2[ai_outlier_like, 0], ai_x2[ai_outlier_like, 1],
        s=10, c="#d95f02", alpha=0.8, linewidths=0,
        label="AI: GNoME → outlier-like",
    )
    ax.set_xlabel("Frozen ICSD PCA-1")
    ax.set_ylabel("Frozen ICSD PCA-2")
    ax.set_title("AI (GNoME) vs Human (ICSD) on the frozen structural map")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
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

    gnome_records = load_gnome_summary(
        Path(args.gnome_summary_csv),
        sample_size=args.sample_size,
        seed=args.seed,
        max_sites=args.max_sites,
    )

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.n_jobs,
        initializer=init_gnome_worker,
        initargs=(args.gnome_by_id_zip, args.wl_iters, args.local_mode),
    ) as ex:
        for item in ex.map(featurize_gnome_record, gnome_records, chunksize=16):
            results.append(item)

    kept = [(rec, emb) for ok, rec, emb, _ in results if ok and emb is not None]
    failures = [
        {"material_id": rec.material_id, "formula": rec.reduced_formula, "detail": err}
        for ok, rec, _, err in results
        if not ok
    ]
    if not kept:
        raise RuntimeError("No GNoME structures featurized successfully")

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
    summary = {
        "sample_size_requested": args.sample_size,
        "n_summary_records_loaded": len(gnome_records),
        "n_featurized": len(recs),
        "n_failures": len(failures),
        "icsd_centroid_distance_threshold_p95": threshold,
        "n_outlier_like": int(np.count_nonzero(outlier_like)),
        "outlier_like_ratio": float(np.mean(outlier_like)),
        "top_existing_communities": top_communities,
        "mean_nearest_centroid_distance": float(np.mean(nearest_dist)),
    }

    (out_dir / "gnome_frontier_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "gnome_frontier_failures.json").write_text(json.dumps(failures[:500], indent=2))

    with (out_dir / "gnome_frontier_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "material_id",
                "reduced_formula",
                "decomposition_energy_per_atom",
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
                    "reduced_formula": rec.reduced_formula,
                    "decomposition_energy_per_atom": rec.decomp_e,
                    "assigned_community": int(comm),
                    "nearest_centroid_distance": float(dist),
                    "outlier_like": bool(is_out),
                    "pca1": float(xy[0]),
                    "pca2": float(xy[1]),
                }
            )

    plot_frontier(X2, np.array(community_labels), Y2, outlier_like, out_dir / "gnome_frontier_pca.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
