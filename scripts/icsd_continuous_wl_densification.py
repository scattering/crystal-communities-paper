#!/usr/bin/env python3
"""
Prototype ICSD chronology / densification pipeline.

This script is designed for small-sample TACC validation before a full ICSD run.
It:
1. Reads ICSD metadata from a CSV.
2. Reads CIFs directly from the ICSD zip archive.
3. Builds a lightweight continuous-WL style embedding from:
   - occupancy-weighted elemental features
   - local CrystalNN geometry summaries
   - a few global lattice features
4. Clusters embeddings with HDBSCAN.
5. Computes chronology-aware exploration vs densification summaries.

The implementation is intentionally lightweight and explicit rather than
optimized for maximum throughput.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import random
import sys
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS

try:
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime on TACC
    hdbscan = None

from icsd_densify_worker import (
    ID_COLUMN_CANDIDATES,
    YEAR_COLUMN_CANDIDATES,
    Record,
    featurize_record,
    featurize_record_batch,
    find_column,
    init_worker,
    parse_icsd_id,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype ICSD continuous-WL densification analysis.")
    parser.add_argument("--icsd-zip", help="Path to ICSD_CIFs.zip")
    password_group = parser.add_mutually_exclusive_group()
    password_group.add_argument("--zip-password", help="Password for encrypted ICSD zip members (legacy; prefer --zip-password-stdin)")
    password_group.add_argument(
        "--zip-password-stdin",
        action="store_true",
        help="Read the encrypted archive password from one line of standard input",
    )
    parser.add_argument("--cif-root", help="Path to extracted FindIt_CIFs directory")
    parser.add_argument("--index-csv", required=True, help="Path to ICSD_index.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument("--sample-size", type=int, default=250, help="Number of ICSD entries to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sites", type=int, default=256, help="Skip structures larger than this many sites")
    parser.add_argument("--wl-iters", type=int, default=3)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel worker processes for structure featurization")
    parser.add_argument("--chunk-size", type=int, default=32, help="Records per dispatched batch in process mode")
    parser.add_argument(
        "--local-mode",
        choices=["matminer_ops", "fast_local", "chem_only"],
        default="matminer_ops",
        help="Local geometry descriptor backend",
    )
    parser.add_argument("--cache-dir", help="Optional directory for cached per-structure embeddings")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Write lightweight progress checkpoints every N newly processed structures",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["thread", "process"],
        default="thread",
        help="Parallel backend for structure featurization",
    )
    parser.add_argument("--run-umap", action="store_true", help="Also compute a 2D UMAP for visualization")
    args = parser.parse_args(argv)
    if args.zip_password_stdin:
        args.zip_password = sys.stdin.readline().rstrip("\r\n")
        if not args.zip_password:
            parser.error("--zip-password-stdin requires a non-empty input line")
    return args


def configure_warnings() -> None:
    noisy_modules = [
        r".*pymatgen\.core\.structure",
        r".*pymatgen\.io\.cif",
        r".*pymatgen\.core\.local_env",
        r".*matminer\.featurizers\.site\.fingerprint",
    ]
    for module in noisy_modules:
        warnings.filterwarnings("ignore", category=UserWarning, module=module)
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*This module has been moved to the pymatgen\.core\.local_env module.*",
    )


def feature_metadata(args: argparse.Namespace) -> dict:
    """Whitelist feature provenance; never serialize the argument namespace."""
    return {
        "feature_version": FEATURE_VERSION,
        "neighbor_settings": dict(NEIGHBOR_SETTINGS),
        "local_mode": args.local_mode,
        "wl_iters": int(args.wl_iters),
        "max_sites": int(args.max_sites),
    }


def feature_metadata_json(args: argparse.Namespace) -> str:
    return json.dumps(feature_metadata(args), sort_keys=True, separators=(",", ":"))


def cache_key(rec: Record, args: argparse.Namespace) -> str:
    settings_hash = hashlib.sha256(feature_metadata_json(args).encode("utf-8")).hexdigest()[:16]
    return (
        f"icsd_{rec.icsd_id:06d}"
        f"_{FEATURE_VERSION}"
        f"_{settings_hash}"
        ".npz"
    )


def load_cached_result(cache_dir: Path, rec: Record, args: argparse.Namespace):
    path = cache_dir / cache_key(rec, args)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["feature_metadata"]) != feature_metadata_json(args):
                return None
            status = str(data["status"])
            if status == "ok":
                embedding = data["embedding"]
                if embedding.ndim != 1 or not np.isfinite(embedding).all():
                    return None
                diagnostics = feature_diagnostics(json.loads(str(data["feature_diagnostics"])))
                return True, rec, embedding, diagnostics
            if status != "fail":
                return None
            failure = {"icsd_id": rec.icsd_id, "reason": str(data["reason"]), "detail": str(data["detail"])}
            return False, rec, None, safe_failure(failure, args)
    except Exception:
        return None


def store_cached_result(cache_dir: Path, rec: Record, args: argparse.Namespace, ok: bool, emb, failure):
    path = cache_dir / cache_key(rec, args)
    if ok:
        np.savez_compressed(
            path,
            status="ok",
            embedding=emb,
            feature_metadata=feature_metadata_json(args),
            feature_diagnostics=json.dumps(feature_diagnostics(failure), sort_keys=True),
        )
    else:
        failure = safe_failure(failure, args)
        np.savez_compressed(
            path,
            status="fail",
            reason=failure.get("reason", "unknown"),
            detail=failure.get("detail", ""),
            feature_metadata=feature_metadata_json(args),
        )


def safe_failure(failure: dict, args: argparse.Namespace) -> dict:
    """Do not persist arbitrary exception text from a password-protected input.

    Exception details can include an escaped or truncated credential, so exact
    string replacement alone is insufficient. The ICSD id and exception type
    remain available for diagnosing protected-input failures.
    """
    secret = getattr(args, "zip_password", None)
    reason = str(failure.get("reason", "unknown"))
    if secret:
        reason = reason.replace(secret, "[redacted]")
    return {
        "icsd_id": failure.get("icsd_id"),
        "reason": reason,
        "detail": "[protected-input exception detail omitted]" if secret else str(failure.get("detail", "")),
    }


def feature_diagnostics(info: dict) -> dict:
    """Keep only numeric/boolean worker diagnostics in persistent artifacts."""
    return {
        "n_sites": int(info["n_sites"]),
        "ordered": bool(info["ordered"]),
        "sites_with_unrepresented_cn_mass": int(info["sites_with_unrepresented_cn_mass"]),
        "max_unrepresented_cn_mass": float(info["max_unrepresented_cn_mass"]),
    }


def summarize_feature_diagnostics(diagnostics: list[dict]) -> dict:
    return {
        "n_structures": len(diagnostics),
        "n_disordered": sum(not info["ordered"] for info in diagnostics),
        "n_structures_with_unrepresented_cn_mass": sum(
            info["sites_with_unrepresented_cn_mass"] > 0 for info in diagnostics
        ),
        "n_sites_with_unrepresented_cn_mass": sum(
            info["sites_with_unrepresented_cn_mass"] for info in diagnostics
        ),
        "max_unrepresented_cn_mass": max(
            (info["max_unrepresented_cn_mass"] for info in diagnostics), default=0.0
        ),
    }


def batched(items: list[Record], size: int) -> list[list[Record]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def atomic_write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        temp_name = tmp.name
    Path(temp_name).replace(path)


def write_progress_checkpoint(
    out_dir: Path,
    sample_size: int,
    kept_records: list[Record],
    failures: list[dict],
    processed_new: int,
    pending_remaining: int,
    args: argparse.Namespace,
    diagnostics: list[dict],
) -> None:
    top_failures = Counter(item.get("reason", "unknown") for item in failures).most_common(10)
    payload = {
        **feature_metadata(args),
        "feature_diagnostics": summarize_feature_diagnostics(diagnostics),
        "sample_size_requested": int(sample_size),
        "n_processed_total": int(len(kept_records) + len(failures)),
        "n_success": int(len(kept_records)),
        "n_failures": int(len(failures)),
        "n_processed_new_this_run": int(processed_new),
        "n_pending_remaining": int(pending_remaining),
        "top_failure_reasons": top_failures,
    }
    atomic_write_json(out_dir / "progress.json", payload)


def load_index_records(index_csv: Path) -> list[Record]:
    with open(index_csv, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {index_csv}")
        id_col = find_column(reader.fieldnames, ID_COLUMN_CANDIDATES)
        year_col = find_column(reader.fieldnames, YEAR_COLUMN_CANDIDATES)
        if id_col is None:
            raise ValueError(f"Could not find an ICSD id column in {index_csv}")

        records: list[Record] = []
        for row in reader:
            raw_id = str(row.get(id_col, "")).strip()
            icsd_id = parse_icsd_id(raw_id)
            if icsd_id is None:
                continue

            year = None
            if year_col:
                raw_year = str(row.get(year_col, "")).strip()
                if raw_year:
                    try:
                        year = int(float(raw_year))
                    except ValueError:
                        year = None
            records.append(Record(icsd_id=icsd_id, year=year, row=row))
    return records


def decade_from_year(year: int | None) -> str:
    if year is None:
        return "unknown"
    return f"{(year // 10) * 10}s"


def summarize_densification(records: list[Record], labels: np.ndarray) -> dict:
    cluster_birth_year: dict[int, int] = {}
    for rec, label in zip(records, labels):
        if label < 0 or rec.year is None:
            continue
        if label not in cluster_birth_year:
            cluster_birth_year[label] = rec.year
        else:
            cluster_birth_year[label] = min(cluster_birth_year[label], rec.year)

    by_decade: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cluster_sizes = Counter(int(v) for v in labels if int(v) >= 0)

    for rec, label in zip(records, labels):
        decade = decade_from_year(rec.year)
        by_decade[decade]["n_total"] += 1
        if int(label) < 0:
            by_decade[decade]["n_outlier"] += 1
            continue
        birth = cluster_birth_year.get(int(label))
        if birth is not None and rec.year is not None and rec.year > birth:
            by_decade[decade]["n_existing_cluster"] += 1
        elif birth is not None and rec.year == birth:
            by_decade[decade]["n_cluster_birth_point"] += 1

    for decade, stats in by_decade.items():
        total = stats["n_total"] or 1.0
        stats["outlier_ratio"] = stats["n_outlier"] / total
        stats["existing_cluster_ratio"] = stats["n_existing_cluster"] / total
        stats["cluster_birth_point_ratio"] = stats["n_cluster_birth_point"] / total

    return {
        "n_clusters": int(len(cluster_sizes)),
        "largest_clusters": cluster_sizes.most_common(20),
        "cluster_birth_year": {str(k): int(v) for k, v in sorted(cluster_birth_year.items())},
        "by_decade": {k: dict(v) for k, v in sorted(by_decade.items())},
    }


def main() -> int:
    configure_warnings()
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.icsd_zip and not args.cif_root:
        print("ERROR: provide either --icsd-zip or --cif-root.", file=sys.stderr)
        return 2

    if hdbscan is None:
        print("ERROR: hdbscan is not installed in the active environment.", file=sys.stderr)
        return 2

    records = load_index_records(Path(args.index_csv))
    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.sample_size, len(records)))

    features = []
    kept_records: list[Record] = []
    diagnostics: list[dict] = []
    failures = []
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    pending: list[Record] = []
    for rec in sample:
        cached = load_cached_result(cache_dir, rec, args) if cache_dir is not None else None
        if cached is None:
            pending.append(rec)
            continue
        ok, kept_rec, emb, failure = cached
        if ok:
            features.append(emb)
            kept_records.append(kept_rec)
            diagnostics.append(feature_diagnostics(failure))
        else:
            failures.append(safe_failure(failure, args))

    processed_new = 0
    write_progress_checkpoint(out_dir, args.sample_size, kept_records, failures, processed_new, len(pending), args, diagnostics)

    if args.n_jobs <= 1:
        init_worker(args.icsd_zip, args.zip_password, args.cif_root, args.wl_iters, args.max_sites, args.local_mode)
        for rec in pending:
            ok, kept_rec, emb, failure = featurize_record(rec)
            if ok:
                features.append(emb)
                kept_records.append(kept_rec)
                diagnostics.append(feature_diagnostics(failure))
            else:
                failures.append(safe_failure(failure, args))
            if cache_dir is not None:
                store_cached_result(cache_dir, rec, args, ok, emb, failure)
            processed_new += 1
            if processed_new % args.checkpoint_every == 0:
                write_progress_checkpoint(
                    out_dir,
                    args.sample_size,
                    kept_records,
                    failures,
                    processed_new,
                    max(len(pending) - processed_new, 0),
                    args,
                    diagnostics,
                )
    else:
        executor_cls: type[concurrent.futures.Executor]
        executor_kwargs: dict = {"max_workers": args.n_jobs}
        if args.parallel_backend == "process":
            executor_cls = concurrent.futures.ProcessPoolExecutor
            executor_kwargs["initializer"] = init_worker
            executor_kwargs["initargs"] = (
                args.icsd_zip,
                args.zip_password,
                args.cif_root,
                args.wl_iters,
                args.max_sites,
                args.local_mode,
            )
        else:
            init_worker(args.icsd_zip, args.zip_password, args.cif_root, args.wl_iters, args.max_sites, args.local_mode)
            executor_cls = concurrent.futures.ThreadPoolExecutor

        with executor_cls(**executor_kwargs) as executor:
            if args.parallel_backend == "process":
                for batch, results in zip(
                    batched(pending, args.chunk_size),
                    executor.map(featurize_record_batch, batched(pending, args.chunk_size), chunksize=1),
                ):
                    for rec, (ok, kept_rec, emb, failure) in zip(batch, results):
                        if ok:
                            features.append(emb)
                            kept_records.append(kept_rec)
                            diagnostics.append(feature_diagnostics(failure))
                        else:
                            failures.append(safe_failure(failure, args))
                        if cache_dir is not None:
                            store_cached_result(cache_dir, rec, args, ok, emb, failure)
                        processed_new += 1
                    if processed_new % args.checkpoint_every == 0:
                        write_progress_checkpoint(
                            out_dir,
                            args.sample_size,
                            kept_records,
                            failures,
                            processed_new,
                            max(len(pending) - processed_new, 0),
                            args,
                            diagnostics,
                        )
            else:
                for rec, (ok, kept_rec, emb, failure) in zip(pending, executor.map(featurize_record, pending)):
                    if ok:
                        features.append(emb)
                        kept_records.append(kept_rec)
                        diagnostics.append(feature_diagnostics(failure))
                    else:
                        failures.append(safe_failure(failure, args))
                    if cache_dir is not None:
                        store_cached_result(cache_dir, rec, args, ok, emb, failure)
                    processed_new += 1
                    if processed_new % args.checkpoint_every == 0:
                        write_progress_checkpoint(
                            out_dir,
                            args.sample_size,
                            kept_records,
                            failures,
                            processed_new,
                            max(len(pending) - processed_new, 0),
                            args,
                            diagnostics,
                        )

    write_progress_checkpoint(out_dir, args.sample_size, kept_records, failures, processed_new, 0, args, diagnostics)

    with open(out_dir / "failures.json", "w") as handle:
        json.dump(failures, handle, indent=2)

    if not features:
        top_failures = Counter(item["reason"] for item in failures).most_common(10)
        print(json.dumps({"n_failures": len(failures), "top_failure_reasons": top_failures}, indent=2), file=sys.stderr)
        print("ERROR: no structures were featurized successfully.", file=sys.stderr)
        return 3

    X = np.vstack(features)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca_dim = min(args.pca_dim, Xs.shape[0], Xs.shape[1])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    Xp = pca.fit_transform(Xs)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(Xp)

    summary = {
        **feature_metadata(args),
        "feature_diagnostics": summarize_feature_diagnostics(diagnostics),
        "sample_size_requested": int(args.sample_size),
        "sample_size_featurized": int(len(kept_records)),
        "n_failures": int(len(failures)),
        "n_outliers": int(np.sum(labels < 0)),
        "outlier_ratio": float(np.mean(labels < 0)),
        "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "densification": summarize_densification(kept_records, labels),
    }

    if args.run_umap:
        try:
            import umap  # type: ignore
        except ImportError:
            umap = None
        if umap is None:
            print("WARNING: --run-umap requested but umap-learn is not installed.", file=sys.stderr)
        else:
            reducer = umap.UMAP(n_components=2, random_state=args.seed)
            um = reducer.fit_transform(Xp)
            np.save(out_dir / "umap_2d.npy", um)

    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_dir / "sample_assignments.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["icsd_id", "year", "cluster"])
        for rec, label in zip(kept_records, labels):
            writer.writerow([rec.icsd_id, rec.year if rec.year is not None else "", int(label)])

    np.save(out_dir / "features.npy", X)
    np.save(out_dir / "features_pca.npy", Xp)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
