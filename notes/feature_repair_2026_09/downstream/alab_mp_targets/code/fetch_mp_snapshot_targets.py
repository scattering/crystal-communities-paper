#!/usr/bin/env python3
"""Fetch the 57 A-Lab MP target documents from the public versioned MP S3 collection.

The script makes one sequential anonymous HTTPS GET per source shard, verifies the
archived SHA-256/size, and keeps only requested documents. Raw shards stay in a
user-selected scratch cache and must not be added to the repository.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, gzip, hashlib, json, os, shutil, urllib.parse, urllib.request
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
BASE_URL = "https://materialsproject-build.s3.amazonaws.com/"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", choices=("materials", "summary"), default="materials")
    p.add_argument("--targets", type=Path, default=BUNDLE / "source/alab_targets.csv")
    p.add_argument("--seed-manifest", type=Path)
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Scratch directory for the 36 compressed shards (about 185 MiB for materials).")
    p.add_argument("--out-docs", type=Path, required=True)
    p.add_argument("--out-report", type=Path, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    if args.seed_manifest is None:
        args.seed_manifest = BUNDLE / f"source/mp_2022_10_28_{args.collection}_source_manifest.json"
    seed = json.loads(args.seed_manifest.read_text())
    expected_prefix = f"collections/2022-10-28/{args.collection}/"
    if seed.get("source") != f"s3://materialsproject-build/{expected_prefix}":
        raise ValueError(f"Seed manifest is not the requested {args.collection!r} collection")
    target_rows = list(csv.DictReader(args.targets.open()))
    ids = {row["mp_id"] for row in target_rows}
    if len(target_rows) != 57 or len(ids) != 57:
        raise ValueError("Expected 57 distinct A-Lab target MP IDs")
    target_key = {row["mp_id"]: row["source_object"] for row in seed["targets"]}
    if set(target_key) != ids:
        raise ValueError("Seed manifest target IDs do not match target CSV")
    object_meta = {row["key"]: row for row in seed["source_objects"]}
    keys = sorted(set(target_key.values()))
    if len(keys) != 36 or any(not key.startswith(expected_prefix) or ".." in key for key in keys):
        raise ValueError("Unexpected source-object set")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    retrieved = []
    for index, key in enumerate(keys, 1):
        expected = object_meta[key]
        cache_name = hashlib.sha256(key.encode()).hexdigest()[:16] + ".jsonl.gz"
        cached = args.cache_dir / cache_name
        downloaded = False
        headers = {}
        if not cached.is_file() or cached.stat().st_size != int(expected["bytes"]) or sha256(cached) != expected["sha256"]:
            tmp = cached.with_suffix(".tmp")
            url = BASE_URL + urllib.parse.quote(key, safe="/=")
            request = urllib.request.Request(url, headers={"User-Agent": "crystal-communities-reproducibility/1"})
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
                headers = {name.lower(): value for name, value in response.headers.items()}
            os.replace(tmp, cached)
            downloaded = True
        actual_size = cached.stat().st_size
        actual_sha = sha256(cached)
        if actual_size != int(expected["bytes"]) or actual_sha != expected["sha256"]:
            raise RuntimeError(f"Archived checksum/size mismatch for {key}")
        wanted_here = {mid for mid, source_key in target_key.items() if source_key == key}
        with gzip.open(cached, "rt", encoding="utf-8") as handle:
            for line in handle:
                doc = json.loads(line)
                mid = str(doc.get("material_id", ""))
                if mid in wanted_here:
                    if mid in found:
                        raise RuntimeError(f"Duplicate target document {mid}")
                    found[mid] = doc
        retrieved.append({"key": key, "cache_file": str(cached), "downloaded_this_run": downloaded,
                          "bytes": actual_size, "sha256": actual_sha,
                          "etag": headers.get("etag", expected.get("etag")),
                          "last_modified": headers.get("last-modified", expected.get("last_modified"))})
        print(f"{index:02d}/{len(keys)} {key} targets={len(found)}/{len(ids)}", flush=True)
    missing = sorted(ids - set(found))
    if missing or any(not found[mid].get("structure") for mid in found):
        raise RuntimeError(f"Missing or structureless target documents: {missing}")
    ordered = {mid: found[mid] for mid in sorted(found)}
    args.out_docs.parent.mkdir(parents=True, exist_ok=True)
    args.out_docs.write_text(json.dumps(ordered, indent=2) + "\n")
    report = {"collection": args.collection, "source": seed["source"],
              "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "anonymous_https_no_credentials": True, "sequential_requests": True,
              "n_targets": len(ordered), "n_source_objects": len(keys), "missing": missing,
              "out_docs": str(args.out_docs), "out_docs_sha256": sha256(args.out_docs),
              "seed_manifest": str(args.seed_manifest), "seed_manifest_sha256": sha256(args.seed_manifest),
              "source_objects": retrieved}
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("collection", "n_targets", "n_source_objects", "missing", "out_docs_sha256")}, indent=2))

if __name__ == "__main__":
    main()
