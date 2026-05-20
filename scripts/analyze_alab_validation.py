#!/usr/bin/env python3
"""External validation of 𝒜ᵢ against A-Lab autonomous-synthesis outcomes.

Supports SI §S3.5. The A-Lab dataset (Szymanski et al. 2023) reports
autonomous synthesis outcomes (synthesis_yield / corrected_outcome) for
~58 targets selected from MP. This script projects each A-Lab target
into the frozen ICSD structural map using
``icsd_densify_worker.build_structure_embedding`` (the same featurizer
the entire manuscript uses), assigns it to a community, computes its
nearest-centroid distance and z-scored 𝒜ᵢ at ``--observation-year``
(default 2023), and tests whether *successfully synthesized* targets
have systematically lower 𝒜ᵢ than failed ones.

Pipeline (per A-Lab target CIF):
  1. Read CIF text from the A-Lab zip.
  2. Embed via ``build_structure_embedding`` with ``--wl-iters`` rounds
     (matching production ICSD featurization), ``--local-mode``
     defaulting to matminer_ops.
  3. Project into the frozen ICSD PCA basis loaded from
     ``--icsd-features``.
  4. Assign to nearest Louvain community (centroids from
     ``--community-assignments``); flag in/out of basin against the
     within-community 95th-percentile centroid-distance threshold.
  5. Score 𝒜ᵢ from ``--node-events`` history at
     ``--observation-year``.

Inputs:
  --icsd-features, --community-assignments, --node-events,
  --alab-zip, --wl-iters, --local-mode, --observation-year.

Outputs (under ``--output-dir``):
  alab_validation_records.csv      Per-target community, distance,
                                    in_basin, 𝒜ᵢ, A-Lab outcome.
  alab_validation_summary.json     Group means by outcome.
  alab_validation_failures.json    Per-target parse / embedding errors.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from icsd_densify_worker import build_structure_embedding


@dataclass
class AlabTarget:
    formula: str
    raw_result: str
    mp_id: str
    corrected_outcome: str
    conclusion: str
    cif_member: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project A-Lab targets into the frozen ICSD structural map.")
    parser.add_argument("--icsd-features", required=True)
    parser.add_argument("--community-assignments", required=True)
    parser.add_argument("--node-events", required=True)
    parser.add_argument("--alab-zip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wl-iters", type=int, default=3)
    parser.add_argument("--local-mode", choices=["matminer_ops", "fast_local"], default="matminer_ops")
    parser.add_argument("--observation-year", type=int, default=2023)
    return parser.parse_args()


def parse_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def parse_float(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var) or 1.0


def zscore(v: float, m: float, s: float) -> float:
    return (v - m) / s if s else 0.0


def raw_accessibility(distance: float, core_threshold: float, size: float, community_age: float) -> float:
    norm_dist = distance / max(core_threshold, 1e-6)
    return math.log1p(norm_dist) - 0.5 * math.log1p(size) - 0.5 * math.log1p(max(community_age, 0.0))


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


def load_community_metadata(assign_path: Path, node_events_path: Path) -> tuple[dict[int, dict[str, float]], float, float]:
    sizes = Counter()
    births: dict[int, int] = {}
    core_thresholds: dict[int, list[float]] = defaultdict(list)
    raw_values: list[float] = []

    with assign_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            comm = parse_int(row.get("community", ""))
            year = parse_int(row.get("year", ""))
            if comm is None or comm < 0:
                continue
            sizes[comm] += 1
            if year is not None:
                births[comm] = min(year, births.get(comm, year))

    with node_events_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        event_rows = []
        for row in reader:
            comm = parse_int(row.get("community", ""))
            thresh = parse_float(row.get("core_threshold", ""))
            dist = parse_float(row.get("distance_to_centroid", ""))
            year = parse_int(row.get("year", ""))
            if comm is not None and comm >= 0 and thresh is not None:
                core_thresholds[comm].append(thresh)
            if comm is not None and comm >= 0 and dist is not None and year is not None:
                event_rows.append((comm, year, dist))

    meta = {}
    for comm, size in sizes.items():
        meta[comm] = {
            "size": float(size),
            "birth_year": float(births.get(comm, 2010)),
            "core_threshold": float(mean(core_thresholds.get(comm, [])) or 1.0),
        }

    for comm, year, dist in event_rows:
        m = meta.get(comm)
        if m is None:
            continue
        age = year - m["birth_year"]
        raw_values.append(raw_accessibility(dist, m["core_threshold"], m["size"], age))

    return meta, mean(raw_values), stdev(raw_values)


def centroid_thresholds(Xp: np.ndarray, labels: list[int]) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    communities = sorted({c for c in labels if c is not None and c >= 0})
    index = {c: i for i, c in enumerate(communities)}
    centroids = np.zeros((len(communities), Xp.shape[1]), dtype=float)
    counts = np.zeros(len(communities), dtype=float)
    thresholds: dict[int, float] = {}
    by_comm: dict[int, list[float]] = defaultdict(list)
    for x, c in zip(Xp, labels):
        if c is None or c < 0:
            continue
        idx = index[c]
        centroids[idx] += x
        counts[idx] += 1
    for i, c in enumerate(communities):
        centroids[i] /= max(counts[i], 1.0)
    for x, c in zip(Xp, labels):
        if c is None or c < 0:
            continue
        d = float(np.linalg.norm(x - centroids[index[c]]))
        by_comm[c].append(d)
    for c, vals in by_comm.items():
        thresholds[c] = float(np.quantile(vals, 0.95)) if vals else 0.0
    return np.array(communities, dtype=int), centroids, thresholds


def xlsx_rows_from_bytes(payload: bytes) -> list[list[str]]:
    with zipfile.ZipFile(BytesIO(payload)) as xzf:
        sst: list[str] = []
        if "xl/sharedStrings.xml" in xzf.namelist():
            root = ET.fromstring(xzf.read("xl/sharedStrings.xml"))
            for si in root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                sst.append("".join(t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")))

        rels = ET.fromstring(xzf.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        wb = ET.fromstring(xzf.read("xl/workbook.xml"))
        sh = next(iter(wb.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheets")))
        rid = sh.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        root = ET.fromstring(xzf.read("xl/" + relmap[rid]))
        sheet_data = root.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData")

        def cell_value(c: ET.Element) -> str:
            t = c.attrib.get("t")
            v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
            if v is None:
                isel = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}is")
                if isel is not None:
                    return "".join(x.text or "" for x in isel.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
                return ""
            val = v.text or ""
            if t == "s":
                return sst[int(val)]
            return val

        return [[cell_value(c) for c in row] for row in sheet_data]


def load_alab_targets(zip_path: Path) -> list[AlabTarget]:
    aliases = {
        "Y3Ga3In2O12": "Y3In2Ga3O12",
        "Ba6Ta2Na2V2O17": "Ba6Na2Ta2V2O17",
    }

    with zipfile.ZipFile(zip_path) as zf:
        csv_rows = list(csv.DictReader(io.StringIO(zf.read("20230502 Synthesis Results with Recipes.csv").decode("utf-8", errors="replace"))))
        xlsx_rows = xlsx_rows_from_bytes(zf.read("Refinement-Table.xlsx"))
        names = zf.namelist()

    raw_by_formula = {r["Target"]: r for r in csv_rows}

    conclusions: dict[str, str] = {}
    current_formula = None
    for vals in xlsx_rows:
        if vals and vals[0]:
            current_formula = vals[0]
        canonical = aliases.get(current_formula, current_formula)
        if canonical in raw_by_formula and len(vals) > 1 and vals[1]:
            conclusions[canonical] = vals[1]

    cifs_by_formula: dict[str, str] = {}
    for name in names:
        if not name.lower().endswith(".cif") or name.startswith("__MACOSX/"):
            continue
        parts = Path(name).parts
        if len(parts) < 2:
            continue
        formula = aliases.get(parts[1], parts[1])
        if formula not in cifs_by_formula or name.startswith("Manual_Refinement_Results/"):
            cifs_by_formula[formula] = name

    targets: list[AlabTarget] = []
    for row in csv_rows:
        formula = row["Target"]
        conclusion = conclusions.get(formula, "")
        low = conclusion.lower()
        if row["Result"] == "Success (offline)":
            corrected = "offline_recovery"
        elif "inconclusive" in low:
            corrected = "inconclusive"
        elif formula in conclusions:
            corrected = "made"
        else:
            corrected = "not_obtained"
        targets.append(
            AlabTarget(
                formula=formula,
                raw_result=row["Result"],
                mp_id=row["Materials Project ID"],
                corrected_outcome=corrected,
                conclusion=conclusion,
                cif_member=cifs_by_formula.get(formula),
            )
        )
    return targets


def open_structure_from_zip(zip_path: Path, member: str) -> Structure:
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read(member).decode("utf-8", errors="replace")
    return Structure.from_str(text, fmt="cif")


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
    pca32 = PCA(n_components=32, random_state=42)
    Xp = pca32.fit_transform(Xs)
    communities, centroids, thresholds = centroid_thresholds(Xp, community_labels)

    community_meta, mu, sigma = load_community_metadata(Path(args.community_assignments), Path(args.node_events))
    targets = load_alab_targets(Path(args.alab_zip))

    records = []
    failures = []
    for target in targets:
        if not target.cif_member:
            failures.append({"formula": target.formula, "detail": "missing_cif"})
            continue
        try:
            structure = open_structure_from_zip(Path(args.alab_zip), target.cif_member)
            emb = build_structure_embedding(structure, wl_iters=args.wl_iters, local_mode=args.local_mode)
            Ys = scaler.transform(np.asarray(emb, dtype=float).reshape(1, -1))
            Yp = pca32.transform(Ys)[0]
            dists = np.linalg.norm(centroids - Yp[None, :], axis=1)
            idx = int(np.argmin(dists))
            comm = int(communities[idx])
            dist = float(dists[idx])
            thr = thresholds.get(comm, 0.0)
            meta = community_meta.get(comm)
            if meta is None:
                raise RuntimeError(f"missing community metadata for {comm}")
            age = args.observation_year - meta["birth_year"]
            raw = raw_accessibility(dist, meta["core_threshold"], meta["size"], age)
            score = zscore(raw, mu, sigma)
            records.append(
                {
                    "formula": target.formula,
                    "mp_id": target.mp_id,
                    "raw_result": target.raw_result,
                    "corrected_outcome": target.corrected_outcome,
                    "assigned_community": comm,
                    "nearest_centroid_distance": dist,
                    "core_threshold_p95": thr,
                    "outlier_like": bool(dist > thr),
                    "accessibility_score": score,
                    "cif_member": target.cif_member,
                    "conclusion": target.conclusion.replace("\n", " ").strip(),
                }
            )
        except Exception as exc:
            failures.append({"formula": target.formula, "detail": str(exc)[:300], "cif_member": target.cif_member})

    with (out_dir / "alab_validation_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "formula",
                "mp_id",
                "raw_result",
                "corrected_outcome",
                "assigned_community",
                "nearest_centroid_distance",
                "core_threshold_p95",
                "outlier_like",
                "accessibility_score",
                "cif_member",
                "conclusion",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    by_outcome: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        by_outcome[str(row["corrected_outcome"])].append(row)

    summary = {
        "n_targets_csv": len(targets),
        "n_scored": len(records),
        "n_failures": len(failures),
        "raw_result_counts": dict(Counter(t.raw_result for t in targets)),
        "corrected_outcome_counts": dict(Counter(t.corrected_outcome for t in targets)),
        "outcome_means": {
            k: {
                "mean_accessibility": mean([float(r["accessibility_score"]) for r in rows]),
                "frontier_rate": mean([1.0 if bool(r["outlier_like"]) else 0.0 for r in rows]),
            }
            for k, rows in by_outcome.items()
        },
        "inconclusive_formulas": sorted([t.formula for t in targets if t.corrected_outcome == "inconclusive"]),
        "not_obtained_formulas": sorted([t.formula for t in targets if t.corrected_outcome == "not_obtained"]),
    }
    (out_dir / "alab_validation_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "alab_validation_failures.json").write_text(json.dumps(failures, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
