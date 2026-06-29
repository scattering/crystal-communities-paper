"""Shared helpers for the per-source frontier-projection scripts.

The five ``analyze_<source>_frontier.py`` producers
(``gnome``, ``mp``, ``jarvis``, ``alexandria``, and the generic
``analyze_external_cif_zip_frontier`` used for MatterGen) historically
duplicated three helper functions byte-for-byte:

  * ``parse_int`` / ``parse_float`` — tolerant CSV cell coercion,
  * ``load_community_rows`` — read the production
    ``community_assignments.csv`` (icsd_id, year, community),
  * ``centroid_thresholds`` — compute per-community centroids in
    PCA space and the 95th-percentile within-community distance
    that defines the in-basin / frontier classification used
    throughout the manuscript.

Pulling them here lets per-source producers focus on the
source-specific record dataclass + ingestion path, and makes a
schema or threshold change a single-edit operation.

Schema invariant
----------------
The shared centroid / threshold logic in ``centroid_thresholds``
defines the *95th percentile of within-community Euclidean distance
in the frozen ICSD PCA basis* as the in-basin cutoff. A record whose
nearest-centroid distance exceeds the cutoff for its assigned
community is labelled ``outlier_like = True`` (frontier). This single
threshold definition is the calibration used by every figure in the
Nature manuscript.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def require_zenodo_file(path, what: str | None = None) -> Path:
    """Verify a data input exists; raise FileNotFoundError with a Zenodo
    pointer if not.

    Use at the top of ``main()`` in figure / analysis scripts (right after
    argparse + Path conversion) to give a reader a more actionable error
    than a bare ``FileNotFoundError`` when they have not yet downloaded
    the manuscript's Zenodo data bundle.

    Parameters
    ----------
    path
        The path to verify (``str`` or ``Path``).
    what
        Optional one-line description of the file's role, included in
        the error message. Example: ``"the formula-overlap summary
        driving Figure 4"``.

    Returns
    -------
    Path
        The verified path (same object, converted to ``Path``).

    Raises
    ------
    FileNotFoundError
        With a multi-line message pointing at the Zenodo bundle and
        the README's "Zenodo data bundle required" section.
    """
    p = Path(path)
    if p.exists():
        return p
    role = f"\n  ({what})" if what else ""
    raise FileNotFoundError(
        f"Required data file not found: {p}{role}\n\n"
        f"This file is part of the Zenodo data bundle for the manuscript\n"
        f"'Computed materials proposals depart from the structural\n"
        f"memory of experimental discovery.' Download the bundle before\n"
        f"running this script:\n\n"
        f"    zenodo_get 10.5281/zenodo.20046302  # concept DOI, always points to latest version -o notes/\n\n"
        f"See README.md → 'Zenodo data bundle required for figure\n"
        f"reproduction' and docs/SCHEMA.md for the full file inventory."
    )


def parse_int(text: str) -> int | None:
    """Coerce a CSV cell to int, returning None on empty/garbage.

    Accepts integer strings, integer-valued floats ("17.0"), and
    whitespace; everything else returns None rather than raising.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def parse_float(text: str) -> float | None:
    """Coerce a CSV cell to float, returning None on empty/garbage."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def load_community_rows(path: Path) -> list[dict[str, int | None]]:
    """Load (icsd_id, year, community) rows from the production CSV.

    Reads ``notes/icsd_community_assignments/community_assignments_labels3.csv``
    (the canonical 167.5K-entry table). Missing or malformed values are
    coerced to None rather than dropped, so downstream code can decide
    how to handle them.
    """
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


def centroid_thresholds(
    Xp: np.ndarray,
    labels: list[int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-community centroids + the 95th-percentile within-community distance.

    Parameters
    ----------
    Xp
        ``(N, D)`` PCA-projected coordinates of the ICSD reference
        sample (one row per ICSD entry).
    labels
        Length-``N`` per-row community labels; entries with label
        ``None`` or ``< 0`` are excluded (these are the noise points
        from HDBSCAN / Louvain that did not get a community).

    Returns
    -------
    communities : ``(K,) int``
        Sorted community ids, with the centroid index in the second
        return value matching this order.
    centroids : ``(K, D) float``
        Mean position of each community in the input PCA basis.
    threshold : ``float``
        95th-percentile within-community Euclidean distance, pooled
        across all communities. This is the single in-basin cutoff
        used by every frontier-projection script and figure in the
        Nature manuscript. A record whose nearest-centroid distance
        exceeds this value is classified as ``outlier_like`` (frontier).
    """
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
