from __future__ import annotations

import base64
import csv
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from pymatgen.core import Structure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


APP_ROOT = Path(__file__).resolve().parent
FIG_ROOT = APP_ROOT.parent / "figures" / "icsd_densification"
REPO_ROOT = APP_ROOT.parent.parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

from icsd_densify_worker import build_structure_embedding
from analyze_alab_validation import (
    centroid_thresholds,
    load_community_metadata,
    load_community_rows,
    raw_accessibility,
    zscore,
)


def env_path(name: str, fallback: str = "") -> Path | None:
    value = os.environ.get(name, fallback).strip()
    return Path(value) if value else None


ICSD_FEATURES_PATH = env_path("ICSD_FEATURES_PATH")
ICSD_COMMUNITY_ASSIGNMENTS_PATH = env_path("ICSD_COMMUNITY_ASSIGNMENTS_PATH")
ICSD_NODE_EVENTS_PATH = env_path("ICSD_NODE_EVENTS_PATH")
ICSD_PROTOTYPE_LABELS_PATH = env_path("ICSD_PROTOTYPE_LABELS_PATH")
ICSD_CANONICAL_LABELS_PATH = env_path(
    "ICSD_CANONICAL_LABELS_PATH",
    str(REPO_ROOT / "notes" / "canonical_family_names_labels3.csv"),
)
ICSD_REPRESENTATIVES_PATH = env_path(
    "ICSD_REPRESENTATIVES_PATH",
    str(REPO_ROOT / "notes" / "functional_community_representatives_top20.csv"),
)
# Output of scripts/infer_community_families.py — heuristic textbook-family
# names ("spinel", "olivine", "garnet", "delafossite", ...) for communities
# whose top-20 members have a dominant (space group, stoichiometry class)
# signature. Used as a fallback after canonical-family curation but before
# the AflowPrototypeMatcher / CIF-systematic-name JSON, because the
# inferred labels are far more chemically informative than e.g. the bare
# stoichiometric "Y0.844Cu1.5Se2" that the prototype JSON usually returns.
ICSD_INFERRED_FAMILIES_PATH = env_path(
    "ICSD_INFERRED_FAMILIES_PATH",
    str(REPO_ROOT / "notes" / "community_families_inferred.csv"),
)
# Optional graph-aware community layout from scripts/build_community_layout.py.
# When set, the dedicated Community Map page draws centroids in this layout
# instead of the raw PCA scatter so overlapping basins separate visually.
ICSD_COMMUNITY_LAYOUT_PATH = env_path(
    "ICSD_COMMUNITY_LAYOUT_PATH",
    str(REPO_ROOT / "notes" / "community_layout.csv"),
)
# Optional directory of CIF files keyed by ICSD id (e.g. 153958.cif). When set,
# the click-to-drill modal can render a 3Dmol.js view of the centroid CIF; when
# unset, the modal degrades to the exemplar table only.
ICSD_CIF_DIR = env_path("ICSD_CIF_DIR")
DEMO_LOCAL_MODE = os.environ.get("ICSD_DEMO_LOCAL_MODE", "matminer_ops")
DEMO_WL_ITERS = int(os.environ.get("ICSD_DEMO_WL_ITERS", "3"))
DEMO_OBSERVATION_YEAR = int(os.environ.get("ICSD_DEMO_OBSERVATION_YEAR", "2025"))
DEMO_SAMPLE_SIZE = int(os.environ.get("ICSD_DEMO_SAMPLE_SIZE", "12000"))
DEMO_RANDOM_SEED = int(os.environ.get("ICSD_DEMO_RANDOM_SEED", "42"))


def load_canonical_label_records(path: Path | None) -> dict[int, dict[str, str]]:
    """Return {community_id: {label, confidence, evidence, notes, raw_label,
    centroid_icsd_id}} for graph_community rows that have a canonical name.

    Empty / sentinel canonical names ("", "unknown", "n/a") are dropped — those
    communities will fall through to the prototype-matcher label and finally to
    "community N" via resolve_community_label().
    """
    if path is None or not path.exists():
        return {}
    out: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("kind") or "").strip() != "graph_community":
                continue
            label = (row.get("canonical_family_name") or "").strip()
            if not label or label.lower() in {"unknown", "n/a"}:
                continue
            try:
                comm = int(row["id"])
            except (TypeError, ValueError):
                continue
            out[comm] = {
                "label": label,
                "confidence": (row.get("confidence") or "").strip(),
                "evidence": (row.get("evidence") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
                "raw_label": (row.get("raw_label") or "").strip(),
                "centroid_icsd_id": (row.get("centroid_icsd_id") or "").strip(),
            }
    return out


def load_canonical_label_map(path: Path | None) -> dict[int, str]:
    """Convenience: just the {community: canonical_name} map."""
    return {c: rec["label"] for c, rec in load_canonical_label_records(path).items()}


def load_inferred_family_map(path: Path | None) -> dict[int, str]:
    """Load community_families_inferred.csv (output of
    scripts/infer_community_families.py). Returns {community: inferred_family}
    for rows where inferred_family is non-empty. Rows with no dominant
    signature are silently skipped — those communities will fall through to
    the prototype-matcher label."""
    if path is None or not path.exists():
        return {}
    out: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("inferred_family") or "").strip()
            if not label:
                continue
            try:
                comm = int(row["community"])
            except (KeyError, TypeError, ValueError):
                continue
            out[comm] = label
    return out


def load_prototype_label_map(path: Path | None) -> dict[int, str]:
    """Load community_prototype_labels.json (the AflowPrototypeMatcher /
    systematic-name fallback emitted by icsd_graph_community_postprocess).
    Returns {community: label}."""
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[int, str] = {}
    if not isinstance(data, list):
        return out
    for row in data:
        try:
            comm = int(row["community"])
        except (KeyError, TypeError, ValueError):
            continue
        label = (row.get("label") or "").strip()
        if label:
            out[comm] = label
    return out


def resolve_community_label(
    comm: int,
    canonical: dict[int, str],
    prototype: dict[int, str],
    inferred: dict[int, str] | None = None,
) -> str:
    """Single source of truth for community labels everywhere in the dashboard.

    Order: canonical curated family name → heuristic textbook-family inference
    (community_families_inferred.csv) → prototype-matcher label (AflowPrototype
    name when available, else the CIF systematic name, else the raw
    stoichiometric formula) → 'community N' as last resort. The heuristic
    inference is preferred over the prototype JSON because labels like
    "spinel (MgAl2O4-type)" are far more chemically informative than the bare
    stoichiometric formulas (e.g. "Y0.844Cu1.5Se2") the prototype matcher
    typically returns.
    """
    canon = canonical.get(comm)
    if canon:
        return canon
    if inferred is not None:
        inf = inferred.get(comm)
        if inf:
            return inf
    proto = prototype.get(comm)
    if proto:
        return proto
    return f"community {comm}"


def load_community_layout(path: Path | None) -> dict[int, dict[str, float]]:
    """Load the graph-aware community layout produced by
    scripts/build_community_layout.py. Returns
    {community: {x, y, size, intercommunity_edge_count, top_neighbor}}.
    """
    if path is None or not path.exists():
        return {}
    out: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                comm = int(row["community"])
                out[comm] = {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "size": int(row["size"]),
                    "intercommunity_edge_count": int(row.get("intercommunity_edge_count") or 0),
                    "top_neighbor": int(row["top_neighbor"]) if (row.get("top_neighbor") or "").strip() else -1,
                }
            except (KeyError, TypeError, ValueError):
                continue
    return out


def load_representatives(path: Path | None, top_k: int = 10) -> dict[int, list[dict[str, str]]]:
    """Load top-k centroid-nearest exemplars per community from the CSV emitted
    by extract_functional_community_representatives.py. Rows are sorted by
    rank_by_centroid_distance ascending. Returns {community: [exemplar, ...]}.
    """
    if path is None or not path.exists():
        return {}
    out: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                comm = int(row["community"])
                rank = int(row["rank_by_centroid_distance"])
            except (KeyError, TypeError, ValueError):
                continue
            if rank > top_k:
                continue
            out.setdefault(comm, []).append({
                "rank": rank,
                "icsd_id": (row.get("icsd_id") or "").strip(),
                "name": (row.get("name") or "").strip(),
                "publication_year": (row.get("publication_year") or "").strip(),
                "sym_group": (row.get("sym_group") or "").strip(),
                "Bravais": (row.get("Bravais") or "").strip(),
                "centroid_distance": (row.get("centroid_distance") or "").strip(),
                "a": (row.get("a") or "").strip(),
                "b": (row.get("b") or "").strip(),
                "c": (row.get("c") or "").strip(),
            })
    for comm in out:
        out[comm].sort(key=lambda r: r["rank"])
    return out


@dataclass
class FrozenMap:
    scaler: StandardScaler
    pca32: PCA
    pca2: PCA
    communities: np.ndarray
    centroids: np.ndarray
    centroids_xy: np.ndarray
    thresholds: dict[int, float]
    community_meta: dict[int, dict[str, float]]
    canonical_labels: dict[int, str]
    canonical_records: dict[int, dict[str, str]]
    inferred_labels: dict[int, str]
    prototype_labels: dict[int, str]
    representatives: dict[int, list[dict[str, str]]]
    community_layout: dict[int, dict[str, float]]
    accessibility_mu: float
    accessibility_sigma: float
    background_xy: np.ndarray
    background_comm: np.ndarray
    background_birth: np.ndarray
    background_label: list[str]
    background_xlim: tuple[float, float]
    background_ylim: tuple[float, float]

    def label(self, comm: int) -> str:
        return resolve_community_label(
            int(comm), self.canonical_labels, self.prototype_labels, self.inferred_labels
        )

    def label_source(self, comm: int) -> str:
        if int(comm) in self.canonical_labels:
            return "canonical"
        if int(comm) in self.inferred_labels:
            return "inferred"
        if int(comm) in self.prototype_labels:
            return "prototype"
        return "fallback"


APP_STATUS = "ready"
if not all([ICSD_FEATURES_PATH, ICSD_COMMUNITY_ASSIGNMENTS_PATH, ICSD_NODE_EVENTS_PATH]):
    APP_STATUS = "missing_paths"


def image_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    return f"data:{mime};base64,{payload}"


FIGURES = {
    "graph_time_ratios": FIG_ROOT / "graph_time_ratios.png",
    "prototype_collapse": FIG_ROOT / "prototype_collapse_space_groups.png",
    "stepping_stone": FIG_ROOT / "tri_stepping_stone_counts.png",
    # gnome_frontier_pca_labeled.png is a relabeled overlay of the original
    # script output: it adds the missing "Human (ICSD background)" legend entry
    # and an AI-vs-Human title. Regenerated by tools/patch_gnome_legend.py
    # (or implicitly by re-running analyze_gnome_frontier.py once that script
    # is also re-rendered with the updated plot_frontier labels).
    "gnome_frontier": FIG_ROOT / "gnome_frontier_pca_labeled.png",
    "graph_growth_gif": FIG_ROOT / "icsd_graph_growth.gif",
}

INTERACTIVE_VIEWERS = {
    "graph_view_html": FIG_ROOT / "icsd_graph_view.html",
    "connectivity_view_html": FIG_ROOT / "icsd_graph_connectivity_view_labels4.html",
}


METRICS = [
    ("167,500", "ICSD entries featurized in production (92.4% of 181,362 requested; 4.8% pymatgen CIF parser failures, 2.8% over the 256-site cap)"),
    ("44.2", "Mean space groups absorbed by the ten largest continuous basins"),
    ("82.9%", "New formulas joining existing communities (TRI-shared subset: 13,739 of 16,582 classifiable formulas in the Aykol-2019 thermodynamic-stability network)"),
    ("Humans > MatterGen > GNoME", "Stable held-out frontier ordering across historical cutoffs"),
]


CARD_STYLE = {
    "background": "rgba(255, 250, 242, 0.92)",
    "border": "1px solid rgba(217, 203, 183, 0.95)",
    "borderRadius": "20px",
    "boxShadow": "0 20px 50px rgba(24, 32, 40, 0.08)",
}


def metric_card(value: str, label: str) -> html.Div:
    return html.Div(
        [
            html.Div(value, style={"fontSize": "1.6rem", "fontWeight": "700", "marginBottom": "6px"}),
            html.Div(label, style={"color": "#5b6672", "lineHeight": "1.45", "fontSize": "0.92rem"}),
        ],
        style={
            "padding": "16px",
            "borderRadius": "16px",
            "background": "rgba(255,255,255,0.78)",
            "border": "1px solid rgba(217, 203, 183, 0.9)",
        },
    )


def figure_card(
    title: str,
    body: str,
    image_path: Path,
    *,
    href: str | None = None,
    badge: str | None = None,
) -> html.Div:
    img = html.Img(
        src=image_data_uri(image_path),
        style={
            "width": "100%",
            "height": "270px",
            # `contain` preserves the entire figure (no cropped axis labels);
            # `cover` was clipping the bottom of bar charts and the rotated
            # x-axis tick labels in the stepping-stone and prototype panels.
            "objectFit": "contain",
            "background": "rgba(255,255,255,0.85)",
            "display": "block",
            "borderBottom": "1px solid rgba(217, 203, 183, 0.9)",
        },
    )
    media: Any = img
    if href:
        media = html.A(
            img,
            href=href,
            target="_blank",
            rel="noopener",
            title="Open the full interactive viewer in a new tab",
            style={"display": "block"},
        )
    badge_node: Any = None
    if badge:
        badge_node = html.Div(
            badge,
            style={
                "display": "inline-block",
                "marginBottom": "8px",
                "padding": "3px 8px",
                "borderRadius": "999px",
                "border": "1px solid rgba(15,109,97,0.35)",
                "color": "#0f6d61",
                "fontSize": "11px",
                "letterSpacing": "0.08em",
                "textTransform": "uppercase",
            },
        )
    body_children: list[Any] = []
    if badge_node is not None:
        body_children.append(badge_node)
    body_children.append(html.H3(title, style={"margin": "0 0 8px", "fontSize": "1.14rem"}))
    body_children.append(
        html.P(body, style={"margin": "0", "color": "#5b6672", "lineHeight": "1.55"})
    )
    return html.Div(
        [
            media,
            html.Div(body_children, style={"padding": "18px 20px 20px"}),
        ],
        style={**CARD_STYLE, "overflow": "hidden"},
    )


def config_hint() -> html.Div:
    if APP_STATUS == "ready":
        return html.Div(
            "Frozen-map paths detected. Upload scoring is enabled.",
            style={
                "padding": "14px 16px",
                "borderRadius": "14px",
                "background": "rgba(15,109,97,0.08)",
                "color": "#35514c",
                "lineHeight": "1.55",
                "marginBottom": "18px",
            },
        )
    return html.Div(
        [
            html.Div("Frozen-map scoring is not configured yet.", style={"fontWeight": "700", "marginBottom": "6px"}),
            html.Div(
                "Set ICSD_FEATURES_PATH, ICSD_COMMUNITY_ASSIGNMENTS_PATH, and ICSD_NODE_EVENTS_PATH in the DANSE2 app environment to enable upload scoring. ICSD_PROTOTYPE_LABELS_PATH is optional but recommended — it surfaces AflowPrototype / CIF systematic names for communities that don't yet have a curated canonical name.",
                style={"lineHeight": "1.55"},
            ),
        ],
        style={
            "padding": "14px 16px",
            "borderRadius": "14px",
            "background": "rgba(181, 98, 0, 0.10)",
            "color": "#6b4f1f",
            "lineHeight": "1.55",
            "marginBottom": "18px",
        },
    )


@lru_cache(maxsize=1)
def load_frozen_map() -> FrozenMap:
    if APP_STATUS != "ready":
        raise RuntimeError("Frozen-map paths are not configured.")
    X = np.load(ICSD_FEATURES_PATH)
    community_rows = load_community_rows(ICSD_COMMUNITY_ASSIGNMENTS_PATH)
    if len(X) != len(community_rows):
        raise ValueError(f"ICSD features rows ({len(X)}) != community rows ({len(community_rows)})")

    community_labels = [int(r["community"]) if r["community"] is not None else -1 for r in community_rows]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca32 = PCA(n_components=min(32, Xs.shape[0], Xs.shape[1]), random_state=DEMO_RANDOM_SEED)
    Xp = pca32.fit_transform(Xs)
    communities, centroids, thresholds = centroid_thresholds(Xp, community_labels)
    community_meta, mu, sigma = load_community_metadata(ICSD_COMMUNITY_ASSIGNMENTS_PATH, ICSD_NODE_EVENTS_PATH)
    canonical_records = load_canonical_label_records(ICSD_CANONICAL_LABELS_PATH)
    canonical_labels = {c: rec["label"] for c, rec in canonical_records.items()}
    inferred_labels = load_inferred_family_map(ICSD_INFERRED_FAMILIES_PATH)
    prototype_labels = load_prototype_label_map(ICSD_PROTOTYPE_LABELS_PATH)
    representatives = load_representatives(ICSD_REPRESENTATIVES_PATH, top_k=10)
    community_layout = load_community_layout(ICSD_COMMUNITY_LAYOUT_PATH)

    valid_idx = np.where(np.asarray(community_labels) >= 0)[0]
    rng = np.random.default_rng(DEMO_RANDOM_SEED)
    if len(valid_idx) > DEMO_SAMPLE_SIZE:
        sample_idx = np.sort(rng.choice(valid_idx, size=DEMO_SAMPLE_SIZE, replace=False))
    else:
        sample_idx = valid_idx

    pca2 = PCA(n_components=2, random_state=DEMO_RANDOM_SEED)
    background_xy = pca2.fit_transform(Xp[sample_idx])
    background_comm = np.asarray([community_labels[i] for i in sample_idx], dtype=int)
    centroids_xy = pca2.transform(centroids)

    background_birth = np.array(
        [
            float(community_meta.get(int(c), {}).get("birth_year", 2010.0))
            for c in background_comm
        ],
        dtype=float,
    )
    background_label = [
        resolve_community_label(int(c), canonical_labels, prototype_labels, inferred_labels)
        for c in background_comm
    ]

    pad_x = 0.05 * (float(np.ptp(background_xy[:, 0])) or 1.0)
    pad_y = 0.05 * (float(np.ptp(background_xy[:, 1])) or 1.0)
    background_xlim = (
        float(background_xy[:, 0].min() - pad_x),
        float(background_xy[:, 0].max() + pad_x),
    )
    background_ylim = (
        float(background_xy[:, 1].min() - pad_y),
        float(background_xy[:, 1].max() + pad_y),
    )

    return FrozenMap(
        scaler=scaler,
        pca32=pca32,
        pca2=pca2,
        communities=communities,
        centroids=centroids,
        centroids_xy=centroids_xy,
        thresholds=thresholds,
        community_meta=community_meta,
        canonical_labels=canonical_labels,
        canonical_records=canonical_records,
        inferred_labels=inferred_labels,
        prototype_labels=prototype_labels,
        representatives=representatives,
        community_layout=community_layout,
        accessibility_mu=mu,
        accessibility_sigma=sigma,
        background_xy=background_xy,
        background_comm=background_comm,
        background_birth=background_birth,
        background_label=background_label,
        background_xlim=background_xlim,
        background_ylim=background_ylim,
    )


def parse_upload(contents: str) -> Structure:
    try:
        _, payload = contents.split(",", 1)
    except ValueError as exc:
        raise ValueError("Upload payload was not a valid data URI.") from exc
    text = base64.b64decode(payload).decode("utf-8", errors="replace")
    return Structure.from_str(text, fmt="cif")


def structure_formula(structure: Structure) -> str:
    return structure.composition.reduced_formula


def score_structure(structure: Structure) -> dict[str, Any]:
    fmap = load_frozen_map()
    embedding = build_structure_embedding(
        structure,
        wl_iters=DEMO_WL_ITERS,
        local_mode=DEMO_LOCAL_MODE,
    )
    Ys = fmap.scaler.transform(np.asarray(embedding, dtype=float).reshape(1, -1))
    Yp = fmap.pca32.transform(Ys)[0]
    dists = np.linalg.norm(fmap.centroids - Yp[None, :], axis=1)
    idx = int(np.argmin(dists))
    comm = int(fmap.communities[idx])
    dist = float(dists[idx])
    threshold = float(fmap.thresholds.get(comm, 0.0))
    meta = fmap.community_meta.get(comm)
    if meta is None:
        raise RuntimeError(f"Missing community metadata for {comm}")
    age = DEMO_OBSERVATION_YEAR - meta["birth_year"]
    raw = raw_accessibility(dist, meta["core_threshold"], meta["size"], age)
    score = float(zscore(raw, fmap.accessibility_mu, fmap.accessibility_sigma))
    xy = fmap.pca2.transform(Yp.reshape(1, -1))[0]
    centroid_xy = fmap.centroids_xy[idx]
    family_name = fmap.label(comm)
    label_source = fmap.label_source(comm)
    canonical_record = fmap.canonical_records.get(comm)
    return {
        "formula": structure_formula(structure),
        "n_sites": len(structure),
        "community": comm,
        "community_label": family_name,
        "label_source": label_source,
        "canonical_confidence": (canonical_record or {}).get("confidence", ""),
        "canonical_evidence": (canonical_record or {}).get("evidence", ""),
        "centroid_icsd_id": (canonical_record or {}).get("centroid_icsd_id", ""),
        "distance": dist,
        "threshold": threshold,
        "frontier": bool(dist > threshold),
        "accessibility": score,
        "community_size": int(meta["size"]),
        "community_birth_year": int(meta["birth_year"]),
        "xy": xy,
        "centroid_xy": centroid_xy,
    }


def summary_panel(result: dict[str, Any]) -> html.Div:
    frontier_label = "Frontier-like" if result["frontier"] else "In-basin"
    frontier_color = "#8a2f2f" if result["frontier"] else "#0f6d61"
    label_source = result.get("label_source", "fallback")
    canonical_conf = (result.get("canonical_confidence") or "").lower()
    if label_source == "canonical":
        if canonical_conf == "high":
            source_text, source_color = "Curated (high confidence)", "#0f6d61"
        elif canonical_conf == "medium":
            source_text, source_color = "Curated (medium confidence)", "#0f6d61"
        elif canonical_conf == "low":
            source_text, source_color = "Curated (low confidence)", "#b56200"
        else:
            source_text, source_color = "Curated", "#0f6d61"
    elif label_source == "inferred":
        source_text, source_color = "Inferred textbook family (heuristic)", "#34915d"
    elif label_source == "prototype":
        source_text, source_color = "Prototype matcher / CIF systematic name", "#5b6672"
    else:
        source_text, source_color = "No prototype assigned", "#5b6672"
    return html.Div(
        [
            html.Div(
                [
                    html.Div(result["formula"], style={"fontSize": "1.6rem", "fontWeight": "700"}),
                    html.Div(
                        result.get("community_label", f"community {result['community']}"),
                        style={"marginTop": "4px", "color": "#5b6672", "fontSize": "0.95rem"},
                    ),
                    html.Div(
                        source_text,
                        title=(result.get("canonical_evidence") or ""),
                        style={
                            "marginTop": "4px",
                            "color": source_color,
                            "fontSize": "0.78rem",
                            "letterSpacing": "0.04em",
                            "textTransform": "uppercase",
                            "fontWeight": "600",
                        },
                    ),
                    html.Div(
                        frontier_label,
                        style={
                            "display": "inline-block",
                            "marginTop": "10px",
                            "padding": "6px 10px",
                            "borderRadius": "999px",
                            "background": "rgba(255,255,255,0.82)",
                            "border": f"1px solid {frontier_color}",
                            "color": frontier_color,
                            "fontWeight": "700",
                        },
                    ),
                ],
                style={"marginBottom": "14px"},
            ),
            html.Div(
                [
                    metric_card(f"{result['community']}", "Nearest structural basin"),
                    metric_card(f"{result['accessibility']:.2f}", "Accessibility score A_i"),
                    metric_card(f"{result['distance']:.3f}", "Centroid distance"),
                    metric_card(f"{result['threshold']:.3f}", "Community p95 threshold"),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "12px"},
            ),
            html.Div(
                f"Community size {result['community_size']} and first observed around {result['community_birth_year']}. Uploaded structure has {result['n_sites']} sites.",
                style={"marginTop": "16px", "color": "#5b6672", "lineHeight": "1.55"},
            ),
        ],
        style={**CARD_STYLE, "padding": "24px"},
    )


def _community_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 3:
        return np.empty(0), np.empty(0)
    try:
        from scipy.spatial import ConvexHull  # local import: scipy is already a transitive dep
    except ImportError:  # pragma: no cover
        return np.empty(0), np.empty(0)
    hull = ConvexHull(points)
    seq = list(hull.vertices) + [hull.vertices[0]]
    pts = points[seq]
    return pts[:, 0], pts[:, 1]


def placement_figure(result: dict[str, Any] | None) -> go.Figure:
    fmap = load_frozen_map()
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=fmap.background_xy[:, 0],
            y=fmap.background_xy[:, 1],
            mode="markers",
            marker={
                "size": 4,
                "color": fmap.background_birth,
                "colorscale": "Viridis",
                "cmin": float(fmap.background_birth.min()),
                "cmax": float(fmap.background_birth.max()),
                "opacity": 0.45,
                "colorbar": {
                    "title": {"text": "community<br>birth year", "side": "right"},
                    "thickness": 12,
                    "len": 0.55,
                    "x": 1.02,
                },
            },
            text=[
                f"{lbl}<br>community {int(c)}<br>birth ~{int(b)}"
                for lbl, c, b in zip(fmap.background_label, fmap.background_comm, fmap.background_birth)
            ],
            # plain python int list, not a numpy array — Plotly otherwise
            # encodes numpy arrays as typed-array dicts ({bdata, dtype}) which
            # round-trip through plotly.js as TypedArrays whose elements do
            # not surface as `customdata` on plotly_click events. Without this
            # cast clicks would fire (POST 200) but the callback would see
            # `customdata=None` and render the placeholder.
            customdata=[int(c) for c in fmap.background_comm],
            hovertemplate="%{text}<extra></extra>",
            name=f"ICSD sample (n={len(fmap.background_xy)})",
        )
    )

    # All known centroids: clickable, regardless of whether an upload was scored.
    # Sized by community size, colored by birth year, hover shows the canonical-
    # or-prototype label. The customdata carries the community id so the click
    # callback can resolve the drill-down without re-doing geometry math.
    centroid_birth = np.array(
        [float(fmap.community_meta.get(int(c), {}).get("birth_year", 2010.0)) for c in fmap.communities],
        dtype=float,
    )
    centroid_size = np.array(
        [float(fmap.community_meta.get(int(c), {}).get("size", 50.0)) for c in fmap.communities],
        dtype=float,
    )
    # Compress the dynamic range so the smallest communities (size ~30) stay
    # visibly clickable next to the largest (size 1000+). log1p prevents the
    # mega-basins from saturating the marker pool while keeping long-tail
    # communities readable.
    log_sizes = np.log1p(centroid_size)
    centroid_marker_size = 9.0 + 9.0 * (log_sizes - log_sizes.min()) / max(log_sizes.max() - log_sizes.min(), 1e-6)
    centroid_text = [
        f"<b>{fmap.label(int(c))}</b><br>community {int(c)}<br>"
        f"size {int(centroid_size[i])} | birth ~{int(centroid_birth[i])}"
        f"<br><i>click to drill down</i>"
        for i, c in enumerate(fmap.communities)
    ]
    fig.add_trace(
        go.Scatter(
            x=fmap.centroids_xy[:, 0],
            y=fmap.centroids_xy[:, 1],
            mode="markers",
            marker={
                "size": centroid_marker_size,
                "color": centroid_birth,
                "colorscale": "Viridis",
                "cmin": float(fmap.background_birth.min()),
                "cmax": float(fmap.background_birth.max()),
                "opacity": 0.85,
                "line": {"width": 1.0, "color": "white"},
                "symbol": "circle",
                "showscale": False,
            },
            text=centroid_text,
            customdata=[int(c) for c in fmap.communities],
            hovertemplate="%{text}<extra></extra>",
            name=f"basin centroids (n={len(fmap.communities)})",
        )
    )

    if result is not None:
        comm = int(result["community"])
        comm_mask = fmap.background_comm == comm
        comm_points = fmap.background_xy[comm_mask]
        if len(comm_points) > 0:
            fig.add_trace(
                go.Scattergl(
                    x=comm_points[:, 0],
                    y=comm_points[:, 1],
                    mode="markers",
                    marker={"size": 6, "color": "rgba(15,109,97,0.6)", "line": {"width": 0.7, "color": "#0f6d61"}},
                    hoverinfo="skip",
                    name=f"members of nearest basin ({len(comm_points)} shown)",
                )
            )
            hx, hy = _community_hull(comm_points)
            if hx.size > 0:
                fig.add_trace(
                    go.Scatter(
                        x=hx, y=hy,
                        mode="lines",
                        line={"color": "rgba(15,109,97,0.55)", "dash": "dot", "width": 1.5},
                        fill="toself",
                        fillcolor="rgba(15,109,97,0.07)",
                        hoverinfo="skip",
                        name="basin convex hull (subsample)",
                    )
                )
        cxy = result.get("centroid_xy")
        upload_xy = result["xy"]
        if cxy is not None:
            fig.add_trace(
                go.Scatter(
                    x=[float(cxy[0]), float(upload_xy[0])],
                    y=[float(cxy[1]), float(upload_xy[1])],
                    mode="lines",
                    line={"color": "#8a2f2f" if result["frontier"] else "#0f6d61", "width": 1.4, "dash": "dash"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[float(cxy[0])],
                    y=[float(cxy[1])],
                    mode="markers+text",
                    marker={"size": 18, "color": "#0f6d61", "symbol": "star", "line": {"width": 1.5, "color": "white"}},
                    text=[result.get("community_label", f"community {comm}")],
                    textposition="bottom center",
                    textfont={"size": 11, "color": "#0f6d61"},
                    customdata=[int(comm)],
                    hovertemplate=(
                        f"{result.get('community_label', 'community ' + str(comm))}<br>"
                        f"community {comm}<br>"
                        f"birth {result['community_birth_year']}<br>"
                        f"size {result['community_size']}<br>"
                        f"<i>click to drill down</i><extra></extra>"
                    ),
                    name="nearest basin centroid",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[float(upload_xy[0])],
                y=[float(upload_xy[1])],
                mode="markers",
                marker={
                    "size": 16,
                    "color": "#8a2f2f" if result["frontier"] else "#0f6d61",
                    "line": {"width": 2, "color": "white"},
                    "symbol": "diamond" if result["frontier"] else "circle",
                },
                hovertemplate=(
                    f"<b>{result['formula']}</b> (uploaded)<br>"
                    f"basin: {result.get('community_label', 'community ' + str(comm))}<br>"
                    f"distance to centroid: {result['distance']:.3f}<br>"
                    f"basin p95 threshold: {result['threshold']:.3f}<br>"
                    f"accessibility A_i: {result['accessibility']:.2f}<br>"
                    f"frontier: {result['frontier']}<extra></extra>"
                ),
                name=result["formula"],
            )
        )

    fig.update_layout(
        margin={"l": 30, "r": 90, "t": 36, "b": 36},
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "title": "Frozen PCA-1",
            "zeroline": False,
            "gridcolor": "rgba(0,0,0,0.07)",
            "range": list(fmap.background_xlim),
        },
        yaxis={
            "title": "Frozen PCA-2",
            "zeroline": False,
            "gridcolor": "rgba(0,0,0,0.07)",
            "range": list(fmap.background_ylim),
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "font": {"size": 11}},
    )
    return fig


def community_sunburst_figure(top_individual_buckets: int = 25) -> go.Figure:
    """Two-level sunburst: curated family name → individual community.

    Communities without a curated canonical family are bucketed under
    'Uncategorized' (or 'Uncategorized (top {N})' if there are many);
    the largest `top_individual_buckets` uncategorized communities get their
    own slice so users can hover them, the rest collapse into a single
    'other uncategorized' slice.

    Slice value = community size. Slice color = birth year (averaged up to
    parent slices automatically by Plotly).
    """
    fmap = load_frozen_map()
    if fmap.communities.size == 0:
        return go.Figure()

    family_to_children: dict[str, list[int]] = {}
    uncategorized: list[int] = []
    for c in (int(x) for x in fmap.communities):
        canonical = fmap.canonical_labels.get(c)
        if canonical:
            family_to_children.setdefault(canonical, []).append(c)
        else:
            uncategorized.append(c)

    def comm_size(c: int) -> int:
        return int(fmap.community_meta.get(c, {}).get("size", 1))

    def comm_birth(c: int) -> float:
        return float(fmap.community_meta.get(c, {}).get("birth_year", 2010.0))

    rows: list[dict[str, Any]] = []  # {id, label, parent, value, color, hover}

    # Compute total first so the root and family slices have proper sums
    # (branchvalues='total' requires parent.value >= sum of children).
    family_totals = {fam: sum(comm_size(c) for c in kids) for fam, kids in family_to_children.items()}
    uncategorized.sort(key=lambda c: -comm_size(c))
    keep = uncategorized[:top_individual_buckets]
    roll = uncategorized[top_individual_buckets:]
    keep_total = sum(comm_size(c) for c in keep)
    roll_total = sum(comm_size(c) for c in roll)
    unc_total = keep_total + roll_total
    grand_total = sum(family_totals.values()) + unc_total

    rows.append({
        "id": "root",
        "label": "ICSD structural communities",
        "parent": "",
        "value": grand_total,
        "color": None,
        "hover": f"{int(fmap.communities.size)} communities, {grand_total} structures",
    })

    # Curated families
    for family, children in sorted(family_to_children.items(), key=lambda kv: -family_totals[kv[0]]):
        family_id = f"family:{family}"
        rows.append({
            "id": family_id,
            "label": family,
            "parent": "root",
            "value": family_totals[family],
            "color": sum(comm_birth(c) * comm_size(c) for c in children) / max(family_totals[family], 1),
            "hover": f"<b>{family}</b><br>{len(children)} communities · {family_totals[family]} structures",
        })
        for c in sorted(children, key=lambda x: -comm_size(x)):
            rows.append({
                "id": f"comm:{c}",
                "label": f"community {c}",
                "parent": family_id,
                "value": comm_size(c),
                "color": comm_birth(c),
                "hover": f"<b>{family}</b><br>community {c}<br>size {comm_size(c)} · birth ~{int(comm_birth(c))}",
            })

    if uncategorized:
        unc_id = "family:__uncategorized__"
        unc_birth = (
            sum(comm_birth(c) * comm_size(c) for c in uncategorized) / max(unc_total, 1)
        )
        rows.append({
            "id": unc_id,
            "label": "Uncategorised",
            "parent": "root",
            "value": unc_total,
            "color": unc_birth,
            "hover": (
                f"<b>Uncategorised</b><br>{len(uncategorized)} communities · {unc_total} structures<br>"
                "<i>not yet assigned a canonical family name</i>"
            ),
        })
        for c in keep:
            rows.append({
                "id": f"comm:{c}",
                "label": fmap.label(c),
                "parent": unc_id,
                "value": comm_size(c),
                "color": comm_birth(c),
                "hover": f"community {c}<br>{fmap.label(c)}<br>size {comm_size(c)} · birth ~{int(comm_birth(c))}",
            })
        if roll:
            roll_birth = sum(comm_birth(c) * comm_size(c) for c in roll) / max(roll_total, 1)
            rows.append({
                "id": "comm:__rollup__",
                "label": f"smaller uncategorised (n={len(roll)})",
                "parent": unc_id,
                "value": roll_total,
                "color": roll_birth,
                "hover": f"{len(roll)} smaller uncategorised communities · {roll_total} structures",
            })

    labels = [r["label"] for r in rows]
    parents = [r["parent"] for r in rows]
    values = [r["value"] for r in rows]
    hover = [r["hover"] for r in rows]
    ids = [r["id"] for r in rows]
    color_vals = [r["color"] for r in rows]
    valid_colors = [c for c in color_vals if c is not None]
    cmin = float(min(valid_colors)) if valid_colors else 1900.0
    cmax = float(max(valid_colors)) if valid_colors else 2025.0
    colors_for_marker = [float(c) if c is not None else cmin for c in color_vals]

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            marker={
                "colors": colors_for_marker,
                "colorscale": "Viridis",
                "cmin": cmin,
                "cmax": cmax,
                "colorbar": {"title": {"text": "birth year", "side": "right"}, "thickness": 12, "len": 0.6, "x": 1.02},
            },
            insidetextorientation="radial",
            maxdepth=3,
        )
    )
    fig.update_layout(
        margin={"l": 10, "r": 90, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def community_map_figure(min_size: int = 200, label_top_n: int = 25) -> go.Figure:
    """Force-directed community map: each centroid positioned by the
    inter-community k-NN graph layout from scripts/build_community_layout.py.
    Edges between communities are drawn with width / opacity proportional to
    the number of cross-community k-NN edges between them.
    Falls back to PCA centroid positions if no layout file is loaded.
    """
    fmap = load_frozen_map()
    using_graph_layout = bool(fmap.community_layout)
    fig = go.Figure()

    # Filter to communities at or above the requested size threshold so the
    # map doesn't try to render every long-tail basin at once. Ensure we
    # always have at least 10 communities so the slider can never produce an
    # empty plot.
    all_comm_ids = [int(c) for c in fmap.communities]
    all_sizes = np.array(
        [float(fmap.community_meta.get(c, {}).get("size", 1.0)) for c in all_comm_ids],
        dtype=float,
    )
    keep_mask = all_sizes >= float(min_size)
    if keep_mask.sum() < 10:
        keep_idx = np.argsort(-all_sizes)[:max(10, int(keep_mask.sum()))]
        keep_mask = np.zeros_like(all_sizes, dtype=bool)
        keep_mask[keep_idx] = True
    keep_indices = np.where(keep_mask)[0]
    comm_ids = [all_comm_ids[i] for i in keep_indices]
    sizes = all_sizes[keep_indices]

    n = len(comm_ids)
    layout_xy = np.empty((n, 2), dtype=float)
    in_layout_mask: list[bool] = []
    if using_graph_layout:
        layout_xs = [entry["x"] for entry in fmap.community_layout.values()]
        layout_ys = [entry["y"] for entry in fmap.community_layout.values()]
        if layout_xs and layout_ys:
            xmin, xmax = float(min(layout_xs)), float(max(layout_xs))
            ymin, ymax = float(min(layout_ys)), float(max(layout_ys))
        else:
            xmin = xmax = ymin = ymax = 0.0
        # Rescale PCA centroids of the *filtered* set into the same box.
        cx = fmap.centroids_xy[keep_indices, 0]
        cy = fmap.centroids_xy[keep_indices, 1]
        c_xmin, c_xmax = float(cx.min()), float(cx.max())
        c_ymin, c_ymax = float(cy.min()), float(cy.max())

        def _rescale(v: float, src_lo: float, src_hi: float, dst_lo: float, dst_hi: float) -> float:
            if src_hi - src_lo < 1e-12 or dst_hi - dst_lo < 1e-12:
                return (dst_lo + dst_hi) / 2.0
            return dst_lo + (v - src_lo) * (dst_hi - dst_lo) / (src_hi - src_lo)

        for i, c in enumerate(comm_ids):
            entry = fmap.community_layout.get(c)
            if entry is None:
                layout_xy[i] = (
                    _rescale(float(cx[i]), c_xmin, c_xmax, xmin, xmax),
                    _rescale(float(cy[i]), c_ymin, c_ymax, ymin, ymax),
                )
                in_layout_mask.append(False)
            else:
                layout_xy[i] = (entry["x"], entry["y"])
                in_layout_mask.append(True)
        x_label, y_label = "graph layout x", "graph layout y"
    else:
        layout_xy = fmap.centroids_xy[keep_indices].copy()
        in_layout_mask = [False] * n
        x_label, y_label = "Frozen PCA-1", "Frozen PCA-2"

    births = np.array(
        [float(fmap.community_meta.get(c, {}).get("birth_year", 2010.0)) for c in comm_ids],
        dtype=float,
    )
    log_sizes = np.log1p(sizes)
    marker_size = 12.0 + 18.0 * (log_sizes - log_sizes.min()) / max(log_sizes.max() - log_sizes.min(), 1e-6)

    edge_x: list[float] = []
    edge_y: list[float] = []
    drawn_pairs: set[tuple[int, int]] = set()
    if using_graph_layout:
        idx_by_comm = {c: i for i, c in enumerate(comm_ids)}
        for c in comm_ids:
            entry = fmap.community_layout.get(c)
            if entry is None:
                continue
            nb = entry["top_neighbor"]
            if nb < 0 or nb not in idx_by_comm:
                continue
            pair = (min(int(c), int(nb)), max(int(c), int(nb)))
            if pair in drawn_pairs:
                continue
            drawn_pairs.add(pair)
            i = idx_by_comm[c]
            j = idx_by_comm[int(nb)]
            edge_x.extend([layout_xy[i, 0], layout_xy[j, 0], None])
            edge_y.extend([layout_xy[i, 1], layout_xy[j, 1], None])
        if edge_x:
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    mode="lines",
                    line={"color": "rgba(15,109,97,0.18)", "width": 1.0},
                    hoverinfo="skip",
                    showlegend=False,
                    name="inter-community spanning edges",
                )
            )

    fallback_note = '<br><span style="color:#b56200">[PCA fallback — below layout min size]</span>'
    hover_text = [
        f"<b>{fmap.label(c)}</b><br>community {c}<br>size {int(sizes[i])} | birth ~{int(births[i])}"
        + ("" if in_layout_mask[i] else fallback_note)
        + "<br><i>click to drill down</i>"
        for i, c in enumerate(comm_ids)
    ]
    point_opacity = [0.95 if in_layout_mask[i] else 0.4 for i in range(n)]

    # Hide in-figure text labels by default — at hundreds of centroids they
    # collapse into an unreadable pile. Names still appear on hover. The
    # top-`label_top_n` slot is reserved for the largest few communities,
    # rendered with a slight upward offset to minimise pile-up.
    if label_top_n > 0:
        top_n = min(label_top_n, n)
        label_threshold = float(np.partition(sizes, -top_n)[-top_n])
        visible_label = [
            fmap.label(c) if (in_layout_mask[i] and sizes[i] >= label_threshold) else ""
            for i, c in enumerate(comm_ids)
        ]
        marker_mode = "markers+text"
    else:
        visible_label = [""] * n
        marker_mode = "markers"

    fig.add_trace(
        go.Scatter(
            x=layout_xy[:, 0],
            y=layout_xy[:, 1],
            mode=marker_mode,
            marker={
                "size": marker_size,
                "color": births,
                "colorscale": "Viridis",
                "cmin": float(births.min()) if len(births) else 1900.0,
                "cmax": float(births.max()) if len(births) else 2025.0,
                "opacity": point_opacity,
                "line": {"width": 1.2, "color": "white"},
                "colorbar": {
                    "title": {"text": "community<br>birth year", "side": "right"},
                    "thickness": 12, "len": 0.55, "x": 1.02,
                },
            },
            text=visible_label,
            textposition="top center",
            textfont={"size": 9, "color": "#333333"},
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            customdata=[int(c) for c in comm_ids],
            name=f"communities (n={n})",
        )
    )

    fig.update_layout(
        margin={"l": 30, "r": 90, "t": 36, "b": 36},
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": x_label, "zeroline": False, "gridcolor": "rgba(0,0,0,0.07)"},
        yaxis={"title": y_label, "zeroline": False, "gridcolor": "rgba(0,0,0,0.07)"},
        showlegend=False,
        annotations=[{
            "text": (
                "graph-aware layout (k-NN spring layout)" if using_graph_layout
                else "fallback PCA layout — set ICSD_COMMUNITY_LAYOUT_PATH for the graph-aware view"
            ),
            "xref": "paper", "yref": "paper", "x": 0.0, "y": 1.04,
            "xanchor": "left", "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 11, "color": "#5b6672"},
        }],
    )
    return fig


def overview_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Structural History Demo",
                                style={
                                    "display": "inline-block",
                                    "padding": "6px 10px",
                                    "border": "1px solid rgba(15,109,97,0.28)",
                                    "borderRadius": "999px",
                                    "fontSize": "12px",
                                    "letterSpacing": "0.12em",
                                    "textTransform": "uppercase",
                                    "color": "#0f6d61",
                                    "marginBottom": "14px",
                                },
                            ),
                            html.H1(
                                "Human discovery mostly densifies known crystal basins.",
                                style={"fontSize": "3.4rem", "lineHeight": "1.02", "margin": "0 0 16px"},
                            ),
                            html.P(
                                "This Dash app is the DANSE2-facing front end for the ICSD structural-history project. "
                                "It combines the paper’s core figures with upload-and-score evaluation of new crystal structures "
                                "against the frozen historical map.",
                                style={"color": "#5b6672", "lineHeight": "1.65", "fontSize": "1.05rem", "maxWidth": "58ch"},
                            ),
                        ],
                        style={**CARD_STYLE, "padding": "32px", "minHeight": "320px"},
                    ),
                    html.Div(
                        [
                            html.H2("Current paper-state summary", style={"margin": "0 0 18px", "fontSize": "1.7rem"}),
                            html.Div(
                                [metric_card(value, label) for value, label in METRICS],
                                style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "12px"},
                            ),
                        ],
                        style={**CARD_STYLE, "padding": "28px"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1.3fr 0.9fr",
                    "gap": "24px",
                    "marginBottom": "22px",
                },
            ),
            html.Div(
                [
                    html.H2("Forest of structural communities", style={"margin": "0 0 8px"}),
                    html.P(
                        "Each curated family is one slice; uncategorised communities are bucketed at the bottom. Slice value is community size; color is birth year. Click a slice to drill into it. Use the Community Map page for the spatial view.",
                        style={"margin": "0 0 18px", "color": "#5b6672", "lineHeight": "1.55"},
                    ),
                    dcc.Graph(
                        id="overview-sunburst",
                        figure=community_sunburst_figure() if APP_STATUS == "ready" else go.Figure(),
                        style={"height": "520px"},
                    ),
                ],
                style={**CARD_STYLE, "padding": "24px", "marginBottom": "22px"},
            ),
            html.Div(
                [
                    html.H2("Core figures", style={"margin": "0 0 8px"}),
                    html.P(
                        "These manuscript-supporting assets provide the historical context for the upload scorer.",
                        style={"margin": "0 0 18px", "color": "#5b6672", "lineHeight": "1.55"},
                    ),
                    html.Div(
                        [
                            figure_card(
                                "Century-scale community birth collapse",
                                "The share of structures that open new communities declines while occupancy of existing basins comes to dominate the record.",
                                FIGURES["graph_time_ratios"],
                            ),
                            figure_card(
                                "Prototype collapse",
                                "Continuous basins absorb many nominally distinct space groups, which is why the representation is more robust than discrete prototype counting.",
                                FIGURES["prototype_collapse"],
                            ),
                            figure_card(
                                "Stepping-stone mechanism",
                                "TRI-linked formulas overwhelmingly enter old structural neighborhoods instead of founding new ones.",
                                FIGURES["stepping_stone"],
                            ),
                            figure_card(
                                "AI vs human continuation",
                                "Public generative outputs are more frontier-like than ordinary held-out ICSD continuation, with MatterGen closer to the human trajectory than GNoME.",
                                FIGURES["gnome_frontier"],
                            ),
                            figure_card(
                                "Animated community growth",
                                "Decade-by-decade densification of the ICSD structural map: faded historical points fix the eye while each frame highlights the new arrivals. Click the image to open the full interactive 3D viewer.",
                                FIGURES["graph_growth_gif"],
                                href="/viewers/icsd_graph_view.html",
                                badge="Animation",
                            ),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "18px"},
                    ),
                ]
            ),
        ]
    )


def scoring_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Upload and score a CIF", style={"margin": "0 0 10px"}),
                    html.P(
                        "Upload a CIF to compute its structural embedding and place it relative to the frozen ICSD map.",
                        style={"margin": "0 0 18px", "color": "#5b6672", "lineHeight": "1.6"},
                    ),
                    config_hint(),
                    dcc.Upload(
                        id="cif-upload",
                        children=html.Div(
                            ["Drag and drop a CIF here or ", html.Span("browse", style={"color": "#0f6d61", "fontWeight": "600"})]
                        ),
                        style={
                            "width": "100%",
                            "padding": "38px 24px",
                            "borderWidth": "2px",
                            "borderStyle": "dashed",
                            "borderColor": "#b7a791",
                            "borderRadius": "18px",
                            "textAlign": "center",
                            "background": "rgba(255,255,255,0.72)",
                            "marginBottom": "18px",
                        },
                        multiple=False,
                    ),
                    html.Div(
                        id="upload-status",
                        children="No CIF uploaded yet.",
                        style={
                            "padding": "16px 18px",
                            "borderRadius": "14px",
                            "background": "rgba(15,109,97,0.08)",
                            "color": "#35514c",
                            "lineHeight": "1.55",
                        },
                    ),
                ],
                style={**CARD_STYLE, "padding": "28px"},
            ),
            html.Div(id="score-summary", style={"marginTop": "18px"}),
            html.Div(
                [
                    dcc.Graph(id="score-plot", figure=placement_figure(None) if APP_STATUS == "ready" else go.Figure(), style={"height": "560px"}),
                    html.Div(
                        "Tip: click any centroid (or any background point) to drill into that community.",
                        style={"color": "#5b6672", "fontSize": "0.85rem", "marginTop": "6px"},
                    ),
                ],
                style={**CARD_STYLE, "padding": "16px", "marginTop": "18px"},
            ),
            html.Div(id="community-detail", style={"marginTop": "18px"}),
            dcc.Store(id="community-detail-scroll"),
        ]
    )


def community_map_layout() -> html.Div:
    """Stand-alone Community Map page: graph-aware centroid layout + the same
    click-to-drill modal used on the Score page. Reuses the score-plot id and
    community-detail id so the existing callbacks fire without modification."""
    # Pick a sensible default min size: largest of (200, the size that yields
    # roughly the top-50 communities). Keeps the initial view readable on
    # both synth (12 communities) and real ICSD (5000+ communities).
    default_min_size = 200
    try:
        if APP_STATUS == "ready":
            fmap = load_frozen_map()
            sizes_sorted = sorted(
                (int(fmap.community_meta.get(int(c), {}).get("size", 0)) for c in fmap.communities),
                reverse=True,
            )
            if sizes_sorted:
                default_min_size = max(50, sizes_sorted[min(49, len(sizes_sorted) - 1)])
    except Exception:
        default_min_size = 200
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Community map", style={"margin": "0 0 10px"}),
                    html.P(
                        "Each marker is one structural community. Position is taken from a "
                        "force-directed (spring) layout of the inter-community k-NN graph "
                        "rather than raw PCA, so basins that overlap in PCA can separate "
                        "here. Marker size encodes community size; color encodes birth year. "
                        "Click any community to drill into its exemplars.",
                        style={"margin": "0 0 18px", "color": "#5b6672", "lineHeight": "1.6"},
                    ),
                    config_hint(),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Min community size", style={"fontWeight": "600", "marginBottom": "6px"}),
                                    dcc.Slider(
                                        id="community-map-min-size",
                                        min=10,
                                        max=1000,
                                        step=10,
                                        value=default_min_size,
                                        marks={10: "10", 50: "50", 100: "100", 200: "200", 500: "500", 1000: "1000"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                                style={"flex": "1", "marginRight": "20px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Show top-N labels", style={"fontWeight": "600", "marginBottom": "6px"}),
                                    dcc.Slider(
                                        id="community-map-label-top-n",
                                        min=0,
                                        max=50,
                                        step=5,
                                        value=15,
                                        marks={0: "0", 10: "10", 25: "25", 50: "50"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                                style={"width": "260px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "flex-start", "marginTop": "16px"},
                    ),
                ],
                style={**CARD_STYLE, "padding": "24px"},
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="score-plot",
                        figure=community_map_figure(min_size=default_min_size, label_top_n=15),
                        style={"height": "640px"},
                    ),
                ],
                style={**CARD_STYLE, "padding": "16px", "marginTop": "18px"},
            ),
            # Pin the upload + summary placeholders so the score-page callback
            # signature still matches when the user navigates between pages.
            html.Div(id="upload-status", style={"display": "none"}),
            html.Div(id="score-summary", style={"display": "none"}),
            dcc.Upload(id="cif-upload", children="", style={"display": "none"}),
            html.Div(id="community-detail", style={"marginTop": "18px"}),
            dcc.Store(id="community-detail-scroll"),
        ]
    )


def _molviewer_srcdoc(cif_text: str) -> str:
    """Self-contained HTML page that loads 3Dmol.js and renders the given CIF
    as ball-and-stick + sphere. Returned to an iframe srcdoc."""
    import html as html_escape

    cif_js = json.dumps(cif_text)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>html,body,#viewer{margin:0;padding:0;width:100%;height:100%;background:#f4ede1}</style>"
        "<script src='https://3Dmol.org/build/3Dmol-min.js'></script>"
        "</head><body><div id='viewer'></div>"
        "<script>"
        "(function(){var v=$3Dmol.createViewer('viewer',{backgroundColor:'#f4ede1'});"
        f"v.addModel({cif_js},'cif',{{doAssembly:true}});"
        "v.addUnitCell();"
        "v.setStyle({},{stick:{radius:0.14},sphere:{scale:0.30}});"
        "v.zoomTo();v.render();v.zoom(1.1);})();"
        "</script></body></html>"
    )


def community_detail_panel(comm: int) -> html.Div:
    """Drill-down for a single community: canonical record + top-k exemplars +
    optional 3Dmol view of the centroid CIF if ICSD_CIF_DIR is configured.
    """
    fmap = load_frozen_map()
    if int(comm) not in set(int(c) for c in fmap.communities):
        return html.Div(
            f"No data for community {comm}.",
            style={**CARD_STYLE, "padding": "20px", "color": "#5b6672"},
        )
    label = fmap.label(int(comm))
    src = fmap.label_source(int(comm))
    canonical = fmap.canonical_records.get(int(comm))
    meta = fmap.community_meta.get(int(comm), {})
    reps = fmap.representatives.get(int(comm), [])
    centroid_icsd = (canonical or {}).get("centroid_icsd_id") or (reps[0]["icsd_id"] if reps else "")

    confidence = (canonical or {}).get("confidence", "").lower()
    if src == "canonical":
        src_chip_color = "#0f6d61" if confidence == "high" else ("#b56200" if confidence == "low" else "#0f6d61")
        src_chip_text = f"Curated · {confidence or 'unspecified'}"
    elif src == "inferred":
        src_chip_color = "#34915d"
        src_chip_text = "Inferred textbook family (heuristic)"
    elif src == "prototype":
        src_chip_color = "#5b6672"
        src_chip_text = "Prototype matcher / CIF systematic name"
    else:
        src_chip_color = "#5b6672"
        src_chip_text = "No prototype assigned"

    header = html.Div(
        [
            html.Div(
                [
                    html.Div(label, style={"fontSize": "1.45rem", "fontWeight": "700"}),
                    html.Div(
                        f"community {int(comm)}  ·  size {int(meta.get('size', 0))}  ·  birth ~{int(meta.get('birth_year', 0))}",
                        style={"color": "#5b6672", "marginTop": "2px"},
                    ),
                    html.Div(
                        src_chip_text,
                        style={
                            "marginTop": "6px",
                            "color": src_chip_color,
                            "fontSize": "0.8rem",
                            "fontWeight": "600",
                            "letterSpacing": "0.05em",
                            "textTransform": "uppercase",
                        },
                    ),
                ],
            ),
        ],
        style={"marginBottom": "14px"},
    )

    canonical_evidence = (canonical or {}).get("evidence") or ""
    canonical_notes = (canonical or {}).get("notes") or ""
    evidence_block: list[Any] = []
    if canonical_evidence:
        evidence_block.append(
            html.Div([html.B("Evidence: "), canonical_evidence], style={"marginTop": "8px", "lineHeight": "1.5"})
        )
    if canonical_notes:
        evidence_block.append(
            html.Div([html.B("Notes: "), canonical_notes], style={"marginTop": "6px", "lineHeight": "1.5", "color": "#5b6672"})
        )

    if reps:
        rep_rows = [
            html.Tr([
                html.Th("#", style={"textAlign": "left", "padding": "4px 8px"}),
                html.Th("ICSD ID", style={"textAlign": "left", "padding": "4px 8px"}),
                html.Th("Formula", style={"textAlign": "left", "padding": "4px 8px"}),
                html.Th("Year", style={"textAlign": "left", "padding": "4px 8px"}),
                html.Th("Sym", style={"textAlign": "left", "padding": "4px 8px"}),
                html.Th("Bravais", style={"textAlign": "left", "padding": "4px 8px"}),
                html.Th("dist", style={"textAlign": "right", "padding": "4px 8px"}),
            ])
        ]
        for rep in reps:
            rep_rows.append(
                html.Tr([
                    html.Td(rep["rank"], style={"padding": "4px 8px", "color": "#5b6672"}),
                    html.Td(rep["icsd_id"], style={"padding": "4px 8px", "fontFamily": "monospace"}),
                    html.Td(rep["name"], style={"padding": "4px 8px"}),
                    html.Td(rep["publication_year"], style={"padding": "4px 8px"}),
                    html.Td(rep["sym_group"], style={"padding": "4px 8px"}),
                    html.Td(rep["Bravais"], style={"padding": "4px 8px"}),
                    html.Td(rep["centroid_distance"][:6] if rep["centroid_distance"] else "", style={"padding": "4px 8px", "textAlign": "right", "color": "#5b6672"}),
                ])
            )
        rep_table = html.Table(
            rep_rows,
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "fontSize": "0.92rem",
                "marginTop": "12px",
            },
        )
        rep_block = html.Div(
            [
                html.H3(f"Top {len(reps)} centroid-nearest exemplars", style={"margin": "10px 0 4px", "fontSize": "1.05rem"}),
                rep_table,
            ]
        )
    else:
        rep_block = html.Div(
            "No representative-exemplars CSV is loaded for this community. Set ICSD_REPRESENTATIVES_PATH or run extract_functional_community_representatives.py.",
            style={"marginTop": "12px", "color": "#5b6672", "lineHeight": "1.5"},
        )

    cif_block: list[Any] = []
    if centroid_icsd and ICSD_CIF_DIR is not None:
        cif_path = ICSD_CIF_DIR / f"{centroid_icsd}.cif"
        if cif_path.exists():
            try:
                cif_text = cif_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                cif_text = ""
            if cif_text:
                cif_block.append(
                    html.Div(
                        [
                            html.H3(
                                f"Centroid structure (ICSD {centroid_icsd})",
                                style={"margin": "16px 0 6px", "fontSize": "1.05rem"},
                            ),
                            # Iframe srcdoc keeps 3Dmol.js execution sandboxed
                            # from Dash's React tree, so the script reliably
                            # runs whether the panel is rendered on first load
                            # or via a click callback.
                            html.Iframe(
                                srcDoc=_molviewer_srcdoc(cif_text),
                                style={
                                    "width": "100%",
                                    "height": "360px",
                                    "border": "1px solid rgba(217, 203, 183, 0.9)",
                                    "borderRadius": "10px",
                                    "background": "#f4ede1",
                                },
                            ),
                        ]
                    )
                )

    if not cif_block:
        cif_block.append(
            html.Div(
                "Centroid structure thumbnail not available. To enable inline 3Dmol previews, set ICSD_CIF_DIR to a directory of <icsd_id>.cif files.",
                style={"marginTop": "16px", "color": "#5b6672", "fontSize": "0.85rem", "lineHeight": "1.5"},
            )
        )

    return html.Div(
        [header, *evidence_block, rep_block, *cif_block],
        style={**CARD_STYLE, "padding": "24px"},
    )


def build_app() -> Dash:
    # suppress_callback_exceptions: each route only mounts a subset of the
    # components (score-plot, community-detail, community-detail-scroll only
    # exist on /score and /communities). Without this flag Dash refuses to
    # register the click + scroll callbacks at startup because the components
    # aren't in the initial Overview layout — which is exactly why centroid
    # clicks were silently dropped before.
    app = Dash(
        __name__,
        title="ICSD Structural History Demo",
        suppress_callback_exceptions=True,
    )

    # The interactive Plotly viewers live in the figures/ tree (alongside the
    # paper) rather than under Dash's auto-served assets/ directory, so register
    # explicit Flask routes for the named viewers exposed in the overview cards.
    from flask import abort, send_file

    @app.server.route("/viewers/<path:asset_name>")
    def serve_external_viewer(asset_name: str):
        # restrict to a small whitelist so this route can never be used to
        # exfiltrate arbitrary files from the host
        whitelist = {
            "icsd_graph_view.html": INTERACTIVE_VIEWERS["graph_view_html"],
            "icsd_graph_connectivity_view.html": INTERACTIVE_VIEWERS["connectivity_view_html"],
        }
        target = whitelist.get(asset_name)
        if target is None or not target.exists():
            abort(404)
        return send_file(str(target), mimetype="text/html")

    app.layout = html.Div(
        [
            dcc.Location(id="url"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("ICSD Structural History", style={"margin": 0, "fontSize": "1.8rem"}),
                            html.Div(
                                [
                                    dcc.Link("Overview", href="/", style={"marginRight": "18px", "color": "#0f6d61", "textDecoration": "none", "fontWeight": "600"}),
                                    dcc.Link("Community map", href="/communities", style={"marginRight": "18px", "color": "#0f6d61", "textDecoration": "none", "fontWeight": "600"}),
                                    dcc.Link("Score a CIF", href="/score", style={"color": "#0f6d61", "textDecoration": "none", "fontWeight": "600"}),
                                ]
                            ),
                        ],
                        style={
                            "maxWidth": "1280px",
                            "margin": "0 auto",
                            "padding": "20px",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                        },
                    )
                ],
                style={"borderBottom": "1px solid rgba(217,203,183,0.9)", "background": "rgba(255,250,242,0.8)"},
            ),
            html.Div(id="page-body", style={"maxWidth": "1280px", "margin": "0 auto", "padding": "28px 20px 44px"}),
        ],
        style={
            "minHeight": "100vh",
            "background": "linear-gradient(180deg, #f7f2ea 0%, #f1ebe2 100%)",
            "fontFamily": 'Georgia, "Iowan Old Style", "Palatino Linotype", serif',
            "color": "#182028",
        },
    )

    @app.callback(Output("page-body", "children"), Input("url", "pathname"))
    def route(pathname: str | None):
        if pathname == "/score":
            return scoring_layout()
        if pathname == "/communities":
            return community_map_layout()
        return overview_layout()

    @app.callback(
        Output("upload-status", "children"),
        Output("score-summary", "children"),
        Output("score-plot", "figure"),
        Input("cif-upload", "contents"),
        Input("cif-upload", "filename"),
        prevent_initial_call=True,
    )
    def score_upload(contents: str | None, filename: str | None):
        if APP_STATUS != "ready":
            placeholder = go.Figure()
            placeholder.update_layout(
                margin={"l": 20, "r": 20, "t": 30, "b": 20},
                plot_bgcolor="white",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis={"visible": False},
                yaxis={"visible": False},
                annotations=[{
                    "text": "Configure frozen-map paths to enable scoring.",
                    "xref": "paper", "yref": "paper",
                    "showarrow": False, "font": {"size": 16, "color": "#5b6672"},
                }],
            )
            if not contents:
                return "No CIF uploaded yet.", html.Div(), placeholder
            return "Upload received, but frozen-map paths are not configured on this server.", html.Div(), placeholder

        # Render the historical map with no upload, so the user sees the full
        # frozen background and colorbar even before scoring anything.
        baseline_fig = placement_figure(None)
        baseline_fig.update_layout(
            annotations=[{
                "text": "Drop a CIF above to place it on this frozen historical map.",
                "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.99,
                "xanchor": "center", "yanchor": "top",
                "showarrow": False, "font": {"size": 12, "color": "#5b6672"},
            }],
        )
        if not contents:
            return "No CIF uploaded yet.", html.Div(), baseline_fig

        try:
            structure = parse_upload(contents)
            result = score_structure(structure)
        except Exception as exc:
            return f"{filename or 'Upload'} could not be scored: {exc}", html.Div(), baseline_fig

        status = (
            f"Scored {filename or 'uploaded CIF'} as {result['formula']}. "
            f"Nearest basin {result['community']} ({result.get('community_label', '')}) "
            f"with A_i = {result['accessibility']:.2f}."
        )
        return status, summary_panel(result), placement_figure(result)

    # Clientside scroll: when the community-detail panel re-renders from a
    # centroid click, smooth-scroll it into view so the user gets visible
    # feedback even when the plot is taller than the viewport. Without this,
    # the panel renders below the fold and the click looks like it did
    # nothing. We trigger only on actual click events, not on the page-load
    # placeholder render.
    app.clientside_callback(
        """
        function(clickData) {
            if (!clickData || !clickData.points) return window.dash_clientside.no_update;
            requestAnimationFrame(function() {
                requestAnimationFrame(function() {
                    var el = document.getElementById('community-detail');
                    if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
                });
            });
            return Date.now();
        }
        """,
        Output("community-detail-scroll", "data"),
        Input("score-plot", "clickData"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("community-detail", "children"),
        Input("score-plot", "clickData"),
        Input("score-summary", "children"),
        prevent_initial_call=False,
    )
    def render_community_detail(clickData: dict | None, _summary: Any) -> Any:
        # Click events expose customdata for the clicked point. Centroid traces
        # set customdata to the community id; the background trace passes the
        # background_comm array so a background click also drills down.
        if APP_STATUS != "ready":
            return html.Div()
        comm: int | None = None
        if clickData and "points" in clickData and clickData["points"]:
            cd = clickData["points"][0].get("customdata")
            if isinstance(cd, list):
                cd = cd[0] if cd else None
            try:
                comm = int(cd) if cd is not None else None
            except (TypeError, ValueError):
                comm = None
        if comm is None:
            return html.Div(
                "Click any centroid (or any background point) to drill into a community.",
                style={
                    "padding": "16px 18px",
                    "borderRadius": "14px",
                    "background": "rgba(255,250,242,0.7)",
                    "color": "#5b6672",
                    "lineHeight": "1.55",
                },
            )
        return community_detail_panel(comm)

    @app.callback(
        Output("score-plot", "figure", allow_duplicate=True),
        Input("community-map-min-size", "value"),
        Input("community-map-label-top-n", "value"),
        prevent_initial_call=True,
    )
    def update_community_map(min_size: int | None, label_top_n: int | None) -> go.Figure:
        if APP_STATUS != "ready":
            return go.Figure()
        return community_map_figure(
            min_size=int(min_size or 200),
            label_top_n=int(label_top_n if label_top_n is not None else 15),
        )

    return app


app = build_app()
server = app.server


if __name__ == "__main__":
    app.run(debug=True, port=8050)
