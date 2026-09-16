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


APP_ROOT = Path(__file__).resolve().parent
# Locate sibling scripts/ directory. Works in two layouts:
#   <repo>/dashboard/dash_app.py            (companion repo: crystal-communities-paper)
#   <repo>/resources/web_demo/dash_app.py   (working repo: crystal-communities)
SCRIPTS_ROOT = None
for candidate in (APP_ROOT.parent / "scripts", APP_ROOT.parent.parent / "scripts"):
    if (candidate / "icsd_densify_worker.py").exists():
        SCRIPTS_ROOT = candidate
        REPO_ROOT = candidate.parent
        break
if SCRIPTS_ROOT is None:
    raise RuntimeError(
        "dash_app.py: could not locate sibling scripts/ directory containing "
        "icsd_densify_worker.py."
    )
# Figures directory. The dashboard's demo assets (interactive-graph HTML,
# growth GIF, supporting PNGs) live in different places across layouts:
#   working repo:    resources/figures/icsd_densification/   (full set)
#   companion repo:  figures/icsd_densification/              (curated subset)
# The top-level companion figures/ holds only the 9 journal-named PNGs,
# which the dashboard does NOT reference, so it must not be selected.
# Pick the first candidate that actually contains a recognisable demo
# asset; fall back to the last candidate so paths still resolve (missing
# individual files degrade gracefully via image_data_uri()).
_FIG_CANDIDATES = (
    REPO_ROOT / "resources" / "figures" / "icsd_densification",
    REPO_ROOT / "figures" / "icsd_densification",
    REPO_ROOT / "figures",
)
FIG_ROOT = next(
    (c for c in _FIG_CANDIDATES
     if (c / "pipeline_schematic_repaired.png").exists()
     or (c / "icsd_graph_view.html").exists() or any(c.glob("*frontier*.png"))),
    _FIG_CANDIDATES[-1],
)
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
from frozen_backend import load_bundle, score as score_saved_structure


def env_path(name: str, fallback: str = "") -> Path | None:
    value = os.environ.get(name, fallback).strip()
    return Path(value) if value else None


DASHBOARD_MANIFEST_PATH = env_path("ICSD_DASHBOARD_MANIFEST")
ICSD_CIF_DIR = None  # Public dashboard does not expose licensed ICSD structures.
DEMO_OBSERVATION_YEAR = int(os.environ.get("ICSD_DEMO_OBSERVATION_YEAR", "2019"))
DEMO_SAMPLE_SIZE = int(os.environ.get("ICSD_DEMO_SAMPLE_SIZE", "12000"))
DEMO_RANDOM_SEED = 42


def resolve_community_label(
    comm: int,
    canonical: dict[int, str],
    prototype: dict[int, str],
    inferred: dict[int, str] | None = None,
) -> str:
    """Use only descriptions supplied by the validated repaired bundle."""
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


@dataclass
class FrozenMap:
    bundle: dict
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


APP_STATUS = "missing_paths"
APP_STATUS_ERROR = ""
FROZEN_BUNDLE = None
if DASHBOARD_MANIFEST_PATH is not None:
    try:
        FROZEN_BUNDLE = load_bundle(DASHBOARD_MANIFEST_PATH)
        APP_STATUS = "ready"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        APP_STATUS = "incompatible_features"
        APP_STATUS_ERROR = str(exc)


def image_data_uri(path: Path) -> str | None:
    # The companion repo intentionally ships only a subset of the
    # working-repo demo assets. A missing figure must degrade to a
    # placeholder card, never crash the page-routing callback.
    if not path.exists():
        return None
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
    "graph_time_ratios": FIG_ROOT / "temporal_cliff_stacked_area_repaired.png",
    "prototype_collapse": FIG_ROOT / "pipeline_schematic_repaired.png",
    "stepping_stone": FIG_ROOT / "formula_graph_tri_comparison_repaired.png",
    "gnome_frontier": FIG_ROOT / "fig3_5source_calibration_repaired.png",
    "graph_growth_gif": FIG_ROOT / "synth_prior_quadrant_repaired.png",
}
if FROZEN_BUNDLE is not None:
    FIGURES = {key: FROZEN_BUNDLE["paths"][f"figure_{key}"] for key in FIGURES}

# Legacy precomputed viewers contain obsolete coordinates and labels. The live
# community-map route uses only the validated repaired artifact bundle.
INTERACTIVE_VIEWERS = {
    "graph_view_html": FIG_ROOT / "icsd_graph_view_repaired.html",
    "connectivity_view_html": FIG_ROOT / "icsd_graph_connectivity_view_repaired.html",
}
METRICS = [
    ("167,392", "ICSD entries encoded with CrystalWeave (92.3% of 181,362 requested)"),
    ("99.3", "Mean distinct space groups in the ten largest communities"),
    ("91.8%", "TRI-shared formulas entering existing communities (16,112 of 17,556 classifiable formulas)"),
    ("Five computed cohorts", "Held-out ICSD has a higher in-basin rate in independently trained historical CrystalWeave maps"),
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
    data_uri = image_data_uri(image_path)
    if data_uri is None:
        # Asset not bundled in this repository — render a labelled
        # placeholder so the page still renders.
        img: Any = html.Div(
            f"Figure not bundled in this repository ({image_path.name})",
            style={
                "width": "100%",
                "height": "270px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "textAlign": "center",
                "padding": "0 18px",
                "color": "#9aa5b1",
                "fontSize": "0.92rem",
                "background": "rgba(255,255,255,0.6)",
                "borderBottom": "1px solid rgba(217, 203, 183, 0.9)",
            },
        )
    else:
        img = html.Img(
            src=data_uri,
            style={
                "width": "100%",
                "height": "270px",
                # `contain` preserves the entire figure (no cropped axis
                # labels); `cover` was clipping the bottom of bar charts
                # and the rotated x-axis tick labels in the stepping-stone
                # and prototype panels.
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
            f"Verified repaired map loaded. Historical cost uses observation year {DEMO_OBSERVATION_YEAR}. Upload scoring is enabled.",
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
            html.Div(
                "Frozen-map scoring requires compatible ICSD features."
                if APP_STATUS_ERROR else "Frozen-map scoring is not configured yet.",
                style={"fontWeight": "700", "marginBottom": "6px"},
            ),
            html.Div(
                APP_STATUS_ERROR or "Set ICSD_DASHBOARD_MANIFEST to the verified repaired artifact manifest to enable scoring.",
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
    if APP_STATUS != "ready" or FROZEN_BUNDLE is None:
        raise RuntimeError(APP_STATUS_ERROR or "A repaired dashboard manifest is required.")
    bundle = FROZEN_BUNDLE
    basis = bundle["basis"]
    communities = basis["communities"]
    centroids = basis["centroids"]
    labels = bundle["labels"]
    thresholds = {int(c): float(t) for c, t in zip(communities, basis["p95"])}
    community_meta = {int(c): {"size": int(n), "birth_year": int(y), "core_threshold": float(t)}
                      for c, n, y, t in zip(communities, basis["counts"], basis["birth_years"], basis["p50"])}
    canonical_records = {}
    for group in bundle["families"]["groups"]:
        for c in group["communities"]:
            canonical_records[int(c)] = {"label": group["name"], "confidence": "descriptive",
                                         "evidence": group.get("space_group_evidence", ""),
                                         "notes": group["label_status"], "centroid_icsd_id": ""}
    canonical_labels = {c: r["label"] for c, r in canonical_records.items()}
    representatives = {}
    for row in bundle["representatives"]:
        c, rank = int(row["community"]), int(row["rank_by_centroid_distance"])
        if rank <= 10:
            representatives.setdefault(c, []).append({"rank": rank, "icsd_id": row["icsd_id"],
                "name": row["reduced_formula"], "publication_year": row["year"],
                "sym_group": row["space_group"], "Bravais": "", "centroid_distance": row["centroid_distance"],
                "a": "", "b": "", "c": ""})
    community_layout = load_community_layout(bundle["paths"]["layout"])
    valid_idx = np.flatnonzero(labels >= 0)
    rng = np.random.default_rng(DEMO_RANDOM_SEED)
    sample_idx = (np.sort(rng.choice(valid_idx, size=DEMO_SAMPLE_SIZE, replace=False))
                  if len(valid_idx) > DEMO_SAMPLE_SIZE else valid_idx)
    background_xy = np.asarray(bundle["pca"][sample_idx, :2])
    background_comm = labels[sample_idx]
    background_birth = np.array([community_meta[int(c)]["birth_year"] for c in background_comm])
    background_label = [canonical_labels.get(int(c), f"community {c}") for c in background_comm]
    pads = .05 * np.maximum(np.ptp(background_xy, axis=0), 1.)
    moments = bundle["accessibility"]
    return FrozenMap(bundle=bundle, communities=communities, centroids=centroids,
        centroids_xy=centroids[:, :2], thresholds=thresholds, community_meta=community_meta,
        canonical_labels=canonical_labels, canonical_records=canonical_records,
        inferred_labels={}, prototype_labels={}, representatives=representatives,
        community_layout=community_layout, accessibility_mu=moments["icsd_raw_mu"],
        accessibility_sigma=moments["icsd_raw_sigma"], background_xy=background_xy,
        background_comm=background_comm, background_birth=background_birth, background_label=background_label,
        background_xlim=(float(background_xy[:, 0].min()-pads[0]), float(background_xy[:, 0].max()+pads[0])),
        background_ylim=(float(background_xy[:, 1].min()-pads[1]), float(background_xy[:, 1].max()+pads[1])))


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
    result = score_saved_structure(structure, fmap.bundle, DEMO_OBSERVATION_YEAR)
    comm = result["community"]
    record = fmap.canonical_records.get(comm, {})
    result.update(community_label=fmap.label(comm), label_source=fmap.label_source(comm),
                  canonical_confidence=record.get("confidence", ""),
                  canonical_evidence=record.get("evidence", ""), centroid_icsd_id="")
    return result


def year_text(year) -> str:
    return str(int(year)) if year is not None and year >= 0 else "unknown"


def birth_colors(years):
    return [float(y) if y is not None and y >= 0 else "#aaaaaa" for y in years]


def birth_min(years):
    known = [float(y) for y in years if y is not None and y >= 0]
    return min(known) if known else 1900.0


def summary_panel(result: dict[str, Any]) -> html.Div:
    birth_note = (f"First observed in {result['community_birth_year']}." if result["community_birth_year"] >= 0 else "Birth year unavailable; the cost uses the production age fallback of 2010.")
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
            source_text, source_color = "Checked community description", "#0f6d61"
    elif label_source == "inferred":
        source_text, source_color = "Inferred textbook family (heuristic)", "#34915d"
    elif label_source == "prototype":
        source_text, source_color = "Prototype matcher / CIF systematic name", "#5b6672"
    else:
        source_text, source_color = "No checked family description", "#5b6672"
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
                        [
                            html.Span(
                                frontier_label,
                                style={
                                    "display": "inline-block",
                                    "padding": "6px 10px",
                                    "marginRight": "8px",
                                    "borderRadius": "999px",
                                    "background": "rgba(255,255,255,0.82)",
                                    "border": f"1px solid {frontier_color}",
                                    "color": frontier_color,
                                    "fontWeight": "700",
                                },
                            ),

                        ],
                        style={"marginTop": "10px"},
                    ),

                ],
                style={"marginBottom": "14px"},
            ),
            html.Div(
                [
                    metric_card(f"{result['community']}", "Nearest structural basin"),
                    metric_card(f"{result['accessibility']:.2f}", "Historical accessibility cost A_i"),
                    metric_card(f"{result['distance']:.3f}", "Centroid distance"),
                    metric_card(f"{result['threshold']:.3f}", "Community p95 threshold"),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "12px"},
            ),
            html.Div(
                f"Community size {result['community_size']}. {birth_note} Uploaded structure has {result['n_sites']} sites. Cost evaluated at {result['observation_year']}; it is not a calibrated synthesis-success score. Community membership does not establish atomic-prototype identity.",
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
                "color": birth_colors(fmap.background_birth),
                "colorscale": "Viridis",
                "cmin": birth_min(fmap.background_birth),
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
                f"{lbl}<br>community {int(c)}<br>birth ~{year_text(b)}"
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
        f"size {int(centroid_size[i])} | birth ~{year_text(centroid_birth[i])}"
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
                "color": birth_colors(centroid_birth),
                "colorscale": "Viridis",
                "cmin": birth_min(fmap.background_birth),
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
                    line={"color": "#444", "width": 1.2, "dash": "dot"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[float(cxy[0])],
                    y=[float(cxy[1])],
                    mode="markers+text",
                    marker={"size": 22, "color": "#d400ff", "symbol": "star", "line": {"width": 2.0, "color": "#1a1a1a"}},
                    text=[result.get("community_label", f"community {comm}")],
                    textposition="bottom center",
                    textfont={"size": 11, "color": "#9b00b8"},
                    customdata=[int(comm)],
                    hovertemplate=(
                        f"{result.get('community_label', 'community ' + str(comm))}<br>"
                        f"community {comm}<br>"
                        f"birth {year_text(result['community_birth_year'])}<br>"
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
                    # Fluorescent orange, oversized, dark-outlined: must read
                    # instantly against the Viridis ICSD cloud and the teal
                    # nearest-basin members/hull regardless of in-basin vs
                    # frontier status.
                    "size": 26,
                    "color": "#ff6d00",
                    "line": {"width": 2.5, "color": "#1a1a1a"},
                    "symbol": "star-diamond" if result["frontier"] else "star",
                },
                hovertemplate=(
                    f"<b>{result['formula']}</b> (uploaded)<br>"
                    f"basin: {result.get('community_label', 'community ' + str(comm))}<br>"
                    f"distance to centroid: {result['distance']:.3f}<br>"
                    f"basin p95 threshold: {result['threshold']:.3f}<br>"
                    f"accessibility A_i: {result['accessibility']:.2f}<br>"
                    f"frontier: {result['frontier']}<extra></extra>"
                ),
                name=f"&#9733; {result['formula']} (uploaded)",
            )
        )

    # Axis range: default to the trimmed background cloud, but always
    # widen so the uploaded point and its nearest centroid are inside the
    # view (otherwise an edge/frontier placement gets hard-clipped and the
    # user sees a legend entry for a marker that isn't on screen).
    xr = list(fmap.background_xlim)
    yr = list(fmap.background_ylim)
    if result is not None:
        px = [float(result["xy"][0])]
        py = [float(result["xy"][1])]
        if result.get("centroid_xy") is not None:
            px.append(float(result["centroid_xy"][0]))
            py.append(float(result["centroid_xy"][1]))
        xr = [min(xr[0], *px), max(xr[1], *px)]
        yr = [min(yr[0], *py), max(yr[1], *py)]
        dx = (xr[1] - xr[0]) * 0.05 or 1.0
        dy = (yr[1] - yr[0]) * 0.05 or 1.0
        xr = [xr[0] - dx, xr[1] + dx]
        yr = [yr[0] - dy, yr[1] + dy]

    fig.update_layout(
        margin={"l": 30, "r": 90, "t": 36, "b": 36},
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "title": "Frozen PCA-1",
            "zeroline": False,
            "gridcolor": "rgba(0,0,0,0.07)",
            "range": xr,
        },
        yaxis={
            "title": "Frozen PCA-2",
            "zeroline": False,
            "gridcolor": "rgba(0,0,0,0.07)",
            "range": yr,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "font": {"size": 11}},
    )
    return fig


def unconfigured_figure(
    what: str = "This view",
) -> go.Figure:
    """Informative empty-state for the frozen-map-backed plots.

    Without a verified dashboard manifest these plots
    have no data. Returning a bare ``go.Figure()`` looks broken; this
    renders a centred explanation instead.
    """
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": (
                    f"{what} requires ICSD features matching the active encoder.<br>"
                    "Regenerate the features and frozen basis, then reload."
                ) if APP_STATUS_ERROR else (
                    f"{what} needs the frozen-map data bundle.<br>"
                    "Set ICSD_DASHBOARD_MANIFEST<br>to the verified repaired "
                    "data bundle, then reload."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14, "color": "#5b6672"},
                "align": "center",
            }
        ],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
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

    def weighted_birth(communities):
        known = [c for c in communities if comm_birth(c) >= 0]
        return (sum(comm_birth(c) * comm_size(c) for c in known) / sum(comm_size(c) for c in known)) if known else None

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
            "color": weighted_birth(children),
            "hover": f"<b>{family}</b><br>{len(children)} communities · {family_totals[family]} structures",
        })
        for c in sorted(children, key=lambda x: -comm_size(x)):
            rows.append({
                "id": f"comm:{c}",
                "label": f"community {c}",
                "parent": family_id,
                "value": comm_size(c),
                "color": comm_birth(c),
                "hover": f"<b>{family}</b><br>community {c}<br>size {comm_size(c)} · birth ~{year_text(comm_birth(c))}",
            })

    if uncategorized:
        unc_id = "family:__uncategorized__"
        unc_birth = (
            weighted_birth(uncategorized)
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
                "hover": f"community {c}<br>{fmap.label(c)}<br>size {comm_size(c)} · birth ~{year_text(comm_birth(c))}",
            })
        if roll:
            roll_birth = weighted_birth(roll)
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
    valid_colors = [c for c in color_vals if c is not None and c >= 0]
    cmin = float(min(valid_colors)) if valid_colors else 1900.0
    cmax = float(max(valid_colors)) if valid_colors else 2025.0
    colors_for_marker = birth_colors(color_vals)

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
        f"<b>{fmap.label(c)}</b><br>community {c}<br>size {int(sizes[i])} | birth ~{year_text(births[i])}"
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
                "color": birth_colors(births),
                "colorscale": "Viridis",
                "cmin": birth_min(births),
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
                else "saved PCA display for communities outside the graph-layout size cutoff"
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
                                "Interactive companion to the manuscript: explore the frozen ICSD structural map "
                                "and score new crystal structures against it. "
                                "Combines the paper’s core figures with upload-and-score evaluation of new structures "
                                "against the frozen historical map.",
                                style={"color": "#5b6672", "lineHeight": "1.65", "fontSize": "1.05rem", "maxWidth": "58ch"},
                            ),
                        ],
                        style={**CARD_STYLE, "padding": "32px", "minHeight": "320px"},
                    ),
                    html.Div(
                        [
                            html.H2("Dataset at a glance", style={"margin": "0 0 18px", "fontSize": "1.7rem"}),
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
                        "Each curated family is one slice; uncategorised communities are bucketed at the bottom. Slice value is community size; color is known birth year (grey when unavailable). Click a slice to drill into it. Use the Community Map page for the spatial view.",
                        style={"margin": "0 0 18px", "color": "#5b6672", "lineHeight": "1.55"},
                    ),
                    dcc.Graph(
                        id="overview-sunburst",
                        figure=community_sunburst_figure() if APP_STATUS == "ready" else unconfigured_figure("The community forest"),
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
                                "Experimental reference pipeline",
                                "The repaired representation organizes structures into neighborhoods spanning multiple space groups; it does not certify prototype identity.",
                                FIGURES["prototype_collapse"],
                            ),
                            figure_card(
                                "Structural and thermodynamic networks",
                                "TRI-linked formulas overwhelmingly enter old structural neighborhoods instead of founding new ones.",
                                FIGURES["stepping_stone"],
                            ),
                            figure_card(
                                "Computed cohorts and held-out ICSD",
                                "Historical CrystalWeave maps place held-out ICSD in-basin more often than all five computed cohorts. GNoME shows greater familiarity under Graphlet and historical AMD maps.",
                                FIGURES["gnome_frontier"],
                            ),
                            figure_card(
                                "Two axes of historical precedent",
                                "Structural proximity and reduced-formula precedent describe different aspects of the experimental record.",
                                FIGURES["graph_growth_gif"],
                                badge="Historical map",
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
                    dcc.Graph(id="score-plot", figure=placement_figure(None) if APP_STATUS == "ready" else unconfigured_figure("Structure placement"), style={"height": "560px"}),
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
                        figure=(
                            community_map_figure(min_size=default_min_size, label_top_n=15)
                            if APP_STATUS == "ready"
                            else unconfigured_figure("The community map")
                        ),
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
        src_chip_text = "No checked family description"

    header = html.Div(
        [
            html.Div(
                [
                    html.Div(label, style={"fontSize": "1.45rem", "fontWeight": "700"}),
                    html.Div(
                        f"community {int(comm)}  ·  size {int(meta.get('size', 0))}  ·  birth ~{year_text(meta.get('birth_year', -1))}",
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
            "No central exemplars were included for this community in the repaired evidence bundle.",
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
                    "text": (
                        "Compatible ICSD features are required to enable scoring."
                        if APP_STATUS_ERROR else "Configure frozen-map paths to enable scoring."
                    ),
                    "xref": "paper", "yref": "paper",
                    "showarrow": False, "font": {"size": 16, "color": "#5b6672"},
                }],
            )
            if not contents:
                return "No CIF uploaded yet.", html.Div(), placeholder
            return (
                f"Upload could not be scored: {APP_STATUS_ERROR}"
                if APP_STATUS_ERROR else "Upload received, but frozen-map paths are not configured on this server."
            ), html.Div(), placeholder

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
            return unconfigured_figure("The community map")
        return community_map_figure(
            min_size=int(min_size or 200),
            label_top_n=int(label_top_n if label_top_n is not None else 15),
        )

    return app


app = build_app()
server = app.server


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
