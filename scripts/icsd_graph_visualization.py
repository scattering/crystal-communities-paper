#!/usr/bin/env python3
"""Animated and interactive views of ICSD graph communities through time.

Outputs (under --output-dir):
  icsd_graph_growth.gif        2D animation, fixed axis limits, faded history,
                               highlighted new arrivals per decade
  icsd_graph_view.html         Plotly viewer (2D or 3D) with decade slider and
                               per-community trails over time
  top_communities_for_view.json   Resolved labels actually rendered

Visual fixes vs the previous version:
  * Axis labels match the projection (PCA / UMAP / Precomputed); the previous
    matplotlib animation always said "PCA 1 / PCA 2" even when UMAP was used.
  * Axis limits are computed once from the full point cloud and held fixed
    across frames so the eye does not interpret autoscale jitter as motion.
  * Top-N defaults to 10 communities (was 25); canonical family names are
    used when available; long stoichiometric strings are abbreviated.
  * Centroid labels routed through a simple greedy de-overlap (or adjustText
    when installed) so labels no longer pile on top of each other.
  * Marker size encodes community size (sqrt scaled); color encodes community
    birth decade with a sequential colormap so the eye can see "old vs new".
  * Per-decade frame fades historical points to alpha~0.12 and renders the
    current decade's new arrivals at full opacity with a thin outline. This
    is the actual claim of the paper -- new structures densify known basins.
  * 3D HTML adds per-community trails: each community's centroid trajectory
    over decades is drawn as a faint line so growth in 3D is visible.
  * HTML payload is reduced by downsampling background structures (kept ones
    are still all top-community members; only outliers are subsampled).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None

try:  # optional but strongly preferred
    from adjustText import adjust_text  # type: ignore
except ImportError:  # pragma: no cover
    adjust_text = None


# ----------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--community-dir", required=True)
    parser.add_argument("--time-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coords-file", default="features_pca.npy")
    parser.add_argument("--top-communities-file", default="top_communities.json")
    parser.add_argument("--community-labels-file", default="community_prototype_labels.json")
    parser.add_argument("--canonical-labels-csv", default=None)
    parser.add_argument("--assignments-file", default="community_assignments.csv")
    parser.add_argument("--top-n-communities", type=int, default=10)
    parser.add_argument("--recent-decades-only", action="store_true")
    parser.add_argument("--start-decade", default=None)
    parser.add_argument("--end-decade", default=None)
    parser.add_argument(
        "--projection-method",
        choices=["pca", "umap", "precomputed"],
        default="umap",
        help="2D/3D projection. PCA is preferred for paper figures because it is "
             "an exact out-of-sample map; UMAP gives more visual separation but "
             "should not be reused as a frozen embedding.",
    )
    parser.add_argument("--html-dimensions", choices=["2d", "3d"], default="3d")
    parser.add_argument(
        "--max-background-nodes",
        type=int,
        default=8000,
        help="Cap on background (non-top-community) structures included in the "
             "interactive HTML to keep file size and DOM cost reasonable.",
    )
    parser.add_argument(
        "--label-max-chars",
        type=int,
        default=28,
        help="Truncate centroid labels longer than this many characters with an ellipsis.",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Render non-top-community structures faintly. Default is to drop "
             "them so the figure conveys the top-community story without periphery noise.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------- I/O helpers
def load_assignments(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_year = row.get("year", "").strip()
            rows.append(
                {
                    "icsd_id": int(row["icsd_id"]),
                    "year": int(raw_year) if raw_year else None,
                    "community": int(row["community"]),
                }
            )
    return rows


def normalize_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {";", ".", "?", "unknown", "Unknown", "n/a", "N/A"}:
        return None
    return text


def load_better_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for row in data:
        community = int(row["community"])
        label = normalize_label(row.get("label"))
        if label:
            out[community] = label
    return out


def load_canonical_labels(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("kind") != "graph_community":
                continue
            label = normalize_label(row.get("canonical_family_name"))
            if not label:
                continue
            out[int(row["id"])] = label
    return out


# --------------------------------------------------------------- decade utils
def decade_from_year(year: int | None) -> str:
    if year is None:
        return "unknown"
    return f"{(year // 10) * 10}s"


def decade_sort_key(decade: str) -> tuple[int, str]:
    if decade == "unknown":
        return (10**9, decade)
    return (int(decade[:-1]), decade)


def decade_in_window(decade: str, start: str | None, end: str | None) -> bool:
    if decade == "unknown":
        return False
    if start is not None and decade < start:
        return False
    if end is not None and decade > end:
        return False
    return True


# ---------------------------------------------------------------- projection
def project_coords(coords: np.ndarray, method: str, n_components: int) -> np.ndarray:
    if method == "precomputed":
        return coords[:, : min(coords.shape[1], n_components)]
    if coords.shape[1] <= n_components:
        return coords[:, :n_components]
    if method == "umap":
        if umap is None:
            raise RuntimeError("projection-method=umap requires umap-learn")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=30,
            min_dist=0.08,
            metric="euclidean",
            random_state=42,
        )
        return reducer.fit_transform(coords)
    return PCA(n_components=n_components, random_state=42).fit_transform(coords)


def projection_axis_labels(method: str, n: int) -> list[str]:
    name = {"pca": "PCA", "umap": "UMAP", "precomputed": "Embedding"}[method]
    return [f"{name} {i + 1}" for i in range(n)]


# ----------------------------------------------------------- label utilities
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+\.?\d*|\.\d+)?")
_LONG_FORMULA_HINT = re.compile(r"\d+\.\d{2,}")


def shorten_label(label: str, max_chars: int) -> str:
    label = label.strip()
    if len(label) <= max_chars:
        return label
    if _LONG_FORMULA_HINT.search(label):
        # Heavy compositional string with weird stoichiometries -- collapse to the
        # element list to avoid pretending a precise stoichiometry is meaningful.
        elements = [match.group(1) for match in _FORMULA_TOKEN.finditer(label)]
        seen: list[str] = []
        for e in elements:
            if e not in seen:
                seen.append(e)
            if sum(len(x) for x in seen) + len(seen) > max_chars - 3:
                break
        if seen:
            condensed = "".join(seen) + "*"
            if len(condensed) <= max_chars:
                return condensed
    return label[: max_chars - 1] + "…"


def resolve_labels(
    top_communities: list[dict[str, object]],
    better: dict[int, str],
    canonical: dict[int, str],
    max_chars: int,
) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in top_communities:
        community = int(item["community"])
        candidate = canonical.get(community) or better.get(community) or str(item.get("label") or community)
        out[community] = shorten_label(str(candidate), max_chars)
    return out


def birth_decade_color_map(top_communities: list[dict[str, object]]):
    births = [int(item["birth_year"]) for item in top_communities if item.get("birth_year") is not None]
    if not births:
        norm = colors.Normalize(vmin=1900, vmax=2020)
    else:
        norm = colors.Normalize(vmin=min(births), vmax=max(births))
    cmap = plt.get_cmap("viridis")

    def lookup(birth_year: int | None) -> str:
        if birth_year is None:
            return "#888888"
        return colors.to_hex(cmap(norm(int(birth_year))))

    return lookup, cmap, norm


def marker_size_from_community_size(size: int, base: float = 36.0, gain: float = 4.5) -> float:
    return base + gain * math.sqrt(max(int(size), 1))


# ------------------------------------------------------------- label placement
def greedy_label_offsets(
    centroids: np.ndarray,
    text_widths: np.ndarray,
    text_height: float,
    radius: float,
) -> np.ndarray:
    """Tiny greedy de-overlap: nudge labels onto a small ring around each
    centroid so collisions are reduced. Returns label anchor coordinates with
    the same shape as ``centroids``. Used when adjustText is not available."""
    angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    placed = np.empty_like(centroids)
    occupied: list[tuple[float, float, float, float]] = []  # x0, y0, x1, y1
    order = np.argsort(-text_widths)  # widest first
    for idx in order:
        cx, cy = centroids[idx]
        w = text_widths[idx]
        h = text_height
        best = None
        best_score = math.inf
        for off in offsets:
            ax_ = cx + off[0]
            ay_ = cy + off[1]
            x0, y0, x1, y1 = ax_ - w / 2, ay_ - h / 2, ax_ + w / 2, ay_ + h / 2
            overlap = 0.0
            for ox0, oy0, ox1, oy1 in occupied:
                if x0 < ox1 and x1 > ox0 and y0 < oy1 and y1 > oy0:
                    overlap += (min(x1, ox1) - max(x0, ox0)) * (min(y1, oy1) - max(y0, oy0))
            if overlap < best_score:
                best_score = overlap
                best = (ax_, ay_, x0, y0, x1, y1)
                if overlap == 0:
                    break
        ax_, ay_, x0, y0, x1, y1 = best
        occupied.append((x0, y0, x1, y1))
        placed[idx] = (ax_, ay_)
    return placed


# -------------------------------------------------------------- 2D animation
def write_animation(
    coords2d: np.ndarray,
    rows: list[dict[str, object]],
    top_communities: list[dict[str, object]],
    label_map: dict[int, str],
    color_lookup,
    cmap,
    norm,
    out_path: Path,
    recent_decades_only: bool,
    start_decade: str | None,
    end_decade: str | None,
    axis_labels: list[str],
    include_outliers: bool,
) -> None:
    top_ids = [int(item["community"]) for item in top_communities]
    top_set = set(top_ids)
    community_size = {int(item["community"]): int(item.get("size", 0) or 0) for item in top_communities}
    community_birth = {int(item["community"]): item.get("birth_year") for item in top_communities}

    decades = sorted(
        {decade_from_year(row["year"]) for row in rows if row["year"] is not None},
        key=decade_sort_key,
    )
    if recent_decades_only:
        decades = [d for d in decades if d >= "1980s"]
    if start_decade is not None:
        decades = [d for d in decades if d >= start_decade]
    if end_decade is not None:
        decades = [d for d in decades if d <= end_decade]
    if not decades:
        raise ValueError("No decades available for animation")

    # Restrict to top-community structures we will actually render.
    top_indices = [
        idx
        for idx, row in enumerate(rows)
        if row["year"] is not None
        and int(row["community"]) in top_set
        and decade_in_window(decade_from_year(row["year"]), decades[0], decades[-1])
    ]
    if not top_indices:
        raise ValueError("No top-community structures inside the requested window")

    fixed_xy = coords2d[top_indices]
    pad_x = 0.05 * (float(np.ptp(fixed_xy[:, 0])) or 1.0)
    pad_y = 0.05 * (float(np.ptp(fixed_xy[:, 1])) or 1.0)
    xlim = (fixed_xy[:, 0].min() - pad_x, fixed_xy[:, 0].max() + pad_x)
    ylim = (fixed_xy[:, 1].min() - pad_y, fixed_xy[:, 1].max() + pad_y)

    by_decade_index: dict[str, list[int]] = {d: [] for d in decades}
    for idx in top_indices:
        d = decade_from_year(rows[idx]["year"])
        if d in by_decade_index:
            by_decade_index[d].append(idx)

    bg_indices: list[int] = []
    if include_outliers:
        bg_indices = [
            idx
            for idx, row in enumerate(rows)
            if row["year"] is not None
            and int(row["community"]) not in top_set
            and decade_in_window(decade_from_year(row["year"]), decades[0], decades[-1])
        ]

    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=170)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.grid(True, color="#e6e6e6", linewidth=0.7)

    # Colorbar must be created *once* on its own axes before animation begins.
    # `ax.clear()` inside update() wipes scatter artists but leaves the colorbar
    # axis intact, so adding a colorbar per frame stacked them across the canvas.
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label("community birth year", fontsize=9)

    def update(frame_idx: int):
        decade = decades[frame_idx]
        ax.clear()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.grid(True, color="#e6e6e6", linewidth=0.7)
        ax.set_title(
            f"ICSD graph communities through time -- {decade}\n"
            f"top {len(top_communities)} communities, color = community birth year",
            fontsize=11,
        )

        if include_outliers and bg_indices:
            ax.scatter(coords2d[bg_indices, 0], coords2d[bg_indices, 1], s=3, color="#cccccc", alpha=0.18)

        for community in top_ids:
            comm_idx = [
                i for i in top_indices
                if int(rows[i]["community"]) == community and decade_from_year(rows[i]["year"]) <= decade
            ]
            if not comm_idx:
                continue
            color = color_lookup(community_birth.get(community))
            new_idx = [i for i in comm_idx if decade_from_year(rows[i]["year"]) == decade]
            old_idx = [i for i in comm_idx if i not in set(new_idx)]
            if old_idx:
                xy = coords2d[old_idx]
                ax.scatter(xy[:, 0], xy[:, 1], s=8, color=color, alpha=0.18, linewidths=0)
            if new_idx:
                xy = coords2d[new_idx]
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=18,
                    color=color,
                    alpha=0.92,
                    edgecolors="#222222",
                    linewidths=0.4,
                )

        centroid_xy = []
        centroid_size = []
        centroid_color = []
        centroid_text = []
        for community in top_ids:
            members = [
                i for i in top_indices
                if int(rows[i]["community"]) == community and decade_from_year(rows[i]["year"]) <= decade
            ]
            if not members:
                continue
            cx, cy = coords2d[members].mean(axis=0)
            centroid_xy.append((cx, cy))
            centroid_size.append(marker_size_from_community_size(community_size.get(community, len(members))))
            centroid_color.append(color_lookup(community_birth.get(community)))
            centroid_text.append(label_map.get(community, str(community)))

        if centroid_xy:
            cxy = np.asarray(centroid_xy)
            ax.scatter(
                cxy[:, 0],
                cxy[:, 1],
                s=centroid_size,
                c=centroid_color,
                edgecolors="#111111",
                linewidths=0.9,
                zorder=5,
            )
            text_artists = []
            for (x, y), label in zip(centroid_xy, centroid_text):
                text_artists.append(
                    ax.text(
                        x,
                        y,
                        label,
                        fontsize=8.5,
                        color="#111111",
                        ha="center",
                        va="center",
                        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5),
                        zorder=6,
                    )
                )
            if adjust_text is not None:
                adjust_text(
                    text_artists,
                    ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5),
                    expand_points=(1.2, 1.4),
                    expand_text=(1.05, 1.2),
                )
            else:
                widths = np.asarray([0.04 * (xlim[1] - xlim[0]) * max(len(t), 6) / 12.0 for t in centroid_text])
                anchors = greedy_label_offsets(
                    np.asarray(centroid_xy),
                    widths,
                    0.04 * (ylim[1] - ylim[0]),
                    radius=0.04 * max(xlim[1] - xlim[0], ylim[1] - ylim[0]),
                )
                for artist, (ax_, ay_), (cx, cy) in zip(text_artists, anchors, centroid_xy):
                    artist.set_position((ax_, ay_))
                    if (ax_ - cx) ** 2 + (ay_ - cy) ** 2 > 1e-12:
                        ax.plot([cx, ax_], [cy, ay_], color="#bbbbbb", lw=0.5, zorder=4)

    anim = FuncAnimation(fig, update, frames=len(decades), interval=1200, repeat_delay=1500)
    anim.save(out_path, writer=PillowWriter(fps=1))
    plt.close(fig)


# ------------------------------------------------------- interactive HTML
def write_plotly_html(
    coords: np.ndarray,
    rows: list[dict[str, object]],
    top_communities: list[dict[str, object]],
    label_map: dict[int, str],
    color_lookup,
    norm,
    out_path: Path,
    start_decade: str | None,
    end_decade: str | None,
    html_dimensions: str,
    axis_labels: list[str],
    max_background_nodes: int,
) -> None:
    top_ids = [int(item["community"]) for item in top_communities]
    top_set = set(top_ids)
    community_meta = {
        int(item["community"]): {
            "label": label_map[int(item["community"])],
            "birth_year": item.get("birth_year"),
            "size": int(item.get("size", 0) or 0),
            "color": color_lookup(item.get("birth_year")),
        }
        for item in top_communities
    }

    decades = sorted(
        {
            decade_from_year(row["year"])
            for row in rows
            if row["year"] is not None and decade_in_window(decade_from_year(row["year"]), start_decade, end_decade)
        },
        key=decade_sort_key,
    )
    if not decades:
        raise ValueError("No decades available in the requested window for HTML view")

    n_dim = 3 if html_dimensions == "3d" else 2

    def vec(idx: int) -> tuple[float, float, float]:
        x = float(coords[idx, 0])
        y = float(coords[idx, 1])
        z = float(coords[idx, 2]) if coords.shape[1] > 2 and n_dim == 3 else 0.0
        return x, y, z

    node_records = []
    for idx, row in enumerate(rows):
        community = int(row["community"])
        decade = decade_from_year(row["year"])
        if community not in top_set or row["year"] is None or not decade_in_window(decade, start_decade, end_decade):
            continue
        x, y, z = vec(idx)
        node_records.append(
            {
                "idx": idx,
                "community": community,
                "decade": decade,
                "year": int(row["year"]),
                "x": x,
                "y": y,
                "z": z,
                "label": community_meta[community]["label"],
                "color": community_meta[community]["color"],
            }
        )

    background = []
    bg_pool = [
        idx
        for idx, row in enumerate(rows)
        if row["year"] is not None
        and int(row["community"]) not in top_set
        and decade_in_window(decade_from_year(row["year"]), start_decade, end_decade)
    ]
    if bg_pool:
        rng = np.random.default_rng(42)
        if len(bg_pool) > max_background_nodes:
            bg_pool = rng.choice(bg_pool, size=max_background_nodes, replace=False).tolist()
        for idx in bg_pool:
            x, y, z = vec(idx)
            background.append(
                {
                    "idx": int(idx),
                    "decade": decade_from_year(rows[idx]["year"]),
                    "year": int(rows[idx]["year"]),
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

    centroid_trails: dict[int, list[dict[str, float]]] = {community: [] for community in top_ids}
    for community in top_ids:
        for decade in decades:
            members = [
                row for row in node_records if row["community"] == community and decade_sort_key(row["decade"]) <= decade_sort_key(decade)
            ]
            if not members:
                continue
            cx = float(np.mean([m["x"] for m in members]))
            cy = float(np.mean([m["y"] for m in members]))
            cz = float(np.mean([m["z"] for m in members]))
            centroid_trails[community].append({"decade": decade, "x": cx, "y": cy, "z": cz, "n": len(members)})

    plot_type = "scatter3d" if n_dim == 3 else "scattergl"
    if n_dim == 3:
        axis_block = (
            "scene: {{xaxis: {{title: {x!r}}}, yaxis: {{title: {y!r}}}, zaxis: {{title: {z!r}}}, "
            "bgcolor: '#ffffff'}},"
        ).format(x=axis_labels[0], y=axis_labels[1], z=axis_labels[2] if len(axis_labels) > 2 else "Axis 3")
    else:
        axis_block = (
            "xaxis: {{title: {x!r}, zeroline: false, gridcolor: 'rgba(0,0,0,0.07)'}}, "
            "yaxis: {{title: {y!r}, zeroline: false, gridcolor: 'rgba(0,0,0,0.07)'}},"
        ).format(x=axis_labels[0], y=axis_labels[1])

    trails_payload = [
        {
            "community": community,
            "label": community_meta[community]["label"],
            "color": community_meta[community]["color"],
            "points": pts,
        }
        for community, pts in centroid_trails.items()
        if pts
    ]

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ICSD Graph Communities</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: 'Inter', system-ui, sans-serif; background: #f7f7f7; color: #111; }}
    #controls {{
      display: flex; gap: 12px; align-items: center; padding: 10px 16px;
      background: #f0f0f0; border-bottom: 1px solid #d7d7d7;
      position: sticky; top: 0; z-index: 20;
    }}
    #plot {{ width: 100vw; height: calc(100vh - 56px); }}
    #decadeValue {{ min-width: 90px; font-weight: 700; }}
    input[type=range] {{ width: min(520px, 60vw); }}
    label {{ font-size: 12px; color: #444; }}
    .legend {{ font-size: 12px; color: #333; }}
  </style>
</head>
<body>
  <div id="controls">
    <button id="playBtn" type="button">Play</button>
    <button id="pauseBtn" type="button">Pause</button>
    <label for="decadeSlider">Decade</label>
    <input id="decadeSlider" type="range" min="0" max="{len(decades) - 1}" step="1" value="0">
    <span id="decadeValue">{decades[0]}</span>
    <label for="trailToggle"><input id="trailToggle" type="checkbox" checked> show centroid trails</label>
    <label for="bgToggle"><input id="bgToggle" type="checkbox" checked> show background structures</label>
  </div>
  <div id="plot"></div>
  <script>
    const decades = {json.dumps(decades)};
    const nodeRecords = {json.dumps(node_records)};
    const background = {json.dumps(background)};
    const topIds = {json.dumps(top_ids)};
    const communityMeta = {json.dumps(community_meta)};
    const trails = {json.dumps(trails_payload)};
    const nDim = {n_dim};
    const dimMin = {float(norm.vmin)};
    const dimMax = {float(norm.vmax)};

    function decadeRank(d) {{ return decades.indexOf(d); }}

    function buildFrame(decadeIdx) {{
      const decade = decades[decadeIdx];
      const rank = decadeRank(decade);
      const showBg = document.getElementById('bgToggle').checked;
      const showTrails = document.getElementById('trailToggle').checked;

      const oldX=[], oldY=[], oldZ=[], oldColor=[], oldText=[];
      const newX=[], newY=[], newZ=[], newColor=[], newText=[];
      for (const row of nodeRecords) {{
        if (decadeRank(row.decade) > rank) continue;
        const isNew = (row.decade === decade);
        const tgtX = isNew ? newX : oldX;
        const tgtY = isNew ? newY : oldY;
        const tgtZ = isNew ? newZ : oldZ;
        const tgtC = isNew ? newColor : oldColor;
        const tgtT = isNew ? newText : oldText;
        tgtX.push(row.x); tgtY.push(row.y); tgtZ.push(row.z);
        tgtC.push(row.color);
        tgtT.push(`${{row.label}}<br>community ${{row.community}}<br>year ${{row.year}}`);
      }}

      const bgX=[], bgY=[], bgZ=[];
      if (showBg) {{
        for (const row of background) {{
          if (decadeRank(row.decade) > rank) continue;
          bgX.push(row.x); bgY.push(row.y); bgZ.push(row.z);
        }}
      }}

      const cX=[], cY=[], cZ=[], cColor=[], cText=[], cLabel=[], cSize=[];
      for (const community of topIds) {{
        const trail = trails.find(t => t.community === community);
        if (!trail) continue;
        const visible = trail.points.filter(p => decadeRank(p.decade) <= rank);
        if (!visible.length) continue;
        const last = visible[visible.length - 1];
        cX.push(last.x); cY.push(last.y); cZ.push(last.z);
        const meta = communityMeta[String(community)] || communityMeta[community];
        cColor.push(meta.color);
        cLabel.push(meta.label);
        cSize.push(8 + 2.5 * Math.sqrt(meta.size || last.n));
        cText.push(`${{meta.label}}<br>community ${{community}}<br>birth ${{meta.birth_year}}<br>size ${{meta.size}}`);
      }}

      const trailX=[], trailY=[], trailZ=[];
      if (showTrails) {{
        for (const trail of trails) {{
          const visible = trail.points.filter(p => decadeRank(p.decade) <= rank);
          for (let i = 0; i < visible.length; i++) {{
            trailX.push(visible[i].x); trailY.push(visible[i].y); trailZ.push(visible[i].z);
          }}
          trailX.push(null); trailY.push(null); trailZ.push(null);
        }}
      }}
      return {{decade, oldX, oldY, oldZ, oldColor, oldText, newX, newY, newZ, newColor, newText, bgX, bgY, bgZ, cX, cY, cZ, cColor, cText, cLabel, cSize, trailX, trailY, trailZ}};
    }}

    const initial = buildFrame(0);
    const baseMarker = nDim === 3 ? {{}} : {{line: {{width: 0}}}};
    const bgTrace = {{
      x: initial.bgX, y: initial.bgY, z: initial.bgZ,
      mode: 'markers', type: '{plot_type}',
      marker: Object.assign({{color: '#bdbdbd', size: nDim === 3 ? 2 : 3, opacity: 0.18}}, baseMarker),
      hoverinfo: 'skip', name: 'other ICSD'
    }};
    const oldNodeTrace = {{
      x: initial.oldX, y: initial.oldY, z: initial.oldZ,
      mode: 'markers', type: '{plot_type}',
      marker: Object.assign({{color: initial.oldColor, size: nDim === 3 ? 3 : 4, opacity: 0.30}}, baseMarker),
      hoverinfo: 'skip', name: 'historical'
    }};
    const newNodeTrace = {{
      x: initial.newX, y: initial.newY, z: initial.newZ,
      text: initial.newText,
      mode: 'markers', type: '{plot_type}',
      marker: {{color: initial.newColor, size: nDim === 3 ? 4 : 6, opacity: 0.95, line: {{width: 0.6, color: '#222'}}}},
      hovertemplate: '%{{text}}<extra></extra>',
      name: 'new this decade'
    }};
    const trailTrace = {{
      x: initial.trailX, y: initial.trailY, z: initial.trailZ,
      mode: 'lines', type: '{plot_type}',
      line: {{color: 'rgba(60,60,60,0.35)', width: 1.5}},
      hoverinfo: 'skip', name: 'centroid trails'
    }};
    const centroidTrace = {{
      x: initial.cX, y: initial.cY, z: initial.cZ,
      text: initial.cLabel, hovertext: initial.cText,
      mode: 'markers+text', type: '{plot_type}',
      textposition: 'top center',
      hovertemplate: '%{{hovertext}}<extra></extra>',
      marker: {{color: initial.cColor, size: initial.cSize, line: {{color: '#111', width: 1}}}},
      textfont: {{size: 11, color: '#111'}},
      name: 'community centroid'
    }};

    const layout = {{
      title: 'ICSD graph communities through time -- ' + initial.decade,
      {axis_block}
      paper_bgcolor: '#f7f7f7',
      plot_bgcolor: '#ffffff',
      margin: {{l: 50, r: 50, t: 50, b: 50}},
      legend: {{orientation: 'h', y: -0.05}},
      annotations: [{{
        xref: 'paper', yref: 'paper', x: 0.99, y: 0.99,
        xanchor: 'right', yanchor: 'top', showarrow: false,
        text: '<span style="color:#777">color = community birth year</span>'
      }}]
    }};
    Plotly.newPlot('plot', [bgTrace, oldNodeTrace, newNodeTrace, trailTrace, centroidTrace], layout, {{responsive: true}});

    const slider = document.getElementById('decadeSlider');
    const decadeValue = document.getElementById('decadeValue');
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    document.getElementById('trailToggle').addEventListener('change', () => render(Number(slider.value)));
    document.getElementById('bgToggle').addEventListener('change', () => render(Number(slider.value)));

    function render(idx) {{
      const f = buildFrame(idx);
      decadeValue.textContent = f.decade;
      Plotly.restyle('plot', {{x: [f.bgX], y: [f.bgY], z: [f.bgZ]}}, [0]);
      Plotly.restyle('plot', {{x: [f.oldX], y: [f.oldY], z: [f.oldZ], 'marker.color': [f.oldColor]}}, [1]);
      Plotly.restyle('plot', {{x: [f.newX], y: [f.newY], z: [f.newZ], text: [f.newText], 'marker.color': [f.newColor]}}, [2]);
      Plotly.restyle('plot', {{x: [f.trailX], y: [f.trailY], z: [f.trailZ]}}, [3]);
      Plotly.restyle('plot', {{x: [f.cX], y: [f.cY], z: [f.cZ], text: [f.cLabel], hovertext: [f.cText], 'marker.color': [f.cColor], 'marker.size': [f.cSize]}}, [4]);
      Plotly.relayout('plot', {{title: 'ICSD graph communities through time -- ' + f.decade}});
    }}

    slider.addEventListener('input', e => render(Number(e.target.value)));
    let timer = null;
    playBtn.addEventListener('click', () => {{
      if (timer !== null) return;
      timer = window.setInterval(() => {{
        const next = Number(slider.value) + 1;
        if (next >= decades.length) {{ clearInterval(timer); timer = null; return; }}
        slider.value = String(next);
        render(next);
      }}, 1100);
    }});
    pauseBtn.addEventListener('click', () => {{ if (timer !== null) {{ clearInterval(timer); timer = null; }} }});
  </script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


# -------------------------------------------------------------- entry point
def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    community_dir = Path(args.community_dir)
    time_dir = Path(args.time_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = np.load(run_dir / args.coords_file)
    n_dim_html = 3 if args.html_dimensions == "3d" else 2
    coords_projected = project_coords(coords, args.projection_method, n_dim_html)
    coords_2d = coords_projected[:, :2]
    rows = load_assignments(community_dir / args.assignments_file)
    top_communities = json.loads((time_dir / args.top_communities_file).read_text())[: args.top_n_communities]
    better = load_better_labels(community_dir / args.community_labels_file)
    canonical = load_canonical_labels(Path(args.canonical_labels_csv) if args.canonical_labels_csv else None)
    label_map = resolve_labels(top_communities, better, canonical, args.label_max_chars)
    color_lookup, cmap, norm = birth_decade_color_map(top_communities)
    axis_labels = projection_axis_labels(args.projection_method, n_dim_html)

    if args.recent_decades_only and args.start_decade is None:
        args.start_decade = "1980s"

    write_plotly_html(
        coords_projected,
        rows,
        top_communities,
        label_map,
        color_lookup,
        norm,
        out_dir / "icsd_graph_view.html",
        args.start_decade,
        args.end_decade,
        args.html_dimensions,
        axis_labels,
        args.max_background_nodes,
    )
    write_animation(
        coords_2d,
        rows,
        top_communities,
        label_map,
        color_lookup,
        cmap,
        norm,
        out_dir / "icsd_graph_growth.gif",
        args.recent_decades_only,
        args.start_decade,
        args.end_decade,
        axis_labels[:2],
        include_outliers=args.include_outliers,
    )
    with (out_dir / "top_communities_for_view.json").open("w") as handle:
        json.dump(
            [
                {
                    "community": int(item["community"]),
                    "label": label_map[int(item["community"])],
                    "raw_label": item.get("label"),
                    "birth_year": item.get("birth_year"),
                    "size": item.get("size"),
                }
                for item in top_communities
            ],
            handle,
            indent=2,
        )
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
