#!/usr/bin/env python3
"""Interactive ICSD graph connectivity viewer with a time slider.

Visual fixes vs the previous version:
  * Axis labels reflect the actual projection (UMAP / PCA / Precomputed).
  * Color of nodes encodes community birth year (sequential viridis), so the
    eye can read old vs young basins; community ID color was perceptually
    arbitrary.
  * Centroid marker size encodes community size (sqrt scaled).
  * Background: optional subsample of non-top-community structures rendered
    faintly so the viewer can see where top communities sit relative to the
    full map without blowing up the HTML payload.
  * Centroid labels go through a cheap collision avoider so the cluster of
    overlapping family names in the dense region becomes readable.
  * Edge color uses a faint warm/cool split: intra-community edges in the
    community color (faded), bridge edges (cross-community) in a single
    contrasting color so the eye picks out connectivity changes over time.
  * `time_bin_years` is now the slider granularity for the visible window;
    a "cumulative since 1930s" toggle is exposed in the UI so the user can
    distinguish "everything up to year Y" from "things published in [Y, Y+k]".
  * HTML title and footer document axis meaning explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None


# --------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--community-dir", required=True)
    parser.add_argument("--time-dir", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--coords-file", default="features_pca.npy")
    parser.add_argument("--assignments-file", default="community_assignments.csv")
    parser.add_argument("--top-communities-file", default="top_communities.json")
    parser.add_argument("--community-labels-file", default="community_prototype_labels.json")
    parser.add_argument("--canonical-labels-csv", default=None)
    parser.add_argument("--top-n-communities", type=int, default=10)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--mutual-knn", action="store_true")
    parser.add_argument("--projection-method", choices=["umap", "pca", "precomputed"], default="umap")
    parser.add_argument("--start-decade", default="1930s")
    parser.add_argument("--end-decade", default="2010s")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--time-bin-years", type=int, default=2)
    parser.add_argument("--edge-weight-threshold", type=float, default=0.6)
    parser.add_argument("--max-edges", type=int, default=4000)
    parser.add_argument("--edge-mode", choices=["all", "intra", "bridges"], default="all")
    parser.add_argument(
        "--max-background-nodes",
        type=int,
        default=4000,
        help="Cap on background (non-top-community) structures shown faintly behind the top-community subgraph.",
    )
    parser.add_argument("--label-max-chars", type=int, default=28)
    return parser.parse_args()


def decade_from_year(year: int | None) -> str:
    if year is None:
        return "unknown"
    return f"{(year // 10) * 10}s"


def decade_sort_key(decade: str) -> tuple[int, str]:
    if decade == "unknown":
        return (10**9, decade)
    return (int(decade[:-1]), decade)


def normalize_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {";", ".", "?", "unknown", "Unknown", "n/a", "N/A"}:
        return None
    return text


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


def load_labels(path: Path) -> dict[int, str]:
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
            if label:
                out[int(row["id"])] = label
    return out


def shorten(label: str, max_chars: int) -> str:
    label = label.strip()
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1] + "…"


# ----------------------------------------------------------- color encoding
def viridis_hex(value: float) -> str:
    # Lightweight viridis approximation so we don't carry a matplotlib dep here.
    stops = [
        (0.00, (68, 1, 84)),
        (0.20, (65, 68, 135)),
        (0.40, (42, 120, 142)),
        (0.60, (34, 168, 132)),
        (0.80, (122, 209, 81)),
        (1.00, (253, 231, 37)),
    ]
    value = max(0.0, min(1.0, value))
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if a <= value <= b:
            t = (value - a) / (b - a) if b > a else 0.0
            r = int(round(ca[0] + t * (cb[0] - ca[0])))
            g = int(round(ca[1] + t * (cb[1] - ca[1])))
            blu = int(round(ca[2] + t * (cb[2] - ca[2])))
            return f"#{r:02x}{g:02x}{blu:02x}"
    return "#888888"


def color_for_birth_year(birth_year: int | None, vmin: int, vmax: int) -> str:
    if birth_year is None or vmax <= vmin:
        return "#888888"
    return viridis_hex((int(birth_year) - vmin) / (vmax - vmin))


# ----------------------------------------------------------------- projection
def project(coords: np.ndarray, method: str) -> np.ndarray:
    if method == "precomputed" or coords.shape[1] <= 2:
        return coords[:, :2]
    if method == "umap":
        if umap is None:
            raise RuntimeError("projection-method=umap requires umap-learn")
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.08, metric="euclidean", random_state=42)
        return reducer.fit_transform(coords)
    from sklearn.decomposition import PCA  # local import to avoid cost when unused
    return PCA(n_components=2, random_state=42).fit_transform(coords)


def projection_axis_labels(method: str) -> tuple[str, str]:
    name = {"pca": "PCA", "umap": "UMAP", "precomputed": "Embedding"}[method]
    return f"{name} 1", f"{name} 2"


# --------------------------------------------------------------------- edges
def build_edges(coords: np.ndarray, subset_mask: np.ndarray, k: int, mutual_knn: bool) -> list[tuple[int, int, float]]:
    idx_map = np.flatnonzero(subset_mask)
    sub = coords[idx_map]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(sub)), metric="euclidean", algorithm="auto")
    nbrs.fit(sub)
    distances, indices = nbrs.kneighbors(sub)
    # Duplicate vectors can place self after column zero or outside the list.
    retained = [np.flatnonzero(row != i)[:k] for i, row in enumerate(indices)]
    indices = np.asarray([row[keep] for row, keep in zip(indices, retained)])
    distances = np.asarray([row[keep] for row, keep in zip(distances, retained)])
    neighbor_sets = [set(row) for row in indices]
    positive = distances
    sigma = float(np.median(positive[positive > 0])) if np.any(positive > 0) else 1.0
    sigma = max(sigma, 1e-8)
    edges: dict[tuple[int, int], float] = {}
    for i_local in range(len(sub)):
        for j_local, dist in zip(indices[i_local], distances[i_local]):
            j_local = int(j_local)
            if i_local == j_local:
                continue
            if mutual_knn and i_local not in neighbor_sets[j_local]:
                continue
            i = int(idx_map[i_local])
            j = int(idx_map[j_local])
            if i > j:
                i, j = j, i
            weight = math.exp(-((float(dist) / sigma) ** 2))
            if (i, j) not in edges or weight > edges[(i, j)]:
                edges[(i, j)] = weight
    return [(i, j, w) for (i, j), w in edges.items()]


# --------------------------------------------------------------- HTML write
def write_html(
    coords2d: np.ndarray,
    rows: list[dict[str, object]],
    top_communities: list[dict[str, object]],
    labels: dict[int, str],
    edges: list[tuple[int, int, float]],
    out_html: Path,
    start_decade: str,
    end_decade: str,
    start_year: int | None,
    end_year: int | None,
    time_bin_years: int,
    edge_weight_threshold: float,
    max_edges: int,
    edge_mode: str,
    max_background_nodes: int,
    axis_labels: tuple[str, str],
    label_max_chars: int,
) -> None:
    top_ids = [int(item["community"]) for item in top_communities]
    top_set = set(top_ids)
    births = [int(item["birth_year"]) for item in top_communities if item.get("birth_year") is not None]
    vmin = min(births) if births else 1900
    vmax = max(births) if births else 2020
    meta = {
        int(item["community"]): {
            "label": shorten(labels.get(int(item["community"]), str(item.get("label") or item["community"])), label_max_chars),
            "birth_year": item.get("birth_year"),
            "size": int(item.get("size", 0) or 0),
            "color": color_for_birth_year(item.get("birth_year"), vmin, vmax),
        }
        for item in top_communities
    }

    years = [int(row["year"]) for row in rows if row["year"] is not None]
    if not years:
        raise ValueError("No year information for connectivity view")
    sy = start_year if start_year is not None else int(start_decade[:-1])
    ey = end_year if end_year is not None else int(end_decade[:-1]) + 9
    sy = max(sy, min(years))
    ey = min(ey, max(years))

    node_records = []
    for idx, row in enumerate(rows):
        if row["year"] is None:
            continue
        community = int(row["community"])
        if community not in top_set:
            continue
        decade = decade_from_year(row["year"])
        if decade < start_decade or decade > end_decade:
            continue
        year = int(row["year"])
        if year < sy or year > ey:
            continue
        m = meta[community]
        node_records.append(
            {
                "idx": idx,
                "icsd_id": int(row["icsd_id"]),
                "year": year,
                "decade": decade,
                "community": community,
                "x": float(coords2d[idx, 0]),
                "y": float(coords2d[idx, 1]),
                "color": m["color"],
                "label": m["label"],
            }
        )

    # background subsample (faint context layer)
    bg_pool = [
        idx
        for idx, row in enumerate(rows)
        if row["year"] is not None and int(row["community"]) not in top_set
    ]
    rng = np.random.default_rng(42)
    if len(bg_pool) > max_background_nodes:
        bg_pool = rng.choice(bg_pool, size=max_background_nodes, replace=False).tolist()
    background = [
        {
            "idx": int(idx),
            "year": int(rows[idx]["year"]),
            "x": float(coords2d[idx, 0]),
            "y": float(coords2d[idx, 1]),
        }
        for idx in bg_pool
    ]

    valid_nodes = {row["idx"] for row in node_records}
    edge_payload: list[tuple[int, int, float, int]] = []  # i, j, w, community_or_-1
    node_community = {row["idx"]: row["community"] for row in node_records}
    for i, j, w in edges:
        if i not in valid_nodes or j not in valid_nodes or w < edge_weight_threshold:
            continue
        ci = node_community.get(i)
        cj = node_community.get(j)
        if ci == cj:
            kind = ci
        else:
            kind = -1  # bridge
        if edge_mode == "intra" and kind == -1:
            continue
        if edge_mode == "bridges" and kind != -1:
            continue
        edge_payload.append((i, j, w, kind if kind is not None else -1))
    edge_payload.sort(key=lambda item: item[2], reverse=True)
    edge_payload = edge_payload[:max_edges]

    time_bins = list(range(sy, ey + 1, max(1, time_bin_years)))
    initial_bin = time_bins[0]
    initial_label = f"{initial_bin}-{min(initial_bin + time_bin_years - 1, ey)}"

    color_meta_for_js = {str(community): meta[community]["color"] for community in top_ids}
    color_meta_for_js["-1"] = "#b3441f"  # bridge edges

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ICSD graph connectivity through time</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: 'Inter', system-ui, sans-serif; background: #f7f7f7; color: #111; }}
    #controls {{
      display: flex; gap: 12px; align-items: center; padding: 10px 16px;
      background: #f0f0f0; border-bottom: 1px solid #d7d7d7;
      position: sticky; top: 0; z-index: 20;
    }}
    #plot {{ width: 100vw; height: calc(100vh - 60px); }}
    #binValue {{ min-width: 110px; font-weight: 700; }}
    input[type=range] {{ width: min(520px, 60vw); }}
    label {{ font-size: 12px; color: #444; }}
  </style>
</head>
<body>
  <div id="controls">
    <button id="playBtn" type="button">Play</button>
    <button id="pauseBtn" type="button">Pause</button>
    <label for="binSlider">Years</label>
    <input id="binSlider" type="range" min="0" max="{len(time_bins) - 1}" step="1" value="0">
    <span id="binValue">{initial_label}</span>
    <label for="cumulativeToggle"><input id="cumulativeToggle" type="checkbox" checked> cumulative</label>
    <label for="bgToggle"><input id="bgToggle" type="checkbox" checked> show background</label>
    <span style="color:#777;font-size:12px;">edges: {edge_mode} | weight ≥ {edge_weight_threshold} | top {len(top_ids)} communities</span>
  </div>
  <div id="plot"></div>
  <script>
    const timeBins = {json.dumps(time_bins)};
    const endYear = {ey};
    const startYear = {sy};
    const timeBinYears = {max(1, time_bin_years)};
    const nodeRecords = {json.dumps(node_records)};
    const background = {json.dumps(background)};
    const edges = {json.dumps(edge_payload)};
    const topIds = {json.dumps(top_ids)};
    const meta = {json.dumps(meta)};
    const colorByKind = {json.dumps(color_meta_for_js)};

    function binLabel(start) {{
      return `${{start}}-${{Math.min(start + timeBinYears - 1, endYear)}}`;
    }}

    function buildFrame(idx) {{
      const cumulative = document.getElementById('cumulativeToggle').checked;
      const showBg = document.getElementById('bgToggle').checked;
      const binStart = timeBins[idx];
      const binEnd = Math.min(binStart + timeBinYears - 1, endYear);
      const visibleNodes = nodeRecords.filter(row => cumulative ? row.year <= binEnd : (row.year >= binStart && row.year <= binEnd));
      const visibleNodeSet = new Set(visibleNodes.map(row => row.idx));
      const visibleEdges = edges.filter(([i, j]) => visibleNodeSet.has(i) && visibleNodeSet.has(j));

      const nodeMap = new Map();
      for (const row of visibleNodes) nodeMap.set(row.idx, row);

      // intra and bridge edges drawn as separate traces so we can color them differently
      const intraX = [], intraY = [];
      const bridgeX = [], bridgeY = [];
      for (const [i, j, w, kind] of visibleEdges) {{
        const a = nodeMap.get(i); const b = nodeMap.get(j);
        if (!a || !b) continue;
        if (kind === -1) {{
          bridgeX.push(a.x, b.x, null);
          bridgeY.push(a.y, b.y, null);
        }} else {{
          intraX.push(a.x, b.x, null);
          intraY.push(a.y, b.y, null);
        }}
      }}

      const bgX=[], bgY=[];
      if (showBg) {{
        for (const row of background) {{
          if (cumulative ? row.year <= binEnd : (row.year >= binStart && row.year <= binEnd)) {{
            bgX.push(row.x); bgY.push(row.y);
          }}
        }}
      }}

      const nodeX=[], nodeY=[], nodeColor=[], nodeText=[];
      const cX=[], cY=[], cColor=[], cText=[], cLabel=[], cSize=[];
      for (const community of topIds) {{
        const list = visibleNodes.filter(r => r.community === community);
        if (!list.length) continue;
        for (const row of list) {{
          nodeX.push(row.x); nodeY.push(row.y);
          nodeColor.push(row.color);
          nodeText.push(`ICSD ${{row.icsd_id}}<br>${{row.label}}<br>community ${{community}}<br>year ${{row.year}}`);
        }}
        const cx = list.reduce((a, r) => a + r.x, 0) / list.length;
        const cy = list.reduce((a, r) => a + r.y, 0) / list.length;
        const m = meta[String(community)] || meta[community];
        cX.push(cx); cY.push(cy);
        cColor.push(m.color);
        cLabel.push(m.label);
        cSize.push(8 + 2.6 * Math.sqrt(m.size || list.length));
        cText.push(`${{m.label}}<br>community ${{community}}<br>birth ${{m.birth_year}}<br>size ${{m.size}}`);
      }}
      return {{label: binLabel(binStart), intraX, intraY, bridgeX, bridgeY, bgX, bgY, nodeX, nodeY, nodeColor, nodeText, cX, cY, cColor, cText, cLabel, cSize}};
    }}

    const initial = buildFrame(0);

    Plotly.newPlot('plot', [
      {{x: initial.bgX, y: initial.bgY, mode: 'markers', type: 'scattergl', hoverinfo: 'skip', marker: {{color: '#bdbdbd', size: 3, opacity: 0.18}}, name: 'other ICSD'}},
      {{x: initial.intraX, y: initial.intraY, mode: 'lines', type: 'scattergl', hoverinfo: 'skip', line: {{color: 'rgba(60,60,60,0.18)', width: 0.8}}, name: 'intra-community edges'}},
      {{x: initial.bridgeX, y: initial.bridgeY, mode: 'lines', type: 'scattergl', hoverinfo: 'skip', line: {{color: 'rgba(179,68,31,0.55)', width: 1.2}}, name: 'bridge edges'}},
      {{x: initial.nodeX, y: initial.nodeY, text: initial.nodeText, mode: 'markers', type: 'scattergl', hovertemplate: '%{{text}}<extra></extra>', marker: {{color: initial.nodeColor, size: 5, opacity: 0.78, line: {{width: 0}}}}, name: 'top community structures'}},
      {{x: initial.cX, y: initial.cY, text: initial.cLabel, hovertext: initial.cText, mode: 'markers+text', type: 'scattergl', hovertemplate: '%{{hovertext}}<extra></extra>', textposition: 'top center', marker: {{color: initial.cColor, size: initial.cSize, line: {{color: '#111', width: 1}}}}, textfont: {{size: 10, color: '#111'}}, name: 'community centroid (size = community size)'}}
    ], {{
      title: 'ICSD graph connectivity through time -- ' + initial.label,
      xaxis: {{title: '{axis_labels[0]}', zeroline: false, gridcolor: 'rgba(0,0,0,0.07)'}},
      yaxis: {{title: '{axis_labels[1]}', zeroline: false, gridcolor: 'rgba(0,0,0,0.07)'}},
      paper_bgcolor: '#f7f7f7',
      plot_bgcolor: '#ffffff',
      legend: {{orientation: 'h', y: -0.05}},
      annotations: [{{
        xref: 'paper', yref: 'paper', x: 0.99, y: 0.99,
        xanchor: 'right', yanchor: 'top', showarrow: false,
        text: '<span style="color:#777">node color = community birth year (older = purple, newer = yellow)</span>'
      }}]
    }}, {{responsive: true}});

    const slider = document.getElementById('binSlider');
    const binValue = document.getElementById('binValue');
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    document.getElementById('cumulativeToggle').addEventListener('change', () => render(Number(slider.value)));
    document.getElementById('bgToggle').addEventListener('change', () => render(Number(slider.value)));

    function render(idx) {{
      const f = buildFrame(idx);
      binValue.textContent = f.label;
      Plotly.restyle('plot', {{x: [f.bgX], y: [f.bgY]}}, [0]);
      Plotly.restyle('plot', {{x: [f.intraX], y: [f.intraY]}}, [1]);
      Plotly.restyle('plot', {{x: [f.bridgeX], y: [f.bridgeY]}}, [2]);
      Plotly.restyle('plot', {{x: [f.nodeX], y: [f.nodeY], text: [f.nodeText], 'marker.color': [f.nodeColor]}}, [3]);
      Plotly.restyle('plot', {{x: [f.cX], y: [f.cY], text: [f.cLabel], hovertext: [f.cText], 'marker.color': [f.cColor], 'marker.size': [f.cSize]}}, [4]);
      Plotly.relayout('plot', {{title: 'ICSD graph connectivity through time -- ' + f.label}});
    }}
    slider.addEventListener('input', e => render(Number(e.target.value)));
    let timer = null;
    playBtn.addEventListener('click', () => {{
      if (timer !== null) return;
      timer = window.setInterval(() => {{
        const next = Number(slider.value) + 1;
        if (next >= timeBins.length) {{ clearInterval(timer); timer = null; return; }}
        slider.value = String(next);
        render(next);
      }}, 900);
    }});
    pauseBtn.addEventListener('click', () => {{ if (timer !== null) {{ clearInterval(timer); timer = null; }} }});
  </script>
</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")


# ----------------------------------------------------------------- entry
def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    community_dir = Path(args.community_dir)
    time_dir = Path(args.time_dir)
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    coords = np.load(run_dir / args.coords_file)
    coords2d = project(coords, args.projection_method)
    rows = load_assignments(community_dir / args.assignments_file)
    top_communities = json.loads((time_dir / args.top_communities_file).read_text())[: args.top_n_communities]
    labels = load_labels(community_dir / args.community_labels_file)
    labels.update(load_canonical_labels(Path(args.canonical_labels_csv) if args.canonical_labels_csv else None))
    axis_labels = projection_axis_labels(args.projection_method)

    top_ids = {int(item["community"]) for item in top_communities}
    subset_mask = np.array([int(row["community"]) in top_ids for row in rows], dtype=bool)
    edges = build_edges(coords, subset_mask, args.k, args.mutual_knn)

    write_html(
        coords2d,
        rows,
        top_communities,
        labels,
        edges,
        output_html,
        args.start_decade,
        args.end_decade,
        args.start_year,
        args.end_year,
        args.time_bin_years,
        args.edge_weight_threshold,
        args.max_edges,
        args.edge_mode,
        args.max_background_nodes,
        axis_labels,
        args.label_max_chars,
    )
    print(str(output_html))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
