# Interactive dashboard

A [Plotly Dash](https://dash.plotly.com/) application for exploring the
167,500-entry ICSD structural embedding interactively and for scoring
your own CIF against the frozen ICSD reference frame.

Two views:

- **Overview** — the community map (structural basins), per-community
  members, year distributions, curated/inferred family labels, and the
  five external-source overlays used in the manuscript.
- **Score a CIF** — upload a CIF; the app computes its structural
  embedding with the same `build_structure_embedding(...)` used in the
  manuscript, projects it into the frozen ICSD map, and returns the
  nearest structural basin, centroid distance, the community's
  95th-percentile threshold, the in-basin vs frontier classification,
  the structural-accessibility score 𝒜ᵢ, and a 2-D placement on a
  sampled historical background.

`index.html` is a static landing page (figure hub) that can be served
alongside or independently of the Dash app.

## Requirements

- The conda environment from the repository root `environment.yml`
  (`conda env create -f environment.yml && conda activate
  crystal-communities`). **CIF upload-scoring requires `matminer`**
  (pinned in `environment.yml`); without it the structure embedding
  silently falls back to the wrong dimensionality and scoring fails
  with a feature-count mismatch.
- The Zenodo data bundle (see the repository `README.md`). The app
  loads the frozen-map artifacts on startup and holds them in memory.

## Running it locally

The three core artifacts are supplied through environment variables so
the app can be pointed at a Zenodo bundle unpacked anywhere. From the
repository root, with the bundle unpacked under `notes/`:

```bash
export ICSD_FEATURES_PATH="notes/features.npy"
export ICSD_COMMUNITY_ASSIGNMENTS_PATH="notes/icsd_community_assignments/community_assignments_labels3.csv"
export ICSD_NODE_EVENTS_PATH="notes/node_temporal_events.csv"   # schema: docs/SCHEMA.md

cd dashboard
python dash_app.py
```

Then open <http://localhost:8050>. If the three core paths are not set,
the app still starts but the CIF-scoring view is disabled and reports
which variables are missing.

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `ICSD_FEATURES_PATH` | yes | Frozen ICSD matminer feature matrix (`features.npy`). |
| `ICSD_COMMUNITY_ASSIGNMENTS_PATH` | yes | Canonical community-label table. |
| `ICSD_NODE_EVENTS_PATH` | yes | Per-node community-birth/attachment events table. |
| `ICSD_PROTOTYPE_LABELS_PATH` | no | AflowPrototype / CIF systematic-name labels per community. |
| `ICSD_CANONICAL_LABELS_PATH` | no | Curated family names (defaults to `notes/canonical_family_names_labels3.csv`). |
| `ICSD_REPRESENTATIVES_PATH` | no | Per-community representative exemplars (defaults to `notes/functional_community_representatives_top20.csv`). |
| `ICSD_INFERRED_FAMILIES_PATH` | no | Heuristic textbook-family names (defaults to `notes/community_families_inferred.csv`). |
| `ICSD_COMMUNITY_LAYOUT_PATH` | no | Graph-aware community layout (defaults to `notes/community_layout.csv`); falls back to a PCA scatter if absent. |
| `ICSD_CIF_DIR` | no | Directory of CIFs keyed by ICSD id; enables the 3Dmol.js centroid view in the drill-down modal. |
| `ICSD_DEMO_SAMPLE_SIZE` | no | Background sample size for the placement scatter (default `12000`). |
| `ICSD_DEMO_WL_ITERS` | no | Message-passing rounds for the upload embedding (default `3`, matching production). |
| `ICSD_DEMO_OBSERVATION_YEAR` | no | Observation-year used for accessibility scoring (default `2025`). |
| `ICSD_DEMO_RANDOM_SEED` | no | Seed for the background sub-sample (default `42`). |

The exact bundle filenames are documented in
[`../docs/SCHEMA.md`](../docs/SCHEMA.md).

## Resource notes

No GPU is required. 2–4 CPU cores and 8–16 GB RAM are sufficient; the
dominant cost is holding the frozen-map artifacts in memory rather than
recomputing them per request.
