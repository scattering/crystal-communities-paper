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
  sampled historical background. The result also includes a
  `structural_match_tier` (categorical) and `small_community_caveat`
  (boolean) — see [Interpreting the score result](#interpreting-the-score-result) below.

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

## Interpreting the score result

`score_structure(...)` returns a dict with the following fields. Two of
them — `structural_match_tier` and `small_community_caveat` — are
purely reporting-layer additions; the manuscript's in-basin definition
(95th-percentile threshold on within-community centroid distances) is
unchanged.

| Field | Type | Meaning |
|---|---|---|
| `community` | int | Nearest community id (Louvain partition, `labels3` pass). |
| `community_label` | str | Resolved family name (canonical → inferred → prototype → fallback). |
| `community_size` | int | Number of training-set members in the community. |
| `distance` | float | Euclidean distance to the community centroid in the frozen PCA-32 space. |
| `threshold` | float | The community's 95th-percentile in-basin threshold (paper definition). |
| `frontier` | bool | `True` iff `distance > threshold` — the manuscript's "out of basin" criterion. **This is the paper-faithful classification.** |
| `accessibility` | float | Structural-accessibility score 𝒜ᵢ (combines centroid distance with size + age weighting). |
| `structural_match_tier` | str | Categorical, absolute-distance based: `VERY HIGH` if `distance ≤ 0.5`; `HIGH` if `distance ≤ threshold`; `NEAR` if `distance ≤ 2 × threshold`; `DISTANT` otherwise. |
| `small_community_caveat` | bool | `True` iff `community_size < 20` or `threshold < 0.1`. Surfaces the small-community statistical-tightness edge case (see below). |
| `xy` | array | Uploaded structure's 2-D placement on the sampled background. |
| `centroid_xy` | array | The nearest community's centroid in the same 2-D space. |

### Why the tier + caveat

The manuscript's in-basin classification uses a 95th-percentile
threshold per community. For very small communities (e.g., ≤20 members
that are mostly near-duplicate refinements of one parent structure),
the percentile collapses to near zero as a statistical artifact of the
small sample. A new structurally near-identical upload can then end up
formally `frontier=True` despite being centroid-close in absolute terms.

The `structural_match_tier` provides a complementary categorical signal
based on the absolute centroid distance (independent of percentile
collapse). `small_community_caveat=True` flags the cases where the
percentile threshold is statistically unreliable and the dashboard
surfaces a "near-textbook structural identity" annotation in the result
panel. The `frontier` boolean itself is unchanged from the paper's
definition; the tier is additive UX, not a redefinition.

The four tier cutoffs (0.5, τ, 2τ) are fixed defaults; the absolute
0.5 floor is calibrated so that distances at or below it correspond to
matched-structure near-identity at the level the embedding can resolve
(empirically: replicate ICSD refinements of the same crystallographic
entry cluster within < 0.1; structurally near-identical compounds
within < 0.5; the same family within ~1–3; cross-family at ~5+).

## Resource notes

No GPU is required. 2–4 CPU cores and 8–16 GB RAM are sufficient; the
dominant cost is holding the frozen-map artifacts in memory rather than
recomputing them per request.
