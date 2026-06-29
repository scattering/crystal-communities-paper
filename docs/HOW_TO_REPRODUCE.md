# How to reproduce the manuscript's figures

This guide assumes you have cloned `crystal-communities-paper`, created
the conda environment from `environment.yml`, and downloaded the Zenodo
data bundle. Total wall-clock for all nine submitted figures on a
laptop, given the bundle: roughly five minutes.

## Step 1 — Set up the environment

```bash
git clone git@github.com:scattering/crystal-communities-paper.git
cd crystal-communities-paper
conda env create -f environment.yml
conda activate crystal-communities
```

The pinned versions match the production TACC Stampede3 environment
that produced every figure in the manuscript. Reproduction with
mismatched versions may produce visually equivalent but byte-different
PNGs (matplotlib font hinting drift); the numerical content of the
figures is reproducible across the same `environment.yml`.

## Step 2 — Download the Zenodo bundle

```bash
# Zenodo DOI 10.5281/zenodo.20046302 (activates publicly at paper acceptance)
mkdir -p notes
zenodo_get 10.5281/zenodo.20046302  # concept DOI, always points to latest version -o notes/   # ~330 MB
```

Unpack the bundle so its contents land **directly under `notes/`**
(i.e. `notes/features.npy`, `notes/icsd_community_assignments/...`).
Every `make_fig_*.py` / `analyze_*.py` script's default input paths
are repo-anchored to `notes/`; a script run with no path flags
expects the bundle there.

This repository intentionally keeps the Git checkout small: it includes
code and submitted figures, while the derived CSV/JSON/NPY artifacts
needed to regenerate most figures live in the Zenodo bundle. Raw ICSD
CIFs are not redistributed; full re-featurization from CIFs requires an
ICSD license and the TACC-scale workflow documented in `scripts/tacc/`.

After download, the layout should be:

```
notes/
├── features.npy                                # 280 MB
├── icsd_community_assignments/
│   └── community_assignments_labels3.csv
├── external_frontier_runs/{gnome,mattergen,mp,jarvis,alexandria}_frontier_*/
│   └── *_frontier_records.csv + summary.json
├── graph_time_summary.json                     # Figure 1 input
├── composition_matched_ai_summary.json
├── composition_matched_ai_records.csv
├── formula_synth_prior_summary.json
├── icsd_first_report_formulas/split_{1980,1990,2000,2010}/
├── temporal_null_runs/temporal_null_summary.json
├── renaissance_*_summary*.json
├── renaissance_survey_top.json
├── community_layout.csv
└── per_community_thresholds_fullmap_p95.json
```

The full schema is in `docs/SCHEMA.md`.

## Step 3 — Regenerate each figure

Each `scripts/make_fig_*.py` reads one or two JSON/CSV files from the
Zenodo bundle and writes a single PNG. The PNGs in `figures/` are the
exact submission outputs.

To sanity-check a fresh checkout before downloading data, run
`python scripts/smoke_reproduce.py`. This only compiles scripts and
renders the data-free pipeline schematic.

### Main-text figures

| Figure | Producer | Reads | Output |
|---|---|---|---|
| 1. Temporal cliff | `make_fig_temporal_cliff.py --time-json notes/graph_time_summary.json --output temporal_cliff_stacked_area.png` | `graph_time_summary.json` | `temporal_cliff_stacked_area.png` |
| 2. Renaissance validation | `make_fig_renaissance_validation.py` | `community_assignments_labels3.csv`, `renaissance_survey_top.json` | `fig2_renaissance_validation.png` |
| 3. Five-source calibration | `make_fig_5source_calibration.py` | `features.npy`, 5 × `*_frontier_records.csv`, `composition_matched_ai_summary.json` | `fig3_5source_calibration.png` |
| 4. Synthesizability-prior quadrant | `make_fig_synth_prior_quadrant.py` | `formula_synth_prior_summary.json` | `synth_prior_quadrant.png` |

### Extended Data figures

| Figure | Producer | Output |
|---|---|---|
| ED 1. End-to-end pipeline overview | `make_fig_pipeline_schematic.py` | `pipeline_schematic.png` |
| ED 2. Year-shuffle null | `make_fig_temporal_null.py` | `temporal_null_birth_ratio.png` |
| ED 3. Composition-matched bar chart | `make_fig_composition_matched_ai.py --summary composition_matched_ai_summary.json` | `composition_matched_ai.png` |
| ED 4. Targeted renaissance probes | `analyze_renaissance_extra.py` (writes its own PNG) | `renaissance_extra.png` |
| ED 5. Structural-accessibility boxplot | `analyze_structural_accessibility.py` (writes its own PNG) | `structural_accessibility_boxplot.png` |

### Supporting Information figures

These are produced as side effects of the analysis scripts named in
the SI:

- `formula_graph_tri_comparison.png` — `compare_tri_structural_network.py`
- `renaissance_communities.png` — `analyze_renaissance_communities.py`
- `bridge_complexity_boxplot.png` — `analyze_bridge_chemistry.py`
- `prototype_collapse_named.png` — `make_fig_prototype_collapse_named.py`
- `stepping_stone_sankey.png` — `make_fig_stepping_stone_sankey.py`
- `kde_topo_ai_overlay.png` — `make_fig_kde_topographical.py`
- `pipeline_schematic.png` — `make_fig_pipeline_schematic.py`
- `tri_degree_vs_fragmentation_entropy.png` — `analyze_tri_structural_roles.py`
- `renaissance_survey.png` — `analyze_renaissance_survey.py`
- `graphlet_stability.png` — `make_fig_graphlet_stability.py --prod
  notes/graph_time_summary.json --graphlet
  experiments/graphlet_compare/results/graphlet_graph_time_summary.json
  --output figures/icsd_densification/graphlet_stability.png`
  (SI §S9). The two input JSONs are the production temporal-replay
  summary and the graphlet-pipeline equivalent; the graphlet
  comparison machinery (mutual-kNN k=16, Louvain resolution 1.0,
  byte-identical `compute_temporal_metrics` call on graphlet-derived
  community labels) runs on TACC, requires ICSD-licensed CIFs, and
  lives in the private working repo. The aggregate JSON output is
  released on Zenodo alongside the other derived analysis artifacts.

## Step 4 — Verify

A pixel-level diff against `figures/Figure_N.png` should match
modulo antialiasing (matplotlib font / freetype version drift).
The underlying numerical content (bar heights, scatter coordinates,
quadrant percentages) reproduces byte-identically across runs of the
same `environment.yml`.

## Re-running on TACC (production scale)

The SLURM wrappers in `scripts/tacc/` document the exact partition,
node count, walltime, and arguments used for each production run:

- `run_icsd_densification_skxdev.sh` — 2.1 hr SKX-dev featurization
- `run_{gnome,mp,jarvis,alexandria,mattergen}_frontier_skxdev.sh` — per-source projection
- `run_composition_matched_ai_5src_skxdev.sh` — Figure 3c bar chart
- `run_make_fig_5source_skxdev.sh` — Figure 3 panel rendering
- `run_icsd_theoretical_audit_skxdev.sh` — SI §S1.7 CIF-header audit
- `run_k_resolution_proper_sweep.sh` — SI §S1.8 graph-partition sensitivity

You will need:
- TACC allocation (we used `CDA24014` under contract to NIST + ACCESS
  PHY250007)
- A valid ICSD license from FIZ Karlsruhe and your own copy of the
  ICSD CIF source (required only for re-featurization from raw CIFs,
  never for re-running the figure or analysis scripts against the
  Zenodo bundle)
