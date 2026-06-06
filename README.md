# crystal-communities-paper

Companion repository for

> **Computed materials proposals depart from the structural memory
> of experimental discovery**
> Nguyen, Cao, Chu, Lemoff, Kienzle, Ratcliff (submitted, 2026)

This repository contains the code, figures, and an interactive dashboard
needed to reproduce every analysis in the manuscript and to apply the
framework to new external structure samples.

The working-repository (with development history, ablations, and
exploratory code) lives at `https://github.com/scattering/crystal-
communities`; this repo is the frozen, curated subset that produced
the submitted manuscript.

## ⚠ Zenodo data bundle required for figure reproduction

**This repository contains code + figures only. Every figure renderer
except the data-free pipeline schematic requires the derived-data
Zenodo bundle** (~330 MB). Download it before running anything beyond
the smoke test:

```bash
# Zenodo DOI 10.5281/zenodo.20046303 (activates publicly at paper acceptance)
zenodo_get 10.5281/zenodo.20046303 -o notes/
```

### What works on a fresh clone (no data download)

- `python scripts/smoke_reproduce.py` — sanity check; compiles all scripts and
  renders the data-free pipeline schematic
- `python scripts/make_fig_pipeline_schematic.py` — Extended Data Figure 1

### What needs the Zenodo bundle

- Every other `make_fig_*.py` figure renderer
- Every `analyze_*.py` script that consumes ICSD-derived inputs
- The interactive dashboard

Scripts that need files not present locally will raise
`FileNotFoundError` on the missing path; the file is in the Zenodo
bundle by the same name. The full file inventory is documented in
[`docs/SCHEMA.md`](docs/SCHEMA.md) and the step-by-step recipe in
[`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md).

### Why the data isn't in git

The largest artifact (`features.npy`, the frozen 167,500-row ICSD
feature matrix) is 280 MB; primary tables like
`community_assignments_labels3.csv` are 2.9 MB and over 10 MB total
across per-record CSVs. Keeping these on Zenodo with their own DOI
gives them citable provenance independent of code churn, and avoids
mixing licensed-data redistribution with the MIT-licensed code in
this repository. The Zenodo deposit is CC-BY-4.0.

## What's here

```
crystal-communities-paper/
├── README.md          # this file
├── LICENSE            # MIT, with NIST disclaimer; CC-BY-4.0 referenced for data
├── environment.yml    # pinned conda env (matches TACC Stampede3 production)
├── CITATION.cff       # so GitHub renders a "Cite this repo" button
│
├── figures/           # the 9 PNGs as submitted
│   ├── Figure_1.png … Figure_4.png            # main text
│   └── Extended_Data_Figure_1.png … _5.png    # extended data
│
├── scripts/           # 40 production .py + 11 TACC SLURM wrappers
│   ├── make_fig_*.py            # 10 figure renderers
│   ├── analyze_*_frontier.py    # 5 per-source projection producers
│   ├── analyze_external_cif_zip_frontier.py    # generic CIF-zip projector
│   ├── analyze_*.py             # composition-matched, formula-overlap,
│   │                            # renaissance survey, synthesis-retrodiction,
│   │                            # accessibility, TRI comparison, Kononova/A-Lab
│   │                            # validation, etc.
│   ├── icsd_densify_worker.py        # production featurization
│   ├── icsd_graph_community_postprocess.py    # community detection
│   ├── frontier_common.py            # shared helpers
│   └── tacc/                         # SLURM wrappers (provenance)
│
├── dashboard/         # Plotly Dash app for interactive exploration
│   ├── dash_app.py
│   ├── index.html
│   └── README.md
│
└── docs/
    ├── HOW_TO_REPRODUCE.md     # step-by-step from a clean machine
    ├── HOW_TO_EXTEND.md        # project your own external CIFs into the ICSD frame
    └── SCHEMA.md               # data dictionary for the Zenodo bundle
```

## Quickstart

```bash
git clone git@github.com:scattering/crystal-communities-paper.git
cd crystal-communities-paper
conda env create -f environment.yml
conda activate crystal-communities

# Pull the Zenodo bundle (~330 MB) so its contents land under ./notes/
# (e.g. ./notes/features.npy). The figure scripts default to notes/.
zenodo_get 10.5281/zenodo.20046303 -o notes/

# Regenerate any main-text figure, e.g. the synthesizability-prior quadrant:
python scripts/make_fig_synth_prior_quadrant.py
```

See `docs/HOW_TO_REPRODUCE.md` for the complete chain of analysis →
figure dependencies.

## Two intended uses

**1. Reproduce every figure.** The 10 `make_fig_*.py` scripts consume
small JSON/CSV artifacts from the Zenodo bundle and emit the figure
PNGs verbatim. Running all 10 takes under five minutes on a laptop
once the Zenodo bundle is downloaded.

**2. Extend the framework to your own structures.** The
`analyze_external_cif_zip_frontier.py` script accepts an arbitrary
ZIP of CIFs and projects them into the same frozen ICSD reference
frame used throughout the manuscript. Output is a per-CIF record
table with `assigned_community`, `nearest_centroid_distance`,
`outlier_like`, `pca1`, `pca2`. See `docs/HOW_TO_EXTEND.md` for the
recipe. Anyone with a new generative-AI structure release, a new
DFT-screened candidate set, or a laboratory CIF library can compute
the same calibrated structural-accessibility coordinate against ICSD
without re-engineering.

## Interactive dashboard

A live deployment of the dashboard is hosted at
**<https://crystalcommunities.org/>** — open it in any browser to
explore the structural community map (with curated family labels
for cuprates, Fe-pnictides 1111 / 122 / 111, lacunar spinels,
perovskites, Laves phases, and other manuscript-anchored families),
and upload your own CIF for upload-and-score evaluation against the
frozen ICSD reference frame. The CIF-scoring result includes the
manuscript's in-basin classification plus two reporting-layer
signals — a categorical structural-match tier (VERY HIGH / HIGH /
NEAR / DISTANT, based on absolute centroid distance) and a
small-community caveat annotation for cases where the
95th-percentile threshold is statistically tight; see
`dashboard/README.md` for the full result schema.

`dashboard/dash_app.py` is the Plotly Dash application that powers
the deployment. To run it locally instead of using the hosted
version:

```bash
cd dashboard
python dash_app.py
```

Then open `http://localhost:8050`. The app loads the Zenodo bundle on
startup, so the bundle must be present locally.

## Compute platform

Production featurization, frontier-projection, and renaissance-survey
runs were executed on the Texas Advanced Computing Center (TACC)
Stampede3 cluster under contract to NIST and through ACCESS allocation
PHY250007. The TACC SLURM wrappers (`scripts/tacc/run_*.sh`) document
the exact node, partition, and arguments used for each production run.
Local re-runs of any individual figure on a laptop take seconds to
minutes from the Zenodo bundle.

## Data

> Zenodo DOI: [10.5281/zenodo.20046303](https://doi.org/10.5281/zenodo.20046303) (activates publicly at paper acceptance).

Bundle contents and column-level schema are documented in
[`docs/SCHEMA.md`](docs/SCHEMA.md). **No raw ICSD CIFs are distributed
here or on Zenodo** — ICSD is licensed by FIZ Karlsruhe. Only
structure-derived embeddings, integer ICSD IDs, formulas, and distances
are released; users who want to regenerate the embedding from raw CIFs
must obtain their own ICSD license. The framework's downstream-use
entry point (`analyze_external_cif_zip_frontier.py`) requires only the
Zenodo bundle, not the ICSD license itself.

## License

Code: MIT, with NIST public-domain disclaimer for U.S.-Government-
employee contributions. See `LICENSE` for full text.

Data (Zenodo bundle): CC-BY-4.0.

## Citing this work

Until the manuscript appears in print, please cite as:

> Nguyen, D., Cao, K., Chu, B., Lemoff, N., Kienzle, P.,
> Ratcliff, W. *Computed materials proposals depart from the
> structural memory of experimental discovery.* Submitted (2026).
> Code: https://github.com/scattering/crystal-communities-paper.
> Data: Zenodo DOI [10.5281/zenodo.20046303](https://doi.org/10.5281/zenodo.20046303).

A `CITATION.cff` is provided for GitHub's "Cite this repository"
button.

## Contact

Questions about the analysis, the dashboard, or the Zenodo bundle:
`william.ratcliff@nist.gov`.
