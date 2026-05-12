# crystal-communities-paper

Companion repository for

> **Computational materials proposals depart from the structural memory
> of experimental discovery**
> Nguyen, Cao, Chu, Lemoff, Coomans, Kienzle, Ratcliff (submitted, 2026)

This repository contains the code, figures, and an interactive dashboard
needed to reproduce every analysis in the manuscript and to apply the
framework to new external structure samples. Derived data (the
167,500-row frozen ICSD embedding, per-source frontier records,
community assignments, summary JSONs) are deposited separately on
Zenodo under CC-BY-4.0; the Zenodo DOI is referenced below and in the
manuscript data-availability statement.

The working-repository (with development history, ablations, and
exploratory code) lives at `https://github.com/scattering/crystal-
communities`; this repo is the frozen, curated subset that produced
the submitted manuscript.

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

# Pull the Zenodo bundle (~330 MB) into ./notes/zenodo/
# (DOI to be assigned at publication; see manuscript data-availability statement)

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

`dashboard/dash_app.py` is a Plotly Dash application for exploring
the 167,500-entry ICSD embedding interactively: communities, members,
year distributions, and overlays from the five external sources.
Run locally:

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

The 280 MB ICSD feature matrix (`features.npy`) and derived data tables
are on Zenodo under CC-BY-4.0:

> Zenodo DOI: *to be assigned at publication*

The schema is documented in `docs/SCHEMA.md`. **No raw ICSD CIFs are
distributed here or on Zenodo** — ICSD is licensed by FIZ Karlsruhe.
Only structure-derived embeddings, integer ICSD IDs, formulas, and
distances are released.

## License

Code: MIT, with NIST public-domain disclaimer for U.S.-Government-
employee contributions. See `LICENSE` for full text.

Data (Zenodo bundle): CC-BY-4.0.

## Citing this work

Until the manuscript appears in print, please cite as:

> Nguyen, D., Cao, K., Chu, B., Lemoff, N., Coomans, W., Kienzle, P.,
> Ratcliff, W. *Computational materials proposals depart from the
> structural memory of experimental discovery.* Submitted (2026).
> Code: https://github.com/scattering/crystal-communities-paper.
> Data: Zenodo DOI [TBA].

A `CITATION.cff` is provided for GitHub's "Cite this repository"
button.

## Contact

Questions about the analysis, the dashboard, or the Zenodo bundle:
`william.ratcliff@nist.gov`.
