# Crystal communities

Scientific software and figures for **Structural memory links experimental discovery and computational novelty**, by Dan Nguyen, Karen Cao, Brian Chu, Nick
Lemoff, Paul Kienzle and William Ratcliff II (2026).

The software maps crystal structures into experimental structural neighborhoods,
replays those neighborhoods through publication time, and compares external
materials collections through structural and formula precedent. This repository
contains the analysis code, numerical tests, figures and dashboard. Manuscript
source files are not part of this repository.

## Representations

| Representation | Information represented | Encoder |
|---|---|---|
| **CrystalWeave** | Chemistry and coordination, three weighted neighbor-aggregation rounds, pooled site descriptors and cell geometry; 213 components | `scripts/icsd_densify_worker.py` |
| **Magpie-22** | Propagated structural ablation using 22 Magpie elemental properties; 4,491 components | `scripts/icsd_ablation_paper_text_worker.py` |
| **Graphlet (CrystalNN)** | Normalized local property and geometry distributions using geometric CrystalNN neighbors | `experiments/graphlet_compare/graphlet_features.py` |
| **Graphlet (VoronoiNN)** | The same local-histogram adaptation using radius-screened VoronoiNN neighbors | `experiments/graphlet_compare/graphlet_features.py` |
| **AMD-100** | 100 average minimum distances describing species-independent periodic geometry | `experiments/amd_compare/amd_features.py` |

CrystalWeave successfully represents **167,392 ICSD entries**. Its filtered
Louvain partition contains **2,939 communities** and 154,025 assigned entries;
13,367 entries remain outliers. The two Graphlet variants are our 64-channel
adaptation and are named separately throughout the analysis. Magpie-22 denotes
this propagated representation, rather than a composition-only Magpie baseline.

For each representation, a query is assigned to its nearest community centroid
and compared with that community's 95th-percentile member distance. Formula
analyses separately compare element sets, scale-invariant compositions,
anonymous stoichiometries and nearest-ICSD Element Mover's Distance (ElMD).

## Data and release status

The current code uses the repaired feature version
`crystal-features-v2-geometric-crystalnn`. **The corresponding revised data
release is not yet published.** Its reserved version-specific DOI is
[10.5281/zenodo.22700590](https://doi.org/10.5281/zenodo.22700590); it will become
available when the draft is published. The
[Zenodo concept record](https://doi.org/10.5281/zenodo.20046302) currently resolves
to an earlier published dataset. Those older features, partitions and fitted
states cannot reproduce this revision and must not be mixed with the current
encoders.

Reproduction requires the matching derived-analysis archive and, for operations
that read them, the separate feature-matrix bundles. Archives preserve their
repository-relative paths under `notes/`; the numerical data are kept outside
Git. Every matrix must retain its ordered row identifiers and provenance. Raw
ICSD CIFs and the licensed source index are excluded; recomputing descriptors
from those inputs requires independent ICSD access.

## Setup

```bash
git clone https://github.com/scattering/crystal-communities-paper.git
cd crystal-communities-paper
conda env create -f environment.yml
conda activate crystal-communities
python -m pip install -r requirements-analysis-extra.txt
```

`environment-recorded.json` records analysis dependency versions. Individual
analysis manifests specify the environment and inputs for each run.

With the matching data installed, regenerate Figures 3 and 4 with:

```bash
python scripts/make_fig_external_precedent_revised.py
```

See [reproduction instructions](docs/HOW_TO_REPRODUCE.md), the
[data dictionary](docs/SCHEMA.md), and [new-structure scoring](docs/HOW_TO_EXTEND.md)
for the current entry points, frozen states and required files. Some older
scripts retain historical default paths; the reproduction guide identifies the
current producers and explicit inputs.

## What the comparisons measure

Temporal neighborhood densification persists across the five representations.
External-cohort basin occupancy depends on the descriptor and reference map:
GNoME combines rare exact ICSD formula precedent with familiar local Graphlet
and periodic-distance patterns. Its lower CrystalWeave occupancy remains a
feature-specific contrast, rather than a universal ranking of material sources.
The A-Lab analysis compares pre-experiment target structures with synthesis
outcomes, including both Graphlet maps and AMD as controls. These analyses
measure structural precedent and campaign-specific outcome associations;
they do not turn a basin label into a probability of synthesis success.

## Dashboard

`dashboard/dash_app.py` explores the saved CrystalWeave map and scores uploaded
CIFs against the same frozen basis. It requires a validated artifact manifest;
without one the app displays which data are missing and disables scoring.
See [dashboard setup](dashboard/README.md). Updating this checkout does not
update the separately hosted service at [crystalcommunities.org](https://crystalcommunities.org/).

## Repository layout

- `scripts/`: encoders, projection, regeneration and figure producers.
- `experiments/`: Graphlet, AMD and other representation comparisons.
- `tests/`: numerical and provenance checks; some require matching data.
- `notes/`: analysis-specific source modules at their original import paths;
  derived datasets are supplied separately.
- `figures/`: current figure exports, including `icsd_densification/` outputs.
- `dashboard/`: Dash application and version-bound scoring backend.
- `docs/`: setup, artifact schema and reuse instructions.

## Citation and license

Nguyen, D., Cao, K., Chu, B., Lemoff, N., Kienzle, P. & Ratcliff II, W.
*Structural memory links experimental discovery and computational novelty.*
Manuscript under review (2026). See [CITATION.cff](CITATION.cff) for metadata.
When citing a dataset, use the specific released version that you analyzed.

Code is provided under the terms in [LICENSE](LICENSE), with third-party
attribution in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Derived-data
archives carry their own license and version metadata.
