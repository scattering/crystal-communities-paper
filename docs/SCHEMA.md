# Data dictionary — `crystal-communities` Zenodo bundle

This document describes the columns of every CSV and the shape of every
JSON in the Zenodo deposit accompanying the manuscript "Computational
materials proposals depart from the structural memory of experimental
discovery." A reviewer should be able to understand the structure of
each artifact without reading the producer code.

The bundle is roughly 330 MB. Heavy artifacts (most importantly
`features.npy`, the 167.5K × 213-dimensional matminer feature matrix
that drives the frozen ICSD reference frame) are hosted on Zenodo
rather than GitHub.

---

## Top-level layout

```
zenodo/
├── features.npy                                   # 280 MB — frozen ICSD matminer features
├── icsd_community_assignments/
│   └── community_assignments_labels3.csv          # primary community labels
├── community_layout.csv                            # graph-layout coordinates
├── canonical_family_names_labels3.csv              # curated family names per community
├── community_families_inferred.csv                 # LLM-inferred family names per community
├── external_frontier_runs/
│   ├── gnome_frontier_20260419/
│   │   ├── gnome_frontier_records.csv
│   │   ├── gnome_frontier_summary.json
│   │   └── gnome_frontier_failures.json
│   ├── mattergen_frontier_20260419/...
│   ├── mp_frontier_20260427/...
│   ├── jarvis_frontier_20260427/...
│   └── alexandria_frontier_20260427/...
├── composition_matched_ai_records.csv             # 12.3 MB
├── composition_matched_ai_summary.json
├── formula_synth_prior_summary.json
├── formula_synth_prior_table.md
├── formula_graph_shared.csv
├── formula_graph_summary.json
├── icsd_first_report_formulas/
│   ├── first_report_formulas.csv
│   ├── split_1980/{first_report_formulas.csv, post_cutoff_accessibility_records.csv}
│   ├── split_1990/...
│   ├── split_2000/...
│   └── split_2010/...
├── temporal_null_runs/
│   └── temporal_null_summary.json
├── renaissance_survey_summary.json
├── renaissance_survey_top.json
├── renaissance_communities_summary.json
├── renaissance_extra_summary.json
└── functional_review/
    ├── functional_community_representatives_top20.csv
    ├── functional_community_review_top50_labeled.csv
    └── external_functional_review_top25_labeled.csv
```

---

## `features.npy`

Numpy save of an `(N=167500, D=213)` float32 array. Each row is one
ICSD entry's matminer descriptor vector (213-dimensional matminer-ops
preset) augmented with three rounds of message passing on a Voronoi
neighbor graph. Row order matches the row order of every CSV that
carries an `icsd_id` column (i.e. row $i$ of `features.npy`
corresponds to the $i$-th ICSD entry in the canonical assignment
table). Columns are not individually labeled; they are intended as a
black-box feature matrix.

This is the input to every frontier-projection script. A reviewer
who wants to regenerate any figure must download `features.npy`
first.

---

## `community_assignments_labels3.csv`

The canonical 167,500-row table of community labels. One row per ICSD
entry.

| column      | type | description                                                                                          |
| ----------- | ---- | ---------------------------------------------------------------------------------------------------- |
| `icsd_id`   | int  | ICSD collection code.                                                                                |
| `year`      | int  | Year of first publication for that entry (from ICSD's "year of report" field).                       |
| `community` | int  | Louvain community label. `-1` denotes HDBSCAN/Louvain noise (excluded from threshold computation).   |

The `labels3` suffix corresponds to the production Louvain settings
documented in Methods: `k=16` mutual-kNN graph in 32-dimensional PCA
space, resolution $\gamma=1.0$, three rounds of Voronoi message passing.

---

## `community_layout.csv`

Per-community 2D coordinates for the figure renderings.

| column      | type   | description                                  |
| ----------- | ------ | -------------------------------------------- |
| `community` | int    | Louvain community label.                     |
| `x`         | float  | Layout x-coordinate (used in graph figures). |
| `y`         | float  | Layout y-coordinate.                         |
| `size`      | int    | Number of ICSD entries in the community.     |
| `edges`     | int    | Number of within-community graph edges.      |

---

## `external_frontier_runs/<source>/<source>_frontier_records.csv`

The per-CIF projection of one external structure source onto the
frozen ICSD reference frame. **Five files**, one per source. Each
producer is `scripts/analyze_<source>_frontier.py` (or the generic
`scripts/analyze_external_cif_zip_frontier.py` for MatterGen).

Columns shared across all five sources:

| column                          | type            | description                                                                                                          |
| ------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------- |
| `material_id`                   | str             | Source-native identifier (`mp-12345`, `JVASP-12345`, `agm005737469`, `mattergen-public/cif00012`, etc.).             |
| `reduced_formula`               | str             | pymatgen reduced formula. Used for the synthesizability-prior formula-match check.                                   |
| `assigned_community`            | int             | Nearest-centroid Louvain community in the frozen ICSD basis.                                                         |
| `nearest_centroid_distance`     | float (eV-free) | Euclidean distance to the assigned community's centroid in the 32-D PCA basis.                                       |
| `outlier_like`                  | bool            | True iff `nearest_centroid_distance` exceeds the 95th-percentile within-community threshold (the "frontier" classification). |
| `pca1`                          | float           | First PCA component of the frozen-ICSD basis, projected for this CIF.                                                |
| `pca2`                          | float           | Second PCA component.                                                                                                |

Source-specific columns:

| source     | extra columns                                                                |
| ---------- | ---------------------------------------------------------------------------- |
| GNoME      | `decomposition_energy_per_atom`, `nsites`                                    |
| MatterGen  | `zip_member`, `family`                                                       |
| MP         | `energy_above_hull`, `spg`                                                   |
| JARVIS     | `ehull`, `formation_energy`, `spg`                                           |
| Alexandria | `energy_above_hull` (renamed from upstream `e_above_hull`), `spg`            |

### B2 disclaimer — hull-distance columns are **not** numerically interchangeable

The four "hull distance" quantities listed above carry the upstream
provider's column name and definition; they are conceptually similar
but differ in subtle ways. **Cross-source energy comparisons should
be made with care.**

- **MP** `energy_above_hull` is the standard convex-hull distance from
  Materials Project's own DFT-PBE convex hull, in eV/atom.
- **JARVIS** `ehull` is the same convention but computed against the
  JARVIS-DFT-3D hull, which uses the OptB88vdW exchange-correlation
  functional rather than PBE. Numerical values can differ from MP's
  by tens of meV/atom on the same composition.
- **Alexandria** `energy_above_hull` (renamed by our analyzer from
  Alexandria's upstream `e_above_hull`) is computed against
  Alexandria's own MP+Alexandria-merged hull at the PBE level,
  release tag 2025-07-02.
- **GNoME** `decomposition_energy_per_atom` is **not** an
  energy-above-hull. It is the energy of the most-favorable
  decomposition reaction, computed by Google DeepMind's
  self-consistent VASP-PBE pipeline against a custom reference set.
  For an on-hull entry, `energy_above_hull` would be 0; the
  GNoME `decomposition_energy_per_atom` is the negative formation
  energy of the most-favorable competing reaction, with a different
  sign convention.

The manuscript figures **do not** consume these energy columns;
all main-text claims rest on the shared structural columns
(`assigned_community`, `nearest_centroid_distance`, `outlier_like`,
`pca1`, `pca2`, `reduced_formula`). The energy columns are
preserved for downstream users who may want to stratify on hull
distance, but they are not directly comparable across sources.

---

## `external_frontier_runs/<source>/<source>_frontier_summary.json`

Aggregate statistics per source.

```json
{
  "source": "GNoME",
  "n_records": 5000,
  "n_in_basin": 3784,
  "n_frontier": 1216,
  "frontier_rate": 0.2432,
  "frontier_rate_ci95": [0.2317, 0.2549],
  "threshold_p95": 0.4521,
  "thresholds_per_community": {"0": 0.32, "1": 0.41, ...}
}
```

---

## `composition_matched_ai_records.csv`

Per-record table for the composition-matched AI vs held-out ICSD
control. ~12.3 MB.

| column           | type   | description                                                                                                  |
| ---------------- | ------ | ------------------------------------------------------------------------------------------------------------ |
| `cutoff`         | int    | Held-out training cutoff year (1990, 2000, 2010).                                                            |
| `series`         | str    | One of `ICSD`, `GNoME`, `MatterGen`, `MP`, `JARVIS`, `Alexandria`.                                           |
| `id`             | str    | Source-native identifier; for ICSD this is the `cif_id`.                                                     |
| `formula`        | str    | Reduced formula.                                                                                             |
| `stratum_coarse` | str    | Coarse composition stratum: `(dominant anion class, n_unique_elements)`. Used by the manuscript main-text bar. |
| `stratum_anon`   | str    | pymatgen anonymized stoichiometry pattern (e.g. `"ABC3"`). Used by the manuscript "anonymized" bar.          |
| `in_basin`       | int    | 0 or 1 — whether the record was classified in-basin against the cutoff's training-only thresholds.           |

## `composition_matched_ai_summary.json`

Aggregate Wilson-CI statistics consumed by `make_fig_composition_matched_ai.py`.
Schema:

```json
{
  "cutoffs": [
    {
      "cutoff": 1990,
      "matchings": {
        "coarse":     { "icsd_unmatched": {...}, "by_source": { "GNoME": {...}, ... } },
        "anonymized": { "icsd_unmatched": {...}, "by_source": { "GNoME": {...}, ... } }
      }
    },
    ...
  ]
}
```

Each `by_source[<NAME>]` block carries:
- `unmatched`: rates / Wilson CIs computed without composition matching
  (the source's full sample);
- `matched`: rates / CIs after restricting to composition strata that
  also appear in held-out ICSD (this is what the figure's colored
  bar shows);
- `icsd_matched_to_source`: rates / CIs of held-out ICSD restricted
  to the same stratum intersection (the figure's grey bar);
- `n_strata_common`: number of composition strata present in both
  the source and ICSD (printed above each cutoff in the figure).

---

## `formula_synth_prior_summary.json`

The 2×2 quadrant counts driving Figure 4. Schema:

```json
{
  "definition": "in-basin × post-1980-ICSD-formula-match crosstab",
  "icsd_reference_set": "post-1980 ICSD reduced formulas",
  "sources": {
    "GNoME":      { "n": 5000, "quadrants": { "in_basin_and_formula_match": 0,
                                              "in_basin_and_no_formula_match": 3784,
                                              "frontier_and_formula_match":  0,
                                              "frontier_and_no_formula_match": 1216 } },
    "MatterGen":  { ... },
    "MP":         { ... },
    "JARVIS":     { ... },
    "Alexandria": { ... }
  }
}
```

---

## `icsd_first_report_formulas/`

Per-cutoff held-out splits of the ICSD with first-report year metadata.
Used by the renaissance-survey and synthesizability-prior pipelines.

`first_report_formulas.csv` (full, no cutoff applied):

| column                        | type    | description                                                                       |
| ----------------------------- | ------- | --------------------------------------------------------------------------------- |
| `cif_id`                      | int     | ICSD collection code.                                                             |
| `reduced_formula`             | str     | pymatgen reduced formula.                                                         |
| `year`                        | int     | First-report year.                                                                |
| `assigned_community`          | int     | Production community label.                                                       |
| `nearest_centroid_distance`   | float   | Same metric as in `*_frontier_records.csv`.                                       |
| `is_in_basin`                 | int 0/1 | 1 if ≤ 95th-percentile within-community threshold; 0 otherwise.                   |
| `A_i`                         | float   | Structural-accessibility score 𝒜ᵢ; lower = closer to historical experimental basin. |

`split_<YEAR>/` subdirectories carry the held-out partition for the
training cutoff year `<YEAR>`. Within each subdirectory:

- `first_report_formulas.csv` — same schema as above, restricted to
  entries with `year > cutoff`.
- `post_cutoff_accessibility_records.csv` — same schema with two
  additional columns (`threshold` and `assigned_threshold`) recording
  the cutoff-specific 95th-percentile threshold used for the
  `is_in_basin` classification.

---

## `temporal_null_runs/temporal_null_summary.json`

Year-shuffle null model for Extended Data Fig 1. Per-decade observed
new-community-formation rate vs the 5/50/95-percentile null bands
under year permutation with the partition fixed.

```json
{
  "n_shuffles": 200,
  "seed": 42,
  "decades": [
    { "decade": "1930s", "observed_birth_ratio": 0.402,
      "null_p05": 0.061, "null_p50": 0.102, "null_p95": 0.149 },
    ...
  ]
}
```

---

## `renaissance_*_summary*.json`

The renaissance-survey outputs consumed by Fig 2 and the SI top-20
table.

`renaissance_survey_top.json` carries the top-20 communities by
step-change score, each with:

```json
{
  "community": 6425,
  "n_members": 378,
  "best_event_year": 1987,
  "fold": 17.3,
  "rate_pre": 0.18,
  "rate_post": 3.12,
  "score": 12.4,
  "representative_formula_examples": [...]
}
```

The other JSONs (`renaissance_communities_summary.json`,
`renaissance_extra_summary.json`) carry the targeted-probe results
for cuprates / CMR manganites / Fe-pnictides / TMDs / hybrid
perovskites / community 2349 documented in SI §S8.

---

## Functional-review tables

Curated tables of community representatives used to construct
Extended Data Table 1 and the SI §S4 functional-class stratification.

`functional_community_representatives_top20.csv` — one row per top-20
community by member count:

| column                              | type | description                                            |
| ----------------------------------- | ---- | ------------------------------------------------------ |
| `community`                         | int  | Louvain label.                                         |
| `n_members`                         | int  | Member count.                                          |
| `representative_formula`            | str  | Most central member by within-community centrality.    |
| `representative_icsd_id`            | int  | ICSD code for the representative.                      |
| `inferred_family`                   | str  | LLM-inferred family name (audited; see SI §S4).        |
| `curated_family`                    | str  | Final curated family name used in the manuscript.      |

The `*_review_top*` tables are auditing aids: each row is a community
with five representative members listed for visual inspection. These
are the working tables behind the SI §S4 manual labeling and have
no direct figure consumer.

---

## Provenance and reproducibility

Every artifact in this bundle was produced by a script in this
repository under one of:

- `scripts/icsd_densify_worker.py` — the production featurization (the
  one that emits `features.npy`); cluster-resident.
- `scripts/icsd_graph_community_postprocess.py` — community detection
  + post-processing.
- `scripts/analyze_<source>_frontier.py` — per-source frontier
  projection.
- `scripts/analyze_external_cif_zip_frontier.py` — generic CIF-zip
  projector (used for MatterGen).
- `scripts/analyze_composition_matched_ai.py`,
  `scripts/analyze_temporal_null.py`,
  `scripts/analyze_renaissance_survey.py`,
  `scripts/analyze_synthesis_retrodiction.py`,
  `scripts/analyze_structural_accessibility.py`.

The TACC SLURM job wrappers under `scripts/tacc/` document the exact
command-line invocations, partitions, and walltimes used for the
manuscript production runs.
