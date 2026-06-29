# How to project new structures into the ICSD reference frame

The most reusable piece of this work is the **frozen ICSD reference
frame**: 167,500 ICSD entries embedded into a 32-D PCA basis with a
fixed Louvain community partition and per-community 95th-percentile
within-community distance thresholds. Anyone with a new set of
crystal structures — a new generative model's output, a new DFT-
screened candidate library, a laboratory CIF collection — can
project them into the same coordinate and obtain the same
synthesizability-prior table the manuscript reports for the five
external sources.

This document walks through the two supported entry points.

## Entry point 1 — Project a zip of CIFs

`scripts/analyze_external_cif_zip_frontier.py` is a generic CIF-zip
projector. It was used for the MatterGen-public release in the
manuscript; it accepts any other CIF-zip input via the
`--dataset-label` flag.

```bash
python scripts/analyze_external_cif_zip_frontier.py \
  --icsd-features         notes/features.npy \
  --community-assignments notes/icsd_community_assignments/community_assignments_labels3.csv \
  --cif-zip               /path/to/your-source.zip \
  --dataset-label         MyNewSource \
  --exclude-pattern       symmetrized \
  --n-jobs                16 \
  --output-dir            notes/external_frontier_runs/MyNewSource_2026
```

Outputs (in `--output-dir`):

- `MyNewSource_frontier_records.csv` — per-CIF table with columns
  `material_id, zip_member, family, reduced_formula,
  assigned_community, nearest_centroid_distance, outlier_like,
  pca1, pca2`
- `MyNewSource_frontier_summary.json` — aggregate counts, thresholds,
  frontier rate
- `MyNewSource_frontier_failures.json` — per-CIF parse / embedding
  errors
- `MyNewSource_frontier_pca.png` — overlay scatter on the ICSD basin

The schema of the records CSV matches what the production five
sources emit, so any downstream script that consumes a
`*_frontier_records.csv` will work on your output unchanged.

## Entry point 2 — Reuse one of the per-source producers

If your structures come from a structured source (a database with a
specific API or schema rather than loose CIFs), the per-source
producers are easier templates than the generic CIF-zip:

- `analyze_mp_frontier.py` — Materials Project API
- `analyze_jarvis_frontier.py` — JARVIS-DFT figshare JSON
- `analyze_alexandria_frontier.py` — Alexandria bz2 JSON shards
- `analyze_gnome_frontier.py` — GNoME public-release zip

Each is ~200–300 lines, structured as: (1) source-specific loader,
(2) shared featurization and projection, (3) shared classification.
The shared parts live in `frontier_common.py` (centroids, thresholds,
parsing). Forking one of these is the path to integrating with a
non-CIF data source.

## What "in-basin" means

A projected structure is classified `outlier_like = True` (frontier)
if its nearest-centroid distance in the 32-D PCA basis exceeds the
**per-community** 95th-percentile within-community centroid distance
for its assigned community. This is the per-community threshold
convention used uniformly in Figures 3c and 4 of the manuscript and
in the composition-matched control in SI §S7.

If you need the legacy pooled-global-threshold convention (single
scalar `τ` computed as the 95th percentile of within-community
distances pooled across all communities — the convention used in
an earlier development version of this work), it can be obtained
from the same records CSV by re-classifying records with
`nearest_centroid_distance ≤ pooled_p95_threshold`. The pooled
threshold is recorded in `notes/per_community_thresholds_fullmap_p95.json`.

## Producing the synthesizability-prior quadrant for your source

Once you have `MyNewSource_frontier_records.csv`, run:

```bash
python scripts/analyze_formula_synth_prior.py \
  --icsd-formulas-dir notes/icsd_first_report_formulas \
  --source MyNewSource notes/external_frontier_runs/MyNewSource_2026/MyNewSource_frontier_records.csv \
  --per-community-thresholds notes/per_community_thresholds_fullmap_p95.json \
  --output-summary notes/MyNewSource_formula_synth_prior_summary.json \
  --output-table   notes/MyNewSource_formula_synth_prior_table.md
```

This emits the same 4-cell quadrant table the manuscript reports for
the five canonical sources, with Wilson 95% confidence intervals.

## Producing a calibrated held-out comparison against your source

To put your source against held-out ICSD at the 1990/2000/2010
cutoffs (the manuscript's Figure 3c bar chart, plus a new column for
your source), run `analyze_composition_matched_ai.py` with your
records CSV passed as `--ai-source MyNewSource <path>`.

The exact invocation is in the production SLURM wrapper
`scripts/tacc/run_composition_matched_ai_5src_skxdev.sh`; add a sixth
`--ai-source MyNewSource <path>` line to that command.

## What you'll need

To run any of the above end-to-end:

- The Zenodo bundle (specifically `features.npy`,
  `community_assignments_labels3.csv`, and
  `per_community_thresholds_fullmap_p95.json`)
- The conda environment from `environment.yml`
- The CIF/structure input for your source

You do NOT need:

- An ICSD license (the frozen-ICSD reference frame is in the Zenodo
  bundle as embeddings + integer IDs; raw ICSD CIFs are not
  redistributed)
- A TACC allocation (the projection of one external source against
  the frozen frame takes minutes-to-tens-of-minutes on a laptop,
  depending on source size and featurization cost)

## Citing this if you use it

If you publish a comparison of your new source against the ICSD
reference frame using these scripts, please cite the manuscript
(DOI to be assigned at publication) and the Zenodo data deposit
([10.5281/zenodo.20046302](https://doi.org/10.5281/zenodo.20046302)).
