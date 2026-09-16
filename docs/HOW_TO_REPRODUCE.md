# Reproduce the analyses and figures

The current encoders and figures use the repaired CrystalWeave reference and
its matched comparison maps. The required revised numerical dataset is
published as version 3 at
[10.5281/zenodo.22700590](https://doi.org/10.5281/zenodo.22700590), paired with
source tag
[`npj-revision-2026-09`](https://github.com/scattering/crystal-communities-paper/tree/npj-revision-2026-09)
at commit `0f4ae1ae788800d93d1beacf2d8529e7bcca641b`. Use this dataset version;
earlier releases contain different features and fitted states and do not
supply the inputs needed below.

## Environment and data placement

```bash
conda env create -f environment.yml
conda activate crystal-communities
python -m pip install -r requirements-analysis-extra.txt
```

Extract the matching derived-analysis archive into the repository root,
preserving its paths. Separate feature bundles carry their own row identifiers
and provenance. Check the release manifest and SHA-256 checksums before use.
`environment-recorded.json` describes the recorded scientific environment;
per-analysis manifests take precedence for a particular run.

Every distributed feature matrix, projection, partition and basin state carries
the feature version `crystal-features-v2-geometric-crystalnn`; the loaders
refuse artifacts from any other feature version.

Throughout this guide, `D` denotes
`notes/feature_repair_2026_09/downstream`. This name is an archive path, not an
environment variable. See [SCHEMA.md](SCHEMA.md) for frozen-state locations.
Raw ICSD structures and the licensed index are needed only when rebuilding
analyses that consume them; they are not distributed in the archive.

## Current main figures

From the repository root, after the matching archive is installed:

```bash
python scripts/make_fig_temporal_cliff_revised.py \
  --exclusive-csv notes/feature_repair_2026_09/downstream/temporal/production/fig1_exclusive_by_decade.csv \
  --output figures/icsd_densification/temporal_cliff_stacked_area_repaired.png

python scripts/make_repaired_renaissance_figures.py \
  --community-evidence notes/feature_repair_2026_09/downstream/community_evidence \
  --output-dir figures/icsd_densification

CRYSTAL_COMMUNITIES_FIGURE_DIR=figures/icsd_densification \
  python scripts/make_fig_external_precedent_revised.py
```

The first command renders Figure 1 from mutually exclusive temporal events.
The second renders the community-growth figures from checked memberships and
historical descriptions. The third renders Figures 3 and 4 from the saved
external projections, independently trained historical CrystalWeave maps,
repaired five-representation comparison, formula layers and A-Lab outcomes.
`CRYSTAL_COMMUNITIES_DOWNSTREAM` can override the third command's data root.

Supplementary figures use the current
`scripts/regenerate_repaired_*` producers and analysis-specific modules under
`D`. Their manifests record the input files and hashes. Older `make_fig_*` and
`analyze_*` helpers may default to pre-repair paths; use explicit current inputs
when invoking those helpers. Matching numerical inputs and dependency versions
are required; image bytes can also depend on font and rendering versions.

The supplementary display items form one sequence: Figures 1–12 and Tables
1–60. Figures 1–5 and Tables 1–3 formerly carried Extended Data labels; some
historical filenames retain those names. Candidate quadrants A–D are Figures
9–12, paired with Tables 47–50. `scripts/select_repaired_candidates.py` selects
the records and `scripts/render_repaired_candidates.py` extracts and renders
their public-source structures using `scripts/render_representative_candidates.py`.
The extraction manifest records source entries and CIF hashes. Rendering
preserves relaxed atomic coordinates and repeats only sites on numerical cell
boundaries; `tests/test_candidate_render_geometry.py` checks this behavior.

## Recompute projections and comparisons

- `scripts/prepare_repaired_projection_basis.py` reconstructs the CrystalWeave
  scaler/PCA from matching features and verifies it against saved coordinates
  before writing a basis.
- `scripts/regenerate_external_projection.py` re-encodes the frozen external
  cohort identifiers from local source files and scores them against the saved
  CrystalWeave basis. It checks encoder version, settings and source hashes.
- `scripts/analyze_cutoff_trained_retrospective.py` fits historical CrystalWeave
  maps independently at each cutoff.
- `scripts/external_representation_sensitivity.py` and the analysis-specific
  producers under `D` implement the Magpie-22 and Graphlet comparisons.
- `D/representations/amd-basin-repair-20260910/` contains the current AMD basin
  repair and comparison producers. Use its outputs instead of the earlier
  AMD external-map summaries.
- `D/alab_representation_controls_20260910/` contains the Graphlet and AMD
  target encodings, outcome comparison and exact-statistics verification.

Use each command's `--help` and its saved manifest for explicit inputs. The
four large external cohorts each begin with 5,000 frozen attempts; MatterGen
uses 386. Success counts differ between encoders. Cross-representation results
join successful identifiers, never matrix positions.

## Auxiliary chemistry and disorder analyses

The three auxiliary workflows have portable entry points:

| Analysis | Entry point and required inputs |
|---|---|
| Nearest-ICSD ElMD | `scripts/compute_nearest_icsd_elmd.py --icsd-index INDEX --cohort-root COHORTS --output-dir OUTPUT`; the complete GNoME release uses `scripts/compute_full_gnome_elmd.py --icsd-index INDEX --gnome-summary SUMMARY --output-dir OUTPUT` |
| Disorder-aware matching | `scripts/run_disorder_structure_match.py --manifest PAIRS --cif-root CIFS --mattergen-source MATTERGEN --output-dir OUTPUT` |
| ICSD site-occupancy flags | `scripts/extract_icsd_partial_occupancy.py ARCHIVE OUTPUT`; merge scan/rescan tables with `scripts/merge_icsd_partial_occupancy.py INPUTS --output OUTPUT` |

Replace the uppercase placeholders with your paths. The cohort root is
`D/external`; it supplies the four `attempted_cohort.csv` tables and the
MatterGen frontier-record table. ElMD uses the installed ElMD 0.5.15
modified-Pettifor lookup by default. `scripts/validate_elmd_library.py` checks
the distance kernel against that library; `scripts/verify_recovered_elmd.py`
compares regenerated cohort outputs and an indexed-search sample with the
saved results.

The matcher uses MatterGen source at commit
`92423660a8bd70e83679086e88f88596d484dc16`. Its CIF directory contains `gnome/`
and `icsd/` subdirectories. `scripts/prepare_disorder_match_manifest.py` can
extract the eight input columns from a saved pair table, excluding all saved
outcomes. The test is restricted to declared exact-formula candidate pairs;
it does not estimate structure novelty across the full GNoME release.

The ICSD index and CIF archive require separately licensed access. The
occupancy scanner asks for an encrypted archive's password interactively;
`--unencrypted` is available for unencrypted input. Its flag tests whether a
parsed numeric site occupancy differs from one by more than `1e-8`, including
occupancies above one. These auxiliary results are separate from the
structural basin test.

## Verify and reuse

```bash
python -m unittest discover -s tests
```

Tests that need the matching numerical archive are skipped when it is absent.
The data-free smoke check is `python scripts/smoke_reproduce.py`. Keep the
feature version, encoder hashes, matrix row identifiers, PCA state, partition
and radius calibration together when reusing a map.

See [HOW_TO_EXTEND.md](HOW_TO_EXTEND.md) to score new CIFs without refitting the
reference, and [dashboard/README.md](../dashboard/README.md) to prepare a local
interactive explorer.
