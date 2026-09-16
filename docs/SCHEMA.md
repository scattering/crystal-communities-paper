# Current artifact schema

The revised analysis archive preserves repository-relative paths. Let `D`
denote `notes/feature_repair_2026_09/downstream` below. The current numerical
release is not yet published; older Zenodo files with similar names belong to
an earlier representation and cannot substitute for these artifacts.

## CrystalWeave reference

The accepted raw matrix has shape **(167392, 213)** and dtype `float64`; its
saved PCA coordinates have shape **(167392, 32)**. The fitted map has 2,939
communities with 154,025 members; 13,367 accepted entries are unassigned
outliers. Feature version:
`crystal-features-v2-geometric-crystalnn`.

| Artifact | Contents |
|---|---|
| Feature bundle `features.npy` | Raw CrystalWeave vectors; ordered IDs and provenance accompany the matrix |
| `D/inputs/features_pca.npy` | Saved PCA coordinates in the reference row order |
| Reference `sample_assignments.csv` | Accepted `icsd_id`, publication `year` and initial cluster label |
| Reference `graph/community_assignments.csv` | Accepted `icsd_id`, `year` and final `community`; negative labels denote unassigned entries |
| Reference `time/node_temporal_events.csv` | Per-record temporal event classification tied to the same IDs, years and communities |
| `D/reference_basis/projection_basis.npz` | Fitted transform, community centers, member radii and provenance |
| `D/reference_basis/community_thresholds.json` | Community-specific p95 and median distances, membership counts and birth years |
| `D/community_evidence/` | Full-member profiles, descriptive family groups, representative records and community layout |

The archived reference directory is
`notes/feature_repair_2026_09/full_run_results/production`. It contains the three
assignment/event paths above and is separate from `D` and the feature bundles.

`projection_basis.npz` contains `scaler_mean`, `scaler_scale`, `pca_mean`,
`pca_components`, `communities`, `centroids`, `p95`, `p50`, `counts`,
`birth_years` and JSON `metadata`. Each centroid/radius row corresponds to the
same position in `communities`. Metadata binds the basis to its encoder
version, neighbor settings, source hashes and reference-file hashes.

CrystalWeave uses weighted geometric CrystalNN neighbors and occupancy-weighted
site chemistry. The 213 components comprise mean, maximum and variance pooling
of 68-component site vectors after three aggregation rounds, plus nine global
features. The code and per-run diagnostics record explicit missing-property
conventions; this is not a matrix of unlabelled composition-only descriptors.

## Other fitted maps

| Map | Frozen state under `D` |
|---|---|
| Independently fitted historical CrystalWeave | `cutoff_trained/T1990/cutoff_map.npz`, with corresponding T2000/T2010 directories |
| Magpie-22 | `external_representation/magpie/basis/transform.npz` and `basis_full.npz` / `basis_T*.npz` |
| Graphlet (CrystalNN) | `external_representation/graphlet/basis/`, including `bin_edges.json` |
| Graphlet (VoronoiNN) | `representation_mechanism_diagnostics/voronoi_external/results/basis/`, with its own bins |
| AMD-100 full map | `representations/amd-full-partition/amd_scaler.json` and `representations/amd-basin-repair-20260910/results_fixed/full_map_fixed_partition_repair/basis.npz` |
| Independently fitted historical AMD-100 | `representations/amd-basin-repair-20260910/results_cutoff/cutoff_1990/`, with corresponding 2000/2010 directories; `summary.json` records the scaler and `basis.npz` the centers/radii |

Each representation has a different successful population. Use ordered record
IDs to join matrices and tables. Magpie-22 and Graphlet historical sensitivities
retain their full-data coordinates and partition, restricting members used for
centers and radii. The historical CrystalWeave and AMD maps are independently
fitted through each cutoff. Full-map ICSD occupancy is an in-sample comparison.
The repaired AMD full map uses 667 eligible positive-radius centers and
reassigns every query after the declared size and radius filters.

## External records

Current CrystalWeave projections are in
`D/external/<source>/*_frontier_records.csv`. Successful external rows total
20,349: GNoME 5,000, MatterGen 386, MP 4,999, JARVIS 4,964 and Alexandria 5,000.

| Column | Meaning |
|---|---|
| `material_id` | Source-native record identifier; pair with source when joining cohorts |
| `reduced_formula` | Display formula; use the separate canonical formula layers for identity tests |
| `n_sites`, `n_elements` | Parsed structure size and element count |
| `assigned_community` | Nearest eligible centroid's community label |
| `nearest_centroid_distance` | Euclidean distance in the representation's scoring coordinates |
| `community_threshold_p95` | 95th percentile of that community's member-to-centroid distances |
| `in_basin` | Distance is less than or equal to the assigned community's p95 radius |
| `outlier_like` | Complement of `in_basin` |
| `outlier_like_pooled` | Separately retained historical pooled-threshold diagnostic |
| `pca1`, `pca2` | Display coordinates; scoring uses all retained components |
| `feature_row` | Position in this source's feature matrix |

Source-specific columns preserve upstream metadata. Energy columns from
different databases have different reference hulls and conventions and are
not a common energy scale.

The current five-map comparison is
`D/representations/amd-basin-repair-20260910/five_representation_common_support.csv`
with JSON/Markdown companions. Its common identifiers comprise 83,661 ICSD,
5,000 GNoME, 384 MatterGen, 4,969 MP, 4,930 JARVIS and 4,982 Alexandria records.
This shared evaluation population leaves each map's fitted state unchanged.

## Formula, chemistry and outcomes

`D/formula_layers/` contains the scale-invariant formula analyses:

- Element sets are alphabetically ordered element identities; normalized
  element ratios sum to one.
- Ordered rational compositions use pymatgen's integer formula and factor,
  greatest-common-divisor reduction and alphabetical element ordering.
- Anonymous stoichiometries retain the reduced coefficient multiset while
  replacing element identities.
- Partial-occupancy flags identify a parsed numeric site occupancy differing from one
  by more than `1e-8`, including overoccupancy. Composition comparisons use the
  same element set and a maximum
  atomic-fraction difference of `1e-6`, with separately reported sensitivities.
- `nearest_icsd_elmd_records.csv` identifies each source record and its nearest
  reference formula, ICSD ID, year and ElMD for both all-ICSD and the
  post-1980-through-2015 reference. ElMD is the published chemistry-aware
  distance, using the recorded ElMD 0.5.15 modified-Pettifor scale.
- `scale_invariant_quadrants.json` combines the formula and CrystalWeave
  structural-precedent flags; `scored_support_layers.json` summarizes formula
  layers on structurally scored populations.

Disorder-aware structure matching is a separate identity screen. Its saved
pair results must not be interpreted as whole-cohort basin membership.

`D/alab_mp_targets/output/` contains the CrystalWeave projection of 57
pre-experiment A-Lab targets. `D/alab_representation_controls_20260910/` contains
both Graphlet and AMD controls. Outcome comparisons use the same 51 definitive
made/not-obtained outcomes; offline recoveries and inconclusive outcomes remain
separately identified. These counts refer to targets, not post-experiment
product structures.

## Integrity

Keep release manifests, checksums and analysis provenance alongside the data.
Record IDs, formulas, years and matrix positions are distinct fields; never
infer row alignment across files from equal row counts. Dashboard manifests
hash every loaded artifact and check encoder compatibility, IDs, partition
counts and reference provenance before enabling scoring.
