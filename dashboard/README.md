# CrystalWeave dashboard

The Dash application explores the saved 167,392-entry CrystalWeave map and its
2,939 retained communities. It shows checked community descriptions, temporal
context, representative records and current figures, and can score an uploaded
CIF against the same frozen 32-component reference.

The app loads a version-bound artifact manifest. Encoder or data mismatches
disable scoring. No new PCA or community partition is fitted at startup.
The source in this checkout and the separately hosted service are independent;
a source update alone does not update the deployment.

## Install and prepare

Use the repository's analysis environment, then install the dashboard packages:

```bash
python -m pip install -r requirements-dashboard.txt
```

The matching revised numerical release is awaiting publication. Once those
artifacts are available locally, prepare a manifest from the repository root:

```bash
python scripts/prepare_repaired_dashboard.py \
  --downstream notes/feature_repair_2026_09/downstream \
  --production notes/feature_repair_2026_09/full_run_results/production \
  --pca notes/feature_repair_2026_09/downstream/inputs/features_pca.npy \
  --figure-dir figures/icsd_densification \
  --out-dir output/dashboard-bundle
```

`--production` must point to the matching reference directory containing
`sample_assignments.csv`, `graph/community_assignments.csv` and
`time/node_temporal_events.csv`. The archive preserves the reference directory shown above.
The downstream directory supplies the frozen basis, community evidence and
accessibility summary. Raw ICSD CIFs and the raw feature matrix are not
required for this dashboard.

Preparation writes `manifest.json` with relative artifact paths and SHA-256
hashes, plus `validation.json`. Validation checks encoder settings and hashes,
reference identities and years, PCA dimensions, partition counts, community
profiles, representatives and layout. Moving a bundle requires retaining those
relative paths or regenerating the manifest from the same verified files.

## Run locally

```bash
export ICSD_DASHBOARD_MANIFEST="$PWD/output/dashboard-bundle/manifest.json"
python dashboard/dash_app.py
```

Open <http://localhost:8050>. Without a valid manifest, the app explains the
missing-data problem and leaves scoring disabled. This command does not deploy
the application to a server.

| Environment variable | Purpose |
|---|---|
| `ICSD_DASHBOARD_MANIFEST` | Required for map exploration and CIF scoring; path to the validated manifest |
| `ICSD_DEMO_OBSERVATION_YEAR` | Historical-accessibility observation year; default 2019 |
| `ICSD_DEMO_SAMPLE_SIZE` | Number of reference points displayed in the placement background; default 12,000 |

`index.html` is the static landing page. `frozen_backend.py` exposes the
`load_bundle` and `score` API used by the application and the
[new-structure scoring guide](../docs/HOW_TO_EXTEND.md).

## Scoring fields

| Field | Meaning |
|---|---|
| `community` | Nearest community centroid in the full 32-component reference |
| `distance` | Euclidean centroid distance |
| `threshold` | Assigned community's 95th-percentile member distance |
| `frontier` | True when distance exceeds the threshold; equality is in basin |
| `d_over_tau` | Distance divided by threshold, or null for a zero threshold |
| `community_size` | Number of fitted community members |
| `community_birth_year` | Earliest dated member; negative when unknown |
| `accessibility` | Standardized descriptive coordinate combining distance, size and age |
| `xy`, `centroid_xy` | First two PCA components for display |
| `diagnostics` | Encoding and occupancy diagnostics |

A basin label describes structural precedent in CrystalWeave. It is not an
atomic-identity test or a probability of successful synthesis. The A-Lab
outcome association is evaluated separately at the campaign level. Descriptive
community labels reflect full membership and should not be read as certified
prototype assignments for every member.

The displayed background subsample changes only the visualization. The full
saved reference and all 32 components determine a query's score. For undated
communities, the accessibility calculation retains its recorded 2010 fallback
and reports that imputation explicitly.
