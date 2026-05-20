# TACC production wrappers — frozen provenance, not portable scripts

These `run_*.sh` files are the **exact SLURM batch scripts used to
produce the manuscript's large derived-data artifacts** on the Texas
Advanced Computing Center (TACC) Stampede3 cluster. They are included
for provenance and methods transparency, **not** as something a reader
is expected to run.

## What they document

Each wrapper records, verbatim, the partition, node count, walltime,
module environment, and command-line arguments of one production run:

- `create_icsd_densify_env.sh` — builds the cluster-resident conda env.
- `run_icsd_densification_skxdev.sh` — the ~2.1 hr SKX featurization
  that emits `features.npy` (the 280 MB frozen ICSD feature matrix at
  the root of the entire pipeline).
- `run_{gnome,mp,jarvis,alexandria,mattergen}_frontier_skxdev.sh` —
  per-source projection of each external structure set onto the frozen
  ICSD reference frame.
- `run_composition_matched_ai_5src_skxdev.sh` — the composition-matched
  AI-vs-ICSD control behind Figure 3c.
- `run_make_fig_5source_skxdev.sh` — Figure 3 panel rendering at
  production scale.
- `run_icsd_theoretical_audit_skxdev.sh` — the SI §S1.7 CIF-header
  audit (aggregate-only output; no per-CIF data leaves TACC).
- `run_k_resolution_proper_sweep.sh` — the SI §S1.8 graph-partition
  sensitivity sweep.
- `run_functional_frontier_stratification.sh` — the SI §S4
  functional-class stratification.

## Why they are not runnable as-is

They are intentionally non-portable. Reproducing these runs requires:

- a TACC allocation (the manuscript used `CDA24014` under contract to
  NIST plus ACCESS allocation `PHY250007`);
- a valid ICSD license from FIZ Karlsruhe and your own copy of the
  ICSD CIF source (required only for re-featurization from raw CIFs,
  never for re-running the figure or analysis scripts);
- the Stampede3 `skx`/`skx-dev` partitions and module stack.

The absolute paths (`/work2/<project>/<user>/stampede3/...`) and
allocation identifiers are kept deliberately: scrubbing them would
weaken the provenance record, and none of them are credentials.

## What you actually want instead

To regenerate any **figure** on a laptop from the Zenodo data bundle
(no TACC, no ICSD license), follow
[`../../docs/HOW_TO_REPRODUCE.md`](../../docs/HOW_TO_REPRODUCE.md).
To project **your own** structures into the frozen ICSD frame, see
[`../../docs/HOW_TO_EXTEND.md`](../../docs/HOW_TO_EXTEND.md).
