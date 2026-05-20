#!/bin/bash
#SBATCH -J mattergen-fr
#SBATCH -A CDA24014
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:20:00
#SBATCH -o /work2/09870/williamratcliff/stampede3/icsd_graph_runs/mattergen_frontier_20260419/slurm-%j.out
#SBATCH -e /work2/09870/williamratcliff/stampede3/icsd_graph_runs/mattergen_frontier_20260419/slurm-%j.err
#
# Project the MatterGen-public structure release into the frozen ICSD
# structural map and emit per-CIF frontier records compatible with
# make_fig_5source_calibration.py and analyze_formula_synth_prior.py.
#
# Producer: scripts/analyze_external_cif_zip_frontier.py (generic CIF
# zip projector; same script handles any external CIF bundle via
# --dataset-label).
#
# This wrapper is the exact command that produced the
# `mattergen-public_frontier_records.csv` consumed by every figure in
# the Nature manuscript. Recorded here so the producer is documented
# alongside the other four sources (run_{alexandria,gnome,jarvis,mp}_
# frontier_skxdev.sh).

set -euo pipefail

WORKROOT=/work2/09870/williamratcliff/stampede3
FEATURES=$WORKROOT/icsd_densify_runs/full_matminer_ops_spr_20260418/features.npy
COMMUNITIES=$WORKROOT/icsd_graph_runs/full_matminer_ops_spr_20260419_rerun/community_assignments.csv
CIF_ZIP=$WORKROOT/mattergen/data-release/cifs.zip
OUT=$WORKROOT/icsd_graph_runs/mattergen_frontier_20260419

mkdir -p "$OUT"

source $WORKROOT/conda/etc/profile.d/conda.sh
conda activate $WORKROOT/conda_envs/icsd_densify

python $WORKROOT/crystal-communities/scripts/analyze_external_cif_zip_frontier.py \
  --icsd-features         "$FEATURES" \
  --community-assignments "$COMMUNITIES" \
  --cif-zip               "$CIF_ZIP" \
  --dataset-label         MatterGen-public \
  --exclude-pattern       symmetrized \
  --n-jobs                16 \
  --output-dir            "$OUT"
