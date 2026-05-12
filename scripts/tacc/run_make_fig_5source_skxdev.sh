#!/bin/bash
#SBATCH -J fig-5src
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:15:00
#SBATCH -o /work2/09870/williamratcliff/stampede3/icsd_graph_runs/fig_5source_20260428/slurm-%j.out
#SBATCH -e /work2/09870/williamratcliff/stampede3/icsd_graph_runs/fig_5source_20260428/slurm-%j.err

set -euo pipefail

WORKROOT=/work2/09870/williamratcliff/stampede3
REPO=$WORKROOT/crystal-communities
OUT=$WORKROOT/icsd_graph_runs/fig_5source_20260428

FEATURES=$WORKROOT/icsd_densify_runs/full_matminer_ops_spr_20260418/features.npy
GNOME=$WORKROOT/icsd_graph_runs/gnome_frontier_20260419/gnome_frontier_records.csv
MATTERGEN=$WORKROOT/icsd_graph_runs/mattergen_frontier_20260419/mattergen-public_frontier_records.csv
MP=$WORKROOT/icsd_graph_runs/mp_frontier_20260427/mp_frontier_records.csv
JARVIS=$WORKROOT/icsd_graph_runs/jarvis_frontier_20260427/jarvis_frontier_records.csv
ALEX=$WORKROOT/icsd_graph_runs/alexandria_frontier_20260427/alexandria_frontier_records.csv
SUMMARY=$WORKROOT/icsd_graph_runs/composition_matched_ai_5src_20260427/composition_matched_ai_summary.json

mkdir -p "$OUT"

source $WORKROOT/conda/etc/profile.d/conda.sh
conda activate $WORKROOT/conda_envs/icsd_densify

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python3 "$REPO/scripts/make_fig_5source_calibration.py" \
  --features "$FEATURES" \
  --gnome "$GNOME" \
  --mattergen "$MATTERGEN" \
  --mp "$MP" \
  --jarvis "$JARVIS" \
  --alexandria "$ALEX" \
  --composition-matched-summary "$SUMMARY" \
  --output "$OUT/fig3_5source_calibration.png"

echo "wrote $OUT/fig3_5source_calibration.png"
