#!/bin/bash
#SBATCH -J jarvis-frontier
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -o /work2/09870/williamratcliff/stampede3/icsd_graph_runs/jarvis_frontier_20260427/slurm-%j.out
#SBATCH -e /work2/09870/williamratcliff/stampede3/icsd_graph_runs/jarvis_frontier_20260427/slurm-%j.err

set -euo pipefail

WORKROOT=/work2/09870/williamratcliff/stampede3
REPO=$WORKROOT/crystal-communities
OUT=$WORKROOT/icsd_graph_runs/jarvis_frontier_20260427
JARVIS_JSON=$WORKROOT/reference_data/jarvis_dft/jdft_3d-12-12-2022.json

FEATURES=$WORKROOT/icsd_densify_runs/full_matminer_ops_spr_20260418/features.npy
ASSIGN=$WORKROOT/icsd_graph_runs/full_matminer_ops_spr_20260419_labels3/community_assignments.csv

mkdir -p "$OUT"

source $WORKROOT/conda/etc/profile.d/conda.sh
conda activate $WORKROOT/conda_envs/icsd_densify

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"

python3 "$REPO/scripts/analyze_jarvis_frontier.py" \
  --icsd-features "$FEATURES" \
  --community-assignments "$ASSIGN" \
  --jarvis-json "$JARVIS_JSON" \
  --min-ehull 0.05 \
  --max-ehull 0.5 \
  --sample-size 5000 \
  --seed 42 \
  --max-sites 256 \
  --wl-iters 3 \
  --local-mode matminer_ops \
  --n-jobs 16 \
  --output-dir "$OUT"

echo "wrote outputs to $OUT"
