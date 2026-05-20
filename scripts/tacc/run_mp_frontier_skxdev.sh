#!/bin/bash
#SBATCH -J mp-frontier
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -o /work2/09870/williamratcliff/stampede3/icsd_graph_runs/mp_frontier_20260427/slurm-%j.out
#SBATCH -e /work2/09870/williamratcliff/stampede3/icsd_graph_runs/mp_frontier_20260427/slurm-%j.err

set -euo pipefail

WORKROOT=/work2/09870/williamratcliff/stampede3
REPO=$WORKROOT/crystal-communities
OUT=$WORKROOT/icsd_graph_runs/mp_frontier_20260427
CACHE=$WORKROOT/reference_data/mp_theoretical_candidates_20260427.jsonl

FEATURES=$WORKROOT/icsd_densify_runs/full_matminer_ops_spr_20260418/features.npy
ASSIGN=$WORKROOT/icsd_graph_runs/full_matminer_ops_spr_20260419_labels3/community_assignments.csv

mkdir -p "$OUT"

source $WORKROOT/conda/etc/profile.d/conda.sh
conda activate $WORKROOT/conda_envs/icsd_densify

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"

python3 "$REPO/scripts/analyze_mp_frontier.py" \
  --icsd-features "$FEATURES" \
  --community-assignments "$ASSIGN" \
  --candidate-cache "$CACHE" \
  --candidate-pool-size 20000 \
  --max-e-above-hull 0.2 \
  --sample-size 5000 \
  --seed 42 \
  --max-sites 256 \
  --wl-iters 3 \
  --local-mode matminer_ops \
  --n-jobs 16 \
  --output-dir "$OUT"

echo "wrote outputs to $OUT"
