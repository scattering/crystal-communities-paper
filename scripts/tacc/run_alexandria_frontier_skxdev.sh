#!/bin/bash
#SBATCH -J alex-frontier
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -o /work2/09870/williamratcliff/stampede3/icsd_graph_runs/alexandria_frontier_20260427/slurm-%j.out
#SBATCH -e /work2/09870/williamratcliff/stampede3/icsd_graph_runs/alexandria_frontier_20260427/slurm-%j.err

set -euo pipefail

WORKROOT=/work2/09870/williamratcliff/stampede3
REPO=$WORKROOT/crystal-communities
OUT=$WORKROOT/icsd_graph_runs/alexandria_frontier_20260427
ALEX_CACHE=$WORKROOT/reference_data/alexandria_pbe_2025_07_02

FEATURES=$WORKROOT/icsd_densify_runs/full_matminer_ops_spr_20260418/features.npy
ASSIGN=$WORKROOT/icsd_graph_runs/full_matminer_ops_spr_20260419_labels3/community_assignments.csv

mkdir -p "$OUT" "$ALEX_CACHE"

source $WORKROOT/conda/etc/profile.d/conda.sh
conda activate $WORKROOT/conda_envs/icsd_densify

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"

# Use 3 source files spaced across the 58-file release (indices 0, 19, 38) for
# ~300K candidates; filter to off-hull predictions (e_above_hull >= 0.05) to
# skip experimentally-known structures; sample 5K from the filtered pool.
python3 "$REPO/scripts/analyze_alexandria_frontier.py" \
  --icsd-features "$FEATURES" \
  --community-assignments "$ASSIGN" \
  --alexandria-dir "$ALEX_CACHE" \
  --n-source-files 3 \
  --source-file-stride 19 \
  --min-e-above-hull 0.05 \
  --max-e-above-hull 0.5 \
  --sample-size 5000 \
  --seed 42 \
  --max-sites 256 \
  --wl-iters 3 \
  --local-mode matminer_ops \
  --n-jobs 16 \
  --output-dir "$OUT"

echo "wrote outputs to $OUT"
