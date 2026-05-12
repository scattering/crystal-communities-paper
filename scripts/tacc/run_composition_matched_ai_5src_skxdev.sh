#!/bin/bash
#SBATCH -J cmatch-5src
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -o /work2/09870/williamratcliff/stampede3/icsd_graph_runs/composition_matched_ai_5src_20260427/slurm-%j.out
#SBATCH -e /work2/09870/williamratcliff/stampede3/icsd_graph_runs/composition_matched_ai_5src_20260427/slurm-%j.err

set -euo pipefail

WORKROOT=/work2/09870/williamratcliff/stampede3
REPO=$WORKROOT/crystal-communities
OUT=$WORKROOT/icsd_graph_runs/composition_matched_ai_5src_20260427

FEATURES=$WORKROOT/icsd_densify_runs/full_matminer_ops_spr_20260418/features.npy
ASSIGN=$WORKROOT/icsd_graph_runs/full_matminer_ops_spr_20260419_labels3/community_assignments.csv
PCDIR=$WORKROOT/icsd_graph_runs/synthesis_retrodiction_20260419_v3
GNOME=$WORKROOT/icsd_graph_runs/gnome_frontier_20260419/gnome_frontier_records.csv
MATTERGEN=$WORKROOT/icsd_graph_runs/mattergen_frontier_20260419/mattergen-public_frontier_records.csv
MP=$WORKROOT/icsd_graph_runs/mp_frontier_20260427/mp_frontier_records.csv
ALEX=$WORKROOT/icsd_graph_runs/alexandria_frontier_20260427/alexandria_frontier_records.csv
JARVIS=$WORKROOT/icsd_graph_runs/jarvis_frontier_20260427/jarvis_frontier_records.csv

mkdir -p "$OUT"

source $WORKROOT/conda/etc/profile.d/conda.sh
conda activate $WORKROOT/conda_envs/icsd_densify

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python3 "$REPO/scripts/analyze_composition_matched_ai.py" \
  --features "$FEATURES" \
  --community-assignments "$ASSIGN" \
  --sample-assignments "$ASSIGN" \
  --post-cutoff-dir "$PCDIR" \
  --post-cutoff-pattern "split_{cutoff}/post_cutoff_accessibility_records.csv" \
  --ai-source GNoME "$GNOME" \
  --ai-source MatterGen "$MATTERGEN" \
  --ai-source MP "$MP" \
  --ai-source Alexandria "$ALEX" \
  --ai-source JARVIS "$JARVIS" \
  --cutoffs 1990 2000 2010 \
  --threshold-percentile 95 \
  --output-summary "$OUT/composition_matched_ai_summary.json" \
  --output-records "$OUT/composition_matched_ai_records.csv" \
  --output-figure "$OUT/composition_matched_ai.png"

echo "wrote outputs to $OUT"
