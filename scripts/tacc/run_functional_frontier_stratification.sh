#!/bin/bash
#SBATCH -J func-front
#SBATCH -A CDA24014
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o /work2/09870/williamratcliff/stampede3/icsd_graph_runs/functional_frontier_stratification_20260419/slurm-%j.out
#SBATCH -e /work2/09870/williamratcliff/stampede3/icsd_graph_runs/functional_frontier_stratification_20260419/slurm-%j.err

set -euo pipefail

ROOT=/work2/09870/williamratcliff/stampede3/crystal-communities
OUT=/work2/09870/williamratcliff/stampede3/icsd_graph_runs/functional_frontier_stratification_20260419

mkdir -p "$OUT"

python3 "$ROOT/scripts/analyze_functional_frontier_stratification.py" \
  --community-labels "$ROOT/notes/functional_community_label_seed_2026-04-19.csv" \
  --heldout-records /work2/09870/williamratcliff/stampede3/icsd_graph_runs/synthesis_retrodiction_20260419_v3/split_2010/post_cutoff_accessibility_records.csv \
  --gnome-records /work2/09870/williamratcliff/stampede3/icsd_graph_runs/gnome_frontier_20260419/gnome_frontier_records.csv \
  --mattergen-records /work2/09870/williamratcliff/stampede3/icsd_graph_runs/mattergen_frontier_20260419/mattergen-public_frontier_records.csv \
  --output-dir "$OUT"
