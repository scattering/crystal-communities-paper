#!/bin/bash
#SBATCH -J k-res-proper
#SBATCH -A CDA24014
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH -o /work2/09870/williamratcliff/stampede3/icsd_graph_runs/k_resolution_proper_sweep/slurm-%j.out
#SBATCH -e /work2/09870/williamratcliff/stampede3/icsd_graph_runs/k_resolution_proper_sweep/slurm-%j.err
#
# Proper graph-partition sensitivity sweep — Tier-2 audit item 9.
#
# WHY THIS EXISTS: an earlier ad-hoc sweep used sample_assignments.csv
# (the densify-step partition, ~4,400 communities) as the production
# baseline. That is NOT the manuscript's actual partition. The
# manuscript uses community_assignments_labels3.csv (the
# post-processed Louvain partition with prototype labels and
# min_component_size=8 filtering, ~6,800 communities) produced by
# icsd_graph_community_postprocess.py.
#
# This script reproduces the manuscript pipeline at varied (k,
# resolution), then re-runs analyze_composition_matched_ai.py against
# each variant to test whether the "held-out > MatterGen > {GNoME ≈
# MP-theoretical} > JARVIS > Alexandria" ordering is robust to graph-
# partition choice.
#
# Compute estimate: ~30 min postprocess + ~5 min composition_matched
# per variant = ~35 min × 5 variants = ~3 hours wall clock. SLURM
# wall-time set to 6 hours for safety. Uses skx (production), not
# skx-dev, because the dev queue caps at 2h.

set -euo pipefail

WORKROOT=/work2/09870/williamratcliff/stampede3
REPO=$WORKROOT/crystal-communities
DENSIFY=$WORKROOT/icsd_densify_runs/full_matminer_ops_spr_20260418
PCDIR=$WORKROOT/icsd_graph_runs/synthesis_retrodiction_20260419_v3

OUTBASE=$WORKROOT/icsd_graph_runs/k_resolution_proper_sweep
mkdir -p "$OUTBASE"

source $WORKROOT/conda/etc/profile.d/conda.sh
conda activate $WORKROOT/conda_envs/icsd_densify

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Variants to sweep. Production is k=16, resolution=1.0.
VARIANTS=(
  "k8_res1p0:8:1.0"
  "k16_res0p5:16:0.5"
  "k16_res2p0:16:2.0"
  "k32_res1p0:32:1.0"
)

# Per-source frontier-record CSVs (consumed by composition_matched)
GNOME=$WORKROOT/icsd_graph_runs/gnome_frontier_20260419/gnome_frontier_records.csv
MATTERGEN=$WORKROOT/icsd_graph_runs/mattergen_frontier_20260419/mattergen-public_frontier_records.csv
MP=$WORKROOT/icsd_graph_runs/mp_frontier_20260427/mp_frontier_records.csv
JARVIS=$WORKROOT/icsd_graph_runs/jarvis_frontier_20260427/jarvis_frontier_records.csv
ALEX=$WORKROOT/icsd_graph_runs/alexandria_frontier_20260427/alexandria_frontier_records.csv

for v in "${VARIANTS[@]}"; do
  IFS=":" read -r tag K RES <<< "$v"
  OUT="$OUTBASE/$tag"
  mkdir -p "$OUT/postprocess" "$OUT/composition_matched"
  echo
  echo "=========================================================="
  echo "=== variant $tag (k=$K, resolution=$RES) ==="
  echo "=========================================================="

  # 1) Re-run icsd_graph_community_postprocess.py at this (k, res).
  #    The postprocess script reuses the densify-step features_pca.npy
  #    and writes a fresh community_assignments.csv.
  python3 "$REPO/scripts/icsd_graph_community_postprocess.py" \
    --run-dir "$DENSIFY" \
    --output-dir "$OUT/postprocess" \
    --k "$K" \
    --mutual-knn \
    --resolution "$RES" \
    --min-component-size 8 \
    --min-community-size 4

  # 2) Re-run analyze_composition_matched_ai.py against the new partition,
  #    at the production p=95 per-community threshold. The held-out post-
  #    cutoff records are reused as-is (their is_in_basin column is fixed
  #    at the production threshold; only the AI sources are re-classified
  #    against the new partition's per-community thresholds).
  python3 "$REPO/scripts/analyze_composition_matched_ai.py" \
    --features "$DENSIFY/features.npy" \
    --community-assignments "$OUT/postprocess/community_assignments.csv" \
    --sample-assignments "$OUT/postprocess/community_assignments.csv" \
    --post-cutoff-dir "$PCDIR" \
    --post-cutoff-pattern "split_{cutoff}/post_cutoff_accessibility_records.csv" \
    --ai-source GNoME "$GNOME" \
    --ai-source MatterGen "$MATTERGEN" \
    --ai-source MP "$MP" \
    --ai-source Alexandria "$ALEX" \
    --ai-source JARVIS "$JARVIS" \
    --cutoffs 1990 2000 2010 \
    --threshold-percentile 95 \
    --output-summary "$OUT/composition_matched/composition_matched_ai_summary.json" \
    --output-records "$OUT/composition_matched/composition_matched_ai_records.csv" \
    --output-figure "$OUT/composition_matched/composition_matched_ai.png"

done

echo
echo "DONE. Per-variant outputs at $OUTBASE/{$(IFS=,; echo "${VARIANTS[*]%%:*}")}/composition_matched/"
echo "Aggregate the per-cutoff per-source rates from each summary.json into"
echo "a single comparison table for the SI by running:"
echo "  python3 $REPO/scripts/aggregate_k_resolution_sweep.py $OUTBASE"
echo "(aggregator script not yet written — write when sweep completes.)"
