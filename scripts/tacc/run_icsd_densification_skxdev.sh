#!/bin/bash
#SBATCH -J icsd-densify
#SBATCH -p skx-dev
#SBATCH -A CDA24014
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /scratch/09870/williamratcliff/icsd_densify.%j.out
#SBATCH -e /scratch/09870/williamratcliff/icsd_densify.%j.err

set -euo pipefail

WORKROOT="${WORKROOT:-/work2/09870/williamratcliff/stampede3}"
REPO_DIR="${REPO_DIR:-$WORKROOT/crystal-communities}"
CONDA_ROOT="${CONDA_ROOT:-$WORKROOT/conda}"
ENV_PATH="${ENV_PATH:-$WORKROOT/conda_envs/icsd_densify}"
ICSD_ZIP="${ICSD_ZIP:-$WORKROOT/reference_data/ICSD_CIFs.zip}"
ICSD_ZIP_PASSWORD="${ICSD_ZIP_PASSWORD:-}"
ICSD_CIF_ROOT="${ICSD_CIF_ROOT:-}"
ICSD_INDEX="${ICSD_INDEX:-$WORKROOT/reference_data/ICSD_index.csv}"
OUT_DIR="${OUT_DIR:-$WORKROOT/icsd_densify_runs/sample_${SLURM_JOB_ID}}"
SAMPLE_SIZE="${SAMPLE_SIZE:-250}"
WL_ITERS="${WL_ITERS:-3}"
MAX_SITES="${MAX_SITES:-256}"
PCA_DIM="${PCA_DIM:-32}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-8}"
MIN_SAMPLES="${MIN_SAMPLES:-4}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
LOCAL_MODE="${LOCAL_MODE:-matminer_ops}"
CACHE_DIR="${CACHE_DIR:-}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-thread}"
PROFILE_OUT="${PROFILE_OUT:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

PY_ARGS=(
  --index-csv "$ICSD_INDEX"
  --output-dir "$OUT_DIR"
  --sample-size "$SAMPLE_SIZE"
  --wl-iters "$WL_ITERS"
  --max-sites "$MAX_SITES"
  --pca-dim "$PCA_DIM"
  --min-cluster-size "$MIN_CLUSTER_SIZE"
  --min-samples "$MIN_SAMPLES"
  --n-jobs "$N_JOBS"
  --chunk-size "$CHUNK_SIZE"
  --local-mode "$LOCAL_MODE"
  --parallel-backend "$PARALLEL_BACKEND"
  --checkpoint-every "$CHECKPOINT_EVERY"
)

if [ -n "$CACHE_DIR" ]; then
  PY_ARGS+=(--cache-dir "$CACHE_DIR")
fi

if [ -n "$ICSD_CIF_ROOT" ]; then
  PY_ARGS+=(--cif-root "$ICSD_CIF_ROOT")
else
  PY_ARGS+=(--icsd-zip "$ICSD_ZIP")
  if [ -n "$ICSD_ZIP_PASSWORD" ]; then
    PY_ARGS+=(--zip-password "$ICSD_ZIP_PASSWORD")
  fi
fi

if [ -n "$PROFILE_OUT" ]; then
  python -m cProfile -o "$PROFILE_OUT" scripts/icsd_continuous_wl_densification.py "${PY_ARGS[@]}"
else
  python scripts/icsd_continuous_wl_densification.py "${PY_ARGS[@]}"
fi

echo "Wrote outputs to $OUT_DIR"
