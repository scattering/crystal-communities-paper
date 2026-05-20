#!/bin/bash
set -euo pipefail

WORKROOT="${WORKROOT:-/work2/09870/williamratcliff/stampede3}"
CONDA_ROOT="${CONDA_ROOT:-$WORKROOT/conda}"
ENV_PATH="${ENV_PATH:-$WORKROOT/conda_envs/icsd_densify}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if [ -d "$ENV_PATH" ]; then
  echo "Environment already exists: $ENV_PATH"
  exit 0
fi

conda create -y -p "$ENV_PATH" python=3.11
conda activate "$ENV_PATH"

python -m pip install --upgrade pip
python -m pip install \
  matminer \
  numpy \
  scipy \
  pandas \
  scikit-learn \
  hdbscan \
  umap-learn \
  pymatgen

python - <<'PY'
from importlib.metadata import version
import numpy
import sklearn
print("numpy", numpy.__version__)
print("sklearn", sklearn.__version__)
print("pymatgen", version("pymatgen"))
PY
