# Score new crystal structures

The frozen CrystalWeave reference represents 167,392 ICSD entries and uses a
32-component PCA basis with 2,939 retained communities. A new structure is
encoded with the same settings and projected without refitting the scaler,
PCA, partition or community radii.

This workflow requires the matching revised artifact bundle, whose public
release is pending. An earlier published feature matrix is incompatible with
the current encoder. Raw ICSD CIFs are not needed to score new structures
against a supplied fitted reference.

## Prepare a validated reference

Follow [dashboard setup](../dashboard/README.md) to prepare a local manifest
from the saved basis, assignments, events, community evidence and figures.
The preparation command validates the encoder and all loaded artifacts.
Set `ICSD_DASHBOARD_MANIFEST` to the resulting `manifest.json`.

## Score one CIF or a collection

From the repository root, using the project environment:

```python
import os
import sys
from pathlib import Path

sys.path[:0] = [str(Path("scripts").resolve()), str(Path("dashboard").resolve())]
from pymatgen.core import Structure
from frozen_backend import load_bundle, score

bundle = load_bundle(os.environ["ICSD_DASHBOARD_MANIFEST"])
structure = Structure.from_file("my_structure.cif")
result = score(structure, bundle)
print({key: result[key] for key in (
    "formula", "community", "distance", "threshold", "frontier"
)})
```

Load the bundle once and call `score` for each structure. Keep your source
identifier with every result and save parse/feature failures separately.
Structures must contain 1–256 sites. The encoder uses the average
crystallographic site skeleton for partial occupancies and preserves their
occupancy-weighted chemistry. It does not infer a particular local ordering
of a disordered structure.

The backend validates the feature version, source hashes, projection state,
row identities and partition. Changing the encoder requires a newly matched
reference, rather than relabelling the old bundle's provenance.

## Interpret the output

`community` identifies the nearest centroid. `distance` is its Euclidean
distance in all 32 PCA components; `threshold` is that community's p95 member
distance. `frontier` is true when `distance > threshold`; equality is in basin.
`xy` contains the first two components for display. A visually nearby point in
a two-dimensional plot need not be nearby in the full scoring space.

The result also contains membership size, community birth information,
feature diagnostics and a continuous historical-accessibility coordinate.
The continuous coordinate is descriptive and is not a calibrated synthesis
probability. Family descriptions summarize communities rather than certify
an uploaded structure's atomic identity.

## Reproduce the fixed external cohorts

`scripts/regenerate_external_projection.py` reads the frozen cohort IDs and
local structures for GNoME, MatterGen, MP, JARVIS or Alexandria. For example:

```bash
python scripts/regenerate_external_projection.py \
  --source mattergen \
  --source-path /path/to/mattergen-release.zip \
  --historical-records /path/to/frozen-mattergen-records.csv \
  --basis notes/feature_repair_2026_09/downstream/reference_basis/projection_basis.npz \
  --out-dir output/mattergen \
  --n-jobs 8
```

This command reproduces a declared sample; it is not a sampler for an arbitrary
new library. Use the manifest's matching source release and cohort table.
Older generic `analyze_*_frontier.py` commands retain historical paths and
threshold conventions; the frozen backend and regeneration command above are
the current scoring entry points.

## Compare representations and formula precedent

Each alternative representation needs its own encoder, fitted transform,
partition and member-radius calibration. The two Graphlet maps also have
separate frozen histogram bins and neighbor rules. Keep successful IDs when
joining maps, and report failed encodings and population differences.

Formula precedent is computed separately. Use the canonical formula-layer
utilities and occupancy-aware reference under
`notes/feature_repair_2026_09/downstream/formula_layers/`; direct equality of
pymatgen display strings is not the composition test. A formula/structure
quadrant reports two forms of experimental precedent, not a calibrated
synthesis-success rate. See [SCHEMA.md](SCHEMA.md) for the formula layers,
nearest-ICSD ElMD and separate disorder-aware matching analysis.
