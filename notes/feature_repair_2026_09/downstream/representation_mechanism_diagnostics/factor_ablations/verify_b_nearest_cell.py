#!/usr/bin/env python3
"""Independently recompute the corrected B nearest-cell table from retained arrays."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'consistent_transform_20260909/results'
NAME = 'B_cdf_zscore_pca32_crystalweave_protocol'
SOURCES = {'icsd': 'ICSD', 'gnome': 'GNoME', 'mattergen': 'MatterGen', 'mp': 'MP', 'jarvis': 'JARVIS', 'alexandria': 'Alexandria'}


def main():
    path = RESULTS / 'maps' / NAME / 'projections.npz'
    with np.load(path, allow_pickle=False) as data:
        p = {k: data[k] for k in data.files}
    common = json.loads((HERE / 'inputs/common_support_ids.json').read_text())['populations']
    with (RESULTS / 'nearest_cell_recalibration.csv').open() as f:
        rows = list(csv.DictReader(f))
    checked = 0
    for row in rows:
        assert row['ablation'] == NAME
        cutoff = None if row['variant'] == 'full_map_nonnoise' else int(row['variant'][1:5])
        calibration = p['icsd_label'] >= 0
        if cutoff is not None:
            calibration &= (p['icsd_year'] >= 0) & (p['icsd_year'] <= cutoff)
        groups = {}
        for cell, distance in zip(p['icsd_nearest'][calibration], p['icsd_distance'][calibration]):
            groups.setdefault(int(cell), []).append(float(distance))
        radii = {cell: np.quantile(distances, .95, method='linear') for cell, distances in groups.items()}
        source = next(k for k, v in SOURCES.items() if v == row['population'])
        keys = p['icsd_ids'].astype(str) if source == 'icsd' else p[source + '_keys'].astype(str)
        selected = np.ones(len(keys), dtype=bool)
        if source == 'icsd' and cutoff is not None:
            selected &= p['icsd_year'] > cutoff
        if row['support'] == 'common':
            selected &= np.isin(keys, np.asarray(common[row['population']]['ids']).astype(str))
        inside = np.array([int(c) in radii and d <= radii[int(c)] for c, d in zip(p[source + '_nearest'], p[source + '_distance'])])
        n, count = int(selected.sum()), int(inside[selected].sum())
        assert (n, count) == (int(row['n']), int(row['n_in_basin']))
        assert abs(count / n - float(row['rate'])) < 1e-14
        assert int(row['n_calibration']) == int(calibration.sum())
        assert int(row['n_calibrated_cells']) == len(groups)
        assert int(row['n_absent_cells']) == len(p['p95']) - len(groups)
        checked += 1
    report = {'status': 'passed', 'nearest_cell_rows_checked': checked, 'projections_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (HERE / 'consistent_transform_20260909/nearest_cell_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
