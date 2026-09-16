#!/usr/bin/env python3
"""Diagnostic only: frozen production-PCA coordinates in saved graphlet-protocol partitions."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[5]
DOWN = ROOT / 'notes/feature_repair_2026_09/downstream'
OUT = Path(__file__).resolve().parent
EXTERNAL = Path('/tmp/representation_mechanism_inputs/external')
sys.path.insert(0, str(ROOT / 'scripts'))
from external_representation_sensitivity import community_basis, classify, rate

INPUTS = {}
CHECKS = []

def digest(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    INPUTS[str(path)] = {'sha256': value, 'bytes': path.stat().st_size}
    return value

def read_json(path):
    digest(path)
    return json.loads(Path(path).read_text())

def read_csv(path, key=None):
    digest(path)
    return pd.read_csv(path, dtype={key: str} if key else None)

def load_array(path):
    digest(path)
    return np.load(path, allow_pickle=False)

def checked_projection(keys, coords, basis, path):
    nearest, distance, flags = classify(coords, basis)
    # Independent SciPy oracle for a deterministic sample, including distances.
    idx = np.random.default_rng(42).choice(len(coords), min(128, len(coords)), replace=False)
    oracle = cdist(coords[idx], basis['centroids'])
    assert np.array_equal(oracle.argmin(axis=1), nearest[idx])
    assert np.allclose(oracle[np.arange(len(idx)), nearest[idx]], distance[idx], rtol=1e-11, atol=1e-11)
    frame = pd.DataFrame({'record_key': list(map(str, keys)),
        'assigned_community': basis['communities'][nearest],
        'nearest_centroid_distance': distance,
        'community_threshold_p95': basis['p95'][nearest], 'in_basin': flags})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    CHECKS.append({'path': str(path.relative_to(OUT)), 'independent_cdist_rows': len(idx)})
    return frame.set_index('record_key'), rate(flags)

def main():
    full = ROOT / 'notes/feature_repair_2026_09/full_run_results/production'
    rp = DOWN / 'representations/graphlet-dated-replay'
    reference_meta = read_json(DOWN / 'reference_basis/projection_basis_provenance.json')
    feature_ids_path = full / 'sample_assignments.csv'
    coordinate_path = DOWN / 'inputs/features_pca.npy'
    assert digest(feature_ids_path) == reference_meta['reference_files']['sample_assignments']['sha256']
    assert digest(coordinate_path) == reference_meta['reference_files']['icsd_pca']['sha256']
    all_ids = read_csv(feature_ids_path)['icsd_id'].to_numpy()
    all_coords = load_array(coordinate_path)
    assert all_coords.shape == (len(all_ids), 32) and len(set(all_ids)) == len(all_ids)
    rows_by_id = {int(i): k for k, i in enumerate(all_ids)}
    canonical_saved = load_array(DOWN / 'reference_basis/projection_basis.npz')
    canonical_basis = {k: canonical_saved[k] for k in ['communities', 'centroids', 'p95', 'counts']}
    canonical_icsd = read_csv(DOWN / 'external_representation/production_reference/icsd_full.csv', 'record_key').set_index('record_key')
    graphlet_icsd = read_csv(DOWN / 'external_representation/graphlet/basis/icsd_full.csv', 'record_key').set_index('record_key')
    ids = np.array(sorted(map(int, graphlet_icsd.index)))
    assert len(ids) == 150247
    coords = np.asarray(all_coords[[rows_by_id[int(i)] for i in ids]], dtype=np.float64)
    source_arrays, source_ids, canonical_external, graphlet_external = {}, {}, {}, {}
    coverage = {}
    for source in ['gnome', 'mattergen', 'mp', 'jarvis', 'alexandria']:
        directory = EXTERNAL / source
        source_ids[source] = read_json(directory / 'feature_ids.json')
        source_arrays[source] = load_array(directory / 'features_pca.npy')
        meta = read_json(directory / 'feature_metadata.json')
        assert meta['feature_version'] == reference_meta['feature_version']
        assert meta['neighbor_settings'] == reference_meta['neighbor_settings']
        assert meta['worker_sha256'] == reference_meta['worker_sha256']
        assert source_arrays[source].shape == (len(source_ids[source]), 32)
        slug = 'mattergen-public' if source == 'mattergen' else source
        original = read_csv(DOWN / f'external/{source}/{slug}_frontier_records.csv')
        original_key = 'zip_member' if source == 'mattergen' else 'material_id'
        assert source_ids[source] == list(original[original_key])
        assert len(set(source_ids[source])) == len(source_ids[source])
        frame, _ = checked_projection(source_ids[source], source_arrays[source], canonical_basis,
            OUT / 'canonical' / f'{source}.csv')
        assert np.array_equal(frame.assigned_community.to_numpy(), original.assigned_community.to_numpy())
        assert np.allclose(frame.nearest_centroid_distance, original.nearest_centroid_distance, atol=1e-10, rtol=1e-10)
        assert np.array_equal(frame.in_basin.to_numpy(), original.in_basin.to_numpy())
        canonical_external[source] = frame
        graphlet_external[source] = read_csv(DOWN / f'external_representation/graphlet/external/{source}/projection_full.csv', 'record_key').set_index('record_key')
        assert frame.index.is_unique and graphlet_external[source].index.is_unique
        coverage[source] = {'production_successes': len(frame), 'graphlet_successes': len(graphlet_external[source]),
            'common_successes': len(frame.index.intersection(graphlet_external[source].index))}
    predictions = {'canonical_production': {'ICSD': canonical_icsd.loc[list(map(str, ids))], **canonical_external},
                   'CrystalNN_graphlet': {'ICSD': graphlet_icsd.loc[list(map(str, ids))], **graphlet_external}}
    maps = {}
    for kind in ['raw', 'standardized']:
        map_name = f'production_pca_{kind}_same_protocol'
        part = read_csv(rp / map_name / 'community_assignments.csv').set_index('icsd_id')
        metadata = read_json(rp / map_name / 'partition.json')
        assert set(part.index) == set(ids) and len(part) == len(ids)
        assert metadata['production_pca_sha256'] == digest(coordinate_path)
        assert metadata['production_pca_ids_sha256'] == digest(feature_ids_path)
        labels = part.loc[ids, 'community'].to_numpy()
        x, external_x = coords, source_arrays
        if kind == 'standardized':
            scaler = read_json(rp / 'pca_subset_scaler.json')
            mean, scale = np.array(scaler['mean']), np.array(scaler['scale'])
            assert scaler['n_samples'] == len(ids)
            assert np.allclose(coords.mean(axis=0), mean, rtol=1e-10, atol=1e-10)
            assert np.allclose(coords.std(axis=0), scale, rtol=1e-10, atol=1e-10)
            x = (coords - mean) / scale
            external_x = {s: (v - mean) / scale for s, v in source_arrays.items()}
        basis = community_basis(x, labels, np.ones(len(ids), dtype=bool))
        assert len(basis['communities']) == metadata['n_communities']
        assert int((labels < 0).sum()) == metadata['n_noise']
        np.savez_compressed(OUT / f'{kind}_basis.npz', **basis)
        entry = {}
        entry['ICSD'], _ = checked_projection(ids, x, basis, OUT / kind / 'ICSD.csv')
        for source in source_arrays:
            entry[source], _ = checked_projection(source_ids[source], external_x[source], basis, OUT / kind / f'{source}.csv')
        predictions[map_name] = entry
        maps[map_name] = {'n_entries': len(ids), 'n_communities': len(basis['communities']),
            'n_noise': int((labels < 0).sum()), 'partition_metadata': metadata}
    rates, paired = [], []
    for name, entries in predictions.items():
        icsd_rate = rate(entries['ICSD'].in_basin)
        rates.append({'map': name, 'source': 'ICSD', **icsd_rate})
        for source in source_arrays:
            common = canonical_external[source].index.intersection(graphlet_external[source].index)
            selected = entries[source].loc[common]
            r = rate(selected.in_basin)
            rates.append({'map': name, 'source': source, **r,
                'ICSD_minus_source_pp': 100 * (icsd_rate['in_basin_fraction'] - r['in_basin_fraction'])})
            if name in ['production_pca_raw_same_protocol', 'production_pca_standardized_same_protocol']:
                baseline = canonical_external[source].loc[common].in_basin.to_numpy()
                current = selected.in_basin.to_numpy()
                paired.append({'map': name, 'source': source, 'n': len(common),
                    'canonical_out_to_in': int((~baseline & current).sum()),
                    'canonical_in_to_out': int((baseline & ~current).sum())})
    pd.DataFrame(rates).to_csv(OUT / 'common_support_rates.csv', index=False)
    pd.DataFrame(paired).to_csv(OUT / 'paired_changes.csv', index=False)
    report = {'diagnostic': 'Frozen production PCA coordinates under existing graphlet-protocol partitions',
        'command': 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python ' + str(Path(__file__).relative_to(ROOT)),
        'common_ICSD_n': len(ids), 'coverage': coverage, 'maps': maps, 'rates': rates,
        'paired_changes': paired, 'checks': CHECKS,
        'limitations': [
            'Full-record-map descriptive sensitivity, not held-out historical validation.',
            'Canonical map uses its original full population/partition. Same-protocol maps change the partition protocol AND fitting population, while keeping production coordinates frozen.',
            'Raw production-PCA same-protocol map and CrystalNN graphlet map use identical 150247 fitting/evaluation IDs and the same graphlet graph protocol; descriptor/geometry changes remain bundled.',
            'Standardized variant additionally rescales the32 production PCA coordinates using the saved subset scaler.',
            'No Voronoi external projection comparison is made here; its available independent partition uses a different146186-ID cohort.',
            'External full32-D coordinates were retrieved from the saved producer. They were independently checked against every canonical assignment, distance and flag; raw213-D external features were not retrieved.'
        ], 'inputs': INPUTS, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame(rates)[['map','source','n','n_in_basin','in_basin_fraction','ICSD_minus_source_pp']].to_string(index=False))
    print('All independent distance checks and canonical per-record reproduction checks passed.')

if __name__ == '__main__':
    main()
