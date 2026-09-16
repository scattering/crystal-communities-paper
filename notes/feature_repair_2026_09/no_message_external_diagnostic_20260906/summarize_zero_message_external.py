"""Summarize an isolated zero-message external projection; no model is refitted.

Uses existing centroid/classification helpers and verifies saved/full-transform
agreement before comparing full-record and historical-radius populations.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
from scipy.spatial.distance import cdist
from external_representation_sensitivity import community_basis, projection_rows, rate
from external_frozen_cohorts import record_key
from prepare_repaired_projection_basis import sha256_file


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('run-dir', 'reference-dir', 'production-dir', 'graphlet-assignments'):
        parser.add_argument('--' + arg, type=Path, required=True)
    a = parser.parse_args()
    inputs = {}

    def source(path):
        inputs[str(path)] = sha256_file(path)
        return path

    ref_meta = json.loads(source(a.reference_dir / 'summary.json').read_text())
    assert ref_meta['wl_iters'] == 0
    provenance = json.loads(source(a.run_dir / 'basis/projection_basis_provenance.json').read_text())
    assert provenance['wl_iters'] == 0 and provenance['fit_transform_reproduces_saved_pca']
    if not provenance['transform_vs_saved_allclose']:
        raise ValueError('Saved zero-message coordinates differ from the external transform; resolve this before reporting.')
    ref = read_rows(source(a.reference_dir / 'graph/community_assignments.csv'))
    coords = np.load(source(a.reference_dir / 'features_pca.npy'), allow_pickle=False)
    assert len(ref) == len(coords) == 167392
    ids = np.array([int(r['icsd_id']) for r in ref])
    labels = np.array([int(r['community']) for r in ref])
    years = np.array([int(r['year']) if r['year'].strip() else -1 for r in ref])
    assert len(set(ids)) == len(ids)
    common_icsd = {int(r['icsd_id']) for r in read_rows(source(a.graphlet_assignments))}
    assert len(common_icsd) == 150247 and common_icsd <= set(ids)
    full_mask = np.ones(len(ids), dtype=bool)
    bases = {}
    report = {'status':'complete', 'wl_iters':0,
              'scope':'Full fitted zero-message map and partition. Historical rows restrict centroid/radius members and evaluate later ICSD; these are not cutoff-trained maps.',
              'n_reference':len(ids), 'n_communities':len(set(labels[labels >= 0])),
              'n_noise':int(np.sum(labels < 0)), 'pca_coordinate_verification':provenance,
              'reference_rates':{}, 'sources':{}, 'input_sha256':inputs}
    for key, cutoff in [('full',None),('T1990',1990),('T2000',2000),('T2010',2010)]:
        training = full_mask if cutoff is None else ((years >= 0) & (years <= cutoff))
        evaluation = full_mask if cutoff is None else (years > cutoff)
        basis = community_basis(coords, labels, training)
        bases[key] = basis
        rows, stats = projection_rows(ids[evaluation], coords[evaluation], basis)
        write_rows(a.run_dir / f'icsd_{key}.csv', rows)
        shared = [r['in_basin'] for r in rows if int(r['record_key']) in common_icsd]
        report['reference_rates'][key] = {'all':stats, 'graphlet_common_ids':rate(shared), 'n_basins':len(basis['communities'])}
    for name in ('gnome','mattergen','mp','jarvis','alexandria'):
        slug = 'mattergen-public' if name == 'mattergen' else name
        directory = a.run_dir / 'external' / name
        metadata = json.loads(source(directory / f'{slug}_frontier_summary.json').read_text())
        assert metadata['wl_iters'] == 0 and metadata['threshold_mode'] == 'per_community_p95'
        external = np.load(source(directory / 'features_pca.npy'), allow_pickle=False)
        keys = json.loads(source(directory / 'feature_ids.json').read_text())
        assert len(keys) == len(external) == metadata['n_featurized']
        attempts = read_rows(source(directory / 'attempted_cohort.csv'))
        assert len(attempts) == metadata['n_records_loaded'] == (386 if name == 'mattergen' else 5000)
        production = read_rows(source(a.production_dir / name / f'{slug}_frontier_records.csv'))
        prod_keys = {record_key(r) for r in production}
        assert len(set(keys)) == len(keys)
        item = {'attempted':len(attempts), 'successes':len(keys), 'failures':metadata['n_failures'],
                'success_ids_equal_production':set(keys) == prod_keys,
                'n_shared_production_successes':len(set(keys) & prod_keys), 'rates':{}}
        for key,basis in bases.items():
            rows, stats = projection_rows(keys, external, basis)
            write_rows(directory / f'projection_{key}.csv', rows)
            shared = [r['in_basin'] for r in rows if r['record_key'] in prod_keys]
            item['rates'][key] = {'all':stats, 'production_shared_ids':rate(shared)}
            # Independent distance calculation on a fixed spread of rows.
            selected = np.unique(np.linspace(0,len(keys)-1,min(128,len(keys)),dtype=int))
            distances = cdist(external[selected], basis['centroids'])
            for ix,d in zip(selected,distances):
                chosen = int(np.argmin(d)); row = rows[ix]
                assert row['assigned_community'] == int(basis['communities'][chosen])
                assert np.isclose(row['nearest_centroid_distance'],d[chosen],rtol=1e-11,atol=1e-11)
                assert row['in_basin'] == bool(d[chosen] <= basis['p95'][chosen])
            if key == 'full':
                assert stats['n_in_basin'] == metadata['n_in_basin']
        report['sources'][name] = item
    (a.run_dir / 'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# Zero-message external projection diagnostic','',report['scope'],'',
             '| Reference | ICSD | Shared ICSD | GNoME | MatterGen | MP | JARVIS | Alexandria |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,ref in report['reference_rates'].items():
        rates = [ref['all']['in_basin_fraction'],ref['graphlet_common_ids']['in_basin_fraction']]+[report['sources'][n]['rates'][key]['all']['in_basin_fraction'] for n in report['sources']]
        lines.append('| '+key+' | '+' | '.join(f'{100*r:.4f}%' for r in rates)+' |')
    (a.run_dir / 'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
