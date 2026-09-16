#!/usr/bin/env python3
"""Measure both existing neighbor rules on the frozen public cohorts only."""
from pathlib import Path
import bz2
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import warnings
import zipfile

import numpy as np
from pymatgen.core import Structure

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'stage'
if not ROOT.exists():
    ROOT = next(p for p in HERE.parents if (p / 'scripts/crystal_neighbors.py').exists())
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from experiments.graphlet_compare import graphlet_features as gf

BASE = Path(os.environ.get('WORK', '/path/to/tacc/work'))  # TACC work root
PATHS = {
    'gnome': BASE / 'reference_data/gnome_data/by_id.zip',
    'mattergen': BASE / 'mattergen/data-release/cifs.zip',
    'mp': BASE / 'reference_data/mp_theoretical_candidates_20260427.jsonl',
    'jarvis': BASE / 'reference_data/jarvis_dft/jdft_3d-12-12-2022.json',
    'alexandria': BASE / 'reference_data/alexandria_pbe_2025_07_02',
}
METHODS = ('crystalnn', 'voronoinn')
warnings.filterwarnings('ignore', module='pymatgen')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def summarize_neighbors(structure, method):
    """Average per-site neighbor weight, retaining every distinct periodic image."""
    infos = gf._structure_neighbor_info(structure, neighbor_method=method)
    cn = np.zeros(len(structure))
    n_contacts = 0
    self_image_mass = 0.0
    for i, entries in enumerate(infos):
        seen = set()
        for entry in entries:
            j = int(entry['site_index'])
            shift = np.asarray(entry['site'].frac_coords) - structure[j].frac_coords
            image = np.rint(shift).astype(int)
            if not np.allclose(shift, image, rtol=0, atol=1e-6):
                raise ValueError('Neighbor is not an integer periodic image')
            identity = (j, *image.tolist())
            if identity in seen:
                raise ValueError('Duplicate periodic neighbor identity')
            seen.add(identity)
            weight = float(entry['weight'])
            cn[i] += weight
            n_contacts += 1
            if i == j:
                self_image_mass += weight
    if not np.isfinite(cn).all() or np.any(cn <= 0):
        raise ValueError('Invalid coordination vector')
    return {
        'mean_cn': float(cn.mean()),
        'median_site_cn': float(np.median(cn)),
        'min_site_cn': float(cn.min()),
        'max_site_cn': float(cn.max()),
        'n_contacts': n_contacts,
        'total_contact_weight': float(cn.sum()),
        'self_image_weight_fraction': float(self_image_mass / cn.sum()),
    }


def extract(task):
    row, kind, raw = task
    ident = {'source': row['source'], 'material_id': row['material_id']}
    raw_bytes = raw if isinstance(raw, bytes) else json.dumps(raw, sort_keys=True).encode()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    try:
        if kind == 'cif':
            s = Structure.from_str(raw.decode('utf-8'), fmt='cif')
        elif kind == 'jarvis':
            s = Structure(raw['lattice_mat'], raw['elements'], raw['coords'],
                          coords_are_cartesian=bool(raw.get('cartesian', False)))
        else:
            s = Structure.from_dict(raw)
        if not s.is_ordered:
            raise ValueError('Disordered public record')
        if len(s) != int(row['n_sites']):
            raise ValueError('Site count differs from frozen manifest')
        if len(s.composition.elements) != int(row['n_elements']):
            raise ValueError('Element count differs from frozen manifest')
        common = dict(ident, n_sites=len(s), n_elements=len(s.composition.elements),
                      parsed_formula=s.composition.reduced_formula,
                      volume_per_site=float(s.volume / len(s)),
                      input_record_sha256=raw_hash)
    except Exception as exc:
        return [], [dict(ident, neighbor_method=m, stage='parse', error=repr(exc)) for m in METHODS]
    results, failures = [], []
    for method in METHODS:
        try:
            results.append(dict(common, neighbor_method=method, **summarize_neighbors(s, method)))
        except Exception as exc:
            failures.append(dict(ident, neighbor_method=method, stage='neighbors', error=repr(exc)))
    return results, failures


def read_alex(task):
    path, wanted = task
    with bz2.open(path, 'rt') as f:
        data = json.load(f)
    found = {}
    for e in data['entries'] if isinstance(data, dict) else data:
        mid = str((e.get('data') or {}).get('mat_id') or e.get('entry_id'))
        if mid in wanted:
            if mid in found:
                raise ValueError('Duplicate Alexandria ID within shard: ' + mid)
            found[mid] = e['structure']
    return found


def load_tasks(source, rows):
    wanted = {r['material_id']: r for r in rows}
    tasks = []
    if source in ('gnome', 'mattergen'):
        with zipfile.ZipFile(PATHS[source]) as z:
            names = set(z.namelist())
            for mid, row in wanted.items():
                choices = [mid] if source == 'mattergen' else [mid + '.cif', mid + '.CIF', 'by_id/' + mid + '.cif', 'by_id/' + mid + '.CIF']
                available = [name for name in choices if name in names]
                if len(available) != 1:
                    raise ValueError('Ambiguous or missing ZIP identity: ' + mid)
                tasks.append((row, 'cif', z.read(available[0])))
    elif source == 'mp':
        with PATHS[source].open() as f:
            for line in f:
                e = json.loads(line)
                if e['material_id'] in wanted:
                    tasks.append((wanted[e['material_id']], 'dict', e['structure']))
    elif source == 'jarvis':
        with PATHS[source].open() as f:
            data = json.load(f)
        for e in data:
            if e.get('jid') in wanted:
                tasks.append((wanted[e['jid']], 'jarvis', e['atoms']))
    else:
        paths = [PATHS[source] / f'alexandria_{i:05d}.json.bz2' for i in (0, 19, 38)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
            for part in pool.map(read_alex, [(p, set(wanted)) for p in paths]):
                tasks.extend((wanted[mid], 'dict', sd) for mid, sd in part.items())
    found = [row['material_id'] for row, _, _ in tasks]
    if len(found) != len(set(found)) or set(found) != set(wanted):
        raise ValueError(f'Input identity mismatch for {source}: {len(found)} / {len(wanted)}')
    return tasks


def main():
    started = time.time()
    manifest = json.loads((HERE / 'cohort_manifest.json').read_text())
    if isinstance(manifest, dict):
        manifest = manifest['records']
    protocol = json.loads((HERE / 'protocol.json').read_text())
    assert digest(HERE / 'cohort_manifest.json') == protocol['manifest_sha256']
    assert digest(HERE / 'matched_pairs.csv') == protocol['pairs_sha256']
    out = HERE / 'results'
    out.mkdir(exist_ok=False)
    workers = int(os.environ.get('COORDINATION_WORKERS', '48'))
    failures = []
    counts = {}
    fieldnames = ['source', 'material_id', 'n_sites', 'n_elements', 'parsed_formula',
                  'volume_per_site', 'input_record_sha256', 'neighbor_method', 'mean_cn',
                  'median_site_cn', 'min_site_cn', 'max_site_cn', 'n_contacts',
                  'total_contact_weight', 'self_image_weight_fraction']
    with (out / 'coordination.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source in PATHS:
            rows = [r for r in manifest if r['source'] == source]
            print(f'Loading {source}: {len(rows)} structures', flush=True)
            tasks = load_tasks(source, rows)
            counts[source] = {'requested': len(tasks), 'successful': {m: 0 for m in METHODS}}
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                for i, (result, errors) in enumerate(pool.map(extract, tasks, chunksize=4), 1):
                    writer.writerows(result)
                    failures.extend(errors)
                    for row in result:
                        counts[source]['successful'][row['neighbor_method']] += 1
                    if i % 500 == 0:
                        f.flush()
                        print(f'{source}: {i}/{len(tasks)}; elapsed {time.time()-started:.1f}s', flush=True)
            f.flush()
            dump(out / 'failures.json', failures)
            print(json.dumps(counts[source]), flush=True)
    inputs = [HERE / n for n in ('cohort_manifest.json', 'matched_pairs.csv', 'protocol.json', 'extract_coordination.py')]
    inputs += [ROOT / 'scripts/crystal_neighbors.py', ROOT / 'experiments/graphlet_compare/graphlet_features.py']
    dump(out / 'extraction_summary.json', {'counts': counts, 'failures': len(failures),
         'runtime_seconds': time.time()-started, 'workers': workers})
    dump(out / 'provenance.json', {'job_id': os.environ.get('SLURM_JOB_ID'),
         'input_hashes': {str(p): digest(p) for p in inputs},
         'neighbor_settings': {m: gf.neighbor_settings(m) for m in METHODS},
         'packages': {p: importlib.metadata.version(p) for p in ('pymatgen', 'numpy', 'scipy')},
         'public_source_paths': {s: str(p) for s, p in PATHS.items()},
         'licensed_icsd_structures_used': False})
    print('COMPLETE', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
