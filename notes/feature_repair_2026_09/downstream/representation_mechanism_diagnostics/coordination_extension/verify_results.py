#!/usr/bin/env python3
"""Check accounting, frozen joins and exact reproduction of the earlier pilot."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    checks = []
    def check(ok, label):
        checks.append({'check': label, 'passed': bool(ok)})
        if not ok:
            raise AssertionError(label)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    for filename, field in [('cohort_manifest.json', 'manifest_sha256'), ('matched_pairs.csv', 'pairs_sha256')]:
        check(hashlib.sha256((HERE / filename).read_bytes()).hexdigest() == protocol[field], filename + ' frozen hash')
    manifest = json.loads((HERE / 'cohort_manifest.json').read_text())
    identities = {(r['source'], r['material_id']) for r in manifest}
    check(len(identities) == len(manifest) == 20265, 'Exact unique public-cohort size')
    rows = list(csv.DictReader((HERE / 'results/coordination.csv').open()))
    data = {(r['source'], r['material_id'], r['neighbor_method']): r for r in rows}
    check(len(data) == len(rows), 'Unique measured record/method identities')
    failures = json.loads((HERE / 'results/failures.json').read_text())
    failed = {(r['source'], r['material_id'], r['neighbor_method']) for r in failures}
    expected = {(s, i, m) for s, i in identities for m in ('crystalnn', 'voronoinn')}
    check(len(failed) == len(failures), 'Unique failure identities')
    check(not (set(data) & failed) and set(data) | failed == expected, 'Complete success/failure accounting')
    for key, row in data.items():
        cn, n, mass = float(row['mean_cn']), int(row['n_sites']), float(row['total_contact_weight'])
        check(np.isfinite(cn) and cn > 0 and np.isclose(cn * n, mass, rtol=1e-12, atol=1e-10), str(key) + ' coordination mass')
        check(float(row['min_site_cn']) - 1e-10 <= cn <= float(row['max_site_cn']) + 1e-10, str(key) + ' site range')
        if key[2] == 'voronoinn':
            check(np.isclose(mass, int(row['n_contacts']), rtol=0, atol=1e-10), str(key) + ' unit-weight count')
    pilot = HERE.parent / 'bonding_cohort_pilot'
    old_rows = json.loads((pilot / 'sample.json').read_text())
    old_metadata = json.loads((pilot / 'results/structure_metadata.json').read_text())
    reproduced = 0
    max_difference = 0.0
    for old in old_rows:
        source, mid = old['source'], old['material_id']
        tag = source + '_' + hashlib.sha256(mid.encode()).hexdigest()[:20]
        old_cn = old_metadata[tag]['mean_weighted_cn']
        check((source, mid, 'crystalnn') in data, 'Pilot target present ' + tag)
        delta = abs(float(data[source, mid, 'crystalnn']['mean_cn']) - old_cn)
        max_difference = max(max_difference, delta)
        check(delta < 1e-8, 'Pilot coordination reproduced ' + tag)
        reproduced += 1
    summary = json.loads((HERE / 'results/summary.json').read_text())
    pairs = list(csv.DictReader((HERE / 'matched_pairs.csv').open()))
    for method in ('crystalnn', 'voronoinn'):
        for analysis in ('all', 'unused'):
            for source in ('mattergen', 'mp', 'jarvis', 'alexandria'):
                selected = [r for r in pairs if r['source'] == source and r['analysis'] == analysis]
                differences = [float(data['gnome', r['gnome_id'], method]['mean_cn']) - float(data[source, r['comparator_id'], method]['mean_cn'])
                               for r in selected if ('gnome', r['gnome_id'], method) in data and (source, r['comparator_id'], method) in data]
                saved = summary['matched'][method][analysis][source]
                check(saved['n_planned'] == len(selected) and saved['n_complete'] == len(differences), f'{method}/{analysis}/{source} pair accounting')
                check(np.isclose(saved['primary']['median_paired_difference'], np.median(differences), rtol=0, atol=1e-12), f'{method}/{analysis}/{source} median reproduced')
    result = {'passed': len(checks), 'failed': 0, 'records': len(manifest), 'successful_measurements': len(rows), 'failed_measurements': len(failed),
              'pilot_records_reproduced': reproduced, 'maximum_pilot_coordination_difference': max_difference,
              'frozen_hashes_verified': True, 'all_matched_medians_independently_reproduced': True}
    (HERE / 'results/verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
