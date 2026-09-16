#!/usr/bin/env python3
"""Recalibrate saved cutoff-map radii in nearest-centroid cells; never refit maps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def transform(features, fitted):
    return (((np.asarray(features, dtype=np.float64) - fitted['scaler_mean']) /
             fitted['scaler_scale']) - fitted['pca_mean']) @ fitted['pca_components'].T


def nearest(X, centers):
    """Match the original producer's top-three/exact-distance assignment rule."""
    chosen = np.empty(len(X), dtype=np.int64)
    distance = np.empty(len(X), dtype=float)
    norms = np.sum(centers * centers, axis=1)
    k = min(3, len(centers))
    exact_ties = 0
    for start in range(0, len(X), 1024):
        x = X[start:start + 1024]
        d2 = np.sum(x*x, axis=1)[:, None] + norms - 2*x @ centers.T
        cand = np.argpartition(d2, k - 1, axis=1)[:, :k]
        distances = np.linalg.norm(x[:, None, :] - centers[cand], axis=2)
        local = distances.argmin(axis=1)
        d = distances[np.arange(len(x)), local]
        exact_ties += int(((distances == d[:, None]).sum(axis=1) > 1).sum())
        chosen[start:start + len(x)] = cand[np.arange(len(x)), local]
        distance[start:start + len(x)] = d
    return chosen, distance, exact_ties


def radii_by_cell(chosen, distance, calibration_mask, n_centers):
    counts = np.bincount(chosen[calibration_mask], minlength=n_centers)
    radii = np.full(n_centers, np.nan)
    for position in np.flatnonzero(counts):
        radii[position] = np.quantile(distance[calibration_mask & (chosen == position)], .95)
    return counts, radii


def quadrant(inside, matched):
    a, b = np.asarray(inside, bool), np.asarray(matched, bool)
    assert len(a) == len(b)
    n, n11 = len(a), int((a & b).sum())
    ni, nf = int(a.sum()), int(b.sum())
    return {'n': n, 'in_basin_and_match': n11, 'in_basin_and_no_match': ni-n11,
        'frontier_and_match': nf-n11, 'frontier_and_no_match': n-ni-nf+n11,
        'p_in_basin': ni/n if n else None, 'p_match': nf/n if n else None,
        'enrichment_ratio_obs_over_independence': n*n11/(ni*nf) if ni*nf else None}


def paired_enrichment_bootstrap(old, new, formula, rng, n_boot):
    n = len(old)
    if not n:
        return {'n_boot': 0}
    index = 4*np.asarray(old, int) + 2*np.asarray(new, int) + np.asarray(formula, int)
    counts = np.bincount(index, minlength=8)
    draws = rng.multinomial(n, counts/n, size=n_boot)
    cells = np.arange(8)
    match = (cells & 1) > 0
    ratios = []
    for bit in [4, 2]:
        inside = (cells & bit) > 0
        ni, nf, n11 = draws[:, inside].sum(1), draws[:, match].sum(1), draws[:, inside & match].sum(1)
        denominator = ni*nf
        ratios.append(np.divide(n*n11.astype(float), denominator, out=np.full(n_boot, np.nan), where=denominator > 0))
    def interval(values):
        valid = values[np.isfinite(values)]
        return list(map(float, np.quantile(valid, [.025,.975]))) if len(valid) else [None,None]
    return {'n_boot': n_boot, 'original_enrichment_ci95': interval(ratios[0]),
        'recalibrated_enrichment_ci95': interval(ratios[1]),
        'paired_enrichment_difference_ci95': interval(ratios[1]-ratios[0]),
        'interpretation': 'Paired row bootstrap conditional on saved maps and calibrated radii; calibration/refit uncertainty and community dependence are not represented.'}


def rate_row(frame, cutoff, policy, source):
    mask = frame.nearest_cell_calibrated.to_numpy(bool)
    old = frame.in_basin.to_numpy(bool)
    new = frame.recalibrated_in_basin.to_numpy(float)
    n, eligible = len(frame), int(mask.sum())
    old_count = int(old[mask].sum())
    new_count = int(np.nansum(new))
    missing = n-eligible
    return {'cutoff': cutoff, 'policy': policy, 'source': source,
        'n_total': n, 'n_calibrated': eligible, 'n_uncalibrated': missing,
        'original_all_n_in_basin': int(old.sum()), 'original_all_rate': float(old.mean()) if n else None,
        'original_supported_n_in_basin': old_count, 'original_supported_rate': old_count/eligible if eligible else None,
        'recalibrated_n_in_basin': new_count, 'recalibrated_rate': new_count/eligible if eligible else None,
        'full_denominator_lower_bound': new_count/n if n else None,
        'full_denominator_upper_bound': (new_count+missing)/n if n else None,
        'out_to_in': int((~old[mask] & (new[mask] == 1)).sum()),
        'in_to_out': int((old[mask] & (new[mask] == 0)).sum())}


def rescore(frame, communities, radii):
    out = frame.copy()
    positions = np.searchsorted(communities, frame.assigned_community.to_numpy(int))
    assert np.array_equal(communities[positions], frame.assigned_community.to_numpy(int))
    thresholds = radii[positions]
    valid = np.isfinite(thresholds)
    out['nearest_cell_threshold_p95'] = thresholds
    out['nearest_cell_calibrated'] = valid
    out['recalibrated_in_basin'] = np.where(valid, (frame.nearest_centroid_distance.to_numpy() <= thresholds).astype(float), np.nan)
    return out


def self_test():
    centers = np.array([[0.,0.],[10.,0.],[100.,100.]])
    x = np.array([[0.,0.],[1.,0.],[9.,0.],[10.,0.]])
    c, d, ties = nearest(x, centers)
    assert np.array_equal(c, cdist(x, centers).argmin(axis=1))
    counts, r = radii_by_cell(c,d,np.ones(4,bool),3)
    assert counts.tolist() == [2,2,0] and np.allclose(r[:2],[.95,.95]) and np.isnan(r[2])
    table = pd.DataFrame({'assigned_community':[0,2], 'nearest_centroid_distance':[.5,1.], 'in_basin':[0,1]})
    scored = rescore(table,np.arange(3),r)
    assert scored.nearest_cell_calibrated.tolist() == [True,False]
    assert scored.recalibrated_in_basin.iloc[0] == 1 and np.isnan(scored.recalibrated_in_basin.iloc[1])
    q = quadrant([1,1,0,0],[1,0,1,0])
    assert q['enrichment_ratio_obs_over_independence'] == 1
    b = paired_enrichment_bootstrap([1,1,0,0],[1,1,0,0],[1,0,1,0],np.random.default_rng(1),100)
    assert b['paired_enrichment_difference_ci95'] == [0.,0.]
    print('Self-test passed: nearest assignment, p95 calibration, empty-cell exclusion, quadrant arithmetic, paired bootstrap.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--sample-assignments', type=Path)
    parser.add_argument('--cutoff-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cutoffs', nargs='+', type=int, default=[1990,2000,2010])
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for name in ['features','sample_assignments','cutoff_root','output_dir']:
        if getattr(args,name) is None:
            parser.error('--' + name.replace('_','-') + ' is required')
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {}
    def pin(path, expected=None):
        value = sha(path)
        if expected is not None:
            assert value == expected, f'Input hash mismatch: {path}'
        inputs[str(path)] = {'sha256':value,'bytes':Path(path).stat().st_size}
    summary_path = args.cutoff_root / 'cutoff_trained_retrospective_summary.json'
    pin(summary_path)
    summary = json.loads(summary_path.read_text())
    pin(args.features, summary['inputs']['features']['sha256'])
    pin(args.sample_assignments, summary['inputs']['sample_assignments']['sha256'])
    features = np.load(args.features, mmap_mode='r', allow_pickle=False)
    sample = pd.read_csv(args.sample_assignments, keep_default_na=False)
    ids = sample.icsd_id.to_numpy(int)
    assert features.shape == (len(ids),213) and len(set(ids)) == len(ids)
    positions_by_id = {int(value):position for position,value in enumerate(ids)}
    rates, joint, blocks = [], [], {}
    for cutoff in args.cutoffs:
        saved = summary['cutoffs'][str(cutoff)]
        for info in saved['outputs'].values():
            pin(args.cutoff_root / info['path'], info['sha256'])
        leaf = args.cutoff_root / f'T{cutoff}'
        out = args.output_dir / f'T{cutoff}'
        out.mkdir(parents=True,exist_ok=True)
        train = pd.read_csv(leaf / 'training_partition.csv',keep_default_na=False)
        held = pd.read_csv(leaf / 'heldout_entries.csv',keep_default_na=False)
        external = pd.read_csv(leaf / 'external_classifications.csv',keep_default_na=False)
        with np.load(leaf / 'cutoff_map.npz',allow_pickle=False) as archive:
            fitted = {k:archive[k] for k in archive.files}
        communities, centers, old_radii = fitted['communities'], fitted['centroids'], fitted['radii_p95']
        assert np.all(np.diff(communities)>0)
        assert len(train) == saved['n_train_dated'] and len(held) == saved['n_heldout_dated']
        assert train.icsd_id.is_unique and held.icsd_id.is_unique
        assert set(train.icsd_id).isdisjoint(set(held.icsd_id))
        assert (train.year <= cutoff).all() and (held.year > cutoff).all()
        row_positions = [positions_by_id[int(i)] for i in train.icsd_id]
        coordinates = transform(features[row_positions],fitted)
        assert coordinates.shape == (len(train),32) and np.isfinite(coordinates).all()
        labels = train.cutoff_community.to_numpy(int)
        errors_c, errors_r = [], []
        for position,community in enumerate(communities):
            members = coordinates[labels == community]
            assert len(members) == fitted['community_sizes'][position]
            centroid = members.mean(axis=0)
            radius = np.quantile(np.linalg.norm(members-centroid,axis=1),.95)
            errors_c.append(np.max(np.abs(centroid-centers[position])))
            errors_r.append(abs(radius-old_radii[position]))
        assert max(errors_c) < 1e-8 and max(errors_r) < 1e-8
        chosen, distance, ties = nearest(coordinates,centers)
        audit = np.random.default_rng(cutoff).choice(len(train),128,replace=False)
        oracle = cdist(coordinates[audit],centers)
        assert np.allclose(oracle.min(axis=1),distance[audit],rtol=1e-11,atol=1e-11)
        # Independently verify a heldout raw-to-map sample against its stored distance.
        audit_h = np.random.default_rng(cutoff+1).choice(len(held),128,replace=False)
        h = held.iloc[audit_h]
        hx = transform(features[[positions_by_id[int(i)] for i in h.icsd_id]],fitted)
        hd = cdist(hx,centers)
        hp = np.searchsorted(communities,h.assigned_community.to_numpy(int))
        assert np.allclose(hd[np.arange(len(h)),hp],h.nearest_centroid_distance,rtol=1e-10,atol=1e-10)
        assert np.allclose(hd.min(axis=1),h.nearest_centroid_distance,rtol=1e-10,atol=1e-10)
        for frame in [held,external]:
            p = np.searchsorted(communities,frame.assigned_community.to_numpy(int))
            assert np.array_equal(communities[p],frame.assigned_community.to_numpy(int))
            assert np.allclose(old_radii[p],frame.community_threshold_p95,rtol=1e-12,atol=1e-12)
            assert np.array_equal((frame.nearest_centroid_distance.to_numpy() <= old_radii[p]),frame.in_basin.to_numpy(bool))
        # Verify all saved old quadrant counts/ratios before changing thresholds.
        parseable = held[held.formula != ''].copy()
        assert parseable.formula_identity.ne('').all()
        first_formula = parseable.sort_values(['year','icsd_id'],kind='stable').drop_duplicates('formula_identity',keep='first')
        for reference,analysis in saved['analyses'].items():
            for unit_name,unit in [('per_entry',parseable),('per_formula',first_formula)]:
                q = quadrant(unit.in_basin,unit[f'formula_match_{reference}'])
                for key,value in q.items():
                    expected = analysis[unit_name]['quadrant'][key]
                    assert (value is None and expected is None) or np.isclose(value,expected,rtol=1e-12,atol=1e-12)
        calibration = train.copy()
        calibration['nearest_community'] = communities[chosen]
        calibration['nearest_centroid_distance'] = distance
        cell_table = pd.DataFrame({'community':communities,'original_member_count':fitted['community_sizes'],
            'original_member_radius_p95':old_radii})
        blocks[str(cutoff)] = {'n_training':len(train),'n_training_original_noise':int((labels<0).sum()),
            'n_communities':len(communities),'training_exact_top3_distance_ties':ties,
            'max_reconstructed_original_centroid_error':float(max(errors_c)),
            'max_reconstructed_original_radius_error':float(max(errors_r)),
            'independent_training_distance_sample':128,'independent_heldout_transform_sample':128,'policies':{}}
        for policy,mask in [('all_training',np.ones(len(train),bool)),('original_nonnoise_only',labels>=0)]:
            counts,radii = radii_by_cell(chosen,distance,mask,len(communities))
            cell_table[policy+'_calibration_count'] = counts
            cell_table[policy+'_radius_p95'] = radii
            calibration[policy+'_in_basin'] = np.where(np.isfinite(radii[chosen]),distance<=radii[chosen],np.nan)
            rescored_h,rescored_e = rescore(held,communities,radii),rescore(external,communities,radii)
            policy_out = out / policy
            policy_out.mkdir(exist_ok=True)
            rescored_h.to_csv(policy_out/'heldout_entries.csv',index=False)
            rescored_e.to_csv(policy_out/'external_classifications.csv',index=False)
            block = {'n_calibration_rows':int(mask.sum()),'n_nonempty_cells':int((counts>0).sum()),
                'n_empty_cells':int((counts==0).sum()),'empty_community_ids':communities[counts==0].tolist(),
                'n_cells_below_10_training_rows':int(((counts>0)&(counts<10)).sum()),
                'empirical_calibration_acceptance':float((distance[mask] <= radii[chosen[mask]]).mean())}
            blocks[str(cutoff)]['policies'][policy] = block
            hrate = rate_row(rescored_h,cutoff,policy,'ICSD')
            rates.append(hrate)
            for source,source_rows in rescored_e.groupby('source',sort=True):
                sr = rate_row(source_rows,cutoff,policy,source)
                sr['ICSD_minus_source_pp'] = 100*(hrate['recalibrated_rate']-sr['recalibrated_rate']) if sr['recalibrated_rate'] is not None else None
                sr['original_supported_ICSD_minus_source_pp'] = 100*(hrate['original_supported_rate']-sr['original_supported_rate']) if sr['original_supported_rate'] is not None else None
                rates.append(sr)
            rng = np.random.default_rng(42+cutoff+(0 if policy=='all_training' else 10000))
            units = [('per_entry',rescored_h[rescored_h.formula!=''])]
            # Select first formula records BEFORE excluding unsupported cells.
            units.append(('per_formula',rescored_h.set_index('icsd_id').loc[first_formula.icsd_id].reset_index()))
            for unit_name,unit in units:
                valid = unit[unit.nearest_cell_calibrated].copy()
                for reference in saved['analyses']:
                    match = valid[f'formula_match_{reference}'].to_numpy(bool)
                    old,new = valid.in_basin.to_numpy(bool),valid.recalibrated_in_basin.to_numpy(bool)
                    joint.append({'cutoff':cutoff,'policy':policy,'unit':unit_name,'formula_reference':reference,
                        'n_original_unit':len(unit),'n_calibrated_unit':len(valid),
                        'original_same_support':quadrant(old,match),'recalibrated':quadrant(new,match),
                        'bootstrap':paired_enrichment_bootstrap(old,new,match,rng,args.n_boot)})
            for source,unit in rescored_e.groupby('source',sort=True):
                valid = unit[unit.nearest_cell_calibrated]
                for reference in saved['analyses']:
                    match = valid[f'formula_match_{reference}'].to_numpy(bool)
                    joint.append({'cutoff':cutoff,'policy':policy,'unit':'external_'+source,'formula_reference':reference,
                        'n_original_unit':len(unit),'n_calibrated_unit':len(valid),
                        'original_same_support':quadrant(valid.in_basin,match),
                        'recalibrated':quadrant(valid.recalibrated_in_basin,match)})
            print(f'T={cutoff} {policy}: ICSD={hrate["recalibrated_rate"]:.6f}; empty cells={block["n_empty_cells"]}',flush=True)
        calibration.to_csv(out/'training_nearest_assignments.csv',index=False)
        cell_table.to_csv(out/'cell_calibration.csv',index=False)
    pd.DataFrame(rates).to_csv(args.output_dir/'rates.csv',index=False)
    (args.output_dir/'quadrants_and_enrichment.json').write_text(json.dumps(joint,indent=2,allow_nan=False)+'\n')
    report = {'status':'complete','purpose':'Sensitivity to the mismatch between Louvain-member and nearest-centroid-cell radius calibration in saved independently cutoff-trained maps.',
        'command':shlex.join([sys.executable]+sys.argv),'inputs':inputs,'script_sha256':sha(__file__),
        'protocol':{'cutoffs':args.cutoffs,'percentile':95,'n_boot':args.n_boot,
            'all_training':'Calibrate nearest-centroid radii from every cutoff training ICSD row, including original map noise.',
            'original_nonnoise_only':'Calibrate nearest-centroid radii using only originally nonnoise training rows, retaining the original calibration population.',
            'centroids_and_assignments':'Frozen fitted centers; training rows assigned with original producer top-three/exact-distance rule. Evaluation assignments and distances unchanged.',
            'empty_cell_policy':'No threshold imputation. Mark unsupported; report paired original/recalibrated rates on supported rows and full-denominator lower/upper bounds.',
            'formula_policy':'All four saved formula-reference flags unchanged. Per-formula units use the saved element-sorted normalized atomic-fraction identity and retain the earliest postcutoff record, selected before exclusion.',
            'limits':'Diagnostic chosen after inspecting the mismatch. Thresholds remain training-calibrated, but use of training data for calibration is not independent calibration validation. Bootstrap intervals condition on maps and radii; no new permutation tests or feature/map fitting are performed.'},
        'cutoffs':blocks,'rates':rates,'runtime_seconds':time.time()-started}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(pd.DataFrame(rates)[['cutoff','policy','source','n_calibrated','recalibrated_rate','ICSD_minus_source_pp']].to_string(index=False),flush=True)
    print('Completed; all provenance, original quadrant and independent distance checks passed.',flush=True)


if __name__ == '__main__':
    main()
