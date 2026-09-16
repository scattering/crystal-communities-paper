#!/usr/bin/env python3
"""Independent arithmetic validation of the retrieved cutoff calibration outputs."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'cutoff_trained'
NEW = HERE / 'results'
checks = 0
scored_rows = 0
training_rows = 0
cell_checks = 0


def ck(condition, description):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(description)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(frame, inside, reference):
    observed = pd.crosstab(frame[inside].astype(bool), frame[reference].astype(bool)).reindex(
        index=[False, True], columns=[False, True], fill_value=0).to_numpy()
    n00,n01,n10,n11 = map(int, observed.ravel())
    n = n00+n01+n10+n11
    denominator = (n10+n11)*(n01+n11)
    return {'n':n,'in_basin_and_match':n11,'in_basin_and_no_match':n10,
        'frontier_and_match':n01,'frontier_and_no_match':n00,
        'p_in_basin':(n10+n11)/n if n else None,
        'p_match':(n01+n11)/n if n else None,
        'enrichment_ratio_obs_over_independence':n*n11/denominator if denominator else None}


report = json.loads((NEW/'report.json').read_text())
original = json.loads((OLD/'cutoff_trained_retrospective_summary.json').read_text())
ck(report['status']=='complete','producer completed')
refresh = report['formula_refresh']
ck(refresh['status']=='complete','formula refresh completed')
ck(refresh['structural_producer_script_sha256']==report['script_sha256'],
   'original structural producer hash retained')
ck(refresh['refresh_script_sha256']==digest(HERE/'refresh_formula_results.py'),
   'exact formula-refresh script')
ck(refresh['cutoff_summary_sha256']==digest(OLD/'cutoff_trained_retrospective_summary.json'),
   'final cutoff summary pinned')
ck(refresh['cutoff_reporting_manifest_sha256']==digest(OLD/'reporting/reporting_manifest.json'),
   'final cutoff reporting manifest pinned')
ck(refresh['formula_identity']=='element-sorted normalized atomic fractions rounded to 12 decimal places',
   'declared normalized formula identity')
rates = pd.read_csv(NEW/'rates.csv')
joint = json.loads((NEW/'quadrants_and_enrichment.json').read_text())
bootstrap_checks = []
for cutoff in [1990,2000,2010]:
    saved = original['cutoffs'][str(cutoff)]
    for value in saved['outputs'].values():
        ck(digest(OLD/value['path'])==value['sha256'],'original packaged file '+value['path'])
    old_train = pd.read_csv(OLD/f'T{cutoff}/training_partition.csv',keep_default_na=False)
    train = pd.read_csv(NEW/f'T{cutoff}/training_nearest_assignments.csv',keep_default_na=False)
    pd.testing.assert_frame_equal(train[old_train.columns],old_train)
    checks += 1
    training_rows += len(train)
    cells = pd.read_csv(NEW/f'T{cutoff}/cell_calibration.csv').set_index('community')
    ck(set(train.nearest_community).issubset(set(cells.index)),'valid training nearest communities')
    for policy in ['all_training','original_nonnoise_only']:
        mask = np.ones(len(train),bool) if policy=='all_training' else train.cutoff_community.to_numpy() >= 0
        grouped = train[mask].groupby('nearest_community').nearest_centroid_distance
        for community,row in cells.iterrows():
            values = np.sort(grouped.get_group(community).to_numpy()) if community in grouped.groups else np.array([])
            ck(len(values)==row[policy+'_calibration_count'],'cell count')
            if len(values):
                location = .95*(len(values)-1)
                lower,upper = int(np.floor(location)),int(np.ceil(location))
                independent = values[lower]+(location-lower)*(values[upper]-values[lower])
                ck(np.isclose(independent,row[policy+'_radius_p95'],rtol=1e-12,atol=1e-12),'independent sorted/interpolated p95')
            else:
                ck(np.isnan(row[policy+'_radius_p95']),'empty cell not imputed')
            cell_checks += 1
        old_tables = {
            'ICSD':pd.read_csv(OLD/f'T{cutoff}/heldout_entries.csv',keep_default_na=False),
            'external':pd.read_csv(OLD/f'T{cutoff}/external_classifications.csv',keep_default_na=False)}
        new_tables = {}
        for kind,old in old_tables.items():
            name = 'heldout_entries.csv' if kind=='ICSD' else 'external_classifications.csv'
            new = pd.read_csv(NEW/f'T{cutoff}/{policy}/{name}',keep_default_na=False)
            pd.testing.assert_frame_equal(new[old.columns],old,check_exact=False,rtol=1e-12,atol=1e-12)
            checks += 1
            expected_threshold = new.assigned_community.map(cells[policy+'_radius_p95']).to_numpy()
            ck(np.isfinite(expected_threshold).all(),'no unsupported evaluation rows')
            ck(new.nearest_cell_calibrated.to_numpy(bool).all(),'all evaluations marked supported')
            ck(np.allclose(expected_threshold,new.nearest_cell_threshold_p95,rtol=1e-12,atol=1e-12),'all per-record thresholds')
            expected = new.nearest_centroid_distance.to_numpy() <= expected_threshold
            ck(np.array_equal(expected,new.recalibrated_in_basin.to_numpy(bool)),'all per-record flags')
            scored_rows += len(new)
            new_tables[kind] = new
        sources = {'ICSD':new_tables['ICSD']}
        sources.update(dict(tuple(new_tables['external'].groupby('source',sort=True))))
        for source,frame in sources.items():
            rate = rates[(rates.cutoff==cutoff)&(rates.policy==policy)&(rates.source==source)].iloc[0]
            a,b = frame.in_basin.to_numpy(bool),frame.recalibrated_in_basin.to_numpy(bool)
            for key,value in {'n_total':len(frame),'n_calibrated':len(frame),'n_uncalibrated':0,
                'original_all_n_in_basin':int(a.sum()),'original_all_rate':float(a.mean()),
                'recalibrated_n_in_basin':int(b.sum()),'recalibrated_rate':float(b.mean()),
                'out_to_in':int((~a&b).sum()),'in_to_out':int((a&~b).sum())}.items():
                ck(np.isclose(value,rate[key],rtol=1e-12,atol=1e-12),'rate '+source+' '+key)
            if source!='ICSD':
                gap = 100*(sources['ICSD'].recalibrated_in_basin.mean()-b.mean())
                ck(np.isclose(gap,rate.ICSD_minus_source_pp,rtol=1e-12,atol=1e-12),'source gap')
        held = sources['ICSD'][sources['ICSD'].formula!='']
        assert held.formula_identity.ne('').all()
        units = {'per_entry':held,'per_formula':held.sort_values(['year','icsd_id'],kind='stable').drop_duplicates('formula_identity',keep='first')}
        units.update({'external_'+k:v for k,v in sources.items() if k!='ICSD'})
        for block in [b for b in joint if b['cutoff']==cutoff and b['policy']==policy]:
            frame = units[block['unit']]
            reference = 'formula_match_'+block['formula_reference']
            for result,column in [('original_same_support','in_basin'),('recalibrated','recalibrated_in_basin')]:
                q = table(frame,column,reference)
                for key,value in q.items():
                    target = block[result][key]
                    ck((value is None and target is None) or (value is not None and target is not None and np.isclose(value,target,rtol=1e-12,atol=1e-12)), 'quadrant '+key)
            if block['formula_reference']=='all_year_le_T_index' and block['unit'] in ['per_entry','per_formula']:
                # Fresh bootstrap from the independently reconstructed 2x2 table.
                q = table(frame,'recalibrated_in_basin',reference)
                counts = np.array([q['frontier_and_no_match'],q['frontier_and_match'],q['in_basin_and_no_match'],q['in_basin_and_match']])
                n = int(counts.sum())
                draw = np.random.default_rng(cutoff+len(frame)).multinomial(n,counts/n,size=2000)
                ratio = n*draw[:,3]/((draw[:,2]+draw[:,3])*(draw[:,1]+draw[:,3]))
                interval = np.quantile(ratio,[.025,.975])
                ck(interval[0]>1,'independent bootstrap retains positive primary enrichment')
                bootstrap_checks.append({'cutoff':cutoff,'policy':policy,'unit':block['unit'],
                    'independent_ci95':interval.tolist(),'producer_ci95':block['bootstrap']['recalibrated_enrichment_ci95']})

result = {'status':'passed','checks':checks,'training_rows_verified':training_rows,
    'calibrated_community_policy_pairs':cell_checks,'evaluation_rows_verified_across_policies':scored_rows,
    'quadrant_blocks_verified':len(joint),'independent_primary_enrichment_bootstraps':bootstrap_checks,
    'script_sha256':digest(Path(__file__)),
    'formula_refresh_script_sha256':digest(HERE/'refresh_formula_results.py'),
    'limitations':'Local verification starts from retrieved training nearest-community/distance rows. The raw213-D matrix remains remote; the producer independently reconstructed original centroids/radii and checked sampled raw training/heldout projections. This local audit independently verifies p95 estimation, coverage, unchanged evaluation inputs, every new flag/rate/gap/quadrant, and fresh conditional primary-enrichment bootstrap intervals.'}
(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['independent_primary_enrichment_bootstraps','limitations']},indent=2))
