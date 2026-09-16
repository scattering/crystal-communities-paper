#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, csv, hashlib, json, math, sys, zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def b(v): return str(v).strip().lower() == 'true'

def f(v): return float(v)

def summarize(vals, flags):
    vals=np.asarray(vals,float); flags=np.asarray(flags,bool)
    return {'n':int(len(vals)),'mean_score':float(vals.mean()),'sd_score':float(vals.std(ddof=1)) if len(vals)>1 else None,
            'median_score':float(np.median(vals)),'min_score':float(vals.min()),'max_score':float(vals.max()),
            'n_in_basin':int(flags.sum()),'in_basin_rate':float(flags.mean())}

def bootstrap_ci(x,y,func,n=50000,seed=20260905):
    rng=np.random.default_rng(seed); out=np.empty(n)
    for i in range(n): out[i]=func(rng.choice(x,len(x),replace=True),rng.choice(y,len(y),replace=True))
    return [float(q) for q in np.quantile(out,[.025,.975])]

def probability_lower(x,y):
    x=np.asarray(x); y=np.asarray(y)
    return float(((x[:,None] < y[None,:]).sum()+.5*(x[:,None] == y[None,:]).sum())/(len(x)*len(y)))

def outcome_test(records, success=('made',), failure=('not_obtained',), seed=20260905):
    x=np.array([r['accessibility_score'] for r in records if r['corrected_outcome'] in success],float)
    y=np.array([r['accessibility_score'] for r in records if r['corrected_outcome'] in failure],float)
    xi=np.array([r['in_basin'] for r in records if r['corrected_outcome'] in success],bool)
    yi=np.array([r['in_basin'] for r in records if r['corrected_outcome'] in failure],bool)
    u1=stats.mannwhitneyu(x,y,alternative='less',method='exact')
    u2=stats.mannwhitneyu(x,y,alternative='two-sided',method='exact')
    wt1=stats.ttest_ind(x,y,equal_var=False,alternative='less')
    wt2=stats.ttest_ind(x,y,equal_var=False,alternative='two-sided')
    table=np.array([[xi.sum(),len(xi)-xi.sum()],[yi.sum(),len(yi)-yi.sum()]])
    fish1=stats.fisher_exact(table,alternative='greater')
    fish2=stats.fisher_exact(table,alternative='two-sided')
    rng=np.random.default_rng(seed); pooled=np.r_[x,y]; obs=float(x.mean()-y.mean()); nperm=100000
    perm=np.empty(nperm)
    for i in range(nperm):
        z=rng.permutation(pooled); perm[i]=z[:len(x)].mean()-z[len(x):].mean()
    return {'success_outcomes':list(success),'failure_outcomes':list(failure),
            'success':summarize(x,xi),'failure':summarize(y,yi),
            'mean_score_difference_success_minus_failure':obs,
            'mean_score_difference_bootstrap_95ci':bootstrap_ci(x,y,lambda a,c:a.mean()-c.mean(),seed=seed),
            'median_score_difference_success_minus_failure':float(np.median(x)-np.median(y)),
            'median_score_difference_bootstrap_95ci':bootstrap_ci(x,y,lambda a,c:np.median(a)-np.median(c),seed=seed+1),
            'probability_random_success_has_lower_score':probability_lower(x,y),
            'probability_lower_bootstrap_95ci':bootstrap_ci(x,y,probability_lower,n=20000,seed=seed+2),
            'mann_whitney_u_success_greater_count':float(u1.statistic),'mann_whitney_exact_p_one_sided_success_lower':float(u1.pvalue),
            'mann_whitney_exact_p_two_sided':float(u2.pvalue),
            'welch_t':float(wt1.statistic),'welch_p_one_sided_success_lower':float(wt1.pvalue),'welch_p_two_sided':float(wt2.pvalue),
            'mean_difference_permutation_p_one_sided_success_lower':float((1+(perm<=obs).sum())/(1+nperm)),
            'in_basin_table_rows_success_failure_cols_in_out':table.tolist(),
            'in_basin_risk_difference_success_minus_failure':float(xi.mean()-yi.mean()),
            'in_basin_risk_difference_bootstrap_95ci':bootstrap_ci(xi.astype(float),yi.astype(float),lambda a,c:a.mean()-c.mean(),seed=seed+3),
            'fisher_odds_ratio':float(fish1.statistic),'fisher_exact_p_one_sided_success_more_in_basin':float(fish1.pvalue),
            'fisher_exact_p_two_sided':float(fish2.pvalue), 'n_permutations':nperm,'bootstrap_seed':seed}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scripts',type=Path,required=True); ap.add_argument('--docs',type=Path,required=True)
    ap.add_argument('--source-summary',type=Path,required=True); ap.add_argument('--targets',type=Path,required=True)
    ap.add_argument('--basis',type=Path,required=True); ap.add_argument('--score-summary',type=Path,required=True)
    ap.add_argument('--refined-records',type=Path,required=True); ap.add_argument('--refined-pca',type=Path,required=True)
    ap.add_argument('--alab-zip',type=Path); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--n-jobs',type=int,default=16)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.scripts))
    from regenerate_external_projection import featurize, initialize_worker, project_to_communities
    from crystal_neighbors import FEATURE_VERSION, NEIGHBOR_SETTINGS
    from prepare_repaired_projection_basis import sha256_file
    from analyze_alab_validation import load_alab_targets, open_structure_from_zip
    docs=json.loads(args.docs.read_text()); source=json.loads(args.source_summary.read_text())
    targets=list(csv.DictReader(args.targets.open()))
    assert len(targets)==57 and len({r['mp_id'] for r in targets})==57 and set(docs)=={r['mp_id'] for r in targets}
    source_targets={r['mp_id']:r for r in source['targets']}; assert source['n_found']==57 and source['n_structures']==57
    records=[]
    for r in targets:
        d=docs[r['mp_id']]; st=Structure.from_dict(d['structure'])
        assert st.composition.reduced_composition==Composition(r['formula']).reduced_composition
        records.append({**r,'material_id':r['mp_id'],'reduced_formula':st.composition.reduced_formula,
                        'snapshot_spacegroup_number':(d.get('symmetry') or {}).get('number'),
                        'snapshot_spacegroup_symbol':(d.get('symmetry') or {}).get('symbol'),
                        'source_object':source_targets[r['mp_id']]['source_object'],'_kind':'pymatgen','_structure':d['structure']})
    with np.load(args.basis,allow_pickle=False) as z:
        basis={k:z[k] for k in z.files if k!='metadata'}; basis_meta=json.loads(str(z['metadata']))
    worker_hashes={name:sha256_file(args.scripts/name) for name in ('icsd_densify_worker.py','crystal_neighbors.py')}
    assert basis_meta['feature_version']==FEATURE_VERSION and basis_meta['neighbor_settings']==NEIGHBOR_SETTINGS
    assert basis_meta['worker_sha256']==worker_hashes and basis_meta['fit_transform_reproduces_saved_pca'] and basis_meta['row_ids_and_years_verified']
    initargs=('mp',str(args.docs),basis_meta['wl_iters'],None)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs,initializer=initialize_worker,initargs=initargs) as pool:
        results=list(pool.map(featurize,records,chunksize=2))
    failures=[{**p,**diag} for ok,p,v,diag in results if not ok]
    (args.out/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    assert len(failures)==0 and len(results)==57
    vectors=np.vstack([r[2] for r in results]); projected,nearest,distances=project_to_communities(vectors,basis)
    moments=json.loads(args.score_summary.read_text()); mu=float(moments['icsd_raw_mu']); sigma=float(moments['icsd_raw_sigma'])
    out=[]
    for i,((ok,pub,vec,diag),j,dist) in enumerate(zip(results,nearest,distances)):
        p50=float(basis['p50'][j]); p95=float(basis['p95'][j]); size=int(basis['counts'][j]); birth=int(basis['birth_years'][j]); age=max(2023-birth,0)
        raw=math.log1p(float(dist)/max(p50,1e-6))-.5*math.log1p(size)-.5*math.log1p(age)
        out.append({**pub,'feature_row':i,'assigned_community':int(basis['communities'][j]),'nearest_centroid_distance':float(dist),
                    'community_threshold_p95':p95,'community_median_distance':p50,'community_size':size,'community_birth_year':birth,
                    'in_basin':bool(dist<=p95),'outlier_like':bool(dist>p95),'accessibility_raw':raw,'accessibility_score':(raw-mu)/sigma,
                    'pca1':float(projected[i,0]),'pca2':float(projected[i,1]),
                    'ordered':bool(diag['ordered']),'sites_with_unrepresented_cn_mass':int(diag['sites_with_unrepresented_cn_mass']),
                    'max_unrepresented_cn_mass':float(diag['max_unrepresented_cn_mass'])})
    fields=[]
    for r in out:
        for k in r:
            if k not in fields: fields.append(k)
    with (args.out/'target_projection_records.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(out)
    np.save(args.out/'features.npy',vectors); np.save(args.out/'features_pca.npy',projected)
    groups={}
    for name in sorted({r['corrected_outcome'] for r in out}):
        rr=[r for r in out if r['corrected_outcome']==name]; groups[name]=summarize([r['accessibility_score'] for r in rr],[r['in_basin'] for r in rr])
    primary=outcome_test(out,('made',),('not_obtained',))
    physical=outcome_test(out,('made','offline_recovery'),('not_obtained',),seed=20260906)
    conservative=outcome_test(out,('made',),('not_obtained','inconclusive'),seed=20260907)
    # Paired target-vs-refined projection sensitivity for all 42 released refinements.
    ref=list(csv.DictReader(args.refined_records.open())); refp=np.load(args.refined_pca)
    assert len(ref)==len(refp)==42; target_by_mp={r['mp_id']:r for r in out}
    paired=[]
    for r,xp in zip(ref,refp):
        t=target_by_mp[r['mp_id']]; refscore=float(r['accessibility_score'])
        j=np.where(basis['communities']==int(r['assigned_community']))[0]; assert len(j)==1; j=int(j[0])
        raw=math.log1p(float(r['nearest_centroid_distance'])/max(float(basis['p50'][j]),1e-6))-.5*math.log1p(int(basis['counts'][j]))-.5*math.log1p(max(2023-int(basis['birth_years'][j]),0))
        recalc=(raw-mu)/sigma
        paired.append({'mp_id':r['mp_id'],'formula':r['formula'],'corrected_outcome':r['corrected_outcome'],
                       'target_community':t['assigned_community'],'refined_community':int(r['assigned_community']),
                       'same_community':t['assigned_community']==int(r['assigned_community']),
                       'target_in_basin':t['in_basin'],'refined_in_basin':b(r['in_basin']),'in_basin_agreement':t['in_basin']==b(r['in_basin']),
                       'target_score':t['accessibility_score'],'refined_score':refscore,'score_target_minus_refined':t['accessibility_score']-refscore,
                       'target_refined_pca_distance':float(np.linalg.norm(projected[t['feature_row']]-xp)),
                       'refined_score_recompute_abs_error':abs(recalc-refscore)})
    # Optional atomic StructureMatcher against public released refinement CIFs.
    matcher={'attempted':0,'matches':0,'failures':[]}
    if args.alab_zip and args.alab_zip.is_file():
        byformula={r.formula:r for r in load_alab_targets(args.alab_zip)}
        for p in paired:
            a=byformula[p['formula']]; matcher['attempted']+=1
            try:
                s1=Structure.from_dict(docs[p['mp_id']]['structure']); s2=open_structure_from_zip(args.alab_zip,a.cif_member)
                sm=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
                p['structure_matcher_fit']=bool(sm.fit(s1,s2)); matcher['matches']+=int(p['structure_matcher_fit'])
            except Exception as exc:
                p['structure_matcher_fit']=None; matcher['failures'].append({'mp_id':p['mp_id'],'reason':type(exc).__name__,'detail':str(exc)[:200]})
    pfields=list(paired[0])
    with (args.out/'target_vs_refinement_records.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=pfields); w.writeheader(); w.writerows(paired)
    ts=np.array([p['target_score'] for p in paired]); rs=np.array([p['refined_score'] for p in paired]); delta=ts-rs
    pear=stats.pearsonr(ts,rs); spear=stats.spearmanr(ts,rs); wil=stats.wilcoxon(delta,alternative='two-sided')
    pair_summary={'n':len(paired),'same_community_n':sum(p['same_community'] for p in paired),'same_community_rate':float(np.mean([p['same_community'] for p in paired])),
                  'in_basin_agreement_n':sum(p['in_basin_agreement'] for p in paired),'in_basin_agreement_rate':float(np.mean([p['in_basin_agreement'] for p in paired])),
                  'target_in_basin_rate':float(np.mean([p['target_in_basin'] for p in paired])),'refined_in_basin_rate':float(np.mean([p['refined_in_basin'] for p in paired])),
                  'mean_score_target_minus_refined':float(delta.mean()),'median_score_target_minus_refined':float(np.median(delta)),
                  'mean_absolute_score_difference':float(np.abs(delta).mean()),'max_absolute_score_difference':float(np.abs(delta).max()),
                  'pearson_r':float(pear.statistic),'pearson_p':float(pear.pvalue),'spearman_rho':float(spear.statistic),'spearman_p':float(spear.pvalue),
                  'paired_wilcoxon_statistic':float(wil.statistic),'paired_wilcoxon_p_two_sided':float(wil.pvalue),
                  'max_refined_score_recompute_abs_error':max(p['refined_score_recompute_abs_error'] for p in paired),
                  'structure_matcher':matcher,'by_outcome':{}}
    for name in sorted({p['corrected_outcome'] for p in paired}):
        pp=[p for p in paired if p['corrected_outcome']==name]; dd=np.array([p['score_target_minus_refined'] for p in pp])
        pair_summary['by_outcome'][name]={'n':len(pp),'same_community_rate':float(np.mean([p['same_community'] for p in pp])),
            'in_basin_agreement_rate':float(np.mean([p['in_basin_agreement'] for p in pp])),
            'mean_score_target_minus_refined':float(dd.mean()),'median_score_target_minus_refined':float(np.median(dd))}
    summary={'scope':'All 57 pre-experiment MP target structures from the exact MP 2022-10-28 public snapshot, projected with the repaired production encoder and frozen ICSD basis.',
             'observation_year':2023,'outcome_counts':dict(Counter(r['corrected_outcome'] for r in out)),'outcome_descriptives':groups,
             'primary_made_vs_not_obtained':primary,'sensitivity_made_plus_offline_vs_not_obtained':physical,
             'sensitivity_made_vs_not_obtained_plus_inconclusive':conservative,'target_vs_refinement':pair_summary,
             'provenance':{'docs_sha256':sha256(args.docs),'source_summary_sha256':sha256(args.source_summary),'targets_sha256':sha256(args.targets),
                'basis_sha256':sha256(args.basis),'score_summary_sha256':sha256(args.score_summary),'refined_records_sha256':sha256(args.refined_records),
                'refined_pca_sha256':sha256(args.refined_pca),'alab_zip_sha256':sha256(args.alab_zip) if args.alab_zip and args.alab_zip.is_file() else None,
                'feature_version':FEATURE_VERSION,'neighbor_settings':NEIGHBOR_SETTINGS,'worker_sha256':worker_hashes,'basis_metadata':basis_meta,
                'script_sha256':sha256(Path(__file__))}}
    (args.out/'target_projection_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'outcome_descriptives':groups,'primary':primary,'target_vs_refinement':pair_summary},indent=2,allow_nan=False))

if __name__=='__main__': main()
