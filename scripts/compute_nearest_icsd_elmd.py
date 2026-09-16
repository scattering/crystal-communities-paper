#!/usr/bin/env python3
"""Exact nearest-ICSD ElMD (modified Pettifor 1-D EMD), recovered original producer."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from numba import njit, prange, get_num_threads

ROOT=Path(__file__).resolve().parents[1]
ICSD_INDEX = COHORT_ROOT = OUT = LOOKUP = None

@njit(parallel=True, cache=True)
def nearest_sparse(qpos,qmass,qptr,rpos,rmass,rptr):
    """Exact L1 NN between cumulative distribution vectors, with monotone pruning."""
    nq=qptr.size-1; nr=rptr.size-1
    outd=np.empty(nq,np.float64); outi=np.empty(nq,np.int64)
    for qi in prange(nq):
        qs=qptr[qi]; qe=qptr[qi+1]
        best=1e300; besti=-1
        for ri in range(nr):
            i=qs; j=rptr[ri]; re=rptr[ri+1]
            cumulative_delta=0.0; previous_rank=0; cost=0.0
            while i<qe or j<re:
                if j>=re or (i<qe and qpos[i]<rpos[j]):
                    rank=qpos[i]
                elif i>=qe or rpos[j]<qpos[i]:
                    rank=rpos[j]
                else:
                    rank=qpos[i]
                if rank!=previous_rank:
                    cost += abs(cumulative_delta)*(rank-previous_rank)
                    if cost>=best: # remaining increments are nonnegative
                        break
                    previous_rank=rank
                if i<qe and qpos[i]==rank:
                    cumulative_delta += qmass[i]; i += 1
                if j<re and rpos[j]==rank:
                    cumulative_delta -= rmass[j]; j += 1
            if cost<best:
                best=cost; besti=ri
        outd[qi]=best; outi[qi]=besti
    return outd,outi

def formula_to_sparse(s):
    comp=Composition(str(s)).element_composition
    by_rank={}
    total=0.0
    for el, amt in comp.items():
        a=float(amt)
        if not np.isfinite(a) or a<0:
            raise ValueError(f'nonfinite/negative amount in {s!r}')
        if a==0: continue
        sym=el.symbol
        if sym not in LOOKUP:
            raise KeyError(f'{sym} missing from ElMD modified-Pettifor lookup')
        rank=int(LOOKUP[sym])
        by_rank[rank]=by_rank.get(rank,0.0)+a
        total += a
    if total<=0: raise ValueError(f'empty composition {s!r}')
    ranks=np.array(sorted(by_rank),dtype=np.int16)
    mass=np.array([by_rank[int(k)]/total for k in ranks],dtype=np.float64)
    return ranks,mass

def build_csr(formulas):
    pos=[]; mass=[]; ptr=[0]; ok=[]; errors=[]
    for idx,s in enumerate(formulas):
        try:
            p,m=formula_to_sparse(s); pos.extend(p); mass.extend(m); ptr.append(len(pos)); ok.append(idx)
        except Exception as exc:
            errors.append((idx,str(s),repr(exc)))
    return (np.asarray(pos,np.int16),np.asarray(mass,np.float64),np.asarray(ptr,np.int64),
            np.asarray(ok,np.int64),errors)

def dense_distance(p1,m1,p2,m2):
    v1=np.zeros(103);v2=np.zeros(103);v1[p1]=m1;v2[p2]=m2
    return float(np.abs(np.cumsum(v1-v2)).sum())

def load_cohorts():
    out=[]
    for src in ['gnome','mp','jarvis','alexandria']:
        p=COHORT_ROOT/f'{src}/attempted_cohort.csv'
        d=pd.read_csv(p)
        out.append(d[['material_id','reduced_formula']].assign(source=src))
    p=COHORT_ROOT/'mattergen/mattergen-public_frontier_records.csv'
    d=pd.read_csv(p)
    out.append(d[['material_id','reduced_formula']].assign(source='mattergen'))
    return pd.concat(out,ignore_index=True)[['source','material_id','reduced_formula']]

def run_reference(label, refdf, qdf, qp,qm,qptr):
    t=time.time(); rp,rm,rptr,ok,errors=build_csr(refdf.name.astype(str).to_numpy())
    r=refdf.iloc[ok].reset_index(drop=True)
    parse_s=time.time()-t
    t=time.time(); dist,idx=nearest_sparse(qp,qm,qptr,rp,rm,rptr); nn_s=time.time()-t
    hit=r.iloc[idx].reset_index(drop=True)
    result=qdf.copy()
    result[f'{label}_nearest_elmd']=dist
    result[f'{label}_nearest_icsd_formula']=hit['name'].astype(str)
    result[f'{label}_nearest_icsd_id']=hit['cif_names'].to_numpy()
    result[f'{label}_nearest_icsd_year']=hit['publication_year'].to_numpy()
    meta={'label':label,'reference_input_rows':int(len(refdf)),'reference_parseable_rows':int(len(r)),
          'reference_parse_errors':len(errors),'parse_seconds':parse_s,'nearest_seconds':nn_s,
          'query_rows':int(len(qdf)),'numba_threads':int(get_num_threads())}
    return result,meta,errors,(rp,rm,rptr,r)

def summaries(df, labels):
    records=[]
    for src,g in df.groupby('source',sort=False):
      for label in labels:
        x=g[f'{label}_nearest_elmd'].to_numpy(float)
        records.append({'source':src,'reference':label,'n':len(x),'zero_n':int(np.count_nonzero(np.isclose(x,0,atol=1e-12))),
          'zero_pct':100*float(np.mean(np.isclose(x,0,atol=1e-12))), 'mean':float(x.mean()),
          'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),'p25':float(np.quantile(x,.25)),
          'p75':float(np.quantile(x,.75)),'p90':float(np.quantile(x,.9)),'p95':float(np.quantile(x,.95)),
          'maximum':float(x.max())})
    return pd.DataFrame(records)

def main():
    configure()
    wall=time.time()
    ref=pd.read_csv(ICSD_INDEX)
    ref['publication_year']=pd.to_numeric(ref['publication_year'],errors='coerce')
    q=load_cohorts()
    qp,qm,qptr,qok,qerr=build_csr(q.reduced_formula.astype(str).to_numpy())
    if qerr: print('query errors',qerr[:10])
    q=q.iloc[qok].reset_index(drop=True)
    all_result,all_meta,all_err,all_arrays=run_reference('all_icsd',ref,q,qp,qm,qptr)
    post=ref[(ref.publication_year>1980)&(ref.publication_year<=2015)].copy()
    post_result,post_meta,post_err,post_arrays=run_reference('post1980_through2015_icsd',post,q,qp,qm,qptr)
    result=all_result.copy()
    for c in post_result.columns:
      if c not in ['source','material_id','reduced_formula']: result[c]=post_result[c]
    # Exact dense-CDF validation on 500 random pairs against the same sparse vectors.
    rng=np.random.default_rng(941); errs=[]
    rp,rm,rptr,rdf=all_arrays
    for _ in range(500):
      qi=int(rng.integers(len(q)));ri=int(rng.integers(len(rdf)))
      qs,qe=qptr[qi:qi+2];rs,re=rptr[ri:ri+2]
      expected=dense_distance(qp[qs:qe],qm[qs:qe],rp[rs:re],rm[rs:re])
      # single-reference exact kernel, exercised independently from production block
      got=nearest_sparse(qp[qs:qe],qm[qs:qe],np.array([0,qe-qs],dtype=np.int64),rp[rs:re],rm[rs:re],np.array([0,re-rs],dtype=np.int64))[0][0]
      errs.append(abs(expected-got))
    validation={'pairs':len(errs),'max_abs_error_sparse_vs_dense_cdf':float(max(errs)),
      'mean_abs_error_sparse_vs_dense_cdf':float(np.mean(errs))}
    result.to_csv(OUT/'nearest_icsd_elmd_records.csv',index=False)
    summ=summaries(result,['all_icsd','post1980_through2015_icsd'])
    summ.to_csv(OUT/'nearest_icsd_elmd_summary.csv',index=False)
    metadata={'metric':'Element Movers Distance using modified Pettifor scale (default ElMD metric)',
      'identity':'For this 1-D elemental scale, ElMD is exactly the L1 distance between cumulative normalized-composition vectors.',
      'algorithm':'Sparse union-rank sweep, exhaustive over every reference composition, with exact monotone lower-bound early exit.',
      'query_parse_errors':qerr,'all_reference':all_meta,'post1980_through2015_reference':post_meta,
      'all_reference_parse_errors_first20':all_err[:20], 'post_reference_parse_errors_first20':post_err[:20],
      'validation':validation,'wall_seconds':time.time()-wall}
    json.dump(metadata,open(OUT/'nearest_icsd_elmd_metadata.json','w'),indent=2)
    print(summ.to_string(index=False));print(json.dumps(metadata,indent=2))


def configure(argv=None):
    """Configure input paths without changing the recovered numerical algorithm."""
    import argparse
    import importlib.util
    from numba import set_num_threads
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--icsd-index', type=Path, default=ROOT / 'notes/feature_repair_2026_09/downstream/inputs/ICSD_index.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--elmd-lookup', type=Path, help='ElMD 0.5.15 el_lookup/mod_petti.json; defaults to the installed ElMD package')
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--cohort-root', type=Path, default=ROOT / 'notes/feature_repair_2026_09/downstream/external')
    args = parser.parse_args(argv)
    lookup_path = args.elmd_lookup
    if lookup_path is None:
        spec = importlib.util.find_spec('ElMD')
        if spec is None or not spec.submodule_search_locations:
            parser.error('Install ElMD==0.5.15 or provide --elmd-lookup')
        lookup_path = Path(next(iter(spec.submodule_search_locations))) / 'el_lookup/mod_petti.json'
    global ICSD_INDEX, OUT, LOOKUP, COHORT_ROOT
    ICSD_INDEX, OUT = args.icsd_index, args.output_dir
    COHORT_ROOT = args.cohort_root
    LOOKUP = json.loads(lookup_path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    set_num_threads(args.threads)

if __name__=='__main__': main()
