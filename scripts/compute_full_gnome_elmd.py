#!/usr/bin/env python3
"""Exact full-release GNoME nearest-ICSD ElMD with original projection bounds."""
from __future__ import annotations
import csv,gzip,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
from pymatgen.core import Composition
from numba import njit,prange,get_num_threads
ROOT=Path(__file__).resolve().parents[1]
ICSD_INDEX = SRC = OUT = LOOKUP = None

@njit(cache=True)
def pair_cost(qp,qm,qs,qe,rp,rm,rs,re,cutoff):
 i=qs;j=rs;cum=0.0;prev=0;cost=0.0
 while i<qe or j<re:
  if j>=re or (i<qe and qp[i]<rp[j]):rank=qp[i]
  elif i>=qe or rp[j]<qp[i]:rank=rp[j]
  else:rank=qp[i]
  if rank!=prev:
   cost += abs(cum)*(rank-prev)
   if cost>cutoff:return cost
   prev=rank
  if i<qe and qp[i]==rank:cum+=qm[i];i+=1
  if j<re and rp[j]==rank:cum-=rm[j];j+=1
 return cost

@njit(parallel=True,cache=True)
def brute_nearest(qp,qm,qptr,rp,rm,rptr):
 nq=len(qptr)-1;nr=len(rptr)-1;od=np.empty(nq);oi=np.empty(nq,np.int64)
 for qi in prange(nq):
  best=1e300;bi=-1
  for ri in range(nr):
   v=pair_cost(qp,qm,qptr[qi],qptr[qi+1],rp,rm,rptr[ri],rptr[ri+1],best)
   if v<best:best=v;bi=ri
  od[qi]=best;oi[qi]=bi
 return od,oi

@njit(parallel=True,cache=True)
def make_projections(pos,mass,ptr,F):
 n=len(ptr)-1;P=F.shape[0];out=np.empty((P,n))
 for i in prange(n):
  for z in range(P):
   v=0.0
   for j in range(ptr[i],ptr[i+1]):v += mass[j]*F[z,pos[j]]
   out[z,i]=v
 return out

@njit(parallel=True,cache=True)
def indexed_nearest(qp,qm,qptr,qproj,rp,rm,rptr,rproj,orders,sorted_values,K=4):
 """Exact NN: every projection is 1-Lipschitz, hence |E f_q-E f_r| <= W1."""
 nq=len(qptr)-1;nr=len(rptr)-1;P=orders.shape[0]
 od=np.empty(nq);oi=np.empty(nq,np.int64);tested=np.empty(nq,np.int64);exact=np.empty(nq,np.int64)
 for qi in prange(nq):
  best=1e300;bi=-1;nt=0;ne=0
  # Obtain a valid upper bound from nearby points in each scalar projection.
  for z in range(P):
   k=np.searchsorted(sorted_values[z],qproj[z,qi]);lo=max(0,k-K);hi=min(nr,k+K)
   for jj in range(lo,hi):
    ri=orders[z,jj]
    v=pair_cost(qp,qm,qptr[qi],qptr[qi+1],rp,rm,rptr[ri],rptr[ri+1],best);ne+=1
    if v<best:best=v;bi=ri
  # Any point closer than best must lie in every projection interval. Iterate the smallest.
  bz=0;blo=0;bhi=nr
  for z in range(P):
   lo=np.searchsorted(sorted_values[z],qproj[z,qi]-best)
   hi=np.searchsorted(sorted_values[z],qproj[z,qi]+best,side='right')
   if hi-lo<bhi-blo:bz=z;blo=lo;bhi=hi
  for jj in range(blo,bhi):
   ri=orders[bz,jj];nt+=1;lb=0.0
   for z in range(P):
    a=abs(qproj[z,qi]-rproj[z,ri])
    if a>lb:lb=a
   if lb>best:continue
   v=pair_cost(qp,qm,qptr[qi],qptr[qi+1],rp,rm,rptr[ri],rptr[ri+1],best);ne+=1
   if v<best:best=v;bi=ri
  od[qi]=best;oi[qi]=bi;tested[qi]=nt;exact[qi]=ne
 return od,oi,tested,exact

def sparse_formula(s):
 c=Composition(str(s)).element_composition;d={};tot=0.0
 for el,a0 in c.items():
  a=float(a0)
  if not np.isfinite(a) or a<0:raise ValueError(s)
  if a==0:continue
  rank=int(LOOKUP[el.symbol]);d[rank]=d.get(rank,0.0)+a;tot+=a
 if tot<=0:raise ValueError(s)
 p=np.array(sorted(d),np.int16);m=np.array([d[int(x)]/tot for x in p],np.float64)
 return p,m

def load_queries():
 ids=[];forms=[];strata=[];pos=[];mass=[];ptr=[0];fails=[];collisions=0
 with SRC.open(newline='',encoding='utf-8',errors='replace') as f:
  for line,r in enumerate(csv.DictReader(f),start=2):
   mid=(r.get('MaterialId') or '').strip();form=(r.get('Reduced Formula') or '').strip();raw=r.get('Is Train')
   st='true' if raw=='True' else ('false' if raw=='False' else 'blank')
   try:
    c=Composition(form).element_composition
    ranks=[int(LOOKUP[e.symbol]) for e in c]
    if len(ranks)!=len(set(ranks)):collisions+=1
    p,m=sparse_formula(form)
   except Exception as e:fails.append({'line':line,'material_id':mid,'formula':form,'error':repr(e)});continue
   ids.append(mid);forms.append(form);strata.append(st);pos.extend(p);mass.extend(m);ptr.append(len(pos))
   if len(ids)%100000==0:print('parsed queries',len(ids),flush=True)
 return ids,forms,strata,np.array(pos,np.int16),np.array(mass),np.array(ptr,np.int64),fails,collisions

def load_reference(mask):
 d=pd.read_csv(ICSD_INDEX)
 d.publication_year=pd.to_numeric(d.publication_year,errors='coerce');d=d.loc[mask(d)].reset_index(drop=True)
 seen={};pos=[];mass=[];ptr=[0];rows=[];fails=[];collisions=0
 for i,r in d.iterrows():
  form=str(r['name'])
  try:
   c=Composition(form).element_composition;ranks=[int(LOOKUP[e.symbol]) for e in c]
   if len(ranks)!=len(set(ranks)):collisions+=1
   p,m=sparse_formula(form)
  except Exception as e:fails.append({'row':int(i),'formula':form,'error':repr(e)});continue
  key=(tuple(map(int,p)),tuple(np.round(m,14)))
  if key in seen:continue
  seen[key]=len(rows);pos.extend(p);mass.extend(m);ptr.append(len(pos));rows.append((form,r['cif_names'],r['publication_year']))
 meta=pd.DataFrame(rows,columns=['formula','cif_id','year'])
 return np.array(pos,np.int16),np.array(mass),np.array(ptr,np.int64),meta,fails,collisions,len(d)

def funcs():
 x=np.arange(103,dtype=float);fs=[x]
 for t in [10,25,40,55,70,85,100]:fs.append(np.abs(x-t))
 rng=np.random.default_rng(20260905)
 for _ in range(8):fs.append(np.r_[0.0,np.cumsum(rng.choice(np.array([-1.,1.]),102))])
 F=np.array(fs)
 assert np.max(np.abs(np.diff(F,axis=1)))<=1+1e-15
 return F

def one_ref(label,qp,qm,qptr,qproj,ref):
 rp,rm,rptr,meta,fails,collisions,input_n=ref
 F=funcs();t=time.time();rproj=make_projections(rp,rm,rptr,F)
 orders=np.argsort(rproj,axis=1).astype(np.int32);svals=np.take_along_axis(rproj,orders,axis=1)
 d,i,tested,exact=indexed_nearest(qp,qm,qptr,qproj,rp,rm,rptr,rproj,orders,svals);sec=time.time()-t
 return {'d':d,'i':i,'tested':tested,'exact':exact,'meta':meta}, {'label':label,'reference_input_n':input_n,'reference_unique_elmd_vectors':len(meta),'parse_failures':fails,'rank_collision_formula_count_before_dedup':collisions,'seconds':sec,'candidate_interval_quantiles':{str(q):float(np.quantile(tested,q)) for q in [0,.5,.9,.99,1]},'exact_evaluation_quantiles':{str(q):float(np.quantile(exact,q)) for q in [0,.5,.9,.99,1]}}

def summarize(ids,strata,results):
 out={}
 for st in ['pooled','true','false','blank']:
  mask=np.ones(len(ids),bool) if st=='pooled' else np.array(strata)==st;q={'n':int(mask.sum()),'references':{}}
  for label,z in results.items():
   x=z['d'][mask]
   q['references'][label]={'zero_count':int(np.sum(x<=1e-12)),'zero_rate':float(np.mean(x<=1e-12)),'mean':float(x.mean()),
    'quantiles':{str(v):float(np.quantile(x,v)) for v in [0,.1,.25,.5,.75,.9,.95,.99,1]},
    'ecdf':{str(t):float(np.mean(x<=t+1e-12)) for t in [.1,.25,.5,1,2,3,5]}}
  out[st]=q
 return out

def main():
 configure()
 wall=time.time();ids,forms,strata,qp,qm,qptr,qfails,qcoll=load_queries();F=funcs();qproj=make_projections(qp,qm,qptr,F)
 allref=load_reference(lambda d:np.ones(len(d),bool));postref=load_reference(lambda d:(d.publication_year>1980)&(d.publication_year<=2015))
 results={};metas={}
 for label,ref in [('all_index',allref),('post1980_through2015',postref)]:
  z,m=one_ref(label,qp,qm,qptr,qproj,ref);results[label]=z;metas[label]=m;print(label,m,flush=True)
 # Exact brute-force validation on random 250 queries against both deduplicated reference sets.
 rng=np.random.default_rng(20260905);sel=np.sort(rng.choice(len(ids),250,replace=False));validation={}
 sp=[];sm=[];si=[0]
 for qi in sel:sp.extend(qp[qptr[qi]:qptr[qi+1]]);sm.extend(qm[qptr[qi]:qptr[qi+1]]);si.append(len(sp))
 sp=np.array(sp,np.int16);sm=np.array(sm);si=np.array(si,np.int64)
 for label,ref in [('all_index',allref),('post1980_through2015',postref)]:
  bd,bi=brute_nearest(sp,sm,si,ref[0],ref[1],ref[2]);got=results[label]['d'][sel]
  validation[label]={'n':len(sel),'max_abs_distance_error':float(np.max(np.abs(bd-got))),'all_within_1e-12':bool(np.all(np.abs(bd-got)<=1e-12))}
 summary={'metric':'ElMD 0.5.15 default modified-Pettifor metric','algorithm':'exact exhaustive nearest neighbor with necessary 1-Lipschitz projection bounds; no approximate search','source':{'path':str(SRC),'sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'records':len(ids),'parse_failures':qfails,'within_formula_rank_collisions':qcoll,'is_train_counts':{x:int(strata.count(x)) for x in ['true','false','blank']}},'references':metas,'validation':validation,'strata':summarize(ids,strata,results),'numba_threads':get_num_threads(),'wall_seconds':None}
 # Per-record public-source output.
 with gzip.open(OUT/'full_gnome_nearest_icsd_elmd_records.csv.gz','wt',newline='') as f:
  w=csv.writer(f);w.writerow(['material_id','reduced_formula','is_train','all_index_nearest_elmd','all_index_nearest_icsd_formula','all_index_nearest_icsd_id','all_index_nearest_icsd_year','post1980_nearest_elmd','post1980_nearest_icsd_formula','post1980_nearest_icsd_id','post1980_nearest_icsd_year'])
  for j in range(len(ids)):
   a=results['all_index'];p=results['post1980_through2015'];ar=a['meta'].iloc[a['i'][j]];pr=p['meta'].iloc[p['i'][j]]
   w.writerow([ids[j],forms[j],strata[j],a['d'][j],ar.formula,ar.cif_id,ar.year,p['d'][j],pr.formula,pr.cif_id,pr.year])
 summary['wall_seconds']=time.time()-wall
 (OUT/'full_gnome_nearest_icsd_elmd_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)

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
    parser.add_argument('--gnome-summary', type=Path, required=True)
    args = parser.parse_args(argv)
    lookup_path = args.elmd_lookup
    if lookup_path is None:
        spec = importlib.util.find_spec('ElMD')
        if spec is None or not spec.submodule_search_locations:
            parser.error('Install ElMD==0.5.15 or provide --elmd-lookup')
        lookup_path = Path(next(iter(spec.submodule_search_locations))) / 'el_lookup/mod_petti.json'
    global ICSD_INDEX, OUT, LOOKUP, SRC
    ICSD_INDEX, OUT = args.icsd_index, args.output_dir
    SRC = args.gnome_summary
    LOOKUP = json.loads(lookup_path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    set_num_threads(args.threads)

if __name__=='__main__':main()
