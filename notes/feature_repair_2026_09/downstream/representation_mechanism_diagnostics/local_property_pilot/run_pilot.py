#!/usr/bin/env python3
"""Frozen 100-structure continuous local-property pilot; no map refit or DFT."""
from pathlib import Path
import os,sys,json,csv,hashlib,getpass,zipfile,time,warnings,concurrent.futures
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'stage'),str(HERE/'stage/scripts'),str(HERE)]
from pymatgen.core import Structure,Composition
from scipy.stats import wasserstein_distance
import metrics as m
warnings.filterwarnings('ignore')
BASE=Path(os.environ.get('WORK','/path/to/tacc/work'))  # TACC work root
REP=BASE/'feature_repair_runs/downstream_v2_20260905/representations'
OUT=HERE/'results'
SEED=20260909
N_SHUFFLES=32

def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def writecsv(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def get_raw(cache):
 return m.distributions(cache['props'],m.FixedEdges(cache['i'],cache['j'],cache['w']))
def extract(task):
 tag,sd=task
 try:
  s=Structure.from_dict(sd);props=m.ordered_site_properties(s)
  nn=m.gf._structure_neighbor_info(s,neighbor_method='crystalnn')
  e=m.fixed_edges(s,nn)
  if not len(e.w) or e.w.sum()<=0:raise ValueError('No positive neighbor mass')
  path=OUT/'cache'/f'{tag}.npz';np.savez_compressed(path,props=props,i=e.i,j=e.j,w=e.w)
  raw,missing=m.distributions(props,e,return_missing_mass=True)
  info={'tag':tag,'formula':s.composition.reduced_formula,'n_sites':len(s),'n_elements':len(s.composition.elements), 'n_directed_edges':len(e.w),'total_edge_weight':float(e.w.sum()),'self_image_weight_fraction':float(e.w[e.i==e.j].sum()/e.w.sum()),'distinct_labels':len(np.unique(props[:,0])),'missing_mass':missing,'cif_ordered':bool(s.is_ordered)}
  return tag,info,None
 except Exception as exc:return tag,None,type(exc).__name__+': '+str(exc)[:180]
def load(tag):
 with np.load(OUT/'cache'/f'{tag}.npz') as z:return {k:z[k] for k in z.files}
def aggregate(ch):
 out={}
 for f in ['f1','f2_mean','f2_absdiff']:
  vals=[v for k,v in ch.items() if k.startswith(f+'/')]
  if not vals:raise ValueError('No available channels for '+f)
  out[f]=float(np.mean(vals))
 out['pair']=float((out['f2_mean']+out['f2_absdiff'])/2)
 return out

def compare(task):
 key,a,b,scales=task
 A=load(a);B=load(b);rawa=get_raw(A);rawb=get_raw(B)
 omitted=[k for k in scales if not len(rawa[k][0]) or not len(rawb[k][0])]
 scales={k:v for k,v in scales.items() if k not in omitted}
 actual=m.distance(rawa,rawb,scales);scores=aggregate(actual)
 seed=SEED+int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
 rng=np.random.default_rng(seed);null=[];null_channels=[];f1error=0
 edges=m.FixedEdges(A['i'],A['j'],A['w'])
 for _ in range(N_SHUFFLES):
  raw=m.distributions(m.shuffled_properties(A['props'],rng=rng),edges)
  for name,(v,w) in rawa.items():
   if name.startswith('f1/') and len(v):
    vv,ww=raw[name];f1error=max(f1error,float(wasserstein_distance(v,vv,u_weights=w,v_weights=ww)))
  ch=m.distance(raw,rawb,scales)
  if set(ch)!=set(actual):raise ValueError('Shuffle changed scored channel availability')
  null.append(aggregate(ch));null_channels.append(ch)
 if f1error>1e-9:raise ValueError('Shuffle changed f1 distribution')
 summary={'comparison_key':key,'query':a,'reference':b,'n_scored_channels':len(actual),'omitted_channels':omitted,'f1_shuffle_max_error':f1error}
 for family,value in scores.items():
  vals=np.array([x[family] for x in null]);median=float(np.median(vals))
  summary.update({family+'_actual':value,family+'_shuffle_median':median,family+'_delta':value-median,family+'_shuffle_sd':float(vals.std()),family+'_rank_p':float((1+sum(vals<=value))/(N_SHUFFLES+1))})
 channel=[{'comparison_key':key,'channel':k,'actual':v,'shuffle_median':float(np.median([x[k] for x in null_channels]))} for k,v in actual.items()]
 return summary,channel,null

def main():
 started=time.time();OUT.mkdir(exist_ok=False);(OUT/'cache').mkdir()
 sample=json.loads((HERE/'sample.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text())
 assert len(sample)==protocol['n_targets']==100
 assert len({r['material_id'] for r in sample})==100
 assert hashlib.sha256('\n'.join(r['material_id'] for r in sample).encode()).hexdigest()==protocol['selected_ids_sha256']
 # A password stays in memory only and is never included in outputs or subprocess arguments.
 zip_path=BASE/'reference_data/ICSD_CIFs.zip'
 z=zipfile.ZipFile(zip_path);members={int(Path(x.filename).stem.split('_')[-1]):x.filename for x in z.infolist() if Path(x.filename).stem.startswith('icsd_') and x.filename.endswith('.cif')}
 encrypted=any(z.getinfo(v).flag_bits&1 for v in list(members.values())[:1])
 password=getpass.getpass('ICSD archive passphrase (memory only): ').encode() if encrypted else None
 print('Reading reference coordinates and matching frozen targets',flush=True)
 iz_cache={};parse_fail=[]
 def icif(i):
  if i not in iz_cache:
   raw=z.read(members[i],pwd=password);s=Structure.from_str(raw.decode('utf-8',errors='replace'),fmt='cif')
   if not s.is_ordered:raise ValueError('Disordered reference')
   iz_cache[i]=(s,hashlib.sha256(raw).hexdigest())
  return iz_cache[i][0]
 ids=np.asarray(json.loads((REP/'amd_features/features.ids.json').read_text()),dtype=int)
 amd=np.asarray(np.load(REP/'amd_features/features.npy',mmap_mode='r'),dtype=float)
 scaler=json.loads((REP/'amd-full-partition/amd_scaler.json').read_text());X=(amd-np.array(scaler['mean']))/np.array(scaler['scale'])
 common=json.loads((HERE/'common_support_ids.json').read_text())['populations']['ICSD']['ids'];eligible=np.isin(ids,np.asarray(common,dtype=int))
 meta={int(r['cif_names']):r for r in csv.DictReader((BASE/'reference_data/ICSD_index.csv').open())}
 nels=np.array([len(Composition(meta[int(i)]['name']).elements) for i in ids])
 prod=BASE/'feature_repair_runs/full_v2_20260904/production';prows=list(csv.DictReader((prod/'sample_assignments.csv').open()))
 pdata=np.load(prod/'features.npy',mmap_mode='r');counts={int(r['icsd_id']):int(round(pdata[k,-1])) for k,r in enumerate(prows)}
 nsites=np.array([counts.get(int(i),0) for i in ids]);index={int(v):i for i,v in enumerate(ids)}
 groot=BASE/'feature_repair_runs/amd_external_20260906/results/gnome'
 gids=json.loads((groot/'feature_ids.json').read_text());gindex={str(v):i for i,v in enumerate(gids)}
 G=(np.asarray(np.load(groot/'amd_features.npy'),dtype=float)-scaler['mean'])/scaler['scale']
 pairs=[];tasks={};gz=zipfile.ZipFile(BASE/'reference_data/gnome_data/by_id.zip');gnames=set(gz.namelist());source_hashes={}
 def nearest(coord,ne,ns,exclude_zero=False):
  d=np.linalg.norm(X-coord,axis=1);mask=eligible&(np.abs(nels-ne)<=1)&(nsites>=max(1,ns/2))&(nsites<=2*ns)
  if exclude_zero:mask&=d>1e-8
  order=np.lexsort((ids,d));attempts=0
  for k in order[mask[order]]:
   attempts+=1
   try:s=icif(int(ids[k]));return int(ids[k]),float(d[k]),attempts,int(mask.sum())
   except Exception as exc:parse_fail.append({'icsd_id':int(ids[k]),'error':type(exc).__name__})
   if attempts>=50:break
  raise ValueError('No ordered reference among first 50 eligible AMD neighbors')
 for row in sample:
  mid=row['material_id']
  try:
   name=next(n for n in [mid+'.cif',mid+'.CIF','by_id/'+mid+'.cif','by_id/'+mid+'.CIF',mid+'.vasp.cif'] if n in gnames)
   raw=gz.read(name);s=Structure.from_str(raw.decode(),fmt='cif');assert s.is_ordered
   a,da,tries,nelig=nearest(G[gindex[mid]],len(s.composition.elements),len(s))
   sa=icif(a);b,db,btries,bnelig=nearest(X[index[a]],len(sa.composition.elements),len(sa),True)
   row=dict(row,icsd_reference=a,icsd_comparator=b,amd_distance=da,icsd_baseline_amd_distance=db,reference_attempts=tries,eligible_pool=nelig,reference_formula=sa.composition.reduced_formula,comparator_formula=icif(b).composition.reduced_formula)
   pairs.append(row);tasks['gnome_'+mid]=s.as_dict();tasks['icsd_'+str(a)]=sa.as_dict();tasks['icsd_'+str(b)]=icif(b).as_dict();source_hashes['gnome_'+mid]=hashlib.sha256(raw).hexdigest()
  except Exception as exc:parse_fail.append({'material_id':mid,'error':type(exc).__name__+': '+str(exc)[:120]})
 password=None;z.close();gz.close()
 source_hashes.update({'icsd_'+str(i):h for i,(_,h) in iz_cache.items()})
 if not pairs:raise RuntimeError('No target matched; first failures: '+str(parse_fail[:3]))
 matched_count=len(pairs)
 dump(OUT/'selection.json',pairs);dump(OUT/'parse_failures.json',parse_fail);dump(OUT/'cif_hashes.json',source_hashes)
 print('Matched',len(pairs),'targets;',len(tasks),'unique structures',flush=True)
 workers=int(os.getenv('PILOT_WORKERS','32'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:extracted=list(pool.map(extract,tasks.items()))
 infos={k:v for k,v,e in extracted if e is None};failures={k:e for k,v,e in extracted if e is not None}
 dump(OUT/'structure_metadata.json',infos);dump(OUT/'feature_failures.json',failures)
 pairs=[r for r in pairs if all(k in infos for k in ['gnome_'+r['material_id'],'icsd_'+str(r['icsd_reference']),'icsd_'+str(r['icsd_comparator'])])]
 if not pairs:raise RuntimeError('No analyzed targets after feature failures')
 print('Extracted',len(infos),'structures;',len(pairs),'complete comparisons; computing shuffles',flush=True)
 refs=sorted({k for r in pairs for k in ['icsd_'+str(r['icsd_reference']),'icsd_'+str(r['icsd_comparator'])]})
 raws=[get_raw(load(k)) for k in refs];scales={};scaleinfo={}
 for ch in m.CHANNEL_NAMES:
  valid=[r[ch] for r in raws if len(r[ch][0])];means=[float(np.dot(v,w)) for v,w in valid];seconds=[float(np.dot(v*v,w)) for v,w in valid]
  sd=float(np.sqrt(max(0,np.mean(seconds)-np.mean(means)**2))) if valid else 0
  scaleinfo[ch]={'sd':sd,'n_reference_structures':len(valid)}
  if sd>1e-12:scales[ch]=sd
 dump(OUT/'scales.json',scaleinfo)
 jobs=[]
 for r in pairs:
  mid=r['material_id'];a='icsd_'+str(r['icsd_reference']);b='icsd_'+str(r['icsd_comparator'])
  jobs.extend([(mid,'gnome_'+mid,a,scales),('baseline_'+mid,a,b,scales)])
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:res=list(pool.map(compare,jobs))
 bykey={a['comparison_key']:a for a,_,_ in res};summaries=[]
 for r in pairs:
  out=dict(r);a=bykey[r['material_id']];bb=bykey['baseline_'+r['material_id']]
  out.update({k:v for k,v in a.items() if k not in ['comparison_key','query','reference']});out.update({'baseline_'+k:v for k,v in bb.items() if k not in ['comparison_key','query','reference']});summaries.append(out)
 writecsv(OUT/'per_target.csv',summaries);writecsv(OUT/'per_channel.csv',[x for _,cs,_ in res for x in cs]);dump(OUT/'null_scores.json',{a['comparison_key']:ns for a,_,ns in res})
 # ICSD–ICSD baseline and target effects are descriptive in this stratified pilot.
 summary={'n_selected':len(sample),'n_matched':matched_count,'n_analyzed':len(summaries),'n_unique_reference_structures':len(refs),'n_shuffles':N_SHUFFLES,'n_channels':len(scales),'runtime_seconds':time.time()-started,'groups':{}}
 for group in ['all_pilot']+list(dict.fromkeys(r['cell'] for r in summaries)):
  rr=[r for r in summaries if group=='all_pilot' or r['cell']==group];entry={'n':len(rr)}
  for fam in ['f1','f2_mean','f2_absdiff','pair']:
   entry[fam]={k:float(np.median([r[fam+'_'+k] for r in rr])) for k in ['actual','shuffle_median','delta']}
   entry[fam]['baseline_actual']=float(np.median([r['baseline_'+fam+'_actual'] for r in rr]));entry[fam]['n_actual_below_shuffle_median']=sum(r[fam+'_delta'] < -1e-10 for r in rr);entry[fam]['n_null_informative']=sum(r[fam+'_shuffle_sd']>1e-10 for r in rr)
  summary['groups'][group]=entry
 summary['checks']={'f1_shuffle_max_error':max(r['f1_shuffle_max_error'] for r in summaries),'all_finite_scores':all(np.isfinite(r['pair_actual']) for r in summaries),'ordered_only':all(x['cif_ordered'] for x in infos.values()),'complete_frozen_sample_analyzed':len(summaries)==len(sample), 'selected_ids_hash_verified':True}
 inputs=[BASE/'reference_data/ICSD_index.csv',prod/'sample_assignments.csv',prod/'features.npy',HERE/'common_support_ids.json',HERE/'sample.json',HERE/'protocol.json',HERE/'metrics.py',HERE/'run_pilot.py',HERE/'stage/experiments/graphlet_compare/graphlet_features.py',HERE/'stage/scripts/crystal_neighbors.py',REP/'amd_features/features.npy',REP/'amd_features/features.ids.json',REP/'amd-full-partition/amd_scaler.json',groot/'amd_features.npy',groot/'feature_ids.json']
 dump(OUT/'provenance.json',{'input_hashes':{str(p):sha(p) for p in inputs},'slurm_job_id':os.getenv('SLURM_JOB_ID'),'numpy':np.__version__,'protocol':json.loads((HERE/'protocol.json').read_text())})
 dump(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
