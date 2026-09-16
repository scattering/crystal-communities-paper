#!/usr/bin/env python3
"""Frozen cross-cohort chemistry/contact/coordination comparison; no map fitting."""
from pathlib import Path
import os,sys,json,csv,hashlib,getpass,zipfile,time,warnings,concurrent.futures,bz2
import numpy as np
import importlib.metadata
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'stage'),str(HERE/'stage/scripts'),str(HERE)]
from pymatgen.core import Structure,Composition
import bond_metrics as m
warnings.filterwarnings('ignore')
BASE=Path(os.environ.get('WORK','/path/to/tacc/work'))  # TACC work root
REP=BASE/'feature_repair_runs/downstream_v2_20260905/representations'
AMD=BASE/'feature_repair_runs/amd_external_20260906/results'
OUT=HERE/'results'
SEED=20260909
N_SHUFFLES=32
SOURCES=['gnome','mattergen','mp','jarvis','alexandria']
SOURCE_PATHS={'gnome':BASE/'reference_data/gnome_data/by_id.zip','mattergen':BASE/'mattergen/data-release/cifs.zip','mp':BASE/'reference_data/mp_theoretical_candidates_20260427.jsonl','jarvis':BASE/'reference_data/jarvis_dft/jdft_3d-12-12-2022.json','alexandria':BASE/'reference_data/alexandria_pbe_2025_07_02'}
def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def writecsv(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def key(row):return row.get('record_key') or row.get('zip_member') or row.get('source_record',{}).get('zip_member') or row['material_id']
def tag(row):return row['source']+'_'+hashlib.sha256(key(row).encode()).hexdigest()[:20]
def read_alex(task):
 path,wanted=task;found={}
 with bz2.open(path,'rt') as f:data=json.load(f)
 for e in data['entries'] if isinstance(data,dict) else data:
  mid=str((e.get('data') or {}).get('mat_id') or e.get('entry_id'))
  if mid in wanted:found[mid]=e['structure']
 return found

def read_public(sample):
 tasks={};hashes={};failures=[]
 for source in SOURCES:
  rows=[r for r in sample if r['source']==source];wanted={r['material_id'] for r in rows};raws={};parsed={}
  if source in ['gnome','mattergen']:
   with zipfile.ZipFile(SOURCE_PATHS[source]) as z:
    names=set(z.namelist())
    for r in rows:
     mid=r['material_id'];name=key(r) if source=='mattergen' else next(n for n in [mid+'.cif',mid+'.CIF','by_id/'+mid+'.cif','by_id/'+mid+'.CIF'] if n in names)
     raw=z.read(name);raws[key(r)]=raw
     parsed[key(r)]=Structure.from_str(raw.decode('utf-8',errors='replace'),fmt='cif')
  elif source=='mp':
   with SOURCE_PATHS[source].open() as f:
    for line in f:
     e=json.loads(line);mid=e['material_id']
     if mid in wanted:raws[mid]=line.encode();parsed[mid]=Structure.from_dict(e['structure'])
  elif source=='jarvis':
   with SOURCE_PATHS[source].open() as f:raw=json.load(f)
   for e in raw:
    mid=e.get('jid')
    if mid in wanted:
     a=e['atoms'];raws[mid]=json.dumps(e,sort_keys=True).encode();parsed[mid]=Structure(a['lattice_mat'],a['elements'],a['coords'],coords_are_cartesian=bool(a.get('cartesian',False)))
  else:
   paths=[SOURCE_PATHS[source]/f'alexandria_{i:05d}.json.bz2' for i in [0,19,38]]
   with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
    for part in pool.map(read_alex,[(p,wanted) for p in paths]):
     for mid,sd in part.items():
      if mid in parsed:raise ValueError('Duplicate Alexandria ID '+mid)
      raws[mid]=json.dumps(sd,sort_keys=True).encode();parsed[mid]=Structure.from_dict(sd)
  for r in rows:
   k=key(r)
   if k not in parsed:failures.append({'source':source,'record_key':k,'stage':'public_parse','error':'missing source key'});continue
   s=parsed[k]
   if not s.is_ordered:failures.append({'source':source,'record_key':k,'stage':'public_parse','error':'disordered query'});continue
   if len(s)!=int(r['n_sites']):raise ValueError('Frozen query site count differs: '+k)
   if len(s.composition.elements)!=int(r['n_elements']):raise ValueError('Frozen query element count differs: '+k)
   tasks[tag(r)]=s.as_dict();hashes[tag(r)]=hashlib.sha256(raws[k]).hexdigest()
  print('Loaded public',source,len(rows),'selected;',sum(t.startswith(source+'_') for t in tasks),'parsed',flush=True)
 return tasks,hashes,failures

def extract(task):
 name,sd=task
 try:
  s=Structure.from_dict(sd);c=m.extract(s)
  np.savez_compressed(OUT/'cache'/f'{name}.npz',**c)
  info={'formula':s.composition.reduced_formula,'n_sites':len(s),'n_elements':len(s.composition.elements),'volume_per_site':float(s.volume/len(s)),'density':float(s.density),'mean_weighted_cn':float(c['cn'].mean()),'n_contacts':len(c['w']),'contact_weight':float(c['w'].sum()),'self_image_weight_fraction':float(c['w'][c['i']==c['j']].sum()/c['w'].sum()),'elements':sorted(e.symbol for e in s.composition.elements)}
  return name,info,None
 except Exception as exc:return name,None,type(exc).__name__+': '+str(exc)[:200]
def load(name):
 with np.load(OUT/'cache'/f'{name}.npz') as z:return {k:z[k] for k in z.files}
def compare(task):
 name,a,b,scales=task;A=load(a);B=load(b);va=m.vectors(A);vb=m.vectors(B)
 actual={f:m.weighted_energy(v[0],v[1],vb[f][0],vb[f][1],scales[f]) for f,v in va.items()}
 marg={f:list(map(float,m.marginal_w1(v[0],v[1],vb[f][0],vb[f][1],scales[f]))) for f,v in va.items()}
 seed=SEED+int(hashlib.sha256(name.encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed);null=[];gerr=0
 for _ in range(N_SHUFFLES):
  v=m.vectors(A,permutation=rng.permutation(len(A['z'])))
  if not np.array_equal(v['geometry'][0],va['geometry'][0]):raise ValueError('Shuffle changed geometry')
  ns={f:m.weighted_energy(x[0],x[1],vb[f][0],vb[f][1],scales[f]) if f!='geometry' else actual[f] for f,x in v.items()};null.append(ns)
 result={'comparison_key':name,'query_tag':a,'reference_tag':b}
 for f,act in actual.items():
  values=np.array([v[f] for v in null]);med=float(np.median(values))
  result.update({f+'_actual':act,f+'_shuffle_median':med,f+'_delta':act-med,f+'_shuffle_sd':float(values.std())})
 return result,marg,null

def main():
 started=time.time();OUT.mkdir(exist_ok=False);(OUT/'cache').mkdir()
 sample=json.loads((HERE/'sample.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text())
 assert sha(HERE/'sample.json')==protocol['sample_sha256']
 assert len({(r['source'],key(r)) for r in sample})==len(sample)
 public,hashes,failures=read_public(sample)
 z=zipfile.ZipFile(BASE/'reference_data/ICSD_CIFs.zip');members={int(Path(x.filename).stem.split('_')[-1]):x.filename for x in z.infolist() if Path(x.filename).stem.startswith('icsd_') and x.filename.endswith('.cif')}
 password=getpass.getpass('ICSD archive passphrase (memory only): ').encode()
 print('Matching ordered ICSD references by frozen AMD coordinates',flush=True)
 icache={};parse_skips=[]
 def icif(i):
  if i not in icache:
   raw=z.read(members[i],pwd=password);s=Structure.from_str(raw.decode('utf-8',errors='replace'),fmt='cif')
   if not s.is_ordered:raise ValueError('disordered reference')
   icache[i]=(s,hashlib.sha256(raw).hexdigest())
  return icache[i][0]
 ids=np.array(json.loads((REP/'amd_features/features.ids.json').read_text()),dtype=int);index={int(v):i for i,v in enumerate(ids)}
 scaler=json.loads((REP/'amd-full-partition/amd_scaler.json').read_text());X=(np.asarray(np.load(REP/'amd_features/features.npy'),dtype=float)-scaler['mean'])/scaler['scale']
 common=json.loads((HERE/'common_support_ids.json').read_text())['populations']['ICSD']['ids'];eligible=np.isin(ids,np.asarray(common,dtype=int))
 meta={int(r['cif_names']):r for r in csv.DictReader((BASE/'reference_data/ICSD_index.csv').open())};nels=np.array([len(Composition(meta[int(i)]['name']).elements) for i in ids])
 prod=BASE/'feature_repair_runs/full_v2_20260904/production';prows=list(csv.DictReader((prod/'sample_assignments.csv').open()));pdata=np.load(prod/'features.npy',mmap_mode='r');counts={int(r['icsd_id']):int(round(pdata[k,-1])) for k,r in enumerate(prows)};nsites=np.array([counts.get(int(i),0) for i in ids])
 def nearest(coord,ne,ns,exclude_zero=False):
  d=np.linalg.norm(X-coord,axis=1);mask=eligible&(np.abs(nels-ne)<=1)&(nsites>=max(1,ns/2))&(nsites<=2*ns)
  if exclude_zero:mask&=d>1e-8
  order=np.lexsort((ids,d))
  for attempt,k in enumerate(order[mask[order]][:50],1):
   try:icif(int(ids[k]));return int(ids[k]),float(d[k])
   except Exception as exc:parse_skips.append({'icsd_id':int(ids[k]),'error':type(exc).__name__})
  raise ValueError('No ordered reference within first 50 eligible neighbors')
 old={r['material_id']:r for r in json.loads((HERE/'gnome_previous_selection.json').read_text())};selected=[];tasks=dict(public)
 for source in SOURCES:
  sourceids=json.loads((AMD/source/'feature_ids.json').read_text());gi={str(v):i for i,v in enumerate(sourceids)};G=(np.asarray(np.load(AMD/source/'amd_features.npy'),dtype=float)-scaler['mean'])/scaler['scale']
  for row in [r for r in sample if r['source']==source and tag(r) in public]:
   try:
    coord=G[gi[key(row)]]
    if source=='gnome':
     o=old[row['material_id']];a=int(o['icsd_reference']);b=int(o['icsd_comparator']);da=float(np.linalg.norm(X[index[a]]-coord));db=float(np.linalg.norm(X[index[a]]-X[index[b]]))
     if abs(da-float(o['amd_distance']))>1e-8:raise ValueError('Prior GNoME AMD reference changed')
    else:
     a,da=nearest(coord,int(row['n_elements']),int(row['n_sites']));sa=icif(a);b,db=nearest(X[index[a]],len(sa.composition.elements),len(sa),True)
    sa=icif(a);sb=icif(b);tasks['icsd_'+str(a)]=sa.as_dict();tasks['icsd_'+str(b)]=sb.as_dict()
    row={k:v for k,v in row.items() if k!='source_record'}
    selected.append(dict(row,query_tag=tag(row),icsd_reference=a,icsd_comparator=b,amd_distance=da,icsd_baseline_amd_distance=db,reference_formula=sa.composition.reduced_formula,comparator_formula=sb.composition.reduced_formula))
   except Exception as exc:failures.append({'source':source,'record_key':key(row),'stage':'matching','error':type(exc).__name__+': '+str(exc)[:150]})
 password=None;z.close();hashes.update({'icsd_'+str(i):h for i,(_,h) in icache.items()})
 dump(OUT/'selection.json',selected);dump(OUT/'parse_skips.json',parse_skips);dump(OUT/'cif_hashes.json',hashes)
 print('Matched',len(selected),'targets;',len(tasks),'unique structures',flush=True)
 workers=int(os.getenv('PILOT_WORKERS','32'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:extracted=list(pool.map(extract,tasks.items()))
 infos={k:v for k,v,e in extracted if not e};featurefails={k:e for k,v,e in extracted if e};dump(OUT/'structure_metadata.json',infos);dump(OUT/'feature_failures.json',featurefails)
 valid=[]
 for row in selected:
  required=[row['query_tag'],'icsd_'+str(row['icsd_reference']),'icsd_'+str(row['icsd_comparator'])]
  if all(k in infos for k in required):valid.append(row)
  else:failures.append({'source':row['source'],'record_key':key(row),'stage':'features','error':'; '.join(featurefails.get(k,'') for k in required if k not in infos)})
 dump(OUT/'failures.json',failures)
 if not valid:raise RuntimeError('No valid comparisons')
 refs=sorted({k for r in valid for k in ['icsd_'+str(r['icsd_reference']),'icsd_'+str(r['icsd_comparator'])]})
 vectors=[m.vectors(load(k)) for k in refs];scales={};scaleinfo={}
 for family in ['joint','chemistry','geometry']:
  moments=np.array([np.sum(v[family][0]*v[family][1][:,None],axis=0) for v in vectors]);seconds=np.array([np.sum(v[family][0]**2*v[family][1][:,None],axis=0) for v in vectors]);sd=np.sqrt(np.maximum(0,seconds.mean(axis=0)-moments.mean(axis=0)**2))
  if np.any(sd<=1e-12):raise ValueError('Constant ICSD scaling channel in '+family)
  scales[family]=sd.tolist();scaleinfo[family]={'sd':sd.tolist(),'mean':moments.mean(axis=0).tolist(),'n_reference_structures':len(refs)}
 dump(OUT/'scales.json',scaleinfo)
 dump(OUT/'covalent_radii.json',m.CovalentRadius.radius)
 print('Extracted',len(infos),'structures;',len(valid),'complete targets; computing joint and component comparisons',flush=True)
 jobs=[]
 for r in valid:
  q=r['query_tag'];a='icsd_'+str(r['icsd_reference']);b='icsd_'+str(r['icsd_comparator']);jobs.extend([(q,q,a,scales),('baseline_'+q,a,b,scales)])
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(compare,jobs))
 bykey={r['comparison_key']:r for r,_,_ in results};rows=[]
 for row in valid:
  q=row['query_tag'];info=infos[q];out=dict(row);out.update({k:v for k,v in info.items() if k not in ['formula','n_sites','n_elements','elements']});out.update({k:v for k,v in bykey[q].items() if k not in ['comparison_key','query_tag','reference_tag']});out.update({'baseline_'+k:v for k,v in bykey['baseline_'+q].items() if k not in ['comparison_key','query_tag','reference_tag']});rows.append(out)
 writecsv(OUT/'per_target.csv',rows);dump(OUT/'marginal_distances.json',{r['comparison_key']:ma for r,ma,_ in results});dump(OUT/'null_scores.json',{r['comparison_key']:ns for r,_,ns in results})
 summary={'n_selected':len(sample),'n_matched':len(selected),'n_analyzed':len(rows),'n_icsd_references':len(refs),'n_shuffles':N_SHUFFLES,'runtime_seconds':time.time()-started,'cohorts':{}}
 for source in SOURCES:
  rr=[r for r in rows if r['source']==source];summary['cohorts'][source]={'n_selected':sum(r['source']==source for r in sample),'n_analyzed':len(rr),'median_amd_distance':float(np.median([r['amd_distance'] for r in rr])) if rr else None}
  for f in ['joint','chemistry','geometry']:
   summary['cohorts'][source][f]={k:float(np.median([r[f+'_'+k] for r in rr])) for k in ['actual','delta','shuffle_median']} if rr else None
 inputs=[HERE/f for f in ['sample.json','protocol.json','run_bonding.py','bond_metrics.py','common_support_ids.json','gnome_previous_selection.json','stage/scripts/crystal_neighbors.py','stage/experiments/graphlet_compare/graphlet_features.py']]+[REP/'amd_features/features.npy',REP/'amd_features/features.ids.json',REP/'amd-full-partition/amd_scaler.json',BASE/'reference_data/ICSD_index.csv',prod/'sample_assignments.csv',prod/'features.npy']+[AMD/s/f for s in SOURCES for f in ['feature_ids.json','amd_features.npy']]
 dump(OUT/'provenance.json',{'input_hashes':{str(p):sha(p) for p in inputs},'slurm_job_id':os.getenv('SLURM_JOB_ID'),'protocol':protocol,'numpy':np.__version__,'packages':{x:importlib.metadata.version(x) for x in ['pymatgen','scipy','numpy']}})
 dump(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
