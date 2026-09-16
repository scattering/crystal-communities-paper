#!/usr/bin/env python3
"""Project all 57 pre-experiment A-Lab MP targets into the repaired frozen AMD-100 basin map."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,platform,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from pymatgen.core import Composition,Structure

ROOT=Path(__file__).resolve().parents[4]
DOWN=ROOT/'notes/feature_repair_2026_09/downstream'
DEFAULT_DOCS=DOWN/'alab_mp_targets/source/mp_2022_10_28_summary_target_docs.json'
DEFAULT_TARGETS=DOWN/'alab_mp_targets/source/alab_targets.csv'
DEFAULT_BASIS=DOWN/'representations/amd-basin-repair-20260910/results_fixed/full_map_fixed_partition_repair/basis.npz'
DEFAULT_SCALER=DOWN/'representations/amd-full-partition/amd_scaler.json'
DEFAULT_OUT=DOWN/'alab_representation_controls_20260910/amd'

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def fractions(c):
 c=Composition(c); total=sum(c.get_el_amt_dict().values())
 return {k:v/total for k,v in c.get_el_amt_dict().items()}

def featurize(item):
 mp_id,doc,formula=item; stage='structure'
 try:
  s=Structure.from_dict(doc['structure'])
  if not 0<len(s)<=256: raise ValueError('Unsupported site count')
  a,b=fractions(s.composition),fractions(formula)
  if set(a)!=set(b) or max(abs(a[k]-b[k]) for k in a)>1e-7:
   raise ValueError('Target formula and snapshot structure composition differ')
  stage='amd_periodicset_adapter'
  import amd
  from amd.io import periodicset_from_pymatgen_structure
  ps=periodicset_from_pymatgen_structure(s)
  stage='amd_100'; v=np.asarray(amd.AMD(ps,100),dtype=np.float32)
  if v.shape!=(100,) or not np.isfinite(v).all() or np.any(v<0) or np.any(np.diff(v)<-1e-6):
   raise ValueError('Invalid AMD-100 vector')
  return {'mp_id':mp_id,'n_sites':len(s),'snapshot_formula':s.composition.formula,'composition_verified':True},v,None
 except Exception as e:
  return {'mp_id':mp_id},None,{'mp_id':mp_id,'stage':stage,'exception_type':type(e).__name__,'message':str(e)}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--docs',type=Path,default=DEFAULT_DOCS); ap.add_argument('--targets',type=Path,default=DEFAULT_TARGETS)
 ap.add_argument('--basis',type=Path,default=DEFAULT_BASIS); ap.add_argument('--scaler',type=Path,default=DEFAULT_SCALER)
 ap.add_argument('--out',type=Path,default=DEFAULT_OUT); ap.add_argument('--workers',type=int,default=2)
 a=ap.parse_args(); started=time.time()
 if not 1<=a.workers<=2: raise ValueError('workers must be 1 or 2')
 a.out.mkdir(parents=True,exist_ok=True)
 protected=['features.npy','transformed_coordinates.npy','feature_ids.json','projection_records.csv','failures.json','run_metadata.json']
 if any((a.out/x).exists() for x in protected): raise FileExistsError('Refusing to overwrite AMD target outputs')
 docs=json.loads(a.docs.read_text())
 with a.targets.open(newline='') as f: targets=list(csv.DictReader(f))
 ids=[r['mp_id'] for r in targets]
 if len(ids)!=57 or len(set(ids))!=57 or set(ids)!=set(docs): raise ValueError('Expected identical 57 unique target IDs')
 items=[(r['mp_id'],docs[r['mp_id']],r['formula']) for r in targets]
 with ProcessPoolExecutor(max_workers=a.workers) as ex: results=list(ex.map(featurize,items))
 failures=[x[2] for x in results if x[2] is not None]
 (a.out/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
 if failures: raise RuntimeError(f'{len(failures)} AMD failures; retained failure report')
 features=np.stack([x[1] for x in results]).astype(np.float32)
 scaler=json.loads(a.scaler.read_text()); mean=np.asarray(scaler['mean'],dtype=np.float64); scale=np.asarray(scaler['scale'],dtype=np.float64)
 if mean.shape!=(100,) or scale.shape!=(100,) or np.any(scale<=0): raise ValueError('Invalid AMD scaler')
 transformed=(features.astype(np.float64)-mean)/scale
 z=np.load(a.basis); communities=z['communities']; centers=z['centroids']; radii=z['radii']
 if centers.shape!=(len(communities),100) or radii.shape!=(len(communities),) or np.any(radii<=0): raise ValueError('Invalid repaired positive-radius basis')
 from scipy.spatial.distance import cdist
 distances_all=cdist(transformed,centers,metric='euclidean'); nearest=distances_all.argmin(axis=1)
 distances=np.linalg.norm(transformed-centers[nearest],axis=1); inside=distances<=radii[nearest]
 np.save(a.out/'features.npy',features); np.save(a.out/'transformed_coordinates.npy',transformed)
 (a.out/'feature_ids.json').write_text(json.dumps(ids,indent=2)+'\n')
 fields=['mp_id','corrected_outcome','feature_row','assigned_community','nearest_centroid_distance','community_threshold_p95','in_basin','formula','n_sites','composition_verified']
 with (a.out/'projection_records.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for i,(target,res) in enumerate(zip(targets,results)):
   w.writerow({'mp_id':target['mp_id'],'corrected_outcome':target['corrected_outcome'],'feature_row':i,'assigned_community':int(communities[nearest[i]]),'nearest_centroid_distance':float(distances[i]),'community_threshold_p95':float(radii[nearest[i]]),'in_basin':bool(inside[i]),'formula':target['formula'],'n_sites':res[0]['n_sites'],'composition_verified':True})
 inputs={str(p):sha256(p) for p in [a.docs,a.targets,a.basis,a.scaler]}
 outputs={x:sha256(a.out/x) for x in protected[:-1]}
 meta={'status':'PASS','scope':'57 pre-experiment A-Lab MP target structures projected into repaired frozen full-map AMD-100 basis; no fitting or radius floor','n_targets':57,'n_successful':57,'n_failures':0,'n_in_basin':int(inside.sum()),'workers':a.workers,'runtime_seconds':time.time()-started,'descriptor':'amd.io.periodicset_from_pymatgen_structure then amd.AMD(periodic_set, 100), stored float32','projection':'saved full-map scaler; exact Euclidean nearest eligible centroid; direct norm; distance <= strictly positive saved member-p95 radius','checks':{'unique_exact_ids':True,'snapshot_compositions_match_targets':True,'all_vectors_finite_nonnegative_monotone':True,'all_radii_strictly_positive':True,'no_query_deletion':True,'no_refit':True,'no_radius_floor':True},'inputs':inputs,'outputs':outputs,'software':{'python':platform.python_version(),'python_executable':__import__('sys').executable,'numpy':np.__version__,'pymatgen':importlib.metadata.version('pymatgen'),'scipy':importlib.metadata.version('scipy'),'average-minimum-distance':importlib.metadata.version('average-minimum-distance'),'isolated_amd_runtime':'/tmp/crystal-alab-amd-runtime-20260910','install_command':'python -m pip install --no-deps --target /tmp/crystal-alab-amd-runtime-20260910 average-minimum-distance==1.6.1'},'producer_sha256':sha256(Path(__file__))}
 (a.out/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(json.dumps({k:meta[k] for k in ['status','n_targets','n_successful','n_in_basin','runtime_seconds']},indent=2))
if __name__=='__main__': main()
