"""Independently verify saved A-Lab encodings, frozen-map projections, and outcomes."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import hypergeom

HERE=Path(__file__).resolve().parent;DOWN=HERE.parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'experiments/graphlet_compare')]
import graphlet_features as gf

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
def B(v):
 assert str(v).lower() in ('true','false','1','0')
 return str(v).lower() in ('true','1')
def main():
 targets=readcsv(DOWN/'alab_mp_targets/source/alab_targets.csv');truth={r['mp_id']:r for r in targets};docs=json.loads((DOWN/'alab_mp_targets/source/mp_2022_10_28_summary_target_docs.json').read_text())
 assert len(truth)==len(docs)==57 and set(truth)==set(docs)
 reps={'graphlet_crystalnn':DOWN/'external_representation/graphlet/basis/basis_full.npz','graphlet_voronoinn':DOWN/'representation_mechanism_diagnostics/voronoi_external/results/basis/basis_full.npz','amd':DOWN/'representations/amd-basin-repair-20260910/results_fixed/full_map_fixed_partition_repair/basis.npz'}
 out={}
 for name,bpath in reps.items():
  folder=HERE/name;rows=readcsv(folder/'projection_records.csv');ids=json.loads((folder/'feature_ids.json').read_text());assert len(ids)==len(set(ids))==len(rows)==57 and set(ids)==set(truth)
  features=np.load(folder/'features.npy',allow_pickle=False);assert features.shape[0]==57 and np.isfinite(features).all()
  coords=np.load(folder/('transformed_coordinates.npy' if name=='amd' else 'coordinates.npy'),allow_pickle=False)
  assert np.isfinite(coords).all()
  errors={}
  if name=='amd':
   scalerpath=DOWN/'representations/amd-full-partition/amd_scaler.json';scaler=json.loads(scalerpath.read_text());mean=np.array(scaler['mean']);scale=np.array(scaler['scale']);expected=(features.astype(float)-mean)/scale;assert np.array_equal(coords,expected)
   assert features.shape==(57,100) and (features>=0).all() and (np.diff(features,axis=1)>=-1e-6).all();errors['scaled_coordinate_max_abs_error']=0.0
  else:
   expected=np.cumsum(features,axis=2).reshape(57,1280).astype(np.float32);assert np.array_equal(expected,coords)
   assert features.shape==(57,64,20) and (features>=0).all() and np.allclose(features.sum(axis=2),1,atol=2e-6,rtol=0)
   edges=json.loads((bpath.parent/'bin_edges.json').read_text());maxerr=0
   for idx,mid in enumerate(ids):
    with np.load(folder/'raw_features'/f'{mid}.npz',allow_pickle=False) as raw:
     assert set(raw.files)==set(gf.REGISTRY.all)
     for k,channel in enumerate(gf.REGISTRY.all):
      vw=raw[channel];assert vw.ndim==2 and vw.shape[1]==2 and len(vw)>0 and np.isfinite(vw).all() and (vw[:,1]>=0).all()
      lo,hi=edges[channel];pos=(np.clip(vw[:,0],lo,hi)-lo)*(20/(hi-lo));near=np.rint(pos);pos=np.where(abs(pos-near)<=1e-10,near,pos)
      hist,_=np.histogram(pos,bins=np.arange(21,dtype=float),weights=vw[:,1]);hist=hist/hist.sum();err=float(np.max(abs(hist-features[idx,k])));maxerr=max(maxerr,err)
   assert maxerr<2e-6,(name,maxerr);errors['independent_raw_weighted_histogram_max_abs_error']=maxerr;errors['raw_histogram_channels_checked']=57*64;errors['histogram_to_cdf_exact']=True
  with np.load(bpath,allow_pickle=False) as z:basis=dict(z)
  centers=basis['centroids'].astype(float);communities=basis['communities'];radii=basis.get('p95',basis.get('radii'))
  # scipy cdist is independent of the Graphlet producer's dot-product search.
  distances=cdist(coords.astype(float),centers,metric='euclidean');nearest=distances.argmin(axis=1);selected=distances[np.arange(57),nearest];flags=selected<=radii[nearest]
  recorded=np.array([float(r['nearest_centroid_distance']) for r in rows]);maxerr=float(np.max(abs(selected-recorded)));assert np.allclose(selected,recorded,atol=2e-10,rtol=2e-10),(name,maxerr)
  for idx,(mid,r) in enumerate(zip(ids,rows)):
   assert r['mp_id']==mid and int(r['feature_row'])==idx and r['corrected_outcome']==truth[mid]['corrected_outcome']
   assert int(r['assigned_community'])==int(communities[nearest[idx]])
   assert float(r['community_threshold_p95'])==float(radii[nearest[idx]])
   assert B(r['in_basin'])==bool(flags[idx]),(name,mid)
   assert int(r['n_sites'])==len(docs[mid]['structure']['sites'])
  out[name]={'status':'PASS','n_targets':57,'n_in_basin_all_outcomes':int(flags.sum()),'n_zero_radius_target_assignments':int(np.sum(radii[nearest]==0)),
   'min_absolute_distance_to_threshold':float(np.min(abs(selected-radii[nearest]))),'min_relative_distance_to_threshold':float(np.min(abs(selected-radii[nearest])/np.maximum(radii[nearest],1e-20))),
   'independent_nearest_distance_max_abs_error':maxerr,'all_community_radius_and_boolean_fields_match':True,'ids_outcomes_and_feature_rows_match':True,'encoding_checks':errors,
   'basis_path':str(bpath.relative_to(DOWN)),'basis_sha256':sha(bpath),'projection_sha256':sha(folder/'projection_records.csv'),'feature_sha256':sha(folder/'features.npy')}
 # Independently enumerate Fisher's null table distribution and verify every table count.
 comparison=json.loads((HERE/'outcome_comparison.json').read_text());statchecks={}
 for name,record in comparison['representations'].items():
  rr=readcsv(DOWN/record['projection_path']);r=[x for x in rr if x['corrected_outcome'] in ('made','not_obtained')];assert len(r)==51
  cells=[[sum(x['corrected_outcome']==outcome and B(x['in_basin'])==flag for x in r) for flag in [True,False]] for outcome in ['made','not_obtained']]
  expected=record['common_successful_support'];assert cells==expected['table_rows_made_not_obtained_cols_in_frontier']
  aa,bb=cells[0];cc,dd=cells[1];M=aa+bb+cc+dd;n=aa+cc;N=aa+bb;support=np.arange(max(0,N-(M-n)),min(N,n)+1);probs=hypergeom.pmf(support,M,n,N);p=float(probs[probs<=hypergeom.pmf(aa,M,n,N)*(1+1e-12)].sum())
  assert abs(p-expected['fisher_exact_p_two_sided'])<2e-12
  statchecks[name]={'status':'PASS','recomputed_table':cells,'enumerated_fisher_p_two_sided':p}
 report={'status':'PASS','representations':out,'outcome_checks':statchecks,'n_queries_checked':171,'n_graphlet_weighted_channels_checked':7296,'verifier_sha256':sha(Path(__file__)),'outcome_comparison_sha256':sha(HERE/'outcome_comparison.json')}
 (HERE/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
