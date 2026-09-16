#!/usr/bin/env python3
"""Independently verify AMD eligibility, training membership and every projection."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[5];D=ROOT/'notes/feature_repair_2026_09/downstream';P=D/'representations';I=D/'inputs/amd_repair_20260910';OUT=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return list(csv.DictReader(Path(path).open()))
def load(path):return json.loads(Path(path).read_text())
ids=np.asarray(load(P/'amd_features/features.ids.json'));raw=np.load(I/'icsd_features.npy').astype(float)
records={int(r['icsd_id']):r for r in read(P/'amd-full-partition/amd/community_assignments.csv')};years=np.array([int(float(records[int(i)]['year'])) if records[int(i)]['year'] else -1 for i in ids]);labels=np.array([int(records[int(i)]['community']) for i in ids]);support=load(D/'representation_mechanism_diagnostics/factor_ablations/inputs/common_support_ids.json');support=support.get('populations',support)
support={k.lower():set(map(str,v['ids'] if isinstance(v,dict) else v)) for k,v in support.items()}
all_results={}
for folder in sorted(x for x in a.results.iterdir() if x.is_dir() and (x/'summary.json').exists()):
 s=load(folder/'summary.json');b=np.load(folder/'basis.npz');centers=b['centroids'];radii=b['radii'];communities=b['communities'];lookup={int(c):i for i,c in enumerate(communities)}
 assert np.isfinite(centers).all() and np.isfinite(radii).all() and (radii>0).all() and (b['counts']>=4).all()
 if s['arm']=='fixed_partition_filtering':
  scaler=load(P/'amd-full-partition/amd_scaler.json');mean=np.array(scaler['mean']);scale=np.array(scaler['scale']);train_raw=raw;train_ids=ids;train_labels=labels
  e=np.load(I/'graph_edges.npy');u=e[:,0].astype(int);v=e[:,1].astype(int);g=coo_matrix((np.ones(len(u)*2),(np.r_[u,v],np.r_[v,u])),shape=(len(ids),len(ids))).tocsr();nc,cl=connected_components(g,directed=False);keep=np.bincount(cl)[cl]>=8
  assert (~keep).sum()==s['n_nodes_component_filtered']
 else:
  t=s['cutoff'];mask=(years>=0)&(years<=t);train_raw=raw[mask];train_ids=ids[mask];mean=np.array(s['scaler']['mean']);scale=np.array(s['scaler']['scale']);np.testing.assert_allclose(mean,train_raw.mean(axis=0),rtol=1e-12,atol=1e-12);expected_scale=train_raw.std(axis=0);expected_scale[expected_scale==0]=1;np.testing.assert_allclose(scale,expected_scale,rtol=1e-10,atol=1e-12)
  ass=read(folder/'training_assignments.csv');assert [int(r['icsd_id']) for r in ass]==train_ids.tolist();train_labels=np.array([int(r['louvain_label']) for r in ass]);keep=train_labels>=0
  assert len(train_raw)==s['n_training'];assert not set(train_ids)&set(ids[years>t])
 xtrain=(train_raw-mean)/scale
 for j,c in enumerate(communities):
  members=xtrain[(train_labels==c)&keep];assert len(members)==b['counts'][j] and len(members)>=4
  center=members.mean(axis=0);radius=np.quantile(np.linalg.norm(members-center,axis=1),.95);assert not np.all(members==members[0]);np.testing.assert_allclose(center,centers[j],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(radius,radii[j],rtol=1e-12,atol=1e-12)
 checked={}
 for source,stats in s['populations'].items():
  rows=read(folder/(source+'_projection.csv'))
  qids=list(map(str,ids)) if source=='icsd' else list(map(str,load(P/'amd-external-full-map/results'/source/'feature_ids.json')))
  assert [r['record_key'] for r in rows]==qids
  qr=raw if source=='icsd' else np.load(I/(source+'_features.npy')).astype(float);x=(qr-mean)/scale
  chosen=np.array([lookup[int(r['assigned_community'])] for r in rows]);stored=np.array([float(r['nearest_centroid_distance']) for r in rows]);thresholds=np.array([float(r['community_radius_p95']) for r in rows]);flags=np.array([r['in_basin']=='True' for r in rows]);direct=np.linalg.norm(x-centers[chosen],axis=1)
  np.testing.assert_allclose(stored,direct,rtol=1e-12,atol=1e-12);np.testing.assert_array_equal(thresholds,radii[chosen]);np.testing.assert_array_equal(flags,direct<=thresholds)
  max_excess=0.0
  for start in range(0,len(x),512):
   nearest=cdist(x[start:start+512],centers).min(axis=1);excess=direct[start:start+512]-nearest;max_excess=max(max_excess,float(excess.max(initial=0)));np.testing.assert_allclose(direct[start:start+512],nearest,rtol=1e-11,atol=1e-11)
  selected=np.array([q in support[source] for q in qids]);full=stats['full'];common=stats['five_way_common_support'];assert len(rows)==full['n'] and int(flags.sum())==full['n_in_basin'];assert int(selected.sum())==common['n'] and int(flags[selected].sum())==common['n_in_basin']
  if source=='icsd' and s['arm']=='independent_cutoff_refit':
   later=(years>s['cutoff'])&(years<=s['max_year']);late=s['populations']['icsd']['later_through_max_year'];assert int(later.sum())==late['n'] and int(flags[later].sum())==late['n_in_basin']
  checked[source]={'rows_checked':len(rows),'in_basin':int(flags.sum()),'max_distance_above_scipy_nearest':max_excess}
 all_results[folder.name]={'status':'PASS','n_centers':len(centers),'summary_sha256':sha(folder/'summary.json'),'all_centroids_radii_reconstructed':True,'all_query_rows_retained':True,'every_nearest_distance_checked_with_scipy_cdist':True,'populations':checked}
 print(folder.name,'PASS',flush=True)
report={'status':'PASS','producer_results':str(a.results),'verifier_sha256':sha(__file__),'arms':all_results};assert all_results
(a.results/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n')
