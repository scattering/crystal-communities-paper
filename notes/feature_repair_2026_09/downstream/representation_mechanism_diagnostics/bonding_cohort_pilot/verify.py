"""Verify frozen cohort identities, cached geometry, source hashes and saved scores."""
from pathlib import Path
import json,csv,hashlib,sys
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[4];R=P/'results'
sys.path.insert(0,str(P));import bond_metrics as m
checks={}
def add(name,value):checks[name]=bool(value)
def key(r):return r.get('record_key') or r.get('zip_member') or r.get('source_record',{}).get('zip_member') or r['material_id']
sample=json.loads((P/'sample.json').read_text());rows=list(csv.DictReader((R/'per_target.csv').open()));protocol=json.loads((P/'protocol.json').read_text());provenance=json.loads((R/'provenance.json').read_text());null=json.loads((R/'null_scores.json').read_text());scales=json.loads((R/'scales.json').read_text());summary=json.loads((R/'summary.json').read_text())
add('frozen_sample_sha',hashlib.sha256((P/'sample.json').read_bytes()).hexdigest()==protocol['sample_sha256'])
selected={(r['source'],key(r)) for r in sample};analyzed={(r['source'],key(r)) for r in rows}
add('no_replacement_targets',analyzed<=selected);add('unique_analyzed',len(analyzed)==len(rows));add('count_matches_summary',len(rows)==summary['n_analyzed'])
for remote,h in provenance['input_hashes'].items():
 name=remote.split('/bonding_cohort_pilot_20260909/')[-1]
 if name.startswith('stage/'):p=ROOT/name[6:]
 elif name=='common_support_ids.json':p=P.parent/'factor_ablations/inputs/common_support_ids.json'
 elif name=='gnome_previous_selection.json':p=P.parent/'local_property_pilot/results/selection.json'
 elif not name.startswith('/'):p=P/name
 else:continue
 add('source_hash:'+name,hashlib.sha256(p.read_bytes()).hexdigest()==h)
for row in rows:
 for pre,k in [('',row['query_tag']),('baseline_','baseline_'+row['query_tag'])]:
  add('null_count:'+k,len(null[k])==32)
  for f in ['joint','chemistry','geometry']:
   vals=np.array([x[f] for x in null[k]]);median=np.median(vals);actual=float(row[pre+f+'_actual'])
   add('finite_nonnegative:'+k+':'+f,np.isfinite(vals).all() and (vals>=0).all() and np.isfinite(actual) and actual>=0)
   add('null_reduction:'+k+':'+f,abs(median-float(row[pre+f+'_shuffle_median']))<1e-10 and abs(actual-median-float(row[pre+f+'_delta']))<1e-10)
   if f=='geometry':add('geometry_invariant:'+k,np.max(abs(vals-actual))==0)
for path in (R/'cache').glob('*.npz'):
 with np.load(path) as z:c={k:z[k] for k in z.files}
 add('geometry_cache:'+path.stem,np.all(c['d']>0) and np.all(c['w']>=0) and c['w'].sum()>0 and np.allclose(c['cn'],np.bincount(c['i'],weights=c['w'],minlength=len(c['z']))))
# Independently recompute selected full joint/component scores from saved contact caches.
rng=np.random.default_rng(20260909)
for index in rng.choice(len(rows),min(10,len(rows)),replace=False):
 row=rows[index]
 with np.load(R/'cache'/f"{row['query_tag']}.npz") as z:A={k:z[k] for k in z.files}
 with np.load(R/'cache'/f"icsd_{row['icsd_reference']}.npz") as z:B={k:z[k] for k in z.files}
 va,vb=m.vectors(A),m.vectors(B)
 for family in va:
  actual=m.weighted_energy(*va[family],*vb[family],scales[family]['sd'])
  add('recomputed:'+row['query_tag']+':'+family,np.isclose(actual,float(row[family+'_actual']),atol=1e-9,rtol=1e-9))
result={'checks':len(checks),'passed':sum(checks.values()),'failed':[k for k,v in checks.items() if not v],'frozen_target_count':len(sample),'analyzed_count':len(rows),'all_targets_analyzed':selected==analyzed}
(R/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert not result['failed']
