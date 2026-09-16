"""Verify frozen inputs, coverage, and every saved null reduction locally."""
from pathlib import Path
import json,csv,hashlib,numpy as np
p=Path(__file__).resolve().parent;root=p.parents[4];r=p/'results'
prov=json.loads((r/'provenance.json').read_text());checks={}
for remote,h in prov['input_hashes'].items():
 name=remote.split('/local_property_pilot_20260909/')[-1]
 if name.startswith('stage/'):local=root/name[6:]
 elif name=='common_support_ids.json':local=p.parent/'factor_ablations/inputs/common_support_ids.json'
 elif not name.startswith('/'):local=p/name
 else:continue
 checks['source_hash:'+name]=hashlib.sha256(local.read_bytes()).hexdigest()==h
rows=list(csv.DictReader((r/'per_target.csv').open()));sample=json.loads((p/'sample.json').read_text());channels=list(csv.DictReader((r/'per_channel.csv').open()));null=json.loads((r/'null_scores.json').read_text())
checks['exact_frozen_sample']=len(rows)==100 and {x['material_id'] for x in rows}=={x['material_id'] for x in sample}
checks['all_30_channels']=all(int(x['n_scored_channels'])==int(x['baseline_n_scored_channels'])==30 for x in rows)
checks['6000_channel_records']=len(channels)==6000
checks['null_f1_invariant']=all(float(x['f1_shuffle_max_error'])==float(x['baseline_f1_shuffle_max_error'])==0 for x in rows)
checks['no_feature_failures']=not json.loads((r/'feature_failures.json').read_text())
for row in rows:
 for prefix,key in [('',row['material_id']),('baseline_','baseline_'+row['material_id'])]:
  for fam in ['f1','f2_mean','f2_absdiff','pair']:
   md=np.median([x[fam] for x in null[key]])
   checks['null_reduction:'+key+':'+fam]=bool(abs(md-float(row[prefix+fam+'_shuffle_median']))<1e-12 and abs(float(row[prefix+fam+'_delta'])-(float(row[prefix+fam+'_actual'])-md))<1e-12)
summary={'checks':len(checks),'passed':int(sum(checks.values())),'failed':[k for k,v in checks.items() if not v]}
(r/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
print('unique reference IDs',len({x['icsd_reference'] for x in rows}),'parse skip events',len(json.loads((r/'parse_failures.json').read_text())))
print('structures with missing mass',sum(any(v>1e-12 for v in x['missing_mass'].values()) for x in json.loads((r/'structure_metadata.json').read_text()).values()))
assert not summary['failed']
