#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=next(parent for parent in Path(__file__).resolve().parents if (parent/'.git').exists())
DOWN=REPO/'notes/feature_repair_2026_09/downstream'
AMD=HERE/'results'
OUT=HERE
SOURCES=('gnome','mattergen','mp','jarvis','alexandria')
LABELS={'gnome':'GNoME','mattergen':'MatterGen','mp':'MP','jarvis':'JARVIS','alexandria':'Alexandria'}
REPS={
 'CrystalWeave': {'icsd':DOWN/'external_representation/production_reference/icsd_full.csv','external':{s:DOWN/'external'/s/(('mattergen-public' if s=='mattergen' else s)+'_frontier_records.csv') for s in SOURCES}},
 'Magpie': {'icsd':DOWN/'external_representation/consistent_transform/basis/icsd_full.csv','root':DOWN/'external_representation/consistent_transform/external'},
 'CrystalNN graphlets': {'icsd':DOWN/'external_representation/graphlet/basis/icsd_full.csv','root':DOWN/'external_representation/graphlet/external'},
 'VoronoiNN graphlets': {'icsd':DOWN/'representation_mechanism_diagnostics/voronoi_external/results/basis/icsd_full.csv','root':DOWN/'representation_mechanism_diagnostics/voronoi_external/results/external'},
 'AMD': {'icsd':AMD/'icsd_projection_full.csv','root':AMD},
}
EXPECTED={'ICSD':83661,'GNoME':5000,'MatterGen':384,'MP':4969,'JARVIS':4930,'Alexandria':4982}

def digest_ids(values): return hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest()
def digest_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def retained_path(path):
 try: return path.relative_to(REPO).as_posix()
 except ValueError: return str(path)
def extpath(spec,s): return spec['external'][s] if 'external' in spec else spec['root']/s/'projection_full.csv'
def load(path,key):
 f=pd.read_csv(path,dtype={key:str},float_precision='round_trip')
 if f[key].isna().any() or not f[key].is_unique: raise ValueError(path)
 values=f.in_basin.astype(str).str.lower()
 if not values.isin(['true','false']).all(): raise ValueError(path)
 flags=values.eq('true')
 calc=f.nearest_centroid_distance.le(f.community_threshold_p95)
 if not flags.equals(calc): raise ValueError(f'flags do not reproduce {path}')
 flags.index=f[key].astype(str); flags.index.name='record_key'; return flags

projections={}; inputs={}
for rep,spec in REPS.items():
 projections[rep]={}
 paths={'ICSD':spec['icsd'],**{LABELS[s]:extpath(spec,s) for s in SOURCES}}
 for pop,path in paths.items():
  key='record_key'
  if rep=='CrystalWeave' and pop!='ICSD': key='zip_member' if pop=='MatterGen' else 'material_id'
  projections[rep][pop]=load(path,key)
  inputs[retained_path(path)]={'sha256':digest_file(path),'bytes':path.stat().st_size}
pops=('ICSD',)+tuple(LABELS[s] for s in SOURCES)
common={pop:set.intersection(*(set(projections[r][pop].index) for r in REPS)) for pop in pops}
for pop in pops:
 if len(common[pop])!=EXPECTED[pop]: raise ValueError((pop,len(common[pop])))
rows=[]
for rep in REPS:
 ic=projections[rep]['ICSD'].loc[sorted(common['ICSD'])]; ir=float(ic.mean())
 rows.append({'representation':rep,'population':'ICSD','n_common':len(ic),'n_in_basin':int(ic.sum()),'in_basin_fraction':ir,'icsd_minus_source_percentage_points':''})
 for pop in pops[1:]:
  f=projections[rep][pop].loc[sorted(common[pop])]; rate=float(f.mean())
  rows.append({'representation':rep,'population':pop,'n_common':len(f),'n_in_basin':int(f.sum()),'in_basin_fraction':rate,'icsd_minus_source_percentage_points':100*(ir-rate)})
with (OUT/'five_representation_common_support.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report={'status':'complete','scope':'Exact evaluation-identifier intersection across CrystalWeave, Magpie, CrystalNN graphlets, VoronoiNN graphlets and AMD. Every map retains its own fitted coordinates, partition, centroids and member-p95 radii.','common_populations':{p:{'n':len(common[p]),'ids_sha256':digest_ids(common[p])} for p in pops},'rows':rows,'inputs':inputs,'checks':{'unique_ids':True,'saved_flags_reproduced':True,'exact_five_representation_intersection':True},'script_sha256':digest_file(Path(__file__))}
(OUT/'five_representation_common_support.json').write_text(json.dumps(report,indent=2)+'\n')
by={r:{x['population']:x for x in rows if x['representation']==r} for r in REPS}
lines=['# Five-representation exact-common-support full-map comparison','', 'Cells are in-basin percentages on the exact common evaluation populations. Each representation retains its own full fitted map and member-p95 basin geometry.','', '| Representation | ICSD | GNoME | MatterGen | MP | JARVIS | Alexandria |','|---|---:|---:|---:|---:|---:|---:|']
for rep in REPS:
 lines.append('| '+rep+' | '+' | '.join(f"{100*by[rep][p]['in_basin_fraction']:.2f}%" for p in pops)+' |')
(OUT/'five_representation_common_support.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
