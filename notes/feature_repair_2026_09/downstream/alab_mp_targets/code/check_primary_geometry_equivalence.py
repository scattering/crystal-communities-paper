#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'source/mp_2022_10_28_summary_target_docs.json'
SS=ROOT/'source/mp_2022_10_28_summary_source_manifest.json'
M=ROOT/'source/mp_2022_10_28_materials_target_docs.json'
MS=ROOT/'source/mp_2022_10_28_materials_source_manifest.json'
OUT=ROOT/'verification/primary_materials_geometry_equivalence.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
s=json.load(S.open()); m=json.load(M.open()); ss=json.load(SS.open()); ms=json.load(MS.open())
assert set(s)==set(m) and len(s)==57
rows=[]
for mid in sorted(s):
 a=s[mid]['structure']; b=m[mid]['structure']
 lat=a['lattice']['matrix']==b['lattice']['matrix']
 n=len(a['sites'])==len(b['sites'])
 species=n and all(x['species']==y['species'] for x,y in zip(a['sites'],b['sites']))
 abc=n and all(x['abc']==y['abc'] for x,y in zip(a['sites'],b['sites']))
 xyz=n and all(x['xyz']==y['xyz'] for x,y in zip(a['sites'],b['sites']))
 siteprops=n and all(x.get('properties')==y.get('properties') for x,y in zip(a['sites'],b['sites']))
 ca=json.loads(json.dumps(a)); cb=json.loads(json.dumps(b)); ca['lattice'].pop('pbc',None); cb['lattice'].pop('pbc',None); ca.pop('properties',None); cb.pop('properties',None)
 clean=ca==cb
 assert lat and n and species and abc and xyz and siteprops and clean
 so=next(x['source_object'] for x in ss['targets'] if x['mp_id']==mid); mo=next(x['source_object'] for x in ms['targets'] if x['mp_id']==mid)
 rows.append({'mp_id':mid,'summary_object':so,'materials_object':mo,'n_sites':len(a['sites']),'exact_lattice_matrix':lat,'exact_ordered_species':species,
              'exact_fractional_coordinates':abc,'exact_cartesian_coordinates':xyz,'exact_site_properties':siteprops,
              'structure_dict_equal_after_removing_summary_only_empty_properties_and_pbc':clean})
res={'result':'All 57 geometries are byte-for-value identical in lattice, ordered species, fractional coordinates, Cartesian coordinates, and site properties.',
 'n_targets':57,'n_exact_geometry_matches':sum(all(r[k] for k in ['exact_lattice_matrix','exact_ordered_species','exact_fractional_coordinates','exact_cartesian_coordinates','exact_site_properties']) for r in rows),
 'raw_structure_dict_equal_n':sum(s[mid]['structure']==m[mid]['structure'] for mid in s),
 'raw_dict_difference_explanation':'summary structures add lattice.pbc=[true,true,true] and an empty top-level structure.properties object; materials structures omit those serialization metadata fields.',
 'clean_structure_dict_equal_n':sum(r['structure_dict_equal_after_removing_summary_only_empty_properties_and_pbc'] for r in rows),
 'summary_collection':{'uri':ss['source'],'n_source_objects':len(ss['source_objects']),'compressed_bytes':sum(x['bytes'] for x in ss['source_objects']),'docs_sha256':sha(S),'manifest_sha256':sha(SS)},
 'primary_materials_collection':{'uri':ms['source'],'n_source_objects':len(ms['source_objects']),'compressed_bytes':sum(x['bytes'] for x in ms['source_objects']),'docs_sha256':sha(M),'manifest_sha256':sha(MS)},
 'source_object_manifests_include':'S3 object key, ETag, Content-Length, Last-Modified, downloaded byte count, and SHA-256.',
 'targets':rows,'script_sha256':sha(Path(__file__))}
OUT.write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({k:v for k,v in res.items() if k!='targets'},indent=2))
