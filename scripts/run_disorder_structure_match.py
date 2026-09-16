#!/usr/bin/env python3
"""Recover the original GNoME–ICSD disorder-aware structure comparison.

Inputs are an exact-formula candidate-pair CSV and already extracted CIFs.
The CIF root contains gnome/<id>.cif and icsd/<id>.cif. The original three
citation-loop repairs are generated in the output directory if preprocessed
reference files are not supplied under icsd_sanitized/<id>.cif.
This program neither downloads nor decrypts source structures. See
docs/HOW_TO_REPRODUCE.md for inputs and the pinned MatterGen source revision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
from collections import Counter,defaultdict
from pathlib import Path


def sanitize_reference_cif(source, destination):
 """Replay the original citation-loop removal, without modifying source CIFs."""
 s=source.read_bytes().decode('latin1').replace('\r\n','\n').replace('\r','\n')
 start=s.find('loop_\n_citation_id')
 end=s.find('\n_publ_author_name',start)
 if start<0 or end<=start:
  raise ValueError(f'Expected original citation-loop boundaries absent in {source.name}')
 s=s[:start]+s[end+1:]
 destination.parent.mkdir(parents=True,exist_ok=True)
 destination.write_bytes(s.encode('latin1'))


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--manifest',type=Path,required=True,help='Exact-formula GNoME–ICSD candidate-pair CSV')
 parser.add_argument('--cif-root',type=Path,required=True,help='Directory containing gnome/ and icsd/; icsd_sanitized/ is optional')
 parser.add_argument('--mattergen-source',type=Path,required=True,help='MatterGen source root at commit 92423660a8bd70e83679086e88f88596d484dc16')
 parser.add_argument('--output-dir',type=Path,required=True,help='Destination for pair CSV, summary, and parse audit')
 args=parser.parse_args()
 expected_sources={
  'mattergen/evaluation/utils/structure_matcher.py':'c7c909c7db452657593978f6c4ed060fc9a92cd5861eb2f3e78940d933baa834',
  'mattergen/evaluation/utils/globals.py':'ba7f916b331c7c59eafb977978dbf99d9658aeaed9f1648d29528393099a060c',
 }
 for relative,expected_sha in expected_sources.items():
  source=args.mattergen_source/relative
  if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected_sha:
   parser.error(f'MatterGen source does not match the pinned public commit: {relative}')
 sys.path.insert(0,str(args.mattergen_source.resolve()))
 from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher,DefaultOrderedStructureMatcher
 from pymatgen.core.structure_matcher import StructureMatcher,OrderDisorderElementComparator,FrameworkComparator,ElementComparator
 from pymatgen.io.cif import CifParser
 from pymatgen.core import Structure

 O=args.output_dir;S=args.cif_root
 O.mkdir(parents=True,exist_ok=True)
 with args.manifest.open(newline='') as manifest_file:
  manifest=list(csv.DictReader(manifest_file))
 parse_meta={'gnome':{},'icsd':{}}

 def load_default(path):
  with warnings.catch_warnings(record=True) as ww:
   warnings.simplefilter('always'); st=Structure.from_file(path)
  return st,'Structure.from_file defaults',[str(w.message) for w in ww]

 def load_icsd(iid):
  path=S/'icsd'/f'{iid}.cif'
  # These two fully occupied CIFs give explicit Wyckoff multiplicities, but their
  # refined coordinates lie just outside pymatgen's 1e-4 special-position
  # tolerance.  A 5e-4 tolerance restores the CIF-reported multiplicities and
  # formula exactly (and is below the last quoted coordinate uncertainty).
  if iid in {'251076','423677'}:
   with warnings.catch_warnings(record=True) as ww:
    warnings.simplefilter('always')
    ss=CifParser(path,occupancy_tolerance=1.10,site_tolerance=5e-4,frac_tolerance=1e-4,check_cif=False).parse_structures(primitive=False,on_error='raise')
   return ss[0],'CifParser site_tolerance=5e-4 to reproduce CIF Wyckoff multiplicities and reported formula',[str(w.message) for w in ww]
  try:return load_default(path)
  except Exception as first:
   # Minimal occupancy-tolerance escalation, keeping original site tolerance.
   last=first
   for ot in [1.01,1.02,1.05,1.10,1.20]:
    try:
     with warnings.catch_warnings(record=True) as ww:
      warnings.simplefilter('always')
      ss=CifParser(path,occupancy_tolerance=ot,site_tolerance=1e-4,frac_tolerance=1e-4,check_cif=False).parse_structures(primitive=False,on_error='raise')
     if ss:return ss[0],f'CifParser occupancy_tolerance={ot}, site_tolerance=1e-4; original default error: {type(first).__name__}',[str(w.message) for w in ww]
    except Exception as e:last=e
   # Three CIFs contain a malformed unrelated citation loop and approximate special coordinates.
   clean=S/'icsd_sanitized'/f'{iid}.cif'
   if not clean.exists() and iid in {'247182','247184','247187'}:
    clean=O/'icsd_sanitized'/f'{iid}.cif'
    sanitize_reference_cif(path,clean)
   if clean.exists():
    with warnings.catch_warnings(record=True) as ww:
     warnings.simplefilter('always')
     ss=CifParser(clean,occupancy_tolerance=1.10,site_tolerance=0.002,frac_tolerance=1e-4,check_cif=False).parse_structures(primitive=False,on_error='raise')
    if ss:return ss[0],f'scratch CIF with malformed citation loop removed; occupancy_tolerance=1.10, site_tolerance=0.002; original default error: {type(first).__name__}',[str(w.message) for w in ww]
   raise last

 def st_meta(st,method,warns,path):
  return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'parse_method':method,'warnings':warns,'n_sites':len(st),'num_atoms':float(st.composition.num_atoms),'is_ordered':bool(st.is_ordered),'composition':str(st.composition),'reduced_formula':st.composition.reduced_formula,'volume':float(st.volume),'density':float(st.density)}

 gstruct={};istruct={}
 for gid in sorted({r['gnome_id'] for r in manifest}):
  p=S/'gnome'/f'{gid}.cif';st,method,ww=load_default(p);gstruct[gid]=st;parse_meta['gnome'][gid]=st_meta(st,method,ww,p)
 for iid in sorted({r['icsd_id'] for r in manifest},key=int):
  p=S/'icsd'/f'{iid}.cif';st,method,ww=load_icsd(iid);istruct[iid]=st;parse_meta['icsd'][iid]=st_meta(st,method,ww,p)
 print('parsed',len(gstruct),'GNoME and',len(istruct),'ICSD; ordered ICSD',sum(x.is_ordered for x in istruct.values()),flush=True)
 (O/'structure_match_parse_audit_rescued.json').write_text(json.dumps(parse_meta,indent=2))

 # MatterGen publication/repository rule is primary. Other matchers expose its permissiveness.
 matchers={
  'mattergen_default_disordered':DefaultDisorderedStructureMatcher(),
  'mattergen_default_ordered_oxidation_stripped':DefaultOrderedStructureMatcher(),
  'pymatgen_order_disorder_direct_oxidation_stripped':StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,comparator=OrderDisorderElementComparator(),attempt_supercell=True,allow_subset=True),
  'pymatgen_framework_geometry':StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,comparator=FrameworkComparator(),attempt_supercell=True,allow_subset=True),
 }
 rows=[];t0=time.time()
 for idx,r in enumerate(manifest,1):
  g=gstruct[r['gnome_id']];i=istruct[r['icsd_id']]
  x=dict(r);x.update(gnome_parsed_formula=g.composition.reduced_formula,gnome_n_sites=len(g),gnome_ordered=g.is_ordered,icsd_parsed_formula=i.composition.reduced_formula,icsd_n_sites=len(i),icsd_ordered=i.is_ordered,icsd_parse_method=parse_meta['icsd'][r['icsd_id']]['parse_method'])
  for name,matcher in matchers.items():
   t=time.time()
   try:
    if name.endswith('_oxidation_stripped'):
     gg=g.copy();gg.remove_oxidation_states();ii=i.copy();ii.remove_oxidation_states()
    else:
     gg,ii=g,i
    v=bool(matcher.fit(gg,ii));err=''
   except Exception as e:
    v=False;err=f'{type(e).__name__}: {e}'
   x[name+'_match']=int(v);x[name+'_error']=err;x[name+'_seconds']=time.time()-t
  rows.append(x)
  print(idx,'/',len(manifest),r['gnome_id'],r['icsd_id'],'partial',r['icsd_partial_occupancy'],'matches',[x[n+'_match'] for n in matchers], 'sec',round(sum(x[n+'_seconds'] for n in matchers),3),flush=True)

 fieldnames=list(rows[0])
 pairs_path=O/'gnome_icsd_disorder_structure_match_pairs.csv'
 with pairs_path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)

 # Candidate-level result for all-index and post-1980 reference scopes.
 def candidate_summary(scope_rows,name):
  by=defaultdict(list)
  for r in scope_rows:by[r['gnome_id']].append(r)
  out={}
  for gid,rr in by.items():
   out[gid]={'formula':rr[0]['gnome_formula'],'is_train':rr[0]['gnome_is_train'],'n_reference_records':len(rr),'n_full_occupancy_references':sum(x['icsd_partial_occupancy']=='0' for x in rr),'n_partial_occupancy_references':sum(x['icsd_partial_occupancy']=='1' for x in rr),'matched_reference_ids':[x['icsd_id'] for x in rr if x[name+'_match']],'match':any(x[name+'_match'] for x in rr),'errors':[{'icsd_id':x['icsd_id'],'error':x[name+'_error']} for x in rr if x[name+'_error']]}
  return out
 summary={'method':{
  'mattergen_source_checkout':str(args.mattergen_source.resolve()),'mattergen_git_commit':'92423660a8bd70e83679086e88f88596d484dc16','mattergen_matcher':'DefaultDisorderedStructureMatcher: ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True, OrderDisorderElementComparator, attempt_supercell=True, allow_subset=True, composition gate atol=0.01 rtol=0.1; ordered approximant heuristic radius difference <=0.3 and electronegativity difference <=1.0','direction':'GNoME proposal first, ICSD reference second, as MatterGen DatasetMatcher does','strict_sensitivity':'MatterGen DefaultOrderedStructureMatcher at pymatgen defaults after oxidation-state removal','direct_disorder_sensitivity':'base pymatgen StructureMatcher with MatterGen geometric/comparator settings and oxidation states removed, but without MatterGen composition gate/ordered-alloy conversion','framework_sensitivity':'same geometry settings while ignoring species; this is diagnostic and is not a structure-identity result'},
  'parse':{'gnome_n':len(gstruct),'gnome_ordered_n':sum(s.is_ordered for s in gstruct.values()),'icsd_n':len(istruct),'icsd_ordered_n':sum(s.is_ordered for s in istruct.values()),'icsd_disordered_n':sum(not s.is_ordered for s in istruct.values()),'icsd_rescued_n':sum(not m['parse_method'].startswith('Structure.from_file') for m in parse_meta['icsd'].values())},
  'pair_counts':{},'candidate_counts':{},'candidates':{},'runtime_seconds':time.time()-t0}
 scopes={'all_index':rows,'post1980_through2015':[r for r in rows if r['post1980_through2015']=='1']}
 for scope,rr in scopes.items():
  summary['pair_counts'][scope]={'pairs':len(rr),'full_occupancy_pairs':sum(r['icsd_partial_occupancy']=='0' for r in rr),'partial_occupancy_pairs':sum(r['icsd_partial_occupancy']=='1' for r in rr)}
  summary['candidate_counts'][scope]={'candidates':len({r['gnome_id'] for r in rr})}
  summary['candidates'][scope]={}
  for name in matchers:
   cs=candidate_summary(rr,name);matched=sum(x['match'] for x in cs.values());errs=sum(bool(x['errors']) for x in cs.values())
   fullmatched=sum(any(x[name+'_match'] and x['icsd_partial_occupancy']=='0' for x in rr if x['gnome_id']==gid) for gid in cs)
   partialmatched=sum(any(x[name+'_match'] and x['icsd_partial_occupancy']=='1' for x in rr if x['gnome_id']==gid) for gid in cs)
   summary['candidate_counts'][scope][name]={'matched_candidates':matched,'match_rate_among_exact_formula_candidates':matched/len(cs),'matched_to_any_full_occupancy_reference':fullmatched,'matched_to_any_partial_occupancy_reference':partialmatched,'candidates_with_matcher_errors':errs}
   summary['candidates'][scope][name]=cs
 summary_path=O/'gnome_icsd_disorder_structure_match_summary.json';summary_path.write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary['candidate_counts'],indent=2),flush=True);print('WROTE',pairs_path,summary_path,flush=True)


if __name__=='__main__':
 main()
