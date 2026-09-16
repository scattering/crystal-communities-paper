"""Compare the same pre-experiment A-Lab outcomes across frozen representations.

No fitting, threshold selection, continuous-score extension, or outcome relabeling.
"""
from __future__ import annotations
import csv, hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats.contingency import odds_ratio

HERE=Path(__file__).resolve().parent
DOWNSTREAM=HERE.parent
TARGETS=DOWNSTREAM/'alab_mp_targets/source/alab_targets.csv'
MODELS={
 'CrystalWeave':DOWNSTREAM/'alab_mp_targets/output/target_projection_records.csv',
 'Graphlet (CrystalNN)':HERE/'graphlet_crystalnn/projection_records.csv',
 'Graphlet (VoronoiNN)':HERE/'graphlet_voronoinn/projection_records.csv',
 'AMD-100':HERE/'amd/projection_records.csv',
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
def boolean(x):
 if isinstance(x,bool):return x
 assert str(x).lower() in ('true','false','1','0'),x
 return str(x).lower() in ('true','1')
def number(x):
 x=float(x)
 return None if math.isnan(x) else 'Infinity' if x==math.inf else '-Infinity' if x==-math.inf else x
def wilson(k,n):
 if not n:return None
 z=stats.norm.ppf(.975);p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;e=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [float(c-e),float(c+e)]
def evaluate(records,ids):
 rows=[records[i] for i in ids]
 a=sum(r['corrected_outcome']=='made' and boolean(r['in_basin']) for r in rows)
 b=sum(r['corrected_outcome']=='made' and not boolean(r['in_basin']) for r in rows)
 c=sum(r['corrected_outcome']=='not_obtained' and boolean(r['in_basin']) for r in rows)
 d=sum(r['corrected_outcome']=='not_obtained' and not boolean(r['in_basin']) for r in rows)
 tab=np.array([[a,b],[c,d]],dtype=np.int64)
 odds=odds_ratio(tab,kind='conditional');ci=odds.confidence_interval(.95)
 return {'n_definitive':int(tab.sum()),'n_made':a+b,'n_not_obtained':c+d,'table_rows_made_not_obtained_cols_in_frontier':tab.tolist(),
  'made_in_basin':a,'n_in_basin':a+c,'made_frontier':b,'n_frontier':b+d,
  'made_fraction_in_basin':a/(a+c) if a+c else None,'made_fraction_frontier':b/(b+d) if b+d else None,
  'made_fraction_in_basin_wilson95':wilson(a,a+c),'made_fraction_frontier_wilson95':wilson(b,b+d),
  'conditional_exact_odds_ratio':number(odds.statistic),'conditional_exact_95ci':[number(ci.low),number(ci.high)],
  'sample_cross_product_odds_ratio':number(stats.fisher_exact(tab).statistic),
  'fisher_exact_p_two_sided':float(stats.fisher_exact(tab,alternative='two-sided').pvalue),
  'fisher_exact_p_one_sided_positive':float(stats.fisher_exact(tab,alternative='greater').pvalue),
  'classification_is_constant':not(a+c) or not(b+d)}
def main():
 targets=readcsv(TARGETS);truth={r['mp_id']:r for r in targets};assert len(targets)==len(truth)==57
 assert Counter(r['corrected_outcome'] for r in targets)=={'made':36,'not_obtained':15,'offline_recovery':2,'inconclusive':4}
 definitive=[r['mp_id'] for r in targets if r['corrected_outcome'] in ('made','not_obtained')];assert len(definitive)==51
 records={};out={}
 for name,path in MODELS.items():
  rr=readcsv(path);byid={r['mp_id']:r for r in rr};assert len(rr)==len(byid) and set(byid)<=set(truth),name
  for i,r in byid.items():assert r['corrected_outcome']==truth[i]['corrected_outcome'],(name,i)
  records[name]=byid;ids=[i for i in definitive if i in byid]
  assert ids,(name,'no definitive outcome features')
  out[name]={'projection_path':str(path.relative_to(DOWNSTREAM)),'projection_sha256':sha(path),'n_targets_represented':len(byid),'missing_target_ids':sorted(set(truth)-set(byid)),
   'represented_by_outcome':dict(Counter(r['corrected_outcome'] for r in rr)),'own_successful_support':evaluate(byid,ids)}
 common=[i for i in definitive if all(i in x for x in records.values())]
 for name,rr in records.items():
  out[name]['common_successful_support']=evaluate(rr,common)
  out[name]['n_basin_assignments_agree_with_crystalweave_on_common']=sum(boolean(rr[i]['in_basin'])==boolean(records['CrystalWeave'][i]['in_basin']) for i in common)
 assert out['CrystalWeave']['own_successful_support']['table_rows_made_not_obtained_cols_in_frontier']==[[28,8],[4,11]]
 assert abs(out['CrystalWeave']['own_successful_support']['conditional_exact_odds_ratio']-9.103087538769064)<1e-10
 # The three requested alternative-map checks form one exploratory robustness family.
 others=[n for n in MODELS if n!='CrystalWeave'];order=sorted(others,key=lambda n:out[n]['common_successful_support']['fisher_exact_p_two_sided']);last=0
 for rank,name in enumerate(order):
  last=max(last,min(1,(len(order)-rank)*out[name]['common_successful_support']['fisher_exact_p_two_sided']))
  out[name]['common_successful_support']['holm_adjusted_p_three_alternative_maps']=last
 result={'status':'complete','protocol':{'target_source':'Pre-experiment MP snapshot 2022-10-28; corrected published A-Lab outcomes.','targets_sha256':sha(TARGETS),
  'n_targets':57,'primary_n':51,'success':'made','failure':'not_obtained','excluded_outcomes':{'offline_recovery':2,'inconclusive':4},
  'coordinates_and_bases':'Existing frozen full-record maps; AMD uses repaired eligible positive-radius basins. No refitting or threshold selection on A-Lab.',
  'effect':'Conditional maximum-likelihood odds ratio with exact 95% confidence interval, table rows made/not_obtained and columns in/frontier.',
  'significance':'Two-sided Fisher exact p; nominal p plus Holm adjustment across the three requested alternative maps. No inference of between-map superiority from differences in significance.',
  'scope':'Representation sensitivity within one selected campaign, not an additional experimental validation cohort.'},
  'common_definitive_ids':common,'n_common_definitive':len(common),'representations':out,'software':{'numpy':np.__version__,'scipy':__import__('scipy').__version__},'producer_sha256':sha(Path(__file__))}
 (HERE/'outcome_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 flat=[]
 for name,x in out.items():
  y=x['common_successful_support'];flat.append({'representation':name,'n_targets_represented':x['n_targets_represented'],**{k:v for k,v in y.items() if not isinstance(v,(dict,list))},'exact_ci_low':y['conditional_exact_95ci'][0],'exact_ci_high':y['conditional_exact_95ci'][1]})
 fields=list(dict.fromkeys(k for r in flat for k in r))
 with (HERE/'outcome_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
 def fmt(x):return 'undefined' if x is None else str(x) if isinstance(x,str) else f'{x:.4g}'
 lines=['# A-Lab outcome comparison across frozen representations','',f'Complete targets: 57; definitive outcomes: 51; common successfully represented definitive targets: {len(common)}.','',
 '| Representation | Targets represented | Made / in basin | Made / frontier | Conditional OR [exact 95% CI] | Fisher two-sided p | Holm p (3 alternatives) |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for name,x in out.items():
  y=x['common_successful_support'];lo,hi=y['conditional_exact_95ci'];lines.append(f"| {name} | {x['n_targets_represented']}/57 | {y['made_in_basin']}/{y['n_in_basin']} | {y['made_frontier']}/{y['n_frontier']} | {fmt(y['conditional_exact_odds_ratio'])} [{fmt(lo)}, {fmt(hi)}] | {fmt(y['fisher_exact_p_two_sided'])} | {fmt(y.get('holm_adjusted_p_three_alternative_maps'))} |")
 lines+=['','All rates use pre-experiment target structures and the same corrected definitive outcomes. Full-record coordinate transforms, partitions and basin radii are frozen; no outcome-based optimization is performed. The maps use their existing ICSD reference populations and do not become independently validated by sharing these targets. Differences in within-map p values do not establish differences between representation performance.','']
 (HERE/'outcome_comparison.md').write_text('\n'.join(lines));print('\n'.join(lines))
if __name__=='__main__':main()
