#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import optimize, stats
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import StratifiedTable

ROOT=Path(__file__).resolve().parents[1]
TARGET_CSV=ROOT/'output/target_projection_records.csv'
PAIR_CSV=ROOT/'output/target_vs_refinement_records.csv'
REF_CSV=ROOT.parent/'alab/alab_validation_records.csv'
SCORE_SUMMARY=ROOT.parent/'accessibility/structural_accessibility_summary.json'
OUT=ROOT/'verification/outcome_sensitivity.json'

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def B(x): return str(x).lower()=='true'
def auc_lower(x,y):
 x=np.asarray(x,float); y=np.asarray(y,float)
 return float(((x[:,None]<y).sum()+.5*(x[:,None]==y).sum())/(x.size*y.size))
def boot_auc(x,y,n=50000,seed=0):
 x=np.asarray(x,float); y=np.asarray(y,float); rng=np.random.default_rng(seed); vals=[]
 for k in range(0,n,1000):
  z=min(1000,n-k); a=x[rng.integers(x.size,size=(z,x.size))]; b=y[rng.integers(y.size,size=(z,y.size))]
  vals.append(((a[:,:,None]<b[:,None,:]).sum((1,2))+.5*(a[:,:,None]==b[:,None,:]).sum((1,2)))/(x.size*y.size))
 q=np.quantile(np.concatenate(vals),[.025,.975]); return [float(v) for v in q]
def boot_diff(x,y,fn=np.mean,n=50000,seed=0):
 x=np.asarray(x,float); y=np.asarray(y,float); rng=np.random.default_rng(seed); vals=np.empty(n)
 for i in range(n):vals[i]=fn(rng.choice(x,x.size,replace=True))-fn(rng.choice(y,y.size,replace=True))
 return [float(v) for v in np.quantile(vals,[.025,.975])]
def fisher_or_exact_ci(tab,alpha=.05):
 tab=np.asarray(tab,int); a,b,c,d=tab.ravel(); M=int(tab.sum()); n=int(a+c); N=int(a+b)
 lo=max(0,N-(M-n)); hi=min(N,n)
 def dist(z): return stats.nchypergeom_fisher(M,n,N,math.exp(z))
 def root(fun): return math.exp(optimize.brentq(fun,-50,50,xtol=1e-13))
 cmle=0.0 if a==lo else math.inf if a==hi else root(lambda z:dist(z).mean()-a)
 low=0.0 if a==lo else root(lambda z:dist(z).sf(a-1)-alpha/2)
 high=math.inf if a==hi else root(lambda z:dist(z).cdf(a)-alpha/2)
 sample=(a*d)/(b*c) if b*c else math.inf if a*d else float('nan')
 one=stats.fisher_exact(tab,alternative='greater'); two=stats.fisher_exact(tab,alternative='two-sided')
 finite=lambda v: float(v) if math.isfinite(v) else None
 return {'table_rows_success_failure_cols_in_basin_frontier':tab.tolist(),'sample_cross_product_odds_ratio':finite(sample),
         'conditional_maximum_likelihood_odds_ratio':finite(cmle),'conditional_exact_95ci':[finite(low),finite(high)],
         'fisher_exact_p_one_sided_success_more_in_basin':float(one[1]),'fisher_exact_p_two_sided':float(two[1])}
def wilson(k,n,z=1.959963984540054):
 p=k/n; den=1+z*z/n; cen=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [cen-half,cen+half]
def compare(rows,success,failure,scorekey,seed):
 sr=[r for r in rows if r['corrected_outcome'] in success]; fr=[r for r in rows if r['corrected_outcome'] in failure]
 x=np.array([float(r[scorekey]) for r in sr]); y=np.array([float(r[scorekey]) for r in fr]); xi=np.array([B(r['in_basin']) for r in sr]); yi=np.array([B(r['in_basin']) for r in fr])
 auc=auc_lower(x,y); ci=boot_auc(x,y,seed=seed); u=stats.mannwhitneyu(x,y,alternative='less',method='exact')
 tab=[[int(xi.sum()),int((~xi).sum())],[int(yi.sum()),int((~yi).sum())]]
 ff=fisher_or_exact_ci(tab)
 # outcome fractions conditional on location
 nin=int(xi.sum()+yi.sum()); nout=int((~xi).sum()+(~yi).sum()); kin=int(xi.sum()); kout=int((~xi).sum())
 return {'success_outcomes':list(success),'failure_outcomes':list(failure),'n_success':len(sr),'n_failure':len(fr),
         'success_score_mean':float(x.mean()),'failure_score_mean':float(y.mean()),'mean_difference_success_minus_failure':float(x.mean()-y.mean()),
         'mean_difference_bootstrap_95ci':boot_diff(x,y,seed=seed+10),'success_score_median':float(np.median(x)),'failure_score_median':float(np.median(y)),
         'auc_lower_score_predicts_success':auc,'auc_stratified_bootstrap_95ci':ci,
         'rank_biserial_positive_means_success_lower':2*auc-1,'rank_biserial_bootstrap_95ci':[2*ci[0]-1,2*ci[1]-1],
         'mann_whitney_u':float(u.statistic),'mann_whitney_exact_p_one_sided_success_lower':float(u.pvalue),
         **ff,'success_rate_among_in_basin':kin/nin,'success_rate_among_in_basin_wilson_95ci':wilson(kin,nin),
         'success_rate_among_frontier':kout/nout,'success_rate_among_frontier_wilson_95ci':wilson(kout,nout),
         'in_basin_risk_difference_success_minus_failure':float(xi.mean()-yi.mean())}

def paired_summary(pair,target_by,ref_by,year,mu,sigma,seed):
 vals=[]
 for p in pair:
  mid=p['mp_id']; t=target_by[mid]; r=ref_by[mid]
  def score(q):
   raw=math.log1p(float(q['nearest_centroid_distance'])/max(float(q['community_median_distance']),1e-6))-.5*math.log1p(int(q['community_size']))-.5*math.log1p(max(year-int(q['community_birth_year']),0))
   return (raw-mu)/sigma
  vals.append((score(t),score(r),B(p['target_in_basin']),B(p['refined_in_basin'])))
 ts=np.array([v[0] for v in vals]); rs=np.array([v[1] for v in vals]); d=ts-rs
 rng=np.random.default_rng(seed); bm=[]; bmed=[]
 for _ in range(50000):
  q=rng.choice(d,d.size,replace=True); bm.append(q.mean()); bmed.append(np.median(q))
 both=sum(a and b for _,_,a,b in vals); ton=sum(a and not b for _,_,a,b in vals); ron=sum((not a) and b for _,_,a,b in vals); neither=sum((not a) and (not b) for _,_,a,b in vals)
 discord=ton+ron; mcp=1.0 if discord==0 else stats.binomtest(min(ton,ron),discord,.5,alternative='two-sided').pvalue
 wil=stats.wilcoxon(d,alternative='two-sided'); pear=stats.pearsonr(ts,rs); spear=stats.spearmanr(ts,rs)
 return {'observation_year':year,'n':len(d),'mean_target_minus_refined_score':float(d.mean()),'mean_delta_paired_bootstrap_95ci':[float(x) for x in np.quantile(bm,[.025,.975])],
         'median_target_minus_refined_score':float(np.median(d)),'median_delta_paired_bootstrap_95ci':[float(x) for x in np.quantile(bmed,[.025,.975])],
         'mean_absolute_score_difference':float(np.abs(d).mean()),'max_absolute_score_difference':float(np.abs(d).max()),
         'paired_wilcoxon_statistic':float(wil.statistic),'paired_wilcoxon_p_two_sided':float(wil.pvalue),
         'pearson_r':float(pear[0]),'spearman_rho':float(spear[0]),
         'basin_table_rows_target_in_out_cols_refined_in_out':[[both,ton],[ron,neither]],'mcnemar_exact_p_two_sided':float(mcp)}

def main():
 rows=list(csv.DictReader(TARGET_CSV.open())); pair=list(csv.DictReader(PAIR_CSV.open())); refs=list(csv.DictReader(REF_CSV.open())); ref_by={r['mp_id']:r for r in refs}; target_by={r['mp_id']:r for r in rows}
 mom=json.load(SCORE_SUMMARY.open()); mu=mom['icsd_raw_mu']; sigma=mom['icsd_raw_sigma']
 for r in rows:
  raw=math.log1p(float(r['nearest_centroid_distance'])/max(float(r['community_median_distance']),1e-6))-.5*math.log1p(int(r['community_size']))-.5*math.log1p(max(2022-int(r['community_birth_year']),0))
  r['score_2022']=(raw-mu)/sigma; r['score_2023']=float(r['accessibility_score'])
 defs={'made_vs_not_obtained':(('made',),('not_obtained',)),
       'autonomous_made_vs_autonomous_failure':(('made',),('not_obtained','offline_recovery')),
       'eventually_obtained_vs_not_obtained':(('made','offline_recovery'),('not_obtained',))}
 comparisons={}
 for year,key in [(2022,'score_2022'),(2023,'score_2023')]:
  comparisons[str(year)]={name:compare(rows,*groups,key,20260905+100*(year-2022)+i) for i,(name,groups) in enumerate(defs.items())}
 primary=[r for r in rows if r['corrected_outcome'] in ('made','not_obtained')]
 # Leave one assigned-community cluster out.
 loo=[]
 for c in sorted({int(r['assigned_community']) for r in primary}):
  q=[r for r in primary if int(r['assigned_community'])!=c]
  ss=[r for r in q if r['corrected_outcome']=='made']; ff=[r for r in q if r['corrected_outcome']=='not_obtained']
  tab=[[sum(B(r['in_basin']) for r in ss),sum(not B(r['in_basin']) for r in ss)],
       [sum(B(r['in_basin']) for r in ff),sum(not B(r['in_basin']) for r in ff)]]
  fi=fisher_or_exact_ci(tab)
  loo.append({'omitted_community':c,'n_omitted':len(primary)-len(q),
              'auc':auc_lower([r['score_2023'] for r in ss],[r['score_2023'] for r in ff]),
              'fisher_one_sided_p':fi['fisher_exact_p_one_sided_success_more_in_basin'],
              'fisher_two_sided_p':fi['fisher_exact_p_two_sided'],
              'sample_or':fi['sample_cross_product_odds_ratio']})
 # Community-cluster bootstrap for risk difference and AUC (primary comparison).
 clusters=defaultdict(list)
 for r in primary:clusters[int(r['assigned_community'])].append(r)
 ids=sorted(clusters); rng=np.random.default_rng(20260909); brd=[]; bauc=[]
 for _ in range(30000):
  rr=[]
  for c in rng.choice(ids,len(ids),replace=True):rr.extend(clusters[int(c)])
  s=[r for r in rr if r['corrected_outcome']=='made']; f=[r for r in rr if r['corrected_outcome']=='not_obtained']
  if not s or not f:continue
  brd.append(np.mean([B(r['in_basin']) for r in s])-np.mean([B(r['in_basin']) for r in f]))
  bauc.append(auc_lower([r['score_2023'] for r in s],[r['score_2023'] for r in f]))
 # Cluster-robust logit, unadjusted and adjusted for phosphate and log nsites.
 y=np.array([r['corrected_outcome']=='made' for r in primary],float); basin=np.array([B(r['in_basin']) for r in primary],float); groups=np.array([int(r['assigned_community']) for r in primary])
 phosphate=np.array(['P' in __import__('re').findall(r'[A-Z][a-z]?',r['formula']) for r in primary],float); logn=np.log(np.array([float(r['n_sites']) for r in primary]))
 def clogit(X,names):
  fit=sm.GLM(y,sm.add_constant(X),family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':groups,'use_correction':True})
  idx=names.index('in_basin')+1; co=float(fit.params[idx]); se=float(fit.bse[idx]); z=co/se
  return {'predictors':names,'in_basin_log_odds':co,'cluster_robust_se':se,'odds_ratio':math.exp(co),'wald_95ci':[math.exp(co-1.95996398454*se),math.exp(co+1.95996398454*se)],'wald_z':z,'wald_p_two_sided':float(2*stats.norm.sf(abs(z))),'n_community_clusters':len(set(groups)),'converged':bool(fit.converged)}
 clog={'unadjusted':clogit(basin[:,None],['in_basin']),'adjusted_phosphate_log_nsites':clogit(np.c_[basin,phosphate,logn],['in_basin','contains_P','log_nsites'])}
 # Exact stratification by phosphate presence.
 strats=[]; tables=[]
 for label,pval in [('non_P',0),('contains_P',1)]:
  q=[r for r,pp in zip(primary,phosphate) if pp==pval]
  tab=[]
  for out in ['made','not_obtained']:
   u=[r for r in q if r['corrected_outcome']==out]; tab.append([sum(B(r['in_basin']) for r in u),sum(not B(r['in_basin']) for r in u)])
  tables.append(np.array(tab,int)); strats.append({'stratum':label,'n':len(q),**fisher_or_exact_ci(tab)})
 # exact one-sided convolution under common null OR=1 conditional on each stratum's margins
 pmf=np.array([1.0]); observed=0
 for tab in tables:
  a,b,c,d=tab.ravel(); M=tab.sum(); n=a+c; N=a+b; lo=max(0,N-(M-n)); hi=min(N,n); xs=np.arange(lo,hi+1); p=stats.hypergeom.pmf(xs,M,n,N)
  arr=np.zeros(hi+1); arr[xs]=p; pmf=np.convolve(pmf,arr); observed+=a
 exact_strat_p=float(pmf[observed:].sum())
 st=StratifiedTable(np.stack(tables,axis=2)); mh=st.test_null_odds()
 robustness={'primary_n':len(primary),'n_assigned_communities':len(ids),
  'leave_one_community_out':{'n_runs':len(loo),'one_sided_fisher_p_range':[min(r['fisher_one_sided_p'] for r in loo),max(r['fisher_one_sided_p'] for r in loo)],
   'two_sided_fisher_p_range':[min(r['fisher_two_sided_p'] for r in loo),max(r['fisher_two_sided_p'] for r in loo)],
   'auc_range':[min(r['auc'] for r in loo),max(r['auc'] for r in loo)],'runs':loo},
  'community_cluster_bootstrap':{'n_resamples_requested':30000,'n_valid':len(brd),'risk_difference_95ci':[float(x) for x in np.quantile(brd,[.025,.975])],
    'auc_95ci':[float(x) for x in np.quantile(bauc,[.025,.975])]},'cluster_robust_logistic':clog,
  'phosphate_stratification':{'strata':strats,'exact_conditional_one_sided_p_common_null':exact_strat_p,'mantel_haenszel_common_odds_ratio':float(st.oddsratio_pooled),
    'mantel_haenszel_chi2':float(mh.statistic),'mantel_haenszel_p_two_sided':float(mh.pvalue)}}
 paired={'year_2022':paired_summary(pair,target_by,ref_by,2022,mu,sigma,20260920),'year_2023':paired_summary(pair,target_by,ref_by,2023,mu,sigma,20260921),
         'same_community_n':sum(B(p['same_community']) for p in pair),'same_community_rate':sum(B(p['same_community']) for p in pair)/len(pair),
         'in_basin_agreement_n':sum(B(p['in_basin_agreement']) for p in pair),'in_basin_agreement_rate':sum(B(p['in_basin_agreement']) for p in pair)/len(pair),
         'structure_matcher_fit_n':sum(B(p['structure_matcher_fit']) for p in pair),'structure_matcher_attempted_n':sum(p['structure_matcher_fit']!='' for p in pair)}
 result={'scope':'Outcome and sensitivity statistics on all 57 public pre-experiment MP target geometries; continuous scores lower means more structurally accessible.',
  'outcome_counts':dict(Counter(r['corrected_outcome'] for r in rows)),'comparisons_by_observation_year':comparisons,
  'robustness_primary_made_vs_not_obtained':robustness,'paired_target_vs_released_refinement':paired,
  'interpretation_guardrails':[
   'Basin membership is structural-distance threshold membership and does not depend on observation year.',
   'Observation-year sensitivity changes only the community-age component of the continuous score; the frozen ICSD normalization is retained so comparisons use the same scale.',
   'The A-Lab cohort was selected for a particular autonomous solid-state workflow, and outcomes within projected communities are not fully independent.',
   'The basin threshold was not designed or preregistered using A-Lab outcomes; these are external supporting associations, not calibrated synthesis probabilities.'
  ],'provenance':{'target_records_sha256':sha256(TARGET_CSV),'paired_records_sha256':sha256(PAIR_CSV),'refined_records_sha256':sha256(REF_CSV),'score_summary_sha256':sha256(SCORE_SUMMARY),'script_sha256':sha256(Path(__file__))}}
 OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'outcome_counts':result['outcome_counts'],'2022':comparisons['2022'],'2023':comparisons['2023'],'robustness':robustness,'paired':paired},indent=2,allow_nan=False))
if __name__=='__main__':main()
