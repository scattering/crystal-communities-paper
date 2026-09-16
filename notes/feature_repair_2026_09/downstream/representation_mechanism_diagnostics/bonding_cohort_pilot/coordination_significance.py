"""Post-hoc multiplicity sensitivity for the four coordination contrasts."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import postprocess as pp
P=Path(__file__).resolve().parent;R=P/'results';rows=list(csv.DictReader((R/'per_target.csv').open()));nboot=20000
out={'status':'post-hoc significance follow-up; original analyses preserved','metric':'median within-pair GNoME minus comparator difference in per-structure mean weighted geometric CrystalNN coordination','n_bootstrap':nboot,'family':'four coordination source comparisons; does not account for selection of coordination after examining multiple physical summaries','multiplicity':'Bonferroni: each percentile interval has nominal 98.75% coverage (quantiles 0.00625,0.99375), targeting approximate simultaneous 95% coverage across four comparisons','resampling':'same connected components as the prior matched analysis; reused GNoME/ICSD IDs retained together','seed':20260909,'source_sha256':hashlib.sha256((R/'per_target.csv').read_bytes()).hexdigest(),'comparisons':{}}
for source in pp.COMPARATORS:
 pairs=pp.make_pairs(rows,source);clusters=pp.pair_clusters(pairs);v=np.array([float(x['gnome']['mean_weighted_cn'])-float(x['other']['mean_weighted_cn']) for x in pairs]);rng=pp.deterministic_rng('coordination_multiplicity_followup|'+source)
 bs=np.array([np.median(v[pp.resample_indices(clusters,rng)]) for _ in range(nboot)])
 result={'n_pairs':len(pairs),'n_clusters':len(clusters),'median_paired_difference':float(np.median(v)),'pointwise_ci95':np.quantile(bs,[.025,.975]).tolist(),'bonferroni_ci9875':np.quantile(bs,[.00625,.99375]).tolist(),'gnome_higher_pairs':int(np.sum(v>1e-10)),'gnome_lower_pairs':int(np.sum(v<-1e-10)),'tied_pairs':int(np.sum(np.abs(v)<=1e-10))}
 out['comparisons'][source]=result;print(source,result)
(R/'coordination_significance.json').write_text(json.dumps(out,indent=2)+'\n')
