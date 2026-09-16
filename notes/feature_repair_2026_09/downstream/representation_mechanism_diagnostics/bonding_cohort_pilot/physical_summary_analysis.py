"""Post-hoc physical description of the frozen contact samples; no new extraction."""
from pathlib import Path
import json,csv,numpy as np
import postprocess as pp
P=Path(__file__).resolve().parent;R=P/'results'
rows=list(csv.DictReader((R/'per_target.csv').open()))
FIELDS=['mean_weighted_cn','mean_relative_contact','mean_electronegativity_difference','mean_covalent_radius','volume_per_site']
for row in rows:
 with np.load(R/'cache'/f"{row['query_tag']}.npz") as z:c={k:z[k] for k in z.files}
 w=c['w']/c['w'].sum();i,j=c['i'],c['j'];ratio=c['d']/(c['rcov'][i]+c['rcov'][j])
 row.update(mean_relative_contact=float(w@ratio),mean_electronegativity_difference=float(w@abs(c['X'][i]-c['X'][j])),same_element_contact_fraction=float(w@(c['z'][i]==c['z'][j])),mean_covalent_radius=float(c['rcov'].mean()))
fields=['source','material_id','matched_gnome_id','cell']+FIELDS+['same_element_contact_fraction']
with (R/'physical_contact_summaries.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])
out={'status':'post-hoc descriptive diagnostic from cached contacts; not a preregistered primary endpoint','matched_comparisons':{}}
for source in pp.COMPARATORS:
 pairs=pp.make_pairs(rows,source);clusters=pp.pair_clusters(pairs)
 out['matched_comparisons'][source]={f:pp.metric_comparison(pairs,clusters,f,source) for f in FIELDS}
(R/'physical_contact_comparisons.json').write_text(json.dumps(out,indent=2)+'\n')
for source,d in out['matched_comparisons'].items():
 print(source,{f:{k:d[f][k] for k in ['gnome_median','comparator_median','median_paired_difference','median_paired_difference_ci95']} for f in ['mean_weighted_cn','mean_relative_contact']})
