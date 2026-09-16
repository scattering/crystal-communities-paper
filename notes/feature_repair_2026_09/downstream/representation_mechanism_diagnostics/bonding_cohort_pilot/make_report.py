"""Build the comparative scientific report and a compact exportable figure."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;R=P/'results'
j=json.loads((R/'bootstrap_summary.json').read_text());physical=json.loads((R/'physical_contact_comparisons.json').read_text());nested=json.loads((R/'nested_chemistry_check.json').read_text());verify=json.loads((R/'verification.json').read_text())
SOURCES=['mattergen','mp','jarvis','alexandria'];NAMES={'mattergen':'MatterGen','mp':'MP-theoretical','jarvis':'JARVIS','alexandria':'Alexandria'}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(11.8,3.5),sharey=True)
configs=[('Joint contact similarity\nto ICSD','joint_actual','primary','#285e8e'),('Weighted coordination','mean_weighted_cn','descriptive','#7861a9'),('Contact length /\nsummed covalent radii','mean_relative_contact','descriptive','#00877c')]
for ax,(title,field,kind,color) in zip(axes,configs):
 for k,s in enumerate(SOURCES):
  d=j['pair_comparisons'][s]['metrics'][field] if kind=='primary' else physical['matched_comparisons'][s][field]
  x=d['median_paired_difference'];lo,hi=d['median_paired_difference_ci95'];ax.errorbar(x,k,xerr=[[x-lo],[hi-x]],fmt='o',color=color,capsize=3,ms=6,lw=1.6)
 ax.axvline(0,color='#888888',lw=.8,ls='--');ax.set_title(title,pad=12);ax.grid(axis='x',alpha=.14);ax.set_xlabel('GNoME − comparator');ax.set_ylim(3.6,-.6)
axes[0].set_yticks(range(4),[NAMES[s]+' (n='+str(j['pair_comparisons'][s]['n_pairs'])+')' for s in SOURCES]);axes[0].set_xlabel('GNoME − comparator\nPositive: GNoME farther from ICSD');axes[0].set_xlim(-.48,.57);axes[1].set_xlim(-.15,2.0);axes[2].set_xlim(-.18,.045)
fig.suptitle('Direct contact environments in the matched pilot',x=.04,ha='left',fontweight='bold',fontsize=14,y=1.01)
fig.tight_layout(w_pad=2.2)
for ext in ['png','pdf','svg']:fig.savefig(R/('bonding_cohort_comparison.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
lines=['# How GNoME differs from the other computed cohorts','',
'Completed 2026-09-09, Stampede3 job **3484071**. The frozen sample contains **467 targets**: 100 GNoME, 82 MatterGen, 100 MP-theoretical, 85 JARVIS and 100 Alexandria. All targets and 854 unique ICSD reference/comparator structures were extracted successfully. Six metric tests and '+str(verify['passed'])+'/'+str(verify['checks'])+' saved-input/result checks passed.','',
'**The clearest physical distinction is higher geometric coordination in GNoME, accompanied by shorter contacts relative to atomic size in three of the four matched comparisons. The direct combined chemistry–distance–coordination comparison does not establish that GNoME has more experimentally familiar bonding than the other computed libraries as a group.**','',
'## Direct comparison with the other sources','',
'The primary score is a standard multivariate energy statistic between each query’s local contact distribution and an independently geometry-selected ICSD reference. It jointly records contact distance divided by summed covalent radii, the mean and difference of endpoint electronegativities, and the mean and difference of endpoint weighted coordination. Smaller scores mean closer agreement. No universal “valid bond” threshold is imposed.','',
'| Comparator | Matched pairs | GNoME−comparator joint score, median paired difference (95% CI) | Reading |','|---|---:|---:|---|']
readings={'mattergen':'No clear unadjusted separation','mp':'MP is closer to ICSD in the direct comparison','jarvis':'No clear separation','alexandria':'No clear separation'}
for s in SOURCES:
 d=j['pair_comparisons'][s]['metrics']['joint_actual'];lo,hi=d['median_paired_difference_ci95'];lines.append(f"| {NAMES[s]} | {j['pair_comparisons'][s]['n_pairs']} | {d['median_paired_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {readings[s]} |")
lines+=['',
'MP has closer AMD-selected ICSD references as well. In the predeclared limited regression adjustment for reference geometry, element count and site count, the GNoME−MP contrast becomes +0.034 [−0.178, +0.244]. The adjusted MatterGen contrast favors GNoME (−0.263 [−0.542, −0.018]); JARVIS and Alexandria remain unresolved. These sensitivities preclude a general cross-library bonding-familiarity ranking. They do not establish equivalence or a causal explanation of source differences.','',
'## What physically distinguishes the matched GNoME structures','',
'The following are post-hoc descriptive summaries of the frozen contact caches, reported separately from the predeclared primary comparison. Each comparison uses exactly its paired GNoME subset and the same conservative cluster bootstrap. Positive coordination differences mean more geometric neighbors on average; a negative normalized contact-length difference means a shorter contact relative to the participating atoms’ radii.','',
'| Comparator | Difference in mean weighted coordination (95% CI) | Difference in mean normalized contact length (95% CI) |','|---|---:|---:|']
for s in SOURCES:
 ds=physical['matched_comparisons'][s];cells=[]
 for f in ['mean_weighted_cn','mean_relative_contact']:
  d=ds[f];lo,hi=d['median_paired_difference_ci95'];cells.append(f"{d['median_paired_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
 lines.append('| '+NAMES[s]+' | '+' | '.join(cells)+' |')
lines+=['',
'GNoME has larger constituent covalent radii in every matched comparison. Volume per atom is generally unresolved, apart from the JARVIS contrast. Therefore the measured signature is **higher coordination and shorter atom-size-normalized contacts**, rather than a demonstrated increase in bulk density or a specific close-packed prototype. It is conditional on the declared geometric CrystalNN rule and on the matched sample.','',
'![Matched contact comparison](results/bonding_cohort_comparison.png)','',
'The first panel is the predeclared joint score; the other two are descriptive follow-ups. All bars are 95% cluster-bootstrap intervals. The plotted quantity is the median of within-pair differences, not the difference between the two cohort medians.','',
'## What the representations tell us','',
'Graphlet stores separate normalized histograms of elemental properties, contact lengths and triplet descriptors. Its implementation divides each channel by total weight (`graphlet_features.py`, `_histogram`), so it retains distribution shape while omitting absolute contact/triplet totals; coordination can still influence those shapes indirectly. It also does not retain a general joint distribution tying every chemical property to its specific distance and coordination. CrystalWeave explicitly carries coordination fingerprints, pooled chemical/neighbor information and cell descriptors. AMD measures species-blind periodic distances. Magpie-22 expands CrystalWeave’s chemical-property basis, with an unresolved GNoME ordering after the existing cell-mixture control.','',
'The new physical signature identifies concrete properties these descriptions handle differently. High Graphlet occupancy can coexist with higher coordination and altered atom-size-normalized contacts; it should not be interpreted as a direct measure of how experimentally conventional all bonding environments are. The existing feature-block and basin-calibration controls remain the stronger interventions on the occupancy reversal. This pilot does not establish that either of the two physical summary statistics alone causes the sign change.','',
'## Relation to the earlier 8.1% result','',
'The earlier pilot established an 8.1% median shuffle advantage for 40 Graphlet-only targets using an average of separate one-dimensional elemental-property Wasserstein distances. That result remains correct under its stated definition. It was too narrow to support a general bonding-conservation interpretation. The present joint contact comparison shows no GNoME shuffle advantage: its median actual−shuffle delta is +0.020 overall and +0.030 in the Graphlet-only group; only 39/100 and 15/40 targets, respectively, improve over their shuffled controls. This does not imply that shuffling produces physically better crystals. The score measures similarity to a particular independently selected ICSD reference, not stability or bond validity.','',
'A post-hoc nested check holds the energy statistic, ICSD scales and reference IDs fixed and retains only the two electronegativity coordinates. GNoME’s delta is −0.009 [−0.026, +0.004] overall and −0.006 [−0.037, +0.024] in the Graphlet-only stratum. Thus there is no robust GNoME advantage even in that nested chemical comparison. The change from the earlier result cannot be assigned solely to adding distances and coordination; the way distributions are compared also matters. The nested and full-dimensional scores are not an additive decomposition. All primary outputs were preserved.','',
'## Proposed paper framing','',
'> GNoME combines nearly unprecedented element–stoichiometry combinations with local descriptor distributions that often lie within experimental Graphlet basins. A chemistry- and cell-size-matched contact analysis identifies higher geometric coordination than in the four comparator cohorts, with generally shorter contacts relative to atomic size. Direct joint comparisons of chemistry, contact distances and coordination give no uniform GNoME advantage in similarity to ICSD. The representation contrast therefore distinguishes forms of structural precedent rather than establishing a universal ordering of experimental familiarity.','',
'This can be presented as a small comparative SI check with one concise physical-result sentence in the main text. The findings describe the selected releases and their matched support; different stability filters, chemical coverage and generation strategies remain intertwined. No generation ancestry or synthesis outcome is inferred from these contacts. The manuscript/response/Word artifacts have not been rebuilt for this new pilot.','',
'## Reproduction, cost and references','',
'`sample_comparators.py` followed by `map_join.py` regenerates the frozen 467-record sample and saved five-map joins. `protocol.json` was frozen before new outcomes. `run_bonding.py` produced contact caches and primary comparisons; `postprocess.py` supplies the predeclared analysis. `physical_summary_analysis.py` and `nested_chemistry_check.py` are explicitly post-hoc diagnostics. `verify.py` checks the sample, staged code hashes, all null reductions and independently recomputed score examples.','',
'Slurm reports **3 minutes 17 seconds on 48 allocated CPUs: 2.63 allocated core-hours**. Summed actual CPU time was 30 minutes; runner wall time was 186.9 seconds. No DFT or map fitting was performed. Passwords were not persisted; only derived ICSD data were retrieved.','',
'Covalent radii: [Cordero et al., 2008](https://doi.org/10.1039/B801115J). Distribution statistic: [Székely and Rizzo, 2013](https://doi.org/10.1016/j.jspi.2013.03.018). These are published ingredients; the particular contact descriptor and matched experiment are our diagnostic, not a published universal bonding-validity score.','']
(P/'README.md').write_text('\n'.join(lines))
(R/'accounting.json').write_text(json.dumps({'job_id':'3484071','state':'COMPLETED','elapsed_seconds':197,'allocated_cpus':48,'allocated_cpu_seconds':9456,'allocated_core_hours':9456/3600,'total_cpu_seconds':1800.431,'runner_seconds':186.92237091064453},indent=2)+'\n')
print('Report and figure written')
