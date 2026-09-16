"""Build the short scientific interpretation and reproducible paired examples."""
from pathlib import Path
import json,csv,numpy as np,hashlib
p=Path(__file__).resolve().parent;r=p/'results'
rows=list(csv.DictReader((r/'per_target.csv').open()));j=json.loads((r/'bootstrap_summary.json').read_text())
lines=['# Direct GNoME local-property pilot','',
'Completed 2026-09-09, Stampede3 job **3484011**. All 100 frozen GNoME targets were analyzed, spanning 90 receiving Graphlet (CrystalNN) communities. The run used 190 unique ICSD reference/comparator structures; all 290 structures supplied all 30 continuous property channels, with no parse, feature or missing-property exclusions. Five metric tests and 812 saved-result/input checks passed.','',
'**Finding.** The Graphlet-only group has a modest local-arrangement signal beyond overall composition: its neighbor-pair property distributions agree better with independently geometry-selected ICSD references than after labels are shuffled. This agreement survives removal of the coarse histogram bins. The same kind of shuffle advantage appears in other groups, so the pilot supports local chemical ordering without identifying it as the specific cause of the Graphlet–CrystalWeave occupancy reversal.','',
'## Results','',
'Each score averages normalized continuous Wasserstein distances across ten neighbor-pair property means and ten absolute differences. Negative actual-minus-shuffle deltas mean better agreement with the ICSD reference than the shuffled target. Intervals are 2,000 shared-reference cluster bootstrap percentiles. Percent improvements are medians of per-target improvements relative to that target’s shuffle median; they are not differences of group medians.','',
'| GNoME group | n | Actual−shuffle median (95% CI) | Median relative improvement | Targets better than shuffle | ICSD baseline score / GNoME score |',
'|---|---:|---:|---:|---:|---:|']
for g,d in j['groups'].items():
 x=d['metrics']['pair'];rr=[z for z in rows if z['cell']==g];ci=x['target_delta_cluster_bootstrap_ci95'];rel=np.median([-float(z['pair_delta'])/float(z['pair_shuffle_median']) for z in rr])*100
 lines.append(f"| {g} | {len(rr)} | {x['target_delta_median']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {rel:.1f}% | {sum(float(z['pair_delta']) < -1e-10 for z in rr)}/{len(rr)} | {x['baseline_actual_median']:.3f} / {x['target_actual_median']:.3f} |")
lines+=['',
'For the primary Graphlet-only group, the separately declared pair-mean and pair-difference effects are −0.029 [−0.049, −0.016] and −0.033 [−0.053, −0.016]. Thus the combined result does not arise from averaging one positive and one negative family. Every target has a nontrivial shuffled pair distribution; one-site marginals are exactly unchanged.','',
'ICSD–ICSD comparisons have smaller property distances and stronger shuffle advantages. However, their geometry matches are also closer: in the Graphlet-only stratum, median standardized AMD distances are 0.663 for ICSD–ICSD and 1.186 for GNoME–ICSD. The baseline is descriptive and does not establish that the chemistry alone causes the difference. The four groups have different chemistry/geometry mixtures, and no group-difference test or full-cohort prevalence claim is warranted.','',
'## Paired examples','',
'These three Graphlet-only examples are selected mechanically at zero-based ranks 10, 20 and 30 after sorting the 40 targets by actual-minus-shuffle pair distance and material ID. The independent reference search uses AMD geometry, element count within one, and site count between half and twice the target; references are not asserted to be structural parents.','',
'| GNoME ID | GNoME formula | ICSD reference | Reference formula | Actual pair distance | Shuffle median | Difference |',
'|---|---|---:|---|---:|---:|---:|']
rr=sorted([x for x in rows if x['cell']=='graphlet-in / CrystalWeave-out'],key=lambda x:(float(x['pair_delta']),x['material_id']))
examples=[]
for k in [10,20,30]:
 x=rr[k];examples.append({f:x[f] for f in ['material_id','reduced_formula','icsd_reference','reference_formula','pair_actual','pair_shuffle_median','pair_delta','amd_distance']})
 lines.append(f"| {x['material_id']} | {x['reduced_formula']} | {x['icsd_reference']} | {x['reference_formula']} | {float(x['pair_actual']):.3f} | {float(x['pair_shuffle_median']):.3f} | {float(x['pair_delta']):+.3f} |")
lines+=['',
'## Consequence for the paper','',
'A proportionate sentence is: **“In a stratified 100-structure pilot, GNoME neighbor-pair elemental-property distributions showed modest agreement with independently geometry-selected ICSD references beyond composition-preserving label shuffles, including among structures accepted only by the Graphlet map.”**','',
'This adds direct evidence beyond occupancy and coarse histograms. It does not demonstrate substitution ancestry, experimental synthesizability, equivalent electronic bonding, or a complete explanation of the map reversal. The feature-block and calibration controls remain the direct evidence for why basin occupancy changes. The pilot also does not test three-site Graphlet channels, the VoronoiNN variant or an alternative to AMD reference selection. A short SI paragraph can carry the protocol and numerical result; the main text can retain its existing measured language about familiar local distributions. No expanded computation is needed to support that bounded interpretation.','',
'## Cost and provenance','',
'Slurm reports COMPLETED, elapsed 93 seconds on 48 allocated CPUs: **1.24 allocated core-hours**, with 420.709 seconds (7.01 minutes) summed actual CPU time. The scientific runner recorded 81.375 seconds; the allocation was capped at 20 minutes. No DFT, new map fitting or parent tracing was performed. Only derived data were retrieved; the ICSD archive passphrase was held in memory and not recorded.','',
'Inputs, source hashes and protocol: `results/provenance.json`. Frozen targets: `sample.json`. Full results and shuffled distributions: `results/per_target.csv`, `per_channel.csv`, `null_scores.json`; bootstrap: `bootstrap_summary.json`; verification: `verification.json`. The postprocessor conditions on the selected references and ICSD normalization scales. The sample is stratified and cannot estimate whole-cohort prevalence.','']
(p/'README.md').write_text('\n'.join(lines))
(r/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')
(r/'accounting.json').write_text(json.dumps({'job_id':'3484011','state':'COMPLETED','allocated_cpus':48,'elapsed_seconds':93,'allocated_cpu_seconds':4464,'allocated_core_hours':1.24,'total_cpu_seconds':420.709,'runner_seconds':81.37520122528076,'time_limit_seconds':1200},indent=2)+'\n')
