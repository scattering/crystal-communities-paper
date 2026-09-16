#!/usr/bin/env python3
"""Replace only the AMD row on unchanged five-way evaluation identifiers."""
from pathlib import Path
import csv,json,hashlib
P=Path(__file__).resolve().parent;old=P.parent/'amd-external-full-map/five_representation_common_support.csv';source=P/'results_fixed/full_map_fixed_partition_repair/summary.json';s=json.loads(source.read_text());rows=list(csv.DictReader(old.open()));fields=list(rows[0]);ref=s['populations']['icsd']['five_way_common_support']['in_basin_fraction']
for r in rows:
 if r['representation']!='AMD':continue
 result=s['populations'][r['population'].lower()]['five_way_common_support'];assert int(r['n_common'])==result['n'];r.update(n_in_basin=result['n_in_basin'],in_basin_fraction=result['in_basin_fraction'],icsd_minus_source_percentage_points='' if r['population']=='ICSD' else 100*(ref-result['in_basin_fraction']))
out=P/'five_representation_common_support.csv'
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(P/'five_representation_common_support_provenance.json').write_text(json.dumps({'description':'Unchanged exact five-way evaluation populations. Only AMD basin eligibility and reassignment updated; other representations retain original in-sample full-map reference.','historical_table':str(old),'historical_table_sha256':sha(old),'repaired_amd_summary':str(source),'repaired_amd_summary_sha256':sha(source),'output_sha256':sha(out)},indent=2)+'\n')
print(out)
