from pathlib import Path
import csv,json,hashlib,numpy as np
from pymatgen.core import Composition
base=Path(__file__).resolve().parent.parent
rows=list(csv.DictReader((base/'gnome_profile/gnome_profile_records.csv').open()))
rng=np.random.default_rng(20260909)
quotas={'graphlet-in / CrystalWeave-out':40,'in both':40,'CrystalWeave-in / graphlet-out':10,'out of both':10}
selected=[]
for group,n in quotas.items():
 pool=[r for r in rows if r['cell']==group];rng.shuffle(pool)
 counts_comm={};counts_chem={}
 for k in range(n):
  def chem(r):
   els={e.symbol for e in Composition(r['reduced_formula']).elements}
   for label,subset in [('oxide',{'O'}),('halide',{'F','Cl','Br','I'}),('chalcogenide',{'S','Se','Te'}),('pnictide',{'N','P','As'}),('HBC',{'H','B','C'})]:
    if els & subset:return label
   return 'other'
  pick=min(pool,key=lambda r:(counts_comm.get(r['comm_Graphlet_CrystalNN'],0),counts_chem.get(chem(r),0)))
  pool.remove(pick);c=chem(pick);counts_comm[pick['comm_Graphlet_CrystalNN']]=counts_comm.get(pick['comm_Graphlet_CrystalNN'],0)+1;counts_chem[c]=counts_chem.get(c,0)+1
  selected.append({k:pick[k] for k in ['material_id','reduced_formula','n_sites','n_elements','cell','comm_Graphlet_CrystalNN'] }|{'chemistry_stratum':c})
out=base/'local_property_pilot';(out/'sample.json').write_text(json.dumps(selected,indent=2)+'\n')
print('Frozen pilot:',len(selected),'targets;',len({r['comm_Graphlet_CrystalNN'] for r in selected}),'receiving communities')
