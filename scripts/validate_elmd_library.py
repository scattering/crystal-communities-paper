#!/usr/bin/env python3
"""Replay the recovered 250-pair check against the ElMD 0.5.15 library."""
from __future__ import annotations
import argparse,importlib.metadata,importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd
from pymatgen.core import Composition
from ElMD import ElMD
lookup = None

def sparse(f):
 c=Composition(str(f)).element_composition
 d={}; tot=sum(float(a) for a in c.values())
 for e,a in c.items(): d[int(lookup[e.symbol])]=d.get(int(lookup[e.symbol]),0)+float(a)/tot
 return np.array(sorted(d)),np.array([d[k] for k in sorted(d)])

def ours(a,b):
 pa,ma=sparse(a);pb,mb=sparse(b);v=np.zeros(103);w=np.zeros(103);v[pa]=ma;w[pb]=mb
 return np.abs(np.cumsum(v-w)).sum()

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--icsd-index', type=Path, default=root/'notes/feature_repair_2026_09/downstream/inputs/ICSD_index.csv')
    parser.add_argument('--cohort-records', type=Path, default=root/'notes/feature_repair_2026_09/downstream/formula_layers/nearest_icsd_elmd_records.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.find_spec('ElMD')
    global lookup
    lookup = json.loads((Path(next(iter(spec.submodule_search_locations)))/'el_lookup/mod_petti.json').read_text())
    r = pd.read_csv(args.icsd_index)
    qs = pd.read_csv(args.cohort_records)
    rng = np.random.default_rng(419)
    rows = []
    for _ in range(250):
        a=str(qs.iloc[int(rng.integers(len(qs)))].reduced_formula);b=str(r.iloc[int(rng.integers(len(r)))]['name'])
        x=float(ours(a,b)); y=float(ElMD(a,metric='mod_petti').elmd(ElMD(b,metric='mod_petti')))
        rows.append((a,b,x,y,abs(x-y)))
    out=pd.DataFrame(rows,columns=['query_formula','icsd_formula','sparse_exact','reference_ElMD_0.5.15','absolute_error'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output_dir/'elmd_library_validation_pairs.csv',index=False)
    summary={'library':'ElMD '+importlib.metadata.version('ElMD'),'metric':'mod_petti','random_pairs':len(out),'seed':419,'max_absolute_error':float(out.absolute_error.max()),'mean_absolute_error':float(out.absolute_error.mean()),'all_within_1e-12':bool((out.absolute_error<=1e-12).all()),'pairwise_csv':str(args.output_dir/'elmd_library_validation_pairs.csv')}
    (args.output_dir/'elmd_library_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not summary['all_within_1e-12']:raise SystemExit('ElMD library comparison failed')

if __name__=='__main__':main()
