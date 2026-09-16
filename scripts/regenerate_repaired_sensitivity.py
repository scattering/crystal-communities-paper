#!/usr/bin/env python3
"""Cutoff-fitted PCA control and genuine alternate-partition reassignment."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notes/review_2026_08"))
from fig3c_corrected_denominators import cutoff_map, heldout_classify, rate_block
from prepare_repaired_projection_basis import community_reference
from regenerate_repaired_downstream import SOURCE_SLUGS, source_flags


def nearest(x, centers):
    indices, distances = [], []
    for start in range(0, len(x), 1000):
        block = x[start:start+1000]
        exact = cdist(block, centers, metric='euclidean')
        pos = exact.argmin(1)
        indices.extend(pos)
        distances.extend(exact[np.arange(len(block)),pos])
    return np.asarray(indices), np.asarray(distances)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--production',type=Path,required=True)
    ap.add_argument('--downstream',type=Path,required=True)
    ap.add_argument('--stages',nargs='+',choices=['truncated','variants'],default=['truncated','variants'])
    args=ap.parse_args()
    sample=pd.read_csv(args.production/'sample_assignments.csv',keep_default_na=False)
    labels=pd.read_csv(args.production/'graph/community_assignments.csv',keep_default_na=False)
    ids=sample.icsd_id.to_numpy(int)
    years=pd.to_numeric(sample.year,errors='coerce').fillna(-1).to_numpy(int)
    comm=labels.community.to_numpy(int)
    x=np.load(args.production/'features_pca.npy')
    assert np.array_equal(ids,labels.icsd_id) and len(x)==len(ids)
    out=args.downstream/'projection_sensitivity';out.mkdir(exist_ok=True)
    if 'truncated' in args.stages:
        raw=np.load(args.production/'features.npy',mmap_mode='r')
        results={}
        for cutoff in [1990,2000,2010]:
            train=(years>0)&(years<=cutoff)
            scaler=StandardScaler().fit(raw[train])
            pca=PCA(n_components=32,random_state=42).fit(scaler.transform(raw[train]))
            tx=pca.transform(scaler.transform(raw))
            scored={}
            for name,coords in [('full_record_PCA',x),('cutoff_fitted_PCA',tx)]:
                cids,cent,thr,_,_=cutoff_map(coords,years,comm,cutoff)
                h=heldout_classify(coords,ids,years,comm,cutoff,cids,cent,thr)
                scored[name]=rate_block(int(h.in_basin.sum()),len(h))
                h.to_csv(out/f'{name}_heldout_{cutoff}.csv',index=False)
            scored['change_percentage_points']=100*(scored['cutoff_fitted_PCA']['rate']-scored['full_record_PCA']['rate'])
            results[str(cutoff)]=scored
            print('Cutoff PCA',cutoff,scored,flush=True)
        (out/'cutoff_pca_summary.json').write_text(json.dumps({'membership_scope':'full-record Louvain held fixed in both controls; only scaler/PCA refitted at cutoff','cutoffs':results},indent=2)+'\n')
    if 'variants' in args.stages:
        ext={name:np.load(args.downstream/'external'/slug/'features_pca.npy') for name,slug in SOURCE_SLUGS.items()}
        variants={'production':args.production/'graph/community_assignments.csv'}
        variants.update({p.parent.parent.name:p for p in sorted((args.downstream/'partition_sensitivity').glob('*/graph/community_assignments.csv'))})
        if len(variants)!=5: raise ValueError('Expected production plus four completed variant partitions')
        results={}
        for name,path in variants.items():
            lab=pd.read_csv(path,keep_default_na=False)
            assert np.array_equal(ids,lab.icsd_id)
            lc=lab.community.to_numpy(int)
            rows=[(int(i),int(y) if y>0 else None,int(c)) for i,y,c in zip(ids,years,lc)]
            arrays,_=community_reference(x,rows)
            mapped={}
            for source,ex in ext.items():
                nearest_idx,dist=nearest(ex,arrays['centroids'])
                mapped[source]=pd.DataFrame({'assigned_community':arrays['communities'][nearest_idx], 'nearest_centroid_distance':dist})
                mapped[source].to_csv(out/f'{name}_{source}_assignments.csv',index=False)
            rec={'n_communities':len(arrays['communities']),'n_noise':int((lc<0).sum()),'cutoffs':{}}
            for cutoff in [1990,2000,2010]:
                cids,cent,thr,thresholds,_=cutoff_map(x,years,lc,cutoff)
                h=heldout_classify(x,ids,years,lc,cutoff,cids,cent,thr)
                rates={'ICSD':rate_block(int(h.in_basin.sum()),len(h))}
                rates.update({source:rate_block(int(source_flags(frame,thresholds).sum()),len(frame)) for source,frame in mapped.items()})
                rec['cutoffs'][str(cutoff)]=rates
            results[name]=rec
            print('Partition',name,rec,flush=True)
            (out/'partition_source_rates.json').write_text(json.dumps({'reassignment':'Every external embedding is assigned to nearest full-map centroid of each variant; cutoff-specific radii then applied, all records retained','variants':results},indent=2)+'\n')


if __name__=='__main__': main()
