#!/usr/bin/env python3
"""Reproduce the canonical merged ICSD occupancy table byte-for-byte."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
FIELDS=["icsd_id","has_partial_occupancy","n_occupancy_values","n_unknown_occupancy_values","min_occupancy","max_occupancy"]
def main()->None:
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    frames=[]
    for path in a.inputs:
        frame=pd.read_csv(path)
        if list(frame.columns)!=FIELDS:raise ValueError(f'Unexpected columns in {path}: {list(frame.columns)}')
        frames.append(frame)
    merged=pd.concat(frames,ignore_index=True)
    if merged.icsd_id.duplicated().any():raise ValueError('Duplicate ICSD identifiers')
    merged=merged.sort_values('icsd_id')
    a.output.parent.mkdir(parents=True,exist_ok=True);merged.to_csv(a.output,index=False)
if __name__=='__main__':main()
