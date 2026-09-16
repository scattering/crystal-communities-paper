#!/usr/bin/env python3
"""Portable wrapper for the recovered ICSD occupancy scanner."""
from __future__ import annotations
import argparse, csv, getpass, json, re, time, zipfile
from pathlib import Path
import gemmi
FIELDS=["icsd_id","has_partial_occupancy","n_occupancy_values","n_unknown_occupancy_values","min_occupancy","max_occupancy"]
ID_RE=re.compile(r"icsd_(\d+)\.cif$",re.I)

def classify_values(values:list[str])->tuple[int,int,int,float|str,float|str]:
    """Apply the recovered producer's Gemmi parsing and 1e-8 rule exactly."""
    observed=[]; unknown=0
    for value in values:
        try:number=gemmi.cif.as_number(value)
        except Exception:number=float("nan")
        if number != number: unknown += 1
        else: observed.append(float(number))
    partial=any(abs(value-1.0)>1e-8 for value in observed)
    return int(partial),len(values),unknown,min(observed) if observed else "",max(observed) if observed else ""

def scan_archive(archive:Path,password:bytes|None)->tuple[list[tuple],list[dict[str,object]]]:
    rows=[]; failures=[]
    with zipfile.ZipFile(archive) as zf:
        members=[name for name in zf.namelist() if name.lower().endswith('.cif')]
        for member in members:
            match=ID_RE.search(member)
            if not match: continue
            icsd_id=int(match.group(1))
            try:
                raw=zf.read(member,pwd=password)
                doc=gemmi.cif.read_string(raw.decode('utf-8',errors='replace'))
                values=list(doc.sole_block().find_values('_atom_site_occupancy'))
                rows.append((icsd_id,*classify_values(values)))
            except Exception as exc:
                failures.append({'icsd_id':icsd_id,'member':member,'error':f'{type(exc).__name__}: {exc}'})
    return rows,failures

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('archive',type=Path); parser.add_argument('output',type=Path)
    parser.add_argument('--unencrypted',action='store_true',help='Read an unencrypted test/public ZIP without prompting.')
    args=parser.parse_args(); password=None if args.unencrypted else getpass.getpass('ICSD archive password: ').encode(); started=time.time()
    rows,failures=scan_archive(args.archive,password); args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle); writer.writerow(FIELDS); writer.writerows(rows)
    metadata={'archive':str(args.archive),'output':str(args.output),'records':len(rows),'partial':sum(row[1] for row in rows),'failures':failures,'rule':'any explicit _atom_site_occupancy value differs from 1 by more than 1e-8; missing occupancy is treated as full occupancy','elapsed_seconds':time.time()-started,'recovered_original_sha256':'a708762cd09bc66e24c973082f23d662cbb9d1ab96f5d8de9552a4e2cf7a04cf'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)); print(json.dumps({k:v for k,v in metadata.items() if k!='failures'},indent=2)); print(f'failure_count={len(failures)}')
if __name__=='__main__': main()
