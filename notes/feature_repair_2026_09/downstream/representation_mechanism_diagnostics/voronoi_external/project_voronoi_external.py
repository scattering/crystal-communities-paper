"""Isolated public-cohort projection into the existing Voronoi graphlet map.

The frozen graphlet encoder is called with neighbor_method='voronoinn'. Shared
cohort, cache, centroid, radius and classification helpers are imported unchanged.
No bin, PCA or partition fitting occurs. Outputs retain the generic independent
external-projection verifier's schema, with kind='graphlet_voronoinn'.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'experiments/graphlet_compare')]
import numpy as np
import graphlet_features as gf
from crystal_neighbors import FEATURE_VERSION
from external_frozen_cohorts import load_cohort, public_fields, record_key
from external_representation_sensitivity import (
    SOURCES, aligned_reference, community_basis, csv_rows, dump, load_cache,
    projection_rows, save_cache, sha, summarize, write_csv,
)

KIND = 'graphlet_voronoinn'
WORKER_EDGES = None
WORKER_ARCHIVE = None
EXPECTED = {'features':149757, 'map':146186, 'communities':1598, 'noise':8942}


def encoder():
    names = ['experiments/graphlet_compare/graphlet_features.py',
             'scripts/crystal_neighbors.py']
    return {'kind':KIND, 'feature_version':gf.GRAPHLET_FEATURE_VERSIONS['voronoinn'],
            'production_feature_version':FEATURE_VERSION,
            'neighbor_method':'voronoinn', 'neighbor_settings':gf.neighbor_settings('voronoinn'),
            'representation':gf.GRAPHLET_REPRESENTATION,
            'code_sha256':{name:sha(ROOT/name) for name in names},
            'missing_property_policy':'Unavailable observations are omitted within channels; a channel without usable positive mass fails. The radius screen rejects any site without an available Slater radius.'}


def validate_cdf(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 1280 or not np.isfinite(values).all():
        raise ValueError('Invalid graphlet matrix dimensions or values')
    for start in range(0,len(values),4096):
        chunk=values[start:start+4096].reshape(-1,64,20)
        if (np.any(chunk < -2e-6) or np.any(chunk > 1+2e-6)
                or np.any(np.diff(chunk,axis=2) < -2e-6)
                or np.any(np.abs(chunk[:,:,-1]-1) > 2e-6)):
            raise ValueError('Values are not normalized cumulative distributions')


def inspect_reference(a):
    meta_path=a.feature_dir/'feature_preparation.json'
    meta=json.loads(meta_path.read_text())
    partition_path=a.partition_dir/'partition.json'
    partition=json.loads(partition_path.read_text())
    declaration=encoder()
    if (not meta['complete'] or meta['kind'] != 'graphlet'
            or meta['feature_version'] != FEATURE_VERSION
            or meta['neighbor_method'] != 'voronoinn'
            or meta['neighbor_settings'] != declaration['neighbor_settings']
            or meta['graphlet_feature_version'] != declaration['feature_version']):
        raise ValueError('Reference is not the frozen radius-screened Voronoi encoder')
    for name,digest in declaration['code_sha256'].items():
        if meta['source_sha256'].get(name) != digest:
            raise ValueError(f'Current encoder differs from frozen producer: {name}')
    if (partition['feature_preparation_sha256'] != sha(meta_path)
            or partition['graphlet_neighbor_method'] != 'voronoinn'
            or partition['graphlet_neighbor_settings'] != declaration['neighbor_settings']
            or partition['graphlet_feature_version'] != declaration['feature_version']
            or partition['metric'] != 'euclidean' or not partition['dated_only']
            or partition['n_entries'] != EXPECTED['map']
            or partition['n_communities'] != EXPECTED['communities']
            or partition['n_noise'] != EXPECTED['noise']):
        raise ValueError('Partition does not match the completed dated Voronoi map')
    ids_path=a.feature_dir/'features.ids.json'
    bins_path=a.feature_dir/'bin_edges.json'
    if sha(ids_path) != meta['ids_sha256'] or sha(bins_path) != meta['bin_edges_sha256']:
        raise ValueError('Reference IDs or bins differ from their frozen hashes')
    feature_ids=json.loads(ids_path.read_text())
    if len(feature_ids) != EXPECTED['features'] or len(set(feature_ids)) != len(feature_ids):
        raise ValueError('Unexpected successful Voronoi feature population')
    edges=json.loads(bins_path.read_text())
    if set(edges) != set(gf.REGISTRY.all) or any(len(v)!=2 or not np.isfinite(v).all() or v[1]<=v[0] for v in edges.values()):
        raise ValueError('Invalid frozen graphlet bin ranges')
    assignments_path=a.partition_dir/'community_assignments.csv'
    assignments=csv_rows(assignments_path)
    indices,ids,years,labels=aligned_reference(feature_ids,assignments)
    if (len(ids) != EXPECTED['map'] or np.any(years < 0)
            or len(set(labels[labels>=0])) != EXPECTED['communities']
            or int(np.sum(labels<0)) != EXPECTED['noise']):
        raise ValueError('Partition assignment population/counts mismatch')
    partition_ids=json.loads((a.partition_dir/'ids.json').read_text())
    if list(map(int,ids)) != list(map(int,partition_ids)):
        raise ValueError('Partition assignment ID order differs from saved map')
    files={str(p):sha(p) for p in [meta_path,partition_path,ids_path,bins_path,assignments_path,a.partition_dir/'ids.json']}
    return meta,edges,indices,ids,years,labels,files


def preflight(a):
    meta,edges,indices,ids,years,labels,files=inspect_reference(a)
    result={'status':'metadata_verified_not_executed', 'n_feature_successes':EXPECTED['features'],
            'n_fitted_map_members':len(ids),'n_undated_feature_successes_outside_map':EXPECTED['features']-len(ids),
            'n_communities':EXPECTED['communities'],'n_noise':EXPECTED['noise'],
            'encoder':encoder(),'input_sha256':files,'expected_features_sha256':meta['features_sha256']}
    if a.out_dir:
        a.out_dir.mkdir(parents=True,exist_ok=True);dump(a.out_dir/'preflight.json',result)
    print(json.dumps(result,indent=2))


def basis(a):
    meta,edges,indices,ids,years,labels,files=inspect_reference(a)
    for package in ('numpy','pymatgen'):
        if importlib.metadata.version(package) != meta['packages'][package]:
            raise ValueError(f'Use the frozen producer environment for numerical execution: {package}')
    features_path=a.feature_dir/'features.npy'
    if sha(features_path) != meta['features_sha256']:
        raise ValueError('Reference CDF matrix differs from the frozen producer hash')
    raw=np.load(features_path,mmap_mode='r',allow_pickle=False)
    if raw.shape != (EXPECTED['features'],1280):
        raise ValueError('Wrong reference CDF shape')
    validate_cdf(raw)
    coordinates=raw[indices]
    a.out_dir.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.feature_dir/'bin_edges.json',a.out_dir/'bin_edges.json')
    reports={}
    for cutoff in [None,1990,2000,2010]:
        key='full' if cutoff is None else f'T{cutoff}'
        training=np.ones(len(ids),dtype=bool) if cutoff is None else years<=cutoff
        evaluation=np.ones(len(ids),dtype=bool) if cutoff is None else years>cutoff
        fitted=community_basis(coordinates,labels,training)
        np.savez_compressed(a.out_dir/f'basis_{key}.npz',**fitted)
        rows,stats=projection_rows(ids[evaluation],coordinates[evaluation],fitted)
        write_csv(a.out_dir/f'icsd_{key}.csv',rows)
        reports[key]={**stats,'cutoff':cutoff,'n_training_rows_including_noise':int(training.sum()),
                      'n_training_nonnoise':int(np.sum(training & (labels>=0))), 'n_communities':len(fitted['communities'])}
        print(json.dumps({'reference':key,**stats}),flush=True)
    files[str(features_path)]=meta['features_sha256']
    dump(a.out_dir/'basis_manifest.json', {**encoder(),'input_sha256':files,
         'n_feature_rows':len(raw),'n_map_rows':len(ids),'n_map_noise':int(np.sum(labels<0)),
         'n_map_undated':0,'n_feature_rows_outside_map':len(raw)-len(ids),
         'reference_rates':reports,'metric':'Euclidean on unscaled 1280-D CDF',
         'radius_rule':'Arithmetic centroid and member-distance numpy.quantile(p95); assign nearest Euclidean centroid then distance <= its radius. Noise defines no basin.',
         'scope':'The full fitted map contains all 146,186 dated Voronoi successes. The 3,571 undated feature successes are outside this partition. Historical rows retain this full partition and frozen bins, restrict centroid/radius members through the cutoff, and evaluate later ICSD. They are not cutoff-trained maps.',
         'script_sha256':sha(__file__),
         'driver_dependencies_sha256':{name:sha(ROOT/name) for name in ['scripts/external_representation_sensitivity.py','scripts/external_frozen_cohorts.py','scripts/verify_external_representation_sensitivity.py']},
         'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pymatgen','matminer']}})


def initialize_worker(source,path,edges):
    global WORKER_EDGES,WORKER_ARCHIVE
    WORKER_EDGES=edges;WORKER_ARCHIVE=None
    if source in ('gnome','mattergen'):
        WORKER_ARCHIVE=zipfile.ZipFile(Path(path)/'by_id.zip' if source=='gnome' else path)


def featurize(record):
    from pymatgen.core import Structure
    public=public_fields(record);stage='parse_structure'
    try:
        kind=record['_kind']
        if kind=='pymatgen':structure=Structure.from_dict(record['_structure'])
        elif kind=='jarvis':
            from analyze_jarvis_frontier import jarvis_atoms_to_structure
            structure=jarvis_atoms_to_structure(record['_structure'])
        else:
            mid=record['material_id']
            names=[record['zip_member']] if kind=='mattergen' else [f'{mid}.cif',f'{mid}.CIF',f'by_id/{mid}.cif',f'by_id/{mid}.CIF',f'{mid}.vasp.cif']
            for name in names:
                try:data=WORKER_ARCHIVE.read(name);break
                except KeyError:continue
            else:raise KeyError('Missing public CIF')
            structure=Structure.from_str(data.decode('utf-8',errors='replace'),fmt='cif')
        stage='site_count'
        if not 0<len(structure)<=256:raise ValueError('Unsupported site count')
        public['reduced_formula']=public.get('reduced_formula') or structure.composition.reduced_formula
        public['n_sites']=len(structure);stage='voronoi_graphlet_features'
        raw=gf.all_raw_features(structure,neighbor_method='voronoinn')
        vector=np.cumsum(gf.histogram_tensor(raw,WORKER_EDGES),axis=1).reshape(-1).astype(np.float32)
        validate_cdf(vector.reshape(1,-1))
        return public,vector,None
    except Exception as exc:
        return public,None,{'stage':stage,'exception_type':type(exc).__name__,'detail':str(exc)[:250]}


def features(a):
    manifest_path=a.basis_dir/'basis_manifest.json';manifest=json.loads(manifest_path.read_text())
    declaration=encoder()
    if any(manifest.get(k)!=v for k,v in declaration.items()):
        raise ValueError('External encoder differs from the frozen Voronoi basis')
    for package,version in manifest['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'External and reference runtime differ: {package}')
    selected,old_keys,files,cohort=load_cohort(a.source,a.source_path,a.historical_records)
    keys=[record_key(r) for r in selected]
    if keys != [record_key(r) for r in csv_rows(a.attempted_cohort)]:
        raise ValueError('Attempted external IDs/order differ from the frozen cohort')
    sources=[{'path':str(p),'sha256':sha(p)} for p in files]
    edges=json.loads((a.basis_dir/'bin_edges.json').read_text())
    signature=hashlib.sha256(json.dumps({**declaration,'sources':sources,'edges':edges},sort_keys=True).encode()).hexdigest()
    a.out_dir.mkdir(parents=True,exist_ok=True);cache=a.out_dir/'feature_cache';cache.mkdir(exist_ok=True)
    write_csv(a.out_dir/'attempted_cohort.csv',[public_fields(r) for r in selected]);dump(a.out_dir/'attempted_ids.json',keys)
    def cache_path(key):return cache/(hashlib.sha256((signature+key).encode()).hexdigest()+'.npz')
    results={};pending=[]
    for record in selected:
        key=record_key(record);cached=load_cache(cache_path(key),signature,key,1280)
        if cached is None:pending.append(record)
        else:validate_cdf(cached[1].reshape(1,-1));results[key]=cached
    n_cached=len(results)
    print(f'{a.source}: {len(keys)} attempts, {n_cached} cached, {len(pending)} pending',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.n_jobs,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,initargs=(a.source,str(a.source_path),edges)) as pool:
        for i,result in enumerate(pool.map(featurize,pending,chunksize=4),1):
            record,vector,error=result;key=record_key(record);results[key]=result
            if error is None:save_cache(cache_path(key),signature,record,vector)
            if i%250==0 or i==len(pending):print(f'{a.source}: computed {i}/{len(pending)}',flush=True)
    ordered=[results[key] for key in keys];successful=[r for r in ordered if r[2] is None]
    failures=[{**r,**e} for r,v,e in ordered if e is not None];dump(a.out_dir/'failures.json',failures)
    if not successful or len(successful)+len(failures)!=len(keys):raise ValueError('Incomplete attempt accounting or no successes')
    coordinates=np.stack([r[1] for r in successful]);success_keys=[record_key(r[0]) for r in successful]
    np.save(a.out_dir/'coordinates.npy',coordinates);dump(a.out_dir/'feature_ids.json',success_keys)
    write_csv(a.out_dir/'successful_records.csv',[r[0] for r in successful]);reports={}
    for key,reference in manifest['reference_rates'].items():
        with np.load(a.basis_dir/f'basis_{key}.npz',allow_pickle=False) as saved:fitted=dict(saved)
        rows,stats=projection_rows(success_keys,coordinates,fitted);write_csv(a.out_dir/f'projection_{key}.csv',rows)
        reports[key]={**stats,'icsd_reference':reference,'icsd_minus_external_percentage_points':100*(reference['in_basin_fraction']-stats['in_basin_fraction']),
                      'external_fraction_all_attempts_lower_bound':stats['n_in_basin']/len(keys),
                      'external_fraction_all_attempts_upper_bound':(stats['n_in_basin']+len(failures))/len(keys)}
    dump(a.out_dir/'summary.json',{**declaration,'source':a.source,'source_files':sources,'cohort':cohort,'signature':signature,
         'n_attempted':len(keys),'n_successful':len(successful),'n_failures':len(failures),'n_cached':n_cached,'rates':reports,
         'basis_manifest_sha256':sha(manifest_path),'attempted_ids_sha256':sha(a.out_dir/'attempted_ids.json'),
         'feature_ids_sha256':sha(a.out_dir/'feature_ids.json'),'coordinates_sha256':sha(a.out_dir/'coordinates.npy'),
         'historical_records_sha256':sha(a.historical_records),'attempted_cohort_sha256':sha(a.attempted_cohort),'script_sha256':sha(__file__)})
    print(json.dumps({'source':a.source,'successes':len(successful),'failures':len(failures),'rates':reports}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name,fun in [('preflight',preflight),('basis',basis)]:
        s=sub.add_parser(name)
        for arg in ('feature-dir','partition-dir'):s.add_argument('--'+arg,type=Path,required=True)
        s.add_argument('--out-dir',type=Path,required=name=='basis');s.set_defaults(function=fun)
    s=sub.add_parser('features')
    for arg in ('source-path','historical-records','attempted-cohort','basis-dir','out-dir'):s.add_argument('--'+arg,type=Path,required=True)
    s.add_argument('--source',choices=SOURCES,required=True);s.add_argument('--n-jobs',type=int,default=48);s.set_defaults(function=features)
    s=sub.add_parser('summarize');s.add_argument('--run-dir',type=Path,required=True);s.set_defaults(function=summarize)
    a=p.parse_args()
    if getattr(a,'n_jobs',1)<1:p.error('n-jobs must be positive')
    a.function(a)


if __name__=='__main__':main()
