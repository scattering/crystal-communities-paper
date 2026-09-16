#!/usr/bin/env python3
"""Independent arithmetic, projection, provenance, and package-integrity checks."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def boolean(value) -> bool:
    return str(value).strip().lower() == "true"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--basis", type=Path)
    p.add_argument("--score-summary", type=Path)
    p.add_argument("--report", type=Path, help="Optional JSON output; omit for a read-only check.")
    return p.parse_args()

def main():
    args = parse_args(); root = args.bundle.resolve()
    basis_path = args.basis or root.parent / "reference_basis/projection_basis.npz"
    score_path = args.score_summary or root.parent / "accessibility/structural_accessibility_summary.json"
    checks=[]
    def check(name, condition, detail=None):
        condition=bool(condition); checks.append({"name":name,"passed":condition,"detail":detail})
        if not condition: raise AssertionError(f"{name}: {detail}")
    # Package file hashes, if the finalized manifest exists.
    checksum_path=root/'SHA256SUMS'
    n_hashes=0
    if checksum_path.is_file():
        for line in checksum_path.read_text().splitlines():
            digest, rel=line.split(None,1); rel=rel.strip(); path=root/rel
            check(f"package_sha256:{rel}",path.is_file() and sha256(path)==digest)
            n_hashes+=1
    summary_docs=json.load((root/'source/mp_2022_10_28_summary_target_docs.json').open())
    materials_docs=json.load((root/'source/mp_2022_10_28_materials_target_docs.json').open())
    targets=list(csv.DictReader((root/'source/alab_targets.csv').open()))
    ids={r['mp_id'] for r in targets}
    check('57_unique_target_ids',len(targets)==len(ids)==57)
    check('both_snapshot_exports_cover_targets',set(summary_docs)==set(materials_docs)==ids)
    for mid in sorted(ids):
        a=summary_docs[mid]['structure']; b=materials_docs[mid]['structure']
        check(f'geometry_lattice:{mid}',a['lattice']['matrix']==b['lattice']['matrix'])
        check(f'geometry_sites:{mid}',len(a['sites'])==len(b['sites']) and all(
              x['species']==y['species'] and x['abc']==y['abc'] and x['xyz']==y['xyz'] and x.get('properties')==y.get('properties')
              for x,y in zip(a['sites'],b['sites'])))
    for collection in ('summary','materials'):
        manifest=json.load((root/f'source/mp_2022_10_28_{collection}_source_manifest.json').open())
        check(f'{collection}_manifest_counts',manifest['n_targets']==manifest['n_found']==manifest['n_structures']==57 and len(manifest['source_objects'])==36)
        check(f'{collection}_manifest_object_provenance',all(set(('key','bytes','sha256','etag','last_modified','content_length'))<=set(o) for o in manifest['source_objects']))
    records=list(csv.DictReader((root/'output/target_projection_records.csv').open()))
    pairs=list(csv.DictReader((root/'output/target_vs_refinement_records.csv').open()))
    features=np.load(root/'output/features.npy'); projected=np.load(root/'output/features_pca.npy')
    check('record_count_and_ids',len(records)==57 and {r['mp_id'] for r in records}==ids)
    check('feature_shapes',features.shape==(57,213) and projected.shape==(57,32))
    check('feature_arrays_finite',np.isfinite(features).all() and np.isfinite(projected).all())
    check('outcome_counts',Counter(r['corrected_outcome'] for r in records)==Counter({'made':36,'not_obtained':15,'inconclusive':4,'offline_recovery':2}))
    with np.load(basis_path,allow_pickle=False) as z:
        basis={k:z[k] for k in z.files if k!='metadata'}
    repro=((features-basis['scaler_mean'])/basis['scaler_scale']-basis['pca_mean']) @ basis['pca_components'].T
    projection_error=float(np.max(np.abs(repro-projected)))
    check('pca_transform_reproduced',projection_error<1e-12,projection_error)
    moments=json.load(score_path.open()); mu=float(moments['icsd_raw_mu']); sigma=float(moments['icsd_raw_sigma'])
    max_distance_error=max_score_error=0.0
    for i,r in enumerate(records):
        check(f'feature_row:{i}',int(r['feature_row'])==i)
        distances=np.linalg.norm(basis['centroids']-projected[i],axis=1); j=int(np.argmin(distances)); d=float(distances[j])
        max_distance_error=max(max_distance_error,abs(d-float(r['nearest_centroid_distance'])))
        check(f'assigned_community:{i}',int(r['assigned_community'])==int(basis['communities'][j]))
        check(f'in_basin:{i}',boolean(r['in_basin'])==(d<=float(basis['p95'][j])))
        age=max(2023-int(basis['birth_years'][j]),0)
        raw=math.log1p(d/max(float(basis['p50'][j]),1e-6))-.5*math.log1p(int(basis['counts'][j]))-.5*math.log1p(age)
        score=(raw-mu)/sigma; max_score_error=max(max_score_error,abs(score-float(r['accessibility_score'])))
    check('nearest_distance_reproduced',max_distance_error<1e-12,max_distance_error)
    check('accessibility_score_reproduced',max_score_error<1e-12,max_score_error)
    made=[r for r in records if r['corrected_outcome']=='made']; failed=[r for r in records if r['corrected_outcome']=='not_obtained']
    table=[[sum(boolean(r['in_basin']) for r in made),sum(not boolean(r['in_basin']) for r in made)],
           [sum(boolean(r['in_basin']) for r in failed),sum(not boolean(r['in_basin']) for r in failed)]]
    check('primary_basin_table',table==[[28,8],[4,11]],table)
    fisher=stats.fisher_exact(table,alternative='two-sided')
    fisher_p=float(fisher[1] if isinstance(fisher,tuple) else fisher.pvalue)
    check('primary_fisher_p',abs(fisher_p-0.0011125025618931553)<1e-14,fisher_p)
    x=np.array([float(r['accessibility_score']) for r in made]); y=np.array([float(r['accessibility_score']) for r in failed])
    auc=float(((x[:,None]<y).sum()+.5*(x[:,None]==y).sum())/(x.size*y.size))
    check('primary_auc',abs(auc-0.5907407407407408)<1e-15,auc)
    by_id={r['mp_id']:r for r in records}
    check('paired_count',len(pairs)==42 and len({r['mp_id'] for r in pairs})==42)
    check('paired_target_fields',all(abs(float(p['target_score'])-float(by_id[p['mp_id']]['accessibility_score']))<1e-14 and boolean(p['target_in_basin'])==boolean(by_id[p['mp_id']]['in_basin']) for p in pairs))
    check('paired_agreement_counts',sum(boolean(p['same_community']) for p in pairs)==34 and sum(boolean(p['in_basin_agreement']) for p in pairs)==38)
    result={"all_passed":all(c['passed'] for c in checks),"n_checks":len(checks),"n_manifested_files_checked":n_hashes,
            "max_projection_abs_error":projection_error,"max_nearest_distance_abs_error":max_distance_error,
            "max_accessibility_score_abs_error":max_score_error,"primary_basin_table":table,"primary_fisher_two_sided_p":fisher_p,
            "primary_auc":auc,"basis":str(basis_path),"basis_sha256":sha256(basis_path),"score_summary":str(score_path),"score_summary_sha256":sha256(score_path),
            "verifier_sha256":sha256(Path(__file__)),"checks":checks}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))

if __name__=='__main__':
    main()
