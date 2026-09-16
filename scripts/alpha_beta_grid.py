#!/usr/bin/env python3
"""(alpha, beta) grid for the ED Fig. 5 group ordering, reproducing the
production scorer of scripts/analyze_structural_accessibility.py exactly
(mean-of-constant core_threshold == per-community median, sample std for
z-scoring, 2019 age reference for GNoME, production role assignment with
bridge precedence and non-exclusive birth), under BOTH GNoME splits:
  pooled   = outlier_like column (d > pooled p95 = 4.756)  [original script]
  percomm  = in_basin column of quadrant_assignments.csv (d <= tau_c p95)
Writes alpha_beta_grid.json and alpha_beta_grid.md next to this file.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import analyze_structural_accessibility_revised as R  # noqa: E402
N = REPO / 'notes'
OUT = Path(__file__).resolve().parent
meta = R.load_community_metadata(N / 'icsd_community_assignments/community_assignments_labels3.csv', N / 'node_temporal_events.csv')
A, B, C, role = [], [], [], []
with (N / 'node_temporal_events.csv').open(newline='', encoding='utf-8') as h:
    for row in csv.DictReader(h):
        comm = R.parse_int(row.get('community', '')); year = R.parse_int(row.get('year', '')); dist = R.parse_float(row.get('distance_to_centroid', ''))
        if comm is None or comm < 0 or year is None or dist is None: continue
        m = meta.get(comm)
        if m is None: continue
        A.append(np.log1p(dist / max(m['core_threshold'], 1e-6))); B.append(np.log1p(m['size'])); C.append(np.log1p(max(year - m['birth_year'], 0.0)))
        et = row.get('event_type', ''); cp = row.get('core_periphery', '').lower(); br = row.get('is_bridge_attachment', '').lower() == 'true'
        role.append((et == 'community_birth', 'bridge' if br else cp))
A, B, C = map(np.asarray, (A, B, C))
birth = np.array([r[0] for r in role]); cls = np.array([r[1] for r in role])
masks = {'ICSD core': cls == 'core', 'ICSD periphery': cls == 'periphery', 'ICSD bridge': cls == 'bridge', 'ICSD birth': birth}
inb = R.load_gnome_in_basin(N / 'review_2026_08/quadrant_assignments.csv')
GA, GB, GC, pooled_in, percomm_in = [], [], [], [], []
with (N / 'external_frontier_runs/gnome_frontier_20260419/gnome_frontier_records.csv').open(newline='', encoding='utf-8') as h:
    for row in csv.DictReader(h):
        comm = R.parse_int(row.get('assigned_community', '')); dist = R.parse_float(row.get('nearest_centroid_distance', ''))
        if comm is None or dist is None or comm not in meta: continue
        m = meta[comm]
        GA.append(np.log1p(dist / max(m['core_threshold'], 1e-6))); GB.append(np.log1p(m['size'])); GC.append(np.log1p(max(2019 - m['birth_year'], 0.0)))
        pooled_in.append(row.get('outlier_like', '').lower() != 'true'); percomm_in.append(inb[row['material_id']])
GA, GB, GC = map(np.asarray, (GA, GB, GC)); pooled_in = np.array(pooled_in); percomm_in = np.array(percomm_in)
print('ICSD n', len(A), 'GNoME n', len(GA), 'pooled in-basin', pooled_in.sum(), 'percomm in-basin', percomm_in.sum())
grid = np.round(np.linspace(0, 1, 6), 1)
ORDER = ['ICSD core', 'ICSD periphery', 'ICSD bridge', 'ICSD birth', 'GNoME in-basin', 'GNoME frontier']
results = []
for a in grid:
    for b in grid:
        raw = A - a * B - b * C; mu = raw.mean(); sd = raw.std(ddof=1); z = (raw - mu) / sd
        gz = (GA - a * GB - b * GC - mu) / sd
        cell = {'alpha': float(a), 'beta': float(b)}
        for split, inmask in (('pooled', pooled_in), ('percomm', percomm_in)):
            gm = {k: float(z[v].mean()) for k, v in masks.items()}
            gm['GNoME in-basin'] = float(gz[inmask].mean()); gm['GNoME frontier'] = float(gz[~inmask].mean())
            # criterion S = the one coded in analyze_accessibility_sensitivity.py / Methods sentence
            S = gm['ICSD core'] < gm['ICSD periphery'] < gm['ICSD bridge'] < gm['ICSD birth'] and gm['GNoME in-basin'] < gm['GNoME frontier']
            # criterion K = the caption chain core < periphery < bridge < GNoME in-basin < GNoME frontier
            K = gm['ICSD core'] < gm['ICSD periphery'] < gm['ICSD bridge'] < gm['GNoME in-basin'] < gm['GNoME frontier']
            # criterion K2 = revised chain core < periphery < GNoME in-basin < bridge < GNoME frontier < birth
            K2 = gm['ICSD core'] < gm['ICSD periphery'] < gm['GNoME in-basin'] < gm['ICSD bridge'] < gm['GNoME frontier'] < gm['ICSD birth']
            cell[split] = {'means': gm, 'S_methods': bool(S), 'K_caption_chain': bool(K), 'K2_revised_chain': bool(K2), 'order': ' < '.join(sorted(ORDER, key=gm.get))}
        results.append(cell)
def tally(split, key):
    ok = [c for c in results if c[split][key]]; bad = [(c['alpha'], c['beta']) for c in results if not c[split][key]]
    return len(ok), bad
lines = ['# (alpha, beta) 6x6 grid — ordering verdicts', '', f'ICSD scored n={len(A)}; GNoME n={len(GA)}; pooled in-basin={int(pooled_in.sum())}, per-community in-basin={int(percomm_in.sum())}', '']
for split, label in (('pooled', 'ORIGINAL split (pooled outlier_like)'), ('percomm', 'REVISED split (per-community p95 in_basin)')):
    lines.append(f'## {label}')
    for key, desc in (('S_methods', 'Methods criterion: core<periphery<bridge<birth AND in-basin<frontier'), ('K_caption_chain', 'Caption chain: core<periphery<bridge<in-basin<frontier'), ('K2_revised_chain', 'Revised chain: core<periphery<in-basin<bridge<frontier<birth')):
        n_ok, bad = tally(split, key)
        lines.append(f'- {desc}: holds at **{n_ok} of 36** cells; failures: {bad if bad else "none"}')
    lines.append('')
    lines.append('| alpha | beta | core | periph | bridge | birth | GN in-basin | GN frontier | S | K | K2 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for c in results:
        g = c[split]['means']
        lines.append(f"| {c['alpha']:.1f} | {c['beta']:.1f} | {g['ICSD core']:+.3f} | {g['ICSD periphery']:+.3f} | {g['ICSD bridge']:+.3f} | {g['ICSD birth']:+.3f} | {g['GNoME in-basin']:+.3f} | {g['GNoME frontier']:+.3f} | {'Y' if c[split]['S_methods'] else 'n'} | {'Y' if c[split]['K_caption_chain'] else 'n'} | {'Y' if c[split]['K2_revised_chain'] else 'n'} |")
    lines.append('')
(OUT / 'alpha_beta_grid.md').write_text('\n'.join(lines) + '\n')
(OUT / 'alpha_beta_grid.json').write_text(json.dumps(results, indent=1))
print('\n'.join(l for l in lines if l.startswith('- ') or l.startswith('## ')))
# production cell echo
for c in results:
    if c['alpha'] == 0.5 and c['beta'] == 0.5:
        for s in ('pooled', 'percomm'): print(s, {k: round(v, 3) for k, v in c[s]['means'].items()})

# ---------------------------------------------------------------------------
# Extension: role convention x split x age reference x std convention.
#   roles 'prod' = production script (bridge > core/periphery; birth NON-exclusive, also counted in core/periphery/bridge)
#   roles 'excl' = analyze_accessibility_sensitivity.py (birth > bridge > core > periphery, exclusive)
# ---------------------------------------------------------------------------
excl_masks = {'ICSD birth': birth, 'ICSD bridge': (cls == 'bridge') & ~birth, 'ICSD core': (cls == 'core') & ~birth, 'ICSD periphery': (cls == 'periphery') & ~birth}
GC2015 = np.log1p(np.maximum(2015 - np.array([0.0]), 0))  # placeholder, replaced below
# rebuild GNoME age arrays for 2015 reference
births_by_comm = {c: m['birth_year'] for c, m in meta.items()}
GB_comm = []
with (N / 'external_frontier_runs/gnome_frontier_20260419/gnome_frontier_records.csv').open(newline='', encoding='utf-8') as h:
    for row in csv.DictReader(h):
        comm = R.parse_int(row.get('assigned_community', '')); dist = R.parse_float(row.get('nearest_centroid_distance', ''))
        if comm is None or dist is None or comm not in meta: continue
        GB_comm.append(meta[comm]['birth_year'])
GB_comm = np.asarray(GB_comm)
GC_by_ref = {2019: np.log1p(np.maximum(2019 - GB_comm, 0)), 2015: np.log1p(np.maximum(2015 - GB_comm, 0))}
assert np.allclose(GC_by_ref[2019], GC)
CAPTION = {'ICSD core': -0.32, 'ICSD bridge': 0.72, 'ICSD birth': 1.78, 'GNoME in-basin': 0.55, 'GNoME frontier': 1.20}
cells = [(a, b) for a in grid for b in grid] + [(0.5, 0.5)]
ext = {}
best = []
for roles, mk in (('prod', masks), ('excl', excl_masks)):
    for split, inmask in (('pooled', pooled_in), ('percomm', percomm_in)):
        for ref in (2019, 2015):
            for ddof in (1, 0):
                S_ok, K_ok, K2_ok = [], [], []
                for a, b in cells:
                    raw = A - a * B - b * C; mu = raw.mean(); sd = raw.std(ddof=ddof); z = (raw - mu) / sd
                    gz = (GA - a * GB - b * GC_by_ref[ref] - mu) / sd
                    gm = {k: float(z[v].mean()) for k, v in mk.items()}
                    gm['GNoME in-basin'] = float(gz[inmask].mean()); gm['GNoME frontier'] = float(gz[~inmask].mean())
                    dev = max(abs(gm[k] - v) for k, v in CAPTION.items())
                    best.append((dev, roles, split, ref, ddof, a, b, {k: round(v, 3) for k, v in gm.items()}))
                    if (a, b) == (0.5, 0.5):
                        ext[(roles, split, ref, ddof)] = {k: round(v, 3) for k, v in gm.items()}
                        continue
                    S_ok.append((gm['ICSD core'] < gm['ICSD periphery'] < gm['ICSD bridge'] < gm['ICSD birth'] and gm['GNoME in-basin'] < gm['GNoME frontier'], (a, b)))
                    K_ok.append((gm['ICSD core'] < gm['ICSD periphery'] < gm['ICSD bridge'] < gm['GNoME in-basin'] < gm['GNoME frontier'], (a, b)))
                    K2_ok.append((gm['ICSD core'] < gm['ICSD periphery'] < gm['GNoME in-basin'] < gm['ICSD bridge'] < gm['GNoME frontier'] < gm['ICSD birth'], (a, b)))
                print(f"roles={roles:4s} split={split:7s} ref={ref} ddof={ddof}: S(methods) {sum(o for o,_ in S_ok)}/36 fail={[c for o,c in S_ok if not o]} | K(caption chain) {sum(o for o,_ in K_ok)}/36 | K2(revised chain) {sum(o for o,_ in K2_ok)}/36")
print()
print('(0.5, 0.5) means by convention:')
for k, v in ext.items(): print('  ', k, v)
print()
best.sort(key=lambda t: t[0])
print('closest matches to caption vector (-0.32, +0.72, +1.78, +0.55, +1.20) [max abs dev, roles, split, ref, ddof, alpha, beta]:')
for t in best[:5]: print('  ', round(t[0], 3), t[1:])
