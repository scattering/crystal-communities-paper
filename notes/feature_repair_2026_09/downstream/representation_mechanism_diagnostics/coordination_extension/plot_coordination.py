#!/usr/bin/env python3
"""Export matched effects with the prior-sample exclusion sensitivity."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
s = json.loads((HERE / 'results/summary.json').read_text())
names = ['MatterGen', 'MP-theoretical', 'JARVIS', 'Alexandria']
sources = ['mattergen', 'mp', 'jarvis', 'alexandria']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, method, title in zip(axes, ['crystalnn', 'voronoinn'], ['Geometric CrystalNN', 'Radius-screened VoronoiNN']):
    ax.axvline(0, color='#888888', lw=1, zorder=0)
    for analysis, offset, color, marker, label in [('all', -.10, '#126782', 'o', 'Full matched comparison'),
                                                  ('unused', .10, '#cf7830', 's', 'Earlier sample excluded')]:
        records = [s['matched'][method][analysis][c]['primary'] for c in sources]
        values = np.array([r['median_paired_difference'] for r in records])
        bounds = np.array([r['bonferroni_ci9875_cluster_percentile'] for r in records])
        ax.errorbar(values, np.arange(4) + offset, xerr=np.vstack((values - bounds[:, 0], bounds[:, 1] - values)),
                    fmt=marker, color=color, ms=5, capsize=3, lw=1.4, label=label)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_yticks(np.arange(4), names)
    ax.set_ylim(3.5, -.6)
    ax.set_xlabel('Median paired GNoME − comparator coordination')
    ax.grid(axis='x', alpha=.15)
axes[0].legend(loc='lower left', bbox_to_anchor=(-.04, -0.38), frameon=False, ncol=2)
fig.suptitle('Does GNoME have higher coordination?', fontsize=14, fontweight='bold', y=.98)
fig.text(.5, .035, 'Intervals: 20,000 formula-cluster bootstrap resamples; four-comparison adjustment within each neighbor rule.',
         ha='center', fontsize=8.5, color='#555555')
fig.subplots_adjust(left=.14, right=.98, bottom=.29, top=.82, wspace=.25)
for suffix in ('png', 'pdf', 'svg'):
    fig.savefig(HERE / f'results/coordination_comparison.{suffix}', dpi=200, facecolor='white')
