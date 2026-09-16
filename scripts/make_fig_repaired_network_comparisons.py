#!/usr/bin/env python3
"""Render manuscript network comparisons from regenerated per-formula tables."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--networks', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(a.networks / 'formula_graph/formula_graph_shared.csv', keep_default_na=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    pairs = [('tri_discovery', 'first_year', 'TRI discovery year', 'First ICSD year'),
             ('tri_deg', 'struct_deg', 'TRI degree', 'Structural graph degree')]
    for i, (ax, (xc, yc, xl, yl)) in enumerate(zip(axes, pairs)):
        xy = frame[[xc, yc]].apply(pd.to_numeric, errors='coerce').dropna()
        x, y = xy[xc].to_numpy(), xy[yc].to_numpy()
        rho = spearmanr(x, y).statistic
        if i:
            x, y = np.log10(1+x), np.log10(1+y)
            xl, yl = 'log₁₀(1 + TRI degree)', 'log₁₀(1 + structural degree)'
        ax.hexbin(x, y, gridsize=38, mincnt=1, bins='log', cmap='viridis')
        ax.set(xlabel=xl, ylabel=yl, title=f"{'ab'[i]}   Spearman ρ = {rho:.3f}; n = {len(x):,}")
    fig.savefig(a.output_dir / 'formula_graph_tri_comparison_repaired.png', dpi=240)
    fig.savefig(a.output_dir / 'formula_graph_tri_comparison_repaired.svg')
    plt.close(fig)
    frame = pd.read_csv(a.networks / 'tri_roles/tri_structural_role_records.csv', keep_default_na=False)
    xy = frame[['tri_deg', 'icsd_fragmentation_entropy']].apply(pd.to_numeric, errors='coerce').dropna()
    fig, ax = plt.subplots(figsize=(6.8, 4.6), layout='constrained')
    ax.hexbin(np.log10(1+xy.tri_deg), xy.icsd_fragmentation_entropy, gridsize=40,
              mincnt=1, bins='log', cmap='viridis')
    rho = spearmanr(xy.tri_deg, xy.icsd_fragmentation_entropy).statistic
    ax.set(xlabel='log₁₀(1 + TRI degree)', ylabel='Structural fragmentation entropy',
           title=f'Spearman ρ = {rho:.3f}; n = {len(xy):,}')
    fig.savefig(a.output_dir / 'tri_fragmentation_entropy_repaired.png', dpi=240)
    fig.savefig(a.output_dir / 'tri_fragmentation_entropy_repaired.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
