#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWN = ROOT / 'notes/feature_repair_2026_09/downstream'
DOWN = Path(os.environ.get('CRYSTAL_COMMUNITIES_DOWNSTREAM', DEFAULT_DOWN)).resolve()
FORM = DOWN / 'formula_layers'
OUT = Path(os.environ.get(
    'CRYSTAL_COMMUNITIES_FIGURE_DIR',
    ROOT / 'resources/figures/icsd_densification',
)).resolve()

SOURCE_ORDER = ['GNoME','MatterGen','MP','JARVIS','Alexandria']
DISPLAY = {'GNoME':'GNoME','MatterGen':'MatterGen','MP':'MP-theoretical','JARVIS':'JARVIS-DFT','Alexandria':'Alexandria'}
COLORS = {'ICSD':'#252525','GNoME':'#f47c26','MatterGen':'#8b2bb3','MP':'#13786c','JARVIS':'#267fb5','Alexandria':'#d9252a'}
MARKERS = {'ICSD':'o','GNoME':'o','MatterGen':'D','MP':'o','JARVIS':'s','Alexandria':'^'}

INPUTS = {
 'pca': DOWN/'inputs/features_pca.npy',
 'gnome': DOWN/'external/gnome/gnome_frontier_records.csv',
 'mattergen': DOWN/'external/mattergen/mattergen-public_frontier_records.csv',
 'mp': DOWN/'external/mp/mp_frontier_records.csv',
 'jarvis': DOWN/'external/jarvis/jarvis_frontier_records.csv',
 'alexandria': DOWN/'external/alexandria/alexandria_frontier_records.csv',
 'cutoffs': DOWN/'cutoff_trained/cutoff_trained_retrospective_summary.json',
 'representation': DOWN/'representations/amd-basin-repair-20260910/five_representation_common_support.csv',
 'formula_layers': FORM/'scored_support_layers.json',
 'elmd': FORM/'nearest_icsd_elmd_records.csv',
 'quadrants': FORM/'scale_invariant_quadrants.json',
 'alab': DOWN/'alab_mp_targets/output/target_projection_summary.json',
}

plt.rcParams.update({
 'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':11.5,
 'axes.labelsize':10.5, 'xtick.labelsize':9.3, 'ytick.labelsize':9.3,
 'legend.fontsize':8.7, 'axes.linewidth':0.8, 'savefig.facecolor':'white'
})

def panel_label(ax, letter):
    ax.text(-0.12, 1.055, letter, transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

def load_ext():
    out={}
    for key in SOURCE_ORDER:
        p=INPUTS[key.lower()]
        df=pd.read_csv(p)
        out[key]=df
    return out

def subsample(ext, n=386, seed=42):
    rng=np.random.default_rng(seed); out={}
    for key in SOURCE_ORDER:
        df=ext[key]
        if len(df)>n:
            idx=np.sort(rng.choice(len(df),n,replace=False)); out[key]=df.iloc[idx].copy()
        else: out[key]=df.copy()
    return out

def kde_background(X, seed=42, sample=12000, gridn=150, trim=.005, bw=1.6):
    lo=np.quantile(X,trim,axis=0); hi=np.quantile(X,1-trim,axis=0)
    keep=np.all((X>=lo)&(X<=hi),axis=1); z=X[keep]
    rng=np.random.default_rng(seed)
    if len(z)>sample: z=z[rng.choice(len(z),sample,replace=False)]
    kde=gaussian_kde(z.T,bw_method='scott'); kde.set_bandwidth(kde.factor*bw)
    xx=np.linspace(lo[0],hi[0],gridn); yy=np.linspace(lo[1],hi[1],gridn)
    XX,YY=np.meshgrid(xx,yy); dens=kde(np.vstack([XX.ravel(),YY.ravel()])).reshape(XX.shape)
    rel=(dens-dens.min())/(dens.max()-dens.min())
    return XX,YY,rel,lo,hi

def overlay_panel(ax, bg, ext, keys, title):
    XX,YY,Z,lo,hi=bg
    cf=ax.contourf(XX,YY,Z,levels=np.linspace(0,1,14),cmap='Blues_r',alpha=.88)
    for key in keys:
        df=ext[key]; x=df.pca1.to_numpy(); y=df.pca2.to_numpy()
        front=df.outlier_like.astype(str).str.lower().eq('true').to_numpy()
        for flag,alpha,size,edge,lw,suffix in [(False,.55,17,'white',.3,'in basin'),(True,.95,30,'white',.75,'frontier')]:
            take=front==flag
            if take.any(): ax.scatter(x[take],y[take],s=size,c=COLORS[key],marker=MARKERS[key],alpha=alpha,
                edgecolors=edge,linewidths=lw,label=f'{DISPLAY[key]}: {suffix}',rasterized=True)
    ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1])
    ax.set_xlabel('CrystalWeave PCA 1'); ax.set_ylabel('CrystalWeave PCA 2')
    ax.set_title(title,pad=6)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),frameon=False,ncol=len(keys),handletextpad=.25,columnspacing=.8)
    return cf

def fig3():
    X=np.load(INPUTS['pca'],mmap_mode='r')[:,:2]
    ext_all=load_ext(); ext=subsample(ext_all)
    bg=kde_background(np.asarray(X))
    fig=plt.figure(figsize=(14.8,10.8))
    gs=fig.add_gridspec(2,2,height_ratios=[1.04,1],left=.065,right=.965,bottom=.090,top=.925,hspace=.43,wspace=.28)
    axa=fig.add_subplot(gs[0,0]); cf=overlay_panel(axa,bg,ext,['GNoME','MatterGen'],'AI-associated cohorts in CrystalWeave')
    panel_label(axa,'a')
    axb=fig.add_subplot(gs[0,1]); overlay_panel(axb,bg,ext,['MP','JARVIS','Alexandria'],'DFT cohorts in the same map')
    panel_label(axb,'b')
    cax=fig.add_axes([.968,.585,.009,.245]); cb=fig.colorbar(cf,cax=cax,ticks=[0,.25,.5,.75,1]); cb.set_label('Relative ICSD density',fontsize=9)

    # Independently cutoff-trained CrystalWeave rates.
    axc=fig.add_subplot(gs[1,0]); panel_label(axc,'c')
    dat=json.loads(INPUTS['cutoffs'].read_text()); years=sorted(map(int,dat['cutoffs']))
    x=np.arange(len(years)); offsets=np.linspace(-.10,.10,6)
    for off,key in zip(offsets,['ICSD']+SOURCE_ORDER):
        vals=[]; lo=[]; hi=[]
        for y in years:
            row=dat['cutoffs'][str(y)]
            if key=='ICSD': d=row['heldout_rate']
            else: d=row['external_source_rates'][key]['all_records']
            vals.append(100*d['rate']); lo.append(100*d['wilson95'][0]); hi.append(100*d['wilson95'][1])
        vals=np.asarray(vals); lo=np.asarray(lo); hi=np.asarray(hi)
        axc.errorbar(x+off,vals,yerr=[vals-lo,hi-vals],color=COLORS[key],marker=MARKERS[key],
                     lw=1.45,ms=5,capsize=2,label='ICSD held out' if key=='ICSD' else DISPLAY[key])
    axc.set_xticks(x); axc.set_xticklabels(years); axc.set_ylim(25,82); axc.set_yticks(np.arange(30,81,10))
    axc.set_xlabel('Historical cutoff'); axc.set_ylabel('In-basin structures (%)')
    axc.set_title('Independently trained CrystalWeave maps',pad=6)
    axc.grid(axis='y',color='#e5e5e5',lw=.7); axc.set_axisbelow(True)
    axc.legend(loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,columnspacing=.8,handletextpad=.3)

    # Full-record exact-common-support comparison across all five representations.
    axd=fig.add_subplot(gs[1,1]); panel_label(axd,'d')
    rep=pd.read_csv(INPUTS['representation'])
    rename={'CrystalWeave':'CrystalWeave','Magpie':'Magpie-22 variant','CrystalNN graphlets':'Graphlet (CrystalNN)','VoronoiNN graphlets':'Graphlet (VoronoiNN)','AMD':'AMD-100'}
    rep['label']=rep.representation.map(rename).fillna(rep.representation)
    order=[x for x in ['CrystalWeave','Magpie-22 variant','Graphlet (CrystalNN)','Graphlet (VoronoiNN)','AMD-100'] if x in set(rep.label)]
    rows=SOURCE_ORDER
    mat=np.full((len(rows),len(order)),np.nan); ref=[]
    for j,rname in enumerate(order):
        sub=rep[rep.label==rname]
        ir=float(sub.loc[sub.population=='ICSD','in_basin_fraction'].iloc[0]); ref.append(ir)
        for i,key in enumerate(rows):
            pop='MP' if key=='MP' else key
            sr=float(sub.loc[sub.population==pop,'in_basin_fraction'].iloc[0])
            mat[i,j]=100*(sr-ir) # computed minus ICSD
    vmax=max(15,np.nanmax(np.abs(mat)))
    im=axd.imshow(mat,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax))
    ticklabs=[{'CrystalWeave':'CrystalWeave','Magpie-22 variant':'Magpie-22\nvariant','Graphlet (CrystalNN)':'Graphlet\n(CrystalNN)','Graphlet (VoronoiNN)':'Graphlet\n(VoronoiNN)','AMD-100':'AMD-100'}[z] for z in order]
    axd.set_xticks(np.arange(len(order))); axd.set_xticklabels(ticklabs)
    axd.set_yticks(np.arange(len(rows))); axd.set_yticklabels([DISPLAY[x] for x in rows])
    for i in range(len(rows)):
        for j in range(len(order)):
            val=mat[i,j]; color='white' if abs(val)>vmax*.48 else '#222'
            label=f'{val:+.1f}'
            if rows[i]=='GNoME' and order[j]=='Magpie-22 variant': label += '*'
            axd.text(j,i,label,ha='center',va='center',fontsize=9,color=color,fontweight='bold')
    for j,v in enumerate(ref): axd.text(j,-.76,f'ICSD {100*v:.1f}%',ha='center',va='bottom',fontsize=8.5,color='#333')
    axd.set_ylim(len(rows)-.5,-1.02)
    axd.set_title('Full-map occupancy relative to in-sample ICSD',pad=6)
    axd.set_ylabel('Computed cohort')
    cbd=fig.colorbar(im,ax=axd,fraction=.045,pad=.025); cbd.set_label('Computed minus ICSD (percentage points)',fontsize=9)
    axd.text(.01,.02,'* map-region reweighting: unresolved',transform=axd.transAxes,ha='left',fontsize=7.8,color='white',fontweight='bold')
    axd.text(.99,.02,'negative: ICSD higher',transform=axd.transAxes,ha='right',fontsize=8.5,color='#444')

    fig.suptitle('Computed cohorts in experimental reference maps',fontsize=15,fontweight='bold',y=.982)
    fig.text(.5,.018,'Panels a,b show 386 structures per cohort (seed 42; MatterGen in full). Panel c uses all scored records. Panel d uses exact five-representation common support and an in-sample ICSD reference.',ha='center',fontsize=8.8,color='#444')
    for ax in [axa,axb,axc,axd]:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for extn in ['png','svg','pdf']:
        fig.savefig(OUT/f'fig3_5source_calibration_repaired.{extn}',dpi=300 if extn=='png' else None,bbox_inches='tight')
    plt.close(fig)


def wilson(k,n,z=1.959963984540054):
    p=k/n; den=1+z*z/n; cen=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return cen-half,cen+half

def fig4():
    fig=plt.figure(figsize=(14.8,10.0))
    gs=fig.add_gridspec(2,2,left=.075,right=.97,bottom=.115,top=.92,hspace=.38,wspace=.30)
    # a. Established formula layers, scored cohorts only.
    ax=fig.add_subplot(gs[0,0]); panel_label(ax,'a')
    fl=json.loads(INPUTS['formula_layers'].read_text())['sources']['scored_post1980']
    keys=SOURCE_ORDER; jsrc={'MP':'MP-theoretical',**{k:k for k in keys if k!='MP'}}
    measures=[('integer_formula_exact_rate','Exact formula'),('element_set_rate','Element set'),('anonymous_integer_stoichiometry_rate','Anonymous stoichiometry')]
    mcolors=['#3a5674','#d38128','#5b9d78']; x=np.arange(len(keys)); width=.23
    for h,(field,label) in enumerate(measures):
        vals=[100*fl[jsrc[k]][field] for k in keys]
        bars=ax.bar(x+(h-1)*width,vals,width,color=mcolors[h],label=label,edgecolor='white',lw=.4)
        for b,v in zip(bars,vals):
            if v<.05: txt='0'
            elif v<10: txt=f'{v:.1f}'
            else: txt=f'{v:.0f}'
            ax.text(b.get_x()+b.get_width()/2,v+1.3,txt,ha='center',va='bottom',fontsize=7.8,rotation=90 if v<8 else 0)
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[k].replace('-theoretical','').replace('-DFT','') for k in keys],rotation=18,ha='right')
    ax.set_ylim(0,108); ax.set_yticks(np.arange(0,101,20)); ax.set_ylabel('Matching structures (%)')
    ax.set_title('Composition precedent in analyzed post-1980 ICSD',pad=6)
    ax.legend(loc='upper left',frameon=False,ncol=3,columnspacing=.8,handletextpad=.35)
    ax.grid(axis='y',color='#e5e5e5',lw=.7); ax.set_axisbelow(True)

    # b. ElMD summaries on the structurally scored records.
    ax=fig.add_subplot(gs[0,1]); panel_label(ax,'b')
    ed=pd.read_csv(INPUTS['elmd']); emap={'gnome':'GNoME','mattergen':'MatterGen','mp':'MP','jarvis':'JARVIS','alexandria':'Alexandria'}; ed['key']=ed.source.map(emap)
    ext=load_ext(); groups=[]
    for key in keys:
        ids=set(ext[key].material_id.astype(str)); g=ed[(ed.key==key)&(ed.material_id.astype(str).isin(ids))]
        groups.append(g.post1980_through2015_icsd_nearest_elmd.dropna().to_numpy())
    y=np.arange(len(keys))
    # 10-90 whisker, IQR thick line, median dot: a cutoff-free distribution summary.
    for yi,key,g in zip(y,keys,groups):
        q=np.quantile(g,[.1,.25,.5,.75,.9])
        ax.plot([q[0],q[4]],[yi,yi],color=COLORS[key],lw=1.5,alpha=.7)
        ax.plot([q[1],q[3]],[yi,yi],color=COLORS[key],lw=7,solid_capstyle='butt',alpha=.55)
        ax.scatter(q[2],yi,s=45,c=COLORS[key],edgecolors='white',lw=.7,zorder=3)
        ax.text(q[2]+.10,yi-.20,f'{q[2]:.2f}',fontsize=8.5,color=COLORS[key],fontweight='bold')
    ax.set_yticks(y); ax.set_yticklabels([DISPLAY[k] for k in keys]); ax.invert_yaxis(); ax.set_xlim(-.05,4.1)
    ax.set_xlabel('Nearest-ICSD Element Mover’s Distance (lower = closer)')
    ax.set_title('Nearest ElMD to dated post-1980 ICSD\n(median, IQR, and 10–90% range)',pad=6)
    ax.grid(axis='x',color='#e5e5e5',lw=.7); ax.set_axisbelow(True)

    # c. Corrected scale-invariant formula × CrystalWeave full-map quadrant shares.
    ax=fig.add_subplot(gs[1,0]); panel_label(ax,'c')
    qd=json.loads(INPUTS['quadrants'].read_text()); cats=[
      ('in_basin__formula_match','both precedents','#356b66'),
      ('in_basin__no_formula_match','structure only','#4f8dbc'),
      ('frontier__formula_match','formula only','#d98b35'),
      ('frontier__no_formula_match','neither','#b9bcc2')]
    left=np.zeros(len(keys)); yy=np.arange(len(keys))
    for field,label,color in cats:
        vals=np.array([100*qd['MP-theoretical' if k=='MP' else k]['scale_invariant_quadrant'].get(field,0)/qd['MP-theoretical' if k=='MP' else k]['n'] for k in keys])
        bars=ax.barh(yy,vals,left=left,color=color,height=.68,label=label,edgecolor='white',lw=.6)
        for i,(v,l) in enumerate(zip(vals,left)):
            if v>=6: ax.text(l+v/2,i,f'{v:.0f}%',ha='center',va='center',fontsize=8.5,color='white' if color!='#b9bcc2' else '#222',fontweight='bold')
        left+=vals
    ax.set_yticks(yy); ax.set_yticklabels([DISPLAY[k] for k in keys]); ax.invert_yaxis(); ax.set_xlim(0,100); ax.set_xlabel('Structures (%)')
    ax.set_title('CrystalWeave basin × formula precedent',pad=6)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.20),ncol=4,frameon=False,columnspacing=.8,handletextpad=.35)

    # d. Direct A-Lab outcome association.
    ax=fig.add_subplot(gs[1,1]); panel_label(ax,'d')
    vals=[28/32,8/19]; ks=[28,8]; ns=[32,19]; labs=['In basin','Frontier']; cols=['#356b66','#d98b35']
    cis=[wilson(k,n) for k,n in zip(ks,ns)]; xp=np.arange(2)
    bars=ax.bar(xp,np.array(vals)*100,width=.58,color=cols,edgecolor='white',lw=.6)
    lo=np.array([c[0] for c in cis])*100; hi=np.array([c[1] for c in cis])*100; vv=np.array(vals)*100
    ax.errorbar(xp,vv,yerr=[vv-lo,hi-vv],fmt='none',ecolor='#222',capsize=5,lw=1.2)
    for b,v,k,n in zip(bars,vv,ks,ns): ax.text(b.get_x()+b.get_width()/2,v-8,f'{v:.1f}%\n({k}/{n})',ha='center',va='top',fontweight='bold',color='white')
    ax.set_xticks(xp); ax.set_xticklabels(labs); ax.set_ylim(0,105); ax.set_ylabel('Targets made (%)')
    ax.set_title('A-Lab outcomes by CrystalWeave basin status',pad=6)
    ax.text(.5,.88,'Fisher two-sided p = 0.0011',transform=ax.transAxes,ha='center',fontsize=9.3)
    ax.grid(axis='y',color='#e5e5e5',lw=.7); ax.set_axisbelow(True)

    fig.suptitle('Structural and compositional precedent',fontsize=15,fontweight='bold',y=.978)
    fig.text(.5,.022,'Formula layers compare scale-invariant compositions; partial-occupancy ICSD references use normalized fractions. Structural panels use the declared CrystalWeave community-envelope rule; ElMD has no pass/fail threshold.',ha='center',fontsize=8.8,color='#444')
    for ax in fig.axes:
        if hasattr(ax,'spines'):
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for extn in ['png','svg','pdf']:
        fig.savefig(OUT/f'synth_prior_quadrant_repaired.{extn}',dpi=300 if extn=='png' else None,bbox_inches='tight')
    plt.close(fig)


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures', nargs='+', choices=['3', '4'], default=['3', '4'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if '3' in args.figures: fig3()
    if '4' in args.figures: fig4()
    prov={'script':str(Path(__file__).resolve()),'script_sha256':sha256(Path(__file__)), 'rebuilt_figures':args.figures,'inputs':{k:{'path':str(v),'sha256':sha256(v)} for k,v in INPUTS.items()},
      'figure3':{'overlay':'386 per cohort, seed 42; MatterGen in full; shared CrystalWeave PCA/KDE background','panel_c':'independently cutoff-trained CrystalWeave rates and Wilson 95% intervals','panel_d':'computed-cohort minus in-sample ICSD occupancy on unchanged exact five-representation common support; AMD uses repaired positive-radius basin eligibility and reassignment'},
      'figure4':{'panel_a':'scored-cohort scale-invariant exact formula, element-set and anonymous-stoichiometry rates against 81,493-record occupancy-aware analyzed post-1980 first-report reference','panel_b':'nearest ElMD to complete dated post-1980-through-2015 ICSD, restricted to CrystalWeave-scored cohort IDs; 10th/25th/50th/75th/90th percentiles','panel_c':'scale-invariant formula flag crossed with full-map CrystalWeave basin flag','panel_d':'made fraction among 51 definitive A-Lab outcomes, Wilson 95% intervals'},
      'outputs':[str(OUT/f'{name}.{e}') for name in ['fig3_5source_calibration_repaired','synth_prior_quadrant_repaired'] for e in ['png','svg','pdf']]}
    (OUT/'fig3_fig4_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    print(json.dumps(prov,indent=2))
if __name__=='__main__': main()
