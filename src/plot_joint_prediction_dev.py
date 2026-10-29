"""Publication-ready figures from complete development result tables."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(root:Path,analysis:Path,out:Path)->None:
    """Export vector PDFs, PNG previews, and figures without excluding seeds."""
    out.mkdir(parents=True,exist_ok=False)
    read=lambda p:list(csv.DictReader(p.open()))
    suite=read(root/'suite_by_seed.csv');pair=read(root/'paired_summary.csv');strata=read(analysis/'stratified_paired_summary.csv')
    arms=['N','OT','NT','ZG','NG','RG','AG'];seeds=[42,43,44]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'figure.dpi':150})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for seed,color in zip(seeds,['#0072B2','#D55E00','#009E73']):
        vals=[float(next(r['mae'] for r in suite if r['seed']==str(seed) and r['arm']==arm)) for arm in arms]
        ax.plot(arms,vals,'o-',label=f'Data seed {seed}',color=color,lw=1.4)
    ax.set(ylabel='Six-scenario mean absolute error',title='All 21 models: paired development results');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.16));ax.grid(axis='y',alpha=.2);save(fig,'01_all_models')
    views=['correct_clean','correct_drift1','correct_bias4','delay12_clean','delay12_drift1','delay12_bias4'];comps=['NT-OT','NT-N','ZG-NT','NG-ZG','NG-N','NG-RG','NG-AG']
    mat=np.array([[float(next(r['mean'] for r in pair if r['comparison']==c and r['view']==v)) for v in views] for c in comps]);limit=np.max(abs(mat))
    fig,ax=plt.subplots(figsize=(9,5.2));im=ax.imshow(mat,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(6),['Correct\nClean','Correct\nWeak drift','Correct\nStrong bias','Delayed\nClean','Delayed\nWeak drift','Delayed\nStrong bias']);ax.set_yticks(range(len(comps)),comps)
    for i in range(len(comps)):
        for j in range(6):ax.text(j,i,f'{mat[i,j]:+.4f}',ha='center',va='center',fontsize=9,color='white' if abs(mat[i,j])>.7*limit else '#1a1a1a')
    ax.set_title('Paired MAE change by scenario (mean of three data seeds)');fig.colorbar(im,ax=ax,label='MAE difference; blue = improvement');save(fig,'02_scenario_effects')
    fig,ax=plt.subplots(figsize=(8,4.4));xx=np.arange(3)
    for offset,view,color,label in [(-.13,'correct_bias4','#0072B2','Correct record'),(.13,'delay12_bias4','#D55E00','12-step delayed record')]:
        rr=[next(r for r in strata if r['comparison']=='NG-N' and r['view']==view and r['stratum']==f'channel_{j}') for j in range(3)]
        ax.errorbar(xx+offset,[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],fmt='o',capsize=4,color=color,label=label)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(xx,['Temperature fault','Humidity fault','Vibration fault']);ax.set(ylabel='Reference correction minus no-score MAE',title='Strong-bias benefit depends on the affected channel');ax.legend();ax.text(.01,-.24,'Mean ± SD across 3 data seeds; error bars are not confidence intervals.',transform=ax.transAxes,fontsize=9);ax.grid(axis='y',alpha=.2);save(fig,'03_channel_effects')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plot(a.root,a.analysis,a.output)
