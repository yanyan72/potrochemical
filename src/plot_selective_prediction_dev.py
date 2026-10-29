"""Figure export for selective fusion; all scenarios and data seeds retained."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(root:Path,out:Path)->None:
    """Export three directly reusable PNG/PDF figures from archived tables."""
    out.mkdir(parents=True,exist_ok=False);read=lambda p:list(csv.DictReader(p.open()));pairs=read(root/'paired_by_seed.csv');summary=read(root/'paired_summary.csv');metrics=read(root/'metrics_by_seed.csv')
    views=['correct_clean','correct_drift1','correct_bias4','delay12_clean','delay12_drift1','delay12_bias4'];labels=['Correct\nClean','Correct\nWeak drift','Correct\nStrong bias','Delayed\nClean','Delayed\nWeak drift','Delayed\nStrong bias'];seeds=[42,43,44]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    comps=['SG-N','SG-SR','SG-RG','SG-NG','SG-SA'];mat=np.array([[float(next(r['mean'] for r in summary if r['comparison']==c and r['view']==v and r['stratum']=='all')) for v in views] for c in comps]);lim=np.max(np.abs(mat))
    fig,ax=plt.subplots(figsize=(9,4.5));im=ax.imshow(mat,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(6),labels);ax.set_yticks(range(5),comps)
    for i in range(5):
        for j in range(6):ax.text(j,i,f'{mat[i,j]:+.4f}',ha='center',va='center',fontsize=9,color='white' if abs(mat[i,j])>.7*lim else 'black')
    ax.set_title('Selective correction: paired MAE changes across six scenarios');fig.colorbar(im,ax=ax,label='MAE difference; blue = improvement');save(fig,'01_scenario_comparisons')
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,arm,title in zip(axes,['NG','SG'],['Continuous correction (NG)','Selective correction (SG)']):
        for seed,color in zip(seeds,['#0072B2','#D55E00','#009E73']):
            get=lambda a,v:float(next(r['mae'] for r in metrics if r['seed']==str(seed) and r['arm']==a and r['view']==v and r['stratum']=='all'))
            d=[np.mean([get(arm,v)-get('N',v) for v in views if v.endswith(k)]) for k in ['clean','drift1','bias4']];ax.plot(range(3),d,'o-',label=f'Seed {seed}',color=color)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),['Clean','Weak drift','Strong bias']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('MAE change relative to no-score N');axes[1].legend();fig.suptitle('Recovery versus normal-information loss (record conditions averaged)',y=1.04);save(fig,'02_tradeoff_by_seed')
    fig,ax=plt.subplots(figsize=(8,4.4));xx=np.arange(3)
    for dx,v,color,label in [(-.13,'correct_bias4','#0072B2','Correct record'),(.13,'delay12_bias4','#D55E00','Delayed record')]:
        rr=[next(r for r in summary if r['comparison']=='SG-N' and r['view']==v and r['stratum']==f'channel_{j}') for j in range(3)];ax.errorbar(xx+dx,[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],fmt='o',capsize=4,color=color,label=label)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(xx,['Temperature fault','Humidity fault','Vibration fault']);ax.set(ylabel='Selective correction minus no-score MAE',title='Strong-bias effects by affected channel');ax.legend();ax.text(.01,-.24,'Mean ± SD across 3 data seeds; not confidence intervals.',transform=ax.transAxes,fontsize=9);save(fig,'03_channel_effects')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plot(a.root,a.output)
