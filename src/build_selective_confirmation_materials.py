"""Create manuscript tables and static scientific plots from frozen confirmation."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/rq3_selective_confirmation_20261030_v1')
FIG=Path('results/rq3_selective_confirmation_figures_20261030_v1')
FIG.mkdir(exist_ok=False)
def table(n):return list(csv.DictReader((ROOT/n).open()))
rows=table('metrics_by_repeat.csv');pairs=table('paired_summary.csv');byseed=table('paired_by_seed.csv');cfg=json.loads((ROOT/'config.json').read_text())
views=cfg['views'];arms=cfg['arms'];comps=['SG-N','SG-SR','SG-NG','SG-SA'];names=['Correct / clean','Correct / weak drift','Correct / strong bias','Delayed / clean','Delayed / weak drift','Delayed / strong bias']
def get(comp,view='suite',stratum='all',target='arrival_proxy'):
    return next(r for r in pairs if r['target']==target and r['comparison']==comp and r['view']==view and r['stratum']==stratum)
def mae(arm,view,seed=None,target='arrival_proxy'):
    return np.mean([float(r['mae']) for r in rows if r['target']==target and r['arm']==arm and r['view']==view and r['stratum']=='all' and (seed is None or int(r['seed'])==seed)])
def save(name):
    plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=180);plt.savefig(FIG/(name+'.pdf'));plt.close()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8,4.5));colors=['#147d92','#dc8245','#4a69a1','#895292']
for i,(co,col) in enumerate(zip(comps,colors)):
    r=get(co);m=float(r['mean']);lo=float(r['ci_low']);hi=float(r['ci_high']);ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=col,capsize=4,lw=2,label='Mean and 95% t interval' if i==0 else None)
    v=[float(r['mae_difference']) for r in byseed if r['target']=='arrival_proxy' and r['comparison']==co and r['view']=='suite' and r['stratum']=='all'];ax.scatter(v,np.full(5,i)+np.linspace(-.12,.12,5),s=20,color=col,alpha=.5)
ax.axvline(0,color='grey',ls='--');ax.set_yticks(range(4),comps);ax.invert_yaxis();ax.set_xlabel('Paired MAE difference (negative favors SG)');ax.set_title('Independent holdout: five data seeds, two optimization repeats');ax.legend(loc='best',fontsize=8);save('01_primary_confirmation')
fig,ax=plt.subplots(figsize=(10,4.8));x=np.arange(6);w=.19
for i,(co,col) in enumerate(zip(comps,colors)):
    means=[float(get(co,v)['mean']) for v in views];ax.bar(x+(i-1.5)*w,means,w,label=co,color=col)
ax.axhline(0,color='grey',lw=.8);ax.set_xticks(x,[s.replace(' / ','\n') for s in names]);ax.set_ylabel('Mean paired MAE difference');ax.set_title('Six prespecified scenarios (all seeds and optimization repeats retained)');ax.legend(ncol=4);save('02_six_scenarios')
fig,axs=plt.subplots(1,3,figsize=(11,4.2),sharey=True)
for ax,kind,title in zip(axs,['clean','drift1','bias4'],['Clean observations','Weak drift','Strong bias']):
    for arm,col in [('NG','#999999'),('SG','#147d92'),('SA','#895292')]:
        vals=[np.mean([mae(arm,f'{rec}_{kind}',s)-mae('N',f'{rec}_{kind}',s) for rec in ['correct','delay12']]) for s in cfg['seeds']];ax.plot(cfg['seeds'],vals,'o-',label=arm,color=col)
    ax.axhline(0,color='grey',ls='--');ax.set_xticks(cfg['seeds']);ax.set_title(title);ax.set_xlabel('Data seed')
axs[0].set_ylabel('MAE difference from N');axs[-1].legend();fig.suptitle('Recovery versus information retention: optimizer-averaged seed results');save('03_tradeoff_by_seed')
fig,axs=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
for ax,view in zip(axs,['correct_bias4','delay12_bias4']):
    for i,(co,col) in enumerate(zip(['SG-N','SG-SA'],[colors[0],colors[3]])):
        vals=[float(get(co,view,f'channel_{j}')['mean']) for j in range(3)];ax.bar(np.arange(3)+(i-.5)*.32,vals,.32,label=co,color=col)
    ax.axhline(0,color='grey',ls='--');ax.set_xticks(range(3),['Temperature','Humidity','Vibration']);ax.set_title('Correct records' if view.startswith('correct') else 'Delayed records');ax.legend()
axs[0].set_ylabel('Mean paired MAE difference');fig.suptitle('Strong-bias channel attribution (prespecified descriptive strata)');save('04_channel_attribution')
lines=['# Confirmation tables — 2026-10-30','', 'Differences are SG minus comparator; negative favors SG. Main target is a synthetic temperature-exposure proxy. Optimization repeats are averaged within each data seed before inference.','', '## Primary comparisons','', '| Comparison | Mean difference | SD | 95% t interval | Improving seeds | Leave-one-seed mean range |','|---|---:|---:|---|---:|---|']
for co in comps:
    r=get(co);lines.append(f"| {co} | {float(r['mean']):+.6f} | {float(r['sd']):.6f} | [{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}] | {r['negative_seeds']}/5 | [{float(r['loo_min']):+.6f}, {float(r['loo_max']):+.6f}] |")
lines+=['','## Six-scenario MAE','', '| Scenario | N | NG | SG | SR | SA | SG−N | SG−SR | SG−NG | SG−SA |','|---|'+'---:|'*9]
for v in views:lines.append('| '+v+' | '+' | '.join([f'{mae(a,v):.6f}' for a in arms]+[f"{float(get(co,v)['mean']):+.6f}" for co in comps])+' |')
lines+=['','## Every data seed and scenario','', '| Seed | Scenario | N | NG | SG | SR | SA | SG−N | SG−SR | SG−NG | SG−SA |','|---:|---|'+'---:|'*9]
for s in cfg['seeds']:
    for v in ['suite']+views:
        lines.append(f'| {s} | {v} | '+' | '.join([f'{mae(a,v,s):.6f}' for a in arms]+[f'{mae("SG",v,s)-mae(co.split("-")[1],v,s):+.6f}' for co in comps])+' |')
lines+=['','## Optimization-repeat sensitivity','', '| Repeat | Comparison | Mean difference | 95% t interval | Improving seeds |','|---:|---|---:|---|---:|']
for r in table('optimizer_sensitivity.csv'):
    if r['target']=='arrival_proxy' and r['view']=='suite' and r['stratum']=='all':lines.append(f"| {r['repeat']} | {r['comparison']} | {float(r['mean']):+.6f} | [{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}] | {r['negative_seeds']}/5 |")
lines+=['','## Independent-label negative control','', '| Arm | Null-target suite MAE |','|---|---:|']
for a in ['train_mean','N','SG','SR']:lines.append(f'| {a} | {mae(a,"suite",target="null_proxy"):.6f} |')
lines+=['','| Comparison | Mean difference | 95% t interval | Improving seeds |','|---|---:|---|---:|']
for co in ['SG-N','SG-SR']:
    r=get(co,target='null_proxy');lines.append(f"| {co} | {float(r['mean']):+.6f} | [{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}] | {r['negative_seeds']}/5 |")
lines+=['','## Secondary delay-six stress views','', '| View | N | SG | SG−N | SG−SR | SG−NG | SG−SA |','|---|'+'---:|'*6]
for v in cfg['secondary_views']:lines.append('| '+v+' | '+' | '.join([f'{mae(a,v):.6f}' for a in ['N','SG']]+[f"{float(get(co,v)['mean']):+.6f}" for co in comps])+' |')
Path('docs/RQ3_CONFIRMATION_TABLES_20261030.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'suite_mae':{a:mae(a,'suite') for a in arms},'figures':str(FIG)}))
