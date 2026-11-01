"""Complete paired summaries and mechanism windows from saved scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path('results');OUT=ROOT/'analysis_v1';OUT.mkdir(exist_ok=False)
d=pd.read_csv(ROOT/'factorial_validation_v1/metrics_by_seed.csv',dtype={'channels':str});d=d[(d.stratum=='all')&(d.regime==-1)]
metrics=['point_recall','bystander_fpr','clean_fpr','mode_error','event_added_recall','all_channels_added_recall','exact_set_event_recall','posterior_tv','true_p_loss'];pairs=[]
for keys,f in d.groupby(['lag','channels','magnitude']):
 for metric in metrics:
  a=f.pivot(index='seed',columns='arm',values=metric)
  for baseline in ['joint12','soft3','hard']:
   diff=a.factorial24-a[baseline]
   if diff.isna().all():continue
   sd=diff.std(ddof=1);h=t.ppf(.975,4)*sd/np.sqrt(5)
   pairs.append(dict(lag=keys[0],channels=keys[1],magnitude=keys[2],metric=metric,baseline=baseline,baseline_mean=a[baseline].mean(),candidate_mean=a.factorial24.mean(),mean=diff.mean(),sd=sd,ci_low=diff.mean()-h,ci_high=diff.mean()+h,negative=int((diff<0).sum()),positive=int((diff>0).sum()),**{f'seed_{s}':v for s,v in diff.items()}))
p=pd.DataFrame(pairs);p.to_csv(OUT/'paired_comparisons.csv',index=False)
windows=[]
for sd in sorted((ROOT/'factorial_validation_v1').glob('seed_*')):
 with np.load(sd/'inputs_and_scores.npz') as z:
  for vi,meta in enumerate(json.loads((sd/'views.json').read_text())):
   if meta['magnitude']==0:continue
   for ai,arm in enumerate(z['arms']):
    a=z['alarm'][vi,ai];mask=z['fault_mask'][vi];starts=z['starts'];q=z['q'][vi,ai];truth=z['truth']
    for win in ['event','post6','post_rest']:
     nc=fp=nt=err=0
     for i,s in enumerate(starts):
      end=s+24;b,e=(s,end) if win=='event' else (end,min(end+6,len(truth[i]))) if win=='post6' else (min(end+6,len(truth[i])),len(truth[i]))
      active=mask[i].any(0);normal=(~mask[i,b:e])&active[None,:]
      nc+=normal.sum();fp+=(a[i,b:e]&normal).sum();nt+=e-b;err+=(q[i,b:e].argmax(-1)!=truth[i,b:e]).sum()
     windows.append(dict(seed=int(sd.name[5:]),arm=str(arm),window=win,**meta,formerly_faulty_clean_fp=int(fp),formerly_faulty_clean_cells=int(nc),formerly_faulty_clean_fpr=float(fp/nc) if nc else np.nan,mode_error=float(err/nt)))
pd.DataFrame(windows).to_csv(OUT/'post_event_windows.csv',index=False)
print(p[(p.channels.isin(['01','02','12']))&(p.magnitude==4)&(p.lag==12)&(p.baseline=='joint12')&(p.metric.isin(['point_recall','bystander_fpr','mode_error']))].to_string(index=False))
w=pd.DataFrame(windows);print(w[(w.channels.isin(['02','12']))&(w.magnitude==4)&w.arm.isin(['joint12','factorial24'])].groupby(['window','arm'])[['formerly_faulty_clean_fpr','mode_error']].mean().to_string())
rq=pd.read_csv(ROOT/'rq1_confirmation_v1/metrics_by_seed.csv');sel=rq[(rq.channel==-1)&(rq.regime=='all')]
print(sel.groupby(['magnitude','method'])[['clean_fpr','recall']].mean().to_string())
