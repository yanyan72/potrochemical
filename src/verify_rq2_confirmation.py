"""Independent sufficient-count check from archived alarms and masks."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
root=Path('results/confirmation_20261102/run_v1');rows=0
for rel,h in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h
for sd in sorted(root.glob('seed_*')):
 with np.load(sd/'scores.npz') as z, np.load(sd/'inputs.npz') as inp:
  z={k:z[k] for k in z.files};inp={k:inp[k] for k in inp.files}
  df=pd.read_csv(sd/'batch_metrics.csv');views=json.loads((sd/'views.json').read_text());vlookup={r['view']:i for i,r in enumerate(views)};arms=list(z['arms']);np.testing.assert_allclose(z['q'].sum(-1),1,atol=1e-12)
  for r in df.itertuples():
   vi=vlookup[r.view];ai=arms.index(r.arm);i=r.batch;a=z['alarm'][vi,ai,i];mask=z['fault_mask'][vi,i];truth=inp['recorded_regime'][i];take=np.ones(len(truth),bool) if r.regime==-1 else truth==r.regime;sup=take[:,None]
   assert (a&mask&sup).sum()==r.tp and (mask&sup).sum()==r.fault_cells
   assert (a&~mask&sup).sum()==r.clean_fp and (~mask&sup).sum()==r.clean_cells
   win=np.zeros(len(truth),bool);s=z['starts'][i];win[s:s+24]=True;by=win[:,None]&~mask.any(0)[None,:]&sup
   assert (a&by).sum()==r.bystander_fp and by.sum()==r.bystander_cells
   assert ((z['q'][vi,ai,i].argmax(-1)!=truth)&take).sum()==r.mode_errors;rows+=1
# Independent seed-primary recount directly from batch sufficient counts.
f=pd.concat([pd.read_csv(sd/'batch_metrics.csv',dtype={'channels':str}) for sd in sorted(root.glob('seed_*'))]);f=f[(f.regime==-1)&(f.magnitude==4)&f.channels.isin(['01','02','12'])]
g=f.groupby(['seed','arm','view'])[['tp','fault_cells','bystander_fp','bystander_cells']].sum();g['point_recall']=g.tp/g.fault_cells;g['bystander_fpr']=g.bystander_fp/g.bystander_cells
for r in json.loads((root/'primary.json').read_text()):
 a=g.groupby(['seed','arm'])[r['metric']].mean().unstack();np.testing.assert_allclose((a.full24-a.restricted12).to_numpy(),r['seed_differences'],atol=1e-14)
result=dict(rows=rows,passed=True,primary_recomputed=True);Path('results/confirmation_20261102/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
d=pd.read_csv(root/'metrics_by_seed.csv',dtype={'channels':str});d=d[(d.stratum=='all')&(d.regime==-1)]
print(d[(d.channels=='none')|((d.channels=='2')&(d.magnitude==4)&(d.lag==12))].groupby(['lag','channels','arm'])[['point_recall','bystander_fpr','clean_fpr']].mean().to_string())
