"""Posthoc hypothesis-mass interpretation on fixed archived candidate outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .rq2_factorial_fault import factorial_filter
root=Path('results/factorial_validation_v1');rows=[]
for sd in sorted(root.glob('seed_*')):
 pars=json.loads((sd/'parameters.json').read_text());views=json.loads((sd/'views.json').read_text())
 with np.load(sd/'inputs_and_scores.npz') as z:
  ai=list(z['arms']).index('factorial24')
  for vi,meta in enumerate(views):
   result=factorial_filter(z['observations'][vi],z['records'][vi],pars['model']['modes'],pars['candidate']);np.testing.assert_array_equal(result['mode'],z['q'][vi,ai]);bits=result['masks'];mass=result['fault_mask'];window=np.zeros(z['truth'].shape,bool)
   for i,s in enumerate(z['starts']):window[i,s:s+24]=True
   actual=z['fault_mask'][vi].any(1);correct=(bits[None,:,:]==actual[:,None,:]).all(-1)
   exact=(mass*correct[:,None,:]).sum(-1);multi=mass[:,:,bits.sum(1)>=2].sum(-1)
   row=dict(seed=int(sd.name[5:]),**meta,exact_mask_mass=float(exact[window].mean()),multi_fault_mass=float(multi[window].mean()),expected_fault_count=float((mass@bits.sum(1))[window].mean()))
   rows.append(row)
pd.DataFrame(rows).to_csv('results/analysis_v1/posterior_hypothesis_mass.csv',index=False)
d=pd.DataFrame(rows);print(d[(d.magnitude.isin([0,4]))].groupby(['lag','channels','magnitude'])[['exact_mask_mass','multi_fault_mass']].mean().to_string())
