"""Rebuild frozen scores and independently count detection/intervention diagnostics."""
import csv,json
from pathlib import Path
import numpy as np
import rq3_selective_confirmation as c
import rq3_uncertainty_a0 as a0
from rq3_selective_prediction_dev import trigger
root=Path('results/rq3_selective_confirmation_20261030_v1');cfg=c.read(root/'config.json');checked=0;summary=[]
for seed in cfg['seeds']:
    sd=root/'training'/f'seed_{seed}';ed=root/'evaluation'/f'seed_{seed}';v=a0.npz(ed/'inputs.npz');events=c.read(ed/'events.json');views=cfg['views']+cfg['secondary_views'];model=c.read(sd/'score_model.json')
    b=c.bundle(v['observations'],v['recorded_regime'],events,np.asarray(c.read(sd/'fault_scales.json')),model,c.read(sd/'candidate.json'),c.read(sd/'score_config.json'),c.read(sd/'view_config.json'),views)
    rows=list(csv.DictReader((ed/'score_metrics.csv').open()))
    for r in rows:
        vi=views.index(r['view']);ci=views.index(r['view'].split('_')[0]+'_clean');scores=b['S'] if r['arm']=='SG' else b['ablationS'];alarm=scores[vi]<cfg['trigger_score'];clean_alarm=scores[ci]<cfg['trigger_score'];mask=b['mask'][vi]
        select=[i for i,e in enumerate(events) if r['stratum']=='all' or e['channel']==int(r['stratum'].split('_')[-1])];event_any=[];added=[];byfp=bycells=0
        for i in select:
            e=events[i];ch=e['channel'];s=e['start'];end=e['end']
            if mask[i].any():
                event_any.append(alarm[i,s:end,ch].any());added.append((alarm[i,s:end,ch]&~clean_alarm[i,s:end,ch]).any());other=[j for j in range(3) if j!=ch];byfp+=alarm[i,s:end][:,other].sum();bycells+=(end-s)*2
        nmask=mask[select];al=alarm[select]
        expected=dict(clean_fp=int((al&~nmask).sum()),clean_cells=int((~nmask).sum()),tp=int((al&nmask).sum()),fault_cells=int(nmask.sum()),bystander_fp=int(byfp),bystander_cells=int(bycells))
        for k,val in expected.items():assert int(r[k])==val,(seed,r,k)
        if added:np.testing.assert_allclose([float(r['event_any']),float(r['event_added']),float(r['bystander_fpr']),float(r['point_recall'])],[np.mean(event_any),np.mean(added),byfp/bycells,expected['tp']/expected['fault_cells']],atol=1e-14,rtol=0)
        checked+=1
        for key in ['clean_fpr','point_recall','bystander_fpr','event_added']:
            if r[key]!='':summary.append(dict(seed=seed,arm=r['arm'],view=r['view'],stratum=r['stratum'],metric=key,value=float(r[key])))
    for r in csv.DictReader((ed/'activation.csv').open()):
        if r['arm']=='SR':continue
        vi=views.index(r['view']);ch=int(r['channel']);score=b['ablationS'][vi] if r['arm']=='SA' else b['S'][vi];w=1-score if r['arm']=='NG' else trigger(score,cfg['trigger_score']);mask=b['mask'][vi,:,:,ch] if r['region']=='fault' else ~b['mask'][vi,:,:,ch]
        np.testing.assert_allclose([float(r['activation_rate']),float(r['mean_strength'])],[(w[:,:,ch][mask]>0).mean(),w[:,:,ch][mask].mean()],rtol=0,atol=1e-14)
    print('verified diagnostics',seed,flush=True)
agg=[]
for key in sorted({(r['arm'],r['view'],r['stratum'],r['metric']) for r in summary}):
    values=[r['value'] for r in summary if (r['arm'],r['view'],r['stratum'],r['metric'])==key]
    agg.append(dict(zip(['arm','view','stratum','metric'],key),mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),outer_seeds=len(values)))
a0.csv_write(root/'score_diagnostics_summary.csv',agg)
a0.dump(root/'diagnostic_verification.json',dict(passed=True,score_metric_rows=checked,scoring_rebuilt_from_visible_inputs=True,counts_independently_recomputed=True,note='mode-related columns in per-seed score_metrics use the common full-candidate q; do not interpret those columns as ablation-specific mode-inference accuracy'))
