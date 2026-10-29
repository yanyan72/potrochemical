"""Replay persisted scores and independently recount event/cell metrics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import rq3_uncertainty_a0 as m

def verify(root):
    cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    for path,h in manifest.items():assert m.digest(root/path)==h,path
    summary=list(csv.DictReader((root/'metrics_by_seed.csv').open()))
    lookup={(int(r['seed']),r['split'],r['arm'],r['view'],r['stratum']):r for r in summary}
    max_ratio=0.;max_q=0.;rows_checked=0;trajectories=0
    diagnostics=[]
    for seed in cfg['seeds']:
        sd=root/f'seed_{seed}';model=json.loads((sd/'model.json').read_text())
        for split in cfg['splits']:
            events=json.loads((sd/f'{split}_events.json').read_text());rows=list(csv.DictReader((sd/f'{split}_batch_metrics.csv').open()))
            bykey={(r['view'],r['arm'],r['sequence_id']):r for r in rows}
            for view in cfg['views']:
                a=m.npz(sd/f'{split}_{view}.npz');rec,kind=view.split('_');clean=m.npz(sd/f'{split}_{rec}_clean.npz')
                for arm in cfg['arms']:
                    if arm=='legacy':ratio,alarm=m.legacy_score(a['observations'],a['recorded_regime'],model['legacy'])
                    else:
                        raw,q=m.matched_raw(a['observations'],a['recorded_regime'],model['modes'],cfg,arm,a['true_regime_evaluation_only'] if arm=='oracle_matched' else None)
                        ratio=raw/model['thresholds'][arm];alarm=ratio>1
                        if arm.startswith('soft'):
                            max_q=max(max_q,float(np.max(np.abs(q-a[arm+'_q']))));assert np.allclose(q,a[arm+'_q'],rtol=0,atol=1e-7)
                    max_ratio=max(max_ratio,float(np.max(np.abs(ratio-a[arm+'_ratio']))))
                    assert np.allclose(ratio,a[arm+'_ratio'],atol=1e-6,rtol=1e-7)
                    assert np.array_equal(alarm,a[arm+'_alarm']);trajectories+=len(alarm)
                    totals=dict(clean_fp=0,clean_cells=0,tp=0,fault_cells=0,bystander_fp=0,bystander_cells=0)
                    fresh=[];added=[]
                    for i,e in enumerate(events):
                        mask=a['fault_mask'][i];al=alarm[i];co=clean[arm+'_alarm'][i];j=e['channel'];s=e['start'];end=e['end']
                        r=bykey[(view,arm,str(a['sequence_id'][i]))]
                        counters=dict(clean_fp=int(al[~mask].sum()),clean_cells=int((~mask).sum()),tp=int(al[mask].sum()),fault_cells=int(mask.sum()))
                        active=bool(mask.any());other=[ch for ch in range(3) if ch!=j]
                        counters['bystander_fp']=int(al[s:end][:,other].sum()) if active else 0
                        counters['bystander_cells']=(end-s)*2 if active else 0
                        onset=[t for t in range(s,end) if al[t,j] and (t==0 or not al[t-1,j])]
                        counters.update(event_any=int(active and al[s:end,j].any()),event_fresh=int(active and bool(onset)),event_added=int(active and np.any(al[s:end,j]&~co[s:end,j])),event_pre_active=int(active and al[s-1,j]),delay_capped=(onset[0]-s) if active and onset else end-s)
                        for key,value in counters.items():assert int(r[key])==value,(seed,split,view,arm,i,key)
                        for key in totals:totals[key]+=counters[key]
                        if active:fresh.append(counters['event_fresh']);added.append(counters['event_added'])
                        rows_checked+=1
                        if split=='val' and active and arm.startswith('soft'):
                            qs=a[arm+'_q'][i,s:end];cq=clean[arm+'_q'][i,s:end];truth=a['true_regime_evaluation_only'][i,s:end,None]
                            wrong=(qs.argmax(-1)!=truth)[:,other];prior=(cq.argmax(-1)!=truth)[:,other]
                            alarms=al[s:end][:,other]
                            diagnostics.append(dict(seed=seed,view=view,arm=arm,sequence_id=str(a['sequence_id'][i]),fault_channel=j,
                                clean_channel_cells=wrong.size,clean_channel_mode_errors=int(wrong.sum()),paired_clean_mode_errors=int(prior.sum()),
                                clean_channel_alarms=int(alarms.sum()),alarms_with_wrong_mode=int((alarms&wrong).sum())))
                    rr=lookup[(seed,split,arm,view,'all')]
                    for key,value in totals.items():assert int(rr[key])==value,(key,rr[key],value)
                    for key,arr in [('event_fresh',fresh),('event_added',added)]:
                        if arr:assert abs(float(rr[key])-np.mean(arr))<1e-12
                        else:assert rr[key]==''
    decision=json.loads((root/'decision.json').read_text())
    assert decision['advance_A1']==any(decision[k]['pass_screen'] for k in ('soft_all','soft_loo'))
    assert decision['rq3_supported'] is False  # This stage has no quality predictions.
    return dict(passed=True,manifest_files=len(manifest),batch_arm_view_rows_recounted=rows_checked,
                score_trajectories_replayed=trajectories,alarm_replay_exact=True,max_float32_ratio_error=max_ratio,max_float32_probability_error=max_q,
                inference='3 outer data seeds; development; no p values; no quality prediction'),diagnostics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    r,d=verify(a.root);m.dump(a.output/'verification.json',r);m.csv_write(a.output/'posthoc_mode_bystander_diagnostics.csv',d);print(json.dumps(r))
