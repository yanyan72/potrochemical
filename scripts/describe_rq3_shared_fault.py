"""Read-only descriptive contrasts from the complete fixed experiment."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path('results/rq3_prediction/rq3_shared_fault_20261023_v1')
OUT=Path('results/rq3_prediction/rq3_shared_fault_diagnostics_20261023')


def ci(v: list[float]) -> dict:
    """Summarize five seed differences without treating paired views as repeats."""
    v=np.array(v);h=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
    return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),ci_low=float(v.mean()-h),ci_high=float(v.mean()+h),negative_seeds=int((v<0).sum()))


def main() -> None:
    """Save all predefined channel degradations and descriptive summaries."""
    cfg=json.loads((ROOT/'config.json').read_text());rows=[];global_rows=[];score=[]
    for seed in cfg['seeds']:
        sd=ROOT/f'seed_{seed}/evaluation'
        with np.load(sd/'predictions.npz') as p:
            err=np.abs(p['prediction']-p['targets'][None,None,:,:]);arms=list(p['arms']);views=list(p['views']);names=list(p['target_names'])
        ev=json.loads((sd/'test/events.json').read_text());ch=np.array([e['channel'] for e in ev])
        for ti,target in enumerate(names):
            for ai,arm in enumerate(arms):
                global_rows.append(dict(seed=seed,target=target,arm=arm,suite_mae=float(err[:,ai,:,ti].mean()),original_mae=float(err[0,ai,:,ti].mean())))
            for vi,view in enumerate(views):
                if not view.startswith(('bias','drift','missing')):continue
                for c in range(3):
                    mask=ch==c
                    for ai,arm in enumerate(arms):
                        rows.append(dict(seed=seed,target=target,view=view,channel=c,arm=arm,batches=int(mask.sum()),mae=float(err[vi,ai,mask,ti].mean()),original_same_batches_mae=float(err[0,ai,mask,ti].mean()),degradation=float((err[vi,ai,mask,ti]-err[0,ai,mask,ti]).mean())))
        with (sd/'score_metrics.csv').open() as f:score.extend(csv.DictReader(f))
    OUT.mkdir(exist_ok=False)
    for name,data in [('channel_degradation_by_seed.csv',rows),('global_by_seed.csv',global_rows)]:
        with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    global_summary=[]
    for target in names:
        for arm in arms:
            rr=[r for r in global_rows if r['target']==target and r['arm']==arm]
            global_summary.append(dict(target=target,arm=arm,suite=ci([r['suite_mae'] for r in rr]),original=ci([r['original_mae'] for r in rr])))
    channel_summary=[]
    for target in names:
        for view in ['bias_4','drift_1']:
            for c in range(3):
                for arm in arms:
                    rr=[r for r in rows if r['target']==target and r['view']==view and r['channel']==c and r['arm']==arm]
                    channel_summary.append(dict(target=target,view=view,channel=c,arm=arm,degradation=ci([r['degradation'] for r in rr])))
    score_summary=[]
    for view in sorted({r['view'] for r in score}):
        for stratum in ['all','channel:0','channel:1','channel:2']:
            rr=[r for r in score if r['view']==view and r['stratum']==stratum]
            score_summary.append(dict(view=view,stratum=stratum,**{k:ci([float(r[k]) for r in rr]) for k in ['point_recall','fresh_event_recall','availability']}))
    doc=dict(global_metrics=global_summary,channel_degradation=channel_summary,score=score_summary)
    (OUT/'descriptive_summary.json').write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps(dict(global_metrics=[r for r in global_summary if r['target']=='arrival_proxy'],weak_drift=next(r for r in score_summary if r['view']=='drift_1' and r['stratum']=='all')),indent=2))

if __name__=='__main__':main()
