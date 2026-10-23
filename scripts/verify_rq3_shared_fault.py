"""Independent arithmetic, provenance and budget audit from persisted arrays."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def csv_rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def verify(root):
    cfg=json.loads((root/'config.json').read_text());checks=dict(files=0,metrics=0,pairs=0,intervals=0,models=0)
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest.items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        assert actual==expected,name
        checks['files']+=1
    raw={}
    for seed in cfg['seeds']:
        tr=root/f'seed_{seed}/training';ev=root/f'seed_{seed}/evaluation'
        m=json.loads((tr/'model_parameters.json').read_text());history=csv_rows(tr/'training_history.csv')
        with np.load(tr/'schedule.npz',allow_pickle=False) as s:
            assert s['shared'].shape==(cfg['epochs'],cfg['counts']['train'])
            expected=np.random.default_rng(cfg['schedule_seed_offset']+seed).integers(0,10,size=s['shared'].shape,dtype=np.int64)
            np.testing.assert_array_equal(s['shared'],expected);assert not s['clean'].any()
        for target in cfg['targets']:
            selected=[v for k,v in m.items() if k.endswith('__'+target)]
            for key in ['initial_state_sha256','shuffle_sha256','parameter_count','optimizer_updates']:
                assert len({v[key] for v in selected})==1
        for k,v in m.items():
            arm,target=k.split('__');hh=[r for r in history if r['arm']==arm and r['target']==target]
            assert len(hh)==cfg['epochs']
            best=min(hh,key=lambda r:float(r['val_suite_mae']))
            assert v['selected_epoch']==int(best['epoch'])
            assert v['optimizer_updates']==cfg['epochs']*int(np.ceil(cfg['counts']['train']/cfg['batch_size']))
            assert v['training_examples']==cfg['epochs']*cfg['counts']['train']
            assert v['replay_max_abs_error']==0
            checks['models']+=1
        for p in [tr,ev]:
            fair=json.loads((p/'fairness.json').read_text())
            for condition in ['clean','shared']:
                rows=[r for r in fair if r['arm'].startswith(condition+'_')]
                for key in set(rows[0])-{'arm'}:assert len({r[key] for r in rows})==1
        with np.load(ev/'predictions.npz',allow_pickle=False) as p:
            pred=p['prediction'];y=p['targets'];arms=list(p['arms']);views=list(p['views']);names=list(p['target_names']);fresh=p['fresh_alarm'];ids=p['sequence_id']
        with np.load(ev/'test/evaluation_only.npz',allow_pickle=False) as truth:
            np.testing.assert_array_equal(y,truth['targets']);np.testing.assert_array_equal(ids,truth['sequence_id'])
            dc=json.loads((root/'new_generator_config.json').read_text());proxy=dc['proxy']
            independent=proxy['intercept']+proxy['coefficient']*np.maximum(truth['clean_process'][...,0]-proxy['threshold'],0).sum(axis=1)+truth['label_noise'].sum(axis=1)
            np.testing.assert_allclose(y[:,0],independent,atol=1e-12,rtol=0)
        events=json.loads((ev/'test/events.json').read_text());ch=np.array([e['channel'] for e in events]);dur=np.array([e['end_exclusive']-e['start'] for e in events])
        for r in csv_rows(ev/'metrics.csv'):
            vi,ai,ti=views.index(r['view']),arms.index(r['arm']),names.index(r['target']);s=r['stratum']
            if s=='all':mask=np.ones(len(y),bool)
            elif s.startswith('channel:'):mask=ch==int(s.split(':')[1])
            elif s.startswith('duration:'):mask=dur==int(s.split(':')[1])
            elif s=='fresh_alarm':mask=fresh[vi]
            elif s=='no_fresh_alarm':mask=~fresh[vi]
            else:raise AssertionError(s)
            e=pred[vi,ai,mask,ti]-y[mask,ti]
            assert int(r['batches'])==mask.sum()
            assert abs(float(r['mae'])-np.mean(abs(e)))<1e-12
            assert abs(float(r['rmse'])-np.sqrt(np.mean(e*e)))<1e-12
            raw[seed,r['view'],r['arm'],r['target'],s]=float(np.mean(abs(e)));checks['metrics']+=1
        # Scores audited with independently implemented event ranges and alarm onset.
        # Recompute from allowed inputs using the frozen scorer, never refitting.
        from src.rq3_paired_data import apply_view
        from src.rq3_prediction_pilot import score_view,view_specs
        with np.load(ev/'test/inputs.npz',allow_pickle=False) as f:inputs={k:f[k] for k in f.files}
        frozen=json.loads((tr/'frozen_reference.json').read_text())
        data=Path('rq3_paired_20261019_rebuild_v1')
        if not data.exists():data=Path('data/processed/rq3_paired_20261019_rebuild_v1')
        scales=np.array(json.loads((data/f'seed_{seed}/fault_scales.json').read_text())['scales'])
        score_lookup={}
        for view,kw in view_specs(cfg):
            if not view.startswith(('bias','drift','missing')):continue
            v,mask=apply_view(inputs,events,scales,**kw);sc=score_view(v,dc['regimes'],frozen['scoring'])
            alarm=sc['alarm'];on=alarm & ~np.concatenate([np.zeros_like(alarm[:,:1]),alarm[:,:-1]],axis=1)
            for s in ['all','channel:0','channel:1','channel:2']:
                selected=np.ones(len(y),bool) if s=='all' else ch==int(s[-1]);counts=[];hits=[];avail=[];den=[]
                for i in np.flatnonzero(selected):
                    e=events[i];sl=(i,slice(e['start'],e['end_exclusive']),e['channel'])
                    counts.append(alarm[sl].sum());hits.append(on[sl].any());avail.append(sc['available'][sl].mean());den.append(e['end_exclusive']-e['start'])
                score_lookup[view,s]=(sum(counts)/sum(den),np.mean(hits),np.mean(avail))
        for r in csv_rows(ev/'score_metrics.csv'):
            np.testing.assert_allclose([float(r[k]) for k in ['point_recall','fresh_event_recall','availability']],score_lookup[r['view'],r['stratum']],atol=1e-12,rtol=0)
    paired=csv_rows(root/'paired_by_seed.csv')
    for r in paired:
        seed=int(r['seed']);a,b=r['comparison'].split('-');s=r['stratum'];target=r['target'];view=r['view']
        vv=views if view=='suite_equal_weight' else [view]
        value=np.mean([raw[seed,v,a,target,s]-raw[seed,v,b,target,s] for v in vv])
        assert abs(value-float(r['mae_difference']))<1e-12;checks['pairs']+=1
    for r in csv_rows(root/'paired_summary.csv'):
        selected=[p for p in paired if all(p[k]==r[k] for k in ['comparison','view','stratum','target'])]
        v=np.array([float(p['mae_difference']) for p in selected]);sd=v.std(ddof=1);h=t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v))
        np.testing.assert_allclose([float(r[k]) for k in ['mean','sd','ci_low','ci_high']],[v.mean(),sd,v.mean()-h,v.mean()+h],atol=1e-12,rtol=0)
        assert int(r['negative_seeds'])==(v<0).sum();checks['intervals']+=1
    checks['passed']=True
    return checks

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/rq3_prediction/rq3_shared_fault_20261023_v1');parser.add_argument('--output',required=True)
    args=parser.parse_args();result=verify(Path(args.root));Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
