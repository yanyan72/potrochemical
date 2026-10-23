"""Independent test namespace correction, frozen before viewing research test metrics.

The original implementation and completed training are immutable. A smoke test
used four test prefixes in namespace20261023. This evaluator uses2026102301 for
all five seeds, preserving generation law, model selection and evaluation math.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from .rq3_shared_fault import (initialize,lock_training,generate_split,save_sample,context,
    read_npz,make_bundle,condition_setup,random_trajectory,arm_features,centers,
    PlainLSTM,infer,apply_view,view_specs,event_vectors,metric_rows,write_csv,write_json,
    array_hash,arm_names,seal,sha256)

def evaluate_fresh_seed(data: Path, ridge: Path, cfg: dict, out: Path, commit: str, seed: int) -> dict:
    """Evaluate newly generated test once, only after all training is locked."""
    dc = initialize(data,ridge,cfg,out,commit); lock_training(out,cfg)
    if seed not in cfg['seeds']: raise ValueError(seed)
    sd = out/f'seed_{seed}'/'evaluation'; sd.mkdir(exist_ok=False)
    write_json(sd/'test_used.json', dict(seed=seed, used=True, reuse_for_selection=False))
    test_dc = dict(dc, namespace=2026102301, dataset_id=cfg['run_id']+'_fresh_test')
    sample = generate_split(test_dc,seed,'test'); save_sample(sd/'test',sample)
    frozen,train,yt,scales,tb = context(data,ridge,cfg,dc,seed)
    tr = out/f'seed_{seed}'/'training'
    vid = read_npz(tr/'validation'/'inputs.npz')['sequence_id']
    if set(sample['inputs']['sequence_id']) & (set(vid)|set(train['sequence_id'])): raise ValueError('test overlap')
    eb = make_bundle(sample['inputs'],sample['events'],scales,frozen,dc,cfg)
    pars = json.loads((tr/'preprocessing.json').read_text()); models = json.loads((tr/'model_parameters.json').read_text())
    pred, feature_hashes = [], []
    for condition in ('clean','shared'):
        _,bank = condition_setup(tb,train,cfg,condition)
        random,ri = random_trajectory(sample['inputs']['sequence_id'],train['sequence_id'],bank,seed,cfg['random_control_seed'])
        np.savez_compressed(sd/f'{condition}_random_indices.npz',test=ri)
        common = None
        for group in cfg[condition+'_groups']:
            arm = condition+'_'+group
            x = arm_features(eb,group,pars[condition]['scaler'],random,centers(frozen,dc))
            if group in ('N','T','R','O'):
                h = array_hash(x[...,:12])
                if common is not None and h!=common: raise ValueError('test common input mismatch')
                common=h;feature_hashes.append(dict(arm=arm,common_sha256=h))
            pp=[]
            for ti,target in enumerate(cfg['targets']):
                key=arm+'__'+target;p=models[key]
                model=PlainLSTM(15,cfg['hidden_size'])
                model.load_state_dict({k:torch.from_numpy(v) for k,v in read_npz(tr/(key+'.npz')).items()})
                pp.append(infer(model,x.reshape(-1,*x.shape[2:]),p['target_center'],p['target_scale'],cfg['batch_size']).reshape(len(x),len(sample['targets'])))
            pred.append(np.stack(pp,axis=-1))
    pred.append(np.broadcast_to(yt.mean(axis=0),(len(eb['views']),len(sample['targets']),2)).copy())
    prediction=np.stack(pred,axis=1);arms=arm_names(cfg)+['mean']
    fresh=np.zeros((len(eb['views']),len(sample['targets'])),bool);score_rows=[]
    for vi,(view,kw) in enumerate(view_specs(cfg)):
        s=eb['scores'][vi]
        if view.startswith(('bias','drift','missing')):
            _,mask=apply_view(sample['inputs'],sample['events'],scales,**kw)
            ev=event_vectors(eb['scores'][0],s,sample['events'],mask);fresh[vi]=ev['fresh_alarm']
            ch=np.asarray([e['channel'] for e in sample['events']])
            for label,selected in [('all',np.ones(len(ch),bool))]+[(f'channel:{c}',ch==c) for c in range(3)]:
                if not selected.any(): continue
                score_rows.append(dict(seed=seed,view=view,stratum=label,point_recall=float(ev['event_alarm_count'][selected].sum()/ev['event_cells'][selected].sum()),fresh_event_recall=float(ev['fresh_alarm'][selected].mean()),availability=float(ev['availability'][selected].mean())))
    np.savez_compressed(sd/'predictions.npz',prediction=prediction,targets=sample['targets'],sequence_id=sample['inputs']['sequence_id'],arms=arms,views=[v for v,_ in view_specs(cfg)],target_names=cfg['targets'],fresh_alarm=fresh)
    write_csv(sd/'metrics.csv',metric_rows(prediction,sample['targets'],sample['events'],seed,cfg,arms,fresh))
    write_csv(sd/'score_metrics.csv',score_rows);write_json(sd/'fairness.json',feature_hashes)
    meta=dict(seed=seed,batches=len(sample['targets']),models=len(models),test_used=True,namespace=2026102301)
    write_json(sd/'metadata.json',meta);seal(sd)
    return meta


def main() -> None:
    """Bind the amended evaluator, lock all trained models, then evaluate one seed."""
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--commit',required=True,help='Original training implementation commit')
    p.add_argument('--evaluation-commit',required=True)
    p.add_argument('--data',default='rq3_paired_20261019_rebuild_v1')
    p.add_argument('--ridge',default='results/rq3_prediction/rq3_ridge_pilot_20261019_v1')
    p.add_argument('--output',default='results/rq3_prediction/rq3_shared_fault_20261023_v1')
    p.add_argument('--config',default='configs/rq3_shared_fault_20261023.yaml')
    args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());out=Path(args.output)
    dc=initialize(Path(args.data),Path(args.ridge),cfg,out,args.commit);locks=lock_training(out,cfg)
    identity=dict(evaluation_commit=args.evaluation_commit,evaluator_sha256=sha256(Path(__file__)),namespace=2026102301,
                  original_training_commit=args.commit,training_manifests=locks,
                  reason='isolate formal test from four automatic smoke-test samples; no research test metrics inspected before amendment')
    path=out/'evaluation_identity.json'
    if path.exists():
        if json.loads(path.read_text())!=identity:raise ValueError('evaluation identity changed')
    else:
        write_json(path,identity)
        write_json(out/'evaluation_generator_config.json',dict(dc,namespace=2026102301,dataset_id=cfg['run_id']+'_fresh_test'))
    print(json.dumps(evaluate_fresh_seed(Path(args.data),Path(args.ridge),cfg,out,args.commit,args.seed)),flush=True)

if __name__=='__main__':main()
