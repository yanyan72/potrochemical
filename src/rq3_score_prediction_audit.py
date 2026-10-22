"""Separate frozen-score consistency diagnostics from saved predictive benefit."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from sklearn.metrics import average_precision_score
from .rq3_paired_data import load_inputs, apply_view
from .rq3_prediction_pilot import score_view, view_specs, aggregate, write_csv, write_json
from .rq3_lstm_pilot import check_files
from .rq3_release_audit import sha256


def npz(path:Path) -> dict[str,np.ndarray]:
    """Load pickle-free arrays."""
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def read_csv(path:Path) -> list[dict[str,str]]:
    """Read stored ID-aligned records."""
    with path.open(newline='') as f:return list(csv.DictReader(f))


def event_vectors(original:dict[str,np.ndarray],observed:dict[str,np.ndarray],events:list[dict],
                  mask:np.ndarray) -> dict[str,np.ndarray]:
    """Compute matched event statistics, distinguishing new from continued alarms."""
    alarm=observed['alarm'];old=original['alarm']
    fresh=alarm & ~np.concatenate([np.zeros_like(alarm[:,:1]),alarm[:,:-1]],axis=1)
    oldfresh=old & ~np.concatenate([np.zeros_like(old[:,:1]),old[:,:-1]],axis=1)
    out={k:[] for k in ['score_drop','availability','any_alarm','fresh_alarm','original_fresh_alarm',
        'event_alarm_count','event_cells','bystander_alarm_count','original_bystander_alarm_count','bystander_cells']}
    for i,e in enumerate(events):
        ch,start,end=[int(e[k]) for k in ['channel','start','end_exclusive']]
        active=observed['available'][i,start:end,ch]
        cells=np.zeros_like(mask[i]);cells[start:end,ch]=True
        if not np.array_equal(cells,mask[i]):raise ValueError('event mask mismatch')
        clean=[j for j in range(3) if j!=ch]
        drop=(original['reliability'][i,start:end,ch][active]-observed['reliability'][i,start:end,ch][active]).mean() if active.any() else np.nan
        vals=[drop,float(active.mean()),bool(alarm[i,start:end,ch].any()),bool(fresh[i,start:end,ch].any()),
              bool(oldfresh[i,start:end,ch].any()),int(alarm[i,start:end,ch].sum()),end-start,
              int(alarm[i,start:end][:,clean].sum()),int(old[i,start:end][:,clean].sum()),2*(end-start)]
        for k,v in zip(out,vals):out[k].append(v)
    return {k:np.asarray(v) for k,v in out.items()}


def prediction_rows(errors:np.ndarray,views:list[str],channels:np.ndarray,fresh:np.ndarray,
                    seed:int,targets:list[str]) -> list[dict]:
    """Describe saved B3-B2 errors on matched batches; no model is refitted."""
    # errors: architecture, view, (B2,B3), batch, target.
    delta=errors[:,:,1]-errors[:,:,0];n=len(channels);out=[]
    for vi,view in enumerate(views):
        masks={'all':np.ones(n,bool)}
        if view.startswith(('bias','drift','missing')):
            masks.update({f'channel:{j}':channels==j for j in range(3)})
        if view.startswith(('bias','drift')):
            masks.update({'fresh_alarm':fresh[vi],'no_fresh_alarm':~fresh[vi]})
        for name,mask in masks.items():
            if not mask.any():continue
            for ai,arch in enumerate(['ridge','lstm']):
                for ti,target in enumerate(targets):
                    d=delta[ai,vi,mask,ti]
                    out.append(dict(seed=seed,architecture=arch,view=view,stratum=name,target=target,
                        batches=int(mask.sum()),b2_mae=float(errors[ai,vi,0,mask,ti].mean()),
                        b3_mae=float(errors[ai,vi,1,mask,ti].mean()),mae_difference=float(d.mean()),
                        improvement_fraction=float((d<0).mean()),
                        b2_degradation=float((errors[ai,vi,0,mask,ti]-errors[ai,0,0,mask,ti]).mean()),
                        b3_degradation=float((errors[ai,vi,1,mask,ti]-errors[ai,0,1,mask,ti]).mean()),
                        incremental_degradation=float((d-delta[ai,0,mask,ti]).mean())))
    return out


def run(data:Path,ridge:Path,lstm:Path,cfg:dict[str,Any],output:Path,commit:str) -> dict[str,Any]:
    """Read existing val inputs and predictions, saving a new immutable audit."""
    if cfg['evaluation_split']!='val' or cfg['no_refitting'] is not True:
        raise ValueError('audit is development only')
    if len(commit)!=40:raise ValueError('published implementation commit required')
    for folder,pin in [(data,'data_manifest_sha256'),(ridge,'ridge_manifest_sha256'),(lstm,'lstm_manifest_sha256')]:
        if sha256(folder/'manifest.json')!=cfg[pin]:raise ValueError('source manifest mismatch')
        check_files(folder,json.loads((folder/'manifest.json').read_text()))
    sources=json.loads((lstm/'run_identity.json').read_text())['source_sha256']
    check_files(Path(__file__).resolve().parents[1],sources)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',cfg)
    write_json(output/'identity.json',dict(implementation_commit=commit,source_sha256={**sources,
        'src/rq3_score_prediction_audit.py':sha256(Path(__file__))},evaluation='posthoc_validation',refitting=False))
    dc=yaml.safe_load((data/'config.yaml').read_text());specs=view_specs(cfg)
    names=[name for name,_ in specs];score_rows=[];predict_rows=[];coverage=[]
    for seed in cfg['seeds']:
        folder=output/f'seed_{seed}';folder.mkdir()
        inputs=load_inputs(data,seed,'val')
        events=read_csv(data/f'seed_{seed}/val/events.csv')
        if [e['sequence_id'] for e in events]!=list(inputs['sequence_id']):raise ValueError('event IDs')
        target_rows=read_csv(data/f'seed_{seed}/val/targets.csv')
        if [e['sequence_id'] for e in target_rows]!=list(inputs['sequence_id']):raise ValueError('target IDs')
        y=np.array([[float(r[t]) for t in cfg['targets']] for r in target_rows])
        rp=json.loads((ridge/f'seed_{seed}/preprocessing_and_score.json').read_text())
        scales=np.array(json.loads((data/f'seed_{seed}/fault_scales.json').read_text())['scales'])
        original=score_view(inputs,dc['regimes'],rp['scoring'])
        sr=[];cv=[];drops=[];fresh=[];availability=[]
        for name,kw in specs:
            view,mask=apply_view(inputs,events,scales,**kw)
            scored=score_view(view,dc['regimes'],rp['scoring'])
            cv.append(dict(seed=seed,view=name,score_coverage=float(scored['available'].mean()),
                all_cell_alarm_rate=float(scored['alarm'].mean()),numerical_fault=bool(name.startswith(('bias','drift')))))
            if name.startswith(('bias','drift','missing')):
                v=event_vectors(original,scored,events,mask)
                drops.append(v['score_drop']);fresh.append(v['fresh_alarm']);availability.append(v['availability'])
                if name=='missing':
                    # Missingness is unscoreable, not a numerical anomaly detection success.
                    continue
                for stratum,selected in [('all',np.ones(len(events),bool))]+[(f'channel:{j}',np.array([int(e['channel'])==j for e in events])) for j in range(3)]:
                    if not selected.any():continue
                    valid=scored['available'][selected]
                    if not valid.all():raise ValueError('numerical faults unexpectedly unscoreable')
                    truth=mask[selected].ravel();anom=(1-scored['reliability'][selected]).ravel()
                    sr.append(dict(seed=seed,view=name,stratum=stratum,batches=int(selected.sum()),
                        prevalence=float(truth.mean()),point_ap=float(average_precision_score(truth,anom)),
                        point_recall=float(v['event_alarm_count'][selected].sum()/v['event_cells'][selected].sum()),
                        event_any_alarm=float(v['any_alarm'][selected].mean()),event_fresh_alarm=float(v['fresh_alarm'][selected].mean()),
                        original_fresh_alarm=float(v['original_fresh_alarm'][selected].mean()),
                        bystander_fpr=float(v['bystander_alarm_count'][selected].sum()/v['bystander_cells'][selected].sum()),
                        original_bystander_fpr=float(v['original_bystander_alarm_count'][selected].sum()/v['bystander_cells'][selected].sum()),
                        mean_score_drop=float(v['score_drop'][selected].mean())))
            else:
                drops.append(np.full(len(y),np.nan));fresh.append(np.zeros(len(y),bool));availability.append(np.ones(len(y)))
        pred=[]
        for folder_model in (ridge,lstm):
            a=npz(folder_model/f'seed_{seed}/validation_predictions.npz')
            if not np.array_equal(a['sequence_id'],inputs['sequence_id']) or not np.array_equal(a['targets'],y) or list(a['views'])!=names:
                raise ValueError('prediction/target/view pairing mismatch')
            idx=[list(a['groups']).index(g) for g in ['B2','B3']]
            pred.append(a['prediction'][:,idx])
        errors=np.abs(np.stack(pred)-y[None,None,None])
        channels=np.array([int(e['channel']) for e in events]);f=np.array(fresh)
        pr=prediction_rows(errors,names,channels,f,seed,cfg['targets'])
        np.savez_compressed(folder/'paired_batches.npz',sequence_id=inputs['sequence_id'],views=np.array(names),
            channels=channels,targets=y,absolute_errors=errors,score_drop=np.array(drops),fresh_alarm=f,
            event_score_availability=np.array(availability),target_names=np.array(cfg['targets']))
        write_csv(folder/'score_metrics.csv',sr);write_csv(folder/'prediction_pairs.csv',pr);write_csv(folder/'coverage.csv',cv)
        write_json(folder/'manifest.json',{p.name:sha256(p) for p in folder.iterdir() if p.is_file()})
        score_rows+=sr;predict_rows+=pr;coverage+=cv
    write_csv(output/'score_metrics_by_seed.csv',score_rows);write_csv(output/'prediction_pairs_by_seed.csv',predict_rows)
    score_values=['prevalence','point_ap','point_recall','event_any_alarm','event_fresh_alarm','original_fresh_alarm','bystander_fpr','original_bystander_fpr','mean_score_drop']
    prediction_values=['b2_mae','b3_mae','mae_difference','improvement_fraction','b2_degradation','b3_degradation','incremental_degradation']
    write_csv(output/'score_summary.csv',aggregate(score_rows,['view','stratum'],score_values))
    write_csv(output/'prediction_summary.csv',aggregate(predict_rows,['architecture','view','stratum','target'],prediction_values))
    write_csv(output/'coverage_by_seed.csv',coverage)
    write_csv(output/'coverage_summary.csv',aggregate(coverage,['view'],['score_coverage','all_cell_alarm_rate']))
    for path in [data,ridge,lstm]:check_files(path,json.loads((path/'manifest.json').read_text()))
    meta=dict(unique_development_batches=sum(len(npz(output/f'seed_{s}/paired_batches.npz')['sequence_id']) for s in cfg['seeds']),
        score_metric_rows=len(score_rows),prediction_pair_rows=len(predict_rows),source_files_unchanged=True,
        new_training=False,new_test_predictions=False,analysis_status='posthoc_development_only')
    write_json(output/'metadata.json',meta)
    write_json(output/'manifest.json',{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--implementation-commit',required=True)
    p.add_argument('--data',type=Path,default=Path('rq3_paired_20261019_rebuild_v1'))
    p.add_argument('--ridge',type=Path,default=Path('results/rq3_prediction/rq3_ridge_pilot_20261019_v1'))
    p.add_argument('--lstm',type=Path,default=Path('results/rq3_prediction/rq3_lstm_pilot_20261020_v1'))
    a=p.parse_args();print(json.dumps(run(a.data,a.ridge,a.lstm,yaml.safe_load(a.config.read_text()),a.output,a.implementation_commit)))
