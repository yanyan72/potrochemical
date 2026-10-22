"""Paired local measurement-noise study using frozen, archived RQ3 models."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .rq3_paired_data import INPUT_KEYS, generate_split, load_inputs
from .rq3_prediction_pilot import (features, predict, random_scores,
                                  score_view, write_csv, write_json)
from .rq3_lstm_pilot import PlainLSTM, check_files, infer, transform_sequence
from .rq3_final_test import ARCHITECTURES, GROUPS, COMPARISONS, read_rows, seed_interval, utc_now
from .rq3_release_audit import sha256

DEFAULT_DATA = Path('rq3_paired_20261019_rebuild_v1')
DEFAULT_RIDGE = Path('results/rq3_prediction/rq3_ridge_pilot_20261019_v1')
DEFAULT_LSTM = Path('results/rq3_prediction/rq3_lstm_pilot_20261020_v1')
DEFAULT_FINAL = Path('results/rq3_prediction/rq3_final_test_20261021_v1')


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """Load numerical archives without pickle."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def validate_config(cfg: dict[str, Any]) -> None:
    """Reject changes outside the registered sensitivity design."""
    if (cfg['noise_factors'] != [0.5, 1., 2.] or cfg['refit'] is not False
            or cfg['source_split'] != 'val' or cfg['inject_faults'] is not False
            or tuple(cfg['groups']) != GROUPS or tuple(cfg['architectures']) != ARCHITECTURES
            or cfg['targets'] != ['arrival_proxy', 'null_proxy']
            or type(cfg['batches_per_seed']) is not int or cfg['batches_per_seed'] < 2
            or len(set(cfg['seeds'])) != len(cfg['seeds']) or not cfg['seeds']
            or type(cfg['namespace']) is not int or cfg['namespace'] < 0):
        raise ValueError('unsupported sensitivity design')


def source_check(data: Path, ridge: Path, lstm: Path, final: Path,
                 cfg: dict[str, Any]) -> dict[str, str]:
    """Check fixed input bytes; the old final test is hashed, never predicted."""
    validate_config(cfg)
    for folder, key in [(data,'data_manifest_sha256'), (ridge,'ridge_manifest_sha256'),
                        (lstm,'lstm_manifest_sha256'), (final,'source_final_manifest_sha256')]:
        if sha256(folder/'manifest.json') != cfg[key]:
            raise ValueError('source manifest mismatch: '+key)
        check_files(folder, json.loads((folder/'manifest.json').read_text()))
    dc = yaml.safe_load((data/'config.yaml').read_text())
    if dc['namespace'] == cfg['namespace'] or not set(cfg['seeds']) <= set(dc['seeds']):
        raise ValueError('new namespace and existing model seed mapping required')
    sources = json.loads((final/'identity.json').read_text())['source_sha256']
    check_files(Path(__file__).resolve().parents[1], sources)
    return {**sources, 'src/rq3_noise_sensitivity.py':sha256(Path(__file__))}


def generate_paired(dc: dict[str, Any], cfg: dict[str, Any], seed: int
                    ) -> tuple[list[dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    """Generate three matched observations with unchanged process and labels."""
    validate_config(cfg)
    if seed not in cfg['seeds'] or cfg['namespace'] == dc['namespace']:
        raise ValueError('seed/namespace mismatch')
    c = copy.deepcopy(dc)
    c.update(dataset_id=cfg['run_id'], namespace=cfg['namespace'], seeds=cfg['seeds'])
    c['counts']['val'] = cfg['batches_per_seed']
    samples = []
    for factor in cfg['noise_factors']:
        level = copy.deepcopy(c)
        level['measurement_noise_std'] = (np.asarray(dc['measurement_noise_std']) * factor).tolist()
        samples.append(generate_split(level, seed, 'val'))
    baseline = samples[1]
    clean = baseline['truth']['clean_process'][:, :baseline['inputs']['observations'].shape[1]]
    noise = baseline['inputs']['observations'] - clean
    maximum = 0.
    for factor, sample in zip(cfg['noise_factors'], samples):
        for key in baseline['truth']:
            if not np.array_equal(sample['truth'][key], baseline['truth'][key]):
                raise ValueError('latent pairing mismatch')
        if not np.array_equal(sample['targets'], baseline['targets']) or sample['events'] != baseline['events']:
            raise ValueError('target/event pairing mismatch')
        for key in INPUT_KEYS - {'observations'}:
            if not np.array_equal(sample['inputs'][key], baseline['inputs'][key]):
                raise ValueError('input identity pairing mismatch')
        maximum = max(maximum, float(np.max(np.abs(sample['inputs']['observations'] - clean - factor*noise))))
    if maximum > 1e-12:
        raise ValueError('noise scaling mismatch')
    audit = dict(status='pass', max_noise_scaling_residual=maximum,
                 latent_targets_ids_equal=True, unique_batches=cfg['batches_per_seed'],
                 paired_observation_versions=3*cfg['batches_per_seed'], injected_faults=0)
    return [s['inputs'] for s in samples], baseline, audit


def predict_views(views: list[dict[str, np.ndarray]], data: Path, ridge: Path,
                  lstm: Path, cfg: dict[str, Any], seed: int) -> np.ndarray:
    """Infer from visible inputs only; labels, events and latent values are rejected."""
    if not views:
        raise ValueError('empty views')
    for view in views:
        if set(view) != INPUT_KEYS or view['observations'].shape[1:] != (118,3):
            raise ValueError('visible input schema mismatch')
        if (np.any(view['available_time'] > 119)
                or not np.array_equal(view['time_step'], np.arange(118))
                or not np.array_equal(view['sequence_id'], views[0]['sequence_id'])):
            raise ValueError('future time or mismatched batch IDs')
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    dc = yaml.safe_load((data/'config.yaml').read_text())
    rp = json.loads((ridge/f'seed_{seed}/preprocessing_and_score.json').read_text())
    lp = json.loads((lstm/f'seed_{seed}/parameters.json').read_text())
    lc = json.loads((lstm/'config.json').read_text())
    if lp['base'] != rp:
        raise ValueError('archived preprocessing mismatch')
    bank = score_view(load_inputs(data,seed,'train'), dc['regimes'],rp['scoring'])['reliability']
    random = random_scores(views[0]['sequence_id'],bank,seed,cfg['random_control_seed'])
    scores = [score_view(v,dc['regimes'],rp['scoring']) for v in views]
    pred = np.empty((2,len(views),len(GROUPS),len(views[0]['sequence_id']),2))
    rstate = read_npz(ridge/f'seed_{seed}/ridge_models.npz')
    lstate = read_npz(lstm/f'seed_{seed}/models.npz')
    pred[:,:,0] = rstate['mean']
    for gi, group in enumerate(GROUPS[1:],1):
        rm = {k:rstate[f'{group}__{k}'] for k in ('center','scale','weights','intercept')}
        models = []
        for target in cfg['targets']:
            key=group+'__'+target
            detail=lp['models'][key]
            model=PlainLSTM(detail['input_channels'],lc['hidden_size'])
            model.load_state_dict({k:torch.from_numpy(lstate[key+'__'+k].copy()) for k in model.state_dict()})
            models.append((model,detail))
        for vi,(view,score) in enumerate(zip(views,scores)):
            flat=features(view,score,rp['preprocessing'],group,random)
            pred[0,vi,gi]=predict(flat,rm)
            x=transform_sequence(flat.reshape(len(flat),118,-1),lp['groups'][group])
            for ti,(model,detail) in enumerate(models):
                pred[1,vi,gi,:,ti]=infer(model,x,detail['target_center'],detail['target_scale'],lc['batch_size'])
    if not np.isfinite(pred).all():
        raise ValueError('nonfinite predictions')
    return pred


def replay(data: Path, ridge: Path, lstm: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    """Confirm identical archived original-val predictions on all model groups."""
    if output.exists():
        raise FileExistsError(output)
    checks=[]
    for seed in cfg['seeds']:
        view=load_inputs(data,seed,'val')
        p=predict_views([view],data,ridge,lstm,cfg,seed)
        for ai,folder in enumerate((ridge,lstm)):
            old=read_npz(folder/f'seed_{seed}/validation_predictions.npz')
            idx=[list(old['groups']).index(g) for g in GROUPS]
            expected=old['prediction'][0,idx]
            delta=float(np.abs(p[ai,0]-expected).max())
            if not np.array_equal(old['sequence_id'],view['sequence_id']) or delta>1e-6:
                raise ValueError('val replay mismatch')
            checks.append(dict(seed=seed,architecture=ARCHITECTURES[ai],max_abs_difference=delta))
    result=dict(status='pass',checks=checks,refitting=False,old_test_predictions=False)
    write_json(output,result)
    return result


def evaluate(pred: np.ndarray, y: np.ndarray, regimes: np.ndarray, cfg: dict[str, Any],
             seed: int) -> tuple[list[dict],list[dict]]:
    """Calculate seed-level errors and paired changes relative to 1x noise."""
    if pred.shape != (2,3,7,len(y),2) or y.shape != (len(y),2):
        raise ValueError('prediction/target axes mismatch')
    if not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite evaluation')
    rows,paired=[],[]
    masks={'all':np.ones(len(y),bool), 'transit_smooth':regimes[:,-1]==1, 'transit_rough':regimes[:,-1]==2}
    err=pred-y[None,None,None]
    for ai,arch in enumerate(ARCHITECTURES):
        for stratum,mask in masks.items():
            if not mask.any():
                continue
            ae=np.abs(err[ai][:,:,mask,:]).mean(axis=2)
            rmse=np.sqrt((err[ai][:,:,mask,:]**2).mean(axis=2))
            for fi,factor in enumerate(cfg['noise_factors']):
                for gi,group in enumerate(GROUPS):
                    for ti,target in enumerate(cfg['targets']):
                        rows.append(dict(seed=seed,architecture=arch,noise_factor=factor,group=group,
                            target=target,stratum=stratum,batches=int(mask.sum()),mae=float(ae[fi,gi,ti]),
                            rmse=float(rmse[fi,gi,ti]),mae_change_vs_1x=float(ae[fi,gi,ti]-ae[1,gi,ti])))
                for better,base in COMPARISONS:
                    bi,ci=GROUPS.index(better),GROUPS.index(base)
                    for ti,target in enumerate(cfg['targets']):
                        delta=ae[fi,bi,ti]-ae[fi,ci,ti]
                        interaction=delta-(ae[1,bi,ti]-ae[1,ci,ti])
                        paired.append(dict(seed=seed,architecture=arch,noise_factor=factor,
                            comparison=better+'-'+base,target=target,stratum=stratum,
                            mae_difference=float(delta),interaction_vs_1x=float(interaction)))
    return rows,paired


def summarize_rows(rows: list[dict], keys: list[str], values: list[str], cfg: dict[str,Any]) -> list[dict]:
    """Require one value per seed and report descriptive seed-level intervals."""
    bins:dict[tuple,list[dict]]={}
    for row in rows:
        bins.setdefault(tuple(row[k] for k in keys),[]).append(row)
    out=[]
    for key,group in sorted(bins.items()):
        if sorted(int(r['seed']) for r in group)!=sorted(cfg['seeds']):
            raise ValueError('missing or duplicated seed')
        row=dict(zip(keys,key));row['data_repeats']=len(group)
        for value in values:
            stat=seed_interval([float(r[value]) for r in group],cfg['confidence_level'])
            row.update({value+'_mean':stat['mae_difference_mean'],value+'_sd':stat['mae_difference_sd'],
                        value+'_ci_low':stat['ci_low'],value+'_ci_high':stat['ci_high']})
        row['interval_use']='descriptive_only'
        out.append(row)
    return out


def seal(folder: Path) -> None:
    """Hash every artifact in a completed folder, excluding its own manifest."""
    path=folder/'manifest.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path,{str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()})


def initialize(data: Path,ridge: Path,lstm: Path,final: Path,cfg:dict[str,Any],output:Path,
               protocol_commit:str,implementation_commit:str) -> None:
    """Create immutable run identity or verify an interrupted matching run."""
    if any(not re.fullmatch('[0-9a-f]{40}',x) for x in [protocol_commit,implementation_commit]):
        raise ValueError('published commits required')
    sources=source_check(data,ridge,lstm,final,cfg)
    if output.exists():
        if (output/'manifest.json').exists():
            raise FileExistsError(output)
        old=json.loads((output/'identity.json').read_text())
        if (json.loads((output/'config.json').read_text())!=cfg or old['source_sha256']!=sources
                or old['protocol_commit']!=protocol_commit or old['implementation_commit']!=implementation_commit):
            raise ValueError('interrupted run identity mismatch')
        return
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',cfg)
    write_json(output/'identity.json',dict(protocol_commit=protocol_commit,implementation_commit=implementation_commit,
        source_sha256=sources,python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        started_utc=utc_now(),data_source='new_synthetic_sensitivity',refitting=False,old_test_predictions=False))
    write_json(output/'generator_base_config.json',yaml.safe_load((data/'config.yaml').read_text()))


def run_seed(data:Path,ridge:Path,lstm:Path,final:Path,cfg:dict[str,Any],output:Path,
             protocol_commit:str,implementation_commit:str,seed:int) -> dict[str,Any]:
    """Persist newly generated inputs before inference, then evaluate and seal."""
    initialize(data,ridge,lstm,final,cfg,output,protocol_commit,implementation_commit)
    folder=output/f'seed_{seed}'
    folder.mkdir(exist_ok=False)
    dc=yaml.safe_load((data/'config.yaml').read_text())
    views,baseline,audit=generate_paired(dc,cfg,seed)
    for fi,view in enumerate(views):
        np.savez_compressed(folder/f'inputs_level_{fi}.npz',**view)
    np.savez_compressed(folder/'evaluation_only.npz',**baseline['truth'],targets=baseline['targets'])
    # Predictor receives only allowed inputs; labels/truth are never arguments.
    pred=predict_views(views,data,ridge,lstm,cfg,seed)
    np.savez_compressed(folder/'predictions.npz',prediction=pred,sequence_id=views[0]['sequence_id'],
        architectures=np.array(ARCHITECTURES),groups=np.array(GROUPS),noise_factors=cfg['noise_factors'],targets=np.array(cfg['targets']))
    saved=read_npz(folder/'evaluation_only.npz')
    if not np.array_equal(saved['sequence_id'],views[0]['sequence_id']):
        raise ValueError('label ID mismatch')
    rows,paired=evaluate(pred,saved['targets'],views[0]['recorded_regime'],cfg,seed)
    write_csv(folder/'metrics_by_seed.csv',rows)
    write_csv(folder/'paired_by_seed.csv',paired)
    source_check(data,ridge,lstm,final,cfg)
    result=dict(seed=seed,completed_utc=utc_now(),audit=audit,models_refitted=0,
        source_files_unchanged=True,predictions_saved_before_evaluation=True,new_synthetic_data=True)
    write_json(folder/'metadata.json',result)
    seal(folder)
    return result


def summarize(output:Path,cfg:dict[str,Any]) -> dict[str,Any]:
    """Aggregate the completed seed artifacts once without selecting results."""
    if (output/'manifest.json').exists() or (output/'metrics_summary.csv').exists():
        raise FileExistsError(output)
    rows,paired=[],[]
    for seed in cfg['seeds']:
        folder=output/f'seed_{seed}'
        check_files(folder,json.loads((folder/'manifest.json').read_text()))
        rows+=read_rows(folder/'metrics_by_seed.csv');paired+=read_rows(folder/'paired_by_seed.csv')
    write_csv(output/'metrics_by_seed.csv',rows);write_csv(output/'paired_by_seed.csv',paired)
    keys=['architecture','noise_factor','group','target','stratum']
    write_csv(output/'metrics_summary.csv',summarize_rows(rows,keys,['mae','rmse','mae_change_vs_1x'],cfg))
    keys=['architecture','noise_factor','comparison','target','stratum']
    write_csv(output/'paired_summary.csv',summarize_rows(paired,keys,['mae_difference','interaction_vs_1x'],cfg))
    meta=dict(completed_utc=utc_now(),run_id=cfg['run_id'],scope=cfg['scope'],
        unique_batches=len(cfg['seeds'])*cfg['batches_per_seed'],paired_versions=len(cfg['seeds'])*cfg['batches_per_seed']*3,
        independent_model_data_pairs=len(cfg['seeds']),new_synthetic_data=True,refitting=False,
        prior_test_status='used_unchanged',old_test_predictions=False,interval_use='descriptive_only')
    write_json(output/'metadata.json',meta);seal(output)
    return meta


def main() -> None:
    """Run registered generation/inference, validation replay or aggregation."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    for name,default in [('data',DEFAULT_DATA),('ridge',DEFAULT_RIDGE),('lstm',DEFAULT_LSTM),('final',DEFAULT_FINAL)]:
        parser.add_argument('--'+name,type=Path,default=default)
    parser.add_argument('--protocol-commit',default='');parser.add_argument('--implementation-commit',default='')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--seed',type=int);mode.add_argument('--replay',action='store_true');mode.add_argument('--summarize',action='store_true')
    a=parser.parse_args();cfg=yaml.safe_load(a.config.read_text());validate_config(cfg)
    source_check(a.data,a.ridge,a.lstm,a.final,cfg)
    if a.replay:
        result=replay(a.data,a.ridge,a.lstm,cfg,a.output)
    elif a.summarize:
        result=summarize(a.output,cfg)
    else:
        result=run_seed(a.data,a.ridge,a.lstm,a.final,cfg,a.output,a.protocol_commit,a.implementation_commit,a.seed)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
