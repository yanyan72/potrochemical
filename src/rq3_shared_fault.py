"""Fixed-budget shared-fault RQ3 controls; oracle arms are diagnostic only."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml
from scipy.stats import t as student_t
from .rq3_paired_data import generate_split, apply_view, load_inputs, INPUT_KEYS
from .rq3_prediction_pilot import (features, score_view, targets, view_specs,
                                  write_json, write_csv, SOURCE_FILES)
from .rq3_lstm_pilot import PlainLSTM, fit_sequence_scaler, transform_sequence, infer, check_files
from .rq3_release_audit import sha256
from .rq3_score_prediction_audit import event_vectors


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """Read arrays without pickle."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def array_hash(a: np.ndarray) -> str:
    """Hash shape, type and contiguous array bytes for equality evidence."""
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def seal(path: Path) -> dict[str, str]:
    """Seal a completed stage; refuse to replace an existing manifest."""
    if (path / 'manifest.json').exists():
        raise FileExistsError(path)
    m = {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob('*')) if p.is_file()}
    write_json(path / 'manifest.json', m)
    return m


def arm_names(cfg: dict) -> list[str]:
    """Return the ten registered learned arms in fixed order."""
    return [f'{c}_{g}' for c in ('clean', 'shared') for g in cfg[c + '_groups']]


def train_events(ids: np.ndarray, length: int, dc: dict, seed: int) -> list[dict]:
    """Apply the frozen event distribution in a new RNG namespace to old train."""
    out = []
    for i, sid in enumerate(ids):
        rng = np.random.default_rng(np.random.SeedSequence([dc['namespace'], seed, 2, i, 7]))
        d = int(rng.choice(dc['faults']['durations']))
        start = int(rng.integers(0, length - d + 1))
        out.append(dict(sequence_id=str(sid), channel=int(rng.integers(3)), start=start,
                        end_exclusive=start+d, sign=int(rng.choice([-1, 1]))))
    return out


def make_bundle(inputs: dict, events: list[dict], scales: np.ndarray,
                frozen: dict, dc: dict, cfg: dict) -> dict:
    """Build shared views/scores; ideal masks remain separate from deployable X."""
    if set(inputs) != INPUT_KEYS:
        raise ValueError('prediction inputs must be allowlisted')
    views, scores, bases, oracle = [], [], [], []
    for _, kw in view_specs(cfg):
        v, mask = apply_view(inputs, events, scales, **kw)
        s = score_view(v, dc['regimes'], frozen['scoring'])
        b = features(v, s, frozen['preprocessing'], 'B2').reshape(*v['observations'].shape[:2], 12)
        wrong = v['recorded_regime'] != inputs['recorded_regime']
        ideal = ~(mask | wrong[..., None])
        views.append(v); scores.append(s); bases.append(b); oracle.append(ideal.astype(float))
    return dict(views=views, scores=scores, base=np.stack(bases), oracle=np.stack(oracle), events=events)


def random_trajectory(ids: np.ndarray, train_ids: np.ndarray, bank: np.ndarray,
                      seed: int, control: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw pooled train trajectories, excluding all views of the receiving train ID."""
    # bank has view,batch,time,channel axes. No targets enter the draw.
    nv, n = bank.shape[:2]
    lookup = {str(s): i for i, s in enumerate(train_ids)}
    ix = []
    for sid in ids:
        v = int.from_bytes(hashlib.sha256(f'{control}:{seed}:{sid}'.encode()).digest()[:8], 'big')
        own = lookup.get(str(sid))
        if own is None:
            j = v % (nv*n)
            ix.append((j//n, j % n))
        else:
            j = v % (nv*(n-1)); b = j % (n-1)
            ix.append((j//(n-1), b + (b >= own)))
    index = np.asarray(ix, dtype=np.int64)
    return bank[index[:, 0], index[:, 1]].copy(), index


def centers(frozen: dict, dc: dict) -> np.ndarray:
    """Unstandardize frozen per-record reference centers, without truth values."""
    ref = frozen['scoring']['reference']
    return np.asarray([ref['references'][r]['center'] for r in dc['regimes']]) * ref['scaler_scale'] + ref['scaler_center']


def gate(x: np.ndarray, center: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Fixed shrinkage formula: center + w*(observed-center)."""
    return center + weight * (x-center)


def arm_features(bundle: dict, group: str, scaler: dict, random: np.ndarray,
                 reference_centers: np.ndarray) -> np.ndarray:
    """Create 15-dimensional inputs; N/T/R never consume ideal masks."""
    if group not in ('N', 'T', 'R', 'O', 'TG', 'RG', 'OG'):
        raise ValueError(group)
    out = []
    for i, (v, s) in enumerate(zip(bundle['views'], bundle['scores'])):
        b = bundle['base'][i].copy()
        w = np.zeros_like(v['observations'])
        if group.startswith('T'):
            w = np.where(s['available'], s['reliability'], 0.)
        elif group.startswith('R'):
            w = np.where(s['available'], random, 0.)
        elif group.startswith('O'):
            w = np.where(s['available'], bundle['oracle'][i], 0.)
        if group.endswith('G'):
            b[..., :3] = gate(b[..., :3], reference_centers[v['recorded_regime']], w)
        b = transform_sequence(b, scaler)
        extra = np.zeros_like(w) if group == 'N' or group.endswith('G') else w
        out.append(np.concatenate([b, extra], axis=-1).astype(np.float32))
    return np.stack(out)


def fit_model(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
              schedule: np.ndarray, cfg: dict, seed: int, target: int
              ) -> tuple[dict, dict, list[dict], np.ndarray]:
    """Train exactly the registered budget with matched initialization and order."""
    ms = cfg['model_seed_offset'] + seed*10 + target
    ss = cfg['shuffle_seed_offset'] + seed*10 + target
    torch.manual_seed(ms)
    model = PlainLSTM(15, cfg['hidden_size'])
    initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    yc, ys = float(y.mean()), float(y.std())
    if ys <= 0 or x.shape[-1] != 15 or schedule.shape != (cfg['epochs'], len(y)):
        raise ValueError('training contract')
    tx = torch.from_numpy(x); ty = torch.from_numpy(((y-yc)/ys).astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    shuffle = np.random.default_rng(ss)
    histories, orders = [], []
    best, best_epoch, selected = float('inf'), -1, None
    vv = vx.reshape(-1, *vx.shape[2:]); yy = np.tile(vy, len(vx))
    for epoch in range(cfg['epochs']):
        order = shuffle.permutation(len(y)); orders.append(order)
        model.train(); loss_total = 0.
        for start in range(0, len(y), cfg['batch_size']):
            ix = order[start:start+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(tx[schedule[epoch, ix], ix])-ty[ix])**2).mean()
            if not torch.isfinite(loss): raise ValueError('nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step(); loss_total += float(loss.detach())*len(ix)
        pred = infer(model, vv, yc, ys, cfg['batch_size'])
        val = float(np.abs(pred-yy).mean())
        if val < best:
            best, best_epoch = val, epoch+1
            selected = {k: p.detach().clone() for k, p in model.state_dict().items()}
        histories.append(dict(epoch=epoch+1, train_standardized_mse=loss_total/len(y), val_suite_mae=val))
    model.load_state_dict(selected)
    pred = infer(model, vv, yc, ys, cfg['batch_size']).reshape(len(vx), len(vy))
    state = {k: p.numpy() for k, p in selected.items()}
    # Reconstruct from numpy checkpoint, independently initialized.
    replay = PlainLSTM(15, cfg['hidden_size'])
    replay.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()})
    delta = float(np.max(np.abs(infer(replay, vv, yc, ys, cfg['batch_size']).reshape(pred.shape)-pred)))
    if delta != 0: raise ValueError('checkpoint replay mismatch')
    for h in histories: h['selected'] = h['epoch'] == best_epoch
    params = dict(target_center=yc, target_scale=ys, model_seed=ms, shuffle_seed=ss,
                  initial_state_sha256=initial, shuffle_sha256=array_hash(np.stack(orders)),
                  schedule_sha256=array_hash(schedule), selected_epoch=best_epoch, val_suite_mae=best,
                  optimizer_updates=cfg['epochs']*math.ceil(len(y)/cfg['batch_size']),
                  training_examples=cfg['epochs']*len(y), input_channels=15,
                  parameter_count=sum(p.numel() for p in model.parameters()), replay_max_abs_error=delta)
    return state, params, histories, pred


def initialize(data: Path, ridge: Path, cfg: dict, out: Path, commit: str) -> dict:
    """Check immutable sources and bind each run to code, configuration and inputs."""
    torch.set_num_threads(cfg['threads']); torch.use_deterministic_algorithms(True)
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('implementation commit required')
    for root, pin in ((data, 'data_manifest_sha256'), (ridge, 'ridge_manifest_sha256')):
        if sha256(root/'manifest.json') != cfg[pin]: raise ValueError(pin)
        check_files(root, json.loads((root/'manifest.json').read_text()))
    if sha256(Path(__file__).with_name('rq3_paired_data.py')) != cfg['generator_sha256']:
        raise ValueError('generator changed')
    src = Path(__file__).parent
    names = sorted(set(SOURCE_FILES) | {'rq3_shared_fault.py', 'rq3_lstm_pilot.py', 'rq3_score_prediction_audit.py', '__init__.py'})
    identity = dict(implementation_commit=commit, source_sha256={n: sha256(src/n) for n in names},
                    data_source='synthetic_new_samples_old_train_reused', python=platform.python_version(),
                    dependencies={p: importlib.metadata.version(p) for p in ['torch','numpy','scipy','PyYAML']})
    dc = yaml.safe_load((data/'config.yaml').read_text()); dc['namespace'] = cfg['namespace']
    dc['dataset_id'] = cfg['run_id']+'_new_validation_test'
    dc['counts'].update(cfg['counts'])
    if out.exists():
        if (out/'manifest.json').exists(): raise FileExistsError('completed run')
        if json.loads((out/'config.json').read_text()) != cfg or json.loads((out/'identity.json').read_text()) != identity:
            raise ValueError('run identity changed')
    else:
        out.mkdir(parents=True)
        write_json(out/'config.json', cfg); write_json(out/'identity.json', identity)
        write_json(out/'new_generator_config.json', dc)
    return dc


def save_sample(path: Path, sample: dict) -> None:
    """Persist new base data, keeping hidden process and targets outside X."""
    path.mkdir(exist_ok=False)
    np.savez_compressed(path/'inputs.npz', **sample['inputs'])
    np.savez_compressed(path/'evaluation_only.npz', **sample['truth'], targets=sample['targets'])
    write_json(path/'events.json', sample['events'])


def context(data: Path, ridge: Path, cfg: dict, dc: dict, seed: int) -> tuple:
    """Recreate training-only pools and common features without reading new test."""
    frozen = json.loads((ridge/f'seed_{seed}/preprocessing_and_score.json').read_text())
    train = load_inputs(data, seed, 'train')
    if len(train['sequence_id']) != cfg['counts']['train']: raise ValueError('train size')
    yt = targets(data, seed, 'train', train['sequence_id'], cfg['targets'])
    scales = np.asarray(json.loads((data/f'seed_{seed}/fault_scales.json').read_text())['scales'])
    events = train_events(train['sequence_id'], train['observations'].shape[1], dc, seed)
    tb = make_bundle(train, events, scales, frozen, dc, cfg)
    return frozen, train, yt, scales, tb


def condition_setup(tb: dict, train: dict, cfg: dict, condition: str) -> tuple:
    """Fit only from the prescribed train condition; score bank includes its views."""
    indices = [0] if condition == 'clean' else list(range(len(tb['views'])))
    raw = tb['base'][indices]
    scaler = fit_sequence_scaler(raw.reshape(-1, *raw.shape[2:]))
    bank = np.stack([np.where(tb['scores'][i]['available'], tb['scores'][i]['reliability'], 0.) for i in indices])
    return scaler, bank


def train_seed(data: Path, ridge: Path, cfg: dict, out: Path, commit: str, seed: int) -> dict:
    """Train twenty models with shared fault samples, stopping at the fixed budget."""
    dc = initialize(data, ridge, cfg, out, commit)
    if seed not in cfg['seeds']: raise ValueError(seed)
    sd = out/f'seed_{seed}'/'training'
    sd.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frozen, train, yt, scales, tb = context(data, ridge, cfg, dc, seed)
    val = generate_split(dc, seed, 'val'); save_sample(sd/'validation', val)
    if set(train['sequence_id']) & set(val['inputs']['sequence_id']): raise ValueError('split overlap')
    vb = make_bundle(val['inputs'], val['events'], scales, frozen, dc, cfg)
    schedule = np.random.default_rng(cfg['schedule_seed_offset']+seed).integers(0, len(tb['views']), size=(cfg['epochs'],len(yt)), dtype=np.int64)
    np.savez_compressed(sd/'schedule.npz', shared=schedule, clean=np.zeros_like(schedule), sequence_id=train['sequence_id'])
    write_json(sd/'train_events.json', tb['events'])
    write_json(sd/'frozen_reference.json', frozen)
    pars, histories, models, preds, fairness = {}, [], {}, [], []
    for condition in ('clean', 'shared'):
        scaler, bank = condition_setup(tb, train, cfg, condition)
        rt, rti = random_trajectory(train['sequence_id'], train['sequence_id'], bank, seed, cfg['random_control_seed'])
        rv, rvi = random_trajectory(val['inputs']['sequence_id'], train['sequence_id'], bank, seed, cfg['random_control_seed'])
        np.savez_compressed(sd/f'{condition}_random_indices.npz', train=rti, val=rvi)
        pars[condition] = dict(scaler=scaler)
        reference_centers = centers(frozen, dc)
        sh = np.zeros_like(schedule) if condition == 'clean' else schedule
        common_hash = None
        for group in cfg[condition+'_groups']:
            name = condition+'_'+group
            x = arm_features(tb, group, scaler, rt, reference_centers)
            vx = arm_features(vb, group, scaler, rv, reference_centers)
            if group in ('N', 'T', 'R', 'O'):
                hashes = (array_hash(x[..., :12]), array_hash(vx[..., :12]))
                if common_hash is not None and hashes != common_hash: raise ValueError('unfair common features')
                common_hash = hashes
                fairness.append(dict(arm=name, train_common_sha256=hashes[0], val_common_sha256=hashes[1]))
            pp = []
            for ti, target in enumerate(cfg['targets']):
                key = name+'__'+target
                state, par, hist, pred = fit_model(x, yt[:, ti], vx, val['targets'][:,ti], sh, cfg, seed, ti)
                # Persist each completed model immediately for interruption recovery/evidence.
                np.savez_compressed(sd/(key+'.npz'), **state)
                write_json(sd/(key+'.json'), par)
                write_csv(sd/(key+'_history.csv'), hist)
                models[key] = par; pp.append(pred)
                histories.extend([dict(arm=name, target=target, **h) for h in hist])
                print(json.dumps(dict(seed=seed, arm=name, target=target, selected_epoch=par['selected_epoch'], elapsed_seconds=round(time.monotonic()-started,1))), flush=True)
            preds.append(np.stack(pp,axis=-1))
    # Exact initialization, order and budget matching, across all ten arms.
    for target in cfg['targets']:
        ms = [v for k,v in models.items() if k.endswith('__'+target)]
        for field in ('initial_state_sha256','shuffle_sha256','optimizer_updates','parameter_count'):
            if len({m[field] for m in ms}) != 1: raise ValueError('fairness:'+field)
        if len({models[a+'__'+target]['schedule_sha256'] for a in arm_names(cfg) if a.startswith('shared_')}) != 1:
            raise ValueError('different shared schedules')
    write_json(sd/'preprocessing.json', pars); write_json(sd/'model_parameters.json', models)
    write_json(sd/'fairness.json', fairness)
    write_csv(sd/'training_history.csv', histories)
    np.savez_compressed(sd/'validation_predictions.npz', prediction=np.stack(preds,axis=1), targets=val['targets'], sequence_id=val['inputs']['sequence_id'], arms=arm_names(cfg), views=[v for v,_ in view_specs(cfg)], target_names=cfg['targets'])
    meta = dict(seed=seed, models=len(models), total_optimizer_updates=sum(m['optimizer_updates'] for m in models.values()), validation_used_for_selection=True, test_generated=False, elapsed_seconds=time.monotonic()-started)
    write_json(sd/'metadata.json', meta); seal(sd)
    return meta


def lock_training(out: Path, cfg: dict) -> dict:
    """Require all seed training manifests before any new test sample is generated."""
    locks = {}
    for seed in cfg['seeds']:
        sd = out/f'seed_{seed}'/'training'
        if not (sd/'manifest.json').exists(): raise ValueError('all training must finish before test')
        check_files(sd, json.loads((sd/'manifest.json').read_text()))
        locks[str(seed)] = sha256(sd/'manifest.json')
    path = out/'training_lock.json'
    if path.exists():
        if json.loads(path.read_text()) != locks: raise ValueError('training changed after lock')
    else: write_json(path, locks)
    return locks


def metric_rows(pred: np.ndarray, y: np.ndarray, events: list[dict], seed: int,
                cfg: dict, arms: list[str], fresh: np.ndarray) -> list[dict]:
    """Compute paired views and fixed strata after inference, with no fitting."""
    channels = np.asarray([int(e['channel']) for e in events])
    duration = np.asarray([int(e['end_exclusive'])-int(e['start']) for e in events])
    out = []
    for vi,(view,_) in enumerate(view_specs(cfg)):
        strata = {'all': np.ones(len(y), bool)}
        if view.startswith(('bias','drift','missing')):
            strata.update({f'channel:{c}':channels==c for c in range(3)})
            strata.update({f'duration:{d}':duration==d for d in (4,12,24)})
        if view.startswith(('bias','drift')):
            strata.update({'fresh_alarm':fresh[vi], 'no_fresh_alarm':~fresh[vi]})
        for s, mask in strata.items():
            if not mask.any(): continue
            for ai,a in enumerate(arms):
                for ti,t in enumerate(cfg['targets']):
                    e = pred[vi,ai,mask,ti]-y[mask,ti]
                    out.append(dict(seed=seed,view=view,arm=a,target=t,stratum=s,batches=int(mask.sum()),mae=float(np.abs(e).mean()),rmse=float(np.sqrt((e**2).mean()))))
    return out


def evaluate_seed(data: Path, ridge: Path, cfg: dict, out: Path, commit: str, seed: int) -> dict:
    """Evaluate newly generated test once, only after all training is locked."""
    dc = initialize(data,ridge,cfg,out,commit); lock_training(out,cfg)
    if seed not in cfg['seeds']: raise ValueError(seed)
    sd = out/f'seed_{seed}'/'evaluation'; sd.mkdir(exist_ok=False)
    write_json(sd/'test_used.json', dict(seed=seed, used=True, reuse_for_selection=False))
    sample = generate_split(dc,seed,'test'); save_sample(sd/'test',sample)
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
    meta=dict(seed=seed,batches=len(sample['targets']),models=len(models),test_used=True)
    write_json(sd/'metadata.json',meta);seal(sd)
    return meta


def interval(values: list[float]) -> dict:
    """Five independent seed paired means and descriptive t uncertainty."""
    v=np.asarray(values);sd=float(v.std(ddof=1));h=float(student_t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v)))
    return dict(mean=float(v.mean()),sd=sd,ci_low=float(v.mean()-h),ci_high=float(v.mean()+h),negative_seeds=int((v<0).sum()),seeds=len(v))


def summarize(out: Path,cfg: dict) -> dict:
    """Report all results; never select the best-performing group as primary."""
    import csv
    if (out/'manifest.json').exists(): raise FileExistsError('completed run')
    lock_training(out,cfg);rows=[];scores=[]
    for seed in cfg['seeds']:
        sd=out/f'seed_{seed}'/'evaluation';check_files(sd,json.loads((sd/'manifest.json').read_text()))
        with (sd/'metrics.csv').open() as f:rows.extend(list(csv.DictReader(f)))
        with (sd/'score_metrics.csv').open() as f:scores.extend(list(csv.DictReader(f)))
    write_csv(out/'metrics_by_seed.csv',rows);write_csv(out/'score_by_seed.csv',scores)
    pairs=[('shared_T','shared_N'),('shared_T','shared_R'),('shared_N','clean_N'),('shared_T','clean_T'),('shared_R','clean_R'),('shared_O','shared_T'),('shared_O','shared_N'),('shared_TG','shared_T'),('shared_TG','shared_RG'),('shared_TG','shared_N'),('shared_OG','shared_TG'),('shared_OG','shared_N'),('shared_RG','shared_R'),('shared_OG','shared_O')]
    lookup={(int(r['seed']),r['view'],r['arm'],r['target'],r['stratum']):float(r['mae']) for r in rows}
    strata=sorted({(r['view'],r['stratum']) for r in rows});pairrows=[]
    for seed in cfg['seeds']:
        for target in cfg['targets']:
            for a,b in pairs:
                for view,s in strata:
                    if (seed,view,a,target,s) not in lookup:continue
                    pairrows.append(dict(seed=seed,target=target,comparison=a+'-'+b,view=view,stratum=s,mae_difference=lookup[seed,view,a,target,s]-lookup[seed,view,b,target,s]))
                delta=np.mean([lookup[seed,v,a,target,'all']-lookup[seed,v,b,target,'all'] for v,_ in view_specs(cfg)])
                pairrows.append(dict(seed=seed,target=target,comparison=a+'-'+b,view='suite_equal_weight',stratum='all',mae_difference=float(delta)))
    write_csv(out/'paired_by_seed.csv',pairrows)
    keys=sorted({(r['target'],r['comparison'],r['view'],r['stratum']) for r in pairrows});summary=[]
    for k in keys:
        vals=[r['mae_difference'] for r in pairrows if (r['target'],r['comparison'],r['view'],r['stratum'])==k]
        summary.append(dict(zip(['target','comparison','view','stratum'],k))|interval(vals))
    write_csv(out/'paired_summary.csv',summary)
    main=[r for r in summary if r['target']=='arrival_proxy' and r['comparison'] in ('shared_T-shared_N','shared_T-shared_R') and r['view']=='suite_equal_weight' and r['stratum']=='all']
    supported=all(r['ci_high']<0 and r['negative_seeds']==len(cfg['seeds']) for r in main) and len(main)==2
    result=dict(rq3_supported_under_registered_criterion=supported,primary=main,models=len(arm_names(cfg))*len(cfg['targets'])*len(cfg['seeds']),source='synthetic',scope='same frozen synthetic mechanism; adaptive research history; no industrial validation')
    write_json(out/'conclusion.json',result);seal(out)
    return result


def main() -> None:
    """Run one immutable training/evaluation stage or final summary."""
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate','summary'])
    p.add_argument('--config',default='configs/rq3_shared_fault_20261023.yaml');p.add_argument('--data',default='rq3_paired_20261019_rebuild_v1')
    p.add_argument('--ridge',default='results/rq3_prediction/rq3_ridge_pilot_20261019_v1');p.add_argument('--output',default='results/rq3_prediction/rq3_shared_fault_20261023_v1')
    p.add_argument('--commit',required=True);p.add_argument('--seed',type=int)
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());out=Path(a.output)
    if a.stage=='summary':result=summarize(out,cfg)
    else:result=(train_seed if a.stage=='train' else evaluate_seed)(Path(a.data),Path(a.ridge),cfg,out,a.commit,a.seed)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
