"""Frozen 50+15 confirmation; separate training lock and one fresh-test evaluation."""
from __future__ import annotations
import argparse, csv, importlib.util, json, platform
from pathlib import Path
import numpy as np
import torch
import yaml
from scipy.stats import t as student_t
import rq3_uncertainty_a0 as a0
import rq3_joint_prediction_dev as base
import rq3_selective_prediction_dev as selective
from rq3_shared_mode_fault_development import score_raw


def read(p: Path) -> dict:
    """Read persisted JSON without mutation."""
    return json.loads(p.read_text())


def bundle(x: np.ndarray, r: np.ndarray, ev: list, scales: np.ndarray,
           model: dict, candidate: dict, scorecfg: dict, viewcfg: dict, views: list) -> dict:
    """Frozen scoring with explicit lag parsing, no quality targets or hidden truth."""
    m=model['modes']; physical=np.asarray(m['mu'])*m['scale']+m['center']
    rows={k:[] for k in ['x','r','mask','S','ablationS','q','center','ablation_center']}
    for view in views:
        rec,kind=view.split('_'); lag=0 if rec=='correct' else int(rec.removeprefix('delay'))
        xv,rv,mask=a0.paired_view(x,r,ev,scales,kind,lag,viewcfg)
        raw,_,p=score_raw(xv,rv,m,candidate,scorecfg,'joint_full')
        ra,_,pa=score_raw(xv,rv,m,candidate,scorecfg,'no_fault_hypothesis')
        values=dict(x=xv,r=rv,mask=mask,S=np.exp(-.5*(raw/model['thresholds']['joint_full'])**2),
                    ablationS=np.exp(-.5*(ra/model['thresholds']['no_fault_hypothesis'])**2),
                    q=p['mode'],center=p['mode']@physical,ablation_center=pa['mode']@physical)
        for key,value in values.items(): rows[key].append(value)
    return {key:np.stack(value) for key,value in rows.items()}


def features(b: dict, arm: str, scaler: dict, random: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply exactly one frozen arm with common non-observation channels."""
    return (base.inputs_for(b,arm,scaler,random) if arm in ['N','NG'] else
            selective.selective_inputs(b,arm,scaler,random,cfg['trigger_score']))


def prepare_score(seed: int, data: Path, legacy: Path, oldroot: Path, scorer: Path,
                  cfg: dict, candidate: dict, scorecfg: dict, viewcfg: dict) -> dict:
    """Reuse fitted development scores; extend identical calibration to seeds45/46."""
    if (scorer/f'seed_{seed}/model.json').exists():return read(scorer/f'seed_{seed}/model.json')
    frozen=read(legacy/f'results/rq3_prediction/rq3_shared_fault_20261023_v1/seed_{seed}/training/frozen_reference.json')['scoring']
    m=a0.fit_modes(a0.npz(data/f'seed_{seed}/reference_fit/inputs.npz'),frozen,viewcfg)
    cal=a0.npz(data/f'seed_{seed}/reference_cal/inputs.npz'); thresholds={}
    for arm in ['joint_full','no_fault_hypothesis']:
        values=[score_raw(cal['observations'],a0.delayed(cal['recorded_regime'],lag),m,candidate,scorecfg,arm)[0] for lag in [0,12]]
        thresholds[arm]=np.quantile(np.concatenate(values).reshape(-1,3),scorecfg['calibration_quantile'],axis=0).tolist()
    return dict(modes=m,thresholds=thresholds,candidate=candidate)


def train_seed(args: argparse.Namespace) -> None:
    """Fit13 models with fixed matched inputs and two independent optimizer streams."""
    cfg=read(args.config); seed=args.seed; out=args.output; data=args.data
    assert seed in cfg['seeds']; torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    assert a0.digest(data/'manifest.json')==cfg['source_dataset_manifest']; dm=read(data/'manifest.json')
    source={}; paths=[data/'manifest.json',data/'config.yaml']
    paths += [data/f'seed_{seed}/{split}/{name}' for split in ['reference_fit','reference_cal','train','val'] for name in (['inputs.npz','targets.csv'] if split in ['train','val'] else ['inputs.npz'])]
    paths += [data/f'seed_{seed}/fault_scales.json',args.scorer/'config.json',args.scorer/'candidate_config.json',args.a0_root/'config.json']
    if (args.scorer/f'seed_{seed}').exists():paths += [args.scorer/f'seed_{seed}/model.json']
    else: paths += [args.legacy/f'results/rq3_prediction/rq3_shared_fault_20261023_v1/seed_{seed}/training/frozen_reference.json']
    prev=args.base_root/f'seed_{seed}'
    if prev.exists(): paths += [prev/p for p in ['train_events.json','val_events.json','scaler.json','schedule_and_random.npz','fairness.json']]
    for p in paths:
        source[str(p)]=a0.digest(p)
        if p.is_relative_to(data) and str(p.relative_to(data)) in dm:assert source[str(p)]==dm[str(p.relative_to(data))]
    if not out.exists():
        out.mkdir(parents=True);a0.dump(out/'config.json',cfg)
        a0.dump(out/'identity.json',dict(implementation_commit=args.commit,python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,data_source='synthetic',source_hashes={p.name:a0.digest(p) for p in Path(__file__).parent.glob('rq3_*.py')}))
    else:assert read(out/'config.json')==cfg and read(out/'identity.json')['implementation_commit']==args.commit
    sd=out/'training'/f'seed_{seed}';sd.mkdir(parents=True,exist_ok=False);a0.dump(sd/'source_hashes.json',source)
    candidate=read(args.scorer/'candidate_config.json');scorecfg=read(args.scorer/'config.json');viewcfg=read(args.a0_root/'config.json')
    for name,obj in [('candidate',candidate),('score_config',scorecfg),('view_config',viewcfg)]:a0.dump(sd/(name+'.json'),obj)
    model=prepare_score(seed,data,args.legacy,args.a0_root,args.scorer,cfg,candidate,scorecfg,viewcfg);a0.dump(sd/'score_model.json',model)
    scales=np.asarray(read(data/f'seed_{seed}/fault_scales.json')['scales']);a0.dump(sd/'fault_scales.json',scales.tolist())
    bundles={};ys={};ids={};random={};ix={}
    for split in ['train','val']:
        v=a0.npz(data/f'seed_{seed}/{split}/inputs.npz');ids[split]=v['sequence_id'];assert len(ids[split])==cfg['counts'][split]
        if prev.exists():ev=read(prev/f'{split}_events.json')
        else:
            first=a0.events_for(ids[split][:60],v['recorded_regime'][:60],viewcfg,seed,split)
            ev=base.events(ids[split],v['recorded_regime'],first,cfg,seed,split)
        a0.dump(sd/f'{split}_events.json',ev)
        bundles[split]=bundle(v['observations'],v['recorded_regime'],ev,scales,model,candidate,scorecfg,viewcfg,cfg['views'])
        ys[split]={target:base.labels(data,seed,split,ids[split],target) for target in [cfg['target'],cfg['negative_control']['target']]}
    assert not set(ids['train'])&set(ids['val']);tb,vb=bundles['train'],bundles['val']
    scaler=base.scaler_fit(base.base_features(tb));a0.dump(sd/'scaler.json',scaler)
    for split in ['train','val']:random[split],ix[split]=base.draw_random(ids[split],ids['train'],tb['S'],seed,cfg['random_control_seed'])
    schedule=np.random.default_rng(np.random.SeedSequence([cfg['schedule_seed'],seed])).integers(0,len(cfg['views']),size=(cfg['epochs'],len(ids['train'])),dtype=np.int64)
    if prev.exists():
        assert scaler==read(prev/'scaler.json');previous=a0.npz(prev/'schedule_and_random.npz')
        np.testing.assert_array_equal(schedule,previous['schedule'])
        for split in ['train','val']:
            np.testing.assert_array_equal(ix[split],previous[split+'_random'])
            fair=next(r for r in read(prev/'fairness.json') if r['arm']=='NG')
            assert base.ah(features(bundles[split],'NG',scaler,random[split],cfg))==fair[split+'_features_sha256']
    np.savez_compressed(sd/'schedule_and_random.npz',schedule=schedule,train_random=ix['train'],val_random=ix['val'],train_ids=ids['train'],val_ids=ids['val'])
    signatures={};fairness=[];models=[]
    for target,repetitions,arms in [(cfg['target'],range(2),cfg['arms']),(cfg['negative_control']['target'],[0],cfg['negative_control']['arms'])]:
        for rep in repetitions:
            run_cfg={**cfg,**cfg['optimization_repeats'][rep]}
            for arm in arms:
                key=f'{target}_opt{rep}_{arm}';md=sd/key;md.mkdir()
                x=features(tb,arm,scaler,random['train'],cfg);vx=features(vb,arm,scaler,random['val'],cfg)
                for b,arr,rnd in [(tb,x,random['train']),(vb,vx,random['val'])]:np.testing.assert_array_equal(arr[...,3:],base.inputs_for(b,'N',scaler,rnd)[...,3:])
                state,par,hist,pred=base.fit(x,ys['train'][target],vx,ys['val'][target],schedule,run_cfg)
                sig=tuple(par[k] for k in ['initial_sha256','shuffle_sha256','schedule_sha256','optimizer_updates','parameter_count'])
                if rep in signatures:assert sig==signatures[rep]
                signatures[rep]=sig
                np.savez_compressed(md/'model.npz',**state);a0.dump(md/'parameters.json',par);a0.csv_write(md/'history.csv',hist)
                replay=base.Predictor(cfg['input_channels'],cfg['hidden_size']);replay.load_state_dict({k:torch.from_numpy(v) for k,v in a0.npz(md/'model.npz').items()})
                rp=base.predict(replay,vx.reshape(-1,*vx.shape[2:]),par['target_center'],par['target_scale'],cfg['batch_size']).reshape(pred.shape);np.testing.assert_array_equal(pred,rp)
                # Original first-repetition models must replay exactly on seeds42-44.
                prior_model=None
                if target==cfg['target'] and rep==0 and prev.exists():
                    prior_model=(prev/f'{arm}_predictions.npz') if arm in ['N','NG'] else args.selective_root/f'seed_{seed}/{arm}_predictions.npz'
                    np.testing.assert_array_equal(pred,a0.npz(prior_model)['pred'])
                np.savez_compressed(md/'validation.npz',pred=pred,y=ys['val'][target],sequence_id=ids['val'],views=cfg['views'])
                fairness.append(dict(key=key,target=target,repeat=rep,arm=arm,train_features_sha256=base.ah(x),val_features_sha256=base.ah(vx),replay_exact=True,old_prediction_exact=prior_model is not None,**par));models.append(key)
                print(json.dumps(dict(stage='train',seed=seed,key=key,val_mae=par['val_mae'],epoch=par['selected_epoch'])),flush=True)
    assert signatures[0][0]!=signatures[1][0] and signatures[0][1]!=signatures[1][1]
    a0.dump(sd/'fairness.json',fairness);a0.dump(sd/'completion.json',dict(models=models,count=len(models),source_unchanged=True))
    for p,h in source.items():assert a0.digest(p)==h
    base.seal(sd)


def lock_training(out: Path) -> None:
    """Seal all training before fresh test can be generated."""
    cfg=read(out/'config.json'); count=0
    for seed in cfg['seeds']:
        sd=out/'training'/f'seed_{seed}';m=read(sd/'manifest.json')
        for p,h in m.items():assert a0.digest(sd/p)==h
        count+=read(sd/'completion.json')['count']
    assert count==65
    base.seal(out/'training');a0.dump(out/'training_lock.json',dict(models=count,manifest_sha256=a0.digest(out/'training/manifest.json'),config_sha256=a0.digest(out/'config.json'),method_frozen=True))


def fresh_events(ids: np.ndarray, records: np.ndarray, cfg: dict, seed: int) -> list:
    """Balanced fixed events independent of target and model; never selected by outcome."""
    ec={**cfg,'event_seed':cfg['test_event_seed']}
    return base.events(ids,records,[],ec,seed,'test')


def evaluate_seed(args: argparse.Namespace) -> None:
    """Generate once after lock, predict without targets, then save evaluation labels."""
    out=args.output;cfg=read(out/'config.json');seed=args.seed;lock=read(out/'training_lock.json')
    assert lock['models']==65 and a0.digest(out/'training/manifest.json')==lock['manifest_sha256']
    assert a0.digest(out/'config.json')==lock['config_sha256'];assert args.lock_commit
    for p,h in read(out/'training/manifest.json').items():assert a0.digest(out/'training'/p)==h
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    sd=out/'training'/f'seed_{seed}';ed=out/'evaluation'/f'seed_{seed}';ed.mkdir(parents=True,exist_ok=False)
    a0.dump(ed/'test_use.json',dict(namespace=cfg['test_namespace'],training_lock_commit=args.lock_commit,source='new_synthetic_holdout',not_enterprise=True))
    spec=importlib.util.spec_from_file_location('frozen_generator',args.legacy/'src/rq3_paired_data.py');gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    gc=yaml.safe_load((args.data/'config.yaml').read_text());gc.update(namespace=cfg['test_namespace'],dataset_id=cfg['run_id']+'_test');gc['counts']['test']=cfg['test_batches_per_seed']
    a0.dump(ed/'generator_config.json',gc);a0.dump(ed/'generator_identity.json',dict(sha256=a0.digest(args.legacy/'src/rq3_paired_data.py')))
    g=gen.generate_split(gc,seed,'test');v=g['inputs'];np.savez_compressed(ed/'inputs.npz',**v)
    np.savez_compressed(ed/'evaluation_only.npz',**g['truth'],targets=g['targets'],target_names=['arrival_proxy','null_proxy']);del g
    prior=a0.npz(sd/'schedule_and_random.npz');assert not set(v['sequence_id'])&set(np.r_[prior['train_ids'],prior['val_ids']])
    ev=fresh_events(v['sequence_id'],v['recorded_regime'],cfg,seed);a0.dump(ed/'events.json',ev)
    model=read(sd/'score_model.json');candidate=read(sd/'candidate.json');scorecfg=read(sd/'score_config.json');viewcfg=read(sd/'view_config.json');scales=np.asarray(read(sd/'fault_scales.json'));views=cfg['views']+cfg['secondary_views']
    b=bundle(v['observations'],v['recorded_regime'],ev,scales,model,candidate,scorecfg,viewcfg,views)
    tr=a0.npz(args.data/f'seed_{seed}/train/inputs.npz');tb=bundle(tr['observations'],tr['recorded_regime'],read(sd/'train_events.json'),scales,model,candidate,scorecfg,viewcfg,cfg['views'])
    random,ix=base.draw_random(v['sequence_id'],tr['sequence_id'],tb['S'],seed,cfg['random_control_seed']);del tb
    np.savez_compressed(ed/'evaluation_metadata.npz',sequence_id=v['sequence_id'],channel=[e['channel'] for e in ev],offset=[e['offset'] for e in ev],end_regime=v['recorded_regime'][:,-1],random_indices=ix)
    predictions=[];keys=[];scaler=read(sd/'scaler.json')
    for arm in cfg['arms']:
        x=features(b,arm,scaler,random,cfg)
        for row in read(sd/'fairness.json'):
            if row['arm']!=arm:continue
            md=sd/row['key'];p=read(md/'parameters.json');modelnn=base.Predictor(cfg['input_channels'],cfg['hidden_size']);modelnn.load_state_dict({k:torch.from_numpy(w) for k,w in a0.npz(md/'model.npz').items()})
            pred=base.predict(modelnn,x.reshape(-1,*x.shape[2:]),p['target_center'],p['target_scale'],cfg['batch_size']).reshape(len(views),len(v['sequence_id']));predictions.append(pred);keys.append(row['key'])
    # Persist all13 predictions before any evaluation-target read.
    np.savez_compressed(ed/'predictions.npz',pred=np.stack(predictions),keys=keys,views=views,sequence_id=v['sequence_id'])
    target=a0.npz(ed/'evaluation_only.npz');np.testing.assert_array_equal(target['sequence_id'],v['sequence_id'])
    a0.dump(ed/'prediction_complete.json',dict(models=len(keys),prediction_sha256=a0.digest(ed/'predictions.npz'),targets_read_only_after_prediction=True))
    diag=[];score_rows=[]
    for vi,view in enumerate(views):
        clean=views.index(view.split('_')[0]+'_clean')
        for arm,score in [('SG',b['S']),('SA',b['ablationS'])]:
            alarm=score[vi]<cfg['trigger_score'];clean_alarm=score[clean]<cfg['trigger_score']
            # Detection counts use common q only for descriptive regime metrics.
            rr=a0.per_batch(alarm,clean_alarm,b['mask'][vi],np.repeat(b['q'][vi,:,:,None,:],3,axis=2),b['r'][vi],v['recorded_regime'],ev)
            for stratum,subset in [('all',rr)]+[(f'channel_{j}',[r for r in rr if r['channel']==j]) for j in range(3)]:
                score_rows.append(dict(seed=seed,arm=arm,view=view,stratum=stratum,**a0.summarize_rows(subset)))
        for arm in ['NG','SG','SR','SA']:
            score=b['ablationS'][vi] if arm=='SA' else random if arm=='SR' else b['S'][vi]
            w=1-score if arm=='NG' else selective.trigger(score,cfg['trigger_score'])
            for ch in range(3):
                for region,mask in [('normal',~b['mask'][vi,:,:,ch]),('fault',b['mask'][vi,:,:,ch])]:
                    if mask.any():diag.append(dict(seed=seed,arm=arm,view=view,channel=ch,region=region,n=int(mask.sum()),activation_rate=float((w[:,:,ch][mask]>0).mean()),mean_strength=float(w[:,:,ch][mask].mean())))
    a0.csv_write(ed/'score_metrics.csv',score_rows);a0.csv_write(ed/'activation.csv',diag);base.seal(ed)
    print(json.dumps(dict(stage='test',seed=seed,batches=len(v['sequence_id']),models=len(keys),views=len(views))),flush=True)


def paired_stats(values: np.ndarray) -> dict:
    """Outer data-seed t interval; callers average optimizer repeats first."""
    values=np.asarray(values,float);n=len(values);mean=float(values.mean());sd=float(values.std(ddof=1));se=sd/np.sqrt(n);half=float(student_t.ppf(.975,n-1)*se)
    return dict(mean=mean,sd=sd,ci_low=mean-half,ci_high=mean+half,outer_seeds=n,negative_seeds=int((values<0).sum()),loo_min=float(min(np.delete(values,i).mean() for i in range(n))),loo_max=float(max(np.delete(values,i).mean() for i in range(n))))


def summarize(out: Path) -> None:
    """Full paired stratification without treating correlated views/batches as replicates."""
    cfg=read(out/'config.json');rows=[];pairs=[]
    for seed in cfg['seeds']:
        ed=out/'evaluation'/f'seed_{seed}';p=a0.npz(ed/'predictions.npz');y=a0.npz(ed/'evaluation_only.npz');meta=a0.npz(ed/'evaluation_metadata.npz')
        groups={'all':np.ones(len(meta['sequence_id']),bool)};groups.update({f'channel_{j}':meta['channel']==j for j in range(3)});groups.update({f'offset_{j}':meta['offset']==j for j in [-6,18]});groups.update({f'end_regime_{j}':meta['end_regime']==j for j in range(3) if np.any(meta['end_regime']==j)})
        for target,reps,arms in [(cfg['target'],range(2),cfg['arms']),(cfg['negative_control']['target'],[0],cfg['negative_control']['arms'])]:
            yi=list(y['target_names']).index(target);truth=y['targets'][:,yi]
            for rep in reps:
                error={}
                for arm in arms+['train_mean']:
                    if arm=='train_mean':pred=np.full((len(p['views']),len(truth)),read(out/'training'/f'seed_{seed}'/f'{target}_opt{rep}_N/parameters.json')['target_center'])
                    else:pred=p['pred'][list(p['keys']).index(f'{target}_opt{rep}_{arm}')]
                    error[arm]=np.abs(pred-truth[None,:])
                    for view in ['suite']+list(p['views']):
                        indices=range(len(cfg['views'])) if view=='suite' else [list(p['views']).index(view)]
                        for group,mask in groups.items():
                            e=error[arm][list(indices)][:,mask];d=(pred-truth[None,:])[list(indices)][:,mask]
                            rows.append(dict(seed=seed,target=target,repeat=rep,arm=arm,view=view,stratum=group,n_batches=int(mask.sum()),mae=float(e.mean()),rmse=float(np.sqrt((d*d).mean()))))
                for a,bb in cfg['comparisons']:
                    if a not in arms or bb not in arms:continue
                    diff=error[a]-error[bb]
                    for view in ['suite']+list(p['views']):
                        vi=None if view=='suite' else list(p['views']).index(view);indices=range(len(cfg['views'])) if vi is None else [vi]
                        ci=None if vi is None else list(p['views']).index(view.split('_')[0]+'_clean')
                        for group,mask in groups.items():
                            pairs.append(dict(seed=seed,target=target,repeat=rep,comparison=f'{a}-{bb}',view=view,stratum=group,mae_difference=float(diff[list(indices)][:,mask].mean()),degradation_difference=None if vi is None else float((diff[vi]-diff[ci])[mask].mean())))
    a0.csv_write(out/'metrics_by_repeat.csv',rows);a0.csv_write(out/'paired_by_repeat.csv',pairs)
    collapsed=[]
    for key in sorted({(r['seed'],r['target'],r['comparison'],r['view'],r['stratum']) for r in pairs}):
        rr=[r for r in pairs if (r['seed'],r['target'],r['comparison'],r['view'],r['stratum'])==key]
        collapsed.append(dict(zip(['seed','target','comparison','view','stratum'],key),optimization_repeats=len(rr),mae_difference=float(np.mean([r['mae_difference'] for r in rr]))))
    a0.csv_write(out/'paired_by_seed.csv',collapsed);agg=[];byopt=[]
    for key in sorted({(r['target'],r['comparison'],r['view'],r['stratum']) for r in collapsed}):
        rr=[r for r in collapsed if (r['target'],r['comparison'],r['view'],r['stratum'])==key];assert len(rr)==5
        agg.append(dict(zip(['target','comparison','view','stratum'],key),**paired_stats([r['mae_difference'] for r in rr])))
        for rep in ([0,1] if key[0]==cfg['target'] else [0]):
            rr=[r for r in pairs if (r['target'],r['comparison'],r['view'],r['stratum'])==key and r['repeat']==rep]
            byopt.append(dict(zip(['target','comparison','view','stratum'],key),repeat=rep,**paired_stats([r['mae_difference'] for r in rr])))
    a0.csv_write(out/'paired_summary.csv',agg);a0.csv_write(out/'optimizer_sensitivity.csv',byopt)
    print(json.dumps([r for r in agg if r['view']=='suite' and r['stratum']=='all']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','lock','evaluate','summarize']);p.add_argument('--data',type=Path);p.add_argument('--legacy',type=Path);p.add_argument('--a0-root',type=Path);p.add_argument('--scorer',type=Path);p.add_argument('--base-root',type=Path);p.add_argument('--selective-root',type=Path);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int);p.add_argument('--commit');p.add_argument('--lock-commit');args=p.parse_args()
    if args.command=='train':train_seed(args)
    elif args.command=='lock':lock_training(args.output)
    elif args.command=='evaluate':evaluate_seed(args)
    else:summarize(args.output)
