"""Selective correction with the existing score boundary and frozen training design."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
import rq3_uncertainty_a0 as a0
import rq3_joint_prediction_dev as base

def trigger(score:np.ndarray,threshold:float)->np.ndarray:
    """Continuous correction strength, zero inside the existing acceptance region."""
    if not 0<threshold<=1 or not np.isfinite(score).all() or np.any((score<0)|(score>1)):raise ValueError('invalid score')
    return np.maximum(0.,1-score/threshold)

def selective_inputs(bundle:dict,arm:str,scaler:dict,random:np.ndarray,threshold:float)->np.ndarray:
    """Keep common features fixed and modify only the three observation channels."""
    z=base.inputs_for(bundle,'N',scaler,random).astype(float)
    if arm=='SG':score=bundle['S'];ref=bundle['center']
    elif arm=='SR':score=np.broadcast_to(random,bundle['S'].shape);ref=bundle['center']
    elif arm=='SA':score=bundle['ablationS'];ref=bundle['ablation_center']
    else:raise ValueError(arm)
    # Use original float64 normalized observations, then cast once like base.inputs_for.
    x=(bundle['x']-np.asarray(scaler['center'])[:3])/np.asarray(scaler['scale'])[:3]
    c=(ref-np.asarray(scaler['center'])[:3])/np.asarray(scaler['scale'])[:3]
    w=trigger(score,threshold);z[...,:3]=x+w*(c-x)
    return z.astype(np.float32)

def run(data:Path,a0root:Path,scorer:Path,prior:Path,cfg:dict,out:Path,commit:str,seed:int)->None:
    """Reuse all previous event, scaler, random-draw and training schedule artifacts."""
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    if a0.digest(prior/'manifest.json')!=cfg['base_result_manifest_sha256']:raise ValueError('base version')
    manifest=json.loads((prior/'manifest.json').read_text())
    for rel,h in manifest.items():assert a0.digest(prior/rel)==h
    prev=prior/f'seed_{seed}';source=json.loads((prev/'source_hashes.json').read_text())
    for p,h in source.items():assert a0.digest(p)==h
    if not out.exists():
        out.mkdir(parents=True);a0.dump(out/'config.json',cfg);a0.dump(out/'identity.json',dict(implementation_commit=commit,source_sha256={p.name:a0.digest(p) for p in [Path(__file__),Path(base.__file__),Path(a0.__file__),Path(__file__).with_name('rq3_shared_mode_fault.py'),Path(__file__).with_name('rq3_shared_mode_fault_development.py')]},data_source='synthetic_reused',test_used=False,new_latent_processes=0))
    else:assert json.loads((out/'config.json').read_text())==cfg and json.loads((out/'identity.json').read_text())['implementation_commit']==commit
    sd=out/f'seed_{seed}';sd.mkdir(exist_ok=False);a0.dump(sd/'source_hashes.json',source)
    old=json.loads((a0root/f'seed_{seed}/model.json').read_text());new=json.loads((scorer/f'seed_{seed}/model.json').read_text());candidate=json.loads((scorer/'candidate_config.json').read_text());scorecfg=json.loads((scorer/'config.json').read_text());viewcfg=json.loads((a0root/'config.json').read_text());scales=np.asarray(json.loads((data/f'seed_{seed}/fault_scales.json').read_text())['scales'])
    scaler=json.loads((prev/'scaler.json').read_text());schedule=a0.npz(prev/'schedule_and_random.npz');bundles={};ys={};ids={}
    for split in ['train','val']:
        v=a0.npz(data/f'seed_{seed}/{split}/inputs.npz');ids[split]=v['sequence_id'];np.testing.assert_array_equal(ids[split],schedule[split+'_ids']);ev=json.loads((prev/f'{split}_events.json').read_text())
        bundles[split]=base.score_bundle(v['observations'],v['recorded_regime'],ev,scales,old,new,candidate,scorecfg,viewcfg);ys[split]=base.labels(data,seed,split,ids[split],cfg['target'])
    tb,vb=bundles['train'],bundles['val'];random={}
    for split in ['train','val']:
        ix=schedule[split+'_random'];random[split]=tb['S'][ix[:,0],ix[:,1]].copy()
        # Recreate old common features and match original recorded identities.
        full=base.inputs_for(bundles[split],'NG',scaler,random[split]);oldfair=next(r for r in json.loads((prev/'fairness.json').read_text()) if r['arm']=='NG')
        assert base.ah(full)==oldfair[split+'_features_sha256']
    oldpar=json.loads((prev/'N_parameters.json').read_text());fair=[];predictions=[];diag=[]
    for arm in cfg['arms']:
        x=selective_inputs(tb,arm,scaler,random['train'],cfg['trigger_score']);vx=selective_inputs(vb,arm,scaler,random['val'],cfg['trigger_score'])
        for split,arr in [('train',x),('val',vx)]:np.testing.assert_array_equal(arr[...,3:],base.inputs_for(bundles[split],'N',scaler,random[split])[...,3:])
        state,par,hist,pred=base.fit(x,ys['train'],vx,ys['val'],schedule['schedule'],cfg)
        for key in ['initial_sha256','shuffle_sha256','schedule_sha256','optimizer_updates','parameter_count','target_center','target_scale']:assert par[key]==oldpar[key],key
        np.savez_compressed(sd/f'{arm}_model.npz',**state);a0.dump(sd/f'{arm}_parameters.json',par);a0.csv_write(sd/f'{arm}_history.csv',hist)
        model=base.Predictor(cfg['input_channels'],cfg['hidden_size']);model.load_state_dict({k:torch.from_numpy(v) for k,v in a0.npz(sd/f'{arm}_model.npz').items()});rp=base.predict(model,vx.reshape(-1,*vx.shape[2:]),par['target_center'],par['target_scale'],cfg['batch_size']).reshape(pred.shape);np.testing.assert_array_equal(pred,rp)
        np.savez_compressed(sd/f'{arm}_predictions.npz',pred=pred);predictions.append(pred);fair.append(dict(arm=arm,replay_error=0.,train_features_sha256=base.ah(x),val_features_sha256=base.ah(vx),**par));print(json.dumps(dict(seed=seed,arm=arm,mae=par['val_mae'],epoch=par['selected_epoch'],seconds=par['elapsed_seconds'])),flush=True)
        ss=vb['S'] if arm=='SG' else np.broadcast_to(random['val'],vb['S'].shape) if arm=='SR' else vb['ablationS'];w=trigger(ss,cfg['trigger_score'])
        for vi,view in enumerate(cfg['views']):
            for ch in range(3):
                for region,mask in [('normal',~vb['mask'][vi,:,:,ch]),('fault',vb['mask'][vi,:,:,ch])]:
                    if mask.any():diag.append(dict(seed=seed,arm=arm,view=view,channel=ch,region=region,n=int(mask.sum()),activation_rate=float((w[vi,:,:,ch][mask]>0).mean()),mean_strength=float(w[vi,:,:,ch][mask].mean())))
    np.savez_compressed(sd/'predictions.npz',pred=np.stack(predictions),arms=cfg['arms'],views=cfg['views'],y=ys['val'],sequence_id=ids['val']);shutil.copyfile(prev/'evaluation_metadata.npz',sd/'evaluation_metadata.npz');a0.csv_write(sd/'activation.csv',diag);a0.dump(sd/'fairness.json',fair)
    for p,h in source.items():assert a0.digest(p)==h
    a0.dump(sd/'completion.json',dict(models=3,source_unchanged=True));base.seal(sd)

def summarize(out:Path,prior:Path)->None:
    """Combine stored old and new predictions; preserve all scenario/seed strata."""
    cfg=json.loads((out/'config.json').read_text());rows=[];pairs=[];suite=[]
    for seed in cfg['seeds']:
        new=a0.npz(out/f'seed_{seed}/predictions.npz');old=a0.npz(prior/f'seed_{seed}/predictions.npz');meta=a0.npz(out/f'seed_{seed}/evaluation_metadata.npz');np.testing.assert_array_equal(old['y'],new['y']);np.testing.assert_array_equal(old['sequence_id'],new['sequence_id'])
        arms=list(old['arms'])+list(new['arms']);p=np.concatenate([old['pred'],new['pred']]);e=np.abs(p-new['y'][None,None,:]);np.savez_compressed(out/f'seed_{seed}_combined.npz',pred=p,y=new['y'],arms=arms,views=cfg['views'],sequence_id=new['sequence_id'])
        groups={'all':np.ones(len(new['y']),bool)};groups.update({f'channel_{j}':meta['channel']==j for j in range(3)});groups.update({f'offset_{o}':meta['offset']==o for o in [-6,18]});groups.update({f'end_regime_{j}':meta['end_regime']==j for j in range(3)})
        for ai,arm in enumerate(arms):
            suite.append(dict(seed=seed,arm=arm,mae=float(e[ai].mean())))
            for vi,view in enumerate(cfg['views']):
                for group,mask in groups.items():
                    if mask.any():rows.append(dict(seed=seed,arm=arm,view=view,stratum=group,n=int(mask.sum()),mae=float(e[ai,vi,mask].mean()),rmse=float(np.sqrt(np.square(p[ai,vi,mask]-new['y'][mask]).mean()))))
        for a,b in cfg['comparisons']:
            d=e[arms.index(a)]-e[arms.index(b)];pairs.append(dict(seed=seed,comparison=f'{a}-{b}',view='suite',stratum='all',mae_difference=float(d.mean()),degradation_difference=None))
            for vi,view in enumerate(cfg['views']):
                ci=cfg['views'].index(view.split('_')[0]+'_clean')
                for group,mask in groups.items():
                    if mask.any():pairs.append(dict(seed=seed,comparison=f'{a}-{b}',view=view,stratum=group,mae_difference=float(d[vi,mask].mean()),degradation_difference=float((d[vi,mask]-d[ci,mask]).mean())))
    agg=[]
    for key in sorted({(r['comparison'],r['view'],r['stratum']) for r in pairs}):
        rr=[r for r in pairs if (r['comparison'],r['view'],r['stratum'])==key];v=np.array([r['mae_difference'] for r in rr]);agg.append(dict(comparison=key[0],view=key[1],stratum=key[2],mean=float(v.mean()),sd=float(v.std(ddof=1)),negative_seeds=int((v<0).sum()),outer_seeds=len(v)))
    for name,rs in [('metrics_by_seed',rows),('suite_by_seed',suite),('paired_by_seed',pairs),('paired_summary',agg)]:a0.csv_write(out/(name+'.csv'),rs)
    base.seal(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','summarize']);p.add_argument('--data',type=Path);p.add_argument('--a0-root',type=Path);p.add_argument('--scorer-root',type=Path);p.add_argument('--base-root',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.command=='summarize':summarize(a.output,a.base_root)
    else:run(a.data,a.a0_root,a.scorer_root,a.base_root,json.loads(a.config.read_text()),a.output,a.commit,a.seed)
