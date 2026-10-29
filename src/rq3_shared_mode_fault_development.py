"""One pre-specified candidate and two internal ablations on old development data."""
import argparse,csv,json,platform,sys,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri
import rq3_uncertainty_a0 as a0
from rq3_shared_mode_fault import joint_filter

def score_raw(x,r,model,candidate,cfg,arm):
    kw={'kappa':0.} if arm=='no_fault_hypothesis' else {'fault_memory':False} if arm=='no_fault_memory' else {}
    out=joint_filter(x,r,model,candidate,**kw)
    if arm=='no_fault_hypothesis':
        # Exact analytical collapse avoids amplifying ~1e-16 posterior differences
        # through inverse-normal CDF tails. Check the 12-state implementation first.
        collapsed=a0.mode_filter(x,r,model,candidate)
        np.testing.assert_allclose(out['mode'],collapsed,atol=1e-12,rtol=0)
        out['mode']=collapsed
    mu=np.asarray(model['mu']);sd=np.sqrt(np.diagonal(np.asarray(model['cov']),axis1=1,axis2=2))
    z=((x-np.asarray(model['center']))/model['scale'])[:,:,:,None]-mu.T[None,None,:,:]
    z=z/sd.T[None,None,:,:]
    q=np.repeat(out['mode'][:,:,None,:],3,axis=2)
    u=(q*ndtr(z)).sum(-1)
    residual=ndtri(np.clip(u,cfg['cdf_clip'],1-cfg['cdf_clip']))
    return np.abs(a0.smooth(residual,r,cfg['alpha'])),q,out

def decision(summary,baselines,cfg):
    def get(seed,arm,view,stratum='all'):
        src=summary if arm in cfg['arms'] else baselines
        return next(r for r in src if r['seed']==seed and r['split']=='val' and r['arm']==arm and r['view']==view and r['stratum']==stratum)
    g=cfg['gate'];checks=[];relative={};vibration_gains=[]
    for seed in cfg['seeds']:
        row={'seed':seed}
        for view in ['correct_clean','delay12_clean']:
            f=get(seed,'joint_full',view)['clean_fpr'];base=get(seed,'soft_all',view)['clean_fpr']
            row[view+'_fpr']=f;row[view+'_acceptable']=bool(f<=g['maximum_clean_fpr'] and f-base<=g['maximum_clean_fpr_increase_vs_soft_all'])
        for view in ['correct_bias4','delay12_bias4']:
            full=get(seed,'joint_full',view);loo=get(seed,'soft_loo',view)
            row[view+'_bystander_lower']=bool(full['bystander_fpr']<loo['bystander_fpr'])
            row[view+'_bystander_below_cap']=bool(full['bystander_fpr']<=g['maximum_bystander_fpr'])
            row[view+'_event_retained']=bool(full['event_added']-loo['event_added']>=-g['maximum_event_loss_vs_loo'])
            for ch in [0,1]:
                change=get(seed,'joint_full',view,f'channel_{ch}')['event_added']-get(seed,'soft_all',view,f'channel_{ch}')['event_added']
                row[view+f'_channel_{ch}_retained']=bool(change>=-g['maximum_other_channel_event_loss_vs_soft_all'])
            vibration_gains.append(get(seed,'joint_full',view,'channel_2')['event_added']-get(seed,'soft_all',view,'channel_2')['event_added'])
        checks.append(row)
    for view in ['correct_bias4','delay12_bias4']:
        before=np.mean([get(s,'soft_loo',view)['bystander_fpr'] for s in cfg['seeds']])
        after=np.mean([get(s,'joint_full',view)['bystander_fpr'] for s in cfg['seeds']])
        relative[view]=dict(before=float(before),after=float(after),relative_reduction=float((before-after)/before) if before else 0.)
    passed=all(v for r in checks for k,v in r.items() if k!='seed' and not k.endswith('_fpr'))
    passed=passed and all(v['relative_reduction']>=g['minimum_loo_bystander_relative_reduction'] for v in relative.values()) and np.mean(vibration_gains)>=g['minimum_vibration_event_gain_vs_soft_all']
    return dict(candidate='joint_full',pass_development_screen=bool(passed),allow_21_prediction_development_fits=bool(passed),rq3_supported=False,
                checks=checks,bystander_comparison=relative,vibration_event_gains=vibration_gains,mean_vibration_event_gain=float(np.mean(vibration_gains)),
                selection='no alternate ablation promotion',scope='adaptive development, three data seeds, no quality predictor')

def run(root,data,candidate,cfg,source_identity,out,commit):
    if out.exists():raise FileExistsError(out)
    if a0.git_blob(root/'manifest.json')!=source_identity['a0_manifest_git_blob']:raise ValueError('A0 identity mismatch')
    if a0.git_blob(Path(a0.__file__))!=source_identity['a0_module_git_blob']:raise ValueError('A0 code identity mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for rel,h in manifest.items():assert a0.digest(root/rel)==h
    if a0.digest(data/'manifest.json')!=cfg['dataset_manifest_sha256']:raise ValueError('dataset manifest mismatch')
    dm=json.loads((data/'manifest.json').read_text());calhash={}
    for seed in cfg['seeds']:
        rel=f'seed_{seed}/reference_cal/inputs.npz';h=a0.digest(data/rel)
        if h!=dm[rel]:raise ValueError('calibration input mismatch')
        calhash[rel]=h
    original_cfg=json.loads((root/'config.json').read_text());baselines=[]
    for r in csv.DictReader((root/'metrics_by_seed.csv').open()):
        row={}
        for k,v in r.items():row[k]=v if k in ['split','arm','view','stratum'] else (None if v=='' else float(v))
        baselines.append(row)
    out.mkdir(parents=True);a0.dump(out/'config.json',cfg);a0.dump(out/'candidate_config.json',candidate)
    a0.dump(out/'input_identity.json',dict(a0=source_identity,calibration_sha256=calhash))
    a0.dump(out/'identity.json',dict(implementation_commit=commit,attempt_directory=out.name,python=sys.version,numpy=np.__version__,platform=platform.platform(),quality_models_trained=0,new_latent_processes=0,test_access=False))
    started=time.perf_counter();summary=[];checks=[]
    for seed in cfg['seeds']:
        old=root/f'seed_{seed}';sd=out/f'seed_{seed}';sd.mkdir();model=json.loads((old/'model.json').read_text());m=model['modes']
        cal=a0.npz(data/f'seed_{seed}/reference_cal/inputs.npz');thresholds={};calibration=[]
        for arm in cfg['arms']:
            values=[]
            for lag in [0,12]:
                r=a0.delayed(cal['recorded_regime'],lag)
                raw,q,posterior=score_raw(cal['observations'],r,m,candidate,cfg,arm)
                if arm=='no_fault_hypothesis':
                    base,bq=a0.matched_raw(cal['observations'],r,m,original_cfg,'soft_all')
                    np.testing.assert_allclose(q,bq,atol=1e-12,rtol=0);np.testing.assert_allclose(raw,base,atol=1e-11,rtol=0)
                values.append(raw)
            threshold=np.quantile(np.concatenate(values).reshape(-1,3),cfg['calibration_quantile'],axis=0);thresholds[arm]=threshold.tolist()
            if arm=='no_fault_hypothesis':np.testing.assert_allclose(threshold,model['thresholds']['soft_all'],atol=1e-11,rtol=0)
            for lag,v in zip([0,12],values):calibration.append(dict(arm=arm,lag=lag,fpr=float((v>threshold).mean())))
        a0.dump(sd/'model.json',dict(modes=m,thresholds=thresholds,calibration=calibration,candidate=candidate))
        for split in cfg['splits']:
            events=json.loads((old/f'{split}_events.json').read_text());clean={};rows_all=[]
            ratio_arrays=[];alarm_arrays=[];mode_arrays=[];fault_arrays=[];joint_arrays=[]
            for view in original_cfg['views']:
                v=a0.npz(old/f'{split}_{view}.npz');rec,kind=view.split('_');rat=[];als=[];qs=[];fs=[]
                for arm in cfg['arms']:
                    raw,q,p=score_raw(v['observations'],v['recorded_regime'],m,candidate,cfg,arm)
                    ratio=raw/thresholds[arm];alarm=ratio>1
                    if arm=='no_fault_hypothesis':
                        np.testing.assert_allclose(ratio,v['soft_all_ratio'],atol=1e-6,rtol=1e-7)
                        np.testing.assert_array_equal(alarm,v['soft_all_alarm'])
                    if kind=='clean':clean[(rec,arm)]=alarm.copy()
                    rows=a0.per_batch(alarm,clean[(rec,arm)],v['fault_mask'],q,v['recorded_regime'],v['true_regime_evaluation_only'],events)
                    for row in rows:row.update(seed=seed,split=split,view=view,arm=arm)
                    rows_all.extend(rows)
                    for label,select in [('all',rows)]+[(f'channel_{j}',[r for r in rows if r['channel']==j]) for j in range(3)]+[(f'offset_{o}',[r for r in rows if r['offset']==o]) for o in original_cfg['start_offsets']]:
                        summary.append(dict(seed=seed,split=split,view=view,arm=arm,stratum=label,**a0.summarize_rows(select)))
                    rat.append(ratio.astype(np.float32));als.append(alarm);qs.append(p['mode'].astype(np.float32));fs.append(p['fault'].astype(np.float32))
                    if arm=='joint_full':joint_arrays.append(p['joint'].astype(np.float32))
                ratio_arrays.append(np.stack(rat));alarm_arrays.append(np.stack(als));mode_arrays.append(np.stack(qs));fault_arrays.append(np.stack(fs))
            np.savez_compressed(sd/f'{split}_scores.npz',ratio=np.stack(ratio_arrays),alarm=np.stack(alarm_arrays),mode=np.stack(mode_arrays),fault=np.stack(fault_arrays),joint_full_posterior=np.stack(joint_arrays),views=original_cfg['views'],arms=cfg['arms'],sequence_id=v['sequence_id'])
            a0.csv_write(sd/f'{split}_batch_metrics.csv',rows_all)
            checks.append(dict(seed=seed,split=split,no_fault_hypothesis_reduces_to_soft_all=True,alarm_replay_exact=True,batches=len(events)))
        print(json.dumps(dict(seed=seed,completed=True)),flush=True)
    a0.csv_write(out/'metrics_by_seed.csv',summary);a0.dump(out/'checks.json',checks)
    result=decision(summary,baselines,cfg);a0.dump(out/'decision.json',result)
    for rel,h in manifest.items():assert a0.digest(root/rel)==h
    for rel,h in calhash.items():assert a0.digest(data/rel)==h
    a0.dump(out/'source_unchanged.json',dict(a0_files=len(manifest),calibration_files=len(calhash),unchanged=True,elapsed_seconds=time.perf_counter()-started))
    a0.dump(out/'manifest.json',{str(p.relative_to(out)):a0.digest(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a0-root',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--source-identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(a.a0_root,a.data,json.loads(a.candidate.read_text()),json.loads(a.config.read_text()),json.loads(a.source_identity.read_text()),a.output,a.commit)
