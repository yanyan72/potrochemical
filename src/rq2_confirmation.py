"""Frozen fresh scoring experiment and factorial support ablation."""
from __future__ import annotations
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtr,ndtri
from scipy.stats import t
from . import rq12_stress as base
from .rq2_restricted_fault import restricted_filter
from .rq2_factorial_fault import factorial_filter

def raw_score(x: np.ndarray,r: np.ndarray,model: dict,candidate: dict,cfg: dict,arm: str) -> tuple[np.ndarray,np.ndarray]:
    """Scorer does not receive masks, truth or targets."""
    if arm=='joint12':return base.score(x,r,model,candidate,cfg,arm)
    q=(factorial_filter if arm=='full24' else restricted_filter)(x,r,model,candidate)['mode']
    v=(x-model['center'])/model['scale'];mu=np.asarray(model['mu']);sd=np.sqrt(np.diagonal(model['cov'],axis1=1,axis2=2))
    u=(ndtr((v[:,:,:,None]-mu.T)/sd.T)*q[:,:,None,:]).sum(-1)
    return abs(base.a0.smooth(ndtri(np.clip(u,cfg['cdf_clip'],1-cfg['cdf_clip'])),r,cfg['alpha'])),q

def run(a: argparse.Namespace) -> None:
    """One fixed run, saving new inputs before scores and checking frozen sources."""
    started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());c=cfg['rq2'];c['schedule_seed']=2026110202;c['batches_per_seed']=32
    arms=['joint12','full24','restricted12'];gc=yaml.safe_load((a.data/'config.yaml').read_text());gc.update(namespace=2026110201,dataset_id='rq2_confirmation_20261102');gc['counts']['test']=32
    base.dump(a.output/'config.json',dict(base=cfg,generator=gc,arms=arms,primary='full24-restricted12 across six strong dual-fault views'))
    spec=importlib.util.spec_from_file_location('gen',a.generator);gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    sources={str(a.generator):base.digest(a.generator),str(a.config):base.digest(a.config),str(a.data/'config.yaml'):base.digest(a.data/'config.yaml')};rows=[]
    for seed in cfg['seeds']:
        sd=a.output/f'seed_{seed}';sd.mkdir();pars_path=a.scorers/f'seed_{seed}/parameters.json';pars=json.loads(pars_path.read_text());sources[str(pars_path)]=base.digest(pars_path);m=pars['model']['modes'];candidate=pars['candidate'];scales=np.asarray(pars['scales'])
        calp=a.data/f'seed_{seed}/reference_cal/inputs.npz';sources[str(calp)]=base.digest(calp);cal=base.a0.npz(calp);threshold={}
        for arm in arms:
            vals=[raw_score(cal['observations'],base.a0.delayed(cal['recorded_regime'],lag),m,candidate,c,arm)[0] for lag in c['lags']]
            threshold[arm]=np.quantile(np.concatenate(vals).reshape(-1,3),.99,axis=0)
        np.testing.assert_allclose(threshold['joint12'],pars['thresholds']['joint12'],atol=1e-11)
        np.testing.assert_allclose(threshold['full24'],pars['thresholds']['factorial24'],atol=1e-11)
        g=gen.generate_split(gc,seed,'test');inp=g['inputs'];x=inp['observations'];truth=inp['recorded_regime'];np.savez_compressed(sd/'inputs.npz',**inp)
        # Preserve all generator arrays, without passing them to scoring.
        for k,v in g.items():
            if k!='inputs' and isinstance(v,dict) and all(isinstance(vv,np.ndarray) for vv in v.values()):np.savez_compressed(sd/(k+'.npz'),**v)
        starts,signs,offsets=base.schedule(truth,seed,c);cache={};alarms=[];posteriors=[];masks=[];views=[];batch=[]
        for lag in c['lags']:
            for channels,mag in [([],0)]+[(ch,mag) for mag in c['magnitudes'] for ch in c['channel_sets']]:
                obs,mask=base.intervene(x,truth,scales,channels,mag,starts,signs,c['duration']);r=base.a0.delayed(truth,lag);view=f'lag{lag}_channels'+(''.join(map(str,channels)) if channels else 'none')+f'_bias{mag}'
                meta=dict(view=view,lag=lag,channels=''.join(map(str,channels)) if channels else 'none',magnitude=mag);views.append(meta);va=[];vq=[]
                for arm in arms:
                    raw,q=raw_score(obs,r,m,candidate,c,arm);alarm=raw>threshold[arm]
                    np.testing.assert_allclose(q.sum(-1),1,atol=1e-12)
                    if not channels:cache[lag,arm]=(alarm,q)
                    ca,cq=cache[lag,arm];rr=base.metric_rows(alarm,ca,q,cq,mask,truth,starts,signs,offsets,c['duration'])
                    for row in rr:row.update(seed=seed,arm=arm,**meta)
                    batch.extend(rr);va.append(alarm);vq.append(q)
                masks.append(mask);alarms.append(va);posteriors.append(vq)
        base.dump(sd/'parameters.json',dict(**pars,confirmation_thresholds={k:v.tolist() for k,v in threshold.items()}));base.dump(sd/'views.json',views)
        np.savez_compressed(sd/'scores.npz',alarm=alarms,q=posteriors,fault_mask=masks,starts=starts,signs=signs,offsets=offsets,arms=arms)
        pd.DataFrame(batch).to_csv(sd/'batch_metrics.csv',index=False);rows.extend(batch);print(seed,flush=True)
    f=base.aggregate(pd.DataFrame(rows));f.to_csv(a.output/'metrics_by_seed.csv',index=False)
    selected=f[(f.stratum=='all')&(f.regime==-1)&(f.magnitude==4)&f.channels.isin(['01','02','12'])];means=selected.groupby(['seed','arm'])[['point_recall','bystander_fpr','mode_error']].mean();results=[]
    for metric in ['point_recall','bystander_fpr']:
        tab=means[metric].unstack();delta=tab.full24-tab.restricted12;mu=float(delta.mean());sd=float(delta.std(ddof=1));half=float(t.ppf(.975,4)*sd/np.sqrt(5));p=float(2*t.sf(abs(mu/(sd/np.sqrt(5))),4))
        results.append(dict(metric=metric,mean_difference=mu,seed_differences=delta.tolist(),ci95=[mu-half,mu+half],p=p,group_means=tab.mean().to_dict()))
    prev=0.
    for rank,idx in enumerate(np.argsort([r['p'] for r in results])):prev=max(prev,min(1.,(2-rank)*results[idx]['p']));results[idx]['p_holm']=prev
    base.dump(a.output/'primary.json',results)
    for p,h in sources.items():assert base.digest(Path(p))==h
    base.dump(a.output/'source_hashes.json',sources);base.finish(a.output,started,dict(implementation_commit=a.commit,new_batches=160,namespace=2026110201,prediction_fits=0));print(json.dumps(results),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();
    for name in ['config','data','scorers','generator','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);run(p.parse_args())
