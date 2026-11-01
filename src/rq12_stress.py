"""Bounded compound-fault diagnostic and independent reference confirmation."""
from __future__ import annotations
import argparse, hashlib, itertools, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtr, ndtri
from scipy.stats import t as student_t
from . import rq3_uncertainty_a0 as a0
from .rq3_shared_mode_fault import joint_filter
from .sensor_reliability import score_sensors


def dump(p: Path, obj: object) -> None:
    """Write finite JSON."""
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def digest(p: Path) -> str:
    """Hash exact bytes."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def schedule(r: np.ndarray, seed: int, cfg: dict) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Balanced sign/offset schedule independent of values and score outcomes."""
    n=len(r); cells=list(itertools.product([-1,1],[-1,1],cfg['offsets']))
    if n%len(cells):raise ValueError('batch count must balance eight sign/offset cells')
    cells=np.array(cells*(n//len(cells)))
    rng=np.random.default_rng(np.random.SeedSequence([cfg['schedule_seed'],seed]))
    cells=cells[rng.permutation(n)]
    dep=np.array([np.flatnonzero(z[1:]!=z[:-1])[0]+1 for z in r])
    starts=dep+cells[:,2]
    if (starts<0).any() or (starts+cfg['duration']>r.shape[1]).any():raise ValueError('event outside observation')
    return starts,cells[:,:2],cells[:,2]


def intervene(x: np.ndarray,r: np.ndarray,scales: np.ndarray,channels: list[int],magnitude: float,starts: np.ndarray,signs: np.ndarray,duration: int) -> tuple[np.ndarray,np.ndarray]:
    """Apply fixed biases only within the supplied event mask; never alter source."""
    obs=x.copy();mask=np.zeros_like(x,dtype=bool)
    for i,s in enumerate(starts):
        for k,j in enumerate(channels):
            obs[i,s:s+duration,j]+=magnitude*scales[r[i,s],j]*signs[i,k]
            mask[i,s:s+duration,j]=True
    return obs,mask


def score(x: np.ndarray,r: np.ndarray,model: dict,candidate: dict,cfg: dict,arm: str) -> tuple[np.ndarray,np.ndarray]:
    """Causal residual and posterior using visible inputs only."""
    if arm=='joint12':q=joint_filter(x,r,model,candidate)['mode']
    elif arm=='soft3':q=a0.mode_filter(x,r,model,candidate)
    elif arm=='hard':q=np.eye(3)[r]
    elif arm=='factorial24':
        from .rq2_factorial_fault import factorial_filter
        q=factorial_filter(x,r,model,candidate)['mode']
    else:raise ValueError(arm)
    v=(x-model['center'])/model['scale'];mu=np.asarray(model['mu']);sd=np.sqrt(np.diagonal(model['cov'],axis1=1,axis2=2))
    z=(v[:,:,:,None]-mu.T)/sd.T
    u=(ndtr(z)*q[:,:,None,:]).sum(-1)
    raw=np.abs(a0.smooth(ndtri(np.clip(u,cfg['cdf_clip'],1-cfg['cdf_clip'])),r,cfg['alpha']))
    return raw,q


def metric_rows(alarm: np.ndarray,clean_alarm: np.ndarray,q: np.ndarray,clean_q: np.ndarray,mask: np.ndarray,truth: np.ndarray,starts: np.ndarray,signs: np.ndarray,offsets: np.ndarray,duration: int) -> list[dict]:
    """Batch-level sufficient counts; both-channel detection is not top-1."""
    rows=[]
    for i,s in enumerate(starts):
        active=mask[i].any(0);window=np.zeros(len(truth[i]),bool);window[s:s+duration]=True
        normal=~mask[i];bystander=window[:,None]&~active[None,:]
        inevent=mask[i];added=alarm[i]&~clean_alarm[i]&inevent
        detected=added.any(0)[active];eventall=alarm[i].any(0)[active]
        trueprob=q[i,np.arange(len(truth[i])),truth[i]]
        cleangain=clean_q[i,np.arange(len(truth[i])),truth[i]]-trueprob
        for regime in [-1,0,1,2]:
            valid=np.ones(len(window),bool) if regime==-1 else truth[i]==regime
            support=valid[:,None];fault=mask[i]&support;norm=normal&support;by=bystander&support
            row={'batch':i,'regime':regime,'offset':int(offsets[i]),'sign_pattern':str(tuple(int(x) for x in signs[i])),
                 'tp':int((alarm[i]&fault).sum()),'fault_cells':int(fault.sum()),'clean_fp':int((alarm[i]&norm).sum()),'clean_cells':int(norm.sum()),
                 'bystander_fp':int((alarm[i]&by).sum()),'bystander_cells':int(by.sum()),
                 'mode_errors':int(((q[i].argmax(-1)!=truth[i])&valid).sum()),'time_cells':int(valid.sum()),
                 'mode_brier_sum':float(((q[i]-np.eye(3)[truth[i]])**2).sum(-1)[valid].sum()),
                 'true_p_loss_sum':float(cleangain[valid].sum()),
                 'posterior_tv_sum':float((.5*np.abs(q[i]-clean_q[i]).sum(-1))[valid].sum()),
                 'event_channels':int(active.sum()) if regime==-1 else 0,'event_added_channels':int(detected.sum()) if regime==-1 else 0,
                 'fault_batches':int(active.any()) if regime==-1 else 0,'all_channels_added':int(active.any() and detected.all()) if regime==-1 else 0,
                 'exact_set_added':int(active.any() and np.array_equal(added.any(0),active) and not alarm[i,s:s+duration,~active].any()) if regime==-1 else 0}
            for j in range(3):
                row[f'tp_{j}']=int((alarm[i,:,j]&fault[:,j]).sum());row[f'fault_{j}']=int(fault[:,j].sum())
                row[f'normal_fp_{j}']=int((alarm[i,:,j]&norm[:,j]).sum());row[f'normal_{j}']=int(norm[:,j].sum())
            rows.append(row)
    return rows


RATIOS={'point_recall':('tp','fault_cells'),'clean_fpr':('clean_fp','clean_cells'),'bystander_fpr':('bystander_fp','bystander_cells'),'mode_error':('mode_errors','time_cells'),'mode_brier':('mode_brier_sum','time_cells'),'true_p_loss':('true_p_loss_sum','time_cells'),'posterior_tv':('posterior_tv_sum','time_cells'),'event_added_recall':('event_added_channels','event_channels'),'all_channels_added_recall':('all_channels_added','fault_batches'),'exact_set_event_recall':('exact_set_added','fault_batches')}


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counts within each seed, never treat views as independent seeds."""
    tables=[]
    base=['seed','view','lag','channels','magnitude','arm','regime']
    counts=list(dict.fromkeys(v for pair in RATIOS.values() for v in pair))
    for stratum,extra in [('all',[]),('offset',['offset']),('sign',['sign_pattern'])]:
        df=frame.groupby(base+extra,dropna=False)[counts].sum().reset_index()
        df['stratum']=stratum
        for name,(num,den) in RATIOS.items():df[name]=df[num]/df[den].replace(0,np.nan)
        tables.append(df)
    return pd.concat(tables,ignore_index=True)


def finish(out: Path,started: float,identity: dict) -> None:
    """Save run identity and immutable output hashes."""
    identity.update(elapsed_seconds=time.perf_counter()-started,python=platform.python_version(),numpy=np.__version__)
    dump(out/'identity.json',identity)
    dump(out/'manifest.json',{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()})


def diagnostic(args: argparse.Namespace,cfg: dict) -> None:
    """Run all predefined stress views on existing development observations."""
    started=time.perf_counter();out=args.output;out.mkdir(parents=True,exist_ok=False);dump(out/'config.json',cfg)
    settings=cfg['rq2'];arms=settings['arms']+(['factorial24'] if args.candidate else [])
    start=settings['validation_start'] if args.candidate else settings['development_start'];count=settings['batches_per_seed']
    allrows=[];sources={};calibration=[]
    for seed in cfg['seeds']:
        sd=out/f'seed_{seed}';sd.mkdir()
        def read_json(p: Path) -> dict:
            sources[str(p)]=digest(p);return json.loads(p.read_text())
        model=read_json(args.scorers/f'seed_{seed}/score_model.json');candidate=read_json(args.scorers/f'seed_{seed}/candidate.json')
        scales=np.asarray(read_json(args.scorers/f'seed_{seed}/fault_scales.json'))
        ip=args.data/f'seed_{seed}/val/inputs.npz';cp=args.data/f'seed_{seed}/reference_cal/inputs.npz'
        sources[str(ip)]=digest(ip);sources[str(cp)]=digest(cp);inputs=a0.npz(ip);cal=a0.npz(cp)
        x=inputs['observations'][start:start+count];truth=inputs['recorded_regime'][start:start+count];ids=inputs['sequence_id'][start:start+count]
        starts,signs,offsets=schedule(truth,seed,settings);thresholds={}
        for arm in arms:
            vals=[score(cal['observations'],a0.delayed(cal['recorded_regime'],lag),model['modes'],candidate,settings,arm)[0] for lag in settings['lags']]
            tau=np.quantile(np.concatenate(vals).reshape(-1,3),settings['calibration_quantile'],axis=0)
            if arm in ['joint12','soft3']:
                frozen=model['thresholds']['joint_full' if arm=='joint12' else 'no_fault_hypothesis']
                np.testing.assert_allclose(tau,frozen,rtol=0,atol=1e-11);tau=np.asarray(frozen)
            thresholds[arm]=tau
            calibration.append({'seed':seed,'arm':arm,'thresholds':tau.tolist()})
        dump(sd/'parameters.json',{'model':model,'candidate':candidate,'thresholds':{a:t.tolist() for a,t in thresholds.items()},'scales':scales.tolist()})
        viewmeta=[];xs=[];rs=[];masks=[];alarms=[];qs=[];ratios=[];clean_cache={};rows=[]
        for lag in settings['lags']:
            cases=[([],0)]+[(ch,mag) for mag in settings['magnitudes'] for ch in settings['channel_sets']]
            for ch,mag in cases:
                view=f'lag{lag}_channels'+(''.join(map(str,ch)) if ch else 'none')+f'_bias{mag}'
                obs,mask=intervene(x,truth,scales,ch,mag,starts,signs,settings['duration']);records=a0.delayed(truth,lag)
                viewmeta.append({'view':view,'lag':lag,'channels':''.join(map(str,ch)) if ch else 'none','magnitude':mag})
                va=[];vq=[];vr=[]
                for arm in arms:
                    raw,q=score(obs,records,model['modes'],candidate,settings,arm);ratio=raw/thresholds[arm];alarm=ratio>1
                    if not ch:clean_cache[lag,arm]=(alarm,q)
                    ca,cq=clean_cache[lag,arm]
                    rr=metric_rows(alarm,ca,q,cq,mask,truth,starts,signs,offsets,settings['duration'])
                    for row in rr:row.update(seed=seed,arm=arm,**viewmeta[-1])
                    rows.extend(rr);va.append(alarm);vq.append(q);vr.append(ratio)
                xs.append(obs);rs.append(records);masks.append(mask);alarms.append(va);qs.append(vq);ratios.append(vr)
        np.savez_compressed(sd/'inputs_and_scores.npz',observations=np.array(xs),records=np.array(rs),fault_mask=np.array(masks),alarm=np.array(alarms),q=np.array(qs),ratio=np.array(ratios),truth=truth,clean=x,sequence_id=ids,starts=starts,signs=signs,offsets=offsets,arms=arms,views=[v['view'] for v in viewmeta])
        dump(sd/'views.json',viewmeta);pd.DataFrame(rows).to_csv(sd/'batch_metrics.csv',index=False);allrows.extend(rows)
        print(json.dumps({'seed':seed,'stage':args.command,'views':len(viewmeta),'arms':arms}),flush=True)
    df=pd.DataFrame(allrows);aggregate(df).to_csv(out/'metrics_by_seed.csv',index=False);dump(out/'calibration.json',calibration)
    for p,h in sources.items():assert digest(Path(p))==h
    dump(out/'source_hashes.json',sources)
    finish(out,started,{'implementation_commit':args.commit,'source_commit':cfg['source_commit'],'scope':'development validation; existing val batches','batch_slice':[start,start+count],'base_batches':count*len(cfg['seeds']),'prediction_fits':0,'candidate':bool(args.candidate)})


def fresh_fixed(config: dict,seed: int,namespace: int,n: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Original E1b Gaussian-factor equations in a distinct evaluation-only stream."""
    loading=np.asarray(config['factor_loadings']);rho=config['ar_rho'];length=config['sequence_length'];xs=[];labels=[];ids=[]
    for j,(name,profile) in enumerate(config['regimes'].items()):
        for i in range(n):
            rng=np.random.default_rng(np.random.SeedSequence([seed,namespace,j,i]));latent=np.empty(length);latent[0]=rng.normal()
            for t in range(1,length):latent[t]=rho*latent[t-1]+np.sqrt(1-rho*rho)*rng.normal()
            z=latent[:,None]*loading+rng.normal(size=(length,3))*np.sqrt(1-loading**2)
            xs.append(np.asarray(profile['mean'])+np.asarray(profile['std'])*z);labels.append(name);ids.append(f'rq1_{namespace}_{seed}_{name}_{i}')
    return np.array(xs),np.array(labels),np.array(ids)


def rq1(args: argparse.Namespace,cfg: dict) -> None:
    """One frozen-reference independent confirmation; never refit or tune."""
    started=time.perf_counter();out=args.output;out.mkdir(parents=True,exist_ok=False);settings=cfg['rq1'];base=yaml.safe_load(args.base_config.read_text());dump(out/'config.json',cfg)
    rows=[];sources={str(args.base_config):digest(args.base_config)}
    for seed in cfg['seeds']:
        sd=out/f'seed_{seed}';sd.mkdir();p=args.references/f'seed_{seed}/references.json';sources[str(p)]=digest(p);refs=json.loads(p.read_text())
        clean,regimes,ids=fresh_fixed(base,seed,settings['namespace'],settings['sequences_per_regime']);n,t,_=clean.shape;rl=np.repeat(regimes,t)
        vals=[];masks=[];alarm_views=[];ratio_views=[]
        for mag in [0]+settings['magnitudes']:
            obs=clean.copy();mask=np.zeros_like(obs,dtype=bool)
            for i,label in enumerate(regimes):
                if not mag:continue
                j=i%3;sign=1 if (i//3)%2 else -1;s=settings['start'];end=s+settings['duration']
                obs[i,s:end,j]+=sign*mag*base['regimes'][label]['std'][j];mask[i,s:end,j]=True
            va=[];vr=[]
            for method in settings['methods']:
                scored=score_sensors(obs.reshape(-1,3),rl,refs[method]);alarm=scored['alarm'].reshape(n,t,3);va.append(alarm);vr.append(scored['deviation_ratio'].reshape(n,t,3))
                for regime in ['all',*base['regimes']]:
                    take=np.ones(n,bool) if regime=='all' else regimes==regime
                    for channel in [-1,0,1,2]:
                        a=alarm[take] if channel==-1 else alarm[take,:,channel:channel+1];m=mask[take] if channel==-1 else mask[take,:,channel:channel+1]
                        tp=int((a&m).sum());fp=int((a&~m).sum());pos=int(m.sum());neg=int((~m).sum())
                        rows.append({'seed':seed,'method':method,'magnitude':mag,'regime':regime,'channel':channel,'tp':tp,'fp':fp,'fault_cells':pos,'clean_cells':neg,'recall':tp/pos if pos else np.nan,'clean_fpr':fp/neg,'f1':2*tp/(2*tp+fp+pos-tp) if 2*tp+fp+pos-tp else np.nan})
            vals.append(obs);masks.append(mask);alarm_views.append(va);ratio_views.append(vr)
        np.savez_compressed(sd/'inputs_and_scores.npz',observations=np.array(vals),fault_mask=np.array(masks),alarm=np.array(alarm_views),ratio=np.array(ratio_views),regimes=regimes,sequence_id=ids,magnitudes=[0]+settings['magnitudes'],methods=settings['methods'])
        dump(sd/'references.json',refs)
    df=pd.DataFrame(rows);df.to_csv(out/'metrics_by_seed.csv',index=False);primary=[]
    for name,left,right,mag,metric in [('coverage','pooled_marginal','single_marginal',0,'clean_fpr'),('conditioning','regime_marginal','pooled_marginal',2,'recall')]:
        v=df[(df.regime=='all')&(df.channel==-1)&(df.magnitude==mag)].pivot(index='seed',columns='method',values=metric);d=(v[left]-v[right]).to_numpy();mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/np.sqrt(len(d));half=float(student_t.ppf(.975,len(d)-1)*se);p=float(2*student_t.sf(abs(mean/se),len(d)-1)) if se else 0.
        primary.append({'name':name,'contrast':left+' minus '+right,'metric':metric,'magnitude':mag,'differences':d.tolist(),'mean':mean,'sd':sd,'ci95':[mean-half,mean+half],'p_two_sided':p,'n':len(d)})
    order=np.argsort([r['p_two_sided'] for r in primary]);prev=0.
    for k,i in enumerate(order):prev=max(prev,min(1.,(len(primary)-k)*primary[i]['p_two_sided']));primary[i]['p_holm']=prev
    dump(out/'primary.json',primary);dump(out/'source_hashes.json',sources)
    for p,h in sources.items():assert digest(Path(p))==h
    finish(out,started,{'implementation_commit':args.commit,'source_commit':cfg['source_commit'],'scope':'independent same-family synthetic confirmation; frozen references','namespace':settings['namespace'],'base_sequences':len(cfg['seeds'])*3*settings['sequences_per_regime'],'prediction_fits':0})
    print(json.dumps(primary),flush=True)


def main() -> None:
    """Run one explicitly selected bounded stage."""
    p=argparse.ArgumentParser();p.add_argument('command',choices=['diagnostic','rq1']);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--data',type=Path);p.add_argument('--scorers',type=Path);p.add_argument('--references',type=Path,default=Path('sources'));p.add_argument('--base-config',type=Path,default=Path('configs/logistics_e1b.yaml'));p.add_argument('--candidate',action='store_true');args=p.parse_args();cfg=json.loads(args.config.read_text())
    (diagnostic if args.command=='diagnostic' else rq1)(args,cfg)

if __name__=='__main__':main()
