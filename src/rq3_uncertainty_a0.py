"""Frozen A0 development screen; never generates latent data or fits quality targets."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import platform
import sys
from pathlib import Path
import numpy as np
import scipy
from scipy.special import logsumexp, ndtr, ndtri

REGIMES = ['warehouse', 'transit_smooth', 'transit_rough']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_blob(path):
    b = Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

def fit_modes(reference_inputs, frozen, cfg):
    """Only reliable reference observations/records and frozen reference parameters."""
    r = reference_inputs['recorded_regime']
    counts = np.full((3,3), cfg['transition_pseudocount'])
    np.add.at(counts, (r[:,:-1].ravel(), r[:,1:].ravel()), 1)
    initial = np.bincount(r[:,0], minlength=3)+cfg['initial_pseudocount']
    ref = frozen['reference']; center=np.asarray(ref['scaler_center']); scale=np.asarray(ref['scaler_scale'])
    mu=np.array([ref['references'][z]['center'] for z in REGIMES])
    cov=np.array([ref['references'][z]['covariance'] for z in REGIMES])
    return dict(center=center.tolist(), scale=scale.tolist(), mu=mu.tolist(), cov=cov.tolist(),
                transition=(counts/counts.sum(axis=1,keepdims=True)).tolist(), initial=(initial/initial.sum()).tolist())

def mode_filter(x, records, model, cfg, exclude=None):
    """Forward filter accepts no event metadata, truth, labels, IDs or future values."""
    keep=[j for j in range(3) if j!=exclude]
    v=((x-np.asarray(model['center']))/np.asarray(model['scale']))[:,:,keep]
    mu=np.asarray(model['mu'])[:,keep]; cov=np.asarray(model['cov'])[:,keep][:,:,keep]
    inv=np.linalg.inv(cov); ld=np.linalg.slogdet(cov)[1]
    delta=v[:,:,None,:]-mu[None,None,:,:]
    ll=-.5*(np.einsum('ntki,kij,ntkj->ntk',delta,inv,delta)+ld+len(keep)*np.log(2*np.pi))
    n,t,_=x.shape; result=np.empty((n,t,3)); trans=np.asarray(model['transition'])
    q=np.broadcast_to(model['initial'],(n,3)).copy(); p=cfg['record_correct_likelihood']
    for k in range(t):
        prior=q if k==0 else q@trans
        rec=np.where(np.arange(3)[None,:]==records[:,k,None], p, (1-p)/2)
        logq=np.log(np.maximum(prior,1e-300))+np.log(rec)+ll[:,k]
        q=np.exp(logq-logsumexp(logq,axis=1,keepdims=True)); result[:,k]=q
    return result

def smooth(residual, records, alpha):
    out=np.empty_like(residual); state=np.zeros((len(residual),3))
    for t in range(residual.shape[1]):
        if t: state[records[:,t]!=records[:,t-1]]=0
        state=alpha*residual[:,t]+(1-alpha)*state; out[:,t]=state
    return out

def legacy_score(x, records, frozen):
    ref=frozen['reference']; v=(x-ref['scaler_center'])/ref['scaler_scale']; z=np.empty_like(x); th=np.empty_like(x)
    for k,name in enumerate(REGIMES):
        r=ref['references'][name]; take=records==k
        z[take]=(v[take]-r['center'])/np.sqrt(np.diag(r['covariance']))
        th[take]=frozen['thresholds'][name]
    filtered=smooth(z,records,frozen['alpha']); ratio=np.abs(filtered)/th
    return ratio, ratio>1

def matched_raw(x, records, model, cfg, arm, oracle_records=None):
    """Return absolute filtered residual and channel-specific mode probabilities."""
    if arm not in ('hard_matched','soft_all','soft_loo','oracle_matched'): raise ValueError(arm)
    n,t,p=x.shape
    if arm=='oracle_matched':
        if oracle_records is None: raise ValueError('oracle requires diagnostic records')
        q=np.repeat(np.eye(3)[oracle_records][:,:,None,:],3,axis=2)
    elif arm=='hard_matched':
        q=np.repeat(np.eye(3)[records][:,:,None,:],3,axis=2)
    elif arm=='soft_all':
        q=np.repeat(mode_filter(x,records,model,cfg)[:,:,None,:],3,axis=2)
    else:
        q=np.stack([mode_filter(x,records,model,cfg,j) for j in range(3)],axis=2)
    v=(x-np.asarray(model['center']))/np.asarray(model['scale'])
    mu=np.asarray(model['mu']); sd=np.sqrt(np.diagonal(np.asarray(model['cov']),axis1=1,axis2=2))
    z=(v[:,:,:,None]-mu.T[None,None,:,:])/sd.T[None,None,:,:]
    u=np.sum(q*ndtr(z),axis=-1)
    residual=ndtri(np.clip(u,cfg['cdf_clip'],1-cfg['cdf_clip']))
    return np.abs(smooth(residual,records,cfg['alpha'])), q

def delayed(r, lag):
    return r[:,np.maximum(np.arange(r.shape[1])-lag,0)].copy()

def events_for(ids, truth, cfg, seed, split):
    table=list(itertools.product(range(3),(-1,1),cfg['start_offsets']))*5
    if len(ids)!=len(table): raise ValueError('frozen balanced event count')
    rng=np.random.default_rng(np.random.SeedSequence([cfg['schedule_seed'],seed,0 if split=='train' else 1]))
    order=rng.permutation(len(table)); events=[]
    for i,sid in enumerate(ids):
        changes=np.flatnonzero(truth[i,1:]!=truth[i,:-1])+1
        if len(changes)==0: raise ValueError('missing departure')
        departure=int(changes[0]); ch,sign,offset=table[order[i]]; start=departure+offset
        if not 0<=start<start+cfg['duration']<=truth.shape[1]: raise ValueError('event boundary')
        events.append(dict(sequence_id=str(sid),channel=ch,sign=sign,departure=departure,
                           offset=offset,start=start,end=start+cfg['duration']))
    return events

def paired_view(x, original_records, events, scales, kind, lag, cfg):
    """Compose new observation intervention; no latent generator or labels involved."""
    out=x.copy(); mask=np.zeros_like(out,dtype=bool)
    if kind!='clean':
        for i,e in enumerate(events):
            j,s,end=e['channel'],e['start'],e['end']; magnitude=cfg['magnitudes'][kind]
            amplitude=magnitude*scales[original_records[i,s],j]*e['sign']
            ramp=np.arange(1,end-s+1)/(end-s) if kind=='drift1' else 1
            out[i,s:end,j]+=amplitude*ramp; mask[i,s:end,j]=True
    return out, delayed(original_records,lag), mask

def per_batch(alarm, clean_alarm, mask, q, r, true_r, events):
    rows=[]; n,t,p=alarm.shape
    for i,e in enumerate(events):
        ch,s,end=e['channel'],e['start'],e['end']; active=bool(mask[i].any())
        fresh=alarm[i,:,ch]&~np.r_[False,alarm[i,:-1,ch]]
        hits=np.flatnonzero(fresh[s:end]); added=alarm[i,s:end,ch]&~clean_alarm[i,s:end,ch]
        clean=~mask[i]; by=np.ones((end-s,p),bool);by[:,ch]=False
        window=np.zeros((t,p),bool);window[e['departure']:e['departure']+12]=True
        truth_onehot=np.eye(3)[true_r[i]][:,None,:]
        ent=-np.sum(q[i]*np.log(np.maximum(q[i],1e-300)),axis=-1)
        rwrong=r[i]!=true_r[i]
        qm=q[i].argmax(-1); correct=qm==true_r[i,:,None]
        recprob=np.take_along_axis(q[i],np.broadcast_to(r[i,:,None,None],(t,p,1)),axis=-1)[...,0]
        suspect=recprob<.5
        transition_error=(~correct)[window]
        row=dict(sequence_id=e['sequence_id'],channel=ch,offset=e['offset'],sign=e['sign'],has_fault=int(active),
                 clean_fp=int((alarm[i]&clean).sum()),clean_cells=int(clean.sum()),
                 clean_all_fpr=float(alarm[i].mean()) if not active else None,
                 tp=int((alarm[i]&mask[i]).sum()),fault_cells=int(mask[i].sum()),
                 event_any=int(active and alarm[i,s:end,ch].any()),event_fresh=int(active and len(hits)>0),
                 event_added=int(active and added.any()),event_pre_active=int(active and alarm[i,s-1,ch]),
                 delay_capped=int(hits[0]) if active and len(hits) else cfg_duration(e),
                 bystander_fp=int(alarm[i,s:end][by].sum()) if active else 0,
                 bystander_cells=int(by.sum()) if active else 0,
                 transition_fp=int((alarm[i]&window&clean).sum()),transition_clean_cells=int((window&clean).sum()),
                 mode_accuracy=float(correct.mean()),mode_brier=float(np.square(q[i]-truth_onehot).sum(-1).mean()),
                 mode_entropy=float(ent.mean()),mode_transition_error=float(transition_error.mean()),
                 wrong_record_detected=int((suspect&rwrong[:,None]).sum()),wrong_record_cells=int(rwrong.sum()*p),
                 correct_record_flagged=int((suspect&~rwrong[:,None]).sum()),correct_record_cells=int((~rwrong).sum()*p))
        rows.append(row)
    return rows

def cfg_duration(e): return e['end']-e['start']

def summarize_rows(rows):
    counts=['clean_fp','clean_cells','tp','fault_cells','bystander_fp','bystander_cells','transition_fp','transition_clean_cells','wrong_record_detected','wrong_record_cells','correct_record_flagged','correct_record_cells']
    s={k:sum(r[k] for r in rows) for k in counts}; s['batches']=len(rows)
    for name,a,b in [('clean_fpr','clean_fp','clean_cells'),('point_recall','tp','fault_cells'),('bystander_fpr','bystander_fp','bystander_cells'),('transition_clean_fpr','transition_fp','transition_clean_cells'),('wrong_record_recall','wrong_record_detected','wrong_record_cells'),('correct_record_fpr','correct_record_flagged','correct_record_cells')]:
        s[name]=s[a]/s[b] if s[b] else None
    faults=[r for r in rows if r['has_fault']]
    for k in ['event_any','event_fresh','event_added','event_pre_active','delay_capped']:
        s[k]=float(np.mean([r[k] for r in faults])) if faults else None
    for k in ['mode_accuracy','mode_brier','mode_entropy','mode_transition_error']:
        s[k]=float(np.mean([r[k] for r in rows]))
    return s

def decide(summary,cfg):
    lookup={(r['seed'],r['arm'],r['view']):r for r in summary if r['split']=='val' and r['stratum']=='all'}
    g=cfg['gate']; result={}
    for arm in ['soft_all','soft_loo']:
        dr=[]; base=[]; checks=[]
        for seed in cfg['seeds']:
            get=lambda a,v:lookup[(seed,a,v)]
            h=get('hard_matched','delay12_clean')['clean_fpr']; v=get(arm,'delay12_clean')['clean_fpr']
            dr.append(h-v);base.append(h)
            tests=dict(seed=seed,delay_fpr_lower=v<h,
                correct_fpr_ok=get(arm,'correct_clean')['clean_fpr']-get('hard_matched','correct_clean')['clean_fpr']<=g['max_correct_fpr_increase'])
            for view in ['correct_bias4','delay12_bias4']:
                tests[view+'_event_ok']=get(arm,view)['event_added']-get('hard_matched',view)['event_added']>=-g['max_bias_event_recall_loss']
                tests[view+'_bystander_ok']=get(arm,view)['bystander_fpr']-get('hard_matched',view)['bystander_fpr']<=g['max_bystander_increase']
            checks.append(tests)
        abs_gain=float(np.mean(dr));rel_gain=abs_gain/float(np.mean(base))
        passed=all(all(v for k,v in c.items() if k!='seed') for c in checks) and abs_gain>=g['delay_fpr_absolute_reduction'] and rel_gain>=g['delay_fpr_relative_reduction']
        result[arm]=dict(pass_screen=passed,absolute_fpr_reduction=abs_gain,relative_fpr_reduction=rel_gain,checks=checks)
    diffs=[lookup[(s,'soft_loo',v)]['event_added']-lookup[(s,'soft_all',v)]['event_added'] for s in cfg['seeds'] for v in ['correct_bias4','delay12_bias4']]
    result['loo_incremental']={'paired_event_differences':diffs,'mean':float(np.mean(diffs)), 'passes_incremental_screen': bool(result['soft_loo']['pass_screen'] and np.mean(diffs)>=g['loo_vs_all_event_gain'] and min(diffs)>=-g['max_bias_event_recall_loss'])}
    result['advance_A1']=any(result[a]['pass_screen'] for a in ['soft_all','soft_loo'])
    result['rq3_supported']=False;result['scope']='descriptive development screen; no quality predictor fitted'
    return result

def main_run(data, legacy_root, cfg, expected, out, commit):
    if out.exists(): raise FileExistsError(out)
    if digest(data/'manifest.json')!=cfg['dataset_manifest_sha256']: raise ValueError('manifest mismatch')
    manifest=json.loads((data/'manifest.json').read_text()); used={}
    def checked(path, remote):
        sha=git_blob(path)
        if sha!=expected.get(remote): raise ValueError('remote/source mismatch: '+remote)
        h=digest(path)
        if path.is_relative_to(data):
            rel=str(path.relative_to(data))
            if rel in manifest and h!=manifest[rel]: raise ValueError('source manifest mismatch')
        used[str(path)]=dict(remote_path=remote,sha256=h,git_blob=sha)
    prefix='data/processed/'+cfg['dataset_id']+'/'
    def check_data(rel):
        p=data/rel; checked(p,prefix+rel); return p
    for seed in cfg['seeds']:
        for split in ['reference_fit','reference_cal','train','val']:check_data(f'seed_{seed}/{split}/inputs.npz')
        for split in cfg['splits']:
            check_data(f'seed_{seed}/{split}/targets.csv');check_data(f'evaluation_only/seed_{seed}/{split}/truth.npz')
        check_data(f'seed_{seed}/fault_scales.json')
        rel=f'results/rq3_prediction/rq3_shared_fault_20261023_v1/seed_{seed}/training/frozen_reference.json'
        checked(legacy_root/rel,rel)
    checked(legacy_root/'src/temporal_reliability.py','src/temporal_reliability.py')
    spec=importlib.util.spec_from_file_location('legacy_temporal',legacy_root/'src/temporal_reliability.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    out.mkdir(parents=True);dump(out/'config.json',cfg);dump(out/'source_inputs.json',used)
    dump(out/'identity.json',dict(implementation_commit=commit,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),source='synthetic_development_reused',test_read=False))
    batchrows=[]; summaries=[]; checks=[]
    for seed in cfg['seeds']:
        sd=out/f'seed_{seed}';sd.mkdir()
        frozen=json.loads((legacy_root/f'results/rq3_prediction/rq3_shared_fault_20261023_v1/seed_{seed}/training/frozen_reference.json').read_text())['scoring']
        fit=npz(data/f'seed_{seed}/reference_fit/inputs.npz');cal=npz(data/f'seed_{seed}/reference_cal/inputs.npz')
        model=fit_modes(fit,frozen,cfg);thresholds={};calibration=[]
        for arm in cfg['arms'][1:]:
            vals=[]
            for lag in [0,cfg['delay']]:
                r=delayed(cal['recorded_regime'],lag)
                raw,_=matched_raw(cal['observations'],r,model,cfg,arm,cal['recorded_regime'] if arm=='oracle_matched' else None)
                vals.append(raw)
            threshold=np.quantile(np.concatenate(vals).reshape(-1,3),cfg['normal_quantile'],axis=0)
            thresholds[arm]=threshold.tolist()
            for lag,raw in zip([0,cfg['delay']],vals):calibration.append(dict(arm=arm,lag=lag,fpr=float((raw>threshold).mean())))
        dump(sd/'model.json',dict(modes=model,thresholds=thresholds,legacy=frozen,calibration=calibration))
        scales=np.asarray(json.loads((data/f'seed_{seed}/fault_scales.json').read_text())['scales'])
        for split in cfg['splits']:
            inputs=npz(data/f'seed_{seed}/{split}/inputs.npz');n=cfg['batches_per_split']
            x=inputs['observations'][:n].copy();r0=inputs['recorded_regime'][:n].copy();ids=inputs['sequence_id'][:n]
            truth=npz(data/f'evaluation_only/seed_{seed}/{split}/truth.npz');true_r=truth['true_regime'][:n,:x.shape[1]]
            assert np.array_equal(r0,true_r) and np.array_equal(ids,truth['sequence_id'][:n])
            with (data/f'seed_{seed}/{split}/targets.csv').open() as f: labels=list(csv.DictReader(f))[:n]
            assert [r['sequence_id'] for r in labels]==list(ids)
            assert np.max(inputs['available_time'])<=119
            events=events_for(ids,true_r,cfg,seed,split);dump(sd/f'{split}_events.json',events);csv_write(sd/f'{split}_shared_targets.csv',labels)
            all_clean={};pair_x={};allrows=[]
            for view in cfg['views']:
                rec,kind=view.split('_');lag=0 if rec=='correct' else cfg['delay']
                xv,rv,mask=paired_view(x,r0,events,scales,kind,lag,cfg)
                if rec=='correct':pair_x[kind]=xv.copy()
                else:assert np.array_equal(xv,pair_x[kind])
                archive={'sequence_id':ids,'recorded_regime':rv,'fault_mask':mask,'observations':xv,'true_regime_evaluation_only':true_r}
                for arm in cfg['arms']:
                    if arm=='legacy':
                        ratio,alarm=legacy_score(xv,rv,frozen);q=np.repeat(np.eye(3)[rv][:,:,None,:],3,axis=2)
                        refold=old.score_temporal(xv,np.asarray(REGIMES)[rv],frozen)
                        assert np.array_equal(alarm,refold['alarm']);assert np.allclose(ratio,refold['deviation_ratio'],rtol=0,atol=1e-12)
                    else:
                        raw,q=matched_raw(xv,rv,model,cfg,arm,true_r if arm=='oracle_matched' else None)
                        ratio=raw/thresholds[arm];alarm=ratio>1
                    if kind=='clean':all_clean[(rec,arm)]=alarm.copy()
                    rows=per_batch(alarm,all_clean[(rec,arm)],mask,q,rv,true_r,events)
                    for row in rows:row.update(seed=seed,split=split,view=view,arm=arm)
                    batchrows.extend(rows);allrows.extend(rows)
                    archive[arm+'_ratio']=ratio.astype(np.float32);archive[arm+'_alarm']=alarm
                    if arm.startswith('soft'):archive[arm+'_q']=q.astype(np.float32)
                    for name,selected in [('all',rows)]+[(f'channel_{j}',[a for a in rows if a['channel']==j]) for j in range(3)]+[(f'offset_{o}',[a for a in rows if a['offset']==o]) for o in cfg['start_offsets']]:
                        summaries.append(dict(seed=seed,split=split,view=view,arm=arm,stratum=name,**summarize_rows(selected)))
                np.savez_compressed(sd/f'{split}_{view}.npz',**archive)
            checks.append(dict(seed=seed,split=split,batches=n,labels_shared=True,paired_observations_equal=True,legacy_replay=True))
            csv_write(sd/f'{split}_batch_metrics.csv',allrows)
        print(json.dumps(dict(seed=seed,status='completed',views=12,arms=5)),flush=True)
    csv_write(out/'metrics_by_seed.csv',summaries)
    dump(out/'decision.json',decide(summaries,cfg));dump(out/'checks.json',checks)
    for path,e in used.items():
        if digest(path)!=e['sha256']:raise ValueError('source changed during run')
    dump(out/'source_unchanged.json',dict(files=len(used),unchanged=True))
    dump(out/'manifest.json',{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(decide(summaries,cfg),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--legacy-root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--source-git',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();main_run(a.data,a.legacy_root,json.loads(a.config.read_text()),json.loads(a.source_git.read_text()),a.output,a.commit)
