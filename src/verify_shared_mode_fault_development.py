"""Independent score arithmetic/counts; production filter replay is identified separately."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri
import rq3_uncertainty_a0 as a0
from rq3_shared_mode_fault import joint_filter

def read(path):
    rows=[]
    for r in csv.DictReader(path.open()):
        rows.append({k:(v if k in ['split','arm','view','stratum','sequence_id'] else None if v=='' else float(v)) for k,v in r.items()})
    return rows

def verify(root,source):
    manifest=json.loads((root/'manifest.json').read_text())
    for p,h in manifest.items():assert a0.digest(root/p)==h,p
    cfg=json.loads((root/'config.json').read_text());candidate=json.loads((root/'candidate_config.json').read_text())
    summaries=read(root/'metrics_by_seed.csv');base=read(source/'metrics_by_seed.csv')
    checked=0;max_error=0.;recount={}
    for seed in cfg['seeds']:
        model=json.loads((root/f'seed_{seed}/model.json').read_text());m=model['modes']
        mu=np.asarray(m['mu']);sd=np.sqrt(np.diagonal(np.asarray(m['cov']),axis1=1,axis2=2))
        for split in cfg['splits']:
            ar=a0.npz(root/f'seed_{seed}/{split}_scores.npz');rows=read(root/f'seed_{seed}/{split}_batch_metrics.csv')
            lookup={(r['view'],r['arm'],r['sequence_id']):r for r in rows}
            events=json.loads((source/f'seed_{seed}/{split}_events.json').read_text())
            for vi,view in enumerate(ar['views']):
                v=a0.npz(source/f'seed_{seed}/{split}_{view}.npz');rec,kind=view.split('_');clean_vi=list(ar['views']).index(rec+'_clean')
                for ai,arm in enumerate(ar['arms']):
                    kw={'kappa':0.} if arm=='no_fault_hypothesis' else {'fault_memory':False} if arm=='no_fault_memory' else {}
                    p=joint_filter(v['observations'],v['recorded_regime'],m,candidate,**kw)
                    q=p['mode']
                    if arm=='no_fault_hypothesis':
                        collapsed=a0.mode_filter(v['observations'],v['recorded_regime'],m,candidate)
                        np.testing.assert_allclose(q,collapsed,atol=1e-12,rtol=0);q=collapsed
                    np.testing.assert_allclose(q,ar['mode'][vi,ai],atol=6e-8,rtol=0)
                    np.testing.assert_allclose(p['fault'],ar['fault'][vi,ai],atol=6e-8,rtol=0)
                    if arm=='joint_full':np.testing.assert_allclose(p['joint'],ar['joint_full_posterior'][vi],atol=6e-8,rtol=0)
                    # Independent CDF and recurrence, not score_raw or a0.smooth.
                    normalized=(v['observations']-m['center'])/m['scale']
                    cdf=np.zeros_like(normalized)
                    for z in range(3):cdf+=q[:,:,z,None]*ndtr((normalized-mu[z])/sd[z])
                    residual=ndtri(np.clip(cdf,cfg['cdf_clip'],1-cfg['cdf_clip']))
                    state=np.zeros((len(residual),3));filtered=np.empty_like(residual)
                    for t in range(residual.shape[1]):
                        if t:state[v['recorded_regime'][:,t]!=v['recorded_regime'][:,t-1]]=0
                        state=cfg['alpha']*residual[:,t]+(1-cfg['alpha'])*state;filtered[:,t]=state
                    ratio=np.abs(filtered)/model['thresholds'][arm];alarm=ratio>1
                    np.testing.assert_allclose(ratio,ar['ratio'][vi,ai],atol=1e-6,rtol=1e-7)
                    np.testing.assert_array_equal(alarm,ar['alarm'][vi,ai]);max_error=max(max_error,float(np.abs(ratio-ar['ratio'][vi,ai]).max()))
                    if arm=='no_fault_hypothesis':np.testing.assert_array_equal(alarm,v['soft_all_alarm'])
                    independent=[]
                    for i,e in enumerate(events):
                        row=lookup[(view,arm,str(ar['sequence_id'][i]))];mask=v['fault_mask'][i];al=alarm[i];s,end,j=e['start'],e['end'],e['channel'];active=kind!='clean';other=[c for c in range(3) if c!=j]
                        counts=dict(clean_fp=int(al[~mask].sum()),clean_cells=int((~mask).sum()),tp=int(al[mask].sum()),fault_cells=int(mask.sum()),bystander_fp=int(al[s:end][:,other].sum()) if active else 0,bystander_cells=(end-s)*2 if active else 0,event_added=int(active and np.any(al[s:end,j]&~ar['alarm'][clean_vi,ai,i,s:end,j])))
                        for k,n in counts.items():assert row[k]==n,(seed,split,view,arm,i,k)
                        independent.append(dict(**counts,channel=j,offset=e['offset'],active=active));checked+=1
                    strata=[('all',independent)]+[(f'channel_{j}',[r for r in independent if r['channel']==j]) for j in range(3)]+[(f'offset_{o}',[r for r in independent if r['offset']==o]) for o in [-6,18]]
                    for stratum,selected in strata:
                        key=(seed,split,arm,view,stratum);stored=next(r for r in summaries if (r['seed'],r['split'],r['arm'],r['view'],r['stratum'])==key)
                        vals={}
                        for name,num,den in [('clean_fpr','clean_fp','clean_cells'),('point_recall','tp','fault_cells'),('bystander_fpr','bystander_fp','bystander_cells')]:
                            d=sum(r[den] for r in selected);vals[name]=sum(r[num] for r in selected)/d if d else None
                        faults=[r for r in selected if r['active']];vals['event_added']=np.mean([r['event_added'] for r in faults]) if faults else None
                        for k,n in vals.items():assert stored[k] is None if n is None else abs(stored[k]-n)<1e-12
                        recount[key]=vals
    def get(s,arm,view,stratum='all'):
        if arm in cfg['arms']:return recount[s,'val',arm,view,stratum]
        return next(r for r in base if r['seed']==s and r['split']=='val' and r['arm']==arm and r['view']==view and r['stratum']==stratum)
    g=cfg['gate'];conditions=[];gains=[]
    for s in cfg['seeds']:
        for v in ['correct_clean','delay12_clean']:
            f=get(s,'joint_full',v)['clean_fpr'];conditions += [f<=g['maximum_clean_fpr'],f-get(s,'soft_all',v)['clean_fpr']<=g['maximum_clean_fpr_increase_vs_soft_all']]
        for v in ['correct_bias4','delay12_bias4']:
            f=get(s,'joint_full',v);b=get(s,'soft_loo',v)
            conditions += [f['bystander_fpr']<b['bystander_fpr'],f['bystander_fpr']<=g['maximum_bystander_fpr'],f['event_added']-b['event_added']>=-g['maximum_event_loss_vs_loo']]
            for j in [0,1]:conditions.append(get(s,'joint_full',v,f'channel_{j}')['event_added']-get(s,'soft_all',v,f'channel_{j}')['event_added']>=-g['maximum_other_channel_event_loss_vs_soft_all'])
            gains.append(get(s,'joint_full',v,'channel_2')['event_added']-get(s,'soft_all',v,'channel_2')['event_added'])
    for v in ['correct_bias4','delay12_bias4']:
        b=np.mean([get(s,'soft_loo',v)['bystander_fpr'] for s in cfg['seeds']]);f=np.mean([get(s,'joint_full',v)['bystander_fpr'] for s in cfg['seeds']]);conditions.append((b-f)/b>=g['minimum_loo_bystander_relative_reduction'])
    conditions.append(np.mean(gains)>=g['minimum_vibration_event_gain_vs_soft_all'])
    d=json.loads((root/'decision.json').read_text());assert all(conditions)==d['pass_development_screen'];assert d['rq3_supported'] is False
    return dict(passed=True,manifest_files=len(manifest),batch_view_arm_records_recounted=checked,summary_strata_recounted=len(recount),alarm_replay_exact=True,max_float32_ratio_error=max_error,decision_independently_recounted=True,filter_replay_uses_production_core=True,independent_parts=['CDF and EWMA arithmetic','event and point counts','summary metrics','development decision'],no_fault_ablation_equals_old_soft_all=True,quality_models_trained=0,rq3_supported=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--a0-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=verify(a.root,a.a0_root);a0.dump(a.output,result);print(json.dumps(result))
