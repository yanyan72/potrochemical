"""Independent fixed-reference arithmetic and stored event recount for W1-R."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri
import rq3_uncertainty_a0 as a0

def verify(root,source):
    manifest=json.loads((root/'manifest.json').read_text())
    for p,h in manifest.items():assert a0.digest(root/p)==h,p
    cfg=json.loads((source/'config.json').read_text());checked=0;max_error=0.
    primary=[]
    for seed in cfg['seeds']:
        model=json.loads((source/f'seed_{seed}/model.json').read_text());m=model['modes']
        mu=np.asarray(m['mu']);sd=np.sqrt(np.diagonal(np.asarray(m['cov']),axis1=1,axis2=2))
        for split in cfg['splits']:
            archive=a0.npz(root/f'seed_{seed}/{split}_scores.npz')
            metrics=list(csv.DictReader((root/f'seed_{seed}/{split}_batch_metrics.csv').open()))
            lookup={(r['view'],r['base_arm'],r['sequence_id']):r for r in metrics}
            events=json.loads((source/f'seed_{seed}/{split}_events.json').read_text())
            for vi,view in enumerate(cfg['views']):
                v=a0.npz(source/f'seed_{seed}/{split}_{view}.npz');rec,kind=view.split('_')
                z=((v['observations']-m['center'])/m['scale']-mu[v['true_regime_evaluation_only']])/sd[v['true_regime_evaluation_only']]
                z=ndtri(np.clip(ndtr(z),cfg['cdf_clip'],1-cfg['cdf_clip']))
                filt=np.empty_like(z);state=np.zeros((len(z),3))
                for t in range(z.shape[1]):
                    if t:state[v['recorded_regime'][:,t]!=v['recorded_regime'][:,t-1]]=0
                    state=.2*z[:,t]+.8*state;filt[:,t]=state
                clean_vi=cfg['views'].index(rec+'_clean')
                for ai,arm in enumerate(['soft_all','soft_loo']):
                    ratio=np.abs(filt)/model['thresholds'][arm];alarm=ratio>1
                    np.testing.assert_array_equal(alarm,archive['alarm'][vi,ai])
                    np.testing.assert_allclose(ratio,archive['ratio'][vi,ai],atol=1e-6,rtol=1e-7)
                    max_error=max(max_error,float(np.abs(ratio-archive['ratio'][vi,ai]).max()))
                    for i,e in enumerate(events):
                        r=lookup[(view,arm,str(archive['sequence_id'][i]))];mask=v['fault_mask'][i];al=alarm[i]
                        s,end,j=e['start'],e['end'],e['channel'];active=kind!='clean';other=[c for c in range(3) if c!=j]
                        counts=dict(clean_fp=int(al[~mask].sum()),clean_cells=int((~mask).sum()),tp=int(al[mask].sum()),fault_cells=int(mask.sum()),
                            bystander_fp=int(al[s:end][:,other].sum()) if active else 0,bystander_cells=(end-s)*2 if active else 0,
                            event_added=int(active and np.any(al[s:end,j]&~archive['alarm'][clean_vi,ai,i,s:end,j])))
                        for k,value in counts.items():assert int(r[k])==value,(seed,split,view,arm,i,k)
                        checked+=1
    d=json.loads((root/'decision.json').read_text());rows=d['primary_seed_rows'];gate=json.loads((root/'config.json').read_text())['w1_diagnostic_gate']
    before=np.mean([r['bystander_fpr_before'] for r in rows]);after=np.mean([r['bystander_fpr_after'] for r in rows]);rel=(before-after)/before
    passed=all(r['bystander_fpr_after']<r['bystander_fpr_before'] and r['event_added_after']-r['event_added_before']>=-gate['maximum_event_recall_loss'] for r in rows) and rel>=gate['minimum_relative_bystander_reduction']
    assert bool(passed)==d['continue_one_candidate_development'];assert abs(rel-d['relative_reduction'])<1e-12
    return dict(passed=True,manifest_files=len(manifest),batch_view_arm_records_recounted=checked,alarm_replay_exact=True,max_float32_ratio_error=max_error,
                decision_arithmetic_verified=True,candidate_real_data_performance_tested=False,quality_models_trained=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--a0-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=verify(a.root,a.a0_root);a0.dump(a.output,result);print(json.dumps(result))
