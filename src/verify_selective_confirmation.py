"""Independent arithmetic and provenance verification of confirmation artifacts."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import t

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def table(p):return list(csv.DictReader(p.open()))
def close(a,b):np.testing.assert_allclose(float(a),b,rtol=1e-11,atol=1e-12)

def verify(root):
    cfg=read(root/'config.json');count=updates=0;seconds=0.;olds=0
    for seed in cfg['seeds']:
        sd=root/'training'/f'seed_{seed}';ed=root/'evaluation'/f'seed_{seed}'
        for folder in [sd,ed]:
            for rel,h in read(folder/'manifest.json').items():assert digest(folder/rel)==h,(folder,rel)
        for p,h in read(sd/'source_hashes.json').items():assert digest(Path(p))==h,p
        fair=read(sd/'fairness.json');assert len(fair)==13
        for rep in [0,1]:
            signatures={tuple(r[k] for k in ['initial_sha256','shuffle_sha256','schedule_sha256','optimizer_updates','parameter_count']) for r in fair if r['repeat']==rep};assert len(signatures)==1
        assert fair[0]['initial_sha256']!=next(r for r in fair if r['repeat']==1)['initial_sha256']
        for f in fair:
            md=sd/f['key'];p=read(md/'parameters.json');hist=table(md/'history.csv');model=npz(md/'model.npz')
            assert sum(v.size for v in model.values())==6689==p['parameter_count'];assert len(hist)==40 and p['optimizer_updates']==320
            best=min(range(len(hist)),key=lambda i:float(hist[i]['val_suite_mae']));assert best+1==p['selected_epoch'];close(p['val_mae'],float(hist[best]['val_suite_mae']))
            val=npz(md/'validation.npz');close(p['val_mae'],np.abs(val['pred']-val['y']).mean());count+=1;updates+=320;seconds+=p['elapsed_seconds'];olds+=f['old_prediction_exact']
        data=npz(ed/'inputs.npz');truth=npz(ed/'evaluation_only.npz');meta=npz(ed/'evaluation_metadata.npz');pred=npz(ed/'predictions.npz');gc=read(ed/'generator_config.json')
        np.testing.assert_array_equal(data['sequence_id'],truth['sequence_id']);np.testing.assert_array_equal(data['sequence_id'],pred['sequence_id']);assert len(data['sequence_id'])==500 and pred['pred'].shape==(13,8,500)
        history=npz(sd/'schedule_and_random.npz');assert not set(data['sequence_id'])&set(np.r_[history['train_ids'],history['val_ids']])
        np.testing.assert_array_equal(data['recorded_regime'],truth['true_regime'][:,:118]);assert data['available_time'].max()<=119
        proxy=gc['proxy'];noiseless=proxy['intercept']+proxy['coefficient']*np.maximum(truth['clean_process'][:,:,0]-proxy['threshold'],0).sum(1)
        np.testing.assert_allclose(noiseless+truth['label_noise'].sum(1),truth['targets'][:,0],rtol=0,atol=1e-13)
        for i in range(500):
            rng=lambda stream:np.random.default_rng(np.random.SeedSequence([cfg['test_namespace'],seed,4,i,stream]))
            assert truth['targets'][i,1]==rng(6).normal(proxy['null_mean'],proxy['null_std'])
            noise=rng(3).normal(size=(180,3))*gc['measurement_noise_std'];np.testing.assert_array_equal(data['observations'][i],(truth['clean_process'][i]+noise)[:118])
        for event in read(ed/'events.json'):
            i=list(data['sequence_id']).index(event['sequence_id']);departure=np.flatnonzero(data['recorded_regime'][i,1:]!=data['recorded_regime'][i,:-1])[0]+1
            assert event['start']==departure+event['offset'] and event['end']-event['start']==12
    metrics=table(root/'metrics_by_repeat.csv');lookup={};cache={}
    for seed in cfg['seeds']:
        ed=root/'evaluation'/f'seed_{seed}';cache[seed]=(npz(ed/'predictions.npz'),npz(ed/'evaluation_only.npz'),npz(ed/'evaluation_metadata.npz'))
    for row in metrics:
        seed=int(row['seed']);rep=int(row['repeat']);target=row['target'];p,y,m=cache[seed];view=row['view'];group=row['stratum']
        mask=np.ones(500,bool) if group=='all' else m['channel' if group.startswith('channel') else 'offset' if group.startswith('offset') else 'end_regime']==int(group.split('_')[-1])
        vi=list(range(6)) if view=='suite' else [list(p['views']).index(view)];yi=list(y['target_names']).index(target)
        if row['arm']=='train_mean':a=np.full((8,500),read(root/'training'/f'seed_{seed}'/f'{target}_opt{rep}_N/parameters.json')['target_center'])
        else:a=p['pred'][list(p['keys']).index(f"{target}_opt{rep}_{row['arm']}")]
        d=(a-y['targets'][:,yi])[vi][:,mask];assert int(row['n_batches'])==mask.sum();close(row['mae'],abs(d).mean());close(row['rmse'],np.sqrt((d*d).mean()))
        lookup[(seed,target,rep,row['arm'],view,group)]=float(row['mae'])
    pairs=table(root/'paired_by_repeat.csv');pl={}
    for r in pairs:
        s=int(r['seed']);o=int(r['repeat']);a,b=r['comparison'].split('-');key=(s,r['target'],o);v=r['view'];g=r['stratum']
        d=lookup[key+(a,v,g)]-lookup[key+(b,v,g)];close(r['mae_difference'],d);pl[(s,r['target'],o,r['comparison'],v,g)]=d
        if v!='suite':cv=v.split('_')[0]+'_clean';close(r['degradation_difference'],d-(lookup[key+(a,cv,g)]-lookup[key+(b,cv,g)]))
    collapsed=table(root/'paired_by_seed.csv');cl={}
    for r in collapsed:
        key=(int(r['seed']),r['target'],r['comparison'],r['view'],r['stratum']);reps=[0,1] if r['target']==cfg['target'] else [0]
        value=np.mean([pl[key[:2]+(o,)+key[2:]] for o in reps]);close(r['mae_difference'],value);assert int(r['optimization_repeats'])==len(reps);cl[key]=value
    for name in ['paired_summary.csv','optimizer_sensitivity.csv']:
        for r in table(root/name):
            key=(r['target'],r['comparison'],r['view'],r['stratum'])
            v=np.array([cl[(s,)+key] if name=='paired_summary.csv' else pl[(s,key[0],int(r['repeat']))+key[1:]] for s in cfg['seeds']]);sd=v.std(ddof=1);half=t.ppf(.975,4)*sd/np.sqrt(5)
            close(r['mean'],v.mean());close(r['sd'],sd);close(r['ci_low'],v.mean()-half);close(r['ci_high'],v.mean()+half);assert int(r['negative_seeds'])==sum(v<0);assert int(r['outer_seeds'])==5
            close(r['loo_min'],min(np.delete(v,i).mean() for i in range(5)));close(r['loo_max'],max(np.delete(v,i).mean() for i in range(5)))
    assert count==65 and updates==20800 and olds==15
    result=dict(passed=True,models=count,main_models=50,null_models=15,updates=updates,epochs=2600,training_seconds=seconds,old_development_predictions_exact=olds,new_test_batches=2500,metric_rows=len(metrics),paired_repeat_rows=len(pairs),paired_seed_rows=len(collapsed),summary_rows=len(table(root/'paired_summary.csv')),sources_unchanged=True,outer_seeds=5,main_optimizer_repeats=2)
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);verify(p.parse_args().root)
