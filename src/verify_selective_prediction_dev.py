"""Independent result arithmetic and matched training budget verification."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def verify(root:Path,prior:Path)->dict:
    """Check every reported seed/scenario/stratum from stored predictions."""
    cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    for p,h in manifest.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h
    cache={};models=0;seconds=0
    read=lambda p:list(csv.DictReader(p.open()))
    for seed in cfg['seeds']:
        sd=root/f'seed_{seed}';z=dict(np.load(root/f'seed_{seed}_combined.npz'));meta=dict(np.load(sd/'evaluation_metadata.npz'));old=dict(np.load(prior/f'seed_{seed}/predictions.npz'));cache[seed]=(z,meta)
        np.testing.assert_array_equal(z['pred'][:7],old['pred']);np.testing.assert_array_equal(z['y'],old['y']);np.testing.assert_array_equal(z['sequence_id'],old['sequence_id'])
        baseline=json.loads((prior/f'seed_{seed}/N_parameters.json').read_text())
        for p,h in json.loads((sd/'source_hashes.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        for arm in cfg['arms']:
            par=json.loads((sd/f'{arm}_parameters.json').read_text());hist=read(sd/f'{arm}_history.csv');weights=dict(np.load(sd/f'{arm}_model.npz'));pp=dict(np.load(sd/f'{arm}_predictions.npz'))['pred'];np.testing.assert_array_equal(pp,z['pred'][list(z['arms']).index(arm)])
            for k in ['initial_sha256','shuffle_sha256','schedule_sha256','optimizer_updates','parameter_count','target_center','target_scale']:assert par[k]==baseline[k]
            assert len(hist)==40 and sum(w.size for w in weights.values())==6689 and par['optimizer_updates']==320
            best=min(range(40),key=lambda i:float(hist[i]['val_suite_mae']));assert par['selected_epoch']==best+1;assert abs(np.abs(pp-z['y']).mean()-par['val_mae'])<1e-12;models+=1;seconds+=par['elapsed_seconds']
    def mask_for(m:dict,name:str,n:int)->np.ndarray:
        if name=='all':return np.ones(n,bool)
        for prefix,key in [('channel_','channel'),('offset_','offset'),('end_regime_','end_regime')]:
            if name.startswith(prefix):return m[key]==int(name[len(prefix):])
        raise ValueError(name)
    rows=read(root/'metrics_by_seed.csv');pairs=read(root/'paired_by_seed.csv');agg=read(root/'paired_summary.csv')
    for r in rows:
        z,m=cache[int(r['seed'])];mask=mask_for(m,r['stratum'],len(z['y']));e=z['pred'][list(z['arms']).index(r['arm']),list(z['views']).index(r['view']),mask]-z['y'][mask]
        assert mask.sum()==int(r['n']);assert abs(np.abs(e).mean()-float(r['mae']))<1e-12;assert abs(np.sqrt(np.square(e).mean())-float(r['rmse']))<1e-12
    for r in pairs:
        z,m=cache[int(r['seed'])];a,b=r['comparison'].split('-');e=np.abs(z['pred']-z['y']);d=e[list(z['arms']).index(a)]-e[list(z['arms']).index(b)]
        if r['view']=='suite':value=d.mean()
        else:
            vi=list(z['views']).index(r['view']);ci=list(z['views']).index(r['view'].split('_')[0]+'_clean');mask=mask_for(m,r['stratum'],len(z['y']));value=d[vi,mask].mean();assert abs((d[vi,mask]-d[ci,mask]).mean()-float(r['degradation_difference']))<1e-12
        assert abs(value-float(r['mae_difference']))<1e-12
    for r in agg:
        vs=np.array([float(p['mae_difference']) for p in pairs if all(p[k]==r[k] for k in ['comparison','view','stratum'])]);assert len(vs)==3;assert abs(vs.mean()-float(r['mean']))<1e-12;assert abs(vs.std(ddof=1)-float(r['sd']))<1e-12;assert (vs<0).sum()==int(r['negative_seeds'])
    return dict(passed=True,new_models=models,updates=models*320,epochs=models*40,training_seconds=seconds,metric_rows=len(rows),paired_rows=len(pairs),paired_summary_rows=len(agg),manifest_files=len(manifest),old_predictions_exact=True,matched_initialization_budget_schedule=True,source_hashes_unchanged=True,outer_data_seeds=3,optimizer_seeds=1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=verify(a.root,a.base_root)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
