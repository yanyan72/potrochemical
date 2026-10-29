"""Independent arithmetic, archived training budget, and source integrity checks."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def read(path:Path)->list:
    """Read a table without using the experiment's summarizer."""
    return list(csv.DictReader(path.open()))

def verify(root:Path)->dict:
    """Recompute every table entry directly from stored per-batch predictions."""
    cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    for p,h in manifest.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
    metrics=read(root/'metrics_by_seed.csv');pairs=read(root/'paired_by_seed.csv');summary=read(root/'paired_summary.csv');suites=read(root/'suite_by_seed.csv')
    cache={};models=0;history_rows=0;common_initial=set();total_seconds=0
    for seed in cfg['seeds']:
        sd=root/f'seed_{seed}';z=dict(np.load(sd/'predictions.npz',allow_pickle=False));meta=dict(np.load(sd/'evaluation_metadata.npz',allow_pickle=False));cache[seed]=(z,meta)
        for p,h in json.loads((sd/'source_hashes.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        ar=dict(np.load(sd/'schedule_and_random.npz',allow_pickle=False));assert not set(ar['train_ids'])&set(ar['val_ids']);assert np.all(ar['train_random'][:,1]!=np.arange(1000));assert ar['schedule'].shape==(40,1000)
        fairness=json.loads((sd/'fairness.json').read_text());assert len({tuple(f['shared_other_features_sha256']) for f in fairness})==1
        for ai,arm in enumerate(cfg['arms']):
            params=json.loads((sd/f'{arm}_parameters.json').read_text());hist=read(sd/f'{arm}_history.csv');weights=dict(np.load(sd/f'{arm}_model.npz',allow_pickle=False));pred=dict(np.load(sd/f'{arm}_predictions.npz',allow_pickle=False))['pred']
            np.testing.assert_array_equal(pred,z['pred'][ai]);assert sum(v.size for v in weights.values())==6689;assert params['optimizer_updates']==320;assert len(hist)==40
            best=min(range(40),key=lambda i:float(hist[i]['val_suite_mae']));assert params['selected_epoch']==best+1;assert abs(float(hist[best]['val_suite_mae'])-np.abs(pred-z['y']).mean())<1e-12
            assert params['model_seed']==cfg['model_seed'];assert params['shuffle_seed']==cfg['shuffle_seed'];common_initial.add(params['initial_sha256']);total_seconds+=params['elapsed_seconds'];models+=1;history_rows+=len(hist)
        assert len({f['shuffle_sha256'] for f in fairness})==1;assert len({f['schedule_sha256'] for f in fairness})==1
    assert len(common_initial)==1
    for r in metrics:
        z,m=cache[int(r['seed'])];mask=np.ones(len(z['y']),bool);stratum=r['stratum']
        if stratum.startswith('channel_'):mask=m['channel']==int(stratum[8:])
        elif stratum.startswith('offset_'):mask=m['offset']==int(stratum[7:])
        elif stratum.startswith('end_regime_'):mask=m['end_regime']==int(stratum[11:])
        e=z['pred'][cfg['arms'].index(r['arm']),cfg['views'].index(r['view']),mask]-z['y'][mask]
        assert int(r['n'])==mask.sum();assert abs(float(r['mae'])-np.abs(e).mean())<1e-12;assert abs(float(r['rmse'])-np.sqrt(np.square(e).mean()))<1e-12
    for r in suites:
        z,_=cache[int(r['seed'])];value=np.abs(z['pred'][cfg['arms'].index(r['arm'])]-z['y']).mean();assert abs(value-float(r['mae']))<1e-12
    for r in pairs:
        z,_=cache[int(r['seed'])];a,b=r['comparison'].split('-');e=np.abs(z['pred']-z['y']);d=e[cfg['arms'].index(a)]-e[cfg['arms'].index(b)];v=d.mean() if r['view']=='suite' else d[cfg['views'].index(r['view'])].mean();assert abs(v-float(r['mae_difference']))<1e-12
    for r in summary:
        v=np.array([float(p['mae_difference']) for p in pairs if p['comparison']==r['comparison'] and p['view']==r['view']]);assert len(v)==3
        assert abs(v.mean()-float(r['mean']))<1e-12;assert abs(v.std(ddof=1)-float(r['sd']))<1e-12;assert (v<0).sum()==int(r['negative_seeds'])
    return dict(passed=True,models=models,epochs=history_rows,updates=models*320,metrics=len(metrics),paired_rows=len(pairs),paired_summary_rows=len(summary),suite_rows=len(suites),manifest_files=len(manifest),common_initialization=True,common_budget=True,all_source_hashes_unchanged=True,total_training_seconds=total_seconds,outer_data_seeds=3,optimization_seeds=1,scope='development, no new test')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    v=verify(a.root);a.output.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
