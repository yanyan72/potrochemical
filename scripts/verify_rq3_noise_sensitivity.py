"""Independently audit stored noise-study inputs and recompute all result tables."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import t


def digest(path: Path) -> str:
    """Hash a persisted artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path) -> dict[str,np.ndarray]:
    """Read arrays without pickle."""
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def rows(path: Path) -> list[dict[str,str]]:
    """Read CSV records."""
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def hashes(root: Path, manifest: dict[str,str]) -> None:
    """Check every declared file."""
    for name,h in manifest.items():
        if digest(root/name)!=h:
            raise AssertionError('hash mismatch '+str(root/name))


def match(actual: list[dict],expected: list[dict],keys:list[str],values:list[str]) -> None:
    """Verify keyed tables without relying on their row order."""
    def key(row:dict) -> tuple:
        return tuple(str(row[k]) for k in keys)
    lookup={key(r):r for r in actual}
    assert len(lookup)==len(actual)==len(expected), 'duplicate/missing rows'
    for row in expected:
        got=lookup[key(row)]
        for v in values:
            assert np.isclose(float(got[v]),float(row[v]),rtol=1e-10,atol=1e-10),(key(row),v,got[v],row[v])


def summaries(records:list[dict],keys:list[str],values:list[str],seeds:list[int]) -> list[dict]:
    """Recompute means, sample SD and t intervals from five paired units."""
    bins:dict[tuple,list[dict]]={}
    for r in records:
        bins.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for key,group in bins.items():
        assert sorted(int(r['seed']) for r in group)==sorted(seeds)
        out=dict(zip(keys,key));out['data_repeats']=len(group)
        for value in values:
            a=np.array([float(r[value]) for r in group]);mean=a.mean();sd=a.std(ddof=1)
            half=t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a))
            out.update({value+'_mean':mean,value+'_sd':sd,value+'_ci_low':mean-half,value+'_ci_high':mean+half})
        result.append(out)
    return result


def verify(run:Path,data:Path,ridge:Path,lstm:Path,final:Path) -> dict[str,Any]:
    """Check source identity, generation semantics, pairing and all numeric outputs."""
    cfg=json.loads((run/'config.json').read_text())
    dc=json.loads((run/'generator_base_config.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text());hashes(run,manifest)
    for path,pin in [(data,'data_manifest_sha256'),(ridge,'ridge_manifest_sha256'),
                     (lstm,'lstm_manifest_sha256'),(final,'source_final_manifest_sha256')]:
        assert digest(path/'manifest.json')==cfg[pin]
        hashes(path,json.loads((path/'manifest.json').read_text()))
    identity=json.loads((run/'identity.json').read_text())
    hashes(Path(__file__).resolve().parents[1],identity['source_sha256'])
    assert cfg['namespace']!=dc['namespace'] and cfg['noise_factors']==[.5,1.,2.]
    assert cfg['refit'] is False and cfg['inject_faults'] is False and cfg['confidence_level']==.95
    metrics,paired=[],[];ids=set();max_residual=0.
    for seed in cfg['seeds']:
        folder=run/f'seed_{seed}';meta=json.loads((folder/'metadata.json').read_text())
        assert meta['models_refitted']==0 and meta['source_files_unchanged']
        base=npz(folder/'evaluation_only.npz');p=npz(folder/'predictions.npz')
        views=[npz(folder/f'inputs_level_{i}.npz') for i in range(3)]
        y=base['targets'];n=cfg['batches_per_seed'];reg=base['true_regime'][:,:118]
        assert y.shape==(n,2) and base['clean_process'].shape==(n,180,3)
        expected_ids=[f"rq3_{cfg['namespace']}_{seed}_val_{i:06d}" for i in range(n)]
        assert list(base['sequence_id'])==expected_ids==list(p['sequence_id'])
        assert not ids.intersection(expected_ids);ids.update(expected_ids)
        # Independent RNG reconstruction: val is split index 3; streams 3..6.
        measurement=np.empty((n,118,3));noise=np.empty((n,2));null=np.empty(n)
        for i in range(n):
            def rng(stream:int) -> np.random.Generator:
                return np.random.default_rng(np.random.SeedSequence([cfg['namespace'],seed,3,i,stream]))
            measurement[i]=(rng(3).normal(size=(180,3))*dc['measurement_noise_std'])[:118]
            noise[i]=[rng(4).normal(scale=dc['proxy']['batch_std']),rng(5).normal(scale=dc['proxy']['assay_std'])]
            null[i]=rng(6).normal(dc['proxy']['null_mean'],dc['proxy']['null_std'])
        assert np.array_equal(noise,base['label_noise']) and np.array_equal(null,y[:,1])
        target=dc['proxy']['intercept']+dc['proxy']['coefficient']*np.maximum(base['clean_process'][:,:,0]-dc['proxy']['threshold'],0).sum(axis=1)+noise.sum(axis=1)
        assert np.allclose(target,y[:,0],rtol=0,atol=1e-12)
        for factor,view in zip(cfg['noise_factors'],views):
            assert set(view)=={'observations','recorded_regime','sequence_id','time_step','available_time'}
            assert np.array_equal(view['recorded_regime'],reg)
            assert np.array_equal(view['sequence_id'],base['sequence_id'])
            assert np.array_equal(view['time_step'],np.arange(118))
            assert np.array_equal(view['available_time'],np.arange(118)+2)
            residual=np.max(np.abs(view['observations']-(base['clean_process'][:,:118]+factor*measurement)))
            max_residual=max(max_residual,float(residual));assert residual<1e-12
        pred=p['prediction'];assert pred.shape==(2,3,7,n,2) and np.isfinite(pred).all()
        assert list(p['architectures'])==['ridge','lstm'] and list(p['groups'])==['mean','B0','B2','B3','B2Q','B3Q','B2R']
        assert list(p['noise_factors'])==cfg['noise_factors'] and list(p['targets'])==cfg['targets']
        mean=npz(ridge/f'seed_{seed}/ridge_models.npz')['mean']
        assert np.array_equal(pred[:,:,0],np.broadcast_to(mean,pred[:,:,0].shape))
        mr,pr=[],[]
        for ai,arch in enumerate(p['architectures']):
            for stratum,mask in [('all',np.ones(n,bool)),('transit_smooth',reg[:,-1]==1),('transit_rough',reg[:,-1]==2)]:
                if not mask.any():continue
                for ti,target_name in enumerate(cfg['targets']):
                    mae=np.zeros((3,7));rmse=np.zeros((3,7))
                    for fi in range(3):
                        for gi in range(7):
                            error=pred[ai,fi,gi,mask,ti]-y[mask,ti]
                            mae[fi,gi]=sum(abs(error))/len(error);rmse[fi,gi]=np.sqrt(sum(error*error)/len(error))
                    for fi,factor in enumerate(cfg['noise_factors']):
                        for gi,group in enumerate(p['groups']):
                            mr.append(dict(seed=seed,architecture=arch,noise_factor=factor,group=group,target=target_name,stratum=stratum,
                                batches=int(mask.sum()),mae=mae[fi,gi],rmse=rmse[fi,gi],mae_change_vs_1x=mae[fi,gi]-mae[1,gi]))
                        for bi,ci in [(3,2),(5,4),(6,2)]:
                            delta=mae[fi,bi]-mae[fi,ci]
                            pr.append(dict(seed=seed,architecture=arch,noise_factor=factor,comparison=p['groups'][bi]+'-'+p['groups'][ci],
                                target=target_name,stratum=stratum,mae_difference=delta,interaction_vs_1x=delta-(mae[1,bi]-mae[1,ci])))
        mk=['seed','architecture','noise_factor','group','target','stratum'];pk=['seed','architecture','noise_factor','comparison','target','stratum']
        match(rows(folder/'metrics_by_seed.csv'),mr,mk,['batches','mae','rmse','mae_change_vs_1x'])
        match(rows(folder/'paired_by_seed.csv'),pr,pk,['mae_difference','interaction_vs_1x'])
        metrics+=mr;paired+=pr
    match(rows(run/'metrics_by_seed.csv'),metrics,mk,['batches','mae','rmse','mae_change_vs_1x'])
    match(rows(run/'paired_by_seed.csv'),paired,pk,['mae_difference','interaction_vs_1x'])
    total_summary=0
    for name,records,keys,values in [('metrics',metrics,mk[1:],['mae','rmse','mae_change_vs_1x']),('paired',paired,pk[1:],['mae_difference','interaction_vs_1x'])]:
        expected=summaries(records,keys,values,cfg['seeds']);actual=rows(run/(name+'_summary.csv'))
        match(actual,expected,keys,['data_repeats']+[v+s for v in values for s in ['_mean','_sd','_ci_low','_ci_high']])
        assert all(r['interval_use']=='descriptive_only' for r in actual)
        total_summary+=len(expected)
    meta=json.loads((run/'metadata.json').read_text())
    assert meta['unique_batches']==len(ids) and meta['paired_versions']==3*len(ids)
    assert meta['refitting'] is False and meta['prior_test_status']=='used_unchanged' and meta['old_test_predictions'] is False
    return dict(status='pass',unique_batches=len(ids),paired_versions=3*len(ids),output_files_hashed=len(manifest),
        metric_rows_recomputed=len(metrics),paired_rows_recomputed=len(paired),summary_rows_recomputed=total_summary,
        maximum_independent_noise_residual=max_residual,label_formula_and_random_streams='pass',
        source_artifacts_unchanged=True,manifest_sha256=digest(run/'manifest.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    for name,default in [('data','rq3_paired_20261019_rebuild_v1'),('ridge','results/rq3_prediction/rq3_ridge_pilot_20261019_v1'),('lstm','results/rq3_prediction/rq3_lstm_pilot_20261020_v1'),('final','results/rq3_prediction/rq3_final_test_20261021_v1')]:
        p.add_argument('--'+name,type=Path,default=Path(default))
    a=p.parse_args();print(json.dumps(verify(a.run,a.data,a.ridge,a.lstm,a.final),indent=2))
