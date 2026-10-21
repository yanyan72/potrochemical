"""Independent arithmetic verification of persisted RQ3 final-test predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def digest(path: Path) -> str:
    """Hash persisted bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV without importing model/evaluation code."""
    with path.open() as f:
        return list(csv.DictReader(f))


def verify(run: Path, data: Path, ridge: Path, lstm: Path) -> dict:
    """Recompute all metrics, paired suite endpoints and intervals independently."""
    cfg = json.loads((run/'config.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    assert set(manifest) == {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()} - {'manifest.json'}
    assert all(digest(run/p) == h for p,h in manifest.items())
    sources = json.loads((run/'source_manifests.json').read_text())
    for name, folder in (('data',data), ('ridge',ridge), ('lstm',lstm)):
        assert digest(folder/'manifest.json') == cfg[name+'_manifest_sha256']
        assert all(digest(folder/p) == h for p,h in sources[name].items())
    m = rows(run/'metrics_by_seed.csv')
    lookup = {(int(r['seed']),r['architecture'],r['view'],r['group'],r['target'],r['stratum']):r for r in m}
    assert len(lookup)==len(m)
    checked = 0
    direct_endpoints = {}
    for seed in cfg['seeds']:
        original_labels = rows(data/f'seed_{seed}/test/targets.csv')
        events = rows(data/f'seed_{seed}/test/events.csv')
        with np.load(data/f'seed_{seed}/test/inputs.npz',allow_pickle=False) as x:
            cutoff = x['recorded_regime'][:,-1]
        with np.load(run/f'seed_{seed}/evaluation_targets.npz',allow_pickle=False) as z:
            labels, ids = z['targets'], z['sequence_id']
        assert list(ids)==[r['sequence_id'] for r in original_labels]==[r['sequence_id'] for r in events]
        assert np.array_equal(labels,[[float(r[name]) for name in cfg['targets']] for r in original_labels])
        with np.load(run/f'seed_{seed}/predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(ids,z['sequence_id'])
            pred = z['prediction']
            assert np.isfinite(pred).all()
            assert list(z['architectures'])==cfg['architectures'] and list(z['groups'])==cfg['groups']
            assert list(z['target_names'])==cfg['targets']
            assert list(z['views'])==['original','bias_1','bias_2','bias_4','drift_1','drift_2','drift_4','missing','delay_6','delay_12']
            for ai,arch in enumerate(z['architectures']):
                for vi,view in enumerate(z['views']):
                    masks = {'all':np.ones(len(ids),bool)}
                    for ri,regime in enumerate(('warehouse','transit_smooth','transit_rough')):
                        masks['cutoff_regime:'+regime]=cutoff==ri
                    if str(view).startswith(('bias','drift','missing')):
                        for ci,ch in enumerate(('temperature','relative_humidity','vibration')):
                            masks['fault_channel:'+ch]=np.array([int(e['channel'])==ci for e in events])
                        for duration in (4,12,24):
                            masks[f'duration:{duration}']=np.array([int(e['end_exclusive'])-int(e['start'])==duration for e in events])
                    for gi,group in enumerate(z['groups']):
                        for ti,target in enumerate(z['target_names']):
                            for stratum,mask in masks.items():
                                if not mask.any():continue
                                row=lookup[seed,str(arch),str(view),str(group),str(target),stratum]
                                errors=pred[ai,vi,gi,mask,ti]-labels[mask,ti]
                                assert int(row['batches'])==mask.sum()
                                assert abs(float(row['mae'])-np.abs(errors).mean())<1e-10
                                assert abs(float(row['rmse'])-np.sqrt(np.mean(errors**2)))<1e-10
                                checked+=1
                for a,b in (('B3','B2'),('B3Q','B2Q'),('B2R','B2')):
                    errors=np.abs(pred[ai,:,list(z['groups']).index(a)]-labels)-np.abs(pred[ai,:,list(z['groups']).index(b)]-labels)
                    for ti,target in enumerate(z['target_names']):
                        for scope,ix in (('original',[0]),('pressure_equal_weight',list(range(1,10))),('suite_equal_weight',list(range(10)))):
                            direct_endpoints[seed,str(arch),a+'-'+b,str(target),scope]=float(errors[ix,:,ti].mean())
    assert checked==len(m)
    paired=rows(run/'paired_by_seed.csv')
    for r in paired:
        s,a,v,c,target=int(r['seed']),r['architecture'],r['view'],r['comparison'],r['target']
        better,base=c.split('-')
        diff=float(lookup[s,a,v,better,target,'all']['mae'])-float(lookup[s,a,v,base,target,'all']['mae'])
        clean=float(lookup[s,a,'original',better,target,'all']['mae'])-float(lookup[s,a,'original',base,target,'all']['mae'])
        assert abs(float(r['mae_difference'])-diff)<1e-10
        assert abs(float(r['degradation_difference'])-diff+clean)<1e-10
    summaries=0
    for name,raw,keys,values in (
        ('metrics_summary',m,['architecture','view','group','target','stratum'],['mae','rmse']),
        ('paired_summary',paired,['architecture','view','comparison','target'],['mae_difference','degradation_difference'])):
        summary=rows(run/(name+'.csv'))
        assert len(summary)==len({tuple(r[k] for k in keys) for r in raw})
        for row in summary:
            group=[r for r in raw if all(r[k]==row[k] for k in keys)]
            assert len(group)==int(row['data_repeats'])==len(cfg['seeds'])
            assert sorted(int(r['seed']) for r in group)==cfg['seeds']
            for value in values:
                numbers=np.array([float(r[value]) for r in group])
                assert abs(float(row[value+'_mean'])-numbers.mean())<1e-10
                assert abs(float(row[value+'_sd'])-numbers.std(ddof=1))<1e-10
            summaries+=1
    erows=rows(run/'endpoints_by_seed.csv')
    assert len(erows)==len(direct_endpoints)
    for r in erows:
        key=(int(r['seed']),r['architecture'],r['comparison'],r['target'],r['scope'])
        assert abs(float(r['mae_difference'])-direct_endpoints[key])<1e-10
    es=rows(run/'endpoints_summary.csv')
    assert len(es)==36
    primary=[]
    for r in es:
        key=tuple(r[k] for k in ('architecture','comparison','target','scope'))
        numbers=np.array([direct_endpoints[(s,*key)] for s in cfg['seeds']])
        mu,sd=float(numbers.mean()),float(numbers.std(ddof=1))
        width=float(t.ppf(.975,len(numbers)-1)*sd/np.sqrt(len(numbers)))
        assert int(r['data_repeats'])==len(numbers) and int(r['df'])==len(numbers)-1
        for k,value in (('mae_difference_mean',mu),('mae_difference_sd',sd),('ci_low',mu-width),('ci_high',mu+width)):
            assert abs(float(r[k])-value)<1e-10
        direction='lower_error' if mu+width<0 else 'higher_error' if mu-width>0 else 'inconclusive'
        assert r['interval_direction']==direction
        if r['primary']=='True':primary.append(r)
    assert len(primary)==1
    assert tuple(primary[0][k] for k in ('architecture','comparison','target','scope'))==('lstm','B3-B2','arrival_proxy','suite_equal_weight')
    meta=json.loads((run/'metadata.json').read_text())
    assert meta['test_status']=='used' and meta['test_predictive_evaluation'] is True and meta['refitting'] is False
    return {'status':'pass','run_manifest_sha256':digest(run/'manifest.json'),'output_files_hashed':len(manifest),
            'input_files_hashed':{k:len(v) for k,v in sources.items()},'metric_rows_recomputed':checked,
            'paired_rows_recomputed':len(paired),'summary_rows_recomputed':summaries,
            'endpoint_rows_recomputed':len(erows),'endpoint_intervals_recomputed':len(es),
            'test_status':'used','verification_refitted_models':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','ridge','lstm'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.run,a.data,a.ridge,a.lstm),indent=2))
