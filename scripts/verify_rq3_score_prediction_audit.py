"""Independent replay of saved B2/B3 absolute errors and diagnostic summaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path:Path) -> dict:
    """Read numpy arrays without pickle."""
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def rows(path:Path) -> list[dict]:
    """Read CSV records."""
    with path.open(newline='') as f:return list(csv.DictReader(f))


def verify(run:Path,ridge:Path,lstm:Path) -> dict:
    """Recompute prediction diagnostics from original archived predictions."""
    cfg=json.loads((run/'config.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    for name,h in manifest.items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h
    checked=0;allrows=[]
    values=['b2_mae','b3_mae','mae_difference','improvement_fraction','b2_degradation','b3_degradation','incremental_degradation']
    for seed in cfg['seeds']:
        a=read(run/f'seed_{seed}/paired_batches.npz');actual=rows(run/f'seed_{seed}/prediction_pairs.csv')
        sources=[]
        for folder in [ridge,lstm]:
            b=read(folder/f'seed_{seed}/validation_predictions.npz')
            assert np.array_equal(a['sequence_id'],b['sequence_id']) and np.array_equal(a['targets'],b['targets'])
            assert list(a['views'])==list(b['views'])
            g=[list(b['groups']).index(n) for n in ['B2','B3']]
            sources.append(abs(b['prediction'][:,g]-b['targets'][None,None]))
        e=np.stack(sources);assert np.array_equal(a['absolute_errors'],e)
        for row in actual:
            ai=['ridge','lstm'].index(row['architecture']);vi=list(a['views']).index(row['view']);ti=list(a['target_names']).index(row['target']);s=row['stratum']
            if s=='all':mask=np.ones(len(a['sequence_id']),bool)
            elif s.startswith('channel:'):mask=a['channels']==int(s[-1])
            elif s=='fresh_alarm':mask=a['fresh_alarm'][vi]
            elif s=='no_fresh_alarm':mask=~a['fresh_alarm'][vi]
            else:raise AssertionError(s)
            b2=e[ai,vi,0,mask,ti];b3=e[ai,vi,1,mask,ti];old2=e[ai,0,0,mask,ti];old3=e[ai,0,1,mask,ti]
            expected=[b2.mean(),b3.mean(),(b3-b2).mean(),(b3<b2).mean(),(b2-old2).mean(),(b3-old3).mean(),((b3-old3)-(b2-old2)).mean()]
            assert int(row['batches'])==int(mask.sum())
            for k,v in zip(values,expected):assert np.isclose(float(row[k]),v,rtol=1e-12,atol=1e-12),(k,row[k],v)
            checked+=1
        allrows+=actual
        for vi,name in enumerate(a['views']):
            if name=='missing':assert np.all(a['event_score_availability'][vi]==0) and np.isnan(a['score_drop'][vi]).all()
            if name.startswith(('bias','drift')):assert np.all(a['event_score_availability'][vi]==1)
    summaries=rows(run/'prediction_summary.csv')
    keys=['architecture','view','stratum','target']
    for summary in summaries:
        selected=[r for r in allrows if all(r[k]==summary[k] for k in keys)]
        assert len({r['seed'] for r in selected})==len(selected)==int(summary['data_repeats'])
        for value in values:
            a=np.array([float(r[value]) for r in selected])
            assert np.isclose(a.mean(),float(summary[value+'_mean']))
            assert np.isclose(a.std(ddof=1) if len(a)>1 else 0.,float(summary[value+'_sd']))
    # Global paired means must reproduce the pre-existing development tables.
    for arch,folder in [('ridge',ridge),('lstm',lstm)]:
        old=rows(folder/'paired_summary.csv')
        for r in summaries:
            if r['architecture']!=arch or r['stratum']!='all':continue
            ref=next(x for x in old if x['comparison']=='B3-B2' and x['view']==r['view'] and x['target']==r['target'])
            assert np.isclose(float(r['mae_difference_mean']),float(ref['mae_difference_mean']),atol=1e-12)
    return dict(status='pass',hashed_output_files=len(manifest),prediction_rows_recomputed=checked,
        prediction_summary_rows_recomputed=len(summaries),saved_errors_exactly_match_original_archives=True,
        development_global_pairs_match_previous_results=True,missing_not_imputed_as_high_trust=True,
        scope='prediction arithmetic independently recomputed; event semantics covered by unit tests; point AP uses sklearn')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--ridge',type=Path,default=Path('results/rq3_prediction/rq3_ridge_pilot_20261019_v1'))
    p.add_argument('--lstm',type=Path,default=Path('results/rq3_prediction/rq3_lstm_pilot_20261020_v1'))
    a=p.parse_args();print(json.dumps(verify(a.run,a.ridge,a.lstm),indent=2))
