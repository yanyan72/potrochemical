"""Independent archive recount: no imports from experimental metric code."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import t

def verify(root: Path, repository_root: Path | None = None) -> dict:
    """Check manifest, saved alarms/posteriors and every batch or RQ1 table row."""
    manifest=json.loads((root/'manifest.json').read_text())
    for rel,h in manifest.items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
    counts=0;rq1=(root/'primary.json').exists();source=json.loads((root/'source_hashes.json').read_text())
    for p,h in source.items():
        resolved=Path(p)
        if not resolved.exists() and repository_root is not None:
            mappings=[('rq3_paired_20261019_rebuild_v1/','data/processed/rq3_paired_20261019_rebuild_v1/'),('rq3_selective_confirmation_20261030_v1/training/','results/rq3_prediction/rq3_selective_confirmation_20261030_v1/training/'),('sources/','results/logistics_e1b/logistics_e1b_20260918_v1/'),('configs/logistics_e1b.yaml','configs/logistics_e1b.yaml')]
            for marker,dest in mappings:
                if marker in p:resolved=repository_root/(dest+p.split(marker,1)[1]);break
        data=resolved.read_bytes()
        # Downloaded E1b JSON/config copies had one added terminal LF only.
        exact=hashlib.sha256(data).hexdigest()==h
        download_copy=('sources/' in p or p=='configs/logistics_e1b.yaml') and hashlib.sha256(data+b'\n').hexdigest()==h
        assert exact or download_copy,p
    for sd in sorted(root.glob('seed_*')):
        with np.load(sd/'inputs_and_scores.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        np.testing.assert_array_equal(a['alarm'],a['ratio']>1)
        assert np.isfinite(a['ratio']).all()
        if rq1:
            df=pd.read_csv(root/'metrics_by_seed.csv');df=df[df.seed==int(sd.name[5:])]
            for row in df.itertuples():
                v=list(a['magnitudes']).index(row.magnitude);arm=list(a['methods']).index(row.method);pick=np.ones(len(a['regimes']),bool) if row.regime=='all' else a['regimes']==row.regime
                alarm=a['alarm'][v,arm,pick];mask=a['fault_mask'][v,pick]
                if row.channel!=-1:alarm=alarm[:,:,row.channel:row.channel+1];mask=mask[:,:,row.channel:row.channel+1]
                assert int((alarm&mask).sum())==row.tp and int((alarm&~mask).sum())==row.fp
                assert int(mask.sum())==row.fault_cells and int((~mask).sum())==row.clean_cells;counts+=1
            continue
        np.testing.assert_allclose(a['q'].sum(-1),1,atol=1e-12)
        views=json.loads((sd/'views.json').read_text());df=pd.read_csv(sd/'batch_metrics.csv',dtype={'channels':str})
        for row in df.itertuples():
            v=list(a['views']).index(row.view);arm=list(a['arms']).index(row.arm);i=row.batch;s=int(a['starts'][i]);end=s+24
            alarm=a['alarm'][v,arm,i];mask=a['fault_mask'][v,i];q=a['q'][v,arm,i];truth=a['truth'][i]
            cv=next(k for k,m in enumerate(views) if m['lag']==row.lag and m['magnitude']==0);clean_alarm=a['alarm'][cv,arm,i];cq=a['q'][cv,arm,i]
            valid=np.ones(len(truth),bool) if row.regime==-1 else truth==row.regime;support=valid[:,None];active=mask.any(0);window=np.zeros(len(truth),bool);window[s:end]=True
            by=window[:,None]&~active[None,:]&support
            assert int((alarm&mask&support).sum())==row.tp;assert int((mask&support).sum())==row.fault_cells
            assert int((alarm&~mask&support).sum())==row.clean_fp;assert int((~mask&support).sum())==row.clean_cells
            assert int((alarm&by).sum())==row.bystander_fp;assert int(by.sum())==row.bystander_cells
            assert int(((q.argmax(-1)!=truth)&valid).sum())==row.mode_errors
            np.testing.assert_allclose(((q-np.eye(3)[truth])**2).sum(-1)[valid].sum(),row.mode_brier_sum,atol=1e-12)
            np.testing.assert_allclose((.5*abs(q-cq).sum(-1))[valid].sum(),row.posterior_tv_sum,atol=1e-12)
            if row.regime==-1:
                added=alarm&~clean_alarm&mask;det=added.any(0)[active]
                assert int(det.sum())==row.event_added_channels;assert int(active.any() and det.all())==row.all_channels_added
            counts+=1
        # Paired perturbation leaves every unmasked observation exactly unchanged.
        np.testing.assert_array_equal(a['observations'][~a['fault_mask']],np.broadcast_to(a['clean'],a['observations'].shape)[~a['fault_mask']])
    if not rq1:
        frames=[pd.read_csv(sd/'batch_metrics.csv',dtype={'channels':str}) for sd in sorted(root.glob('seed_*'))]
        allbatch=pd.concat(frames,ignore_index=True);published=pd.read_csv(root/'metrics_by_seed.csv',dtype={'channels':str})
        definitions={'point_recall':('tp','fault_cells'),'clean_fpr':('clean_fp','clean_cells'),'bystander_fpr':('bystander_fp','bystander_cells'),'mode_error':('mode_errors','time_cells'),'mode_brier':('mode_brier_sum','time_cells'),'true_p_loss':('true_p_loss_sum','time_cells'),'posterior_tv':('posterior_tv_sum','time_cells'),'event_added_recall':('event_added_channels','event_channels'),'all_channels_added_recall':('all_channels_added','fault_batches'),'exact_set_event_recall':('exact_set_added','fault_batches')}
        group=['seed','view','lag','channels','magnitude','arm','regime'];columns=list(dict.fromkeys(v for pair in definitions.values() for v in pair))
        for stratum,extra in [('all',[]),('offset',['offset']),('sign',['sign_pattern'])]:
            expected=allbatch.groupby(group+extra)[columns].sum().sort_index();actual=published[published.stratum==stratum].set_index(group+extra).sort_index()
            for column in columns:np.testing.assert_allclose(actual[column],expected[column],atol=1e-10)
            for key,(num,den) in definitions.items():np.testing.assert_allclose(actual[key],expected[num]/expected[den].replace(0,np.nan),atol=1e-12,equal_nan=True)
    if rq1:
        frame=pd.read_csv(root/'metrics_by_seed.csv')
        for r in json.loads((root/'primary.json').read_text()):
            l,rr=r['contrast'].split(' minus ');f=frame[(frame.regime=='all')&(frame.channel==-1)&(frame.magnitude==r['magnitude'])].pivot(index='seed',columns='method',values=r['metric']);d=(f[l]-f[rr]).to_numpy();np.testing.assert_allclose(d,r['differences'],atol=1e-14);half=t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5);np.testing.assert_allclose([d.mean()-half,d.mean()+half],r['ci95'],atol=1e-14)
    return {'directory':root.name,'manifest_items':len(manifest),'recounted_rows':counts,'all_passed':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--repository-root',type=Path);args=p.parse_args();result=[verify(r,args.repository_root) for r in args.roots];args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
