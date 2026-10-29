"""Post-fit paired mechanism tables and channel-wise correction accounting."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import rq3_uncertainty_a0 as a0
from rq3_joint_prediction_dev import score_bundle

def analyze(root:Path,data:Path,a0root:Path,scorer:Path,out:Path)->None:
    """Analyze every condition without changing fitted models or source data."""
    out.mkdir(parents=True,exist_ok=False);cfg=json.loads((root/'config.json').read_text());pairs=[];account=[]
    for seed in cfg['seeds']:
        sd=root/f'seed_{seed}';p=a0.npz(sd/'predictions.npz');m=a0.npz(sd/'evaluation_metadata.npz');err=np.abs(p['pred']-p['y'][None,None,:])
        for arm,base in cfg['comparisons']:
            ia,ib=cfg['arms'].index(arm),cfg['arms'].index(base)
            for vi,view in enumerate(cfg['views']):
                ci=cfg['views'].index(view.split('_')[0]+'_clean')
                groups={'all':np.ones(len(p['y']),bool)};groups.update({f'channel_{j}':m['channel']==j for j in range(3)});groups.update({f'offset_{o}':m['offset']==o for o in [-6,18]});groups.update({f'end_regime_{j}':m['end_regime']==j for j in range(3)})
                for label,mask in groups.items():
                    if mask.any():pairs.append(dict(seed=seed,comparison=f'{arm}-{base}',view=view,stratum=label,n=int(mask.sum()),mae_difference=float((err[ia,vi,mask]-err[ib,vi,mask]).mean()),degradation_difference=float(((err[ia,vi,mask]-err[ia,ci,mask])-(err[ib,vi,mask]-err[ib,ci,mask])).mean())))
        v=a0.npz(data/f'seed_{seed}/val/inputs.npz');ev=json.loads((sd/'val_events.json').read_text());old=json.loads((a0root/f'seed_{seed}/model.json').read_text());new=json.loads((scorer/f'seed_{seed}/model.json').read_text());candidate=json.loads((scorer/'candidate_config.json').read_text());scorecfg=json.loads((scorer/'config.json').read_text());viewcfg=json.loads((a0root/'config.json').read_text());scales=np.asarray(json.loads((data/f'seed_{seed}/fault_scales.json').read_text())['scales'])
        bundle=score_bundle(v['observations'],v['recorded_regime'],ev,scales,old,new,candidate,scorecfg,viewcfg);train_scale=np.asarray(json.loads((sd/'scaler.json').read_text())['scale'])[:3]
        for vi,view in enumerate(cfg['views']):
            ci=cfg['views'].index(view.split('_')[0]+'_clean');x=bundle['x'][vi];clean=bundle['x'][ci];s=bundle['S'][vi];reference=bundle['center'][vi];f=s*x+(1-s)*reference
            for j in range(3):
                for label,mask in [('all',np.ones(x.shape[:2],bool)),('injected',bundle['mask'][vi,:,:,j]),('not_injected',~bundle['mask'][vi,:,:,j])]:
                    if mask.any():
                        e=(x[:,:,j]-clean[:,:,j])/train_scale[j];b=(reference[:,:,j]-clean[:,:,j])/train_scale[j];w=s[:,:,j];total=((f[:,:,j]-clean[:,:,j])/train_scale[j])**2
                        term1=w*w*e*e;term2=(1-w)**2*b*b;cross=2*w*(1-w)*e*b
                        np.testing.assert_allclose(total,term1+term2+cross,atol=1e-11,rtol=1e-10)
                        account.append(dict(seed=seed,view=view,channel=j,region=label,points=int(mask.sum()),mean_downweight=float((1-w)[mask].mean()),fraction_above_alarm=float((w[mask]<np.exp(-.5)).mean()),observed_clean_mse=float((e*e)[mask].mean()),corrected_clean_mse=float(total[mask].mean()),retained_error_term=float(term1[mask].mean()),reference_bias_term=float(term2[mask].mean()),cross_term=float(cross[mask].mean())))
    a0.csv_write(out/'stratified_paired_effects.csv',pairs);a0.csv_write(out/'correction_error_accounting.csv',account)
    summary=[]
    for key in sorted({(r['comparison'],r['view'],r['stratum']) for r in pairs}):
        selected=[r for r in pairs if (r['comparison'],r['view'],r['stratum'])==key];x=np.array([r['mae_difference'] for r in selected]);d=np.array([r['degradation_difference'] for r in selected])
        summary.append(dict(comparison=key[0],view=key[1],stratum=key[2],mean=float(x.mean()),sd=float(x.std(ddof=1)),negative_seeds=int((x<0).sum()),outer_seeds=len(x),degradation_difference=float(d.mean())))
    a0.csv_write(out/'stratified_paired_summary.csv',summary)
    a0.dump(out/'scope.json',dict(post_hoc_analysis=True,extra_training_models=0,paired_clean_is_noisy_observation_not_latent_truth=True,correction_identity_verified=True,source_result_manifest=a0.digest(root/'manifest.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--a0-root',type=Path,required=True);p.add_argument('--scorer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.root,a.data,a.a0_root,a.scorer_root,a.output)
