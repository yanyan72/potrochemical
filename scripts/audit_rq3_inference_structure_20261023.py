"""Read-only paper-stage audit; no generation, refitting or new hypothesis search."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def main() -> None:
    """Recompute seed-cluster contrasts from batch-paired saved predictions."""
    root=Path('results/rq3_prediction/rq3_shared_fault_20261023_v1')
    out=Path('results/paper_acceptance/paper_stage_20261023_v1')
    out.mkdir(parents=True,exist_ok=False)
    archived=json.loads((root/'conclusion.json').read_text())['primary']
    cfg=json.loads((root/'config.json').read_text())
    rows=[];batch_rows=[];hashes={};all_ids=[];per_seed_models=[]
    for seed in cfg['seeds']:
        predpath=root/f'seed_{seed}/evaluation/predictions.npz'
        hashes[str(predpath)]=hashlib.sha256(predpath.read_bytes()).hexdigest()
        with np.load(predpath,allow_pickle=False) as a:
            ids=a['sequence_id'];arms=list(a['arms']);targets=list(a['target_names'])
            ti=targets.index('arrival_proxy');err=np.abs(a['prediction'][...,ti]-a['targets'][None,None,:,ti])
            assert len(set(ids))==500 and len(a['views'])==10
            all_ids.extend(ids.tolist())
            for other in ['shared_N','shared_R']:
                diff=err[:,arms.index('shared_T')]-err[:,arms.index(other)]
                # Preserve all paired views belonging to one physical simulated batch.
                bd=diff.mean(axis=0)
                np.testing.assert_allclose(bd.mean(),diff.mean(axis=1).mean(),atol=1e-14,rtol=0)
                rows.append(dict(seed=seed,comparison='shared_T-'+other,batches=len(ids),paired_views=10,
                    difference=float(bd.mean()),batch_difference_sd_given_fitted_models=float(bd.std(ddof=1)),
                    conditional_batch_se_not_population_model_se=float(bd.std(ddof=1)/np.sqrt(len(ids)))))
                batch_rows.extend(dict(seed=seed,comparison='shared_T-'+other,sequence_id=str(s),view_averaged_absolute_error_difference=float(d)) for s,d in zip(ids,bd))
        pars=json.loads((root/f'seed_{seed}/training/model_parameters.json').read_text())
        target_models=[m for name,m in pars.items() if name.endswith('__arrival_proxy')]
        assert len(target_models)==10
        for key in ['model_seed','shuffle_seed','initial_state_sha256','shuffle_sha256','parameter_count','optimizer_updates']:
            assert len({m[key] for m in target_models})==1,key
        per_seed_models.append(dict(data_seed=seed,model_seed=target_models[0]['model_seed'],learned_arms=10,
                                   same_model_seed_across_arms=True,independent_model_repeats_within_dataset=1))
    assert len(all_ids)==len(set(all_ids))==2500
    primary=[]
    for comparison in ['shared_T-shared_N','shared_T-shared_R']:
        v=np.array([r['difference'] for r in rows if r['comparison']==comparison])
        sd=v.std(ddof=1);se=sd/np.sqrt(len(v));half=t.ppf(.975,4)*se
        item=dict(comparison=comparison,independent_outer_replicates=5,df=4,mean=float(v.mean()),sd=float(sd),
                  se=float(se),ci_low=float(v.mean()-half),ci_high=float(v.mean()+half),negative_seeds=int((v<0).sum()))
        old=next(r for r in archived if r['comparison']==comparison)
        for k in ['mean','sd','ci_low','ci_high']:
            assert abs(old[k]-item[k])<1e-12,(k,old[k],item[k])
        primary.append(item)
    for name,data in [('seed_paired_recalculation.csv',rows),('batch_paired_differences.csv',batch_rows)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    audit=dict(passed=True,analysis_only=True,new_data_generated=False,new_models_fitted=0,
        original_primary_conclusion_unchanged=True,primary=primary,model_data_mapping=per_seed_models,
        distinct_test_batches=2500,source_sha256=hashes,
        estimand='Expected paired ten-view equal-weight MAE difference over this synthetic data-and-training-seed procedure, with frozen hyperparameters and selection rule.',
        limitations=['Five outer replicates: t intervals require an approximate distributional assumption and are not distribution-free.',
                    'Data/reference/validation/test draws and model/shuffle seed change together; components of variance are not identifiable.',
                    'Batch uncertainty conditional on fitted models is not a replacement for outer-seed uncertainty.',
                    '100 models, 10 paired views, time points and correlated historical rounds are not independent replications.',
                    '1820 exploratory intervals are unadjusted; no familywise discoveries, equivalence, or universal absence of benefit established.',
                    'The two primary improvements are jointly required; do not choose whichever contrast or subgroup looks favorable.'])
    (out/'statistical_structure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,outer_replicates=5,df=4,paired_batch_rows=len(batch_rows),primary=primary),ensure_ascii=False))

if __name__=='__main__':main()
