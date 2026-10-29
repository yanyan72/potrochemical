"""Week-one controlled oracle substitution; no candidate performance or quality fit."""
import argparse,csv,json,platform,sys,time
from pathlib import Path
import numpy as np
import rq3_uncertainty_a0 as a0


def run(root,config,source_identity,output,commit):
    if output.exists():raise FileExistsError(output)
    if a0.git_blob(root/'manifest.json')!=source_identity['a0_manifest_git_blob']:raise ValueError('A0 manifest differs from committed source')
    if a0.git_blob(Path(a0.__file__))!=source_identity['a0_module_git_blob']:raise ValueError('A0 implementation mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for path,h in manifest.items():
        if a0.digest(root/path)!=h:raise ValueError('A0 input mismatch '+path)
    cfg=json.loads((root/'config.json').read_text());output.mkdir(parents=True)
    a0.dump(output/'source_identity.json',source_identity)
    a0.dump(output/'config.json',config)
    a0.dump(output/'identity.json',dict(implementation_commit=commit,python=sys.version,numpy=np.__version__,platform=platform.platform(),stage='W1-R diagnostic',new_candidate_real_data_evaluation=False,quality_models=0,source='reused_synthetic_development'))
    started=time.perf_counter();summary=[];checks=[]
    for seed in cfg['seeds']:
        source=root/f'seed_{seed}';dest=output/f'seed_{seed}';dest.mkdir()
        model=json.loads((source/'model.json').read_text());all_rows=[]
        for split in cfg['splits']:
            events=json.loads((source/f'{split}_events.json').read_text())
            before=list(csv.DictReader((source/f'{split}_batch_metrics.csv').open()))
            original={(r['view'],r['arm'],r['sequence_id']):r for r in before}
            ratios=[];alarms=[];clean={};split_rows=[];baseline_rows=[]
            for view in cfg['views']:
                v=a0.npz(source/f'{split}_{view}.npz');rec,kind=view.split('_')
                raw,q=a0.matched_raw(v['observations'],v['recorded_regime'],model['modes'],cfg,'oracle_matched',v['true_regime_evaluation_only'])
                vr=[];va=[]
                for base in ['soft_all','soft_loo']:
                    ratio=raw/model['thresholds'][base];alarm=ratio>1
                    if kind=='clean':clean[(rec,base)]=alarm.copy()
                    rows=a0.per_batch(alarm,clean[(rec,base)],v['fault_mask'],q,v['recorded_regime'],v['true_regime_evaluation_only'],events)
                    # Recount key outcomes independently of a0.per_batch.
                    for i,e in enumerate(events):
                        s,end,ch=e['start'],e['end'],e['channel'];fault=kind!='clean';other=[j for j in range(3) if j!=ch]
                        nfp=int(alarm[i,s:end][:,other].sum()) if fault else 0
                        event=int(fault and np.any(alarm[i,s:end,ch]&~clean[(rec,base)][i,s:end,ch]))
                        assert rows[i]['bystander_fp']==nfp and rows[i]['event_added']==event
                        row=rows[i];row.update(seed=seed,split=split,view=view,arm=base+'_oracle_reference',base_arm=base)
                        old=original[(view,base,str(v['sequence_id'][i]))]
                        copied={k:old[k] for k in old};copied['base_arm']=base
                        for k in list(copied):
                            if k in ['sequence_id','split','view','arm','base_arm']:continue
                            copied[k]=None if copied[k]=='' else float(copied[k])
                        baseline_rows.append(copied)
                    split_rows.extend(rows);vr.append(ratio.astype(np.float32));va.append(alarm)
                    for stratum,selected in [('all',rows)]+[(f'channel_{j}',[r for r in rows if r['channel']==j]) for j in range(3)]+[(f'offset_{o}',[r for r in rows if r['offset']==o]) for o in cfg['start_offsets']]:
                        summary.append(dict(seed=seed,split=split,view=view,arm=base+'_oracle_reference',base_arm=base,stratum=stratum,**a0.summarize_rows(selected)))
                ratios.append(np.stack(vr));alarms.append(np.stack(va))
            np.savez_compressed(dest/f'{split}_scores.npz',ratio=np.stack(ratios),alarm=np.stack(alarms),views=cfg['views'],base_arms=['soft_all','soft_loo'],sequence_id=v['sequence_id'])
            a0.csv_write(dest/f'{split}_batch_metrics.csv',split_rows)
            # Baselines are archived by reference; summarize without refitting anything.
            for view in cfg['views']:
                for base in ['soft_all','soft_loo']:
                    rows=[r for r in baseline_rows if r['view']==view and r['arm']==base]
                    for stratum,selected in [('all',rows)]+[(f'channel_{j}',[r for r in rows if r['channel']==j]) for j in range(3)]+[(f'offset_{o}',[r for r in rows if r['offset']==o]) for o in cfg['start_offsets']]:
                        summary.append(dict(seed=seed,split=split,view=view,arm=base,base_arm=base,stratum=stratum,**a0.summarize_rows(selected)))
            checks.append(dict(seed=seed,split=split,batches=len(events),thresholds_unchanged=True,independent_count_check=True,quality_labels_read=False))
        print(json.dumps(dict(seed=seed,completed=True)),flush=True)
    a0.csv_write(output/'metrics_by_seed.csv',summary);a0.dump(output/'checks.json',checks)
    pairs=[]
    for r in summary:
        if not r['arm'].endswith('_oracle_reference'):continue
        old=next(x for x in summary if x['seed']==r['seed'] and x['split']==r['split'] and x['view']==r['view'] and x['stratum']==r['stratum'] and x['arm']==r['base_arm'])
        row={k:r[k] for k in ['seed','split','view','base_arm','stratum']}
        for k in ['clean_fpr','point_recall','event_added','event_fresh','bystander_fpr','transition_clean_fpr']:
            row[k+'_before']=old[k];row[k+'_after']=r[k];row[k+'_difference']=None if r[k] is None else r[k]-old[k]
        pairs.append(row)
    a0.csv_write(output/'paired_differences.csv',pairs)
    main=[r for r in pairs if r['split']=='val' and r['view']=='delay12_bias4' and r['base_arm']=='soft_loo' and r['stratum']=='all']
    before=np.mean([r['bystander_fpr_before'] for r in main]);after=np.mean([r['bystander_fpr_after'] for r in main])
    rel=(before-after)/before if before>0 else 0
    gate=config['w1_diagnostic_gate']
    passed=bool(all(r['bystander_fpr_difference']<0 and r['event_added_difference']>=-gate['maximum_event_recall_loss'] for r in main) and rel>=gate['minimum_relative_bystander_reduction'])
    decision=dict(oracle_reference_direction_supported=passed,continue_one_candidate_development=passed,
        candidate_performance_supported=False,rq3_supported=False,bystander_fpr_before=float(before),bystander_fpr_after=float(after),relative_reduction=float(rel),primary_seed_rows=main,
        scope='known-A0 development diagnostic; true modes are not deployable; unchanged thresholds; no quality model')
    a0.dump(output/'decision.json',decision)
    for path,h in manifest.items():assert a0.digest(root/path)==h
    a0.dump(output/'source_unchanged.json',dict(a0_manifest_items_verified=len(manifest),unchanged=True,elapsed_seconds=time.perf_counter()-started))
    a0.dump(output/'manifest.json',{str(p.relative_to(output)):a0.digest(p) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps(decision,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a0-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--source-identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(a.a0_root,json.loads(a.config.read_text()),json.loads(a.source_identity.read_text()),a.output,a.commit)
