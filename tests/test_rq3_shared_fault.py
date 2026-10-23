"""Numerical and leakage guards for the fixed shared-fault experiment."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from src.rq3_shared_fault import (train_events,make_bundle,arm_features,random_trajectory,
    gate,centers,fit_model,lock_training,seal,interval,metric_rows,arm_names)
from src.rq3_paired_data import generate_split
from src.rq3_lstm_pilot import fit_sequence_scaler

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def fixture():
    cfg=yaml.safe_load((ROOT/'configs/rq3_shared_fault_20261023.yaml').read_text())
    dc=yaml.safe_load((ROOT/'configs/rq3_paired_rebuild_20261019_v1.yaml').read_text())
    dc['counts']={k:8 for k in dc['counts']};dc['namespace']=20261023
    frozen=json.loads((ROOT/'results/rq3_prediction/rq3_ridge_pilot_20261019_v1/seed_42/preprocessing_and_score.json').read_text())
    s=generate_split(dc,42,'val')
    b=make_bundle(s['inputs'],s['events'],np.ones((3,3)),frozen,dc,cfg)
    return cfg,dc,frozen,s,b


def test_gate_hand_calculation():
    np.testing.assert_array_equal(gate(np.array([2.,8.,5.]),np.array([4.,4.,4.]),np.array([0.,.5,1.])),[4.,6.,5.])


def test_shared_features_and_oracle_isolation(fixture):
    cfg,dc,frozen,s,b=fixture
    scaler=fit_sequence_scaler(b['base'].reshape(-1,118,12));r=np.full((8,118,3),.4)
    x={g:arm_features(b,g,scaler,r,centers(frozen,dc)) for g in ['N','T','R','O']}
    for g in x:np.testing.assert_array_equal(x[g][...,:12],x['N'][...,:12])
    assert np.all(x['N'][...,12:]==0)
    altered=copy.deepcopy(b);altered['oracle'][:]=np.nan
    for g in ['N','T','R','TG','RG']:
        np.testing.assert_array_equal(arm_features(altered,g,scaler,r,centers(frozen,dc)),arm_features(b,g,scaler,r,centers(frozen,dc)))
    leaked=dict(s['inputs'],targets=s['targets'])
    with pytest.raises(ValueError):make_bundle(leaked,s['events'],np.ones((3,3)),frozen,dc,cfg)


def test_random_trajectories_exclude_self_and_repeat():
    ids=np.array(['a','b','c']);bank=np.arange(2*3*4*3).reshape(2,3,4,3)
    a,ix=random_trajectory(ids,ids,bank,42,1901)
    assert (ix[:,1]!=np.arange(3)).all()
    b,ib=random_trajectory(ids,ids,bank,42,1901)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ix,ib)
    np.testing.assert_array_equal(a,bank[ix[:,0],ix[:,1]])


def test_events_new_namespace_and_pair_labels(fixture):
    cfg,dc,frozen,s,b=fixture
    e=train_events(s['inputs']['sequence_id'],118,dc,42)
    assert e==train_events(s['inputs']['sequence_id'],118,dc,42)
    for v in b['views']:
        assert set(v)==set(s['inputs'])
        np.testing.assert_array_equal(v['sequence_id'],s['inputs']['sequence_id'])
    t=generate_split(dc,42,'test')
    assert not set(t['inputs']['sequence_id'])&set(s['inputs']['sequence_id'])
    np.testing.assert_array_equal(generate_split(dc,42,'val')['targets'],s['targets'])


def test_train_fit_constant_and_evaluation_no_refit(fixture):
    _,dc,frozen,s,b=fixture
    sc=fit_sequence_scaler(b['base'][0]);before=copy.deepcopy(sc)
    mutated=copy.deepcopy(b);mutated['base']+=100
    z=arm_features(mutated,'N',sc,np.zeros((8,118,3)),centers(frozen,dc))
    assert sc==before
    assert np.all(z[...,:12][...,np.array(sc['constant'])]==0)


def test_training_determinism_budget_and_replay(fixture,tmp_path):
    cfg,dc,frozen,s,b=fixture;cfg=copy.deepcopy(cfg);cfg['epochs']=2;cfg['batch_size']=4
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    sc=fit_sequence_scaler(b['base'].reshape(-1,118,12))
    x=arm_features(b,'T',sc,np.zeros((8,118,3)),centers(frozen,dc))
    schedule=np.random.default_rng(77342).integers(10,size=(2,8))
    a,p,h,pred=fit_model(x,s['targets'][:,0],x,s['targets'][:,0],schedule,cfg,42,0)
    aa,pp,hh,pred2=fit_model(x,s['targets'][:,0],x,s['targets'][:,0],schedule,cfg,42,0)
    for k in a:np.testing.assert_array_equal(a[k],aa[k])
    np.testing.assert_array_equal(pred,pred2);assert p==pp and h==hh
    assert p['optimizer_updates']==4 and p['training_examples']==16 and p['replay_max_abs_error']==0
    assert p['selected_epoch']==min(range(1,3),key=lambda e:h[e-1]['val_suite_mae'])
    # Perturb features, not budget or initializer; core fairness remains exact.
    _,other,_,_=fit_model(x*0,s['targets'][:,0],x*0,s['targets'][:,0],schedule,cfg,42,0)
    for k in ['initial_state_sha256','shuffle_sha256','schedule_sha256','optimizer_updates','parameter_count']:
        assert other[k]==p[k]


def test_lock_and_seal_refuse_partial_or_changed_training(tmp_path):
    cfg={'seeds':[42,43]}
    with pytest.raises(ValueError):lock_training(tmp_path,cfg)
    for seed in cfg['seeds']:
        p=tmp_path/f'seed_{seed}'/'training';p.mkdir(parents=True);(p/'model.txt').write_text('frozen');seal(p)
        with pytest.raises(FileExistsError):seal(p)
    lock_training(tmp_path,cfg)
    (tmp_path/'seed_42/training/model.txt').write_text('changed')
    with pytest.raises(ValueError):lock_training(tmp_path,cfg)


def test_metric_and_interval_math(fixture):
    cfg,_,_,s,b=fixture
    arms=arm_names(cfg)+['mean'];y=s['targets']
    pred=np.broadcast_to(y,(10,len(arms),*y.shape)).copy();pred[:,0,:,0]+=2
    rows=metric_rows(pred,y,s['events'],42,cfg,arms,np.zeros((10,8),bool))
    r=next(r for r in rows if r['view']=='original' and r['arm']==arms[0] and r['target']=='arrival_proxy')
    assert r['mae']==r['rmse']==2
    ci=interval([-1.]*5);assert ci['mean']==ci['ci_low']==ci['ci_high']==-1


def test_full_stage_smoke_on_isolated_fixture(fixture,tmp_path):
    from src.rq3_shared_fault import train_seed,evaluate_seed,summarize,read_npz
    from src.rq3_prediction_pilot import write_json,write_csv
    from src.rq3_release_audit import sha256
    cfg,dc,frozen,s,b=fixture;cfg=copy.deepcopy(cfg)
    cfg.update(epochs=2,batch_size=4,seeds=[42],counts=dict(train=8,val=4,test=4),run_id='SMOKE_NOT_RESEARCH')
    data=tmp_path/'data';ridge=tmp_path/'ridge';out=tmp_path/'result'
    d=data/'seed_42'/'train';d.mkdir(parents=True)
    dc['namespace']=20261019
    (data/'config.yaml').write_text(yaml.safe_dump(dc))
    tr=generate_split(dc,42,'train')
    np.savez_compressed(d/'inputs.npz',**tr['inputs'])
    write_csv(d/'targets.csv',[dict(sequence_id=str(sid),arrival_proxy=float(y[0]),null_proxy=float(y[1])) for sid,y in zip(tr['inputs']['sequence_id'],tr['targets'])])
    write_json(d.parent/'fault_scales.json',dict(scales=np.ones((3,3)).tolist()))
    r=ridge/'seed_42';r.mkdir(parents=True);write_json(r/'preprocessing_and_score.json',frozen)
    seal(data);seal(ridge)
    cfg['data_manifest_sha256']=sha256(data/'manifest.json');cfg['ridge_manifest_sha256']=sha256(ridge/'manifest.json')
    train_seed(data,ridge,cfg,out,'a'*40,42)
    evaluate_seed(data,ridge,cfg,out,'a'*40,42)
    p=read_npz(out/'seed_42/evaluation/predictions.npz')
    assert p['prediction'].shape==(10,11,4,2) and np.isfinite(p['prediction']).all()
    with pytest.raises(FileExistsError):train_seed(data,ridge,cfg,out,'a'*40,42)
