"""Pairing, target integrity, frozen-inference and noise-contrast arithmetic."""
import copy
from pathlib import Path
import numpy as np
import pytest
import yaml
from src import rq3_noise_sensitivity as ns


def config():
    return yaml.safe_load(Path('configs/rq3_noise_sensitivity_20261022.yaml').read_text())


def test_paired_noise_reproducible_and_labels_unchanged():
    cfg=config();cfg.update(batches_per_seed=8,namespace=2026102201)
    dc=yaml.safe_load(Path('configs/rq3_paired_data.yaml').read_text())
    for seed in [42,43]:
        a,base,audit=ns.generate_paired(dc,cfg,seed)
        b,other,_=ns.generate_paired(dc,cfg,seed)
        assert audit['max_noise_scaling_residual']<1e-12
        for av,bv in zip(a,b):
            for key in av:assert np.array_equal(av[key],bv[key])
        assert np.array_equal(base['targets'],other['targets'])
        clean=base['truth']['clean_process'][:,:118]
        for f,v in zip(cfg['noise_factors'],a):
            assert np.max(abs(v['observations']-clean-f*(a[1]['observations']-clean)))<1e-12
        assert all('2026102201' in s for s in a[0]['sequence_id'])


@pytest.mark.parametrize('change',[{'noise_factors':[1.,2.,4.]},{'refit':True},{'source_split':'test'},{'inject_faults':True}])
def test_scope_guards(change):
    cfg=config();cfg.update(change)
    with pytest.raises(ValueError):ns.validate_config(cfg)


def test_predictor_rejects_truth_and_future_inputs_before_io(tmp_path):
    cfg=config();cfg['batches_per_seed']=2
    dc=yaml.safe_load(Path('configs/rq3_paired_data.yaml').read_text())
    views,_,_=ns.generate_paired(dc,cfg,42)
    bad=copy.deepcopy(views);bad[0]['targets']=np.zeros((2,2))
    with pytest.raises(ValueError,match='schema'):ns.predict_views(bad,tmp_path,tmp_path,tmp_path,cfg,42)
    bad=copy.deepcopy(views);bad[0]['available_time'][-1]=120
    with pytest.raises(ValueError,match='future'):ns.predict_views(bad,tmp_path,tmp_path,tmp_path,cfg,42)


def test_paired_difference_and_interaction_hand_calculation():
    cfg=config();y=np.zeros((4,2));pred=np.ones((2,3,7,4,2))
    pred[:,:,3]+=np.array([1.,2.,4.])[None,:,None,None]
    regs=np.array([[1]*118,[1]*118,[2]*118,[2]*118])
    rows,paired=ns.evaluate(pred,y,regs,cfg,42)
    chosen=[r for r in paired if r['architecture']=='lstm' and r['comparison']=='B3-B2' and r['target']=='arrival_proxy' and r['stratum']=='all']
    assert [r['mae_difference'] for r in chosen]==[1.,2.,4.]
    assert [r['interaction_vs_1x'] for r in chosen]==[-1.,0.,2.]
    assert len(rows)==252 and len(paired)==108


def test_summary_rejects_pseudoreplication():
    cfg=config();rows=[dict(seed=s,key='same',value=float(s-41)) for s in cfg['seeds']]
    got=ns.summarize_rows(rows,['key'],['value'],cfg)[0]
    assert got['data_repeats']==5 and got['value_mean']==3
    assert abs(got['value_sd']-np.sqrt(2.5))<1e-12
    assert abs(got['value_ci_high']-(3+2.776445105*np.sqrt(.5)))<1e-8
    with pytest.raises(ValueError,match='seed'):ns.summarize_rows(rows+[rows[0]],['key'],['value'],cfg)


def test_archived_models_small_integration_and_no_overwrite(tmp_path,monkeypatch):
    data=ns.DEFAULT_DATA if ns.DEFAULT_DATA.exists() else Path('data/processed/rq3_paired_20261019_rebuild_v1')
    if not data.exists() or not (ns.DEFAULT_FINAL/'manifest.json').exists():
        pytest.skip('requires published archived RQ3 artifacts')
    cfg=config();cfg.update(seeds=[42,43],batches_per_seed=8,namespace=2026102202,run_id='isolated_noise_fixture')
    import src.rq3_prediction_pilot as ridge
    import src.rq3_lstm_pilot as lstm
    def forbidden(*args,**kwargs):raise AssertionError('refitting forbidden')
    for name in ['fit_preprocessor','fit_score','ridge_path','select_model']:monkeypatch.setattr(ridge,name,forbidden)
    monkeypatch.setattr(lstm,'train_model',forbidden)
    original=ns.load_inputs;loaded=[]
    def guarded(root,seed,split,**kwargs):
        loaded.append(split);assert split=='train'
        return original(root,seed,split,**kwargs)
    monkeypatch.setattr(ns,'load_inputs',guarded)
    output=tmp_path/'run'
    for seed in cfg['seeds']:
        result=ns.run_seed(data,ns.DEFAULT_RIDGE,ns.DEFAULT_LSTM,ns.DEFAULT_FINAL,cfg,output,'0'*40,'0'*40,seed)
        assert result['models_refitted']==0
    with pytest.raises(FileExistsError):ns.run_seed(data,ns.DEFAULT_RIDGE,ns.DEFAULT_LSTM,ns.DEFAULT_FINAL,cfg,output,'0'*40,'0'*40,42)
    ns.summarize(output,cfg)
    from scripts.verify_rq3_noise_sensitivity import verify
    result=verify(output,data,ns.DEFAULT_RIDGE,ns.DEFAULT_LSTM,ns.DEFAULT_FINAL)
    assert result['status']=='pass' and result['unique_batches']==16
    assert set(loaded)=={'train'}
