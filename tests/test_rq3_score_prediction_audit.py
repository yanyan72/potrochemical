"""Check event and matched-prediction arithmetic without fitting models."""
import numpy as np
import pytest
from src.rq3_score_prediction_audit import event_vectors,prediction_rows


def scores(alarm):
    return dict(alarm=alarm,available=np.ones_like(alarm,bool),reliability=np.ones_like(alarm,float))


def test_continuing_alarm_is_not_fresh_and_bystanders_separate():
    alarm=np.zeros((2,8,3),bool);alarm[0,1:5,0]=True;alarm[1,3:5,0]=True;alarm[:,2:5,1]=True
    old=scores(np.zeros_like(alarm));obs=scores(alarm);obs['reliability'][:,:,0]=.5
    events=[dict(channel=0,start=2,end_exclusive=5)]*2
    mask=np.zeros_like(alarm);mask[:,2:5,0]=True
    v=event_vectors(old,obs,events,mask)
    assert list(v['any_alarm'])==[True,True] and list(v['fresh_alarm'])==[False,True]
    assert list(v['event_alarm_count'])==[3,2] and list(v['bystander_alarm_count'])==[3,3]
    assert np.array_equal(v['score_drop'],[.5,.5])
    mask[0,0,0]=True
    with pytest.raises(ValueError,match='mask'):event_vectors(old,obs,events,mask)


def test_missing_score_is_unavailable_not_zero_trust():
    a=np.zeros((1,8,3),bool);old=scores(a);obs=scores(a)
    obs['available'][0,2:5,0]=False;obs['reliability'][0,2:5,0]=np.nan
    mask=np.zeros_like(a);mask[0,2:5,0]=True
    v=event_vectors(old,obs,[dict(channel=0,start=2,end_exclusive=5)],mask)
    assert v['availability'][0]==0 and np.isnan(v['score_drop'][0]) and not v['fresh_alarm'][0]


def test_same_batch_prediction_difference_and_degradation():
    errors=np.ones((2,2,2,2,2));errors[:,0,1]=2
    errors[:,1,0]=3;errors[:,1,1]=1
    out=prediction_rows(errors,['original','bias_1'],np.array([0,1]),np.array([[False,False],[True,False]]),42,['arrival_proxy','null_proxy'])
    row=next(r for r in out if r['architecture']=='lstm' and r['view']=='bias_1' and r['stratum']=='all' and r['target']=='arrival_proxy')
    assert row['mae_difference']==-2 and row['incremental_degradation']==-3
    assert row['b2_degradation']==2 and row['b3_degradation']==-1 and row['improvement_fraction']==1
