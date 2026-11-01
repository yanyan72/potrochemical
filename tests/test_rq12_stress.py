"""Core scientific invariants for scoring-only studies."""
import json
from pathlib import Path
import numpy as np
import yaml
from src.rq12_stress import intervene,schedule,metric_rows,score,fresh_fixed
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/rq12_20261101.json').read_text())['rq2']
CAND={'variance_inflation_kappa':4.,'fault_initial':[.97,.01,.01,.01], 'fault_transition':[[.997,.001,.001,.001],[.1,.9,0,0],[.1,0,.9,0],[.1,0,0,.9]],'record_correct_likelihood':.9}
MODEL={'center':[0,0,0],'scale':[1,1,1],'mu':[[0,0,0],[1,1,1],[2,2,2]],'cov':[np.eye(3).tolist()]*3,'transition':[[.98,.01,.01],[.01,.98,.01],[.01,.01,.98]],'initial':[.34,.33,.33]}

def test_intervention_exact_mask_and_source():
 x=np.zeros((8,80,3));r=np.zeros((8,80),int);r[:,30:]=1
 starts,signs,offsets=schedule(r,42,CFG);assert len(set(map(tuple,np.c_[signs,offsets])))==8
 y,m=intervene(x,r,np.ones((3,3)),[0,2],4,starts,signs,24)
 assert m.sum()==8*24*2;np.testing.assert_array_equal(y[~m],x[~m]);assert not x.any();assert set(np.abs(y[m]))=={4}
 np.testing.assert_array_equal(schedule(r,42,CFG)[1],signs)

def test_causality_and_normalized_modes():
 x=np.random.default_rng(101).normal(size=(2,50,3));r=np.zeros((2,50),int);r[:,25:]=1
 for arm in ['joint12','soft3','hard']:
  a,q=score(x,r,MODEL,CAND,CFG,arm);b,z=score(x[:,:31],r[:,:31],MODEL,CAND,CFG,arm)
  np.testing.assert_allclose(a[:,:31],b,atol=1e-12);np.testing.assert_allclose(q[:,:31],z,atol=1e-12);np.testing.assert_allclose(q.sum(-1),1,atol=1e-12)

def test_no_fault_reduction():
 x=np.random.default_rng(13).normal(size=(2,30,3));r=np.zeros((2,30),int);c=dict(CAND,variance_inflation_kappa=0)
 a,q=score(x,r,MODEL,c,CFG,'joint12');b,z=score(x,r,MODEL,c,CFG,'soft3')
 np.testing.assert_allclose(q,z,atol=1e-12);np.testing.assert_allclose(a,b,atol=1e-10)

def test_both_channel_and_bystander_counts():
 mask=np.zeros((1,10,3),bool);mask[:,2:6,:2]=True;a=mask.copy();a[:,3,2]=True;q=np.zeros((1,10,3));q[:,:,0]=1
 rows=metric_rows(a,np.zeros_like(a),q,q,mask,np.zeros((1,10),int),np.array([2]),np.array([[1,-1]]),np.array([0]),4)
 r=rows[0];assert r['tp']==8 and r['fault_cells']==8;assert r['bystander_fp']==1 and r['bystander_cells']==4
 assert r['event_added_channels']==2 and r['all_channels_added']==1 and r['exact_set_added']==0
 assert r['posterior_tv_sum']==0 and r['true_p_loss_sum']==0

def test_independent_stream_and_original_equations():
 c=yaml.safe_load((ROOT/'configs/logistics_e1b.yaml').read_text());x,r,ids=fresh_fixed(c,42,2026110102,2);y,rr,j=fresh_fixed(c,42,2026110102,2);z,_,_=fresh_fixed(c,42,2026110103,2)
 np.testing.assert_array_equal(x,y);assert not np.array_equal(x,z);assert len(set(ids))==len(ids)
 a,_,_=fresh_fixed(c,42,3,2);rng=np.random.default_rng(np.random.SeedSequence([42,3,0,0]));latent=np.empty(160);latent[0]=rng.normal()
 for t in range(1,160):latent[t]=c['ar_rho']*latent[t-1]+np.sqrt(1-c['ar_rho']**2)*rng.normal()
 loading=np.array(c['factor_loadings']);v=latent[:,None]*loading+rng.normal(size=(160,3))*np.sqrt(1-loading**2)
 expected=np.array(c['regimes']['warehouse']['mean'])+np.array(c['regimes']['warehouse']['std'])*v
 np.testing.assert_array_equal(a[0],expected)
