import numpy as np
from src.rq2_factorial_fault import factorial_filter
from src.rq2_restricted_fault import restricted_filter
from test_rq12_stress import MODEL,CAND

def test_structural_ablation():
 x=np.random.default_rng(50).normal(size=(2,40,3));r=np.zeros((2,40),int)
 a=restricted_filter(x,r,MODEL,CAND,restrict=False);b=factorial_filter(x,r,MODEL,CAND)
 np.testing.assert_allclose(a['joint'],b['joint'],atol=1e-13)
 c=restricted_filter(x,r,MODEL,CAND);assert c['joint'].shape[-1]==4;assert (c['masks'].sum(-1)<=1).all();np.testing.assert_allclose(c['joint'].sum((-1,-2)),1,atol=1e-12)
 d=restricted_filter(x[:,:20],r[:,:20],MODEL,CAND);np.testing.assert_allclose(d['joint'],c['joint'][:,:20],atol=1e-12)
 z=dict(CAND,variance_inflation_kappa=0);np.testing.assert_allclose(restricted_filter(x,r,MODEL,z)['mode'],factorial_filter(x,r,MODEL,z)['mode'],atol=1e-12)
