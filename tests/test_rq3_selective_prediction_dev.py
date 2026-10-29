import unittest
import numpy as np
from rq3_selective_prediction_dev import trigger,selective_inputs
from rq3_joint_prediction_dev import base_features,scaler_fit,inputs_for

class SelectiveTests(unittest.TestCase):
    def test_trigger_boundary(self):
        t=np.exp(-.5);s=np.array([0.,t/2,t,1.]);np.testing.assert_allclose(trigger(s,t),[1,.5,0,0])
    def test_acceptance_and_common_features(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(2,8,5,3));b=dict(x=x,r=np.zeros(x.shape[:-1],int),S=np.ones_like(x),oldS=np.ones_like(x),ablationS=np.ones_like(x),q=np.full_like(x,1/3),center=np.full_like(x,2),ablation_center=np.full_like(x,-1));sc=scaler_fit(base_features(b));random=np.ones(x.shape[1:]);before=x.copy()
        for arm in ['SG','SR','SA']:np.testing.assert_array_equal(selective_inputs(b,arm,sc,random,np.exp(-.5)),inputs_for(b,'N',sc,random))
        b['S'][:]=0;z=selective_inputs(b,'SG',sc,random,np.exp(-.5));expected=(b['center']-np.array(sc['center'])[:3])/np.array(sc['scale'])[:3];np.testing.assert_allclose(z[...,:3],expected,atol=1e-6);np.testing.assert_array_equal(x,before)
    def test_monotonicity_and_bounds(self):
        s=np.linspace(0,1,1001);w=trigger(s,np.exp(-.5));self.assertTrue(np.all(np.diff(w)<=0));self.assertTrue(np.all((w>=0)&(w<=1)))
    def test_invalid(self):
        for s in [np.array([np.nan]),np.array([-.1]),np.array([1.1])]:
            with self.assertRaises(ValueError):trigger(s,.6)

if __name__=='__main__':unittest.main()
