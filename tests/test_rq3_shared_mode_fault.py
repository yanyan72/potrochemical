"""Mathematical invariants; independent tiny fixtures, no study data evaluation."""
import copy,json,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rq3_shared_mode_fault import joint_filter
from rq3_uncertainty_a0 import mode_filter

class SharedPosteriorTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/rq3_shared_mode_fault_candidate_20261029.json').read_text())
        self.model=dict(center=[0,0,0],scale=[1,2,3],mu=[[0,0,0],[1,2,1],[-1,1,2]],
            cov=[[[1,.2,.1],[.2,1.4,.3],[.1,.3,.8]]]*3,
            transition=[[.97,.02,.01],[.01,.98,.01],[.01,.01,.98]],initial=[.8,.1,.1])
        self.x=np.random.default_rng(20991030).normal(size=(3,19,3))
        self.r=np.zeros((3,19),int);self.r[:,10:]=1

    def test_normalization_and_fault_mass(self):
        out=joint_filter(self.x,self.r,self.model,self.cfg)
        np.testing.assert_allclose(out['joint'].sum((-1,-2)),1,atol=1e-12)
        self.assertTrue(np.all(out['fault'][...,1:].sum(-1)<=1+1e-12))
        self.assertTrue(np.all(out['joint']>=0))

    def test_exact_no_inflation_reduction(self):
        expected=mode_filter(self.x,self.r,self.model,self.cfg)
        actual=joint_filter(self.x,self.r,self.model,self.cfg,kappa=0)['mode']
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)

    def test_causal_prefix(self):
        out=joint_filter(self.x,self.r,self.model,self.cfg)
        short=joint_filter(self.x[:,:12],self.r[:,:12],self.model,self.cfg)
        for key in out:np.testing.assert_allclose(out[key][:,:12],short[key],atol=1e-12,rtol=0)

    def test_channel_permutation(self):
        perm=[2,0,1];model=copy.deepcopy(self.model)
        for key in ('center','scale'):model[key]=np.asarray(model[key])[perm].tolist()
        model['mu']=np.asarray(model['mu'])[:,perm].tolist()
        model['cov']=np.asarray(model['cov'])[:,perm][:,:,perm].tolist()
        a=joint_filter(self.x,self.r,self.model,self.cfg)
        b=joint_filter(self.x[:,:,perm],self.r,model,self.cfg)
        np.testing.assert_allclose(a['mode'],b['mode'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(a['fault'][:,:,[0,3,1,2]],b['fault'],atol=1e-12,rtol=0)

    def test_inputs_unchanged_and_repeatable(self):
        oldx=self.x.copy();oldr=self.r.copy();a=joint_filter(self.x,self.r,self.model,self.cfg)
        b=joint_filter(self.x,self.r,self.model,self.cfg)
        np.testing.assert_array_equal(oldx,self.x);np.testing.assert_array_equal(oldr,self.r)
        np.testing.assert_array_equal(a['joint'],b['joint'])

    def test_identical_observation_causes_remain_ambiguous(self):
        a=joint_filter(self.x,self.r,self.model,self.cfg)
        b=joint_filter(self.x.copy(),self.r.copy(),self.model,self.cfg)
        # No hidden cause label is provided, so identical inputs cannot distinguish it.
        np.testing.assert_array_equal(a['joint'],b['joint'])

    def test_no_fault_memory_ablation_is_normalized(self):
        out=joint_filter(self.x,self.r,self.model,self.cfg,fault_memory=False)
        np.testing.assert_allclose(out['joint'].sum((-1,-2)),1,atol=1e-12)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):joint_filter(self.x,self.r,self.model,self.cfg,kappa=-1)
        x=self.x.copy();x[0,0,0]=np.nan
        with self.assertRaises(ValueError):joint_filter(x,self.r,self.model,self.cfg)

if __name__=='__main__':unittest.main(verbosity=2)
