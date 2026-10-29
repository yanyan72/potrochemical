"""Small mathematical/fairness checks; fixture data are not research samples."""
import unittest
import numpy as np
import torch
from rq3_joint_prediction_dev import inputs_for,base_features,scaler_fit,draw_random,fit,Predictor,events

class PredictionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(991);x=rng.normal(size=(2,8,5,3));q=np.ones_like(x)/3
        self.b=dict(x=x,r=np.zeros(x.shape[:-1],int),S=np.full_like(x,.4),oldS=np.full_like(x,.6),ablationS=np.full_like(x,.7),q=q,center=np.full_like(x,.5),ablation_center=np.full_like(x,-.3))
        self.sc=scaler_fit(base_features(self.b));self.random=np.full(x.shape[1:],.25)
    def test_common_features(self):
        arrays={a:inputs_for(self.b,a,self.sc,self.random) for a in ['N','OT','NT','ZG','NG','RG','AG']}
        for a,x in arrays.items():
            self.assertEqual(x.shape[-1],18);np.testing.assert_array_equal(x[...,3:15],arrays['N'][...,3:15])
        for a in ['OT','NT']:np.testing.assert_array_equal(arrays[a][...,:15],arrays['N'][...,:15])
    def test_gate_limits_and_reference_term(self):
        b={k:v.copy() for k,v in self.b.items()};b['S'][:]=1
        np.testing.assert_array_equal(inputs_for(b,'NG',self.sc,self.random),inputs_for(b,'N',self.sc,self.random))
        b['S'][:]=0;ng=inputs_for(b,'NG',self.sc,self.random);zg=inputs_for(b,'ZG',self.sc,self.random)
        np.testing.assert_array_equal(zg[...,:3],0)
        expected=(b['center']-np.array(self.sc['center'])[:3])/np.array(self.sc['scale'])[:3]
        np.testing.assert_allclose(ng[...,:3],expected,atol=1e-6)
    def test_random_train_only_and_no_self(self):
        ids=np.array(['a','b','c']);bank=np.arange(2*3*4*3).reshape(2,3,4,3)
        rand,ix=draw_random(ids,ids,bank,42,59);self.assertTrue(np.all(ix[:,1]!=np.arange(3)))
        np.testing.assert_array_equal(rand,bank[ix[:,0],ix[:,1]])
        v,j=draw_random(np.array(['val']),ids,bank,42,59);np.testing.assert_array_equal(v,bank[j[:,0],j[:,1]])
    def test_parameter_capacity(self):
        self.assertEqual(sum(p.numel() for p in Predictor().parameters()),6689)
    def test_tiny_budget_and_replay(self):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        x=inputs_for(self.b,'N',self.sc,self.random);y=np.arange(8,dtype=float)
        cfg=dict(model_seed=1,shuffle_seed=2,hidden_size=3,learning_rate=.003,weight_decay=.0001,epochs=2,batch_size=4,grad_clip=1.)
        schedule=np.zeros((2,8),int);s,p,h,pred=fit(x,y,x,y,schedule,cfg)
        s2,p2,h2,pred2=fit(x,y,x,y,schedule,cfg)
        self.assertEqual(p['optimizer_updates'],4);self.assertEqual(p['initial_sha256'],p2['initial_sha256']);np.testing.assert_array_equal(pred,pred2)
        self.assertEqual(len(h),2);self.assertEqual(sum(r['selected'] for r in h),1)
    def test_event_extension_and_no_input_mutation(self):
        r=np.zeros((24,118),int);r[:,30:]=1;ids=np.array([f'fixture_{i}' for i in range(24)])
        cfg=dict(event_seed=123);before=r.copy();ev=events(ids,r,[],cfg,42,'train')
        self.assertEqual({e['channel'] for e in ev},{0,1,2});self.assertTrue(all(e['end']-e['start']==12 for e in ev));np.testing.assert_array_equal(r,before)
        self.assertEqual(events(ids,r,ev[:12],cfg,42,'train')[:12],ev[:12])

if __name__=='__main__':unittest.main()
