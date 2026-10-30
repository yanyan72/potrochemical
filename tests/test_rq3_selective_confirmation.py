"""Focused checks of confirmation plumbing, not additional research fits."""
import json,unittest
from pathlib import Path
import numpy as np
import torch
import rq3_selective_confirmation as c
import rq3_joint_prediction_dev as b
import rq3_uncertainty_a0 as a0

class ConfirmationTests(unittest.TestCase):
    def test_outer_seed_statistics(self):
        r=c.paired_stats([-1,-2,-3,-4,-5]);self.assertEqual(r['mean'],-3);self.assertEqual(r['outer_seeds'],5);self.assertAlmostEqual(r['sd'],np.sqrt(2.5));self.assertLess(r['ci_high'],0)
    def test_balanced_events_are_label_free(self):
        ids=np.array([str(i) for i in range(24)]);rec=np.zeros((24,118),int);rec[:,30:]=1
        ev=c.fresh_events(ids,rec,{'test_event_seed':2026103002},42)
        self.assertEqual(len({(e['channel'],e['sign'],e['offset']) for e in ev}),12)
        self.assertTrue(all(e['end']-e['start']==12 for e in ev));self.assertEqual(ev,c.fresh_events(ids,rec,{'test_event_seed':2026103002},42))
    def test_explicit_lag6_and_frozen_features(self):
        root=Path('results/shared_mode_fault_development_20261029_v2');m=c.read(root/'seed_42/model.json');cc=c.read(root/'candidate_config.json');sc=c.read(root/'config.json')
        vc=c.read(Path('../a0_20261029/results/rq3_uncertainty_a0_20261029_v1/config.json'))
        v=a0.npz(Path('../a0_20261029/results/rq3_uncertainty_a0_20261029_v1/seed_42/val_correct_clean.npz'));x=v['observations'][:2];r=v['recorded_regime'][:2];ev=c.fresh_events(v['sequence_id'][:2],r,{'test_event_seed':12345},42)
        q=c.bundle(x,r,ev,np.ones((3,3)),m,cc,sc,vc,['correct_clean','delay6_clean','delay12_clean'])
        np.testing.assert_array_equal(q['r'][1],a0.delayed(r,6));self.assertFalse(np.array_equal(q['r'][1],q['r'][2]))
        scaler=b.scaler_fit(b.base_features(q));rnd=q['S'][0];cfg={'trigger_score':np.exp(-.5)}
        n=c.features(q,'N',scaler,rnd,cfg)
        for arm in ['NG','SG','SR','SA']:np.testing.assert_array_equal(c.features(q,arm,scaler,rnd,cfg)[...,3:],n[...,3:])
    def test_two_optimizer_streams_repeatable(self):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(123)
        x=rng.normal(size=(2,8,6,18)).astype(np.float32);y=rng.normal(size=8);schedule=np.zeros((2,8),dtype=np.int64)
        cfg=dict(hidden_size=32,batch_size=4,epochs=2,learning_rate=.003,weight_decay=.0001,grad_clip=1,model_seed=29001,shuffle_seed=39001)
        a=b.fit(x,y,x,y,schedule,cfg);aa=b.fit(x,y,x,y,schedule,cfg);np.testing.assert_array_equal(a[3],aa[3]);bb=b.fit(x,y,x,y,schedule,{**cfg,'model_seed':29002,'shuffle_seed':39002})
        self.assertNotEqual(a[1]['initial_sha256'],bb[1]['initial_sha256']);self.assertNotEqual(a[1]['shuffle_sha256'],bb[1]['shuffle_sha256']);self.assertEqual(a[1]['parameter_count'],6689)

if __name__=='__main__':unittest.main()
