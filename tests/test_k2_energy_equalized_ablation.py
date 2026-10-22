"""EXP039 deterministic energy-matched intervention tests, no model forwards."""
import importlib.util,pathlib,sys,unittest
import numpy as np
import torch
P=pathlib.Path(__file__).resolve().parents[1]/"research";sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location("k2_energy_equalized_ablation",P/"k2_energy_equalized_ablation.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_control_selection_reproducible(self):
        x=np.random.default_rng(10).normal(size=(24,5120)).astype(np.float32)
        x[:,0]*=90
        ta,ca=m.select(x,7);tb,cb=m.select(x,7)
        self.assertTrue(np.array_equal(ta,tb))
        self.assertEqual(len(ca),3)
        self.assertIn(0,ta)
        for a,b in zip(ca,cb):
            self.assertTrue(np.array_equal(a,b))
            self.assertEqual(len(a),32)
            self.assertFalse(set(ta)&set(a))
    def test_equalized_per_example_energy(self):
        x=torch.arange(1,1+5120*2,dtype=torch.float32).reshape(1,2,5120).to(torch.bfloat16)/100
        t=np.arange(200,232);r=np.arange(300,332)
        out_t,meta_t=m.modify(x,t,r,"top")
        out_r,meta_r=m.modify(x,t,r,"random")
        self.assertAlmostEqual(meta_t[0],meta_r[0],places=7)
        self.assertLess(meta_t[2],m.TOL)
        self.assertLess(meta_r[2],m.TOL)
        self.assertTrue(torch.equal(x[:,:-1,:],out_t[:,:-1,:]))
        self.assertTrue(torch.equal(x[:,:-1,:],out_r[:,:-1,:]))
        self.assertTrue(torch.equal(x[0,-1,:200],out_t[0,-1,:200]))
        self.assertTrue(torch.equal(x[0,-1,:300],out_r[0,-1,:300]))
    def test_math_fail_closed(self):
        x=torch.zeros(1,2,5120,dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):m.modify(x,np.arange(32),np.arange(32,64),"top")
    def test_report_has_all_pre_registered_pairs(self):
        p=m.pair_stats
        e=np.array([0.,1.,2.]);prob=np.array([.2,.3,.5])
        row=(e,prob,2,False)
        meta=(5.,100.,0.,0)
        val=p([row]*10,[row]*10,[row]*10,[meta]*10,[meta]*10)
        self.assertEqual((val['wins'],val['losses'],val['ties']),(0,0,10))
if __name__=="__main__":unittest.main()
