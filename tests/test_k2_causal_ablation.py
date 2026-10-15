"""EXP038: bounded causal hook math and selection integrity, no model writes."""
import importlib.util, pathlib, sys, unittest
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("k2_causal_ablation",ROOT/"k2_causal_ablation.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TestEXP038(unittest.TestCase):
    def test_selected_vs_ranked_random_disjoint(self):
        rng=np.random.default_rng(7)
        a=rng.normal(size=(24,5120)).astype(np.float32)
        a[:,13]*=100
        t,r,info=m.choose_units(a,7)
        self.assertEqual(len(t),32)
        self.assertEqual(len(r),32)
        self.assertIn(13,t.tolist())
        self.assertFalse(set(t.tolist()) & set(r.tolist()))
        self.assertGreater(info["top32_energy"],info["random32_energy"])
    def test_intervention_changes_only_last_token(self):
        x=torch.arange(1*3*5120,dtype=torch.float32).reshape(1,3,5120)
        targeted=np.arange(32)
        control=np.arange(32,64)
        selection={7:(targeted,control,{}),14:(targeted,control,{})}
        k,h=m.hook_for("layer7_top32",selection)
        self.assertEqual(k,7)
        out=h(None,(x,))[0]
        self.assertTrue(torch.equal(out[:,:-1,:],x[:,:-1,:]))
        self.assertTrue(torch.equal(out[:,-1,32:],x[:,-1,32:]))
        self.assertEqual(int(torch.count_nonzero(out[:,-1,:32])),0)
        self.assertNotEqual(float(x[0,-1,0]),0.)
        self.assertTrue(torch.equal(x,torch.arange(15360,dtype=torch.float32).reshape(1,3,5120)))
    def test_noop_is_identity(self):
        lay,fn=m.hook_for("identity",{})
        x=torch.ones(1,2,5120)
        self.assertEqual(lay,7)
        self.assertIs(fn(None,(x,))[0],x)
    def test_parity_fail_closed(self):
        a=np.array([1.,2.,3.]);p=np.array([.1,.2,.7])
        reference=[(a.copy(),p.copy(),2,True)]*10
        self.assertEqual(m.summarize(reference,reference,"identity")["flips"],0)
        mutated=list(reference)
        mutated[0]=(a+0.0001,p,2,True)
        with self.assertRaises(RuntimeError):
            m.summarize(mutated,reference,"identity")
    def test_eval_selection_receipt(self):
        dev=m.dev_cohort()
        test=m.evaluation_cohort()
        self.assertEqual(len(dev),24)
        self.assertEqual(len(test),10)
        self.assertEqual(m.DEV_SHA,"c1eac7118a917cc4505795767640e8321c671a346a41fa7bb9a22bf74e95577e")
        self.assertEqual(m.EVAL_SHA,"fbf36a541ab0622ccca046390e9da24b0602393f501a2f40a68825e17f14045c")
if __name__=="__main__":unittest.main()
