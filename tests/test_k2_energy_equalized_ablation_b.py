"""EXP039B: strict pre-registered 24-step BF16 solver and integrity regression."""
import importlib.util
import pathlib
import sys
import unittest
import numpy as np
import torch
P=pathlib.Path(__file__).resolve().parents[1]/"research";sys.path.insert(0,str(P))
from k2_energy_equalized_ablation import modify as naive
spec=importlib.util.spec_from_file_location("k2_energy_equalized_ablation_b",P/"k2_energy_equalized_ablation_b.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_protocol_distinct_and_tolerance_unchanged(self):
        self.assertEqual(m.TOL,.005)
        self.assertEqual(m.PROTOCOL_SHA,"d377a47a7446170610f17346df0a109a98e1256ec1660e5e14b5ed7c6f2e112a")
    def test_exact_realized_budget_bfloat_and_no_mutation(self):
        g=torch.Generator().manual_seed(20261015)
        coords_a=np.arange(96,128);coords_b=np.arange(256,288)
        for i in range(30):
            v=torch.randn((1,2,5120),generator=g).to(torch.bfloat16)
            x=v.clone()
            y,a=m.modify(x,coords_a,coords_b,"top")
            z,b=m.modify(x,coords_a,coords_b,"random")
            self.assertEqual(a[0],b[0])
            self.assertLessEqual(a[2],.005)
            self.assertLessEqual(b[2],.005)
            self.assertTrue(torch.equal(y[:,:-1,:],x[:,:-1,:]))
            self.assertTrue(torch.equal(z[:,:-1,:],x[:,:-1,:]))
            self.assertTrue(torch.equal(x,v))
    def test_never_degrades_naive_matching(self):
        g=torch.Generator().manual_seed(425)
        t=np.arange(60,92);r=np.arange(300,332)
        for i in range(18):
            x=torch.randn(1,2,5120,generator=g).to(torch.bfloat16)
            for group in ("top","random"):
                improved,rec=m.modify(x,t,r,group)
                try:
                    old,oldrec=naive(x,t,r,group)
                    self.assertLessEqual(rec[2],oldrec[2]+1e-7)
                except RuntimeError:
                    # Original EXP039 fails closed; EXP039B must still
                    # honor the same numerical 0.5% threshold.
                    self.assertLessEqual(rec[2],.005)
if __name__=="__main__":unittest.main()
