"""EXP036 fail-closed, no-inference regression tests."""
import importlib.util
import pathlib
import sys
import unittest
import numpy as np
P=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location("k2_factorial_disjoint",P/"k2_factorial_disjoint.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class FactorialTests(unittest.TestCase):
    def test_article_disjoint_preflight(self):
        x=m.cohort()
        self.assertEqual(len(x),24)
        self.assertEqual(sum(y[1] for y in x),12)
        self.assertEqual(sum(y[-1]==0 for y in x),6)
        self.assertEqual(sum(y[-1]==1 for y in x),6)
        self.assertEqual(sum(y[-1]==2 for y in x),12)
        self.assertEqual(len({y[0] for y in x}),12)
    def test_cka_invariance(self):
        x=np.arange(24*8,dtype=float).reshape(24,8)
        self.assertAlmostEqual(m.cka(x,x),1,places=12)
        self.assertAlmostEqual(m.cka(x,3*x),1,places=12)
    def test_cka_rejects_wrong_n(self):
        with self.assertRaises(RuntimeError):
            m.cka(np.ones((23,5)),np.ones((23,7)))
    def test_report_custody_and_row_count(self):
        fake={"total":0,"answerable":0,"unknown":0,"nll":0.,
         "margin":0.,"A":0,"B":0,"C":0}
        rec={k:dict.fromkeys(("abs","top32","cos","cka","null"),0.)
             for k in m.LAYERS}
        t=m.render([(0.0,"default",fake,rec)])
        self.assertIn(m.PROTOCOL_SHA,t)
        self.assertIn(m.GROUP_SHA,t)
        self.assertIn("alpha",t.lower())
if __name__=="__main__":unittest.main()