"""EXP050 preregistered question/source 2x2 matched-length foil test."""
import pathlib
import sys
import unittest
P=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(P))
import k2_qasper_question_swap_fixture as fx
import k2_qasper_question_swap_eval as exp
class TestEXP050(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=fx.build()
    def test_receipts_and_paper_question_pair(self):
        self.assertEqual(fx.digest(self.rows),exp.COHORT_SHA)
        self.assertEqual(len(exp.frozen()),32)
        self.assertEqual(len({r["topic"] for r in self.rows}),8)
        for a,b in zip(self.rows[::2],self.rows[1::2]):
            self.assertEqual((a["kind"],b["kind"]),fx.KINDS)
            self.assertEqual(a["row"]["context"],b["row"]["context"])
            self.assertNotEqual(a["row"]["question"],b["row"]["question"])
            self.assertEqual((a["gold"],a["foil"]),(b["gold"],b["foil"]))
            self.assertEqual(a["n"],b["n"])
    def test_source_original_and_withheld_pairs(self):
        for j in range(0,32,4):
            a,b,c,d=self.rows[j:j+4]
            self.assertEqual((a["arm"],b["arm"],c["arm"],d["arm"]),
                             ("original","original","withheld","withheld"))
            self.assertEqual(a["row"]["question"],c["row"]["question"])
            self.assertEqual(b["row"]["question"],d["row"]["question"])
    def test_factorial_difference_in_differences(self):
        out={}
        for mode in fx.MODES:
            a=[]
            for r in self.rows:
                # Synthetic: gold is preferred more for the supported question
                # only in original evidence, with no question advantage if withheld.
                score=4. if r["arm"]=="original" and r["kind"]=="supported" else 1.
                a.append({"sum_margin":score,"mean_margin":score/2})
            out[mode]=a
        groups,q,d=exp.stats(self.rows,out)
        self.assertEqual((len(groups),len(q),len(d)),(8,4,2))
        self.assertEqual(d[0]["positive"],8)
        self.assertAlmostEqual(d[0]["mean"],3.)
        text=exp.report(groups,q,d,[200]*32,[6]*32,55.)
        self.assertIn("128 real",text)
        self.assertNotIn(self.rows[0]["row"]["question"],text)
    def test_repeat_baseline_fail_closed(self):
        fake=[{"mode":m,"arm":a,"kind":k,"mean":0,"median":0,"gold":0}
          for m in fx.MODES for a in fx.ARMS for k in fx.KINDS]
        with self.assertRaises(RuntimeError):
            exp.base_repeat_gate(fake)
if __name__=="__main__":unittest.main()
