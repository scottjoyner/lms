"""EXP048 source/hash custody and candidate likelihood tests; no real model loads."""
import pathlib
import sys
import unittest
import torch
P=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(P))
import k2_qasper_likelihood_fixture as fx
import k2_qasper_likelihood_eval as exp

class TestEXP048(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=fx.inputs()
    def test_frozen_custody(self):
        self.assertEqual(len(exp.frozen_preflight()),32)
        self.assertEqual(fx.digest(self.rows),exp.COHORT_SHA)
        self.assertEqual({(r["kind"],r["arm"]) for r in self.rows},
                         {(kind,arm) for kind in fx.CATEGORIES for arm in fx.ARMS})
        self.assertEqual(len({r["topic"] for r in self.rows}),8)
    def test_answer_candidate_and_source_pair(self):
        for i in range(0,len(self.rows),2):
            a,b=self.rows[i:i+2]
            self.assertEqual((a["kind"],a["topic"]),(b["kind"],b["topic"]))
            self.assertEqual((a["arm"],b["arm"]),("original","withheld"))
            self.assertEqual(a["choice_answer"],b["choice_answer"])
            self.assertEqual(a["choice_unknown"]," UNKNOWN")
            self.assertEqual(b["row"]["context"],fx.MISSING)
            self.assertEqual(a["row"]["question"],b["row"]["question"])
            if a["kind"]=="supported":
                self.assertEqual(a["row"]["gold"],a["choice_answer"].strip())
                self.assertIn(a["row"]["gold"],a["row"]["context"])
    def test_margins_and_paired_evidence_stats(self):
        scores={}
        for mode in fx.MODES:
            scores[mode]=[]
            for x in self.rows:
                m=2.0 if x["arm"]=="original" else -1.0
                if x["kind"]=="human_unanswerable":m=-m
                scores[mode].append({"margin_mean":m,"margin_sum":2*m})
        grouped,paired=exp.aggregate(self.rows,scores)
        self.assertEqual(len(grouped),8)
        self.assertEqual(len(paired),4)
        self.assertEqual(next(r for r in grouped if r["kind"]=="supported" and
          r["arm"]=="original")["answer_favored"],8)
        self.assertEqual(next(r for r in paired if r["kind"]=="supported")["positive"],8)
    def test_report_does_not_expose_source(self):
        vals={m:[{"margin_mean":.1,"margin_sum":.1} for _ in self.rows] for m in fx.MODES}
        group,pairs=exp.aggregate(self.rows,vals)
        txt=exp.report(group,pairs,[100]*32,[3]*64,8.1)
        self.assertIn("128 native",txt)
        self.assertIn(exp.PROTOCOL_SHA,txt)
        self.assertNotIn(self.rows[0]["row"]["question"],txt)
    def test_logprob_alignment_with_fake_model(self):
        class Fake:
            def __call__(self,input_ids,use_cache,return_dict):
                # Target token IDs [3,4]; every position's target is
                # strongly preferred to controls; candidate logP finite.
                ids=input_ids[0]
                logits=torch.zeros((1,len(ids),7),dtype=torch.float32)
                for j in range(len(ids)-1):
                    logits[0,j,ids[j+1]]=5.
                return type("O",(),{"logits":logits})()
        with unittest.mock.patch.object(exp,"available_gib",return_value=50):
            out=exp.candidate_logprob(Fake(),[1]*22,[3,4])
        self.assertEqual(out["n"],2)
        self.assertTrue(-.2<out["mean"]<0)
        self.assertAlmostEqual(out["sum"],2*out["mean"])
if __name__=="__main__":unittest.main()
