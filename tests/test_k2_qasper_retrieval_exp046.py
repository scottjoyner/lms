"""EXP046: paper-disjoint BM25 retrieval, correct evidence, immutable cohort."""
import pathlib
import sys
import unittest
R=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(R))
import k2_qasper_retrieval_fixture as fx
import k2_qasper_retrieval_eval as runner
from k2_exp046_reporting import hit_conditioned,report
from k2_qasper_evidence_fixture import build as exp045_build

class TestEXP046(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.stats=fx.examples()
    def test_no_paper_overlap_with_exp045(self):
        a={x["pid"] for x in exp045_build()}
        b={x["pid"] for x in self.rows}
        self.assertEqual(len(a),12)
        self.assertEqual(len(b),12)
        self.assertFalse(a&b)
    def test_source_and_receipt(self):
        self.assertEqual(len(self.rows),60)
        self.assertEqual(fx.receipt(self.rows,self.stats),runner.COHORT_SHA)
        self.assertEqual({k:sum(x["kind"]==k for x in self.rows) for k in fx.KINDS},
                         {k:12 for k in fx.KINDS})
        self.assertEqual(len(runner.frozen_sources()[0]),60)
    def test_retrieval_gold_coverage(self):
        d=fx.retrieval_audit(self.stats)
        self.assertEqual((d["answer_gold_any_index"],d["answer_gold_top1"],
                          d["answer_gold_top3"]),(12,3,7))
        for r in self.rows:
            if r["kind"]=="oracle_supported":
                self.assertIn(r["gold"],r["context"])
            if r["kind"].startswith("bm25_unanswer"):
                self.assertEqual(r["gold"],"UNKNOWN")
    def test_bm25_reproducibility(self):
        docs=["a cat sat on a tree","the quick cat jumped","dogs and rain water"]
        first=fx.rank("quick cat",docs)
        self.assertEqual(first[0],1)
        self.assertEqual(first,fx.rank("quick cat",docs))
        self.assertEqual(sorted(first),[0,1,2])
    def test_prompt_formation(self):
        for r in self.rows:
            txt=fx.prompt(r)
            self.assertTrue(txt.endswith("\nAnswer:"))
            self.assertIn("\nEvidence:\n",txt)
            self.assertIn(r["question"],txt)
    def test_reporting_conditioned_counts(self):
        fake=[fx.score(r,r["gold"],hit_newline=True) for r in self.rows]
        for item in fake:item["native_eos"]=0
        d=hit_conditioned(self.rows,fake,self.stats)
        self.assertEqual(d[("bm25_answer_top1","gold_in_source")]["n"],3)
        self.assertEqual(d[("bm25_answer_top3","gold_missing")]["n"],5)
        counts=fx.aggregate(self.rows,fake)
        for k in fx.KINDS:counts[k]["native_eos"]=0
        s=report({"default":counts,"yarn":counts},[245]*120,17.3,
                 self.stats,{"default":d,"yarn":d})
        self.assertIn("Full-paper BM25",s)
        self.assertIn(runner.PROTOCOL_SHA,s)
        self.assertIn(runner.COHORT_SHA,s)
    def test_strict_score(self):
        a=next(x for x in self.rows if x["kind"]=="oracle_supported")
        self.assertEqual(fx.score(a,a["gold"])["strict"],1)
        self.assertEqual(fx.score(a,"Answer: "+a["gold"])["strict"],0)
        b=next(x for x in self.rows if x["kind"]=="bm25_unanswer_top1")
        self.assertEqual(fx.score(b,"UNKNOWN")["strict"],1)
if __name__=="__main__":unittest.main()
