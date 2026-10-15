"""EXP043: human QuAC source custody, abstention and scorer checks (model-free)."""
import sys,pathlib,unittest,hashlib
P=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(P))
import k2_quac_independent_fixture as q
import k2_quac_abstention as exp

class TestEXP043(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=q.build()
    def test_official_source_hash_and_cohort(self):
        self.assertEqual(hashlib.sha256(q.SOURCE.read_bytes()).hexdigest(),q.SOURCE_SHA)
        self.assertEqual(q.receipt(self.cases),exp.COHORT_SHA)
        self.assertEqual(len(self.cases),36)
        self.assertEqual(len({r["title"] for r in self.cases}),12)
        self.assertEqual({k:sum(r["kind"]==k for r in self.cases)
          for k in q.CONDITIONS},
          {"answerable":12,"unanswerable":12,"withheld":6,"swapped":6})
    def test_human_gold_supported_and_unknown_labels(self):
        for row in self.cases:
            if row["kind"]=="answerable":
                self.assertIn(row["gold"],row["context"])
                self.assertNotEqual(row["gold"],"UNKNOWN")
                self.assertNotIn(row["gold"].casefold(),row["history"].casefold())
            elif row["kind"]=="unanswerable":
                self.assertEqual(row["gold"],"UNKNOWN")
                self.assertGreater(len(row["context"]),300)
            else:
                self.assertEqual(row["gold"],"UNKNOWN")
                self.assertNotIn("Prior dialog:",q.prompt(row))
    def test_prompt_content_and_history(self):
        for row in self.cases:
            s=q.prompt(row)
            self.assertIn("Passage:\n",s)
            self.assertTrue(s.endswith("\nAnswer:"))
            self.assertNotIn("\nA)",s)
            self.assertNotIn("\nB)",s)
            self.assertIn("Current question:",s)
    def test_strict_and_lenient_scoring(self):
        ans=next(c for c in self.cases if c["kind"]=="answerable")
        good=q.score(ans,ans["gold"])
        self.assertEqual(good["strict"],1)
        self.assertEqual(good["f1"],1.)
        extra=q.score(ans,"The answer is "+ans["gold"]+".")
        self.assertEqual(extra["strict"],0)
        self.assertLess(extra["f1"],1.)
        bad=next(c for c in self.cases if c["kind"]=="unanswerable")
        self.assertEqual(q.score(bad,"UNKNOWN")["strict"],1)
        self.assertEqual(q.score(bad,"No answer is given")["strict"],0)
    def test_aggregate_fixed_cohort(self):
        scored=[q.score(x,x["gold"],hit_newline=True) for x in self.cases]
        grouped=q.aggregate(self.cases,scored)
        self.assertEqual(sum(d["n"] for d in grouped.values()),36)
        self.assertEqual(sum(d["strict"] for d in grouped.values()),36)
        self.assertTrue(all(d["mean_f1"]==1. for d in grouped.values()))
        for k in grouped:grouped[k]["native_eos"]=0
        report=exp.report({"default":grouped,"yarn":grouped},[312]*72,33.)
        self.assertIn(exp.PROTOCOL_SHA,report)
        self.assertIn("Human-annotated QuAC",report)
        self.assertIn("withheld",report)
    def test_frozen_checkpoint_and_protocol(self):
        self.assertEqual(exp.PROTOCOL_SHA,"90e252068fc23caeb233ba08164073f311d3495dfcbb8927cc94c4f1e47d7e69")
        self.assertEqual(len(exp.frozen_sources()),36)
if __name__=="__main__":unittest.main()
