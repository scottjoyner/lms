"""EXP045 frozen QASPER scientific evidence, human labels and decoy controls."""
import sys,pathlib,unittest,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(ROOT))
import k2_qasper_evidence_fixture as fixture
import k2_qasper_evidence_eval as exp

class TestScientificEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=fixture.build()
    def test_official_source_and_exact_selection(self):
        self.assertEqual(hashlib.sha256(fixture.SOURCE.read_bytes()).hexdigest(),fixture.SOURCE_SHA)
        self.assertEqual(fixture.receipt(self.rows),exp.COHORT_SHA)
        self.assertEqual(len(self.rows),48)
        self.assertEqual(len({r["pid"] for r in self.rows}),12)
        self.assertEqual({k:sum(r["kind"]==k for r in self.rows) for k in fixture.KINDS},
                         {k:12 for k in fixture.KINDS})
    def test_human_evidence_and_gold_exclusion(self):
        for row in self.rows:
            if row["kind"]=="supported":
                self.assertIn(row["gold"],row["context"])
                self.assertNotEqual(row["gold"],"UNKNOWN")
            elif row["kind"]=="answerable_decoy":
                gold=next(r["gold"] for r in self.rows if r["kind"]=="supported" and r["pid"]==row["pid"])
                self.assertNotIn(gold.casefold(),row["context"].casefold())
                self.assertEqual(row["gold"],"UNKNOWN")
            elif row["kind"]=="human_unanswerable":
                self.assertEqual(row["gold"],"UNKNOWN")
                self.assertGreater(len(row["context"]),100)
            else:
                self.assertEqual(row["context"],"The passage was withheld.")
    def test_prompt_answer_and_no_choices(self):
        for row in self.rows:
            txt=fixture.prompt(row)
            self.assertIn("Excerpt:\n",txt)
            self.assertTrue(txt.endswith("\nAnswer:"))
            self.assertIn(row["question"],txt)
    def test_score_exact_and_lenient(self):
        row=next(r for r in self.rows if r["kind"]=="supported")
        self.assertEqual(fixture.score(row,row["gold"])["strict"],1)
        self.assertEqual(fixture.score(row,"Because "+row["gold"])["strict"],0)
        abstain=next(r for r in self.rows if r["kind"]=="human_unanswerable")
        self.assertEqual(fixture.score(abstain,"UNKNOWN.")["strict"],1)
        self.assertEqual(fixture.score(abstain,"No evidence")["strict"],0)
    def test_source_protocol_and_aggregate(self):
        self.assertEqual(len(exp.frozen_sources()),48)
        vals=[fixture.score(x,x["gold"],hit_newline=True) for x in self.rows]
        for v in vals:v["native_eos"]=0
        a=fixture.aggregate(self.rows,vals)
        for k in fixture.KINDS:
            self.assertEqual(a[k]["strict"],12)
            a[k]["native_eos"]=0
        out=exp.report({"default":a,"yarn":a},[151]*96,7.)
        self.assertIn(exp.PROTOCOL_SHA,out)
        self.assertIn(exp.COHORT_SHA,out)
        self.assertIn("QASPER",out)
if __name__=="__main__":unittest.main()
