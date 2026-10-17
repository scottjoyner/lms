"""EXP041 no-model source, score, and publication fail-closed controls."""
import sys
import pathlib
import unittest
PATH=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(PATH))
import k2_novel_source_fixture as fixture
import k2_new_source_generation as run

class SourceTests(unittest.TestCase):
    def test_checksum_and_balance(self):
        cases=fixture.build()
        self.assertEqual(len(cases),24)
        self.assertEqual(fixture.corpus_receipt(cases),run.FIXTURE_SHA)
        self.assertEqual({k:sum(x["kind"]==k for x in cases)
          for k in fixture.CASES},
          {"answerable":8,"unanswerable":8,"redacted":4,"swapped":4})
        self.assertEqual(len({x["topic"] for x in cases}),8)
    def test_source_depends_on_passage(self):
        cases=fixture.build()
        for c in cases:
            if c["kind"]=="answerable":
                self.assertIn(c["gold"],c["source"])
                self.assertNotEqual(c["gold"],"UNKNOWN")
            elif c["kind"] in ("redacted","swapped"):
                self.assertEqual(c["gold"],"UNKNOWN")
                self.assertNotIn(" carried the "+fixture.ARTIFACTS[c["topic"]],c["source"])
            else:
                self.assertEqual(c["gold"],"UNKNOWN")
                self.assertNotIn("repaired",c["source"])
    def test_prompt_plain_text_and_not_choice_selection(self):
        for c in fixture.build():
            p=fixture.prompt(c)
            self.assertIn("\nPassage:\n",p)
            self.assertIn("\nQuestion: ",p)
            self.assertTrue(p.endswith("\nAnswer:"))
            self.assertNotIn("A)",p)
            self.assertNotIn("B)",p)
    def test_scoring_strict_vs_lenient(self):
        a=next(x for x in fixture.build() if x["kind"]=="answerable")
        ok=fixture.score(a,a["gold"]+".")
        self.assertEqual(ok["strict"],1)
        extra=fixture.score(a,"The answer is "+a["gold"]+".")
        self.assertEqual(extra["strict"],0)
        self.assertEqual(extra["contains_gold"],1)
        self.assertEqual(extra["first_line"],0)
        another=fixture.score(a,a["gold"]+"\nThis is more.")
        self.assertEqual(another["strict"],0)
        self.assertEqual(another["first_line"],1)
        unknown=next(x for x in fixture.build() if x["kind"]=="unanswerable")
        self.assertEqual(fixture.score(unknown,"  UNKNOWN.  ")["strict"],1)
        self.assertEqual(fixture.score(unknown,"Cannot determine")["strict"],0)
    def test_group_report_and_preflight_custody(self):
        c=fixture.build()
        scores=[fixture.score(x,x["gold"]) for x in c]
        result=fixture.score_group(c,scores)
        self.assertEqual([(k,result[k]["strict"]) for k in fixture.CASES],
                         [(k,result[k]["n"]) for k in fixture.CASES])
        self.assertEqual(run.frozen_sources(),c)
        output=run.render(c,{"default":result,"yarn":result},
                          [123]*48,10.1)
        self.assertIn(run.PROTOCOL_SHA,output)
        self.assertIn(run.FIXTURE_SHA,output)
        self.assertIn("not a human-annotated",output.lower())
if __name__=="__main__":unittest.main()
