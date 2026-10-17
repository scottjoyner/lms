"""EXP047 original 8-paper cohort, frozen parser, paired scoring and guards."""
import unittest
import sys
import pathlib
R=pathlib.Path(__file__).resolve().parents[1]/'research';sys.path.insert(0,str(R))
import k2_qasper_decode_fixture as f
import k2_qasper_decode_eval as e
from k2_qasper_evidence_fixture import build as exp045
from k2_qasper_retrieval_fixture import examples as exp046
class Test047(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=f.cohort()
 def test_disjoint(self):
  old={r['pid'] for r in exp045()}|{r['pid'] for r in exp046()[0]}
  self.assertEqual(len(old),24)
  self.assertEqual(len({x['pid'] for x in self.rows}),8)
  self.assertFalse(old&{x['pid'] for x in self.rows})
 def test_custody(self):
  self.assertEqual(f.receipt(self.rows),e.COHORT_SHA)
  self.assertEqual(len(e.frozen()),16)
  self.assertEqual({k:sum(r['kind']==k for r in self.rows) for k in f.KINDS},{'supported':8,'human_unanswerable':8})
 def test_parser_no_gold_or_substring_guess(self):
  self.assertEqual(f.extract_span('Answer: NanoLab. More text'),'NanoLab')
  self.assertEqual(f.extract_span('The answer is UNKNOWN\nMore'),'UNKNOWN')
  self.assertEqual(f.extract_span('The value 3.14 is reported.'),'The value 3.14 is reported')
  self.assertEqual(f.extract_span('A: Sample; unrelated'),'Sample')
 def test_scoring_truth(self):
  ans=next(x for x in self.rows if x['kind']=='supported')
  sc=f.score(ans,'Answer: '+ans['gold']+'.',hit_newline=True)
  self.assertEqual(sc['strict'],0)
  self.assertEqual(sc['extracted_exact'],1)
  no=next(x for x in self.rows if x['kind']=='human_unanswerable')
  self.assertEqual(f.score(no,'UNKNOWN')['strict'],1)
 def test_aggregate_and_report(self):
  scores=[f.score(r,r['gold'],hit_newline=True) for r in self.rows]
  d=f.aggregate(self.rows,scores)
  self.assertEqual(d['supported']['strict'],8)
  self.assertEqual(d['human_unanswerable']['extracted_exact'],8)
  results={(m,c):d for m in f.MODES for c in f.CAPS}
  pairs={(m,k):dict(strict_gained=0,strict_lost=0,span_gained=0,span_lost=0,prefix_equal=8) for m in f.MODES for k in f.KINDS}
  s=e.report(results,pairs,None,25.,[180]*16)
  self.assertIn(e.PROTOCOL_SHA,s)
  self.assertIn('64 actual',s)
if __name__=='__main__':unittest.main()
