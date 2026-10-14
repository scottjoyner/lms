"""EXP030 comparison receipt, source integrity, no unsupported answer claims."""
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
import k2_human_output_comparison as compare
RESEARCH=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
EXPECTED='a47b18bc55bc8f5de83be03726ee4161973e8231684d2f65ebe1eac7b524afc0'
class TestMeasuredEXP030Comparison(unittest.TestCase):
 def test_measured_markdown_exact_sha(self):
  self.assertEqual(compare.sha(RESEARCH/'EXPERIMENT-030-COMPARISON.md'),EXPECTED)
 def test_reproducible_comparison_and_known_counts(self):
  measured=compare.inputs(RESEARCH)
  self.assertEqual(set(measured),set(compare.REPORTS))
  counts=measured['strict32']
  self.assertTrue(counts)
  self.assertEqual((counts['short']['exact'],counts['short']['first_gold'],
                    counts['short']['unknown'],counts['short']['caps']),(0,5,0,16))
  self.assertEqual((counts['gap']['exact'],counts['gap']['first_gold'],
                    counts['gap']['unknown'],counts['gap']['caps']),(0,5,0,16))
  budget=measured['budget64']
  self.assertEqual((budget['short']['lenient'],budget['short']['first_gold'],
                    budget['short']['caps']),(8,6,9))
  self.assertEqual((budget['gap']['lenient'],budget['gap']['first_gold'],
                    budget['gap']['caps']),(7,4,13))
  self.assertEqual(compare.render(measured),(RESEARCH/'EXPERIMENT-030-COMPARISON.md').read_text())
 def test_fail_closed_on_tampered_scientific_report(self):
  with tempfile.TemporaryDirectory() as d:
   tmp=Path(d)
   for file in (compare.BASE[0],'EXPERIMENT-030-PROTOCOL.md'):
    (tmp/file).write_bytes((RESEARCH/file).read_bytes())
   for name,_ in compare.REPORTS.values():
    (tmp/name).write_bytes((RESEARCH/name).read_bytes())
   with (tmp/compare.REPORTS['strict32'][0]).open('a') as f:f.write('extra')
   with self.assertRaises(RuntimeError):compare.inputs(tmp)
 def test_not_confusing_first_word_with_strict_answer(self):
  s=(RESEARCH/'EXPERIMENT-030-COMPARISON.md').read_text()
  self.assertIn('Never count first-word match as whole',s)
  self.assertIn('eight-source',s)
  self.assertIn('K2 grounded QA stays NO-GO',s)
