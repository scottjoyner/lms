"""EXP034 fail-closed Markdown-only scientific source reconciliation."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import k2_evidence_swap_compare as m
R=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
EXPECTED='b16671fe3d50e8da57571e9911f9e3a54128e948c3d789b54c1282149ac0d54b'
class Compare(unittest.TestCase):
 def test_immutable_markdown_receipt(self):
  self.assertEqual(m.sha(R/'EXPERIMENT-034-COMPARISON.md'),EXPECTED)
  self.assertEqual(set(m.inputs(R)),set(m.REPORTS))
 def test_real_full_logits_human_negative_pattern(self):
  d=m.inputs(R)
  k=d['K2 plain'];q=d['Qwen native chat']
  self.assertEqual((k['answerable']['original'],k['unanswerable']['original']), (8,5))
  self.assertEqual((q['answerable']['original'],q['unanswerable']['original']), (8,1))
  self.assertEqual([d[model][group]['positive'] for model in d for group in ('answerable','unanswerable')],[8,8,8,8])
  self.assertEqual(m.render(d),(R/'EXPERIMENT-034-COMPARISON.md').read_text())
 def test_corrupt_source_denied_without_partial_report(self):
  with tempfile.TemporaryDirectory() as folder:
   tmp=Path(folder)
   for name,_ in m.REPORTS.values():
    (tmp/name).write_bytes((R/name).read_bytes())
   (tmp/'EXPERIMENT-034-PROTOCOL.md').write_bytes((R/'EXPERIMENT-034-PROTOCOL.md').read_bytes())
   (tmp/m.REPORTS['K2 plain'][0]).write_text('BAD')
   with self.assertRaises(RuntimeError):m.inputs(tmp)
 def test_no_cross_model_absolute_logprob_ranking(self):
  report=(R/'EXPERIMENT-034-COMPARISON.md').read_text()
  self.assertIn('Do not compare the absolute mean logP',report)
  self.assertIn('K2 grounded QA remains NO-GO',report)
