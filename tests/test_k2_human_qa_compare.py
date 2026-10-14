"""EXP029 exact-report source-custody and no-raw-data comparative audit."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import k2_human_qa_compare as c

ROOT=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
EXPECTED_SHA='6d78895839a4efd9c6962c7942a4b75cc69f8a70f0afa6d81a7fd4535ae44a1e'
class HumanQaCompare(unittest.TestCase):
 def test_report_sha_and_precise_source_matches(self):
  self.assertEqual(c.digest(ROOT/'EXPERIMENT-029-COMPARISON.md'),EXPECTED_SHA)
  self.assertEqual(set(c.inputs(ROOT)),set(c.SHA))
 def test_counts_and_template_provenance(self):
  x=c.inputs(ROOT)
  a=x['K2'];b=x['Qwen (native chat)']
  self.assertEqual([a[p]['exact'] for p in c.POSITIONS],[0,0,0])
  self.assertEqual([a[p]['contains'] for p in c.POSITIONS],[7,8,6])
  self.assertEqual([a[p]['unknown'] for p in c.POSITIONS],[0,0,0])
  self.assertEqual([a[p]['caps'] for p in c.POSITIONS],[9,11,14])
  self.assertEqual([b[p]['exact'] for p in c.POSITIONS],[5,5,5])
  self.assertEqual([b[p]['unknown'] for p in c.POSITIONS],[2,6,6])
  self.assertEqual([b[p]['caps'] for p in c.POSITIONS],[1,0,0])
 def test_human_markdown_replay_deterministic(self):
  x=c.inputs(ROOT)
  doc=c.render(x)
  self.assertEqual(doc,(ROOT/'EXPERIMENT-029-COMPARISON.md').read_text())
  self.assertIn('different published',doc)
  self.assertNotIn('input_ids',doc)
 def test_tampered_source_denied(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for name,_ in c.SHA.values():
    (root/name).write_bytes((ROOT/name).read_bytes())
   (root/c.SHA['K2'][0]).write_text('corrupt input')
   with self.assertRaises(RuntimeError):c.inputs(root)
