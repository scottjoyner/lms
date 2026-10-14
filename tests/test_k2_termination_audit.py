"""EXP031 source and synthetic token-marker accounting; no live model needed."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
sys.path.insert(0,'/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
import k2_termination_audit as aud
import k2_human_grounded_qa as prior
R=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
class NativeStopWitness(unittest.TestCase):
 def test_registration_and_ids(self):
  self.assertEqual(aud.sha(R/'EXPERIMENT-031-PROTOCOL.md'),aud.PROTOCOL)
  self.assertEqual(aud.END_IDS,{'ifm_message_end':64019,'eot_id':64024,
    'end_of_thought':64026,'end_of_solution':64028,'assistant_end':64040})
 def test_exact_original_human_gold_subset(self):
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained('/home/scott/git/.latent-k2-mopd_249-research',
       local_files_only=True,trust_remote_code=False)
  rows=aud.preflight(R,prior,tok)
  self.assertEqual(len(rows),8)
  self.assertEqual(sum(r['impossible'] for r in rows),4)
  self.assertEqual({r['article_index'] for r in rows},{0,1,2,3})
  self.assertTrue(all(r['position']=='short' for r in rows))
 def test_model_terminator_synthetic_positive_and_negative_controls(self):
  special={1,64019,64024,64026,64028,64040,64255}
  r=aud.score_tokens([8,64028,27,1,64255],1,special)
  self.assertTrue(r['eos'])
  self.assertFalse(r['cap'])
  self.assertEqual(r['eos_pos'],3)
  self.assertEqual(r['marks']['end_of_solution'],[1])
  self.assertEqual(r['other_special_count'],0)
  r=aud.score_tokens([34]*32,1,special)
  self.assertFalse(r['eos'])
  self.assertTrue(r['cap'])
  self.assertFalse(any(r['marks'].values()))
 def test_published_terminator_audit_when_complete(self):
  report=R/'EXPERIMENT-031-TERMINATORS.md'
  if not report.exists():self.skipTest('Read-only live witness not yet complete')
  self.assertEqual(aud.sha(report),'a0e7db3d681eb81cc3880715aab9f9542948a4008b1e61516ad7e8c1499beb50')
  text=report.read_text()
  self.assertIn(aud.PROTOCOL,text)
  self.assertIn('| end_of_solution | 64028 |',text)
  self.assertIn('| ifm_message_end | 64019 | 1 | 1 | 13–13 |',text)
  self.assertIn('Native EOS observed: **1/8**.',text)
  self.assertIn('All 32 new tokens exhausted without native EOS: **7/8**.',text)
  self.assertIn('Any five audited alternate terminators: **1/8**.',text)
  self.assertIn('native EOS',text)
