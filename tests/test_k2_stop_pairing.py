"""EXP032C exact pairing chronology, synthetic controls, source integrity."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
import k2_stop_pairing as a
RESEARCH=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')

class Timeline(unittest.TestCase):
 def test_custody(self):
  self.assertEqual(a.sha(RESEARCH/'EXPERIMENT-032C-STOP-PAIRING-PROTOCOL.md'),a.PROTOCOL)
  self.assertEqual(a.sha(RESEARCH/'EXPERIMENT-032B-STOP-POSTHOC.md'),a.EXP032B_SHA)
 def test_alt_immediately_followed_by_native_eos(self):
  r=a.timeline([11,a.ALT,a.EOS,64255,64255])
  self.assertEqual((r['cell'],r['order'],r['adjacency'],r['cap']),
                   ('both','alt_before_eos','1',0))
 def test_alt_multiple_tokens_before_native_eos(self):
  r=a.timeline([11,a.ALT,20,25,30,a.EOS])
  self.assertEqual((r['cell'],r['order'],r['adjacency']),
                   ('both','alt_before_eos','>3'))
 def test_alt_without_eos_and_eos_without_alt(self):
  self.assertEqual(a.timeline([a.ALT]+[3]*31)['cell'],'alt_only')
  self.assertEqual(a.timeline([12,a.EOS,64255])['cell'],'eos_only')
  self.assertEqual(a.timeline([12]*32)['cell'],'neither')
  self.assertEqual(a.timeline([12]*32)['cap'],1)
 def test_ignore_PAD_after_eos_as_unemitted(self):
  r=a.timeline([12,a.EOS,64255,a.ALT])
  self.assertEqual((r['cell'],r['alt']),('eos_only',None))
 def test_pinned_source_subcohort_short(self):
  import k2_human_grounded_qa as prior
  import k2_posthoc_boundary as earlier
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained('/home/scott/git/.latent-k2-mopd_249-research',
              local_files_only=True,trust_remote_code=False)
  rows,_=earlier.preflight(RESEARCH,prior,tok)
  cohort=[r for r in rows if r['position']=='short']
  self.assertEqual(len(cohort),16)
  self.assertEqual(sum(r['impossible'] for r in cohort),8)
 def test_exact_report_if_completed(self):
  p=RESEARCH/'EXPERIMENT-032C-STOP-PAIRING.md'
  if not p.is_file():return
  text=p.read_text()
  self.assertIn(a.PROTOCOL,text)
  self.assertIn('Native EOS',text)
  self.assertIn('Strict UNKNOWN',text)
