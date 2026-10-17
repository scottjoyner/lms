"""Analytic EXP028A rotary phase evidence regression; no model weights loaded."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import k2_rotary_phase as phase

R=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
EXPECTED='3612d1e0012e79233fc4f25ec03884480cdc4fbc415bb6e57047dc477d29404b'
class NativeRotaryPhaseTests(unittest.TestCase):
 def test_analytic_zero_equal_schedules(self):
  inv=torch.linspace(.1,1.,32)
  for gap,mean,max_,rms,changed in phase.phase_diagnostics(inv,inv):
   self.assertAlmostEqual(mean,0.,places=12)
   self.assertAlmostEqual(max_,0.,places=12)
   self.assertLess(rms,1e-6)
   self.assertEqual(changed,0)
 def test_valid_input_guard(self):
  with self.assertRaises(ValueError):
   phase.phase_diagnostics(torch.ones(31),torch.ones(32))
 def test_source_sha_and_report_table(self):
  self.assertEqual(phase.sha(R/'results_exp018_intermediate_k2_20261014.json'),phase.REF_SHA)
  p=R/'EXPERIMENT-028A-ROTARY-PHASE.md'
  self.assertEqual(phase.sha(p),EXPECTED)
  contents=p.read_text()
  self.assertIn('| 384 | 0.649853 | 2.999577 | 0.904647 | 7/32 |',contents)
  self.assertIn('| 128 | 0.248937 | 1.298580 | 0.454921 | 0/32 |',contents)
  self.assertIn('NO neural inference',contents)
  self.assertIn('not a completed long-context model test',contents)
