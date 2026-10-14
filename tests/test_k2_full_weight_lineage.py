"""EXP025 fixed SHA and all-255 model parameter closure, synthetic aggregate controls."""
import hashlib
import importlib.util
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('k2_full_weight_lineage',ROOT/'research/k2_full_weight_lineage.py')
import sys
sys.path.insert(0,str(ROOT/'research'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
RESEARCH=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
REPORT=RESEARCH/'EXPERIMENT-025-FULL-WEIGHT-DRIFT.md'
SHA='169507cd918359b189abf208c5708af5a301a8017096b23642864f455f86821a'

class TestFullK2Weights(unittest.TestCase):
 def test_complete_255_and_1b_accounting(self):
  self.assertEqual(m.EXPECTED_TOTAL,1078285824)
  self.assertEqual(m.EXPECTED_LAYERS,880889856)
  self.assertEqual(m.EXPECTED_TOTAL-m.EXPECTED_LAYERS,197395968)
  self.assertEqual(len(m.EXTRAS),3)
  self.assertEqual(len(m.EXPECTED_COS),4)
 def test_report_sha_and_complete_exact_native_stages(self):
  self.assertEqual(hashlib.sha256(REPORT.read_bytes()).hexdigest(),SHA)
  t=REPORT.read_text()
  for stage in m.drift.STAGES:
   self.assertIn(stage,t)
  self.assertIn('1,078,285,824',t)
  self.assertIn('all 255',t.lower())
 def test_report_drifts(self):
  s=REPORT.read_text()
  self.assertIn('| mid2_47684 → rl_merged | 0.999985252 | 0.00127096 | 29.99% |',s)
  self.assertIn('| rl_merged → mopd_249 | 0.999999952 | 0.00006913 | 3.93% |',s)
  self.assertIn('1.94%',s)
 def test_reconciled_previous_tensor_scope(self):
  for edge,(cosine,l2) in m.EXPECTED_COS.items():
   self.assertGreater(cosine,0.9)
   self.assertGreaterEqual(l2,0)
  self.assertEqual(len({m.drift.tname(i,k) for i in range(28) for k in m.drift.KINDS}),252)
 def test_failclosed_low_ram(self):
  with self.assertRaises(RuntimeError):
   m.check_host_ram(10**6)
