"""EXP026 fixed RoPE factorial under the separate EXP019 fact-neutral fixture."""
import hashlib
import importlib.util
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('k2_rope_fact_control',ROOT/'research/k2_rope_fact_control.py')
import sys
sys.path.insert(0,str(ROOT/'research'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
R=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
OUT=R/'EXPERIMENT-026-FACT-ROPE.md'
EXPECTED_SHA='4452696eb2b567332112b392baac941a403c718053ccb33377dbe6cca020653c'

class FactNeutralRoPETests(unittest.TestCase):
 def test_two_axes_complete(self):
  self.assertEqual(set(m.STAGES),{'mid2_47684','rl_merged'})
  self.assertEqual(set(m.ROPES),{'default','yarn'})
 def test_pinned_prior_and_source(self):
  self.assertEqual(m.sha(R/'results_exp019_fact_bias_20261014.json'),m.PRIOR_SHA)
  self.assertEqual(len(m.CORPUS_SHA),64)
 def test_metric_inventory(self):
  self.assertEqual(len(m.FIELDS),4)
  self.assertIn('factual_pair_both_correct',m.FIELDS)
  self.assertIn('fact_signal_positive',m.FIELDS)
 def test_completed_report_if_present(self):
  if not OUT.exists():self.skipTest('Four real frozen-model stages not yet completed')
  s=OUT.read_text()
  self.assertEqual(hashlib.sha256(OUT.read_bytes()).hexdigest(),EXPECTED_SHA)
  expected={
   ('mid2_47684','default'):(129,1,86,85),
   ('mid2_47684','yarn'):(171,56,92,105),
   ('rl_merged','default'):(128,0,92,90),
   ('rl_merged','yarn'):(178,61,98,111),
  }
  for (stage,rope),(raw,both,debiased,signal) in expected.items():
   self.assertIn(f'| {stage} | {rope} | {raw} | {both} | {debiased} | {signal} |',s)
  self.assertIn('| Raw factual accuracy | +8 count(s) |',s)
  self.assertIn('| Both option orders correct | +6 count(s) |',s)
  for left in m.STAGES:
   for right in m.ROPES:
    self.assertIn('| '+left+' | '+right+' |',s)
  for fragment in ('mid2_rope','rl_rope','default_weights','yarn_weights','native_transition'):
   self.assertIn(fragment,s)
  self.assertIn('32 surname',s)
  self.assertIn('## Factorial interaction',s)
  self.assertGreater(len(s.splitlines()),75)
