"""EXP032 strict source locking and neuron null/similarity controls."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
sys.path.insert(0,'/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
import k2_neuron_sniffer as m
import k2_human_grounded_qa as prior
R=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
class TestSniffer(unittest.TestCase):
 def test_protocol_and_source(self):
  self.assertEqual(m.sha(R/'EXPERIMENT-032-PROTOCOL.md'),m.PROTOCOL)
  self.assertEqual(len(m.STAGES),3)
  self.assertEqual(m.LAYERS['mid2_47684'],m.LAYERS['rl_merged'])
  self.assertNotEqual(m.LAYERS['rl_merged'],m.LAYERS['qwen08_chat'])
 def test_exact_frozen_subset(self):
  from transformers import AutoTokenizer
  for stage,model in (('mid2_47684','/home/scott/git/.latent-k2-mid2_47684-research'),
                      ('qwen08_chat','/home/scott/git/.latent-qwen35-0p8b-model')):
   tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
   rows=m.source_prompts(prior,tok,stage)
   self.assertEqual(len(rows),16)
   self.assertEqual([r['article_index'] for r in rows],[i for i in range(8) for _ in range(2)])
 def test_synthetic_identical_cka_and_row_null(self):
  rng=np.random.default_rng(20261014)
  x=rng.standard_normal((16,128))
  self.assertAlmostEqual(m.cka(x,x),1.0,places=9)
  # n=16 << width=128: centered Gram matrices may have high shuffled CKA.
  # This is exactly why the actual neuron analysis requires a shuffled null.
  shuffled=m.cka(x,x[rng.permutation(16)])
  self.assertGreaterEqual(shuffled,0)
  self.assertLess(shuffled,1)
  self.assertGreater(m.cka(x,x)-shuffled,.01)
 def test_same_coordinate_and_nonhomology_guard(self):
  rng=np.random.default_rng(4)
  x=rng.standard_normal((16,128))
  cosine,overlap=m.same_coordinate(x,x)
  self.assertAlmostEqual(cosine,1.0,places=8)
  self.assertEqual(overlap,32)
  with self.assertRaises(ValueError):m.same_coordinate(x,x[:,:64])
 def test_exact_paired_signflip_max_and_width(self):
  rng=np.random.default_rng(9)
  x=rng.normal(size=(16,256))
  stats=m.neuron_stats(x)
  self.assertEqual(stats['width'],256)
  self.assertGreater(stats['top32_mass'],0)
  self.assertLessEqual(stats['null_fraction'],1)
  self.assertGreaterEqual(stats['null_fraction'],2/256)
 def test_report_if_measured(self):
  path=R/'EXPERIMENT-032-NEURON-SNIFFER.md'
  if not path.exists():self.skipTest('Live inference has not finished')
  t=path.read_text()
  self.assertIn(m.PROTOCOL,t)
  self.assertIn('within-K2',t.lower())
  self.assertIn('not',t.lower())
