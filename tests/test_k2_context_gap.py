"""EXP028 source-pinned equal-length gap/frequency pilot preflight and guards."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import k2_context_gap as e

R=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
class ContextPilot(unittest.TestCase):
 def test_frozen_source_and_protocol(self):
  self.assertEqual(e.sha(R/'EXPERIMENT-028-PROTOCOL.md'),e.PROTOCOL_SHA)
  self.assertEqual(e.sha(R/'results_exp018_intermediate_k2_20261014.json'),e.EXP018_SHA)
  self.assertEqual(e.sha(R/'EXPERIMENT-025-FULL-WEIGHT-DRIFT.md'),e.PRIOR_WEIGHT_SHA)
  self.assertEqual(len(e.DERIVED_SHA),64)
  self.assertEqual(e.GROUPS,(0,8))
 def test_two_frequency_same_amplitude(self):
  self.assertEqual(e.FREQUENCIES,('default','yarn'))
  self.assertEqual(e.TARGET_TOKENS,(128,384))
  self.assertEqual(e.PLACEMENTS,('prefix','gap'))
 def test_fixture_and_exact_token_matched_pair(self):
  from transformers import AutoTokenizer
  from pathlib import Path
  from k2_rope_components_fixture import make
  tok=AutoTokenizer.from_pretrained('/home/scott/git/.latent-k2-mid2_47684-research',local_files_only=True,trust_remote_code=False)
  rows,sha=e.build(tok)
  self.assertEqual(sha,e.DERIVED_SHA)
  self.assertEqual(len(rows),128)
  self.assertEqual(len({r['prompt'] for r in rows}),128)
  for n in e.TARGET_TOKENS:
   for g in e.GROUPS:
    for state in (0,1):
     for role in ('actor','second_party'):
      for condition in ('factual','neutral'):
       for order in (0,1):
        a=[r for r in rows if (r['target'],r['group'],r['state'],r['role'],r['condition'],r['order'])==(n,g,state,role,condition,order)]
        self.assertEqual(len(a),2)
        self.assertLessEqual(abs(a[0]['tokens']-a[1]['tokens']),4)
        self.assertGreater(abs(a[0]['gap_tokens']-a[1]['gap_tokens']),80)
 def test_aggregation_and_original_answer_order(self):
  scored={}
  for target in e.TARGET_TOKENS:
   for place in e.PLACEMENTS:
    for group in e.GROUPS:
     for state in (0,1):
      for role in ('actor','second_party'):
       for cond in ('factual','neutral'):
        for order in (0,1):
         scored[target,place,group,state,role,cond,order]={
            'margin':1. if cond=='factual' else -1.,
            'correct':int(cond=='factual'),'raw_A':0}
  for target in e.TARGET_TOKENS:
   for place in e.PLACEMENTS:
    x=e.metric(scored,target,place)
    self.assertEqual(x['raw'],16)
    self.assertEqual(x['both'],8)
    self.assertEqual(x['debiased'],8)
    self.assertEqual(x['positive_fact'],8)
    self.assertEqual(x['neutral_A'],0)

class ContextReportFormatting(unittest.TestCase):
 def test_deterministic_human_readable_report(self):
  import hashlib
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained('/home/scott/git/.latent-k2-mid2_47684-research',local_files_only=True,trust_remote_code=False)
  rows,receipt=e.build(tok)
  fake={}
  for freq in e.FREQUENCIES:
   for target in e.TARGET_TOKENS:
    for place in e.PLACEMENTS:
     fake[freq,target,place]={
       'raw':8,'both':0,'debiased':4,'positive_fact':5,'neutral_A':0,
       'roles':{'actor':{'raw':4,'both':0,'debiased':2,'positive_fact':3},
                'second_party':{'raw':4,'both':0,'debiased':2,'positive_fact':2}}}
  doc=e.report(receipt,rows,fake,'7bf45b3582c2e0348c2cde47d5366588c357e771',26)
  self.assertIn('| 128 | gap | yarn | 8 | 0 | 4 | 5 | 0 |',doc)
  self.assertIn('| 384 | prefix | default | 8 | 0 | 4 | 5 | 0 |',doc)
  self.assertIn(e.PROTOCOL_SHA,doc)
  self.assertIn(e.DERIVED_SHA,doc)
  self.assertEqual(doc,e.report(receipt,rows,fake,'7bf45b3582c2e0348c2cde47d5366588c357e771',26))
