"""EXP030 exact source, prompt-only vs budget-only isolation, score and custody."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
sys.path.insert(0,'/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
import k2_human_output_contract as m
import k2_human_grounded_qa as prior

EVIDENCE=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
EXPECTED={
 'strict32':'22a5f0e07d3dde27cfb4236405ba424e204e159331bf83e4cd0da559a2259838',
 'budget64':'1b7053530fe70ffc8b875ca39ff75e65701e1d79058ad022bc49fe61cce7d108',
}

class TestContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from transformers import AutoTokenizer
  cls.tok=AutoTokenizer.from_pretrained('/home/scott/git/.latent-k2-mopd_249-research',
          local_files_only=True,trust_remote_code=False)
 def test_full_source_and_protocol_sha(self):
  self.assertEqual(m.sha(EVIDENCE/'EXPERIMENT-030-PROTOCOL.md'),m.PROTOCOL_SHA)
  self.assertEqual(m.sha(EVIDENCE/'EXPERIMENT-029-K2-HUMAN-QA.md'),m.PARENT_SHA)
  self.assertEqual(m.PARENT_SHA,'0ab5ac2d744362969b97618b32ce9ad536ae32bda820a4bfac79410cc7c17c71')
 def test_model_arms_change_one_axis_only(self):
  a,receipt1=m.prepare(prior,self.tok,'strict32')
  b,receipt2=m.prepare(prior,self.tok,'budget64')
  self.assertEqual(receipt1,EXPECTED['strict32'])
  self.assertEqual(receipt2,EXPECTED['budget64'])
  self.assertEqual(len(a),32)
  self.assertEqual(len(b),32)
  for row1,row2 in zip(a,b):
   self.assertEqual(row1['qa_id'],row2['qa_id'])
   self.assertEqual(row1['position'],row2['position'])
   self.assertTrue(row1['prompt'].startswith(m.STRICT_INSTRUCTION))
   self.assertTrue(row2['prompt'].startswith(m.BASE_INSTRUCTION))
   self.assertEqual(row1['prompt'][len(m.STRICT_INSTRUCTION):],
                    row2['prompt'][len(m.BASE_INSTRUCTION):])
  self.assertEqual(m.ARMS,{'strict32':32,'budget64':64})
 def test_deterministic_full_answer_vs_lexical_prefix(self):
  answers,_,_=prior.source_rows()
  factual=next(a for a in answers if not a['impossible'])
  neutral=next(a for a in answers if a['impossible'])
  factual={**factual,'position':'short','prompt_tokens':200}
  neutral={**neutral,'position':'short','prompt_tokens':200}
  gold=factual['answers'][0]
  correct=m.scored(prior,factual,gold,False)
  self.assertEqual((correct['strict'],correct['first_gold']),(1,1))
  verbose=m.scored(prior,factual,gold+' because I think so',True)
  self.assertEqual((verbose['strict'],verbose['first_gold'],verbose['hit_cap']),(0,1,1))
  u=m.scored(prior,neutral,'UNKNOWN because context missing',False)
  self.assertEqual((u['strict'],u['first_unknown'],u['nonabstain_impossible']),(0,1,1))
 def test_eight_plus_eight_metric_aggregation(self):
  pairs,_,_=prior.source_rows()
  results=[]
  for placement in m.PLACEMENTS:
   for row in pairs:
    response='UNKNOWN' if row['impossible'] else row['answers'][0]
    results.append(m.scored(prior,{**row,'position':placement,
                         'prompt_tokens':200},response,False))
  for placement in m.PLACEMENTS:
   x=m.metrics(results,placement)
   self.assertEqual(x['strict'],8)
   self.assertEqual(x['unknown'],8)
   self.assertEqual(x['first_gold'],8)
   self.assertEqual(x['first_unknown'],8)
   self.assertEqual(x['caps'],0)
 def test_parent_baseline_readonly(self):
  baseline=m.parent_baseline(EVIDENCE)
  self.assertEqual((baseline['short']['exact'],baseline['short']['unknown'],
                    baseline['short']['caps']),(0,0,9))
  self.assertEqual((baseline['gap']['exact'],baseline['gap']['unknown'],
                    baseline['gap']['caps']),(0,0,14))
 def test_report_if_present(self):
  for arm in m.ARMS:
   p=EVIDENCE/f'EXPERIMENT-030-{arm.upper()}.md'
   if not p.exists():continue
   s=p.read_text()
   for needle in (m.PROTOCOL_SHA,m.PARENT_SHA,EXPECTED[arm],
                  '| short |','| gap |','Strict UNKNOWN /8','CC BY-SA 4.0'):
    self.assertIn(needle,s)
