"""EXP029 human-annotated SQuAD2 selection, custody, inference-result contract."""
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
import k2_human_grounded_qa as m

EVIDENCE=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
GOLD_SELECTION='aad6e578544e2a3f100ab4b5c93d7ee6158fbfbe14bd426b8bffd9f9d1cde23b'
K2_TOKEN_SOURCE='6513d65b456a0f52d230e386078d732a4b2800663c4d9b11c76b3cd569651c2a'
QWEN_TOKEN_SOURCE='e27009abf39759a4b34a18665da2db87fc2a11aa9b53ecc9229ce37d4129f251'

class HumanQA(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows,cls.receipt,cls.count=m.source_rows()
 def test_official_human_qa_custody(self):
  self.assertEqual(m.sha(m.SOURCE),m.SOURCE_SHA)
  self.assertEqual(m.sha(EVIDENCE/'EXPERIMENT-029-PROTOCOL.md'),m.PROTOCOL_SHA)
  self.assertEqual(self.receipt,GOLD_SELECTION)
  self.assertEqual(self.count,90)
  self.assertEqual(len(self.rows),16)
  self.assertEqual(len(set(r['article_index'] for r in self.rows)),8)
  self.assertEqual(len({r['topic_digest'] for r in self.rows}),8)
  self.assertEqual(sum(r['impossible'] for r in self.rows),8)
 def test_span_identity_and_no_false_human_review_claim(self):
  for item in self.rows:
   if item['impossible']:self.assertEqual(item['answers'],())
   else:
    self.assertTrue(all(a in item['context'] for a in item['answers']))
    self.assertTrue(all(a.isalpha() or '-' in a for a in item['answers']))
 def test_exact_answer_vs_lenient_scoring(self):
  example=next(x for x in self.rows if not x['impossible'])
  gold=example['answers'][0]
  one={**example,'position':'short','prompt_tokens':300}
  strict=m.score_one(one,gold+'.',False)
  self.assertEqual(strict['strict'],1)
  longer=m.score_one(one,'The correct answer is '+gold+'.',False)
  self.assertEqual(longer['strict'],0)
  self.assertEqual(longer['lenient'],1)
  unknown=m.score_one(one,'UNKNOWN',True)
  self.assertEqual(unknown['strict'],0)
  self.assertEqual(unknown['unexpected_unknown'],1)
  self.assertEqual(unknown['hit_cap'],1)
 def test_unanswerable_abstention_and_nonabstention(self):
  case=next(x for x in self.rows if x['impossible'])
  row={**case,'position':'short','prompt_tokens':400}
  exact=m.score_one(row,'UNKNOWN.',False)
  self.assertEqual(exact['strict'],1)
  self.assertEqual(exact['nonabstain_impossible'],0)
  verbose=m.score_one(row,'I think UNKNOWN due to missing context',False)
  self.assertEqual(verbose['strict'],0)
  self.assertEqual(verbose['contains_unknown'],1)
  self.assertEqual(verbose['nonabstain_impossible'],1)
 def test_three_positions_and_paired_lengths(self):
  from transformers import AutoTokenizer
  for stage,model,expected in (
      ('mopd_249','/home/scott/git/.latent-k2-mopd_249-research',K2_TOKEN_SOURCE),
      ('qwen08_chat','/home/scott/git/.latent-qwen35-0p8b-model',QWEN_TOKEN_SOURCE)):
   tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
   rows,digest,filler=m.generate_rows(tok,self.rows,stage)
   self.assertEqual(digest,expected)
   self.assertEqual(len(rows),48)
   self.assertEqual(filler,768)
   for i in range(16):
    base,prefix,gap=rows[i],rows[i+16],rows[i+32]
    self.assertEqual(base['qa_id'],prefix['qa_id'])
    self.assertEqual(gap['qa_id'],prefix['qa_id'])
    self.assertLessEqual(abs(prefix['prompt_tokens']-gap['prompt_tokens']),6)
    self.assertGreater(prefix['prompt_tokens']-base['prompt_tokens'],700)
 def test_aggregate_and_layout(self):
  got=[]
  for pos in m.POSITIONS:
   for r in self.rows:
    got.append(m.score_one({**r,'position':pos,'prompt_tokens':300},
                    'UNKNOWN' if r['impossible'] else r['answers'][0],False))
  for pos in m.POSITIONS:
   s=m.summarize(got,pos)
   self.assertEqual(s['answerable_exact'],8)
   self.assertEqual(s['unanswerable_UNKNOWN'],8)
   self.assertEqual(s['token_caps'],0)
 def test_completed_k2_report_exact_custody(self):
  p=EVIDENCE/'EXPERIMENT-029-K2-HUMAN-QA.md'
  self.assertEqual(m.sha(p),'0ab5ac2d744362969b97618b32ce9ad536ae32bda820a4bfac79410cc7c17c71')
  doc=p.read_text()
  self.assertIn('| short | 164–307 | 0 | 7 | 0 | 8 | 0 | 9 |',doc)
  self.assertIn('| prefix | 933–1076 | 0 | 8 | 0 | 8 | 0 | 11 |',doc)
  self.assertIn('| gap | 933–1076 | 0 | 6 | 0 | 8 | 0 | 14 |',doc)
  self.assertIn('Published native chat template used: **False**',doc)
  self.assertIn(m.PROTOCOL_SHA,doc)
 def test_completed_qwen_report_exact_custody(self):
  p=EVIDENCE/'EXPERIMENT-029-QWEN-HUMAN-QA.md'
  self.assertEqual(m.sha(p),'a166f1727620bc2cc162f07c388584a97201bf2064e233cb92c87c2d3d60cd85')
  doc=p.read_text()
  self.assertIn('| short | 173–310 | 5 | 6 | 2 | 6 | 0 | 1 |',doc)
  self.assertIn('| prefix | 942–1079 | 5 | 6 | 6 | 2 | 1 | 0 |',doc)
  self.assertIn('| gap | 942–1079 | 5 | 6 | 6 | 2 | 1 | 0 |',doc)
  self.assertIn('Published native chat template used: **True**',doc)
  self.assertIn(m.PROTOCOL_SHA,doc)
 def test_result_if_present(self):
  for tag in ('K2','QWEN'):
   p=EVIDENCE/f'EXPERIMENT-029-{tag}-HUMAN-QA.md'
   if not p.exists():continue
   content=p.read_text()
   self.assertIn(GOLD_SELECTION,content)
   self.assertIn('CC BY-SA 4.0',content)
   for pos in m.POSITIONS:self.assertIn('| '+pos+' |',content)
   self.assertIn('32-token cap',content)
