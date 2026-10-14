"""EXP034 human-gold data integrity, source derangement, candidate likelihood tests."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
sys.path.insert(0,'/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
import k2_evidence_swap as e
import k2_human_grounded_qa as prior

EVIDENCE=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
K2_SHA='163a8a9f27d108dc7e824c4b70b70031abcd96e4ba6432f76e1b2613368588d2'
QWEN_SHA='7195c2f3e3d5661769120b366e11cc04a04940617c615a0e9a70dce81b7d009a'
DONORS='bd5004731519d1b5b6c994885e5f519bb7ed5209616e329937812395a9759380'

class EvidenceSwap(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from transformers import AutoTokenizer
  cls.k2=AutoTokenizer.from_pretrained('/home/scott/git/.latent-k2-mopd_249-research',
                local_files_only=True,trust_remote_code=False)
  cls.qwen=AutoTokenizer.from_pretrained('/home/scott/git/.latent-qwen35-0p8b-model',
                local_files_only=True,trust_remote_code=False)
  cls.cases,cls.source_receipt,cls.eligible=prior.source_rows()
 def test_preregistered_source_sha_and_8_human_topics(self):
  self.assertEqual(e.sha(EVIDENCE/'EXPERIMENT-034-PROTOCOL.md'),e.PROTOCOL_SHA)
  self.assertEqual(self.source_receipt,e.SQUAD_RECEIPT)
  self.assertEqual(len(self.cases),16)
  self.assertEqual(self.eligible,90)
  self.assertEqual(sum(r['impossible'] for r in self.cases),8)
 def test_derangement_does_not_consult_gold_labels(self):
  p,sizes,shifts,h=e.source_donors(self.cases,self.k2)
  self.assertEqual(h,DONORS)
  self.assertEqual(sorted(shifts),[3,4,6,7,14,20,64,64])
  self.assertTrue(all(i!=p[i] for i in range(8)))
  self.assertEqual(set(p),set(range(8)))
 def test_qwen_pad_from_published_tokenizer_not_multimodal_config(self):
  import json
  config=json.loads(Path('/home/scott/git/.latent-qwen35-0p8b-model/config.json').read_text())
  self.assertNotIn('pad_token_id',config)
  self.assertEqual(self.qwen.pad_token_id,248044)
  self.assertEqual(self.k2.pad_token_id,64255)
 def test_full_native_source_and_candidate_suffixes(self):
  for stage,tok,sha in [('mopd_249',self.k2,K2_SHA),('qwen08_chat',self.qwen,QWEN_SHA)]:
   rows,receipt,donor,shifts,leak=e.prep(EVIDENCE,prior,tok,stage)
   self.assertEqual(len(rows),64)
   self.assertEqual(receipt,sha)
   self.assertEqual(donor,DONORS)
   self.assertEqual(leak,0)
   for row in rows:
    self.assertGreaterEqual(len(row['suffix']),1)
    self.assertEqual(row['suffix'],row['full'][row['prefix_len']:])
    self.assertLess(row['prefix_len'],e.MAX_PROMPT)
 def test_complete_full_suffix_score_summary(self):
  fake=[]
  for article in range(8):
   for impossible in (False,True):
    for source in ('original','donor'):
     for choice in ('answer','unknown'):
      value=0. if choice=='unknown' else (2. if source=='original' else 1.)
      fake.append({'article':article,'impossible':impossible,
                   'source':source,'choice':choice,'lp':value})
  result=e.summarize(fake)
  self.assertEqual(result['answerable']['positive_source_gain'],8)
  self.assertEqual(result['unanswerable']['positive_source_gain'],8)
  self.assertEqual(result['answerable']['mean_source_gain'],1.)
  with self.assertRaises(RuntimeError):e.summarize(fake[:-1])
 def test_measured_k2_full_candidate_source_swap(self):
  p=EVIDENCE/'EXPERIMENT-034-K2-EVIDENCE-SWAP.md'
  self.assertEqual(e.sha(p),'eea7b12d7de64a74057721006028b2cbef2e70c113d3dd10cfde3610d1cf2661')
  doc=p.read_text()
  self.assertIn('| answerable | 8 | 0 | 8 | +13.99098 | +13.70894 | +7.7336..+22.1475 |',doc)
  self.assertIn('| unanswerable | 5 | 0 | 8 | +10.97771 | +10.05620 | +4.9251..+21.7904 |',doc)
  self.assertIn('all 64 candidate strings completed',doc.lower())
  self.assertIn(e.PROTOCOL_SHA,doc)
  self.assertIn(DONORS,doc)
 def test_measured_qwen_full_candidate_source_swap(self):
  p=EVIDENCE/'EXPERIMENT-034-QWEN-EVIDENCE-SWAP.md'
  self.assertEqual(e.sha(p),'403105ee08da20907de6850574a340544397bb98b5da55f830c13d00685f014d')
  doc=p.read_text()
  self.assertIn('| answerable | 8 | 0 | 8 | +30.46691 | +31.33203 | +17.5657..+39.3851 |',doc)
  self.assertIn('| unanswerable | 1 | 0 | 8 | +17.03037 | +14.53505 | +9.3552..+26.8731 |',doc)
  self.assertIn('Published native chat template used: **True**',doc)
  self.assertIn('all 64 candidate strings completed',doc.lower())
 def test_reports_when_present(self):
  for m in ('K2','QWEN'):
   report=EVIDENCE/f'EXPERIMENT-034-{m}-EVIDENCE-SWAP.md'
   if not report.exists():continue
   text=report.read_text()
   self.assertIn(e.PROTOCOL_SHA,text)
   self.assertIn(DONORS,text)
   self.assertIn('| answerable |',text)
   self.assertIn('| unanswerable |',text)
   self.assertIn('all 64 candidate',text.lower())
