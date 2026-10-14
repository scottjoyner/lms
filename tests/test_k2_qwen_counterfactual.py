"""EXP032E source/question counterfactual Qwen verifier controls."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import k2_qwen_counterfactual as exp
import k2_qwen_candidate_guard as parent
R=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')

class CounterfactualVerifierTests(unittest.TestCase):
 def test_source_protocol_and_provenance(self):
  self.assertEqual(exp.sha(R/'EXPERIMENT-032E-COUNTERFACTUAL-JUDGE-PROTOCOL.md'),exp.PROTOCOL)
  self.assertEqual(exp.sha(R/'EXPERIMENT-032D-QWEN-VERIFIER.md'),exp.PARENT)
 def test_donor_absent_candidate(self):
  rows=[{'article_index':0,'context':'Paris is in France'},
        {'article_index':1,'context':'Berlin is in Germany'},
        {'article_index':2,'context':'London is in Britain'}]
  text,fallback=exp.donors(rows,0,'Paris')
  self.assertFalse(fallback)
  self.assertNotIn('Paris',text)
 def test_fallback_never_mentions_candidate(self):
  rows=[{'article_index':0,'context':'A shared word'},
        {'article_index':1,'context':'Another shared word'}]
  donor,fallback=exp.donors(rows,0,'shared')
  self.assertTrue(fallback)
  self.assertEqual(donor,exp.EMPTY_EVIDENCE)
 def test_fixed_3_way_controled_prompts_and_questions(self):
  rows=[]
  for i in range(8):
   rows += [{'article_index':i,'context':f'City{i} is a destination.',
            'question':f'Where was traveller {i}?'} for _ in range(2)]
  rules=[{'question_exclusion':'City0'}]+[{}]*15
  proposals, fallback=exp.counterfactuals(rows,rules,[0])
  self.assertEqual(len(proposals),3)
  self.assertEqual({r['mode'] for r in proposals},set(exp.CONDITIONS))
  self.assertTrue(all('Proposed candidate: City0' in r['prompt'] for r in proposals))
  original=next(r for r in proposals if r['mode']=='original')
  changed=next(r for r in proposals if r['mode']=='question_swap')
  self.assertNotEqual(original['prompt'],changed['prompt'])
 def test_fail_closed_on_replay_mismatch(self):
  sample={mode:{i:('YES' if i<10 else 'NO') for i in range(11)} for mode in exp.CONDITIONS}
  _,summary=exp.summarize(sample,0,'synthetic-hash')
  self.assertEqual(summary['original'],{'YES':10,'NO':1,'MALFORMED':0})
  sample['original'][0]='MALFORMED'
  with self.assertRaises(RuntimeError):exp.summarize(sample,0,'synthetic-hash')
 def test_result_when_present(self):
  p=R/'EXPERIMENT-032E-COUNTERFACTUAL-JUDGE.md'
  if p.exists():
   doc=p.read_text()
   self.assertIn(exp.PROTOCOL,doc)
   self.assertIn(exp.PARENT,doc)
   for mode in exp.CONDITIONS:self.assertIn('| '+mode+' |',doc)

 def test_exact_completed_counterfactual_result(self):
  path=R/'EXPERIMENT-032E-COUNTERFACTUAL-JUDGE.md'
  self.assertEqual(exp.sha(path),'3333b2e05361e4626a9790e5e2f1c1b106c160b13d52e91bfa79f6567cb33966')
  doc=path.read_text()
  for snippet in (
      '| original | 10 | 1 | 0 |',
      '| evidence_swap | 0 | 11 | 0 |',
      '| question_swap | 1 | 10 | 0 |',
      'Original-YES→evidence_swap-NO: **10/10**.',
      'Original-YES→question_swap-NO: **9/10**.',
      'fixed digits-only empty-evidence fallback count: **2/11**.',
      exp.PROTOCOL,exp.PARENT):
   self.assertIn(snippet,doc)
