"""EXP032B post-hoc stop/evidence gate source, negative and report controls."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
sys.path.insert(0,'/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
import k2_posthoc_boundary as m
import k2_human_grounded_qa as prior
E=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')

class FakeTokenizer:
 eos_token_id=1
 def decode(self,ids,skip_special_tokens=True):
  mapping={10:' Paris',11:' because',12:' detailed',13:' UNKNOWN',14:' The',
           1:'<|endoftext|>',64019:'<|ifm|im_end|>'}
  return ''.join(mapping[i] for i in ids if not(skip_special_tokens and i in (1,64019)))

class PostHocBoundaryTests(unittest.TestCase):
 def test_exact_immutable_registration(self):
  self.assertEqual(m.sha(E/'EXPERIMENT-032B-STOP-POSTHOC-PROTOCOL.md'),m.PROTOCOL_SHA)
  self.assertEqual(m.sha(E/'EXPERIMENT-029-K2-HUMAN-QA.md'),m.EXP029_RESULTS)
  self.assertEqual(m.sha(E/'EXPERIMENT-031-TERMINATORS.md'),m.EXP031_RESULTS)
 def test_32_source_instances_exact_parent_digests(self):
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained('/home/scott/git/.latent-k2-mopd_249-research',
                                     local_files_only=True,trust_remote_code=False)
  rows,receipt=m.preflight(E,prior,tok)
  self.assertEqual(receipt,m.SOURCE_SELECTION)
  self.assertEqual(len(rows),32)
  self.assertEqual(sum(r['impossible'] for r in rows),16)
  self.assertEqual(len({r['qa_id'] for r in rows}),16)
 def test_actual_alt_marker_changes_cut_but_not_native_stop(self):
  tok=FakeTokenizer()
  outputs,meta=m.interpret([10,64019,11]+[12]*29,tok,'Paris is documented','Where is the place?')
  self.assertEqual(outputs['native'],' Paris because'+' detailed'*29)
  self.assertEqual(outputs['alt_cut'],' Paris')
  self.assertEqual(outputs['first_word'],'Paris')
  self.assertEqual(outputs['lexical_gate'],'Paris')
  self.assertEqual(outputs['question_exclusion_gate'],'Paris')
  self.assertEqual(outputs['always_unknown'],'UNKNOWN')
  self.assertEqual((meta['cap'],meta['alt'],meta['alt_position'],meta['alt_changed']),(1,1,1,1))
 def test_question_gate_rejects_source_visible_candidate(self):
  outputs,_=m.interpret([10,11]*16,FakeTokenizer(),
                         'Paris is a city','Who visited Paris?')
  self.assertEqual(outputs['lexical_gate'],'Paris')
  self.assertEqual(outputs['question_exclusion_gate'],'UNKNOWN')
 def test_lexical_gate_not_truth_or_entailment(self):
  # The external rule DOES NOT see gold or is_impossible. Negative
  # examples can still contain a source word that the rule wrongly accepts.
  outputs,_=m.interpret([10]*32,FakeTokenizer(),
                        'Paris is a city','Who sold the nonexistent treaty?')
  self.assertEqual(outputs['lexical_gate'],'Paris')
  self.assertEqual(outputs['question_exclusion_gate'],'Paris')
 def test_fallback_unknown_when_no_lexical_source(self):
  outputs,_=m.interpret([13]*32,FakeTokenizer(),
                      'Unknown not in context','Who spoke?')
  self.assertEqual(outputs['first_word'],'UNKNOWN')
  self.assertEqual(outputs['lexical_gate'],'UNKNOWN')
 def test_score_aggregation_factual_negative_separation(self):
  scored=[]
  for i in range(16):
   scored.append({'placement':'short','strategy':'native',
                  'unanswerable':i>=8,'strict':int(i==0 or i==8),
                  'lenient':int(i==0),'nonabstention':int(i>=9),
                  'unexpected_unknown':0,'changed_from_native':0})
  r=m.summarize(scored,'short','native')
  self.assertEqual((r['strict'],r['lenient'],r['unknown'],r['negative_nonabstention']),
                   (1,1,1,7))
 def test_result_hash_and_summary_if_completed(self):
  p=E/'EXPERIMENT-032B-STOP-POSTHOC.md'
  if not p.is_file():return
  text=p.read_text()
  self.assertIn(m.PROTOCOL_SHA,text)
  self.assertIn(m.EXP029_RESULTS,text)
  self.assertIn('| short | native |',text)
  self.assertIn('| gap | always_unknown |',text)
