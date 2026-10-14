"""EXP032F pinned source-disjoint human-gold fixture and independent denial."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
sys.path.insert(0,'/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')
import k2_disjoint_squad_fixture as source
import k2_qwen_disjoint_holdout as exp
import k2_qwen_candidate_guard as parent
import k2_human_grounded_qa as prior
R=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026')

class ArticleDisjoint(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows,cls.sha,cls.eligible=source.select(prior)
 def test_exact_protocol_and_new_source_receipts(self):
  self.assertEqual(exp.sha(R/'EXPERIMENT-032F-ARTICLE-DISJOINT-PROTOCOL.md'),exp.PROTOCOL)
  self.assertEqual(self.sha,exp.SOURCE_SELECTION)
  self.assertEqual(len(self.rows),32)
  self.assertEqual(self.eligible,55)
 def test_original_eight_titles_excluded(self):
  previous,_,_=prior.source_rows()
  titles={x['topic_digest'] for x in previous}
  selected={x['topic_digest'] for x in self.rows}
  self.assertEqual(len(selected),16)
  self.assertFalse(titles&selected)
  self.assertEqual(len(titles),8)
 def test_questions_pair_and_human_gold_spans(self):
  self.assertEqual(sum(x['impossible'] for x in self.rows),16)
  for i in range(16):
   a,b=self.rows[2*i:2*i+2]
   self.assertEqual(a['article_index'],b['article_index'])
   self.assertFalse(a['impossible'])
   self.assertTrue(b['impossible'])
   self.assertTrue(all(gold in a['context'] for gold in a['answers']))
   self.assertEqual(b['answers'],())
   self.assertTrue(a['prompt'].startswith('Answer the question using only the passage.'))
 def test_frozen_verifier_same_as_previous(self):
  self.assertEqual(exp.parent, parent)
  self.assertEqual(exp.ARMS,('first_word','lexical_gate','question_exclusion','qwen_verifier','always_unknown'))
 def test_aggregate_math_16_positive_and_negative(self):
  fake=[]
  for i in range(32):
   fake.append({'strategy':'qwen_verifier','impossible':i>=16,'exact':int(i<16 or i>=16),
        'accepted':i<16,'nonabstain_negative':0,'positive_unknown':0})
  self.assertEqual(exp.summary(fake,'qwen_verifier'),{
       'correct':16,'incorrect_accepted':0,'negative_false_accept':0,
       'negative_unknown':16,'positive_unknown':0})
 def test_completed_report_if_present(self):
  f=R/'EXPERIMENT-032F-ARTICLE-DISJOINT.md'
  if f.exists():
   s=f.read_text()
   self.assertIn(exp.PROTOCOL,s)
   self.assertIn(exp.SOURCE_SELECTION,s)
   for arm in exp.ARMS:self.assertIn('| '+arm+' |',s)
