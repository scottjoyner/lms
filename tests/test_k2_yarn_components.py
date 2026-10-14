"""EXP027 no-inference source, masking, data-accounting, and report contracts."""
import hashlib
import sys
import unittest
import torch
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'research'))
import k2_rope_components_fixture as fixture
import k2_yarn_components as exp

class TestYaRNComponents(unittest.TestCase):
 def test_source_disjoint_and_locked(self):
  fixture.check_disjoint()
  rows=fixture.make()
  self.assertEqual(fixture.receipt(rows)['sha256'],exp.FIXTURE_SHA)
  self.assertEqual(len(rows),256)
  self.assertEqual(sum(r['condition']=='factual' for r in rows),128)
  self.assertEqual(sum(r['condition']=='neutral' for r in rows),128)
  self.assertEqual(len({r['prompt'] for r in rows}),256)
 def test_complete_role_order_pairs(self):
  rows=fixture.make()
  for g in range(16):
   group=[x for x in rows if x['group']==g]
   self.assertEqual(len(group),16)
   for role in ('actor','second_party'):
    for state in (0,1):
     for cond in ('factual','neutral'):
      cases=[r for r in group if r['state']==state and r['role']==role and r['condition']==cond]
      self.assertEqual(sorted(r['order'] for r in cases),[0,1])
      self.assertNotEqual(cases[0]['correct'],cases[1]['correct'])
 def test_episode_aggregation_on_synthetic_margin(self):
  scored={}
  for row in fixture.make():
   key=(row['group'],row['state'],row['role'],row['condition'],row['order'])
   m=1. if row['condition']=='factual' else -.25
   scored[key]={'signed':m,'raw':(m if row['correct']=='A' else -m),
                'correct':int(m>0),'neutral':row['condition']=='neutral','role':row['role']}
  summary=exp.aggregate(exp.episodes(scored))
  self.assertEqual(summary['overall']['raw_correct'],128)
  self.assertEqual(summary['overall']['both_correct'],64)
  self.assertEqual(summary['overall']['debiased_correct'],64)
  self.assertEqual(summary['overall']['positive_fact_signal'],64)
  self.assertEqual(summary['overall']['neutral_A_count'],64)
  self.assertEqual(summary['roles']['actor']['raw_correct'],64)
 def test_pairing_and_bootstrap_same_sample(self):
  scored={}
  for row in fixture.make():
   key=(row['group'],row['state'],row['role'],row['condition'],row['order'])
   scored[key]={'signed':2.,'raw':2.,'correct':1,'neutral':False,'role':row['role']}
  episodes=exp.episodes(scored)
  a,lo,hi=exp.boot(episodes,episodes,'both_correct')
  self.assertEqual((a,lo,hi),(0.,0.,0.))
 def test_full_factorial_definition_and_protocol_custody(self):
  self.assertEqual(set(exp.CORNERS),{('default','default'),('yarn','default'),('default','yarn'),('yarn','yarn')})
  protocol=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-027-PROTOCOL.md')
  self.assertEqual(exp.sha(protocol),exp.PROTOCOL_SHA)
 def test_rotary_component_isolation_synthetic(self):
  class FakeRotary(torch.nn.Module):
   def __init__(self,config,device=None):
    super().__init__()
    self.config=config
    f=torch.as_tensor(config['freq'],dtype=torch.float32)
    self.register_buffer('inv_freq',f.clone(),persistent=False)
    self.register_buffer('original_inv_freq',f.clone(),persistent=False)
    self.attention_scaling=float(config['scale'])
  orig=FakeRotary({'freq':[1.,.25,.015],'scale':1.0})
  yarn=FakeRotary({'freq':[1.,.16,.004],'scale':1.277258872})
  for freq,amp in exp.CORNERS:
   rot=exp.setup_rope(orig,yarn,freq,amp)
   expected_freq=orig if freq=='default' else yarn
   expected_amp=orig if amp=='default' else yarn
   self.assertTrue(torch.equal(rot.inv_freq,expected_freq.inv_freq))
   self.assertEqual(rot.attention_scaling,expected_amp.attention_scaling)
   self.assertTrue(torch.equal(orig.inv_freq,torch.tensor([1.,.25,.015])))

 def test_result_receipt_if_present(self):
  report=Path('/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-027-YARN-COMPONENTS.md')
  if not report.exists():self.skipTest('Frozen model inference not completed')
  s=report.read_text()
  self.assertEqual(hashlib.sha256(report.read_bytes()).hexdigest(),'d37b5e4babfc3db23ba2a02745e15f987a1552e82404741ef9bfb41f8a797ff2')
  for f,a,raw,both,debiased,signal,neutral in [
   ('default','default',64,0,37,40,0),
   ('yarn','default',64,0,34,39,0),
   ('default','yarn',75,24,39,43,6),
   ('yarn','yarn',74,24,38,44,0)]:
   self.assertIn(f'| {f} | {a} | {raw} | {both} | {debiased} | {signal} | {neutral} |',s)
  self.assertIn('**1.277258872**',s)
  self.assertIn('**26** positions',s)
  self.assertIn('## Controlled main contrasts',s)
  self.assertEqual(sum(f'| {a} | {b} |' in s for a,b in exp.CORNERS),4)
  self.assertIn(exp.FIXTURE_SHA,s)
  self.assertIn('human-reviewed',s)
