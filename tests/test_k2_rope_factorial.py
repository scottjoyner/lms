"""EXP024 independent reproducibility and fixed-weight/RoPE evidence contracts."""
import importlib.util
import hashlib
import unittest
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location(
    "k2_rope_factorial",BASE/"research/k2_rope_factorial.py")
factorial=importlib.util.module_from_spec(spec)
spec.loader.exec_module(factorial)

RESEARCH=Path("/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026")
REPORT=RESEARCH/"EXPERIMENT-024-ROPE-FACTORIAL.md"
EXPECTED_SHA="ffd0752b7d160bb0d9582b2d3c73b05a47703c74a3acf5acf692ce3d42b967a3"

class Exp024FactorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md=REPORT.read_text()
        cls.digest=hashlib.sha256(REPORT.read_bytes()).hexdigest()

    def test_report_exact_sha_and_original_custody(self):
        self.assertEqual(self.digest,EXPECTED_SHA)
        self.assertEqual(len(factorial.K2_STAGES),2)
        self.assertEqual(set(factorial.ROPES),{"default","yarn"})
        self.assertEqual(len(factorial.SAMPLED),5)
        self.assertIn(factorial.SOURCE_SHA,self.md)
        self.assertIn(factorial.WEIGHT_REPORT_SHA,self.md)
        self.assertIn(factorial.CORPUS_SHA,self.md)

    def test_complete_two_by_two_and_native_replays(self):
        expected={
            ("mid2_47684","default"):(149,84,65),
            ("mid2_47684","yarn"):(160,114,46),
            ("rl_merged","default"):(156,92,64),
            ("rl_merged","yarn"):(161,116,45),
        }
        for (stage,rope),(total,actor,recipient) in expected.items():
            self.assertIn(
                f"| {stage} | {rope} | {total} | {actor} | {recipient} |",
                self.md)
        for effect in ("+11/256","+5/256","+7/256","+1/256","-6/256"):
            self.assertIn(effect,self.md)

    def test_rope_vs_weight_geometry_contrast(self):
        self.assertIn(
            "| mid2 fixed weights: default→YaRN | 0.9360 | 0.5568 |",self.md)
        self.assertIn(
            "| RL fixed weights: default→YaRN | 0.9367 | 0.4797 |",self.md)
        self.assertIn(
            "| default RoPE fixed: mid2→RL | 0.9979 | 0.9957 |",self.md)
        self.assertIn(
            "| YaRN fixed: mid2→RL | 0.9979 | 0.9499 |",self.md)
        self.assertIn(
            "| observed native mid2/default→RL/YaRN | 0.9332 | 0.4669 |",self.md)

    def test_five_crosslayer_grids_each_four_conditions(self):
        self.assertEqual(self.md.count("#### Mean-pooling whole-fact"),5)
        self.assertEqual(self.md.count("#### Last-token whole-fact"),5)
        self.assertEqual(self.md.count("#### mean signed-role"),5)
        self.assertEqual(self.md.count("#### last signed-role"),5)
        self.assertGreater(len(self.md.splitlines()),250)

    def test_sign_control_on_synthetic_hidden(self):
        rng=np.random.default_rng(24)
        x=rng.normal(size=(64,8))
        a={(depth,pool):x for depth in factorial.SAMPLED
          for pool in ("mean","last")}
        def cka(u,v):
            from numpy.linalg import norm
            u=u-u.mean(0);v=v-v.mean(0)
            cu=u@u.T;cv=v@v.T
            return float((cu*cv).sum()/(norm(cu)*norm(cv)))
        m=factorial.cka_matrix(type("C",(),{"centered_cka":staticmethod(cka)}),
                                a,a,"mean",True)
        self.assertEqual(len(m),5)
        self.assertAlmostEqual(m[-1][-1],1.,places=10)
        self.assertAlmostEqual(cka(x,-x),1.,places=10)

    def test_no_raw_prompt_or_model_tensors_exported(self):
        self.assertIn("No prompts/labels were generated independently",self.md)
        self.assertNotIn("input_ids",self.md)
        self.assertNotIn("raw_logits",self.md)
        self.assertIn("controlled",self.md.lower())
