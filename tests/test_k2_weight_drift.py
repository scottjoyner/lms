"""Synthetic arithmetic tests for EXP023 without loading published large models."""
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("k2_weight_drift",ROOT/"research/k2_weight_drift.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class WeightDriftTest(unittest.TestCase):
    def test_tensor_family_paths(self):
        self.assertEqual(len(m.KINDS),9)
        self.assertEqual(m.tname(13,"attn.q_proj"),
                         "model.layers.13.self_attn.q_proj.weight")
        self.assertEqual(m.tname(27,"mlp.down_proj"),
                         "model.layers.27.mlp.down_proj.weight")
        self.assertEqual(m.tname(0,"input_layernorm"),
                         "model.layers.0.input_layernorm.weight")

    def test_unit_cosine_and_unchanged_exact_count(self):
        x={"nx":10.,"ny":10.,"nd":0.,"dot":10.,"n":4,"changed":0,"equal":True}
        v=m.merge([x])
        self.assertAlmostEqual(v["cosine"],1.)
        self.assertAlmostEqual(v["delta_l2"],0.)
        self.assertEqual(v["same_tensors"],1)
        self.assertEqual(v["changed_fraction"],0.)

    def test_mixed_parameter_weighted_not_tensor_weighted(self):
        x={"nx":5.,"ny":5.,"nd":0.,"dot":5.,"n":4,"changed":0,"equal":True}
        y={"nx":100.,"ny":100.,"nd":100.,"dot":50.,"n":100,
           "changed":25,"equal":False}
        v=m.merge([x,y])
        self.assertEqual(v["same_tensors"],1)
        self.assertEqual(v["tensors"],2)
        self.assertEqual(v["n"],104)
        self.assertAlmostEqual(v["changed_fraction"],25/104)
        self.assertAlmostEqual(v["delta_l2"],math.sqrt(100/105))

    def test_safetensors_actual_tensor_triplet(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            name=m.tname(0,"attn.q_proj")
            x=torch.tensor([1.,2.,3.,4.],dtype=torch.bfloat16)
            y=torch.tensor([1.,2.,4.,4.],dtype=torch.bfloat16)
            save_file({name:x},str(p/"old.safetensors"))
            save_file({name:y},str(p/"new.safetensors"))
            old={"shards":{"a":p/"old.safetensors"},"tensor_to_file":{name:"a"}}
            new={"shards":{"a":p/"new.safetensors"},"tensor_to_file":{name:"a"}}
            got=m.tensor_pair(old,new,name)
            self.assertEqual(got["n"],4)
            self.assertEqual(got["changed"],1)
            self.assertFalse(got["equal"])
            self.assertAlmostEqual(got["nx"],30)
            self.assertAlmostEqual(got["ny"],37)
            self.assertAlmostEqual(got["nd"],1)
            self.assertAlmostEqual(got["dot"],33)
            self.assertAlmostEqual(m.merge([got])["changed_fraction"],.25)

    def test_stage_layout_and_pins(self):
        self.assertEqual(len(m.STAGES),5)
        self.assertEqual(len(m.REFERENCE_SHA),64)
        self.assertEqual(len(set(m.STAGES)),5)
