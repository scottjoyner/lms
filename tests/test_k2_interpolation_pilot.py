"""No-network, no-checkpoint-write regression tests for EXP035."""
import importlib.util
import pathlib
import unittest

import torch
SCRIPT=pathlib.Path(__file__).resolve().parents[1]/"research"/"k2_interpolation_pilot.py"
SPEC=importlib.util.spec_from_file_location("k2_interpolation_pilot",SCRIPT)
k2=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(k2)

class InterpolationUnitTest(unittest.TestCase):
    def test_alpha_gate_and_endpoints(self):
        base=torch.tensor([1.,2.,3.],dtype=torch.bfloat16)
        target=torch.tensor([3.,2.,1.],dtype=torch.bfloat16)
        for alpha in (0,.25,.5,.75,1):
            result=k2.interpolate_tensor(base,target,alpha)
            self.assertEqual(result.dtype,torch.bfloat16)
            self.assertEqual(result.tolist(),
                             [float(x) for x in (1+2*alpha,2,3-2*alpha)])
        self.assertEqual(base.tolist(),[1,2,3])
        self.assertEqual(target.tolist(),[3,2,1])
        self.assertIs(k2.interpolate_tensor(base,target,0),base)
        self.assertIs(k2.interpolate_tensor(base,target,1),target)
    def test_fail_closed_shape_dtype(self):
        with self.assertRaises(ValueError):
            k2.interpolate_tensor(torch.ones(2,dtype=torch.bfloat16),
                                  torch.ones(3,dtype=torch.bfloat16),.5)
        with self.assertRaises(ValueError):
            k2.interpolate_tensor(torch.ones(2),torch.ones(2),.5)
        with self.assertRaises(ValueError):
            k2.interpolate_tensor(torch.tensor([float("nan")],dtype=torch.bfloat16),
                                  torch.ones(1,dtype=torch.bfloat16),.5)
    def test_fixture_balance(self):
        choices=[row[-1] for row in k2.STIMULI]
        self.assertEqual(len(choices),8)
        self.assertEqual({c:choices.count(c) for c in "ABC"},{"A":3,"B":3,"C":2})
        self.assertTrue(all("Source:" in x for x,_ in k2.make_prompts()))
    def test_pinned_assets_read_only_preflight(self):
        index,_,_=k2.preflight()
        self.assertEqual(k2.check_shard_layout(*index),k2.EXPECTED_PARAMS)
        self.assertEqual(len(index[0]),255)
        self.assertEqual(len(index[1]),255)
if __name__=="__main__":unittest.main()
