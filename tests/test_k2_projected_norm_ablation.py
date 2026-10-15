"""EXP040: projection-energy constraint and no-weight-mutation checks."""
import importlib.util,pathlib,sys,unittest
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(R))
spec=importlib.util.spec_from_file_location("k2_projected_norm_ablation",R/"k2_projected_norm_ablation.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_projection_match_quantized(self):
        gen=torch.Generator().manual_seed(4040)
        W=torch.randn(1536,5120,generator=gen,dtype=torch.float32)
        t=np.arange(15,47);r=np.arange(128,160)
        for i in range(8):
            x=torch.randn(1,2,5120,generator=gen).to(torch.bfloat16)
            orig=x.clone();Worig=W.clone()
            y,a=m.match_projected(x,W,t,r,"top")
            z,b=m.match_projected(x,W,t,r,"random")
            self.assertAlmostEqual(a[0],b[0],places=6)
            self.assertLessEqual(a[2],m.TOL)
            self.assertLessEqual(b[2],m.TOL)
            self.assertTrue(torch.equal(y[:,:-1,:],x[:,:-1,:]))
            self.assertTrue(torch.equal(z[:,:-1,:],x[:,:-1,:]))
            self.assertTrue(torch.equal(orig,x))
            self.assertTrue(torch.equal(Worig,W))
    def test_invalid_shapes_fail_closed(self):
        W=torch.zeros(1536,5120,dtype=torch.float32)
        x=torch.zeros(1,1,5120,dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            m.match_projected(x,W,np.arange(32),np.arange(32,64),"top")
        with self.assertRaises(RuntimeError):
            m.match_projected(x,W[:100,:],np.arange(32),np.arange(32,64),"top")
        with self.assertRaises(RuntimeError):
            m.match_projected(x,W,np.arange(32),np.arange(32),"top")
    def test_full_protocol_pinned(self):
        self.assertEqual(m.PROTOCOL_SHA,"1cad05cb07fa9e35d38b42d181a082c5777e6a7b8d23dd29a96a395fe1067916")
        self.assertEqual(m.REPS,(0,1))
        self.assertEqual(m.TOL,.005)
if __name__=="__main__":unittest.main()
