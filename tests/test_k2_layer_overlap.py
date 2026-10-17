"""LMS K2 lineage matrix: pinned custody, dense matrix, alignment and fail-closed."""
import importlib.util
import pathlib
import unittest

HERE=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("k2_layer_overlap",HERE/"research/k2_layer_overlap.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

DATA=pathlib.Path("/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026")

class K2LayerOverlapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=module.pinned(DATA)

    def test_exact_sha256_and_six_checkpoint_order(self):
        self.assertEqual(tuple(self.e["018"]["checkpoint_order"]),module.STAGES)
        self.assertEqual(len(module.PINS),4)
        self.assertEqual(self.e["018"]["source_groups"],32)

    def test_real_full_layer_overlaps_each_transition(self):
        for transition,rows in self.e["018"]["adjacent_layerwise_CKA"].items():
            self.assertEqual(len(rows),50,transition)
            for pooling in ("mean","last"):
                for metric in ("whole_CKA","signed_role_CKA"):
                    m=module.grid(rows,pooling,metric)
                    self.assertEqual((len(m),len(m[0])),(5,5))
                    self.assertTrue(all(v is None or 0<=v<=1.00001 for row in m for v in row))

    def test_full_cross_family_matrix(self):
        for pair in ("k2_pretrain_vs_qwen08","k2_release_vs_qwen08"):
            cells=self.e["016"]["cross_model_cka"][pair]["comparisons"]
            self.assertEqual(len(cells),50)
            m=module.grid(cells,"mean","whole_cka",module.K2,module.QWEN)
            self.assertEqual(len(m),5)

    def test_identity_model_perfect(self):
        cells=self.e["018"]["adjacent_layerwise_CKA"]["mopd_249__to__main"]
        for pool in ("mean","last"):
            m=module.grid(cells,pool,"whole_CKA")
            d=module.depth_diagnostics(m)
            self.assertAlmostEqual(d["final"],1,places=8)

    def test_diagonal_gap_and_5factorial_alignment(self):
        cells=self.e["018"]["adjacent_layerwise_CKA"]["mid2_47684__to__rl_merged"]
        m=module.grid(cells,"last","whole_CKA")
        d=module.depth_diagnostics(m)
        self.assertGreater(d["final"],0.4)
        self.assertLess(d["final"],0.6)
        self.assertGreaterEqual(d["optimal_mean"],d["diag_mean"]-1e-9)
        self.assertEqual(len(d["optimal_assignment"]),5)

    def test_tampering_and_duplicate_fail_closed(self):
        rows=list(self.e["018"]["adjacent_layerwise_CKA"]["rl_merged__to__mopd_249"])
        with self.assertRaises(ValueError):
            module.grid(rows[:-1],"mean","whole_CKA")
        with self.assertRaises(ValueError):
            module.grid(rows+[rows[0]],"mean","whole_CKA")

    def test_markdown_render_reproducible_without_inference(self):
        rendered=module.report(self.e)
        self.assertIn("K2 model lineage:",rendered)
        self.assertIn("## K2↔Qwen cross-family matrices",rendered)
        self.assertIn("## Behavior lineage",rendered)
        self.assertEqual(rendered.count("####"),28)
        self.assertTrue(len(rendered)>10000)

if __name__=="__main__":unittest.main()
