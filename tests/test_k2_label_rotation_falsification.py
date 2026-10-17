"""EXP037 deterministic source-disjoint label rotations; offline-only tests."""
import importlib.util
import pathlib
import sys
import unittest
P=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location("k2_label_rotation_falsification",P/"k2_label_rotation_falsification.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestEXP037(unittest.TestCase):
    def test_five_article_groups_disjoint(self):
        items=m.cohort()
        self.assertEqual(len(items),10)
        self.assertEqual(len({x[0] for x in items}),5)
        self.assertEqual(sum(x[1] for x in items),5)
    def test_rotation_is_bijection(self):
        for gold in (0,2):
            self.assertEqual(set((gold+r)%3 for r in range(3)),{0,1,2})
        # index-to-semantic inversion used by the evaluator
        for gold in range(3):
            for rot in range(3):
                predicted=(gold+rot)%3
                self.assertEqual((predicted-rot)%3,gold)
    def test_manuscript_output_has_semantic_robustness(self):
        fake={"total":0,"answerable":0,"unknown":0,"all3":0,"ans_all3":0,
             "unk_all3":0,"semantic_stable":0,"rot_correct":"0/0/0",
             "pred_labels":"0/0/0","nll":0.0}
        g={(k,r):1.0 for k in m.LAYERS for r in (1,2)}
        rows={a:{mode:(fake,g) for mode in ("default","yarn")} for a in (0.,1.)}
        output=m.report(rows)
        self.assertIn(m.PROTOCOL_SHA,output)
        self.assertIn("Semantic choice stable",output)
        self.assertEqual(sum(x.startswith("| 0 | default |") for x in output.splitlines()),3)
        self.assertEqual(sum(x.startswith("| 1 | yarn |") for x in output.splitlines()),3)
if __name__=="__main__":unittest.main()
