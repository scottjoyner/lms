"""EXP041B only syntactic guard and source custody repair, no inference."""
import sys,pathlib,unittest
P=pathlib.Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(P))
import k2_new_source_generation_b as runner
from k2_novel_source_fixture import build,prompt,corpus_receipt
class Tests(unittest.TestCase):
    def test_unchanged_scientific_fixture(self):
        self.assertEqual(runner.FIXTURE_SHA,corpus_receipt(build()))
        self.assertEqual(runner.PROTOCOL_SHA,"f5459e7c1c382db2f322d9c4ffd47889d99848c7945e1d6c7ad79d310f6eab4c")
        self.assertEqual(runner.B_PROTOCOL_SHA,"634892caa5fdc3870b0001ed4ebf9b8db26c26eeb446a8ccdd2368d0b4b505f4")
    def test_all_prompt_lengths_match_new_syntactic_bound(self):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(runner.MID),
                        local_files_only=True,trust_remote_code=False)
        lens=[len(tok.encode(prompt(x),add_special_tokens=True)) for x in build()]
        self.assertEqual(len(lens),24)
        self.assertTrue(all(20<x<650 for x in lens))
    def test_base_fixture_scoring_unchanged(self):
        from k2_novel_source_fixture import score,score_group
        x=build();s=[score(c,c["gold"]) for c in x]
        d=score_group(x,s)
        self.assertEqual(sum(c["strict"] for c in d.values()),24)
        report=runner.render(x,{"default":d,"yarn":d},[95]*48,9.)
        self.assertIn(runner.B_PROTOCOL_SHA,report)
        self.assertIn("EXP-041B",report)
if __name__=="__main__":unittest.main()
