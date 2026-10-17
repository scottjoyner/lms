"""EXP053 tokenizer preparation portability and tamper audit, never model forwards."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(R))
import exp053_tokenizer_preflight as t
class TokenizerPreflight(unittest.TestCase):
    def test_frozen_spec_matches(self):
        self.assertTrue(t.SPEC.is_file())
        self.assertEqual(t.filehash(t.SPEC),t.SPEC_SHA)
        self.assertEqual(t.filehash(t.PLAN),t.PLAN_SHA)
    def test_64_article_report_has_no_forward_calls_or_private_text(self):
        p=R/"EXP053-TOKENIZER-PREFLIGHT-2026-10-16.json"
        self.assertTrue(p.is_file())
        report=json.loads(p.read_text())
        self.assertEqual(report["model_forward_passes"],0)
        self.assertEqual(report["human_reviewed_articles"],0)
        self.assertEqual(report["candidate_articles"],64)
        self.assertEqual(report["source_conditions"],3)
        self.assertEqual(report["label_rotations"],2)
        self.assertEqual(len(report["models"]),3)
        self.assertFalse(report["approved_only_final_decoy_checked"])
        for m in report["models"].values():
            self.assertEqual(m["source_prompts_tokenized"],384)
            self.assertEqual(m["label_boundary_checks_passed"],1536)
            self.assertEqual(m["model_forward_passes"],0)
            self.assertLess(m["prompt_tokens_max"],4096)
            self.assertEqual(len(set(m["candidate_answer_token_ids"].values())),4)
        self.assertFalse(any(x in report for x in ("articles","prompts","case_ids","answer_key")))
        script=(R/"exp053_tokenizer_preflight.py").read_text()
        self.assertNotIn("AutoModelFor",script)
        self.assertNotIn("from torch",script)
    def test_question_and_option_rotation_preserve_semantics(self):
        case={"question":"What was stated?","options":
              {"A":"first","B":"second","C":"third","D":"fourth"}}
        self.assertEqual(t.options_for(case,0),
                         {"A":"first","B":"second","C":"third","D":"fourth"})
        self.assertEqual(t.options_for(case,1),
                         {"A":"third","B":"fourth","C":"first","D":"second"})
        s=t.prompt(case["question"],t.options_for(case,0),"sealed source text")
        self.assertTrue(s.endswith("Answer:"))
        self.assertNotIn("Answer: ",s)
        self.assertIn("Passage:\nsealed source text\n",s)
        self.assertIn("A) first\nB) second\n",s)
    def test_tampered_spec_or_plan_denied_before_tokenization(self):
        with patch.object(t,"SPEC_SHA","0"*64):
            with self.assertRaisesRegex(ValueError,"HOLD"):
                t.validate_custody()
    def test_unknown_rotation_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            t.options_for({"options":dict(zip("ABCD","wxyz"))},4)
    def test_deterministic_source_arm_cardinality(self):
        private=Path("/media/scott/SSD_4TB/exp053-private-20261016/packet.json")
        if not private.is_file():self.skipTest("private research packet absent on hosted CI")
        cases={c["case_id"]:c for c in json.loads(private.read_text())["cases"]}
        rows=list(t.prompts(cases))
        self.assertEqual(len(rows),384)
        self.assertEqual(len({(c,r,a) for c,r,a,_ in rows}),384)
        self.assertTrue(all(s.endswith("Answer:") for _,_,_,s in rows))
        self.assertFalse(any("sealed_exam_key" in s for _,_,_,s in rows))
if __name__=="__main__":unittest.main()
