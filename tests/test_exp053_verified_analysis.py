"""Synthetic source receipt checks: NO actual model scores or human approvals."""
import sys
import tempfile
import unittest
from pathlib import Path

R=Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(R))
import exp053_verified_analysis as v

PRIVATE=Path("/media/scott/SSD_4TB/exp053-private-20261016")
HAS_PACKET=(PRIVATE/"packet.json").is_file()

class VerifiedScoreReplay(unittest.TestCase):
    @unittest.skipUnless(HAS_PACKET,"private source not installed on hosted CI")
    def test_preflight_source_receipt_locked(self):
        packet,manifest,scan=v.require_sources()
        self.assertEqual(len(packet["cases"]),64)
        self.assertEqual(len(manifest["rows"]),64)
        self.assertEqual(len(scan["models"]),3)

    def fixture(self):
        if not HAS_PACKET:self.skipTest("private source not installed")
        packet,manifest,scan=v.require_sources()
        cases={c["case_id"]:c for c in packet["cases"]}
        cid=sorted(cases)[0]
        opts=v.options_for(cases[cid],0)
        model=sorted(scan["models"])[0]
        approved={x:{"supported":"A","foil":"B"} for x in sorted(cases)[:40]}
        rec={"case_id":cid,"rotation":0,"arm":"supported","model":model,
             "article_sha256":v.sha(cases[cid]["article"]),
             "context_article_sha256":v.sha(cases[cid]["article"]),
             "option_order_sha256":v.digest(opts),
             "prompt_sha256":v.sha(v.prompt(cases[cid]["question"],opts,cases[cid]["article"])),
             "tokenizer_sha256":scan["models"][model]["tokenizer_json_sha256"]}
        rec.update({"gold_label":"A","foil_label":"B",
                    "checkpoint_sha256":"a"*64,"mode":"raw",
                    "gold_logp":-1.5,"foil_logp":-2.0,
                    "prompt_token_count":300,"gold_token_count":1,
                    "foil_token_count":1})
        return packet,scan,approved,rec

    @staticmethod
    def weights(scan):
        return {model:"a"*64 for model in scan["models"]}

    def test_no_independently_pinned_weight_receipt_denied(self):
        packet,scan,approved,rec=self.fixture()
        self.assertEqual(v.WEIGHT_REPORT_SHA,"")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            v.admitted_weight_inventory(scan)

    def test_wrong_checkpoint_digest_denied_even_if_source_matches(self):
        packet,scan,approved,rec=self.fixture()
        inventory=self.weights(scan)
        rec["checkpoint_sha256"]="b"*64
        with self.assertRaisesRegex(ValueError,"HOLD"):
            v.check_private_archive([rec],approved,packet,scan,inventory)
        rec["checkpoint_sha256"]="a"*64
        self.assertEqual(v.check_private_archive([rec],approved,packet,scan,inventory),1)

    def test_only_source_exact_prompt_accepted_as_receipt(self):
        packet,scan,approved,rec=self.fixture()
        self.assertEqual(v.check_private_archive([rec],approved,packet,scan,self.weights(scan)),1)

    def test_changed_prompt_hash_denied(self):
        packet,scan,approved,rec=self.fixture()
        rec["prompt_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"HOLD"):
            v.check_private_archive([rec],approved,packet,scan,self.weights(scan))

    def test_option_rotation_hash_denied(self):
        packet,scan,approved,rec=self.fixture()
        rec["option_order_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"HOLD"):
            v.check_private_archive([rec],approved,packet,scan,self.weights(scan))

    def test_tokenizer_misidentification_denied(self):
        packet,scan,approved,rec=self.fixture()
        rec["tokenizer_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"HOLD"):
            v.check_private_archive([rec],approved,packet,scan,self.weights(scan))

    def test_unapproved_paper_denied(self):
        packet,scan,approved,rec=self.fixture()
        approved.pop(rec["case_id"])
        approved[sorted({c["case_id"] for c in packet["cases"]}-set(approved)-{rec["case_id"]})[0]]={
            "supported":"A","foil":"B"}
        with self.assertRaisesRegex(ValueError,"HOLD"):
            v.check_private_archive([rec],approved,packet,scan,self.weights(scan))

    @unittest.skipUnless(HAS_PACKET,"private source not installed on hosted CI")
    def test_missing_human_review_refuses_report_before_scores(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            trust=d/"trust.json"
            trust.write_text("{}")
            output=R/"EXP053-DO-NOT-CREATE-FROM-SYNTHETIC.json"
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError,"HOLD"):
                v.replay(PRIVATE,trust,PRIVATE/"no-scores.json",output)
            self.assertFalse(output.exists())

    def test_public_metrics_are_not_inference_engines(self):
        source=(R/"exp053_verified_analysis.py").read_text()
        self.assertNotIn("AutoModelFor",source)
        self.assertNotIn("import torch",source)
        self.assertNotIn("from transformers",source)

if __name__=="__main__":
    unittest.main()
