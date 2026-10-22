"""Prospective EXP053 score-receipt binding tests; entirely synthetic article text.

No actual RACE data, reviewer approvals, model checkpoints or model scores.
"""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

RESEARCH=Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(RESEARCH))
import exp053_verified_analysis as verification

def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

class SyntheticSourceAndLabelBinding(unittest.TestCase):
    def setUp(self):
        self.packet={"cases":[
            {"case_id":f"fake_case_{i:02d}",
             "article":f"Synthetic case {i} contains independent no-license source text.",
             "question":"Which fictional alternative is supported by this dummy paragraph?",
             "options":{"A":"Synthesized A","B":"Synthesized B",
                        "C":"Synthesized C","D":"Synthesized D"}}
            for i in range(40)]}
        self.approved={c["case_id"]:{"supported":"A","foil":"B"}
                       for c in self.packet["cases"]}
        self.scan=json.loads(verification.PREFLIGHT.read_text())
        self.model=sorted(self.scan["models"])[0]
        self.checkpoints={k:"c"*64 for k in self.scan["models"]}
        self.case=self.packet["cases"][0]
        self.next_case=self.packet["cases"][1]

    def receipt(self,arm="supported",rotation=0):
        case=self.case
        opts=verification.options_for(case,rotation)
        passage=(case["article"] if arm=="supported" else
                 verification.WITHHELD if arm=="withheld" else self.next_case["article"])
        gold,foil=(("A","B") if rotation==0 else ("C","D"))
        return {"case_id":case["case_id"],"rotation":rotation,"arm":arm,
                "model":self.model,"article_sha256":digest(case["article"]),
                "context_article_sha256":("WITHHELD" if arm=="withheld" else digest(passage)),
                "option_order_sha256":verification.digest(opts),
                "prompt_sha256":digest(verification.prompt(case["question"],opts,passage)),
                "tokenizer_sha256":self.scan["models"][self.model]["tokenizer_json_sha256"],
                "checkpoint_sha256":self.checkpoints[self.model],
                "gold_label":gold,"foil_label":foil,
                "prompt_token_count":318,"gold_token_count":1,"foil_token_count":1,
                "gold_logp":-1.4,"foil_logp":-2.8,"mode":"raw"}

    def verify(self,r):
        return verification.check_private_archive(
            [r],self.approved,self.packet,self.scan,self.checkpoints)

    def test_all_source_and_rotation_combinations_accept_only_synthetic_receipt(self):
        for arm in ("supported","withheld","decoy"):
            for rotation in (0,1):
                with self.subTest(arm=arm,rotation=rotation):
                    self.assertEqual(self.verify(self.receipt(arm,rotation)),1)

    def test_article_identity_mismatch_denied(self):
        r=self.receipt()
        r["article_sha256"]=digest("different fabricated article")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.verify(r)

    def test_supported_or_decoy_context_mismatch_denied(self):
        for arm in ("supported","decoy","withheld"):
            r=self.receipt(arm)
            r["context_article_sha256"]=digest("substituted synthetic source")
            with self.assertRaisesRegex(ValueError,"HOLD"):
                self.verify(r)

    def test_wrong_next_approved_decoy_article_denied(self):
        r=self.receipt("decoy")
        r["context_article_sha256"]=digest(self.packet["cases"][2]["article"])
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.verify(r)

    def test_swapped_gold_or_foil_denied_without_model_scores(self):
        for rotation in (0,1):
            for label in ("gold_label","foil_label"):
                r=self.receipt(rotation=rotation)
                r[label]="D" if rotation==0 else "A"
                with self.assertRaisesRegex(ValueError,"HOLD"):
                    self.verify(r)

    def test_swapped_answer_text_rotation_denied(self):
        r=self.receipt(rotation=1)
        r["option_order_sha256"]=verification.digest(
            verification.options_for(self.case,0))
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.verify(r)

    def test_prompt_question_or_passage_changed_denied(self):
        r=self.receipt()
        r["prompt_sha256"]=digest("a different fabricated prompt")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.verify(r)

    def test_extra_or_missing_fields_denied_including_private_source(self):
        base=self.receipt()
        for change in ("extra","missing"):
            r=copy.deepcopy(base)
            if change=="extra":r["private_article_text"]="Never upload this"
            else:r.pop("context_article_sha256")
            with self.assertRaisesRegex(ValueError,"HOLD"):
                self.verify(r)

    def test_absent_checkpoint_authorization_denied(self):
        r=self.receipt()
        with self.assertRaisesRegex(ValueError,"HOLD"):
            verification.check_private_archive(
                [r],self.approved,self.packet,self.scan,None)

    def test_wrong_weight_digest_and_wrong_tokenizer_denied(self):
        for field in ("checkpoint_sha256","tokenizer_sha256"):
            r=self.receipt()
            r[field]="d"*64
            with self.assertRaisesRegex(ValueError,"HOLD"):
                self.verify(r)

    def test_nonstring_case_identity_denied_explicitly(self):
        for value in (["fake_case_00"],None,42):
            r=self.receipt()
            r["case_id"]=value
            with self.assertRaisesRegex(ValueError,"HOLD"):
                self.verify(r)

    def test_corrupt_candidate_lengths_and_mode_denied(self):
        for field,value in (("prompt_token_count",4097),
                            ("gold_token_count",2),("foil_token_count",2),
                            ("mode","chat")):
            r=self.receipt()
            r[field]=value
            with self.assertRaisesRegex(ValueError,"HOLD"):
                self.verify(r)

    def test_boolean_one_token_counts_not_accepted_as_integer(self):
        for key in ("gold_token_count","foil_token_count"):
            row=self.receipt()
            row[key]=True
            with self.assertRaisesRegex(ValueError,"HOLD"):
                self.verify(row)

    def test_private_packet_selection_bound_to_caller_not_global(self):
        # A caller-selected directory cannot silently fall back to canonical
        # private packet.json when the supplied path is missing.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"HOLD"):
                verification.require_sources(d)

if __name__=="__main__":unittest.main()
