"""EXP053 fail-closed software checks; ephemeral test keys are NOT human approvals."""
import base64
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
except ImportError:  # Portable hosted CI must never require private reviewer keys.
    Ed25519PrivateKey=None
    serialization=None
R=Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(R))
import exp053_race_review as x
PRIVATE=Path("/media/scott/SSD_4TB/exp053-private-20261016")
HAS_PRIVATE=(PRIVATE/"packet.json").is_file()
HAS_SOURCE=x.SOURCE.is_file() and importlib.util.find_spec("pyarrow") is not None
ATTEST="I independently reviewed these source passages and responses; no model predictions or other reviewer labels were provided to me."
class TestEXP053(unittest.TestCase):
    @unittest.skipUnless(HAS_SOURCE,"private local RACE source and pyarrow unavailable")
    def test_64_article_disjoint_cohort(self):
        cases=x.cohort(x.corpus())
        self.assertEqual(len(cases),64)
        self.assertEqual(len({z["article_id"] for z in cases}),64)
        self.assertEqual(len({z["case_id"] for z in cases}),64)
    def test_manifest_no_redistribution(self):
        v=json.loads(x.MANIFEST.read_text())
        self.assertEqual(x.filehash(x.MANIFEST),x.MANIFEST_SHA)
        self.assertEqual(v["candidate_articles"],64)
        self.assertFalse(v["evaluation_admitted"])
        self.assertFalse({"question","article","options"} & set(v["rows"][0]))
    @unittest.skipUnless(HAS_PRIVATE,"private local blinded source packet unavailable")
    def test_packet_sealed_by_manifest(self):
        packet=json.loads((PRIVATE/"packet.json").read_text())
        key=json.loads((PRIVATE/"sealed_exam_key.json").read_text())
        m=json.loads(x.MANIFEST.read_text())
        self.assertEqual(x.digest(packet),m["packet_sha256"])
        self.assertEqual(x.digest(key["answer_key"]),m["sealed_answer_key_sha256"])
    @unittest.skipUnless(HAS_PRIVATE,"private local blinded source packet unavailable")
    def test_unsigned_incomplete_review_denied(self):
        packet=json.loads((PRIVATE/"packet.json").read_text())
        review=json.loads((PRIVATE/"reviewer_A.json").read_text())
        with self.assertRaisesRegex(ValueError,"HOLD"):x.decisions_for(review,packet)
    @unittest.skipUnless(HAS_PRIVATE,"private local blinded source packet unavailable")
    def test_source_quote_requirement(self):
        packet=json.loads((PRIVATE/"packet.json").read_text())
        bad={"decision":"accept","supported_option":"A","foil_option":"B",
             "support_quote":"This quotation does not occur in the passage.",
             "foil_reason":"Unsupported by the article"}
        with self.assertRaisesRegex(ValueError,"HOLD"):x.check_decision(bad,packet["cases"][0])
    @unittest.skipUnless(Ed25519PrivateKey is not None,"cryptography not installed in hosted CI")
    def test_ed25519_signature_and_tamper(self):
        k=Ed25519PrivateKey.generate()
        p=k.public_key().public_bytes(encoding=serialization.Encoding.Raw,
                                      format=serialization.PublicFormat.Raw)
        second=Ed25519PrivateKey.generate().public_key().public_bytes(
            encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)
        trust={"tester":base64.b64encode(p).decode(),
               "other":base64.b64encode(second).decode()}
        record={"schema":"exp053-review-v1","reviewer_id":"tester","role":"independent",
                "packet_sha256":"test-only","attestation":ATTEST,"decisions":[]}
        record["signature"]=k.sign(x.canonical(record)).hex()
        self.assertEqual(x.verify_signed(record,trust),"tester")
        bad=copy.deepcopy(record);bad["packet_sha256"]="modified"
        with self.assertRaisesRegex(ValueError,"HOLD"):x.verify_signed(bad,trust)
    @unittest.skipUnless(Ed25519PrivateKey is not None,"cryptography not installed in hosted CI")
    def test_missing_human_attestation_denied(self):
        k=Ed25519PrivateKey.generate()
        p=k.public_key().public_bytes(encoding=serialization.Encoding.Raw,
                                      format=serialization.PublicFormat.Raw)
        r={"schema":"exp053-review-v1","reviewer_id":"tester","role":"independent",
           "packet_sha256":"test","attestation":"","decisions":[]}
        r["signature"]=k.sign(x.canonical(r)).hex()
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.verify_signed(r,{"tester":base64.b64encode(p).decode()})
    @unittest.skipUnless(HAS_PRIVATE and HAS_SOURCE,"private local source unavailable")
    def test_live_unsigned_gate_stays_hold(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);t=d/"trust.json";t.write_text("{}")
            with self.assertRaisesRegex(ValueError,"HOLD"):
                x.gate(PRIVATE,t,d/"admitted.json")
            self.assertFalse((d/"admitted.json").exists())
    def test_unapproved_reviewer_roster_denied(self):
        self.assertEqual(x.APPROVED_REVIEWER_ROSTER_SHA256,"")
        if not HAS_PRIVATE:self.skipTest("private cohort absent")
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            trust=directory/"self_selected_roster.json"
            trust.write_text("{}")
            with self.assertRaisesRegex(ValueError,"HOLD"):
                x.verified_admission(PRIVATE,trust)

    def test_malformed_review_schema_is_hold(self):
        trust={"reviewer_A":"a"*44,"reviewer_B":"b"*44}
        for payload in (None,[],{},{"schema":"exp053-review-v1",
                                     "reviewer_id":"reviewer_A",
                                     "role":"independent"}):
            with self.assertRaisesRegex(ValueError,"HOLD"):
                x.verify_signed(payload,trust)

    def test_malformed_decision_is_hold(self):
        case={"article":"This passage contains a simple source citation fragment."}
        for payload in (None,[],{"decision":"accept","supported_option":[],
                                 "foil_option":"B","support_quote":None,"foil_reason":12},
                        {"decision":"reject","reject_reason":None}):
            with self.assertRaisesRegex(ValueError,"HOLD"):
                x.check_decision(payload,case)

    def test_review_acceptance_rejects_empty_or_multiletter_answer_and_foil(self):
        # Synthetic-only regression: Python `"" in "ABCD"` and `"AB" in "ABCD"`
        # were TRUE. Signed reviewer labels must be exact A/B/C/D, never
        # accepted as substring matches into the trusted source gate.
        article = ("In this synthetic paragraph, a sealed rainfall gauge "
                   "measured rainfall during a controlled experiment.")
        valid = {"decision":"accept","supported_option":"A","foil_option":"B",
                 "support_quote":"a sealed rainfall gauge measured rainfall",
                 "foil_reason":"The other choice is not supported by the synthetic passage."}
        self.assertEqual(x.check_decision(valid,{"article":article}),"accept")
        invalid = (("supported_option", ""), ("foil_option", ""),
                   ("supported_option", "AB"), ("foil_option", "AB"),
                   ("supported_option", "BCD"), ("foil_option", "BC"),
                   ("supported_option", 1), ("foil_option", None))
        for field,value in invalid:
            with self.subTest(field=field,value=value):
                bad={**valid,field:value}
                with self.assertRaisesRegex(ValueError,"HOLD"):
                    x.check_decision(bad,{"article":article})

        packet={"cases":[{"case_id":"synthetic-case","article":article}]}
        bad_review={"packet_sha256":x.digest(packet),"role":"independent",
                    "decisions":[{"case_id":"synthetic-case",**valid,"foil_option":"AB"}]}
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.decisions_for(bad_review,packet)

    def test_invalid_multiletter_review_refused_before_signer_key_access(self):
        # Signing a review with an invalid foil is blocked before loading any
        # key; this regression creates no key or identity, only synthetic JSON.
        from unittest.mock import patch
        import exp053_sign_review as signer
        article=("Only this synthetic observation contains a detailed "
                 "citation for an invented observation.")
        packet={"schema":"exp053-packet-v1", "cases":[
            {"case_id":"synthetic-case","article":article}]}
        review={"schema":"exp053-review-v1","reviewer_id":"synthetic_test_only",
                "role":"independent","packet_sha256":x.digest(packet),
                "attestation":ATTEST,"signature":"", "decisions":[
                    {"case_id":"synthetic-case","decision":"accept",
                     "supported_option":"A","foil_option":"AB",
                     "support_quote":"this synthetic observation contains a detailed",
                     "foil_reason":"This other choice lacks supporting synthetic evidence."}]}
        with tempfile.TemporaryDirectory(prefix="exp053-invalid-label-signer-test-") as directory:
            root=Path(directory)
            packet_path=root/"synthetic_packet.json"
            draft_path=root/"synthetic_review.json"
            output=root/"must_not_be_signed.json"
            packet_path.write_text(json.dumps(packet))
            draft_path.write_text(json.dumps(review))
            with patch.object(signer,"get_key",side_effect=AssertionError(
                    "invalid review must fail before loading a signing key")):
                with self.assertRaisesRegex(ValueError,"HOLD"):
                    signer.sign(draft_path,packet_path,root/"no-real-key.pem",output)
            self.assertFalse(output.exists())

    def test_unsupported_decision_type_is_hold(self):
        packet={"cases":[{"case_id":"a",
                           "article":"Source sentence long enough for a quote."}]}
        rev={"packet_sha256":x.digest(packet),"role":"independent",
             "decisions":[None]}
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.decisions_for(rev,packet)

    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.build(PRIVATE,x.MANIFEST)
    @unittest.skipUnless(Ed25519PrivateKey is not None,"cryptography not installed in hosted CI")
    def test_reused_signing_key_two_reviewer_ids_denied(self):
        key=Ed25519PrivateKey.generate()
        pub=key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)
        shared=base64.b64encode(pub).decode()
        r={"schema":"exp053-review-v1","reviewer_id":"reviewer_1",
           "role":"independent","packet_sha256":"test",
           "attestation":ATTEST,"decisions":[]}
        r["signature"]=key.sign(x.canonical(r)).hex()
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.verify_signed(r,{"reviewer_1":shared,"reviewer_2":shared})
    def test_third_review_must_cover_exact_disputed_subset(self):
        packet={"cases":[
            {"case_id":"a","article":"A review-supported article passage with explicit evidence.",
             "question":"Who performed the action?","options":{"A":"Ana","B":"Bob"}},
            {"case_id":"b","article":"Another paragraph with enough words for a citation.",
             "question":"What was recorded?","options":{"A":"Blue","B":"Red"}}]}
        review={"packet_sha256":x.digest(packet),"role":"adjudicator",
                "decisions":[{"case_id":"a","decision":"reject",
                              "reject_reason":"No uniquely supported option"}]}
        r=x.decisions_for(review,packet,required_ids={"a"})
        self.assertEqual(set(r),{"a"})
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.decisions_for(review,packet,required_ids={"b"})
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.decisions_for(review,packet)
    def test_corrupt_multiletter_exam_key_never_enters_frozen_cohort(self):
        rows=[{"example_id":f"synthetic_unique_case_{i}",
               "article":(f"Independent synthetic text case {i}; " + "filler "*55),
               "question":"Which fictional answer is supported by this sentence?",
               "options":["north","east","south","west"],"answer":"A"}
              for i in range(64)]
        self.assertEqual(len(x.cohort(rows)),64)
        for bad in ("", "AB", "BCD", None, 1):
            with self.subTest(bad=bad):
                damaged=copy.deepcopy(rows)
                damaged[0]["answer"]=bad
                with self.assertRaisesRegex(ValueError,"HOLD"):
                    x.cohort(damaged)

    def test_same_article_under_64_ids_denied(self):
        repeated=[{"example_id":f"id_{i}","article":"A "*150,
                   "question":f"What happens when condition {i} occurs?",
                   "options":["A1","B1","C1","D1"],"answer":"A"}
                  for i in range(64)]
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.cohort(repeated)
if __name__=="__main__":unittest.main()
