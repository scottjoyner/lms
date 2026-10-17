"""EXP053 fail-closed software checks; ephemeral test keys are NOT human approvals."""
import base64,copy,json,sys,tempfile,unittest
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
R=Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(R))
import exp053_race_review as x
PRIVATE=Path("/media/scott/SSD_4TB/exp053-private-20261016")
ATTEST="I independently reviewed these source passages and responses; no model predictions or other reviewer labels were provided to me."
class TestEXP053(unittest.TestCase):
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
    def test_packet_sealed_by_manifest(self):
        packet=json.loads((PRIVATE/"packet.json").read_text())
        key=json.loads((PRIVATE/"sealed_exam_key.json").read_text())
        m=json.loads(x.MANIFEST.read_text())
        self.assertEqual(x.digest(packet),m["packet_sha256"])
        self.assertEqual(x.digest(key["answer_key"]),m["sealed_answer_key_sha256"])
    def test_unsigned_incomplete_review_denied(self):
        packet=json.loads((PRIVATE/"packet.json").read_text())
        review=json.loads((PRIVATE/"reviewer_A.json").read_text())
        with self.assertRaisesRegex(ValueError,"HOLD"):x.decisions_for(review,packet)
    def test_source_quote_requirement(self):
        packet=json.loads((PRIVATE/"packet.json").read_text())
        bad={"decision":"accept","supported_option":"A","foil_option":"B",
             "support_quote":"This quotation does not occur in the passage.",
             "foil_reason":"Unsupported by the article"}
        with self.assertRaisesRegex(ValueError,"HOLD"):x.check_decision(bad,packet["cases"][0])
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
    def test_missing_human_attestation_denied(self):
        k=Ed25519PrivateKey.generate()
        p=k.public_key().public_bytes(encoding=serialization.Encoding.Raw,
                                      format=serialization.PublicFormat.Raw)
        r={"schema":"exp053-review-v1","reviewer_id":"tester","role":"independent",
           "packet_sha256":"test","attestation":"","decisions":[]}
        r["signature"]=k.sign(x.canonical(r)).hex()
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.verify_signed(r,{"tester":base64.b64encode(p).decode()})
    def test_live_unsigned_gate_stays_hold(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);t=d/"trust.json";t.write_text("{}")
            with self.assertRaisesRegex(ValueError,"HOLD"):
                x.gate(PRIVATE,t,d/"admitted.json")
            self.assertFalse((d/"admitted.json").exists())
    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.build(PRIVATE,x.MANIFEST)
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
    def test_same_article_under_64_ids_denied(self):
        repeated=[{"example_id":f"id_{i}","article":"A "*150,
                   "question":f"What happens when condition {i} occurs?",
                   "options":["A1","B1","C1","D1"],"answer":"A"}
                  for i in range(64)]
        with self.assertRaisesRegex(ValueError,"HOLD"):
            x.cohort(repeated)
if __name__=="__main__":unittest.main()
