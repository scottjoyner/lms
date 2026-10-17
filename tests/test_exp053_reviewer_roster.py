"""Synthetic enrollment and role binding only; no human approval or model inference."""
import base64
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
import exp053_reviewer_roster as roster
import exp053_race_review as review

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    Ed25519PrivateKey=None
    serialization=None


class Enrollment(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/"owner-enrollment-TEST-ONLY.json"
        self.packet="c"*64
        self.a=base64.b64encode(b"a"*32).decode()
        self.b=base64.b64encode(b"b"*32).decode()
        self.value={"schema":roster.SCHEMA,"packet_sha256":self.packet,
                    "owner_custody_reference_sha256":"d"*64,
                    "blind_assignments_confirmed":True,
                    "slots":{"A":{"reviewer_id":"test_reviewer_a","role":"independent",
                                  "public_key_b64":self.a},
                             "B":{"reviewer_id":"test_reviewer_b","role":"independent",
                                  "public_key_b64":self.b}}}

    def check(self,data=None,pin=None,permissions=0o600):
        self.path.write_text(json.dumps(self.value if data is None else data,sort_keys=True)+"\n")
        os.chmod(self.path,permissions)
        actual=hashlib.sha256(self.path.read_bytes()).hexdigest()
        return roster.load_roster(self.path,self.packet,
                                  actual if pin is None else pin,ROOT)

    def test_approved_root_is_still_empty(self):
        self.assertEqual(review.APPROVED_REVIEWER_ROSTER_SHA256,"")
        with self.assertRaisesRegex(ValueError,"HOLD"):self.check(pin="")

    def test_synthetic_slot_map_parses(self):
        r=self.check()
        self.assertEqual(set(r["slots"]),{"A","B"})
        self.assertEqual(set(r["keys"]),{"test_reviewer_a","test_reviewer_b"})

    def test_duplicate_key_or_reviewer_id_rejected(self):
        for field in ("public_key_b64","reviewer_id"):
            r=copy.deepcopy(self.value)
            r["slots"]["B"][field]=r["slots"]["A"][field]
            with self.assertRaisesRegex(ValueError,"HOLD"):self.check(r)

    def test_wrong_slot_role_and_extra_slot_rejected(self):
        r=copy.deepcopy(self.value);r["slots"]["A"]["role"]="adjudicator"
        with self.assertRaisesRegex(ValueError,"HOLD"):self.check(r)
        r=copy.deepcopy(self.value);r["slots"]["other"]=r["slots"]["B"]
        with self.assertRaisesRegex(ValueError,"HOLD"):self.check(r)

    def test_adjudicator_role_enforced(self):
        r=copy.deepcopy(self.value)
        r["slots"]["ADJ"]={"reviewer_id":"test_adjudicator","role":"adjudicator",
                            "public_key_b64":base64.b64encode(b"c"*32).decode()}
        self.assertEqual(set(self.check(r)["slots"]),{"A","B","ADJ"})
        r["slots"]["ADJ"]["role"]="independent"
        with self.assertRaisesRegex(ValueError,"HOLD"):self.check(r)

    def test_packet_or_owner_record_mismatch_rejected(self):
        for field,value in (("packet_sha256","a"*64),
                            ("owner_custody_reference_sha256",""),
                            ("blind_assignments_confirmed",False)):
            r=copy.deepcopy(self.value);r[field]=value
            with self.assertRaisesRegex(ValueError,"HOLD"):self.check(r)

    def test_unapproved_or_world_readable_file_rejected(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):self.check(pin="0"*64)
        with self.assertRaisesRegex(ValueError,"HOLD"):self.check(permissions=0o644)

    def test_unexpected_source_or_bad_key_rejected(self):
        r=copy.deepcopy(self.value);r["article"]="synthetic text must not enter roster"
        with self.assertRaisesRegex(ValueError,"HOLD"):self.check(r)
        r=copy.deepcopy(self.value);r["slots"]["B"]["public_key_b64"]="bad!"
        with self.assertRaisesRegex(ValueError,"HOLD"):self.check(r)

    def test_roster_symlink_rejected(self):
        real=Path(self.tmp.name)/"actual.json";real.write_text("{}")
        self.path.symlink_to(real)
        with self.assertRaisesRegex(ValueError,"HOLD"):
            roster.load_roster(self.path,self.packet,"f"*64,ROOT)


@unittest.skipUnless(Ed25519PrivateKey is not None,"cryptography optional")
class SignedSlotBinding(unittest.TestCase):
    def test_swapped_signers_and_roles_rejected(self):
        ka=Ed25519PrivateKey.generate()
        kb=Ed25519PrivateKey.generate()
        def pub(k):
            return base64.b64encode(k.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw)).decode()
        keys={"test_reviewer_a":pub(ka),"test_reviewer_b":pub(kb)}
        slots={"A":{"reviewer_id":"test_reviewer_a","role":"independent"},
               "B":{"reviewer_id":"test_reviewer_b","role":"independent"}}
        rec={"schema":"exp053-review-v1","reviewer_id":"test_reviewer_a",
             "role":"independent","packet_sha256":"synthetic",
             "attestation":"I independently reviewed these source passages and responses; no model predictions or other reviewer labels were provided to me.",
             "decisions":[]}
        rec["signature"]=ka.sign(review.canonical(rec)).hex()
        self.assertEqual(review.require_owner_assigned_signer(rec,keys,slots,"A"),
                         "test_reviewer_a")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            review.require_owner_assigned_signer(rec,keys,slots,"B")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            review.require_owner_assigned_signer(rec,keys,slots,"ADJ")
        altered=copy.deepcopy(rec);altered["role"]="adjudicator"
        altered["signature"]=ka.sign(review.canonical(
            {k:v for k,v in altered.items() if k!="signature"})).hex()
        with self.assertRaisesRegex(ValueError,"HOLD"):
            review.require_owner_assigned_signer(altered,keys,slots,"A")


if __name__=="__main__":
    unittest.main()
