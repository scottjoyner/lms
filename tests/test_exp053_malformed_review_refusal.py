"""Source-free malformed reviewer inputs must produce sanitized HOLD only."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import exp053_race_review as review

ATTEST = ("I independently reviewed these source passages and responses; "
          "no model predictions or other reviewer labels were provided to me.")


class MalformedReviewRefusal(unittest.TestCase):
    def setUp(self):
        self.packet = {"cases": [
            {"case_id": "synthetic-one", "article": "Synthetic public fixture sentence only."},
            {"case_id": "synthetic-two", "article": "Another synthetic public fixture sentence."},
        ]}
        self.form = {
            "packet_sha256": review.digest(self.packet), "role": "independent",
            "decisions": [
                {"case_id": "synthetic-one", "decision": "reject",
                 "reject_reason": "Ambiguous invented synthetic question"},
                {"case_id": "synthetic-two", "decision": "reject",
                 "reject_reason": "No single invented synthetic answer"},
            ],
        }

    def assert_hold(self, fn, *args):
        with self.assertRaisesRegex(ValueError, r"^HOLD EXP053:") as held:
            fn(*args)
        self.assertNotIn("Synthetic public fixture", str(held.exception))
        self.assertNotIn("private", str(held.exception).lower())

    def test_valid_synthetic_review_still_parses(self):
        parsed = review.decisions_for(self.form, self.packet)
        self.assertEqual(set(parsed), {"synthetic-one", "synthetic-two"})
        self.assertEqual(parsed["synthetic-one"]["decision"], "reject")

    def test_malformed_signed_decision_case_ids_are_hold(self):
        for invalid in ([], {}, ["synthetic-one"], {"fake": "synthetic-one"},
                        None, 1, False, ""):
            with self.subTest(type=type(invalid).__name__, invalid=str(invalid)[:60]):
                altered = copy.deepcopy(self.form)
                altered["decisions"][0]["case_id"] = invalid
                altered["packet_sha256"] = review.digest(self.packet)
                self.assert_hold(review.decisions_for, altered, self.packet)

    def test_malformed_source_case_ids_and_duplicates_are_hold(self):
        for invalid in ([], {}, None, False, 1, ""):
            with self.subTest(invalid=str(invalid)):
                packet = copy.deepcopy(self.packet)
                packet["cases"][0]["case_id"] = invalid
                altered = copy.deepcopy(self.form)
                altered["packet_sha256"] = review.digest(packet)
                self.assert_hold(review.decisions_for, altered, packet)
        packet = copy.deepcopy(self.packet)
        packet["cases"][1]["case_id"] = "synthetic-one"
        altered = copy.deepcopy(self.form)
        altered["packet_sha256"] = review.digest(packet)
        self.assert_hold(review.decisions_for, altered, packet)

    def test_invalid_source_case_shapes_are_hold(self):
        for bad in (None, [], "synthetic", {"case_id": "synthetic-one"},
                    {"case_id": "synthetic-one", "article": None}):
            with self.subTest(bad=str(bad)[:50]):
                packet = {"cases": [bad]}
                candidate = {"packet_sha256": review.digest(packet),
                             "role": "independent", "decisions": []}
                self.assert_hold(review.decisions_for, candidate, packet)
        for packet in (None, {}, [], {"cases": None}):
            with self.subTest(packet=type(packet).__name__):
                candidate = {"packet_sha256": review.digest(packet),
                             "role": "independent", "decisions": []}
                self.assert_hold(review.decisions_for, candidate, packet)

    def test_third_party_subset_rejects_nonstring_and_duplicate_ids(self):
        for subset in ([[]], [{}], [None], ["synthetic-one", "synthetic-one"],
                       "synthetic-one", None):
            if subset is None:
                continue  # None deliberately means A/B full-case scope
            with self.subTest(subset=str(subset)[:80]):
                candidate = {"packet_sha256": review.digest(self.packet),
                             "role": "adjudicator", "decisions": [self.form["decisions"][0]]}
                self.assert_hold(review.decisions_for, candidate,
                                 self.packet, subset)
        adjudicator = {"packet_sha256": review.digest(self.packet),
                       "role": "adjudicator", "decisions": [self.form["decisions"][0]]}
        self.assertEqual(set(review.decisions_for(adjudicator, self.packet,
                                                  {"synthetic-one"})), {"synthetic-one"})

    def test_unhashable_and_empty_trust_keys_are_hold(self):
        unsigned = {
            "schema": "exp053-review-v1", "reviewer_id": "fake-A",
            "role": "independent", "packet_sha256": "synthetic-only",
            "attestation": ATTEST, "decisions": [], "signature": "",
        }
        for bad in ([], {}, None, 5, ""):
            with self.subTest(bad=str(bad)[:40]):
                self.assert_hold(review.verify_signed, unsigned,
                                 {"fake-A": bad, "fake-B": "synthetic-placeholder"})
        self.assert_hold(review.verify_signed, unsigned,
                         {"fake-A": "x", "fake-B": "x"})
        self.assert_hold(review.verify_signed, unsigned, {"fake-A": "x", "": "y"})

    def test_bad_review_is_refused_before_signer_key_access(self):
        try:
            import exp053_sign_review as signer
        except ImportError:
            self.skipTest("cryptography unavailable: human key signer is optional")
        with tempfile.TemporaryDirectory(prefix="exp053-malformed-case-sign-") as folder:
            home = Path(folder)
            packet_path = home / "synthetic-packet.json"
            draft_path = home / "synthetic-draft.json"
            output = home / "must-not-be-signed.json"
            packet_path.write_text(json.dumps(self.packet))
            signed_input = {
                **self.form,
                "schema": "exp053-review-v1",
                "reviewer_id": "synthetic-reviewer",
                "attestation": ATTEST,
                "signature": "",
            }
            signed_input["decisions"][0]["case_id"] = ["synthetic-one"]
            draft_path.write_text(json.dumps(signed_input))
            with patch.object(signer, "get_key", side_effect=AssertionError(
                    "malformed review must never attempt to read a key")):
                self.assert_hold(signer.sign, draft_path, packet_path,
                                 home / "no-key.pem", output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
