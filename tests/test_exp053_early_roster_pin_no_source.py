"""Early human-owner HOLD without reading licensed source or reviewer forms."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import exp053_race_review as review


class UnapprovedRosterCannotReadSource(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="exp053-no-owner-pin-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.private = self.folder / "missing-human-reviews"
        self.roster = self.folder / "missing-owner-approved-roster"
        self.result = self.folder / "must-not-create-admission.json"
        self.source = self.folder / "tiny-synthetic-non-RACE-file"
        self.source.write_bytes(b"PUBLIC-SAFE-NON-RACE-SYNTHETIC-FIXTURE")

    def no_sensitive_file_access(self, action, *, pin):
        with patch.object(review, "APPROVED_REVIEWER_ROSTER_SHA256", pin), \
             patch.object(review, "SOURCE", self.source), \
             patch.object(review, "filehash", side_effect=AssertionError(
                 "unapproved admission must not hash ANY file")), \
             patch.object(Path, "is_file", side_effect=AssertionError(
                 "unapproved admission must not probe source or private file")), \
             patch.object(Path, "read_text", side_effect=AssertionError(
                 "unapproved admission must not open any review JSON")):
            with self.assertRaisesRegex(ValueError, "^HOLD EXP053: owner has not approved"):
                action()

    def test_empty_pin_refuses_before_any_sensitive_read(self):
        self.assertEqual(review.APPROVED_REVIEWER_ROSTER_SHA256, "")
        self.no_sensitive_file_access(
            lambda: review.verified_admission(self.private, self.roster), pin="")

    def test_malformed_pins_all_refused_before_source_access(self):
        for pin in (None, 123, b"a" * 64, {}, [], "b" * 63,
                    "B" * 64, "g" * 64, "a" * 65, "a" * 63 + "\\n"):
            with self.subTest(pin_type=type(pin).__name__, value=str(pin)[:24]):
                self.no_sensitive_file_access(
                    lambda: review.verified_admission(self.private, self.roster), pin=pin)

    def test_gate_does_not_create_output_without_owner_approval(self):
        self.no_sensitive_file_access(
            lambda: review.gate(self.private, self.roster, self.result), pin="")
        self.assertFalse(self.result.exists())

    def test_valid_pin_shape_without_roster_never_reaches_source_hash(self):
        real_hash = review.filehash
        seen = []

        def protected_hash(path):
            seen.append(path)
            if Path(path) == review.MANIFEST:
                return real_hash(path)  # Immutable public, source-free manifest.
            raise AssertionError("roster is absent: no source hashing allowed")

        with patch.object(review, "APPROVED_REVIEWER_ROSTER_SHA256", "a" * 64), \
             patch.object(review, "SOURCE", self.source), \
             patch.object(review, "filehash", side_effect=protected_hash):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
                review.verified_admission(self.private, self.roster)
        self.assertEqual(seen, [])  # Public manifest uses anchored bytes, not filehash.
        self.assertFalse(self.result.exists())

    def test_live_unapproved_pin_remains_empty_without_opening_live_corpus(self):
        self.assertEqual(review.APPROVED_REVIEWER_ROSTER_SHA256, "")
        self.no_sensitive_file_access(
            lambda: review.verified_admission(self.private, self.roster),
            pin=review.APPROVED_REVIEWER_ROSTER_SHA256)


if __name__ == "__main__":
    unittest.main()
