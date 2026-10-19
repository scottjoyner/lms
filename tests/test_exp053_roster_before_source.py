"""Source-free ordering test: exact roster custody before licensed source reads."""

import base64
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import exp053_race_review as review


class ReviewRosterBeforeLicensedSource(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="exp053-synthetic-roster-order-")
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.source = self.home / "SYNTHETIC-not-RACE.bin"
        self.source.write_bytes(b"Only a tiny public-safe synthetic source hash fixture.")
        self.keys = self.home / "synthetic-owner-roster.json"
        self.private_dir = self.home / "absent-no-private-packet"
        self.public_manifest = json.loads(review.MANIFEST.read_text())
        self.assertEqual(review.filehash(review.MANIFEST), review.MANIFEST_SHA)
        self.packet_pin = self.public_manifest["packet_sha256"]
        self.roster = {
            "schema": "exp053-owner-reviewed-roster-v1",
            "packet_sha256": self.packet_pin,
            "owner_custody_reference_sha256": "f" * 64,
            "blind_assignments_confirmed": True,
            "slots": {
                "A": {"reviewer_id": "synthetic_a", "role": "independent",
                      "public_key_b64": base64.b64encode(bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")).decode()},
                "B": {"reviewer_id": "synthetic_b", "role": "independent",
                      "public_key_b64": base64.b64encode(bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")).decode()},
            },
        }

    def write_roster(self, *, patch_fields=None, chmod=0o600):
        payload = {**self.roster, **(patch_fields or {})}
        self.keys.write_bytes((json.dumps(payload, sort_keys=True) + "\n").encode())
        os.chmod(self.keys, chmod)
        return hashlib.sha256(self.keys.read_bytes()).hexdigest()

    def run_source_trap(self, pin):
        real_hash = review.filehash
        calls = []
        def hash_callback(path):
            calls.append(Path(path))
            if Path(path) == review.MANIFEST:
                return real_hash(path)
            raise AssertionError("SOURCE/protocol/private access before roster approval")
        with patch.object(review, "APPROVED_REVIEWER_ROSTER_SHA256", pin), \
             patch.object(review, "SOURCE", self.source), \
             patch.object(review, "filehash", side_effect=hash_callback):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
                review.verified_admission(self.private_dir, self.keys)
        self.assertEqual(calls, [review.MANIFEST])
        self.assertFalse(self.private_dir.exists())

    def test_owner_pin_shape_without_private_roster_does_not_read_source(self):
        self.run_source_trap("a" * 64)

    def test_wrong_owner_pin_even_for_valid_synthetic_roster_does_not_read_source(self):
        self.write_roster()
        self.run_source_trap("a" * 64)

    def test_world_readable_synthetic_roster_denied_before_source(self):
        pin = self.write_roster(chmod=0o644)
        self.run_source_trap(pin)

    def test_wrong_packet_binding_denied_before_source(self):
        pin = self.write_roster(patch_fields={"packet_sha256": "f" * 64})
        self.run_source_trap(pin)

    def test_invalid_blinding_denied_before_source(self):
        pin = self.write_roster(patch_fields={"blind_assignments_confirmed": False})
        self.run_source_trap(pin)

    def test_complete_synthetic_roster_does_not_skip_source_hash(self):
        pin = self.write_roster()
        real_hash = review.filehash
        observed = []
        def mismatch(path):
            observed.append(Path(path))
            if Path(path) == review.MANIFEST:
                return real_hash(path)
            if Path(path) == self.source:
                return "0" * 64
            raise AssertionError("unexpected read or hashing beyond source gate")
        with patch.object(review, "APPROVED_REVIEWER_ROSTER_SHA256", pin), \
             patch.object(review, "SOURCE", self.source), \
             patch.object(review, "filehash", side_effect=mismatch):
            with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source authority changed"):
                review.verified_admission(self.private_dir, self.keys)
        self.assertEqual(observed, [review.MANIFEST, self.source])
        self.assertFalse(self.private_dir.exists())

    def test_pinned_public_manifest_failure_denied_before_any_roster_or_source(self):
        real_hash = review.filehash
        observed = []
        def damaged_manifest(path):
            observed.append(Path(path))
            if Path(path) == review.MANIFEST:
                return "0" * 64
            return real_hash(path)
        with patch.object(review, "APPROVED_REVIEWER_ROSTER_SHA256", "a" * 64), \
             patch.object(review, "SOURCE", self.source), \
             patch.object(review, "filehash", side_effect=damaged_manifest):
            with self.assertRaisesRegex(ValueError, "^HOLD EXP053: immutable public manifest"):
                review.verified_admission(self.private_dir, self.keys)
        self.assertEqual(observed, [review.MANIFEST])


if __name__ == "__main__":
    unittest.main()
