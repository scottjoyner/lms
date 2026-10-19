"""Synthetic-only offline signer ingestion; never open licensed source files."""

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    import exp053_sign_review as signer
    import exp053_race_review as review
except ImportError:
    serialization = None
    Ed25519PrivateKey = None
    signer = None
    review = None


@unittest.skipUnless(signer is not None, "cryptography unavailable")
class SignerInputBoundary(unittest.TestCase):
    def setUp(self):
        private = tempfile.TemporaryDirectory(prefix="exp053-signer-input-synthetic-")
        self.addCleanup(private.cleanup)
        self.home = Path(private.name)
        self.packet = {"cases": [{"case_id": "synthetic-1", "article":
                                  "An invented non-licensed story for a toy review."}]}
        self.form = {
            "schema": "exp053-review-v1", "reviewer_id": "synthetic_only",
            "packet_sha256": review.digest(self.packet), "role": "independent",
            "attestation": signer.ATTEST, "signature": "",
            "decisions": [{"case_id": "synthetic-1", "decision": "reject",
                           "reject_reason": "Synthetic case intentionally ambiguous"}],
        }
        self.packet_path = self.home / "synthetic_packet.json"
        self.draft_path = self.home / "synthetic_review.json"
        self.output = self.home / "no-real-review-signed.json"
        self.packet_path.write_text(json.dumps(self.packet))
        self.draft_path.write_text(json.dumps(self.form))
        self.key = self.home / "ephemeral_test_key.pem"
        self.key_object = Ed25519PrivateKey.generate()
        self.key.write_bytes(self.key_object.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
        self.key.chmod(0o600)

    def sign_should_hold_before_key(self):
        with patch.object(signer, "get_key", side_effect=AssertionError(
                "invalid synthetic input must never reach a private key")):
            with self.assertRaisesRegex(ValueError, "^HOLD EXP053:") as held:
                signer.sign(self.draft_path, self.packet_path, self.key, self.output)
        self.assertFalse(self.output.exists())
        self.assertNotIn(str(self.home), str(held.exception))
        self.assertNotIn("invented non-licensed", str(held.exception))

    def test_valid_synthetic_draft_signs_and_verifies(self):
        with contextlib.redirect_stdout(io.StringIO()):
            signer.sign(self.draft_path, self.packet_path, self.key, self.output)
        actual = json.loads(self.output.read_text())
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o600)
        self.assertEqual(actual["decisions"], self.form["decisions"])
        unsigned = {k: v for k, v in actual.items() if k != "signature"}
        self.key_object.public_key().verify(bytes.fromhex(actual["signature"]),
                                            review.canonical(unsigned))

    def test_bad_adjudicator_decisions_hold_before_key(self):
        for decisions in (None, {}, "bad", [None], [{"case_id": []}],
                          [{"case_id": {"id": 1}}], [{"case_id": ""}], []):
            with self.subTest(decisions=str(decisions)[:50]):
                self.form["role"] = "adjudicator"
                self.form["decisions"] = decisions
                self.draft_path.write_text(json.dumps(self.form))
                self.sign_should_hold_before_key()

    def test_duplicate_adjudicator_scope_refused_before_key(self):
        self.form["role"] = "adjudicator"
        self.form["decisions"] = [self.form["decisions"][0]] * 2
        self.draft_path.write_text(json.dumps(self.form))
        self.sign_should_hold_before_key()

    def test_wrong_or_missing_fields_and_existing_signature_refused(self):
        for change, value in (("schema", "unknown"), ("reviewer_id", None),
                              ("signature", "f" * 128), ("decisions", {}),
                              ("role", "system"), ("attestation", None)):
            with self.subTest(change=change):
                draft = {**self.form, change: value}
                self.draft_path.write_text(json.dumps(draft))
                self.sign_should_hold_before_key()
        self.draft_path.write_text(json.dumps({**self.form, "extra_private": "not allowed"}))
        self.sign_should_hold_before_key()

    def test_nested_duplicate_json_field_in_draft_refused(self):
        raw = self.draft_path.read_bytes()
        old = b'"decision": "reject",'
        new = b'"decision": "accept", "decision": "reject",'
        self.assertIn(old, raw)
        self.draft_path.write_bytes(raw.replace(old, new, 1))
        self.sign_should_hold_before_key()

    def test_duplicate_json_field_in_source_packet_refused(self):
        raw = self.packet_path.read_bytes()
        old = b'"cases":'
        self.packet_path.write_bytes(raw.replace(old, b'"cases": [], "cases":', 1))
        self.sign_should_hold_before_key()

    def test_synthetic_invalid_number_and_invalid_utf8_refused(self):
        for raw in (b'{"bad":NaN}', b'{"bad":1e999}', b'\xff\xff',
                    b'{} trailing'):
            with self.subTest(raw=repr(raw)):
                self.draft_path.write_bytes(raw)
                self.sign_should_hold_before_key()

    def test_symlinked_packet_or_draft_rejected(self):
        for attr in ("packet_path", "draft_path"):
            with self.subTest(attr=attr):
                path = getattr(self, attr)
                original = self.home / ("unlinked_" + attr + ".json")
                path.rename(original)
                path.symlink_to(original)
                self.sign_should_hold_before_key()
                path.unlink()
                original.rename(path)

    def test_os_read_failure_closes_open_fd(self):
        collected = []
        def fail_read(fd, n):
            collected.append(fd)
            raise OSError("synthetic OS read failure")
        with patch.object(signer.os, "read", side_effect=fail_read):
            self.sign_should_hold_before_key()
        self.assertTrue(collected)
        with self.assertRaises(OSError):
            os.fstat(collected[0])

    def test_zero_oversized_and_fifo_refused_before_key(self):
        for raw in (b"", b"x" * (signer.MAX_INPUT_BYTES + 1)):
            with self.subTest(n=len(raw)):
                self.draft_path.write_bytes(raw)
                self.sign_should_hold_before_key()
        if hasattr(os, "mkfifo"):
            self.draft_path.unlink()
            os.mkfifo(self.draft_path, mode=0o600)
            self.sign_should_hold_before_key()

    def test_missing_safety_flags_prevent_key_access(self):
        for name in ("O_NOFOLLOW", "O_NONBLOCK"):
            with self.subTest(name=name), patch.object(signer.os, name, 0):
                self.sign_should_hold_before_key()

    def test_packet_path_swap_during_descriptor_read_is_hold(self):
        old_read = os.read
        replaced = False
        moved = self.home / "moved-synthetic_packet.json"
        replacement = self.home / "untrusted-synthetic-packet.json"
        replacement.write_text(json.dumps({"cases": []}))

        def swap_file(fd, length):
            nonlocal replaced
            if not replaced:
                replaced = True
                self.packet_path.rename(moved)
                replacement.rename(self.packet_path)
            return old_read(fd, length)

        # Exercise the source-packet reader itself: replacement after os.open
        # changes the original inode's ctime and must be refused. No signer
        # key is in scope for this source-only input race test.
        with patch.object(signer.os, "read", side_effect=swap_file):
            with self.assertRaisesRegex(ValueError, "^HOLD EXP053:"):
                signer.read_offline_json(self.packet_path)
        self.assertTrue(replaced)
        self.assertFalse(self.output.exists())
        self.assertEqual(json.loads(self.packet_path.read_text()), {"cases": []})

    def test_valid_synthetic_adjudicator_draft_can_sign(self):
        self.form["role"] = "adjudicator"
        self.draft_path.write_text(json.dumps(self.form))
        with contextlib.redirect_stdout(io.StringIO()):
            signer.sign(self.draft_path, self.packet_path, self.key, self.output)
        signed = json.loads(self.output.read_text())
        self.assertEqual(signed["role"], "adjudicator")
        self.assertEqual(len(signed["signature"]), 128)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
