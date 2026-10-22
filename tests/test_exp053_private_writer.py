"""Synthetic private-output creation tests; no RACE or actual reviewer files."""

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
import exp053_race_review as review


class PrivateReviewWriterSynthetic(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="exp053-private-output-synthetic-")
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.output = self.folder / "synthetic-private-output.json"
        self.obj = {"synthetic": "NO PRIVATE DATA OR HUMAN APPROVAL"}
        self.bytes = review.canonical(self.obj) + b"\n"

    def test_written_file_is_0600_under_umask_022(self):
        old_umask = os.umask(0o022)
        try:
            review.save_private(self.output, self.obj)
        finally:
            os.umask(old_umask)
        self.assertEqual(stat.S_IMODE(os.lstat(self.output).st_mode), 0o600)
        self.assertEqual(self.output.read_bytes(), self.bytes)

    def test_private_mode_is_in_effect_before_any_write(self):
        real_write = os.write
        observed = []
        def inspect_write(fd, contents):
            observed.append(stat.S_IMODE(os.fstat(fd).st_mode))
            return real_write(fd, contents)
        previous = os.umask(0o022)
        try:
            with patch.object(review.os, "write", side_effect=inspect_write):
                review.save_private(self.output, self.obj)
        finally:
            os.umask(previous)
        self.assertEqual(observed, [0o600])
        self.assertEqual(self.output.read_bytes(), self.bytes)

    def test_fchmod_exception_cleans_private_artifact(self):
        seen_modes = []
        def fail_at_fchmod(fd, mode):
            seen_modes.append(stat.S_IMODE(os.fstat(fd).st_mode))
            raise OSError("synthetic fchmod denied before write")
        previous = os.umask(0o022)
        try:
            with patch.object(review.os, "fchmod", side_effect=fail_at_fchmod):
                with self.assertRaises(OSError):
                    review.save_private(self.output, self.obj)
        finally:
            os.umask(previous)
        self.assertEqual(seen_modes, [0o600])
        self.assertFalse(self.output.exists())

    def test_exclusive_output_preserves_existing_bytes(self):
        self.output.write_bytes(b"AN_EXISTING_PRIVATE_ARTIFACT")
        self.output.chmod(0o600)
        with self.assertRaises(FileExistsError):
            review.save_private(self.output, self.obj)
        self.assertEqual(self.output.read_bytes(), b"AN_EXISTING_PRIVATE_ARTIFACT")

    def test_symlink_to_unrelated_file_never_followed(self):
        decoy = self.folder / "unrelated-existing-file"
        decoy.write_bytes(b"DO NOT TOUCH")
        self.output.symlink_to(decoy)
        with self.assertRaises(FileExistsError):
            review.save_private(self.output, self.obj)
        self.assertEqual(decoy.read_bytes(), b"DO NOT TOUCH")
        self.assertTrue(self.output.is_symlink())

    def test_partial_write_retried_without_truncation(self):
        real_write = os.write
        def write_at_most_four(fd, contents):
            return real_write(fd, contents[:4])
        with patch.object(review.os, "write", side_effect=write_at_most_four):
            review.save_private(self.output, self.obj)
        self.assertEqual(self.output.read_bytes(), self.bytes)

    def test_write_error_and_interrupt_cleanup(self):
        for error in (OSError("synthetic write failure"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                def fail_write(fd, contents):
                    os.fstat(fd)
                    raise error
                with patch.object(review.os, "write", side_effect=fail_write):
                    with self.assertRaises(type(error)):
                        review.save_private(self.output, self.obj)
                self.assertFalse(self.output.exists())

    def test_fsync_failure_leaves_no_named_incomplete_file(self):
        real_fsync = os.fsync
        fired = False
        def fail_data_fsync(fd):
            nonlocal fired
            if not fired and stat.S_ISREG(os.fstat(fd).st_mode):
                fired = True
                raise OSError("synthetic fsync disk error")
            return real_fsync(fd)
        with patch.object(review.os, "fsync", side_effect=fail_data_fsync):
            with self.assertRaises(OSError):
                review.save_private(self.output, self.obj)
        self.assertTrue(fired)
        self.assertFalse(self.output.exists())

    def test_replacement_file_preserved_on_write_error(self):
        real_write = os.write
        moved = self.folder / "moved-original-incomplete.json"
        substitute = b"SYNTHETIC THIRD PARTY SUBSTITUTE"
        swapped = False
        def swap_and_fail(fd, contents):
            nonlocal swapped
            if not swapped:
                swapped = True
                self.output.rename(moved)
                self.output.write_bytes(substitute)
                raise OSError("synthetic failure after destination substitution")
            return real_write(fd, contents)
        with patch.object(review.os, "write", side_effect=swap_and_fail):
            with self.assertRaises(OSError):
                review.save_private(self.output, self.obj)
        self.assertTrue(swapped)
        self.assertEqual(self.output.read_bytes(), substitute)
        self.assertTrue(moved.exists())
        self.assertEqual(stat.S_IMODE(moved.stat().st_mode), 0o600)

    def test_parent_directory_swap_denies_unrelated_destination(self):
        parent = self.folder / "private-folder"
        parent.mkdir(mode=0o700)
        out = parent / "payload.json"
        moved_parent = self.folder / "moved-private-folder"
        decoy = self.folder / "untrusted-but-synthetic"
        decoy.mkdir(mode=0o700)
        touched = False
        real_write = os.write
        def swap_parent(fd, contents):
            nonlocal touched
            if not touched:
                touched = True
                parent.rename(moved_parent)
                parent.symlink_to(decoy, target_is_directory=True)
            return real_write(fd, contents)
        with patch.object(review.os, "write", side_effect=swap_parent):
            with self.assertRaises((OSError, ValueError)):
                review.save_private(out, self.obj)
        self.assertTrue(touched)
        self.assertFalse((decoy / "payload.json").exists())
        self.assertFalse((moved_parent / "payload.json").exists())

    def test_missing_secure_file_open_flags_fail_closed(self):
        for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"):
            with self.subTest(flag=name), patch.object(review.os, name, 0):
                with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
                    review.save_private(self.output, self.obj)
                self.assertFalse(self.output.exists())

    def test_private_admission_receipt_reuses_exclusive_writer(self):
        # This is only a writer check. Do NOT invoke the scientific admission gate.
        fixture = {"schema": "exp053-synthetic-only", "human_approval": False}
        review.save_private(self.output, fixture)
        self.assertEqual(json.loads(self.output.read_text()), fixture)

    def test_signed_review_export_uses_same_private_exclusive_writer(self):
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            import exp053_sign_review as sign
        except ImportError:
            self.skipTest("cryptography unavailable")
        k = Ed25519PrivateKey.generate()
        key_path = self.folder / "synthetic-only-review-key.pem"
        key_path.write_bytes(k.private_bytes(serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        key_path.chmod(0o600)
        packet = {"cases": [{"case_id": "case-synthetic", "article": "No real corpus."}]}
        unsigned = {"schema": "exp053-review-v1",
                    "packet_sha256": review.digest(packet),
                    "role": "independent", "reviewer_id": "synthetic-test-only",
                    "attestation": sign.ATTEST, "signature": "",
                    "decisions": [{"case_id": "case-synthetic", "decision": "reject",
                                   "reject_reason": "Source-only synthetic ambiguity"}]}
        packet_file = self.folder / "synthetic_packet.json"
        draft_file = self.folder / "synthetic_draft.json"
        packet_file.write_text(json.dumps(packet))
        draft_file.write_text(json.dumps(unsigned))
        signed_output = self.folder / "synthetic_signed.json"
        with contextlib.redirect_stdout(io.StringIO()):
            sign.sign(draft_file, packet_file, key_path, signed_output)
        data = json.loads(signed_output.read_text())
        self.assertEqual(stat.S_IMODE(os.stat(signed_output).st_mode), 0o600)
        self.assertEqual(data["reviewer_id"], "synthetic-test-only")
        self.assertEqual(len(data["signature"]), 128)
        self.assertEqual(data["decisions"], unsigned["decisions"])
        with self.assertRaises((ValueError, FileExistsError)):
            with contextlib.redirect_stdout(io.StringIO()):
                sign.sign(draft_file, packet_file, key_path, signed_output)


if __name__ == "__main__":
    unittest.main()
