"""Synthetic-only fail-closed key-file reads; NOT independent human enrollment."""

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
except ImportError:  # Optional on hosted CI: no private key workflow is required.
    serialization = None
    Ed25519PrivateKey = None
    signer = None


@unittest.skipUnless(Ed25519PrivateKey is not None, "cryptography not installed")
class PrivateKeyReaderSynthetic(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="exp053-synthetic-private-key-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.key = self.root / "synthetic-only-private-key.pem"
        self.key_obj = Ed25519PrivateKey.generate()
        self.key.write_bytes(self.key_obj.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
        self.key.chmod(0o600)

    def test_valid_ephemeral_ed25519_pem_parses_public_key(self):
        expected = signer.public_b64(self.key_obj)
        self.assertEqual(signer.public_b64(signer.get_key(self.key)), expected)

    def test_symlink_key_refused_even_if_target_mode_is_private(self):
        alias = self.root / "synthetic-alias.pem"
        alias.symlink_to(self.key)
        with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
            signer.get_key(alias)

    def test_hardlink_refused_even_when_private(self):
        second = self.root / "synthetic-hardlink.pem"
        os.link(self.key, second)
        with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
            signer.get_key(self.key)

    def test_world_readable_key_refused(self):
        self.key.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
            signer.get_key(self.key)

    def test_missing_nonfollow_or_nonblock_flag_is_hold(self):
        for name in ("O_NOFOLLOW", "O_NONBLOCK"):
            with self.subTest(name=name), patch.object(signer.os, name, 0):
                with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
                    signer.get_key(self.key)

    def test_oversized_and_empty_file_rejected_without_parsing(self):
        for blob in (b"", b"x" * (signer.MAX_KEY_BYTES + 1)):
            with self.subTest(size=len(blob)):
                self.key.write_bytes(blob)
                with patch.object(signer.serialization, "load_pem_private_key",
                                  side_effect=AssertionError("must not parse invalid size")):
                    with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
                        signer.get_key(self.key)

    def test_nonregular_fifo_rejected_without_blocking(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO unsupported")
        fifo = self.root / "synthetic-fifo"
        os.mkfifo(fifo, mode=0o600)
        with patch.object(signer.serialization, "load_pem_private_key",
                          side_effect=AssertionError("must not parse FIFO")):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
                signer.get_key(fifo)

    def test_non_ed25519_pem_rejected(self):
        from cryptography.hazmat.primitives.asymmetric import ec
        wrong = ec.generate_private_key(ec.SECP256R1())
        self.key.write_bytes(wrong.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
        with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
            signer.get_key(self.key)

    def test_invalid_pem_rejected_without_path_or_key_in_error(self):
        self.key.write_bytes(b"SYNTHETIC_NON_KEY_MARKER_NEVER_PRINTED")
        with self.assertRaisesRegex(ValueError, "HOLD EXP053") as raised:
            signer.get_key(self.key)
        self.assertNotIn(str(self.key), str(raised.exception))
        self.assertNotIn("SYNTHETIC_NON_KEY_MARKER", str(raised.exception))

    def test_key_path_swapped_after_open_still_reads_original_descriptor(self):
        original = self.key
        moved = self.root / "original-file-moved.pem"
        decoy = self.root / "decoy-synthetic.pem"
        decoy_key = Ed25519PrivateKey.generate()
        decoy.write_bytes(decoy_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
        decoy.chmod(0o600)
        real_read = os.read
        swapped = False

        def rename_and_replace(fd, n):
            nonlocal swapped
            if not swapped and stat.S_ISREG(os.fstat(fd).st_mode):
                swapped = True
                original.rename(moved)
                decoy.rename(original)
            return real_read(fd, n)

        # Renaming the still-open inode can change ctime, so a HOLD is the
        # strongest safe result: never silently sign with the replacement.
        with patch.object(signer.os, "read", side_effect=rename_and_replace):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
                signer.get_key(original)
        self.assertTrue(swapped)
        self.assertEqual(signer.public_b64(signer.get_key(moved)),
                         signer.public_b64(self.key_obj))
        self.assertEqual(signer.public_b64(signer.get_key(original)),
                         signer.public_b64(decoy_key))

    def test_mutating_contents_during_read_is_hold(self):
        real_read = os.read
        mutation = False

        def mutate_before_read(fd, n):
            nonlocal mutation
            result = real_read(fd, n)
            if not mutation:
                mutation = True
                with self.key.open("ab") as stream:
                    stream.write(b"\nSYNTHETIC_MUTATION")
            return result

        with patch.object(signer.os, "read", side_effect=mutate_before_read):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
                signer.get_key(self.key)
        self.assertTrue(mutation)

    def test_os_read_error_closes_fd_and_is_sanitized(self):
        fds = []

        def failed_read(fd, n):
            fds.append(fd)
            raise OSError("synthetic low-level read failed")

        with patch.object(signer.os, "read", side_effect=failed_read):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
                signer.get_key(self.key)
        self.assertEqual(len(fds), 1)
        with self.assertRaises(OSError):
            os.fstat(fds[0])


if __name__ == "__main__":
    unittest.main()
