"""Synthetic-only source SHA/Parquet decoder identity binding regression."""
import hashlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import exp053_race_review as exp


class SyntheticPinnedParquetSource(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="exp053-synthetic-source-fd-")
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.original = self.home / "ORIGINAL-INVENTED-PARQUET.bin"
        self.second = self.home / "OTHER-INVENTED-PARQUET.bin"
        self.pointer = self.home / "synthetic-hf-snapshot.parquet"
        self.protocol = self.home / "SYNTHETIC-PROTOCOL.txt"
        self.original.write_bytes(b"SYNTHETIC_SOURCE_A_PUBLIC_TEST_BYTES")
        self.second.write_bytes(b"SYNTHETIC_SOURCE_B_PUBLIC_TEST_BYTES")
        self.pointer.symlink_to(self.original)
        self.protocol.write_bytes(b"SOURCE-FREE-TEST-PROTOCOL")
        self.seen = []

    def fake_parquet(self, read_table=None):
        pq = types.ModuleType("pyarrow.parquet")
        def checked_reader(fileobj, *, use_threads=None, **kwargs):
            self.assertIs(use_threads, False,
                          "source descriptor must not outlive threaded PyArrow readers")
            self.assertFalse(kwargs, "unexpected extra source read options")
            return (self.read_table if read_table is None else read_table)(fileobj)
        pq.read_table = checked_reader
        pyarrow = types.ModuleType("pyarrow")
        pyarrow.__path__ = []
        pyarrow.parquet = pq
        return {"pyarrow": pyarrow, "pyarrow.parquet": pq}

    def read_table(self, fileobj):
        self.assertTrue(hasattr(fileobj, "fileno"),
                        "parquet must be decoded through the hashed file fd")
        self.assertTrue(hasattr(fileobj, "read"))
        self.assertEqual(fileobj.tell(), 0)
        self.seen.append(fileobj.read())
        return types.SimpleNamespace(to_pylist=lambda: [
            {"synthetic_not_a_RACE_article": True} for _ in range(2500)])

    def corpus(self, *, source_sha=None, read_table=None):
        source_hash = source_sha if source_sha is not None else hashlib.sha256(
            self.original.read_bytes()).hexdigest()
        with patch.object(exp, "SOURCE", self.pointer), \
             patch.object(exp, "SOURCE_SHA", source_hash), \
             patch.object(exp, "PROTOCOL", self.protocol), \
             patch.object(exp, "PROTOCOL_SHA", exp.filehash(self.protocol)), \
             patch.dict(sys.modules, self.fake_parquet(read_table)):
            return exp.corpus()

    def test_stable_snapshot_symlink_decodes_exact_pinned_target_fd(self):
        rows = self.corpus()
        self.assertEqual(len(rows), 2500)
        self.assertEqual(self.seen, [self.original.read_bytes()])
        self.assertTrue(self.pointer.is_symlink())

    def test_wrong_source_digest_denied_before_parquet(self):
        with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source bytes"):
            self.corpus(source_sha="0" * 64)
        self.assertFalse(self.seen)

    def test_noncanonical_source_digest_denied(self):
        with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source authority digest"):
            self.corpus(source_sha="Z" * 64)
        self.assertFalse(self.seen)

    def test_snapshot_symlink_changed_during_parquet_read_denied(self):
        def swap_pointer(fileobj):
            table = self.read_table(fileobj)
            self.pointer.unlink()
            self.pointer.symlink_to(self.second)
            return table
        with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source snapshot symlink"):
            self.corpus(read_table=swap_pointer)
        self.assertEqual(self.seen, [self.original.read_bytes()])
        self.assertEqual(self.pointer.resolve(strict=True), self.second)

    def test_inode_swapped_after_hash_but_before_parquet_decoding_denied(self):
        # Even a same-byte replacement at the same target pathname must HOLD.
        moved = self.home / "ORIGINAL-MOVED.bin"
        replacement = self.home / "SYNTHETIC-SAME-BYTES-OTHER-INODE.bin"
        replacement.write_bytes(self.original.read_bytes())
        def replace_inode(fileobj):
            table = self.read_table(fileobj)
            self.original.rename(moved)
            replacement.rename(self.original)
            return table
        with self.assertRaisesRegex(ValueError,
                                    "^HOLD EXP053: source (changed during|resolved inode)"):
            self.corpus(read_table=replace_inode)
        self.assertEqual(self.original.read_bytes(), moved.read_bytes())

    def test_source_content_mutated_during_parquet_decode_denied(self):
        def mutate(fileobj):
            table = self.read_table(fileobj)
            self.original.write_bytes(b"SYNTHETIC_MUTATED_SOURCE_B")
            return table
        with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source changed during"):
            self.corpus(read_table=mutate)

    def test_missing_optional_pyarrow_backend_refuses_without_source_decoding(self):
        with patch.object(exp, "SOURCE", self.pointer), \
             patch.object(exp, "SOURCE_SHA", hashlib.sha256(
                 self.original.read_bytes()).hexdigest()), \
             patch.object(exp, "PROTOCOL", self.protocol), \
             patch.object(exp, "PROTOCOL_SHA", exp.filehash(self.protocol)), \
             patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}):
            with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source parquet backend"):
                exp.corpus()

    def test_missing_no_follow_flags_refuses_before_file_open(self):
        for flag in ("O_PATH", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"):
            with self.subTest(flag=flag), patch.object(exp.os, flag, 0), \
                 patch.object(exp.os, "open", side_effect=AssertionError(
                     "no file source opening with insufficient kernel protections")):
                with self.assertRaisesRegex(ValueError, "^HOLD EXP053: secure source"):
                    self.corpus()

    def test_oversized_sparse_source_denied_before_parquet_decoder(self):
        with self.original.open("wb") as f:
            f.truncate(exp.SOURCE_MAX_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source is not a bounded"):
            self.corpus(source_sha="0" * 64)
        self.assertFalse(self.seen)

    def test_nonregular_source_fifo_denied_without_blocking(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO unsupported")
        self.pointer.unlink()
        os.mkfifo(self.pointer, mode=0o600)
        with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source is not a bounded"):
            self.corpus(source_sha="0" * 64)

    def test_protocol_authority_mismatch_refuses_before_source_open(self):
        with patch.object(exp, "PROTOCOL", self.protocol), \
             patch.object(exp, "PROTOCOL_SHA", "0" * 64), \
             patch.object(exp, "SOURCE", self.pointer), \
             patch.object(exp, "SOURCE_SHA", "0" * 64), \
             patch.object(exp, "_open_no_symlink_ancestors", create=True,
                          side_effect=AssertionError("must not open source")):
            with self.assertRaisesRegex(ValueError, "^HOLD EXP053: preregistered protocol"):
                exp.corpus()


if __name__ == "__main__":
    unittest.main()
