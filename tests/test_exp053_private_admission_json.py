"""Synthetic-only private packet, sealed key, and signed-review read boundary."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import exp053_race_review as admission


class PrivateAdmissionJsonSynthetic(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="exp053-synthetic-private-read-")
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.private = self.home / "owner-private"
        self.private.mkdir(mode=0o700)
        self.input = self.private / "SYNTHETIC-review.json"
        self.payload = {"schema": "SYNTHETIC-ONLY", "public-safe": {"x": 2},
                        "packet_sha256": "invented-no-human-review"}
        self.write()

    def write(self, data=None):
        payload = self.payload if data is None else data
        self.input.write_bytes(json.dumps(payload, sort_keys=True).encode())
        self.input.chmod(0o600)

    def read(self):
        return admission._read_private_admission_json(self.input)

    def test_owner_private_synthetic_json_parses(self):
        self.assertEqual(self.read(), self.payload)

    def test_all_private_gate_reads_are_routed_through_bound_reader(self):
        import inspect
        src = inspect.getsource(admission.verified_admission)
        for fragment in ('private_dir/"packet.json"',
                         'private_dir/"sealed_exam_key.json"',
                         'private_dir/("reviewer_"+slot+".json")',
                         '_read_private_admission_json(adjudicator_file)'):
            self.assertIn(fragment, src)
        for forbidden in ('packet.json").read_text()',
                          'sealed_exam_key.json").read_text()',
                          'adjudicator_file.read_text()'):
            self.assertNotIn(forbidden, src)

    def test_all_nested_duplicate_fields_fail_closed(self):
        for raw in (b'{"packet_sha256":"X","packet_sha256":"Y"}',
                    b'{"nested":{"answer_key":{},"answer_key":{}}}'):
            with self.subTest(raw=raw[:50]):
                self.input.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, '^HOLD EXP053:') as held:
                    self.read()
                self.assertNotIn('answer_key', str(held.exception))
                self.assertNotIn('packet_sha256', str(held.exception))

    def test_invalid_json_nonfinite_and_nonobject_denied(self):
        samples = (b'null', b'[]', b'{"x": NaN}', b'{"x": Infinity}',
                   b'{"x": 1e999}', b'{} trailing', b'\xff', b'')
        for raw in samples:
            with self.subTest(raw=raw[:30]):
                self.input.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
                    self.read()

    def test_static_final_and_ancestor_symlinks_are_denied(self):
        alias = self.private / "SYMLINK.json"
        alias.symlink_to(self.input)
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
            admission._read_private_admission_json(alias)
        directory_alias = self.home / "alias-to-owner-private"
        directory_alias.symlink_to(self.private, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
            admission._read_private_admission_json(directory_alias / self.input.name)

    def test_oversized_files_and_nonregular_fifo_denied(self):
        self.input.write_bytes(b'0' * (admission.PRIVATE_JSON_LIMIT + 1))
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
            self.read()
        self.input.unlink()
        if hasattr(os, 'mkfifo'):
            os.mkfifo(self.input, mode=0o600)
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
                self.read()

    def test_hardlink_and_world_readable_denied(self):
        alias = self.private / "duplicate-link.json"
        os.link(self.input, alias)
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
            self.read()
        alias.unlink()
        self.input.chmod(0o644)
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
            self.read()

    def test_missing_required_no_follow_flags_denied_before_path_access(self):
        for flag in ('O_PATH', 'O_DIRECTORY', 'O_NOFOLLOW', 'O_NONBLOCK'):
            with self.subTest(flag=flag), patch.object(admission.os, flag, 0), \
                    patch.object(admission.os, 'open', side_effect=AssertionError(
                        'no sensitive file opens with an unavailable OS flag')):
                with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
                    self.read()

    def test_wrong_owner_fstat_denied_before_read(self):
        original_fstat = os.fstat
        observed = []
        def wrong_owner(fd):
            info = original_fstat(fd)
            observed.append(fd)
            fields = ('st_mode', 'st_uid', 'st_nlink', 'st_size', 'st_dev',
                      'st_ino', 'st_mtime_ns', 'st_ctime_ns')
            fake = {name: getattr(info, name) for name in fields}
            fake['st_uid'] = os.geteuid() + 10000
            return SimpleNamespace(**fake)
        with patch.object(admission.os, 'fstat', side_effect=wrong_owner), \
             patch.object(admission.os, 'read', side_effect=AssertionError(
                 'private content cannot be read for wrong uid')):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
                self.read()
        self.assertTrue(observed)

    def test_read_failure_closes_file_descriptor(self):
        read_fds = []
        def fail(fd, length):
            read_fds.append(fd)
            raise OSError('synthetic interruption')
        with patch.object(admission.os, 'read', side_effect=fail):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
                self.read()
        self.assertTrue(read_fds)
        for fd in read_fds:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_leaf_replaced_with_identical_bytes_during_read_refused(self):
        moved = self.private / 'moved-synthetic-old.json'
        replacement = self.private / 'synthetic-replacement.json'
        replacement.write_bytes(self.input.read_bytes())
        replacement.chmod(0o600)
        original_read = os.read
        once = False
        def swap(fd, count):
            nonlocal once
            data = original_read(fd, count)
            if not once:
                once = True
                self.input.rename(moved)
                replacement.rename(self.input)
            return data
        with patch.object(admission.os, 'read', side_effect=swap):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
                self.read()
        self.assertTrue(once)
        self.assertEqual(self.input.read_bytes(), moved.read_bytes())

    def test_parent_swapped_into_repository_during_read_refused(self):
        public_repo = self.home / 'synthetic-public-repo'
        public_repo.mkdir(mode=0o700)
        (public_repo / self.input.name).write_bytes(self.input.read_bytes())
        (public_repo / self.input.name).chmod(0o600)
        moved = self.home / 'original-owner-dir'
        original_read = os.read
        once = False
        def swap(fd, count):
            nonlocal once
            data = original_read(fd, count)
            if not once:
                once = True
                self.private.rename(moved)
                self.private.symlink_to(public_repo, target_is_directory=True)
            return data
        with patch.object(admission.os, 'read', side_effect=swap):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
                self.read()
        self.assertTrue(once)

    def test_inside_public_repo_read_denied(self):
        public_repo = self.home / 'synthetic-public-repo'
        source_dir = public_repo / 'research'
        source_dir.mkdir(parents=True)
        self.input = public_repo / 'fake-private-review.json'
        self.write()
        with patch.object(admission, '__file__', str(source_dir / 'exp053_race_review.py')):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053:'):
                self.read()

    def test_empty_owner_roster_still_holds_before_any_sensitive_source_read(self):
        with patch.object(admission, 'APPROVED_REVIEWER_ROSTER_SHA256', ''), \
             patch.object(admission, 'filehash', side_effect=AssertionError(
                 'never inspect source with absent human authorization')):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053: owner has not approved'):
                admission.verified_admission(self.private, self.private / 'missing-roster')


if __name__ == '__main__':
    unittest.main()
