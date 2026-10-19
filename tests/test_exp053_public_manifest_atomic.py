"""Immutable public manifest: same-bytes hash/parse, no roster/source access."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
import exp053_race_review as admission
import exp053_reviewer_roster as roster


class SyntheticPinnedPublicManifest(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(prefix='exp053-synthetic-public-manifest-')
        self.addCleanup(t.cleanup)
        self.home = Path(t.name)
        self.path = self.home / 'SYNTHETIC-PUBLIC-MANIFEST.json'
        self.content = {"schema": "synthetic-public-v1", "packet_sha256": "a" * 64,
                        "sealed_answer_key_sha256": "b" * 64,
                        "candidate_articles": 64}
        self.write()

    def write(self, content=None):
        blob = json.dumps(self.content if content is None else content,
                          sort_keys=True).encode()
        self.path.write_bytes(blob)
        self.pin = hashlib.sha256(blob).hexdigest()
        return blob

    def read(self, *, pin=None):
        with patch.object(admission, 'MANIFEST', self.path), patch.object(
                admission, 'MANIFEST_SHA', self.pin if pin is None else pin):
            return admission._read_pinned_public_manifest()

    def test_real_source_free_pinned_manifest_still_matches(self):
        self.assertEqual(admission._read_pinned_public_manifest()['schema'],
                         'exp053-public-manifest-v1')
        self.assertEqual(self.read(), self.content)

    def test_changed_hash_denied_even_when_json_parses(self):
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053: immutable public manifest'):
            self.read(pin='0' * 64)

    def test_missing_no_follow_capabilities_denied(self):
        for name in ('O_PATH', 'O_DIRECTORY', 'O_NOFOLLOW', 'O_NONBLOCK'):
            with self.subTest(name=name), patch.object(admission.os, name, 0), \
                 patch.object(admission.os, 'open', side_effect=AssertionError(
                     'no manifest opening without all required flags')):
                with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
                    self.read()

    def test_noncanonical_or_bad_json_denied_even_when_hash_pinned(self):
        for blob in (b'{"packet_sha256":"a","packet_sha256":"b"}',
                     b'{"nested":{"foo":1,"foo":2}}',
                     b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}',
                     b'[]', b'null', b'\xff', b''):
            with self.subTest(blob=blob[:35]):
                self.path.write_bytes(blob)
                self.pin = hashlib.sha256(blob).hexdigest()
                with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
                    self.read()

    def test_symlink_hardlink_fifo_and_oversize_denied(self):
        alias = self.home / 'alias.json'
        alias.symlink_to(self.path)
        with patch.object(admission, 'MANIFEST', alias), patch.object(
                admission, 'MANIFEST_SHA', self.pin):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
                admission._read_pinned_public_manifest()
        alias.unlink()
        os.link(self.path, alias)
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
            self.read()
        alias.unlink()
        self.path.write_bytes(b'0' * (admission.PUBLIC_MANIFEST_LIMIT + 1))
        self.pin = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
            self.read()
        self.path.unlink()
        if hasattr(os, 'mkfifo'):
            os.mkfifo(self.path, mode=0o600)
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
                self.read()

    def test_final_file_swap_during_read_cannot_bind_substituted_packet(self):
        """The previous hash-then-read accepted a post-hash packet SHA swap."""
        replacement = self.home / 'replacement.json'
        replacement.write_bytes(json.dumps({**self.content,
            'packet_sha256': 'd' * 64}, sort_keys=True).encode())
        archived = self.home / 'original.json'
        actual_read = os.read
        did_swap = False
        def read_then_swap(fd, length):
            nonlocal did_swap
            result = actual_read(fd, length)
            if not did_swap:
                did_swap = True
                self.path.rename(archived)
                replacement.rename(self.path)
            return result
        with patch.object(admission.os, 'read', side_effect=read_then_swap):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053: immutable public manifest'):
                self.read()
        self.assertTrue(did_swap)
        self.assertEqual(json.loads(self.path.read_text())['packet_sha256'], 'd'*64)

    def test_exact_pinned_source_read_does_not_call_filehash(self):
        with patch.object(admission, 'filehash', side_effect=AssertionError(
                'no second filehash permitted in pinned manifest read')):
            self.assertEqual(self.read()['packet_sha256'], 'a'*64)

    def test_approved_pin_missing_does_not_touch_even_public_manifest(self):
        with patch.object(admission, 'APPROVED_REVIEWER_ROSTER_SHA256', ''), \
             patch.object(admission, '_read_pinned_public_manifest',
                          side_effect=AssertionError('not allowed yet')):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053: owner has not'):
                admission.verified_admission(self.home / 'absent', self.home / 'absent')

    def test_wrong_public_manifest_pin_stops_before_roster_or_source(self):
        with patch.object(admission, 'APPROVED_REVIEWER_ROSTER_SHA256', 'a'*64), \
             patch.object(admission, 'MANIFEST', self.path), \
             patch.object(admission, 'MANIFEST_SHA', '0'*64), \
             patch.object(admission, 'filehash', side_effect=AssertionError(
                 'no licensed source read')), \
             patch.object(roster, 'load_roster', side_effect=AssertionError(
                 'no private reviewer roster read')):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053: immutable public manifest'):
                admission.verified_admission(self.home / 'absent', self.home / 'absent')


if __name__ == '__main__':
    unittest.main()
