"""Invented metadata receipts only; no real weights, RACE or private reviews."""

import copy
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
sys.path.insert(0, str(ROOT / 'tests'))
import exp053_verified_analysis as replay
from test_exp053_weight_receipt_adversarial import fake_receipt, fake_report

MISSING = object()


class SyntheticPinnedReplayReceipts(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(prefix='exp053-synthetic-scoring-receipt-')
        self.addCleanup(t.cleanup)
        self.home = Path(t.name)
        self.path = self.home / 'invented-source-free-receipt.json'
        self.scan, self.models = fake_receipt()
        self.public = fake_report(self.models)
        self.write(self.public)

    def write(self, record):
        raw = json.dumps(record, sort_keys=True).encode()
        self.path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def load(self, pin=MISSING):
        return replay._read_pinned_source_free_receipt(
            self.path, hashlib.sha256(self.path.read_bytes()).hexdigest()
            if pin is MISSING else pin)

    def test_positive_invented_report_is_checked_as_exact_bytes(self):
        pin = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(self.load(pin), self.public)
        with patch.object(replay, 'WEIGHT_REPORT', self.path), \
             patch.object(replay, 'WEIGHT_REPORT_SHA', pin):
            result = replay.admitted_weight_inventory(self.scan)
        self.assertEqual(set(result), set(self.models))

    def test_real_operator_weight_approval_remains_unset(self):
        self.assertEqual(replay.WEIGHT_REPORT_SHA, '')
        with patch.object(replay, 'WEIGHT_REPORT', self.path):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
                replay.admitted_weight_inventory(self.scan)

    def test_synthetic_weight_report_swap_cannot_change_attested_models(self):
        pin = hashlib.sha256(self.path.read_bytes()).hexdigest()
        changed = copy.deepcopy(self.models)
        target = sorted(changed)[0]
        changed[target]['shards'][0]['sha256'] = 'f' * 64
        alternative = fake_report(changed)
        self.assertNotEqual(alternative['models'][target]['checkpoint_receipt_sha256'],
                            self.public['models'][target]['checkpoint_receipt_sha256'])
        replacement = self.home / 'invented-replacement.json'
        replacement.write_bytes(json.dumps(alternative,sort_keys=True).encode())
        original = self.home / 'invented-pinned-original.json'
        original_read = os.read
        once = False
        def swap_after_read(fd, amount):
            nonlocal once
            chunk = original_read(fd, amount)
            if not once:
                once = True
                self.path.rename(original)
                replacement.rename(self.path)
            return chunk
        with patch.object(replay, 'WEIGHT_REPORT', self.path), \
             patch.object(replay, 'WEIGHT_REPORT_SHA', pin), \
             patch.object(replay.os, 'read', side_effect=swap_after_read):
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
                replay.admitted_weight_inventory(self.scan)
        self.assertTrue(once)
        self.assertNotEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), pin)

    def test_raw_json_duplicate_keys_and_numeric_ambiguity_denied(self):
        variants = (b'{"x":1,"x":2}', b'{"nested":{"a":1,"a":2}}',
                    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}',
                    b'[]', b'null', b'\xff', b'{}trailing', b'')
        for raw in variants:
            with self.subTest(raw=raw[:40]):
                self.path.write_bytes(raw)
                pin = hashlib.sha256(raw).hexdigest()
                with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
                    self.load(pin)

    def test_pinned_public_tokenizer_metadata_valid(self):
        got = replay._read_pinned_source_free_receipt(replay.PREFLIGHT,
                                                      replay.PREFLIGHT_SHA)
        self.assertEqual(got['model_forward_passes'], 0)
        self.assertEqual(set(got['models']), set(self.scan['models']))

    def test_wrong_pin_refuses_before_parsing(self):
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
            self.load('0' * 64)
        for bad in (None, '', 'A' * 64, '0' * 63, ['0']):
            with self.subTest(bad=repr(bad)):
                with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
                    self.load(bad)

    def test_final_and_intermediate_symlinks_refuse(self):
        target = self.home / 'invented-symlink.json'
        target.symlink_to(self.path)
        with self.assertRaisesRegex((OSError, ValueError), '.'):
            replay._read_pinned_source_free_receipt(target,
                hashlib.sha256(self.path.read_bytes()).hexdigest())
        alias = self.home / 'invented-symlink-dir'
        folder = self.home / 'invented-real-dir'
        folder.mkdir()
        direct = folder / 'public-receipt.json'
        direct.write_bytes(self.path.read_bytes())
        alias.symlink_to(folder, target_is_directory=True)
        with self.assertRaisesRegex((OSError, ValueError), '.'):
            replay._read_pinned_source_free_receipt(alias/direct.name,
                hashlib.sha256(direct.read_bytes()).hexdigest())

    def test_hardlink_and_oversize_and_fifo_refuse(self):
        alias = self.home / 'additional-hardlink'
        os.link(self.path, alias)
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
            self.load()
        alias.unlink()
        with self.path.open('wb') as fd:
            fd.truncate(replay.PUBLIC_RECEIPT_LIMIT+1)
        with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
            self.load('0'*64)
        self.path.unlink()
        if hasattr(os, 'mkfifo'):
            os.mkfifo(self.path, mode=0o600)
            with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
                self.load('0'*64)

    def test_missing_secure_kernel_flags_refuse_before_file_open(self):
        for flag in ('O_PATH','O_DIRECTORY','O_NOFOLLOW','O_NONBLOCK'):
            with self.subTest(flag=flag), patch.object(replay.os,flag,0), \
                 patch.object(replay.os,'open',side_effect=AssertionError(
                     'unsafe OS flags must fail before file access')):
                with self.assertRaisesRegex(ValueError, '^HOLD EXP053'):
                    self.load('a'*64)

    def test_unreadable_private_packet_uses_existing_guarded_reader(self):
        # Replay must not open or parse a private source with path.read_text.
        import inspect
        source = inspect.getsource(replay.require_sources)
        self.assertIn('_read_private_admission_json(source_packet)', source)
        self.assertIn('_read_pinned_public_manifest()', source)
        self.assertNotIn('source_packet.read_text()', source)
        self.assertNotIn('MANIFEST.read_text()', source)
        self.assertNotIn('PREFLIGHT.read_text()', source)


if __name__ == '__main__':
    unittest.main()
