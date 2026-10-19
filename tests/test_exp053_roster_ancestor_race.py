"""Synthetic roster filesystem provenance: ancestry and final-leaf races."""

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
import exp053_reviewer_roster as roster


class RosterAncestorRaceSynthetic(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="exp053-synthetic-roster-ancestor-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "synthetic-public-repository"
        self.repo.mkdir()
        self.private = self.root / "owner-private"
        self.private.mkdir(mode=0o700)
        self.path = self.private / "synthetic-reviewer-roster.json"
        self.payload = {
            "schema": roster.SCHEMA, "packet_sha256": "c" * 64,
            "owner_custody_reference_sha256": "d" * 64,
            "blind_assignments_confirmed": True,
            "slots": {
                "A": {"reviewer_id": "synthetic_reviewer_A", "role": "independent",
                      "public_key_b64": base64.b64encode(bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")).decode()},
                "B": {"reviewer_id": "synthetic_reviewer_B", "role": "independent",
                      "public_key_b64": base64.b64encode(bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")).decode()},
            },
        }
        self.data = json.dumps(self.payload, sort_keys=True).encode()
        self.path.write_bytes(self.data)
        self.path.chmod(0o600)
        self.pin = hashlib.sha256(self.data).hexdigest()

    def read(self):
        return roster.load_roster(self.path, "c" * 64, self.pin, self.repo)

    def test_normal_owner_private_roster_is_still_valid(self):
        loaded = self.read()
        self.assertEqual(set(loaded["slots"]), {"A", "B"})
        self.assertEqual(loaded["roster_sha256"], self.pin)

    def test_swap_immediate_parent_to_public_repo_before_traversal_refused(self):
        # resolve(strict=True) sees the original outside-repo path. Just
        # before the first descriptor open, an attacker replaces the parent.
        inside = self.repo / self.path.name
        inside.write_bytes(self.data)
        inside.chmod(0o600)
        moved = self.root / "moved-synthetic-owner-private"
        original_open = os.open
        swapped = False

        def swap_parent(filename, *args, **kwargs):
            nonlocal swapped
            if not swapped:
                swapped = True
                self.private.rename(moved)
                self.private.symlink_to(self.repo, target_is_directory=True)
            return original_open(filename, *args, **kwargs)

        with patch.object(roster.os, "open", side_effect=swap_parent):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
                self.read()
        self.assertTrue(swapped)
        self.assertEqual(inside.read_bytes(), self.data)
        self.assertEqual((moved / self.path.name).read_bytes(), self.data)

    def test_swap_distant_ancestor_to_repo_symlink_before_traversal_refused(self):
        parent = self.root / "outer"
        parent.mkdir(mode=0o700)
        nest = parent / "inner"
        nest.mkdir(mode=0o700)
        target = nest / self.path.name
        self.path.rename(target)
        self.path = target
        in_repo = self.repo / "inner"
        in_repo.mkdir()
        (in_repo / target.name).write_bytes(self.data)
        (in_repo / target.name).chmod(0o600)
        moved = self.root / "outer-moved"
        original_open = os.open
        swapped = False

        def swap_ancestor(filename, *args, **kwargs):
            nonlocal swapped
            if not swapped:
                swapped = True
                parent.rename(moved)
                parent.symlink_to(self.repo, target_is_directory=True)
            return original_open(filename, *args, **kwargs)

        with patch.object(roster.os, "open", side_effect=swap_ancestor):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
                self.read()
        self.assertTrue(swapped)
        self.assertTrue((in_repo / target.name).exists())

    def test_ancestor_swapped_during_read_is_refused(self):
        inside = self.repo / self.path.name
        inside.write_bytes(self.data)
        inside.chmod(0o600)
        moved = self.root / "owner-private-moved"
        real_read = os.read
        swapped = False

        def swap_during_read(fd, count):
            nonlocal swapped
            result = real_read(fd, count)
            if not swapped:
                swapped = True
                self.private.rename(moved)
                self.private.symlink_to(self.repo, target_is_directory=True)
            return result

        with patch.object(roster.os, "read", side_effect=swap_during_read):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
                self.read()
        self.assertTrue(swapped)
        self.assertEqual(inside.read_bytes(), self.data)

    def test_leaf_swapped_during_read_to_identical_bytes_is_refused(self):
        moved = self.private / "moved-original-roster.json"
        new = self.private / "replacement-roster.json"
        new.write_bytes(self.data)
        new.chmod(0o600)
        real_read = os.read
        swapped = False

        def swap_during_read(fd, count):
            nonlocal swapped
            result = real_read(fd, count)
            if not swapped:
                swapped = True
                self.path.rename(moved)
                new.rename(self.path)
            return result

        with patch.object(roster.os, "read", side_effect=swap_during_read):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
                self.read()
        self.assertTrue(swapped)
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertEqual(moved.read_bytes(), self.data)

    def test_static_symlinked_ancestor_is_denied(self):
        alias = self.root / "private-alias"
        alias.symlink_to(self.private, target_is_directory=True)
        self.path = alias / self.path.name
        with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
            self.read()

    def test_failed_intermediate_parent_close_does_not_leak_next_dir(self):
        real_open = os.open
        real_close = os.close
        opened = []
        failed = False

        def track_open(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            opened.append(fd)
            return fd

        def fail_first_close(fd):
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("synthetic interrupted ancestor handoff")
            return real_close(fd)

        with patch.object(roster.os, "open", side_effect=track_open), patch.object(
                roster.os, "close", side_effect=fail_first_close):
            with self.assertRaises(OSError):
                roster._open_no_symlink_ancestors(self.path)
        self.assertTrue(failed)
        self.assertGreaterEqual(len(opened), 2)
        for fd in opened:
            with self.subTest(fd=fd), self.assertRaises(OSError):
                os.fstat(fd)

    def test_failed_final_parent_close_does_not_leak_opened_leaf(self):
        real_open = os.open
        real_close = os.close
        opened = []
        failed = False
        leaf_opened = False

        def track_open(*args, **kwargs):
            nonlocal leaf_opened
            fd = real_open(*args, **kwargs)
            opened.append(fd)
            if fd is not None and os.fstat(fd).st_mode & 0o170000 == 0o100000:
                leaf_opened = True
            return fd

        def fail_parent_once(fd):
            nonlocal failed
            if leaf_opened and not failed:
                failed = True
                raise OSError("synthetic failed final parent close")
            return real_close(fd)

        with patch.object(roster.os, "open", side_effect=track_open), patch.object(
                roster.os, "close", side_effect=fail_parent_once):
            with self.assertRaises(OSError):
                roster._open_no_symlink_ancestors(self.path)
        self.assertTrue(leaf_opened)
        self.assertTrue(failed)
        for fd in opened:
            with self.subTest(fd=fd), self.assertRaises(OSError):
                os.fstat(fd)

    def test_keyboard_interrupt_at_intermediate_parent_release_closes_next_dir(self):
        real_open, real_close = os.open, os.close
        opened = []
        failed = False
        def track_open(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            opened.append(fd)
            return fd
        def interrupt(fd):
            nonlocal failed
            if not failed:
                failed = True
                raise KeyboardInterrupt("synthetic research-only interruption")
            return real_close(fd)
        with patch.object(roster.os, "open", side_effect=track_open), patch.object(
                roster.os, "close", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                roster._open_no_symlink_ancestors(self.path)
        self.assertTrue(failed)
        for fd in opened:
            with self.subTest(fd=fd), self.assertRaises(OSError):
                os.fstat(fd)

    def test_missing_opath_fails_before_any_descriptor(self):
        with patch.object(roster.os, "O_PATH", 0), patch.object(
                roster.os, "open", side_effect=AssertionError("no fd on unsupported OS")):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
                self.read()


if __name__ == "__main__":
    unittest.main()
