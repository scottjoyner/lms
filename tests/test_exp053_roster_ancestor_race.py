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
                      "public_key_b64": base64.b64encode(b"A" * 32).decode()},
                "B": {"reviewer_id": "synthetic_reviewer_B", "role": "independent",
                      "public_key_b64": base64.b64encode(b"B" * 32).decode()},
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

    def test_missing_opath_fails_before_any_descriptor(self):
        with patch.object(roster.os, "O_PATH", 0), patch.object(
                roster.os, "open", side_effect=AssertionError("no fd on unsupported OS")):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
                self.read()


if __name__ == "__main__":
    unittest.main()
