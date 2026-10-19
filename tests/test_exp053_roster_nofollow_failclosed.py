"""Synthetic owner-pinned enrollment parser must never downgrade no-follow I/O."""

import base64
import hashlib
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
import exp053_reviewer_roster as roster


class RosterFileBoundarySynthetic(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="exp053-nofollow-roster-synthetic-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / "synthetic-only-owner-roster.json"
        self.packet = "c" * 64
        self.content = {
            "schema": roster.SCHEMA,
            "packet_sha256": self.packet,
            "owner_custody_reference_sha256": "d" * 64,
            "blind_assignments_confirmed": True,
            "slots": {
                "A": {"reviewer_id": "synthetic_A", "role": "independent",
                      "public_key_b64": base64.b64encode(bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")).decode()},
                "B": {"reviewer_id": "synthetic_B", "role": "independent",
                      "public_key_b64": base64.b64encode(bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")).decode()},
            },
        }
        self.data = (json.dumps(self.content, sort_keys=True) + "\n").encode()
        self.path.write_bytes(self.data)
        self.path.chmod(0o600)
        self.pin = hashlib.sha256(self.data).hexdigest()

    def read(self, path=None, pin=None):
        return roster.load_roster(path or self.path, self.packet,
                                  self.pin if pin is None else pin, ROOT)

    def test_owned_private_single_link_roster_still_loads(self):
        approved = self.read()
        self.assertEqual(set(approved["slots"]), {"A", "B"})
        self.assertEqual(approved["roster_sha256"], self.pin)
        self.assertEqual(approved["owner_custody_reference_sha256"], "d" * 64)

    def test_no_follow_and_nonblocking_missing_flags_refuse_before_file_open(self):
        for flag in ("O_NOFOLLOW", "O_NONBLOCK"):
            with self.subTest(flag=flag):
                with patch.object(roster.os, flag, 0), patch.object(
                    roster.os, "open", side_effect=AssertionError(
                        "no protected roster file may be opened")):
                    with self.assertRaisesRegex(ValueError,
                                                "HOLD EXP053 ROSTER: secure nonfollowing"):
                        self.read()

    def test_hardlink_to_same_roster_is_denied_even_with_owner_pin(self):
        alias = self.root / "synthetic-owner-hardlink.json"
        os.link(self.path, alias)
        self.assertEqual(self.path.stat().st_nlink, 2)
        with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
            self.read()
        self.assertEqual(alias.read_bytes(), self.data)

    def test_regular_final_component_symlink_denied(self):
        alias = self.root / "synthetic-owner-symlink.json"
        alias.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
            self.read(path=alias)

    def test_mismatched_owner_fstat_refuses_before_read(self):
        actual_fstat = os.fstat
        observed = []

        def wrong_owner(fd):
            info = actual_fstat(fd)
            observed.append(fd)
            names = ("st_mode", "st_uid", "st_nlink", "st_size", "st_dev",
                     "st_ino", "st_mtime_ns", "st_ctime_ns")
            props = {name: getattr(info, name) for name in names}
            props["st_uid"] = os.geteuid() + 10000
            return SimpleNamespace(**props)

        with patch.object(roster.os, "fstat", side_effect=wrong_owner), patch.object(
                roster.os, "fdopen", side_effect=AssertionError(
                    "wrong-owner file cannot be opened for any content read")):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
                self.read()
        self.assertTrue(observed)

    def test_public_repo_roster_location_remains_denied(self):
        # The protected repo code path is intentionally never modified:
        # the synthetic temporary directory is not the active public repo,
        # but explicitly making it the repo root must still deny.
        with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
            roster.load_roster(self.path, self.packet, self.pin, self.root)

    def test_malformed_pin_refused_before_any_file_open(self):
        with patch.object(roster.os, "open", side_effect=AssertionError("no roster open")):
            with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
                self.read(pin="")


if __name__ == "__main__":
    unittest.main()
