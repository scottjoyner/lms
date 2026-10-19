"""Source-free custodian packet: deterministic, non-authorizing, never remote."""

import contextlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "research"))
import exp053_custodian_handoff_packet as packet


class OfflineHandoff(unittest.TestCase):
    def setUp(self):
        ctx = tempfile.TemporaryDirectory(prefix="exp053-custody-packet-")
        self.addCleanup(ctx.cleanup)
        self.root = Path(ctx.name)
        self.src = self.root / "src"
        self.src.mkdir()
        for item in packet.PINS:
            (self.src / item).write_bytes((ROOT / "research" / item).read_bytes())
        self.dest = self.root / "owner-private"
        self.dest.mkdir(mode=0o700)
        self.output = self.dest / "source-only.zip"

    def prepared(self):
        return packet.render_packet(packet.inventory(source_dir=self.src))

    def test_allowlist_contains_only_seven_source_free_docs(self):
        self.assertEqual(len(packet.PINS), 7)
        self.assertTrue(all(n.endswith((".md", ".json")) for n in packet.PINS))
        self.assertFalse(any(n.endswith(".py") for n in packet.PINS))
        self.assertTrue(all("PRIVATE" not in n and "PASSAGE" not in n for n in packet.PINS))
        info = packet.verify_packet(self.prepared())
        self.assertEqual(info["document_count"], 7)
        self.assertEqual(info["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(info["scientific_execution_authorized"])
        self.assertFalse(info["independent_administration_approved"])
        self.assertFalse(info["witness_durability_approved"])
        self.assertEqual(info["provenance"], "CALLER_ASSEMBLED_UNTRUSTED_SOURCE_ONLY")

    def test_deterministic_bytes_and_reviewed_manifest(self):
        raw = self.prepared()
        self.assertEqual(raw, self.prepared())
        info = packet.verify_packet(raw)
        self.assertEqual(info["packet_sha256"], packet._sha(raw))
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertEqual(len(archive.infolist()), 8)
            self.assertEqual(archive.namelist(), sorted(archive.namelist()))
            manifest = json.loads(archive.read(packet.MANIFEST))
            self.assertEqual(manifest["decision"], "NO_GO_RESEARCH_ONLY")
            for name in packet.PINS:
                data = archive.read("docs/" + name)
                self.assertEqual(packet._sha(data), packet.PINS[name])
                self.assertEqual(manifest["files"]["docs/" + name]["bytes"], len(data))
            self.assertFalse(any(n.startswith("/") or ".." in n for n in archive.namelist()))

    def test_create_user_local_only_exclusive_readable_and_private(self):
        with patch.object(packet, "_account_home", return_value=self.root):
            result = packet.create(self.output, source_dir=self.src)
        self.assertEqual(result["status"], "CREATED_USER_LOCAL_ONLY")
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o600)
        self.assertEqual(packet.verify_packet(self.output.read_bytes())["packet_sha256"],
                         result["packet_sha256"])
        with patch.object(packet, "_account_home", return_value=self.root):
            with self.assertRaises(packet.Hold):
                packet.create(self.output, source_dir=self.src)
        self.assertEqual(result["packet_sha256"], packet._sha(self.output.read_bytes()))

    def test_create_outside_account_home_refused_without_writes(self):
        outside = self.root.parent / "exp053-packet-outside-unwanted.zip"
        with patch.object(packet, "_account_home", return_value=self.root):
            with self.assertRaisesRegex(packet.Hold, "user-local"):
                packet.create(outside, source_dir=self.src)
            with self.assertRaisesRegex(packet.Hold, "user-local"):
                packet.create(self.root / ".." / "escape.zip", source_dir=self.src)
            with self.assertRaisesRegex(packet.Hold, "subfolder"):
                packet.create(self.root / "root.zip", source_dir=self.src)
        self.assertFalse(outside.exists())
        self.assertFalse((self.root / "root.zip").exists())

    def test_unowned_or_world_writable_destination_denied(self):
        open_dir = self.root / "unsafe"
        open_dir.mkdir(mode=0o777)
        open_dir.chmod(0o777)
        with patch.object(packet, "_account_home", return_value=self.root):
            with self.assertRaisesRegex(packet.Hold, "owner-private"):
                packet.create(open_dir / "packet.zip", source_dir=self.src)
        self.assertFalse((open_dir / "packet.zip").exists())

    def test_symlinked_output_parent_denied(self):
        link = self.root / "link"
        link.symlink_to(self.dest, target_is_directory=True)
        with patch.object(packet, "_account_home", return_value=self.root):
            with self.assertRaisesRegex(packet.Hold, "symlinked"):
                packet.create(link / "packet.zip", source_dir=self.src)
        self.assertFalse((self.dest / "packet.zip").exists())

    def test_race_swapping_output_parent_with_symlink_cannot_escape_directory(self):
        # Regression for an actually reproduced TOCTOU bug: the old implementation
        # checked the parent path, then used os.open(full_absolute_path).
        # Replacing that parent with a symlink just before O_CREAT wrote into
        # the unapproved directory. All state here is disposable synthetic data.
        outside = self.root / "not-approved"
        outside.mkdir(mode=0o700)
        moved = self.root / "owner-private-moved"
        ordinary_open = os.open
        swapped = False

        def swap_on_create(path, flags, *args, **kwargs):
            nonlocal swapped
            if (not swapped and str(path) == self.output.name and flags & os.O_CREAT):
                swapped = True
                self.dest.rename(moved)
                self.dest.symlink_to(outside, target_is_directory=True)
            return ordinary_open(path, flags, *args, **kwargs)

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "open", side_effect=swap_on_create):
            with self.assertRaises(packet.Hold):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(swapped)
        self.assertFalse((outside / self.output.name).exists())
        self.assertFalse((moved / self.output.name).exists())

    def test_race_between_directory_validation_and_creation_is_safe(self):
        outside = self.root / "not-approved"
        outside.mkdir(mode=0o700)
        moved = self.root / "owner-private-moved"
        actual_open = packet._safe_output_dirfd
        observed = 0

        def validate_then_swap(path):
            nonlocal observed
            observed += 1
            opened = actual_open(path)
            if observed == 1:
                self.dest.rename(moved)
                self.dest.symlink_to(outside, target_is_directory=True)
            return opened

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet, "_safe_output_dirfd", side_effect=validate_then_swap):
            with self.assertRaises(packet.Hold):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(observed >= 2)
        self.assertFalse((outside / self.output.name).exists())
        self.assertFalse((moved / self.output.name).exists())

    def test_nested_symlinked_output_component_rejected(self):
        inner = self.dest / "inner"
        inner.mkdir(mode=0o700)
        bad = self.dest / "alias"
        bad.symlink_to(inner, target_is_directory=True)
        with patch.object(packet, "_account_home", return_value=self.root):
            with self.assertRaisesRegex(packet.Hold, "symlinked"):
                packet.create(bad / "subdir" / "packet.zip", source_dir=self.src)
        self.assertFalse((inner / "subdir" / "packet.zip").exists())

    def test_success_path_is_no_overwrite_and_retains_source_hash(self):
        with patch.object(packet, "_account_home", return_value=self.root):
            result = packet.create(self.output, source_dir=self.src)
        expected = self.prepared()
        self.assertEqual(packet._sha(expected), result["packet_sha256"])
        self.assertEqual(self.output.read_bytes(), expected)
        self.assertFalse(result["scientific_execution_authorized"])
        self.assertFalse(result["independent_administration_approved"])

    def test_platform_without_nofollow_or_directory_flags_is_rejected(self):
        # Never silently degrade to a path-following create on unsupported OS.
        for flag in ("O_NOFOLLOW", "O_DIRECTORY"):
            with self.subTest(flag=flag), patch.object(packet.os, flag, 0), patch.object(
                    packet, "_account_home", return_value=self.root):
                with self.assertRaises(packet.Hold):
                    packet.create(self.output, source_dir=self.src)
                self.assertFalse(self.output.exists())

    def test_source_changes_are_refused_before_packet_created(self):
        name = next(iter(packet.PINS))
        target = self.src / name
        target.write_bytes(target.read_bytes() + b"\nUNREVIEWED\n")
        with self.assertRaisesRegex(packet.Hold, "digest"):
            packet.inventory(source_dir=self.src)
        with patch.object(packet, "_account_home", return_value=self.root):
            with self.assertRaises(packet.Hold):
                packet.create(self.output, source_dir=self.src)
        self.assertFalse(self.output.exists())

    def test_missing_or_symlinked_source_refused(self):
        name = next(iter(packet.PINS))
        target = self.src / name
        backup = self.root / "real"
        backup.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(backup)
        with self.assertRaises(packet.Hold):
            packet.inventory(source_dir=self.src)
        target.unlink()
        with self.assertRaises(packet.Hold):
            packet.inventory(source_dir=self.src)

    def test_tampered_member_crc_manifest_and_trailing_bytes_denied(self):
        raw = self.prepared()
        for mutated in (raw + b"tamper", raw[:-20], b"ZIP", raw + b"\0" * 200):
            with self.subTest(size=len(mutated)), self.assertRaises(packet.Hold):
                packet.verify_packet(mutated)
        # Build valid ZIP metadata but supply a different allowlisted body.
        with zipfile.ZipFile(io.BytesIO(raw)) as src:
            entries = {n: src.read(n) for n in src.namelist()}
        first = next(n for n in entries if n.startswith("docs/"))
        entries[first] += b"tampered"
        with self.assertRaises(packet.Hold):
            packet.render_packet({k: v for k, v in entries.items() if k != packet.MANIFEST})

    def test_extra_zip_member_and_duplicate_path_denied(self):
        contents = packet.inventory(source_dir=self.src)
        raw = self.prepared()
        with zipfile.ZipFile(io.BytesIO(raw)) as src:
            items = {n: src.read(n) for n in src.namelist()}
        for extra in ("../private-passages.txt", "subfolder/unreviewed.py"):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as zf:
                for name, data in items.items():
                    zf.writestr(name, data)
                zf.writestr(extra, b"not authorized")
            with self.subTest(extra=extra), self.assertRaises(packet.Hold):
                packet.verify_packet(stream.getvalue())
        self.assertEqual(len(contents), 7)

    def test_metadata_mismatch_and_zip_compression_refused(self):
        raw = self.prepared()
        stream = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(raw)) as src:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as dst:
                for name in src.namelist():
                    dst.writestr(name, src.read(name))
        with self.assertRaises(packet.Hold):
            packet.verify_packet(stream.getvalue())

    def test_untrusted_recursive_json_manifest_safely_refused(self):
        contents = packet.inventory(source_dir=self.src)
        raw = packet.render_packet(contents)
        deep = b"[" * 1200 + b"0" + b"]" * 1200
        with zipfile.ZipFile(io.BytesIO(raw)) as src:
            entries = {n: src.read(n) for n in src.namelist()}
        entries[packet.MANIFEST] = deep
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as dst:
            for name in sorted(entries):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                dst.writestr(info, entries[name])
        with self.assertRaises(packet.Hold):
            packet.verify_packet(stream.getvalue())

    def test_readonly_cli_does_not_write_and_never_grants_authority(self):
        before = set(self.root.iterdir())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = packet.main(["--inspect"])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(out.getvalue())["status"], "NO_FILES_WRITTEN")
        self.assertFalse(json.loads(out.getvalue())["scientific_execution_authorized"])
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertNotIn(str(self.root), out.getvalue())

    def test_cli_verify_original_and_fail_closed_on_corrupt_zip(self):
        self.output.write_bytes(self.prepared())
        for changed in (False, True):
            if changed:
                self.output.write_bytes(self.prepared() + b"appended")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = packet.main(["--verify", str(self.output)])
            self.assertEqual(rc, 2)
            report = json.loads(out.getvalue())
            self.assertEqual(report["decision"], "NO_GO_RESEARCH_ONLY")
            self.assertFalse(report["scientific_execution_authorized"])
            if changed:
                self.assertEqual(report["status"], "INVALID_OR_UNVERIFIABLE_PACKET")
            else:
                self.assertEqual(report["status"], "LOCAL_PACKET_BYTES_VERIFIED_ONLY")
            self.assertNotIn(str(self.output), out.getvalue())


if __name__ == "__main__":
    unittest.main()
