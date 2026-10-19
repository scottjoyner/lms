"""Source-free custodian packet: deterministic, non-authorizing, never remote."""

import contextlib
import io
import json
import os
import stat
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

    def test_keyboard_interrupt_during_write_cleans_named_packet(self):
        original_fsync = os.fsync
        interrupted = False

        def signal_during_file_fsync(fd):
            nonlocal interrupted
            if not interrupted and stat.S_ISREG(os.fstat(fd).st_mode):
                interrupted = True
                raise KeyboardInterrupt("synthetic SIGINT")
            return original_fsync(fd)

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fsync", side_effect=signal_during_file_fsync):
            with self.assertRaises(KeyboardInterrupt):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(interrupted)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.dest.iterdir()), [])

    def test_keyboard_interrupt_before_initial_file_fstat_retries_cleanup(self):
        real_fstat = os.fstat
        interrupted = False

        def signal_on_initial_fstat(fd):
            nonlocal interrupted
            info = real_fstat(fd)
            if not interrupted and stat.S_ISREG(info.st_mode) and info.st_size == 0:
                interrupted = True
                raise KeyboardInterrupt("synthetic SIGINT before identity saved")
            return info

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fstat", side_effect=signal_on_initial_fstat):
            with self.assertRaises(KeyboardInterrupt):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(interrupted)
        self.assertFalse(self.output.exists())

    def test_keyboard_interrupt_during_fdopen_closes_raw_fd(self):
        real_fdopen = os.fdopen
        interrupted_fd = []

        def signal_during_fdopen(fd, mode="r", *args, **kwargs):
            if mode == "w+b":
                interrupted_fd.append(fd)
                raise KeyboardInterrupt("synthetic SIGINT before descriptor adoption")
            return real_fdopen(fd, mode, *args, **kwargs)

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fdopen", side_effect=signal_during_fdopen):
            with self.assertRaises(KeyboardInterrupt):
                packet.create(self.output, source_dir=self.src)
        self.assertEqual(len(interrupted_fd), 1)
        with self.assertRaises(OSError):
            os.fstat(interrupted_fd[0])
        self.assertFalse(self.output.exists())

    def test_cli_keyboard_interrupt_is_sanitized_no_go(self):
        out = io.StringIO()
        with patch.object(packet, "inventory", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(out):
            rc = packet.main(["--inspect"])
        self.assertEqual(rc, 2)
        result = json.loads(out.getvalue())
        self.assertEqual(result["status"], "INVALID_OR_UNVERIFIABLE_PACKET")
        self.assertFalse(result["scientific_execution_authorized"])
        self.assertFalse(result["independent_administration_approved"])

    def test_created_empty_packet_is_cleaned_after_initial_fstat_error(self):
        # Confirmed failure in the prior version: successful O_EXCL followed
        # by a one-time metadata read failure left a zero-byte output behind.
        real_fstat = os.fstat
        failed = False

        def initial_file_info_error(fd):
            nonlocal failed
            info = real_fstat(fd)
            if not failed and stat.S_ISREG(info.st_mode) and info.st_size == 0:
                failed = True
                raise OSError("synthetic first regular-file fstat error")
            return info

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fstat", side_effect=initial_file_info_error):
            with self.assertRaises(OSError):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(failed)
        self.assertFalse(self.output.exists())
        self.assertEqual(set(self.dest.iterdir()), set())

    def test_fdopen_failure_closes_fd_and_removes_exclusive_entry(self):
        fdopen = os.fdopen
        opened = []

        def reject_write_wrapper(fd, mode="r", *args, **kwargs):
            if mode == "w+b":
                opened.append(fd)
                raise OSError("synthetic binary descriptor wrapping failure")
            return fdopen(fd, mode, *args, **kwargs)

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fdopen", side_effect=reject_write_wrapper):
            with self.assertRaises(OSError):
                packet.create(self.output, source_dir=self.src)
        self.assertEqual(len(opened), 1)
        with self.assertRaises(OSError):
            os.fstat(opened[0])  # original descriptor MUST NOT remain open
        self.assertFalse(self.output.exists())

    def test_interrupted_flush_after_packet_write_cleans_partial_file(self):
        fsync = os.fsync
        interrupted = False

        def fail_first_file_flush(fd):
            nonlocal interrupted
            if not interrupted and stat.S_ISREG(os.fstat(fd).st_mode):
                interrupted = True
                raise OSError("synthetic persistence failure")
            return fsync(fd)

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fsync", side_effect=fail_first_file_flush):
            with self.assertRaises(OSError):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(interrupted)
        self.assertFalse(self.output.exists())

    def test_failure_cleanup_never_unlinks_substituted_file(self):
        fsync = os.fsync
        moved = self.dest / "original-incomplete.zip"
        fake = b"synthetic substitute that must survive cleanup"
        substituted = False

        def replace_during_fsync(fd):
            nonlocal substituted
            if not substituted and stat.S_ISREG(os.fstat(fd).st_mode):
                substituted = True
                self.output.rename(moved)
                self.output.write_bytes(fake)
                raise OSError("synthetic failure after third-party substitution")
            return fsync(fd)

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fsync", side_effect=replace_during_fsync):
            with self.assertRaises(OSError):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(substituted)
        self.assertEqual(self.output.read_bytes(), fake)
        self.assertEqual(moved.read_bytes(), self.prepared())

    def test_final_output_symlink_swap_after_fsync_is_refused(self):
        # Reproduces the second observed bug: the original code read back
        # bytes from the original descriptor, but never checked that its
        # directory entry still pointed at that inode.
        outside = self.root / "synthetic-decoy.zip"
        outside.write_bytes(b"NOT AN AUTHORIZED CUSTODIAN PACKET")
        moved = self.dest / "relocated-original.zip"
        real_fsync = os.fsync
        swapped = False

        def swap_after_write(fd):
            nonlocal swapped
            real_fsync(fd)
            if not swapped and os.fstat(fd).st_mode & 0o170000 == 0o100000:
                swapped = True
                self.output.rename(moved)
                self.output.symlink_to(outside)

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fsync", side_effect=swap_after_write):
            with self.assertRaises(packet.Hold):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(swapped)
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(outside.read_bytes(), b"NOT AN AUTHORIZED CUSTODIAN PACKET")
        self.assertEqual(moved.read_bytes(), self.prepared())

    def test_final_output_regular_substitution_after_fsync_is_refused(self):
        moved = self.dest / "original-packet-moved.zip"
        fake = b"SYNTHETIC DIFFERENT CONTENT"
        real_fsync = os.fsync
        substituted = False

        def substitute_after_write(fd):
            nonlocal substituted
            real_fsync(fd)
            if not substituted and os.fstat(fd).st_mode & 0o170000 == 0o100000:
                substituted = True
                self.output.rename(moved)
                self.output.write_bytes(fake)

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fsync", side_effect=substitute_after_write):
            with self.assertRaises(packet.Hold):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(substituted)
        self.assertEqual(self.output.read_bytes(), fake)
        self.assertEqual(moved.read_bytes(), self.prepared())

    def test_final_output_hardlink_after_fsync_is_refused(self):
        shadow = self.dest / "unexpected-second-link.zip"
        real_fsync = os.fsync
        linked = False

        def hardlink_after_write(fd):
            nonlocal linked
            real_fsync(fd)
            if not linked and os.fstat(fd).st_mode & 0o170000 == 0o100000:
                linked = True
                os.link(self.output, shadow)

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fsync", side_effect=hardlink_after_write):
            with self.assertRaises(packet.Hold):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(linked)
        self.assertFalse(self.output.exists())
        self.assertEqual(shadow.read_bytes(), self.prepared())

    def test_final_output_becomes_world_readable_after_fsync_is_refused(self):
        real_fsync = os.fsync
        chmodded = False

        def chmod_after_write(fd):
            nonlocal chmodded
            real_fsync(fd)
            if not chmodded and os.fstat(fd).st_mode & 0o170000 == 0o100000:
                chmodded = True
                os.chmod(self.output, 0o644)

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fsync", side_effect=chmod_after_write):
            with self.assertRaises(packet.Hold):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(chmodded)
        self.assertFalse(self.output.exists())

    def test_final_output_deleted_after_fsync_is_refused(self):
        real_fsync = os.fsync
        removed = False

        def delete_after_write(fd):
            nonlocal removed
            real_fsync(fd)
            if not removed and os.fstat(fd).st_mode & 0o170000 == 0o100000:
                removed = True
                self.output.unlink()

        with patch.object(packet, "_account_home", return_value=self.root), patch.object(
                packet.os, "fsync", side_effect=delete_after_write):
            with self.assertRaises(packet.Hold):
                packet.create(self.output, source_dir=self.src)
        self.assertTrue(removed)
        self.assertFalse(self.output.exists())

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
