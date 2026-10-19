"""Synthetic-only owner preflight: never read private RACE or reviewer bytes."""

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import exp053_review_owner_readiness as ready


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class ReviewOwnerPreflight(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="exp053-owner-ready-synthetic-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.protocol = self.root / "protocol.md"
        self.protocol.write_text("synthetic preregistered protocol\n")
        self.source = self.root / "not_race.parquet"
        self.source.write_bytes(b"synthetic placeholder not a RACE dataset")
        self.private = self.root / "private"
        self.private.mkdir(mode=0o700)
        for f in ready.PRIVATE_FILES:
            target = self.private / f
            target.write_bytes(b"test-only non-licensed placeholder")
            target.chmod(0o600)
        self.manifest = self.root / "public-cohort.json"
        self.payload = {
            "schema": "exp053-public-manifest-v1",
            "source": "synthetic non-licensed test reference",
            "source_sha256": ready.review.SOURCE_SHA,
            "protocol_sha256": ready.review.PROTOCOL_SHA,
            "packet_sha256": sha(b"synthetic-packet-reference"),
            "sealed_answer_key_sha256": sha(b"synthetic-key-reference"),
            "candidate_articles": 64,
            "human_reviewed_articles": 0,
            "minimum_human_reviewed": 40,
            "evaluation_admitted": False,
            "copyright": "no RACE passage appears in any test fixture",
            "rows": [
                {"case_id": sha(f"case-{n}".encode())[:20],
                 "article_sha256": sha(f"article-{n}".encode()),
                 "question_sha256": sha(f"question-{n}".encode()),
                 "options_sha256": sha(f"options-{n}".encode()),
                 "article_chars": 250 + n} for n in range(64)
            ],
        }
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.payload, sort_keys=True))

    def inspect(self, *, reviewer_pin=""):
        return ready.inspect(
            manifest_path=self.manifest,
            protocol_path=self.protocol,
            source_path=self.source,
            private_dir=self.private,
            pinned_manifest_sha=sha(self.manifest.read_bytes()),
            pinned_protocol_sha=sha(self.protocol.read_bytes()),
            reviewer_pin=reviewer_pin,
        )

    def test_64_distinct_articles_and_two_real_slots_remain_unapproved(self):
        report = self.inspect()
        self.assertEqual(report["candidate_article_count"], 64)
        self.assertEqual(report["minimum_two_reviewer_case_decisions"], 80)
        self.assertEqual(report["required_distinct_co_reviewed_articles"], 40)
        self.assertEqual(report["reviewer_enrollment_status"], "UNAPPROVED_OWNER_PIN")
        self.assertEqual(report["admitted_human_review_count"], 0)
        self.assertEqual(report["blind_review_slots"], ["A", "B"])
        self.assertFalse(report["private_content_accessed"])
        self.assertFalse(report["scientific_execution_authorized"])
        self.assertFalse(report["independent_human_review_authorized"])
        self.assertTrue(all(x == "PRESENT_METADATA_ONLY_UNVERIFIED"
                            for x in report["private_assets_metadata_only"].values()))
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_fixed_owner_pin_does_not_grant_enrollment_approval(self):
        result = self.inspect(reviewer_pin="a" * 64)
        self.assertEqual(result["reviewer_enrollment_status"], "PIN_CONFIGURED_NOT_IDENTITY_PROOF")
        self.assertEqual(result["admitted_human_review_count"], "UNKNOWN_REQUIRES_PRIVATE_ADMISSION_GATE")
        self.assertFalse(result["independent_human_review_authorized"])

    def test_duplicate_article_hash_is_refused(self):
        self.payload["rows"][1]["article_sha256"] = self.payload["rows"][0]["article_sha256"]
        self.write_manifest()
        with self.assertRaises(ready.Hold):
            self.inspect()

    def test_duplicate_case_id_is_refused(self):
        self.payload["rows"][1]["case_id"] = self.payload["rows"][0]["case_id"]
        self.write_manifest()
        with self.assertRaises(ready.Hold):
            self.inspect()

    def test_extra_private_or_source_text_manifest_field_is_refused(self):
        self.payload["rows"][0]["licensed_source_text"] = "this is not a licensed passage"
        self.write_manifest()
        with self.assertRaises(ready.Hold):
            self.inspect()

    def test_changed_source_key_or_admission_in_manifest_denied(self):
        for key, val in (("source_sha256", "b"*64), ("evaluation_admitted", True),
                         ("human_reviewed_articles", 40), ("minimum_human_reviewed", 0),
                         ("candidate_articles", 63)):
            original = self.payload[key]
            with self.subTest(key=key):
                self.payload[key] = val
                self.write_manifest()
                with self.assertRaises(ready.Hold):
                    self.inspect()
            self.payload[key] = original

    def test_public_manifest_pin_changed_denied_even_if_structurally_valid(self):
        with self.assertRaises(ready.Hold):
            ready.inspect(manifest_path=self.manifest, protocol_path=self.protocol,
                          private_dir=self.private, source_path=self.source,
                          pinned_manifest_sha="0"*64,
                          pinned_protocol_sha=sha(self.protocol.read_bytes()))

    def test_world_readable_private_form_is_blocked(self):
        (self.private / "reviewer_A.json").chmod(0o644)
        report = self.inspect()
        self.assertEqual(report["private_assets_metadata_only"]["reviewer_A.json"], "BLOCKED_UNVERIFIED")
        self.assertEqual(report["private_assets_metadata_only"]["reviewer_B.json"],
                         "PRESENT_METADATA_ONLY_UNVERIFIED")
        self.assertFalse(report["independent_human_review_authorized"])

    def test_world_traversable_private_directory_is_blocked(self):
        self.private.chmod(0o755)
        report = self.inspect()
        self.assertTrue(all(x == "BLOCKED_UNVERIFIED"
                            for x in report["private_assets_metadata_only"].values()))

    def test_symlink_to_a_private_form_is_not_treated_as_safe(self):
        form = self.private / "reviewer_B.json"
        backing = self.root / "synthetic-original"
        form.rename(backing)
        form.symlink_to(backing)
        report = self.inspect()
        self.assertEqual(report["private_assets_metadata_only"]["reviewer_B.json"], "BLOCKED_UNVERIFIED")

    def test_hf_corpus_symlink_is_explicitly_unverified_not_fake_verified(self):
        backing = self.root / "synthetic-blob"
        self.source.rename(backing)
        self.source.symlink_to(backing)
        report = self.inspect()
        self.assertEqual(report["corpus"], "SYMLINK_PRESENT_TARGET_UNVERIFIED")
        self.assertFalse(report["scientific_execution_authorized"])

    def test_broken_symlink_corpus_remains_not_verified(self):
        self.source.unlink()
        self.source.symlink_to(self.root / "missing-blob")
        report = self.inspect()
        self.assertEqual(report["corpus"], "SYMLINK_PRESENT_TARGET_UNVERIFIED")

    def test_private_files_are_never_opened_even_when_all_exist(self):
        real_open = os.open
        unexpected = []
        def guarded_open(path, flags, *args, **kwargs):
            if ((str(path).startswith(str(self.private)) and str(path) != str(self.private)) or
                    str(path) == str(self.source)):
                unexpected.append(str(path))
                raise AssertionError("private content must not be opened")
            return real_open(path, flags, *args, **kwargs)
        with patch.object(ready.os, "open", side_effect=guarded_open):
            report = self.inspect()
        self.assertEqual(unexpected, [])
        self.assertFalse(report["private_content_accessed"])

    def test_symlink_swap_during_directory_relative_stat_is_blocked(self):
        # The previous path-based lstat check could observe the destination
        # directory before a rename and different files after a symlink swap.
        original_stat = os.stat
        moved = self.root / "private-moved"
        substitute = self.root / "synthetic-untrusted-files"
        substitute.mkdir(mode=0o700)
        for name in ready.PRIVATE_FILES:
            (substitute / name).write_bytes(b"untrusted placeholder")
            (substitute / name).chmod(0o600)
        swapped = False

        def swap_on_first_private_stat(path, *args, **kwargs):
            nonlocal swapped
            if (not swapped and path in ready.PRIVATE_FILES and
                    kwargs.get("dir_fd") is not None):
                swapped = True
                self.private.rename(moved)
                self.private.symlink_to(substitute, target_is_directory=True)
            return original_stat(path, *args, **kwargs)

        with patch.object(ready.os, "stat", side_effect=swap_on_first_private_stat):
            result = self.inspect()
        self.assertTrue(swapped)
        self.assertTrue(all(value == "BLOCKED_UNVERIFIED"
                            for value in result["private_assets_metadata_only"].values()))
        self.assertFalse(result["scientific_execution_authorized"])

    def test_missing_secure_directory_flags_refuses_private_presence(self):
        with patch.object(ready.os, "O_DIRECTORY", 0):
            result = self.inspect()
        self.assertTrue(all(x == "BLOCKED_UNVERIFIED"
                            for x in result["private_assets_metadata_only"].values()))
        self.assertFalse(result["independent_human_review_authorized"])

    def test_missing_public_nofollow_or_nonblocking_flag_denies_all_reads(self):
        for name in ("O_NOFOLLOW", "O_NONBLOCK"):
            with self.subTest(flag=name), patch.object(ready.os, name, 0):
                with self.assertRaises(ready.Hold):
                    self.inspect()

    def test_public_fdopen_failure_does_not_leak_a_descriptor(self):
        original = os.fdopen
        captured = []
        def refuse(fd, mode="r", *args, **kwargs):
            if mode == "rb":
                captured.append(fd)
                raise OSError("synthetic public file wrapper failure")
            return original(fd, mode, *args, **kwargs)
        with patch.object(ready.os, "fdopen", side_effect=refuse):
            with self.assertRaises(ready.Hold):
                self.inspect()
        self.assertTrue(captured)
        for fd in captured:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_public_fdopen_keyboard_interrupt_closes_fd_and_propagates(self):
        original = os.fdopen
        captured = []
        def refuse(fd, mode="r", *args, **kwargs):
            if mode == "rb":
                captured.append(fd)
                raise KeyboardInterrupt("synthetic SIGINT during read wrapper")
            return original(fd, mode, *args, **kwargs)
        with patch.object(ready.os, "fdopen", side_effect=refuse):
            with self.assertRaises(KeyboardInterrupt):
                self.inspect()
        self.assertTrue(captured)
        for fd in captured:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_public_file_symlink_is_not_followed(self):
        original = self.root / "original-spec"
        spec = self.root / "symlink-spec"
        original.write_bytes(ready.SPEC.read_bytes())
        spec.symlink_to(original)
        with self.assertRaises(ready.Hold):
            ready.inspect(spec_path=spec)


    def test_nested_json_parser_ambiguous_fields_rejected(self):
        raw = self.manifest.read_bytes()
        # Preserve structure and replace only one occurrence with duplicates.
        edited = raw.replace(b'"candidate_articles": 64,',
                             b'"candidate_articles": 64, "candidate_articles": 64,',1)
        self.assertNotEqual(edited, raw)
        self.manifest.write_bytes(edited)
        with self.assertRaises(ready.Hold):
            self.inspect()

    def test_real_entrypoint_always_reports_source_free_no_go(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            result = ready.main()
        self.assertEqual(result, 2)
        obj = json.loads(capture.getvalue())
        self.assertEqual(obj["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(obj["scientific_execution_authorized"])
        self.assertNotIn(str(self.root), capture.getvalue())
        self.assertNotIn("synthetic placeholder", capture.getvalue())
        self.assertEqual(obj["admitted_human_review_count"], 0)


if __name__ == "__main__":
    unittest.main()
