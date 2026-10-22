"""Synthetic, user-local immutable EXP053 preflight deployment checks."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/"research"))
import exp053_preflight_release as release


class SourceOnlyRelease(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home=Path(self.temp.name)
        self.prefix=self.home/"user-local"/"research-preflight"
        self.source=self.home/"source"
        self.source.mkdir()
        for name in release.FILES:
            (self.source/name).write_bytes((ROOT/"research"/name).read_bytes())

    def _install(self):
        with patch.object(release,"_account_home",return_value=self.home):
            return release.install(self.prefix,source_dir=self.source)

    def test_dry_run_never_writes(self):
        files=release.source_inventory(self.source)
        self.assertEqual(len(files),4)
        self.assertFalse(self.prefix.exists())
        self.assertEqual(len(release.release_id(files)),20)

    def test_installs_complete_content_addressed_release(self):
        result=self._install()
        self.assertEqual(result["result"],"installed")
        installed=Path(result["path"])
        self.assertTrue(installed.is_relative_to(self.home))
        self.assertEqual(set(os.listdir(installed)),{*release.FILES,release.MANIFEST})
        self.assertEqual(result["files"],release.source_inventory(self.source))
        self.assertEqual(release.verify(installed)["release_id"],result["release_id"])
        self.assertEqual((installed/release.FILES[0]).stat().st_mode & 0o777,0o755)
        self.assertEqual((installed/release.FILES[1]).stat().st_mode & 0o777,0o644)
        self.assertEqual((installed/release.FILES[2]).stat().st_mode & 0o777,0o755)
        self.assertEqual((installed/release.FILES[3]).stat().st_mode & 0o777,0o644)

    def test_both_installed_tools_identical_to_reviewed_sources(self):
        result=self._install()
        installed=Path(result["path"])
        for name in ("exp053_witness_host_preflight.py",
                     "exp053_ssh_custody_overlap.py"):
            self.assertEqual(
                (installed/name).read_bytes(), (self.source/name).read_bytes())
        for key in ("scientific_execution_authorized", "NO_GO_RESEARCH_ONLY"):
            self.assertIn(key,(installed/"exp053_ssh_custody_overlap.py").read_text())

    def test_modified_custody_tool_fail_closed(self):
        target=Path(self._install()["path"])
        path=target/"exp053_ssh_custody_overlap.py"
        path.write_bytes(path.read_bytes()+b"\n# unreviewed change\n")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            release.verify(target)

    def test_repeated_install_idempotent_not_overwritten(self):
        first=self._install()
        second=self._install()
        self.assertEqual(second["result"],"already_verified")
        self.assertEqual(first["path"],second["path"])

    def test_modified_release_fail_closed(self):
        installed=Path(self._install()["path"])
        path=installed/release.FILES[0]
        path.write_bytes(path.read_bytes()+b"# changed\n")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            release.verify(installed)
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self._install()

    def test_extra_file_denied(self):
        installed=Path(self._install()["path"])
        (installed/"unreviewed_model.py").write_text("print('not allowed')")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            release.verify(installed)

    def test_outside_user_home_denied_before_creation(self):
        with patch.object(release,"_account_home",return_value=self.home):
            with self.assertRaisesRegex(ValueError,"HOLD"):
                release.install(self.home.parent/"external-research",source_dir=self.source)
        self.assertFalse((self.home.parent/"external-research").exists())

    def test_symlink_source_refused(self):
        original=self.source/release.FILES[0]
        backing=self.home/"source-backup"
        backing.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(backing)
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self._install()

    def test_symlink_prefix_component_refused(self):
        real=self.home/"real"
        real.mkdir()
        link=self.home/"link"
        link.symlink_to(real,target_is_directory=True)
        with patch.object(release,"_account_home",return_value=self.home):
            with self.assertRaisesRegex(ValueError,"HOLD"):
                release.install(link/"research-preflight",source_dir=self.source)

    def test_corrupt_manifest_refused(self):
        installed=Path(self._install()["path"])
        (installed/release.MANIFEST).write_text('{"schema":"other"}\n')
        with self.assertRaisesRegex(ValueError,"HOLD"):
            release.verify(installed)

    def test_source_change_produces_another_release_id(self):
        first=self._install()
        with (self.source/release.FILES[0]).open("ab") as f:
            f.write(b"# another research-only version\n")
        second=self._install()
        self.assertNotEqual(first["release_id"],second["release_id"])
        self.assertEqual(release.verify(first["path"])["release_id"],first["release_id"])

    def test_help_no_install(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(release.main([]),0)
        self.assertEqual(json.loads(output.getvalue())["result"],
                         "dry_run_no_files_written")


if __name__=="__main__":
    unittest.main()
