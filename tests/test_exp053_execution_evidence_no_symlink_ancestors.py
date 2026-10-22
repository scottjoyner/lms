"""All evidence files are invented ephemeral data, never genuine RACE or models."""
import hashlib
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
import exp053_execution_attestation as source


class PrivateEvidenceAncestorSafety(unittest.TestCase):
    def setUp(self):
        context=tempfile.TemporaryDirectory(prefix='exp053-synthetic-evidence-')
        self.addCleanup(context.cleanup)
        self.root=Path(context.name)
        self.private=self.root/'invented-private';self.private.mkdir(mode=0o700)
        self.sub=self.private/'real-nested';self.sub.mkdir(mode=0o700)
        self.path=self.sub/'fiction-only-evidence.json'
        self.payload=b'{"synthetic_only":true,"rows":0}'
        self.path.write_bytes(self.payload)
        self.path.chmod(0o600)

    def reader(self,path=None,private=None,limit=4096):
        return source.private_regular_file(
            self.path if path is None else path,
            self.private if private is None else private,
            small_limit=limit)

    def test_good_owner_private_nested_path_hashes_exact_opened_bytes(self):
        digest,raw=self.reader()
        self.assertEqual(digest,hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(raw,self.payload)
        digest,raw=self.reader(limit=None)
        self.assertEqual(digest,hashlib.sha256(self.payload).hexdigest())
        self.assertIsNone(raw)

    def test_static_symlink_ancestor_refused_despite_resolved_in_root(self):
        alias=self.private/'invented-alias';alias.symlink_to(self.sub,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'HOLD'):
            self.reader(alias/self.path.name)

    def test_symlink_private_directory_refused(self):
        alias=self.root/'symlink-private-dir';alias.symlink_to(self.private,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'HOLD'):
            self.reader(alias/'real-nested'/self.path.name,private=alias)

    def test_outside_root_and_traversal_refused(self):
        outside=self.root/'invented-public.json';outside.write_bytes(self.payload)
        with self.assertRaisesRegex(ValueError,'HOLD'):
            self.reader(outside)
        with self.assertRaisesRegex(ValueError,'HOLD'):
            self.reader(self.private/'real-nested'/'..'/'..'/outside.name)

    def test_leaf_symlink_refused(self):
        link=self.sub/'fiction-link.json';link.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError,'HOLD'):
            self.reader(link)

    def test_private_owner_mode_rejects_group_other_access(self):
        for mode in (0o640,0o604,0o666):
            with self.subTest(mode=mode):
                self.path.chmod(mode)
                with self.assertRaisesRegex(ValueError,'HOLD'):
                    self.reader()
        self.path.chmod(0o600)

    def test_hardlinked_file_denied(self):
        second=self.sub/'invented-hardlink'
        os.link(self.path,second)
        with self.assertRaisesRegex(ValueError,'HOLD'):
            self.reader()

    def test_fifo_is_denied_without_blocking(self):
        if not hasattr(os,'mkfifo'):self.skipTest('FIFO unsupported')
        fifo=self.sub/'invented-fifo'
        os.mkfifo(fifo,mode=0o600)
        with self.assertRaisesRegex(ValueError,'HOLD'):
            self.reader(fifo)

    def test_missing_no_follow_kernel_flags_fail_before_open(self):
        for flag in ('O_PATH','O_DIRECTORY','O_NOFOLLOW','O_NONBLOCK'):
            with self.subTest(flag=flag),patch.object(source.os,flag,0),\
                 patch.object(source.os,'open',side_effect=AssertionError('must not open')):
                with self.assertRaisesRegex(ValueError,'HOLD'):
                    self.reader()

    def test_same_bytes_replaced_after_first_fd_read_denied(self):
        other=self.sub/'replacement-same-payload.json'
        other.write_bytes(self.payload)
        original=self.sub/'moved-original.json'
        original_read=os.read
        swapped=False
        def swap(fd,count):
            nonlocal swapped
            data=original_read(fd,count)
            if data and not swapped:
                swapped=True
                self.path.rename(original)
                other.rename(self.path)
            return data
        with patch.object(source.os,'read',side_effect=swap):
            with self.assertRaisesRegex(ValueError,'HOLD'):
                self.reader()
        self.assertTrue(swapped)
        self.assertEqual(self.path.read_bytes(),self.payload)
        self.assertEqual(original.read_bytes(),self.payload)

    def test_symlink_substitution_midread_denied(self):
        other=self.sub/'invented-fake-replacement.json';other.write_bytes(self.payload)
        original=self.sub/'original-inode.json'
        original_read=os.read
        changed=False
        def replace(fd,count):
            nonlocal changed
            result=original_read(fd,count)
            if result and not changed:
                changed=True
                self.path.rename(original)
                self.path.symlink_to(other)
            return result
        with patch.object(source.os,'read',side_effect=replace):
            with self.assertRaisesRegex(ValueError,'HOLD'):
                self.reader()
        self.assertTrue(changed)
        self.assertFalse(other.is_symlink())

    def test_midread_rewrite_refuses_changed_fd_metadata(self):
        original_read=os.read
        rewritten=False
        def rewrite(fd,count):
            nonlocal rewritten
            data=original_read(fd,count)
            if data and not rewritten:
                rewritten=True
                self.path.write_bytes(b'SYNTHETIC-MUTATED-BYTES')
            return data
        with patch.object(source.os,'read',side_effect=rewrite):
            with self.assertRaisesRegex(ValueError,'HOLD'):
                self.reader()
        self.assertTrue(rewritten)

    def test_wrong_owner_refused(self):
        real_fstat=os.fstat
        def invalid_owner(fd):
            st=real_fstat(fd)
            if stat.S_ISREG(st.st_mode):
                return SimpleNamespace(st_mode=st.st_mode,st_nlink=st.st_nlink,
                    st_uid=os.geteuid()+1,st_size=st.st_size)
            return st
        with patch.object(source.os,'fstat',side_effect=invalid_owner):
            with self.assertRaisesRegex(ValueError,'HOLD'):
                self.reader()

    def test_invalid_limits_refused_without_source_read(self):
        for invalid in (0,-1,True,'4096'):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError,'HOLD'):
                    self.reader(limit=invalid)

    def test_oversized_private_json_refused(self):
        with self.assertRaisesRegex(ValueError,'HOLD'):
            self.reader(limit=10)


if __name__=='__main__':unittest.main()
