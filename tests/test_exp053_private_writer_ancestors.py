"""All paths and files are disposable synthetic writer-custody fixtures."""
import json
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
import exp053_race_review as review


class TestPrivateWriterAncestors(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='exp053-fake-writer-ancestors-')
        self.addCleanup(t.cleanup)
        self.home=Path(t.name)
        self.owned=self.home/'owner-private';self.owned.mkdir(mode=0o700)
        self.alias=self.home/'untrusted-alias';self.alias.symlink_to(self.owned,target_is_directory=True)
        self.payload={'invented':'NO LICENSED RACE OR HUMAN REVIEW'}

    def test_static_symlinked_ancestor_refused_before_any_output_creation(self):
        out=self.alias/'invented-private-review.json'
        with self.assertRaisesRegex((OSError,ValueError), '.*'):
            review.save_private(out,self.payload)
        self.assertFalse(out.exists())
        self.assertFalse((self.owned/out.name).exists())

    def test_symlinked_intermediate_ancestor_refused(self):
        inner=self.owned/'another-nested-private';inner.mkdir(mode=0o700)
        out=self.alias/'another-nested-private'/'synthetic.json'
        with self.assertRaises((OSError,ValueError)):
            review.save_private(out,self.payload)
        self.assertFalse((inner/'synthetic.json').exists())

    def test_secure_nested_real_directory_succeeds_with_mode_0600(self):
        nested=self.owned/'real-nested';nested.mkdir(mode=0o700)
        out=nested/'invented-private-review.json'
        review.save_private(out,self.payload)
        self.assertEqual(json.loads(out.read_text()),self.payload)
        self.assertEqual(stat.S_IMODE(out.stat().st_mode),0o600)

    def test_parent_owner_mismatch_denied_before_output_creation(self):
        actual_fstat=os.fstat
        parent_fd=[]
        def wrong_uid(fd):
            st=actual_fstat(fd)
            if stat.S_ISDIR(st.st_mode):
                parent_fd.append(fd)
                return SimpleNamespace(st_mode=st.st_mode,st_uid=os.geteuid()+1)
            return st
        out=self.owned/'never-created.json'
        with patch.object(review.os,'fstat',side_effect=wrong_uid):
            with self.assertRaisesRegex(ValueError,'HOLD EXP053: private output parent'):
                review.save_private(out,self.payload)
        self.assertTrue(parent_fd)
        self.assertFalse(out.exists())

    def test_missing_opath_holds_before_any_output_creation(self):
        out=self.owned/'synthetic.json'
        with patch.object(review.os,'O_PATH',0):
            with self.assertRaisesRegex(ValueError,'HOLD EXP053'):
                review.save_private(out,self.payload)
        self.assertFalse(out.exists())

    def test_final_parent_symlink_refused(self):
        out=self.alias/'direct.json'
        with self.assertRaises((ValueError,OSError)):
            review.save_private(out,self.payload)
        self.assertFalse((self.owned/'direct.json').exists())

    def test_ancestor_swap_during_write_cleans_up_original_only(self):
        holder=self.home/'holder';holder.mkdir(mode=0o700)
        target=holder/'real-parent';target.mkdir(mode=0o700)
        moved=self.home/'moved-parent'
        decoy=self.home/'decoy';decoy.mkdir(mode=0o700)
        output=target/'invented-review.json'
        real_write=os.write
        swapped=False
        def swap(fd,payload):
            nonlocal swapped
            if not swapped:
                swapped=True
                holder.rename(moved)
                holder.symlink_to(decoy,target_is_directory=True)
            return real_write(fd,payload)
        with patch.object(review.os,'write',side_effect=swap):
            with self.assertRaises((OSError,ValueError)):
                review.save_private(output,self.payload)
        self.assertTrue(swapped)
        self.assertFalse((moved/'real-parent'/'invented-review.json').exists())
        self.assertFalse((decoy/'real-parent'/'invented-review.json').exists())


if __name__=='__main__':unittest.main()
