"""Aggregate output controls: ONLY invented JSON in ephemeral directories."""

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
import exp053_verified_analysis as score


class SyntheticAggregateOutput(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='exp053-fake-aggregate-output-')
        self.addCleanup(tmp.cleanup)
        self.home=Path(tmp.name)
        self.repo=self.home/'invented-repository';self.repo.mkdir(mode=0o700)
        self.research=self.repo/'research';self.research.mkdir(mode=0o700)
        self.out=self.research/'invented-public-aggregate.json'
        self.decoy=self.home/'external';self.decoy.mkdir(mode=0o700)
        self.value={'invented_aggregate':'NO RACE ARTICLE, NO MODEL',
                    'privacy':'aggregates only'}
        # Scorer ROOT means the research/ source directory itself, not repo.
        # Patch only this private synthetic path, never the real repository.
        self.patch_root=patch.object(score,'ROOT',self.research)
        self.patch_root.start();self.addCleanup(self.patch_root.stop)

    def pin(self):
        return score._aggregate_output_parent_identity(self.out)

    def publish(self,pin=None):
        if pin is None:pin=self.pin()
        score._persist_source_free_aggregate(self.out,self.value,pin)

    def test_source_free_aggregate_written_exclusively_with_owner_only_mode(self):
        pin = self.pin()
        self.publish(pin)
        self.assertEqual(json.loads(self.out.read_text()),self.value)
        self.assertEqual(stat.S_IMODE(self.out.stat().st_mode),0o600)
        with self.assertRaises(FileExistsError):
            self.publish(pin)
        self.assertEqual(json.loads(self.out.read_text()),self.value)

    def test_wrong_research_root_denied_before_operator_claim(self):
        self.out=self.decoy/'invented.json'
        with self.assertRaisesRegex(ValueError,'HOLD EXP053: public aggregate'):
            self.pin()
        self.assertFalse(self.out.exists())

    def test_static_parent_symlink_refused(self):
        alias=self.home/'symlinked-research';alias.symlink_to(self.research)
        self.out=alias/'invented.json'
        with self.assertRaises((OSError,ValueError)):
            self.pin()
        self.assertFalse((self.research/self.out.name).exists())

    def test_static_leaf_symlink_refused_without_decoy_mutation(self):
        target=self.decoy/'existing-public-object.json'
        target.write_text('DO NOT TOUCH')
        self.out.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'HOLD EXP053: public aggregate'):
            self.pin()
        self.assertEqual(target.read_text(),'DO NOT TOUCH')

    def test_parent_swapped_after_approval_denied_and_no_external_write(self):
        pin=self.pin()
        moved=self.repo/'old-research';self.research.rename(moved)
        self.research.symlink_to(self.decoy,target_is_directory=True)
        with self.assertRaises((OSError,ValueError)):
            self.publish(pin)
        self.assertFalse((self.decoy/self.out.name).exists())
        self.assertFalse((moved/self.out.name).exists())

    def test_same_owned_new_parent_inode_does_not_reuse_approval(self):
        pin=self.pin()
        moved=self.repo/'old-research';self.research.rename(moved)
        self.research.mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError,'HOLD EXP053: private output parent differs'):
            self.publish(pin)
        self.assertFalse(self.out.exists())
        self.assertFalse((moved/self.out.name).exists())

    def test_would_be_parent_swap_to_external_without_symlink_never_accepted(self):
        pin=self.pin()
        old=self.repo/'old-research';self.research.rename(old)
        self.decoy.rename(self.research)
        with self.assertRaisesRegex(ValueError,'HOLD EXP053: private output parent differs'):
            self.publish(pin)
        self.assertFalse(self.out.exists())

    def test_missing_secure_flags_refuses_before_file_open(self):
        for flag in ('O_PATH','O_DIRECTORY','O_NOFOLLOW','O_NONBLOCK'):
            with self.subTest(flag=flag),patch.object(score.os,flag,0),\
                 patch.object(score.os,'open',side_effect=AssertionError('never open unsafe')):
                with self.assertRaisesRegex(ValueError,'^HOLD EXP053'):
                    self.pin()

    def test_unsafe_group_writable_research_directory_refused(self):
        self.research.chmod(0o770)
        with self.assertRaisesRegex(ValueError,'^HOLD EXP053: public aggregate output directory owner'):
            self.pin()
        self.assertFalse(self.out.exists())

    def test_wrong_expected_parent_identity_denied_before_any_creation(self):
        pin=self.pin()
        with self.assertRaisesRegex(ValueError,'^HOLD EXP053: private output parent differs'):
            self.publish((pin[0],pin[1]+100))
        self.assertFalse(self.out.exists())
        for bad in ('unexpected',[],(1,),None):
            if bad is None:continue
            with self.subTest(value=repr(bad)),self.assertRaises(ValueError):
                self.publish(bad)
        self.assertFalse(self.out.exists())

    def test_existing_aggregate_rejected_before_claim(self):
        self.out.write_text('DO NOT CHANGE')
        with self.assertRaisesRegex(ValueError,'^HOLD EXP053: public aggregate'):
            self.pin()
        self.assertEqual(self.out.read_text(),'DO NOT CHANGE')

    def test_replay_pins_directory_before_consuming_irreversible_claim(self):
        import inspect
        body=inspect.getsource(score.replay)
        self.assertLess(body.index('_aggregate_output_parent_identity(output)'),
                        body.index('consume_unique_run('))
        self.assertLess(body.index('consume_unique_run('),
                        body.index('_persist_source_free_aggregate('))
        self.assertNotIn('output.open("x")',body)

    def test_approved_root_inode_checked_before_ordinary_fresh_write(self):
        approval=self.pin()
        self.assertEqual(approval,(self.research.stat().st_dev,self.research.stat().st_ino))
        self.publish(approval)
        self.assertTrue(self.out.exists())
        self.assertFalse((self.decoy/self.out.name).exists())


if __name__=='__main__':unittest.main()
