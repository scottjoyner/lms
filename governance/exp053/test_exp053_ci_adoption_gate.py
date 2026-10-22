"""Synthetic git-history regression of the pinned required CI governance step.

Only temporary local repos and stub Python files are used. No GitHub network,
policy writes, actual model inference, or real workflow changes occur.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github' / 'workflows' / 'ci.yml'
STEP = '      - name: EXP-053 offline governance gate (when present in checkout)'
NEXT_STEP = '      - name: Build distributable wheel and source archive'


def workflow_script():
    source = WORKFLOW.read_text()
    section = source.split(STEP, 1)
    if len(section) != 2:
        raise AssertionError('required CI governance step removed or renamed')
    step = section[1].split(NEXT_STEP, 1)[0]
    if '        run: |\n' not in step:
        raise AssertionError('governance step run block missing')
    script = textwrap.dedent(step.split('        run: |\n', 1)[1])
    if not script.strip():
        raise AssertionError('empty governance step')
    return script


class RequiredGovernanceStep(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='exp053-ci-adoption-')
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / 'checkout'
        self.repo.mkdir()
        self.bin = Path(self.temporary.name) / 'bin'
        self.bin.mkdir()
        (self.bin / 'python').symlink_to(sys.executable)
        self.git('init', '-q')

    def git(self, *args):
        return subprocess.run(
            ['git', '-C', str(self.repo), *args], capture_output=True, text=True,
            check=True).stdout

    def commit(self, message):
        self.git('add', '-A')
        self.git('-c', 'user.name=Synthetic Test',
                 '-c', 'user.email=synthetic@example.invalid',
                 'commit', '-q', '--allow-empty', '-m', message)

    def add_suite(self, *, include_admin=True):
        base = self.repo / 'governance' / 'exp053'
        base.mkdir(parents=True, exist_ok=True)
        if include_admin:
            (base / 'exp053_admin_activation.py').write_text(
                'print("ADMIN_DRY_RUN_STUB_ZERO_WRITES")\n')
        (base / 'EXP053_validate_ruleset_candidate_2026-10-20.py').write_text(
            'print("POLICY_VALIDATOR_STUB_SUCCESS")\n')
        (base / 'test_exp053_synthetic.py').write_text(
            'import unittest\nclass Test(unittest.TestCase):\n'
            '    def test_success(self):\n        self.assertTrue(True)\n')

    def make_two_parent_pr_merge(self, *, initial_suite, change):
        """Recreate the actual GitHub PR test-merge ancestry with the base as HEAD^.

        Both commits and merge are confined to the disposable repo. No
        protected branch, GitHub API, or repository-wide Git config is touched.
        """
        self.git('config', 'user.name', 'Synthetic Test')
        self.git('config', 'user.email', 'synthetic@example.invalid')
        self.git('checkout', '-q', '-b', 'protected-base')
        (self.repo / 'base.txt').write_text('protected target branch\n')
        if initial_suite:
            self.add_suite()
            if change == 'delete_one_test':
                (self.repo / 'governance/exp053/test_exp053_second.py').write_text(
                    'import unittest\nclass Extra(unittest.TestCase):\n'
                    '    def test_extra(self):\n        self.assertTrue(True)\n')
        self.commit('target snapshot before PR')
        target_sha = self.git('rev-parse', 'HEAD').strip()
        self.git('checkout', '-q', '-b', 'pr-feature')
        if change == 'delete_one_test':
            (self.repo / 'governance/exp053/test_exp053_second.py').unlink()
        elif change == 'delete_suite':
            shutil.rmtree(self.repo / 'governance/exp053')
        elif change == 'add_suite':
            self.add_suite()
        elif change != 'keep_suite':
            raise AssertionError('unknown synthetic change')
        (self.repo / 'feature.txt').write_text('pull request change\n')
        self.commit('feature HEAD')
        feature_sha = self.git('rev-parse', 'HEAD').strip()
        self.git('checkout', '-q', 'protected-base')
        self.git('merge', '-q', '--no-ff', '--no-edit', 'pr-feature')
        # GitHub's generated pull_request merge commit has two parents:
        # HEAD^ is target/base and HEAD^2 is the proposed feature head.
        self.assertEqual(self.git('rev-list', '--parents', '-n', '1', 'HEAD').split(),
                         [self.git('rev-parse', 'HEAD').strip(), target_sha, feature_sha])
        self.assertEqual(self.git('rev-parse', 'HEAD^').strip(), target_sha)
        return self.run_required_step()

    def test_github_style_merge_rejects_removal_of_one_adopted_test(self):
        result = self.make_two_parent_pr_merge(
            initial_suite=True, change='delete_one_test')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Refusing removal of adopted governance test', result.stderr)

    def test_github_style_merge_rejects_removal_of_adopted_directory(self):
        result = self.make_two_parent_pr_merge(
            initial_suite=True, change='delete_suite')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Refusing removal of an already-adopted governance suite',
                      result.stderr)

    def test_github_style_merge_allows_initial_complete_adoption(self):
        result = self.make_two_parent_pr_merge(
            initial_suite=False, change='add_suite')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ADMIN_DRY_RUN_STUB_ZERO_WRITES', result.stdout)
        self.assertIn('Ran 1 test', result.stderr)

    def test_github_style_merge_continues_testing_after_adoption(self):
        result = self.make_two_parent_pr_merge(
            initial_suite=True, change='keep_suite')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ADMIN_DRY_RUN_STUB_ZERO_WRITES', result.stdout)
        self.assertIn('Ran 1 test', result.stderr)

    def run_required_step(self):
        return subprocess.run(
            ['bash', '-euo', 'pipefail', '-c', workflow_script()],
            cwd=self.repo, text=True, capture_output=True, timeout=15,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                 'PATH': str(self.bin) + os.pathsep + os.environ.get('PATH', '')})

    def test_pre_adoption_base_and_head_both_absent_is_allowed(self):
        self.commit('pre adoption baseline')
        self.commit('unrelated source-only change')
        result = self.run_required_step()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('not present in pre-adoption base', result.stdout)

    def test_suite_deletion_after_adoption_is_a_required_check_failure(self):
        self.add_suite()
        self.commit('adopt governance suite in base')
        shutil.rmtree(self.repo / 'governance' / 'exp053')
        self.commit('attempt deleting entire suite')
        result = self.run_required_step()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Refusing removal', result.stderr)

    def test_removing_only_adopted_test_module_fails_required_check(self):
        self.add_suite()
        self.commit('base with governance test module')
        (self.repo / 'governance/exp053/test_exp053_synthetic.py').unlink()
        self.commit('attempt to remove last test')
        result = self.run_required_step()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('zero test modules', result.stderr)

    def test_removing_one_of_two_adopted_test_modules_fails_required_check(self):
        self.add_suite()
        other = self.repo / 'governance/exp053/test_exp053_more.py'
        other.write_text('import unittest\nclass More(unittest.TestCase):\n'
                         '    def test_more(self):\n        self.assertTrue(True)\n')
        self.commit('base with two governance test modules')
        other.unlink()
        self.commit('attempt to drop one test while retaining other')
        result = self.run_required_step()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Refusing removal of adopted governance test', result.stderr)

    def test_initial_adoption_without_test_modules_fails_closed(self):
        self.commit('base without suite')
        self.add_suite()
        (self.repo / 'governance/exp053/test_exp053_synthetic.py').unlink()
        self.commit('attempt governance adoption with zero tests')
        result = self.run_required_step()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('zero test modules', result.stderr)

    def test_existing_governance_suite_runs_tests_and_two_guards(self):
        self.add_suite()
        self.commit('adopt governance suite in base')
        (self.repo / 'non-governance.txt').write_text('unrelated change\n')
        self.commit('ordinary PR keeps suite')
        result = self.run_required_step()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ADMIN_DRY_RUN_STUB_ZERO_WRITES', result.stdout)
        self.assertIn('POLICY_VALIDATOR_STUB_SUCCESS', result.stdout)
        self.assertIn('Ran 1 test', result.stderr)

    def test_partial_new_governance_directory_fails_closed(self):
        self.commit('base without governance')
        self.add_suite(include_admin=False)
        self.commit('partial governance addition')
        result = self.run_required_step()
        self.assertNotEqual(result.returncode, 0)

    def test_missing_base_parent_is_not_treated_as_pre_adoption(self):
        self.commit('root commit has no first parent')
        result = self.run_required_step()
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
