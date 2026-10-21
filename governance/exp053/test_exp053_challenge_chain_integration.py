"""Synthetic end-to-end V21 integration, no live GitHub, POSTs or merge actions.

Unlike the unit tests, this exercises actual Audit.run(), audit_pr(), and the
challenge screen together with one strict GET-only fake GitHub transport.
"""
from __future__ import annotations

import copy
import unittest
from urllib.parse import unquote

from exp053_admin_activation import REPO, verified_evidence, verify_local
from exp053_readonly_challenge_preflight import ChallengeError, screen

BASE_SHA = 'b' * 40
HEAD_SHA = 'a' * 40
MERGE_SHA = 'c' * 40
REQUIRED = tuple(verified_evidence()['verified_successful_check_contexts'])
CHALLENGE_PATH = 'governance/exp053/challenges/exp053-merge-denial-offline-fixture.md'


class GitHubFixture:
    """Exact repository-relative GET contract; fails on any unexpected endpoint."""

    def __init__(self, *, check_selection='merge', applied=True, main_protected=True,
                 corrupt_merge_parents=False, review_events=None, inline_notes=None):
        self.calls = []
        self.check_selection = check_selection
        self.applied = applied
        self.main_protected = main_protected
        self.corrupt_merge_parents = corrupt_merge_parents
        self.review_events = copy.deepcopy(review_events if review_events is not None else [])
        self.inline_notes = copy.deepcopy(inline_notes if inline_notes is not None else [])
        self.branch_names = ['main', *verified_evidence()['baseline']['research_exp053_branches']]
        self.policy = {
            **verify_local(), 'id': 42, 'source_type': 'Repository', 'source': REPO
        }
        self.pr = {
            'number': 49, 'state': 'open', 'draft': False, 'merged_at': None,
            'auto_merge': None, 'mergeable': True, 'merge_commit_sha': MERGE_SHA,
            'head': {'sha': HEAD_SHA, 'repo': {'full_name': REPO}},
            'base': {'ref': 'main', 'sha': BASE_SHA, 'repo': {'full_name': REPO}},
        }
        self.doc = {
            'filename': CHALLENGE_PATH, 'status': 'added', 'additions': 2,
            'deletions': 0, 'changes': 2,
            'patch': '@@ -0,0 +1,2 @@\n+# EXP-053 governance challenge fixture\n+synthetic only',
        }

    def call(self, method, path, payload=None):
        if method != 'GET' or payload is not None:
            raise AssertionError('Write or non-GET attempted')
        self.calls.append(path)
        if path.startswith('/rulesets?includes_parents=true'):
            return [{'id': 42, 'name': self.policy['name'],
                     'target': 'branch', 'enforcement': 'active'}]
        if path == '/rulesets/42':
            return copy.deepcopy(self.policy)
        if path.startswith('/branches?'):
            return [{'name': name} for name in self.branch_names]
        if path.startswith('/rules/branches/'):
            name = unquote(path.split('?', 1)[0][len('/rules/branches/'):])
            if name not in self.branch_names:
                raise AssertionError('Unknown branch effective-policy query')
            if not self.applied and name == 'main':
                return []
            return [{**rule, 'ruleset_id': 42, 'ruleset_source_type': 'Repository',
                     'ruleset_source': REPO} for rule in self.policy['rules']]
        if path.startswith('/branches/'):
            name = unquote(path[len('/branches/'):])
            if name not in self.branch_names:
                raise AssertionError('Unknown protected-branch query')
            return {'name': name, 'protected': not (name == 'main' and not self.main_protected),
                    'commit': {'sha': BASE_SHA if name == 'main' else HEAD_SHA}}
        if path == '/pulls/49':
            return copy.deepcopy(self.pr)
        if path.startswith('/pulls/49/files?'):
            return [copy.deepcopy(self.doc)]
        if path.startswith('/pulls/49/reviews?'):
            return copy.deepcopy(self.review_events)
        if path.startswith('/pulls/49/comments?'):
            return copy.deepcopy(self.inline_notes)
        if path == '/commits/' + MERGE_SHA:
            return {'sha': MERGE_SHA, 'parents': [
                {'sha': BASE_SHA}, {'sha': BASE_SHA if self.corrupt_merge_parents else HEAD_SHA}
            ]}
        if path.startswith('/commits/') and '/check-runs?' in path:
            sha = path.split('/')[2]
            if sha == MERGE_SHA and self.check_selection == 'head':
                return {'total_count': 0, 'check_runs': []}
            if sha not in (HEAD_SHA, MERGE_SHA):
                raise AssertionError('Unexpected checks SHA')
            checks = [{'name': context, 'id': i + 1,
                       'head_sha': sha, 'status': 'completed', 'conclusion': 'success',
                       'app': {'id': 15368}}
                      for i, context in enumerate(REQUIRED)]
            if self.check_selection == 'missing' and sha == MERGE_SHA:
                checks.pop()
            return {'total_count': len(checks), 'check_runs': checks}
        if path in ('/commits/' + MERGE_SHA + '/status',
                    '/commits/' + HEAD_SHA + '/status'):
            return {'statuses': []}
        raise AssertionError('Unexpected GET ' + path)


class CompleteChallengeChain(unittest.TestCase):
    def test_genuine_audit_and_ci_selection_merge_candidate_only(self):
        fake = GitHubFixture()
        result = screen(fake, 49, 'main')
        self.assertEqual(result['status'], 'HUMAN_MERGE_DENIAL_INSPECTION_CANDIDATE_ONLY')
        self.assertEqual(result['ci_source'], 'TEST_MERGE_COMMIT')
        self.assertEqual(result['evaluated_ci_sha'], MERGE_SHA)
        self.assertEqual(result['protected_base_sha'], BASE_SHA)
        self.assertFalse(result['merge_denial_attested'])
        self.assertFalse(result['merge_action_performed'])
        self.assertEqual(result['writes_performed'], 0)
        self.assertGreaterEqual(sum(route.startswith('/rules/branches/') for route in fake.calls), 12)

    def test_real_ci_auditor_falls_back_to_head_only_if_no_merge_activity(self):
        fake = GitHubFixture(check_selection='head')
        result = screen(fake, 49, 'main')
        self.assertEqual(result['ci_source'], 'HEAD_FALLBACK_NO_MERGE_STATUS')
        self.assertEqual(result['evaluated_ci_sha'], HEAD_SHA)

    def test_real_policy_audit_rejects_unapplied_target(self):
        fake = GitHubFixture(applied=False)
        with self.assertRaisesRegex(ChallengeError, 'effective governance rules'):
            screen(fake, 49, 'main')
        self.assertFalse(any(route.startswith('/commits/') for route in fake.calls))

    def test_real_policy_audit_rejects_unprotected_branch(self):
        fake = GitHubFixture(main_protected=False)
        with self.assertRaisesRegex(ChallengeError, 'effective governance rules'):
            screen(fake, 49, 'main')

    def test_real_chain_rejects_preexisting_changes_requested(self):
        fake = GitHubFixture(review_events=[{'id': 3, 'state': 'CHANGES_REQUESTED'}])
        with self.assertRaisesRegex(ChallengeError, 'empty submitted review history'):
            screen(fake, 49, 'main')
        self.assertFalse(any(route.startswith('/commits/') for route in fake.calls))

    def test_real_chain_rejects_inline_review_discussion(self):
        fake = GitHubFixture(inline_notes=[{'id': 4, 'body': 'blocking discussion'}])
        with self.assertRaisesRegex(ChallengeError, 'zero inline review comments'):
            screen(fake, 49, 'main')
        self.assertFalse(any(route.startswith('/commits/') for route in fake.calls))

    def test_real_ci_audit_rejects_missing_merge_check(self):
        fake = GitHubFixture(check_selection='missing')
        with self.assertRaisesRegex(ChallengeError, 'CI selection'):
            screen(fake, 49, 'main')

    def test_real_ci_audit_rejects_wrong_merge_parents(self):
        fake = GitHubFixture(corrupt_merge_parents=True)
        with self.assertRaisesRegex(ChallengeError, 'CI selection'):
            screen(fake, 49, 'main')


if __name__ == '__main__':
    unittest.main()
