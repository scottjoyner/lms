"""Offline synthetic tests: no requests leave the process."""
import json
import unittest
from exp053_merge_ref_ci_audit import audit_pr
from exp053_pr_head_ci_audit import GithubReadOnly, EVIDENCE, SnapshotError

CTX = json.loads(EVIDENCE.read_text())['verified_successful_check_contexts']
BASE='a'*40
HEAD='b'*40
MERGE='c'*40

class Fake:
    def __init__(self, merge_checks=False, head_checks=True, other_merge_job=False,
                 wrong_parents=False, drift=False, queued=False, legacy_failed=False,
                 legacy_merge_only=False, number=46):
        self.merge_checks=merge_checks; self.head_checks=head_checks
        self.other_merge_job=other_merge_job; self.wrong_parents=wrong_parents
        self.drift=drift; self.queued=queued; self.legacy_failed=legacy_failed
        self.legacy_merge_only=legacy_merge_only; self.pull_calls=0
        self.number=number

    def __call__(self, path):
        path=path.split('?')[0]
        if path==f'/pulls/{self.number}':
            self.pull_calls+=1
            return {'merge_commit_sha':('d'*40 if self.drift and self.pull_calls>1 else MERGE),
                    'head':{'sha':HEAD},'base':{'sha':BASE}}
        if path==f'/commits/{MERGE}':
            return {'parents':[{'sha':'e'*40 if self.wrong_parents else BASE},{'sha':HEAD}]}
        if path in (f'/commits/{MERGE}/status',f'/commits/{HEAD}/status'):
            legacy = []
            if self.legacy_merge_only and path.endswith(MERGE+'/status'):
                legacy = [{'id':1,'created_at':'2026-10-20T15:00:00Z','context':'unrelated', 'state':'success'}]
            if self.legacy_failed and path.endswith('/status'):
                legacy.append({'id':2,'created_at':'2026-10-20T15:00:00Z',
                               'context':CTX[0], 'state':'failure'})
            return {'statuses':legacy}
        if path in (f'/commits/{MERGE}/check-runs',f'/commits/{HEAD}/check-runs'):
            merge=path.endswith(MERGE+'/check-runs')
            full=self.merge_checks if merge else self.head_checks
            checks=[]
            if full:
                for i,name in enumerate(CTX):
                    checks.append({'id':i+1,'created_at':'2026-10-20T15:00:00Z',
                                   'name':name,'head_sha':MERGE if merge else HEAD,
                                   'app':{'id':15368}, 'status':'queued' if self.queued else 'completed',
                                   'conclusion':None if self.queued else 'success'})
            if merge and self.other_merge_job:
                checks.append({'id':200,'name':'unrelated', 'head_sha':MERGE, 'app':{'id':15368},
                               'status':'completed','conclusion':'success'})
            return {'total_count':len(checks),'check_runs':checks}
        raise AssertionError('unexpected route '+path)

class MergeRefTests(unittest.TestCase):
    def test_isolated_challenge_requires_explicit_opt_in(self):
        fixture=GithubReadOnly(Fake(number=49))
        with self.assertRaisesRegex(SnapshotError, 'allowlisted'):
            audit_pr(fixture,49,CTX)
        result=audit_pr(GithubReadOnly(Fake(number=49)),49,CTX,
                        allow_isolated_challenge=True)
        self.assertEqual(result['status'],'CI_SELECTION_VERIFIED_HEAD')

    def test_missing_merge_checks_selects_head_per_docs(self):
        r=audit_pr(GithubReadOnly(Fake()),46,CTX)
        self.assertEqual(r['status'],'CI_SELECTION_VERIFIED_HEAD')
        self.assertEqual(r['selected_commit_sha'],HEAD)
        self.assertEqual(r['merge_commit_check_run_count'],0)
        self.assertIn('not proof',r['claim_limit'])
    def test_no_checks_on_any_sha_stays_unverified(self):
        self.assertEqual(audit_pr(GithubReadOnly(Fake(head_checks=False)),46,CTX)['status'],
                         'CI_SELECTION_UNVERIFIED')
    def test_merge_checks_select_merge(self):
        r=audit_pr(GithubReadOnly(Fake(merge_checks=True)),46,CTX)
        self.assertEqual(r['status'],'CI_SELECTION_VERIFIED_MERGE')
    def test_queued_merge_checks_block_selected_evidence(self):
        r=audit_pr(GithubReadOnly(Fake(merge_checks=True,queued=True)),46,CTX)
        self.assertEqual(r['passing_required_checks_on_selected_commit'],0)
        self.assertEqual(r['status'],'CI_SELECTION_UNVERIFIED')
    def test_unrelated_merge_job_prevents_optimistic_head_fallback(self):
        r=audit_pr(GithubReadOnly(Fake(other_merge_job=True)),46,CTX)
        self.assertEqual(r['checks_selected_from'],'TEST_MERGE_COMMIT')
        self.assertEqual(r['status'],'CI_SELECTION_UNVERIFIED')
    def test_legacy_merge_status_prevents_head_fallback(self):
        r=audit_pr(GithubReadOnly(Fake(legacy_merge_only=True)),46,CTX)
        self.assertEqual(r['checks_selected_from'],'TEST_MERGE_COMMIT')
        self.assertEqual(r['status'],'CI_SELECTION_UNVERIFIED')
    def test_named_legacy_failure_blocks_otherwise_successful_check(self):
        r=audit_pr(GithubReadOnly(Fake(legacy_failed=True)),46,CTX)
        self.assertEqual(r['status'],'CI_SELECTION_UNVERIFIED')
        self.assertEqual(r['colliding_legacy_statuses_not_successful'],[CTX[0]])
    def test_wrong_parent_blocks_identity(self):
        r=audit_pr(GithubReadOnly(Fake(merge_checks=True,wrong_parents=True)),46,CTX)
        self.assertFalse(r['parents_match_current_base_and_head'])
        self.assertEqual(r['status'],'CI_SELECTION_UNVERIFIED')
    def test_pr_change_during_scan_holds(self):
        r=audit_pr(GithubReadOnly(Fake(merge_checks=True,drift=True)),46,CTX)
        self.assertFalse(r['snapshot_stable_during_read'])
        self.assertEqual(r['status'],'CI_SELECTION_UNVERIFIED')
    def test_allowlist(self):
        with self.assertRaises(SnapshotError):
            audit_pr(GithubReadOnly(Fake()),32,CTX)

if __name__=='__main__':
    unittest.main()
