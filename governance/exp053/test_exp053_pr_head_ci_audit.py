"""Offline fixtures, no network or production/science data."""
import copy
import json
import unittest
from pathlib import Path
from urllib.parse import unquote
from exp053_pr_head_ci_audit import GithubReadOnly, SnapshotError, run, BASE, PR_NUMBERS, EXPECTED_STACK, verified_checks

EVIDENCE=json.loads((BASE/'EXP053_ruleset_validation_evidence_2026-10-20.json').read_text())
PIN={x['branch_or_pr']:x['sha'] for x in EVIDENCE['check_contexts_verified_at']}
CTX=EVIDENCE['verified_successful_check_contexts']


class Fake:
    def __init__(self):
        self.shifted_sha_by_pr={}
        self.bad_check=None
        self.bad_app=False
        self.draft_removed=None
        self.approved_review=False
        self.bad_base_pr=None
        self.calls=[]

    def __call__(self,path):
        self.calls.append(path)
        clean=path.split('?')[0]
        if clean=='/branches/main':
            return {'commit':{'sha':PIN['main']}}
        if clean.startswith('/pulls/'):
            fields=clean.split('/')
            number=int(fields[2])
            if len(fields)>3:
                if fields[3]=='reviews':
                    return [{'state':'APPROVED','user':{'login':'test-only-person'}}] if self.approved_review else []
                raise SnapshotError('unexpected nested endpoint')
            baseline_base,baseline_head=EXPECTED_STACK[number]
            return {'head':{'sha':self.shifted_sha_by_pr.get(number,PIN[f'PR #{number}']),
                            'ref':baseline_head},
                    'base':{'ref':'research/unexpected-parent' if number==self.bad_base_pr else baseline_base},
                    'state':'open','draft':number!=self.draft_removed,'merged':False,
                    'user':{'login':'test-author'},'requested_reviewers':[]}
        if clean.startswith('/commits/') and clean.endswith('/check-runs'):
            sha=clean.split('/')[2]
            checks=[]
            for k,name in enumerate(CTX):
                checks.append({'id':k+1,'name':name,'head_sha':sha,
                               'status':'completed',
                               'conclusion':'failure' if name==self.bad_check else 'success',
                               'app':{'id':1 if self.bad_app and k==0 else 15368}})
            return {'total_count':len(checks),'check_runs':checks}
        raise SnapshotError('unexpected fixture route')


class Tests(unittest.TestCase):
    def test_all_green_is_ci_only(self):
        x=run(GithubReadOnly(Fake()))
        self.assertEqual(x['snapshot_status'],'HEAD_CI_SNAPSHOT_VERIFIED')
        self.assertEqual(x['scientific_status'],'NO_GO_RESEARCH_ONLY')
        self.assertEqual(len(x['checks_by_head']),11)
        self.assertEqual(len(x['prs']),10)
    def test_changed_head_invalidates_previous_ci_snapshot(self):
        f=Fake(); f.shifted_sha_by_pr[46]='f'*40
        x=run(GithubReadOnly(f))
        self.assertFalse(x['baseline_head_shas_unchanged'])
        self.assertEqual(x['snapshot_status'],'HOLD_REFRESH_OR_REVIEW')
    def test_retargeted_stacked_pr_fails(self):
        f=Fake(); f.bad_base_pr=43
        x=run(GithubReadOnly(f))
        self.assertFalse(x['stack_base_head_ancestry_unchanged'])
        self.assertEqual(x['snapshot_status'],'HOLD_REFRESH_OR_REVIEW')
    def test_ci_failure_holds(self):
        f=Fake();f.bad_check=CTX[-1]
        x=run(GithubReadOnly(f))
        self.assertEqual(x['snapshot_status'],'HOLD_REFRESH_OR_REVIEW')
    def test_wrong_application_holds(self):
        f=Fake();f.bad_app=True
        x=run(GithubReadOnly(f))
        self.assertFalse(x['all_required_checks_succeeded_on_current_heads'])
    def test_nondraft_holds(self):
        f=Fake();f.draft_removed=46
        x=run(GithubReadOnly(f))
        self.assertFalse(x['all_sensitive_prs_remain_draft'])
    def test_human_review_not_scientific_authority(self):
        f=Fake();f.approved_review=True
        x=run(GithubReadOnly(f))
        self.assertEqual(x['scientific_status'],'NO_GO_RESEARCH_ONLY')
        self.assertEqual(x['prs'][0]['reported_approval_events_not_authority'],1)
    def test_latest_failed_rerun_rejects_older_success(self):
        sha=PIN['main']; n=CTX[0]
        good={'id':1,'name':n,'head_sha':sha,'status':'completed','conclusion':'success','app':{'id':15368}}
        bad=dict(good,id=2,conclusion='failure')
        self.assertFalse(verified_checks([good,bad],sha,[n])[n]['passed'])
    def test_wrong_sha_does_not_count(self):
        sha=PIN['main']; n=CTX[0]
        self.assertFalse(verified_checks([{'id':1,'name':n,'head_sha':'f'*40,
             'status':'completed','conclusion':'success','app':{'id':15368}}],sha,[n])[n]['passed'])
    def test_network_error_aborts(self):
        def fail(path): raise SnapshotError('test-only denied')
        with self.assertRaises(SnapshotError):run(GithubReadOnly(fail))

if __name__=='__main__':
    unittest.main()

# V3 regression: a newly queued rerun with no started_at must not be masked
# by a previously completed success from the same exact SHA.
class RerunFreshnessTests(unittest.TestCase):
    def test_pending_rerun_newer_creation_time_denies_old_success(self):
        sha=PIN['main']; n=CTX[0]
        older={'id':401,'name':n,'head_sha':sha,'created_at':'2026-10-20T15:01:00Z',
               'started_at':'2026-10-20T15:01:01Z','status':'completed',
               'conclusion':'success','app':{'id':15368}}
        newer={'id':402,'name':n,'head_sha':sha,'created_at':'2026-10-20T15:02:00Z',
               'started_at':None,'status':'queued','conclusion':None,'app':{'id':15368}}
        val=verified_checks([newer,older],sha,[n])[n]
        self.assertFalse(val['passed'])
        self.assertEqual(val['reason'],'LATEST_NOT_SUCCESSFUL')

    def test_duplicate_runs_missing_creation_time_are_ambiguous(self):
        sha=PIN['main']; n=CTX[0]
        a={'id':403,'name':n,'head_sha':sha,'status':'completed','conclusion':'success',
           'app':{'id':15368}}
        b={**a,'id':404}
        val=verified_checks([a,b],sha,[n])[n]
        self.assertFalse(val['passed'])
        self.assertEqual(val['reason'],'AMBIGUOUS_CHECK_HISTORY')

    def test_mismatched_app_does_not_supply_success(self):
        sha=PIN['main']; n=CTX[0]
        v={'id':405,'name':n,'head_sha':sha,'status':'completed','conclusion':'success',
           'app':{'id':999}}
        self.assertFalse(verified_checks([v],sha,[n])[n]['passed'])
