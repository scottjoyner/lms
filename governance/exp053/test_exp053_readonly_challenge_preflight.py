"""Synthetic-only V10 challenge candidate tests; never call GitHub."""
import copy
import unittest
from unittest.mock import patch
from exp053_readonly_challenge_preflight import screen,ChallengeError

SHA='a'*40
BASE='b'*40
MERGE='c'*40
PR={'number':49,'state':'open','draft':False,'merged_at':None,
    'mergeable':True,'merge_commit_sha':MERGE,
    'head':{'sha':SHA,'repo':{'full_name':'scottjoyner/lms'}},
    'base':{'sha':BASE,'ref':'main',
            'repo':{'full_name':'scottjoyner/lms'}}}

class Fake:
    def __init__(self, pr=None, rules=True, reviews=None, drift=False):
        self.pr=copy.deepcopy(PR if pr is None else pr)
        self.rules=rules
        self.reviews=reviews if reviews is not None else []
        self.drift=drift
        self.calls=[]
        self.pr_reads=0
    def call(self,method,path,payload=None):
        if method!='GET' or payload is not None:
            raise AssertionError('Read-only preflight attempted a write')
        self.calls.append(path)
        if path.startswith('/rulesets?'):
            return ([{'name':'EXP-053 require independent human review and verified CI',
                      'enforcement':'active','target':'branch','id':42}]
                    if self.rules else [])
        if path.startswith('/pulls/49/reviews?'):
            return self.reviews
        if path=='/pulls/49':
            self.pr_reads+=1
            p=copy.deepcopy(self.pr)
            if self.drift and self.pr_reads>1:
                p['head']['sha']='d'*40
            return p
        raise AssertionError('Unexpected GET '+path)

class Screening(unittest.TestCase):
    def setUp(self):
        self.audit=patch('exp053_verify_live_enforcement.Audit')
        self.ci=patch('exp053_merge_ref_ci_audit.audit_pr')
        self.audit_cls=self.audit.start()
        self.ci_fun=self.ci.start()
        self.audit_cls.return_value.run.return_value={
            'governance_status':'POLICY_METADATA_VERIFIED'}
        self.ci_fun.return_value={
            'status':'CI_SELECTION_VERIFIED_HEAD',
            'selected_commit_sha':SHA,
            'checks_selected_from':'HEAD_FALLBACK_NO_MERGE_STATUS'}
        self.addCleanup(self.audit.stop)
        self.addCleanup(self.ci.stop)
    def test_success_is_candidate_only(self):
        f=Fake()
        result=screen(f,49,'main')
        self.assertEqual(result['status'],'HUMAN_MERGE_DENIAL_INSPECTION_CANDIDATE_ONLY')
        self.assertFalse(result['merge_denial_attested'])
        self.assertFalse(result['merge_action_performed'])
        self.assertEqual(result['writes_performed'],0)
        self.assertTrue(all(x.startswith('/') for x in f.calls))
        self.assertTrue(self.ci_fun.call_args.kwargs['allow_isolated_challenge'])
    def test_no_ruleset_holds_before_pr(self):
        f=Fake(rules=False)
        with self.assertRaisesRegex(ChallengeError,'ruleset'):
            screen(f,49,'main')
        self.assertEqual(f.pr_reads,0)
    def test_draft_is_not_valid_denial(self):
        p=copy.deepcopy(PR);p['draft']=True
        with self.assertRaisesRegex(ChallengeError,'non-draft'):
            screen(Fake(pr=p),49,'main')
    def test_conflicted_pr_not_valid(self):
        p=copy.deepcopy(PR);p['mergeable']=False
        with self.assertRaisesRegex(ChallengeError,'conflict-free'):
            screen(Fake(pr=p),49,'main')
    def test_old_security_pr_never_qualifies(self):
        with self.assertRaisesRegex(ChallengeError,'Existing EXP-053 PR'):
            screen(Fake(),46,'main')
    def test_already_approved_is_not_missing_approval(self):
        with self.assertRaisesRegex(ChallengeError,'Already approved'):
            screen(Fake(reviews=[{'state':'APPROVED'}]),49,'main')
    def test_head_change_rejected(self):
        with self.assertRaisesRegex(ChallengeError,'changed'):
            screen(Fake(drift=True),49,'main')
    def test_bad_ci_selection_rejected(self):
        self.ci_fun.return_value={'status':'CI_SELECTION_UNVERIFIED'}
        with self.assertRaisesRegex(ChallengeError,'CI selection'):
            screen(Fake(),49,'main')
    def test_metadata_only_does_not_qualify(self):
        self.audit_cls.return_value.run.return_value={
            'governance_status':'NO_GO_GOVERNANCE'}
        with self.assertRaisesRegex(ChallengeError,'not verified'):
            screen(Fake(),49,'main')
    def test_wrong_branch_rejected(self):
        with self.assertRaisesRegex(ChallengeError,'Unexpected target'):
            screen(Fake(),49,'research/exp053-nonmatching')
    def test_nested_branch_rejected_without_network(self):
        f=Fake()
        with self.assertRaisesRegex(ChallengeError,'flat EXP-053'):
            screen(f,49,'research/exp053-child/nested')
        self.assertEqual(f.calls,[])
    def test_invalid_pr_rejected_without_network(self):
        f=Fake()
        with self.assertRaisesRegex(ChallengeError,'Invalid pull request'):
            screen(f,0,'main')
        self.assertEqual(f.calls,[])

if __name__=='__main__':
    unittest.main()
