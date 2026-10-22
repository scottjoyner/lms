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

DOC={'filename':'governance/exp053/challenges/exp053-merge-denial-example.md',
     'status':'added','additions':2,'deletions':0,'changes':2,
     'patch':'@@ -0,0 +1,2 @@\\n+# EXP-053 governance challenge fixture\\n+No production or scientific content.'}

class Fake:
    def __init__(self, pr=None, rules=True, reviews=None, drift=False, review_drift=False, policy_drift=False, list_drift=False, mergeable_drift=False, files=None, file_drift=False, target_drift=False, target_unprotected=False, target_wrong_sha=False, comments=None, comment_drift=False):
        self.pr=copy.deepcopy(PR if pr is None else pr)
        self.rules=rules
        self.reviews=reviews if reviews is not None else []
        self.drift=drift
        self.calls=[]
        self.pr_reads=0
        self.review_reads=0
        self.comments=comments if comments is not None else []
        self.comment_drift=comment_drift
        self.comment_reads=0
        self.policy_reads=0
        self.list_reads=0
        self.review_drift=review_drift
        self.policy_drift=policy_drift
        self.list_drift=list_drift
        self.mergeable_drift=mergeable_drift
        self.files=copy.deepcopy([DOC] if files is None else files)
        self.file_drift=file_drift
        self.file_reads=0
        self.target_reads=0
        self.target_drift=target_drift
        self.target_unprotected=target_unprotected
        self.target_wrong_sha=target_wrong_sha
    def call(self,method,path,payload=None):
        if method!='GET' or payload is not None:
            raise AssertionError('Read-only preflight attempted a write')
        self.calls.append(path)
        if path.startswith('/rulesets?'):
            self.list_reads+=1
            if not self.rules:
                return []
            items=[{'name':'EXP-053 require pull requests and pinned CI (zero approvals)',
                      'enforcement':'active','target':'branch','id':42}]
            if self.list_drift and self.list_reads>1:
                items.append({'id':43,'name':'unexpected ruleset'})
            return items
        if path.startswith('/branches/'):
            self.target_reads+=1
            return {'name':'main',
                    'protected':not self.target_unprotected,
                    'commit':{'sha':('c'*40 if self.target_wrong_sha or (self.target_drift and self.target_reads>1) else BASE)}}
        if path=='/rulesets/42':
            self.policy_reads+=1
            return {'id':42,'enforcement':'evaluate' if self.policy_drift and self.policy_reads>1 else 'active'}
        if path.startswith('/pulls/49/files?'):
            self.file_reads+=1
            rows=copy.deepcopy(self.files)
            if self.file_drift and self.file_reads>1:
                rows.append({'filename':'unexpected.py'})
            return rows
        if path.startswith('/pulls/49/reviews?'):
            self.review_reads+=1
            if self.review_drift and self.review_reads>1:
                return self.reviews+[{'id':999,'state':'APPROVED'}]
            return self.reviews
        if path.startswith('/pulls/49/comments?'):
            self.comment_reads+=1
            if self.comment_drift and self.comment_reads>1:
                return self.comments+[{'id':999,'body':'late inline review comment'}]
            return self.comments
        if path=='/pulls/49':
            self.pr_reads+=1
            p=copy.deepcopy(self.pr)
            if self.drift and self.pr_reads>1:
                p['head']['sha']='d'*40
            if self.mergeable_drift and self.pr_reads>1:
                p['mergeable']=False
            return p
        raise AssertionError('Unexpected GET '+path)

class Screening(unittest.TestCase):
    def setUp(self):
        self.audit=patch('exp053_verify_live_enforcement.Audit')
        self.ci=patch('exp053_merge_ref_ci_audit.audit_pr')
        self.audit_cls=self.audit.start()
        self.ci_fun=self.ci.start()
        self.audit_cls.return_value.run.return_value={
            'governance_status':'POLICY_METADATA_VERIFIED',
            'branches':[{'branch':'main','protected':True,'required_rule_set_observed':True}]}
        self.ci_fun.return_value={
            'status':'CI_SELECTION_VERIFIED_HEAD','pr':49,
            'head_sha':SHA,'base_sha':BASE,
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
        with self.assertRaisesRegex(ChallengeError,'empty submitted review history'):
            screen(Fake(reviews=[{'state':'APPROVED'}]),49,'main')
    def test_changes_requested_review_not_suitable_for_isolated_denial(self):
        with self.assertRaisesRegex(ChallengeError, 'empty submitted review history'):
            screen(Fake(reviews=[{'state':'CHANGES_REQUESTED'}]),49,'main')

    def test_commented_review_not_suitable_for_isolated_denial(self):
        with self.assertRaisesRegex(ChallengeError, 'empty submitted review history'):
            screen(Fake(reviews=[{'state':'COMMENTED'}]),49,'main')

    def test_inline_review_comment_not_suitable_for_isolated_denial(self):
        with self.assertRaisesRegex(ChallengeError, 'zero inline review comments'):
            screen(Fake(comments=[{'id':7, 'body':'unresolved discussion'}]),49,'main')

    def test_inline_comment_added_during_audit_is_rejected(self):
        with self.assertRaisesRegex(ChallengeError, 'Inline review comments changed'):
            screen(Fake(comment_drift=True),49,'main')

    def test_clean_challenge_checks_reviews_and_inline_comments_twice(self):
        fake=Fake()
        result=screen(fake,49,'main')
        self.assertEqual(fake.review_reads,2)
        self.assertEqual(fake.comment_reads,2)
        self.assertEqual(result['inline_review_comments_observed'],0)

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
        with self.assertRaisesRegex(ChallengeError,'not individually protected'):
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

    def test_review_added_mid_audit_blocks_candidate(self):
        with self.assertRaisesRegex(ChallengeError,'Review history changed'):
            screen(Fake(review_drift=True),49,'main')

    def test_active_policy_changed_mid_audit_blocks_candidate(self):
        with self.assertRaisesRegex(ChallengeError,'Active policy changed'):
            screen(Fake(policy_drift=True),49,'main')

    def test_policy_inventory_changed_mid_audit_blocks_candidate(self):
        with self.assertRaisesRegex(ChallengeError,'Ruleset inventory changed'):
            screen(Fake(list_drift=True),49,'main')

    def test_mergeability_change_mid_audit_blocks_candidate(self):
        with self.assertRaisesRegex(ChallengeError,'identity/state changed'):
            screen(Fake(mergeable_drift=True),49,'main')

    def test_stale_ci_witness_binding_blocks_candidate(self):
        self.ci_fun.return_value['head_sha']='f'*40
        with self.assertRaisesRegex(ChallengeError,'different PR revision'):
            screen(Fake(),49,'main')

    def test_auto_merge_enabled_blocks_candidate(self):
        p=copy.deepcopy(PR)
        p['auto_merge']={'enabled_by':{'login':'somebody'}}
        with self.assertRaisesRegex(ChallengeError,'conflict-free'):
            screen(Fake(pr=p),49,'main')

    def test_success_rechecks_policy_and_reviews(self):
        fake=Fake()
        screen(fake,49,'main')
        self.assertEqual(fake.review_reads,2)
        self.assertEqual(fake.policy_reads,2)
        self.assertEqual(fake.list_reads,2)

    def test_program_code_change_rejected(self):
        f=copy.deepcopy(DOC)
        f['filename']='src/unsafe.py'
        with self.assertRaisesRegex(ChallengeError,'file scope'):
            screen(Fake(files=[f]),49,'main')

    def test_multiple_changed_files_rejected(self):
        with self.assertRaisesRegex(ChallengeError,'exactly one'):
            screen(Fake(files=[DOC,DOC]),49,'main')

    def test_existing_document_modified_not_disposable(self):
        f=copy.deepcopy(DOC)
        f['status']='modified'
        with self.assertRaisesRegex(ChallengeError,'file scope'):
            screen(Fake(files=[f]),49,'main')

    def test_oversized_fixture_rejected(self):
        f=copy.deepcopy(DOC)
        f['additions']=24
        f['changes']=24
        with self.assertRaisesRegex(ChallengeError,'file scope'):
            screen(Fake(files=[f]),49,'main')

    def test_binary_or_missing_patch_rejected(self):
        f=copy.deepcopy(DOC)
        f.pop('patch')
        with self.assertRaisesRegex(ChallengeError,'file scope'):
            screen(Fake(files=[f]),49,'main')

    def test_marker_missing_rejected(self):
        f=copy.deepcopy(DOC)
        f['patch']='@@ -0,0 +1 @@\\n+Unrelated notes'
        with self.assertRaisesRegex(ChallengeError,'file scope'):
            screen(Fake(files=[f]),49,'main')

    def test_changed_file_roster_rejected(self):
        with self.assertRaisesRegex(ChallengeError,'file inventory changed'):
            screen(Fake(file_drift=True),49,'main')

    def test_success_reads_fixture_twice(self):
        f=Fake()
        screen(f,49,'main')
        self.assertEqual(f.file_reads,2)

    def test_target_not_in_effective_audit_is_rejected(self):
        self.audit_cls.return_value.run.return_value['branches']=[]
        with self.assertRaisesRegex(ChallengeError,'not individually protected'):
            screen(Fake(),49,'main')

    def test_unprotected_target_is_rejected(self):
        with self.assertRaisesRegex(ChallengeError,'identity/protection'):
            screen(Fake(target_unprotected=True),49,'main')

    def test_pr_base_must_match_live_protected_branch_head(self):
        with self.assertRaisesRegex(ChallengeError,'differs from current'):
            screen(Fake(target_wrong_sha=True),49,'main')

    def test_protected_target_moving_mid_audit_is_rejected(self):
        with self.assertRaisesRegex(ChallengeError,'branch changed'):
            screen(Fake(target_drift=True),49,'main')

    def test_effective_audit_must_report_actual_protection(self):
        self.audit_cls.return_value.run.return_value['branches']=[
            {'branch':'main','protected':False,'required_rule_set_observed':True}]
        with self.assertRaisesRegex(ChallengeError,'not individually protected'):
            screen(Fake(),49,'main')

    def test_success_reads_target_head_twice(self):
        f=Fake()
        screen(f,49,'main')
        self.assertEqual(f.target_reads,2)

if __name__=='__main__':
    unittest.main()
