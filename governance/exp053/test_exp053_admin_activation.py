"""Activation tests simulate API mutations in memory; NEVER reach GitHub."""
import copy
import json
import unittest
import tempfile
from unittest.mock import patch
from exp053_admin_activation import (main, activate, preflight_api, verify_local,
                                      ActivationError, CONFIRMATION, REPO, BASELINE,
                                      ReadOnlyGitHub)
from pathlib import Path
from hashlib import sha256
from urllib.parse import unquote
from urllib.error import HTTPError

EVIDENCE=json.loads(Path(BASELINE).read_text())
REVIEWERS=('independent-human-one', 'independent-human-two')


def synthetic_activate(api, confirmation, reviewers=REVIEWERS, *, attested=True):
    return activate(api, confirmation, reviewers, reviewer_consent_attested=attested)


class FakeAdmin:
    def __init__(self, existing=False, changed=False, missing_ci=False, wrong_app=False, moving_head=False, protected=False, duplicate_ci=False, changed_roster=False, ruleset_appears_late=False, roster_changes_prepost=False, head_moves_prepost=False, protection_appears_prepost=False, readback_mutator=None, reviewer_permission='write', reviewer_drift=False, reviewer_identity=None, reviewer_type='User'):
        self.existing=existing
        self.changed=changed
        self.calls=[]
        self.created=None
        self.missing_ci=missing_ci
        self.wrong_app=wrong_app
        self.moving_head=moving_head
        self.protected=protected
        self.duplicate_ci=duplicate_ci
        self.branch_reads={}
        self.changed_roster=changed_roster
        self.branch_listings=0
        self.ruleset_reads=0
        self.ruleset_appears_late=ruleset_appears_late
        self.roster_changes_prepost=roster_changes_prepost
        self.head_moves_prepost=head_moves_prepost
        self.protection_appears_prepost=protection_appears_prepost
        self.readback_mutator=readback_mutator
        self.reviewer_permission=reviewer_permission
        self.reviewer_drift=reviewer_drift
        self.reviewer_identity=reviewer_identity
        self.reviewer_type=reviewer_type
        self.reviewer_reads={}
    def call(self,method,path,payload=None):
        self.calls.append(method+' '+path)
        if method=='GET' and path.startswith('/collaborators/') and path.endswith('/permission'):
            login=unquote(path.split('/')[2])
            self.reviewer_reads[login]=self.reviewer_reads.get(login,0)+1
            permission=('read' if self.reviewer_drift and self.reviewer_reads[login]>1
                        else self.reviewer_permission)
            return {'permission':permission,
                    'user':{'login':self.reviewer_identity or login,'type':self.reviewer_type}}
        if method=='GET' and path.startswith('/rulesets?'):
            self.ruleset_reads+=1
            return [{'id':11}] if self.existing or (self.ruleset_appears_late and self.ruleset_reads>1) else []
        if method=='GET' and path.startswith('/branches?'):
            self.branch_listings+=1
            names=['main']+EVIDENCE['baseline']['research_exp053_branches']
            if ((self.changed_roster and self.branch_listings>1) or
                    (self.roster_changes_prepost and self.branch_listings>2)):
                names=names+['research/exp053-new-after-inventory']
            return [{'name':x} for x in names]
        if method=='GET' and path.startswith('/branches/'):
            name=unquote(path[len('/branches/'):])
            self.branch_reads[name]=self.branch_reads.get(name,0)+1
            head=sha256(name.encode()).hexdigest()[:40]
            if ((self.moving_head and self.branch_reads[name]>1) or
                    (self.head_moves_prepost and self.branch_reads[name]>2)):
                head='0'*40
            protected=(self.protected or
                       (self.protection_appears_prepost and self.branch_reads[name]>2))
            return {'name':name,'commit':{'sha':head},'protected':protected}
        if method=='GET' and path.startswith('/commits/') and path.endswith('/check-runs?per_page=100'):
            head=path[len('/commits/'):].split('/')[0]
            checks=[{'name':name,'status':'completed','conclusion':'success',
                     'head_sha':head,'app':{'id':1 if self.wrong_app else 15368}}
                    for name in EVIDENCE['verified_successful_check_contexts']]
            if self.missing_ci:
                checks.pop()
            if self.duplicate_ci:
                checks.append(copy.deepcopy(checks[0]))
            return {'total_count':len(checks),'check_runs':checks}
        if method=='POST' and path=='/rulesets':
            self.created=copy.deepcopy(payload)
            return {'id': 54}
        if method=='GET' and path=='/rulesets/54':
            fetched={**(self.created or {}), 'id':54, 'source_type':'Repository',
                     'source':'someone-else/lms' if self.changed else REPO}
            if self.readback_mutator is not None:
                # Allow compact in-place mutators; they return None.
                changed=self.readback_mutator(fetched)
                if changed is not None:
                    fetched=changed
            return fetched
        raise AssertionError('unexpected call '+method+' '+path)

class Tests(unittest.TestCase):
    def test_mutated_evidence_denied_before_api_calls(self):
        with tempfile.TemporaryDirectory() as d:
            altered = Path(d)/'evidence.json'
            changed = copy.deepcopy(EVIDENCE)
            changed['baseline']['research_exp053_branches'] = changed['baseline']['research_exp053_branches'][:-1]
            altered.write_text(json.dumps(changed))
            fake = FakeAdmin()
            with patch('exp053_admin_activation.BASELINE', altered):
                with self.assertRaisesRegex(ActivationError, 'Evidence SHA-256'):
                    synthetic_activate(fake, CONFIRMATION)
            self.assertEqual(fake.calls, [])

    def test_default_dry_run_refuses_misleading_reviewer_arguments(self):
        self.assertEqual(main(['--reviewer', REVIEWERS[0]]), 2)
        self.assertEqual(main(['--reviewer-consent-attested']), 2)
        self.assertEqual(main(['--confirm', CONFIRMATION]), 2)

    def test_activation_refuses_missing_reviewers_before_api_read(self):
        fake=FakeAdmin()
        with self.assertRaisesRegex(ActivationError, 'Two explicitly nominated'):
            activate(fake, CONFIRMATION, reviewer_consent_attested=True)
        self.assertEqual(fake.calls, [])

    def test_activation_refuses_unattested_human_consent_before_api_read(self):
        fake=FakeAdmin()
        with self.assertRaisesRegex(ActivationError, 'consent/custody'):
            activate(fake, CONFIRMATION, REVIEWERS)
        self.assertEqual(fake.calls, [])

    def test_activation_refuses_author_and_duplicate_accounts_before_api_read(self):
        for reviewers in [('scottjoyner', REVIEWERS[1]),
                          (REVIEWERS[0], REVIEWERS[0].upper()),
                          (REVIEWERS[0], '../unsafe'),
                          ('', REVIEWERS[1])]:
            with self.subTest(reviewers=reviewers):
                fake=FakeAdmin()
                with self.assertRaises(ActivationError):
                    synthetic_activate(fake, CONFIRMATION, reviewers)
                self.assertEqual(fake.calls, [])

    def test_readonly_commenter_cannot_satisfy_write_review_requirement(self):
        fake=FakeAdmin(reviewer_permission='read')
        with self.assertRaisesRegex(ActivationError, 'write permission'):
            synthetic_activate(fake, CONFIRMATION)
        self.assertFalse(any(x.startswith('POST') for x in fake.calls))
        self.assertFalse(any('/branches?' in x for x in fake.calls))

    def test_wrong_reviewer_identity_or_bot_is_not_accepted(self):
        for opts in ({'reviewer_identity':'unrelated'},
                     {'reviewer_type':'Bot'}):
            with self.subTest(opts=opts):
                fake=FakeAdmin(**opts)
                with self.assertRaisesRegex(ActivationError, 'identity'):
                    synthetic_activate(fake, CONFIRMATION)
                self.assertFalse(any(x.startswith('POST') for x in fake.calls))

    def test_reviewer_permission_revoked_during_ci_scan_blocks_post(self):
        fake=FakeAdmin(reviewer_drift=True)
        with self.assertRaisesRegex(ActivationError, 'write permission'):
            synthetic_activate(fake, CONFIRMATION)
        self.assertEqual(fake.reviewer_reads, {k:2 for k in REVIEWERS[:1]} | {REVIEWERS[1]:1})
        self.assertFalse(any(x.startswith('POST') for x in fake.calls))

    def test_reviewer_eligibility_rechecked_on_synthetic_success(self):
        fake=FakeAdmin()
        self.assertEqual(synthetic_activate(fake, CONFIRMATION),54)
        self.assertEqual(fake.reviewer_reads, {k:2 for k in REVIEWERS})
        self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)

    def test_cli_never_activates_with_only_reviewer_names(self):
        self.assertEqual(main(['--activate','--reviewer',REVIEWERS[0],
                               '--reviewer',REVIEWERS[1]]),2)

    def test_default_offline_dryrun(self):
        self.assertEqual(main([]),0)
    def test_activate_flag_alone_does_not_write(self):
        self.assertEqual(main(['--activate']),2)
    def test_candidate_sha_and_review_gate(self):
        r=verify_local()
        self.assertEqual(r['rules'][0]['parameters']['required_approving_review_count'],2)
    def test_preflight_existing_ruleset_refuses_post(self):
        fake=FakeAdmin(existing=True)
        with self.assertRaises(ActivationError):synthetic_activate(fake, CONFIRMATION)
        self.assertFalse(any(x.startswith('POST') for x in fake.calls))
    def test_wrong_phrase_no_api_calls(self):
        fake=FakeAdmin()
        with self.assertRaises(ActivationError):synthetic_activate(fake,'wrong')
        self.assertEqual(fake.calls,[])
    def test_new_ruleset_during_ci_scan_prevents_post(self):
        fake=FakeAdmin(ruleset_appears_late=True)
        with self.assertRaisesRegex(ActivationError, 'Rulesets now exist'):
            synthetic_activate(fake,CONFIRMATION)
        self.assertEqual(fake.ruleset_reads,2)
        self.assertFalse(any(x.startswith('POST') for x in fake.calls))

    def test_late_target_roster_change_prevents_post(self):
        fake=FakeAdmin(roster_changes_prepost=True)
        with self.assertRaisesRegex(ActivationError, 'roster changed before policy POST'):
            synthetic_activate(fake,CONFIRMATION)
        self.assertEqual(fake.branch_listings,3)
        self.assertFalse(any(x.startswith('POST') for x in fake.calls))

    def test_late_branch_head_change_blocks_policy_creation(self):
        fake=FakeAdmin(head_moves_prepost=True)
        with self.assertRaisesRegex(ActivationError,'head/protection changed before policy POST'):
            synthetic_activate(fake,CONFIRMATION)
        self.assertEqual(fake.ruleset_reads,2)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_late_branch_protection_change_blocks_policy_creation(self):
        fake=FakeAdmin(protection_appears_prepost=True)
        with self.assertRaisesRegex(ActivationError,'head/protection changed before policy POST'):
            synthetic_activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_before_create_checks_every_target_head_three_times(self):
        fake=FakeAdmin()
        self.assertEqual(synthetic_activate(fake,CONFIRMATION),54)
        self.assertEqual(len(fake.branch_reads),12)
        self.assertEqual(set(fake.branch_reads.values()),{3})
        self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)

    def test_synthetic_success_rechecks_before_single_post(self):
        fake=FakeAdmin()
        self.assertEqual(synthetic_activate(fake,CONFIRMATION),54)
        self.assertEqual(fake.ruleset_reads,2)
        self.assertEqual(fake.branch_listings,3)
        self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)

    def test_synthetic_optin_creates_exactly_once(self):
        fake=FakeAdmin()
        self.assertEqual(synthetic_activate(fake,CONFIRMATION),54)
        self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)
    def test_missing_current_check_prevents_write(self):
        fake=FakeAdmin(missing_ci=True)
        with self.assertRaisesRegex(ActivationError,'Missing or ambiguous current'):
            synthetic_activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_foreign_github_app_prevents_write(self):
        fake=FakeAdmin(wrong_app=True)
        with self.assertRaisesRegex(ActivationError,'source-bound'):
            synthetic_activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_moving_branch_head_prevents_write(self):
        fake=FakeAdmin(moving_head=True)
        with self.assertRaisesRegex(ActivationError,'moved'):
            synthetic_activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_existing_protection_needs_re_review(self):
        fake=FakeAdmin(protected=True)
        with self.assertRaisesRegex(ActivationError,'identity/protection/head'):
            synthetic_activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_duplicate_check_name_prevents_write(self):
        fake=FakeAdmin(duplicate_ci=True)
        with self.assertRaisesRegex(ActivationError,'Missing or ambiguous'):
            synthetic_activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_changed_target_roster_prevents_write(self):
        fake=FakeAdmin(changed_roster=True)
        with self.assertRaisesRegex(ActivationError,'roster changed'):
            synthetic_activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_anonymous_rate_limit_is_not_permission_denial(self):
        headers={'X-RateLimit-Remaining':'0','X-RateLimit-Reset':'1792553031'}
        failure=HTTPError('https://api.github.com/repos/scottjoyner/lms/rulesets',
                          403,'rate limited',headers,None)
        with patch('exp053_admin_activation.urlopen', side_effect=failure) as network:
            with patch.dict('exp053_admin_activation.os.environ', {'GITHUB_TOKEN':''}):
                with self.assertRaisesRegex(ActivationError,
                        'RATE_LIMITED_READ_ONLY.*reset_epoch=1792553031'):
                    ReadOnlyGitHub().call('GET','/rulesets')
            network.assert_called_once()

    def test_403_without_rate_limit_is_not_misclassified(self):
        failure=HTTPError('https://api.github.com/repos/scottjoyner/lms/rulesets',
                          403,'permission denied',{'X-RateLimit-Remaining':'17'},None)
        with patch('exp053_admin_activation.urlopen', side_effect=failure) as network:
            with self.assertRaisesRegex(ActivationError,'rejected: HTTP 403'):
                ReadOnlyGitHub().call('GET','/rulesets')
            network.assert_called_once()

    def test_bad_rate_reset_header_not_leaked(self):
        failure=HTTPError('https://api.github.com/repos/scottjoyner/lms/rulesets',
                          429,'secret diagnostic',{'X-RateLimit-Remaining':'0',
                          'X-RateLimit-Reset':'SECRET_URI_AND_IP'},None)
        with patch('exp053_admin_activation.urlopen', side_effect=failure) as network:
            with self.assertRaisesRegex(ActivationError,
                    'RATE_LIMITED_READ_ONLY.*reset_epoch=unknown') as caught:
                ReadOnlyGitHub().call('GET','/rulesets')
            self.assertNotIn('SECRET_URI_AND_IP',str(caught.exception))
            network.assert_called_once()

    def test_readonly_transport_never_reaches_network_for_mutations(self):
        fake=ReadOnlyGitHub()
        with patch('exp053_admin_activation.urlopen') as network:
            for method, path, payload in [
                ('POST', '/rulesets', {}),
                ('GET', '/rulesets', {}),
                ('GET', 'https://evil.invalid/path', None),
            ]:
                with self.assertRaises(ActivationError):
                    fake.call(method, path, payload)
            network.assert_not_called()

    def test_live_preflight_cli_uses_get_only_fake(self):
        fake=FakeAdmin()
        with patch('exp053_admin_activation.ReadOnlyGitHub', return_value=fake):
            self.assertEqual(main(['--live-preflight']), 0)
        self.assertEqual(fake.branch_listings, 2)
        self.assertFalse(any(call.startswith('POST') for call in fake.calls))

    def test_live_preflight_cli_fails_closed_for_existing_ruleset(self):
        fake=FakeAdmin(existing=True)
        with patch('exp053_admin_activation.ReadOnlyGitHub', return_value=fake):
            self.assertEqual(main(['--live-preflight']), 2)
        self.assertFalse(any(call.startswith('POST') for call in fake.calls))

    def test_live_preflight_refuses_admin_confirmation(self):
        with patch('exp053_admin_activation.ReadOnlyGitHub') as live:
            self.assertEqual(main(['--live-preflight','--confirm',CONFIRMATION]), 2)
            live.assert_not_called()

    def test_created_ruleset_readback_critical_controls(self):
        fake=FakeAdmin()
        self.assertEqual(synthetic_activate(fake,CONFIRMATION),54)
        self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)
        self.assertEqual(sum(x=='GET /rulesets/54' for x in fake.calls),1)

    def test_created_ruleset_readback_denies_missing_critical_controls(self):
        scenarios = {
            'wrong_identity': lambda p:p.update(id=999),
            'inactive': lambda p:p.update(enforcement='evaluate'),
            'wrong_branch_target': lambda p:p.update(target='tag'),
            'expanded_scope': lambda p:p['conditions']['ref_name']['include'].append('refs/heads/other'),
            'deleted_main_scope': lambda p:p['conditions']['ref_name']['include'].remove('refs/heads/main'),
            'new_bypass': lambda p:p['bypass_actors'].append({'actor_id':1,'actor_type':'OrganizationAdmin','bypass_mode':'always'}),
            'missing_rules': lambda p:p.pop('rules'),
            'missing_force_push_protection': lambda p:p['rules'].__setitem__(2,{'type':'non_fast_forward_DISABLED'}),
            'duplicate_rule': lambda p:p['rules'].__setitem__(3,copy.deepcopy(p['rules'][0])),
            'weakened_two_approvals': lambda p:p['rules'][0]['parameters'].update(required_approving_review_count=1),
            'lost_stale_review_guard': lambda p:p['rules'][0]['parameters'].update(dismiss_stale_reviews_on_push=False),
            'lost_last_push_guard': lambda p:p['rules'][0]['parameters'].update(require_last_push_approval=False),
            'lost_thread_guard': lambda p:p['rules'][0]['parameters'].update(required_review_thread_resolution=False),
            'lost_creation_exception': lambda p:p['rules'][1]['parameters'].update(do_not_enforce_on_create=False),
            'relaxed_ci_freshness': lambda p:p['rules'][1]['parameters'].update(strict_required_status_checks_policy=False),
            'missing_ci_context': lambda p:p['rules'][1]['parameters']['required_status_checks'].pop(),
            'extra_ci_context': lambda p:p['rules'][1]['parameters']['required_status_checks'].append({'context':'extra','integration_id':15368}),
            'foreign_ci_app': lambda p:p['rules'][1]['parameters']['required_status_checks'][0].update(integration_id=42),
            'weak_allowed_merges': lambda p:p['rules'][0]['parameters'].update(allowed_merge_methods=['merge']),
            'missing_pull_parameters': lambda p:p['rules'][0].pop('parameters'),
        }
        for name, mutator in scenarios.items():
            with self.subTest(name=name):
                fake=FakeAdmin(readback_mutator=mutator)
                with self.assertRaisesRegex(ActivationError, 'do not retry POST'):
                    synthetic_activate(fake,CONFIRMATION)
                self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)
                self.assertEqual(sum(x=='GET /rulesets/54' for x in fake.calls),1)

    def test_readback_allows_additional_output_fields_and_reordered_checks(self):
        def normalize(p):
            p['node_id']='opaque_github_generated_id'
            p['rules'][0]['parameters']['repository_added_field']=True
            p['rules'][1]['parameters']['required_status_checks'].reverse()
        fake=FakeAdmin(readback_mutator=normalize)
        self.assertEqual(synthetic_activate(fake,CONFIRMATION),54)
        self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)

    def test_readback_wrong_origin_fails_closed_no_retry(self):
        fake=FakeAdmin(changed=True)
        with self.assertRaises(ActivationError):synthetic_activate(fake,CONFIRMATION)
        self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)

if __name__=='__main__':
    unittest.main()
