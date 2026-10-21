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

class FakeAdmin:
    def __init__(self, existing=False, changed=False, missing_ci=False, wrong_app=False, moving_head=False, protected=False, duplicate_ci=False, changed_roster=False):
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
    def call(self,method,path,payload=None):
        self.calls.append(method+' '+path)
        if method=='GET' and path.startswith('/rulesets?'):
            return [{'id':11}] if self.existing else []
        if method=='GET' and path.startswith('/branches?'):
            self.branch_listings+=1
            names=['main']+EVIDENCE['baseline']['research_exp053_branches']
            if self.changed_roster and self.branch_listings>1:
                names=names+['research/exp053-new-after-inventory']
            return [{'name':x} for x in names]
        if method=='GET' and path.startswith('/branches/'):
            name=unquote(path[len('/branches/'):])
            self.branch_reads[name]=self.branch_reads.get(name,0)+1
            head=sha256(name.encode()).hexdigest()[:40]
            if self.moving_head and self.branch_reads[name]>1:
                head='0'*40
            return {'name':name,'commit':{'sha':head},'protected':self.protected}
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
            return {**(self.created or {}), 'source_type':'Repository',
                    'source':'someone-else/lms' if self.changed else REPO}
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
                    activate(fake, CONFIRMATION)
            self.assertEqual(fake.calls, [])

    def test_default_offline_dryrun(self):
        self.assertEqual(main([]),0)
    def test_activate_flag_alone_does_not_write(self):
        self.assertEqual(main(['--activate']),2)
    def test_candidate_sha_and_review_gate(self):
        r=verify_local()
        self.assertEqual(r['rules'][0]['parameters']['required_approving_review_count'],2)
    def test_preflight_existing_ruleset_refuses_post(self):
        fake=FakeAdmin(existing=True)
        with self.assertRaises(ActivationError):activate(fake, CONFIRMATION)
        self.assertFalse(any(x.startswith('POST') for x in fake.calls))
    def test_wrong_phrase_no_api_calls(self):
        fake=FakeAdmin()
        with self.assertRaises(ActivationError):activate(fake,'wrong')
        self.assertEqual(fake.calls,[])
    def test_synthetic_optin_creates_exactly_once(self):
        fake=FakeAdmin()
        self.assertEqual(activate(fake,CONFIRMATION),54)
        self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)
    def test_missing_current_check_prevents_write(self):
        fake=FakeAdmin(missing_ci=True)
        with self.assertRaisesRegex(ActivationError,'Missing or ambiguous current'):
            activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_foreign_github_app_prevents_write(self):
        fake=FakeAdmin(wrong_app=True)
        with self.assertRaisesRegex(ActivationError,'source-bound'):
            activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_moving_branch_head_prevents_write(self):
        fake=FakeAdmin(moving_head=True)
        with self.assertRaisesRegex(ActivationError,'moved'):
            activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_existing_protection_needs_re_review(self):
        fake=FakeAdmin(protected=True)
        with self.assertRaisesRegex(ActivationError,'identity/protection/head'):
            activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_duplicate_check_name_prevents_write(self):
        fake=FakeAdmin(duplicate_ci=True)
        with self.assertRaisesRegex(ActivationError,'Missing or ambiguous'):
            activate(fake,CONFIRMATION)
        self.assertFalse(any(c.startswith('POST') for c in fake.calls))

    def test_changed_target_roster_prevents_write(self):
        fake=FakeAdmin(changed_roster=True)
        with self.assertRaisesRegex(ActivationError,'roster changed'):
            activate(fake,CONFIRMATION)
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

    def test_readback_wrong_origin_fails_closed_no_retry(self):
        fake=FakeAdmin(changed=True)
        with self.assertRaises(ActivationError):activate(fake,CONFIRMATION)
        self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)

if __name__=='__main__':
    unittest.main()
