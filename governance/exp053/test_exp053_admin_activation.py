"""Activation tests simulate API mutations in memory; NEVER reach GitHub."""
import copy
import json
import unittest
import tempfile
from unittest.mock import patch
from exp053_admin_activation import (main, activate, preflight_api, verify_local,
                                      ActivationError, CONFIRMATION, REPO, BASELINE)
from pathlib import Path

EVIDENCE=json.loads(Path(BASELINE).read_text())

class FakeAdmin:
    def __init__(self, existing=False, changed=False):
        self.existing=existing
        self.changed=changed
        self.calls=[]
        self.created=None
    def call(self,method,path,payload=None):
        self.calls.append(method+' '+path)
        if method=='GET' and path.startswith('/rulesets?'):
            return [{'id':11}] if self.existing else []
        if method=='GET' and path.startswith('/branches?'):
            return [{'name':x} for x in ['main']+EVIDENCE['baseline']['research_exp053_branches']]
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
    def test_readback_wrong_origin_fails_closed_no_retry(self):
        fake=FakeAdmin(changed=True)
        with self.assertRaises(ActivationError):activate(fake,CONFIRMATION)
        self.assertEqual(sum(x.startswith('POST') for x in fake.calls),1)

if __name__=='__main__':
    unittest.main()
