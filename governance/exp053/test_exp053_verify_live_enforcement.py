"""Synthetic-only negative cases for EXP-053 audit; no network, no science data."""
import copy
import json
import unittest
import tempfile
from unittest.mock import patch
from urllib.error import HTTPError
from pathlib import Path
from urllib.parse import unquote
from exp053_verify_live_enforcement import Audit, ApiError, BASE

CANDIDATE=json.loads((BASE/'EXP053_github_ruleset_candidate_2026-10-20.json').read_text())
EVIDENCE=json.loads((BASE/'EXP053_ruleset_validation_evidence_2026-10-20.json').read_text())
BRANCHES=['main']+EVIDENCE['baseline']['research_exp053_branches']

class FakeServer:
    def __init__(self, active=True, bypass_visibility=True, branch_rules=True):
        self.active=active
        self.bypass_visibility=bypass_visibility
        self.branch_rules=branch_rules
        self.names=list(BRANCHES)
        self.bad_branch=None
        self.rewrite_app=False
        self.wrong_source=False
        self.duplicate_context=False
        self.extra_installed_context=False
        self.extra_effective_context=False
        self.duplicate_installed_ci_rule=False
        self.duplicate_effective_ci_rule=False
        self.installed_parameter_mutator=None
        self.effective_parameter_mutator=None
        self.reorder_contexts=False
        self.foreign_effective=False
        self.wrong_effective_id=False
        self.mutate_policy_after=False
        self.policy_reads=0
        self.branch_listings=0
        self.add_branch_on_refresh=False
        self.drop_branch_on_refresh=False

    def __call__(self,path):
        path=path.split('?')[0]
        if path=='/rulesets':
            return [{'id': 42,'name':CANDIDATE['name'],'target':'branch',
                     'enforcement':'active'}] if self.active else []
        if path=='/rulesets/42':
            self.policy_reads += 1
            data=copy.deepcopy(CANDIDATE)
            data['id']=42
            if self.mutate_policy_after and self.policy_reads > 1:
                data['enforcement']='evaluate'
            data['source_type']='Repository'
            data['source']='scottjoyner/lms'
            if self.wrong_source:
                data['source']='different-owner/different-repo'
            if not self.bypass_visibility:
                del data['bypass_actors']
            if self.extra_installed_context:
                data['rules'][1]['parameters']['required_status_checks'].append(
                    {'context':'unapproved-extra-required-job','integration_id':15368})
            if self.duplicate_installed_ci_rule:
                data['rules'].append({'type':'required_status_checks',
                                     'parameters':{'strict_required_status_checks_policy':True,
                                      'required_status_checks':[
                                          {'context':'unapproved-extra-required-job','integration_id':15368}]}})
            if self.reorder_contexts:
                data['rules'][1]['parameters']['required_status_checks'].reverse()
            if self.installed_parameter_mutator:
                self.installed_parameter_mutator(data['rules'])
            return data
        if path=='/branches':
            self.branch_listings += 1
            snapshot = list(self.names)
            if self.branch_listings >= 2 and self.add_branch_on_refresh:
                snapshot.append('research/exp053-new-unscanned-target')
            if self.branch_listings >= 2 and self.drop_branch_on_refresh:
                snapshot.pop()
            return [{'name':x} for x in snapshot]
        if path.startswith('/branches/'):
            name=unquote(path[len('/branches/'):])
            return {'name':name,'protected':self.active}
        if path.startswith('/rules/branches/'):
            name=unquote(path[len('/rules/branches/'):])
            if not (self.active and self.branch_rules):
                return []
            rules=copy.deepcopy(CANDIDATE['rules'])
            if name==self.bad_branch:
                rules=[x for x in rules if x['type']!='required_status_checks']
            if self.duplicate_context:
                rules[1]['parameters']['required_status_checks'].append(copy.deepcopy(rules[1]['parameters']['required_status_checks'][0]))
            if self.extra_effective_context:
                rules[1]['parameters']['required_status_checks'].append(
                    {'context':'unapproved-extra-required-job','integration_id':15368})
            if self.duplicate_effective_ci_rule:
                rules.append({'type':'required_status_checks',
                              'parameters':{'strict_required_status_checks_policy':True,
                               'required_status_checks':[
                                   {'context':'unapproved-extra-required-job','integration_id':15368}]}})
            if self.reorder_contexts:
                rules[1]['parameters']['required_status_checks'].reverse()
            if self.effective_parameter_mutator:
                self.effective_parameter_mutator(rules)
            if self.rewrite_app:
                rules[1]['parameters']['required_status_checks'][0]['integration_id']=1
            for rule in rules:
                rule.update(ruleset_id=99 if self.wrong_effective_id else 42,
                            ruleset_source='scottjoyner/lms', ruleset_source_type='Repository')
            if self.foreign_effective:
                for rule in rules:
                    rule['ruleset_id']=99
            return rules
        raise ApiError('unexpected fixture path '+path)

class EnforcementTests(unittest.TestCase):
    def test_mutated_branch_baseline_rejected_before_api(self):
        with tempfile.TemporaryDirectory() as d:
            altered = Path(d)/'evidence.json'
            changed = copy.deepcopy(EVIDENCE)
            changed['baseline']['research_exp053_branches'] = []
            altered.write_text(json.dumps(changed))
            called = []
            def never_get(path):
                called.append(path)
                raise AssertionError('No API calls expected for tampered evidence')
            with patch('exp053_verify_live_enforcement.EVIDENCE', altered):
                with self.assertRaisesRegex(ApiError, 'historical branch/CI evidence changed'):
                    Audit(never_get).run()
            self.assertEqual(called, [])

    def test_all_passing_is_metadata_only(self):
        report=Audit(FakeServer()).run()
        self.assertEqual(report['governance_status'],'POLICY_METADATA_VERIFIED')
        self.assertEqual(report['scientific_status'],'NO_GO_RESEARCH_ONLY')
        self.assertIn('no negative merge challenge',report['enforcement_claim_limit'])
        self.assertEqual(len(report['branches']),12)

    def test_no_rulesets_critical_hold(self):
        report=Audit(FakeServer(active=False)).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertEqual(report['branches'][0]['effective_rules'],[])

    def test_redacted_bypass_never_assumed_empty(self):
        report=Audit(FakeServer(bypass_visibility=False)).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertFalse(next(x for x in report['events'] if x['code']=='bypass_visibility')['ok'])

    def test_one_missing_branch_ci_blocks_status(self):
        server=FakeServer();server.bad_branch=BRANCHES[-1]
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertFalse(report['branches'][-1]['required_rule_set_observed'])

    def test_wrong_status_check_source_blocks(self):
        server=FakeServer();server.rewrite_app=True
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')

    def test_extra_installed_status_context_must_fail_exact_policy(self):
        server=FakeServer()
        server.extra_installed_context=True
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertFalse(next(e for e in report['events']
                              if e['code']=='installed_policy.checks')['ok'])

    def test_extra_effective_status_context_must_fail_exact_policy(self):
        server=FakeServer()
        server.extra_effective_context=True
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertFalse(next(e for e in report['events']
                              if e['code']=='effective:main.checks')['ok'])

    def test_extra_installed_status_rule_cannot_hide_unapproved_required_job(self):
        server=FakeServer()
        server.duplicate_installed_ci_rule=True
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertFalse(next(e for e in report['events']
                              if e['code']=='installed_policy.checks')['ok'])

    def test_extra_effective_status_rule_cannot_hide_unapproved_required_job(self):
        server=FakeServer()
        server.duplicate_effective_ci_rule=True
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertFalse(next(e for e in report['events']
                              if e['code']=='effective:main.checks')['ok'])

    @staticmethod
    def modified_critical_params():
        return (
            ('three_instead_of_two_approvals', 'reviews',
             lambda rules: rules[0]['parameters'].update(required_approving_review_count=3)),
            ('added_code_owner_requirement', 'reviews',
             lambda rules: rules[0]['parameters'].update(require_code_owner_review=True)),
            ('restricted_merge_methods', 'reviews',
             lambda rules: rules[0]['parameters'].update(allowed_merge_methods=['merge'])),
            ('lost_creation_exception', 'checks',
             lambda rules: rules[1]['parameters'].update(do_not_enforce_on_create=False)),
            ('missing_pinned_review_parameter', 'reviews',
             lambda rules: rules[0]['parameters'].pop('require_code_owner_review')),
        )

    def test_installed_policy_rejects_nonapproved_critical_review_parameters(self):
        for name, event, mutator in self.modified_critical_params():
            with self.subTest(name=name):
                server=FakeServer()
                server.installed_parameter_mutator=mutator
                report=Audit(server).run()
                self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
                self.assertFalse(next(x for x in report['events']
                                      if x['code']=='installed_policy.'+event)['ok'])

    def test_effective_policy_rejects_nonapproved_critical_review_parameters(self):
        for name, event, mutator in self.modified_critical_params():
            with self.subTest(name=name):
                server=FakeServer()
                server.effective_parameter_mutator=mutator
                report=Audit(server).run()
                self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
                self.assertFalse(next(x for x in report['events']
                                      if x['code']=='effective:main.'+event)['ok'])

    def test_harmless_merge_method_reordering_and_output_metadata_pass(self):
        def reorder_and_add_metadata(rules):
            rules[0]['parameters']['allowed_merge_methods'].reverse()
            rules[0]['parameters']['server_output_only']=False
            rules[1]['parameters']['server_output_only']=False
        server=FakeServer()
        server.installed_parameter_mutator=reorder_and_add_metadata
        server.effective_parameter_mutator=reorder_and_add_metadata
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'POLICY_METADATA_VERIFIED')

    def test_harmless_reordering_of_five_contexts_does_not_fail(self):
        server=FakeServer()
        server.reorder_contexts=True
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'POLICY_METADATA_VERIFIED')

    def test_nested_branch_not_covered_by_wildcard(self):
        server=FakeServer();server.names.append('research/exp053-nested/extra')
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertFalse(next(x for x in report['events'] if x['code']=='branch_inventory')['ok'])

    def test_missing_branch_protection_blocks(self):
        server=FakeServer();server.active=False
        report=Audit(server).run()
        self.assertFalse(any(x['protected'] for x in report['branches']))

    def test_foreign_policy_cannot_pass(self):
        server=FakeServer(); server.wrong_source=True
        result=Audit(server).run()
        self.assertEqual(result['governance_status'],'NO_GO_GOVERNANCE')

    def test_dropped_baseline_branch_holds(self):
        server=FakeServer(); server.names.pop()
        result=Audit(server).run()
        self.assertFalse(next(x for x in result['events'] if x['code']=='baseline_target_retention')['ok'])

    def test_duplicate_ci_context_holds(self):
        server=FakeServer(); server.duplicate_context=True
        result=Audit(server).run()
        self.assertEqual(result['governance_status'],'NO_GO_GOVERNANCE')

    def test_effective_rules_from_foreign_policy_do_not_count(self):
        server=FakeServer(); server.foreign_effective=True
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertFalse(next(e for e in report['events'] if e['code']=='effective_origin:main')['ok'])

    def test_wrong_effective_ruleset_id_holds(self):
        server=FakeServer(); server.wrong_effective_id=True
        self.assertEqual(Audit(server).run()['governance_status'],'NO_GO_GOVERNANCE')

    def test_policy_modified_during_branch_scan_holds(self):
        server=FakeServer(); server.mutate_policy_after=True
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertFalse(next(e for e in report['events'] if e['code']=='policy_stable_across_snapshot')['ok'])

    def test_new_branch_mid_scan_holds_instead_of_certifying_partial_inventory(self):
        server=FakeServer()
        server.add_branch_on_refresh=True
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertEqual(len(report['branches']),len(BRANCHES))
        self.assertFalse(next(e for e in report['events']
                              if e['code']=='branch_inventory_stable')['ok'])
        self.assertEqual(server.branch_listings,2)

    def test_deleted_branch_mid_scan_holds(self):
        server=FakeServer()
        server.drop_branch_on_refresh=True
        report=Audit(server).run()
        self.assertEqual(report['governance_status'],'NO_GO_GOVERNANCE')
        self.assertFalse(next(e for e in report['events']
                              if e['code']=='branch_inventory_stable')['ok'])

    def test_anonymous_rate_limit_is_distinct_from_permission_denial(self):
        failure = HTTPError('https://api.github.com/repos/scottjoyner/lms/rulesets',
                            403, 'quota exceeded',
                            {'X-RateLimit-Remaining':'0',
                             'X-RateLimit-Reset':'1792688541'}, None)
        with patch('exp053_verify_live_enforcement.urlopen', side_effect=failure) as network:
            with self.assertRaisesRegex(
                    ApiError, 'RATE_LIMITED_READ_ONLY.*reset_epoch=1792688541'):
                Audit._fetch_https('/rulesets?includes_parents=true&per_page=100&page=1')
            network.assert_called_once()

    def test_403_with_remaining_quota_is_permission_failure(self):
        failure = HTTPError('https://api.github.com/repos/scottjoyner/lms/rulesets',
                            403, 'forbidden', {'X-RateLimit-Remaining':'19'}, None)
        with patch('exp053_verify_live_enforcement.urlopen', side_effect=failure) as network:
            with self.assertRaisesRegex(ApiError, '^GET returned HTTP 403$'):
                Audit._fetch_https('/rulesets')
            network.assert_called_once()

    def test_429_without_rate_headers_is_explicit_limit(self):
        failure = HTTPError('https://api.github.com/repos/scottjoyner/lms/rulesets',
                            429, 'slow down', {}, None)
        with patch('exp053_verify_live_enforcement.urlopen', side_effect=failure):
            with self.assertRaisesRegex(
                    ApiError, 'RATE_LIMITED_READ_ONLY.*reset_epoch=unknown'):
                Audit._fetch_https('/rulesets')

    def test_rate_limit_response_never_leaks_untrusted_diagnostic_header(self):
        secret='SENSITIVE_REQUESTER_IP_OR_TOKEN'
        failure = HTTPError('https://api.github.com/repos/scottjoyner/lms/rulesets',
                            403, secret,
                            {'X-RateLimit-Remaining':'0',
                             'X-RateLimit-Reset':secret}, None)
        with patch('exp053_verify_live_enforcement.urlopen', side_effect=failure):
            with self.assertRaisesRegex(ApiError, 'reset_epoch=unknown') as caught:
                Audit._fetch_https('/rulesets')
            self.assertNotIn(secret, str(caught.exception))

    def test_api_errors_are_not_silenced(self):
        def fails(path): raise ApiError('simulated denied request')
        with self.assertRaises(ApiError):Audit(fails).run()

if __name__=='__main__':
    unittest.main()
