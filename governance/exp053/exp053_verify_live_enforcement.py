#!/usr/bin/env python3
"""Read-only EXP-053 GitHub ruleset enforcement audit.

Never calls a GitHub write endpoint. Public API mode may be used anonymously;
admin review is required to establish bypass actors (redacted to anonymous readers).
The output never promotes EXP-053 science beyond NO_GO_RESEARCH_ONLY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = 'scottjoyner/lms'
ROOT = f'https://api.github.com/repos/{REPO}'
BASE = Path(__file__).resolve().parent
CANDIDATE = BASE / 'EXP053_github_ruleset_candidate_2026-10-20.json'
EVIDENCE = BASE / 'EXP053_ruleset_validation_evidence_2026-10-20.json'
API_VERSION = '2022-11-28'
CANDIDATE_SHA256 = '461c6bd54985f4c1b0523788d3dd56216ff061b8f27a9c5136f561d781012efd'
EVIDENCE_SHA256 = 'dfc303135946a57064597b9097c80cdcbdb471959e6afd61d92bf45113a6a599'
EXPECTED_CONTEXTS = (
    'lint-and-test (3.12)', 'focused-tests (3.12)',
    'qualification-tests (3.12)', 'operational-failure-injection (3.12)',
    'prompt-cache-tests (3.10)',
)

class ApiError(Exception):
    pass

class Audit:
    def __init__(self, transport=None):
        self.transport = transport or self._fetch_https
        self.events = []
        self.branch_results = []
        self.checks = []

    def get(self, path):
        # The transport accepts only repository-relative paths. Never use user-supplied URLs.
        if not path.startswith('/') or '..' in path or '://' in path:
            raise ApiError('unsafe API path')
        return self.transport(path)

    @staticmethod
    def _fetch_https(path):
        req = Request(ROOT + path, headers={
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': API_VERSION,
            'User-Agent': 'EXP053-governance-readonly-verifier',
        })
        # A token is OPTIONAL; never store, log, or include it in results.
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            req.add_header('Authorization', 'Bearer ' + token)
        try:
            with urlopen(req, timeout=15) as response:
                if response.status != 200:
                    raise ApiError(f'GET returned HTTP {response.status}')
                payload = response.read(8_000_001)
                if len(payload) > 8_000_000:
                    raise ApiError('oversized GitHub API response')
                return json.loads(payload)
        except HTTPError as exc:
            # Anonymous GitHub quota exhaustion is *missing evidence*, not
            # proof that governance reads require additional repository access.
            # The API's diagnostic body could contain a requester IP or other
            # sensitive details; never echo it or unvalidated header values.
            headers = exc.headers
            remaining = headers.get('X-RateLimit-Remaining') if headers else None
            reset = headers.get('X-RateLimit-Reset') if headers else None
            if exc.code == 429 or (exc.code == 403 and remaining == '0'):
                reset = (reset if isinstance(reset, str) and reset.isdecimal()
                         and len(reset) <= 12 else 'unknown')
                raise ApiError(
                    f'RATE_LIMITED_READ_ONLY: GitHub GET quota exhausted; '
                    f'reset_epoch={reset}; no write attempted') from None
            raise ApiError(f'GET returned HTTP {exc.code}') from None
        except (URLError, TimeoutError, ValueError) as exc:
            raise ApiError(f'GET failed: {type(exc).__name__}') from None

    def paginated(self, path, maximum_pages=10):
        output = []
        for page in range(1, maximum_pages+1):
            suffix = '?' if '?' not in path else '&'
            batch = self.get(f'{path}{suffix}per_page=100&page={page}')
            if not isinstance(batch, list):
                raise ApiError(f'Expected list from {path}')
            output.extend(batch)
            if len(batch) < 100:
                return output
        raise ApiError(f'Pagination exceeds {maximum_pages} pages: {path}')

    def record(self, ok, code, message):
        self.events.append({'ok': bool(ok), 'code': code, 'message': message})
        return bool(ok)

    @staticmethod
    def ruletypes(rules):
        out = {}
        for rule in rules:
            if not isinstance(rule, dict) or 'type' not in rule:
                continue
            out.setdefault(rule['type'], []).append(rule)
        return out

    def policy_matches(self, rules, candidate, prefix):
        """Verify one effective rules collection contains all critical controls."""
        found = self.ruletypes(rules)
        # A second rule of the same type can add an unreviewed merge blocker
        # while the first rule still satisfies a naive any() search. Require
        # the entire approved rule-type set, including exact cardinality.
        expected_types = {r['type'] for r in candidate['rules']}
        shape_ok = (isinstance(rules, list) and
                    len(rules) == len(expected_types) and
                    set(found) == expected_types and
                    all(len(found[typ]) == 1 for typ in expected_types))
        self.record(shape_ok, prefix+'.types',
                    'Exactly four approved rule types without additions or duplicates')
        pr_rules = found.get('pull_request', [])
        pr_ok = any(
            r.get('parameters', {}).get('required_approving_review_count', 0) >= 2
            and all(r.get('parameters', {}).get(x) is True for x in (
                'dismiss_stale_reviews_on_push', 'require_last_push_approval',
                'required_review_thread_resolution'))
            for r in pr_rules
        )
        self.record(pr_ok, prefix+'.reviews', 'Two approvals; stale, last-push and thread guards')
        expected = {r['context']: r['integration_id'] for r in candidate['rules'][1]['parameters']['required_status_checks']}
        contexts_ok = False
        for r in (found.get('required_status_checks', []) if shape_ok else []):
            params = r.get('parameters') or {}
            checks = params.get('required_status_checks') or []
            mapping = {c.get('context'): c.get('integration_id') for c in checks if isinstance(c, dict)}
            # Exact identities and cardinality are authorization-critical. A
            # *sixth* required job (or a duplicate) can deadlock legitimate
            # merges even though the five pinned checks all still pass.
            # Check order is not semantically meaningful after GitHub readback.
            distinct = len(mapping) == len(checks)
            if (distinct and len(checks) == len(expected) and
                    params.get('strict_required_status_checks_policy') is True
                    and mapping == expected):
                contexts_ok = True
                break
        self.record(contexts_ok, prefix+'.checks',
                    'Exactly five approved GitHub Actions contexts pinned to app 15368 and strict')
        for typ in ('non_fast_forward', 'deletion'):
            self.record(bool(found.get(typ)), prefix+'.'+typ, f'{typ} protection active')
        return shape_ok and pr_ok and contexts_ok and all(
            found.get(t) for t in ('non_fast_forward', 'deletion'))

    def run(self):
        raw_candidate = CANDIDATE.read_bytes()
        if hashlib.sha256(raw_candidate).hexdigest() != CANDIDATE_SHA256:
            raise ApiError('Local candidate changed since authorized handoff')
        candidate = json.loads(raw_candidate)
        raw_evidence = EVIDENCE.read_bytes()
        if hashlib.sha256(raw_evidence).hexdigest() != EVIDENCE_SHA256:
            raise ApiError('Pinned historical branch/CI evidence changed')
        evidence = json.loads(raw_evidence)
        self.record(evidence.get('baseline', {}).get('scientific_authorization') == 'NO_GO_RESEARCH_ONLY',
                    'science_hold', 'No scientific authorization or model inference')
        checks = (candidate['rules'][1]['parameters']['required_status_checks'])
        fixed_baseline = (
            candidate.get('target') == 'branch' and candidate.get('enforcement') == 'active'
            and candidate.get('bypass_actors') == []
            and candidate.get('conditions', {}).get('ref_name', {}).get('include')
                == ['refs/heads/main', 'refs/heads/research/exp053*']
            and candidate.get('conditions', {}).get('ref_name', {}).get('exclude') == []
            and tuple(c.get('context') for c in checks) == EXPECTED_CONTEXTS
            and all(c.get('integration_id') == 15368 for c in checks)
            and tuple(evidence.get('verified_successful_check_contexts', ())) == EXPECTED_CONTEXTS
        )
        self.record(fixed_baseline, 'candidate_integrity',
                    'Pinned scope, five check identities and no bypass in local proposal')
        summaries = self.paginated('/rulesets?includes_parents=true')
        self.record(bool(summaries), 'rulesets_present', f'{len(summaries)} repository/parent rulesets returned')

        candidates = []
        for row in summaries:
            if row.get('enforcement') != 'active' or row.get('target') != 'branch':
                continue
            ruleset_id = row.get('id')
            if type(ruleset_id) is not int:
                continue
            details = self.get(f'/rulesets/{ruleset_id}')
            if not isinstance(details, dict):
                continue
            if details.get('name') == candidate['name']:
                candidates.append(details)
        self.record(len(candidates) == 1, 'named_active_ruleset',
                    f'{len(candidates)} active rulesets with exact candidate name')
        policy = candidates[0] if len(candidates) == 1 else None
        policy_id = policy.get('id') if policy else None
        if policy:
            self.record(type(policy_id) is int and policy_id > 0 and policy.get('enforcement') == 'active' and policy.get('target') == 'branch',
                        'policy_identity', 'Pinned repository ruleset has a stable ID, active enforcement and branch target')
            self.record(policy.get('source_type') == 'Repository' and policy.get('source') == REPO,
                        'ruleset_origin', 'Ruleset belongs to expected repository')
            refs = (policy.get('conditions') or {}).get('ref_name') or {}
            target_ok = refs.get('include') == candidate['conditions']['ref_name']['include'] and refs.get('exclude') == []
            self.record(target_ok, 'branch_scope', 'Main and research/exp053* targeted, no exclusions')
            # The GitHub API redacts bypass_actors for callers without write access.
            # Missing property is UNKNOWN, never treat it as [].
            bypass = policy.get('bypass_actors', None)
            self.record(bypass == [], 'bypass_visibility',
                        'No explicit bypass actors' if bypass == [] else 'Bypass actors absent/redacted or nonempty: admin must verify')
            self.policy_matches(policy.get('rules') or [], candidate, 'installed_policy')
        else:
            self.record(False, 'ruleset_origin', 'No authenticated repository policy identity')
            self.record(False, 'branch_scope', 'No exact active ruleset available')
            self.record(False, 'bypass_visibility', 'No verified no-bypass policy')
            self.record(False, 'installed_policy', 'Active required rule set cannot be inspected')

        branches = self.paginated('/branches')
        names = sorted({b.get('name') for b in branches if isinstance(b.get('name'), str) and
                        (b['name'] == 'main' or b['name'].startswith('research/exp053'))})
        baseline = set(['main'] + evidence['baseline']['research_exp053_branches'])
        unexpected_missing = sorted(baseline - set(names))
        self.record(not unexpected_missing, 'baseline_target_retention',
                    'Previous EXP-053 branches remain observable' if not unexpected_missing
                    else 'Baseline branches missing: ' + ', '.join(unexpected_missing))
        if not names or 'main' not in names:
            self.record(False, 'branch_inventory', 'Missing main from branch listing')
        elif any('/' in name[len('research/exp053'):] for name in names if name != 'main'):
            self.record(False, 'branch_inventory', 'Nested exp053 branches not covered by candidate glob')
        else:
            self.record(True, 'branch_inventory', f'{len(names)} matching existing branches inspected')
        for name in names:
            encoded = quote(name, safe='')
            info = self.get('/branches/' + encoded)
            effective = self.paginated('/rules/branches/' + encoded)
            protected = info.get('protected') is True
            # GitHub returns the origin of each effective rule. A union of rules
            # from unrelated rulesets must never masquerade as our selected policy.
            owned = [rule for rule in effective if isinstance(rule, dict) and
                     rule.get('ruleset_id') == policy_id and
                     rule.get('ruleset_source_type') == 'Repository' and
                     rule.get('ruleset_source') == REPO]
            self.record(bool(owned), 'effective_origin:'+name,
                        f'{len(owned)} rules trace to the exact pinned repository ruleset')
            result = self.policy_matches(owned, candidate, 'effective:'+name)
            self.record(protected, 'protected:'+name, 'GitHub branch protected flag')
            self.branch_results.append({'branch': name, 'protected': protected,
                                        'effective_rules': sorted(self.ruletypes(effective)),
                                        'required_rule_set_observed': bool(result)})
        self.record(len(names) >= 12, 'scope_regression', f'{len(names)} eligible branches, baseline was 12')
        # A policy may change during multi-branch observation. Fail closed if
        # its bytes are different at the end; otherwise the snapshot is mixed.
        if policy is not None and type(policy_id) is int and policy_id > 0:
            end_policy = self.get(f'/rulesets/{policy_id}')
            stable = (isinstance(end_policy, dict) and
                      json.dumps(policy, sort_keys=True) == json.dumps(end_policy, sort_keys=True))
            self.record(stable, 'policy_stable_across_snapshot',
                        'Repository ruleset unchanged throughout the effective-branch read')
        # Re-enumerate targets after individual branch reads. A branch created or
        # deleted during the scan must never inherit a claim based on an older
        # inventory (the policy re-read above detects ruleset drift separately).
        refreshed = self.paginated('/branches')
        end_names = sorted({b.get('name') for b in refreshed if isinstance(b, dict)
                            and isinstance(b.get('name'), str)
                            and (b['name'] == 'main' or b['name'].startswith('research/exp053'))})
        self.record(names == end_names, 'branch_inventory_stable',
                    'EXP-053 target branch inventory unchanged during read-only audit'
                    if names == end_names else
                    'EXP-053 target branch inventory changed during audit; rerun from scratch')
        status = 'POLICY_METADATA_VERIFIED' if all(item['ok'] for item in self.events) else 'NO_GO_GOVERNANCE'
        return {'repository': REPO, 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                'governance_status': status, 'scientific_status': 'NO_GO_RESEARCH_ONLY',
                'enforcement_claim_limit': 'API metadata only; no negative merge challenge performed',
                'events': self.events, 'branches': self.branch_results,
                'verified_critical_checks': evidence.get('verified_successful_check_contexts', [])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json-out', type=Path, help='Write sanitized audit to local file')
    args=parser.parse_args()
    try:
        report=Audit().run()
    except (ApiError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({'governance_status':'NO_GO_GOVERNANCE',
                          'scientific_status':'NO_GO_RESEARCH_ONLY',
                          'error':str(exc)},indent=2),file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    return 0 if report['governance_status']=='POLICY_METADATA_VERIFIED' else 2

if __name__=='__main__':
    raise SystemExit(main())
