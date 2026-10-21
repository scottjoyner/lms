#!/usr/bin/env python3
"""Guarded administrator-only EXP-053 GitHub ruleset installer.

Default mode is completely offline and NEVER writes. No automatic invocation is
appropriate: activate ONLY after a separate accountable administrator chooses to
accept the policy, its reviewer/custody impact, and recovery procedure.

This script creates a repository ruleset (one POST). It NEVER edits an existing
ruleset, dismisses reviews, merges a PR, or changes the research science hold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
from urllib.error import HTTPError, URLError

BASE = Path(__file__).resolve().parent
POLICY = BASE / 'EXP053_github_ruleset_candidate_2026-10-20.json'
SHA256 = '461c6bd54985f4c1b0523788d3dd56216ff061b8f27a9c5136f561d781012efd'
REPO = 'scottjoyner/lms'
URL = f'https://api.github.com/repos/{REPO}'
CONFIRMATION = 'I-ACCEPT-EXP053-TWO-REVIEWER-ENFORCEMENT'
BASELINE = BASE / 'EXP053_ruleset_validation_evidence_2026-10-20.json'
EVIDENCE_SHA256 = 'dfc303135946a57064597b9097c80cdcbdb471959e6afd61d92bf45113a6a599'


class ActivationError(Exception):
    pass


class GitHub:
    def __init__(self, token):
        if not token:
            raise ActivationError('Administrator token is missing; no activation performed')
        self.token = token

    def call(self, method, path, payload=None):
        if method not in ('GET', 'POST') or not path.startswith('/') or '..' in path or '://' in path:
            raise ActivationError('Unsafe repository API endpoint')
        data = None if payload is None else json.dumps(payload, separators=(',', ':')).encode('utf-8')
        req = Request(URL + path, data=data, method=method, headers={
            'Accept': 'application/vnd.github+json',
            'Content-Type': 'application/json',
            'User-Agent': 'EXP053-explicit-admin-activation',
            'X-GitHub-Api-Version': '2022-11-28',
            'Authorization': 'Bearer ' + self.token,
        })
        try:
            with urlopen(req, timeout=20) as result:
                content = result.read(4_000_001)
                if len(content) > 4_000_000:
                    raise ActivationError('API response too large')
                return json.loads(content)
        except HTTPError as exc:
            # Do not print the response body or raw request/token; fail closed.
            raise ActivationError(f'GitHub {method} rejected: HTTP {exc.code}') from None
        except (URLError, TimeoutError, ValueError) as exc:
            raise ActivationError(f'GitHub {method} failed: {type(exc).__name__}') from None


class ReadOnlyGitHub:
    """Dedicated GET-only transport; does not share the POST-capable client."""
    def call(self, method, path, payload=None):
        if (method != 'GET' or payload is not None or
                not path.startswith('/') or '..' in path or '://' in path):
            raise ActivationError('Live preflight forbids writes or unsafe endpoints')
        headers = {
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'EXP053-read-only-live-preflight',
            'X-GitHub-Api-Version': '2022-11-28',
        }
        token = os.environ.get('GITHUB_TOKEN', '')
        if token:
            headers['Authorization'] = 'Bearer ' + token
        req = Request(URL + path, method='GET', headers=headers)
        try:
            with urlopen(req, timeout=20) as result:
                content = result.read(4_000_001)
                if len(content) > 4_000_000:
                    raise ActivationError('Read-only API response too large')
                return json.loads(content)
        except HTTPError as exc:
            # Rate exhaustion is missing evidence, not an admin permission denial.
            # Never leak GitHub's diagnostic response body or requester IP.
            headers = exc.headers
            remaining = headers.get('X-RateLimit-Remaining') if headers else None
            reset = headers.get('X-RateLimit-Reset') if headers else None
            if exc.code in (403, 429) and remaining == '0':
                reset = reset if isinstance(reset, str) and reset.isdecimal() and len(reset) <= 12 else 'unknown'
                raise ActivationError(
                    f'RATE_LIMITED_READ_ONLY: GitHub GET quota exhausted; reset_epoch={reset}; '
                    'no write attempted') from None
            raise ActivationError(f'Read-only GitHub GET rejected: HTTP {exc.code}') from None
        except (URLError, TimeoutError, ValueError) as exc:
            raise ActivationError(f'Read-only GitHub GET failed: {type(exc).__name__}') from None


def verified_evidence():
    # Historic branch/check evidence is authorization-critical, not an editable input.
    raw = BASELINE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EVIDENCE_SHA256:
        raise ActivationError('Evidence SHA-256 changed; administrator re-review required')
    return json.loads(raw)


def verify_local():
    data = POLICY.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ActivationError('Ruleset JSON SHA-256 has changed; administrator re-review required')
    payload = json.loads(data)
    if payload.get('target') != 'branch' or payload.get('enforcement') != 'active':
        raise ActivationError('Candidate must target active branches')
    if payload.get('bypass_actors') != []:
        raise ActivationError('Candidate has bypass actors')
    if (payload.get('conditions') or {}).get('ref_name', {}).get('include') != [
            'refs/heads/main', 'refs/heads/research/exp053*']:
        raise ActivationError('Unexpected target scope')
    if (payload.get('conditions') or {}).get('ref_name', {}).get('exclude') != []:
        raise ActivationError('Unexpected branch exclusion')
    rules = {x['type']: x for x in payload.get('rules', [])}
    if set(rules) != {'pull_request', 'required_status_checks', 'non_fast_forward', 'deletion'}:
        raise ActivationError('Incomplete required rule types')
    rp = rules['pull_request']['parameters']
    if (rp.get('required_approving_review_count') != 2 or
            not all(rp.get(k) is True for k in ('dismiss_stale_reviews_on_push',
                'require_last_push_approval','required_review_thread_resolution'))):
        raise ActivationError('Human approval requirements differ from handoff')
    status = rules['required_status_checks']['parameters']
    contexts = status['required_status_checks']
    evidence = verified_evidence()
    if not status.get('strict_required_status_checks_policy') or \
            [c.get('context') for c in contexts] != evidence['verified_successful_check_contexts'] or \
            len(contexts) != 5 or any(c.get('integration_id') != 15368 for c in contexts):
        raise ActivationError('Status check policy differs from verified handoff')
    return payload


def paginated(api, route):
    items=[]
    for p in range(1,11):
        data=api.call('GET', route + ('&' if '?' in route else '?') + f'per_page=100&page={p}')
        if not isinstance(data,list):
            raise ActivationError('Unexpected GitHub list API response')
        items.extend(data)
        if len(data)<100:
            return items
    raise ActivationError('API listing is truncated')


def preflight_api(api):
    # The owner must independently check legacy protection before enabling;
    # these API reads additionally guard against a ruleset added since handoff.
    existing = paginated(api, '/rulesets?includes_parents=true')
    if existing:
        raise ActivationError('Rulesets now exist: refuse automatic overlap or replacement')
    branches = paginated(api, '/branches')
    names = [x.get('name') for x in branches]
    expected = set(['main'] + verified_evidence()['baseline']['research_exp053_branches'])
    if not expected.issubset(set(names)):
        raise ActivationError('A baseline branch has disappeared; re-review target scope')
    from fnmatch import fnmatchcase
    for name in names:
        if name == 'main':
            pass
        elif isinstance(name, str) and name.startswith('research/exp053'):
            if not (name.startswith('research/') and '/' not in name[len('research/'):]
                    and fnmatchcase(name, 'research/exp053*')):
                raise ActivationError('Nested EXP-053 branch is not matched by proposed glob')
    return names


def verify_current_required_checks(api, branches):
    """Fail closed if current protected-target heads lack pinned, successful CI.

    Historical evidence is a signed-off baseline, not a substitute for checking
    fresh head commits. This function performs GETs only; it never authorizes
    merge or research inference.
    """
    required = tuple(verified_evidence()['verified_successful_check_contexts'])
    expected = set(required)
    if len(required) != 5 or len(expected) != 5:
        raise ActivationError('Pinned required CI context list is incomplete')
    for name in sorted(branches):
        if name != 'main' and not name.startswith('research/exp053'):
            continue
        path = '/branches/' + quote(name, safe='')
        before = api.call('GET', path)
        head = (before.get('commit') or {}).get('sha') if isinstance(before, dict) else None
        if (not isinstance(before, dict) or before.get('name') != name or
                before.get('protected') is not False or not isinstance(head, str) or
                len(head) != 40 or any(c not in '0123456789abcdef' for c in head)):
            raise ActivationError('Unexpected branch identity/protection/head; no policy write')
        result = api.call('GET', f'/commits/{head}/check-runs?per_page=100')
        if not isinstance(result, dict):
            raise ActivationError('Current check-runs API shape is unknown')
        runs = result.get('check_runs')
        count = result.get('total_count')
        if (not isinstance(runs, list) or type(count) is not int or count != len(runs)
                or count > 100):
            raise ActivationError('Current check-runs listing incomplete; no policy write')
        found = {context: [] for context in required}
        for run in runs:
            if not isinstance(run, dict):
                raise ActivationError('Invalid check run; no policy write')
            context = run.get('name')
            if context in expected:
                found[context].append(run)
        for context, matching in found.items():
            if len(matching) != 1:
                raise ActivationError('Missing or ambiguous current required CI check; no policy write')
            run = matching[0]
            if (run.get('head_sha') != head or run.get('status') != 'completed'
                    or run.get('conclusion') != 'success' or
                    (run.get('app') or {}).get('id') != 15368):
                raise ActivationError('Current required CI check is not successful and source-bound')
        after = api.call('GET', path)
        if not isinstance(after, dict) or (after.get('commit') or {}).get('sha') != head:
            raise ActivationError('Branch head moved during CI preflight; no policy write')
    # A target branch created or deleted during this multi-branch scan cannot
    # be certified by a snapshot that did not include it.
    refreshed = paginated(api, '/branches')
    def targets(items):
        return sorted(row['name'] for row in items if isinstance(row, dict)
                      and isinstance(row.get('name'), str)
                      and (row['name'] == 'main' or row['name'].startswith('research/exp053')))
    initial_targets = sorted(name for name in branches if isinstance(name, str)
                             and (name == 'main' or name.startswith('research/exp053')))
    if initial_targets != targets(refreshed):
        raise ActivationError('Target branch roster changed during CI preflight; no policy write')


def activate(api, confirmation):
    if confirmation != CONFIRMATION:
        raise ActivationError('Exact administrator confirmation phrase is required')
    payload = verify_local()
    branches = preflight_api(api)
    verify_current_required_checks(api, branches)
    # Exactly one write. If the response is uncertain do NOT retry the POST.
    created = api.call('POST', '/rulesets', payload)
    identifier = created.get('id') if isinstance(created,dict) else None
    if type(identifier) is not int or identifier < 1:
        raise ActivationError('POST response uncertain; inspect GitHub before ANY retry')
    # Read-back does not automatically authorize merge/scientific execution.
    fetched = api.call('GET', f'/rulesets/{identifier}')
    if not isinstance(fetched,dict) or fetched.get('enforcement') != 'active' or \
            fetched.get('source_type') != 'Repository' or fetched.get('source') != REPO or \
            fetched.get('name') != payload['name'] or fetched.get('bypass_actors') != []:
        raise ActivationError('Installed policy read-back incomplete; do not retry POST; inspect in GitHub')
    return identifier


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--activate', action='store_true', help='Explicitly request one admin policy creation POST')
    mode.add_argument('--live-preflight', action='store_true',
                      help='Read-only current-head CI and branch check; never activates anything')
    parser.add_argument('--confirm', default='', help='Exact accountable-administrator confirmation phrase')
    args = parser.parse_args(argv)
    try:
        payload=verify_local()
        if args.live_preflight:
            if args.confirm:
                raise ActivationError('Read-only mode refuses administrator confirmation phrase')
            api=ReadOnlyGitHub()
            branches=preflight_api(api)
            verify_current_required_checks(api, branches)
            print(json.dumps({'status':'LIVE_READ_ONLY_PREFLIGHT_VALID',
                'repository':REPO,
                'target_branches_checked':sum(1 for name in branches if name=='main' or name.startswith('research/exp053')),
                'writes_performed':0,
                'governance_status':'NO_GO_GOVERNANCE',
                'scientific_status':'NO_GO_RESEARCH_ONLY'},indent=2))
            return 0
        if not args.activate:
            print(json.dumps({'status':'OFFLINE_DRY_RUN_VALID','repository':REPO,
                'payload_sha256': SHA256, 'name':payload['name'],
                'writes_performed':0, 'scientific_status':'NO_GO_RESEARCH_ONLY'}, indent=2))
            return 0
        if args.confirm != CONFIRMATION:
            raise ActivationError('Explicit --confirm phrase missing; no write attempted')
        token=os.environ.get('GITHUB_TOKEN','')
        ident=activate(GitHub(token), args.confirm)
        print(json.dumps({'status':'RULESET_CREATED_READBACK_PARTIAL','id':ident,
                          'remaining':'effective rules, actual merge rejection, reviewers, science custody',
                          'scientific_status':'NO_GO_RESEARCH_ONLY'},indent=2))
        return 0
    except (ActivationError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({'status':'NO_GO_GOVERNANCE', 'error':str(exc),
            'scientific_status':'NO_GO_RESEARCH_ONLY'},indent=2),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
