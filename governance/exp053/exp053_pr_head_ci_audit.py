#!/usr/bin/env python3
"""Read-only source-only EXP-053 PR head + GitHub Actions check freshness auditor.

This is NOT a scientific review, a substitute for branch protection, or merge
permission. Its output always retains scientific_status=NO_GO_RESEARCH_ONLY.
Only GET requests are issued. No secrets or licensed RACE content are loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parent
EVIDENCE = BASE / 'EXP053_ruleset_validation_evidence_2026-10-20.json'
API = 'https://api.github.com/repos/scottjoyner/lms'
CHECK_APP_ID = 15368
PR_NUMBERS = tuple(range(37, 47))
EXPECTED_STACK = {37: ('research/exp053-race-human-review-gate-20261016', 'research/exp053-owner-roster-read-review-20261019'), 38: ('research/exp053-owner-roster-read-review-20261019', 'research/exp053-private-admission-source-review-20261019'), 39: ('research/exp053-private-admission-source-review-20261019', 'research/exp053-source-fd-cohort-review-20261019'), 40: ('research/exp053-source-fd-cohort-review-20261019', 'research/exp053-private-writer-ancestor-review-20261019'), 41: ('research/exp053-private-writer-ancestor-review-20261019', 'research/exp053-sealed-scope-review-20261019'), 42: ('research/exp053-sealed-scope-review-20261019', 'research/exp053-replay-pinned-receipts-review-20261019'), 43: ('research/exp053-replay-pinned-receipts-review-20261019', 'research/exp053-aggregate-fd-review-20261019'), 44: ('research/exp053-aggregate-fd-review-20261019', 'research/exp053-operator-journal-ancestors-review-20261020'), 45: ('research/exp053-operator-journal-ancestors-review-20261020', 'research/exp053-execution-evidence-ancestor-review-20261020'), 46: ('research/exp053-execution-evidence-ancestor-review-20261020', 'research/exp053-scorer-json-numeric-review-20261020')}


class SnapshotError(Exception):
    pass


class GithubReadOnly:
    def __init__(self, transport=None):
        self.transport = transport or self._get_http

    @staticmethod
    def _get_http(path):
        if not path.startswith('/') or '..' in path or '://' in path:
            raise SnapshotError('invalid API path')
        request = Request(API + path, headers={
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'EXP053-readonly-ci-audit',
        })
        # Optional; never serialize, log or print the credential.
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            request.add_header('Authorization', f'Bearer {token}')
        try:
            with urlopen(request, timeout=20) as handle:
                if handle.status != 200:
                    raise SnapshotError(f'API HTTP {handle.status}')
                data = handle.read(8_000_001)
            if len(data) > 8_000_000:
                raise SnapshotError('oversize API response')
            return json.loads(data)
        except HTTPError as exc:
            raise SnapshotError(f'API HTTP {exc.code}') from None
        except (URLError, TimeoutError, ValueError) as exc:
            raise SnapshotError(f'API error: {type(exc).__name__}') from None

    def get(self, path):
        return self.transport(path)

    def all_pages(self, path, maximum_pages=10):
        records = []
        for page in range(1, maximum_pages + 1):
            query = '&' if '?' in path else '?'
            value = self.get(f'{path}{query}per_page=100&page={page}')
            if not isinstance(value, list):
                raise SnapshotError(f'not a list from {path}')
            records.extend(value)
            if len(value) < 100:
                return records
        raise SnapshotError(f'pagination not complete for {path}')

    def all_checks(self, sha, maximum_pages=10):
        checks = []
        for page in range(1, maximum_pages + 1):
            value = self.get(f'/commits/{sha}/check-runs?filter=all&per_page=100&page={page}')
            if not isinstance(value, dict) or not isinstance(value.get('check_runs'), list):
                raise SnapshotError('malformed check-runs response')
            checks.extend(value['check_runs'])
            if len(checks) >= value.get('total_count', 0):
                return checks
            if len(value['check_runs']) < 100:
                raise SnapshotError('check-runs response truncated')
        raise SnapshotError('check-runs exceed pagination bounds')


def verified_checks(items, sha, contexts):
    """Fail closed for queued/retried check runs, even when started_at is absent.

    Do not sort by started_at: GitHub may return a *new queued* rerun whose
    started_at is null, while an older successful run has a started_at value.
    Use the creation timestamp plus immutable check-run ID. In ambiguous
    duplicate-history cases lacking timestamp or ID, do not declare success.
    """
    from datetime import datetime
    grouped = {context: [] for context in contexts}
    for check in items:
        if not isinstance(check, dict) or check.get('name') not in grouped:
            continue
        if not isinstance(check.get('app'), dict) or check['app'].get('id') != CHECK_APP_ID:
            continue
        if check.get('head_sha') != sha:
            continue
        grouped[check['name']].append(check)
    summary = {}
    for context, runs in grouped.items():
        if not runs:
            summary[context] = {'present_on_exact_sha': False, 'passed': False,
                                'conclusion': None, 'reason': 'MISSING'}
            continue
        valid_orders = []
        for item in runs:
            ident = item.get('id')
            created = item.get('created_at')
            if type(ident) is not int or ident <= 0:
                valid_orders = []
                break
            if created is None and len(runs) == 1:
                # Single result from fixture / latest-filter API: no conflict.
                valid_orders.append((("", ident), item))
                continue
            if not isinstance(created, str):
                valid_orders = []
                break
            try:
                stamp = datetime.fromisoformat(created.replace('Z', '+00:00'))
                if stamp.tzinfo is None:
                    valid_orders = []
                    break
            except ValueError:
                valid_orders = []
                break
            valid_orders.append(((stamp.isoformat(), ident), item))
        if len(valid_orders) != len(runs):
            summary[context] = {'present_on_exact_sha': True, 'passed': False,
                                'conclusion': None, 'reason': 'AMBIGUOUS_CHECK_HISTORY'}
            continue
        # Timestamp sorting wins over check-run start or completion order;
        # ID disambiguates multiple runs at the same creation instant.
        value = max(valid_orders, key=lambda p: p[0])[1]
        passed = value.get('status') == 'completed' and value.get('conclusion') == 'success'
        summary[context] = {
            'present_on_exact_sha': True,
            'passed': passed,
            'conclusion': value.get('conclusion'),
            'reason': 'PASSED' if passed else 'LATEST_NOT_SUCCESSFUL',
        }
    return summary


def run(source=None):
    source = source or GithubReadOnly()
    evidence = json.loads(EVIDENCE.read_text())
    contexts = tuple(evidence['verified_successful_check_contexts'])
    assert len(set(contexts)) == len(contexts) == 5
    expected = {item['branch_or_pr']: item['sha'] for item in evidence['check_contexts_verified_at']}
    if set(expected) != {'main', *(f'PR #{n}' for n in PR_NUMBERS)}:
        raise SnapshotError('bad baseline entries')
    findings = []
    checks_by_head = []
    main = source.get('/branches/main')
    main_sha = (main.get('commit') or {}).get('sha')
    if not isinstance(main_sha, str) or len(main_sha) != 40:
        raise SnapshotError('main SHA unavailable')
    checks_by_head.append({'target': 'main', 'sha': main_sha,
                           'baseline_sha_unchanged': main_sha == expected['main'],
                           'contexts': verified_checks(source.all_checks(main_sha), main_sha, contexts)})
    for number in PR_NUMBERS:
        pr = source.get(f'/pulls/{number}')
        sha = (pr.get('head') or {}).get('sha')
        base = (pr.get('base') or {}).get('ref')
        author = (pr.get('user') or {}).get('login')
        branch = (pr.get('head') or {}).get('ref')
        if not isinstance(sha, str) or len(sha) != 40 or not isinstance(branch, str):
            raise SnapshotError(f'PR #{number} lacks head identity')
        reviews = source.all_pages(f'/pulls/{number}/reviews')
        # Do NOT infer valid approval from review count. Stale, dismissed,
        # latest-push and author constraints depend on live GitHub enforcement.
        chain_ok = (base, branch) == EXPECTED_STACK[number]
        summary = {'pr': number, 'stack_base_head_unchanged': chain_ok, 'state': pr.get('state'), 'draft': pr.get('draft'),
                   'merged': pr.get('merged'), 'base': base, 'head': branch,
                   'head_sha': sha, 'baseline_sha_unchanged': sha == expected[f'PR #{number}'],
                   'requested_reviewers': len(pr.get('requested_reviewers') or []),
                   'submitted_review_count': len(reviews),
                   'reported_approval_events_not_authority': sum(
                       1 for review in reviews if review.get('state') == 'APPROVED'),
                   'author': author}
        findings.append(summary)
        checks_by_head.append({'target': f'PR #{number}', 'sha': sha,
                               'baseline_sha_unchanged': summary['baseline_sha_unchanged'],
                               'contexts': verified_checks(source.all_checks(sha), sha, contexts)})
    baseline_unchanged = all(item['baseline_sha_unchanged'] for item in checks_by_head)
    all_checks_pass = all(all(v['passed'] for v in item['contexts'].values()) for item in checks_by_head)
    chain_unchanged = all(item['stack_base_head_unchanged'] for item in findings)
    remain_draft = all(item['state'] == 'open' and item['draft'] is True and item['merged'] is False for item in findings)
    # This is a read-only research snapshot, not approval for merging.
    status = 'HEAD_CI_SNAPSHOT_VERIFIED' if baseline_unchanged and all_checks_pass and remain_draft and chain_unchanged else 'HOLD_REFRESH_OR_REVIEW'
    return {'repository': 'scottjoyner/lms', 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'snapshot_status': status, 'scientific_status': 'NO_GO_RESEARCH_ONLY',
            'authorization': 'NONE; reviews, ruleset activation and science are separate gates',
            'baseline_head_shas_unchanged': baseline_unchanged,
            'all_required_checks_succeeded_on_current_heads': all_checks_pass,
            'all_sensitive_prs_remain_draft': remain_draft,
            'stack_base_head_ancestry_unchanged': chain_unchanged,
            'prs': findings, 'checks_by_head': checks_by_head}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--json-out', type=Path)
    opts = p.parse_args()
    try:
        report = run()
    except (SnapshotError, TypeError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({'snapshot_status': 'HOLD_ERROR', 'scientific_status': 'NO_GO_RESEARCH_ONLY',
                          'error': str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if opts.json_out:
        opts.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return 0 if report['snapshot_status'] == 'HEAD_CI_SNAPSHOT_VERIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
