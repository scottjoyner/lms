#!/usr/bin/env python3
"""Read-only, conservative GitHub PR check-selection witness for EXP-053.

GitHub documents that when the generated test-merge commit has no status/checks,
PR HEAD checks can satisfy a required check. This witness selects the check
source accordingly, but it does NOT prove strict branch freshness, GitHub's
merge-box eligibility, reviewer approvals, governance enforcement, or science GO.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from exp053_pr_head_ci_audit import GithubReadOnly, SnapshotError, EVIDENCE, verified_checks

NUMBERS = tuple(range(37, 47))


def read_statuses(source, sha):
    response = source.get(f'/commits/{sha}/status')
    if not isinstance(response, dict) or not isinstance(response.get('statuses'), list):
        raise SnapshotError('malformed GitHub combined-commit-status response')
    return response['statuses']


def check_legacy_collisions(statuses, contexts):
    """If the same name is a legacy status AND a check run, both must pass."""
    per_name = {context: [] for context in contexts}
    for status in statuses:
        if isinstance(status, dict) and status.get('context') in per_name:
            per_name[status['context']].append(status)
    failing = []
    for name, entries in per_name.items():
        if not entries:
            continue
        # GitHub's combined-status endpoint sorts newest first but we do not
        # assume ordering. Unorderable duplicates fail closed.
        ranked = []
        for entry in entries:
            when = entry.get('created_at')
            ident = entry.get('id')
            if not isinstance(when, str) or type(ident) is not int:
                ranked = []
                break
            try:
                stamp = datetime.fromisoformat(when.replace('Z', '+00:00'))
                if stamp.tzinfo is None:
                    raise ValueError('timezone missing')
            except ValueError:
                ranked = []
                break
            ranked.append((stamp, ident, entry))
        if not ranked or max(ranked, key=lambda row: (row[0], row[1]))[2].get('state') != 'success':
            failing.append(name)
    return failing


def audit_pr(source, number, contexts, *, allow_isolated_challenge=False):
    if number not in NUMBERS and not (allow_isolated_challenge and type(number) is int and number > 0):
        raise SnapshotError('PR not in allowlisted EXP-053 review stack')
    before = source.get(f'/pulls/{number}')
    if not isinstance(before, dict):
        raise SnapshotError('invalid PR metadata')
    head = (before.get('head') or {}).get('sha')
    base = (before.get('base') or {}).get('sha')
    candidate = before.get('merge_commit_sha')
    if not all(isinstance(s, str) and len(s) == 40 and
               all(c in '0123456789abcdef' for c in s) for s in (head, base, candidate)):
        return {'pr': number, 'status': 'CI_SELECTION_UNVERIFIED', 'reason': 'MISSING_SHA',
                'scientific_status': 'NO_GO_RESEARCH_ONLY'}
    parents_obj = source.get(f'/commits/{candidate}')
    if not isinstance(parents_obj, dict) or not isinstance(parents_obj.get('parents'), list):
        raise SnapshotError('malformed merge-commit response')
    parents = [item.get('sha') for item in parents_obj['parents']]
    parents_match = parents == [base, head]
    merge_checks = source.all_checks(candidate)
    merge_statuses = read_statuses(source, candidate)
    # Docs: when test merge commit has NO status, use current head commit.
    # Any merge activity switches selection to merge candidate; this is
    # conservative for pending or unrelated merge checks and does not claim
    # GitHub's final merge-eligibility decision.
    merged_source_has_activity = bool(merge_checks or merge_statuses)
    evaluated_sha = candidate if merged_source_has_activity else head
    source_type = 'TEST_MERGE_COMMIT' if merged_source_has_activity else 'HEAD_FALLBACK_NO_MERGE_STATUS'
    evaluated_checks = merge_checks if merged_source_has_activity else source.all_checks(head)
    evaluated_statuses = merge_statuses if merged_source_has_activity else read_statuses(source, head)
    checks = verified_checks(evaluated_checks, evaluated_sha, contexts)
    colliding_statuses_not_successful = check_legacy_collisions(evaluated_statuses, contexts)
    after = source.get(f'/pulls/{number}')
    unchanged = (after.get('merge_commit_sha') == candidate and
                 (after.get('head') or {}).get('sha') == head and
                 (after.get('base') or {}).get('sha') == base)
    passing = sum(bool(item.get('passed')) for item in checks.values())
    observed = sum(bool(item.get('present_on_exact_sha')) for item in checks.values())
    okay = (parents_match and unchanged and passing == len(contexts) and
            not colliding_statuses_not_successful)
    return {
        'pr': number, 'head_sha': head, 'base_sha': base,
        'generated_merge_sha': candidate, 'parents_match_current_base_and_head': parents_match,
        'snapshot_stable_during_read': unchanged,
        'merge_commit_check_run_count': len(merge_checks),
        'merge_commit_legacy_status_count': len(merge_statuses),
        'required_check_contexts': len(contexts),
        'checks_selected_from': source_type,
        'selected_commit_sha': evaluated_sha,
        'observed_required_checks_on_selected_commit': observed,
        'passing_required_checks_on_selected_commit': passing,
        'colliding_legacy_statuses_not_successful': colliding_statuses_not_successful,
        'selected_check_details': checks,
        'status': ('CI_SELECTION_VERIFIED_HEAD' if okay and not merged_source_has_activity else
                   'CI_SELECTION_VERIFIED_MERGE' if okay else 'CI_SELECTION_UNVERIFIED'),
        'scientific_status': 'NO_GO_RESEARCH_ONLY',
        'claim_limit': ('A CI selection witness is not proof of strict-base freshness, '\
                        'review approvals, GitHub merge eligibility, governance enforcement or scientific authorization.'),
    }


def run(source=None, numbers=(46,)):
    source = source or GithubReadOnly()
    evidence = json.loads(EVIDENCE.read_text())
    contexts = evidence['verified_successful_check_contexts']
    if len(contexts) != len(set(contexts)) or len(contexts) != 5:
        raise SnapshotError('unexpected required-check baseline')
    prs = [audit_pr(source, number, contexts) for number in numbers]
    return {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'repository': 'scottjoyner/lms', 'scientific_status': 'NO_GO_RESEARCH_ONLY',
            'governance_status': 'NO_GO_GOVERNANCE', 'merges_performed': 0, 'evidence': prs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Audit all ten PRs (default: PR 46)')
    parser.add_argument('--json-out', type=Path)
    options = parser.parse_args()
    try:
        report = run(numbers=NUMBERS if options.all else (46,))
    except (SnapshotError, KeyError, TypeError, ValueError) as exc:
        report = {'governance_status': 'NO_GO_GOVERNANCE',
                  'scientific_status': 'NO_GO_RESEARCH_ONLY', 'error': type(exc).__name__}
        print(json.dumps(report, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    if options.json_out:
        options.json_out.write_text(json.dumps(report, indent=2)+'\n')
    return 0 if all(e['status'].startswith('CI_SELECTION_VERIFIED') for e in report['evidence']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
