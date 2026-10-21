#!/usr/bin/env python3
"""Read-only candidate screening for a HUMAN negative merge-denial challenge.

No REST merge mutation is ever issued. Passing this screen proves ONLY that a
non-draft PR is suitable for an independent human to inspect in GitHub's merge
box after governance enforcement is activated. No inference authorization.
"""
from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import quote
from exp053_admin_activation import (ReadOnlyGitHub, ActivationError,
    paginated, verified_evidence, REPO)

class ChallengeError(Exception):
    pass

def screen(api, number, expected_base):
    if type(number) is not int or number < 1:
        raise ChallengeError('Invalid pull request number')
    if expected_base != 'main' and not (expected_base.startswith('research/exp053')
            and '/' not in expected_base[len('research/'):]):
        raise ChallengeError('Challenge target must be main or one flat EXP-053 branch')
    if number in range(37, 49):
        raise ChallengeError('Existing EXP-053 PRs cannot be disposable challenge probes')
    found = paginated(api, '/rulesets?includes_parents=true')
    candidate = [r for r in found if isinstance(r,dict) and
                 r.get('name') == 'EXP-053 require independent human review and verified CI'
                 and r.get('enforcement') == 'active' and r.get('target') == 'branch']
    if len(candidate) != 1:
        raise ChallengeError('Named active ruleset is not present exactly once')
    from exp053_verify_live_enforcement import Audit
    from exp053_merge_ref_ci_audit import audit_pr
    from exp053_pr_head_ci_audit import GithubReadOnly
    state = Audit(transport=lambda route: api.call('GET', route)).run()
    if state['governance_status'] != 'POLICY_METADATA_VERIFIED':
        raise ChallengeError('Active effective governance rules not verified')
    before = api.call('GET', f'/pulls/{number}')
    if not isinstance(before,dict) or before.get('number') != number:
        raise ChallengeError('Malformed challenge PR response')
    head = before.get('head') or {}
    base = before.get('base') or {}
    if (before.get('state') != 'open' or before.get('draft') is not False
            or before.get('merged_at') is not None or before.get('mergeable') is not True):
        raise ChallengeError('PR must be open, non-draft, unmerged, conflict-free')
    if (base.get('ref') != expected_base or
            (base.get('repo') or {}).get('full_name') != REPO or
            (head.get('repo') or {}).get('full_name') != REPO):
        raise ChallengeError('Unexpected target or repository identity')
    reviews = paginated(api, f'/pulls/{number}/reviews')
    if any(not isinstance(r,dict) for r in reviews):
        raise ChallengeError('Unrecognized review history')
    if any(r.get('state') == 'APPROVED' for r in reviews):
        raise ChallengeError('Already approved PR cannot establish missing-review denial')
    contexts=verified_evidence()['verified_successful_check_contexts']
    ci=audit_pr(GithubReadOnly(lambda route: api.call('GET',route)), number, contexts,
                allow_isolated_challenge=True)
    if not ci['status'].startswith('CI_SELECTION_VERIFIED_'):
        raise ChallengeError('Head/test-merge CI selection cannot be established')
    after=api.call('GET',f'/pulls/{number}')
    if (after.get('head') or {}).get('sha') != head.get('sha') or (
            (after.get('base') or {}).get('sha') != base.get('sha') or
            after.get('merge_commit_sha') != before.get('merge_commit_sha') or
            after.get('draft') is not False or after.get('state') != 'open'):
        raise ChallengeError('PR identity/state changed during audit')
    return {
        'status':'HUMAN_MERGE_DENIAL_INSPECTION_CANDIDATE_ONLY',
        'repository':REPO,'pr':number,'base':expected_base,
        'head_sha':head.get('sha'),'evaluated_ci_sha':ci['selected_commit_sha'],
        'ci_source':ci['checks_selected_from'],
        'reviews_observed':len(reviews),'writes_performed':0,
        'merge_denial_attested':False,'merge_action_performed':False,
        'governance_status':'NO_GO_GOVERNANCE',
        'scientific_status':'NO_GO_RESEARCH_ONLY'
    }

def main(argv=None):
    from exp053_verify_live_enforcement import ApiError
    from exp053_pr_head_ci_audit import SnapshotError
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pr',type=int,required=True)
    parser.add_argument('--expect-base',required=True)
    args=parser.parse_args(argv)
    try:
        result=screen(ReadOnlyGitHub(),args.pr,args.expect_base)
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0
    except (ChallengeError,ActivationError,ApiError,SnapshotError,
            KeyError,TypeError,ValueError) as exc:
        print(json.dumps({
            'status':'NO_GO_CHALLENGE','reason':str(exc),
            'governance_status':'NO_GO_GOVERNANCE',
            'scientific_status':'NO_GO_RESEARCH_ONLY',
            'writes_performed':0,
        },indent=2),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
