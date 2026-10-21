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
    selected_id = candidate[0].get('id')
    if type(selected_id) is not int or selected_id < 1:
        raise ChallengeError('Active policy ID missing or invalid')
    installed_before = api.call('GET', f'/rulesets/{selected_id}')
    if (not isinstance(installed_before, dict) or
            installed_before.get('id') != selected_id):
        raise ChallengeError('Active policy identity is unverified')
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
            or before.get('merged_at') is not None or before.get('mergeable') is not True
            or before.get('auto_merge') is not None):
        raise ChallengeError('PR must be open, non-draft, unmerged, conflict-free')
    if (base.get('ref') != expected_base or
            (base.get('repo') or {}).get('full_name') != REPO or
            (head.get('repo') or {}).get('full_name') != REPO):
        raise ChallengeError('Unexpected target or repository identity')
    # A challenge should never repurpose a production/security PR. Keep its
    # entire change set to one small, newly added, human-reviewed text fixture.
    # This scope gate does not determine whether the content itself is safe.
    changes = paginated(api, f'/pulls/{number}/files')
    if len(changes) != 1 or not isinstance(changes[0], dict):
        raise ChallengeError('Challenge must change exactly one documentation fixture')
    file = changes[0]
    name = file.get('filename')
    prefix = 'governance/exp053/challenges/exp053-merge-denial-'
    if (not isinstance(name, str) or not name.startswith(prefix) or
            not name.endswith('.md') or '/' in name[len(prefix):] or
            not 1 <= len(name[len(prefix):-3]) <= 40 or
            not all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name[len(prefix):-3]) or
            file.get('status') != 'added' or
            type(file.get('additions')) is not int or
            not 1 <= file['additions'] <= 15 or
            file.get('deletions') != 0 or
            type(file.get('changes')) is not int or
            file['changes'] != file['additions'] or
            not isinstance(file.get('patch'), str) or
            not 0 < len(file['patch']) <= 2000 or
            '+# EXP-053 governance challenge fixture' not in file['patch']):
        raise ChallengeError('Challenge file scope/content markers are not isolated or bounded')
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
    if (ci.get('pr') != number or ci.get('head_sha') != head.get('sha') or
            ci.get('base_sha') != base.get('sha')):
        raise ChallengeError('Selected CI witness is bound to a different PR revision')
    after=api.call('GET',f'/pulls/{number}')
    if (not isinstance(after,dict) or after.get('number') != number or
            (after.get('head') or {}).get('sha') != head.get('sha') or
            (after.get('base') or {}).get('sha') != base.get('sha') or
            (after.get('base') or {}).get('ref') != expected_base or
            after.get('merge_commit_sha') != before.get('merge_commit_sha') or
            after.get('draft') is not False or after.get('state') != 'open' or
            after.get('mergeable') is not True or after.get('merged_at') is not None or
            after.get('auto_merge') is not None):
        raise ChallengeError('PR identity/state changed during audit')
    reviews_after = paginated(api, f'/pulls/{number}/reviews')
    if reviews_after != reviews:
        raise ChallengeError('Review history changed during challenge audit')
    installed_after = api.call('GET', f'/rulesets/{selected_id}')
    if installed_after != installed_before:
        raise ChallengeError('Active policy changed during challenge audit')
    found_after = paginated(api, '/rulesets?includes_parents=true')
    if found_after != found:
        raise ChallengeError('Ruleset inventory changed during challenge audit')
    changes_after = paginated(api, f'/pulls/{number}/files')
    if changes_after != changes:
        raise ChallengeError('Challenge file inventory changed during audit')
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
