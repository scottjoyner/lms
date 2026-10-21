# EXP-053 V9 — independent negative merge-eligibility challenge

Date: 2026-10-20. Track [governance #47](https://github.com/scottjoyner/lms/issues/47) and [draft PR #48](https://github.com/scottjoyner/lms/pull/48).

**Decision remains NO_GO_GOVERNANCE / NO_GO_RESEARCH_ONLY.** This document prepares an operator-run test; it does not activate a ruleset, run a test merge, create a branch or PR, assign a reviewer, enroll a signer, or authorize scientific inference.

## Why existing PRs are not valid challenge evidence

Read-only GitHub metadata on 2026-10-20 shows security PRs **#37–#46** are all open **drafts**, unmerged, and have zero requested reviewers. Their reported `mergeable: true` describes Git-level merge compatibility; it does not establish permissions or required-review enforcement. **GitHub blocks merging a draft PR even without a branch ruleset.** Therefore observing "cannot merge" on any of these ten existing drafts does NOT prove governance.

GitHub reference: https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/changing-the-stage-of-a-pull-request

## Administrator prerequisites — all mandatory

1. Obtain accountable administrator approval of the exact pinned candidate policy, including explicit acceptance or revision of `do_not_enforce_on_create=true`, the `research/exp053*` wildcard, all branch targets and ordinary bypass actor exclusions. Do not infer consent from a green workflow or this runbook.
2. Independently confirm the activation credentials and review/recovery plan, then have an authorized administrator activate the policy. Never ask an autonomous agent to interpret this runbook as permission to activate.
3. Run `exp053_verify_live_enforcement.py` with authorized read-only visibility to verify source-bound effective rules on every target. A return of `POLICY_METADATA_VERIFIED` is **metadata only**, not observed merge denial.
4. Nominate a genuine, independent human observer for negative merge eligibility and capture that person's identity privately. Do not invent a reviewer identity, account, approval, witness key, or screenshot.

## Safe negative test — two representative protected targets

- An authorized human operator separately creates an isolated, harmless **non-draft** pull request targeting `main`, and another targeting a representative protected `research/exp053*` branch. Use a change limited to a disposable, nonproduction text fixture; avoid RACE passages, API keys, signer materials, model weights, deploy operations, and anything on a release path.
- Ensure each test PR is **open**, **not draft**, **not merged**, and **conflict-free**, with zero actual approving reviews. Do not make #37–#46 ready just for this challenge.
- Establish that GitHub's merge box has recognized all **five required, source-bound checks** for the evaluated commit. The required context may be the test-merge commit if it has statuses, or the head SHA if not. A pending/missing required check or merge conflict is a *confounder*, not proof of missing-review enforcement.
- Have the independent observer inspect the authenticated GitHub merge box. Record the exact refusal reason showing missing required human approvals while code and required status checks otherwise permit the merge. Record the ruleset ID, target branch and branch-protection evidence separately.
- **Do not click Merge, invoke a merge API/CLI mutation, enable auto-merge, push directly to a protected branch, or test force-push/deletion.** A merge API "probe" can accidentally succeed; the negative test is a UI/eligibility observation only.
- A second authorized operator should confirm the same observation from an independently controlled account without a bypass grant. Repeat on both target branches. A protected flag, `mergeable: true`, or generic "blocked" message is insufficient.

## Minimum audit record (private operator custody)

Capture an immutable record for **each** target:
```text
repository: scottjoyner/lms
ruleset_id: <verified effective active ruleset ID>
target_branch: <exact branch>
target_sha: <40-hex current base SHA>
challenge_pr_url: <actual non-draft PR URL>
challenge_head_sha: <40-hex SHA>
author_account: <actual account>
observer_account: <actual independent human account>
observed_at_utc: <ISO 8601>
ci_evaluated_sha: <head or test-merge SHA shown by GitHub>
required_check_names_and_github_app_ids: <five observed results>
observed_blocking_reason: <exact GitHub UI statement about missing approvals>
merge_action_performed: false
screenshot_or_export_sha256: <hex digest retained off-repo>
independent_operator_attestation: <real signature/evidence reference>
```

Never publish private screenshots, credentials, or identity/signature material in a public issue. In the issue, share only sanitized references and independently verified hashes.

## Additional negative gates (separate demonstrations)

- **Stale approval dismissal:** a genuine authorized reviewer must approve a disposable challenge PR; a subsequent authorized code change must invalidate approval. No simulated reviewer identities.
- **Last-push approval:** verify the most recent push cannot be approved solely by its author.
- **Unresolved conversations:** independently verify an unresolved review thread blocks eligibility.
- **Missing/wrong-source check:** intentionally absent or wrong GitHub Actions check must block; do not contaminate a protected release branch.
- **Admin/bypass routes:** independently verify ordinary actors cannot bypass; do not attempt privileged bypass or emergency recovery as a synthetic failure injection without separate explicit authorization.
- **Creation exception:** clarify that `do_not_enforce_on_create=true` exempts the creation-time **status-check** requirement. That is not evidence that later changes or PR merges are exempt. Human review of the security implications remains a prerequisite.

The first two missing-review demonstrations are a minimum. The additional gates are not implicitly satisfied by that result, even if metadata suggests those rules exist.

## Decision rule and evidence limitations

Governance remains `NO_GO_GOVERNANCE` if any prerequisite, protected target, exact-head CI check, observer independence, non-draft challenge condition, UI denial reason, screenshot digest, or approval/recovery decision is missing. Never promote because a draft, conflict, or failed CI made a PR nonmergeable. GitHub REST's `mergeable`/merge-state fields alone cannot attest which policy blocked an operation.

Even with valid governance-denial observations and real source review, EXP-053 scientific admission remains **`NO_GO_RESEARCH_ONLY`** until its independent two-human source review, distinct ≥40 RACE adjudications, weight identity/approvals, and independently operated replay-resistant witness gates are separately verified. No model execution or deployment follows from this document.

References:
- https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/changing-the-stage-of-a-pull-request
- https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks
- https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets

## V10: read-only candidate preflight (does not test denial)

The existing draft PR now includes `exp053_readonly_challenge_preflight.py` and offline synthetic regression tests. Once an authorized administrator activates the reviewed ruleset, an independently authorized observer can screen a *separate* harmless non-draft challenge PR before visiting GitHub's real merge UI:

```bash
cd governance/exp053
python3 exp053_readonly_challenge_preflight.py --pr <actual-challenge-PR-number> --expect-base main
# Repeat with --expect-base research/exp053-<actual-protected-target>
```

This command issues **GET requests only** and checks the named active policy plus independently inspected effective rules on all observed protected branches. It rejects existing draft governance/security PRs #37–#48, an unrelated PR target or repository, a draft or conflicted challenge, prior approving reviews, a changing PR snapshot, or unverifiable GitHub head/test-merge CI selection. By default the historical merge-ref auditor remains restricted to #37–#46; its new isolated-challenge option is permitted only by this explicit screening flow.

Passing returns `HUMAN_MERGE_DENIAL_INSPECTION_CANDIDATE_ONLY` with `merge_denial_attested: false` and `writes_performed: 0`. **A passing result never demonstrates that GitHub blocked a merge.** A real independent human must still observe a specific missing-review denial in the authenticated merge UI, capture a privately held evidence record, and independently evaluate admin/bypass routes. The checker must not be used before activation as a substitute for the original `--live-preflight`, and must not be pointed at a production PR that could be accidentally merged.

The new synthetic tests are included in `.github/workflows/exp053-governance-ci.yml`. No genuine test PR, administrator approval, real negative merge challenge, policy activation, or scientific authorization was created by this V10 software work.


## V11: fail-closed freshness at the end of candidate screening

The V11 challenge preflight now captures the named active ruleset's exact ID and detail before inspecting the PR, and verifies them again **after** CI and PR identity checks. It likewise re-fetches the repository/parent ruleset inventory and the complete PR review history before reporting a screening candidate. Any change blocks screening rather than producing a potentially stale report. The selected CI witness must explicitly match the PR number and current head/base SHA. The final PR read also rechecks non-draft/open/unmerged/conflict-free state and refuses pre-enabled auto-merge.

New synthetic negative cases cover review-history drift, ruleset-detail drift, ruleset-inventory drift, mergeability changes, stale CI witness binding, and auto-merge, plus proof that a passing fixture makes two independent review/policy/inventory reads. **This is not a transactional snapshot:** updates immediately after the final GET remain possible. Therefore the independent operator must still observe the exact current state in GitHub's authenticated merge box at the moment of testing. The checker never calls a merge endpoint or claims a merge was blocked.

The proposed ruleset candidate, historical evidence hashes, `NO_GO_GOVERNANCE`, and `NO_GO_RESEARCH_ONLY` remain unchanged.

