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



## V12 — isolate the entire challenge diff

The V12 screening tool now rejects a prospective challenge PR unless GitHub's complete `/pulls/{number}/files` listing contains **exactly one newly added documentation fixture**. The only accepted path form is:

```text
governance/exp053/challenges/exp053-merge-denial-<lowercase-alphanumeric-or-hyphen-slug>.md
```

The fixture must be new (not a modification or rename), contain **1–15 added lines**, **zero deletions**, and a GitHub text patch no longer than **2,000 characters**. It must include an added Markdown heading `# EXP-053 governance challenge fixture`. For example, a future *human-authorized* disposable fixture could contain:

```markdown
# EXP-053 governance challenge fixture
Read-only governance denial exercise; no production or scientific data.
```

The same file list is fetched again at the end of the audit. A changed file set, additional modified code, a binary/omitted patch, a larger diff, or a different path fails closed. **These mechanical checks cannot prove fixture content is safe or non-sensitive.** The independent human operator must inspect the exact proposed text, target branch and permissions before creating a challenge PR.

This is a **post-activation**, separate-PR operator procedure only. A new fixture cannot be used as evidence until a qualified human witnesses GitHub's missing-approvals denial on an otherwise acceptable, non-draft, conflict-free PR with valid required checks. A markdown-only PR may skip some CI workflows; missing checks must be treated as a confounder rather than a merge-denial proof. No challenge PR was created in V12.

Fleet Commander x1-370: **86/86** synthetic tests passed after eight new changed-file scope and drift regression cases. The existing candidate policy JSON, pinned evidence files, and all governance/scientific release HOLDs remain unchanged.


## V13 — challenge target branch bound to effective protection and exact base HEAD

V13 adds a distinct missing evidence boundary to the read-only challenge candidate preflight. The specific `--expect-base` branch must now be present **exactly once** in the independent effective-rule audit's branch results with both `protected: true` and `required_rule_set_observed: true`. A successful overall policy report no longer suffices when the selected target was not individually inspected.

The preflight also performs GET-only `/branches/{target}` reads **before and after** the PR/CI/file/review scan. Both must show the selected branch, `protected: true`, and the same current HEAD SHA. The PR's `base.sha` must match that current protected-branch SHA. The screening report records `protected_base_sha` for comparison by the independent operator.

This defends against a newly created, unprotected, mismatched or moving target branch being mistaken for a valid test environment. Tests cover an omitted audit target, absent protection, PR base/head mismatch, mid-audit branch movement, an unprotected audited target, and the expected two branch reads. No GET snapshot is atomic: actual enforcement and an independently observed GitHub merge-denial still require contemporaneous human verification.

**Do not activate rulesets, create challenge PRs, or merge automatically.** The research release remains `NO_GO_RESEARCH_ONLY`, regardless of candidate-screening results.

## V21 — cross-module offline challenge integration (synthetic, no GitHub writes)

The earlier V10–V18 screening regressions mocked both the effective GitHub ruleset auditor
(`Audit.run`) and the head/test-merge CI witness (`audit_pr`). The new
`test_exp053_challenge_chain_integration.py` tests the **actual implementations
end to end**, passing only a strictly GET-only synthetic GitHub API fixture
with the pinned candidate ruleset, all 12 target branches, per-branch effective
rule provenance, a harmless new-file PR, and five app-15368 check runs.

Six integration cases exercise the full **candidate-only** path for exact
test-merge checks and the documented no-merge-activity HEAD fallback, plus
fail-closed paths for an unapplied or unprotected target, a missing required
merge check, and a generated merge commit with incorrect parent binding.
They confirm the result is **not an attested merge denial** and never sends
a POST/Merge request. This checks API contracts between the modules, not actual
GitHub policy enforcement, independent reviewer identity, or UI merge denial.

The same test module is now part of the existing offline governance GitHub
Actions workflow, with no network credentials or repository mutation.
An accountable administrator must still approve/activate the ruleset and
commission human-witnessed non-draft negative challenges before claiming
protection. Governance and scientific execution remain NO-GO.

## V22 — isolate review-approval denial from other merge blockers

The read-only challenge-screening path now requires **zero prior submitted
review events** (not merely zero `APPROVED` events), and **zero inline pull
request review comments**. A `CHANGES_REQUESTED` review or unresolved
review discussion can prevent a merge independently of the configured
two-approval rule, making a negative test ambiguous. Both GET-only collections
are read again after CI and policy verification; additions or changes during
the audit fail closed. The output records the observed zero review and zero
inline-comment counts.

This requirement intentionally rejects even harmless prior `COMMENTED`
reviews or resolved historical inline comments: use a **fresh, isolated,
human-inspected challenge PR** rather than attempting to clean up an existing
review history. GitHub's REST inline comments are not an authoritative
GraphQL review-thread resolution report, so this is a conservative screen,
**not** proof that a merge box refused the PR specifically for missing
approvals. The human witness must still verify the actual refusal in GitHub,
all required checks, no other reported blockers, unchanged branch and PR
identity, and no auto-merge or merge action. No policy or scientific
authorization changes follow from passing these synthetic regressions.
