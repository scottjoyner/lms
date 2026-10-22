# EXP-053 governance — current operating policy (V40, 2026-10-22)

**Owner decision supersedes all V1–V39 two-reviewer proposals:** no human approval
or external reviewer is required to merge source-code pull requests. The
independent scientific evidence and research-release requirements remain
separate and are **not waived**.

## Intended repository rules

The pinned proposal in `EXP053_github_ruleset_candidate_2026-10-20.json`
requires an opened pull request with **zero approving reviews**, resolved
conversations, five exact GitHub Actions check contexts bound to app 15368,
strict up-to-date checks, no unapproved bypass actors, and force-push/deletion
protection. It targets `main` and flat `research/exp053*` branches.

The user approved eliminating the two-reviewer blocking requirement. This
does **not** imply that GitHub rules have already been installed or verified.
The candidate's digest is pinned in `exp053_admin_activation.py` and
`exp053_verify_live_enforcement.py`. Both must be updated whenever the
candidate's bytes change. Earlier V1–V39 docket entries are historical and
their two-approver instructions must **not** be followed.

## Delivery / closure sequence

1. Adopt this source, its amended policy, and required CI workflow into
   `main` through the normal PR merge path. No reviewer invitation needed.
2. From an **independently authenticated GitHub admin** session, create the
   candidate repository branch ruleset only after reviewing exact branch
   targeting, plan support and the pinned five GitHub check names. The
   GitHub connector used in this session has no ruleset-write action and
   Fleet Commander has no `gh` or `GITHUB_TOKEN` authenticated session.
   An unprivileged assistant cannot represent successful activation.
3. Verify `GET /rulesets`, every target's effective rules and `protected`
   flag, and current HEAD/CI. No false 'policy active' assertion from
   successfully merging source alone. The old missing-two-approvals challenge
   is **inapplicable** for a zero-approval ruleset; a future real test should
   verify rejection of an otherwise eligible PR with a missing required CI
   check or a disallowed direct push, without actually bypassing protection.
4. Close the obsolete 'require independent GitHub reviewers' governance
   issue only as **superseded**, tracking any uninstalled GitHub ruleset
   separately. Actual GitHub enforcement is not complete until witnessed.

## Scientific research remains held

`NO_GO_RESEARCH_ONLY` persists. Owner-reviewed sources, licensing,
reproducibility, weight provenance, independent rollback protection and
other evidence in EXP-053 research issue #31 are separate obligations.
Source CI green and a zero-review code workflow never authorize licensed
corpus use, model inference or release.

The offline test suite and default administrator dry run perform zero
production writes. Do not embed GitHub tokens or credentials in scripts,
CI logs, repository history or issue comments.
