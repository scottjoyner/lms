> **V40 OVERRIDE (2026-10-22):** Repository owner withdrew the two-reviewer requirement. Current policy is **zero required GitHub approvals**, with PR/CI/branch-integrity guards retained. See [current operator policy](README.md). Earlier two-reviewer and missing-approval negative-challenge instructions below are historical only. Scientific release gates remain independent.

# EXP-053 — Ready-to-review GitHub ruleset (administrator handoff)

**October 20, 2026 | `NO_GO_RESEARCH_ONLY` | PROPOSAL ONLY**

Repo: [`scottjoyner/lms`](https://github.com/scottjoyner/lms). Track governance in [P0 issue #47](https://github.com/scottjoyner/lms/issues/47), scientific gate in [issue #31](https://github.com/scottjoyner/lms/issues/31).

## What was verified

The connected GitHub API reports `main` and all 11 currently observed `research/exp053*` branches **unprotected**, no repository rulesets, and zero assigned/requested or submitted reviewers on draft PRs #37–#46. The detailed branch-protection endpoint returned 403 to the integration. This is a point-in-time observation, not proof of future enforcement.

**CI check names are not invented.** Checks were read from the GitHub checks API on the default branch's commit (`1f64776b3b31c581f88635a5ed3bc2a98a48f0c4`) and the ten exact draft PR heads. Five stable checks were successful at every observed head, all reported by GitHub Actions app ID **15368**:

`lint-and-test (3.12)`, `focused-tests (3.12)`, `qualification-tests (3.12)`, `operational-failure-injection (3.12)`, and `prompt-cache-tests (3.10)`.

## Proposed ruleset

See [ruleset candidate](EXP053_github_ruleset_candidate_2026-10-20.json), [snapshot evidence](EXP053_ruleset_validation_evidence_2026-10-20.json) and the [offline preflight validator](EXP053_validate_ruleset_candidate_2026-10-20.py).

* `enforcement=active` (not evaluate/disabled), `target=branch`.
* Scope `refs/heads/main` and `refs/heads/research/exp053*`. Slash-aware matching covers all 12 currently observed merge targets.
* Require **two independent approving reviews**; dismiss stale approvals on changes; require most recent push approved by another person; resolve review threads.
* Require the **five real checks** above, attributable to GitHub Actions integration ID 15368. Proposed `strict_required_status_checks_policy=true` means a base-branch update can require fresh tests.
* Reject force pushes and deletions. **No explicit bypass actors** (`[]`).

**Important limitations:** two GitHub approvals alone do not attest two independent scientific reviewers. Actual reviewers must be nominated and genuinely accountable, not bots or agents. A repo administrator must confirm the available account plan, protection enforcement behavior, permitted merge methods, and legitimate emergency-recovery procedure. Strict checks and protection of research branches can block otherwise normal development; plan the rollout and validate the target preview. The candidate is not installed.

## Offline preflight (no GitHub credentials required)

Place all three files together and run:

```bash
python3 EXP053_validate_ruleset_candidate_2026-10-20.py --dry-run
```

The validator checks branch glob coverage, prevents an overbroad branch pattern, requires approvals and review-thread controls, checks expected GitHub Actions names/IDs, and verifies that the snapshot makes no scientific approval claim. It does **not** make API calls or alter GitHub.

## V8 live preflight (read-only GitHub GET requests)

From `governance/exp053`, the new guarded helper can independently check the **current** main and research target HEADs, without activating a ruleset:

```bash
python3 exp053_admin_activation.py --live-preflight
```

This command checks that no ruleset has appeared since the historical snapshot; verifies the target branch inventory is unchanged during the scan; and checks each branch's HEAD has all five required, successful GitHub Actions contexts from app `15368`. It refuses missing, duplicate, stale or wrong-source checks, unexpected existing protection, and branch movement. It **only calls GET endpoints** through a dedicated read-only client; `--activate` is mutually exclusive with `--live-preflight`, and the live mode rejects `--confirm`. `GITHUB_TOKEN` is optional for public data and only supplies authentication for GET access, if present. On missing permissions/rate limits, treat a failed run as unverified, not an authorization override.

On 2026-10-20, Fleet Commander x1-370 ran this command against all **12** observed target branches and returned `LIVE_READ_ONLY_PREFLIGHT_VALID` / `writes_performed: 0`. A passing preflight proves current CI *metadata* only: **no active ruleset, actual GitHub merge denial, independent reviewers, bypass custody, or scientific execution approval** follows. Do not run this pre-activation-only check as a replacement for the post-activation effective-rule audit.


## Administrator activation (deliberate manual action)

1. Open [`Settings → Rules → Rulesets`](https://github.com/scottjoyner/lms/settings/rules) as an authorized repository administrator. Review P0 issue #47 and the JSON candidate. Configure a branch ruleset with the matching targets and rules; verify **Active** mode, not Evaluate/Disabled. Confirm bypass/recovery design. Alternatively, an administrator with a GitHub access token bearing **Administration: write** can apply the JSON through the [GitHub REST repository rulesets create endpoint](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset); this has **not** been executed by the assistant.
2. Confirm that GitHub's target preview covers `main` plus *every* existing `research/exp053...` branch and excludes unrelated branches. Check actual branch-rule status for representative protected refs (`main`, the PR #37 base, and the #46 base); a ruleset listing alone isn't enough.
3. Validate that an **unreviewed synthetic PR cannot merge**, that failing or absent required checks block a merge, and that author-only approvals do not count. Do not use actual licensed RACE material to test these controls.
4. Nominate and verify two independent qualified GitHub human reviewers; request their reviews of the exact PR heads #37–#46. Do not assign anyone by guess or fabricate approvals. Review each PR's frozen security spec and read-only, source-free regression evidence.
5. Keep ALL review and scientific gates distinct. Even after enforced branch rules, maintain `NO_GO_RESEARCH_ONLY` until actual separate reviewer custody, ≥40 genuinely independently co-reviewed/adjudicated source-distinct articles, external high-water authority and weight identity are established.

This package is an implementation-ready **proposal**, not an enforced GitHub policy. No rulesets, draft statuses, approval pins, deployments, merges, or real operator/witness credentials were changed while preparing it.

GitHub docs: [REST rulesets parameters](https://docs.github.com/en/rest/repos/rules?apiVersion=2022-11-28), [Available rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets).

## V31 — two named reviewer permission gate

**Additional required human decision, not an authorization to execute:** An
accountable administrator must first verify and document two different real,
consenting non-author GitHub users with sufficient permissions and independent
review custody. The guarded activation CLI now requires exactly two
`--reviewer <github-login>` options and explicit
`--reviewer-consent-attested` *in addition* to the original separate
administrator policy confirmation and authenticated administrative token.
No default or pre-filled logins are provided, and no consent is presumed.

Before any authorized POST, the script reads each named collaborator's
effective GitHub base permission and account type (fail-closed on anything
other than matching `User`, `write` or `admin`), then repeats both reads
right before the possible single POST. These GETs **do not establish**
independent real-world identity/consent; all externally witnessed
administrative decisions, actual policy enforcement checks, and negative
GitHub merge-denial tests remain mandatory. A failure must not trigger
blind POST retry or collaborator invitations. Default execution remains
an offline zero-write dry run.
